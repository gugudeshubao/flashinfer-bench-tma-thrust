/*
 * GDN Prefill v5 — CUDA kernel for B200 (sm100)
 *
 * Processes variable-length sequences with sequential token scan.
 * 
 * Grid: (N=num_seqs, H=8, V_BLOCKS)
 * Block: 128 threads (4 warps)
 *
 * State layout: k-last [N, H, V=128, K=128] float32
 * GVA: num_q_heads=4, num_v_heads=8 → qk_head = v_head // 2
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace gdn {

// Constants
constexpr int PREFILL_D = 128;
constexpr int PREFILL_WARP_SIZE = 32;

// Softplus approximation
__device__ __forceinline__ float prefill_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

/*
 * Prefill kernel — sequential token scan
 *
 * Each block handles one (seq, v_head, v_block) tuple.
 * Processes all tokens in sequence sequentially.
 */
template<int BLOCK_V>
__global__ void gdn_prefill_kernel_v5(
    // Inputs [T, H, D] packed
    const __nv_bfloat16* __restrict__ Q,      // [T, 4, D]
    const __nv_bfloat16* __restrict__ K,      // [T, 4, D]
    const __nv_bfloat16* __restrict__ V,      // [T, 8, D]
    const float* __restrict__ State,          // [N, 8, D, D] k-last
    // Gates
    const float* __restrict__ A_log,          // [8]
    const __nv_bfloat16* __restrict__ A,      // [T, 8]
    const float* __restrict__ DtBias,         // [8]
    const __nv_bfloat16* __restrict__ B_gate, // [T, 8]
    // Sequence info
    const int32_t* __restrict__ CuSeqlens,    // [N+1]
    // Outputs
    __nv_bfloat16* __restrict__ Out,          // [T, 8, D]
    float* __restrict__ NewState,             // [N, 8, D, D]
    // Params
    float scale,
    int stride_q_t, int stride_q_h,
    int stride_k_t, int stride_k_h,
    int stride_v_t, int stride_v_h,
    int stride_s_n, int stride_s_h, int stride_s_v,
    int stride_a_t,
    int stride_b_t,
    int stride_o_t, int stride_o_h,
    int stride_ns_n, int stride_ns_h, int stride_ns_v
) {
    // Grid indices
    const int n = blockIdx.x;           // sequence index
    const int h = blockIdx.y;           // v_head [0, 8)
    const int vb = blockIdx.z;          // v_block [0, D/BLOCK_V)
    const int v0 = vb * BLOCK_V;        // first V element
    const int qk_h = h / 2;             // GVA: 2 v-heads share qk-head
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x; // 128
    
    // Sequence bounds
    const int t_start = CuSeqlens[n];
    const int t_end = CuSeqlens[n + 1];
    const int seq_len = t_end - t_start;
    
    // Head-level constants
    const float alog = A_log[h];
    const float dt_val = DtBias[h];
    
    // Shared memory layout
    extern __shared__ char smem_raw[];
    float* q_smem = (float*)smem_raw;                     // [D]
    float* k_smem = q_smem + PREFILL_D;                   // [D]
    float* v_smem = k_smem + PREFILL_D;                   // [BLOCK_V]
    float* state_smem = v_smem + BLOCK_V;                 // [BLOCK_V * D]
    float* old_v_smem = state_smem + BLOCK_V * PREFILL_D; // [BLOCK_V]
    float* out_smem = old_v_smem + BLOCK_V;               // [BLOCK_V]
    
    // ─── Load initial state [BLOCK_V, D] into shared memory ───────────
    const float* state_ptr = State + n * stride_s_n + h * stride_s_h + v0 * stride_s_v;
    
    for (int i = tid; i < BLOCK_V * PREFILL_D; i += num_threads) {
        int vi = i / PREFILL_D;
        int ki = i % PREFILL_D;
        state_smem[i] = state_ptr[vi * stride_s_v + ki];
    }
    __syncthreads();
    
    // ─── Sequential token scan ────────────────────────────────────────
    for (int tok = 0; tok < seq_len; tok++) {
        int t = t_start + tok;
        
        // Load gates for this token (thread 0)
        __shared__ float g_shared, beta_shared;
        if (tid == 0) {
            float a_val = __bfloat162float(A[t * stride_a_t + h]);
            float b_val = __bfloat162float(B_gate[t * stride_b_t + h]);
            
            float sp = prefill_softplus(a_val + dt_val);
            g_shared = expf(-expf(alog) * sp);
            beta_shared = 1.0f / (1.0f + expf(-b_val));  // sigmoid
        }
        __syncthreads();
        
        float g = g_shared;
        float beta = beta_shared;
        
        // Load Q, K [D] into shared memory
        for (int i = tid; i < PREFILL_D; i += num_threads) {
            q_smem[i] = __bfloat162float(Q[t * stride_q_t + qk_h * stride_q_h + i]);
            k_smem[i] = __bfloat162float(K[t * stride_k_t + qk_h * stride_k_h + i]);
        }
        
        // Load V slice [BLOCK_V]
        for (int i = tid; i < BLOCK_V; i += num_threads) {
            v_smem[i] = __bfloat162float(V[t * stride_v_t + h * stride_v_h + v0 + i]);
        }
        __syncthreads();
        
        // Apply gate decay: S = g * S
        for (int i = tid; i < BLOCK_V * PREFILL_D; i += num_threads) {
            state_smem[i] *= g;
        }
        __syncthreads();
        
        // Compute old_v = S @ k
        if (tid < BLOCK_V) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int ki = 0; ki < PREFILL_D; ki++) {
                sum += state_smem[tid * PREFILL_D + ki] * k_smem[ki];
            }
            old_v_smem[tid] = sum;
        }
        __syncthreads();
        
        // Rank-1 update: S += delta * k^T
        for (int i = tid; i < BLOCK_V * PREFILL_D; i += num_threads) {
            int vi = i / PREFILL_D;
            int ki = i % PREFILL_D;
            float delta = beta * (v_smem[vi] - old_v_smem[vi]);
            state_smem[i] += delta * k_smem[ki];
        }
        __syncthreads();
        
        // Compute out = scale * S @ q
        if (tid < BLOCK_V) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int ki = 0; ki < PREFILL_D; ki++) {
                sum += state_smem[tid * PREFILL_D + ki] * q_smem[ki];
            }
            out_smem[tid] = scale * sum;
        }
        __syncthreads();
        
        // Store output for this token
        for (int i = tid; i < BLOCK_V; i += num_threads) {
            Out[t * stride_o_t + h * stride_o_h + v0 + i] = __float2bfloat16(out_smem[i]);
        }
        __syncthreads();
    }
    
    // ─── Store final state [BLOCK_V, D] ───────────────────────────────
    float* new_state_ptr = NewState + n * stride_ns_n + h * stride_ns_h + v0 * stride_ns_v;
    
    for (int i = tid; i < BLOCK_V * PREFILL_D; i += num_threads) {
        int vi = i / PREFILL_D;
        int ki = i % PREFILL_D;
        new_state_ptr[vi * stride_ns_v + ki] = state_smem[i];
    }
}

// Launcher function
void gdn_prefill_v5_launch(
    const void* Q, const void* K, const void* V, const void* State,
    const void* A_log, const void* A, const void* DtBias, const void* B_gate,
    const void* CuSeqlens,
    void* Out, void* NewState,
    float scale,
    int N, int num_v_heads, int D,
    int stride_q_t, int stride_q_h,
    int stride_k_t, int stride_k_h,
    int stride_v_t, int stride_v_h,
    int stride_s_n, int stride_s_h, int stride_s_v,
    int stride_a_t, int stride_b_t,
    int stride_o_t, int stride_o_h,
    int stride_ns_n, int stride_ns_h, int stride_ns_v,
    int BLOCK_V,
    cudaStream_t stream
) {
    int V_BLOCKS = D / BLOCK_V;
    dim3 grid(N, num_v_heads, V_BLOCKS);
    dim3 block(128);  // 4 warps
    
    // Shared memory: q[D] + k[D] + v[BLOCK_V] + state[BLOCK_V*D] + old_v[BLOCK_V] + out[BLOCK_V]
    size_t smem_size = (PREFILL_D + PREFILL_D + BLOCK_V + BLOCK_V * PREFILL_D + BLOCK_V + BLOCK_V) * sizeof(float);
    
    if (BLOCK_V == 16) {
        gdn_prefill_kernel_v5<16><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State,
            (const float*)A_log, (const __nv_bfloat16*)A, (const float*)DtBias,
            (const __nv_bfloat16*)B_gate,
            (const int32_t*)CuSeqlens,
            (__nv_bfloat16*)Out, (float*)NewState,
            scale,
            stride_q_t, stride_q_h, stride_k_t, stride_k_h,
            stride_v_t, stride_v_h,
            stride_s_n, stride_s_h, stride_s_v,
            stride_a_t, stride_b_t,
            stride_o_t, stride_o_h,
            stride_ns_n, stride_ns_h, stride_ns_v
        );
    } else {  // BLOCK_V == 32
        gdn_prefill_kernel_v5<32><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State,
            (const float*)A_log, (const __nv_bfloat16*)A, (const float*)DtBias,
            (const __nv_bfloat16*)B_gate,
            (const int32_t*)CuSeqlens,
            (__nv_bfloat16*)Out, (float*)NewState,
            scale,
            stride_q_t, stride_q_h, stride_k_t, stride_k_h,
            stride_v_t, stride_v_h,
            stride_s_n, stride_s_h, stride_s_v,
            stride_a_t, stride_b_t,
            stride_o_t, stride_o_h,
            stride_ns_n, stride_ns_h, stride_ns_v
        );
    }
}

}  // namespace gdn

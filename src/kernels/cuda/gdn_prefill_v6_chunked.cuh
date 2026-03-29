/*
 * GDN Prefill v6 — Optimized CUDA kernel with Chunking
 *
 * Key Optimization: Chunk-based processing to increase compute density
 *
 * Problem with v5:
 *   - Process 1 token per iteration
 *   - Arithmetic Intensity = 1 FLOP/byte (memory-bound)
 *
 * Solution in v6:
 *   - Process CHUNK_SIZE tokens together
 *   - Arithmetic Intensity = CHUNK_SIZE FLOP/byte (compute-bound!)
 *   - Better register reuse, fewer global memory accesses
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

constexpr int PREFILL_D_V6 = 128;
constexpr int PREFILL_WARP_SIZE_V6 = 32;

__device__ __forceinline__ float prefill_softplus_v6(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

/*
 * Optimized Prefill kernel with chunking
 *
 * CHUNK_SIZE: Number of tokens processed together (e.g., 4, 8, 16)
 * Higher CHUNK_SIZE = Higher arithmetic intensity = Better compute utilization
 *
 * Arithmetic Intensity Analysis:
 *   - Per token: 2*D*D FLOPs (S@k + S@q) 
 *   - Memory: 2*D*D bytes (state read/write)
 *   - AI_single = 2*D*D / (2*D*D) = 1 FLOP/byte
 *   
 *   - With chunk of C tokens: C * 2*D*D FLOPs
 *   - Memory: 2*D*D bytes (state accessed once, reused C times)
 *   - AI_chunk = C * 2*D*D / (2*D*D) = C FLOP/byte
 *
 * For C=8: AI = 8 FLOP/byte → approaches compute-bound territory!
 */
template<int BLOCK_V, int CHUNK_SIZE>
__global__ void gdn_prefill_kernel_v6_chunked(
    // Inputs
    const __nv_bfloat16* __restrict__ Q,      // [T, 4, D]
    const __nv_bfloat16* __restrict__ K,      // [T, 4, D]
    const __nv_bfloat16* __restrict__ V,      // [T, 8, D]
    const float* __restrict__ State,          // [N, 8, D, D]
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
    int stride_a_t, int stride_b_t,
    int stride_o_t, int stride_o_h,
    int stride_ns_n, int stride_ns_h, int stride_ns_v
) {
    // Grid indices
    const int n = blockIdx.x;           // sequence index
    const int h = blockIdx.y;           // v_head [0, 8)
    const int vb = blockIdx.z;          // v_block
    const int v0 = vb * BLOCK_V;        // first V element
    const int qk_h = h / 2;             // GVA mapping
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x; // 128
    
    // Sequence bounds
    const int t_start = CuSeqlens[n];
    const int t_end = CuSeqlens[n + 1];
    const int seq_len = t_end - t_start;
    
    // Head constants
    const float alog = A_log[h];
    const float dt_val = DtBias[h];
    
    // Shared memory layout
    extern __shared__ char smem_raw[];
    float* q_smem = (float*)smem_raw;                              // [CHUNK_SIZE * D]
    float* k_smem = q_smem + CHUNK_SIZE * PREFILL_D_V6;           // [CHUNK_SIZE * D]
    float* v_smem = k_smem + CHUNK_SIZE * PREFILL_D_V6;           // [CHUNK_SIZE * BLOCK_V]
    float* state_smem = v_smem + CHUNK_SIZE * BLOCK_V;            // [BLOCK_V * D]
    float* g_smem = state_smem + BLOCK_V * PREFILL_D_V6;          // [CHUNK_SIZE]
    float* beta_smem = g_smem + CHUNK_SIZE;                        // [CHUNK_SIZE]
    float* old_v_smem = beta_smem + CHUNK_SIZE;                    // [BLOCK_V]
    float* out_smem = old_v_smem + BLOCK_V;                        // [CHUNK_SIZE * BLOCK_V]
    
    // ─── Load initial state [BLOCK_V, D] ───────────────────────────────
    const float* state_ptr = State + n * stride_s_n + h * stride_s_h + v0 * stride_s_v;
    
    for (int i = tid; i < BLOCK_V * PREFILL_D_V6; i += num_threads) {
        int vi = i / PREFILL_D_V6;
        int ki = i % PREFILL_D_V6;
        state_smem[i] = state_ptr[vi * stride_s_v + ki];
    }
    __syncthreads();
    
    // ─── Process tokens in chunks ──────────────────────────────────────
    int num_chunks = (seq_len + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int chunk_start = chunk * CHUNK_SIZE;
        int chunk_end = min(chunk_start + CHUNK_SIZE, seq_len);
        int actual_chunk_size = chunk_end - chunk_start;
        
        // ── Load chunk of Q, K, V and gates ──────────────────────────
        for (int c = 0; c < actual_chunk_size; c++) {
            int t = t_start + chunk_start + c;
            
            // Load Q[c], K[c] into shared memory
            for (int i = tid; i < PREFILL_D_V6; i += num_threads) {
                q_smem[c * PREFILL_D_V6 + i] = __bfloat162float(Q[t * stride_q_t + qk_h * stride_q_h + i]);
                k_smem[c * PREFILL_D_V6 + i] = __bfloat162float(K[t * stride_k_t + qk_h * stride_k_h + i]);
            }
            
            // Load V slice [BLOCK_V]
            for (int i = tid; i < BLOCK_V; i += num_threads) {
                v_smem[c * BLOCK_V + i] = __bfloat162float(V[t * stride_v_t + h * stride_v_h + v0 + i]);
            }
            
            // Load gates (single thread per chunk element)
            if (tid == c) {
                float a_val = __bfloat162float(A[t * stride_a_t + h]);
                float b_val = __bfloat162float(B_gate[t * stride_b_t + h]);
                
                float sp = prefill_softplus_v6(a_val + dt_val);
                g_smem[c] = expf(-expf(alog) * sp);
                beta_smem[c] = 1.0f / (1.0f + expf(-b_val));
            }
        }
        __syncthreads();
        
        // ── Process chunk tokens with state reuse ────────────────────
        // This is where we get the compute density boost!
        // State is loaded once, used CHUNK_SIZE times
        
        for (int c = 0; c < actual_chunk_size; c++) {
            float g = g_smem[c];
            float beta = beta_smem[c];
            
            // Apply gate decay: S = g * S
            for (int i = tid; i < BLOCK_V * PREFILL_D_V6; i += num_threads) {
                state_smem[i] *= g;
            }
            __syncthreads();
            
            // Compute old_v = S @ k[c]
            if (tid < BLOCK_V) {
                float sum = 0.0f;
                const float* k_ptr = k_smem + c * PREFILL_D_V6;
                #pragma unroll 8
                for (int ki = 0; ki < PREFILL_D_V6; ki++) {
                    sum += state_smem[tid * PREFILL_D_V6 + ki] * k_ptr[ki];
                }
                old_v_smem[tid] = sum;
            }
            __syncthreads();
            
            // Rank-1 update: S += delta * k^T
            const float* k_ptr = k_smem + c * PREFILL_D_V6;
            const float* v_ptr = v_smem + c * BLOCK_V;
            
            for (int i = tid; i < BLOCK_V * PREFILL_D_V6; i += num_threads) {
                int vi = i / PREFILL_D_V6;
                int ki = i % PREFILL_D_V6;
                float delta = beta * (v_ptr[vi] - old_v_smem[vi]);
                state_smem[i] += delta * k_ptr[ki];
            }
            __syncthreads();
            
            // Compute out = scale * S @ q[c]
            if (tid < BLOCK_V) {
                float sum = 0.0f;
                const float* q_ptr = q_smem + c * PREFILL_D_V6;
                #pragma unroll 8
                for (int ki = 0; ki < PREFILL_D_V6; ki++) {
                    sum += state_smem[tid * PREFILL_D_V6 + ki] * q_ptr[ki];
                }
                out_smem[c * BLOCK_V + tid] = scale * sum;
            }
            __syncthreads();
        }
        
        // ── Store outputs for this chunk ─────────────────────────────
        for (int c = 0; c < actual_chunk_size; c++) {
            int t = t_start + chunk_start + c;
            for (int i = tid; i < BLOCK_V; i += num_threads) {
                Out[t * stride_o_t + h * stride_o_h + v0 + i] = 
                    __float2bfloat16(out_smem[c * BLOCK_V + i]);
            }
        }
        __syncthreads();
    }
    
    // ─── Store final state ─────────────────────────────────────────────
    float* new_state_ptr = NewState + n * stride_ns_n + h * stride_ns_h + v0 * stride_ns_v;
    
    for (int i = tid; i < BLOCK_V * PREFILL_D_V6; i += num_threads) {
        int vi = i / PREFILL_D_V6;
        int ki = i % PREFILL_D_V6;
        new_state_ptr[vi * stride_ns_v + ki] = state_smem[i];
    }
}

// Launcher function with CHUNK_SIZE selection
void gdn_prefill_v6_launch(
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
    int CHUNK_SIZE,  // New parameter!
    cudaStream_t stream
) {
    int V_BLOCKS = D / BLOCK_V;
    dim3 grid(N, num_v_heads, V_BLOCKS);
    dim3 block(128);
    
    // Shared memory for chunked processing
    // q[CHUNK*D] + k[CHUNK*D] + v[CHUNK*BLOCK_V] + state[BLOCK_V*D] + g[CHUNK] + beta[CHUNK] + old_v[BLOCK_V] + out[CHUNK*BLOCK_V]
    size_t smem_size = (CHUNK_SIZE * D + CHUNK_SIZE * D + CHUNK_SIZE * BLOCK_V + 
                        BLOCK_V * D + CHUNK_SIZE + CHUNK_SIZE + BLOCK_V + 
                        CHUNK_SIZE * BLOCK_V) * sizeof(float);
    
    // Dispatch based on BLOCK_V and CHUNK_SIZE
    #define LAUNCH_KERNEL(BV, CS) \
        gdn_prefill_kernel_v6_chunked<BV, CS><<<grid, block, smem_size, stream>>>( \
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V, \
            (const float*)State, (const float*)A_log, (const __nv_bfloat16*)A, \
            (const float*)DtBias, (const __nv_bfloat16*)B_gate, (const int32_t*)CuSeqlens, \
            (__nv_bfloat16*)Out, (float*)NewState, scale, \
            stride_q_t, stride_q_h, stride_k_t, stride_k_h, \
            stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v, \
            stride_a_t, stride_b_t, stride_o_t, stride_o_h, \
            stride_ns_n, stride_ns_h, stride_ns_v)
    
    if (BLOCK_V == 16) {
        if (CHUNK_SIZE == 4) { LAUNCH_KERNEL(16, 4); }
        else if (CHUNK_SIZE == 8) { LAUNCH_KERNEL(16, 8); }
        else { LAUNCH_KERNEL(16, 4); }  // default
    } else {  // BLOCK_V == 32
        if (CHUNK_SIZE == 4) { LAUNCH_KERNEL(32, 4); }
        else if (CHUNK_SIZE == 8) { LAUNCH_KERNEL(32, 8); }
        else { LAUNCH_KERNEL(32, 4); }
    }
    
    #undef LAUNCH_KERNEL
}

}  // namespace gdn

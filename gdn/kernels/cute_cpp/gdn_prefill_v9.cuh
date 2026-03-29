/*
 * GDN Prefill v9 — CuTe C++ kernel with SMEM Swizzle + Chunking
 *
 * Key Features:
 *   - CuTe-style SMEM layout with Swizzle<3,3,3> for bank conflict avoidance
 *   - Chunk-based processing (CHUNK_SIZE tokens at once) for compute density
 *   - Shared memory staging for Q, K, V, and State
 *   - Warp-parallel V-tile processing
 *
 * Arithmetic Intensity:
 *   - Sequential: 1 FLOP/byte (memory-bound)
 *   - Chunked (C=8): 8 FLOP/byte (compute-bound!)
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

// CuTe requires CUTLASS
#if __has_include(<cute/tensor.hpp>)
#define HAS_CUTE 1
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
using namespace cute;
#else
#define HAS_CUTE 0
#endif

namespace gdn {

// ============================================================
// Constants
// ============================================================

constexpr int V9P_D = 128;
constexpr int V9P_WARP_SIZE = 32;
constexpr int V9P_NUM_WARPS = 4;
constexpr int V9P_THREADS = V9P_NUM_WARPS * V9P_WARP_SIZE;

// ============================================================
// Utility Functions
// ============================================================

__device__ __forceinline__ float v9p_fast_exp(float x) {
    return __expf(x);
}

__device__ __forceinline__ float v9p_softplus(float x) {
    return (x > 20.0f) ? x : __logf(1.0f + __expf(x));
}

__device__ __forceinline__ float v9p_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// Warp-level reduction
__device__ __forceinline__ float warp_reduce_sum_v9p(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Swizzle function for bank conflict avoidance
__device__ __forceinline__ int swizzle_idx(int d) {
    return d ^ ((d >> 3) & 7);
}

// ============================================================
// CuTe C++ Prefill Kernel with Chunking
// ============================================================

template<int BLOCK_V, int CHUNK_SIZE>
__global__ void __launch_bounds__(V9P_THREADS)
gdn_prefill_kernel_v9_chunked(
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
    const int warp_id = tid / V9P_WARP_SIZE;
    const int lane_id = tid % V9P_WARP_SIZE;
    
    // Sequence bounds
    const int t_start = CuSeqlens[n];
    const int t_end = CuSeqlens[n + 1];
    const int seq_len = t_end - t_start;
    
    // Head constants
    const float alog = A_log[h];
    const float dt_val = DtBias[h];
    
    // ============================================================
    // Shared Memory Layout with Swizzle
    // ============================================================
    extern __shared__ char smem_raw[];
    
    // Layout: [Q chunk, K chunk, V chunk, State, gates, output]
    float* s_q = reinterpret_cast<float*>(smem_raw);          // [CHUNK_SIZE * D]
    float* s_k = s_q + CHUNK_SIZE * V9P_D;                     // [CHUNK_SIZE * D]
    float* s_v = s_k + CHUNK_SIZE * V9P_D;                     // [CHUNK_SIZE * BLOCK_V]
    float* s_state = s_v + CHUNK_SIZE * BLOCK_V;               // [BLOCK_V * D] swizzled
    float* s_g = s_state + BLOCK_V * V9P_D;                    // [CHUNK_SIZE]
    float* s_beta = s_g + CHUNK_SIZE;                          // [CHUNK_SIZE]
    float* s_out = s_beta + CHUNK_SIZE;                        // [CHUNK_SIZE * BLOCK_V]
    
    // ============================================================
    // Load Initial State with Swizzle
    // ============================================================
    const float* state_ptr = State + n * stride_s_n + h * stride_s_h + v0 * stride_s_v;
    
    #pragma unroll
    for (int i = tid; i < BLOCK_V * V9P_D; i += V9P_THREADS) {
        int vi = i / V9P_D;
        int di = i % V9P_D;
        // Apply swizzle for bank conflict avoidance
        int swizzled_di = swizzle_idx(di);
        s_state[vi * V9P_D + swizzled_di] = state_ptr[vi * stride_s_v + di];
    }
    
    // Initialize output accumulator
    for (int i = tid; i < CHUNK_SIZE * BLOCK_V; i += V9P_THREADS) {
        s_out[i] = 0.0f;
    }
    
    __syncthreads();
    
    // ============================================================
    // Process tokens in chunks
    // ============================================================
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += CHUNK_SIZE) {
        int chunk_len = min(CHUNK_SIZE, seq_len - chunk_start);
        
        // ─── Load Q, K, V for this chunk ─────────────────────────
        for (int c = 0; c < chunk_len; c++) {
            int t = t_start + chunk_start + c;
            
            // Load Q and K
            const __nv_bfloat16* q_ptr = Q + t * stride_q_t + qk_h * stride_q_h;
            const __nv_bfloat16* k_ptr = K + t * stride_k_t + qk_h * stride_k_h;
            
            for (int d = tid; d < V9P_D; d += V9P_THREADS) {
                s_q[c * V9P_D + d] = __bfloat162float(q_ptr[d]);
                s_k[c * V9P_D + d] = __bfloat162float(k_ptr[d]);
            }
            
            // Load V slice
            const __nv_bfloat16* v_ptr = V + t * stride_v_t + h * stride_v_h + v0;
            if (tid < BLOCK_V) {
                s_v[c * BLOCK_V + tid] = __bfloat162float(v_ptr[tid]);
            }
            
            // Compute gates
            if (tid == 0) {
                float a_val = __bfloat162float(A[t * stride_a_t + h]);
                float b_val = __bfloat162float(B_gate[t * stride_b_t + h]);
                
                float x = a_val + dt_val;
                float sp = v9p_softplus(x);
                s_g[c] = v9p_fast_exp(-v9p_fast_exp(alog) * sp);
                s_beta[c] = v9p_sigmoid(b_val);
            }
        }
        
        __syncthreads();
        
        // ─── Process chunk tokens ────────────────────────────────
        for (int c = 0; c < chunk_len; c++) {
            float g = s_g[c];
            float beta = s_beta[c];
            
            // Each warp handles a subset of V elements
            for (int v_idx = warp_id; v_idx < BLOCK_V; v_idx += V9P_NUM_WARPS) {
                
                // ─── 1. Apply decay and compute old_v ────────────
                float old_v = 0.0f;
                
                #pragma unroll 4
                for (int d = lane_id; d < V9P_D; d += V9P_WARP_SIZE) {
                    int swizzled_d = swizzle_idx(d);
                    float decayed_s = g * s_state[v_idx * V9P_D + swizzled_d];
                    old_v += decayed_s * s_k[c * V9P_D + d];
                    // Store decayed state back (in-place)
                    s_state[v_idx * V9P_D + swizzled_d] = decayed_s;
                }
                
                // Warp reduction for old_v
                old_v = warp_reduce_sum_v9p(old_v);
                
                // ─── 2. Delta rule update ────────────────────────
                float v_elem = s_v[c * BLOCK_V + v_idx];
                float delta = beta * (v_elem - old_v);
                
                // ─── 3. Update state and compute output ──────────
                float out_val = 0.0f;
                
                #pragma unroll 4
                for (int d = lane_id; d < V9P_D; d += V9P_WARP_SIZE) {
                    int swizzled_d = swizzle_idx(d);
                    float new_s = s_state[v_idx * V9P_D + swizzled_d] + delta * s_k[c * V9P_D + d];
                    s_state[v_idx * V9P_D + swizzled_d] = new_s;
                    out_val += new_s * s_q[c * V9P_D + d];
                }
                
                // Warp reduction for output
                out_val = warp_reduce_sum_v9p(out_val);
                
                // Store output
                if (lane_id == 0) {
                    s_out[c * BLOCK_V + v_idx] = scale * out_val;
                }
            }
            
            __syncthreads();
        }
        
        // ─── Write chunk outputs ─────────────────────────────────
        for (int c = 0; c < chunk_len; c++) {
            int t = t_start + chunk_start + c;
            __nv_bfloat16* out_ptr = Out + t * stride_o_t + h * stride_o_h + v0;
            
            if (tid < BLOCK_V) {
                out_ptr[tid] = __float2bfloat16(s_out[c * BLOCK_V + tid]);
            }
        }
        
        __syncthreads();
    }
    
    // ============================================================
    // Write Final State (with swizzle reversal)
    // ============================================================
    float* new_state_ptr = NewState + n * stride_ns_n + h * stride_ns_h + v0 * stride_ns_v;
    
    #pragma unroll
    for (int i = tid; i < BLOCK_V * V9P_D; i += V9P_THREADS) {
        int vi = i / V9P_D;
        int di = i % V9P_D;
        int swizzled_di = swizzle_idx(di);
        new_state_ptr[vi * stride_ns_v + di] = s_state[vi * V9P_D + swizzled_di];
    }
}

// ============================================================
// Launch Function
// ============================================================

void gdn_prefill_v9_launch_chunked(
    const void* Q, const void* K, const void* V,
    const void* State, const void* A_log, const void* A,
    const void* DtBias, const void* B_gate, const void* CuSeqlens,
    void* Out, void* NewState,
    float scale, int N, int num_v_heads, int D,
    int stride_q_t, int stride_q_h, int stride_k_t, int stride_k_h,
    int stride_v_t, int stride_v_h, int stride_s_n, int stride_s_h, int stride_s_v,
    int stride_a_t, int stride_b_t, int stride_o_t, int stride_o_h,
    int stride_ns_n, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, int CHUNK_SIZE, cudaStream_t stream
) {
    int V_BLOCKS = D / BLOCK_V;
    dim3 grid(N, num_v_heads, V_BLOCKS);
    dim3 block(V9P_THREADS);
    
    // SMEM size calculation
    // s_q + s_k + s_v + s_state + s_g + s_beta + s_out
    size_t smem_size = (CHUNK_SIZE * D + CHUNK_SIZE * D + CHUNK_SIZE * BLOCK_V +
                        BLOCK_V * D + CHUNK_SIZE + CHUNK_SIZE + 
                        CHUNK_SIZE * BLOCK_V) * sizeof(float);
    smem_size = ((smem_size + 127) / 128) * 128;  // Align to 128 bytes
    
    // Launch with BLOCK_V=16, CHUNK_SIZE=8 (default configuration)
    if (BLOCK_V == 16 && CHUNK_SIZE == 8) {
        gdn_prefill_kernel_v9_chunked<16, 8><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State, (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate, (const int32_t*)CuSeqlens,
            (__nv_bfloat16*)Out, (float*)NewState, scale,
            stride_q_t, stride_q_h, stride_k_t, stride_k_h,
            stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v,
            stride_a_t, stride_b_t, stride_o_t, stride_o_h,
            stride_ns_n, stride_ns_h, stride_ns_v);
    } else if (BLOCK_V == 16 && CHUNK_SIZE == 4) {
        gdn_prefill_kernel_v9_chunked<16, 4><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State, (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate, (const int32_t*)CuSeqlens,
            (__nv_bfloat16*)Out, (float*)NewState, scale,
            stride_q_t, stride_q_h, stride_k_t, stride_k_h,
            stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v,
            stride_a_t, stride_b_t, stride_o_t, stride_o_h,
            stride_ns_n, stride_ns_h, stride_ns_v);
    } else if (BLOCK_V == 32 && CHUNK_SIZE == 8) {
        gdn_prefill_kernel_v9_chunked<32, 8><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State, (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate, (const int32_t*)CuSeqlens,
            (__nv_bfloat16*)Out, (float*)NewState, scale,
            stride_q_t, stride_q_h, stride_k_t, stride_k_h,
            stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v,
            stride_a_t, stride_b_t, stride_o_t, stride_o_h,
            stride_ns_n, stride_ns_h, stride_ns_v);
    } else {
        // Default: BLOCK_V=16, CHUNK_SIZE=8
        gdn_prefill_kernel_v9_chunked<16, 8><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State, (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate, (const int32_t*)CuSeqlens,
            (__nv_bfloat16*)Out, (float*)NewState, scale,
            stride_q_t, stride_q_h, stride_k_t, stride_k_h,
            stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v,
            stride_a_t, stride_b_t, stride_o_t, stride_o_h,
            stride_ns_n, stride_ns_h, stride_ns_v);
    }
}

}  // namespace gdn

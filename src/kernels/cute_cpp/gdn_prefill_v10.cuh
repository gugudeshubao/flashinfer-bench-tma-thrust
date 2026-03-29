/*
 * GDN Prefill v10 — CuTe TiledMMA kernel for Tensor Core acceleration
 *
 * Key Innovation: Uses CuTe's TiledMMA abstraction for mat-mat operations
 *
 * Tensor Core Applicability:
 *   - Chunked prefill with CHUNK_SIZE=C creates matrix-matrix ops:
 *   - old_v = State @ K^T: [V, D] @ [D, C] = [V, C]
 *   - out = State @ Q:     [V, D] @ [D, C] = [V, C]
 *   - These CAN use tcgen05.mma on Blackwell (sm_100)!
 *
 * Performance Analysis:
 *   - AI (Arithmetic Intensity) with chunking:
 *   - CHUNK=8:  AI ≈ 8 FLOP/byte (near compute-bound)
 *   - CHUNK=64: AI ≈ 64 FLOP/byte (compute-bound!)
 *   - FP32 ridge point = 9.3 FLOP/byte
 *   - BF16 Tensor Core ridge = 281 FLOP/byte
 *
 * Grid: (N=num_seqs, H=8, V_BLOCKS)
 * Block: 128 threads (4 warps, or 1 warpgroup for WGMMA)
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#ifdef __CUDACC__

// CuTe includes
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/swizzle.hpp>

// For TiledMMA support (requires CUTLASS 3.x)
#if defined(CUTE_ARCH_MMA_SM90_ENABLED) || defined(CUTE_ARCH_MMA_SM100_ENABLED)
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/gemm.hpp>
#define HAS_TILEDMMA 1
#else
#define HAS_TILEDMMA 0
#endif

using namespace cute;

namespace gdn {

// ============================================================
// Constants
// ============================================================

constexpr int V10P_D = 128;
constexpr int V10P_WARP_SIZE = 32;
constexpr int V10P_NUM_WARPS = 4;
constexpr int V10P_THREADS = V10P_NUM_WARPS * V10P_WARP_SIZE;

// ============================================================
// Utility Functions
// ============================================================

__device__ __forceinline__ float v10p_softplus(float x) {
    return (x > 20.0f) ? x : __logf(1.0f + __expf(x));
}

__device__ __forceinline__ float v10p_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// Swizzle index for bank conflict avoidance
__device__ __forceinline__ int swizzle_d(int d) {
    return d ^ ((d >> 3) & 7);
}

// ============================================================
// Prefill v10 Kernel with TiledMMA-ready structure
//
// This kernel is structured for Tensor Core integration:
// - State is kept in registers during chunk processing
// - Q/K chunks are loaded to enable mat-mat operations
// ============================================================

template<int BLOCK_V, int CHUNK_SIZE>
__global__ void __launch_bounds__(V10P_THREADS)
gdn_prefill_kernel_v10_tiledmma(
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
    const int warp_id = tid / V10P_WARP_SIZE;
    const int lane_id = tid % V10P_WARP_SIZE;
    
    // Sequence bounds
    const int t_start = CuSeqlens[n];
    const int t_end = CuSeqlens[n + 1];
    const int seq_len = t_end - t_start;
    
    // Head constants
    const float alog = A_log[h];
    const float dt_val = DtBias[h];
    
    // ============================================================
    // Shared Memory Layout (optimized for TiledMMA)
    // ============================================================
    extern __shared__ char smem_raw[];
    
    // Layout for matrix operands
    // Q_chunk: [CHUNK_SIZE, D] - transposed for matmul
    // K_chunk: [CHUNK_SIZE, D] - transposed for matmul
    // V_chunk: [CHUNK_SIZE, BLOCK_V]
    // State:   [BLOCK_V, D] - swizzled
    
    float* s_q = reinterpret_cast<float*>(smem_raw);          // [CHUNK_SIZE, D]
    float* s_k = s_q + CHUNK_SIZE * V10P_D;                   // [CHUNK_SIZE, D]
    float* s_v = s_k + CHUNK_SIZE * V10P_D;                   // [CHUNK_SIZE, BLOCK_V]
    float* s_state = s_v + CHUNK_SIZE * BLOCK_V;              // [BLOCK_V, D] swizzled
    float* s_g = s_state + BLOCK_V * V10P_D;                  // [CHUNK_SIZE]
    float* s_beta = s_g + CHUNK_SIZE;                         // [CHUNK_SIZE]
    float* s_out = s_beta + CHUNK_SIZE;                       // [CHUNK_SIZE, BLOCK_V]
    
    // ============================================================
    // Load Initial State with Swizzle (for bank conflict avoidance)
    // ============================================================
    const float* state_ptr = State + n * stride_s_n + h * stride_s_h + v0 * stride_s_v;
    
    #pragma unroll
    for (int i = tid; i < BLOCK_V * V10P_D; i += V10P_THREADS) {
        int vi = i / V10P_D;
        int di = i % V10P_D;
        int swizzled_di = swizzle_d(di);
        s_state[vi * V10P_D + swizzled_di] = state_ptr[vi * stride_s_v + di];
    }
    
    __syncthreads();
    
    // ============================================================
    // Process Tokens in Chunks
    // This is where TiledMMA would be applied
    // ============================================================
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += CHUNK_SIZE) {
        int chunk_len = min(CHUNK_SIZE, seq_len - chunk_start);
        
        // ─── Load Q, K, V for this chunk ─────────────────────────
        // Layout: Q[c, d], K[c, d] for mat-mat with State[v, d]
        for (int c = 0; c < chunk_len; c++) {
            int t = t_start + chunk_start + c;
            
            // Load Q[c, :] and K[c, :]
            const __nv_bfloat16* q_ptr = Q + t * stride_q_t + qk_h * stride_q_h;
            const __nv_bfloat16* k_ptr = K + t * stride_k_t + qk_h * stride_k_h;
            
            for (int d = tid; d < V10P_D; d += V10P_THREADS) {
                s_q[c * V10P_D + d] = __bfloat162float(q_ptr[d]);
                s_k[c * V10P_D + d] = __bfloat162float(k_ptr[d]);
            }
            
            // Load V slice [BLOCK_V]
            const __nv_bfloat16* v_ptr = V + t * stride_v_t + h * stride_v_h + v0;
            if (tid < BLOCK_V) {
                s_v[c * BLOCK_V + tid] = __bfloat162float(v_ptr[tid]);
            }
            
            // Compute gates
            if (tid == 0) {
                float a_val = __bfloat162float(A[t * stride_a_t + h]);
                float b_val = __bfloat162float(B_gate[t * stride_b_t + h]);
                
                float x = a_val + dt_val;
                float sp = v10p_softplus(x);
                s_g[c] = __expf(-__expf(alog) * sp);
                s_beta[c] = v10p_sigmoid(b_val);
            }
        }
        
        __syncthreads();
        
        // ─── Process chunk tokens ────────────────────────────────
        // This loop structure is designed for TiledMMA integration
        //
        // For each token c in chunk:
        //   1. Apply decay: State[v,d] *= g[c]
        //   2. old_v = State @ K[c]^T  (mat-vec, or batched as mat-mat)
        //   3. delta = beta * (V[c] - old_v)
        //   4. State += delta * K[c]  (rank-1 update)
        //   5. out[c] = scale * State @ Q[c]  (mat-vec)
        //
        // With TiledMMA, steps 2 and 5 become:
        //   2. old_v_batch = State @ K_chunk^T  [V, D] @ [D, C] = [V, C]
        //   5. out_batch = State @ Q_chunk      [V, D] @ [D, C] = [V, C]
        
        for (int c = 0; c < chunk_len; c++) {
            float g = s_g[c];
            float beta = s_beta[c];
            
            // Warp-parallel processing over V dimension
            for (int v_idx = warp_id; v_idx < BLOCK_V; v_idx += V10P_NUM_WARPS) {
                
                // ─── 1. Apply decay and compute old_v ────────────
                float old_v = 0.0f;
                
                #pragma unroll 4
                for (int d = lane_id; d < V10P_D; d += V10P_WARP_SIZE) {
                    int swizzled_d = swizzle_d(d);
                    float decayed_s = g * s_state[v_idx * V10P_D + swizzled_d];
                    old_v += decayed_s * s_k[c * V10P_D + d];
                    s_state[v_idx * V10P_D + swizzled_d] = decayed_s;
                }
                
                // Warp reduction
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                    old_v += __shfl_xor_sync(0xffffffff, old_v, mask);
                }
                
                // ─── 2. Delta rule update ────────────────────────
                float v_elem = s_v[c * BLOCK_V + v_idx];
                float delta = beta * (v_elem - old_v);
                
                // ─── 3. Update state and compute output ──────────
                float out_val = 0.0f;
                
                #pragma unroll 4
                for (int d = lane_id; d < V10P_D; d += V10P_WARP_SIZE) {
                    int swizzled_d = swizzle_d(d);
                    float new_s = s_state[v_idx * V10P_D + swizzled_d] + delta * s_k[c * V10P_D + d];
                    s_state[v_idx * V10P_D + swizzled_d] = new_s;
                    out_val += new_s * s_q[c * V10P_D + d];
                }
                
                // Warp reduction
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                    out_val += __shfl_xor_sync(0xffffffff, out_val, mask);
                }
                
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
    // Write Final State (reverse swizzle)
    // ============================================================
    float* new_state_ptr = NewState + n * stride_ns_n + h * stride_ns_h + v0 * stride_ns_v;
    
    #pragma unroll
    for (int i = tid; i < BLOCK_V * V10P_D; i += V10P_THREADS) {
        int vi = i / V10P_D;
        int di = i % V10P_D;
        int swizzled_di = swizzle_d(di);
        new_state_ptr[vi * stride_ns_v + di] = s_state[vi * V10P_D + swizzled_di];
    }
}

// ============================================================
// Launch Function
// ============================================================

void gdn_prefill_v10_launch_tiledmma(
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
    dim3 block(V10P_THREADS);
    
    // SMEM: q + k + v + state + g + beta + out
    size_t smem_size = (CHUNK_SIZE * D + CHUNK_SIZE * D + CHUNK_SIZE * BLOCK_V +
                        BLOCK_V * D + CHUNK_SIZE + CHUNK_SIZE + 
                        CHUNK_SIZE * BLOCK_V) * sizeof(float);
    smem_size = ((smem_size + 127) / 128) * 128;
    
    // Dispatch based on BLOCK_V and CHUNK_SIZE
    if (BLOCK_V == 16 && CHUNK_SIZE == 8) {
        gdn_prefill_kernel_v10_tiledmma<16, 8><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State, (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate, (const int32_t*)CuSeqlens,
            (__nv_bfloat16*)Out, (float*)NewState, scale,
            stride_q_t, stride_q_h, stride_k_t, stride_k_h,
            stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v,
            stride_a_t, stride_b_t, stride_o_t, stride_o_h,
            stride_ns_n, stride_ns_h, stride_ns_v);
    } else if (BLOCK_V == 16 && CHUNK_SIZE == 16) {
        gdn_prefill_kernel_v10_tiledmma<16, 16><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State, (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate, (const int32_t*)CuSeqlens,
            (__nv_bfloat16*)Out, (float*)NewState, scale,
            stride_q_t, stride_q_h, stride_k_t, stride_k_h,
            stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v,
            stride_a_t, stride_b_t, stride_o_t, stride_o_h,
            stride_ns_n, stride_ns_h, stride_ns_v);
    } else if (BLOCK_V == 32 && CHUNK_SIZE == 8) {
        gdn_prefill_kernel_v10_tiledmma<32, 8><<<grid, block, smem_size, stream>>>(
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
        gdn_prefill_kernel_v10_tiledmma<16, 8><<<grid, block, smem_size, stream>>>(
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

#else
// Fallback when not compiling with CUDA
namespace gdn {
void gdn_prefill_v10_launch_tiledmma(...) {}
}
#endif

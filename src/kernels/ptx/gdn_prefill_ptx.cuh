/*
 * GDN Prefill — CUDA C++ with Embedded PTX Assembly
 *
 * Optimized prefill kernel using inline PTX for:
 *   - Fast math (ex2.approx, lg2.approx, rcp.approx)
 *   - FMA operations (fma.rn.f32)
 *   - Memory operations with cache hints (ld.global.nc)
 *   - Predicated execution (selp)
 *   - Warp shuffle for reductions
 *
 * Key Optimizations:
 *   1. Chunk-based processing (CHUNK_SIZE tokens at once)
 *   2. PTX fast math for gates (exp, log, sigmoid)
 *   3. FMA chains for dot products
 *   4. Prefetch hints for state access
 *
 * Grid: (N=num_seqs, H=8, V_BLOCKS)
 * Block: 128 threads (4 warps)
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace gdn_ptx {

// ============================================================
// PTX Inline Assembly Primitives (shared with decode)
// ============================================================

// Fast approximate exp2 (2^x)
__device__ __forceinline__ float ptx_exp2_pf(float x) {
    float result;
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

// Fast approximate log2
__device__ __forceinline__ float ptx_log2_pf(float x) {
    float result;
    asm volatile("lg2.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

// Fast approximate reciprocal
__device__ __forceinline__ float ptx_rcp_pf(float x) {
    float result;
    asm volatile("rcp.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

// Fused multiply-add
__device__ __forceinline__ float ptx_fma_pf(float a, float b, float c) {
    float result;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
}

// Non-coherent load
__device__ __forceinline__ float ptx_ld_nc_pf(const float* ptr) {
    float result;
    asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(result) : "l"(ptr));
    return result;
}

// Store with write-back
__device__ __forceinline__ void ptx_st_wb_pf(float* ptr, float val) {
    asm volatile("st.global.wb.f32 [%0], %1;" :: "l"(ptr), "f"(val));
}

// Predicated select
__device__ __forceinline__ float ptx_selp_pf(float a, float b, bool pred) {
    float result;
    asm volatile("selp.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "r"((int)pred));
    return result;
}

// ============================================================
// Derived Math Functions using PTX
// ============================================================

// exp(x) = 2^(x * log2(e))
__device__ __forceinline__ float ptx_exp_pf(float x) {
    constexpr float LOG2E = 1.4426950408889634f;
    return ptx_exp2_pf(x * LOG2E);
}

// log(x) = log2(x) * ln(2)
__device__ __forceinline__ float ptx_log_pf(float x) {
    constexpr float LN2 = 0.6931471805599453f;
    return ptx_log2_pf(x) * LN2;
}

// Softplus: log(1 + exp(x)), branchless
__device__ __forceinline__ float ptx_softplus_pf(float x) {
    float exp_x = ptx_exp_pf(x);
    float log_result = ptx_log_pf(1.0f + exp_x);
    return ptx_selp_pf(x, log_result, x > 20.0f);
}

// Sigmoid: 1 / (1 + exp(-x))
__device__ __forceinline__ float ptx_sigmoid_pf(float x) {
    float exp_neg_x = ptx_exp_pf(-x);
    return ptx_rcp_pf(1.0f + exp_neg_x);
}

// ============================================================
// Constants
// ============================================================

constexpr int PREFILL_D_PTX = 128;
constexpr int PREFILL_WARP_SIZE_PTX = 32;

// ============================================================
// PTX Optimized Prefill Kernel with Chunking
// ============================================================

template<int BLOCK_V, int CHUNK_SIZE>
__global__ void gdn_prefill_kernel_ptx_chunked(
    // Inputs
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ State,
    // Gates
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ A,
    const float* __restrict__ DtBias,
    const __nv_bfloat16* __restrict__ B_gate,
    // Sequence info
    const int32_t* __restrict__ CuSeqlens,
    // Outputs
    __nv_bfloat16* __restrict__ Out,
    float* __restrict__ NewState,
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
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const int vb = blockIdx.z;
    const int v0 = vb * BLOCK_V;
    const int qk_h = h / 2;
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // Sequence bounds
    const int t_start = CuSeqlens[n];
    const int t_end = CuSeqlens[n + 1];
    const int seq_len = t_end - t_start;
    
    // Head constants (use PTX load)
    const float alog = ptx_ld_nc_pf(&A_log[h]);
    const float dt_val = DtBias[h];
    
    // Shared memory
    extern __shared__ char smem_raw[];
    float* q_smem = (float*)smem_raw;
    float* k_smem = q_smem + CHUNK_SIZE * PREFILL_D_PTX;
    float* v_smem = k_smem + CHUNK_SIZE * PREFILL_D_PTX;
    float* state_smem = v_smem + CHUNK_SIZE * BLOCK_V;
    float* g_smem = state_smem + BLOCK_V * PREFILL_D_PTX;
    float* beta_smem = g_smem + CHUNK_SIZE;
    float* old_v_smem = beta_smem + CHUNK_SIZE;
    float* out_smem = old_v_smem + BLOCK_V;
    
    // ─── Load initial state with PTX ───────────────────────────────────
    const float* state_ptr = State + n * stride_s_n + h * stride_s_h + v0 * stride_s_v;
    
    for (int i = tid; i < BLOCK_V * PREFILL_D_PTX; i += num_threads) {
        int vi = i / PREFILL_D_PTX;
        int ki = i % PREFILL_D_PTX;
        state_smem[i] = ptx_ld_nc_pf(&state_ptr[vi * stride_s_v + ki]);
    }
    __syncthreads();
    
    // ─── Process tokens in chunks ──────────────────────────────────────
    int num_chunks = (seq_len + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int chunk_start = chunk * CHUNK_SIZE;
        int chunk_end = min(chunk_start + CHUNK_SIZE, seq_len);
        int actual_chunk_size = chunk_end - chunk_start;
        
        // ── Load chunk data ──────────────────────────────────────────
        for (int c = 0; c < actual_chunk_size; c++) {
            int t = t_start + chunk_start + c;
            
            // Load Q[c], K[c]
            for (int i = tid; i < PREFILL_D_PTX; i += num_threads) {
                q_smem[c * PREFILL_D_PTX + i] = __bfloat162float(Q[t * stride_q_t + qk_h * stride_q_h + i]);
                k_smem[c * PREFILL_D_PTX + i] = __bfloat162float(K[t * stride_k_t + qk_h * stride_k_h + i]);
            }
            
            // Load V slice
            for (int i = tid; i < BLOCK_V; i += num_threads) {
                v_smem[c * BLOCK_V + i] = __bfloat162float(V[t * stride_v_t + h * stride_v_h + v0 + i]);
            }
            
            // Compute gates using PTX fast math
            if (tid == c) {
                float a_val = __bfloat162float(A[t * stride_a_t + h]);
                float b_val = __bfloat162float(B_gate[t * stride_b_t + h]);
                
                // PTX softplus and sigmoid
                float sp = ptx_softplus_pf(a_val + dt_val);
                float exp_alog = ptx_exp_pf(alog);
                g_smem[c] = ptx_exp_pf(-exp_alog * sp);
                beta_smem[c] = ptx_sigmoid_pf(b_val);
            }
        }
        __syncthreads();
        
        // ── Process chunk with PTX FMA ───────────────────────────────
        for (int c = 0; c < actual_chunk_size; c++) {
            float g = g_smem[c];
            float beta = beta_smem[c];
            
            // Apply gate decay: S = g * S (using FMA: g*S + 0)
            for (int i = tid; i < BLOCK_V * PREFILL_D_PTX; i += num_threads) {
                state_smem[i] = ptx_fma_pf(g, state_smem[i], 0.0f);
            }
            __syncthreads();
            
            // Compute old_v = S @ k[c] using FMA chain
            if (tid < BLOCK_V) {
                float sum = 0.0f;
                const float* k_ptr = k_smem + c * PREFILL_D_PTX;
                
                // Unrolled FMA chain for maximum throughput
                #pragma unroll 8
                for (int ki = 0; ki < PREFILL_D_PTX; ki += 4) {
                    sum = ptx_fma_pf(state_smem[tid * PREFILL_D_PTX + ki + 0], k_ptr[ki + 0], sum);
                    sum = ptx_fma_pf(state_smem[tid * PREFILL_D_PTX + ki + 1], k_ptr[ki + 1], sum);
                    sum = ptx_fma_pf(state_smem[tid * PREFILL_D_PTX + ki + 2], k_ptr[ki + 2], sum);
                    sum = ptx_fma_pf(state_smem[tid * PREFILL_D_PTX + ki + 3], k_ptr[ki + 3], sum);
                }
                old_v_smem[tid] = sum;
            }
            __syncthreads();
            
            // Rank-1 update: S += delta * k^T (using FMA)
            const float* k_ptr = k_smem + c * PREFILL_D_PTX;
            const float* v_ptr = v_smem + c * BLOCK_V;
            
            for (int i = tid; i < BLOCK_V * PREFILL_D_PTX; i += num_threads) {
                int vi = i / PREFILL_D_PTX;
                int ki = i % PREFILL_D_PTX;
                float delta = beta * (v_ptr[vi] - old_v_smem[vi]);
                state_smem[i] = ptx_fma_pf(delta, k_ptr[ki], state_smem[i]);
            }
            __syncthreads();
            
            // Compute out = scale * S @ q[c] using FMA chain
            if (tid < BLOCK_V) {
                float sum = 0.0f;
                const float* q_ptr = q_smem + c * PREFILL_D_PTX;
                
                #pragma unroll 8
                for (int ki = 0; ki < PREFILL_D_PTX; ki += 4) {
                    sum = ptx_fma_pf(state_smem[tid * PREFILL_D_PTX + ki + 0], q_ptr[ki + 0], sum);
                    sum = ptx_fma_pf(state_smem[tid * PREFILL_D_PTX + ki + 1], q_ptr[ki + 1], sum);
                    sum = ptx_fma_pf(state_smem[tid * PREFILL_D_PTX + ki + 2], q_ptr[ki + 2], sum);
                    sum = ptx_fma_pf(state_smem[tid * PREFILL_D_PTX + ki + 3], q_ptr[ki + 3], sum);
                }
                out_smem[c * BLOCK_V + tid] = scale * sum;
            }
            __syncthreads();
        }
        
        // ── Store outputs for chunk ──────────────────────────────────
        for (int c = 0; c < actual_chunk_size; c++) {
            int t = t_start + chunk_start + c;
            for (int i = tid; i < BLOCK_V; i += num_threads) {
                Out[t * stride_o_t + h * stride_o_h + v0 + i] = 
                    __float2bfloat16(out_smem[c * BLOCK_V + i]);
            }
        }
        __syncthreads();
    }
    
    // ─── Store final state with PTX ────────────────────────────────────
    float* new_state_ptr = NewState + n * stride_ns_n + h * stride_ns_h + v0 * stride_ns_v;
    
    for (int i = tid; i < BLOCK_V * PREFILL_D_PTX; i += num_threads) {
        int vi = i / PREFILL_D_PTX;
        int ki = i % PREFILL_D_PTX;
        ptx_st_wb_pf(&new_state_ptr[vi * stride_ns_v + ki], state_smem[i]);
    }
}

// ============================================================
// Launcher Function
// ============================================================

inline void gdn_prefill_ptx_launch(
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
    int CHUNK_SIZE,
    cudaStream_t stream
) {
    int V_BLOCKS = D / BLOCK_V;
    dim3 grid(N, num_v_heads, V_BLOCKS);
    dim3 block(128);
    
    size_t smem_size = (CHUNK_SIZE * D + CHUNK_SIZE * D + CHUNK_SIZE * BLOCK_V + 
                        BLOCK_V * D + CHUNK_SIZE + CHUNK_SIZE + BLOCK_V + 
                        CHUNK_SIZE * BLOCK_V) * sizeof(float);
    
    #define LAUNCH_PTX_KERNEL(BV, CS) \
        gdn_prefill_kernel_ptx_chunked<BV, CS><<<grid, block, smem_size, stream>>>( \
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V, \
            (const float*)State, (const float*)A_log, (const __nv_bfloat16*)A, \
            (const float*)DtBias, (const __nv_bfloat16*)B_gate, (const int32_t*)CuSeqlens, \
            (__nv_bfloat16*)Out, (float*)NewState, scale, \
            stride_q_t, stride_q_h, stride_k_t, stride_k_h, \
            stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v, \
            stride_a_t, stride_b_t, stride_o_t, stride_o_h, \
            stride_ns_n, stride_ns_h, stride_ns_v)
    
    if (BLOCK_V == 16) {
        if (CHUNK_SIZE == 4) { LAUNCH_PTX_KERNEL(16, 4); }
        else if (CHUNK_SIZE == 8) { LAUNCH_PTX_KERNEL(16, 8); }
        else { LAUNCH_PTX_KERNEL(16, 4); }
    } else {
        if (CHUNK_SIZE == 4) { LAUNCH_PTX_KERNEL(32, 4); }
        else if (CHUNK_SIZE == 8) { LAUNCH_PTX_KERNEL(32, 8); }
        else { LAUNCH_PTX_KERNEL(32, 4); }
    }
    
    #undef LAUNCH_PTX_KERNEL
}

}  // namespace gdn_ptx

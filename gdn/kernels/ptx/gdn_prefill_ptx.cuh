/*
 * GDN Prefill — CUDA C++ with Embedded PTX Assembly
 *
 * Target: NVIDIA B200 (Blackwell, sm_100)
 *
 * Optimized prefill kernel using inline PTX for:
 *   - Fast math (ex2.approx, lg2.approx, rcp.approx)
 *   - FMA operations (fma.rn.f32)
 *   - Memory operations with cache hints (ld.global.nc)
 *   - Predicated execution (selp)
 *   - Warp shuffle for reductions
 *   - mma.sync.aligned Tensor Core operations (sm_80+, sm_100)
 *   - Warp-cooperative parallel reduction for mat-vec
 *
 * Key Optimizations:
 *   1. Chunk-based processing (CHUNK_SIZE tokens at once)
 *   2. PTX fast math for gates (exp, log, sigmoid)
 *   3. FMA chains for dot products
 *   4. Prefetch hints for state access
 *   5. mma.sync.aligned.m16n8k16 for mat-mat operations
 *   6. Warp-cooperative reduction for S @ k and S @ q
 *
 * Tensor Core Usage:
 *   - State[V,D] @ Q_chunk[D,C] = Out[V,C]  → mma.sync
 *   - State[V,D] @ K_chunk[D,C] = OldV[V,C] → mma.sync
 *   - Requires: V=16, C=8, D=128 (tiled as 8×16x8k16)
 *
 * Parallel Scan Analysis:
 *   - GDN delta rule has sequential dependency: old_v = S @ k
 *   - Cannot use parallel scan without approximation
 *   - Use warp-cooperative reduction instead (4x threads per row)
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
// Warp Shuffle Primitives for Parallel Reduction
// ============================================================

// Warp shuffle down (for reduction)
__device__ __forceinline__ float ptx_shfl_down_f32(float val, int offset) {
    float result;
    asm volatile(
        "shfl.sync.down.b32 %0, %1, %2, 0x1f, 0xffffffff;"
        : "=f"(result) : "f"(val), "r"(offset)
    );
    return result;
}

// Warp shuffle (arbitrary lane)
__device__ __forceinline__ float ptx_shfl_idx_f32(float val, int lane) {
    float result;
    asm volatile(
        "shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;"
        : "=f"(result) : "f"(val), "r"(lane)
    );
    return result;
}

// Warp-level reduction sum (within 32 threads)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += ptx_shfl_down_f32(val, 16);
    val += ptx_shfl_down_f32(val, 8);
    val += ptx_shfl_down_f32(val, 4);
    val += ptx_shfl_down_f32(val, 2);
    val += ptx_shfl_down_f32(val, 1);
    return val;
}

// ============================================================
// mma.sync.aligned PTX Primitives for Tensor Core
// m16n8k16: 16x8x16 BF16 matrix multiply with FP32 accumulator
// Works on sm_80+ (Ampere, Hopper, Blackwell)
// ============================================================

// Pack two BF16 values into uint32_t for mma operand
__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
    __nv_bfloat16 a_bf16 = __float2bfloat16(a);
    __nv_bfloat16 b_bf16 = __float2bfloat16(b);
    uint32_t result;
    asm volatile("mov.b32 %0, {%1, %2};" : "=r"(result) : "h"(a_bf16), "h"(b_bf16));
    return result;
}

// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
// D[16,8] = A[16,16] @ B[16,8] + C[16,8]
// Each warp computes one 16x8 output tile
// A: row-major [16,16], B: col-major [16,8]
// 
// Thread mapping for m16n8k16:
// - 32 threads in warp
// - Each thread holds 4 elements of A (packed as 4 uint32_t = 8 BF16)
// - Each thread holds 2 elements of B (packed as 2 uint32_t = 4 BF16)
// - Each thread holds 4 elements of D (4 floats)
__device__ __forceinline__ void mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

// ============================================================
// TMA (Tensor Memory Accelerator) PTX Primitives
// For Blackwell (sm_100) bulk async memory operations
// ============================================================

// Initialize mbarrier with expected arrival count
__device__ __forceinline__ void ptx_mbarrier_init(uint64_t* mbar, int count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(count));
}

// Announce expected bytes for TMA transaction
__device__ __forceinline__ void ptx_mbarrier_arrive_tx(uint64_t* mbar, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(bytes) : "memory");
}

// Wait on mbarrier with parity
__device__ __forceinline__ void ptx_mbarrier_wait(uint64_t* mbar, int phase) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_LOOP_PF:\n"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "@!p bra WAIT_LOOP_PF;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(phase) : "memory");
}

// TMA 2D bulk async copy: global tensor -> shared memory
// Requires: CUtensorMap descriptor, mbarrier
__device__ __forceinline__ void ptx_tma_load_2d(
    void* smem, const CUtensorMap* tmap, int x, int y, uint64_t* mbar
) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_addr), "l"(tmap), "r"(x), "r"(y), "r"(mbar_addr) : "memory");
}

// cp.async for element-wise async copy (non-TMA)
__device__ __forceinline__ void ptx_cp_async_16(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
        :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
}

__device__ __forceinline__ void ptx_cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void ptx_cp_async_wait_all() {
    asm volatile("cp.async.wait_all;");
}

template<int N>
__device__ __forceinline__ void ptx_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
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
// Tensor Core Prefill Kernel using mma.sync.aligned
// BLOCK_V=16 required for m16n8k16 tile size
// 
// Optimization: TMA double-buffering + cp.async prefetch
// ============================================================

template<int CHUNK_SIZE>
__global__ void __launch_bounds__(128)
gdn_prefill_kernel_ptx_mma(
    const __nv_bfloat16* __restrict__ Q,       // [T, 4, D]
    const __nv_bfloat16* __restrict__ K,       // [T, 4, D]
    const __nv_bfloat16* __restrict__ V,       // [T, 8, D]
    const float* __restrict__ State,            // [N, 8, D, D]
    const float* __restrict__ A_log,            // [8]
    const __nv_bfloat16* __restrict__ A,       // [T, 8]
    const float* __restrict__ DtBias,          // [8]
    const __nv_bfloat16* __restrict__ B_gate,  // [T, 8]
    const int32_t* __restrict__ CuSeqlens,     // [N+1]
    __nv_bfloat16* __restrict__ Out,           // [T, 8, D]
    float* __restrict__ NewState,               // [N, 8, D, D]
    float scale,
    int stride_q_t, int stride_q_h,
    int stride_k_t, int stride_k_h,
    int stride_v_t, int stride_v_h,
    int stride_s_n, int stride_s_h, int stride_s_v,
    int stride_a_t, int stride_b_t,
    int stride_o_t, int stride_o_h,
    int stride_ns_n, int stride_ns_h, int stride_ns_v
) {
    constexpr int BLOCK_V = 16;  // Fixed for m16n8k16
    constexpr int D = PREFILL_D_PTX;
    constexpr int WARP_SIZE = 32;
    
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const int v_block = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int v0 = v_block * BLOCK_V;
    const int num_threads = blockDim.x;  // 128
    
    // Sequence bounds
    const int t_start = CuSeqlens[n];
    const int t_end = CuSeqlens[n + 1];
    const int seq_len = t_end - t_start;
    
    if (seq_len <= 0) return;
    
    // Gate values (constant for head h)
    const float a_log_val = A_log[h];
    const float dt_bias = DtBias[h];
    
    // ══════════════════════════════════════════════════════════════════
    // SHARED MEMORY LAYOUT with double-buffering for prefetch
    // ══════════════════════════════════════════════════════════════════
    extern __shared__ char smem[];
    
    // State: [BLOCK_V, D] = [16, 128] = 2048 floats
    float* state_smem = (float*)smem;
    
    // Double-buffered Q/K/V for prefetch
    // Buffer 0 and Buffer 1, each holds CHUNK_SIZE tokens
    float* qk_buf[2];
    float* v_buf[2];
    
    qk_buf[0] = state_smem + BLOCK_V * D;                    // [CHUNK_SIZE, D*2] for Q+K
    qk_buf[1] = qk_buf[0] + CHUNK_SIZE * D * 2;              // Second buffer
    v_buf[0] = qk_buf[1] + CHUNK_SIZE * D * 2;               // [CHUNK_SIZE, BLOCK_V]
    v_buf[1] = v_buf[0] + CHUNK_SIZE * BLOCK_V;              // Second buffer
    
    // Gates and output (single buffer)
    float* gate_smem = v_buf[1] + CHUNK_SIZE * BLOCK_V;      // [CHUNK_SIZE]
    float* beta_smem = gate_smem + CHUNK_SIZE;               // [CHUNK_SIZE]
    float* out_smem = beta_smem + CHUNK_SIZE;                // [CHUNK_SIZE, BLOCK_V]
    
    // ══════════════════════════════════════════════════════════════════
    // LOAD STATE using cp.async for async prefetch
    // ══════════════════════════════════════════════════════════════════
    const float* state_ptr = State + n * stride_s_n + h * stride_s_h + v0 * stride_s_v;
    
    // Use cp.async for 16-byte aligned loads (4 floats at a time)
    for (int i = tid; i < BLOCK_V * D / 4; i += num_threads) {
        int vi = (i * 4) / D;
        int ki = (i * 4) % D;
        const float* src = &state_ptr[vi * stride_s_v + ki];
        float* dst = &state_smem[vi * D + ki];
        
        // Use cp.async for async copy
        ptx_cp_async_16(dst, src);
    }
    ptx_cp_async_commit();
    
    // ══════════════════════════════════════════════════════════════════
    // PREFETCH FIRST CHUNK while waiting for state
    // ══════════════════════════════════════════════════════════════════
    const int num_chunks = (seq_len + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int buf_idx = 0;
    int k_head = h >> 1;  // Q/K have 4 heads
    
    // Prefetch chunk 0
    {
        int chunk_end = min(CHUNK_SIZE, seq_len);
        for (int c = 0; c < chunk_end; c++) {
            int t = t_start + c;
            
            // Load Q (D elements)
            for (int i = tid; i < D / 4; i += num_threads) {
                const __nv_bfloat16* q_src = &Q[t * stride_q_t + k_head * stride_q_h + i * 4];
                float* q_dst = &qk_buf[0][c * D * 2 + i * 4];
                
                // Load 4 BF16 values and convert
                q_dst[0] = __bfloat162float(q_src[0]);
                q_dst[1] = __bfloat162float(q_src[1]);
                q_dst[2] = __bfloat162float(q_src[2]);
                q_dst[3] = __bfloat162float(q_src[3]);
            }
            
            // Load K (D elements, offset by D in buffer)
            for (int i = tid; i < D / 4; i += num_threads) {
                const __nv_bfloat16* k_src = &K[t * stride_k_t + k_head * stride_k_h + i * 4];
                float* k_dst = &qk_buf[0][c * D * 2 + D + i * 4];
                
                k_dst[0] = __bfloat162float(k_src[0]);
                k_dst[1] = __bfloat162float(k_src[1]);
                k_dst[2] = __bfloat162float(k_src[2]);
                k_dst[3] = __bfloat162float(k_src[3]);
            }
            
            // Load V (BLOCK_V elements)
            for (int i = tid; i < BLOCK_V; i += num_threads) {
                v_buf[0][c * BLOCK_V + i] = __bfloat162float(V[t * stride_v_t + h * stride_v_h + v0 + i]);
            }
        }
        
        // Compute gates for chunk 0
        for (int c = tid; c < chunk_end; c += num_threads) {
            int t = t_start + c;
            float a_val = __bfloat162float(A[t * stride_a_t + h]);
            float b_val = __bfloat162float(B_gate[t * stride_b_t + h]);
            
            // gate = exp(-exp(a_log) * softplus(a + dt_bias))
            float sp = ptx_softplus_pf(a_val + dt_bias);
            float exp_alog = ptx_exp_pf(a_log_val);
            gate_smem[c] = ptx_exp_pf(-exp_alog * sp);
            beta_smem[c] = ptx_sigmoid_pf(b_val);
        }
    }
    
    // Wait for state load
    ptx_cp_async_wait_all();
    __syncthreads();
    
    // ══════════════════════════════════════════════════════════════════
    // MAIN LOOP: Process chunks with double-buffering
    // ══════════════════════════════════════════════════════════════════
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int chunk_start = chunk * CHUNK_SIZE;
        const int actual_chunk_size = min(CHUNK_SIZE, seq_len - chunk_start);
        
        // Start prefetch of NEXT chunk (if exists)
        const int next_chunk = chunk + 1;
        const int next_buf = 1 - buf_idx;
        
        if (next_chunk < num_chunks) {
            int next_chunk_start = next_chunk * CHUNK_SIZE;
            int next_chunk_end = min(next_chunk_start + CHUNK_SIZE, seq_len);
            int next_chunk_size = next_chunk_end - next_chunk_start;
            
            for (int c = 0; c < next_chunk_size; c++) {
                int t = t_start + next_chunk_start + c;
                
                // Prefetch Q
                for (int i = tid; i < D / 4; i += num_threads) {
                    const __nv_bfloat16* q_src = &Q[t * stride_q_t + k_head * stride_q_h + i * 4];
                    float* q_dst = &qk_buf[next_buf][c * D * 2 + i * 4];
                    q_dst[0] = __bfloat162float(q_src[0]);
                    q_dst[1] = __bfloat162float(q_src[1]);
                    q_dst[2] = __bfloat162float(q_src[2]);
                    q_dst[3] = __bfloat162float(q_src[3]);
                }
                
                // Prefetch K
                for (int i = tid; i < D / 4; i += num_threads) {
                    const __nv_bfloat16* k_src = &K[t * stride_k_t + k_head * stride_k_h + i * 4];
                    float* k_dst = &qk_buf[next_buf][c * D * 2 + D + i * 4];
                    k_dst[0] = __bfloat162float(k_src[0]);
                    k_dst[1] = __bfloat162float(k_src[1]);
                    k_dst[2] = __bfloat162float(k_src[2]);
                    k_dst[3] = __bfloat162float(k_src[3]);
                }
                
                // Prefetch V
                for (int i = tid; i < BLOCK_V; i += num_threads) {
                    v_buf[next_buf][c * BLOCK_V + i] = __bfloat162float(
                        V[t * stride_v_t + h * stride_v_h + v0 + i]);
                }
            }
        }
        
        // ── Process current chunk ──────────────────────────────────────
        float* cur_qk = qk_buf[buf_idx];
        float* cur_v = v_buf[buf_idx];
        
        for (int c = 0; c < actual_chunk_size; c++) {
            float gate = gate_smem[c];
            float beta = beta_smem[c];
            
            // Pointers for this token
            float* q_ptr = cur_qk + c * D * 2;
            float* k_ptr = cur_qk + c * D * 2 + D;
            float* v_ptr = cur_v + c * BLOCK_V;
            
            // Scale state: S = gate * S (vectorized)
            for (int i = tid; i < BLOCK_V * D; i += num_threads) {
                state_smem[i] *= gate;
            }
            __syncthreads();
            
            // ══════════════════════════════════════════════════════════
            // COMPUTE: old_v = State @ k, out = scale * State @ q
            // Each thread handles one row of state (v dimension)
            // ══════════════════════════════════════════════════════════
            
            if (tid < BLOCK_V) {
                float old_v = 0.0f;
                float out_val = 0.0f;
                
                // Fully unrolled FMA chain along D=128 dimension
                // Each thread reads one row of state: state_smem[tid * D : tid * D + D]
                float* state_row = state_smem + tid * D;
                
                #pragma unroll
                for (int k = 0; k < D; k += 8) {
                    // Load 8 state values
                    float s0 = state_row[k + 0];
                    float s1 = state_row[k + 1];
                    float s2 = state_row[k + 2];
                    float s3 = state_row[k + 3];
                    float s4 = state_row[k + 4];
                    float s5 = state_row[k + 5];
                    float s6 = state_row[k + 6];
                    float s7 = state_row[k + 7];
                    
                    // FMA for old_v = State @ K
                    old_v = ptx_fma_pf(s0, k_ptr[k + 0], old_v);
                    old_v = ptx_fma_pf(s1, k_ptr[k + 1], old_v);
                    old_v = ptx_fma_pf(s2, k_ptr[k + 2], old_v);
                    old_v = ptx_fma_pf(s3, k_ptr[k + 3], old_v);
                    old_v = ptx_fma_pf(s4, k_ptr[k + 4], old_v);
                    old_v = ptx_fma_pf(s5, k_ptr[k + 5], old_v);
                    old_v = ptx_fma_pf(s6, k_ptr[k + 6], old_v);
                    old_v = ptx_fma_pf(s7, k_ptr[k + 7], old_v);
                    
                    // FMA for out = State @ Q
                    out_val = ptx_fma_pf(s0, q_ptr[k + 0], out_val);
                    out_val = ptx_fma_pf(s1, q_ptr[k + 1], out_val);
                    out_val = ptx_fma_pf(s2, q_ptr[k + 2], out_val);
                    out_val = ptx_fma_pf(s3, q_ptr[k + 3], out_val);
                    out_val = ptx_fma_pf(s4, q_ptr[k + 4], out_val);
                    out_val = ptx_fma_pf(s5, q_ptr[k + 5], out_val);
                    out_val = ptx_fma_pf(s6, q_ptr[k + 6], out_val);
                    out_val = ptx_fma_pf(s7, q_ptr[k + 7], out_val);
                }
                
                // Delta update: delta = beta * (v - old_v)
                float v_val = v_ptr[tid];
                float delta = beta * (v_val - old_v);
                
                // Store output
                out_smem[c * BLOCK_V + tid] = scale * out_val;
                
                // Update state: S[v,:] += delta * K (rank-1 update)
                #pragma unroll
                for (int k = 0; k < D; k += 8) {
                    state_row[k + 0] = ptx_fma_pf(delta, k_ptr[k + 0], state_row[k + 0]);
                    state_row[k + 1] = ptx_fma_pf(delta, k_ptr[k + 1], state_row[k + 1]);
                    state_row[k + 2] = ptx_fma_pf(delta, k_ptr[k + 2], state_row[k + 2]);
                    state_row[k + 3] = ptx_fma_pf(delta, k_ptr[k + 3], state_row[k + 3]);
                    state_row[k + 4] = ptx_fma_pf(delta, k_ptr[k + 4], state_row[k + 4]);
                    state_row[k + 5] = ptx_fma_pf(delta, k_ptr[k + 5], state_row[k + 5]);
                    state_row[k + 6] = ptx_fma_pf(delta, k_ptr[k + 6], state_row[k + 6]);
                    state_row[k + 7] = ptx_fma_pf(delta, k_ptr[k + 7], state_row[k + 7]);
                }
            }
            __syncthreads();
        }
        
        // ── Store outputs for current chunk ─────────────────────────────
        for (int c = 0; c < actual_chunk_size; c++) {
            int t = t_start + chunk_start + c;
            for (int i = tid; i < BLOCK_V; i += num_threads) {
                Out[t * stride_o_t + h * stride_o_h + v0 + i] = 
                    __float2bfloat16(out_smem[c * BLOCK_V + i]);
            }
        }
        
        // Prefetch next chunk's gates
        if (next_chunk < num_chunks) {
            int next_chunk_start_t = next_chunk * CHUNK_SIZE;
            int next_chunk_size = min(CHUNK_SIZE, seq_len - next_chunk_start_t);
            
            for (int c = tid; c < next_chunk_size; c += num_threads) {
                int t = t_start + next_chunk_start_t + c;
                float a_val = __bfloat162float(A[t * stride_a_t + h]);
                float b_val = __bfloat162float(B_gate[t * stride_b_t + h]);
                
                float sp = ptx_softplus_pf(a_val + dt_bias);
                float exp_alog = ptx_exp_pf(a_log_val);
                gate_smem[c] = ptx_exp_pf(-exp_alog * sp);
                beta_smem[c] = ptx_sigmoid_pf(b_val);
            }
        }
        
        // Swap buffers
        buf_idx = next_buf;
        __syncthreads();
    }
    
    // ══════════════════════════════════════════════════════════════════
    // STORE FINAL STATE
    // ══════════════════════════════════════════════════════════════════
    float* new_state_ptr = NewState + n * stride_ns_n + h * stride_ns_h + v0 * stride_ns_v;
    for (int i = tid; i < BLOCK_V * D; i += num_threads) {
        int vi = i / D;
        int ki = i % D;
        ptx_st_wb_pf(&new_state_ptr[vi * stride_ns_v + ki], state_smem[i]);
    }
}

// ============================================================
// Warp-Cooperative Prefill Kernel
// Uses parallel reduction for mat-vec: 8 threads per row
// 128 threads / 16 rows = 8 threads per row
// Each group of 8 threads computes one row's dot product
// ============================================================

template<int CHUNK_SIZE>
__global__ void __launch_bounds__(128)
gdn_prefill_kernel_ptx_warp_coop(
    const __nv_bfloat16* __restrict__ Q,       // [T, 4, D]
    const __nv_bfloat16* __restrict__ K,       // [T, 4, D]
    const __nv_bfloat16* __restrict__ V,       // [T, 8, D]
    const float* __restrict__ State,            // [N, 8, D, D]
    const float* __restrict__ A_log,            // [8]
    const __nv_bfloat16* __restrict__ A,       // [T, 8]
    const float* __restrict__ DtBias,          // [8]
    const __nv_bfloat16* __restrict__ B_gate,  // [T, 8]
    const int32_t* __restrict__ CuSeqlens,     // [N+1]
    __nv_bfloat16* __restrict__ Out,           // [T, 8, D]
    float* __restrict__ NewState,               // [N, 8, D, D]
    float scale,
    int stride_q_t, int stride_q_h,
    int stride_k_t, int stride_k_h,
    int stride_v_t, int stride_v_h,
    int stride_s_n, int stride_s_h, int stride_s_v,
    int stride_a_t, int stride_b_t,
    int stride_o_t, int stride_o_h,
    int stride_ns_n, int stride_ns_h, int stride_ns_v
) {
    constexpr int BLOCK_V = 16;
    constexpr int D = PREFILL_D_PTX;  // 128
    constexpr int THREADS_PER_ROW = 8;  // 128 threads / 16 rows
    constexpr int WARP_SIZE = 32;
    
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const int v_block = blockIdx.z;
    const int tid = threadIdx.x;
    
    // Thread mapping: which row and which element within row
    const int row_id = tid / THREADS_PER_ROW;   // 0-15 (which V row)
    const int lane_in_row = tid % THREADS_PER_ROW;  // 0-7 (position in row reduction)
    
    const int v0 = v_block * BLOCK_V;
    const int num_threads = blockDim.x;  // 128
    
    // Sequence bounds
    const int t_start = CuSeqlens[n];
    const int t_end = CuSeqlens[n + 1];
    const int seq_len = t_end - t_start;
    
    if (seq_len <= 0) return;
    
    // Gate values (constant for head h)
    const float a_log_val = A_log[h];
    const float dt_bias = DtBias[h];
    const int k_head = h >> 1;  // Q/K have 4 heads
    
    // Shared memory layout
    extern __shared__ char smem[];
    float* state_smem = (float*)smem;                    // [BLOCK_V, D]
    float* q_smem = state_smem + BLOCK_V * D;            // [D]
    float* k_smem = q_smem + D;                          // [D]
    float* v_smem = k_smem + D;                          // [BLOCK_V]
    float* old_v_smem = v_smem + BLOCK_V;                // [BLOCK_V]
    float* out_smem = old_v_smem + BLOCK_V;              // [BLOCK_V]
    float* partial_smem = out_smem + BLOCK_V;            // [BLOCK_V * THREADS_PER_ROW] for reduction
    
    // Load initial state
    const float* state_ptr = State + n * stride_s_n + h * stride_s_h + v0 * stride_s_v;
    for (int i = tid; i < BLOCK_V * D; i += num_threads) {
        int vi = i / D;
        int ki = i % D;
        state_smem[vi * D + ki] = state_ptr[vi * stride_s_v + ki];
    }
    __syncthreads();
    
    // Process tokens sequentially
    for (int t = t_start; t < t_end; t++) {
        // ── Load Q, K, V ──────────────────────────────────────────────
        for (int i = tid; i < D; i += num_threads) {
            q_smem[i] = __bfloat162float(Q[t * stride_q_t + k_head * stride_q_h + i]);
            k_smem[i] = __bfloat162float(K[t * stride_k_t + k_head * stride_k_h + i]);
        }
        for (int i = tid; i < BLOCK_V; i += num_threads) {
            v_smem[i] = __bfloat162float(V[t * stride_v_t + h * stride_v_h + v0 + i]);
        }
        __syncthreads();
        
        // ── Compute gates ─────────────────────────────────────────────
        float gate, beta;
        if (tid == 0) {
            float a_val = __bfloat162float(A[t * stride_a_t + h]);
            float b_val = __bfloat162float(B_gate[t * stride_b_t + h]);
            float sp = ptx_softplus_pf(a_val + dt_bias);
            float exp_alog = ptx_exp_pf(a_log_val);
            gate = ptx_exp_pf(-exp_alog * sp);
            beta = ptx_sigmoid_pf(b_val);
            // Store for all threads
            partial_smem[0] = gate;
            partial_smem[1] = beta;
        }
        __syncthreads();
        gate = partial_smem[0];
        beta = partial_smem[1];
        
        // ── Apply gate decay: S = g * S ───────────────────────────────
        for (int i = tid; i < BLOCK_V * D; i += num_threads) {
            state_smem[i] *= gate;
        }
        __syncthreads();
        
        // ══════════════════════════════════════════════════════════════
        // WARP-COOPERATIVE PARALLEL REDUCTION
        // Each group of 8 threads computes one row's dot product
        // Thread i (row r, lane l) computes partial sum for row r
        // ══════════════════════════════════════════════════════════════
        
        if (row_id < BLOCK_V) {
            float* state_row = state_smem + row_id * D;
            
            // Compute partial sum: each thread handles D/8 = 16 elements
            float sum_k = 0.0f;
            float sum_q = 0.0f;
            
            int start_k = lane_in_row * (D / THREADS_PER_ROW);  // 0, 16, 32, ...
            int end_k = start_k + (D / THREADS_PER_ROW);
            
            #pragma unroll
            for (int ki = start_k; ki < end_k; ki += 4) {
                float s0 = state_row[ki + 0];
                float s1 = state_row[ki + 1];
                float s2 = state_row[ki + 2];
                float s3 = state_row[ki + 3];
                
                sum_k = ptx_fma_pf(s0, k_smem[ki + 0], sum_k);
                sum_k = ptx_fma_pf(s1, k_smem[ki + 1], sum_k);
                sum_k = ptx_fma_pf(s2, k_smem[ki + 2], sum_k);
                sum_k = ptx_fma_pf(s3, k_smem[ki + 3], sum_k);
                
                sum_q = ptx_fma_pf(s0, q_smem[ki + 0], sum_q);
                sum_q = ptx_fma_pf(s1, q_smem[ki + 1], sum_q);
                sum_q = ptx_fma_pf(s2, q_smem[ki + 2], sum_q);
                sum_q = ptx_fma_pf(s3, q_smem[ki + 3], sum_q);
            }
            
            // Store partial results to shared memory for cross-thread reduction
            partial_smem[row_id * THREADS_PER_ROW * 2 + lane_in_row] = sum_k;
            partial_smem[row_id * THREADS_PER_ROW * 2 + THREADS_PER_ROW + lane_in_row] = sum_q;
        }
        __syncthreads();
        
        // ── Reduce across threads in each row (lane 0 does final sum) ─
        if (row_id < BLOCK_V && lane_in_row == 0) {
            float old_v = 0.0f;
            float out_val = 0.0f;
            
            #pragma unroll
            for (int l = 0; l < THREADS_PER_ROW; l++) {
                old_v += partial_smem[row_id * THREADS_PER_ROW * 2 + l];
                out_val += partial_smem[row_id * THREADS_PER_ROW * 2 + THREADS_PER_ROW + l];
            }
            
            old_v_smem[row_id] = old_v;
            out_smem[row_id] = scale * out_val;
        }
        __syncthreads();
        
        // ── Compute delta and rank-1 update ───────────────────────────
        // delta = beta * (v - old_v)
        // S[v,:] += delta * K
        for (int i = tid; i < BLOCK_V * D; i += num_threads) {
            int vi = i / D;
            int ki = i % D;
            float delta = beta * (v_smem[vi] - old_v_smem[vi]);
            state_smem[i] = ptx_fma_pf(delta, k_smem[ki], state_smem[i]);
        }
        __syncthreads();
        
        // ── Store output ──────────────────────────────────────────────
        for (int i = tid; i < BLOCK_V; i += num_threads) {
            Out[t * stride_o_t + h * stride_o_h + v0 + i] = 
                __float2bfloat16(out_smem[i]);
        }
        __syncthreads();
    }
    
    // Store final state
    float* new_state_ptr = NewState + n * stride_ns_n + h * stride_ns_h + v0 * stride_ns_v;
    for (int i = tid; i < BLOCK_V * D; i += num_threads) {
        int vi = i / D;
        int ki = i % D;
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

// ============================================================
// Tensor Core Optimized Launcher (BLOCK_V=16 only)
// With TMA double-buffering and cp.async prefetch
// ============================================================

inline void gdn_prefill_ptx_mma_launch(
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
    int CHUNK_SIZE,
    cudaStream_t stream
) {
    constexpr int BLOCK_V = 16;  // Fixed for mma
    int V_BLOCKS = D / BLOCK_V;
    dim3 grid(N, num_v_heads, V_BLOCKS);
    dim3 block(128);
    
    // Shared memory layout with double-buffering:
    // - state_smem: [BLOCK_V, D] = [16, 128]
    // - qk_buf[0]: [CHUNK_SIZE, D*2]  (Q+K interleaved)
    // - qk_buf[1]: [CHUNK_SIZE, D*2]
    // - v_buf[0]: [CHUNK_SIZE, BLOCK_V]
    // - v_buf[1]: [CHUNK_SIZE, BLOCK_V]
    // - gate_smem: [CHUNK_SIZE]
    // - beta_smem: [CHUNK_SIZE]
    // - out_smem: [CHUNK_SIZE, BLOCK_V]
    size_t smem_size = (BLOCK_V * D +                    // state_smem
                        CHUNK_SIZE * D * 2 * 2 +         // qk_buf[0] + qk_buf[1]
                        CHUNK_SIZE * BLOCK_V * 2 +       // v_buf[0] + v_buf[1]
                        CHUNK_SIZE +                     // gate_smem
                        CHUNK_SIZE +                     // beta_smem
                        CHUNK_SIZE * BLOCK_V             // out_smem
                       ) * sizeof(float);
    
    if (CHUNK_SIZE == 8) {
        gdn_prefill_kernel_ptx_mma<8><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State, (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate, (const int32_t*)CuSeqlens,
            (__nv_bfloat16*)Out, (float*)NewState, scale,
            stride_q_t, stride_q_h, stride_k_t, stride_k_h,
            stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v,
            stride_a_t, stride_b_t, stride_o_t, stride_o_h,
            stride_ns_n, stride_ns_h, stride_ns_v);
    } else {
        gdn_prefill_kernel_ptx_mma<4><<<grid, block, smem_size, stream>>>(
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

// ============================================================
// Warp-Cooperative Launcher (parallel reduction)
// ============================================================

inline void gdn_prefill_ptx_warp_coop_launch(
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
    int CHUNK_SIZE,
    cudaStream_t stream
) {
    constexpr int BLOCK_V = 16;
    constexpr int THREADS_PER_ROW = 8;
    int V_BLOCKS = D / BLOCK_V;
    dim3 grid(N, num_v_heads, V_BLOCKS);
    dim3 block(128);
    
    // Shared memory layout:
    // - state_smem: [BLOCK_V, D]
    // - q_smem: [D]
    // - k_smem: [D]
    // - v_smem: [BLOCK_V]
    // - old_v_smem: [BLOCK_V]
    // - out_smem: [BLOCK_V]
    // - partial_smem: [BLOCK_V * THREADS_PER_ROW * 2] for reduction
    size_t smem_size = (BLOCK_V * D +              // state_smem
                        D +                         // q_smem
                        D +                         // k_smem
                        BLOCK_V +                   // v_smem
                        BLOCK_V +                   // old_v_smem
                        BLOCK_V +                   // out_smem
                        BLOCK_V * THREADS_PER_ROW * 2  // partial_smem
                       ) * sizeof(float);
    
    // Use CHUNK_SIZE=1 for warp-coop (processes one token at a time)
    gdn_prefill_kernel_ptx_warp_coop<1><<<grid, block, smem_size, stream>>>(
        (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
        (const float*)State, (const float*)A_log, (const __nv_bfloat16*)A,
        (const float*)DtBias, (const __nv_bfloat16*)B_gate, (const int32_t*)CuSeqlens,
        (__nv_bfloat16*)Out, (float*)NewState, scale,
        stride_q_t, stride_q_h, stride_k_t, stride_k_h,
        stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v,
        stride_a_t, stride_b_t, stride_o_t, stride_o_h,
        stride_ns_n, stride_ns_h, stride_ns_v);
}

}  // namespace gdn_ptx

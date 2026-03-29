/*
 * GDN Decode — CUDA C++ with Embedded PTX Assembly
 *
 * This implementation demonstrates low-level PTX intrinsics for:
 *   - Warp shuffle reductions (shfl.sync.bfly.b32)
 *   - Fast math (ex2.approx, lg2.approx, rcp.approx)
 *   - FMA operations (fma.rn.f32)
 *   - Memory operations with cache hints (ld.global.nc, st.global.wb)
 *   - Async copy (cp.async) for prefetch (Iteration 1)
 *   - Predicated execution
 *   - FP8 state quantization (Iteration 2 - 4x memory compression)
 *
 * Grid: (B, H=8, V_BLOCKS)
 * Block: 128 threads (4 warps)
 *
 * State layout: k-last [B, H, V=128, K=128] float32
 * GVA: num_q_heads=4, num_v_heads=8 → qk_head = v_head // 2
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>  // FP8 support (Iteration 2)
#include <cstdint>

namespace gdn_ptx {

// ============================================================
// PTX Inline Assembly Primitives
// ============================================================

// Warp-level butterfly shuffle (for reductions)
__device__ __forceinline__ float ptx_shfl_xor(float val, int lane_mask) {
    float result;
    asm volatile(
        "shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
        : "=f"(result)
        : "f"(val), "r"(lane_mask)
    );
    return result;
}

// Warp-level direct shuffle (broadcast from lane)
__device__ __forceinline__ float ptx_shfl_idx(float val, int src_lane) {
    float result;
    asm volatile(
        "shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;"
        : "=f"(result)
        : "f"(val), "r"(src_lane)
    );
    return result;
}

// Fast approximate exp2 (2^x)
__device__ __forceinline__ float ptx_exp2(float x) {
    float result;
    asm volatile(
        "ex2.approx.f32 %0, %1;"
        : "=f"(result)
        : "f"(x)
    );
    return result;
}

// Fast approximate log2
__device__ __forceinline__ float ptx_log2(float x) {
    float result;
    asm volatile(
        "lg2.approx.f32 %0, %1;"
        : "=f"(result)
        : "f"(x)
    );
    return result;
}

// Fast approximate reciprocal (1/x)
__device__ __forceinline__ float ptx_rcp(float x) {
    float result;
    asm volatile(
        "rcp.approx.f32 %0, %1;"
        : "=f"(result)
        : "f"(x)
    );
    return result;
}

// Fused multiply-add (a * b + c)
__device__ __forceinline__ float ptx_fma(float a, float b, float c) {
    float result;
    asm volatile(
        "fma.rn.f32 %0, %1, %2, %3;"
        : "=f"(result)
        : "f"(a), "f"(b), "f"(c)
    );
    return result;
}

// Load float with non-coherent cache hint (L2 bypass)
__device__ __forceinline__ float ptx_ld_nc(const float* ptr) {
    float result;
    asm volatile(
        "ld.global.nc.f32 %0, [%1];"
        : "=f"(result)
        : "l"(ptr)
    );
    return result;
}

// ============================================================
// cp.async PTX Primitives (Iteration 1 - Async Prefetch)
// ============================================================

// Async copy 4 bytes from global to shared memory
__device__ __forceinline__ void ptx_cp_async_ca(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;"
        :
        : "r"(smem_addr), "l"(gmem_ptr)
        : "memory"
    );
}

// Async copy 16 bytes (4 floats) from global to shared memory
__device__ __forceinline__ void ptx_cp_async_cg(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :
        : "r"(smem_addr), "l"(gmem_ptr)
        : "memory"
    );
}

// Commit async copy group
__device__ __forceinline__ void ptx_cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

// Wait for N groups (0 = all)
template<int N>
__device__ __forceinline__ void ptx_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" : : "n"(N));
}

// Wait for all async copies
__device__ __forceinline__ void ptx_cp_async_wait_all() {
    asm volatile("cp.async.wait_all;");
}

// Load float4 with cache hint
__device__ __forceinline__ void ptx_ld_nc_v4(float4& dst, const float* ptr) {
    asm volatile(
        "ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w)
        : "l"(ptr)
    );
}

// Store float with write-back cache hint
__device__ __forceinline__ void ptx_st_wb(float* ptr, float val) {
    asm volatile(
        "st.global.wb.f32 [%0], %1;"
        :: "l"(ptr), "f"(val)
    );
}

// Store float4 with write-back
__device__ __forceinline__ void ptx_st_wb_v4(float* ptr, float4 val) {
    asm volatile(
        "st.global.wb.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(ptr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w)
    );
}

// Predicated move (conditional assignment)
__device__ __forceinline__ float ptx_selp(float a, float b, bool pred) {
    float result;
    asm volatile(
        "selp.f32 %0, %1, %2, %3;"
        : "=f"(result)
        : "f"(a), "f"(b), "r"((int)pred)
    );
    return result;
}

// Bar.sync with explicit barrier ID
__device__ __forceinline__ void ptx_bar_sync(int barrier_id) {
    asm volatile("bar.sync %0;" :: "r"(barrier_id));
}

// ============================================================
// FP8 E4M3 Quantization PTX Primitives (Iteration 2)
// ============================================================

// Convert FP32 to FP8 E4M3
__device__ __forceinline__ __nv_fp8_e4m3 ptx_fp32_to_fp8(float val) {
    return __nv_fp8_e4m3(val);
}

// Convert FP8 E4M3 to FP32
__device__ __forceinline__ float ptx_fp8_to_fp32(__nv_fp8_e4m3 val) {
    return float(val);
}

// Pack 4 FP8 values into uint32_t for vectorized memory access
__device__ __forceinline__ uint32_t ptx_pack_fp8x4(
    __nv_fp8_e4m3 a, __nv_fp8_e4m3 b, __nv_fp8_e4m3 c, __nv_fp8_e4m3 d
) {
    uint32_t result;
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&result);
    bytes[0] = *reinterpret_cast<const uint8_t*>(&a);
    bytes[1] = *reinterpret_cast<const uint8_t*>(&b);
    bytes[2] = *reinterpret_cast<const uint8_t*>(&c);
    bytes[3] = *reinterpret_cast<const uint8_t*>(&d);
    return result;
}

// Unpack uint32_t to 4 FP8 values
__device__ __forceinline__ void ptx_unpack_fp8x4(
    uint32_t packed,
    __nv_fp8_e4m3& a, __nv_fp8_e4m3& b, __nv_fp8_e4m3& c, __nv_fp8_e4m3& d
) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&packed);
    a = *reinterpret_cast<const __nv_fp8_e4m3*>(&bytes[0]);
    b = *reinterpret_cast<const __nv_fp8_e4m3*>(&bytes[1]);
    c = *reinterpret_cast<const __nv_fp8_e4m3*>(&bytes[2]);
    d = *reinterpret_cast<const __nv_fp8_e4m3*>(&bytes[3]);
}

// Load uint32_t (packed FP8x4) with cache hint
__device__ __forceinline__ uint32_t ptx_ld_nc_u32(const uint32_t* ptr) {
    uint32_t result;
    asm volatile(
        "ld.global.nc.b32 %0, [%1];"
        : "=r"(result)
        : "l"(ptr)
    );
    return result;
}

// ============================================================
// Derived Math Functions using PTX
// ============================================================

// exp(x) = 2^(x * log2(e)) where log2(e) ≈ 1.4426950408889634
__device__ __forceinline__ float ptx_exp(float x) {
    constexpr float LOG2E = 1.4426950408889634f;
    return ptx_exp2(x * LOG2E);
}

// log(x) = log2(x) / log2(e) = log2(x) * ln(2) where ln(2) ≈ 0.6931471805599453
__device__ __forceinline__ float ptx_log(float x) {
    constexpr float LN2 = 0.6931471805599453f;
    return ptx_log2(x) * LN2;
}

// Softplus: log(1 + exp(x))
// For large x, softplus(x) ≈ x
__device__ __forceinline__ float ptx_softplus(float x) {
    // Branch-free: use predicated select
    float exp_x = ptx_exp(x);
    float log_result = ptx_log(1.0f + exp_x);
    return ptx_selp(x, log_result, x > 20.0f);
}

// Sigmoid: 1 / (1 + exp(-x))
__device__ __forceinline__ float ptx_sigmoid(float x) {
    float exp_neg_x = ptx_exp(-x);
    return ptx_rcp(1.0f + exp_neg_x);
}

// ============================================================
// Warp-level Reduction using PTX Shuffle
// ============================================================

__device__ __forceinline__ float ptx_warp_reduce_sum(float val) {
    // Butterfly reduction pattern
    val += ptx_shfl_xor(val, 16);
    val += ptx_shfl_xor(val, 8);
    val += ptx_shfl_xor(val, 4);
    val += ptx_shfl_xor(val, 2);
    val += ptx_shfl_xor(val, 1);
    return val;
}

// ============================================================
// Constants
// ============================================================

constexpr int D = 128;
constexpr int WARP_SIZE = 32;

// ============================================================
// Main GDN Decode Kernel with PTX
// ============================================================

template<int BLOCK_V>
__global__ void gdn_decode_kernel_ptx(
    // Inputs
    const __nv_bfloat16* __restrict__ Q,      // [B, 4, D]
    const __nv_bfloat16* __restrict__ K,      // [B, 4, D]
    const __nv_bfloat16* __restrict__ V,      // [B, 8, D]
    const float* __restrict__ State,          // [B, 8, D, D]
    // Gates
    const float* __restrict__ A_log,          // [8]
    const __nv_bfloat16* __restrict__ A,      // [B, 8]
    const __nv_bfloat16* __restrict__ DtBias, // [8]
    const __nv_bfloat16* __restrict__ B_gate, // [B, 8]
    // Outputs
    __nv_bfloat16* __restrict__ Out,          // [B, 8, D]
    float* __restrict__ NewState,             // [B, 8, D, D]
    // Params
    float scale,
    int stride_q_b, int stride_q_h,
    int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h,
    int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b,
    int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v
) {
    // Grid indices
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int vb = blockIdx.z;
    const int v0 = vb * BLOCK_V;
    const int qk_h = h / 2;  // GVA mapping
    
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_threads = blockDim.x;
    
    // Shared memory
    extern __shared__ float smem[];
    float* q_smem = smem;                      // [D]
    float* k_smem = q_smem + D;                // [D]
    float* v_smem = k_smem + D;                // [BLOCK_V]
    float* state_smem = v_smem + BLOCK_V;      // [BLOCK_V * D]
    float* reduce_smem = state_smem + BLOCK_V * D;  // [4]
    
    // ─── Load gates using PTX (single thread) ──────────────────────
    __shared__ float g_shared, beta_shared;
    if (tid == 0) {
        float a_val = __bfloat162float(A[b * stride_a_b + h]);
        float dt_val = __bfloat162float(DtBias[h]);
        float alog = ptx_ld_nc(&A_log[h]);
        float b_val = __bfloat162float(B_gate[b * stride_b_b + h]);
        
        // Compute gates using PTX math
        float sp = ptx_softplus(a_val + dt_val);
        float exp_alog = ptx_exp(alog);
        g_shared = ptx_exp(-exp_alog * sp);
        beta_shared = ptx_sigmoid(b_val);
    }
    __syncthreads();
    
    float g = g_shared;
    float beta = beta_shared;
    
    // ─── Load Q, K [D] with PTX vectorized loads ───────────────────
    // Use float4 loads for better memory throughput
    if (tid < D / 4) {
        const float* q_ptr = (const float*)(Q + b * stride_q_b + qk_h * stride_q_h);
        const float* k_ptr = (const float*)(K + b * stride_k_b + qk_h * stride_k_h);
        
        // Convert bf16 to float (4 elements at a time)
        for (int i = tid; i < D; i += num_threads) {
            q_smem[i] = __bfloat162float(Q[b * stride_q_b + qk_h * stride_q_h + i]);
            k_smem[i] = __bfloat162float(K[b * stride_k_b + qk_h * stride_k_h + i]);
        }
    }
    
    // Load V slice [BLOCK_V]
    for (int i = tid; i < BLOCK_V; i += num_threads) {
        v_smem[i] = __bfloat162float(V[b * stride_v_b + h * stride_v_h + v0 + i]);
    }
    __syncthreads();
    
    // ─── Load state tile [BLOCK_V, D] with cp.async (async prefetch) ────
    const float* state_ptr = State + b * stride_s_b + h * stride_s_h + v0 * stride_s_v;
    
    // Use cp.async for async prefetch - overlaps with computation
    for (int i = tid; i < BLOCK_V * D; i += num_threads) {
        int row = i / D;
        int col = i % D;
        ptx_cp_async_ca(&state_smem[i], &state_ptr[row * stride_s_v + col]);
    }
    ptx_cp_async_commit();
    ptx_cp_async_wait<0>();
    __syncthreads();
    
    // ─── Apply gate decay: S = g * S (using PTX FMA) ───────────────
    for (int i = tid; i < BLOCK_V * D; i += num_threads) {
        // state_smem[i] = g * state_smem[i] + 0 (FMA pattern)
        state_smem[i] = ptx_fma(g, state_smem[i], 0.0f);
    }
    __syncthreads();
    
    // ─── Compute old_v = S @ k using PTX reduction ─────────────────
    __shared__ float old_v_smem[64];
    
    // Each thread handles one row's dot product
    if (tid < BLOCK_V) {
        float sum = 0.0f;
        
        // Unrolled dot product with PTX FMA
        #pragma unroll 8
        for (int ki = 0; ki < D; ki += 4) {
            sum = ptx_fma(state_smem[tid * D + ki + 0], k_smem[ki + 0], sum);
            sum = ptx_fma(state_smem[tid * D + ki + 1], k_smem[ki + 1], sum);
            sum = ptx_fma(state_smem[tid * D + ki + 2], k_smem[ki + 2], sum);
            sum = ptx_fma(state_smem[tid * D + ki + 3], k_smem[ki + 3], sum);
        }
        old_v_smem[tid] = sum;
    }
    __syncthreads();
    
    // ─── Compute delta and rank-1 update ───────────────────────────
    // delta[vi] = beta * (v[vi] - old_v[vi])
    // S[vi, :] += delta[vi] * k[:]
    
    for (int i = tid; i < BLOCK_V * D; i += num_threads) {
        int vi = i / D;
        int ki = i % D;
        float delta = beta * (v_smem[vi] - old_v_smem[vi]);
        // S[vi,ki] = S[vi,ki] + delta * k[ki] (FMA)
        state_smem[i] = ptx_fma(delta, k_smem[ki], state_smem[i]);
    }
    __syncthreads();
    
    // ─── Compute out = scale * S @ q with PTX ──────────────────────
    __shared__ float out_smem[64];
    
    if (tid < BLOCK_V) {
        float sum = 0.0f;
        
        #pragma unroll 8
        for (int ki = 0; ki < D; ki += 4) {
            sum = ptx_fma(state_smem[tid * D + ki + 0], q_smem[ki + 0], sum);
            sum = ptx_fma(state_smem[tid * D + ki + 1], q_smem[ki + 1], sum);
            sum = ptx_fma(state_smem[tid * D + ki + 2], q_smem[ki + 2], sum);
            sum = ptx_fma(state_smem[tid * D + ki + 3], q_smem[ki + 3], sum);
        }
        out_smem[tid] = scale * sum;
    }
    __syncthreads();
    
    // ─── Store output [BLOCK_V] ────────────────────────────────────
    for (int i = tid; i < BLOCK_V; i += num_threads) {
        Out[b * stride_o_b + h * stride_o_h + v0 + i] = __float2bfloat16(out_smem[i]);
    }
    
    // ─── Store new state [BLOCK_V, D] with PTX ─────────────────────
    float* new_state_ptr = NewState + b * stride_ns_b + h * stride_ns_h + v0 * stride_ns_v;
    
    for (int i = tid; i < BLOCK_V * D; i += num_threads) {
        int vi = i / D;
        int ki = i % D;
        ptx_st_wb(&new_state_ptr[vi * stride_ns_v + ki], state_smem[i]);
    }
}

// ============================================================
// FP8 Quantized State Kernel (Iteration 2 - 4x memory compression)
// ============================================================

template<int BLOCK_V>
__global__ void gdn_decode_kernel_ptx_fp8(
    // Inputs
    const __nv_bfloat16* __restrict__ Q,      // [B, 4, D]
    const __nv_bfloat16* __restrict__ K,      // [B, 4, D]
    const __nv_bfloat16* __restrict__ V,      // [B, 8, D]
    const uint32_t* __restrict__ State_FP8,   // [B, 8, D, D/4] packed FP8x4
    const float* __restrict__ State_Scale,    // [B, 8, D] per-row scale
    // Gates
    const float* __restrict__ A_log,          // [8]
    const __nv_bfloat16* __restrict__ A,      // [B, 8]
    const __nv_bfloat16* __restrict__ DtBias, // [8]
    const __nv_bfloat16* __restrict__ B_gate, // [B, 8]
    // Outputs
    __nv_bfloat16* __restrict__ Out,          // [B, 8, D]
    uint32_t* __restrict__ NewState_FP8,      // [B, 8, D, D/4] packed FP8x4
    float* __restrict__ NewState_Scale,       // [B, 8, D] per-row scale
    // Params
    float scale,
    int stride_q_b, int stride_q_h,
    int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h,
    int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b,
    int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v
) {
    // Grid indices
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int vb = blockIdx.z;
    const int v0 = vb * BLOCK_V;
    const int qk_h = h / 2;
    
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_threads = blockDim.x;
    
    // Shared memory
    extern __shared__ float smem[];
    float* q_smem = smem;
    float* k_smem = q_smem + D;
    float* v_smem = k_smem + D;
    float* state_smem = v_smem + BLOCK_V;
    float* scale_smem = state_smem + BLOCK_V * D;  // Per-row scales
    float* reduce_smem = scale_smem + BLOCK_V;
    
    // Load gates
    __shared__ float g_shared, beta_shared;
    if (tid == 0) {
        float a_val = __bfloat162float(A[b * stride_a_b + h]);
        float dt_val = __bfloat162float(DtBias[h]);
        float alog = ptx_ld_nc(&A_log[h]);
        float b_val = __bfloat162float(B_gate[b * stride_b_b + h]);
        
        float x = a_val + dt_val;
        float sp = ptx_softplus(x);
        float decay = ptx_exp(-ptx_exp(alog) * sp);
        float beta = ptx_sigmoid(b_val);
        
        g_shared = decay;
        beta_shared = beta;
    }
    
    // Load Q, K with vectorized loads
    const __nv_bfloat16* q_ptr = Q + b * stride_q_b + qk_h * stride_q_h;
    const __nv_bfloat16* k_ptr = K + b * stride_k_b + qk_h * stride_k_h;
    
    for (int d = tid; d < D; d += num_threads) {
        q_smem[d] = __bfloat162float(q_ptr[d]);
        k_smem[d] = __bfloat162float(k_ptr[d]);
    }
    
    // Load V
    const __nv_bfloat16* v_ptr = V + b * stride_v_b + h * stride_v_h + v0;
    for (int i = tid; i < BLOCK_V; i += num_threads) {
        v_smem[i] = __bfloat162float(v_ptr[i]);
    }
    
    // Load per-row scales
    const float* scale_ptr = State_Scale + b * stride_s_b + h * stride_s_h + v0;
    for (int i = tid; i < BLOCK_V; i += num_threads) {
        scale_smem[i] = ptx_ld_nc(&scale_ptr[i]);
    }
    
    // Load FP8 state and dequantize
    const uint32_t* state_fp8_ptr = State_FP8 + b * stride_s_b + h * stride_s_h + v0 * stride_s_v;
    
    for (int i = tid; i < BLOCK_V * (D / 4); i += num_threads) {
        int vi = i / (D / 4);
        int d4_idx = i % (D / 4);
        int d_base = d4_idx * 4;
        
        uint32_t packed = ptx_ld_nc_u32(&state_fp8_ptr[vi * stride_s_v + d4_idx]);
        
        __nv_fp8_e4m3 fp8_0, fp8_1, fp8_2, fp8_3;
        ptx_unpack_fp8x4(packed, fp8_0, fp8_1, fp8_2, fp8_3);
        
        float row_scale = scale_smem[vi];
        state_smem[vi * D + d_base + 0] = ptx_fp8_to_fp32(fp8_0) * row_scale;
        state_smem[vi * D + d_base + 1] = ptx_fp8_to_fp32(fp8_1) * row_scale;
        state_smem[vi * D + d_base + 2] = ptx_fp8_to_fp32(fp8_2) * row_scale;
        state_smem[vi * D + d_base + 3] = ptx_fp8_to_fp32(fp8_3) * row_scale;
    }
    
    __syncthreads();
    
    float g = g_shared;
    float beta = beta_shared;
    
    // Output accumulator in registers
    float* old_v_smem = reduce_smem + 4;  // [BLOCK_V]
    float* out_smem = old_v_smem + BLOCK_V;  // [BLOCK_V]
    
    // Delta rule computation (same as FP32 version)
    for (int vi = warp_id; vi < BLOCK_V; vi += 4) {
        float old_v = 0.0f;
        
        #pragma unroll 4
        for (int d = lane_id; d < D; d += WARP_SIZE) {
            float decayed_s = g * state_smem[vi * D + d];
            old_v = ptx_fma(decayed_s, k_smem[d], old_v);
        }
        
        old_v = ptx_warp_reduce_sum(old_v);
        
        float delta = beta * (v_smem[vi] - old_v);
        
        #pragma unroll 4
        for (int d = lane_id; d < D; d += WARP_SIZE) {
            float decayed_s = g * state_smem[vi * D + d];
            state_smem[vi * D + d] = ptx_fma(delta, k_smem[d], decayed_s);
        }
        
        if (lane_id == 0) {
            old_v_smem[vi] = old_v;
        }
    }
    
    __syncthreads();
    
    // Compute output: out = sum(new_state * q)
    for (int vi = warp_id; vi < BLOCK_V; vi += 4) {
        float out_acc = 0.0f;
        
        #pragma unroll 4
        for (int d = lane_id; d < D; d += WARP_SIZE) {
            out_acc = ptx_fma(state_smem[vi * D + d], q_smem[d], out_acc);
        }
        
        out_acc = ptx_warp_reduce_sum(out_acc);
        
        if (lane_id == 0) {
            out_smem[vi] = scale * out_acc;
        }
    }
    
    __syncthreads();
    
    // Store output
    for (int i = tid; i < BLOCK_V; i += num_threads) {
        Out[b * stride_o_b + h * stride_o_h + v0 + i] = __float2bfloat16(out_smem[i]);
    }
    
    // Compute new per-row scales and quantize to FP8
    uint32_t* new_state_fp8_ptr = NewState_FP8 + b * stride_ns_b + h * stride_ns_h + v0 * stride_ns_v;
    float* new_scale_ptr = NewState_Scale + b * stride_ns_b + h * stride_ns_h + v0;
    
    // Compute max abs per row for scaling
    for (int vi = tid; vi < BLOCK_V; vi += num_threads) {
        float max_abs = 0.0f;
        for (int d = 0; d < D; d++) {
            max_abs = fmaxf(max_abs, fabsf(state_smem[vi * D + d]));
        }
        float new_scale = (max_abs > 1e-6f) ? (max_abs / 400.0f) : 1.0f;
        scale_smem[vi] = new_scale;
        new_scale_ptr[vi] = new_scale;
    }
    
    __syncthreads();
    
    // Quantize and pack to FP8x4
    for (int i = tid; i < BLOCK_V * (D / 4); i += num_threads) {
        int vi = i / (D / 4);
        int d4_idx = i % (D / 4);
        int d_base = d4_idx * 4;
        
        float inv_scale = ptx_rcp(scale_smem[vi]);
        
        __nv_fp8_e4m3 fp8_0 = ptx_fp32_to_fp8(state_smem[vi * D + d_base + 0] * inv_scale);
        __nv_fp8_e4m3 fp8_1 = ptx_fp32_to_fp8(state_smem[vi * D + d_base + 1] * inv_scale);
        __nv_fp8_e4m3 fp8_2 = ptx_fp32_to_fp8(state_smem[vi * D + d_base + 2] * inv_scale);
        __nv_fp8_e4m3 fp8_3 = ptx_fp32_to_fp8(state_smem[vi * D + d_base + 3] * inv_scale);
        
        uint32_t packed = ptx_pack_fp8x4(fp8_0, fp8_1, fp8_2, fp8_3);
        new_state_fp8_ptr[vi * stride_ns_v + d4_idx] = packed;
    }
}

// ============================================================
// Launcher Function
// ============================================================

inline void gdn_decode_ptx_launch(
    const void* Q, const void* K, const void* V, const void* State,
    const void* A_log, const void* A, const void* DtBias, const void* B_gate,
    void* Out, void* NewState,
    float scale,
    int B, int num_v_heads, int D_val,
    int stride_q_b, int stride_q_h,
    int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h,
    int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b,
    int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v,
    int BLOCK_V,
    cudaStream_t stream
) {
    int V_BLOCKS = D_val / BLOCK_V;
    dim3 grid(B, num_v_heads, V_BLOCKS);
    dim3 block(128);
    
    // Shared memory: q[D] + k[D] + v[BLOCK_V] + state[BLOCK_V*D] + reduce[4] + old_v[64] + out[64]
    size_t smem_size = (D + D + BLOCK_V + BLOCK_V * D + 4 + 64 + 64) * sizeof(float);
    
    if (BLOCK_V == 16) {
        gdn_decode_kernel_ptx<16><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State,
            (const float*)A_log, (const __nv_bfloat16*)A, (const __nv_bfloat16*)DtBias,
            (const __nv_bfloat16*)B_gate,
            (__nv_bfloat16*)Out, (float*)NewState,
            scale,
            stride_q_b, stride_q_h, stride_k_b, stride_k_h,
            stride_v_b, stride_v_h,
            stride_s_b, stride_s_h, stride_s_v,
            stride_a_b, stride_b_b,
            stride_o_b, stride_o_h,
            stride_ns_b, stride_ns_h, stride_ns_v
        );
    } else if (BLOCK_V == 32) {
        gdn_decode_kernel_ptx<32><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State,
            (const float*)A_log, (const __nv_bfloat16*)A, (const __nv_bfloat16*)DtBias,
            (const __nv_bfloat16*)B_gate,
            (__nv_bfloat16*)Out, (float*)NewState,
            scale,
            stride_q_b, stride_q_h, stride_k_b, stride_k_h,
            stride_v_b, stride_v_h,
            stride_s_b, stride_s_h, stride_s_v,
            stride_a_b, stride_b_b,
            stride_o_b, stride_o_h,
            stride_ns_b, stride_ns_h, stride_ns_v
        );
    } else {  // BLOCK_V == 64
        gdn_decode_kernel_ptx<64><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State,
            (const float*)A_log, (const __nv_bfloat16*)A, (const __nv_bfloat16*)DtBias,
            (const __nv_bfloat16*)B_gate,
            (__nv_bfloat16*)Out, (float*)NewState,
            scale,
            stride_q_b, stride_q_h, stride_k_b, stride_k_h,
            stride_v_b, stride_v_h,
            stride_s_b, stride_s_h, stride_s_v,
            stride_a_b, stride_b_b,
            stride_o_b, stride_o_h,
            stride_ns_b, stride_ns_h, stride_ns_v
        );
    }
}

// ============================================================
// FP8 Launcher Function (Iteration 2 - 4x memory compression)
// ============================================================

inline void gdn_decode_ptx_fp8_launch(
    const void* Q, const void* K, const void* V,
    const void* State_FP8, const void* State_Scale,
    const void* A_log, const void* A, const void* DtBias, const void* B_gate,
    void* Out, void* NewState_FP8, void* NewState_Scale,
    float scale,
    int B, int num_v_heads, int D_val,
    int stride_q_b, int stride_q_h,
    int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h,
    int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b,
    int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v,
    int BLOCK_V,
    cudaStream_t stream
) {
    int V_BLOCKS = D_val / BLOCK_V;
    dim3 grid(B, num_v_heads, V_BLOCKS);
    dim3 block(128);
    
    // Extra space for per-row scales
    size_t smem_size = (D + D + BLOCK_V + BLOCK_V * D + BLOCK_V + 4 + BLOCK_V + BLOCK_V) * sizeof(float);
    
    if (BLOCK_V == 16) {
        gdn_decode_kernel_ptx_fp8<16><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const uint32_t*)State_FP8, (const float*)State_Scale,
            (const float*)A_log, (const __nv_bfloat16*)A, (const __nv_bfloat16*)DtBias,
            (const __nv_bfloat16*)B_gate,
            (__nv_bfloat16*)Out, (uint32_t*)NewState_FP8, (float*)NewState_Scale,
            scale,
            stride_q_b, stride_q_h, stride_k_b, stride_k_h,
            stride_v_b, stride_v_h,
            stride_s_b, stride_s_h, stride_s_v,
            stride_a_b, stride_b_b,
            stride_o_b, stride_o_h,
            stride_ns_b, stride_ns_h, stride_ns_v
        );
    } else if (BLOCK_V == 32) {
        gdn_decode_kernel_ptx_fp8<32><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const uint32_t*)State_FP8, (const float*)State_Scale,
            (const float*)A_log, (const __nv_bfloat16*)A, (const __nv_bfloat16*)DtBias,
            (const __nv_bfloat16*)B_gate,
            (__nv_bfloat16*)Out, (uint32_t*)NewState_FP8, (float*)NewState_Scale,
            scale,
            stride_q_b, stride_q_h, stride_k_b, stride_k_h,
            stride_v_b, stride_v_h,
            stride_s_b, stride_s_h, stride_s_v,
            stride_a_b, stride_b_b,
            stride_o_b, stride_o_h,
            stride_ns_b, stride_ns_h, stride_ns_v
        );
    } else {  // BLOCK_V == 64
        gdn_decode_kernel_ptx_fp8<64><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const uint32_t*)State_FP8, (const float*)State_Scale,
            (const float*)A_log, (const __nv_bfloat16*)A, (const __nv_bfloat16*)DtBias,
            (const __nv_bfloat16*)B_gate,
            (__nv_bfloat16*)Out, (uint32_t*)NewState_FP8, (float*)NewState_Scale,
            scale,
            stride_q_b, stride_q_h, stride_k_b, stride_k_h,
            stride_v_b, stride_v_h,
            stride_s_b, stride_s_h, stride_s_v,
            stride_a_b, stride_b_b,
            stride_o_b, stride_o_h,
            stride_ns_b, stride_ns_h, stride_ns_v
        );
    }
}

}  // namespace gdn_ptx

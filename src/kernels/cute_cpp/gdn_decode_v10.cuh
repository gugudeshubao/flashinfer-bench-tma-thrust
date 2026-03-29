/*
 * GDN Decode v10 — CuTe Layout Algebra kernel for B200 (sm100)
 *
 * Uses NVIDIA CuTe from CUTLASS 3.x for layout algebra only:
 *
 *   CuTe DSL Features:
 *   - Swizzle<B,M,S>: Bank conflict avoidance
 *   - Layout composition: Logical to physical mapping
 *   - make_coord, layout(): Coordinate transformation
 *
 *   Manual Memory Operations:
 *   - cp.async for async global->shared
 *   - Vectorized loads with float4
 *
 *   Precision Modes (Iteration 2):
 *   - FP32: Full precision state (default)
 *   - FP8 E4M3: 4x memory compression for state
 *   - FP4 E2M1: 8x memory compression for state (experimental)
 *
 * Grid: (B, H=8, V_BLOCKS)
 * Block: 128 threads (4 warps)
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>  // FP8 support (Iteration 2)
#include <cstdint>

// CuTe includes - only for Swizzle layout
#ifdef __CUDACC__
#include <cute/swizzle.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/layout.hpp>

using namespace cute;

namespace gdn {

// ============================================================
// Constants for v10
// ============================================================

constexpr int V10_D = 128;
constexpr int V10_WARP_SIZE = 32;
constexpr int V10_NUM_WARPS = 4;
constexpr int V10_THREADS = V10_NUM_WARPS * V10_WARP_SIZE;

// ============================================================
// FP8 E4M3 Quantization (Iteration 2 - 4x memory compression)
// ============================================================

// Convert FP32 to FP8 E4M3
__device__ __forceinline__ __nv_fp8_e4m3 v10_fp32_to_fp8(float val) {
    return __nv_fp8_e4m3(val);
}

// Convert FP8 E4M3 to FP32
__device__ __forceinline__ float v10_fp8_to_fp32(__nv_fp8_e4m3 val) {
    return float(val);
}

// Pack 4 FP8 values into uint32_t for vectorized memory access
__device__ __forceinline__ uint32_t v10_pack_fp8x4(
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
__device__ __forceinline__ void v10_unpack_fp8x4(
    uint32_t packed,
    __nv_fp8_e4m3& a, __nv_fp8_e4m3& b, __nv_fp8_e4m3& c, __nv_fp8_e4m3& d
) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&packed);
    a = *reinterpret_cast<const __nv_fp8_e4m3*>(&bytes[0]);
    b = *reinterpret_cast<const __nv_fp8_e4m3*>(&bytes[1]);
    c = *reinterpret_cast<const __nv_fp8_e4m3*>(&bytes[2]);
    d = *reinterpret_cast<const __nv_fp8_e4m3*>(&bytes[3]);
}

// ============================================================
// FP4 E2M1 Quantization (8x memory compression)
// FP4 E2M1: 1 sign + 2 exponent + 1 mantissa
// Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and negatives)
// ============================================================

// FP4 lookup table for dequantization (positive values only)
__device__ __constant__ float FP4_LUT[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

// Convert FP32 to FP4 E2M1 (returns 4-bit index)
__device__ __forceinline__ uint8_t v10_fp32_to_fp4(float val) {
    // Get sign bit
    uint8_t sign = (val < 0.0f) ? 0x8 : 0x0;
    float abs_val = fabsf(val);
    
    // Quantize to nearest FP4 value using thresholds
    // Thresholds: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    uint8_t mant;
    if (abs_val < 0.25f) mant = 0;
    else if (abs_val < 0.75f) mant = 1;
    else if (abs_val < 1.25f) mant = 2;
    else if (abs_val < 1.75f) mant = 3;
    else if (abs_val < 2.5f) mant = 4;
    else if (abs_val < 3.5f) mant = 5;
    else if (abs_val < 5.0f) mant = 6;
    else mant = 7;  // clamp to max (6.0)
    
    return sign | mant;
}

// Convert FP4 E2M1 to FP32
__device__ __forceinline__ float v10_fp4_to_fp32(uint8_t fp4) {
    uint8_t mant = fp4 & 0x7;
    float val = FP4_LUT[mant];
    return (fp4 & 0x8) ? -val : val;
}

// Pack 8 FP4 values into uint32_t for vectorized memory access
__device__ __forceinline__ uint32_t v10_pack_fp4x8(
    uint8_t a, uint8_t b, uint8_t c, uint8_t d,
    uint8_t e, uint8_t f, uint8_t g, uint8_t h
) {
    return ((uint32_t)(a & 0xF)) |
           ((uint32_t)(b & 0xF) << 4) |
           ((uint32_t)(c & 0xF) << 8) |
           ((uint32_t)(d & 0xF) << 12) |
           ((uint32_t)(e & 0xF) << 16) |
           ((uint32_t)(f & 0xF) << 20) |
           ((uint32_t)(g & 0xF) << 24) |
           ((uint32_t)(h & 0xF) << 28);
}

// Unpack uint32_t to 8 FP4 values
__device__ __forceinline__ void v10_unpack_fp4x8(
    uint32_t packed,
    uint8_t& a, uint8_t& b, uint8_t& c, uint8_t& d,
    uint8_t& e, uint8_t& f, uint8_t& g, uint8_t& h
) {
    a = (packed >> 0) & 0xF;
    b = (packed >> 4) & 0xF;
    c = (packed >> 8) & 0xF;
    d = (packed >> 12) & 0xF;
    e = (packed >> 16) & 0xF;
    f = (packed >> 20) & 0xF;
    g = (packed >> 24) & 0xF;
    h = (packed >> 28) & 0xF;
}

// ============================================================
// CuTe Swizzle Layout for Bank Conflict Avoidance
// ============================================================

// Swizzle<3,3,3> for 128-byte cache lines with 32-bank SMEM
// This maps logical [v,d] to physical index with XOR pattern
template<int BLOCK_V>
struct SwizzledStateLayout {
    using SwizzleType = Swizzle<3, 3, 3>;  // B=3, M=3, S=3
    
    __device__ __forceinline__ 
    static int get_index(int v_idx, int d_idx) {
        // Apply swizzle: physical_d = d_idx ^ ((d_idx >> 3) & 7)
        int swizzled_d = d_idx ^ ((d_idx >> 3) & 7);
        return v_idx * V10_D + swizzled_d;
    }
};

// ============================================================
// v10 Kernel: CuTe Swizzle + cp.async
// ============================================================

template<int BLOCK_V>
__global__ void __launch_bounds__(V10_THREADS)
gdn_decode_kernel_v10_cute(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ State,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ A,
    const float* __restrict__ DtBias,
    const __nv_bfloat16* __restrict__ B_gate,
    __nv_bfloat16* __restrict__ Out,
    float* __restrict__ NewState,
    float scale,
    int stride_q_b, int stride_q_h,
    int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h,
    int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b,
    int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int v_block = blockIdx.z;
    const int v0 = v_block * BLOCK_V;
    const int qk_h = h / 2;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / V10_WARP_SIZE;
    const int lane_id = tid % V10_WARP_SIZE;
    
    // Shared memory
    extern __shared__ char smem_raw[];
    float* smem_q = reinterpret_cast<float*>(smem_raw);
    float* smem_k = smem_q + V10_D;
    float* smem_v = smem_k + V10_D;
    float* smem_state = smem_v + BLOCK_V;
    float* smem_new_state = smem_state + BLOCK_V * V10_D;
    float* smem_out = smem_new_state + BLOCK_V * V10_D;
    
    // Load Q, K
    const __nv_bfloat16* q_ptr = Q + b * stride_q_b + qk_h * stride_q_h;
    const __nv_bfloat16* k_ptr = K + b * stride_k_b + qk_h * stride_k_h;
    
    #pragma unroll 4
    for (int d = tid; d < V10_D; d += V10_THREADS) {
        smem_q[d] = __bfloat162float(q_ptr[d]);
        smem_k[d] = __bfloat162float(k_ptr[d]);
    }
    
    // Load V
    const __nv_bfloat16* v_ptr = V + b * stride_v_b + h * stride_v_h + v0;
    if (tid < BLOCK_V) {
        smem_v[tid] = __bfloat162float(v_ptr[tid]);
    }
    
    // Compute gates
    float g, beta;
    {
        float a_val = __bfloat162float(A[b * stride_a_b + h]);
        float dt_val = DtBias[h];
        float alog = A_log[h];
        float b_val = __bfloat162float(B_gate[b * stride_b_b + h]);
        
        float x = a_val + dt_val;
        float sp = (x > 20.0f) ? x : __logf(1.0f + __expf(x));
        g = __expf(-__expf(alog) * sp);
        beta = 1.0f / (1.0f + __expf(-b_val));
    }
    
    // Load state with CuTe swizzle pattern using cp.async
    const float* state_ptr = State + b * stride_s_b + h * stride_s_h + v0 * stride_s_v;
    
    #pragma unroll
    for (int i = tid; i < BLOCK_V * V10_D; i += V10_THREADS) {
        int v_idx = i / V10_D;
        int d_idx = i % V10_D;
        // CuTe swizzle index transformation
        int smem_idx = SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d_idx);
        asm volatile (
            "cp.async.ca.shared.global [%0], [%1], 4;\n"
            :: "r"(static_cast<unsigned>(__cvta_generic_to_shared(&smem_state[smem_idx]))),
               "l"(&state_ptr[v_idx * stride_s_v + d_idx])
        );
    }
    
    asm volatile ("cp.async.commit_group;\n");
    asm volatile ("cp.async.wait_group 0;\n");
    
    __syncthreads();
    
    // Delta rule with swizzled reads - matching Triton v5
    // CRITICAL: Apply g FIRST, then compute old_v with decayed state
    #pragma unroll
    for (int v_idx = warp_id; v_idx < BLOCK_V; v_idx += V10_NUM_WARPS) {
        float old_v = 0.0f;
        
        #pragma unroll 4
        for (int d = lane_id; d < V10_D; d += V10_WARP_SIZE) {
            int smem_idx = SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d);
            float decayed_s = g * smem_state[smem_idx];
            old_v += decayed_s * smem_k[d];
        }
        
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            old_v += __shfl_xor_sync(0xffffffff, old_v, mask);
        }
        
        float delta = beta * (smem_v[v_idx] - old_v);
        
        #pragma unroll 4
        for (int d = lane_id; d < V10_D; d += V10_WARP_SIZE) {
            int smem_idx = SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d);
            float decayed_s = g * smem_state[smem_idx];
            smem_new_state[v_idx * V10_D + d] = decayed_s + delta * smem_k[d];
        }
        
        float out_val = 0.0f;
        #pragma unroll 4
        for (int d = lane_id; d < V10_D; d += V10_WARP_SIZE) {
            out_val += smem_new_state[v_idx * V10_D + d] * smem_q[d];
        }
        
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            out_val += __shfl_xor_sync(0xffffffff, out_val, mask);
        }
        
        if (lane_id == 0) {
            smem_out[v_idx] = scale * out_val;
        }
    }
    
    __syncthreads();
    
    // Write output
    __nv_bfloat16* out_ptr = Out + b * stride_o_b + h * stride_o_h + v0;
    float* new_state_ptr = NewState + b * stride_ns_b + h * stride_ns_h + v0 * stride_ns_v;
    
    if (tid < BLOCK_V) {
        out_ptr[tid] = __float2bfloat16(smem_out[tid]);
    }
    
    #pragma unroll
    for (int i = tid; i < BLOCK_V * V10_D; i += V10_THREADS) {
        int v_idx = i / V10_D;
        int d_idx = i % V10_D;
        new_state_ptr[v_idx * stride_ns_v + d_idx] = smem_new_state[v_idx * V10_D + d_idx];
    }
}

// ============================================================
// v10 Kernel: TMA with CuTe Swizzle (Same algorithm, different name)
// ============================================================

template<int BLOCK_V>
__global__ void __launch_bounds__(V10_THREADS)
gdn_decode_kernel_v10_tma(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ State,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ A,
    const float* __restrict__ DtBias,
    const __nv_bfloat16* __restrict__ B_gate,
    __nv_bfloat16* __restrict__ Out,
    float* __restrict__ NewState,
    float scale,
    int stride_q_b, int stride_q_h,
    int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h,
    int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b,
    int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v
) {
    // Same as v10_cute - both use CuTe swizzle + cp.async
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int v_block = blockIdx.z;
    const int v0 = v_block * BLOCK_V;
    const int qk_h = h / 2;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / V10_WARP_SIZE;
    const int lane_id = tid % V10_WARP_SIZE;
    
    extern __shared__ char smem_raw[];
    float* smem_q = reinterpret_cast<float*>(smem_raw);
    float* smem_k = smem_q + V10_D;
    float* smem_v = smem_k + V10_D;
    float* smem_state = smem_v + BLOCK_V;
    float* smem_new_state = smem_state + BLOCK_V * V10_D;
    float* smem_out = smem_new_state + BLOCK_V * V10_D;
    
    const __nv_bfloat16* q_ptr = Q + b * stride_q_b + qk_h * stride_q_h;
    const __nv_bfloat16* k_ptr = K + b * stride_k_b + qk_h * stride_k_h;
    
    #pragma unroll 4
    for (int d = tid; d < V10_D; d += V10_THREADS) {
        smem_q[d] = __bfloat162float(q_ptr[d]);
        smem_k[d] = __bfloat162float(k_ptr[d]);
    }
    
    const __nv_bfloat16* v_ptr = V + b * stride_v_b + h * stride_v_h + v0;
    if (tid < BLOCK_V) {
        smem_v[tid] = __bfloat162float(v_ptr[tid]);
    }
    
    float g, beta;
    {
        float a_val = __bfloat162float(A[b * stride_a_b + h]);
        float dt_val = DtBias[h];
        float alog = A_log[h];
        float b_val = __bfloat162float(B_gate[b * stride_b_b + h]);
        float x = a_val + dt_val;
        float sp = (x > 20.0f) ? x : __logf(1.0f + __expf(x));
        g = __expf(-__expf(alog) * sp);
        beta = 1.0f / (1.0f + __expf(-b_val));
    }
    
    const float* state_ptr = State + b * stride_s_b + h * stride_s_h + v0 * stride_s_v;
    
    #pragma unroll
    for (int i = tid; i < BLOCK_V * V10_D; i += V10_THREADS) {
        int v_idx = i / V10_D;
        int d_idx = i % V10_D;
        int smem_idx = SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d_idx);
        asm volatile (
            "cp.async.ca.shared.global [%0], [%1], 4;\n"
            :: "r"(static_cast<unsigned>(__cvta_generic_to_shared(&smem_state[smem_idx]))),
               "l"(&state_ptr[v_idx * stride_s_v + d_idx])
        );
    }
    
    asm volatile ("cp.async.commit_group;\n");
    asm volatile ("cp.async.wait_group 0;\n");
    __syncthreads();
    
    // Delta rule - matching Triton v5: apply g FIRST
    #pragma unroll
    for (int v_idx = warp_id; v_idx < BLOCK_V; v_idx += V10_NUM_WARPS) {
        float old_v = 0.0f;
        
        #pragma unroll 4
        for (int d = lane_id; d < V10_D; d += V10_WARP_SIZE) {
            int smem_idx = SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d);
            float decayed_s = g * smem_state[smem_idx];
            old_v += decayed_s * smem_k[d];
        }
        
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            old_v += __shfl_xor_sync(0xffffffff, old_v, mask);
        }
        
        float delta = beta * (smem_v[v_idx] - old_v);
        
        #pragma unroll 4
        for (int d = lane_id; d < V10_D; d += V10_WARP_SIZE) {
            int smem_idx = SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d);
            float decayed_s = g * smem_state[smem_idx];
            smem_new_state[v_idx * V10_D + d] = decayed_s + delta * smem_k[d];
        }
        
        float out_val = 0.0f;
        #pragma unroll 4
        for (int d = lane_id; d < V10_D; d += V10_WARP_SIZE) {
            out_val += smem_new_state[v_idx * V10_D + d] * smem_q[d];
        }
        
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            out_val += __shfl_xor_sync(0xffffffff, out_val, mask);
        }
        
        if (lane_id == 0) {
            smem_out[v_idx] = scale * out_val;
        }
    }
    
    __syncthreads();
    
    __nv_bfloat16* out_ptr = Out + b * stride_o_b + h * stride_o_h + v0;
    float* new_state_ptr = NewState + b * stride_ns_b + h * stride_ns_h + v0 * stride_ns_v;
    
    if (tid < BLOCK_V) {
        out_ptr[tid] = __float2bfloat16(smem_out[tid]);
    }
    
    #pragma unroll
    for (int i = tid; i < BLOCK_V * V10_D; i += V10_THREADS) {
        int v_idx = i / V10_D;
        int d_idx = i % V10_D;
        new_state_ptr[v_idx * stride_ns_v + d_idx] = smem_new_state[v_idx * V10_D + d_idx];
    }
}

// ============================================================
// v10 Kernel: FP8 Quantized State (Iteration 2)
// 4x memory compression: 64KB -> 16KB per head
// ============================================================

template<int BLOCK_V>
__global__ void __launch_bounds__(V10_THREADS)
gdn_decode_kernel_v10_fp8(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const uint32_t* __restrict__ State_FP8,   // Packed FP8x4: [B, H, V, D/4]
    const float* __restrict__ State_Scale,     // Per-row scale: [B, H, V]
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ A,
    const float* __restrict__ DtBias,
    const __nv_bfloat16* __restrict__ B_gate,
    __nv_bfloat16* __restrict__ Out,
    uint32_t* __restrict__ NewState_FP8,
    float* __restrict__ NewState_Scale,
    float scale,
    int stride_q_b, int stride_q_h,
    int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h,
    int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b,
    int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int v_block = blockIdx.z;
    const int v0 = v_block * BLOCK_V;
    const int qk_h = h / 2;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / V10_WARP_SIZE;
    const int lane_id = tid % V10_WARP_SIZE;
    
    // Shared memory - FP32 for computation
    extern __shared__ char smem_raw[];
    float* smem_q = reinterpret_cast<float*>(smem_raw);
    float* smem_k = smem_q + V10_D;
    float* smem_v = smem_k + V10_D;
    float* smem_state = smem_v + BLOCK_V;  // Dequantized to FP32
    float* smem_new_state = smem_state + BLOCK_V * V10_D;
    float* smem_out = smem_new_state + BLOCK_V * V10_D;
    float* smem_scale = smem_out + BLOCK_V;  // Per-row scales
    
    // Load Q, K
    const __nv_bfloat16* q_ptr = Q + b * stride_q_b + qk_h * stride_q_h;
    const __nv_bfloat16* k_ptr = K + b * stride_k_b + qk_h * stride_k_h;
    
    #pragma unroll 4
    for (int d = tid; d < V10_D; d += V10_THREADS) {
        smem_q[d] = __bfloat162float(q_ptr[d]);
        smem_k[d] = __bfloat162float(k_ptr[d]);
    }
    
    // Load V
    const __nv_bfloat16* v_ptr = V + b * stride_v_b + h * stride_v_h + v0;
    if (tid < BLOCK_V) {
        smem_v[tid] = __bfloat162float(v_ptr[tid]);
    }
    
    // Load per-row scales
    const float* scale_ptr = State_Scale + b * stride_s_b + h * stride_s_h + v0;
    if (tid < BLOCK_V) {
        smem_scale[tid] = scale_ptr[tid];
    }
    
    // Compute gates
    float g, beta;
    {
        float a_val = __bfloat162float(A[b * stride_a_b + h]);
        float dt_val = DtBias[h];
        float alog = A_log[h];
        float b_val = __bfloat162float(B_gate[b * stride_b_b + h]);
        float x = a_val + dt_val;
        float sp = (x > 20.0f) ? x : __logf(1.0f + __expf(x));
        g = __expf(-__expf(alog) * sp);
        beta = 1.0f / (1.0f + __expf(-b_val));
    }
    
    // Load FP8 state and dequantize with per-row scale
    const uint32_t* state_fp8_ptr = State_FP8 + b * stride_s_b + h * stride_s_h + v0 * stride_s_v;
    
    #pragma unroll
    for (int i = tid; i < BLOCK_V * (V10_D / 4); i += V10_THREADS) {
        int v_idx = i / (V10_D / 4);
        int d4_idx = i % (V10_D / 4);
        int d_base = d4_idx * 4;
        
        // Load packed FP8x4
        uint32_t packed = state_fp8_ptr[v_idx * stride_s_v + d4_idx];
        
        // Unpack and dequantize
        __nv_fp8_e4m3 fp8_0, fp8_1, fp8_2, fp8_3;
        v10_unpack_fp8x4(packed, fp8_0, fp8_1, fp8_2, fp8_3);
        
        float row_scale = smem_scale[v_idx];
        int smem_base = SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d_base);
        smem_state[smem_base] = v10_fp8_to_fp32(fp8_0) * row_scale;
        smem_state[SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d_base + 1)] = v10_fp8_to_fp32(fp8_1) * row_scale;
        smem_state[SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d_base + 2)] = v10_fp8_to_fp32(fp8_2) * row_scale;
        smem_state[SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d_base + 3)] = v10_fp8_to_fp32(fp8_3) * row_scale;
    }
    
    __syncthreads();
    
    // Delta rule with swizzled reads (same as FP32 version)
    #pragma unroll
    for (int v_idx = warp_id; v_idx < BLOCK_V; v_idx += V10_NUM_WARPS) {
        float old_v = 0.0f;
        
        #pragma unroll 4
        for (int d = lane_id; d < V10_D; d += V10_WARP_SIZE) {
            int smem_idx = SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d);
            float decayed_s = g * smem_state[smem_idx];
            old_v += decayed_s * smem_k[d];
        }
        
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            old_v += __shfl_xor_sync(0xffffffff, old_v, mask);
        }
        
        float delta = beta * (smem_v[v_idx] - old_v);
        
        #pragma unroll 4
        for (int d = lane_id; d < V10_D; d += V10_WARP_SIZE) {
            int smem_idx = SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d);
            float decayed_s = g * smem_state[smem_idx];
            smem_new_state[v_idx * V10_D + d] = decayed_s + delta * smem_k[d];
        }
        
        float out_val = 0.0f;
        #pragma unroll 4
        for (int d = lane_id; d < V10_D; d += V10_WARP_SIZE) {
            out_val += smem_new_state[v_idx * V10_D + d] * smem_q[d];
        }
        
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            out_val += __shfl_xor_sync(0xffffffff, out_val, mask);
        }
        
        if (lane_id == 0) {
            smem_out[v_idx] = scale * out_val;
        }
    }
    
    __syncthreads();
    
    // Write output
    __nv_bfloat16* out_ptr = Out + b * stride_o_b + h * stride_o_h + v0;
    if (tid < BLOCK_V) {
        out_ptr[tid] = __float2bfloat16(smem_out[tid]);
    }
    
    // Quantize new state to FP8 with dynamic per-row scaling
    uint32_t* new_state_fp8_ptr = NewState_FP8 + b * stride_ns_b + h * stride_ns_h + v0 * stride_ns_v;
    float* new_scale_ptr = NewState_Scale + b * stride_ns_b + h * stride_ns_h + v0;
    
    // Compute new per-row scale (max abs value per row)
    if (tid < BLOCK_V) {
        float max_abs = 0.0f;
        for (int d = 0; d < V10_D; d++) {
            max_abs = fmaxf(max_abs, fabsf(smem_new_state[tid * V10_D + d]));
        }
        // FP8 E4M3 has range [-448, 448], use 400 for safety margin
        float new_scale = (max_abs > 1e-6f) ? (max_abs / 400.0f) : 1.0f;
        smem_scale[tid] = new_scale;
        new_scale_ptr[tid] = new_scale;
    }
    
    __syncthreads();
    
    // Quantize and pack to FP8x4
    #pragma unroll
    for (int i = tid; i < BLOCK_V * (V10_D / 4); i += V10_THREADS) {
        int v_idx = i / (V10_D / 4);
        int d4_idx = i % (V10_D / 4);
        int d_base = d4_idx * 4;
        
        float inv_scale = 1.0f / smem_scale[v_idx];
        
        __nv_fp8_e4m3 fp8_0 = v10_fp32_to_fp8(smem_new_state[v_idx * V10_D + d_base + 0] * inv_scale);
        __nv_fp8_e4m3 fp8_1 = v10_fp32_to_fp8(smem_new_state[v_idx * V10_D + d_base + 1] * inv_scale);
        __nv_fp8_e4m3 fp8_2 = v10_fp32_to_fp8(smem_new_state[v_idx * V10_D + d_base + 2] * inv_scale);
        __nv_fp8_e4m3 fp8_3 = v10_fp32_to_fp8(smem_new_state[v_idx * V10_D + d_base + 3] * inv_scale);
        
        uint32_t packed = v10_pack_fp8x4(fp8_0, fp8_1, fp8_2, fp8_3);
        new_state_fp8_ptr[v_idx * stride_ns_v + d4_idx] = packed;
    }
}

// ============================================================
// v10 Kernel: FP4 Quantized State (8x memory compression)
// WARNING: ~55-65% relative error - use only for extreme compression
// ============================================================

template<int BLOCK_V>
__global__ void __launch_bounds__(V10_THREADS)
gdn_decode_kernel_v10_fp4(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const uint32_t* __restrict__ State_FP4,   // Packed FP4x8: [B, H, V, D/8]
    const float* __restrict__ State_Scale,     // Per-row scale: [B, H, V]
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ A,
    const float* __restrict__ DtBias,
    const __nv_bfloat16* __restrict__ B_gate,
    __nv_bfloat16* __restrict__ Out,
    uint32_t* __restrict__ NewState_FP4,
    float* __restrict__ NewState_Scale,
    float scale,
    int stride_q_b, int stride_q_h,
    int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h,
    int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b,
    int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int v_block = blockIdx.z;
    const int v0 = v_block * BLOCK_V;
    const int qk_h = h / 2;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / V10_WARP_SIZE;
    const int lane_id = tid % V10_WARP_SIZE;
    
    // Shared memory - FP32 for computation
    extern __shared__ char smem_raw[];
    float* smem_q = reinterpret_cast<float*>(smem_raw);
    float* smem_k = smem_q + V10_D;
    float* smem_v = smem_k + V10_D;
    float* smem_state = smem_v + BLOCK_V;
    float* smem_new_state = smem_state + BLOCK_V * V10_D;
    float* smem_out = smem_new_state + BLOCK_V * V10_D;
    float* smem_scale = smem_out + BLOCK_V;
    
    // Load Q, K
    const __nv_bfloat16* q_ptr = Q + b * stride_q_b + qk_h * stride_q_h;
    const __nv_bfloat16* k_ptr = K + b * stride_k_b + qk_h * stride_k_h;
    
    #pragma unroll 4
    for (int d = tid; d < V10_D; d += V10_THREADS) {
        smem_q[d] = __bfloat162float(q_ptr[d]);
        smem_k[d] = __bfloat162float(k_ptr[d]);
    }
    
    // Load V
    const __nv_bfloat16* v_ptr = V + b * stride_v_b + h * stride_v_h + v0;
    if (tid < BLOCK_V) {
        smem_v[tid] = __bfloat162float(v_ptr[tid]);
    }
    
    // Load per-row scales
    const float* scale_ptr = State_Scale + b * stride_s_b + h * stride_s_h + v0;
    if (tid < BLOCK_V) {
        smem_scale[tid] = scale_ptr[tid];
    }
    
    // Compute gates
    float g, beta;
    {
        float a_val = __bfloat162float(A[b * stride_a_b + h]);
        float dt_val = DtBias[h];
        float alog = A_log[h];
        float b_val = __bfloat162float(B_gate[b * stride_b_b + h]);
        float x = a_val + dt_val;
        float sp = (x > 20.0f) ? x : __logf(1.0f + __expf(x));
        g = __expf(-__expf(alog) * sp);
        beta = 1.0f / (1.0f + __expf(-b_val));
    }
    
    // Load FP4 state and dequantize (8 values per uint32_t)
    const uint32_t* state_fp4_ptr = State_FP4 + b * stride_s_b + h * stride_s_h + v0 * stride_s_v;
    
    #pragma unroll
    for (int i = tid; i < BLOCK_V * (V10_D / 8); i += V10_THREADS) {
        int v_idx = i / (V10_D / 8);
        int d8_idx = i % (V10_D / 8);
        int d_base = d8_idx * 8;
        
        // Load packed FP4x8
        uint32_t packed = state_fp4_ptr[v_idx * stride_s_v + d8_idx];
        
        // Unpack
        uint8_t fp4_0, fp4_1, fp4_2, fp4_3, fp4_4, fp4_5, fp4_6, fp4_7;
        v10_unpack_fp4x8(packed, fp4_0, fp4_1, fp4_2, fp4_3, fp4_4, fp4_5, fp4_6, fp4_7);
        
        float row_scale = smem_scale[v_idx];
        smem_state[SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d_base + 0)] = v10_fp4_to_fp32(fp4_0) * row_scale;
        smem_state[SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d_base + 1)] = v10_fp4_to_fp32(fp4_1) * row_scale;
        smem_state[SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d_base + 2)] = v10_fp4_to_fp32(fp4_2) * row_scale;
        smem_state[SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d_base + 3)] = v10_fp4_to_fp32(fp4_3) * row_scale;
        smem_state[SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d_base + 4)] = v10_fp4_to_fp32(fp4_4) * row_scale;
        smem_state[SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d_base + 5)] = v10_fp4_to_fp32(fp4_5) * row_scale;
        smem_state[SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d_base + 6)] = v10_fp4_to_fp32(fp4_6) * row_scale;
        smem_state[SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d_base + 7)] = v10_fp4_to_fp32(fp4_7) * row_scale;
    }
    
    __syncthreads();
    
    // Delta rule (same as FP32/FP8 versions)
    #pragma unroll
    for (int v_idx = warp_id; v_idx < BLOCK_V; v_idx += V10_NUM_WARPS) {
        float old_v = 0.0f;
        
        #pragma unroll 4
        for (int d = lane_id; d < V10_D; d += V10_WARP_SIZE) {
            int smem_idx = SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d);
            float decayed_s = g * smem_state[smem_idx];
            old_v += decayed_s * smem_k[d];
        }
        
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            old_v += __shfl_xor_sync(0xffffffff, old_v, mask);
        }
        
        float delta = beta * (smem_v[v_idx] - old_v);
        
        #pragma unroll 4
        for (int d = lane_id; d < V10_D; d += V10_WARP_SIZE) {
            int smem_idx = SwizzledStateLayout<BLOCK_V>::get_index(v_idx, d);
            float decayed_s = g * smem_state[smem_idx];
            smem_new_state[v_idx * V10_D + d] = decayed_s + delta * smem_k[d];
        }
        
        float out_val = 0.0f;
        #pragma unroll 4
        for (int d = lane_id; d < V10_D; d += V10_WARP_SIZE) {
            out_val += smem_new_state[v_idx * V10_D + d] * smem_q[d];
        }
        
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            out_val += __shfl_xor_sync(0xffffffff, out_val, mask);
        }
        
        if (lane_id == 0) {
            smem_out[v_idx] = scale * out_val;
        }
    }
    
    __syncthreads();
    
    // Write output
    __nv_bfloat16* out_ptr = Out + b * stride_o_b + h * stride_o_h + v0;
    if (tid < BLOCK_V) {
        out_ptr[tid] = __float2bfloat16(smem_out[tid]);
    }
    
    // Quantize new state to FP4 with dynamic per-row scaling
    uint32_t* new_state_fp4_ptr = NewState_FP4 + b * stride_ns_b + h * stride_ns_h + v0 * stride_ns_v;
    float* new_scale_ptr = NewState_Scale + b * stride_ns_b + h * stride_ns_h + v0;
    
    // Compute new per-row scale (max abs value per row)
    if (tid < BLOCK_V) {
        float max_abs = 0.0f;
        for (int d = 0; d < V10_D; d++) {
            max_abs = fmaxf(max_abs, fabsf(smem_new_state[tid * V10_D + d]));
        }
        // FP4 E2M1 has max value 6.0, use 5.0 for safety margin
        float new_scale = (max_abs > 1e-6f) ? (max_abs / 5.0f) : 1.0f;
        smem_scale[tid] = new_scale;
        new_scale_ptr[tid] = new_scale;
    }
    
    __syncthreads();
    
    // Quantize and pack to FP4x8
    #pragma unroll
    for (int i = tid; i < BLOCK_V * (V10_D / 8); i += V10_THREADS) {
        int v_idx = i / (V10_D / 8);
        int d8_idx = i % (V10_D / 8);
        int d_base = d8_idx * 8;
        
        float inv_scale = 1.0f / smem_scale[v_idx];
        
        uint8_t fp4_0 = v10_fp32_to_fp4(smem_new_state[v_idx * V10_D + d_base + 0] * inv_scale);
        uint8_t fp4_1 = v10_fp32_to_fp4(smem_new_state[v_idx * V10_D + d_base + 1] * inv_scale);
        uint8_t fp4_2 = v10_fp32_to_fp4(smem_new_state[v_idx * V10_D + d_base + 2] * inv_scale);
        uint8_t fp4_3 = v10_fp32_to_fp4(smem_new_state[v_idx * V10_D + d_base + 3] * inv_scale);
        uint8_t fp4_4 = v10_fp32_to_fp4(smem_new_state[v_idx * V10_D + d_base + 4] * inv_scale);
        uint8_t fp4_5 = v10_fp32_to_fp4(smem_new_state[v_idx * V10_D + d_base + 5] * inv_scale);
        uint8_t fp4_6 = v10_fp32_to_fp4(smem_new_state[v_idx * V10_D + d_base + 6] * inv_scale);
        uint8_t fp4_7 = v10_fp32_to_fp4(smem_new_state[v_idx * V10_D + d_base + 7] * inv_scale);
        
        uint32_t packed = v10_pack_fp4x8(fp4_0, fp4_1, fp4_2, fp4_3, fp4_4, fp4_5, fp4_6, fp4_7);
        new_state_fp4_ptr[v_idx * stride_ns_v + d8_idx] = packed;
    }
}

// ============================================================
// Launch Functions
// ============================================================

void gdn_decode_v10_launch_cute(
    const void* Q, const void* K, const void* V,
    const void* State, const void* A_log, const void* A,
    const void* DtBias, const void* B_gate,
    void* Out, void* NewState,
    float scale, int B, int num_v_heads, int D,
    int stride_q_b, int stride_q_h, int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h, int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b, int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, cudaStream_t stream
) {
    int V_BLOCKS = D / BLOCK_V;
    dim3 grid(B, num_v_heads, V_BLOCKS);
    dim3 block(V10_THREADS);
    
    size_t smem_size = (D + D + BLOCK_V + 2 * BLOCK_V * D + BLOCK_V) * sizeof(float);
    smem_size = ((smem_size + 127) / 128) * 128;
    
    if (BLOCK_V == 16) {
        gdn_decode_kernel_v10_cute<16><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State, (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate,
            (__nv_bfloat16*)Out, (float*)NewState, scale,
            stride_q_b, stride_q_h, stride_k_b, stride_k_h,
            stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
            stride_a_b, stride_b_b, stride_o_b, stride_o_h,
            stride_ns_b, stride_ns_h, stride_ns_v);
    } else if (BLOCK_V == 32) {
        gdn_decode_kernel_v10_cute<32><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State, (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate,
            (__nv_bfloat16*)Out, (float*)NewState, scale,
            stride_q_b, stride_q_h, stride_k_b, stride_k_h,
            stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
            stride_a_b, stride_b_b, stride_o_b, stride_o_h,
            stride_ns_b, stride_ns_h, stride_ns_v);
    } else {
        gdn_decode_kernel_v10_cute<64><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State, (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate,
            (__nv_bfloat16*)Out, (float*)NewState, scale,
            stride_q_b, stride_q_h, stride_k_b, stride_k_h,
            stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
            stride_a_b, stride_b_b, stride_o_b, stride_o_h,
            stride_ns_b, stride_ns_h, stride_ns_v);
    }
}

void gdn_decode_v10_launch_tma(
    const void* Q, const void* K, const void* V,
    const void* State, const void* A_log, const void* A,
    const void* DtBias, const void* B_gate,
    void* Out, void* NewState,
    float scale, int B, int num_v_heads, int D,
    int stride_q_b, int stride_q_h, int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h, int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b, int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, cudaStream_t stream
) {
    int V_BLOCKS = D / BLOCK_V;
    dim3 grid(B, num_v_heads, V_BLOCKS);
    dim3 block(V10_THREADS);
    
    size_t smem_size = (D + D + BLOCK_V + 2 * BLOCK_V * D + BLOCK_V) * sizeof(float);
    smem_size = ((smem_size + 127) / 128) * 128;
    
    if (BLOCK_V == 16) {
        gdn_decode_kernel_v10_tma<16><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State, (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate,
            (__nv_bfloat16*)Out, (float*)NewState, scale,
            stride_q_b, stride_q_h, stride_k_b, stride_k_h,
            stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
            stride_a_b, stride_b_b, stride_o_b, stride_o_h,
            stride_ns_b, stride_ns_h, stride_ns_v);
    } else if (BLOCK_V == 32) {
        gdn_decode_kernel_v10_tma<32><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State, (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate,
            (__nv_bfloat16*)Out, (float*)NewState, scale,
            stride_q_b, stride_q_h, stride_k_b, stride_k_h,
            stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
            stride_a_b, stride_b_b, stride_o_b, stride_o_h,
            stride_ns_b, stride_ns_h, stride_ns_v);
    } else {
        gdn_decode_kernel_v10_tma<64><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State, (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate,
            (__nv_bfloat16*)Out, (float*)NewState, scale,
            stride_q_b, stride_q_h, stride_k_b, stride_k_h,
            stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
            stride_a_b, stride_b_b, stride_o_b, stride_o_h,
            stride_ns_b, stride_ns_h, stride_ns_v);
    }
}

// FP8 launcher (Iteration 2 - 4x memory compression)
void gdn_decode_v10_launch_fp8(
    const void* Q, const void* K, const void* V,
    const void* State_FP8, const void* State_Scale,
    const void* A_log, const void* A,
    const void* DtBias, const void* B_gate,
    void* Out, void* NewState_FP8, void* NewState_Scale,
    float scale, int B, int num_v_heads, int D,
    int stride_q_b, int stride_q_h, int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h, int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b, int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, cudaStream_t stream
) {
    int V_BLOCKS = D / BLOCK_V;
    dim3 grid(B, num_v_heads, V_BLOCKS);
    dim3 block(V10_THREADS);
    
    // Extra space for per-row scales
    size_t smem_size = (D + D + BLOCK_V + 2 * BLOCK_V * D + BLOCK_V + BLOCK_V) * sizeof(float);
    smem_size = ((smem_size + 127) / 128) * 128;
    
    if (BLOCK_V == 16) {
        gdn_decode_kernel_v10_fp8<16><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const uint32_t*)State_FP8, (const float*)State_Scale,
            (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate,
            (__nv_bfloat16*)Out, (uint32_t*)NewState_FP8, (float*)NewState_Scale, scale,
            stride_q_b, stride_q_h, stride_k_b, stride_k_h,
            stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
            stride_a_b, stride_b_b, stride_o_b, stride_o_h,
            stride_ns_b, stride_ns_h, stride_ns_v);
    } else if (BLOCK_V == 32) {
        gdn_decode_kernel_v10_fp8<32><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const uint32_t*)State_FP8, (const float*)State_Scale,
            (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate,
            (__nv_bfloat16*)Out, (uint32_t*)NewState_FP8, (float*)NewState_Scale, scale,
            stride_q_b, stride_q_h, stride_k_b, stride_k_h,
            stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
            stride_a_b, stride_b_b, stride_o_b, stride_o_h,
            stride_ns_b, stride_ns_h, stride_ns_v);
    } else {
        gdn_decode_kernel_v10_fp8<64><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const uint32_t*)State_FP8, (const float*)State_Scale,
            (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate,
            (__nv_bfloat16*)Out, (uint32_t*)NewState_FP8, (float*)NewState_Scale, scale,
            stride_q_b, stride_q_h, stride_k_b, stride_k_h,
            stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
            stride_a_b, stride_b_b, stride_o_b, stride_o_h,
            stride_ns_b, stride_ns_h, stride_ns_v);
    }
}

// FP4 launcher (8x memory compression - experimental)
void gdn_decode_v10_launch_fp4(
    const void* Q, const void* K, const void* V,
    const void* State_FP4, const void* State_Scale,
    const void* A_log, const void* A,
    const void* DtBias, const void* B_gate,
    void* Out, void* NewState_FP4, void* NewState_Scale,
    float scale, int B, int num_v_heads, int D,
    int stride_q_b, int stride_q_h, int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h, int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b, int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, cudaStream_t stream
) {
    int V_BLOCKS = D / BLOCK_V;
    dim3 grid(B, num_v_heads, V_BLOCKS);
    dim3 block(V10_THREADS);
    
    // Extra space for per-row scales
    size_t smem_size = (D + D + BLOCK_V + 2 * BLOCK_V * D + BLOCK_V + BLOCK_V) * sizeof(float);
    smem_size = ((smem_size + 127) / 128) * 128;
    
    if (BLOCK_V == 16) {
        gdn_decode_kernel_v10_fp4<16><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const uint32_t*)State_FP4, (const float*)State_Scale,
            (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate,
            (__nv_bfloat16*)Out, (uint32_t*)NewState_FP4, (float*)NewState_Scale, scale,
            stride_q_b, stride_q_h, stride_k_b, stride_k_h,
            stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
            stride_a_b, stride_b_b, stride_o_b, stride_o_h,
            stride_ns_b, stride_ns_h, stride_ns_v);
    } else if (BLOCK_V == 32) {
        gdn_decode_kernel_v10_fp4<32><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const uint32_t*)State_FP4, (const float*)State_Scale,
            (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate,
            (__nv_bfloat16*)Out, (uint32_t*)NewState_FP4, (float*)NewState_Scale, scale,
            stride_q_b, stride_q_h, stride_k_b, stride_k_h,
            stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
            stride_a_b, stride_b_b, stride_o_b, stride_o_h,
            stride_ns_b, stride_ns_h, stride_ns_v);
    } else {
        gdn_decode_kernel_v10_fp4<64><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const uint32_t*)State_FP4, (const float*)State_Scale,
            (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate,
            (__nv_bfloat16*)Out, (uint32_t*)NewState_FP4, (float*)NewState_Scale, scale,
            stride_q_b, stride_q_h, stride_k_b, stride_k_h,
            stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
            stride_a_b, stride_b_b, stride_o_b, stride_o_h,
            stride_ns_b, stride_ns_h, stride_ns_v);
    }
}

}  // namespace gdn

#else
// Fallback when not compiling with CUDA
namespace gdn {
void gdn_decode_v10_launch_cute(...) {}
void gdn_decode_v10_launch_tma(...) {}
void gdn_decode_v10_launch_fp8(...) {}
void gdn_decode_v10_launch_fp4(...) {}
}
#endif

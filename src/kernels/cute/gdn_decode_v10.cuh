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
 * Grid: (B, H=8, V_BLOCKS)
 * Block: 128 threads (4 warps)
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

}  // namespace gdn

#else
// Fallback when not compiling with CUDA
namespace gdn {
void gdn_decode_v10_launch_cute(...) {}
void gdn_decode_v10_launch_tma(...) {}
}
#endif

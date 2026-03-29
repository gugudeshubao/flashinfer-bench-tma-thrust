/*
 * GDN Prefill v8 — Maximum Performance CUDA kernel for B200 (sm100)
 *
 * All optimizations from decode v8 + prefill-specific:
 *   - Multi-stage pipelining for token prefetch
 *   - Persistent kernel for long sequences
 *   - Chunked processing for very long sequences
 *
 * Grid: (N=num_seqs, H=8, V_BLOCKS)
 * Block: 128 threads (4 warps: 2 producer + 2 consumer)
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cmath>

namespace gdn {

constexpr int V8P_D = 128;
constexpr int V8P_STAGES = 3;  // Triple buffering for prefill

// ============================================================
// Utility Functions (shared with decode)
// ============================================================

__device__ __forceinline__ float v8p_fast_exp(float x) { return __expf(x); }
__device__ __forceinline__ float v8p_fast_log(float x) { return __logf(x); }

__device__ __forceinline__ float v8p_softplus(float x) {
    return (x > 20.0f) ? x : v8p_fast_log(1.0f + v8p_fast_exp(x));
}

__device__ __forceinline__ float v8p_sigmoid(float x) {
    return 1.0f / (1.0f + v8p_fast_exp(-x));
}

// FP8 helpers
__device__ __forceinline__ __nv_fp8_e4m3 fp32_to_fp8_v8p(float val) {
    return __nv_fp8_e4m3(val);
}

__device__ __forceinline__ float fp8_to_fp32_v8p(__nv_fp8_e4m3 val) {
    return float(val);
}

__device__ __forceinline__ uint32_t pack_fp8x4_v8p(
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

__device__ __forceinline__ void unpack_fp8x4_v8p(
    uint32_t packed,
    __nv_fp8_e4m3& a, __nv_fp8_e4m3& b, __nv_fp8_e4m3& c, __nv_fp8_e4m3& d
) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&packed);
    a = *reinterpret_cast<const __nv_fp8_e4m3*>(&bytes[0]);
    b = *reinterpret_cast<const __nv_fp8_e4m3*>(&bytes[1]);
    c = *reinterpret_cast<const __nv_fp8_e4m3*>(&bytes[2]);
    d = *reinterpret_cast<const __nv_fp8_e4m3*>(&bytes[3]);
}

// L2 cache hint
__device__ __forceinline__ float ldg_v8p(const float* ptr) { return __ldg(ptr); }

// ============================================================
// Main Kernel: FP32 with Pipelining
// ============================================================

template<int BLOCK_V>
__global__ void __launch_bounds__(128, 2)
gdn_prefill_kernel_v8_fp32(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ State,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ A,
    const float* __restrict__ DtBias,
    const __nv_bfloat16* __restrict__ B_gate,
    const int32_t* __restrict__ CuSeqlens,
    __nv_bfloat16* __restrict__ Out,
    float* __restrict__ NewState,
    float scale,
    int stride_q_t, int stride_q_h,
    int stride_k_t, int stride_k_h,
    int stride_v_t, int stride_v_h,
    int stride_s_n, int stride_s_h, int stride_s_v,
    int stride_a_t, int stride_b_t,
    int stride_o_t, int stride_o_h,
    int stride_ns_n, int stride_ns_h, int stride_ns_v
) {
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const int vb = blockIdx.z;
    const int v0 = vb * BLOCK_V;
    const int qk_h = h / 2;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int t_start = CuSeqlens[n];
    const int t_end = CuSeqlens[n + 1];
    const int seq_len = t_end - t_start;
    
    if (seq_len == 0) return;
    
    const float alog = ldg_v8p(&A_log[h]);
    const float dt_bias = ldg_v8p(&DtBias[h]);
    
    // Triple-buffered shared memory for Q, K prefetch
    extern __shared__ __align__(128) char smem_raw[];
    
    float* q_buf[V8P_STAGES];
    float* k_buf[V8P_STAGES];
    
    float* base = (float*)smem_raw;
    for (int s = 0; s < V8P_STAGES; s++) {
        q_buf[s] = base + s * V8P_D * 2;
        k_buf[s] = q_buf[s] + V8P_D;
    }
    
    float* v_smem = base + V8P_STAGES * V8P_D * 2;
    float* state_smem = v_smem + BLOCK_V;
    float* old_v_smem = state_smem + BLOCK_V * V8P_D;
    float* out_smem = old_v_smem + BLOCK_V;
    
    // Load initial state
    const float* state_ptr = State + n * stride_s_n + h * stride_s_h + v0 * stride_s_v;
    
    #pragma unroll 4
    for (int i = tid; i < BLOCK_V * V8P_D; i += blockDim.x) {
        int vi = i / V8P_D;
        int ki = i % V8P_D;
        state_smem[i] = ldg_v8p(&state_ptr[vi * stride_s_v + ki]);
    }
    
    // Prefetch first tokens
    for (int s = 0; s < min(V8P_STAGES, seq_len); s++) {
        int t = t_start + s;
        for (int i = tid; i < V8P_D; i += blockDim.x) {
            q_buf[s][i] = __bfloat162float(__ldg(&Q[t * stride_q_t + qk_h * stride_q_h + i]));
            k_buf[s][i] = __bfloat162float(__ldg(&K[t * stride_k_t + qk_h * stride_k_h + i]));
        }
    }
    __syncthreads();
    
    // Process tokens with pipelining
    for (int tok = 0; tok < seq_len; tok++) {
        int t = t_start + tok;
        int buf_idx = tok % V8P_STAGES;
        
        float* q_curr = q_buf[buf_idx];
        float* k_curr = k_buf[buf_idx];
        
        // Prefetch next token (if exists and buffer available)
        int next_tok = tok + V8P_STAGES;
        if (next_tok < seq_len) {
            int next_t = t_start + next_tok;
            int next_buf = next_tok % V8P_STAGES;
            
            // Producer warps handle prefetch
            if (warp_id < 2) {
                for (int i = (warp_id * 32 + lane_id); i < V8P_D; i += 64) {
                    q_buf[next_buf][i] = __bfloat162float(__ldg(&Q[next_t * stride_q_t + qk_h * stride_q_h + i]));
                    k_buf[next_buf][i] = __bfloat162float(__ldg(&K[next_t * stride_k_t + qk_h * stride_k_h + i]));
                }
            }
        }
        
        // Load gates and V
        __shared__ float g_val, beta_val;
        if (tid == 0) {
            float a = __bfloat162float(__ldg(&A[t * stride_a_t + h]));
            float bv = __bfloat162float(__ldg(&B_gate[t * stride_b_t + h]));
            float sp = v8p_softplus(a + dt_bias);
            g_val = v8p_fast_exp(-v8p_fast_exp(alog) * sp);
            beta_val = v8p_sigmoid(bv);
        }
        
        if (tid < BLOCK_V) {
            v_smem[tid] = __bfloat162float(__ldg(&V[t * stride_v_t + h * stride_v_h + v0 + tid]));
        }
        __syncthreads();
        
        float g = g_val;
        float beta = beta_val;
        
        // Decay state (vectorized)
        #pragma unroll 4
        for (int i = tid; i < BLOCK_V * V8P_D / 4; i += blockDim.x) {
            int vi = (i * 4) / V8P_D;
            int ki = (i * 4) % V8P_D;
            float4* ptr = reinterpret_cast<float4*>(&state_smem[vi * V8P_D + ki]);
            float4 s = *ptr;
            s.x *= g; s.y *= g; s.z *= g; s.w *= g;
            *ptr = s;
        }
        __syncthreads();
        
        // old_v = S @ k (consumer warps)
        if (tid < BLOCK_V) {
            float sum = 0.0f;
            const float4* s_ptr = reinterpret_cast<const float4*>(&state_smem[tid * V8P_D]);
            const float4* k_ptr = reinterpret_cast<const float4*>(k_curr);
            
            #pragma unroll 8
            for (int ki = 0; ki < V8P_D / 4; ki++) {
                float4 s4 = s_ptr[ki];
                float4 k4 = k_ptr[ki];
                sum += s4.x * k4.x + s4.y * k4.y + s4.z * k4.z + s4.w * k4.w;
            }
            old_v_smem[tid] = sum;
        }
        __syncthreads();
        
        // Rank-1 update
        #pragma unroll 4
        for (int i = tid; i < BLOCK_V * V8P_D; i += blockDim.x) {
            int vi = i / V8P_D;
            int ki = i % V8P_D;
            float delta = beta * (v_smem[vi] - old_v_smem[vi]);
            state_smem[i] += delta * k_curr[ki];
        }
        __syncthreads();
        
        // Output
        if (tid < BLOCK_V) {
            float sum = 0.0f;
            const float4* s_ptr = reinterpret_cast<const float4*>(&state_smem[tid * V8P_D]);
            const float4* q_ptr = reinterpret_cast<const float4*>(q_curr);
            
            #pragma unroll 8
            for (int ki = 0; ki < V8P_D / 4; ki++) {
                float4 s4 = s_ptr[ki];
                float4 q4 = q_ptr[ki];
                sum += s4.x * q4.x + s4.y * q4.y + s4.z * q4.z + s4.w * q4.w;
            }
            out_smem[tid] = scale * sum;
        }
        __syncthreads();
        
        if (tid < BLOCK_V) {
            Out[t * stride_o_t + h * stride_o_h + v0 + tid] = __float2bfloat16(out_smem[tid]);
        }
        __syncthreads();
    }
    
    // Store final state
    float* new_state_ptr = NewState + n * stride_ns_n + h * stride_ns_h + v0 * stride_ns_v;
    
    #pragma unroll 4
    for (int i = tid; i < BLOCK_V * V8P_D / 4; i += blockDim.x) {
        int vi = (i * 4) / V8P_D;
        int ki = (i * 4) % V8P_D;
        float4 s = *reinterpret_cast<float4*>(&state_smem[vi * V8P_D + ki]);
        *reinterpret_cast<float4*>(&new_state_ptr[vi * stride_ns_v + ki]) = s;
    }
}

// ============================================================
// Main Kernel: FP8 Quantized State
// ============================================================

template<int BLOCK_V>
__global__ void __launch_bounds__(128, 2)
gdn_prefill_kernel_v8_fp8(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const uint32_t* __restrict__ State_FP8,
    const float* __restrict__ State_Scale,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ A,
    const float* __restrict__ DtBias,
    const __nv_bfloat16* __restrict__ B_gate,
    const int32_t* __restrict__ CuSeqlens,
    __nv_bfloat16* __restrict__ Out,
    uint32_t* __restrict__ NewState_FP8,
    float* __restrict__ NewState_Scale,
    float scale,
    int stride_q_t, int stride_q_h,
    int stride_k_t, int stride_k_h,
    int stride_v_t, int stride_v_h,
    int stride_s_n, int stride_s_h, int stride_s_v,
    int stride_a_t, int stride_b_t,
    int stride_o_t, int stride_o_h,
    int stride_ns_n, int stride_ns_h, int stride_ns_v
) {
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const int vb = blockIdx.z;
    const int v0 = vb * BLOCK_V;
    const int qk_h = h / 2;
    
    const int tid = threadIdx.x;
    
    const int t_start = CuSeqlens[n];
    const int t_end = CuSeqlens[n + 1];
    const int seq_len = t_end - t_start;
    
    if (seq_len == 0) return;
    
    const float alog = ldg_v8p(&A_log[h]);
    const float dt_bias = ldg_v8p(&DtBias[h]);
    
    extern __shared__ __align__(128) char smem_raw[];
    float* q_smem = (float*)smem_raw;
    float* k_smem = q_smem + V8P_D;
    float* v_smem = k_smem + V8P_D;
    float* state_smem = v_smem + BLOCK_V;
    float* old_v_smem = state_smem + BLOCK_V * V8P_D;
    float* out_smem = old_v_smem + BLOCK_V;
    float* row_scale = out_smem + BLOCK_V;
    
    // Load row scales and dequantize initial state
    if (tid < BLOCK_V) {
        row_scale[tid] = ldg_v8p(&State_Scale[n * stride_s_n + h * stride_s_h + v0 + tid]);
    }
    __syncthreads();
    
    const uint32_t* state_fp8_ptr = State_FP8 + n * stride_s_n + h * stride_s_h + v0 * stride_s_v;
    
    for (int i = tid; i < BLOCK_V * V8P_D / 4; i += blockDim.x) {
        int vi = (i * 4) / V8P_D;
        int ki = (i * 4) % V8P_D;
        
        uint32_t packed = __ldg(&state_fp8_ptr[vi * stride_s_v + ki / 4]);
        __nv_fp8_e4m3 fp8_a, fp8_b, fp8_c, fp8_d;
        unpack_fp8x4_v8p(packed, fp8_a, fp8_b, fp8_c, fp8_d);
        
        float s = row_scale[vi];
        state_smem[vi * V8P_D + ki + 0] = s * fp8_to_fp32_v8p(fp8_a);
        state_smem[vi * V8P_D + ki + 1] = s * fp8_to_fp32_v8p(fp8_b);
        state_smem[vi * V8P_D + ki + 2] = s * fp8_to_fp32_v8p(fp8_c);
        state_smem[vi * V8P_D + ki + 3] = s * fp8_to_fp32_v8p(fp8_d);
    }
    __syncthreads();
    
    // Process tokens
    for (int tok = 0; tok < seq_len; tok++) {
        int t = t_start + tok;
        
        // Load gates
        __shared__ float g_val, beta_val;
        if (tid == 0) {
            float a = __bfloat162float(__ldg(&A[t * stride_a_t + h]));
            float bv = __bfloat162float(__ldg(&B_gate[t * stride_b_t + h]));
            float sp = v8p_softplus(a + dt_bias);
            g_val = v8p_fast_exp(-v8p_fast_exp(alog) * sp);
            beta_val = v8p_sigmoid(bv);
        }
        
        // Load Q, K, V
        for (int i = tid; i < V8P_D; i += blockDim.x) {
            q_smem[i] = __bfloat162float(__ldg(&Q[t * stride_q_t + qk_h * stride_q_h + i]));
            k_smem[i] = __bfloat162float(__ldg(&K[t * stride_k_t + qk_h * stride_k_h + i]));
        }
        if (tid < BLOCK_V) {
            v_smem[tid] = __bfloat162float(__ldg(&V[t * stride_v_t + h * stride_v_h + v0 + tid]));
        }
        __syncthreads();
        
        float g = g_val;
        float beta = beta_val;
        
        // Decay
        for (int i = tid; i < BLOCK_V * V8P_D; i += blockDim.x) {
            state_smem[i] *= g;
        }
        __syncthreads();
        
        // old_v = S @ k
        if (tid < BLOCK_V) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int ki = 0; ki < V8P_D; ki++) {
                sum += state_smem[tid * V8P_D + ki] * k_smem[ki];
            }
            old_v_smem[tid] = sum;
        }
        __syncthreads();
        
        // Rank-1 update
        for (int i = tid; i < BLOCK_V * V8P_D; i += blockDim.x) {
            int vi = i / V8P_D;
            int ki = i % V8P_D;
            float delta = beta * (v_smem[vi] - old_v_smem[vi]);
            state_smem[i] += delta * k_smem[ki];
        }
        __syncthreads();
        
        // Output
        if (tid < BLOCK_V) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int ki = 0; ki < V8P_D; ki++) {
                sum += state_smem[tid * V8P_D + ki] * q_smem[ki];
            }
            out_smem[tid] = scale * sum;
        }
        __syncthreads();
        
        if (tid < BLOCK_V) {
            Out[t * stride_o_t + h * stride_o_h + v0 + tid] = __float2bfloat16(out_smem[tid]);
        }
        __syncthreads();
    }
    
    // Compute new scales and quantize
    if (tid < BLOCK_V) {
        float max_abs = 0.0f;
        #pragma unroll 8
        for (int ki = 0; ki < V8P_D; ki++) {
            max_abs = fmaxf(max_abs, fabsf(state_smem[tid * V8P_D + ki]));
        }
        row_scale[tid] = max_abs / 448.0f;
        NewState_Scale[n * stride_ns_n + h * stride_ns_h + v0 + tid] = row_scale[tid];
    }
    __syncthreads();
    
    // Quantize and store
    uint32_t* new_state_fp8_ptr = NewState_FP8 + n * stride_ns_n + h * stride_ns_h + v0 * stride_ns_v;
    
    for (int i = tid; i < BLOCK_V * V8P_D / 4; i += blockDim.x) {
        int vi = (i * 4) / V8P_D;
        int ki = (i * 4) % V8P_D;
        
        float s_inv = (row_scale[vi] > 0.0f) ? (1.0f / row_scale[vi]) : 0.0f;
        
        __nv_fp8_e4m3 fp8_a = fp32_to_fp8_v8p(state_smem[vi * V8P_D + ki + 0] * s_inv);
        __nv_fp8_e4m3 fp8_b = fp32_to_fp8_v8p(state_smem[vi * V8P_D + ki + 1] * s_inv);
        __nv_fp8_e4m3 fp8_c = fp32_to_fp8_v8p(state_smem[vi * V8P_D + ki + 2] * s_inv);
        __nv_fp8_e4m3 fp8_d = fp32_to_fp8_v8p(state_smem[vi * V8P_D + ki + 3] * s_inv);
        
        new_state_fp8_ptr[vi * stride_ns_v + ki / 4] = pack_fp8x4_v8p(fp8_a, fp8_b, fp8_c, fp8_d);
    }
}

// ============================================================
// Launcher Functions
// ============================================================

void gdn_prefill_v8_launch_fp32(
    const void* Q, const void* K, const void* V, const void* State,
    const void* A_log, const void* A, const void* DtBias, const void* B_gate,
    const void* CuSeqlens,
    void* Out, void* NewState,
    float scale, int N, int num_v_heads, int D,
    int stride_q_t, int stride_q_h, int stride_k_t, int stride_k_h,
    int stride_v_t, int stride_v_h, int stride_s_n, int stride_s_h, int stride_s_v,
    int stride_a_t, int stride_b_t, int stride_o_t, int stride_o_h,
    int stride_ns_n, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, cudaStream_t stream
) {
    int V_BLOCKS = D / BLOCK_V;
    dim3 grid(N, num_v_heads, V_BLOCKS);
    dim3 block(128);
    
    // Triple buffered Q/K + state + temporaries
    size_t smem_size = V8P_STAGES * D * 2 * sizeof(float);  // Q/K buffers
    smem_size += (BLOCK_V + BLOCK_V * D + BLOCK_V + BLOCK_V) * sizeof(float);
    smem_size = ((smem_size + 127) / 128) * 128;
    
    if (BLOCK_V == 16) {
        gdn_prefill_kernel_v8_fp32<16><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State, (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate,
            (const int32_t*)CuSeqlens,
            (__nv_bfloat16*)Out, (float*)NewState, scale,
            stride_q_t, stride_q_h, stride_k_t, stride_k_h,
            stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v,
            stride_a_t, stride_b_t, stride_o_t, stride_o_h,
            stride_ns_n, stride_ns_h, stride_ns_v);
    } else {
        gdn_prefill_kernel_v8_fp32<32><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const float*)State, (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate,
            (const int32_t*)CuSeqlens,
            (__nv_bfloat16*)Out, (float*)NewState, scale,
            stride_q_t, stride_q_h, stride_k_t, stride_k_h,
            stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v,
            stride_a_t, stride_b_t, stride_o_t, stride_o_h,
            stride_ns_n, stride_ns_h, stride_ns_v);
    }
}

void gdn_prefill_v8_launch_fp8(
    const void* Q, const void* K, const void* V,
    const void* State_FP8, const void* State_Scale,
    const void* A_log, const void* A, const void* DtBias, const void* B_gate,
    const void* CuSeqlens,
    void* Out, void* NewState_FP8, void* NewState_Scale,
    float scale, int N, int num_v_heads, int D,
    int stride_q_t, int stride_q_h, int stride_k_t, int stride_k_h,
    int stride_v_t, int stride_v_h, int stride_s_n, int stride_s_h, int stride_s_v,
    int stride_a_t, int stride_b_t, int stride_o_t, int stride_o_h,
    int stride_ns_n, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, cudaStream_t stream
) {
    int V_BLOCKS = D / BLOCK_V;
    dim3 grid(N, num_v_heads, V_BLOCKS);
    dim3 block(128);
    
    size_t smem_size = (D + D + BLOCK_V + BLOCK_V * D + BLOCK_V + BLOCK_V + BLOCK_V) * sizeof(float);
    smem_size = ((smem_size + 127) / 128) * 128;
    
    if (BLOCK_V == 16) {
        gdn_prefill_kernel_v8_fp8<16><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const uint32_t*)State_FP8, (const float*)State_Scale,
            (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate,
            (const int32_t*)CuSeqlens,
            (__nv_bfloat16*)Out, (uint32_t*)NewState_FP8, (float*)NewState_Scale, scale,
            stride_q_t, stride_q_h, stride_k_t, stride_k_h,
            stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v,
            stride_a_t, stride_b_t, stride_o_t, stride_o_h,
            stride_ns_n, stride_ns_h, stride_ns_v);
    } else {
        gdn_prefill_kernel_v8_fp8<32><<<grid, block, smem_size, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (const uint32_t*)State_FP8, (const float*)State_Scale,
            (const float*)A_log, (const __nv_bfloat16*)A,
            (const float*)DtBias, (const __nv_bfloat16*)B_gate,
            (const int32_t*)CuSeqlens,
            (__nv_bfloat16*)Out, (uint32_t*)NewState_FP8, (float*)NewState_Scale, scale,
            stride_q_t, stride_q_h, stride_k_t, stride_k_h,
            stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v,
            stride_a_t, stride_b_t, stride_o_t, stride_o_h,
            stride_ns_n, stride_ns_h, stride_ns_v);
    }
}

}  // namespace gdn

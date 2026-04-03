#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float ptx_sigmoid_approx(float x) {
    constexpr float LOG2E = 1.4426950408889634f;
    float ex2_val;
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(ex2_val) : "f"(-x * LOG2E));
    float denom = 1.0f + ex2_val;
    float recip;
    asm volatile("rcp.approx.f32 %0, %1;" : "=f"(recip) : "f"(denom));
    return recip;
}

__global__ void fused_swiglu_ptx_vec4_kernel(
    const float4* __restrict__ x1,
    const float4* __restrict__ x2,
    float4* __restrict__ out,
    int total_vec4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_vec4) {
        float4 up = x1[idx];
        float4 gate = x2[idx];
        float4 result;
        result.x = gate.x * ptx_sigmoid_approx(gate.x) * up.x;
        result.y = gate.y * ptx_sigmoid_approx(gate.y) * up.y;
        result.z = gate.z * ptx_sigmoid_approx(gate.z) * up.z;
        result.w = gate.w * ptx_sigmoid_approx(gate.w) * up.w;
        out[idx] = result;
    }
}

__global__ void fused_swiglu_ptx_kernel(
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    float* __restrict__ out,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float gate = x2[idx];
        float up = x1[idx];
        out[idx] = gate * ptx_sigmoid_approx(gate) * up;
    }
}

void moe_swiglu_ptx(torch::Tensor x1, torch::Tensor x2, torch::Tensor out) {
    int total = x1.numel();
    int threads = 256;
    uintptr_t x1_ptr = reinterpret_cast<uintptr_t>(x1.data_ptr<float>());
    uintptr_t x2_ptr = reinterpret_cast<uintptr_t>(x2.data_ptr<float>());
    uintptr_t out_ptr = reinterpret_cast<uintptr_t>(out.data_ptr<float>());
    bool aligned_vec4 = ((x1_ptr | x2_ptr | out_ptr) & 0xF) == 0;

    int total_vec4 = aligned_vec4 ? (total / 4) : 0;
    int tail_offset = total_vec4 * 4;

    if (total_vec4 > 0) {
        int blocks_vec4 = (total_vec4 + threads - 1) / threads;
        fused_swiglu_ptx_vec4_kernel<<<blocks_vec4, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const float4*>(x1.data_ptr<float>()),
            reinterpret_cast<const float4*>(x2.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            total_vec4
        );
    }

    int tail = total - tail_offset;
    if (tail > 0) {
        int blocks_tail = (tail + threads - 1) / threads;
        fused_swiglu_ptx_kernel<<<blocks_tail, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x1.data_ptr<float>() + tail_offset,
            x2.data_ptr<float>() + tail_offset,
            out.data_ptr<float>() + tail_offset,
            tail
        );
    }
}

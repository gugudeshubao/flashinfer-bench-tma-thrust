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
    int blocks = (total + threads - 1) / threads;
    fused_swiglu_ptx_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        out.data_ptr<float>(),
        total
    );
}

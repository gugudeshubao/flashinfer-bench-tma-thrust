#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#if __has_include(<cute/tensor.hpp>)
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
using namespace cute;
#define HAS_CUTE 1
#else
#define HAS_CUTE 0
#endif

template<int BLOCK_THREADS>
__global__ void fused_swiglu_cute_kernel(
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    float* __restrict__ out,
    int total
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_THREADS + tid;

    extern __shared__ float smem[];
    float* smem_x1 = smem;
    float* smem_x2 = smem + BLOCK_THREADS;

#if HAS_CUTE
    auto tx1 = make_tensor(make_smem_ptr(smem_x1), make_layout(make_shape(Int<BLOCK_THREADS>{})));
    auto tx2 = make_tensor(make_smem_ptr(smem_x2), make_layout(make_shape(Int<BLOCK_THREADS>{})));

    if (idx < total) {
        tx1(tid) = x1[idx];
        tx2(tid) = x2[idx];
    }
    __syncthreads();

    if (idx < total) {
        float gate = tx2(tid);
        float up = tx1(tid);
        float silu = gate / (1.0f + __expf(-gate));
        out[idx] = silu * up;
    }
#else
    if (idx < total) {
        smem_x1[tid] = x1[idx];
        smem_x2[tid] = x2[idx];
    }
    __syncthreads();

    if (idx < total) {
        float gate = smem_x2[tid];
        float up = smem_x1[tid];
        float silu = gate / (1.0f + __expf(-gate));
        out[idx] = silu * up;
    }
#endif
}

void moe_swiglu_cute(torch::Tensor x1, torch::Tensor x2, torch::Tensor out) {
    constexpr int kBlock = 256;
    int total = x1.numel();
    int blocks = (total + kBlock - 1) / kBlock;
    size_t smem = sizeof(float) * kBlock * 2;
    fused_swiglu_cute_kernel<kBlock><<<blocks, kBlock, smem, at::cuda::getCurrentCUDAStream()>>>(
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        out.data_ptr<float>(),
        total
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_swiglu_cute", &moe_swiglu_cute, "MoE SwiGLU CuTe kernel");
}

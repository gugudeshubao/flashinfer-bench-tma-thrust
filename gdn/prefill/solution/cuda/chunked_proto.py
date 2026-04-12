"""
Experimental chunked prefill prototype backed by a repo-local Blackwell GEMM
shared library plus a CUDA chunk-correction kernel.

This module is intentionally separate from the active solution entrypoint.
It is meant for algorithm exploration and end-to-end prototype benchmarking.
"""

import ctypes
import math
import os
import subprocess
from pathlib import Path

import torch

_PROTO_LIB = None
AUTO_CHUNK_SIZE = 0
DEFAULT_CHUNK_SIZE = AUTO_CHUNK_SIZE
DEBUG_SYNC_ENV = "GDN_CHUNKPROTO_SYNC"


CUDA_SOURCE = r'''
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"
#include <cstdlib>
#include <cuda_runtime.h>

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using ElementA = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::bfloat16_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = float;
using LayoutC = cutlass::layout::ColumnMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using MmaTileShape_MNK = Shape<_256,_128,_64>;
using ClusterShape_MNK = Shape<_2,_2,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

#endif

static int g_last_gemm_batched_path = -1;

extern "C" int get_last_gemm_batched_path() {
  return g_last_gemm_batched_path;
}

extern "C" int run_sm100_bf16_gemm(
    const void* a_ptr,
    const void* b_ptr,
    const void* c_ptr,
    void* d_ptr,
    int m,
    int n,
    int k
) {
#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  return -100;
#else
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  Gemm gemm;
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {m, n, k, 1},
    {static_cast<ElementA const*>(a_ptr), stride_A, static_cast<ElementB const*>(b_ptr), stride_B},
    {{1.0f, 0.0f}, static_cast<ElementC const*>(c_ptr), stride_C, static_cast<ElementC*>(d_ptr), stride_D}
  };

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status can = gemm.can_implement(arguments);
  if (can != cutlass::Status::kSuccess) {
    return static_cast<int>(can);
  }
  cutlass::Status init = gemm.initialize(arguments, workspace.get());
  if (init != cutlass::Status::kSuccess) {
    return static_cast<int>(init);
  }
  cutlass::Status run = gemm.run();
  if (run != cutlass::Status::kSuccess) {
    return static_cast<int>(run);
  }
  return 0;
#endif
}

extern "C" int run_sm100_bf16_gemm_batched(
    const void* a_ptr,
    const void* b_ptr,
    const void* c_ptr,
    void* d_ptr,
    int num_batches,
    int m,
    int n,
    int k,
    long long a_batch_stride,
    long long b_batch_stride,
    long long c_batch_stride,
    long long d_batch_stride
) {
#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  return -100;
#else
  if (num_batches <= 0) {
    g_last_gemm_batched_path = -1;
    return 0;
  }

  const char* a_base = static_cast<const char*>(a_ptr);
  const char* b_base = static_cast<const char*>(b_ptr);
  const char* c_base = static_cast<const char*>(c_ptr);
  char* d_base = static_cast<char*>(d_ptr);

  long long packed_a_batch_stride = static_cast<long long>(m) * k * sizeof(ElementA);
  long long packed_b_batch_stride = static_cast<long long>(n) * k * sizeof(ElementB);
  long long packed_c_batch_stride = static_cast<long long>(m) * n * sizeof(ElementC);
  long long packed_d_batch_stride = static_cast<long long>(m) * n * sizeof(ElementC);

  bool can_use_strided_batched =
      a_batch_stride == packed_a_batch_stride &&
      b_batch_stride == packed_b_batch_stride &&
      c_batch_stride == packed_c_batch_stride &&
      d_batch_stride == packed_d_batch_stride &&
      std::getenv("GDN_CHUNKPROTO_DISABLE_STRIDED_BATCHED") == nullptr;

  if (can_use_strided_batched) {
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, num_batches});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, num_batches});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, num_batches});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, num_batches});

    Gemm gemm;
    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kBatched,
      {m, n, k, num_batches},
      {reinterpret_cast<ElementA const*>(a_base), stride_A, reinterpret_cast<ElementB const*>(b_base), stride_B},
      {{1.0f, 0.0f}, reinterpret_cast<ElementC const*>(c_base), stride_C, reinterpret_cast<ElementC*>(d_base), stride_D}
    };

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status can = gemm.can_implement(arguments);
    if (can == cutlass::Status::kSuccess) {
      cutlass::Status init = gemm.initialize(arguments, workspace.get());
      if (init == cutlass::Status::kSuccess) {
        cutlass::Status run = gemm.run();
        if (run == cutlass::Status::kSuccess) {
          g_last_gemm_batched_path = 1;
          return 0;
        }
      }
    }
    g_last_gemm_batched_path = -2;
  }

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  Gemm gemm;
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {m, n, k, 1},
    {reinterpret_cast<ElementA const*>(a_base), stride_A, reinterpret_cast<ElementB const*>(b_base), stride_B},
    {{1.0f, 0.0f}, reinterpret_cast<ElementC const*>(c_base), stride_C, reinterpret_cast<ElementC*>(d_base), stride_D}
  };

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status can = gemm.can_implement(arguments);
  if (can != cutlass::Status::kSuccess) {
    return static_cast<int>(can);
  }

  for (int batch = 0; batch < num_batches; ++batch) {
    auto a_batch_ptr = reinterpret_cast<ElementA const*>(a_base + batch * a_batch_stride);
    auto b_batch_ptr = reinterpret_cast<ElementB const*>(b_base + batch * b_batch_stride);
    auto c_batch_ptr = reinterpret_cast<ElementC const*>(c_base + batch * c_batch_stride);
    auto d_batch_ptr = reinterpret_cast<ElementC*>(d_base + batch * d_batch_stride);

    arguments.mainloop = {a_batch_ptr, stride_A, b_batch_ptr, stride_B};
    arguments.epilogue = {{1.0f, 0.0f}, c_batch_ptr, stride_C, d_batch_ptr, stride_D};

    cutlass::Status init = gemm.initialize(arguments, workspace.get());
    if (init != cutlass::Status::kSuccess) {
      return static_cast<int>(init);
    }
    cutlass::Status run = gemm.run();
    if (run != cutlass::Status::kSuccess) {
      return static_cast<int>(run);
    }
  }
  g_last_gemm_batched_path = 0;
  return 0;
#endif
}

template <int CHUNK_SIZE>
__global__ void chunk_correction_kernel_fixed(
    const float* __restrict__ old_v_init,
    const float* __restrict__ out_init,
    const float* __restrict__ kk,
    const float* __restrict__ kq,
    const float* __restrict__ v,
    const float* __restrict__ prefix,
    const float* __restrict__ beta,
    const float* __restrict__ state,
    const float* __restrict__ k,
    float scale,
    float* __restrict__ out,
    float* __restrict__ final_state,
    int D
) {
    int vi = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi >= D) {
        return;
    }

    float deltas[CHUNK_SIZE];

    #pragma unroll
    for (int t = 0; t < CHUNK_SIZE; ++t) {
        float old_v = old_v_init[vi * CHUNK_SIZE + t];
        #pragma unroll
        for (int j = 0; j < t; ++j) {
            float decay = prefix[t] / prefix[j];
            old_v += decay * kk[j * CHUNK_SIZE + t] * deltas[j];
        }

        float delta_t = beta[t] * (v[t * D + vi] - old_v);
        deltas[t] = delta_t;

        float out_t = out_init[vi * CHUNK_SIZE + t];
        #pragma unroll
        for (int j = 0; j < t; ++j) {
            float decay = prefix[t] / prefix[j];
            out_t += scale * decay * kq[j * CHUNK_SIZE + t] * deltas[j];
        }
        out_t += scale * kq[t * CHUNK_SIZE + t] * delta_t;
        out[vi * CHUNK_SIZE + t] = out_t;
    }

    float tail_decay = prefix[CHUNK_SIZE - 1];
    for (int d = 0; d < D; ++d) {
        float value = tail_decay * state[vi * D + d];
        #pragma unroll
        for (int j = 0; j < CHUNK_SIZE; ++j) {
            float decay = tail_decay / prefix[j];
            value += decay * deltas[j] * k[j * D + d];
        }
        final_state[vi * D + d] = value;
    }
}

__global__ void chunk_correction_kernel_runtime(
    const float* __restrict__ old_v_init,
    const float* __restrict__ out_init,
    const float* __restrict__ kk,
    const float* __restrict__ kq,
    const float* __restrict__ v,
    const float* __restrict__ prefix,
    const float* __restrict__ beta,
    const float* __restrict__ state,
    const float* __restrict__ k,
    float scale,
    float* __restrict__ out,
    float* __restrict__ final_state,
    int chunk_size,
    int D
) {
    int vi = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi >= D) {
        return;
    }

    float deltas[64];

    for (int t = 0; t < chunk_size; ++t) {
        float old_v = old_v_init[vi * chunk_size + t];
        for (int j = 0; j < t; ++j) {
            float decay = prefix[t] / prefix[j];
            old_v += decay * kk[j * chunk_size + t] * deltas[j];
        }

        float delta_t = beta[t] * (v[t * D + vi] - old_v);
        deltas[t] = delta_t;

        float out_t = out_init[vi * chunk_size + t];
        for (int j = 0; j < t; ++j) {
            float decay = prefix[t] / prefix[j];
            out_t += scale * decay * kq[j * chunk_size + t] * deltas[j];
        }
        out_t += scale * kq[t * chunk_size + t] * delta_t;
        out[vi * chunk_size + t] = out_t;
    }

    float tail_decay = prefix[chunk_size - 1];
    for (int d = 0; d < D; ++d) {
        float value = tail_decay * state[vi * D + d];
        for (int j = 0; j < chunk_size; ++j) {
            float decay = tail_decay / prefix[j];
            value += decay * deltas[j] * k[j * D + d];
        }
        final_state[vi * D + d] = value;
    }
}

extern "C" void run_chunk_correction(
    const void* old_v_init,
    const void* out_init,
    const void* kk,
    const void* kq,
    const void* v,
    const void* prefix,
    const void* beta,
    const void* state,
    const void* k,
    float scale,
    void* out,
    void* final_state,
    int chunk_size,
    int D
) {
    dim3 block(128);
    dim3 grid((D + block.x - 1) / block.x);
    auto old_v_ptr = static_cast<const float*>(old_v_init);
    auto out_init_ptr = static_cast<const float*>(out_init);
    auto kk_ptr = static_cast<const float*>(kk);
    auto kq_ptr = static_cast<const float*>(kq);
    auto v_ptr = static_cast<const float*>(v);
    auto prefix_ptr = static_cast<const float*>(prefix);
    auto beta_ptr = static_cast<const float*>(beta);
    auto state_ptr = static_cast<const float*>(state);
    auto k_ptr = static_cast<const float*>(k);
    auto out_ptr = static_cast<float*>(out);
    auto final_state_ptr = static_cast<float*>(final_state);

    if (chunk_size == 8) {
        chunk_correction_kernel_fixed<8><<<grid, block>>>(
            old_v_ptr, out_init_ptr, kk_ptr, kq_ptr, v_ptr, prefix_ptr, beta_ptr,
            state_ptr, k_ptr, scale, out_ptr, final_state_ptr, D
        );
        return;
    }
    if (chunk_size == 16) {
        chunk_correction_kernel_fixed<16><<<grid, block>>>(
            old_v_ptr, out_init_ptr, kk_ptr, kq_ptr, v_ptr, prefix_ptr, beta_ptr,
            state_ptr, k_ptr, scale, out_ptr, final_state_ptr, D
        );
        return;
    }
    if (chunk_size == 32) {
        chunk_correction_kernel_fixed<32><<<grid, block>>>(
            old_v_ptr, out_init_ptr, kk_ptr, kq_ptr, v_ptr, prefix_ptr, beta_ptr,
            state_ptr, k_ptr, scale, out_ptr, final_state_ptr, D
        );
        return;
    }
    if (chunk_size == 64) {
        chunk_correction_kernel_fixed<64><<<grid, block>>>(
            old_v_ptr, out_init_ptr, kk_ptr, kq_ptr, v_ptr, prefix_ptr, beta_ptr,
            state_ptr, k_ptr, scale, out_ptr, final_state_ptr, D
        );
        return;
    }
    chunk_correction_kernel_runtime<<<grid, block>>>(
        old_v_ptr, out_init_ptr, kk_ptr, kq_ptr, v_ptr, prefix_ptr, beta_ptr,
        state_ptr, k_ptr, scale, out_ptr, final_state_ptr, chunk_size, D
    );
}

template <int CHUNK_SIZE>
__global__ void chunk_correction_heads_kernel_fixed(
    const float* __restrict__ old_v_init,   // [H, V, C]
    const float* __restrict__ out_init,     // [H, V, C]
    const float* __restrict__ kk,           // [H, C, C]
    const float* __restrict__ kq,           // [H, C, C]
    const float* __restrict__ v,            // [H, C, V]
    const float* __restrict__ prefix,       // [H, C]
    const float* __restrict__ beta,         // [H, C]
    const float* __restrict__ state,        // [H, V, D]
    const float* __restrict__ k,            // [H, C, D]
    float scale,
    float* __restrict__ out,                // [H, V, C]
    float* __restrict__ final_state,        // [H, V, D]
    int num_heads,
    int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = num_heads * D;
    if (idx >= total_rows) {
        return;
    }

    int h = idx / D;
    int vi = idx % D;

    const float* old_v_head = old_v_init + h * D * CHUNK_SIZE;
    const float* out_head = out_init + h * D * CHUNK_SIZE;
    const float* kk_head = kk + h * CHUNK_SIZE * CHUNK_SIZE;
    const float* kq_head = kq + h * CHUNK_SIZE * CHUNK_SIZE;
    const float* v_head = v + h * CHUNK_SIZE * D;
    const float* prefix_head = prefix + h * CHUNK_SIZE;
    const float* beta_head = beta + h * CHUNK_SIZE;
    const float* state_head = state + h * D * D;
    const float* k_head = k + h * CHUNK_SIZE * D;
    float* out_dst = out + h * D * CHUNK_SIZE;
    float* final_state_head = final_state + h * D * D;

    float deltas[CHUNK_SIZE];

    #pragma unroll
    for (int t = 0; t < CHUNK_SIZE; ++t) {
        float old_v = old_v_head[vi * CHUNK_SIZE + t];
        #pragma unroll
        for (int j = 0; j < t; ++j) {
            float decay = prefix_head[t] / prefix_head[j];
            old_v += decay * kk_head[j * CHUNK_SIZE + t] * deltas[j];
        }

        float delta_t = beta_head[t] * (v_head[t * D + vi] - old_v);
        deltas[t] = delta_t;

        float out_t = out_head[vi * CHUNK_SIZE + t];
        #pragma unroll
        for (int j = 0; j < t; ++j) {
            float decay = prefix_head[t] / prefix_head[j];
            out_t += scale * decay * kq_head[j * CHUNK_SIZE + t] * deltas[j];
        }
        out_t += scale * kq_head[t * CHUNK_SIZE + t] * delta_t;
        out_dst[vi * CHUNK_SIZE + t] = out_t;
    }

    float tail_decay = prefix_head[CHUNK_SIZE - 1];
    for (int d = 0; d < D; ++d) {
        float value = tail_decay * state_head[vi * D + d];
        #pragma unroll
        for (int j = 0; j < CHUNK_SIZE; ++j) {
            float decay = tail_decay / prefix_head[j];
            value += decay * deltas[j] * k_head[j * D + d];
        }
        final_state_head[vi * D + d] = value;
    }
}

__global__ void chunk_correction_heads_kernel_runtime(
    const float* __restrict__ old_v_init,   // [H, V, C]
    const float* __restrict__ out_init,     // [H, V, C]
    const float* __restrict__ kk,           // [H, C, C]
    const float* __restrict__ kq,           // [H, C, C]
    const float* __restrict__ v,            // [H, C, V]
    const float* __restrict__ prefix,       // [H, C]
    const float* __restrict__ beta,         // [H, C]
    const float* __restrict__ state,        // [H, V, D]
    const float* __restrict__ k,            // [H, C, D]
    float scale,
    float* __restrict__ out,                // [H, V, C]
    float* __restrict__ final_state,        // [H, V, D]
    int num_heads,
    int chunk_size,
    int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = num_heads * D;
    if (idx >= total_rows) {
        return;
    }

    int h = idx / D;
    int vi = idx % D;

    const float* old_v_head = old_v_init + h * D * chunk_size;
    const float* out_head = out_init + h * D * chunk_size;
    const float* kk_head = kk + h * chunk_size * chunk_size;
    const float* kq_head = kq + h * chunk_size * chunk_size;
    const float* v_head = v + h * chunk_size * D;
    const float* prefix_head = prefix + h * chunk_size;
    const float* beta_head = beta + h * chunk_size;
    const float* state_head = state + h * D * D;
    const float* k_head = k + h * chunk_size * D;
    float* out_dst = out + h * D * chunk_size;
    float* final_state_head = final_state + h * D * D;

    float deltas[64];

    for (int t = 0; t < chunk_size; ++t) {
        float old_v = old_v_head[vi * chunk_size + t];
        for (int j = 0; j < t; ++j) {
            float decay = prefix_head[t] / prefix_head[j];
            old_v += decay * kk_head[j * chunk_size + t] * deltas[j];
        }

        float delta_t = beta_head[t] * (v_head[t * D + vi] - old_v);
        deltas[t] = delta_t;

        float out_t = out_head[vi * chunk_size + t];
        for (int j = 0; j < t; ++j) {
            float decay = prefix_head[t] / prefix_head[j];
            out_t += scale * decay * kq_head[j * chunk_size + t] * deltas[j];
        }
        out_t += scale * kq_head[t * chunk_size + t] * delta_t;
        out_dst[vi * chunk_size + t] = out_t;
    }

    float tail_decay = prefix_head[chunk_size - 1];
    for (int d = 0; d < D; ++d) {
        float value = tail_decay * state_head[vi * D + d];
        for (int j = 0; j < chunk_size; ++j) {
            float decay = tail_decay / prefix_head[j];
            value += decay * deltas[j] * k_head[j * D + d];
        }
        final_state_head[vi * D + d] = value;
    }
}

extern "C" void run_chunk_correction_heads(
    const void* old_v_init,
    const void* out_init,
    const void* kk,
    const void* kq,
    const void* v,
    const void* prefix,
    const void* beta,
    const void* state,
    const void* k,
    float scale,
    void* out,
    void* final_state,
    int num_heads,
    int chunk_size,
    int D
) {
    dim3 block(128);
    dim3 grid((num_heads * D + block.x - 1) / block.x);
    auto old_v_ptr = static_cast<const float*>(old_v_init);
    auto out_init_ptr = static_cast<const float*>(out_init);
    auto kk_ptr = static_cast<const float*>(kk);
    auto kq_ptr = static_cast<const float*>(kq);
    auto v_ptr = static_cast<const float*>(v);
    auto prefix_ptr = static_cast<const float*>(prefix);
    auto beta_ptr = static_cast<const float*>(beta);
    auto state_ptr = static_cast<const float*>(state);
    auto k_ptr = static_cast<const float*>(k);
    auto out_ptr = static_cast<float*>(out);
    auto final_state_ptr = static_cast<float*>(final_state);

    if (chunk_size == 8) {
        chunk_correction_heads_kernel_fixed<8><<<grid, block>>>(
            old_v_ptr, out_init_ptr, kk_ptr, kq_ptr, v_ptr, prefix_ptr, beta_ptr,
            state_ptr, k_ptr, scale, out_ptr, final_state_ptr, num_heads, D
        );
        return;
    }
    if (chunk_size == 16) {
        chunk_correction_heads_kernel_fixed<16><<<grid, block>>>(
            old_v_ptr, out_init_ptr, kk_ptr, kq_ptr, v_ptr, prefix_ptr, beta_ptr,
            state_ptr, k_ptr, scale, out_ptr, final_state_ptr, num_heads, D
        );
        return;
    }
    if (chunk_size == 32) {
        chunk_correction_heads_kernel_fixed<32><<<grid, block>>>(
            old_v_ptr, out_init_ptr, kk_ptr, kq_ptr, v_ptr, prefix_ptr, beta_ptr,
            state_ptr, k_ptr, scale, out_ptr, final_state_ptr, num_heads, D
        );
        return;
    }
    if (chunk_size == 64) {
        chunk_correction_heads_kernel_fixed<64><<<grid, block>>>(
            old_v_ptr, out_init_ptr, kk_ptr, kq_ptr, v_ptr, prefix_ptr, beta_ptr,
            state_ptr, k_ptr, scale, out_ptr, final_state_ptr, num_heads, D
        );
        return;
    }
    chunk_correction_heads_kernel_runtime<<<grid, block>>>(
        old_v_ptr, out_init_ptr, kk_ptr, kq_ptr, v_ptr, prefix_ptr, beta_ptr,
        state_ptr, k_ptr, scale, out_ptr, final_state_ptr, num_heads, chunk_size, D
    );
}
'''


def _ensure_cuda_home():
    if os.environ.get("CUDA_HOME"):
        return
    for candidate in ("/usr/local/cuda-12.8", "/usr/local/cuda"):
        if os.path.exists(candidate):
            os.environ["CUDA_HOME"] = candidate
            return


def _maybe_synchronize():
    if os.environ.get(DEBUG_SYNC_ENV) == "1":
        torch.cuda.synchronize()


def _load_library():
    global _PROTO_LIB
    if _PROTO_LIB is not None:
        return _PROTO_LIB

    _ensure_cuda_home()
    build_dir = Path("/tmp/repo_local_blackwell_chunkproto_module")
    build_dir.mkdir(parents=True, exist_ok=True)
    source_path = build_dir / "chunked_prefill_proto_lib.cu"
    library_path = build_dir / "libchunked_prefill_proto.so"
    source_path.write_text(CUDA_SOURCE)

    compile_cmd = [
        "nvcc",
        "-O3",
        "-arch=sm_100a",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--shared",
        "-Xcompiler", "-fPIC",
        "-I", "/opt/cutlass/include",
        "-I", "/opt/cutlass/tools/util/include",
        "-I", "/opt/cutlass/examples/common",
        "-o", str(library_path),
        str(source_path),
    ]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"chunked proto compile failed:\n{result.stderr[:4000]}")

    lib = ctypes.CDLL(str(library_path))

    gemm = lib.run_sm100_bf16_gemm
    gemm.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    gemm.restype = ctypes.c_int

    gemm_batched = lib.run_sm100_bf16_gemm_batched
    gemm_batched.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
    ]
    gemm_batched.restype = ctypes.c_int

    correction = lib.run_chunk_correction
    correction.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    correction.restype = None

    correction_heads = lib.run_chunk_correction_heads
    correction_heads.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    correction_heads.restype = None

    get_last_gemm_batched_path = lib.get_last_gemm_batched_path
    get_last_gemm_batched_path.argtypes = []
    get_last_gemm_batched_path.restype = ctypes.c_int

    _PROTO_LIB = {
        "gemm": gemm,
        "gemm_batched": gemm_batched,
        "correction": correction,
        "correction_heads": correction_heads,
        "get_last_gemm_batched_path": get_last_gemm_batched_path,
    }
    return _PROTO_LIB


def get_last_gemm_batched_path() -> int:
    lib = _load_library()
    return int(lib["get_last_gemm_batched_path"]())


def gemm_state_cols(state_bf16: torch.Tensor, cols_bf16_storage: torch.Tensor) -> torch.Tensor:
    lib = _load_library()
    gemm = lib["gemm"]

    m, k = state_bf16.shape
    n = cols_bf16_storage.shape[0]
    c_storage = torch.empty(n, m, device=state_bf16.device, dtype=torch.float32)
    d_storage = torch.empty(n, m, device=state_bf16.device, dtype=torch.float32)
    rc = gemm(
        state_bf16.data_ptr(),
        cols_bf16_storage.data_ptr(),
        c_storage.data_ptr(),
        d_storage.data_ptr(),
        m,
        n,
        k,
    )
    if rc != 0:
        raise RuntimeError(f"run_sm100_bf16_gemm returned {rc}")
    return d_storage.transpose(0, 1).contiguous()


def gemm_state_cols_pair(
    state_bf16: torch.Tensor,
    first_cols_bf16: torch.Tensor,
    second_cols_bf16: torch.Tensor,
):
    split = first_cols_bf16.shape[0]
    combined_cols = torch.cat([first_cols_bf16, second_cols_bf16], dim=0).contiguous()
    combined = gemm_state_cols(state_bf16, combined_cols)
    return combined[:, :split].contiguous(), combined[:, split:].contiguous()


def gemm_state_cols_batched(state_bf16: torch.Tensor, cols_bf16_storage: torch.Tensor) -> torch.Tensor:
    """
    Batched helper over heads.

    state_bf16: [H, V, D]
    cols_bf16_storage: [H, C, D]
    returns: [H, V, C]
    """
    lib = _load_library()
    gemm_batched = lib["gemm_batched"]

    num_batches, m, k = state_bf16.shape
    n = cols_bf16_storage.shape[1]
    c_storage = torch.empty(num_batches, n, m, device=state_bf16.device, dtype=torch.float32)
    d_storage = torch.empty(num_batches, n, m, device=state_bf16.device, dtype=torch.float32)

    rc = gemm_batched(
        state_bf16.data_ptr(),
        cols_bf16_storage.data_ptr(),
        c_storage.data_ptr(),
        d_storage.data_ptr(),
        num_batches,
        m,
        n,
        k,
        state_bf16.stride(0) * state_bf16.element_size(),
        cols_bf16_storage.stride(0) * cols_bf16_storage.element_size(),
        c_storage.stride(0) * c_storage.element_size(),
        d_storage.stride(0) * d_storage.element_size(),
    )
    if rc != 0:
        raise RuntimeError(f"run_sm100_bf16_gemm_batched returned {rc}")
    return d_storage.permute(0, 2, 1).contiguous()


def gemm_state_cols_pair_batched(
    state_bf16: torch.Tensor,
    first_cols_bf16: torch.Tensor,
    second_cols_bf16: torch.Tensor,
):
    split = first_cols_bf16.shape[1]
    combined_cols = torch.cat([first_cols_bf16, second_cols_bf16], dim=1).contiguous()
    combined = gemm_state_cols_batched(state_bf16, combined_cols)
    return combined[:, :, :split].contiguous(), combined[:, :, split:].contiguous()


def pairwise_k_products(k_tensor: torch.Tensor, q_tensor: torch.Tensor):
    """
    Compute K@K^T and K@Q^T in one matmul by concatenating the RHS columns.

    Accepts tensors shaped [..., C, D] and returns two tensors shaped [..., C, C].
    """
    split = k_tensor.shape[-2]
    rhs = torch.cat([k_tensor, q_tensor], dim=-2).transpose(-1, -2).contiguous()
    products = torch.matmul(k_tensor, rhs).float().contiguous()
    return products[..., :split].contiguous(), products[..., split:].contiguous()


def chunked_chunk_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
):
    lib = _load_library()
    correction = lib["correction"]

    chunk_size, D = q.shape
    prefix = torch.cumprod(g, dim=0)

    k_init = (prefix[:, None] * k).to(torch.bfloat16).contiguous()
    q_init = (prefix[:, None] * q).to(torch.bfloat16).contiguous()
    state_bf16 = state.to(torch.bfloat16).contiguous()

    old_v_init, out_init = gemm_state_cols_pair(state_bf16, k_init, q_init)
    old_v_init = old_v_init.float().contiguous()
    out_init = (out_init.float() * scale).contiguous()

    kk, kq = pairwise_k_products(k, q)
    v_t = v.float().contiguous()
    prefix_c = prefix.float().contiguous()
    beta_c = beta.float().contiguous()
    state_c = state.float().contiguous()
    k_c = k.float().contiguous()

    out = torch.empty(D, chunk_size, device=q.device, dtype=torch.float32)
    final_state = torch.empty_like(state_c)

    correction(
        old_v_init.data_ptr(),
        out_init.data_ptr(),
        kk.data_ptr(),
        kq.data_ptr(),
        v_t.data_ptr(),
        prefix_c.data_ptr(),
        beta_c.data_ptr(),
        state_c.data_ptr(),
        k_c.data_ptr(),
        ctypes.c_float(scale),
        out.data_ptr(),
        final_state.data_ptr(),
        chunk_size,
        D,
    )
    _maybe_synchronize()
    return out.transpose(0, 1).contiguous(), final_state


def _expand_qk_heads(x: torch.Tensor, num_v_heads: int) -> torch.Tensor:
    num_q_heads = x.shape[1]
    ratio = num_v_heads // num_q_heads
    return x.repeat_interleave(ratio, dim=1).permute(1, 0, 2).contiguous()


def chunked_sequence_heads_cuda(
    q_heads: torch.Tensor,
    k_heads: torch.Tensor,
    v_heads: torch.Tensor,
    state: torch.Tensor,
    g_heads: torch.Tensor,
    beta_heads: torch.Tensor,
    scale: float,
):
    lib = _load_library()
    correction_heads = lib["correction_heads"]

    num_v_heads, chunk_size, D = q_heads.shape
    prefix_heads = torch.cumprod(g_heads, dim=1).contiguous()

    state_bf16 = state.to(torch.bfloat16).contiguous()
    k_init = (prefix_heads[:, :, None] * k_heads).to(torch.bfloat16).contiguous()
    q_init = (prefix_heads[:, :, None] * q_heads).to(torch.bfloat16).contiguous()

    old_v_init, out_init = gemm_state_cols_pair_batched(state_bf16, k_init, q_init)
    old_v_init = old_v_init.float().contiguous()   # [H, V, C]
    out_init = (out_init.float() * scale).contiguous()
    kk, kq = pairwise_k_products(k_heads, q_heads)  # [H, C, C]

    state_c = state.float().contiguous()                            # [H, D, D]
    k_c = k_heads.contiguous()                                      # [H, C, D]
    out = torch.empty(num_v_heads, D, chunk_size, device=q_heads.device, dtype=torch.float32)
    final_state = torch.empty_like(state_c)

    correction_heads(
        old_v_init.data_ptr(),
        out_init.data_ptr(),
        kk.data_ptr(),
        kq.data_ptr(),
        v_heads.data_ptr(),
        prefix_heads.data_ptr(),
        beta_heads.data_ptr(),
        state_c.data_ptr(),
        k_c.data_ptr(),
        ctypes.c_float(scale),
        out.data_ptr(),
        final_state.data_ptr(),
        num_v_heads,
        chunk_size,
        D,
    )
    _maybe_synchronize()
    return out.permute(2, 0, 1).contiguous(), final_state


def chunked_sequence_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
):
    num_v_heads = v.shape[1]
    q_heads = _expand_qk_heads(q.float(), num_v_heads)  # [H, C, D]
    k_heads = _expand_qk_heads(k.float(), num_v_heads)  # [H, C, D]
    v_heads = v.float().permute(1, 0, 2).contiguous()   # [H, C, D]
    g_heads = g.float().permute(1, 0).contiguous()      # [H, C]
    beta_heads = beta.float().permute(1, 0).contiguous()
    return chunked_sequence_heads_cuda(
        q_heads,
        k_heads,
        v_heads,
        state,
        g_heads,
        beta_heads,
        scale,
    )


def chunked_prefill_end_to_end(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    chunk_size: int = 8,
):
    total_tokens, _, D = q.shape
    num_v_heads = v.shape[1]
    num_seqs = cu_seqlens.numel() - 1
    device = q.device

    x = a.float() + dt_bias.float()
    g_all = torch.exp(-torch.exp(A_log.float()) * torch.nn.functional.softplus(x))
    beta_all = torch.sigmoid(b.float())

    out = torch.empty(total_tokens, num_v_heads, D, dtype=torch.bfloat16, device=device)
    new_state = state.clone().float()

    for seq_idx in range(num_seqs):
        start = int(cu_seqlens[seq_idx].item())
        end = int(cu_seqlens[seq_idx + 1].item())
        if end <= start:
            continue
        for h in range(num_v_heads):
            qk_h = h // 2
            state_h = new_state[seq_idx, h].clone()
            for chunk_start in range(start, end, chunk_size):
                chunk_end = min(chunk_start + chunk_size, end)
                q_chunk = q[chunk_start:chunk_end, qk_h].float().contiguous()
                k_chunk = k[chunk_start:chunk_end, qk_h].float().contiguous()
                v_chunk = v[chunk_start:chunk_end, h].float().contiguous()
                g_chunk = g_all[chunk_start:chunk_end, h].float().contiguous()
                beta_chunk = beta_all[chunk_start:chunk_end, h].float().contiguous()

                out_chunk, state_h = chunked_chunk_cuda(
                    q_chunk, k_chunk, v_chunk, state_h, g_chunk, beta_chunk, scale
                )
                out[chunk_start:chunk_end, h] = out_chunk.to(torch.bfloat16)
            new_state[seq_idx, h] = state_h

    return out, new_state


def chunked_prefill_end_to_end_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    chunk_size: int = 8,
):
    total_tokens, _, D = q.shape
    num_v_heads = v.shape[1]
    num_seqs = cu_seqlens.numel() - 1
    device = q.device

    x = a.float() + dt_bias.float()
    g_all = torch.exp(-torch.exp(A_log.float()) * torch.nn.functional.softplus(x))
    beta_all = torch.sigmoid(b.float())

    out = torch.empty(total_tokens, num_v_heads, D, dtype=torch.bfloat16, device=device)
    new_state = state.clone().float()

    for seq_idx in range(num_seqs):
        start = int(cu_seqlens[seq_idx].item())
        end = int(cu_seqlens[seq_idx + 1].item())
        if end <= start:
            continue

        q_heads_full = _expand_qk_heads(q[start:end].float(), num_v_heads)
        k_heads_full = _expand_qk_heads(k[start:end].float(), num_v_heads)
        v_heads_full = v[start:end].float().permute(1, 0, 2).contiguous()
        g_heads_full = g_all[start:end].float().permute(1, 0).contiguous()
        beta_heads_full = beta_all[start:end].float().permute(1, 0).contiguous()
        state_seq = new_state[seq_idx].clone()
        seq_len = end - start
        for local_start in range(0, seq_len, chunk_size):
            local_end = min(local_start + chunk_size, seq_len)
            out_chunk, state_seq = chunked_sequence_heads_cuda(
                q_heads_full[:, local_start:local_end, :],
                k_heads_full[:, local_start:local_end, :],
                v_heads_full[:, local_start:local_end, :],
                state_seq,
                g_heads_full[:, local_start:local_end],
                beta_heads_full[:, local_start:local_end],
                scale,
            )
            out[start + local_start:start + local_end] = out_chunk.to(torch.bfloat16)
        new_state[seq_idx] = state_seq

    return out, new_state


def chunked_prefill_end_to_end_uniform_batch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    chunk_size: int = 8,
):
    """
    Uniform-length sequence prototype.

    Assumes all sequences have the same length so we can flatten sequence and
    head dimensions into one batched GEMM / correction launch per chunk.
    """
    total_tokens, num_q_heads, D = q.shape
    num_v_heads = v.shape[1]
    num_seqs = cu_seqlens.numel() - 1
    if num_seqs <= 0:
        return torch.empty_like(v), state.clone()

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if not torch.all(lengths == lengths[0]):
        raise ValueError("chunked_prefill_end_to_end_uniform_batch requires equal-length sequences")

    seq_len = int(lengths[0].item())
    if seq_len <= 0:
        return torch.empty(total_tokens, num_v_heads, D, dtype=torch.bfloat16, device=q.device), state.clone()

    x = a.float() + dt_bias.float()
    g_all = torch.exp(-torch.exp(A_log.float()) * torch.nn.functional.softplus(x))
    beta_all = torch.sigmoid(b.float())

    out = torch.empty(num_seqs, seq_len, num_v_heads, D, dtype=torch.bfloat16, device=q.device)
    state_seq = state.clone().float()

    ratio = num_v_heads // num_q_heads
    q_heads_full = q.view(num_seqs, seq_len, num_q_heads, D).float()
    q_heads_full = q_heads_full.repeat_interleave(ratio, dim=2).permute(0, 2, 1, 3).contiguous()
    k_heads_full = k.view(num_seqs, seq_len, num_q_heads, D).float()
    k_heads_full = k_heads_full.repeat_interleave(ratio, dim=2).permute(0, 2, 1, 3).contiguous()
    v_heads_full = v.view(num_seqs, seq_len, num_v_heads, D).float().permute(0, 2, 1, 3).contiguous()
    g_heads_full = g_all.view(num_seqs, seq_len, num_v_heads).float().permute(0, 2, 1).contiguous()
    beta_heads_full = beta_all.view(num_seqs, seq_len, num_v_heads).float().permute(0, 2, 1).contiguous()
    lib = _load_library()
    correction_heads = lib["correction_heads"]

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        actual_chunk = chunk_end - chunk_start

        q_heads = q_heads_full[:, :, chunk_start:chunk_end, :]
        k_heads = k_heads_full[:, :, chunk_start:chunk_end, :]
        v_heads = v_heads_full[:, :, chunk_start:chunk_end, :]
        g_heads = g_heads_full[:, :, chunk_start:chunk_end]
        beta_heads = beta_heads_full[:, :, chunk_start:chunk_end]
        prefix_heads = torch.cumprod(g_heads, dim=-1).contiguous()

        flat_batches = num_seqs * num_v_heads
        state_bf16 = state_seq.to(torch.bfloat16).contiguous().view(flat_batches, D, D)
        k_init = (prefix_heads[..., None] * k_heads).to(torch.bfloat16).contiguous().view(flat_batches, actual_chunk, D)
        q_init = (prefix_heads[..., None] * q_heads).to(torch.bfloat16).contiguous().view(flat_batches, actual_chunk, D)

        old_v_init, out_init = gemm_state_cols_pair_batched(state_bf16, k_init, q_init)
        old_v_init = old_v_init.float().contiguous()   # [BH, D, C]
        out_init = (out_init.float() * scale).contiguous()

        kk, kq = pairwise_k_products(k_heads, q_heads)
        kk = kk.view(flat_batches, actual_chunk, actual_chunk)
        kq = kq.view(flat_batches, actual_chunk, actual_chunk)
        v_flat = v_heads.contiguous().view(flat_batches, actual_chunk, D)
        prefix_flat = prefix_heads.contiguous().view(flat_batches, actual_chunk)
        beta_flat = beta_heads.contiguous().view(flat_batches, actual_chunk)
        state_flat = state_seq.contiguous().view(flat_batches, D, D)
        k_flat = k_heads.contiguous().view(flat_batches, actual_chunk, D)

        out_flat = torch.empty(flat_batches, D, actual_chunk, device=q.device, dtype=torch.float32)
        final_state_flat = torch.empty_like(state_flat)
        correction_heads(
            old_v_init.data_ptr(),
            out_init.data_ptr(),
            kk.data_ptr(),
            kq.data_ptr(),
            v_flat.data_ptr(),
            prefix_flat.data_ptr(),
            beta_flat.data_ptr(),
            state_flat.data_ptr(),
            k_flat.data_ptr(),
            ctypes.c_float(scale),
            out_flat.data_ptr(),
            final_state_flat.data_ptr(),
            flat_batches,
            actual_chunk,
            D,
        )
        _maybe_synchronize()

        out_heads = out_flat.view(num_seqs, num_v_heads, D, actual_chunk).permute(0, 3, 1, 2).contiguous()
        out[:, chunk_start:chunk_end] = out_heads.to(torch.bfloat16)
        state_seq = final_state_flat.view(num_seqs, num_v_heads, D, D).contiguous()

    return out.view(total_tokens, num_v_heads, D), state_seq


def chunked_prefill_end_to_end_grouped(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    chunk_size: int = 8,
):
    """
    Group sequences by equal length and run the uniform-batch prototype per group.

    This is a practical bridge between the equal-length fast path and true packed
    varlen workloads.
    """
    total_tokens, _, D = q.shape
    num_v_heads = v.shape[1]
    num_seqs = cu_seqlens.numel() - 1
    device = q.device

    out = torch.empty(total_tokens, num_v_heads, D, dtype=torch.bfloat16, device=device)
    new_state = state.clone().float()

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    groups = {}
    for seq_idx, length in enumerate(lengths):
        groups.setdefault(int(length), []).append(seq_idx)

    for length, seq_indices in groups.items():
        if length <= 0:
            continue
        if len(seq_indices) == 1:
            seq_idx = seq_indices[0]
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())
            q_group = q[start:end]
            k_group = k[start:end]
            v_group = v[start:end]
            a_group = a[start:end]
            b_group = b[start:end]
            cu_group = torch.tensor([0, length], dtype=cu_seqlens.dtype, device=device)
            state_group = new_state[seq_idx : seq_idx + 1].clone()
            out_group, updated_state = chunked_prefill_end_to_end_batched(
                q_group,
                k_group,
                v_group,
                state_group,
                A_log,
                a_group,
                dt_bias,
                b_group,
                cu_group,
                scale,
                chunk_size=chunk_size,
            )
            out[start:end] = out_group
            new_state[seq_idx] = updated_state[0]
            continue

        gathered_q = []
        gathered_k = []
        gathered_v = []
        gathered_a = []
        gathered_b = []
        gathered_state = new_state[seq_indices].clone()

        for seq_idx in seq_indices:
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())
            gathered_q.append(q[start:end])
            gathered_k.append(k[start:end])
            gathered_v.append(v[start:end])
            gathered_a.append(a[start:end])
            gathered_b.append(b[start:end])

        q_group = torch.cat(gathered_q, dim=0)
        k_group = torch.cat(gathered_k, dim=0)
        v_group = torch.cat(gathered_v, dim=0)
        a_group = torch.cat(gathered_a, dim=0)
        b_group = torch.cat(gathered_b, dim=0)
        cu_group = torch.arange(
            0,
            len(seq_indices) * length + 1,
            length,
            dtype=cu_seqlens.dtype,
            device=device,
        )

        out_group, state_group = chunked_prefill_end_to_end_uniform_batch(
            q_group,
            k_group,
            v_group,
            gathered_state,
            A_log,
            a_group,
            dt_bias,
            b_group,
            cu_group,
            scale,
            chunk_size=chunk_size,
        )

        for group_idx, seq_idx in enumerate(seq_indices):
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())
            group_start = group_idx * length
            group_end = group_start + length
            out[start:end] = out_group[group_start:group_end]
            new_state[seq_idx] = state_group[group_idx]

    return out, new_state


def _should_use_grouped_varlen(lengths: torch.Tensor) -> bool:
    """
    Grouped varlen only pays off when some lengths repeat.

    When every sequence length is unique, the gather/scatter overhead tends to
    outweigh any benefit from grouping and the simpler batched path is faster.
    """
    if lengths.numel() <= 1:
        return False
    _, counts = torch.unique(lengths, return_counts=True)
    return int(counts.max().item()) > 1


def recommend_chunk_size(cu_seqlens: torch.Tensor) -> int:
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if lengths.numel() == 0:
        return 64
    if torch.all(lengths == lengths[0]) and lengths.numel() > 1:
        return 32
    return 64


def kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
):
    """
    Solution-like wrapper for the experimental chunked prefill path.

    Signature matches the active prefill solution entrypoint so we can compare
    it directly against the Triton baseline without changing call sites.
    """
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(q.shape[-1])
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if chunk_size is None or chunk_size <= 0:
        chunk_size = recommend_chunk_size(cu_seqlens)
    if torch.all(lengths == lengths[0]):
        if lengths.numel() == 1:
            return chunked_prefill_end_to_end_batched(
                q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, float(scale), chunk_size=chunk_size
            )
        return chunked_prefill_end_to_end_uniform_batch(
            q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, float(scale), chunk_size=chunk_size
        )
    if _should_use_grouped_varlen(lengths):
        return chunked_prefill_end_to_end_grouped(
            q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, float(scale), chunk_size=chunk_size
        )
    return chunked_prefill_end_to_end_batched(
        q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, float(scale), chunk_size=chunk_size
    )

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
import time
from pathlib import Path

import torch

_PROTO_LIB = None
AUTO_CHUNK_SIZE = 0
DEFAULT_CHUNK_SIZE = AUTO_CHUNK_SIZE
DEBUG_SYNC_ENV = "GDN_CHUNKPROTO_SYNC"
FUSED_CORRECTION_MIN_BATCHES = 32


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

template <int CHUNK_SIZE>
__global__ void chunk_correction_heads_outdelta_kernel_fixed(
    const float* __restrict__ old_v_init,
    const float* __restrict__ out_init,
    const float* __restrict__ kk,
    const float* __restrict__ kq,
    const float* __restrict__ v,
    const float* __restrict__ prefix,
    const float* __restrict__ beta,
    float scale,
    float* __restrict__ out,
    float* __restrict__ deltas_out,
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
    float* out_dst = out + h * D * CHUNK_SIZE;
    float* delta_dst = deltas_out + h * D * CHUNK_SIZE;

    __shared__ float prefix_sh[CHUNK_SIZE];
    __shared__ float inv_prefix_sh[CHUNK_SIZE];
    __shared__ float beta_sh[CHUNK_SIZE];
    __shared__ float kk_sh[CHUNK_SIZE * CHUNK_SIZE];
    __shared__ float kq_sh[CHUNK_SIZE * CHUNK_SIZE];

    for (int i = threadIdx.x; i < CHUNK_SIZE; i += blockDim.x) {
        prefix_sh[i] = prefix_head[i];
        inv_prefix_sh[i] = 1.0f / prefix_head[i];
        beta_sh[i] = beta_head[i];
    }
    for (int i = threadIdx.x; i < CHUNK_SIZE * CHUNK_SIZE; i += blockDim.x) {
        kk_sh[i] = kk_head[i];
        kq_sh[i] = kq_head[i];
    }
    __syncthreads();

    float delta_scaled[CHUNK_SIZE];

    #pragma unroll
    for (int t = 0; t < CHUNK_SIZE; ++t) {
        float prefix_t = prefix_sh[t];
        float old_v = old_v_head[vi * CHUNK_SIZE + t];
        #pragma unroll
        for (int j = 0; j < t; ++j) {
            old_v += prefix_t * kk_sh[j * CHUNK_SIZE + t] * delta_scaled[j];
        }

        float delta_t = beta_sh[t] * (v_head[t * D + vi] - old_v);
        delta_scaled[t] = delta_t * inv_prefix_sh[t];
        delta_dst[vi * CHUNK_SIZE + t] = delta_t;

        float out_t = out_head[vi * CHUNK_SIZE + t];
        #pragma unroll
        for (int j = 0; j < t; ++j) {
            out_t += scale * prefix_t * kq_sh[j * CHUNK_SIZE + t] * delta_scaled[j];
        }
        out_t += scale * kq_sh[t * CHUNK_SIZE + t] * delta_t;
        out_dst[vi * CHUNK_SIZE + t] = out_t;
    }
}

__global__ void chunk_correction_heads_outdelta_kernel_runtime(
    const float* __restrict__ old_v_init,
    const float* __restrict__ out_init,
    const float* __restrict__ kk,
    const float* __restrict__ kq,
    const float* __restrict__ v,
    const float* __restrict__ prefix,
    const float* __restrict__ beta,
    float scale,
    float* __restrict__ out,
    float* __restrict__ deltas_out,
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
    float* out_dst = out + h * D * chunk_size;
    float* delta_dst = deltas_out + h * D * chunk_size;

    float deltas[64];

    for (int t = 0; t < chunk_size; ++t) {
        float old_v = old_v_head[vi * chunk_size + t];
        for (int j = 0; j < t; ++j) {
            float decay = prefix_head[t] / prefix_head[j];
            old_v += decay * kk_head[j * chunk_size + t] * deltas[j];
        }

        float delta_t = beta_head[t] * (v_head[t * D + vi] - old_v);
        deltas[t] = delta_t;
        delta_dst[vi * chunk_size + t] = delta_t;

        float out_t = out_head[vi * chunk_size + t];
        for (int j = 0; j < t; ++j) {
            float decay = prefix_head[t] / prefix_head[j];
            out_t += scale * decay * kq_head[j * chunk_size + t] * deltas[j];
        }
        out_t += scale * kq_head[t * chunk_size + t] * delta_t;
        out_dst[vi * chunk_size + t] = out_t;
    }
}

extern "C" void run_chunk_correction_heads_outdelta(
    const void* old_v_init,
    const void* out_init,
    const void* kk,
    const void* kq,
    const void* v,
    const void* prefix,
    const void* beta,
    float scale,
    void* out,
    void* deltas_out,
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
    auto out_ptr = static_cast<float*>(out);
    auto deltas_ptr = static_cast<float*>(deltas_out);

    if (chunk_size == 8) {
        chunk_correction_heads_outdelta_kernel_fixed<8><<<grid, block>>>(
            old_v_ptr, out_init_ptr, kk_ptr, kq_ptr, v_ptr, prefix_ptr, beta_ptr,
            scale, out_ptr, deltas_ptr, num_heads, D
        );
        return;
    }
    if (chunk_size == 16) {
        chunk_correction_heads_outdelta_kernel_fixed<16><<<grid, block>>>(
            old_v_ptr, out_init_ptr, kk_ptr, kq_ptr, v_ptr, prefix_ptr, beta_ptr,
            scale, out_ptr, deltas_ptr, num_heads, D
        );
        return;
    }
    if (chunk_size == 32) {
        chunk_correction_heads_outdelta_kernel_fixed<32><<<grid, block>>>(
            old_v_ptr, out_init_ptr, kk_ptr, kq_ptr, v_ptr, prefix_ptr, beta_ptr,
            scale, out_ptr, deltas_ptr, num_heads, D
        );
        return;
    }
    if (chunk_size == 64) {
        chunk_correction_heads_outdelta_kernel_fixed<64><<<grid, block>>>(
            old_v_ptr, out_init_ptr, kk_ptr, kq_ptr, v_ptr, prefix_ptr, beta_ptr,
            scale, out_ptr, deltas_ptr, num_heads, D
        );
        return;
    }
    chunk_correction_heads_outdelta_kernel_runtime<<<grid, block>>>(
        old_v_ptr, out_init_ptr, kk_ptr, kq_ptr, v_ptr, prefix_ptr, beta_ptr,
        scale, out_ptr, deltas_ptr, num_heads, chunk_size, D
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

    correction_heads_outdelta = lib.run_chunk_correction_heads_outdelta
    correction_heads_outdelta.argtypes = [
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
    correction_heads_outdelta.restype = None

    get_last_gemm_batched_path = lib.get_last_gemm_batched_path
    get_last_gemm_batched_path.argtypes = []
    get_last_gemm_batched_path.restype = ctypes.c_int

    _PROTO_LIB = {
        "gemm": gemm,
        "gemm_batched": gemm_batched,
        "correction": correction,
        "correction_heads": correction_heads,
        "correction_heads_outdelta": correction_heads_outdelta,
        "get_last_gemm_batched_path": get_last_gemm_batched_path,
    }
    return _PROTO_LIB


def get_last_gemm_batched_path() -> int:
    lib = _load_library()
    return int(lib["get_last_gemm_batched_path"]())


def _measure_stage_ms(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000.0


def _new_profile(path: str, chunk_size: int) -> dict:
    return {
        "path": path,
        "chunk_size": int(chunk_size),
        "chunks": 0,
        "groups": 0,
        "gate_ms": 0.0,
        "head_expand_ms": 0.0,
        "prefix_pack_ms": 0.0,
        "gemm_ms": 0.0,
        "gram_ms": 0.0,
        "correction_ms": 0.0,
        "state_update_ms": 0.0,
        "group_gather_ms": 0.0,
        "scatter_ms": 0.0,
        "total_ms": 0.0,
        "orchestration_ms": 0.0,
    }


def _merge_profile(dst: dict, src: dict):
    for key in (
        "chunks",
        "groups",
        "gate_ms",
        "head_expand_ms",
        "prefix_pack_ms",
        "gemm_ms",
        "gram_ms",
        "correction_ms",
        "state_update_ms",
        "group_gather_ms",
        "scatter_ms",
    ):
        dst[key] += src.get(key, 0.0)


def _finalize_profile(profile: dict):
    accounted = (
        profile["gate_ms"]
        + profile["head_expand_ms"]
        + profile["prefix_pack_ms"]
        + profile["gemm_ms"]
        + profile["gram_ms"]
        + profile["correction_ms"]
        + profile["state_update_ms"]
        + profile["group_gather_ms"]
        + profile["scatter_ms"]
    )
    profile["orchestration_ms"] += max(profile["total_ms"] - accounted, 0.0)
    return profile


def _run_correction_heads(
    correction_heads,
    old_v_init: torch.Tensor,
    out_init: torch.Tensor,
    kk: torch.Tensor,
    kq: torch.Tensor,
    v_in: torch.Tensor,
    prefix_in: torch.Tensor,
    beta_in: torch.Tensor,
    state_c: torch.Tensor,
    k_c: torch.Tensor,
    scale: float,
):
    num_heads = state_c.shape[0]
    chunk_size = kk.shape[-1]
    D = state_c.shape[-1]

    out = torch.empty(num_heads, D, chunk_size, device=state_c.device, dtype=torch.float32)
    final_state = torch.empty_like(state_c)
    correction_heads(
        old_v_init.data_ptr(),
        out_init.data_ptr(),
        kk.data_ptr(),
        kq.data_ptr(),
        v_in.data_ptr(),
        prefix_in.data_ptr(),
        beta_in.data_ptr(),
        state_c.data_ptr(),
        k_c.data_ptr(),
        ctypes.c_float(scale),
        out.data_ptr(),
        final_state.data_ptr(),
        num_heads,
        chunk_size,
        D,
    )
    _maybe_synchronize()
    return out, final_state


def _run_correction_heads_outdelta(
    correction_heads_outdelta,
    old_v_init: torch.Tensor,
    out_init: torch.Tensor,
    kk: torch.Tensor,
    kq: torch.Tensor,
    v_in: torch.Tensor,
    prefix_in: torch.Tensor,
    beta_in: torch.Tensor,
    scale: float,
):
    num_heads = old_v_init.shape[0]
    D = old_v_init.shape[1]
    chunk_size = old_v_init.shape[2]

    out = torch.empty(num_heads, D, chunk_size, device=old_v_init.device, dtype=torch.float32)
    deltas = torch.empty_like(out)
    correction_heads_outdelta(
        old_v_init.data_ptr(),
        out_init.data_ptr(),
        kk.data_ptr(),
        kq.data_ptr(),
        v_in.data_ptr(),
        prefix_in.data_ptr(),
        beta_in.data_ptr(),
        ctypes.c_float(scale),
        out.data_ptr(),
        deltas.data_ptr(),
        num_heads,
        chunk_size,
        D,
    )
    _maybe_synchronize()
    return out, deltas


def _state_update_from_deltas(
    deltas: torch.Tensor,
    k_rows: torch.Tensor,
    prefix_vals: torch.Tensor,
    base_state: torch.Tensor,
):
    tail_decay = prefix_vals[:, -1].contiguous()
    weighted_k = ((tail_decay[:, None] / prefix_vals).unsqueeze(-1) * k_rows).float().contiguous()
    return tail_decay[:, None, None] * base_state.float().contiguous() + torch.matmul(deltas.float(), weighted_k)


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
    correction_heads_outdelta = lib["correction_heads_outdelta"]

    num_v_heads, chunk_size, D = q_heads.shape
    prefix_heads = torch.cumprod(g_heads, dim=1).contiguous()

    state_bf16 = state.to(torch.bfloat16).contiguous()
    k_init = (prefix_heads[:, :, None] * k_heads).to(torch.bfloat16).contiguous()
    q_init = (prefix_heads[:, :, None] * q_heads).to(torch.bfloat16).contiguous()

    old_v_init, out_init = gemm_state_cols_pair_batched(state_bf16, k_init, q_init)
    old_v_init = old_v_init.float().contiguous()   # [H, V, C]
    out_init = (out_init.float() * scale).contiguous()
    kk, kq = pairwise_k_products(k_heads, q_heads)  # [H, C, C]

    correction_heads_outdelta = lib["correction_heads_outdelta"]
    out, deltas = _run_correction_heads_outdelta(
        correction_heads_outdelta,
        old_v_init,
        out_init,
        kk,
        kq,
        v_heads,
        prefix_heads,
        beta_heads,
        scale,
    )
    final_state = _state_update_from_deltas(deltas, k_heads, prefix_heads, state)
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
    correction_heads_outdelta = lib["correction_heads_outdelta"]

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

        if flat_batches >= FUSED_CORRECTION_MIN_BATCHES:
            out_flat, final_state_flat = _run_correction_heads(
                correction_heads,
                old_v_init,
                out_init,
                kk,
                kq,
                v_flat,
                prefix_flat,
                beta_flat,
                state_flat,
                k_flat,
                scale,
            )
        else:
            out_flat, deltas_flat = _run_correction_heads_outdelta(
                correction_heads_outdelta,
                old_v_init,
                out_init,
                kk,
                kq,
                v_flat,
                prefix_flat,
                beta_flat,
                scale,
            )
            final_state_flat = _state_update_from_deltas(deltas_flat, k_flat, prefix_flat, state_flat)

        out_heads = out_flat.view(num_seqs, num_v_heads, D, actual_chunk).permute(0, 3, 1, 2).contiguous()
        out[:, chunk_start:chunk_end] = out_heads.to(torch.bfloat16)
        state_seq = final_state_flat.view(num_seqs, num_v_heads, D, D).contiguous()

    return out.view(total_tokens, num_v_heads, D), state_seq


def chunked_batch_heads_cuda(
    q_heads_batch: torch.Tensor,
    k_heads_batch: torch.Tensor,
    v_heads_batch: torch.Tensor,
    state_batch: torch.Tensor,
    g_heads_batch: torch.Tensor,
    beta_heads_batch: torch.Tensor,
    scale: float,
):
    """
    Batched chunk update for a set of sequences that share the same actual chunk size.

    Shapes:
      q/k/v:      [N, H, C, D]
      g/beta:     [N, H, C]
      state:      [N, H, D, D]
    Returns:
      out:        [N, C, H, D]
      new_state:  [N, H, D, D]
    """
    num_seqs, num_v_heads, actual_chunk, D = q_heads_batch.shape
    if num_seqs <= 0:
        return (
            torch.empty(0, actual_chunk, num_v_heads, D, dtype=torch.bfloat16, device=q_heads_batch.device),
            state_batch,
        )

    prefix_heads = torch.cumprod(g_heads_batch, dim=-1).contiguous()
    flat_batches = num_seqs * num_v_heads
    state_bf16 = state_batch.to(torch.bfloat16).contiguous().view(flat_batches, D, D)
    k_init = (prefix_heads[..., None] * k_heads_batch).to(torch.bfloat16).contiguous().view(flat_batches, actual_chunk, D)
    q_init = (prefix_heads[..., None] * q_heads_batch).to(torch.bfloat16).contiguous().view(flat_batches, actual_chunk, D)

    old_v_init, out_init = gemm_state_cols_pair_batched(state_bf16, k_init, q_init)
    old_v_init = old_v_init.float().contiguous()
    out_init = (out_init.float() * scale).contiguous()

    kk, kq = pairwise_k_products(k_heads_batch, q_heads_batch)
    kk = kk.view(flat_batches, actual_chunk, actual_chunk)
    kq = kq.view(flat_batches, actual_chunk, actual_chunk)
    v_flat = v_heads_batch.contiguous().view(flat_batches, actual_chunk, D)
    prefix_flat = prefix_heads.contiguous().view(flat_batches, actual_chunk)
    beta_flat = beta_heads_batch.contiguous().view(flat_batches, actual_chunk)
    state_flat = state_batch.contiguous().view(flat_batches, D, D)
    k_flat = k_heads_batch.contiguous().view(flat_batches, actual_chunk, D)

    lib = _load_library()
    correction_heads = lib["correction_heads"]
    correction_heads_outdelta = lib["correction_heads_outdelta"]

    if flat_batches >= FUSED_CORRECTION_MIN_BATCHES:
        out_flat, final_state_flat = _run_correction_heads(
            correction_heads,
            old_v_init,
            out_init,
            kk,
            kq,
            v_flat,
            prefix_flat,
            beta_flat,
            state_flat,
            k_flat,
            scale,
        )
    else:
        out_flat, deltas_flat = _run_correction_heads_outdelta(
            correction_heads_outdelta,
            old_v_init,
            out_init,
            kk,
            kq,
            v_flat,
            prefix_flat,
            beta_flat,
            scale,
        )
        final_state_flat = _state_update_from_deltas(deltas_flat, k_flat, prefix_flat, state_flat)

    out_batch = out_flat.view(num_seqs, num_v_heads, D, actual_chunk).permute(0, 3, 1, 2).contiguous()
    new_state_batch = final_state_flat.view(num_seqs, num_v_heads, D, D).contiguous()
    return out_batch, new_state_batch


def profile_chunked_batch_heads_cuda(
    q_heads_batch: torch.Tensor,
    k_heads_batch: torch.Tensor,
    v_heads_batch: torch.Tensor,
    state_batch: torch.Tensor,
    g_heads_batch: torch.Tensor,
    beta_heads_batch: torch.Tensor,
    scale: float,
):
    num_seqs, num_v_heads, actual_chunk, D = q_heads_batch.shape
    profile = _new_profile("wavefront_chunk", actual_chunk)
    profile["chunks"] = 1

    def build_prefetch():
        prefix_heads = torch.cumprod(g_heads_batch, dim=-1).contiguous()
        flat_batches = num_seqs * num_v_heads
        state_bf16 = state_batch.to(torch.bfloat16).contiguous().view(flat_batches, D, D)
        k_init = (prefix_heads[..., None] * k_heads_batch).to(torch.bfloat16).contiguous().view(flat_batches, actual_chunk, D)
        q_init = (prefix_heads[..., None] * q_heads_batch).to(torch.bfloat16).contiguous().view(flat_batches, actual_chunk, D)
        state_flat = state_batch.contiguous().view(flat_batches, D, D)
        v_flat = v_heads_batch.contiguous().view(flat_batches, actual_chunk, D)
        prefix_flat = prefix_heads.contiguous().view(flat_batches, actual_chunk)
        beta_flat = beta_heads_batch.contiguous().view(flat_batches, actual_chunk)
        k_flat = k_heads_batch.contiguous().view(flat_batches, actual_chunk, D)
        return flat_batches, q_init, k_init, state_bf16, state_flat, v_flat, prefix_flat, beta_flat, k_flat, prefix_heads

    (
        flat_batches,
        q_init,
        k_init,
        state_bf16,
        state_flat,
        v_flat,
        prefix_flat,
        beta_flat,
        k_flat,
        prefix_heads,
    ), ms = _measure_stage_ms(build_prefetch)
    profile["prefix_pack_ms"] += ms

    (old_v_init, out_init), ms = _measure_stage_ms(
        lambda: gemm_state_cols_pair_batched(state_bf16, k_init, q_init)
    )
    profile["gemm_ms"] += ms
    old_v_init = old_v_init.float().contiguous()
    out_init = (out_init.float() * scale).contiguous()

    (kk, kq), ms = _measure_stage_ms(lambda: pairwise_k_products(k_heads_batch, q_heads_batch))
    profile["gram_ms"] += ms
    kk = kk.view(flat_batches, actual_chunk, actual_chunk)
    kq = kq.view(flat_batches, actual_chunk, actual_chunk)

    lib = _load_library()
    correction_heads = lib["correction_heads"]
    correction_heads_outdelta = lib["correction_heads_outdelta"]

    if flat_batches >= FUSED_CORRECTION_MIN_BATCHES:
        (out_flat, final_state_flat), ms = _measure_stage_ms(
            lambda: _run_correction_heads(
                correction_heads,
                old_v_init,
                out_init,
                kk,
                kq,
                v_flat,
                prefix_flat,
                beta_flat,
                state_flat,
                k_flat,
                scale,
            )
        )
        profile["correction_ms"] += ms
    else:
        (out_flat, deltas_flat), ms = _measure_stage_ms(
            lambda: _run_correction_heads_outdelta(
                correction_heads_outdelta,
                old_v_init,
                out_init,
                kk,
                kq,
                v_flat,
                prefix_flat,
                beta_flat,
                scale,
            )
        )
        profile["correction_ms"] += ms

        def build_state():
            return _state_update_from_deltas(deltas_flat, k_flat, prefix_flat, state_flat)

        final_state_flat, ms = _measure_stage_ms(build_state)
        profile["state_update_ms"] += ms

    out_batch = out_flat.view(num_seqs, num_v_heads, D, actual_chunk).permute(0, 3, 1, 2).contiguous()
    new_state_batch = final_state_flat.view(num_seqs, num_v_heads, D, D).contiguous()
    return out_batch, new_state_batch, _finalize_profile(profile)


def chunked_prefill_end_to_end_wavefront(
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
    Varlen wavefront path.

    Sequences are expanded once, then processed by chunk position. Sequences with
    the same actual chunk size at the current position are merged into one batch.
    """
    total_tokens, _, D = q.shape
    num_v_heads = v.shape[1]
    num_seqs = cu_seqlens.numel() - 1
    device = q.device

    x = a.float() + dt_bias.float()
    g_all = torch.exp(-torch.exp(A_log.float()) * torch.nn.functional.softplus(x))
    beta_all = torch.sigmoid(b.float())

    out = torch.empty(total_tokens, num_v_heads, D, dtype=torch.bfloat16, device=device)
    new_state = state.clone().float()

    (
        lengths,
        starts,
        max_len,
        q_heads_pad,
        k_heads_pad,
        v_heads_pad,
        g_heads_pad,
        beta_heads_pad,
        sort_order,
        inverse_order,
        _padded_idx,
    ) = _build_padded_wavefront_cache(q, k, v, g_all, beta_all, cu_seqlens, num_v_heads)
    new_state = state.index_select(0, sort_order).clone().float() if sort_order.numel() else state.clone().float()
    wavefront_plan = _build_wavefront_plan(lengths, starts, chunk_size, device)
    out = torch.empty(total_tokens, num_v_heads, D, dtype=torch.bfloat16, device=device)

    for chunk_start, chunk_groups in wavefront_plan:
        for actual_chunk, group_start, group_end, token_idx in chunk_groups:
            q_batch = q_heads_pad[group_start:group_end, :, chunk_start:chunk_start + actual_chunk, :]
            k_batch = k_heads_pad[group_start:group_end, :, chunk_start:chunk_start + actual_chunk, :]
            v_batch = v_heads_pad[group_start:group_end, :, chunk_start:chunk_start + actual_chunk, :]
            g_batch = g_heads_pad[group_start:group_end, :, chunk_start:chunk_start + actual_chunk]
            beta_batch = beta_heads_pad[group_start:group_end, :, chunk_start:chunk_start + actual_chunk]
            state_batch = new_state[group_start:group_end]

            out_batch, new_state_batch = chunked_batch_heads_cuda(
                q_batch.contiguous(),
                k_batch.contiguous(),
                v_batch.contiguous(),
                state_batch.contiguous(),
                g_batch.contiguous(),
                beta_batch.contiguous(),
                scale,
            )

            out.index_copy_(0, token_idx, out_batch.to(torch.bfloat16).reshape(-1, num_v_heads, D))
            new_state[group_start:group_end] = new_state_batch

    if inverse_order.numel():
        new_state = new_state.index_select(0, inverse_order)
    return out, new_state


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

        seq_idx_tensor, token_idx = _build_group_token_indices(cu_seqlens, seq_indices, length, device)
        gathered_state = new_state.index_select(0, seq_idx_tensor).clone()
        q_group = q.index_select(0, token_idx)
        k_group = k.index_select(0, token_idx)
        v_group = v.index_select(0, token_idx)
        a_group = a.index_select(0, token_idx)
        b_group = b.index_select(0, token_idx)
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

        out.index_copy_(0, token_idx, out_group)
        new_state.index_copy_(0, seq_idx_tensor, state_group)

    return out, new_state


def profile_chunked_sequence_heads_cuda(
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
    correction_heads_outdelta = lib["correction_heads_outdelta"]
    num_v_heads, chunk_size, D = q_heads.shape
    profile = _new_profile("chunk", chunk_size)
    profile["chunks"] = 1

    def build_prefetch():
        prefix_heads = torch.cumprod(g_heads, dim=1).contiguous()
        state_bf16 = state.to(torch.bfloat16).contiguous()
        k_init = (prefix_heads[:, :, None] * k_heads).to(torch.bfloat16).contiguous()
        q_init = (prefix_heads[:, :, None] * q_heads).to(torch.bfloat16).contiguous()
        return prefix_heads, state_bf16, k_init, q_init

    (prefix_heads, state_bf16, k_init, q_init), ms = _measure_stage_ms(build_prefetch)
    profile["prefix_pack_ms"] += ms

    (old_v_init, out_init), ms = _measure_stage_ms(
        lambda: gemm_state_cols_pair_batched(state_bf16, k_init, q_init)
    )
    profile["gemm_ms"] += ms
    old_v_init = old_v_init.float().contiguous()
    out_init = (out_init.float() * scale).contiguous()

    (kk, kq), ms = _measure_stage_ms(lambda: pairwise_k_products(k_heads, q_heads))
    profile["gram_ms"] += ms

    (out, deltas), ms = _measure_stage_ms(
        lambda: _run_correction_heads_outdelta(
            correction_heads_outdelta,
            old_v_init,
            out_init,
            kk,
            kq,
            v_heads,
            prefix_heads,
            beta_heads,
            scale,
        )
    )
    profile["correction_ms"] += ms

    def build_state():
        return _state_update_from_deltas(deltas, k_heads, prefix_heads, state)

    final_state, ms = _measure_stage_ms(build_state)
    profile["state_update_ms"] += ms

    return out.permute(2, 0, 1).contiguous(), final_state, profile


def profile_chunked_prefill_end_to_end_batched(
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
    profile = _new_profile("batched", chunk_size)

    def build_gates():
        x = a.float() + dt_bias.float()
        g_all = torch.exp(-torch.exp(A_log.float()) * torch.nn.functional.softplus(x))
        beta_all = torch.sigmoid(b.float())
        return g_all, beta_all

    (g_all, beta_all), ms = _measure_stage_ms(build_gates)
    profile["gate_ms"] += ms

    torch.cuda.synchronize()
    total_start = time.perf_counter()

    out = torch.empty(total_tokens, num_v_heads, D, dtype=torch.bfloat16, device=device)
    new_state = state.clone().float()

    for seq_idx in range(num_seqs):
        start = int(cu_seqlens[seq_idx].item())
        end = int(cu_seqlens[seq_idx + 1].item())
        if end <= start:
            continue

        def build_heads():
            q_heads_full = _expand_qk_heads(q[start:end].float(), num_v_heads)
            k_heads_full = _expand_qk_heads(k[start:end].float(), num_v_heads)
            v_heads_full = v[start:end].float().permute(1, 0, 2).contiguous()
            g_heads_full = g_all[start:end].float().permute(1, 0).contiguous()
            beta_heads_full = beta_all[start:end].float().permute(1, 0).contiguous()
            return q_heads_full, k_heads_full, v_heads_full, g_heads_full, beta_heads_full

        (q_heads_full, k_heads_full, v_heads_full, g_heads_full, beta_heads_full), ms = _measure_stage_ms(build_heads)
        profile["head_expand_ms"] += ms

        state_seq = new_state[seq_idx].clone()
        seq_len = end - start
        for local_start in range(0, seq_len, chunk_size):
            local_end = min(local_start + chunk_size, seq_len)
            out_chunk, state_seq, chunk_profile = profile_chunked_sequence_heads_cuda(
                q_heads_full[:, local_start:local_end, :],
                k_heads_full[:, local_start:local_end, :],
                v_heads_full[:, local_start:local_end, :],
                state_seq,
                g_heads_full[:, local_start:local_end],
                beta_heads_full[:, local_start:local_end],
                scale,
            )
            _merge_profile(profile, chunk_profile)
            _, ms = _measure_stage_ms(
                lambda: out.__setitem__(
                    slice(start + local_start, start + local_end),
                    out_chunk.to(torch.bfloat16),
                )
            )
            profile["scatter_ms"] += ms
        new_state[seq_idx] = state_seq

    torch.cuda.synchronize()
    profile["total_ms"] = (time.perf_counter() - total_start) * 1000.0
    return out, new_state, _finalize_profile(profile)


def profile_chunked_prefill_end_to_end_uniform_batch(
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
    total_tokens, num_q_heads, D = q.shape
    num_v_heads = v.shape[1]
    num_seqs = cu_seqlens.numel() - 1
    if num_seqs <= 0:
        return torch.empty_like(v), state.clone(), _finalize_profile(_new_profile("uniform", chunk_size))

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if not torch.all(lengths == lengths[0]):
        raise ValueError("profile_chunked_prefill_end_to_end_uniform_batch requires equal-length sequences")

    seq_len = int(lengths[0].item())
    profile = _new_profile("uniform", chunk_size)

    def build_gates():
        x = a.float() + dt_bias.float()
        g_all = torch.exp(-torch.exp(A_log.float()) * torch.nn.functional.softplus(x))
        beta_all = torch.sigmoid(b.float())
        return g_all, beta_all

    (g_all, beta_all), ms = _measure_stage_ms(build_gates)
    profile["gate_ms"] += ms

    torch.cuda.synchronize()
    total_start = time.perf_counter()

    out = torch.empty(num_seqs, seq_len, num_v_heads, D, dtype=torch.bfloat16, device=q.device)
    state_seq = state.clone().float()

    ratio = num_v_heads // num_q_heads

    def build_heads():
        q_heads_full = q.view(num_seqs, seq_len, num_q_heads, D).float()
        q_heads_full = q_heads_full.repeat_interleave(ratio, dim=2).permute(0, 2, 1, 3).contiguous()
        k_heads_full = k.view(num_seqs, seq_len, num_q_heads, D).float()
        k_heads_full = k_heads_full.repeat_interleave(ratio, dim=2).permute(0, 2, 1, 3).contiguous()
        v_heads_full = v.view(num_seqs, seq_len, num_v_heads, D).float().permute(0, 2, 1, 3).contiguous()
        g_heads_full = g_all.view(num_seqs, seq_len, num_v_heads).float().permute(0, 2, 1).contiguous()
        beta_heads_full = beta_all.view(num_seqs, seq_len, num_v_heads).float().permute(0, 2, 1).contiguous()
        return q_heads_full, k_heads_full, v_heads_full, g_heads_full, beta_heads_full

    (q_heads_full, k_heads_full, v_heads_full, g_heads_full, beta_heads_full), ms = _measure_stage_ms(build_heads)
    profile["head_expand_ms"] += ms
    lib = _load_library()
    correction_heads = lib["correction_heads"]
    correction_heads_outdelta = lib["correction_heads_outdelta"]

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        actual_chunk = chunk_end - chunk_start

        def build_prefetch():
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
            state_flat = state_seq.contiguous().view(flat_batches, D, D)
            v_flat = v_heads.contiguous().view(flat_batches, actual_chunk, D)
            prefix_flat = prefix_heads.contiguous().view(flat_batches, actual_chunk)
            beta_flat = beta_heads.contiguous().view(flat_batches, actual_chunk)
            k_flat = k_heads.contiguous().view(flat_batches, actual_chunk, D)
            return flat_batches, q_heads, k_heads, v_flat, prefix_flat, beta_flat, state_bf16, state_flat, k_init, q_init, k_flat

        (
            flat_batches,
            q_heads,
            k_heads,
            v_flat,
            prefix_flat,
            beta_flat,
            state_bf16,
            state_flat,
            k_init,
            q_init,
            k_flat,
        ), ms = _measure_stage_ms(build_prefetch)
        profile["prefix_pack_ms"] += ms

        (old_v_init, out_init), ms = _measure_stage_ms(
            lambda: gemm_state_cols_pair_batched(state_bf16, k_init, q_init)
        )
        profile["gemm_ms"] += ms
        old_v_init = old_v_init.float().contiguous()
        out_init = (out_init.float() * scale).contiguous()

        (kk, kq), ms = _measure_stage_ms(lambda: pairwise_k_products(k_heads, q_heads))
        profile["gram_ms"] += ms
        kk = kk.view(flat_batches, actual_chunk, actual_chunk)
        kq = kq.view(flat_batches, actual_chunk, actual_chunk)

        if flat_batches >= FUSED_CORRECTION_MIN_BATCHES:
            (out_flat, final_state_flat), ms = _measure_stage_ms(
                lambda: _run_correction_heads(
                    correction_heads,
                    old_v_init,
                    out_init,
                    kk,
                    kq,
                    v_flat,
                    prefix_flat,
                    beta_flat,
                    state_flat,
                    k_flat,
                    scale,
                )
            )
            profile["correction_ms"] += ms
        else:
            (out_flat, deltas_flat), ms = _measure_stage_ms(
                lambda: _run_correction_heads_outdelta(
                    correction_heads_outdelta,
                    old_v_init,
                    out_init,
                    kk,
                    kq,
                    v_flat,
                    prefix_flat,
                    beta_flat,
                    scale,
                )
            )
            profile["correction_ms"] += ms

            def build_state():
                return _state_update_from_deltas(deltas_flat, k_flat, prefix_flat, state_flat)

            final_state_flat, ms = _measure_stage_ms(build_state)
            profile["state_update_ms"] += ms
        profile["chunks"] += 1

        def scatter_chunk():
            out_heads = out_flat.view(num_seqs, num_v_heads, D, actual_chunk).permute(0, 3, 1, 2).contiguous()
            out[:, chunk_start:chunk_end] = out_heads.to(torch.bfloat16)
            return final_state_flat.view(num_seqs, num_v_heads, D, D).contiguous()

        state_seq, ms = _measure_stage_ms(scatter_chunk)
        profile["scatter_ms"] += ms

    torch.cuda.synchronize()
    profile["total_ms"] = (time.perf_counter() - total_start) * 1000.0
    return out.view(total_tokens, num_v_heads, D), state_seq, _finalize_profile(profile)


def profile_chunked_prefill_end_to_end_grouped(
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
    device = q.device
    profile = _new_profile("grouped", chunk_size)

    torch.cuda.synchronize()
    total_start = time.perf_counter()

    out = torch.empty(total_tokens, num_v_heads, D, dtype=torch.bfloat16, device=device)
    new_state = state.clone().float()

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    groups = {}
    for seq_idx, length in enumerate(lengths):
        groups.setdefault(int(length), []).append(seq_idx)

    for length, seq_indices in groups.items():
        if length <= 0:
            continue
        profile["groups"] += 1

        if len(seq_indices) == 1:
            seq_idx = seq_indices[0]
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())

            def gather_single():
                q_group = q[start:end]
                k_group = k[start:end]
                v_group = v[start:end]
                a_group = a[start:end]
                b_group = b[start:end]
                cu_group = torch.tensor([0, length], dtype=cu_seqlens.dtype, device=device)
                state_group = new_state[seq_idx : seq_idx + 1].clone()
                return q_group, k_group, v_group, a_group, b_group, cu_group, state_group

            (q_group, k_group, v_group, a_group, b_group, cu_group, state_group), ms = _measure_stage_ms(gather_single)
            profile["group_gather_ms"] += ms

            out_group, updated_state, subprofile = profile_chunked_prefill_end_to_end_batched(
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
            _merge_profile(profile, subprofile)

            _, ms = _measure_stage_ms(
                lambda: (
                    out.__setitem__(slice(start, end), out_group),
                    new_state.__setitem__(seq_idx, updated_state[0]),
                )
            )
            profile["scatter_ms"] += ms
            continue

        def gather_group():
            seq_idx_tensor, token_idx = _build_group_token_indices(cu_seqlens, seq_indices, length, device)
            gathered_state = new_state.index_select(0, seq_idx_tensor).clone()
            q_group = q.index_select(0, token_idx)
            k_group = k.index_select(0, token_idx)
            v_group = v.index_select(0, token_idx)
            a_group = a.index_select(0, token_idx)
            b_group = b.index_select(0, token_idx)
            cu_group = torch.arange(0, len(seq_indices) * length + 1, length, dtype=cu_seqlens.dtype, device=device)
            return q_group, k_group, v_group, a_group, b_group, cu_group, gathered_state, seq_idx_tensor, token_idx

        (
            q_group,
            k_group,
            v_group,
            a_group,
            b_group,
            cu_group,
            gathered_state,
            seq_idx_tensor,
            token_idx,
        ), ms = _measure_stage_ms(gather_group)
        profile["group_gather_ms"] += ms

        out_group, state_group, subprofile = profile_chunked_prefill_end_to_end_uniform_batch(
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
        _merge_profile(profile, subprofile)

        def scatter_group():
            out.index_copy_(0, token_idx, out_group)
            new_state.index_copy_(0, seq_idx_tensor, state_group)

        _, ms = _measure_stage_ms(scatter_group)
        profile["scatter_ms"] += ms

    torch.cuda.synchronize()
    profile["total_ms"] = (time.perf_counter() - total_start) * 1000.0
    return out, new_state, _finalize_profile(profile)


def profile_chunked_prefill_end_to_end_wavefront(
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
    device = q.device
    profile = _new_profile("wavefront", chunk_size)

    def build_gates():
        x = a.float() + dt_bias.float()
        g_all = torch.exp(-torch.exp(A_log.float()) * torch.nn.functional.softplus(x))
        beta_all = torch.sigmoid(b.float())
        return g_all, beta_all

    (g_all, beta_all), ms = _measure_stage_ms(build_gates)
    profile["gate_ms"] += ms

    def build_cache():
        return _build_padded_wavefront_cache(q, k, v, g_all, beta_all, cu_seqlens, num_v_heads)

    (
        lengths,
        starts,
        max_len,
        q_heads_pad,
        k_heads_pad,
        v_heads_pad,
        g_heads_pad,
        beta_heads_pad,
        sort_order,
        inverse_order,
        _padded_idx,
    ), ms = _measure_stage_ms(build_cache)
    profile["head_expand_ms"] += ms
    new_state = state.index_select(0, sort_order).clone().float() if sort_order.numel() else state.clone().float()
    wavefront_plan, ms = _measure_stage_ms(lambda: _build_wavefront_plan(lengths, starts, chunk_size, device))
    profile["orchestration_ms"] += ms

    torch.cuda.synchronize()
    total_start = time.perf_counter()

    out = torch.empty(total_tokens, num_v_heads, D, dtype=torch.bfloat16, device=device)

    for chunk_start, chunk_groups in wavefront_plan:
        for actual_chunk, group_start, group_end, token_idx in chunk_groups:

            def gather_group():
                q_batch = q_heads_pad[group_start:group_end, :, chunk_start:chunk_start + actual_chunk, :].contiguous()
                k_batch = k_heads_pad[group_start:group_end, :, chunk_start:chunk_start + actual_chunk, :].contiguous()
                v_batch = v_heads_pad[group_start:group_end, :, chunk_start:chunk_start + actual_chunk, :].contiguous()
                g_batch = g_heads_pad[group_start:group_end, :, chunk_start:chunk_start + actual_chunk].contiguous()
                beta_batch = beta_heads_pad[group_start:group_end, :, chunk_start:chunk_start + actual_chunk].contiguous()
                state_batch = new_state[group_start:group_end].contiguous()
                return q_batch, k_batch, v_batch, g_batch, beta_batch, state_batch, token_idx

            (q_batch, k_batch, v_batch, g_batch, beta_batch, state_batch, token_idx), ms = _measure_stage_ms(gather_group)
            profile["group_gather_ms"] += ms

            out_batch, new_state_batch, subprofile = profile_chunked_batch_heads_cuda(
                q_batch,
                k_batch,
                v_batch,
                state_batch,
                g_batch,
                beta_batch,
                scale,
            )
            _merge_profile(profile, subprofile)

            def scatter_group():
                out.index_copy_(0, token_idx, out_batch.to(torch.bfloat16).reshape(-1, num_v_heads, D))
                new_state[group_start:group_end] = new_state_batch

            _, ms = _measure_stage_ms(scatter_group)
            profile["scatter_ms"] += ms

    torch.cuda.synchronize()
    profile["total_ms"] = (time.perf_counter() - total_start) * 1000.0
    if inverse_order.numel():
        new_state = new_state.index_select(0, inverse_order)
    return out, new_state, _finalize_profile(profile)


def profile_kernel(
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
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(q.shape[-1])
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if chunk_size is None or chunk_size <= 0:
        chunk_size = recommend_chunk_size(cu_seqlens)
    if torch.all(lengths == lengths[0]):
        if lengths.numel() == 1:
            return profile_chunked_prefill_end_to_end_batched(
                q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, float(scale), chunk_size=chunk_size
            )
        return profile_chunked_prefill_end_to_end_uniform_batch(
            q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, float(scale), chunk_size=chunk_size
        )
    if _should_use_grouped_varlen(lengths):
        return profile_chunked_prefill_end_to_end_grouped(
            q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, float(scale), chunk_size=chunk_size
        )
    return profile_chunked_prefill_end_to_end_wavefront(
        q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, float(scale), chunk_size=chunk_size
    )


def _build_group_token_indices(cu_seqlens: torch.Tensor, seq_indices, length: int, device):
    seq_idx_tensor = torch.as_tensor(seq_indices, device=device, dtype=torch.long)
    starts = cu_seqlens.index_select(0, seq_idx_tensor).to(torch.long)
    offsets = torch.arange(length, device=device, dtype=torch.long)
    token_idx = (starts[:, None] + offsets[None, :]).reshape(-1)
    return seq_idx_tensor, token_idx


def _build_wavefront_token_indices(starts: torch.Tensor, chunk_start: int, actual_chunk: int, device):
    offsets = torch.arange(actual_chunk, device=device, dtype=torch.long)
    return (starts[:, None] + chunk_start + offsets[None, :]).reshape(-1)


def _build_packed_to_padded_token_indices(
    lengths: torch.Tensor,
    starts: torch.Tensor,
    max_len: int,
    device,
    seq_ids: torch.Tensor | None = None,
):
    if lengths.numel() == 0 or max_len <= 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if seq_ids is None:
        seq_ids = torch.arange(lengths.numel(), device=device, dtype=torch.long)
    seq_ids = torch.repeat_interleave(seq_ids.to(torch.long), lengths)
    repeated_starts = torch.repeat_interleave(starts, lengths)
    local_offsets = torch.arange(int(lengths.sum().item()), device=device, dtype=torch.long) - repeated_starts
    return seq_ids * max_len + local_offsets


def _expand_packed_qk_heads(x: torch.Tensor, num_v_heads: int) -> torch.Tensor:
    num_q_heads = x.shape[1]
    ratio = num_v_heads // num_q_heads
    return x.repeat_interleave(ratio, dim=1).contiguous()


def _build_wavefront_plan(lengths: torch.Tensor, starts: torch.Tensor, chunk_size: int, device):
    if lengths.numel() == 0:
        return []
    max_len = int(lengths.max().item())
    plan = []
    for chunk_start in range(0, max_len, chunk_size):
        remaining = lengths - chunk_start
        active_mask = remaining > 0
        if not torch.any(active_mask):
            continue
        active_count = int(active_mask.sum().item())
        actual_active = torch.minimum(
            remaining[:active_count].clamp_min(0),
            torch.full_like(remaining[:active_count], chunk_size),
        )
        chunk_groups = []
        unique_chunks, counts = torch.unique_consecutive(actual_active, return_counts=True)
        offset = 0
        for actual_chunk_tensor, count_tensor in zip(unique_chunks, counts):
            actual_chunk = int(actual_chunk_tensor.item())
            count = int(count_tensor.item())
            group_start = offset
            group_end = offset + count
            token_idx = _build_wavefront_token_indices(
                starts[group_start:group_end],
                chunk_start,
                actual_chunk,
                device,
            )
            chunk_groups.append((actual_chunk, group_start, group_end, token_idx))
            offset = group_end
        plan.append((chunk_start, chunk_groups))
    return plan


def _build_padded_wavefront_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_all: torch.Tensor,
    beta_all: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_v_heads: int,
):
    device = q.device
    orig_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
    num_seqs = orig_lengths.numel()
    orig_starts = cu_seqlens[:-1].to(torch.long)
    max_len = int(orig_lengths.max().item()) if num_seqs else 0
    D = q.shape[-1]

    if num_seqs == 0 or max_len <= 0:
        empty = torch.zeros(num_seqs, num_v_heads, max_len, D, dtype=torch.float32, device=device)
        empty_gate = torch.ones(num_seqs, num_v_heads, max_len, dtype=torch.float32, device=device)
        empty_beta = torch.zeros_like(empty_gate)
        empty_order = torch.empty(0, device=device, dtype=torch.long)
        empty_idx = torch.empty(0, device=device, dtype=torch.long)
        return (
            orig_lengths,
            orig_starts,
            max_len,
            empty,
            empty.clone(),
            empty.clone(),
            empty_gate,
            empty_beta,
            empty_order,
            empty_order,
            empty_idx,
        )

    sort_order = torch.argsort(orig_lengths, descending=True)
    inverse_order = torch.empty_like(sort_order)
    inverse_order[sort_order] = torch.arange(num_seqs, device=device, dtype=torch.long)
    lengths = orig_lengths.index_select(0, sort_order)
    starts = orig_starts.index_select(0, sort_order)

    q_heads_pad = torch.zeros(num_seqs, num_v_heads, max_len, D, dtype=torch.float32, device=device)
    k_heads_pad = torch.zeros_like(q_heads_pad)
    v_heads_pad = torch.zeros_like(q_heads_pad)
    g_heads_pad = torch.ones(num_seqs, num_v_heads, max_len, dtype=torch.float32, device=device)
    beta_heads_pad = torch.zeros_like(g_heads_pad)

    padded_idx = _build_packed_to_padded_token_indices(
        orig_lengths,
        orig_starts,
        max_len,
        device,
        seq_ids=inverse_order,
    )
    linear_shape = (num_seqs * max_len, num_v_heads)

    q_heads_linear = torch.zeros(*linear_shape, D, dtype=torch.float32, device=device)
    k_heads_linear = torch.zeros_like(q_heads_linear)
    v_heads_linear = torch.zeros_like(q_heads_linear)
    g_heads_linear = torch.ones(*linear_shape, dtype=torch.float32, device=device)
    beta_heads_linear = torch.zeros_like(g_heads_linear)

    q_heads_linear.index_copy_(0, padded_idx, _expand_packed_qk_heads(q.float(), num_v_heads))
    k_heads_linear.index_copy_(0, padded_idx, _expand_packed_qk_heads(k.float(), num_v_heads))
    v_heads_linear.index_copy_(0, padded_idx, v.float().contiguous())
    g_heads_linear.index_copy_(0, padded_idx, g_all.float().contiguous())
    beta_heads_linear.index_copy_(0, padded_idx, beta_all.float().contiguous())

    q_heads_pad = q_heads_linear.view(num_seqs, max_len, num_v_heads, D).permute(0, 2, 1, 3).contiguous()
    k_heads_pad = k_heads_linear.view(num_seqs, max_len, num_v_heads, D).permute(0, 2, 1, 3).contiguous()
    v_heads_pad = v_heads_linear.view(num_seqs, max_len, num_v_heads, D).permute(0, 2, 1, 3).contiguous()
    g_heads_pad = g_heads_linear.view(num_seqs, max_len, num_v_heads).permute(0, 2, 1).contiguous()
    beta_heads_pad = beta_heads_linear.view(num_seqs, max_len, num_v_heads).permute(0, 2, 1).contiguous()

    return (
        lengths,
        starts,
        max_len,
        q_heads_pad,
        k_heads_pad,
        v_heads_pad,
        g_heads_pad,
        beta_heads_pad,
        sort_order,
        inverse_order,
        padded_idx,
    )


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
    if torch.all(lengths == lengths[0]):
        if lengths.numel() > 1 and int(lengths[0].item()) >= 64:
            return 32
        return 64
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
    return chunked_prefill_end_to_end_wavefront(
        q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, float(scale), chunk_size=chunk_size
    )

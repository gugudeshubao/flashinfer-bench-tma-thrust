"""
GDN Decode CUDA wrapper.

Current packaged CUDA candidate is the v10 CuTe/TMA family.
Fallback: Triton if CUDA JIT compilation fails.
"""

import math
import os
from pathlib import Path

import torch

try:
    from .backend import BACKEND as _PACKAGED_BACKEND
except Exception:
    try:
        from backend import BACKEND as _PACKAGED_BACKEND
    except Exception:
        _PACKAGED_BACKEND = "cute"

# ============================================================
# CUDA KERNEL SOURCE (EMBEDDED)
# ============================================================

CUDA_SOURCE = r'''
/*
 * GDN Decode v5 — CUDA kernel for B200 (sm100)
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace gdn {

constexpr int D = 128;
constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template<int BLOCK_V>
__global__ void gdn_decode_kernel_v5(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ State,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ DtBias,
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
    const int vb = blockIdx.z;
    const int v0 = vb * BLOCK_V;
    const int qk_h = h / 2;
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    extern __shared__ float smem[];
    float* reduce_smem = smem;
    float* q_smem = smem + 4;
    float* k_smem = q_smem + D;
    float* v_smem = k_smem + D;
    float* state_smem = v_smem + BLOCK_V;
    
    __shared__ float g_shared, beta_shared;
    if (tid == 0) {
        float a_val = __bfloat162float(A[b * stride_a_b + h]);
        float dt_val = __bfloat162float(DtBias[h]);
        float alog = A_log[h];
        float b_val = __bfloat162float(B_gate[b * stride_b_b + h]);
        
        float sp = softplus(a_val + dt_val);
        g_shared = expf(-expf(alog) * sp);
        beta_shared = 1.0f / (1.0f + expf(-b_val));
    }
    __syncthreads();
    
    float g = g_shared;
    float beta = beta_shared;
    
    if (tid < D) {
        q_smem[tid] = __bfloat162float(Q[b * stride_q_b + qk_h * stride_q_h + tid]);
        k_smem[tid] = __bfloat162float(K[b * stride_k_b + qk_h * stride_k_h + tid]);
    }
    
    for (int i = tid; i < BLOCK_V; i += num_threads) {
        v_smem[i] = __bfloat162float(V[b * stride_v_b + h * stride_v_h + v0 + i]);
    }
    __syncthreads();
    
    const float* state_ptr = State + b * stride_s_b + h * stride_s_h + v0 * stride_s_v;
    
    for (int i = tid; i < BLOCK_V * D; i += num_threads) {
        int vi = i / D;
        int ki = i % D;
        state_smem[vi * D + ki] = state_ptr[vi * stride_s_v + ki];
    }
    __syncthreads();
    
    for (int i = tid; i < BLOCK_V * D; i += num_threads) {
        state_smem[i] *= g;
    }
    __syncthreads();
    
    __shared__ float old_v_smem[64];
    
    if (tid < BLOCK_V) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int ki = 0; ki < D; ki++) {
            sum += state_smem[tid * D + ki] * k_smem[ki];
        }
        old_v_smem[tid] = sum;
    }
    __syncthreads();
    
    for (int i = tid; i < BLOCK_V * D; i += num_threads) {
        int vi = i / D;
        int ki = i % D;
        float delta = beta * (v_smem[vi] - old_v_smem[vi]);
        state_smem[vi * D + ki] += delta * k_smem[ki];
    }
    __syncthreads();
    
    __shared__ float out_smem[64];
    
    if (tid < BLOCK_V) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int ki = 0; ki < D; ki++) {
            sum += state_smem[tid * D + ki] * q_smem[ki];
        }
        out_smem[tid] = scale * sum;
    }
    __syncthreads();
    
    for (int i = tid; i < BLOCK_V; i += num_threads) {
        Out[b * stride_o_b + h * stride_o_h + v0 + i] = __float2bfloat16(out_smem[i]);
    }
    
    float* new_state_ptr = NewState + b * stride_ns_b + h * stride_ns_h + v0 * stride_ns_v;
    
    for (int i = tid; i < BLOCK_V * D; i += num_threads) {
        int vi = i / D;
        int ki = i % D;
        new_state_ptr[vi * stride_ns_v + ki] = state_smem[vi * D + ki];
    }
}

void gdn_decode_v5_launch(
    const void* Q, const void* K, const void* V, const void* State,
    const void* A_log, const void* A, const void* DtBias, const void* B_gate,
    void* Out, void* NewState,
    float scale,
    int B, int num_v_heads, int D,
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
    int V_BLOCKS = D / BLOCK_V;
    dim3 grid(B, num_v_heads, V_BLOCKS);
    dim3 block(128);
    
    size_t smem_size = (4 + D + D + BLOCK_V + BLOCK_V * D) * sizeof(float);
    smem_size += 128 * sizeof(float);
    
    if (BLOCK_V == 16) {
        gdn_decode_kernel_v5<16><<<grid, block, smem_size, stream>>>(
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
        gdn_decode_kernel_v5<32><<<grid, block, smem_size, stream>>>(
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
    } else {
        gdn_decode_kernel_v5<64><<<grid, block, smem_size, stream>>>(
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

}  // namespace gdn
'''

_cuda_module = None
_use_triton_fallback = False


def _ensure_cuda_home():
    """Point PyTorch's JIT extension loader at a real CUDA toolkit when available."""
    if os.environ.get("CUDA_HOME"):
        return

    for candidate in ("/usr/local/cuda", "/usr/local/cuda-12.8"):
        if os.path.exists(candidate):
            os.environ["CUDA_HOME"] = candidate
            return


def _read_v10_header_source() -> str:
    bundled = Path(__file__).with_name("gdn_decode_v10.cuh")
    if bundled.exists():
        return bundled.read_text()

    repo_copy = Path(__file__).resolve().parents[3] / "kernels" / "cute_cpp" / "gdn_decode_v10.cuh"
    if repo_copy.exists():
        return repo_copy.read_text()

    raise FileNotFoundError("Could not locate gdn_decode_v10.cuh for decode CUDA packaging.")


def _try_load_cuda_module():
    """Attempt to load v10 CuTe/TMA CUDA kernel via JIT compilation."""
    global _cuda_module, _use_triton_fallback
    
    if _cuda_module is not None:
        return _cuda_module
    
    if _use_triton_fallback:
        return None
    
    try:
        _ensure_cuda_home()
        from torch.utils.cpp_extension import load_inline

        v10_header = _read_v10_header_source()

        wrapper = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void gdn_decode_v10_cute_wrapper(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor State,
    torch::Tensor A_log, torch::Tensor A, torch::Tensor DtBias, torch::Tensor B_gate,
    torch::Tensor Out, torch::Tensor NewState,
    float scale, int BLOCK_V
) {
    int B = Q.size(0);
    int num_v_heads = V.size(1);

    gdn::gdn_decode_v10_launch_cute(
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), State.data_ptr(),
        A_log.data_ptr(), A.data_ptr(), DtBias.data_ptr(), B_gate.data_ptr(),
        Out.data_ptr(), NewState.data_ptr(),
        scale, B, num_v_heads, 128,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        State.stride(0), State.stride(1), State.stride(2),
        A.stride(0), B_gate.stride(0),
        Out.stride(0), Out.stride(1),
        NewState.stride(0), NewState.stride(1), NewState.stride(2),
        BLOCK_V,
        at::cuda::getCurrentCUDAStream()
    );
}

void gdn_decode_v10_tma_wrapper(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor State,
    torch::Tensor A_log, torch::Tensor A, torch::Tensor DtBias, torch::Tensor B_gate,
    torch::Tensor Out, torch::Tensor NewState,
    float scale, int BLOCK_V
) {
    int B = Q.size(0);
    int num_v_heads = V.size(1);

    gdn::gdn_decode_v10_launch_tma(
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), State.data_ptr(),
        A_log.data_ptr(), A.data_ptr(), DtBias.data_ptr(), B_gate.data_ptr(),
        Out.data_ptr(), NewState.data_ptr(),
        scale, B, num_v_heads, 128,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        State.stride(0), State.stride(1), State.stride(2),
        A.stride(0), B_gate.stride(0),
        Out.stride(0), Out.stride(1),
        NewState.stride(0), NewState.stride(1), NewState.stride(2),
        BLOCK_V,
        at::cuda::getCurrentCUDAStream()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_decode_v10_cute", &gdn_decode_v10_cute_wrapper, "GDN Decode v10 CuTe CUDA kernel");
    m.def("gdn_decode_v10_tma", &gdn_decode_v10_tma_wrapper, "GDN Decode v10 TMA CUDA kernel");
}
'''

        _cuda_module = load_inline(
            name='gdn_decode_v10_cuda',
            cpp_sources='',
            cuda_sources=v10_header + "\n" + wrapper,
            functions=['gdn_decode_v10_cute', 'gdn_decode_v10_tma'],
            extra_cuda_cflags=['-O3', '--use_fast_math', '-std=c++17', '-I/opt/cutlass/include'],
            verbose=False,
        )
        return _cuda_module
        
    except Exception as e:
        print(f"[CUDA JIT] Failed to compile CUDA kernel: {e}")
        print("[CUDA JIT] Falling back to Triton implementation")
        _use_triton_fallback = True
        return None


# ============================================================
# TRITON FALLBACK
# ============================================================

import triton
import triton.language as tl


@triton.jit
def _decode_kernel_triton(
    Q, K, V, State,
    A_log, A, DtBias, B_gate,
    Out, NewState,
    scale,
    stride_q_b, stride_q_h,
    stride_k_b, stride_k_h,
    stride_v_b, stride_v_h,
    stride_s_b, stride_s_h, stride_s_v,
    stride_a_b, stride_b_b,
    stride_o_b, stride_o_h,
    stride_ns_b, stride_ns_h, stride_ns_v,
    D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    b    = tl.program_id(0)
    h    = tl.program_id(1)
    vb   = tl.program_id(2)
    v0   = vb * BLOCK_V
    qk_h = h // 2

    a_val  = tl.load(A     + b * stride_a_b + h).to(tl.float32)
    dt_val = tl.load(DtBias + h)
    alog   = tl.load(A_log  + h)
    b_val  = tl.load(B_gate + b * stride_b_b + h).to(tl.float32)

    x  = a_val + dt_val
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g    = tl.exp(-tl.exp(alog) * sp)
    beta = tl.sigmoid(b_val)

    d  = tl.arange(0, D)
    vd = tl.arange(0, BLOCK_V)

    q = tl.load(Q + b * stride_q_b + qk_h * stride_q_h + d).to(tl.float32)
    k = tl.load(K + b * stride_k_b + qk_h * stride_k_h + d).to(tl.float32)
    v = tl.load(V + b * stride_v_b + h * stride_v_h + v0 + vd).to(tl.float32)

    vi = tl.arange(0, BLOCK_V)[:, None]
    ki = tl.arange(0, D)[None, :]
    s_ptr = State + b * stride_s_b + h * stride_s_h + v0 * stride_s_v
    S = tl.load(s_ptr + vi * stride_s_v + ki)

    S     = g * S
    old_v = tl.sum(S * k[None, :], axis=1)
    delta = beta * (v - old_v)
    S     = S + delta[:, None] * k[None, :]
    out   = scale * tl.sum(S * q[None, :], axis=1)

    tl.store(Out + b * stride_o_b + h * stride_o_h + v0 + vd, out.to(tl.bfloat16))
    ns_ptr = NewState + b * stride_ns_b + h * stride_ns_h + v0 * stride_ns_v
    tl.store(ns_ptr + vi * stride_ns_v + ki, S)


def _triton_kernel(q, k, v, state, A_log, a, dt_bias, b, scale, BLOCK_V):
    """Triton fallback implementation."""
    B, _, num_q_heads, D = q.shape
    num_v_heads = v.shape[2]
    device = q.device
    V_BLOCKS = D // BLOCK_V

    q_c = q.squeeze(1).contiguous()
    k_c = k.squeeze(1).contiguous()
    v_c = v.squeeze(1).contiguous()
    a_c = a.squeeze(1).contiguous()
    b_c = b.squeeze(1).contiguous()

    if state is not None:
        S = state.contiguous()
    else:
        S = torch.zeros(B, num_v_heads, D, D, dtype=torch.float32, device=device)

    out   = torch.empty(B, num_v_heads, D, dtype=torch.bfloat16, device=device)
    new_S = torch.empty_like(S)

    _decode_kernel_triton[(B, num_v_heads, V_BLOCKS)](
        q_c, k_c, v_c, S,
        A_log, a_c, dt_bias, b_c,
        out, new_S,
        float(scale),
        q_c.stride(0), q_c.stride(1),
        k_c.stride(0), k_c.stride(1),
        v_c.stride(0), v_c.stride(1),
        S.stride(0), S.stride(1), S.stride(2),
        a_c.stride(0), b_c.stride(0),
        out.stride(0), out.stride(1),
        new_S.stride(0), new_S.stride(1), new_S.stride(2),
        D=128, BLOCK_V=BLOCK_V, num_warps=4,
    )
    return out.unsqueeze(1), new_S


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def kernel(q, k, v, state, A_log, a, dt_bias, b, scale):
    """
    GDN Decode CUDA packaged kernel.
    
    Attempts v10 CuTe/TMA CUDA JIT compilation, falls back to Triton if needed.
    """
    B, _, num_q_heads, D = q.shape
    num_v_heads = v.shape[2]
    device = q.device

    # Adaptive BLOCK_V
    if B <= 16:
        BLOCK_V = 16
    elif B <= 128:
        BLOCK_V = 32
    else:
        BLOCK_V = 64

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)

    # Try CUDA first
    cuda_mod = _try_load_cuda_module()
    
    if cuda_mod is not None:
        # Use CUDA kernel
        q_c = q.squeeze(1).contiguous()
        k_c = k.squeeze(1).contiguous()
        v_c = v.squeeze(1).contiguous()
        a_c = a.squeeze(1).contiguous()
        b_c = b.squeeze(1).contiguous()
        
        if dt_bias.dtype == torch.bfloat16:
            dt_bias = dt_bias.float()
        
        if state is not None:
            S = state.contiguous()
        else:
            S = torch.zeros(B, num_v_heads, D, D, dtype=torch.float32, device=device)
        
        out = torch.empty(B, num_v_heads, D, dtype=torch.bfloat16, device=device)
        new_S = torch.empty_like(S)

        backend = (_PACKAGED_BACKEND or "cute").lower()
        if backend == "tma":
            cuda_mod.gdn_decode_v10_tma(q_c, k_c, v_c, S, A_log, a_c, dt_bias, b_c, out, new_S, float(scale), BLOCK_V)
        else:
            cuda_mod.gdn_decode_v10_cute(q_c, k_c, v_c, S, A_log, a_c, dt_bias, b_c, out, new_S, float(scale), BLOCK_V)
        return out.unsqueeze(1), new_S
    
    else:
        # Triton fallback
        return _triton_kernel(q, k, v, state, A_log, a, dt_bias, b, scale, BLOCK_V)

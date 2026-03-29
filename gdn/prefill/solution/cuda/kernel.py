"""
GDN Prefill v5 — Python wrapper for CUDA kernel

CUDA source: src/kernels/gdn_prefill_v5.cuh
Fallback: Triton (if JIT compilation fails in sandbox)
"""

import math
import os
from pathlib import Path

import torch

# ============================================================
# CUDA KERNEL LOADING
# ============================================================

CUDA_SOURCE_PATH = Path(__file__).parent.parent.parent.parent / "src" / "kernels" / "gdn_prefill_v5.cuh"

_cuda_module = None
_use_triton_fallback = False


def _try_load_cuda_module():
    """Attempt to load CUDA kernel via JIT compilation."""
    global _cuda_module, _use_triton_fallback
    
    if _cuda_module is not None:
        return _cuda_module
    
    if _use_triton_fallback:
        return None
    
    try:
        from torch.utils.cpp_extension import load_inline
        
        cuda_source = CUDA_SOURCE_PATH.read_text()
        
        wrapper = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void gdn_prefill_v5_wrapper(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor State,
    torch::Tensor A_log, torch::Tensor A, torch::Tensor DtBias, torch::Tensor B_gate,
    torch::Tensor CuSeqlens,
    torch::Tensor Out, torch::Tensor NewState,
    float scale, int BLOCK_V
) {
    int N = CuSeqlens.size(0) - 1;
    int num_v_heads = V.size(1);
    
    gdn::gdn_prefill_v5_launch(
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), State.data_ptr(),
        A_log.data_ptr(), A.data_ptr(), DtBias.data_ptr(), B_gate.data_ptr(),
        CuSeqlens.data_ptr(),
        Out.data_ptr(), NewState.data_ptr(),
        scale, N, num_v_heads, 128,
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
    m.def("gdn_prefill_v5", &gdn_prefill_v5_wrapper, "GDN Prefill v5 CUDA kernel");
}
'''
        
        _cuda_module = load_inline(
            name='gdn_prefill_v5_cuda',
            cpp_sources='',
            cuda_sources=cuda_source + wrapper,
            functions=['gdn_prefill_v5'],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
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
def _prefill_kernel_triton(
    Q_ptr, K_ptr, V_ptr, State_ptr,
    A_log_ptr, A_ptr, DtBias_ptr, B_ptr,
    CuSeq_ptr, Out_ptr, NewState_ptr,
    scale,
    stride_q_t, stride_q_h,
    stride_k_t, stride_k_h,
    stride_v_t, stride_v_h,
    stride_s_n, stride_s_h, stride_s_v,
    stride_a_t, stride_b_t,
    stride_o_t, stride_o_h,
    stride_ns_n, stride_ns_h, stride_ns_v,
    D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    n    = tl.program_id(0)
    h    = tl.program_id(1)
    vb   = tl.program_id(2)
    v0   = vb * BLOCK_V
    qk_h = h // 2

    t_start = tl.load(CuSeq_ptr + n).to(tl.int32)
    t_end   = tl.load(CuSeq_ptr + n + 1).to(tl.int32)
    seq_len = t_end - t_start

    alog   = tl.load(A_log_ptr + h)
    dt_val = tl.load(DtBias_ptr + h)

    vi = tl.arange(0, BLOCK_V)[:, None]
    ki = tl.arange(0, D)[None, :]
    s_ptr = State_ptr + n * stride_s_n + h * stride_s_h + v0 * stride_s_v
    S = tl.load(s_ptr + vi * stride_s_v + ki)

    di = tl.arange(0, D)
    vd = tl.arange(0, BLOCK_V)

    for i in range(seq_len):
        t = t_start + i

        a_val = tl.load(A_ptr + t * stride_a_t + h).to(tl.float32)
        b_val = tl.load(B_ptr + t * stride_b_t + h).to(tl.float32)

        x  = a_val + dt_val
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        g    = tl.exp(-tl.exp(alog) * sp)
        beta = tl.sigmoid(b_val)

        kv = tl.load(K_ptr + t * stride_k_t + qk_h * stride_k_h + di).to(tl.float32)
        vv = tl.load(V_ptr + t * stride_v_t + h * stride_v_h + v0 + vd).to(tl.float32)
        qv = tl.load(Q_ptr + t * stride_q_t + qk_h * stride_q_h + di).to(tl.float32)

        S     = g * S
        old_v = tl.sum(S * kv[None, :], axis=1)
        delta = beta * (vv - old_v)
        S     = S + delta[:, None] * kv[None, :]

        ov = scale * tl.sum(S * qv[None, :], axis=1)
        tl.store(Out_ptr + t * stride_o_t + h * stride_o_h + v0 + vd, ov.to(tl.bfloat16))

    ns_ptr = NewState_ptr + n * stride_ns_n + h * stride_ns_h + v0 * stride_ns_v
    tl.store(ns_ptr + vi * stride_ns_v + ki, S)


def _triton_kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, BLOCK_V):
    """Triton fallback implementation."""
    T, num_q_heads, D = q.shape
    num_v_heads = v.shape[1]
    N = cu_seqlens.shape[0] - 1
    device = q.device
    V_BLOCKS = D // BLOCK_V

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    a_c = a.contiguous()
    b_c = b.contiguous()
    cu = cu_seqlens.contiguous()

    if state is not None:
        S = state.contiguous()
    else:
        S = torch.zeros(N, num_v_heads, D, D, dtype=torch.float32, device=device)

    out   = torch.empty(T, num_v_heads, D, dtype=torch.bfloat16, device=device)
    new_S = torch.empty_like(S)

    _prefill_kernel_triton[(N, num_v_heads, V_BLOCKS)](
        q_c, k_c, v_c, S,
        A_log, a_c, dt_bias, b_c,
        cu, out, new_S,
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
    return out, new_S


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    GDN Prefill v5 kernel.
    
    Attempts CUDA JIT compilation, falls back to Triton if sandbox blocks it.
    """
    T, num_q_heads, D = q.shape
    num_v_heads = v.shape[1]
    N = cu_seqlens.shape[0] - 1
    device = q.device

    # Adaptive BLOCK_V
    if N <= 4:
        BLOCK_V = 16
    else:
        BLOCK_V = 32

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)

    # Try CUDA first
    cuda_mod = _try_load_cuda_module()
    
    if cuda_mod is not None:
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        a_c = a.contiguous()
        b_c = b.contiguous()
        cu = cu_seqlens.contiguous()
        
        if dt_bias.dtype == torch.bfloat16:
            dt_bias = dt_bias.float()
        
        if state is not None:
            S = state.contiguous()
        else:
            S = torch.zeros(N, num_v_heads, D, D, dtype=torch.float32, device=device)
        
        out = torch.empty(T, num_v_heads, D, dtype=torch.bfloat16, device=device)
        new_S = torch.empty_like(S)
        
        cuda_mod.gdn_prefill_v5(q_c, k_c, v_c, S, A_log, a_c, dt_bias, b_c, cu, out, new_S, float(scale), BLOCK_V)
        return out, new_S
    
    else:
        return _triton_kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, BLOCK_V)

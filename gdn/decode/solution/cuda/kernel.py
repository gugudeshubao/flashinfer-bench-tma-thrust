"""
GDN Decode v5 — Python wrapper for CUDA kernel

CUDA source: src/kernels/gdn_decode_v5.cuh
Fallback: Triton (if JIT compilation fails in sandbox)
"""

import math
import os
from pathlib import Path

import torch

# ============================================================
# CUDA KERNEL LOADING
# ============================================================

# Path to CUDA source file
CUDA_SOURCE_PATH = Path(__file__).parent.parent.parent.parent / "src" / "kernels" / "gdn_decode_v5.cuh"

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
        
        # Read CUDA source
        cuda_source = CUDA_SOURCE_PATH.read_text()
        
        # Wrapper code to expose Python binding
        wrapper = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void gdn_decode_v5_wrapper(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor State,
    torch::Tensor A_log, torch::Tensor A, torch::Tensor DtBias, torch::Tensor B_gate,
    torch::Tensor Out, torch::Tensor NewState,
    float scale, int BLOCK_V
) {
    int B = Q.size(0);
    int num_v_heads = V.size(1);
    
    gdn::gdn_decode_v5_launch(
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
    m.def("gdn_decode_v5", &gdn_decode_v5_wrapper, "GDN Decode v5 CUDA kernel");
}
'''
        
        _cuda_module = load_inline(
            name='gdn_decode_v5_cuda',
            cpp_sources='',
            cuda_sources=cuda_source + wrapper,
            functions=['gdn_decode_v5'],
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
    GDN Decode v5 kernel.
    
    Attempts CUDA JIT compilation, falls back to Triton if sandbox blocks it.
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
        
        cuda_mod.gdn_decode_v5(q_c, k_c, v_c, S, A_log, a_c, dt_bias, b_c, out, new_S, float(scale), BLOCK_V)
        return out.unsqueeze(1), new_S
    
    else:
        # Triton fallback
        return _triton_kernel(q, k, v, state, A_log, a, dt_bias, b, scale, BLOCK_V)

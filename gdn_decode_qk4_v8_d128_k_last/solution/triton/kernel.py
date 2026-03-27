"""
GDN Decode — v2 Triton kernel
gdn_decode_qk4_v8_d128_k_last

Grid: (B, H=8)  — one Triton program per (batch, v_head).
Each program loads the full 128×128 state tile, applies the GDN delta-rule
update in registers, writes state back, and emits the output vector.
No Python-level loops; all per-head compute is fused in one kernel launch.

GVA: num_q_heads=4, num_v_heads=8  →  qk_head = v_head // 2
State layout: k-last  [B, H, V=128, K=128]  float32
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_kernel(
    Q_ptr, K_ptr, V_ptr, State_ptr,
    A_log_ptr, A_ptr, DtBias_ptr, B_ptr,
    Out_ptr, NewState_ptr,
    scale,
    # tensor strides (post-squeeze, contiguous layout assumed)
    stride_q_b, stride_q_h,        # Q  [B, 4, D]
    stride_k_b, stride_k_h,        # K  [B, 4, D]
    stride_v_b, stride_v_h,        # V  [B, 8, D]
    stride_s_b, stride_s_h, stride_s_v,  # State [B,8,D,D] k-last
    stride_a_b,                    # a  [B, 8]
    stride_b_b,                    # b  [B, 8]
    stride_o_b, stride_o_h,        # Out [B, 8, D]
    stride_ns_b, stride_ns_h, stride_ns_v,
    D: tl.constexpr,               # head_size = 128
):
    b    = tl.program_id(0)
    h    = tl.program_id(1)
    qk_h = h // 2                  # GVA expansion

    # ── gates (scalar per program) ──────────────────────────────────────────
    a_val  = tl.load(A_ptr     + b * stride_a_b + h).to(tl.float32)
    dt_val = tl.load(DtBias_ptr + h)
    alog   = tl.load(A_log_ptr  + h)
    b_val  = tl.load(B_ptr     + b * stride_b_b + h).to(tl.float32)

    x = a_val + dt_val
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))   # softplus, stable
    g    = tl.exp(-tl.exp(alog) * sp)
    beta = tl.sigmoid(b_val)

    # ── q / k / v vectors  [D] ──────────────────────────────────────────────
    d = tl.arange(0, D)
    q = tl.load(Q_ptr + b * stride_q_b + qk_h * stride_q_h + d).to(tl.float32)
    k = tl.load(K_ptr + b * stride_k_b + qk_h * stride_k_h + d).to(tl.float32)
    v = tl.load(V_ptr + b * stride_v_b + h    * stride_v_h + d).to(tl.float32)

    # ── state  [D, D]  (V rows = dim-0, K cols = dim-1) ────────────────────
    v_idx = tl.arange(0, D)[:, None]   # [D, 1]
    k_idx = tl.arange(0, D)[None, :]   # [1, D]
    s_ptr = State_ptr + b * stride_s_b + h * stride_s_h
    S = tl.load(s_ptr + v_idx * stride_s_v + k_idx)  # [D, D] f32

    # ── GDN delta-rule update ───────────────────────────────────────────────
    S       = g * S                                  # decay
    old_v   = tl.sum(S * k[None, :], axis=1)        # [D]  S @ k
    delta   = beta * (v - old_v)                    # [D]
    S       = S + delta[:, None] * k[None, :]       # rank-1 update
    out     = scale * tl.sum(S * q[None, :], axis=1)  # [D]  scale * S @ q

    # ── store outputs ────────────────────────────────────────────────────────
    tl.store(Out_ptr + b * stride_o_b + h * stride_o_h + d,
             out.to(tl.bfloat16))
    ns_ptr = NewState_ptr + b * stride_ns_b + h * stride_ns_h
    tl.store(ns_ptr + v_idx * stride_ns_v + k_idx, S)


def kernel(q, k, v, state, A_log, a, dt_bias, b, scale):
    B, _, num_q_heads, D = q.shape
    num_v_heads = v.shape[2]
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)

    # squeeze T=1, ensure contiguous
    q_c = q.squeeze(1).contiguous()   # [B, 4, D]
    k_c = k.squeeze(1).contiguous()   # [B, 4, D]
    v_c = v.squeeze(1).contiguous()   # [B, 8, D]
    a_c = a.squeeze(1).contiguous()   # [B, 8]
    b_c = b.squeeze(1).contiguous()   # [B, 8]

    if state is not None:
        S = state.contiguous()        # [B, 8, D, D] k-last
    else:
        S = torch.zeros(B, num_v_heads, D, D, dtype=torch.float32, device=device)

    out    = torch.empty(B, num_v_heads, D, dtype=torch.bfloat16, device=device)
    new_S  = torch.empty_like(S)

    _decode_kernel[(B, num_v_heads)](
        q_c, k_c, v_c, S,
        A_log, a_c, dt_bias, b_c,
        out, new_S,
        float(scale),
        q_c.stride(0), q_c.stride(1),
        k_c.stride(0), k_c.stride(1),
        v_c.stride(0), v_c.stride(1),
        S.stride(0), S.stride(1), S.stride(2),
        a_c.stride(0),
        b_c.stride(0),
        out.stride(0), out.stride(1),
        new_S.stride(0), new_S.stride(1), new_S.stride(2),
        D=128,
        num_warps=4,
    )

    return out.unsqueeze(1), new_S   # [B,1,8,D], [B,8,D,D]

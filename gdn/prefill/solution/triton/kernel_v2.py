"""
GDN Prefill — v2 Triton kernel
gdn_prefill_qk4_v8_d128_k_last

Grid: (N=num_seqs, H=8) — one program per (seq, v_head).

v2: Triton kernel with fused delta-rule, full state tile in registers.
Each program handles the full state [K=128, V=128] for one (sequence, head).
Sequential token scan within each program.
  - Eliminates Python loop overhead
  - State in registers during sequence processing
  - Single HBM state read/write per sequence

GVA: num_q_heads=4, num_v_heads=8  →  qk_head = v_head // 2
State layout: k-last  [N, H, V=128, K=128]  float32
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_kernel_v2(
    Q_ptr, K_ptr, V_ptr, State_ptr,
    A_log_ptr, A_ptr, DtBias_ptr, B_ptr,
    CuSeq_ptr, Out_ptr, NewState_ptr,
    scale,
    stride_q_t, stride_q_h,          # Q   [T, 4, D]
    stride_k_t, stride_k_h,          # K   [T, 4, D]
    stride_v_t, stride_v_h,          # V   [T, 8, D]
    stride_s_n, stride_s_h, stride_s_v,  # State [N,8,D,D] k-last
    stride_a_t,                       # a   [T, 8]
    stride_b_t,                       # b   [T, 8]
    stride_o_t, stride_o_h,           # Out [T, 8, D]
    stride_ns_n, stride_ns_h, stride_ns_v,
    D: tl.constexpr,                  # head_size = 128
):
    n    = tl.program_id(0)   # sequence index
    h    = tl.program_id(1)   # v_head index
    qk_h = h // 2             # GVA: 2 v-heads per qk-head

    # ── sequence bounds ──────────────────────────────────────────────────────
    t_start = tl.load(CuSeq_ptr + n    ).to(tl.int32)
    t_end   = tl.load(CuSeq_ptr + n + 1).to(tl.int32)
    seq_len = t_end - t_start

    # ── head-level constants ─────────────────────────────────────────────────
    alog   = tl.load(A_log_ptr   + h)
    dt_val = tl.load(DtBias_ptr  + h)

    # ── load initial state [D, D] ────────────────────────────────────────────
    vi = tl.arange(0, D)[:, None]   # [D, 1]
    ki = tl.arange(0, D)[None, :]   # [1, D]
    s_ptr = State_ptr + n * stride_s_n + h * stride_s_h
    S = tl.load(s_ptr + vi * stride_s_v + ki)   # [D, D] f32

    di = tl.arange(0, D)            # reused for 1-D vector loads

    # ── sequential token scan ────────────────────────────────────────────────
    for i in range(seq_len):
        t = t_start + i

        # per-token gate scalars
        a_val = tl.load(A_ptr + t * stride_a_t + h).to(tl.float32)
        b_val = tl.load(B_ptr + t * stride_b_t + h).to(tl.float32)

        x  = a_val + dt_val
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))  # softplus
        g    = tl.exp(-tl.exp(alog) * sp)
        beta = tl.sigmoid(b_val)

        # k [D], v [D], q [D]
        kv = tl.load(K_ptr + t * stride_k_t + qk_h * stride_k_h + di).to(tl.float32)
        vv = tl.load(V_ptr + t * stride_v_t + h    * stride_v_h + di).to(tl.float32)
        qv = tl.load(Q_ptr + t * stride_q_t + qk_h * stride_q_h + di).to(tl.float32)

        # GDN delta-rule
        S     = g * S
        old_v = tl.sum(S * kv[None, :], axis=1)        # [D]  S @ k
        delta = beta * (vv - old_v)                     # [D]
        S     = S + delta[:, None] * kv[None, :]        # rank-1 update

        # output  o = scale * S @ q
        ov = scale * tl.sum(S * qv[None, :], axis=1)   # [D]
        tl.store(Out_ptr + t * stride_o_t + h * stride_o_h + di,
                 ov.to(tl.bfloat16))

    # ── store final state ─────────────────────────────────────────────────────
    ns_ptr = NewState_ptr + n * stride_ns_n + h * stride_ns_h
    tl.store(ns_ptr + vi * stride_ns_v + ki, S)


def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    T, num_q_heads, D = q.shape
    num_v_heads = v.shape[1]
    N = cu_seqlens.shape[0] - 1
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)

    q_c  = q.contiguous()    # [T, 4, D]
    k_c  = k.contiguous()    # [T, 4, D]
    v_c  = v.contiguous()    # [T, 8, D]
    a_c  = a.contiguous()    # [T, 8]
    b_c  = b.contiguous()    # [T, 8]
    cu   = cu_seqlens.contiguous()

    if state is not None:
        S = state.contiguous()   # [N, 8, D, D] k-last
    else:
        S = torch.zeros(N, num_v_heads, D, D, dtype=torch.float32, device=device)

    out   = torch.empty(T, num_v_heads, D, dtype=torch.bfloat16, device=device)
    new_S = torch.empty_like(S)

    _prefill_kernel_v2[(N, num_v_heads)](
        q_c, k_c, v_c, S,
        A_log, a_c, dt_bias, b_c,
        cu, out, new_S,
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

    return out, new_S

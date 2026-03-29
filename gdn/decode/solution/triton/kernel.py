"""
GDN Decode — v4 Triton kernel
gdn_decode_qk4_v8_d128_k_last

Grid: (B, H=8, V_BLOCKS)  — one program per (batch, v_head, V-tile).

v4: Adaptive BLOCK_V based on batch size for optimal SM occupancy:
  - Small batch (B <= 16): BLOCK_V=16, more programs, better occupancy
  - Medium batch (B <= 128): BLOCK_V=32, balanced
  - Large batch (B > 128): BLOCK_V=64, reduced launch overhead

GVA: num_q_heads=4, num_v_heads=8  →  qk_head = v_head // 2
State layout: k-last  [B, H, V=128, K=128]  float32
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_kernel(
    Q, K, V, State,
    A_log, A, DtBias, B_gate,
    Out, NewState,
    scale,
    stride_q_b, stride_q_h,          # Q  [B, 4, K]
    stride_k_b, stride_k_h,          # K  [B, 4, K]
    stride_v_b, stride_v_h,          # V  [B, 8, D]
    stride_s_b, stride_s_h, stride_s_v,  # State [B,8,D,D] k-last
    stride_a_b,                       # a  [B, 8]
    stride_b_b,                       # b  [B, 8]
    stride_o_b, stride_o_h,           # Out [B, 8, D]
    stride_ns_b, stride_ns_h, stride_ns_v,
    D: tl.constexpr,                  # head_size = 128
    BLOCK_V: tl.constexpr,            # V-tile size (adaptive)
):
    b    = tl.program_id(0)
    h    = tl.program_id(1)
    vb   = tl.program_id(2)           # V-block index in [0, D//BLOCK_V)
    v0   = vb * BLOCK_V               # first V element this program owns
    qk_h = h // 2                     # GVA: 2 v-heads share each qk-head

    # ── gates (scalar per program) ──────────────────────────────────────────
    a_val  = tl.load(A     + b * stride_a_b + h).to(tl.float32)
    dt_val = tl.load(DtBias + h)
    alog   = tl.load(A_log  + h)
    b_val  = tl.load(B_gate + b * stride_b_b + h).to(tl.float32)

    x  = a_val + dt_val
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g    = tl.exp(-tl.exp(alog) * sp)
    beta = tl.sigmoid(b_val)

    # ── q / k [D],  v V-slice [BLOCK_V] ────────────────────────────────────
    d  = tl.arange(0, D)
    vd = tl.arange(0, BLOCK_V)

    q = tl.load(Q + b * stride_q_b + qk_h * stride_q_h + d).to(tl.float32)
    k = tl.load(K + b * stride_k_b + qk_h * stride_k_h + d).to(tl.float32)
    v = tl.load(V + b * stride_v_b  + h    * stride_v_h + v0 + vd).to(tl.float32)

    # ── state V-slice  [BLOCK_V, D] ─────────────────────────────────────────
    vi = tl.arange(0, BLOCK_V)[:, None]
    ki = tl.arange(0, D)[None, :]
    s_ptr = State + b * stride_s_b + h * stride_s_h + v0 * stride_s_v
    S = tl.load(s_ptr + vi * stride_s_v + ki)   # [BLOCK_V, D] f32

    # ── GDN delta-rule ───────────────────────────────────────────────────────
    S     = g * S
    old_v = tl.sum(S * k[None, :], axis=1)      # [BLOCK_V]  S @ k
    delta = beta * (v - old_v)                   # [BLOCK_V]
    S     = S + delta[:, None] * k[None, :]      # rank-1 update
    out   = scale * tl.sum(S * q[None, :], axis=1)  # [BLOCK_V]  scale*S@q

    # ── store ────────────────────────────────────────────────────────────────
    tl.store(Out    + b * stride_o_b  + h * stride_o_h  + v0 + vd,
             out.to(tl.bfloat16))
    ns_ptr = NewState + b * stride_ns_b + h * stride_ns_h + v0 * stride_ns_v
    tl.store(ns_ptr + vi * stride_ns_v + ki, S)


def kernel(q, k, v, state, A_log, a, dt_bias, b, scale):
    B, _, num_q_heads, D = q.shape
    num_v_heads = v.shape[2]
    device = q.device

    # Adaptive BLOCK_V based on batch size
    if B <= 16:
        BLOCK_V = 16   # More parallelism for small batch
    elif B <= 128:
        BLOCK_V = 32   # Balanced
    else:
        BLOCK_V = 64   # Reduced overhead for large batch

    V_BLOCKS = D // BLOCK_V

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)

    q_c = q.squeeze(1).contiguous()   # [B, 4, D]
    k_c = k.squeeze(1).contiguous()
    v_c = v.squeeze(1).contiguous()   # [B, 8, D]
    a_c = a.squeeze(1).contiguous()   # [B, 8]
    b_c = b.squeeze(1).contiguous()

    if state is not None:
        S = state.contiguous()
    else:
        S = torch.zeros(B, num_v_heads, D, D, dtype=torch.float32, device=device)

    out   = torch.empty(B, num_v_heads, D, dtype=torch.bfloat16, device=device)
    new_S = torch.empty_like(S)

    _decode_kernel[(B, num_v_heads, V_BLOCKS)](
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
        BLOCK_V=BLOCK_V,
        num_warps=4,
    )

    return out.unsqueeze(1), new_S   # [B,1,8,D], [B,8,D,D]

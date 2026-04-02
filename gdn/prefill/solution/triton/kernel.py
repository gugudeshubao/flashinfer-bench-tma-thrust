"""
GDN Prefill — v5 Triton kernel with Software Pipelining
gdn_prefill_qk4_v8_d128_k_last

Grid: (N=num_seqs, H=8, V_BLOCKS) — one program per (seq, v_head, V-tile).

v5: Double-buffering with software pipelining:
  - Prefetch next token's Q/K/V while computing current token
  - Overlaps memory latency with computation
  - Adaptive BLOCK_V based on num_seqs

GVA: num_q_heads=4, num_v_heads=8  →  qk_head = v_head // 2
State layout: k-last  [N, H, V=128, K=128]  float32
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_disable_v5 = False
_disable_triton = False
_logged_v5_error = False
_logged_triton_error = False


def _matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.float() @ b.float()


def _torch_fallback_kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """Reference-style fallback used when Triton compilation fails."""
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_k_heads = k.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    x = a.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b.float())

    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    output = torch.zeros(
        (total_seq_len, num_sab_heads, head_size), dtype=torch.bfloat16, device=device
    )
    new_state = torch.zeros(
        (num_seqs, num_sab_heads, head_size, head_size), dtype=torch.float32, device=device
    )

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = seq_end - seq_start

        if seq_len <= 0:
            if state is not None:
                new_state[seq_idx] = state[seq_idx].float()
            continue

        if state is not None:
            state_HKV = state[seq_idx].clone().float().transpose(-1, -2)
        else:
            state_HKV = torch.zeros(
                (num_sab_heads, head_size, head_size), dtype=torch.float32, device=device
            )

        for i in range(seq_len):
            t = seq_start + i
            q_H1K = q_exp[t].unsqueeze(1).float()
            k_H1K = k_exp[t].unsqueeze(1).float()
            v_H1V = v[t].unsqueeze(1).float()
            g_H11 = g[t].unsqueeze(1).unsqueeze(2)
            beta_H11 = beta[t].unsqueeze(1).unsqueeze(2)

            old_state_HKV = g_H11 * state_HKV
            old_v_H1V = _matmul(k_H1K, old_state_HKV)
            new_v_H1V = beta_H11 * v_H1V + (1 - beta_H11) * old_v_H1V
            state_remove = torch.einsum("hkl,hlv->hkv", k_H1K.transpose(-1, -2), old_v_H1V)
            state_update = torch.einsum("hkl,hlv->hkv", k_H1K.transpose(-1, -2), new_v_H1V)
            state_HKV = old_state_HKV - state_remove + state_update

            o_H1V = scale * _matmul(q_H1K, state_HKV)
            output[t] = o_H1V.squeeze(1).to(torch.bfloat16)

        new_state[seq_idx] = state_HKV.transpose(-1, -2)

    return output, new_state


@triton.jit
def _prefill_kernel_v5(
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
    BLOCK_V: tl.constexpr,            # V-tile size (adaptive)
):
    """
    v5: Software pipelining with double-buffered token loads.
    
    Pipeline stages:
      Stage 0: Load token i+1 data (prefetch)
      Stage 1: Compute token i (uses previously loaded data)
    """
    n    = tl.program_id(0)   # sequence index
    h    = tl.program_id(1)   # v_head index
    vb   = tl.program_id(2)   # V-block index in [0, D//BLOCK_V)
    v0   = vb * BLOCK_V       # first V element this program owns
    qk_h = h // 2             # GVA: 2 v-heads per qk-head

    # ── sequence bounds ──────────────────────────────────────────────────────
    t_start = tl.load(CuSeq_ptr + n    ).to(tl.int32)
    t_end   = tl.load(CuSeq_ptr + n + 1).to(tl.int32)
    seq_len = t_end - t_start

    # ── head-level constants ─────────────────────────────────────────────────
    alog   = tl.load(A_log_ptr   + h)
    dt_val = tl.load(DtBias_ptr  + h)

    # ── load initial state V-slice  [BLOCK_V, D] ─────────────────────────────
    vi = tl.arange(0, BLOCK_V)[:, None]   # [BLOCK_V, 1]
    ki = tl.arange(0, D)[None, :]         # [1, D]
    s_ptr = State_ptr + n * stride_s_n + h * stride_s_h + v0 * stride_s_v
    S = tl.load(s_ptr + vi * stride_s_v + ki)   # [BLOCK_V, D] f32

    di = tl.arange(0, D)            # reused for 1-D vector loads
    vd = tl.arange(0, BLOCK_V)      # for v-slice

    # Handle empty sequences - just copy state and return
    if seq_len <= 0:
        ns_ptr = NewState_ptr + n * stride_ns_n + h * stride_ns_h + v0 * stride_ns_v
        tl.store(ns_ptr + vi * stride_ns_v + ki, S)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # Software Pipelining: Process all tokens with prefetching
    # Strategy: Always prefetch next token, clamp index for last iteration
    # ══════════════════════════════════════════════════════════════════════════
    
    # Prefetch first token
    t_curr = t_start
    a_curr = tl.load(A_ptr + t_curr * stride_a_t + h).to(tl.float32)
    b_curr = tl.load(B_ptr + t_curr * stride_b_t + h).to(tl.float32)
    k_curr = tl.load(K_ptr + t_curr * stride_k_t + qk_h * stride_k_h + di).to(tl.float32)
    v_curr = tl.load(V_ptr + t_curr * stride_v_t + h * stride_v_h + v0 + vd).to(tl.float32)
    q_curr = tl.load(Q_ptr + t_curr * stride_q_t + qk_h * stride_q_h + di).to(tl.float32)

    # Main loop: process tokens 0 to seq_len-1
    for i in range(seq_len):
        t = t_start + i
        
        # ── Stage 0: Prefetch next token (clamp to avoid OOB) ────────────────
        # Use min to clamp t_next to valid range (last token stays clamped)
        t_next = tl.minimum(t + 1, t_end - 1)
        
        # These loads will be for the same token on last iteration (harmless)
        a_next = tl.load(A_ptr + t_next * stride_a_t + h).to(tl.float32)
        b_next = tl.load(B_ptr + t_next * stride_b_t + h).to(tl.float32)
        k_next = tl.load(K_ptr + t_next * stride_k_t + qk_h * stride_k_h + di).to(tl.float32)
        v_next = tl.load(V_ptr + t_next * stride_v_t + h * stride_v_h + v0 + vd).to(tl.float32)
        q_next = tl.load(Q_ptr + t_next * stride_q_t + qk_h * stride_q_h + di).to(tl.float32)
        
        # ── Stage 1: Compute current token ───────────────────────────────────
        # Gate computation
        x = a_curr + dt_val
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))  # softplus
        g = tl.exp(-tl.exp(alog) * sp)
        beta = tl.sigmoid(b_curr)

        # GDN delta-rule on V-slice
        S = g * S
        old_v = tl.sum(S * k_curr[None, :], axis=1)        # [BLOCK_V]
        delta = beta * (v_curr - old_v)                     # [BLOCK_V]
        S = S + delta[:, None] * k_curr[None, :]           # rank-1 update

        # Output: o = scale * S @ q
        ov = scale * tl.sum(S * q_curr[None, :], axis=1)   # [BLOCK_V]
        tl.store(Out_ptr + t * stride_o_t + h * stride_o_h + v0 + vd,
                 ov.to(tl.bfloat16))
        
        # ── Rotate buffers: next → current ───────────────────────────────────
        # (On last iteration, this is a no-op since next == current after clamp)
        a_curr = a_next
        b_curr = b_next
        k_curr = k_next
        v_curr = v_next
        q_curr = q_next

    # ── store final state V-slice ─────────────────────────────────────────────
    ns_ptr = NewState_ptr + n * stride_ns_n + h * stride_ns_h + v0 * stride_ns_v
    tl.store(ns_ptr + vi * stride_ns_v + ki, S)


# Keep original kernel for comparison
@triton.jit
def _prefill_kernel(
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
    D: tl.constexpr, BLOCK_V: tl.constexpr,
):
    """Original v4 kernel (for comparison)"""
    n    = tl.program_id(0)
    h    = tl.program_id(1)
    vb   = tl.program_id(2)
    v0   = vb * BLOCK_V
    qk_h = h // 2

    t_start = tl.load(CuSeq_ptr + n    ).to(tl.int32)
    t_end   = tl.load(CuSeq_ptr + n + 1).to(tl.int32)
    seq_len = t_end - t_start

    alog   = tl.load(A_log_ptr   + h)
    dt_val = tl.load(DtBias_ptr  + h)

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
        vv = tl.load(V_ptr + t * stride_v_t + h    * stride_v_h + v0 + vd).to(tl.float32)
        qv = tl.load(Q_ptr + t * stride_q_t + qk_h * stride_q_h + di).to(tl.float32)
        S     = g * S
        old_v = tl.sum(S * kv[None, :], axis=1)
        delta = beta * (vv - old_v)
        S     = S + delta[:, None] * kv[None, :]
        ov = scale * tl.sum(S * qv[None, :], axis=1)
        tl.store(Out_ptr + t * stride_o_t + h * stride_o_h + v0 + vd, ov.to(tl.bfloat16))

    ns_ptr = NewState_ptr + n * stride_ns_n + h * stride_ns_h + v0 * stride_ns_v
    tl.store(ns_ptr + vi * stride_ns_v + ki, S)


def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    GDN Prefill kernel wrapper.
    """
    T, num_q_heads, D = q.shape
    num_v_heads = v.shape[1]
    N = cu_seqlens.shape[0] - 1
    device = q.device

    # Adaptive BLOCK_V based on num_seqs
    if N <= 4:
        BLOCK_V = 16   # More parallelism for few sequences
    else:
        BLOCK_V = 32   # Balanced register usage

    V_BLOCKS = D // BLOCK_V

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)

    q_c  = q.contiguous()
    k_c  = k.contiguous()
    v_c  = v.contiguous()
    a_c  = a.contiguous()
    b_c  = b.contiguous()
    cu   = cu_seqlens.contiguous()

    if state is not None:
        S = state.contiguous()
    else:
        S = torch.zeros(N, num_v_heads, D, D, dtype=torch.float32, device=device)

    out   = torch.empty(T, num_v_heads, D, dtype=torch.bfloat16, device=device)
    new_S = torch.empty_like(S)

    def _launch(kernel_fn):
        kernel_fn[(N, num_v_heads, V_BLOCKS)](
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
            BLOCK_V=BLOCK_V,
            num_warps=4,
        )
        return out, new_S

    global _disable_v5, _disable_triton, _logged_v5_error, _logged_triton_error

    if not _disable_triton:
        if not _disable_v5:
            try:
                return _launch(_prefill_kernel_v5)
            except Exception as exc:
                _disable_v5 = True
                if not _logged_v5_error:
                    print(f"[prefill Triton] v5 compile failed, falling back to v4: {exc}")
                    _logged_v5_error = True

        try:
            return _launch(_prefill_kernel)
        except Exception as exc:
            _disable_triton = True
            if not _logged_triton_error:
                print(f"[prefill Triton] v4 compile failed, falling back to PyTorch: {exc}")
                _logged_triton_error = True

    return _torch_fallback_kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

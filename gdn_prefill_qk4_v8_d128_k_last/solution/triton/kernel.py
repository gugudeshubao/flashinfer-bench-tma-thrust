"""
GDN Prefill kernel - gdn_prefill_qk4_v8_d128_k_last

Gated Delta Net prefill (chunked) with GVA (Grouped Value Attention):
  num_q_heads=4, num_k_heads=4, num_v_heads=8, head_size=128

Inputs use varlen / ragged batch format with cu_seqlens.
State layout: k-last [N, H, V, K]

Algorithm (per token t in sequence):
  g    = exp(-exp(A_log) * softplus(a[t] + dt_bias))
  beta = sigmoid(b[t])
  S     = g * S
  old_v = k[t] @ S
  new_v = beta * v[t] + (1-beta) * old_v
  S     = S + outer(k[t], new_v - old_v)
  o[t]  = scale * q[t] @ S
"""

import math

import torch
import torch.nn.functional as F


def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    GDN prefill variable-length batched forward pass.

    Parameters
    ----------
    q         : [total_seq_len, 4, 128] bfloat16
    k         : [total_seq_len, 4, 128] bfloat16
    v         : [total_seq_len, 8, 128] bfloat16
    state     : [num_seqs, 8, 128, 128] float32, k-last [N, H, V, K]  (optional)
    A_log     : [8] float32
    a         : [total_seq_len, 8] bfloat16
    dt_bias   : [8] float32
    b         : [total_seq_len, 8] bfloat16
    cu_seqlens: [num_seqs+1] int64
    scale     : scalar float32

    Returns
    -------
    output   : [total_seq_len, 8, 128] bfloat16
    new_state: [num_seqs, 8, 128, 128] float32, k-last [N, H, V, K]
    """
    total_seq_len, num_q_heads, K = q.shape
    num_v_heads = v.shape[1]
    num_seqs = cu_seqlens.shape[0] - 1
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    # Pre-compute gates for all tokens
    x = a.float() + dt_bias.float()                               # [T, 8]
    g_all = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [T, 8]
    beta_all = torch.sigmoid(b.float())                            # [T, 8]

    # GVA expansion
    ratio = num_v_heads // num_q_heads  # 2
    q_exp = q.float().repeat_interleave(ratio, dim=1)  # [T, 8, K]
    k_exp = k.float().repeat_interleave(ratio, dim=1)  # [T, 8, K]
    v_f = v.float()                                    # [T, 8, K]

    output = torch.zeros(total_seq_len, num_v_heads, K, dtype=torch.float32, device=device)
    new_state = torch.zeros(num_seqs, num_v_heads, K, K, dtype=torch.float32, device=device)

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        if seq_end <= seq_start:
            continue

        # Initial state: k-last [H, V, K] -> k-first [H, K, V]
        if state is not None:
            S = state[seq_idx].float().transpose(-1, -2).clone()  # [8, K, V]
        else:
            S = torch.zeros(num_v_heads, K, K, dtype=torch.float32, device=device)

        # Sequential scan over tokens in the sequence
        for t in range(seq_start, seq_end):
            g_t = g_all[t]          # [8]
            beta_t = beta_all[t]    # [8]
            k_t = k_exp[t]          # [8, K]
            v_t = v_f[t]            # [8, K]
            q_t = q_exp[t]          # [8, K]

            # Decay
            S = g_t[:, None, None] * S               # [8, K, V]

            # Delta rule update
            old_v = torch.einsum("hk,hkv->hv", k_t, S)           # [8, V]
            new_v = beta_t[:, None] * v_t + (1.0 - beta_t[:, None]) * old_v  # [8, V]
            delta = new_v - old_v                                   # [8, V]
            S = S + torch.einsum("hk,hv->hkv", k_t, delta)        # [8, K, V]

            # Output
            output[t] = scale * torch.einsum("hk,hkv->hv", q_t, S)  # [8, V]

        new_state[seq_idx] = S.transpose(-1, -2)  # [H, K, V] -> [H, V, K] k-last

    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, new_state

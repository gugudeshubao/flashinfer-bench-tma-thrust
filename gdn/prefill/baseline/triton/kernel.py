"""
GDN Prefill kernel - gdn_prefill_qk4_v8_d128_k_last

Direct translation of the reference implementation for correctness baseline.
"""

import math

import torch
import torch.nn.functional as F


def _matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.float() @ b.float()


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
            state_remove = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), old_v_H1V)
            state_update = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), new_v_H1V)
            state_HKV = old_state_HKV - state_remove + state_update

            o_H1V = scale * _matmul(q_H1K, state_HKV)
            output[t] = o_H1V.squeeze(1).to(torch.bfloat16)

        new_state[seq_idx] = state_HKV.transpose(-1, -2)

    return output, new_state

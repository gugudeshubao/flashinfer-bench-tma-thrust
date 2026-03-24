"""
GDN Decode kernel - gdn_decode_qk4_v8_d128_k_last

Gated Delta Net single-token decode with GVA (Grouped Value Attention):
  num_q_heads=4, num_k_heads=4, num_v_heads=8, head_size=128

State layout: k-last [B, H, V, K]

Algorithm:
  g    = exp(-exp(A_log) * softplus(a + dt_bias))  # decay gate per head
  beta = sigmoid(b)                                  # update gate per head

  # Working in k-first layout [K, V]:
  S     = g * S                            # apply decay
  old_v = k @ S                            # [K] @ [K,V] -> [V]
  new_v = beta * v + (1-beta) * old_v     # weighted update
  S     = S + outer(k, new_v - old_v)     # delta rule
  o     = scale * q @ S                   # [K] @ [K,V] -> [V]
"""

import math

import torch
import torch.nn.functional as F


def kernel(q, k, v, state, A_log, a, dt_bias, b, scale):
    """
    GDN decode single-token forward pass.

    Parameters
    ----------
    q      : [B, 1, 4, 128] bfloat16
    k      : [B, 1, 4, 128] bfloat16
    v      : [B, 1, 8, 128] bfloat16
    state  : [B, 8, 128, 128] float32, k-last layout [B, H, V, K]  (optional)
    A_log  : [8] float32
    a      : [B, 1, 8] bfloat16
    dt_bias: [8] float32
    b      : [B, 1, 8] bfloat16
    scale  : scalar float32

    Returns
    -------
    output   : [B, 1, 8, 128] bfloat16
    new_state: [B, 8, 128, 128] float32, k-last layout [B, H, V, K]
    """
    B, T, num_q_heads, K = q.shape
    num_v_heads = v.shape[2]
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    # Compute decay and update gates
    # a: [B, 1, 8], dt_bias: [8], A_log: [8]
    x = a.float() + dt_bias.float()               # [B, 1, 8]
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [B, 1, 8]
    beta = torch.sigmoid(b.float())                # [B, 1, 8]

    # Squeeze seq dim, promote to float32
    q_f = q.float().squeeze(1)       # [B, 4, K]
    k_f = k.float().squeeze(1)       # [B, 4, K]
    v_f = v.float().squeeze(1)       # [B, 8, K]
    g_f = g.squeeze(1)               # [B, 8]
    beta_f = beta.squeeze(1)         # [B, 8]

    # GVA expansion: repeat q,k from 4 heads to 8 heads
    ratio = num_v_heads // num_q_heads  # 2
    q_exp = q_f.repeat_interleave(ratio, dim=1)  # [B, 8, K]
    k_exp = k_f.repeat_interleave(ratio, dim=1)  # [B, 8, K]

    # State: k-last [B, H, V, K] -> k-first [B, H, K, V]
    if state is not None:
        S = state.float().transpose(-1, -2).clone()  # [B, 8, K, V]
    else:
        S = torch.zeros(B, num_v_heads, K, K, dtype=torch.float32, device=device)

    # Apply decay gate: g_f [B, 8] -> broadcast over [B, 8, K, V]
    S = g_f[:, :, None, None] * S                     # [B, 8, K, V]

    # old_v = k @ S  : bmm over [B*8, 1, K] x [B*8, K, V] -> [B*8, 1, V]
    # Equivalent: einsum('bhk,bhkv->bhv', k_exp, S)
    old_v = torch.einsum("bhk,bhkv->bhv", k_exp, S)   # [B, 8, V]

    # new_v = beta * v + (1-beta) * old_v
    new_v = beta_f[:, :, None] * v_f + (1.0 - beta_f[:, :, None]) * old_v  # [B, 8, V]

    # State update: S += outer(k, new_v - old_v)
    delta = new_v - old_v                              # [B, 8, V]
    S = S + torch.einsum("bhk,bhv->bhkv", k_exp, delta)  # [B, 8, K, V]

    # Output: o = scale * q @ S
    out = scale * torch.einsum("bhk,bhkv->bhv", q_exp, S)  # [B, 8, V]

    # Pack outputs
    output = out.unsqueeze(1).to(torch.bfloat16)          # [B, 1, 8, 128]
    new_state = S.transpose(-1, -2)                        # [B, 8, V, K] k-last

    return output, new_state

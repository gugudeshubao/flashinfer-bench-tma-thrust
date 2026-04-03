from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch


def hadamard_transform(x: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    """Walsh-Hadamard transform over the last dimension."""
    last_dim = x.shape[-1]
    if last_dim == 0 or (last_dim & (last_dim - 1)) != 0:
        raise ValueError(f"hadamard_transform requires a power-of-two size, got {last_dim}")

    y = x.float().reshape(-1, last_dim)
    step = 1
    while step < last_dim:
        y = y.view(-1, last_dim // (step * 2), 2, step)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.cat((a + b, a - b), dim=2)
        step *= 2
    y = y.reshape_as(x.float())
    if scale is not None:
        y = y * scale
    return y.to(dtype=x.dtype)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to the last dimension of x."""
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"apply_rotary_emb requires an even head dim, got {x.shape[-1]}")

    x_float = x.float()
    x_complex = torch.view_as_complex(x_float.reshape(*x.shape[:-1], -1, 2))
    freqs = freqs_cis.to(device=x.device)
    while freqs.dim() < x_complex.dim():
        freqs = freqs.unsqueeze(0)
    out = torch.view_as_real(x_complex * freqs).flatten(-2)
    return out.to(dtype=x.dtype)


def build_causal_mask(
    query_len: int,
    key_len: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    start_pos: int = 0,
) -> torch.Tensor:
    """Create an additive causal mask with shape [1, query_len, key_len]."""
    q_pos = torch.arange(query_len, device=device) + start_pos
    k_pos = torch.arange(key_len, device=device)
    mask = torch.full((query_len, key_len), float("-inf"), device=device, dtype=dtype)
    mask = mask.masked_fill(k_pos.unsqueeze(0) <= q_pos.unsqueeze(1), 0.0)
    return mask.unsqueeze(0)


def _normalize_attn_mask(
    attn_mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    query_len: int,
    key_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if attn_mask is None:
        return None

    mask = attn_mask.to(device=device, dtype=torch.float32)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() != 3:
        raise ValueError(f"attn_mask must be [S, T], [1, S, T], or [B, S, T], got {tuple(mask.shape)}")
    if mask.shape[-2:] != (query_len, key_len):
        raise ValueError(
            f"attn_mask trailing dims must be ({query_len}, {key_len}), got {tuple(mask.shape)}"
        )
    if mask.shape[0] == 1:
        mask = mask.expand(batch_size, -1, -1)
    elif mask.shape[0] != batch_size:
        raise ValueError(f"attn_mask batch dim must be 1 or {batch_size}, got {mask.shape[0]}")
    return mask


def compute_index_scores(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_weights: torch.Tensor,
    *,
    index_scale: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Reference token-level indexer.

    index_q:       [B, S, I, D]
    index_k:       [B, T, D]
    index_weights: [B, S, I]
    """
    if index_q.dim() != 4:
        raise ValueError(f"index_q must be [B, S, I, D], got {tuple(index_q.shape)}")
    if index_k.dim() != 3:
        raise ValueError(f"index_k must be [B, T, D], got {tuple(index_k.shape)}")
    if index_weights.dim() != 3:
        raise ValueError(f"index_weights must be [B, S, I], got {tuple(index_weights.shape)}")

    batch_size, query_len, num_index_heads, index_dim = index_q.shape
    if index_k.shape[0] != batch_size or index_k.shape[-1] != index_dim:
        raise ValueError("index_k must match index_q batch and head dim")
    if index_weights.shape != (batch_size, query_len, num_index_heads):
        raise ValueError("index_weights must match [B, S, I]")

    scale = index_scale if index_scale is not None else 1.0 / math.sqrt(index_dim)
    scores = torch.einsum("bsid,btd->bsit", index_q.float(), index_k.float())
    scores = scores * index_weights.float().unsqueeze(-1)
    scores = scores.sum(dim=2) * scale

    mask = _normalize_attn_mask(
        attn_mask,
        batch_size=batch_size,
        query_len=query_len,
        key_len=index_k.shape[1],
        device=index_q.device,
    )
    if mask is not None:
        scores = scores + mask
    return scores


def select_topk_indices(index_scores: torch.Tensor, topk: Optional[int]) -> torch.Tensor:
    if index_scores.dim() != 3:
        raise ValueError(f"index_scores must be [B, S, T], got {tuple(index_scores.shape)}")
    k = index_scores.shape[-1] if topk is None else min(int(topk), index_scores.shape[-1])
    if k <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    return torch.topk(index_scores, k=k, dim=-1, largest=True, sorted=False).indices


def build_sparse_mask(topk_indices: torch.Tensor, key_len: int) -> torch.Tensor:
    if topk_indices.dim() != 3:
        raise ValueError(f"topk_indices must be [B, S, K], got {tuple(topk_indices.shape)}")
    mask = torch.full(
        (*topk_indices.shape[:2], key_len),
        float("-inf"),
        device=topk_indices.device,
        dtype=torch.float32,
    )
    mask.scatter_(-1, topk_indices, 0.0)
    return mask


def _project_kv_dense(
    compressed_kv: torch.Tensor,
    wkv_b: torch.Tensor,
    qk_nope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected = torch.einsum("btc,hoc->btho", compressed_kv.float(), wkv_b.float())
    k_nope, value = torch.split(
        projected,
        [qk_nope_head_dim, projected.shape[-1] - qk_nope_head_dim],
        dim=-1,
    )
    return k_nope, value


@dataclass
class SparseAttentionMetadata:
    topk_indices: torch.Tensor
    index_scores: Optional[torch.Tensor]
    sparse_mask: torch.Tensor


@torch.no_grad()
def prefill_reference(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    wkv_b: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_weights: torch.Tensor,
    *,
    topk: Optional[int] = None,
    topk_indices: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    index_scale: Optional[float] = None,
) -> tuple[torch.Tensor, SparseAttentionMetadata]:
    """
    DeepSeek Sparse Attention prefill baseline over MLA core tensors.

    q_nope:        [B, S, H, C_nope]
    q_pe:          [B, S, H, C_rope]
    compressed_kv: [B, T, R_kv]
    k_pe:          [B, T, C_rope]
    wkv_b:         [H, C_nope + C_v, R_kv]
    index_q:       [B, S, I, D_idx]
    index_k:       [B, T, D_idx]
    index_weights: [B, S, I]
    """
    batch_size, query_len, num_heads, qk_nope_head_dim = q_nope.shape
    _, key_len, rope_dim = k_pe.shape

    if q_pe.shape != (batch_size, query_len, num_heads, rope_dim):
        raise ValueError("q_pe must be [B, S, H, C_rope]")
    if compressed_kv.shape[0] != batch_size or compressed_kv.shape[1] != key_len:
        raise ValueError("compressed_kv must match [B, T, R_kv]")
    if wkv_b.shape[0] != num_heads or wkv_b.shape[1] < qk_nope_head_dim:
        raise ValueError("wkv_b must be [H, C_nope + C_v, R_kv]")

    attn_mask_f = _normalize_attn_mask(
        attn_mask,
        batch_size=batch_size,
        query_len=query_len,
        key_len=key_len,
        device=q_nope.device,
    )

    k_nope, value = _project_kv_dense(compressed_kv, wkv_b, qk_nope_head_dim)
    k = torch.cat(
        (
            k_nope,
            k_pe.float().unsqueeze(2).expand(-1, -1, num_heads, -1),
        ),
        dim=-1,
    )
    q = torch.cat((q_nope.float(), q_pe.float()), dim=-1)

    if topk_indices is None:
        index_scores = compute_index_scores(
            index_q=index_q,
            index_k=index_k,
            index_weights=index_weights,
            index_scale=index_scale,
            attn_mask=attn_mask_f,
        )
        topk_indices = select_topk_indices(index_scores, topk)
    else:
        index_scores = None

    sparse_mask = build_sparse_mask(topk_indices, key_len)
    if attn_mask_f is not None:
        sparse_mask = sparse_mask + attn_mask_f

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bshd,bthd->bsht", q, k) * scale
    probs = torch.softmax(scores + sparse_mask.unsqueeze(2), dim=-1)
    output = torch.einsum("bsht,bthd->bshd", probs.float(), value.float())

    return output.to(dtype=q_nope.dtype), SparseAttentionMetadata(
        topk_indices=topk_indices,
        index_scores=index_scores,
        sparse_mask=sparse_mask,
    )


@torch.no_grad()
def decode_reference(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe_cache: torch.Tensor,
    wkv_b: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_weights: torch.Tensor,
    *,
    topk: Optional[int] = None,
    topk_indices: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    index_scale: Optional[float] = None,
) -> tuple[torch.Tensor, SparseAttentionMetadata]:
    """
    DeepSeek Sparse Attention decode baseline over MLA cache tensors.

    q_nope:        [B, S, H, C_nope]
    q_pe:          [B, S, H, C_rope]
    compressed_kv: [B, T, R_kv]
    k_pe_cache:    [B, T, C_rope]
    wkv_b:         [H, C_nope + C_v, R_kv]
    index_q:       [B, S, I, D_idx]
    index_k:       [B, T, D_idx]
    index_weights: [B, S, I]
    """
    batch_size, query_len, num_heads, qk_nope_head_dim = q_nope.shape
    _, key_len, rope_dim = k_pe_cache.shape

    if q_pe.shape != (batch_size, query_len, num_heads, rope_dim):
        raise ValueError("q_pe must be [B, S, H, C_rope]")
    if compressed_kv.shape[0] != batch_size or compressed_kv.shape[1] != key_len:
        raise ValueError("compressed_kv must match [B, T, R_kv]")
    if wkv_b.shape[0] != num_heads or wkv_b.shape[1] < qk_nope_head_dim:
        raise ValueError("wkv_b must be [H, C_nope + C_v, R_kv]")

    attn_mask_f = _normalize_attn_mask(
        attn_mask,
        batch_size=batch_size,
        query_len=query_len,
        key_len=key_len,
        device=q_nope.device,
    )

    if topk_indices is None:
        index_scores = compute_index_scores(
            index_q=index_q,
            index_k=index_k,
            index_weights=index_weights,
            index_scale=index_scale,
            attn_mask=attn_mask_f,
        )
        topk_indices = select_topk_indices(index_scores, topk)
    else:
        index_scores = None

    sparse_mask = build_sparse_mask(topk_indices, key_len)
    if attn_mask_f is not None:
        sparse_mask = sparse_mask + attn_mask_f

    q_nope_proj = torch.einsum(
        "bshd,hdc->bshc",
        q_nope.float(),
        wkv_b[:, :qk_nope_head_dim].float(),
    )
    scores_nope = torch.einsum("bshc,btc->bsht", q_nope_proj, compressed_kv.float())
    scores_rope = torch.einsum("bshr,btr->bsht", q_pe.float(), k_pe_cache.float())
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(qk_nope_head_dim + rope_dim)
    probs = torch.softmax((scores_nope + scores_rope) * scale + sparse_mask.unsqueeze(2), dim=-1)

    latent = torch.einsum("bsht,btc->bshc", probs.float(), compressed_kv.float())
    value_proj = wkv_b[:, qk_nope_head_dim:].float()
    output = torch.einsum("bshc,hvc->bshv", latent, value_proj)

    return output.to(dtype=q_nope.dtype), SparseAttentionMetadata(
        topk_indices=topk_indices,
        index_scores=index_scores,
        sparse_mask=sparse_mask,
    )

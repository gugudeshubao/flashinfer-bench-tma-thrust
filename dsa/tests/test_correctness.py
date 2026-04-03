"""Correctness tests for the first DSA baseline."""

import sys
from pathlib import Path

import torch


def _add_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_add_repo_root()

from dsa.common.reference import (
    build_causal_mask,
    compute_index_scores,
    decode_reference,
    prefill_reference,
    select_topk_indices,
)


def _make_inputs(batch_size: int, query_len: int, key_len: int, device: str):
    torch.manual_seed(0)
    dtype = torch.float32
    num_heads = 4
    qk_nope_head_dim = 8
    rope_dim = 4
    v_head_dim = 8
    kv_rank = 12
    num_index_heads = 3
    index_dim = 8

    q_nope = torch.randn(batch_size, query_len, num_heads, qk_nope_head_dim, device=device, dtype=dtype)
    q_pe = torch.randn(batch_size, query_len, num_heads, rope_dim, device=device, dtype=dtype)
    compressed_kv = torch.randn(batch_size, key_len, kv_rank, device=device, dtype=dtype)
    k_pe = torch.randn(batch_size, key_len, rope_dim, device=device, dtype=dtype)
    wkv_b = torch.randn(num_heads, qk_nope_head_dim + v_head_dim, kv_rank, device=device, dtype=dtype)
    index_q = torch.randn(batch_size, query_len, num_index_heads, index_dim, device=device, dtype=dtype)
    index_k = torch.randn(batch_size, key_len, index_dim, device=device, dtype=dtype)
    index_weights = torch.randn(batch_size, query_len, num_index_heads, device=device, dtype=dtype)

    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "compressed_kv": compressed_kv,
        "k_pe": k_pe,
        "wkv_b": wkv_b,
        "index_q": index_q,
        "index_k": index_k,
        "index_weights": index_weights,
        "qk_nope_head_dim": qk_nope_head_dim,
    }


def _naive_prefill(inputs: dict, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    q_nope = inputs["q_nope"]
    q_pe = inputs["q_pe"]
    compressed_kv = inputs["compressed_kv"]
    k_pe = inputs["k_pe"]
    wkv_b = inputs["wkv_b"]
    index_q = inputs["index_q"]
    index_k = inputs["index_k"]
    index_weights = inputs["index_weights"]
    qk_nope_head_dim = inputs["qk_nope_head_dim"]

    batch_size, query_len, num_heads, _ = q_nope.shape
    key_len = compressed_kv.shape[1]
    scale = (q_nope.shape[-1] + q_pe.shape[-1]) ** -0.5
    attn_mask = build_causal_mask(query_len, key_len, device=q_nope.device)
    index_scores = compute_index_scores(index_q, index_k, index_weights, attn_mask=attn_mask)
    topk_indices = select_topk_indices(index_scores, topk)

    projected = torch.einsum("btc,hoc->btho", compressed_kv, wkv_b)
    k_nope = projected[..., :qk_nope_head_dim]
    value = projected[..., qk_nope_head_dim:]
    k = torch.cat((k_nope, k_pe.unsqueeze(2).expand(-1, -1, num_heads, -1)), dim=-1)
    q = torch.cat((q_nope, q_pe), dim=-1)

    out = torch.zeros(batch_size, query_len, num_heads, value.shape[-1], device=q_nope.device, dtype=q_nope.dtype)
    for b_idx in range(batch_size):
        for s_idx in range(query_len):
            picked = topk_indices[b_idx, s_idx]
            mask_batch = 0 if attn_mask.shape[0] == 1 else b_idx
            picked_mask = attn_mask[mask_batch, s_idx, picked]
            for h_idx in range(num_heads):
                logits = (k[b_idx, picked, h_idx] * q[b_idx, s_idx, h_idx]).sum(dim=-1) * scale
                logits = logits + picked_mask
                probs = torch.softmax(logits.float(), dim=-1)
                out[b_idx, s_idx, h_idx] = probs @ value[b_idx, picked, h_idx]
    return out, topk_indices


def _naive_decode(inputs: dict, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    q_nope = inputs["q_nope"]
    q_pe = inputs["q_pe"]
    compressed_kv = inputs["compressed_kv"]
    k_pe = inputs["k_pe"]
    wkv_b = inputs["wkv_b"]
    index_q = inputs["index_q"]
    index_k = inputs["index_k"]
    index_weights = inputs["index_weights"]
    qk_nope_head_dim = inputs["qk_nope_head_dim"]

    batch_size, query_len, num_heads, _ = q_nope.shape
    scale = (q_nope.shape[-1] + q_pe.shape[-1]) ** -0.5
    index_scores = compute_index_scores(index_q, index_k, index_weights)
    topk_indices = select_topk_indices(index_scores, topk)

    q_nope_proj = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :qk_nope_head_dim])
    value_proj = wkv_b[:, qk_nope_head_dim:]
    out = torch.zeros(batch_size, query_len, num_heads, value_proj.shape[1], device=q_nope.device, dtype=q_nope.dtype)

    for b_idx in range(batch_size):
        for s_idx in range(query_len):
            picked = topk_indices[b_idx, s_idx]
            for h_idx in range(num_heads):
                nope_logits = torch.einsum(
                    "c,kc->k",
                    q_nope_proj[b_idx, s_idx, h_idx],
                    compressed_kv[b_idx, picked],
                )
                rope_logits = torch.einsum("r,kr->k", q_pe[b_idx, s_idx, h_idx], k_pe[b_idx, picked])
                probs = torch.softmax((nope_logits + rope_logits).float() * scale, dim=-1)
                latent = probs @ compressed_kv[b_idx, picked]
                out[b_idx, s_idx, h_idx] = torch.einsum("c,vc->v", latent, value_proj[h_idx])
    return out, topk_indices


def test_prefill_matches_naive() -> None:
    inputs = _make_inputs(batch_size=2, query_len=9, key_len=9, device="cpu")
    topk = 4
    ref_out, metadata = prefill_reference(
        q_nope=inputs["q_nope"],
        q_pe=inputs["q_pe"],
        compressed_kv=inputs["compressed_kv"],
        k_pe=inputs["k_pe"],
        wkv_b=inputs["wkv_b"],
        index_q=inputs["index_q"],
        index_k=inputs["index_k"],
        index_weights=inputs["index_weights"],
        topk=topk,
        attn_mask=build_causal_mask(9, 9, device="cpu"),
    )
    naive_out, naive_topk = _naive_prefill(inputs, topk=topk)
    assert torch.equal(metadata.topk_indices, naive_topk)
    assert torch.allclose(ref_out, naive_out, atol=1e-5, rtol=1e-5)


def test_decode_matches_naive() -> None:
    inputs = _make_inputs(batch_size=2, query_len=1, key_len=13, device="cpu")
    topk = 5
    ref_out, metadata = decode_reference(
        q_nope=inputs["q_nope"],
        q_pe=inputs["q_pe"],
        compressed_kv=inputs["compressed_kv"],
        k_pe_cache=inputs["k_pe"],
        wkv_b=inputs["wkv_b"],
        index_q=inputs["index_q"],
        index_k=inputs["index_k"],
        index_weights=inputs["index_weights"],
        topk=topk,
    )
    naive_out, naive_topk = _naive_decode(inputs, topk=topk)
    assert torch.equal(metadata.topk_indices, naive_topk)
    assert torch.allclose(ref_out, naive_out, atol=1e-5, rtol=1e-5)


def test_full_topk_recovers_dense_prefill() -> None:
    inputs = _make_inputs(batch_size=1, query_len=7, key_len=7, device="cpu")
    attn_mask = build_causal_mask(7, 7, device="cpu")
    sparse_out, _ = prefill_reference(
        q_nope=inputs["q_nope"],
        q_pe=inputs["q_pe"],
        compressed_kv=inputs["compressed_kv"],
        k_pe=inputs["k_pe"],
        wkv_b=inputs["wkv_b"],
        index_q=inputs["index_q"],
        index_k=inputs["index_k"],
        index_weights=inputs["index_weights"],
        topk=7,
        attn_mask=attn_mask,
    )

    qk_nope_head_dim = inputs["qk_nope_head_dim"]
    projected = torch.einsum("btc,hoc->btho", inputs["compressed_kv"], inputs["wkv_b"])
    k_nope = projected[..., :qk_nope_head_dim]
    value = projected[..., qk_nope_head_dim:]
    num_heads = inputs["q_nope"].shape[2]
    k = torch.cat((k_nope, inputs["k_pe"].unsqueeze(2).expand(-1, -1, num_heads, -1)), dim=-1)
    q = torch.cat((inputs["q_nope"], inputs["q_pe"]), dim=-1)
    logits = torch.einsum("bshd,bthd->bsht", q, k) * (q.shape[-1] ** -0.5)
    dense_out = torch.einsum(
        "bsht,bthd->bshd",
        torch.softmax(logits + attn_mask.unsqueeze(2), dim=-1),
        value,
    )
    assert torch.allclose(sparse_out, dense_out, atol=1e-5, rtol=1e-5)


def main() -> None:
    test_prefill_matches_naive()
    test_decode_matches_naive()
    test_full_topk_recovers_dense_prefill()
    print("All DSA correctness tests passed.")


if __name__ == "__main__":
    main()

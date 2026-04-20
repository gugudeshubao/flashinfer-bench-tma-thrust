"""
DSA decode solution entry.

Decode shares the same MLA latent sparse-attention math as prefill, with
`query_len == 1` and no causal mask. Reuse the prefill Triton path directly so
the two stages stay numerically aligned.
"""

import sys
from pathlib import Path


def _add_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_add_repo_root()

from dsa.prefill.solution.triton.kernel import kernel as prefill_triton_kernel


def kernel(
    q_nope,
    q_pe,
    compressed_kv,
    k_pe_cache,
    wkv_b,
    index_q,
    index_k,
    index_weights,
    topk=None,
    topk_indices=None,
    attn_mask=None,
    softmax_scale=None,
    index_scale=None,
    backend="auto",
    causal_mask_hint=None,
    num_warps_override=None,
    num_stages_override=None,
    weighted_q_kernel_override=None,
    return_metadata=False,
):
    return prefill_triton_kernel(
        q_nope=q_nope,
        q_pe=q_pe,
        compressed_kv=compressed_kv,
        k_pe=k_pe_cache,
        wkv_b=wkv_b,
        index_q=index_q,
        index_k=index_k,
        index_weights=index_weights,
        topk=topk,
        topk_indices=topk_indices,
        attn_mask=attn_mask,
        softmax_scale=softmax_scale,
        index_scale=index_scale,
        backend=backend,
        causal_mask_hint=causal_mask_hint,
        num_warps_override=num_warps_override,
        num_stages_override=num_stages_override,
        weighted_q_kernel_override=weighted_q_kernel_override,
        return_metadata=return_metadata,
    )

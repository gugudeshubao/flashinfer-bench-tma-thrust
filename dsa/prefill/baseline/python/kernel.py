"""PyTorch reference kernel for DeepSeek Sparse Attention prefill."""

import sys
from pathlib import Path


def _add_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_add_repo_root()

from dsa.common.reference import prefill_reference


def kernel(
    q_nope,
    q_pe,
    compressed_kv,
    k_pe,
    wkv_b,
    index_q,
    index_k,
    index_weights,
    topk=None,
    topk_indices=None,
    attn_mask=None,
    softmax_scale=None,
    index_scale=None,
    return_metadata=False,
):
    output, metadata = prefill_reference(
        q_nope=q_nope,
        q_pe=q_pe,
        compressed_kv=compressed_kv,
        k_pe=k_pe,
        wkv_b=wkv_b,
        index_q=index_q,
        index_k=index_k,
        index_weights=index_weights,
        topk=topk,
        topk_indices=topk_indices,
        attn_mask=attn_mask,
        softmax_scale=softmax_scale,
        index_scale=index_scale,
    )
    return (output, metadata) if return_metadata else output

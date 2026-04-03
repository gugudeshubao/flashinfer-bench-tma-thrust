from dataclasses import dataclass
from typing import Optional

import torch

from dsa.common.reference import compute_index_scores, select_topk_indices


@dataclass
class IndexerOutput:
    topk_indices: torch.Tensor
    scores: Optional[torch.Tensor] = None


class DSAIndexer(torch.nn.Module):
    """Reference token-level indexer for DeepSeek sparse attention."""

    def __init__(self, index_scale: Optional[float] = None):
        super().__init__()
        self.index_scale = index_scale

    @torch.no_grad()
    def forward(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        index_weights: torch.Tensor,
        topk: Optional[int] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_scores: bool = False,
    ) -> IndexerOutput:
        scores = compute_index_scores(
            index_q=index_q,
            index_k=index_k,
            index_weights=index_weights,
            index_scale=self.index_scale,
            attn_mask=attn_mask,
        )
        topk_indices = select_topk_indices(scores, topk)
        return IndexerOutput(
            topk_indices=topk_indices,
            scores=scores if return_scores else None,
        )


@torch.no_grad()
def run_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_weights: torch.Tensor,
    topk: Optional[int] = None,
    attn_mask: Optional[torch.Tensor] = None,
    index_scale: Optional[float] = None,
    return_scores: bool = False,
) -> IndexerOutput:
    module = DSAIndexer(index_scale=index_scale)
    return module(
        index_q=index_q,
        index_k=index_k,
        index_weights=index_weights,
        topk=topk,
        attn_mask=attn_mask,
        return_scores=return_scores,
    )

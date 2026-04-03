"""
Hybrid MoE variant.

Use the CUDA/C++ torch extension path for very small batches where Python
overhead matters most, and fall back to the current default implementation for
larger workloads.
"""

import sys
from pathlib import Path


_SMALL_T_THRESHOLD = 4


def _add_repo_roots():
    candidate_roots = ["/root", str(Path(__file__).resolve().parents[3])]
    for root in candidate_roots:
        if Path(root, "moe").exists() and root not in sys.path:
            sys.path.insert(0, root)


def _get_impls():
    _add_repo_roots()
    from moe.solution.triton import kernel as default_impl
    from moe.solution.cute_cpp import kernel as cute_impl

    return default_impl, cute_impl


def kernel(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    default_impl, cute_impl = _get_impls()
    if routing_logits.shape[0] <= _SMALL_T_THRESHOLD:
        return cute_impl.kernel(
            routing_logits,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            local_expert_offset,
            routed_scaling_factor,
        )

    return default_impl.kernel(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        local_expert_offset,
        routed_scaling_factor,
    )

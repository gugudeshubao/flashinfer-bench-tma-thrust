"""
Hybrid tiny-seq variant.

Use the CUDA/C++ torch extension path for the tiniest workloads where Python
overhead dominates, and keep the current default implementation for larger
workloads.
"""

import sys
from pathlib import Path


_TINY_SEQ_THRESHOLD = 2


def _add_repo_roots():
    candidate_roots = ["/root", str(Path(__file__).resolve().parents[3])]
    for root in candidate_roots:
        if Path(root, "moe").exists() and root not in sys.path:
            sys.path.insert(0, root)


def _get_impls():
    _add_repo_roots()
    from moe.solution.triton import kernel as default_impl
    from moe.solution.cute_cpp_torch import runtime as cute_cpp_torch_runtime

    return default_impl, cute_cpp_torch_runtime


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
    default_impl, cute_cpp_torch_runtime = _get_impls()

    if routing_logits.shape[0] <= _TINY_SEQ_THRESHOLD:
        mod = cute_cpp_torch_runtime.get_module()
        if mod is not None:
            return mod.kernel(
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

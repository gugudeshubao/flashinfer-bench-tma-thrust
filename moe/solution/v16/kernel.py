"""
PTX/CUDA extension threshold experiment.

This variant uses the same logic as the current default implementation, but
routes workloads with sequence length <= 8 to the CUDA/C++ torch extension.
"""

import sys
from pathlib import Path


_SEQ_THRESHOLD = 8


def _add_repo_roots():
    candidate_roots = ["/root", str(Path(__file__).resolve().parents[3])]
    for root in candidate_roots:
        if Path(root, "moe").exists() and root not in sys.path:
            sys.path.insert(0, root)


def _get_impls():
    _add_repo_roots()
    from moe.solution.v3 import kernel as stable_impl
    from moe.solution.cute_cpp_torch import runtime as cute_cpp_torch_runtime

    return stable_impl, cute_cpp_torch_runtime


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
    stable_impl, cute_cpp_torch_runtime = _get_impls()
    if routing_logits.shape[0] <= _SEQ_THRESHOLD:
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

    return stable_impl.kernel(
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

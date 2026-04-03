"""
Experimental PTX-threshold tuning variant.

Reuses the default implementation but lowers the PTX SwiGLU threshold so
moderately sized expert batches also use the JIT kernel.
"""

import sys
from pathlib import Path


def _import_default_impl():
    candidate_roots = ["/root", str(Path(__file__).resolve().parents[3])]
    for root in candidate_roots:
        if Path(root, "moe").exists() and root not in sys.path:
            sys.path.insert(0, root)
    from moe.solution.triton import kernel as default_impl

    return default_impl


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
    impl = _import_default_impl()
    impl._SWIGLU_PTX_MIN_ELEMS = 4096
    return impl.kernel(
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

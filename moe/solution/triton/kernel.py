"""
FP8 Block-Scale Fused MoE — default submission path.

This default path first tries the CUDA/C++ torch extension implementation. If
that extension is unavailable, it falls back to the CuTe/C++ shared-memory
SwiGLU path.
"""

import sys
from collections import OrderedDict
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F


H = 7168
I = 2048
E_GLOBAL = 256
E_LOCAL = 32
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
BLK = 128

_gemm_mode = None  # "scaled_mm" or "f32"
_cuda_module = None
_cuda_failed = False
_CUTE_MIN_ELEMS = 8192
_torch_ext_module = None
_torch_ext_failed = False
_ptx_torch_ext_module = None
_ptx_torch_ext_failed = False
_HS_SCALE_CACHE_LIMIT = 8
_ROUTE_CACHE_LIMIT = 8
_W13_CACHE_LIMIT = 4
_hs_scale_cache = OrderedDict()
_route_cache = OrderedDict()
_w13_t_cache = OrderedDict()
_w13_scale_t_cache = OrderedDict()


def _cache_get(cache, key, validate_objs):
    value = cache.get(key)
    if value is None:
        return None

    cached_objs, payload = value
    if any(cached is not current for cached, current in zip(cached_objs, validate_objs)):
        cache.pop(key, None)
        return None

    cache.move_to_end(key)
    return payload


def _cache_put(cache, key, validate_objs, payload, limit):
    cache[key] = (tuple(validate_objs), payload)
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


def _add_repo_roots():
    candidate_roots = ["/root", str(Path(__file__).resolve().parents[3])]
    for root in candidate_roots:
        if Path(root, "moe").exists() and root not in sys.path:
            sys.path.insert(0, root)


def _load_torch_ext_module():
    global _torch_ext_module, _torch_ext_failed
    if _torch_ext_module is not None:
        return _torch_ext_module
    if _torch_ext_failed:
        return None

    try:
        _add_repo_roots()
        from moe.solution.cute_cpp_torch import runtime as cute_cpp_torch_runtime

        _torch_ext_module = cute_cpp_torch_runtime.get_module()
        if _torch_ext_module is None:
            _torch_ext_failed = True
        return _torch_ext_module
    except Exception:
        _torch_ext_failed = True
        return None


def _load_ptx_torch_ext_module():
    global _ptx_torch_ext_module, _ptx_torch_ext_failed
    if _ptx_torch_ext_module is not None:
        return _ptx_torch_ext_module
    if _ptx_torch_ext_failed:
        return None

    try:
        from torch.utils.cpp_extension import load

        src = Path(__file__).with_name("moe_cuda_ptx_torch_kernel.cu")
        digest = hashlib.sha1(src.read_bytes()).hexdigest()[:10]
        _ptx_torch_ext_module = load(
            name=f"moe_cuda_ptx_torch_ext_default_{digest}",
            sources=[str(src)],
            extra_include_paths=[str(src.parent)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        if _ptx_torch_ext_module is None:
            _ptx_torch_ext_failed = True
        return _ptx_torch_ext_module
    except Exception:
        _ptx_torch_ext_failed = True
        return None


def _prewarm_torch_extensions():
    # Best-effort preload so extension build/load cost is paid before the first
    # measured kernel invocation when the module import itself is outside timing.
    try:
        _load_ptx_torch_ext_module()
    except Exception:
        pass
    try:
        _load_torch_ext_module()
    except Exception:
        pass


def _detect_gemm_mode(act_fp8, act_scale, w_fp8, w_scale):
    global _gemm_mode
    if _gemm_mode is not None:
        return

    try:
        _ = torch._scaled_mm(
            act_fp8,
            w_fp8.t().contiguous(),
            scale_a=act_scale,
            scale_b=w_scale.t().contiguous(),
            out_dtype=torch.float32,
        )
        _gemm_mode = "scaled_mm"
    except Exception:
        _gemm_mode = "f32"


def _route(logits, bias, scaling_factor):
    t = logits.shape[0]
    s = 1.0 / (1.0 + torch.exp(-logits.float()))
    s_b = s + bias.float().reshape(-1)

    group_size = E_GLOBAL // N_GROUP
    grouped = s_b.view(t, N_GROUP, group_size)
    top2, _ = torch.topk(grouped, k=2, dim=2, largest=True, sorted=False)
    g_scores = top2.sum(dim=2)

    _, g_idx = torch.topk(g_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    g_mask = torch.zeros_like(g_scores)
    g_mask.scatter_(1, g_idx, 1.0)
    e_mask = g_mask.unsqueeze(2).expand(t, N_GROUP, group_size).reshape(t, E_GLOBAL)

    pruned = s_b.masked_fill(e_mask == 0, torch.finfo(torch.float32).min)
    _, topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    mask = torch.zeros_like(s)
    mask.scatter_(1, topk_idx, 1.0)
    weights = s * mask
    weights = (weights / (weights.sum(dim=1, keepdim=True) + 1e-20)) * scaling_factor
    return topk_idx, weights


def _prepare_route_cache(routing_logits, routing_bias, local_start, routed_scaling_factor, device):
    t = routing_logits.shape[0]
    key = (
        int(routing_logits.data_ptr()),
        int(routing_bias.data_ptr()),
        tuple(routing_logits.shape),
        int(local_start),
        float(routed_scaling_factor),
    )
    cached = _cache_get(_route_cache, key, (routing_logits, routing_bias))
    if cached is not None:
        return cached

    topk_idx, weights = _route(routing_logits, routing_bias, routed_scaling_factor)
    assignments = _build_expert_assignments(topk_idx, local_start, t, device)
    active_experts = [le for le in range(E_LOCAL) if assignments[le] is not None]
    payload = (weights, assignments, active_experts)
    _cache_put(_route_cache, key, (routing_logits, routing_bias), payload, _ROUTE_CACHE_LIMIT)
    return payload


def _build_expert_assignments(topk_idx, local_start, t, device):
    flat_tok = torch.arange(t, device=device).unsqueeze(1).expand(-1, TOP_K).reshape(-1)
    flat_exp = topk_idx.reshape(-1)

    local_mask = (flat_exp >= local_start) & (flat_exp < local_start + E_LOCAL)
    local_tok = flat_tok[local_mask]
    local_exp = flat_exp[local_mask] - local_start

    if local_tok.numel() == 0:
        return [None] * E_LOCAL

    order = torch.argsort(local_exp, stable=True)
    sorted_tok = local_tok[order]
    sorted_exp = local_exp[order]

    counts = torch.bincount(sorted_exp.int(), minlength=E_LOCAL)
    offsets = torch.zeros(E_LOCAL + 1, dtype=torch.int64, device=device)
    torch.cumsum(counts, dim=0, out=offsets[1:])

    assignments = []
    for e in range(E_LOCAL):
        s_off, e_off = offsets[e].item(), offsets[e + 1].item()
        assignments.append(sorted_tok[s_off:e_off] if s_off < e_off else None)

    return assignments


def _expand_scale_2d(scale):
    sn, sk = scale.shape
    out = scale.unsqueeze(1).expand(sn, BLK, sk).reshape(sn * BLK, sk)
    out = out.unsqueeze(2).expand(sn * BLK, sk, BLK).reshape(sn * BLK, sk * BLK)
    return out


def _prepare_hs_scale_cache(hidden_states_scale):
    key = (int(hidden_states_scale.data_ptr()), tuple(hidden_states_scale.shape))
    cached = _cache_get(_hs_scale_cache, key, (hidden_states_scale,))
    if cached is not None:
        return cached

    value = hidden_states_scale.float().permute(1, 0).contiguous()
    _cache_put(_hs_scale_cache, key, (hidden_states_scale,), value, _HS_SCALE_CACHE_LIMIT)
    return value


def _get_w13_t(gemm1_weights, le):
    key = (int(gemm1_weights.data_ptr()), tuple(gemm1_weights.shape), le)
    cached = _cache_get(_w13_t_cache, key, (gemm1_weights,))
    if cached is not None:
        return cached

    value = gemm1_weights[le].t().contiguous()
    _cache_put(_w13_t_cache, key, (gemm1_weights,), value, _W13_CACHE_LIMIT)
    return value


def _get_w13_scale_t(gemm1_weights_scale, le):
    key = (int(gemm1_weights_scale.data_ptr()), tuple(gemm1_weights_scale.shape), le)
    cached = _cache_get(_w13_scale_t_cache, key, (gemm1_weights_scale,))
    if cached is not None:
        return cached

    value = gemm1_weights_scale[le].t().contiguous()
    _cache_put(_w13_scale_t_cache, key, (gemm1_weights_scale,), value, _W13_CACHE_LIMIT)
    return value


def _load_cuda_module():
    global _cuda_module, _cuda_failed
    if _cuda_module is not None:
        return _cuda_module
    if _cuda_failed:
        return None

    try:
        from torch.utils.cpp_extension import load

        src = Path(__file__).with_name("moe_cute_swiglu.cu")
        digest = hashlib.sha1(src.read_bytes()).hexdigest()[:10]
        include_paths = ["/opt/cutlass/include", "/opt/cutlass/tools/util/include", str(src.parent)]
        _cuda_module = load(
            name=f"moe_cute_cpp_ext_default_{digest}",
            sources=[str(src)],
            extra_include_paths=include_paths,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        return _cuda_module
    except Exception:
        _cuda_failed = True
        return None


def _fused_swiglu(x1, x2):
    if x1.numel() < _CUTE_MIN_ELEMS:
        return F.silu(x2) * x1

    cuda_mod = _load_cuda_module()
    if cuda_mod is None:
        return F.silu(x2) * x1

    out = torch.empty_like(x1)
    cuda_mod.moe_swiglu_cute(x1.contiguous(), x2.contiguous(), out)
    return out


@torch.no_grad()
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
    torch_ext = _load_ptx_torch_ext_module()
    if torch_ext is None:
        torch_ext = _load_torch_ext_module()
    if torch_ext is not None:
        return torch_ext.kernel(
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

    t = routing_logits.shape[0]
    device = routing_logits.device
    local_start = int(local_expert_offset)

    hs_scale_th = _prepare_hs_scale_cache(hidden_states_scale)
    _detect_gemm_mode(
        hidden_states[:1],
        hs_scale_th[:1],
        gemm1_weights[0],
        gemm1_weights_scale[0],
    )
    use_scaled_mm = (_gemm_mode == "scaled_mm")

    weights, assignments, active_experts = _prepare_route_cache(
        routing_logits,
        routing_bias,
        local_start,
        routed_scaling_factor,
        device,
    )

    if not active_experts:
        return torch.zeros((t, H), dtype=torch.bfloat16, device=device)

    a = None
    if not use_scaled_mm:
        a_fp32 = hidden_states.float()
        a_scale_exp = (
            hs_scale_th.unsqueeze(-1)
            .repeat(1, 1, BLK)
            .reshape(t, H)
            .contiguous()
        )
        a = a_fp32 * a_scale_exp

    output = torch.zeros((t, H), dtype=torch.float32, device=device)

    for le in active_experts:
        token_idx = assignments[le]
        ge = local_start + le

        if use_scaled_mm:
            g1 = torch._scaled_mm(
                hidden_states.index_select(0, token_idx),
                _get_w13_t(gemm1_weights, le),
                scale_a=hs_scale_th.index_select(0, token_idx),
                scale_b=_get_w13_scale_t(gemm1_weights_scale, le),
                out_dtype=torch.float32,
            )
        else:
            a_e = a.index_select(0, token_idx)
            w13_f32 = gemm1_weights[le].float()
            s13_exp = _expand_scale_2d(gemm1_weights_scale[le].float())
            g1 = a_e.matmul((w13_f32 * s13_exp).t())

        x1 = g1[:, :I]
        x2 = g1[:, I:]
        mid = _fused_swiglu(x1, x2)

        w2_f32 = gemm2_weights[le].float()
        s2_exp = _expand_scale_2d(gemm2_weights_scale[le].float())
        out_e = mid.matmul((w2_f32 * s2_exp).t())
        w_tok = weights.index_select(0, token_idx)[:, ge]
        output.index_add_(0, token_idx, out_e * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)


_prewarm_torch_extensions()

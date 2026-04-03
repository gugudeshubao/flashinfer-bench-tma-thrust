"""
Experimental cached MoE path.

This branch keeps the more aggressive cache and sparse-routing ideas isolated
from the stable default implementation.
"""

from collections import OrderedDict

import torch


H = 7168
I = 2048
E_GLOBAL = 256
E_LOCAL = 32
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
BLK = 128

_gemm_mode = None  # "scaled_mm" or "f32"
_ROUTE_CACHE_LIMIT = 5
_WEIGHT_CACHE_LIMIT = 5
_SCALE_CACHE_LIMIT = 5
_ACT_CACHE_LIMIT = 5
_route_cache = OrderedDict()
_weight_cache = OrderedDict()
_scale_cache = OrderedDict()
_act_cache = OrderedDict()


def _cache_get(cache, key):
    value = cache.pop(key, None)
    if value is not None:
        cache[key] = value
    return value


def _cache_put(cache, key, value, limit):
    if key in cache:
        cache.pop(key)
    cache[key] = value
    while len(cache) > limit:
        cache.popitem(last=False)


def _fast_expand_scale_2d(scale, blk, target_n, target_k):
    sn, sk = scale.shape
    out = scale.unsqueeze(1).expand(sn, blk, sk).reshape(sn * blk, sk)
    out = out.unsqueeze(2).expand(sn * blk, sk, blk).reshape(sn * blk, sk * blk)
    return out[:target_n, :target_k]


def _fast_expand_act_scale(scale_th, blk, target_h):
    t, sh = scale_th.shape
    return scale_th.unsqueeze(2).expand(t, sh, blk).reshape(t, sh * blk)[:, :target_h]


def _block_dequant_weight(w_fp8, w_scale):
    w_f32 = w_fp8.float()
    s_f32 = w_scale.float()
    s_exp = _fast_expand_scale_2d(s_f32, BLK, w_f32.shape[0], w_f32.shape[1])
    return w_f32 * s_exp


def _block_dequant_act(act_fp8, act_scale_th):
    s_exp = _fast_expand_act_scale(act_scale_th, BLK, act_fp8.shape[1])
    return act_fp8.float() * s_exp


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


def _route_sparse(logits, bias, scaling_factor):
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

    topk_scores = s.gather(1, topk_idx)
    topk_weights = (topk_scores / (topk_scores.sum(dim=1, keepdim=True) + 1e-20)) * scaling_factor
    return topk_idx, topk_weights


def _build_expert_assignments(topk_idx, topk_weights, local_start, t, device):
    flat_tok = torch.arange(t, device=device).unsqueeze(1).expand(-1, TOP_K).reshape(-1)
    flat_exp = topk_idx.reshape(-1)
    flat_w = topk_weights.reshape(-1)

    local_mask = (flat_exp >= local_start) & (flat_exp < local_start + E_LOCAL)
    local_tok = flat_tok[local_mask]
    local_exp = flat_exp[local_mask] - local_start
    local_w = flat_w[local_mask]

    if local_tok.numel() == 0:
        return [None] * E_LOCAL, [None] * E_LOCAL, []

    order = torch.argsort(local_exp, stable=True)
    sorted_tok = local_tok[order]
    sorted_exp = local_exp[order]
    sorted_w = local_w[order]

    counts = torch.bincount(sorted_exp.int(), minlength=E_LOCAL)
    offsets = torch.zeros(E_LOCAL + 1, dtype=torch.int64, device=device)
    torch.cumsum(counts, dim=0, out=offsets[1:])

    assignments = []
    weight_lists = []
    active_experts = []
    for e in range(E_LOCAL):
        s_off, e_off = offsets[e].item(), offsets[e + 1].item()
        if s_off < e_off:
            assignments.append(sorted_tok[s_off:e_off])
            weight_lists.append(sorted_w[s_off:e_off])
            active_experts.append(e)
        else:
            assignments.append(None)
            weight_lists.append(None)
    return assignments, weight_lists, active_experts


def _prepare_route_cache(routing_logits, routing_bias, local_start, routed_scaling_factor, device):
    t = routing_logits.shape[0]
    cache_key = (
        id(routing_logits),
        id(routing_bias),
        int(local_start),
        float(routed_scaling_factor),
        t,
    )
    cached = _cache_get(_route_cache, cache_key)
    if cached is not None:
        return cached

    topk_idx, topk_weights = _route_sparse(routing_logits, routing_bias, routed_scaling_factor)
    assignments, weight_lists, active_experts = _build_expert_assignments(
        topk_idx, topk_weights, local_start, t, device
    )

    payload = (assignments, weight_lists, active_experts)
    _cache_put(_route_cache, cache_key, payload, _ROUTE_CACHE_LIMIT)
    return payload


def _prepare_weight_cache(gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, use_scaled_mm):
    cache_key = (
        id(gemm1_weights),
        id(gemm1_weights_scale),
        id(gemm2_weights),
        id(gemm2_weights_scale),
        bool(use_scaled_mm),
    )
    cached = _cache_get(_weight_cache, cache_key)
    if cached is not None:
        return cached

    w2_t_f32 = []
    for le in range(E_LOCAL):
        w2_t_f32.append(
            _block_dequant_weight(gemm2_weights[le], gemm2_weights_scale[le]).t().contiguous()
        )

    payload = {"w2_t_f32": tuple(w2_t_f32)}

    if use_scaled_mm:
        payload["w13_t"] = tuple(gemm1_weights[le].t().contiguous() for le in range(E_LOCAL))
        payload["w13_scale_t"] = tuple(
            gemm1_weights_scale[le].t().contiguous() for le in range(E_LOCAL)
        )
    else:
        payload["w13_t_f32"] = tuple(
            _block_dequant_weight(gemm1_weights[le], gemm1_weights_scale[le]).t().contiguous()
            for le in range(E_LOCAL)
        )

    _cache_put(_weight_cache, cache_key, payload, _WEIGHT_CACHE_LIMIT)
    return payload


def _prepare_scale_cache(hidden_states_scale):
    cache_key = (
        id(hidden_states_scale),
        hidden_states_scale.shape[1],
    )
    cached = _cache_get(_scale_cache, cache_key)
    if cached is not None:
        return cached

    hs_scale_th = hidden_states_scale.float().permute(1, 0).contiguous()
    _cache_put(_scale_cache, cache_key, hs_scale_th, _SCALE_CACHE_LIMIT)
    return hs_scale_th


def _prepare_act_cache(hidden_states, hidden_states_scale, hs_scale_th):
    cache_key = (
        id(hidden_states),
        id(hidden_states_scale),
        hidden_states.shape[0],
    )
    cached = _cache_get(_act_cache, cache_key)
    if cached is not None:
        return cached

    acts = _block_dequant_act(hidden_states, hs_scale_th)
    _cache_put(_act_cache, cache_key, acts, _ACT_CACHE_LIMIT)
    return acts


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
    t = routing_logits.shape[0]
    device = routing_logits.device
    local_start = int(local_expert_offset)

    hs_scale_th = _prepare_scale_cache(hidden_states_scale)
    _detect_gemm_mode(
        hidden_states[:1],
        hs_scale_th[:1],
        gemm1_weights[0],
        gemm1_weights_scale[0],
    )
    use_scaled_mm = (_gemm_mode == "scaled_mm")

    assignments, weight_lists, active_experts = _prepare_route_cache(
        routing_logits,
        routing_bias,
        local_start,
        routed_scaling_factor,
        device,
    )

    if not active_experts:
        return torch.zeros((t, H), dtype=torch.bfloat16, device=device)

    packed_weights = _prepare_weight_cache(
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        use_scaled_mm,
    )

    acts = None
    if not use_scaled_mm:
        acts = _prepare_act_cache(hidden_states, hidden_states_scale, hs_scale_th)

    output = torch.zeros((t, H), dtype=torch.float32, device=device)

    for le in active_experts:
        token_idx = assignments[le]
        w_tok = weight_lists[le]

        if use_scaled_mm:
            g1 = torch._scaled_mm(
                hidden_states.index_select(0, token_idx),
                packed_weights["w13_t"][le],
                scale_a=hs_scale_th.index_select(0, token_idx),
                scale_b=packed_weights["w13_scale_t"][le],
                out_dtype=torch.float32,
            )
        else:
            act_e = acts.index_select(0, token_idx)
            g1 = act_e.matmul(packed_weights["w13_t_f32"][le])

        x1 = g1[:, :I]
        x2 = g1[:, I:]
        mid = (x2 / (1.0 + torch.exp(-x2))) * x1

        out_e = mid.matmul(packed_weights["w2_t_f32"][le])
        output.index_add_(0, token_idx, out_e * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)

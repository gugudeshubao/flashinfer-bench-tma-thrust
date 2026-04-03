"""
Experimental CuTe/C++ MoE variant.

This keeps the default algorithm but swaps SwiGLU to a CUDA extension that uses
CuTe tensor/layout constructs in shared memory. If CuTe compilation is blocked,
it falls back to the exact PyTorch expression.
"""

from pathlib import Path
import hashlib

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
            name=f"moe_cute_cpp_ext_{digest}",
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
    t = routing_logits.shape[0]
    device = routing_logits.device
    local_start = int(local_expert_offset)

    hs_scale_th = hidden_states_scale.float().permute(1, 0).contiguous()
    _detect_gemm_mode(
        hidden_states[:1],
        hs_scale_th[:1],
        gemm1_weights[0],
        gemm1_weights_scale[0],
    )
    use_scaled_mm = (_gemm_mode == "scaled_mm")

    topk_idx, weights = _route(routing_logits, routing_bias, routed_scaling_factor)
    assignments = _build_expert_assignments(topk_idx, local_start, t, device)
    active_experts = [le for le in range(E_LOCAL) if assignments[le] is not None]

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
    cuda_mod = _load_cuda_module()

    for le in active_experts:
        token_idx = assignments[le]
        ge = local_start + le

        if use_scaled_mm:
            g1 = torch._scaled_mm(
                hidden_states.index_select(0, token_idx),
                gemm1_weights[le].t().contiguous(),
                scale_a=hs_scale_th.index_select(0, token_idx),
                scale_b=gemm1_weights_scale[le].t().contiguous(),
                out_dtype=torch.float32,
            )
        else:
            a_e = a.index_select(0, token_idx)
            w13_f32 = gemm1_weights[le].float()
            s13_exp = _expand_scale_2d(gemm1_weights_scale[le].float())
            g1 = a_e.matmul((w13_f32 * s13_exp).t())

        w2_f32 = gemm2_weights[le].float()
        s2_exp = _expand_scale_2d(gemm2_weights_scale[le].float())
        w_tok = weights.index_select(0, token_idx)[:, ge]
        w2_t = (w2_f32 * s2_exp).t().contiguous()

        if cuda_mod is not None and g1.numel() >= _CUTE_MIN_ELEMS:
            cuda_mod.moe_post_cute(
                g1.contiguous(),
                w2_t,
                w_tok.contiguous(),
                token_idx.contiguous(),
                output,
            )
        else:
            x1 = g1[:, :I]
            x2 = g1[:, I:]
            mid = _fused_swiglu(x1, x2)
            out_e = mid.matmul(w2_t)
            output.index_add_(0, token_idx, out_e * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)

"""
FP8 Block-Scale Fused MoE — v6: _scaled_mm with per-tensor scale.

Strategy:
  - Convert block-scale FP8 weights to per-tensor-scale FP8 for _scaled_mm
  - For GEMM2: quantize SwiGLU mid output to FP8, then _scaled_mm
  - Fallback to lazy dequant f32 if _scaled_mm unavailable or inaccurate
  - All with lazy per-expert processing (only active experts)
"""

import torch

# ============================================================================
# Constants
# ============================================================================
H = 7168
I = 2048
E_GLOBAL = 256
E_LOCAL = 32
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
BLK = 128
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0


# ============================================================================
# Helpers
# ============================================================================

def _fast_expand_scale_2d(scale, blk, target_n, target_k):
    """Expand [N//blk, K//blk] scale to [N, K] via expand+reshape."""
    sn, sk = scale.shape
    out = scale.unsqueeze(1).expand(sn, blk, sk).reshape(sn * blk, sk)
    out = out.unsqueeze(2).expand(sn * blk, sk, blk).reshape(sn * blk, sk * blk)
    return out[:target_n, :target_k]


def _block_dequant_weight(w_fp8, w_scale, blk=BLK):
    """Dequant block-scale FP8 weight to f32."""
    w_f32 = w_fp8.float()
    s_f32 = w_scale.float()
    s_exp = _fast_expand_scale_2d(s_f32, blk, w_f32.shape[0], w_f32.shape[1])
    return w_f32 * s_exp


def _block_dequant_act(act_fp8, act_scale_th, blk=BLK):
    """Dequant block-scale FP8 activations to f32. act_scale_th: [T, H//blk]."""
    t, sh = act_scale_th.shape
    s_exp = act_scale_th.unsqueeze(2).expand(t, sh, blk).reshape(t, sh * blk)[:, :act_fp8.shape[1]]
    return act_fp8.float() * s_exp


def _quantize_to_fp8(tensor_f32):
    """Quantize f32 tensor to FP8 with per-tensor scale. Returns (fp8_tensor, scale)."""
    amax = tensor_f32.abs().amax()
    if amax == 0:
        scale = torch.ones(1, device=tensor_f32.device, dtype=torch.float32)
    else:
        scale = amax / FP8_MAX
    fp8_val = (tensor_f32 / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return fp8_val, scale


# ============================================================================
# GEMM mode detection
# ============================================================================
_gemm_mode = None  # "scaled_mm" or "f32"


def _detect_gemm_mode(device):
    """Test if _scaled_mm with per-tensor scalar scale works."""
    global _gemm_mode
    if _gemm_mode is not None:
        return

    try:
        a = torch.randn(4, 128, device=device)
        b = torch.randn(64, 128, device=device)
        a_fp8, a_scale = _quantize_to_fp8(a)
        b_fp8, b_scale = _quantize_to_fp8(b)
        result = torch._scaled_mm(
            a_fp8, b_fp8.t().contiguous(),
            scale_a=a_scale,
            scale_b=b_scale,
            out_dtype=torch.float32,
        )
        # Verify rough correctness
        ref = a.matmul(b.t())
        rel_err = (result - ref).abs().max() / (ref.abs().max() + 1e-6)
        if rel_err < 0.2:
            _gemm_mode = "scaled_mm"
        else:
            _gemm_mode = "f32"
    except Exception:
        _gemm_mode = "f32"


# ============================================================================
# Routing (identical to reference)
# ============================================================================

def _route(logits, bias, scaling_factor):
    T = logits.shape[0]
    s = 1.0 / (1.0 + torch.exp(-logits.float()))
    s_b = s + bias.float().reshape(-1)

    group_size = E_GLOBAL // N_GROUP
    grouped = s_b.view(T, N_GROUP, group_size)
    top2, _ = torch.topk(grouped, k=2, dim=2, largest=True, sorted=False)
    g_scores = top2.sum(dim=2)

    _, g_idx = torch.topk(g_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    g_mask = torch.zeros_like(g_scores)
    g_mask.scatter_(1, g_idx, 1.0)
    e_mask = g_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_GLOBAL)

    pruned = s_b.masked_fill(e_mask == 0, torch.finfo(torch.float32).min)
    _, topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    w = s * M
    w = (w / (w.sum(dim=1, keepdim=True) + 1e-20)) * scaling_factor
    return topk_idx, w


# ============================================================================
# Token permutation
# ============================================================================

def _build_expert_assignments(topk_idx, local_start, T, device):
    flat_tok = torch.arange(T, device=device).unsqueeze(1).expand(-1, TOP_K).reshape(-1)
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


# ============================================================================
# Entry Point
# ============================================================================

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
    T = routing_logits.shape[0]
    device = routing_logits.device
    local_start = int(local_expert_offset)

    _detect_gemm_mode(device)
    use_smm = (_gemm_mode == "scaled_mm")

    # 1) Routing
    topk_idx, weights = _route(routing_logits, routing_bias, routed_scaling_factor)

    # 2) Build expert assignments
    assignments = _build_expert_assignments(topk_idx, local_start, T, device)
    active_experts = [le for le in range(E_LOCAL) if assignments[le] is not None]

    if not active_experts:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    # 3) Dequant activations to f32 (once)
    hs_scale_th = hidden_states_scale.float().permute(1, 0).contiguous()  # [T, H//128]
    A = _block_dequant_act(hidden_states, hs_scale_th)  # [T, H] f32

    # 4) Per-expert compute
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    for le in active_experts:
        token_idx = assignments[le]
        ge = local_start + le
        num_tokens = token_idx.shape[0]

        A_e = A.index_select(0, token_idx)  # [Tk, H] f32

        if use_smm and num_tokens >= 4:
            # === FP8 _scaled_mm path ===
            # GEMM1: quantize A_e to FP8, dequant W13 to f32 then re-quantize per-tensor
            A_e_fp8, a_scale = _quantize_to_fp8(A_e)

            # Dequant W13 to f32, then re-quantize per-tensor
            W13_f32 = _block_dequant_weight(gemm1_weights[le], gemm1_weights_scale[le])
            W13_fp8, w13_scale = _quantize_to_fp8(W13_f32)

            G1 = torch._scaled_mm(
                A_e_fp8, W13_fp8.t().contiguous(),
                scale_a=a_scale,
                scale_b=w13_scale,
                out_dtype=torch.float32,
            )

            # SwiGLU
            X1 = G1[:, :I]
            X2 = G1[:, I:]
            mid = (X2 / (1.0 + torch.exp(-X2))) * X1

            # GEMM2: quantize mid to FP8
            mid_fp8, mid_scale = _quantize_to_fp8(mid)
            W2_f32 = _block_dequant_weight(gemm2_weights[le], gemm2_weights_scale[le])
            W2_fp8, w2_scale = _quantize_to_fp8(W2_f32)

            O = torch._scaled_mm(
                mid_fp8, W2_fp8.t().contiguous(),
                scale_a=mid_scale,
                scale_b=w2_scale,
                out_dtype=torch.float32,
            )
        else:
            # === f32 fallback path ===
            W13_f32 = _block_dequant_weight(gemm1_weights[le], gemm1_weights_scale[le])
            G1 = A_e.matmul(W13_f32.t())

            X1 = G1[:, :I]
            X2 = G1[:, I:]
            mid = (X2 / (1.0 + torch.exp(-X2))) * X1

            W2_f32 = _block_dequant_weight(gemm2_weights[le], gemm2_weights_scale[le])
            O = mid.matmul(W2_f32.t())

        w_tok = weights.index_select(0, token_idx)[:, ge]
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)

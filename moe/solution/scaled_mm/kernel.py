"""
FP8 Block-Scale Fused MoE — torch._scaled_mm optimized version.

Key optimization: Use torch._scaled_mm for FP8 GEMM on B200 (sm_100),
which supports block-scale FP8 Tensor Core operations natively.
This avoids the expensive dequant-to-f32 + f32-matmul path.

Falls back to dequant+matmul if _scaled_mm fails.
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

# ============================================================================
# FP8 GEMM strategies
# ============================================================================
_scaled_mm_mode = None  # None=untested, "block"=block-scale, "fallback"=dequant


def _probe_scaled_mm(act_fp8, act_scale, w_fp8, w_scale):
    """Probe which _scaled_mm mode works on this hardware."""
    global _scaled_mm_mode
    if _scaled_mm_mode is not None:
        return _scaled_mm_mode

    M, K = act_fp8.shape
    N = w_fp8.shape[0]

    # Try block-scale mode (B200 sm_100)
    try:
        w_t = w_fp8.t().contiguous()
        _ = torch._scaled_mm(
            act_fp8,
            w_t,
            scale_a=act_scale,
            scale_b=w_scale.t().contiguous(),
            out_dtype=torch.float32,
        )
        _scaled_mm_mode = "block"
        print("[FP8] Using block-scale torch._scaled_mm (native FP8 TC)")
        return _scaled_mm_mode
    except Exception as exc:
        print(f"[FP8] Block-scale _scaled_mm failed: {exc}")

    _scaled_mm_mode = "fallback"
    print("[FP8] Falling back to dequant + f32 matmul")
    return _scaled_mm_mode


def _fp8_gemm(act_fp8, act_scale, w_fp8, w_scale):
    """FP8 block-scale GEMM — auto-selects best path.

    act_fp8:   [M, K]        fp8_e4m3fn
    act_scale: [M, K//128]   f32
    w_fp8:     [N, K]        fp8_e4m3fn
    w_scale:   [N//128, K//128] f32
    Returns:   [M, N]        f32
    """
    if _scaled_mm_mode == "block":
        w_t = w_fp8.t().contiguous()
        return torch._scaled_mm(
            act_fp8,
            w_t,
            scale_a=act_scale,
            scale_b=w_scale.t().contiguous(),
            out_dtype=torch.float32,
        )

    # Fallback: dequant + f32 matmul
    M, K = act_fp8.shape
    N = w_fp8.shape[0]
    act_f32 = act_fp8.float()
    w_f32 = w_fp8.float()
    act_s = torch.repeat_interleave(act_scale, BLK, dim=1)[:, :K]
    w_s = torch.repeat_interleave(w_scale, BLK, dim=0)[:N]
    w_s = torch.repeat_interleave(w_s, BLK, dim=1)[:, :K]
    return torch.mm(act_f32 * act_s, (w_f32 * w_s).t())


def _dequant_to_fp8_with_scale(tensor_f32):
    """Quantize f32 tensor to fp8 with per-block scale for GEMM2 input.

    tensor_f32: [M, K] f32
    Returns: (tensor_fp8 [M, K], scale [M, K//128])
    """
    M, K = tensor_f32.shape
    num_blocks = (K + BLK - 1) // BLK

    # Pad K to multiple of BLK if needed
    if K % BLK != 0:
        pad = BLK - (K % BLK)
        tensor_f32 = torch.nn.functional.pad(tensor_f32, (0, pad))

    reshaped = tensor_f32.reshape(M, num_blocks, BLK)
    block_max = reshaped.abs().amax(dim=2).clamp(min=1e-12)

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = block_max / fp8_max

    scaled_blocks = reshaped / scale.unsqueeze(2)
    quantized = scaled_blocks.reshape(M, num_blocks * BLK)[:, :K]
    quantized = quantized.to(torch.float8_e4m3fn)

    return quantized, scale


# ============================================================================
# Routing (identical to baseline)
# ============================================================================

def _route(logits, bias, scaling_factor):
    T = logits.shape[0]
    s = torch.sigmoid(logits.float())
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

    sel = torch.zeros_like(s)
    sel.scatter_(1, topk_idx, 1.0)
    w = s * sel
    w = (w / (w.sum(dim=1, keepdim=True) + 1e-20)) * scaling_factor
    return topk_idx, w


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

    # 1) Routing
    topk_idx, weights = _route(routing_logits, routing_bias, routed_scaling_factor)

    # 2) Probe _scaled_mm support
    # hidden_states_scale: [H//128, T] -> need [T, H//128] for act_scale
    hs_scale_TH = hidden_states_scale.float().permute(1, 0).contiguous()  # [T, H//128]
    _probe_scaled_mm(
        hidden_states[:1],
        hs_scale_TH[:1],
        gemm1_weights[0],
        gemm1_weights_scale[0],
    )

    use_fp8_gemm = (_scaled_mm_mode == "block")

    # 3) If not using FP8 GEMM, dequant everything upfront (batch is faster)
    if not use_fp8_gemm:
        A_fp32 = hidden_states.float()
        A_scale_exp = (
            hs_scale_TH.unsqueeze(-1)
            .repeat(1, 1, BLK)
            .reshape(T, H)
            .contiguous()
        )
        A = A_fp32 * A_scale_exp

        W13_fp32 = gemm1_weights.float()
        S13 = gemm1_weights_scale.float()
        S13_exp = torch.repeat_interleave(S13, BLK, dim=1)
        S13_exp = torch.repeat_interleave(S13_exp, BLK, dim=2)
        W13 = W13_fp32 * S13_exp

        W2_fp32 = gemm2_weights.float()
        S2 = gemm2_weights_scale.float()
        S2_exp = torch.repeat_interleave(S2, BLK, dim=1)
        S2_exp = torch.repeat_interleave(S2_exp, BLK, dim=2)
        W2 = W2_fp32 * S2_exp

    # 4) Per-expert compute
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    for le in range(E_LOCAL):
        ge = local_start + le
        if ge < 0 or ge >= E_GLOBAL:
            continue

        sel_mask = (topk_idx == ge).any(dim=1)
        if not sel_mask.any():
            continue

        token_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)

        if use_fp8_gemm:
            # FP8 GEMM path: keep data in fp8
            hs_e = hidden_states.index_select(0, token_idx)     # [Tk, H] fp8
            hs_s = hs_scale_TH.index_select(0, token_idx)       # [Tk, H//128]

            # GEMM1: [Tk, H] x [4096, H]^T -> [Tk, 4096]
            G1 = _fp8_gemm(hs_e, hs_s, gemm1_weights[le], gemm1_weights_scale[le])

            # SwiGLU: silu(X2) * X1
            X1 = G1[:, :I]
            X2 = G1[:, I:]
            silu_X2 = X2 / (1.0 + torch.exp(-X2))
            mid = silu_X2 * X1  # [Tk, I] f32

            # GEMM2: quantize mid to fp8, then FP8 GEMM
            mid_fp8, mid_scale = _dequant_to_fp8_with_scale(mid)
            O = _fp8_gemm(mid_fp8, mid_scale, gemm2_weights[le], gemm2_weights_scale[le])
        else:
            # Fallback: dequant + f32 matmul
            A_e = A.index_select(0, token_idx)
            G1 = A_e.matmul(W13[le].t())

            X1 = G1[:, :I]
            X2 = G1[:, I:]
            silu_X2 = X2 / (1.0 + torch.exp(-X2))
            mid = silu_X2 * X1

            O = mid.matmul(W2[le].t())

        w_tok = weights.index_select(0, token_idx)[:, ge]
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)

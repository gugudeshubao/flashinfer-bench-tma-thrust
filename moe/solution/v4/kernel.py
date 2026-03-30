"""
FP8 Block-Scale Fused MoE — High-Performance v4.

Builds on v2 (1.59x) with bf16 matmul for ~2x GEMM speedup:
  1. Lazy per-expert weight dequant (only active experts)
  2. Dequant to bf16 instead of f32 (2x less memory, faster TC)
  3. bf16 matmul (B200 bf16 TC throughput >> f32)
  4. SwiGLU in f32 for precision, then cast back to bf16
  5. Final accumulation in f32 for correctness
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
# Token permutation (optimized: single pass)
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
        s, t = offsets[e].item(), offsets[e + 1].item()
        assignments.append(sorted_tok[s:t] if s < t else None)
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

    # 1) Routing
    topk_idx, weights = _route(routing_logits, routing_bias, routed_scaling_factor)

    # 2) Build expert assignments
    assignments = _build_expert_assignments(topk_idx, local_start, T, device)
    active_experts = [le for le in range(E_LOCAL) if assignments[le] is not None]

    if not active_experts:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    # 3) Dequant hidden_states to bf16
    hs_scale_TH = hidden_states_scale.float().permute(1, 0).contiguous()  # [T, H//128]
    A_f32 = hidden_states.float()
    A_scale_exp = (
        hs_scale_TH.unsqueeze(-1)
        .repeat(1, 1, BLK)
        .reshape(T, H)
        .contiguous()
    )
    A_bf16 = (A_f32 * A_scale_exp).to(torch.bfloat16)  # [T, H] bf16

    # 4) Per-expert compute with lazy dequant to bf16
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    for le in active_experts:
        token_idx = assignments[le]
        ge = local_start + le

        A_e = A_bf16.index_select(0, token_idx)  # [Tk, H] bf16

        # GEMM1: lazy dequant W13 to bf16
        w13_f32 = gemm1_weights[le].float()
        s13 = gemm1_weights_scale[le].float()
        s13_n = torch.repeat_interleave(s13, BLK, dim=0)
        s13_nk = torch.repeat_interleave(s13_n, BLK, dim=1)
        W13_bf16 = (w13_f32 * s13_nk).to(torch.bfloat16)  # [4096, 7168] bf16

        G1 = torch.mm(A_e, W13_bf16.t())  # [Tk, 4096] bf16

        # SwiGLU in f32 for precision: silu(X2) * X1
        G1_f32 = G1.float()
        X1 = G1_f32[:, :I]
        X2 = G1_f32[:, I:]
        mid = (X2 / (1.0 + torch.exp(-X2))) * X1  # [Tk, I] f32
        mid_bf16 = mid.to(torch.bfloat16)

        # GEMM2: lazy dequant W2 to bf16
        w2_f32 = gemm2_weights[le].float()
        s2 = gemm2_weights_scale[le].float()
        s2_n = torch.repeat_interleave(s2, BLK, dim=0)
        s2_nk = torch.repeat_interleave(s2_n, BLK, dim=1)
        W2_bf16 = (w2_f32 * s2_nk).to(torch.bfloat16)  # [7168, 2048] bf16

        O = torch.mm(mid_bf16, W2_bf16.t()).float()  # [Tk, 7168] f32

        w_tok = weights.index_select(0, token_idx)[:, ge]
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)

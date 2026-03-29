"""
FP8 Block-Scale Fused MoE Kernel for DeepSeek-V3/R1.

Implements the full MoE pipeline:
  1. DeepSeek no-aux routing (sigmoid + group selection + top-k)
  2. Token permutation (sort tokens by expert)
  3. FP8 block-scale GEMM1 (gate + up projection) with Triton
  4. SwiGLU activation
  5. FP8 block-scale GEMM2 (down projection) with Triton
  6. Weighted accumulation back to original token order

Target: NVIDIA B200 (sm_100), FP8 Tensor Core
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Constants
# ============================================================================
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
GEMM1_OUT_SIZE = 4096  # 2 * INTERMEDIATE_SIZE
NUM_EXPERTS_GLOBAL = 256
NUM_LOCAL_EXPERTS = 32
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
BLOCK_SCALE_SIZE = 128


# ============================================================================
# Triton Kernels
# ============================================================================

@triton.jit
def _fp8_gemm_kernel(
    # Pointers
    A_ptr, A_scale_ptr,
    B_ptr, B_scale_ptr,
    C_ptr,
    # Strides for A: [M, K]
    stride_am, stride_ak,
    # Strides for A_scale: [M, K // BLOCK]
    stride_asm, stride_ask,
    # Strides for B: [N, K] (row-major, transposed during compute)
    stride_bn, stride_bk,
    # Strides for B_scale: [N // BLOCK, K // BLOCK]
    stride_bsn, stride_bsk,
    # Strides for C: [M, N]
    stride_cm, stride_cn,
    # Dimensions
    M, N, K: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """FP8 block-scale GEMM: C = dequant(A) @ dequant(B).T

    A: [M, K] fp8, A_scale: [M, K//128] f32
    B: [N, K] fp8, B_scale: [N//128, K//128] f32
    C: [M, N] f32
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Scale block indices for B (N dimension)
    b_scale_n_idx = pid_n * BLOCK_N // BLOCK_SCALE_SIZE

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: [BLOCK_N, BLOCK_K] -> we want [BLOCK_K, BLOCK_N] for matmul
        b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b_ptrs = B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Load A scales: [BLOCK_M]
        a_scale_k_idx = k_start // BLOCK_SCALE_SIZE
        a_scale_ptrs = A_scale_ptr + offs_m * stride_asm + a_scale_k_idx * stride_ask
        a_scale = tl.load(a_scale_ptrs, mask=offs_m < M, other=1.0)

        # Load B scales: [BLOCK_N // BLOCK_SCALE_SIZE] - one per N-block
        # B_scale: [N//128, K//128], index [n_block, k_block]
        b_scale_k_idx = k_start // BLOCK_SCALE_SIZE
        # For each BLOCK_N, we may span multiple scale blocks
        offs_n_scale = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        b_scale_n_indices = offs_n_scale // BLOCK_SCALE_SIZE
        b_scale_ptrs = B_scale_ptr + b_scale_n_indices * stride_bsn + b_scale_k_idx * stride_bsk
        b_scale = tl.load(b_scale_ptrs, mask=offs_n_scale < N, other=1.0)

        # Scale: combined scale = a_scale * b_scale
        combined_scale = a_scale[:, None] * b_scale[None, :]

        # FP8 matmul: a @ b.T
        # a: [BLOCK_M, BLOCK_K], b: [BLOCK_N, BLOCK_K]
        # We need a @ b^T = [BLOCK_M, BLOCK_N]
        ab = tl.dot(a, tl.trans(b))

        acc += ab.to(tl.float32) * combined_scale

    # Store C
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _silu_mul_kernel(
    gate_ptr, up_ptr, out_ptr,
    N,
    stride,
    BLOCK: tl.constexpr,
):
    """SwiGLU: out = silu(gate) * up"""
    pid = tl.program_id(0)
    row = tl.program_id(1)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    gate_ptrs = gate_ptr + row * stride + offs
    up_ptrs = up_ptr + row * stride + offs
    out_ptrs = out_ptr + row * stride + offs

    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)

    # silu(x) = x * sigmoid(x)
    silu_gate = gate * tl.sigmoid(gate)
    result = silu_gate * up

    tl.store(out_ptrs, result, mask=mask)


# ============================================================================
# DeepSeek-V3 No-Aux Routing (PyTorch)
# ============================================================================

def deepseek_routing(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    routed_scaling_factor: float,
) -> tuple:
    """DeepSeek-V3 no-aux routing.

    Returns:
        topk_idx: [T, TOP_K] - selected expert indices
        topk_weights: [T, TOP_K] - normalized routing weights
    """
    seq_len = routing_logits.shape[0]
    device = routing_logits.device

    logits = routing_logits.float()
    bias = routing_bias.float()

    # Sigmoid scores
    scores = torch.sigmoid(logits)
    scores_with_bias = scores + bias

    # Group selection: split into N_GROUP groups
    group_size = NUM_EXPERTS_GLOBAL // N_GROUP  # 32
    scores_grouped = scores_with_bias.view(seq_len, N_GROUP, group_size)

    # Top-2 per group -> group scores
    top2_vals, _ = torch.topk(scores_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)  # [T, N_GROUP]

    # Select top TOPK_GROUP groups
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)

    # Expand group mask to expert mask
    score_mask = group_mask.unsqueeze(2).expand(seq_len, N_GROUP, group_size).reshape(seq_len, NUM_EXPERTS_GLOBAL)

    # Global top-k within kept groups
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = scores_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    # Weights from original scores (without bias), normalized
    expert_mask = torch.zeros_like(scores)
    expert_mask.scatter_(1, topk_idx, 1.0)
    weights = scores * expert_mask
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    # Gather per-token weights for selected experts
    topk_weights = torch.gather(weights, 1, topk_idx)  # [T, TOP_K]

    return topk_idx, topk_weights


# ============================================================================
# Token Permutation
# ============================================================================

def permute_tokens(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_offset: int,
    num_tokens: int,
):
    """Sort tokens by expert for efficient batched GEMM.

    Returns:
        sorted_token_ids: [total_selected] - token indices sorted by expert
        sorted_weights: [total_selected] - corresponding weights
        expert_offsets: [NUM_LOCAL_EXPERTS + 1] - start/end offsets per expert
    """
    device = topk_idx.device

    # Flatten: each token appears TOP_K times
    flat_token_ids = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, TOP_K).reshape(-1)
    flat_expert_ids = topk_idx.reshape(-1)
    flat_weights = topk_weights.reshape(-1)

    # Filter to local experts only
    local_mask = (flat_expert_ids >= local_expert_offset) & (flat_expert_ids < local_expert_offset + NUM_LOCAL_EXPERTS)
    local_token_ids = flat_token_ids[local_mask]
    local_expert_ids = flat_expert_ids[local_mask] - local_expert_offset
    local_weights = flat_weights[local_mask]

    if local_token_ids.numel() == 0:
        empty_offsets = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int64, device=device)
        return local_token_ids, local_weights, empty_offsets

    # Sort by expert id for contiguous access
    sort_indices = torch.argsort(local_expert_ids, stable=True)
    sorted_token_ids = local_token_ids[sort_indices]
    sorted_expert_ids = local_expert_ids[sort_indices]
    sorted_weights = local_weights[sort_indices]

    # Compute expert offsets using bincount
    expert_counts = torch.bincount(sorted_expert_ids.int(), minlength=NUM_LOCAL_EXPERTS)
    expert_offsets = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int64, device=device)
    torch.cumsum(expert_counts, dim=0, out=expert_offsets[1:])

    return sorted_token_ids, sorted_weights, expert_offsets


# ============================================================================
# FP8 Block-Scale GEMM Wrapper
# ============================================================================

def fp8_block_scale_gemm(
    activations: torch.Tensor,       # [M, K] fp8 or f32
    act_scale: torch.Tensor,         # [M, K//128] f32
    weights: torch.Tensor,           # [N, K] fp8
    weight_scale: torch.Tensor,      # [N//128, K//128] f32
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """FP8 block-scale GEMM: C = dequant(A, A_scale) @ dequant(B, B_scale).T

    Uses Triton kernel for the matmul with integrated dequantization.
    Falls back to PyTorch for correctness baseline.
    """
    M, K = activations.shape
    N = weights.shape[0]

    # For now, use PyTorch dequant + matmul as baseline
    # This will be replaced with fused Triton kernel for performance
    act_f32 = activations.float()
    w_f32 = weights.float()

    # Dequantize activations: [M, K]
    # act_scale: [M, K//128] -> expand to [M, K]
    act_scale_expanded = act_scale.unsqueeze(-1).expand(-1, -1, BLOCK_SCALE_SIZE).reshape(M, -1)[:, :K]
    act_dequant = act_f32 * act_scale_expanded

    # Dequantize weights: [N, K]
    # weight_scale: [N//128, K//128] -> expand to [N, K]
    w_scale_expanded = weight_scale.unsqueeze(-1).expand(-1, -1, BLOCK_SCALE_SIZE)
    w_scale_expanded = w_scale_expanded.reshape(weight_scale.shape[0], -1)[:, :K // BLOCK_SCALE_SIZE * BLOCK_SCALE_SIZE]
    w_scale_expanded = weight_scale.unsqueeze(1).expand(-1, BLOCK_SCALE_SIZE, -1).reshape(-1, weight_scale.shape[-1])[:N, :]
    w_scale_expanded = w_scale_expanded.unsqueeze(-1).expand(-1, -1, BLOCK_SCALE_SIZE).reshape(N, -1)[:, :K]
    w_dequant = w_f32 * w_scale_expanded

    # Matmul
    output = torch.matmul(act_dequant, w_dequant.t())

    return output.to(output_dtype)


def fp8_block_scale_gemm_torch(
    activations: torch.Tensor,       # [M, K] fp8
    act_scale: torch.Tensor,         # [scale_k, M] f32 (transposed layout for hidden_states)
    weights: torch.Tensor,           # [N, K] fp8
    weight_scale: torch.Tensor,      # [N//128, K//128] f32
    act_scale_transposed: bool = False,
) -> torch.Tensor:
    """FP8 block-scale GEMM using PyTorch ops.

    Handles the transposed scale layout used by hidden_states_scale.
    """
    M, K = activations.shape
    N = weights.shape[0]

    # Dequantize activations
    act_f32 = activations.float()
    if act_scale_transposed:
        # act_scale: [K//128, M] -> [M, K//128]
        a_scale = act_scale.t().contiguous()
    else:
        a_scale = act_scale

    a_scale_expanded = a_scale.unsqueeze(-1).repeat(1, 1, BLOCK_SCALE_SIZE).reshape(M, -1)[:, :K]
    act_dequant = act_f32 * a_scale_expanded

    # Dequantize weights
    w_f32 = weights.float()
    # weight_scale: [N//128, K//128]
    w_scale_n = weight_scale.unsqueeze(1).repeat(1, BLOCK_SCALE_SIZE, 1).reshape(-1, weight_scale.shape[-1])[:N, :]
    w_scale_nk = w_scale_n.unsqueeze(-1).repeat(1, 1, BLOCK_SCALE_SIZE).reshape(N, -1)[:, :K]
    w_dequant = w_f32 * w_scale_nk

    return torch.matmul(act_dequant, w_dequant.t())


# ============================================================================
# Main Kernel Entry Point
# ============================================================================

def kernel(
    routing_logits: torch.Tensor,       # [seq_len, 256] f32
    routing_bias: torch.Tensor,         # [256] bf16
    hidden_states: torch.Tensor,        # [seq_len, 7168] fp8
    hidden_states_scale: torch.Tensor,  # [56, seq_len] f32
    gemm1_weights: torch.Tensor,        # [32, 4096, 7168] fp8
    gemm1_weights_scale: torch.Tensor,  # [32, 32, 56] f32
    gemm2_weights: torch.Tensor,        # [32, 7168, 2048] fp8
    gemm2_weights_scale: torch.Tensor,  # [32, 56, 16] f32
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """FP8 Block-Scale Fused MoE for DeepSeek-V3/R1.

    Pipeline:
      1. DeepSeek no-aux routing -> topk experts + weights
      2. Token permutation (sort by expert)
      3. Per-expert: GEMM1 -> SwiGLU -> GEMM2
      4. Weighted accumulation
    """
    seq_len = routing_logits.shape[0]
    device = routing_logits.device

    # Step 1: Routing
    topk_idx, topk_weights = deepseek_routing(
        routing_logits, routing_bias, routed_scaling_factor
    )

    # Step 2: Token permutation
    sorted_token_ids, sorted_weights, expert_offsets = permute_tokens(
        topk_idx, topk_weights, local_expert_offset, seq_len
    )

    # Step 3: Per-expert compute
    output = torch.zeros((seq_len, HIDDEN_SIZE), dtype=torch.float32, device=device)

    if sorted_token_ids.numel() == 0:
        return output.to(torch.bfloat16)

    # Dequantize hidden_states once: [seq_len, 7168]
    hs_f32 = hidden_states.float()
    # hidden_states_scale: [56, seq_len] -> [seq_len, 56]
    hs_scale = hidden_states_scale.t().contiguous()
    hs_scale_expanded = hs_scale.unsqueeze(-1).repeat(1, 1, BLOCK_SCALE_SIZE).reshape(seq_len, -1)[:, :HIDDEN_SIZE]
    hidden_dequant = hs_f32 * hs_scale_expanded  # [seq_len, 7168] f32

    for expert_idx in range(NUM_LOCAL_EXPERTS):
        start = expert_offsets[expert_idx].item()
        end = expert_offsets[expert_idx + 1].item()

        if start == end:
            continue

        # Gather tokens for this expert
        token_ids = sorted_token_ids[start:end]
        expert_weights = sorted_weights[start:end]
        num_expert_tokens = end - start

        # Gather dequantized activations: [num_expert_tokens, 7168]
        expert_input = hidden_dequant[token_ids]

        # GEMM1: [num_expert_tokens, 7168] @ [7168, 4096] -> [num_expert_tokens, 4096]
        # gemm1_weights[expert_idx]: [4096, 7168] fp8
        # gemm1_weights_scale[expert_idx]: [32, 56] f32
        w1 = gemm1_weights[expert_idx].float()  # [4096, 7168]
        w1_scale = gemm1_weights_scale[expert_idx]  # [32, 56]

        # Dequantize W1: [4096, 7168]
        w1_scale_n = w1_scale.unsqueeze(1).repeat(1, BLOCK_SCALE_SIZE, 1).reshape(-1, w1_scale.shape[-1])[:GEMM1_OUT_SIZE, :]
        w1_scale_nk = w1_scale_n.unsqueeze(-1).repeat(1, 1, BLOCK_SCALE_SIZE).reshape(GEMM1_OUT_SIZE, -1)[:, :HIDDEN_SIZE]
        w1_dequant = w1 * w1_scale_nk

        gemm1_out = torch.matmul(expert_input, w1_dequant.t())  # [num_expert_tokens, 4096]

        # SwiGLU: split into gate and up
        gate = gemm1_out[:, :INTERMEDIATE_SIZE]   # [num_expert_tokens, 2048]
        up = gemm1_out[:, INTERMEDIATE_SIZE:]      # [num_expert_tokens, 2048]
        silu_gate = gate * torch.sigmoid(gate)
        intermediate = silu_gate * up              # [num_expert_tokens, 2048]

        # GEMM2: [num_expert_tokens, 2048] @ [2048, 7168] -> [num_expert_tokens, 7168]
        w2 = gemm2_weights[expert_idx].float()  # [7168, 2048]
        w2_scale = gemm2_weights_scale[expert_idx]  # [56, 16]

        # Dequantize W2: [7168, 2048]
        w2_scale_n = w2_scale.unsqueeze(1).repeat(1, BLOCK_SCALE_SIZE, 1).reshape(-1, w2_scale.shape[-1])[:HIDDEN_SIZE, :]
        w2_scale_nk = w2_scale_n.unsqueeze(-1).repeat(1, 1, BLOCK_SCALE_SIZE).reshape(HIDDEN_SIZE, -1)[:, :INTERMEDIATE_SIZE]
        w2_dequant = w2 * w2_scale_nk

        expert_output = torch.matmul(intermediate, w2_dequant.t())  # [num_expert_tokens, 7168]

        # Weighted accumulation
        weighted_output = expert_output * expert_weights.unsqueeze(1)
        output.index_add_(0, token_ids, weighted_output)

    return output.to(torch.bfloat16)

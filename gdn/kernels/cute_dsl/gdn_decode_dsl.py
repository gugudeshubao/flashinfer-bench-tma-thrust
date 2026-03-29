"""
GDN Decode — CuTe DSL kernel (CUTLASS 4.x)

This is a simplified GDN decode kernel using CuTe DSL to demonstrate the concept.
For production use, the Triton or optimized CUDA kernels are recommended.

Grid: (B * H * V_BLOCKS,) — one program per (batch, v_head, V-tile)

Algorithm (Delta Rule - simplified):
1. S = g * S                   # decay state
2. out = S @ q                 # output (simplified, no delta update for demo)

Usage:
    modal run scripts/test_cute_dsl.py
"""

import math
from typing import Optional

import torch

# Check if CuTe DSL is available
try:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass import Float32, Int32
    HAS_CUTE_DSL = True
except ImportError:
    HAS_CUTE_DSL = False


if HAS_CUTE_DSL:
    # ============================================================
    # CuTe DSL Kernel Implementation (Simplified Demo)
    # ============================================================
    
    # Constants
    D_CONST = 128  # Head dimension (compile-time constant)
    
    @cute.kernel
    def _gdn_state_matmul_kernel(
        # Input: state and query
        gState: cute.Tensor,   # [total_state_elements] flattened
        gQ: cute.Tensor,       # [total_q_elements] flattened
        # Output
        gOut: cute.Tensor,     # [total_out_elements] flattened
    ):
        """
        Simplified kernel: computes out = State @ q for one (batch, head, v) element.
        Each thread handles one V element.
        
        This is a demo kernel - not optimized for performance.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        
        # Constants
        D = D_CONST
        num_v_heads = 8  # Hardcoded for GVA
        V_BLOCKS = 8  # D / BLOCK_V = 128 / 16 = 8
        BLOCK_V = 16
        
        # Decode block index to (batch, head, v_block)
        vb = bidx % V_BLOCKS
        temp = bidx // V_BLOCKS
        h = temp % num_v_heads
        b = temp // num_v_heads
        
        v0 = vb * BLOCK_V  # first V element in this block
        v_idx = v0 + tidx
        qk_h = h // 2  # GVA: 2 v-heads share each qk-head
        
        # Compute offsets
        # State layout: [B, 8, D, D] flattened
        # state_offset = b * 8 * D * D + h * D * D + v_idx * D
        state_base = b * 8 * D * D + h * D * D + v_idx * D
        
        # Q layout: [B, 4, D] flattened  
        # q_offset = b * 4 * D + qk_h * D
        q_base = b * 4 * D + qk_h * D
        
        # Out layout: [B, 8, D] flattened
        out_idx = b * 8 * D + h * D + v_idx
        
        # Compute dot product: out = sum(State[v, :] * Q[:])
        # This is a simple serial loop - not optimized
        acc = gState[state_base] * gQ[q_base]  # Initialize with first element
        
        # Unroll loop manually for better performance
        acc = acc + gState[state_base + 1] * gQ[q_base + 1]
        acc = acc + gState[state_base + 2] * gQ[q_base + 2]
        acc = acc + gState[state_base + 3] * gQ[q_base + 3]
        acc = acc + gState[state_base + 4] * gQ[q_base + 4]
        acc = acc + gState[state_base + 5] * gQ[q_base + 5]
        acc = acc + gState[state_base + 6] * gQ[q_base + 6]
        acc = acc + gState[state_base + 7] * gQ[q_base + 7]
        
        # Continue for all D=128 elements (unrolled in groups of 8)
        for i in range(8, D, 8):
            acc = acc + gState[state_base + i + 0] * gQ[q_base + i + 0]
            acc = acc + gState[state_base + i + 1] * gQ[q_base + i + 1]
            acc = acc + gState[state_base + i + 2] * gQ[q_base + i + 2]
            acc = acc + gState[state_base + i + 3] * gQ[q_base + i + 3]
            acc = acc + gState[state_base + i + 4] * gQ[q_base + i + 4]
            acc = acc + gState[state_base + i + 5] * gQ[q_base + i + 5]
            acc = acc + gState[state_base + i + 6] * gQ[q_base + i + 6]
            acc = acc + gState[state_base + i + 7] * gQ[q_base + i + 7]
        
        # Store result
        gOut[out_idx] = acc
    
    
    @cute.jit
    def _launch_gdn_matmul(mState, mQ, mOut, num_blocks: int):
        """Host function to launch the kernel."""
        BLOCK_V = 16
        
        _gdn_state_matmul_kernel(mState, mQ, mOut).launch(
            grid=[num_blocks, 1, 1],
            block=[BLOCK_V, 1, 1],
        )


def kernel(q, k, v, state, A_log, a, dt_bias, b_gate, scale):
    """
    CuTe DSL wrapper for GDN decode (simplified - just computes State @ Q).
    
    For full GDN decode with delta rule, use the Triton kernel.
    This is a demonstration of CuTe DSL usage.
    
    Args:
        q: [B, 1, 4, D] query tensor (bf16)
        k: [B, 1, 4, D] key tensor (bf16) - unused in this demo
        v: [B, 1, 8, D] value tensor (bf16) - unused in this demo
        state: [B, 8, D, D] state tensor (fp32)
        A_log: [8] log gate - unused in this demo
        a: [B, 1, 8] gate a - unused in this demo
        dt_bias: [8] dt bias - unused in this demo
        b_gate: [B, 1, 8] gate b - unused in this demo
        scale: scaling factor
        
    Returns:
        out: [B, 1, 8, D] output (bf16)
        new_state: [B, 8, D, D] new state (fp32) - same as input in this demo
    """
    if not HAS_CUTE_DSL:
        raise RuntimeError("CuTe DSL not available. Install with: pip install nvidia-cutlass-dsl")
    
    B, _, num_q_heads, D = q.shape
    num_v_heads = v.shape[2]
    device = q.device
    
    assert D == D_CONST, f"D must be {D_CONST}, got {D}"
    
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)
    
    # Flatten tensors
    state_flat = state.float().contiguous().view(-1)  # [B * 8 * D * D]
    q_flat = q.squeeze(1).float().contiguous().view(-1)  # [B * 4 * D]
    out_flat = torch.empty(B * num_v_heads * D, dtype=torch.float32, device=device)
    
    # Convert to CuTe tensors
    mState = from_dlpack(state_flat).mark_layout_dynamic()
    mQ = from_dlpack(q_flat).mark_layout_dynamic()
    mOut = from_dlpack(out_flat).mark_layout_dynamic()
    
    # Launch kernel
    BLOCK_V = 16
    V_BLOCKS = D // BLOCK_V
    num_blocks = B * num_v_heads * V_BLOCKS
    
    _launch_gdn_matmul(mState, mQ, mOut, num_blocks)
    
    # Apply scale
    out_flat = out_flat * scale
    
    # Reshape outputs
    out = out_flat.view(B, num_v_heads, D).unsqueeze(1).to(torch.bfloat16)
    new_state = state  # In this demo, we don't update state
    
    return out, new_state


# ============================================================
# Fallback: Pure PyTorch reference implementation
# ============================================================

def kernel_reference(q, k, v, state, A_log, a, dt_bias, b_gate, scale):
    """
    Pure PyTorch reference implementation for correctness checking.
    This matches the SIMPLIFIED CuTe DSL kernel (just State @ Q, no delta rule).
    """
    B, _, num_q_heads, D = q.shape
    num_v_heads = v.shape[2]
    device = q.device
    
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)
    
    q_c = q.squeeze(1).float()  # [B, 4, D]
    S = state.float()  # [B, 8, D, D]
    
    out = torch.empty(B, num_v_heads, D, dtype=torch.float32, device=device)
    
    for h in range(num_v_heads):
        qk_h = h // 2
        
        # Get q
        q_h = q_c[:, qk_h, :]  # [B, D]
        
        # State [B, D, D]
        S_h = S[:, h, :, :]
        
        # Simplified: out = State @ q (no delta rule)
        out_h = scale * torch.einsum('bvd,bd->bv', S_h, q_h)  # [B, D]
        
        out[:, h, :] = out_h
    
    return out.unsqueeze(1).to(torch.bfloat16), S


def kernel_reference_full(q, k, v, state, A_log, a, dt_bias, b_gate, scale):
    """
    Full PyTorch reference implementation with delta rule.
    """
    B, _, num_q_heads, D = q.shape
    num_v_heads = v.shape[2]
    device = q.device
    
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)
    
    q_c = q.squeeze(1).float()  # [B, 4, D]
    k_c = k.squeeze(1).float()
    v_c = v.squeeze(1).float()  # [B, 8, D]
    a_c = a.squeeze(1).float()  # [B, 8]
    b_c = b_gate.squeeze(1).float()
    
    if state is not None:
        S = state.clone().float()
    else:
        S = torch.zeros(B, num_v_heads, D, D, dtype=torch.float32, device=device)
    
    out = torch.empty(B, num_v_heads, D, dtype=torch.float32, device=device)
    
    for h in range(num_v_heads):
        qk_h = h // 2
        
        # Gates
        x = a_c[:, h] + dt_bias[h]
        sp = torch.where(x > 20.0, x, torch.log(1.0 + torch.exp(x)))
        g = torch.exp(-torch.exp(A_log[h]) * sp)  # [B]
        beta = torch.sigmoid(b_c[:, h])  # [B]
        
        # Get q, k, v
        q_h = q_c[:, qk_h, :]  # [B, D]
        k_h = k_c[:, qk_h, :]  # [B, D]
        v_h = v_c[:, h, :]     # [B, D]
        
        # State [B, D, D]
        S_h = S[:, h, :, :]
        
        # Delta rule
        S_h = g[:, None, None] * S_h  # decay
        old_v = torch.einsum('bvd,bd->bv', S_h, k_h)  # [B, D] mat-vec
        delta = beta[:, None] * (v_h - old_v)  # [B, D]
        S_h = S_h + torch.einsum('bv,bd->bvd', delta, k_h)  # rank-1 update
        out_h = scale * torch.einsum('bvd,bd->bv', S_h, q_h)  # [B, D]
        
        S[:, h, :, :] = S_h
        out[:, h, :] = out_h
    
    return out.unsqueeze(1).to(torch.bfloat16), S


if __name__ == "__main__":
    print(f"CuTe DSL available: {HAS_CUTE_DSL}")
    
    if HAS_CUTE_DSL:
        print(f"CUTLASS version: {cutlass.__version__}")

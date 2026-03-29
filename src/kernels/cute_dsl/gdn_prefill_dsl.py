"""
GDN Prefill — CuTe DSL kernel (CUTLASS 4.x, MLIR Backend)

Prefill processes multiple tokens per sequence, computing:
  1. S = g * S           (decay state)
  2. old_v = S @ k       (mat-vec)
  3. delta = beta * (v - old_v)
  4. S = S + delta ⊗ k   (rank-1 update)
  5. out = scale * S @ q (mat-vec)

Key Optimization: Chunk-based processing
  - Process CHUNK_SIZE tokens together
  - Arithmetic Intensity: C FLOP/byte (compute-bound for C=8!)

Compilation Pipeline:
    Python DSL → MLIR → LLVM → PTX → SASS

Grid: (N=num_seqs, H=8, V_BLOCKS)

Usage:
    modal run scripts/bench_prefill_all.py
"""

import math
from typing import Optional

import torch

# Check if CuTe DSL is available
try:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    HAS_CUTE_DSL = True
except ImportError:
    HAS_CUTE_DSL = False


# ============================================================
# Constants
# ============================================================
D_CONST = 128
BLOCK_V = 16
NUM_V_HEADS = 8
NUM_Q_HEADS = 4
THREADS_PER_BLOCK = 128


if HAS_CUTE_DSL:
    # ============================================================
    # CuTe DSL Prefill Kernel
    # ============================================================
    
    @cute.kernel
    def _gdn_prefill_kernel_dsl(
        # Inputs (flattened)
        gQ: cute.Tensor,        # [T * 4 * D]
        gK: cute.Tensor,        # [T * 4 * D]
        gV: cute.Tensor,        # [T * 8 * D]
        gState: cute.Tensor,    # [N * 8 * D * D]
        gA_log: cute.Tensor,    # [8]
        gA: cute.Tensor,        # [T * 8]
        gDtBias: cute.Tensor,   # [8]
        gB_gate: cute.Tensor,   # [T * 8]
        # Sequence info
        gCuSeqlens: cute.Tensor,  # [N+1]
        # Outputs
        gOut: cute.Tensor,      # [T * 8 * D]
        gNewState: cute.Tensor, # [N * 8 * D * D]
        # Params
        scale: float,
    ):
        """
        CuTe DSL Prefill kernel (simplified demo).
        
        Processes one (seq, v_head, v_block) per block.
        Full implementation would include chunking for compute density.
        """
        tid = cute.arch.thread_idx()[0]
        n = cute.arch.block_idx()[0]      # sequence
        h = cute.arch.block_idx()[1]      # v_head
        vb = cute.arch.block_idx()[2]     # v_block
        
        D = D_CONST
        v0 = vb * BLOCK_V
        qk_h = h // 2  # GVA
        
        # Get sequence bounds
        t_start = gCuSeqlens[n]
        t_end = gCuSeqlens[n + 1]
        
        # Head constants
        alog = gA_log[h]
        dt_val = gDtBias[h]
        
        # State base offset: [N, 8, D, D] flattened
        state_base = n * 8 * D * D + h * D * D + v0 * D
        
        # Output base offset: [T, 8, D] flattened
        # Note: output indexed by global token position
        
        # Process tokens (simplified: no chunking in this demo)
        v_idx = tid  # Each thread handles one V element
        
        if v_idx < BLOCK_V:
            # Load initial state row
            state_row_base = state_base + v_idx * D
            
            # Initialize state in registers
            s_vals = [0.0] * D
            for d in range(D):
                s_vals[d] = gState[state_row_base + d]
            
            # Process each token
            for t in range(t_start, t_end):
                # Compute gates
                a_val = gA[t * 8 + h]
                b_val = gB_gate[t * 8 + h]
                
                x = a_val + dt_val
                sp = x if x > 20.0 else cute.log(1.0 + cute.exp(x))
                g = cute.exp(-cute.exp(alog) * sp)
                beta = 1.0 / (1.0 + cute.exp(-b_val))
                
                # Load Q, K, V for this token
                qk_base = t * 4 * D + qk_h * D
                v_base = t * 8 * D + h * D + v0
                
                v_elem = gV[v_base + v_idx]
                
                # 1. Decay state and compute old_v
                old_v = 0.0
                for d in range(D):
                    s_vals[d] = g * s_vals[d]
                    k_val = gK[qk_base + d]
                    old_v = old_v + s_vals[d] * k_val
                
                # 2. Delta rule
                delta = beta * (v_elem - old_v)
                
                # 3. Update state and compute output
                out_val = 0.0
                for d in range(D):
                    k_val = gK[qk_base + d]
                    q_val = gQ[qk_base + d]
                    s_vals[d] = s_vals[d] + delta * k_val
                    out_val = out_val + s_vals[d] * q_val
                
                # Store output
                out_base = t * 8 * D + h * D + v0
                gOut[out_base + v_idx] = scale * out_val
            
            # Store final state
            for d in range(D):
                gNewState[state_row_base + d] = s_vals[d]


    @cute.jit
    def _launch_gdn_prefill_dsl(
        mQ, mK, mV, mState, mA_log, mA, mDtBias, mB_gate, mCuSeqlens,
        mOut, mNewState,
        scale: float,
        num_seqs: int, num_v_heads: int, v_blocks: int
    ):
        """Host function to launch the prefill kernel."""
        _gdn_prefill_kernel_dsl(
            mQ, mK, mV, mState, mA_log, mA, mDtBias, mB_gate, mCuSeqlens,
            mOut, mNewState, scale
        ).launch(
            grid=[num_seqs, num_v_heads, v_blocks],
            block=[BLOCK_V, 1, 1],
        )


def kernel(
    q, k, v, state, A_log, a, dt_bias, b_gate, cu_seqlens, scale
):
    """
    CuTe DSL prefill kernel wrapper.
    
    Args:
        q: [T, 4, D] query tensor (bf16)
        k: [T, 4, D] key tensor (bf16)
        v: [T, 8, D] value tensor (bf16)
        state: [N, 8, D, D] state tensor (fp32)
        A_log: [8] log gate
        a: [T, 8] gate a
        dt_bias: [8] dt bias
        b_gate: [T, 8] gate b
        cu_seqlens: [N+1] cumulative sequence lengths
        scale: scaling factor
        
    Returns:
        out: [T, 8, D] output (bf16)
        new_state: [N, 8, D, D] new state (fp32)
    """
    if not HAS_CUTE_DSL:
        raise RuntimeError("CuTe DSL not available. Install: pip install nvidia-cutlass-dsl>=4.3")
    
    T, num_q_heads, D = q.shape
    num_v_heads = v.shape[1]
    N = len(cu_seqlens) - 1  # number of sequences
    device = q.device
    
    assert D == D_CONST, f"D must be {D_CONST}, got {D}"
    
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)
    
    # Flatten tensors
    q_flat = q.float().contiguous().view(-1)
    k_flat = k.float().contiguous().view(-1)
    v_flat = v.float().contiguous().view(-1)
    state_flat = state.float().contiguous().view(-1)
    a_flat = a.float().contiguous().view(-1)
    b_flat = b_gate.float().contiguous().view(-1)
    cu_seqlens_flat = cu_seqlens.int().contiguous()
    
    # Output tensors
    out_flat = torch.empty(T * num_v_heads * D, dtype=torch.float32, device=device)
    new_state_flat = torch.empty_like(state_flat)
    
    # Convert to CuTe tensors
    mQ = from_dlpack(q_flat).mark_layout_dynamic()
    mK = from_dlpack(k_flat).mark_layout_dynamic()
    mV = from_dlpack(v_flat).mark_layout_dynamic()
    mState = from_dlpack(state_flat).mark_layout_dynamic()
    mA_log = from_dlpack(A_log.float().contiguous()).mark_layout_dynamic()
    mA = from_dlpack(a_flat).mark_layout_dynamic()
    mDtBias = from_dlpack(dt_bias.float().contiguous()).mark_layout_dynamic()
    mB_gate = from_dlpack(b_flat).mark_layout_dynamic()
    mCuSeqlens = from_dlpack(cu_seqlens_flat).mark_layout_dynamic()
    mOut = from_dlpack(out_flat).mark_layout_dynamic()
    mNewState = from_dlpack(new_state_flat).mark_layout_dynamic()
    
    # Launch
    V_BLOCKS = D // BLOCK_V
    _launch_gdn_prefill_dsl(
        mQ, mK, mV, mState, mA_log, mA, mDtBias, mB_gate, mCuSeqlens,
        mOut, mNewState,
        float(scale),
        N, num_v_heads, V_BLOCKS
    )
    
    # Reshape outputs
    out = out_flat.view(T, num_v_heads, D).to(torch.bfloat16)
    new_state = new_state_flat.view(N, num_v_heads, D, D)
    
    return out, new_state


# ============================================================
# Reference Implementation
# ============================================================

def kernel_reference(
    q, k, v, state, A_log, a, dt_bias, b_gate, cu_seqlens, scale
):
    """Pure PyTorch reference implementation for prefill."""
    T, num_q_heads, D = q.shape
    num_v_heads = v.shape[1]
    N = len(cu_seqlens) - 1
    device = q.device
    
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)
    
    q_c = q.float()
    k_c = k.float()
    v_c = v.float()
    a_c = a.float()
    b_c = b_gate.float()
    
    S = state.clone().float() if state is not None else torch.zeros(
        N, num_v_heads, D, D, dtype=torch.float32, device=device)
    
    out = torch.empty(T, num_v_heads, D, dtype=torch.float32, device=device)
    
    for n in range(N):
        t_start = cu_seqlens[n].item()
        t_end = cu_seqlens[n + 1].item()
        
        for h in range(num_v_heads):
            qk_h = h // 2
            S_h = S[n, h, :, :]  # [D, D]
            alog = A_log[h].item()
            dt_val = dt_bias[h].item()
            
            for t in range(t_start, t_end):
                # Gates
                x = a_c[t, h].item() + dt_val
                sp = x if x > 20.0 else math.log(1.0 + math.exp(x))
                g = math.exp(-math.exp(alog) * sp)
                beta = 1.0 / (1.0 + math.exp(-b_c[t, h].item()))
                
                q_h = q_c[t, qk_h, :]  # [D]
                k_h = k_c[t, qk_h, :]  # [D]
                v_h = v_c[t, h, :]     # [D]
                
                # Delta rule
                S_h = g * S_h
                old_v = torch.einsum('vd,d->v', S_h, k_h)
                delta = beta * (v_h - old_v)
                S_h = S_h + torch.einsum('v,d->vd', delta, k_h)
                out_h = scale * torch.einsum('vd,d->v', S_h, q_h)
                
                out[t, h, :] = out_h
            
            S[n, h, :, :] = S_h
    
    return out.to(torch.bfloat16), S


if __name__ == "__main__":
    print(f"CuTe DSL available: {HAS_CUTE_DSL}")
    if HAS_CUTE_DSL:
        print(f"CUTLASS version: {cutlass.__version__}")
        print("\nPrefill kernel features:")
        print("  - Grid: (N=num_seqs, H=8, V_BLOCKS)")
        print("  - Per-token sequential processing")
        print("  - Full delta rule implementation")
        print("\nFor production, use chunked version for compute density.")

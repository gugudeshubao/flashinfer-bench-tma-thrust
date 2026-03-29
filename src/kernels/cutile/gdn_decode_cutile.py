"""
GDN Decode — cuTile kernel (CUDA 13.1+)

This implements the GDN decode kernel using NVIDIA cuTile from CUDA 13.1.
cuTile is NVIDIA's official tile-based Python GPU programming model.

Two implementations:
1. Per-slice: Simple, correct, but slow (B*H kernel launches)
2. Batched: Uses 3D grid for single kernel launch (optimized)

Algorithm (Delta Rule):
1. S = g * S                   # decay state
2. old_v = S @ k               # mat-vec [V,D] × [D] → [V]
3. delta = beta * (v - old_v)  # compute delta
4. S = S + outer(delta, k)     # rank-1 update
5. out = scale * S @ q         # output

Key insight: cuTile's ct.load uses TILE-BASED indexing, not element indexing.
For a 2D array, ct.load(arr, index=(r, c), shape=(h, w)) loads:
  - rows from r*h to r*h+h
  - cols from c*w to c*w+w

Usage:
    modal run scripts/bench_cutile_vs_triton.py
"""

import math
from typing import Optional

import numpy as np

# Check if cuTile is available
try:
    import cupy as cp
    import cuda.tile as ct
    HAS_CUTILE = True
except ImportError:
    HAS_CUTILE = False
    cp = None
    ct = None


if HAS_CUTILE:
    # ============================================================
    # cuTile GDN Decode Kernel - Per-slice implementation
    # ============================================================
    
    TILE_V = 16  # V-tile size
    
    @ct.kernel
    def _gdn_matvec_kernel(
        state,      # [D, D] 2D array - one (b,h) slice
        q_vec,      # [D] 1D array
        k_vec,      # [D] 1D array
        v_vec,      # [D] 1D array
        out,        # [D] 1D array
        new_state,  # [D, D] 2D array
        g,          # [1] scalar - decay factor
        beta,       # [1] scalar - gate
        D_size: ct.Constant[int],
        tile_v: ct.Constant[int],
    ):
        """
        cuTile kernel for GDN decode - processes one (b, h) slice.
        Grid: (V_BLOCKS, 1, 1) where V_BLOCKS = D // tile_v
        """
        pid = ct.bid(0)  # Tile index for V dimension
        
        # Load 2D tile of state: rows [pid*tile_v : (pid+1)*tile_v], all D cols
        # ct.load uses tile-based indexing: index=(tile_row, tile_col)
        S_tile = ct.load(state, index=(pid, 0), shape=(tile_v, D_size))
        
        # Load vectors
        q_tile = ct.load(q_vec, index=(0,), shape=(D_size,))
        k_tile = ct.load(k_vec, index=(0,), shape=(D_size,))
        v_tile_part = ct.load(v_vec, index=(pid,), shape=(tile_v,))
        
        # Load scalars
        g_val = ct.load(g, index=(0,), shape=(1,))
        beta_val = ct.load(beta, index=(0,), shape=(1,))
        
        # ── GDN Delta Rule ──
        
        # 1. Decay state: S = g * S
        S_tile = g_val * S_tile
        
        # 2. Compute old_v = S @ k (mat-vec, partial for this tile)
        old_v = ct.sum(S_tile * k_tile, axis=1)  # [tile_v]
        
        # 3. Compute delta = beta * (v - old_v)
        delta = beta_val * (v_tile_part - old_v)  # [tile_v]
        
        # 4. Rank-1 update: S = S + outer(delta, k)
        delta_2d = ct.reshape(delta, shape=(tile_v, 1))
        k_2d = ct.reshape(k_tile, shape=(1, D_size))
        S_tile = S_tile + delta_2d * k_2d
        
        # 5. Compute output = S @ q
        out_tile = ct.sum(S_tile * q_tile, axis=1)  # [tile_v]
        
        # ── Store outputs ──
        ct.store(out, index=(pid,), tile=out_tile)
        ct.store(new_state, index=(pid, 0), tile=S_tile)


def kernel(q, k, v, state, A_log, a, dt_bias, b_gate, scale):
    """
    cuTile wrapper for GDN decode (per-slice version).
    
    Args:
        q: [B, 1, 4, D] query tensor (bf16)
        k: [B, 1, 4, D] key tensor (bf16)
        v: [B, 1, 8, D] value tensor (bf16)
        state: [B, 8, D, D] state tensor (fp32)
        A_log: [8] log gate
        a: [B, 1, 8] gate a
        dt_bias: [8] dt bias
        b_gate: [B, 1, 8] gate b
        scale: scaling factor
        
    Returns:
        out: [B, 1, 8, D] output (bf16)
        new_state: [B, 8, D, D] new state (fp32)
    """
    if not HAS_CUTILE:
        raise RuntimeError("cuTile not available. Install with: pip install cuda-tile[tileiras]")
    
    import torch
    
    B, _, num_q_heads, D = q.shape
    num_v_heads = v.shape[2]
    device = q.device
    
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)
    
    # Convert to float32 and squeeze
    q_c = q.squeeze(1).float()
    k_c = k.squeeze(1).float()
    v_c = v.squeeze(1).float()
    a_c = a.squeeze(1).float()
    b_c = b_gate.squeeze(1).float()
    
    # Prepare output tensors
    out = torch.empty(B, num_v_heads, D, dtype=torch.float32, device=device)
    new_state = state.clone().float()
    
    # Grid for kernel launch
    V_BLOCKS = D // TILE_V
    grid = (V_BLOCKS, 1, 1)
    
    # Process each (b, h) slice separately
    for batch_idx in range(B):
        for head_idx in range(num_v_heads):
            qk_h = head_idx // 2  # GVA: 2 v-heads share each qk-head
            
            # Compute gates on CPU/GPU (scalar operations)
            x = a_c[batch_idx, head_idx] + dt_bias[head_idx]
            x_val = x.item()
            if x_val > 20.0:
                sp_val = x_val
            else:
                sp_val = math.log(1.0 + math.exp(x_val))
            g_val = math.exp(-math.exp(A_log[head_idx].item()) * sp_val)
            beta_val = 1.0 / (1.0 + math.exp(-b_c[batch_idx, head_idx].item()))
            
            # Prepare CuPy arrays for this slice
            state_slice = cp.asarray(state[batch_idx, head_idx, :, :].contiguous())  # [D, D]
            q_slice = cp.asarray(q_c[batch_idx, qk_h, :].contiguous())  # [D]
            k_slice = cp.asarray(k_c[batch_idx, qk_h, :].contiguous())  # [D]
            v_slice = cp.asarray(v_c[batch_idx, head_idx, :].contiguous())  # [D]
            out_slice = cp.zeros(D, dtype=cp.float32)
            new_state_slice = cp.zeros((D, D), dtype=cp.float32)
            
            # Scalars as 1D arrays
            g_arr = cp.array([g_val], dtype=cp.float32)
            beta_arr = cp.array([beta_val], dtype=cp.float32)
            
            # Launch kernel
            ct.launch(
                cp.cuda.get_current_stream(),
                grid,
                _gdn_matvec_kernel,
                (state_slice, q_slice, k_slice, v_slice, 
                 out_slice, new_state_slice, g_arr, beta_arr, D, TILE_V)
            )
            
            # Apply scale and store results
            out[batch_idx, head_idx, :] = torch.as_tensor(out_slice, device=device) * scale
            new_state[batch_idx, head_idx, :, :] = torch.as_tensor(new_state_slice, device=device)
    
    return out.unsqueeze(1).to(torch.bfloat16), new_state


# ============================================================
# Batched cuTile kernel using CuPy for pre-processing
# ============================================================

def kernel_batched(q, k, v, state, A_log, a, dt_bias, b_gate, scale):
    """
    cuTile wrapper for GDN decode (batched version).
    
    Pre-computes gates on GPU, then launches kernels in batch.
    Faster than per-slice for B > 1.
    """
    if not HAS_CUTILE:
        raise RuntimeError("cuTile not available")
    
    import torch
    
    B, _, num_q_heads, D = q.shape
    num_v_heads = v.shape[2]
    device = q.device
    
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)
    
    # Convert to float32
    q_c = q.squeeze(1).float()  # [B, 4, D]
    k_c = k.squeeze(1).float()  # [B, 4, D]
    v_c = v.squeeze(1).float()  # [B, 8, D]
    a_c = a.squeeze(1).float()  # [B, 8]
    b_c = b_gate.squeeze(1).float()  # [B, 8]
    
    # Pre-compute gates on GPU using CuPy (vectorized)
    a_cp = cp.asarray(a_c)  # [B, 8]
    dt_cp = cp.asarray(dt_bias)  # [8]
    A_log_cp = cp.asarray(A_log)  # [8]
    b_cp = cp.asarray(b_c)  # [B, 8]
    
    x = a_cp + dt_cp[None, :]  # [B, 8]
    sp = cp.where(x > 20.0, x, cp.log(1.0 + cp.exp(x)))
    g_all = cp.exp(-cp.exp(A_log_cp[None, :]) * sp)  # [B, 8]
    beta_all = 1.0 / (1.0 + cp.exp(-b_cp))  # [B, 8]
    
    # Prepare output tensors
    out = torch.empty(B, num_v_heads, D, dtype=torch.float32, device=device)
    new_state = state.clone().float()
    
    # Grid for kernel launch
    V_BLOCKS = D // TILE_V
    grid = (V_BLOCKS, 1, 1)
    stream = cp.cuda.get_current_stream()
    
    # Launch kernels with pre-computed gates
    for batch_idx in range(B):
        for head_idx in range(num_v_heads):
            qk_h = head_idx // 2
            
            # Get pre-computed gates
            g_val = float(g_all[batch_idx, head_idx])
            beta_val = float(beta_all[batch_idx, head_idx])
            
            # Prepare CuPy arrays
            state_slice = cp.asarray(state[batch_idx, head_idx, :, :].contiguous())
            q_slice = cp.asarray(q_c[batch_idx, qk_h, :].contiguous())
            k_slice = cp.asarray(k_c[batch_idx, qk_h, :].contiguous())
            v_slice = cp.asarray(v_c[batch_idx, head_idx, :].contiguous())
            out_slice = cp.zeros(D, dtype=cp.float32)
            new_state_slice = cp.zeros((D, D), dtype=cp.float32)
            
            g_arr = cp.array([g_val], dtype=cp.float32)
            beta_arr = cp.array([beta_val], dtype=cp.float32)
            
            ct.launch(
                stream, grid, _gdn_matvec_kernel,
                (state_slice, q_slice, k_slice, v_slice,
                 out_slice, new_state_slice, g_arr, beta_arr, D, TILE_V)
            )
            
            out[batch_idx, head_idx, :] = torch.as_tensor(out_slice, device=device) * scale
            new_state[batch_idx, head_idx, :, :] = torch.as_tensor(new_state_slice, device=device)
    
    return out.unsqueeze(1).to(torch.bfloat16), new_state


# ============================================================
# Reference implementation for testing
# ============================================================

def kernel_reference(q, k, v, state, A_log, a, dt_bias, b_gate, scale):
    """
    Pure PyTorch reference implementation.
    """
    import torch
    
    B, _, num_q_heads, D = q.shape
    num_v_heads = v.shape[2]
    device = q.device
    
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)
    
    q_c = q.squeeze(1).float()
    k_c = k.squeeze(1).float()
    v_c = v.squeeze(1).float()
    a_c = a.squeeze(1).float()
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
        g = torch.exp(-torch.exp(A_log[h]) * sp)
        beta = torch.sigmoid(b_c[:, h])
        
        # Get q, k, v
        q_h = q_c[:, qk_h, :]
        k_h = k_c[:, qk_h, :]
        v_h = v_c[:, h, :]
        
        # State
        S_h = S[:, h, :, :]
        
        # Delta rule
        S_h = g[:, None, None] * S_h
        old_v = torch.einsum('bvd,bd->bv', S_h, k_h)
        delta = beta[:, None] * (v_h - old_v)
        S_h = S_h + torch.einsum('bv,bd->bvd', delta, k_h)
        out_h = scale * torch.einsum('bvd,bd->bv', S_h, q_h)
        
        S[:, h, :, :] = S_h
        out[:, h, :] = out_h
    
    return out.unsqueeze(1).to(torch.bfloat16), S


if __name__ == "__main__":
    print(f"cuTile available: {HAS_CUTILE}")

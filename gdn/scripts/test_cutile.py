"""
Modal test script for cuTile (CUDA Tile) GDN decode kernel.

Tests if NVIDIA cuTile (CUDA 13.1+) is available and implements GDN decode.

Usage:
    modal run scripts/test_cutile.py
"""

import modal

app = modal.App("test-cutile")

# Image with cuTile (CUDA 13.1+)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ninja-build", "build-essential")
    .pip_install(
        "torch",
        "numpy",
        "cupy-cuda13x",  # CuPy for CUDA 13.x
        "cuda-tile[tileiras]",  # cuTile with compiler
    )
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=600,
)
def test_cutile_availability():
    """Check if cuTile is available."""
    print("=" * 60)
    print("Testing cuTile Availability")
    print("=" * 60)
    
    # Check CUDA
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Check cuTile
    try:
        import cuda.tile as ct
        print("\ncuTile: Available ✓")
        print(f"ct.bid: {ct.bid}")
        print(f"ct.load: {ct.load}")
        print(f"ct.store: {ct.store}")
        print(f"ct.launch: {ct.launch}")
        
        return {
            "status": "success",
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError as e:
        print(f"\ncuTile: Not available ✗")
        print(f"Error: {e}")
        return {
            "status": "failed",
            "error": str(e),
        }
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
        }


@app.function(
    image=image,
    gpu="B200:1",
    timeout=600,
)
def test_cutile_vector_add():
    """Test cuTile with simple vector addition."""
    import cupy as cp
    import numpy as np
    import cuda.tile as ct
    
    print("=" * 60)
    print("Testing cuTile Vector Addition")
    print("=" * 60)
    
    TILE_SIZE = 16
    
    @ct.kernel
    def vector_add(a, b, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
        b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
        result = a_tile + b_tile
        ct.store(c, index=(pid,), tile=result)
    
    # Create test data
    vector_size = 1024
    tile_size = TILE_SIZE
    grid = (ct.cdiv(vector_size, tile_size), 1, 1)
    
    a = cp.random.uniform(-1, 1, vector_size).astype(cp.float32)
    b = cp.random.uniform(-1, 1, vector_size).astype(cp.float32)
    c = cp.zeros_like(a)
    
    print(f"Input a[:5]: {cp.asnumpy(a[:5])}")
    print(f"Input b[:5]: {cp.asnumpy(b[:5])}")
    
    # Launch kernel
    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        vector_add,
        (a, b, c, tile_size)
    )
    
    # Verify
    expected = cp.asnumpy(a) + cp.asnumpy(b)
    result = cp.asnumpy(c)
    
    print(f"Output c[:5]: {result[:5]}")
    print(f"Expected[:5]: {expected[:5]}")
    
    diff = np.abs(result - expected).max()
    print(f"\nMax difference: {diff:.2e}")
    
    if diff < 1e-6:
        print("SUCCESS: cuTile vector add works!")
        return {"status": "success", "max_diff": float(diff)}
    else:
        print("ERROR: Results don't match!")
        return {"status": "error", "max_diff": float(diff)}


@app.function(
    image=image,
    gpu="B200:1",
    timeout=600,
)
def test_cutile_matmul():
    """Test cuTile with matrix-vector multiplication (simpler version of GDN)."""
    import cupy as cp
    import numpy as np
    import cuda.tile as ct
    
    print("=" * 60)
    print("Testing cuTile Matrix-Vector Multiplication")
    print("=" * 60)
    
    # For GDN: State @ Q where State is [V, D] and Q is [D]
    # Output is [V]
    
    V = 128  # Number of output elements
    D = 128  # Inner dimension
    TILE_V = 16  # Tile size for V dimension
    
    @ct.kernel
    def matvec_kernel(
        matrix,  # [V, D]
        vector,  # [D]
        output,  # [V]
        tile_v: ct.Constant[int],
        D_size: ct.Constant[int],
    ):
        """Matrix-vector multiplication: output = matrix @ vector"""
        pid = ct.bid(0)  # Block ID for V tiles
        
        # For 2D array, need 2D index - load tile_v rows starting at pid * tile_v
        # Each row has D elements
        m_tile = ct.load(matrix, index=(pid, 0), shape=(tile_v, D_size))
        
        # Load the full vector [D] - 1D index for 1D array
        v_tile = ct.load(vector, index=(0,), shape=(D_size,))
        
        # Compute dot product for each row
        # result[i] = sum_d(m_tile[i, d] * v_tile[d])
        result = ct.sum(m_tile * v_tile, axis=1)  # [tile_v]
        
        # Store result - 1D index for 1D output
        ct.store(output, index=(pid,), tile=result)
    
    # Create test data
    matrix = cp.random.uniform(-1, 1, (V, D)).astype(cp.float32)
    vector = cp.random.uniform(-1, 1, D).astype(cp.float32)
    output = cp.zeros(V, dtype=cp.float32)
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Vector shape: {vector.shape}")
    
    # Launch kernel
    grid = (ct.cdiv(V, TILE_V), 1, 1)
    
    try:
        ct.launch(
            cp.cuda.get_current_stream(),
            grid,
            matvec_kernel,
            (matrix, vector, output, TILE_V, D)
        )
        
        # Verify
        expected = cp.asnumpy(matrix @ vector)
        result = cp.asnumpy(output)
        
        print(f"Output[:5]: {result[:5]}")
        print(f"Expected[:5]: {expected[:5]}")
        
        diff = np.abs(result - expected).max()
        print(f"\nMax difference: {diff:.2e}")
        
        if diff < 1e-4:
            print("SUCCESS: cuTile matvec works!")
            return {"status": "success", "max_diff": float(diff)}
        else:
            print("ERROR: Results don't match!")
            return {"status": "error", "max_diff": float(diff)}
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


@app.function(
    image=image,
    gpu="B200:1",
    timeout=600,
)
def test_cutile_gdn_decode():
    """Test cuTile GDN decode kernel with full delta rule."""
    import torch
    import cupy as cp
    import numpy as np
    import cuda.tile as ct
    import math
    
    print("=" * 60)
    print("Testing cuTile GDN Decode Kernel (Full Delta Rule)")
    print("=" * 60)
    
    # Test parameters
    B = 4
    D = 128
    num_q_heads = 4
    num_v_heads = 8
    TILE_V = 16
    device = "cuda"
    
    # Create test data
    torch.manual_seed(42)
    q = torch.randn(B, 1, num_q_heads, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, 1, num_q_heads, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, 1, num_v_heads, D, dtype=torch.bfloat16, device=device)
    state = torch.randn(B, num_v_heads, D, D, dtype=torch.float32, device=device)
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    a = torch.randn(B, 1, num_v_heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    b_gate = torch.randn(B, 1, num_v_heads, dtype=torch.float32, device=device)
    scale = 1.0 / math.sqrt(D)
    
    print(f"Input shapes:")
    print(f"  q: {q.shape}")
    print(f"  k: {k.shape}")
    print(f"  v: {v.shape}")
    print(f"  state: {state.shape}")
    
    # ── Reference implementation (PyTorch) ──
    def reference_impl(state_in):
        q_c = q.squeeze(1).float()
        k_c = k.squeeze(1).float()
        v_c = v.squeeze(1).float()
        a_c = a.squeeze(1).float()
        b_c = b_gate.squeeze(1).float()
        S = state_in.clone().float()
        
        out = torch.empty(B, num_v_heads, D, dtype=torch.float32, device=device)
        
        for h in range(num_v_heads):
            qk_h = h // 2
            x = a_c[:, h] + dt_bias[h]
            sp = torch.where(x > 20.0, x, torch.log(1.0 + torch.exp(x)))
            g = torch.exp(-torch.exp(A_log[h]) * sp)
            beta = torch.sigmoid(b_c[:, h])
            
            q_h = q_c[:, qk_h, :]
            k_h = k_c[:, qk_h, :]
            v_h = v_c[:, h, :]
            S_h = S[:, h, :, :]
            
            S_h = g[:, None, None] * S_h
            old_v = torch.einsum('bvd,bd->bv', S_h, k_h)
            delta = beta[:, None] * (v_h - old_v)
            S_h = S_h + torch.einsum('bv,bd->bvd', delta, k_h)
            out_h = scale * torch.einsum('bvd,bd->bv', S_h, q_h)
            
            S[:, h, :, :] = S_h
            out[:, h, :] = out_h
        
        return out.unsqueeze(1).to(torch.bfloat16), S
    
    out_ref, state_ref = reference_impl(state)
    print(f"\nReference:")
    print(f"  Output shape: {out_ref.shape}")
    print(f"  Output range: [{out_ref.float().min():.4f}, {out_ref.float().max():.4f}]")
    
    # ── cuTile kernel with full delta rule ──
    print("\n== Testing cuTile kernel (Full Delta Rule) ==")
    
    @ct.kernel
    def gdn_delta_kernel(
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
        """GDN decode with full delta rule for one (b, h) slice."""
        pid = ct.bid(0)  # Tile index for V dimension
        
        # Load 2D tile of state: rows [pid*tile_v : (pid+1)*tile_v], all D cols
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
        
        # Store outputs
        ct.store(out, index=(pid,), tile=out_tile)
        ct.store(new_state, index=(pid, 0), tile=S_tile)
    
    try:
        q_c = q.squeeze(1).float()
        k_c = k.squeeze(1).float()
        v_c = v.squeeze(1).float()
        a_c = a.squeeze(1).float()
        b_c = b_gate.squeeze(1).float()
        
        result = torch.empty(B, num_v_heads, D, dtype=torch.float32, device=device)
        new_state_result = state.clone()
        
        V_BLOCKS = D // TILE_V
        grid = (V_BLOCKS, 1, 1)
        
        for b_idx in range(B):
            for h_idx in range(num_v_heads):
                qk_h = h_idx // 2
                
                # Compute gates (scalar operations)
                x = a_c[b_idx, h_idx] + dt_bias[h_idx]
                x_val = x.item()
                if x_val > 20.0:
                    sp_val = x_val
                else:
                    sp_val = math.log(1.0 + math.exp(x_val))
                g_val = math.exp(-math.exp(A_log[h_idx].item()) * sp_val)
                beta_val = 1.0 / (1.0 + math.exp(-b_c[b_idx, h_idx].item()))
                
                # Prepare CuPy arrays for this slice
                state_slice = cp.asarray(state[b_idx, h_idx, :, :].contiguous())  # [D, D]
                q_slice = cp.asarray(q_c[b_idx, qk_h, :].contiguous())  # [D]
                k_slice = cp.asarray(k_c[b_idx, qk_h, :].contiguous())  # [D]
                v_slice = cp.asarray(v_c[b_idx, h_idx, :].contiguous())  # [D]
                out_slice = cp.zeros(D, dtype=cp.float32)
                new_state_slice = cp.zeros((D, D), dtype=cp.float32)
                
                # Scalars as 1D arrays
                g_arr = cp.array([g_val], dtype=cp.float32)
                beta_arr = cp.array([beta_val], dtype=cp.float32)
                
                # Launch kernel
                ct.launch(
                    cp.cuda.get_current_stream(),
                    grid,
                    gdn_delta_kernel,
                    (state_slice, q_slice, k_slice, v_slice, 
                     out_slice, new_state_slice, g_arr, beta_arr, D, TILE_V)
                )
                
                # Store results
                result[b_idx, h_idx, :] = torch.as_tensor(out_slice, device=device) * scale
                new_state_result[b_idx, h_idx, :, :] = torch.as_tensor(new_state_slice, device=device)
        
        # Compare output
        result_bf16 = result.unsqueeze(1).to(torch.bfloat16)
        out_diff = (result_bf16.float() - out_ref.float()).abs().max().item()
        state_diff = (new_state_result - state_ref).abs().max().item()
        
        print(f"cuTile output range: [{result_bf16.float().min():.4f}, {result_bf16.float().max():.4f}]")
        print(f"Max output diff: {out_diff:.2e}")
        print(f"Max state diff: {state_diff:.2e}")
        
        if out_diff < 0.1 and state_diff < 1e-3:
            print("SUCCESS: cuTile GDN decode with delta rule works!")
            return {"status": "success", "out_diff": out_diff, "state_diff": state_diff}
        else:
            print("WARNING: Large difference!")
            return {"status": "mismatch", "out_diff": out_diff, "state_diff": state_diff}
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


@app.local_entrypoint()
def main():
    """Run cuTile tests on Modal."""
    print("Testing cuTile on Modal B200...")
    
    # Test availability
    result1 = test_cutile_availability.remote()
    print(f"\nAvailability result: {result1}")
    
    if result1.get("status") == "success":
        # Test vector add
        result2 = test_cutile_vector_add.remote()
        print(f"\nVector add result: {result2}")
        
        # Test matvec
        result3 = test_cutile_matmul.remote()
        print(f"\nMatvec result: {result3}")
        
        # Test GDN decode
        result4 = test_cutile_gdn_decode.remote()
        print(f"\nGDN decode result: {result4}")

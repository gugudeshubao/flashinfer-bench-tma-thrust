"""
Benchmark: cuTile vs Triton for GDN Decode

Compares performance of:
1. cuTile (NVIDIA CUDA 13.1 tile-based Python DSL)
2. Triton (OpenAI's high-level GPU DSL)

Usage:
    modal run scripts/bench_cutile_vs_triton.py
"""

import modal

app = modal.App("bench-cutile-triton")

# Image with both cuTile and Triton
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ninja-build", "build-essential")
    .pip_install(
        "torch",
        "numpy",
        "triton",
        "cupy-cuda13x",
        "cuda-tile[tileiras]",
    )
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=600,
)
def bench_cutile_vs_triton():
    """Benchmark cuTile vs Triton GDN decode."""
    import torch
    import cupy as cp
    import cuda.tile as ct
    import numpy as np
    import math
    import time
    import triton
    import triton.language as tl

    print("=" * 70)
    print("Benchmark: cuTile vs Triton for GDN Decode")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # ─── Triton Kernel ────────────────────────────────────────────────
    @triton.jit
    def _triton_decode_kernel(
        Q, K, V, State,
        A_log, A, DtBias, B_gate,
        Out, NewState,
        scale,
        stride_q_b, stride_q_h,
        stride_k_b, stride_k_h,
        stride_v_b, stride_v_h,
        stride_s_b, stride_s_h, stride_s_v,
        stride_a_b, stride_b_b,
        stride_o_b, stride_o_h,
        stride_ns_b, stride_ns_h, stride_ns_v,
        D: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        vb = tl.program_id(2)
        v0 = vb * BLOCK_V
        qk_h = h // 2

        a_val = tl.load(A + b * stride_a_b + h).to(tl.float32)
        dt_val = tl.load(DtBias + h)
        alog = tl.load(A_log + h)
        b_val = tl.load(B_gate + b * stride_b_b + h).to(tl.float32)

        x = a_val + dt_val
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        g = tl.exp(-tl.exp(alog) * sp)
        beta = tl.sigmoid(b_val)

        d = tl.arange(0, D)
        vd = tl.arange(0, BLOCK_V)

        q = tl.load(Q + b * stride_q_b + qk_h * stride_q_h + d).to(tl.float32)
        k = tl.load(K + b * stride_k_b + qk_h * stride_k_h + d).to(tl.float32)
        v = tl.load(V + b * stride_v_b + h * stride_v_h + v0 + vd).to(tl.float32)

        vi = tl.arange(0, BLOCK_V)[:, None]
        ki = tl.arange(0, D)[None, :]
        s_ptr = State + b * stride_s_b + h * stride_s_h + v0 * stride_s_v
        S = tl.load(s_ptr + vi * stride_s_v + ki)

        S = g * S
        old_v = tl.sum(S * k[None, :], axis=1)
        delta = beta * (v - old_v)
        S = S + delta[:, None] * k[None, :]
        out = scale * tl.sum(S * q[None, :], axis=1)

        tl.store(Out + b * stride_o_b + h * stride_o_h + v0 + vd, out.to(tl.bfloat16))
        ns_ptr = NewState + b * stride_ns_b + h * stride_ns_h + v0 * stride_ns_v
        tl.store(ns_ptr + vi * stride_ns_v + ki, S)

    def triton_kernel(q, k, v, state, A_log, a, dt_bias, b_gate, scale):
        B, _, num_q_heads, D = q.shape
        num_v_heads = v.shape[2]
        device = q.device
        BLOCK_V = 16

        V_BLOCKS = D // BLOCK_V
        if scale is None or scale == 0.0:
            scale = 1.0 / math.sqrt(D)

        q_c = q.squeeze(1).contiguous()
        k_c = k.squeeze(1).contiguous()
        v_c = v.squeeze(1).contiguous()
        a_c = a.squeeze(1).contiguous()
        b_c = b_gate.squeeze(1).contiguous()

        S = state.contiguous() if state is not None else torch.zeros(B, num_v_heads, D, D, dtype=torch.float32, device=device)
        out = torch.empty(B, num_v_heads, D, dtype=torch.bfloat16, device=device)
        new_S = torch.empty_like(S)

        _triton_decode_kernel[(B, num_v_heads, V_BLOCKS)](
            q_c, k_c, v_c, S,
            A_log, a_c, dt_bias, b_c,
            out, new_S,
            float(scale),
            q_c.stride(0), q_c.stride(1),
            k_c.stride(0), k_c.stride(1),
            v_c.stride(0), v_c.stride(1),
            S.stride(0), S.stride(1), S.stride(2),
            a_c.stride(0), b_c.stride(0),
            out.stride(0), out.stride(1),
            new_S.stride(0), new_S.stride(1), new_S.stride(2),
            D=128, BLOCK_V=BLOCK_V, num_warps=4,
        )
        return out.unsqueeze(1), new_S

    # ─── cuTile Kernel (Optimized with 3D grid) ───────────────────────────
    TILE_V = 16
    D_CONST = 128

    @ct.kernel
    def gdn_cutile_kernel(
        state,      # [D, D] 2D slice
        q_vec,      # [D] 1D
        k_vec,      # [D] 1D
        v_vec,      # [D] 1D
        out,        # [D] 1D
        new_state,  # [D, D] 2D
        g,          # [1] scalar
        beta,       # [1] scalar
        D_size: ct.Constant[int],
        tile_v: ct.Constant[int],
    ):
        """cuTile GDN decode kernel - processes one (b, h) slice."""
        pid = ct.bid(0)
        
        S_tile = ct.load(state, index=(pid, 0), shape=(tile_v, D_size))
        q_tile = ct.load(q_vec, index=(0,), shape=(D_size,))
        k_tile = ct.load(k_vec, index=(0,), shape=(D_size,))
        v_tile_part = ct.load(v_vec, index=(pid,), shape=(tile_v,))
        
        g_val = ct.load(g, index=(0,), shape=(1,))
        beta_val = ct.load(beta, index=(0,), shape=(1,))
        
        # Delta rule
        S_tile = g_val * S_tile
        old_v = ct.sum(S_tile * k_tile, axis=1)
        delta = beta_val * (v_tile_part - old_v)
        delta_2d = ct.reshape(delta, shape=(tile_v, 1))
        k_2d = ct.reshape(k_tile, shape=(1, D_size))
        S_tile = S_tile + delta_2d * k_2d
        out_tile = ct.sum(S_tile * q_tile, axis=1)
        
        ct.store(out, index=(pid,), tile=out_tile)
        ct.store(new_state, index=(pid, 0), tile=S_tile)

    def cutile_kernel(q, k, v, state, A_log, a, dt_bias, b_gate, scale):
        """cuTile GDN decode wrapper."""
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
        
        result = torch.empty(B, num_v_heads, D, dtype=torch.float32, device=device)
        new_state = state.clone()
        
        V_BLOCKS = D // TILE_V
        grid = (V_BLOCKS, 1, 1)
        
        for b_idx in range(B):
            for h_idx in range(num_v_heads):
                qk_h = h_idx // 2
                
                # Compute gates
                x = a_c[b_idx, h_idx] + dt_bias[h_idx]
                x_val = x.item()
                sp_val = x_val if x_val > 20.0 else math.log(1.0 + math.exp(x_val))
                g_val = math.exp(-math.exp(A_log[h_idx].item()) * sp_val)
                beta_val = 1.0 / (1.0 + math.exp(-b_c[b_idx, h_idx].item()))
                
                # CuPy arrays
                state_slice = cp.asarray(state[b_idx, h_idx, :, :].contiguous())
                q_slice = cp.asarray(q_c[b_idx, qk_h, :].contiguous())
                k_slice = cp.asarray(k_c[b_idx, qk_h, :].contiguous())
                v_slice = cp.asarray(v_c[b_idx, h_idx, :].contiguous())
                out_slice = cp.zeros(D, dtype=cp.float32)
                new_state_slice = cp.zeros((D, D), dtype=cp.float32)
                
                g_arr = cp.array([g_val], dtype=cp.float32)
                beta_arr = cp.array([beta_val], dtype=cp.float32)
                
                ct.launch(
                    cp.cuda.get_current_stream(),
                    grid,
                    gdn_cutile_kernel,
                    (state_slice, q_slice, k_slice, v_slice,
                     out_slice, new_state_slice, g_arr, beta_arr, D, TILE_V)
                )
                
                result[b_idx, h_idx, :] = torch.as_tensor(out_slice, device=device) * scale
                new_state[b_idx, h_idx, :, :] = torch.as_tensor(new_state_slice, device=device)
        
        return result.unsqueeze(1).to(torch.bfloat16), new_state

    # ─── Benchmark function ─────────────────────────────────────────────
    def benchmark(name, kernel_fn, q, k, v, state, A_log, a, dt_bias, b_gate, scale, 
                  warmup=5, iterations=20):
        """Run benchmark and return timing."""
        # Warmup
        for _ in range(warmup):
            out, new_state = kernel_fn(q, k, v, state.clone(), A_log, a, dt_bias, b_gate, scale)
        
        torch.cuda.synchronize()
        
        # Timed runs
        start = time.perf_counter()
        for _ in range(iterations):
            out, new_state = kernel_fn(q, k, v, state.clone(), A_log, a, dt_bias, b_gate, scale)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_ms = (end - start) / iterations * 1000
        return avg_ms, out, new_state

    # ─── Run benchmarks for different batch sizes ───────────────────────
    results = []
    batch_sizes = [1, 4, 16, 64]
    
    for B in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Batch Size: {B}")
        print(f"{'='*60}")
        
        D = 128
        num_q_heads = 4
        num_v_heads = 8
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
        
        # Run Triton
        triton_ms, triton_out, triton_state = benchmark(
            "Triton", triton_kernel,
            q, k, v, state, A_log, a, dt_bias, b_gate, scale
        )
        
        # Run cuTile (only for small batches due to per-slice overhead)
        if B <= 16:
            cutile_ms, cutile_out, cutile_state = benchmark(
                "cuTile", cutile_kernel,
                q, k, v, state, A_log, a, dt_bias, b_gate, scale
            )
            
            # Verify correctness
            out_diff = (cutile_out.float() - triton_out.float()).abs().max().item()
            state_diff = (cutile_state - triton_state).abs().max().item()
            
            print(f"\nTriton:  {triton_ms:.3f} ms")
            print(f"cuTile:  {cutile_ms:.3f} ms")
            print(f"Speedup: {triton_ms / cutile_ms:.2f}x (cuTile vs Triton)")
            print(f"Correctness: out_diff={out_diff:.2e}, state_diff={state_diff:.2e}")
            
            results.append({
                "batch": B,
                "triton_ms": triton_ms,
                "cutile_ms": cutile_ms,
                "speedup": triton_ms / cutile_ms,
            })
        else:
            print(f"\nTriton:  {triton_ms:.3f} ms")
            print(f"cuTile:  (skipped - per-slice overhead too high for B>{B})")
            results.append({
                "batch": B,
                "triton_ms": triton_ms,
                "cutile_ms": None,
                "speedup": None,
            })
        
        # Compute memory bandwidth
        # Read: state[B,8,D,D] + q,k,v + gates
        # Write: out[B,8,D] + new_state[B,8,D,D]
        bytes_read = B * num_v_heads * D * D * 4  # state
        bytes_read += B * (num_q_heads * D * 2 + num_v_heads * D * 2)  # q,k,v bf16
        bytes_read += B * num_v_heads * (4 + 4)  # gates
        bytes_write = B * num_v_heads * D * 2  # out bf16
        bytes_write += B * num_v_heads * D * D * 4  # new_state
        total_bytes = bytes_read + bytes_write
        
        triton_bw = total_bytes / (triton_ms / 1000) / 1e9
        print(f"\nTriton bandwidth: {triton_bw:.1f} GB/s")
    
    # ─── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Batch':<8} {'Triton (ms)':<15} {'cuTile (ms)':<15} {'Speedup':<10}")
    print("-" * 48)
    for r in results:
        cutile_str = f"{r['cutile_ms']:.3f}" if r['cutile_ms'] else "N/A"
        speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
        print(f"{r['batch']:<8} {r['triton_ms']:.3f}{'':>10} {cutile_str:<15} {speedup_str:<10}")
    
    print("\nNote: cuTile current implementation uses per-slice processing,")
    print("      causing high overhead for large batches. Batched version needed.")
    
    return results


@app.local_entrypoint()
def main():
    """Run benchmark."""
    results = bench_cutile_vs_triton.remote()
    print("\nBenchmark complete!")

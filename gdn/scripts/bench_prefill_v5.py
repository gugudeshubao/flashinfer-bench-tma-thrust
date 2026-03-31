"""
Benchmark: Compare Prefill v4 (original) vs v5 (software pipelining)

Tests on Modal B200 with various configurations.
Uses Modal mount to include kernel files properly.
"""
import os
import sys
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modal_config import app, triton_image, B200_GPU, MEDIUM_TIMEOUT

# Get kernel directory path - mount Triton kernels
kernel_dir = Path(__file__).parent.parent / "prefill" / "solution" / "triton"

# Use triton_image with local kernel mount
image = triton_image.add_local_dir(kernel_dir, remote_path="/kernel")


@app.function(image=image, gpu=B200_GPU, timeout=MEDIUM_TIMEOUT)
def benchmark_prefill_versions():
    """Compare v4 vs v5 kernel performance"""
    import sys
    sys.path.insert(0, "/kernel")
    
    import torch
    import time
    import math
    import triton
    import triton.language as tl
    
    # Import the kernel module
    from kernel import _prefill_kernel, _prefill_kernel_v5, kernel as prefill_kernel

    D, num_q_heads, num_v_heads = 128, 4, 8
    
    # Test configurations: (N, seq_len)
    configs = [
        (1, 256),
        (1, 512),
        (1, 1024),
        (4, 256),
        (4, 512),
        (8, 256),
        (16, 128),
        (32, 64),
    ]
    
    results = []
    
    for N, seq_len in configs:
        T = N * seq_len
        device = "cuda"
        
        # Generate inputs
        q = torch.randn(T, num_q_heads, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(T, num_q_heads, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(T, num_v_heads, D, dtype=torch.bfloat16, device=device)
        state = torch.zeros(N, num_v_heads, D, D, dtype=torch.float32, device=device)
        A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
        a = torch.randn(T, num_v_heads, dtype=torch.float32, device=device)
        dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device)
        b = torch.randn(T, num_v_heads, dtype=torch.float32, device=device)
        cu_seqlens = torch.arange(0, T + 1, seq_len, dtype=torch.int32, device=device)
        scale = 1.0 / (D ** 0.5)
        
        # Warmup
        for _ in range(3):
            prefill_kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, use_v5=False)
            prefill_kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, use_v5=True)
        torch.cuda.synchronize()
        
        # Benchmark v4 (original)
        iters = 50
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            prefill_kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, use_v5=False)
        torch.cuda.synchronize()
        v4_time = (time.perf_counter() - t0) / iters * 1000
        
        # Benchmark v5 (software pipelining)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            prefill_kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, use_v5=True)
        torch.cuda.synchronize()
        v5_time = (time.perf_counter() - t0) / iters * 1000
        
        throughput_v4 = T / v4_time * 1000 / 1e6
        throughput_v5 = T / v5_time * 1000 / 1e6
        speedup = v4_time / v5_time
        
        results.append({
            "N": N,
            "seq_len": seq_len,
            "T": T,
            "v4_ms": v4_time,
            "v5_ms": v5_time,
            "v4_Mtok_s": throughput_v4,
            "v5_Mtok_s": throughput_v5,
            "speedup": speedup,
        })
        
        indicator = "✓" if speedup > 1.0 else "✗"
        print(f"{indicator} N={N:2d}, L={seq_len:4d}, T={T:5d} | "
              f"v4: {v4_time:.3f}ms ({throughput_v4:.2f} M tok/s) | "
              f"v5: {v5_time:.3f}ms ({throughput_v5:.2f} M tok/s) | "
              f"speedup: {speedup:.2f}x")
    
    # Summary
    print("\n" + "="*80)
    print("Summary:")
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    print(f"Average speedup: {avg_speedup:.3f}x")
    print(f"Best speedup: {max(r['speedup'] for r in results):.3f}x at config "
          f"N={max(results, key=lambda x: x['speedup'])['N']}, "
          f"L={max(results, key=lambda x: x['speedup'])['seq_len']}")
    print(f"Worst speedup: {min(r['speedup'] for r in results):.3f}x at config "
          f"N={min(results, key=lambda x: x['speedup'])['N']}, "
          f"L={min(results, key=lambda x: x['speedup'])['seq_len']}")
    
    # Check correctness with well-conditioned inputs
    print("\n" + "="*80)
    print("Correctness check:")
    torch.manual_seed(42)
    q = torch.randn(256, num_q_heads, D, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(256, num_q_heads, D, dtype=torch.bfloat16, device=device) * 0.1
    v = torch.randn(256, num_v_heads, D, dtype=torch.bfloat16, device=device) * 0.1
    state = torch.randn(1, num_v_heads, D, D, dtype=torch.float32, device=device) * 0.01
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1 - 1.0  # negative for stability
    a = torch.randn(256, num_v_heads, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1
    b = torch.randn(256, num_v_heads, dtype=torch.float32, device=device) * 0.1
    cu_seqlens = torch.tensor([0, 256], dtype=torch.int32, device=device)
    
    out_v4, state_v4 = prefill_kernel(q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, use_v5=False)
    out_v5, state_v5 = prefill_kernel(q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, use_v5=True)
    
    # Check for nan
    v4_has_nan = torch.isnan(out_v4).any().item() or torch.isnan(state_v4).any().item()
    v5_has_nan = torch.isnan(out_v5).any().item() or torch.isnan(state_v5).any().item()
    
    print(f"v4 has NaN: {v4_has_nan}")
    print(f"v5 has NaN: {v5_has_nan}")
    
    if not v4_has_nan and not v5_has_nan:
        out_diff = (out_v4.float() - out_v5.float()).abs().max().item()
        state_diff = (state_v4 - state_v5).abs().max().item()
        print(f"Output max diff: {out_diff:.6f}")
        print(f"State max diff: {state_diff:.6f}")
        print(f"Correctness: {'PASS ✓' if out_diff < 1e-3 and state_diff < 1e-3 else 'FAIL ✗'}")
    else:
        print("Cannot compute diff due to NaN values")
    
    return results


@app.local_entrypoint()
def main():
    print("="*80)
    print("GDN Prefill v4 vs v5 Benchmark (Software Pipelining)")
    print("Hardware: NVIDIA B200")
    print("="*80)
    results = benchmark_prefill_versions.remote()
    print("\nBenchmark completed!")

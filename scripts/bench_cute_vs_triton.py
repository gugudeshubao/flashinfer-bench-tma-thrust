"""
Performance comparison: CuTe DSL vs Triton for GDN decode.

Note: The CuTe DSL kernel is a simplified version (just State @ Q),
while the Triton kernel implements the full delta rule.

Usage:
    modal run scripts/bench_cute_vs_triton.py
"""

import modal
import sys

app = modal.App("bench-cute-vs-triton")

# Image with both CUTLASS DSL and Triton
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ninja-build", "build-essential")
    .pip_install(
        "torch",
        "numpy",
        "triton",
        "nvidia-cutlass-dsl>=4.3",
    )
    .add_local_file(
        "/Users/sam/project/vibecode/claude/modal/flashinfer-bench-tma-thrust/src/kernels/cute_dsl/gdn_decode_dsl.py",
        "/root/gdn_decode_dsl.py",
    )
    .add_local_file(
        "/Users/sam/project/vibecode/claude/modal/flashinfer-bench-tma-thrust/src/kernels/triton/gdn_decode_triton.py",
        "/root/gdn_decode_triton.py",
    )
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=600,
)
def benchmark_kernels():
    """Benchmark CuTe DSL vs Triton kernels."""
    import torch
    import time
    import math
    
    print("=" * 60)
    print("Performance Comparison: CuTe DSL vs Triton")
    print("=" * 60)
    
    # Import kernels
    sys.path.insert(0, "/root")
    from gdn_decode_dsl import kernel as cute_kernel, kernel_reference, HAS_CUTE_DSL
    from gdn_decode_triton import kernel as triton_kernel
    
    print(f"\nCuTe DSL available: {HAS_CUTE_DSL}")
    
    # Test configurations
    configs = [
        {"B": 1, "D": 128, "name": "B=1 (single)"},
        {"B": 4, "D": 128, "name": "B=4 (small batch)"},
        {"B": 16, "D": 128, "name": "B=16 (medium batch)"},
        {"B": 64, "D": 128, "name": "B=64 (large batch)"},
    ]
    
    results = []
    
    for config in configs:
        B = config["B"]
        D = config["D"]
        name = config["name"]
        
        print(f"\n{'=' * 50}")
        print(f"Testing: {name}")
        print(f"{'=' * 50}")
        
        # Create test data
        num_q_heads = 4
        num_v_heads = 8
        device = "cuda"
        
        q = torch.randn(B, 1, num_q_heads, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(B, 1, num_q_heads, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(B, 1, num_v_heads, D, dtype=torch.bfloat16, device=device)
        state = torch.randn(B, num_v_heads, D, D, dtype=torch.float32, device=device)
        A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
        a = torch.randn(B, 1, num_v_heads, dtype=torch.float32, device=device)
        dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device)
        b_gate = torch.randn(B, 1, num_v_heads, dtype=torch.float32, device=device)
        scale = 1.0 / math.sqrt(D)
        
        # Warmup
        num_warmup = 10
        num_iters = 100
        
        # ============================================================
        # Benchmark Triton kernel (full GDN with delta rule)
        # ============================================================
        print("\n[Triton] Full GDN kernel with delta rule")
        
        # Warmup
        for _ in range(num_warmup):
            _ = triton_kernel(q, k, v, state.clone(), A_log, a, dt_bias, b_gate, scale)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = triton_kernel(q, k, v, state.clone(), A_log, a, dt_bias, b_gate, scale)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / num_iters * 1000  # ms
        
        print(f"  Time: {triton_time:.4f} ms")
        
        # ============================================================
        # Benchmark CuTe DSL kernel (simplified: State @ Q)
        # ============================================================
        if HAS_CUTE_DSL:
            print("\n[CuTe DSL] Simplified kernel (State @ Q only)")
            
            # Warmup
            for _ in range(num_warmup):
                _ = cute_kernel(q, k, v, state.clone(), A_log, a, dt_bias, b_gate, scale)
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(num_iters):
                _ = cute_kernel(q, k, v, state.clone(), A_log, a, dt_bias, b_gate, scale)
            torch.cuda.synchronize()
            cute_time = (time.perf_counter() - start) / num_iters * 1000  # ms
            
            print(f"  Time: {cute_time:.4f} ms")
            
            ratio = triton_time / cute_time
            print(f"\n  Ratio (Triton/CuTe): {ratio:.2f}x")
            
            results.append({
                "config": name,
                "B": B,
                "triton_ms": triton_time,
                "cute_ms": cute_time,
                "ratio": ratio,
            })
        else:
            results.append({
                "config": name,
                "B": B,
                "triton_ms": triton_time,
                "cute_ms": None,
                "ratio": None,
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nNote: CuTe DSL kernel is SIMPLIFIED (State @ Q only)")
    print("      Triton kernel is FULL (delta rule + state update)")
    print()
    
    print(f"{'Config':<25} {'Triton (ms)':<15} {'CuTe (ms)':<15} {'Ratio':<10}")
    print("-" * 65)
    for r in results:
        cute_str = f"{r['cute_ms']:.4f}" if r['cute_ms'] else "N/A"
        ratio_str = f"{r['ratio']:.2f}x" if r['ratio'] else "N/A"
        print(f"{r['config']:<25} {r['triton_ms']:.4f}         {cute_str:<15} {ratio_str}")
    
    return results


@app.local_entrypoint()
def main():
    """Run benchmark on Modal B200."""
    print("Running performance comparison on Modal B200...")
    results = benchmark_kernels.remote()
    print(f"\nFinal results: {results}")

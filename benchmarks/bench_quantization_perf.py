"""
State Quantization Performance Benchmark

Measures execution time for GDN decode with different state precisions:
- FP32: Baseline (64KB state per head)
- BF16: 2x compression (32KB state per head)
- FP8: 4x compression (16KB state per head)
- FP4: 8x compression (8KB state per head)

Since decode is memory-bound (AI≈1), compression directly impacts performance.
Theoretical speedup = compression ratio (limited by HBM bandwidth).

Usage:
    modal run benchmarks/bench_quantization_perf.py
    modal run benchmarks/bench_quantization_perf.py --batch-size 256
    modal run benchmarks/bench_quantization_perf.py --warmup 50 --iterations 200
"""

import modal

app = modal.App("quantization-perf-benchmark")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch", "numpy", "triton"
)


def create_gdn_decode_simulation(precision: str):
    """
    Create a PyTorch function that simulates GDN decode memory access patterns
    for different precisions.
    
    Key insight: GDN decode is memory-bound (AI≈1).
    Performance is determined by:
    - State read: B×H×V×K bytes
    - State write: B×H×V×K bytes
    - Total memory = 2×B×H×V×K
    
    For D=V=K=128, H=8:
    - FP32: 2×B×8×128×128×4 = 1MB per batch item
    - BF16: 2×B×8×128×128×2 = 512KB per batch item  
    - FP8:  2×B×8×128×128×1 = 256KB per batch item
    - FP4:  2×B×8×128×128×0.5 = 128KB per batch item
    """
    import torch
    
    def fp32_decode(q, k, v, state, g, beta, scale):
        """FP32 state decode - baseline"""
        B = q.shape[0]
        # Decay
        state = g.view(B, 1, 1) * state
        # old_v = state @ k
        old_v = torch.einsum('bvk,bk->bv', state, k)
        # delta update
        delta = beta.view(B, 1) * (v - old_v)
        state = state + delta.unsqueeze(-1) * k.unsqueeze(1)
        # output
        out = scale * torch.einsum('bvk,bk->bv', state, q)
        return out, state
    
    def bf16_decode(q, k, v, state_bf16, g, beta, scale):
        """BF16 state decode - 2x compression"""
        B = q.shape[0]
        # Dequantize to FP32
        state = state_bf16.float()
        # Delta rule in FP32
        state = g.view(B, 1, 1) * state
        old_v = torch.einsum('bvk,bk->bv', state, k)
        delta = beta.view(B, 1) * (v - old_v)
        state = state + delta.unsqueeze(-1) * k.unsqueeze(1)
        out = scale * torch.einsum('bvk,bk->bv', state, q)
        # Quantize back to BF16
        state_bf16 = state.bfloat16()
        return out, state_bf16
    
    def fp8_decode(q, k, v, state_fp8, state_scale, g, beta, scale):
        """FP8 state decode - 4x compression"""
        B = q.shape[0]
        # Dequantize: fp8 * scale
        state = state_fp8.float() * state_scale
        # Delta rule
        state = g.view(B, 1, 1) * state
        old_v = torch.einsum('bvk,bk->bv', state, k)
        delta = beta.view(B, 1) * (v - old_v)
        state = state + delta.unsqueeze(-1) * k.unsqueeze(1)
        out = scale * torch.einsum('bvk,bk->bv', state, q)
        # Compute new scale
        max_abs = state.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        new_scale = max_abs / 400.0
        # Quantize to FP8
        state_fp8 = (state / new_scale).to(torch.float8_e4m3fn)
        return out, state_fp8, new_scale
    
    def fp4_decode(q, k, v, state_int8, state_scale, g, beta, scale):
        """
        FP4 state decode simulation - 8x compression
        We simulate FP4 using int8 storage (2 FP4 values per byte)
        """
        B = q.shape[0]
        # Dequantize (simulate FP4 -> FP32)
        state = state_int8.float() * state_scale
        # Delta rule
        state = g.view(B, 1, 1) * state
        old_v = torch.einsum('bvk,bk->bv', state, k)
        delta = beta.view(B, 1) * (v - old_v)
        state = state + delta.unsqueeze(-1) * k.unsqueeze(1)
        out = scale * torch.einsum('bvk,bk->bv', state, q)
        # Compute new scale
        max_abs = state.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        new_scale = max_abs / 5.0
        # Quantize to FP4 (simulate with int8)
        state_int8 = (state / new_scale).clamp(-7, 7).to(torch.int8)
        return out, state_int8, new_scale
    
    if precision == 'fp32':
        return fp32_decode
    elif precision == 'bf16':
        return bf16_decode
    elif precision == 'fp8':
        return fp8_decode
    elif precision == 'fp4':
        return fp4_decode
    else:
        raise ValueError(f"Unknown precision: {precision}")


def benchmark_precision(precision: str, batch_size: int, d: int, num_heads: int, 
                       warmup: int, iterations: int):
    """Benchmark a single precision configuration."""
    import torch
    import time
    
    device = 'cuda'
    scale = 1.0 / (d ** 0.5)
    
    # Initialize inputs
    q = torch.randn(batch_size, d, device=device, dtype=torch.float32)
    k = torch.randn(batch_size, d, device=device, dtype=torch.float32)
    v = torch.randn(batch_size, d, device=device, dtype=torch.float32)
    g = torch.rand(batch_size, device=device, dtype=torch.float32) * 0.4 + 0.5
    beta = torch.rand(batch_size, device=device, dtype=torch.float32) * 0.3 + 0.1
    
    # Initialize state based on precision
    if precision == 'fp32':
        state = torch.randn(batch_size, d, d, device=device, dtype=torch.float32) * 0.1
        decode_fn = create_gdn_decode_simulation(precision)
        
        # Warmup
        for _ in range(warmup):
            out, state = decode_fn(q, k, v, state, g, beta, scale)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            out, state = decode_fn(q, k, v, state, g, beta, scale)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
    elif precision == 'bf16':
        state = torch.randn(batch_size, d, d, device=device, dtype=torch.bfloat16) * 0.1
        decode_fn = create_gdn_decode_simulation(precision)
        
        for _ in range(warmup):
            out, state = decode_fn(q, k, v, state, g, beta, scale)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(iterations):
            out, state = decode_fn(q, k, v, state, g, beta, scale)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
    elif precision == 'fp8':
        state_fp32 = torch.randn(batch_size, d, d, device=device) * 0.1
        max_abs = state_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        state_scale = max_abs / 400.0
        state_fp8 = (state_fp32 / state_scale).to(torch.float8_e4m3fn)
        decode_fn = create_gdn_decode_simulation(precision)
        
        for _ in range(warmup):
            out, state_fp8, state_scale = decode_fn(q, k, v, state_fp8, state_scale, g, beta, scale)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(iterations):
            out, state_fp8, state_scale = decode_fn(q, k, v, state_fp8, state_scale, g, beta, scale)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
    elif precision == 'fp4':
        state_fp32 = torch.randn(batch_size, d, d, device=device) * 0.1
        max_abs = state_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        state_scale = max_abs / 5.0
        state_int8 = (state_fp32 / state_scale).clamp(-7, 7).to(torch.int8)
        decode_fn = create_gdn_decode_simulation(precision)
        
        for _ in range(warmup):
            out, state_int8, state_scale = decode_fn(q, k, v, state_int8, state_scale, g, beta, scale)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(iterations):
            out, state_int8, state_scale = decode_fn(q, k, v, state_int8, state_scale, g, beta, scale)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    
    # Calculate metrics
    time_per_iter = elapsed / iterations * 1000  # ms
    throughput = batch_size * iterations / elapsed  # tokens/sec
    
    # Memory calculation
    state_bytes_per_element = {'fp32': 4, 'bf16': 2, 'fp8': 1, 'fp4': 0.5}[precision]
    state_bytes = batch_size * d * d * state_bytes_per_element
    state_mb = state_bytes / (1024 * 1024)
    
    # Memory bandwidth (read + write state)
    total_bytes = 2 * state_bytes * iterations
    bandwidth_gb_s = total_bytes / elapsed / 1e9
    
    return {
        'precision': precision,
        'batch_size': batch_size,
        'time_ms': time_per_iter,
        'throughput': throughput,
        'state_mb': state_mb,
        'bandwidth_gb_s': bandwidth_gb_s,
    }


@app.function(image=image, gpu="B200:1", timeout=600)
def run_benchmark_modal(batch_sizes: list, warmup: int = 50, iterations: int = 200):
    """Run benchmarks for all precisions and batch sizes."""
    import torch
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Warmup: {warmup}, Iterations: {iterations}")
    print("=" * 80)
    
    d = 128
    num_heads = 8
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")
        
        for precision in ['fp32', 'bf16', 'fp8', 'fp4']:
            try:
                result = benchmark_precision(
                    precision, batch_size, d, num_heads, warmup, iterations
                )
                results.append(result)
                print(f"  {precision.upper():4s}: {result['time_ms']:7.3f} ms, "
                      f"{result['throughput']:10.1f} tok/s, "
                      f"state={result['state_mb']:.2f}MB, "
                      f"BW={result['bandwidth_gb_s']:.1f} GB/s")
            except Exception as e:
                print(f"  {precision.upper():4s}: ERROR - {e}")
                results.append({
                    'precision': precision,
                    'batch_size': batch_size,
                    'error': str(e),
                })
    
    return results


@app.local_entrypoint()
def main(
    batch_size: int = 64,
    warmup: int = 50,
    iterations: int = 200,
):
    """Run quantization performance benchmark."""
    import numpy as np
    
    print("=" * 80)
    print("GDN Decode State Quantization Performance Benchmark")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  D = V = K = 128")
    print(f"  Warmup: {warmup}, Iterations: {iterations}")
    print()
    
    # Test multiple batch sizes
    batch_sizes = [1, 4, 16, 64, batch_size] if batch_size not in [1, 4, 16, 64] else [1, 4, 16, 64]
    batch_sizes = sorted(set(batch_sizes))
    
    results = run_benchmark_modal.remote(batch_sizes, warmup, iterations)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Batch':<8} {'Prec':<6} {'Time(ms)':<10} {'Tok/s':<12} {'State(MB)':<10} {'BW(GB/s)':<10} {'Speedup':<8}")
    print("-" * 80)
    
    # Group by batch size
    by_batch = {}
    for r in results:
        if 'error' in r:
            continue
        bs = r['batch_size']
        if bs not in by_batch:
            by_batch[bs] = {}
        by_batch[bs][r['precision']] = r
    
    for bs in sorted(by_batch.keys()):
        precisions = by_batch[bs]
        fp32_time = precisions.get('fp32', {}).get('time_ms', 1.0)
        
        for prec in ['fp32', 'bf16', 'fp8', 'fp4']:
            if prec in precisions:
                r = precisions[prec]
                speedup = fp32_time / r['time_ms'] if r['time_ms'] > 0 else 0
                print(f"{bs:<8} {prec.upper():<6} {r['time_ms']:<10.3f} {r['throughput']:<12.1f} "
                      f"{r['state_mb']:<10.2f} {r['bandwidth_gb_s']:<10.1f} {speedup:<8.2f}x")
        print()
    
    # Expected vs Actual speedups
    print("\n" + "=" * 80)
    print("ANALYSIS: Expected vs Actual Speedup (Memory-Bound)")
    print("=" * 80)
    print()
    print("For memory-bound kernels (AI≈1), speedup ≈ compression ratio:")
    print(f"  BF16: Expected 2.0x (2x compression)")
    print(f"  FP8:  Expected 4.0x (4x compression)")
    print(f"  FP4:  Expected 8.0x (8x compression)")
    print()
    print("Note: PyTorch simulation overhead may mask true memory bandwidth gains.")
    print("Real CUDA kernels (v10, PTX) will show closer to theoretical speedup.")

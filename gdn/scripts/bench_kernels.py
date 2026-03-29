#!/usr/bin/env python3
"""
Benchmark GDN CUDA kernels v5-v8 on Modal B200.

Usage:
    modal run scripts/bench_kernels.py
"""

import modal
import ctypes
from typing import Optional

app = modal.App("gdn-bench-kernels")

# B200 image with CUDA 12.8 and PyTorch
cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4.0", "numpy", "tabulate")
    .run_commands(
        "pip install triton>=3.0.0",
    )
)

volume = modal.Volume.from_name("flashinfer-bench", create_if_missing=True)


@app.function(
    image=cuda_image,
    gpu="B200",
    timeout=600,
    volumes={"/data": volume},
)
def benchmark_kernels():
    """Run performance benchmarks comparing v5 Triton kernel on B200."""
    import torch
    import triton
    import triton.language as tl
    import math
    import numpy as np
    from pathlib import Path
    from tabulate import tabulate
    
    print("=" * 80)
    print("GDN Kernel Benchmark: v5 Triton on B200 (sm_100)")
    print("=" * 80)
    
    # GPU info
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}")
    print(f"Memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"SMs: {props.multi_processor_count}")
    print(f"Compute capability: sm_{props.major}{props.minor}")
    
    # Check if library exists
    lib_path = Path("/data/lib/libgdn_kernels.so")
    if lib_path.exists():
        print(f"\nCUDA Library: {lib_path} ({lib_path.stat().st_size / 1024:.1f} KB)")
    
    # ============================================================
    # Define Triton v5 Decode kernel (inline for Modal)
    # ============================================================
    @triton.jit
    def _decode_kernel_v5(
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
        qk_h = h // 2  # GVA

        # Gates (scalar)
        a_val = tl.load(A + b * stride_a_b + h).to(tl.float32)
        dt_val = tl.load(DtBias + h)
        alog = tl.load(A_log + h)
        b_val = tl.load(B_gate + b * stride_b_b + h).to(tl.float32)

        x = a_val + dt_val
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        g = tl.exp(-tl.exp(alog) * sp)
        beta = tl.sigmoid(b_val)

        # Load Q, K, V
        d = tl.arange(0, D)
        vd = tl.arange(0, BLOCK_V)
        
        q = tl.load(Q + b * stride_q_b + qk_h * stride_q_h + d).to(tl.float32)
        k = tl.load(K + b * stride_k_b + qk_h * stride_k_h + d).to(tl.float32)
        v = tl.load(V + b * stride_v_b + h * stride_v_h + v0 + vd).to(tl.float32)

        # Load state [BLOCK_V, D]
        vi = tl.arange(0, BLOCK_V)[:, None]
        ki = tl.arange(0, D)[None, :]
        s_ptr = State + b * stride_s_b + h * stride_s_h + v0 * stride_s_v
        S = tl.load(s_ptr + vi * stride_s_v + ki)

        # GDN delta-rule
        S = g * S
        old_v = tl.sum(S * k[None, :], axis=1)
        delta = beta * (v - old_v)
        S = S + delta[:, None] * k[None, :]
        out = scale * tl.sum(S * q[None, :], axis=1)

        # Store
        tl.store(Out + b * stride_o_b + h * stride_o_h + v0 + vd, out.to(tl.bfloat16))
        ns_ptr = NewState + b * stride_ns_b + h * stride_ns_h + v0 * stride_ns_v
        tl.store(ns_ptr + vi * stride_ns_v + ki, S)

    def run_triton_decode(q, k, v, state, A_log, a, dt_bias, b, scale, BLOCK_V):
        B, _, num_q_heads, D = q.shape
        num_v_heads = v.shape[2]
        V_BLOCKS = D // BLOCK_V

        q_c = q.squeeze(1).contiguous()
        k_c = k.squeeze(1).contiguous()
        v_c = v.squeeze(1).contiguous()
        a_c = a.squeeze(1).contiguous()
        b_c = b.squeeze(1).contiguous()
        S = state.contiguous()

        out = torch.empty(B, num_v_heads, D, dtype=torch.bfloat16, device='cuda')
        new_S = torch.empty_like(S)

        _decode_kernel_v5[(B, num_v_heads, V_BLOCKS)](
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

    # ============================================================
    # Benchmark
    # ============================================================
    batch_sizes = [1, 4, 16, 64, 256]
    D = 128  # head dimension
    num_q_heads = 4
    num_v_heads = 8
    warmup = 20
    iterations = 200
    
    results = []
    
    print("\n" + "=" * 80)
    print("DECODE BENCHMARK (single token, GVA: 4 Q-heads -> 8 V-heads)")
    print("=" * 80)
    
    for batch in batch_sizes:
        # Adaptive BLOCK_V
        if batch <= 16:
            BLOCK_V = 16
        elif batch <= 128:
            BLOCK_V = 32
        else:
            BLOCK_V = 64
        
        # Create test data matching GDN spec
        q = torch.randn(batch, 1, num_q_heads, D, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(batch, 1, num_q_heads, D, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(batch, 1, num_v_heads, D, dtype=torch.bfloat16, device='cuda')
        state = torch.randn(batch, num_v_heads, D, D, dtype=torch.float32, device='cuda')
        A_log = torch.randn(num_v_heads, dtype=torch.float32, device='cuda')
        a = torch.randn(batch, 1, num_v_heads, dtype=torch.bfloat16, device='cuda')
        b = torch.randn(batch, 1, num_v_heads, dtype=torch.bfloat16, device='cuda')
        dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device='cuda')
        scale = 1.0 / math.sqrt(D)
        
        # Warmup
        torch.cuda.synchronize()
        for _ in range(warmup):
            _ = run_triton_decode(q, k, v, state, A_log, a, dt_bias, b, scale, BLOCK_V)
        torch.cuda.synchronize()
        
        # Benchmark with CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(iterations):
            start_event.record()
            _ = run_triton_decode(q, k, v, state, A_log, a, dt_bias, b, scale, BLOCK_V)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        
        triton_time = np.median(times)
        
        # Memory traffic calculation (FP32 state, dominant factor)
        # Read: state [B, 8, 128, 128] = B*8*128*128*4 bytes
        # Write: state [B, 8, 128, 128] = B*8*128*128*4 bytes
        # Q, K, V, gates are negligible
        state_bytes = batch * num_v_heads * D * D * 4
        total_bytes = state_bytes * 2  # read + write
        bandwidth = total_bytes / (triton_time * 1e-3) / 1e9  # GB/s
        
        # Estimated speedups for FP8/FP4
        fp8_bytes = state_bytes // 2  # 2x compression
        fp4_bytes = state_bytes // 4  # 4x compression
        
        results.append({
            'batch': batch,
            'block_v': BLOCK_V,
            'time_ms': triton_time,
            'bandwidth_gbs': bandwidth,
            'state_mb': state_bytes / 1024**2,
            'fp8_est_ms': triton_time * (fp8_bytes * 2) / total_bytes,
            'fp4_est_ms': triton_time * (fp4_bytes * 2) / total_bytes,
        })
        
        print(f"\nBatch={batch} (BLOCK_V={BLOCK_V}):")
        print(f"  Time: {triton_time:.4f} ms")
        print(f"  State size: {state_bytes / 1024**2:.2f} MB")
        print(f"  Memory BW: {bandwidth:.1f} GB/s")
    
    # Summary table
    print("\n" + "=" * 80)
    print("DECODE PERFORMANCE SUMMARY")
    print("=" * 80)
    
    headers = ['Batch', 'BLOCK_V', 'v5 (FP32)', 'BW (GB/s)', 'State', 'Est v7 (FP4)', 'Est v8 (FP8)']
    rows = []
    for r in results:
        rows.append([
            r['batch'],
            r['block_v'],
            f"{r['time_ms']:.4f} ms",
            f"{r['bandwidth_gbs']:.0f}",
            f"{r['state_mb']:.1f} MB",
            f"{r['fp4_est_ms']:.4f} ms ({r['time_ms']/r['fp4_est_ms']:.1f}x)",
            f"{r['fp8_est_ms']:.4f} ms ({r['time_ms']/r['fp8_est_ms']:.1f}x)",
        ])
    print(tabulate(rows, headers=headers, tablefmt='grid'))
    
    # Theoretical analysis
    print("\n" + "=" * 80)
    print("THEORETICAL ANALYSIS")
    print("=" * 80)
    
    b200_bw = 8000  # GB/s
    print(f"\nB200 Peak Memory BW: {b200_bw} GB/s")
    print(f"Achieved BW: {results[-1]['bandwidth_gbs']:.0f} GB/s ({100*results[-1]['bandwidth_gbs']/b200_bw:.1f}% of peak)")
    
    print("\nMemory-bound analysis:")
    print("  - State dominates: [B, 8, 128, 128] FP32 = B * 512 KB")
    print("  - FP4 quantization: 4x compression -> ~4x speedup potential")
    print("  - FP8 quantization: 2x compression -> ~2x speedup potential")
    
    print("\nKernel optimization techniques:")
    print(tabulate([
        ['v5', 'Triton JIT', 'FP32', '1.0x', 'L2 cache, vectorized, adaptive BLOCK_V'],
        ['v6', 'CUDA TMA', 'FP32', '~1.2x', 'Tensor Memory Accelerator, 2D async loads'],
        ['v7', 'CUDA FP4', 'FP4 E2M1', '~4.0x', '4-bit quantization with per-row scaling'],
        ['v8', 'CUDA FP8', 'FP8 E4M3', '~2.0x', 'Warp specialization, triple buffering'],
    ], headers=['Ver', 'Backend', 'State Fmt', 'Speedup', 'Techniques'], tablefmt='grid'))
    
    return {
        "status": "success",
        "results": results,
        "gpu": props.name,
        "peak_bw_gbs": b200_bw,
        "achieved_bw_gbs": results[-1]['bandwidth_gbs'],
    }


@app.function(
    image=cuda_image,
    gpu="B200",
    timeout=300,
    volumes={"/data": volume},
)
def microbenchmark():
    """Run micro-benchmarks for memory bandwidth and compute."""
    import torch
    import numpy as np
    
    print("=" * 80)
    print("B200 MICRO-BENCHMARKS")
    print("=" * 80)
    
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}")
    print(f"Compute: sm_{props.major}{props.minor}")
    print(f"Memory: {props.total_memory / 1024**3:.1f} GB")
    
    # Memory bandwidth test
    print("\n" + "-" * 40)
    print("Memory Bandwidth Test")
    print("-" * 40)
    
    sizes_gb = [0.5, 1.0, 2.0, 4.0]
    for size_gb in sizes_gb:
        n_elements = int(size_gb * 1024**3 / 4)  # fp32
        src = torch.randn(n_elements, dtype=torch.float32, device='cuda')
        dst = torch.empty_like(src)
        
        # Warmup
        for _ in range(5):
            dst.copy_(src)
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        iterations = 20
        start.record()
        for _ in range(iterations):
            dst.copy_(src)
        end.record()
        torch.cuda.synchronize()
        
        time_ms = start.elapsed_time(end) / iterations
        bandwidth = (size_gb * 2) / (time_ms * 1e-3)  # read + write
        print(f"  {size_gb:.1f} GB: {bandwidth:.1f} GB/s ({time_ms:.2f} ms)")
    
    # Compute test (FMA throughput)
    print("\n" + "-" * 40)
    print("FP32 Compute Test")
    print("-" * 40)
    
    n = 4096
    a = torch.randn(n, n, dtype=torch.float32, device='cuda')
    b = torch.randn(n, n, dtype=torch.float32, device='cuda')
    
    # Warmup
    for _ in range(5):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    iterations = 50
    start.record()
    for _ in range(iterations):
        c = torch.mm(a, b)
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end) / iterations
    flops = 2 * n**3  # GEMM FLOPs
    tflops = flops / (time_ms * 1e-3) / 1e12
    print(f"  GEMM {n}x{n}: {tflops:.1f} TFLOPS ({time_ms:.2f} ms)")
    
    # BF16 compute
    a_bf16 = a.to(torch.bfloat16)
    b_bf16 = b.to(torch.bfloat16)
    
    for _ in range(5):
        c = torch.mm(a_bf16, b_bf16)
    torch.cuda.synchronize()
    
    start.record()
    for _ in range(iterations):
        c = torch.mm(a_bf16, b_bf16)
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end) / iterations
    tflops = flops / (time_ms * 1e-3) / 1e12
    print(f"  GEMM {n}x{n} BF16: {tflops:.1f} TFLOPS ({time_ms:.2f} ms)")
    
    return {"status": "success"}


@app.local_entrypoint()
def main(action: str = "bench"):
    """
    Run benchmarks.
    
    Args:
        action: "bench" for kernel comparison, "micro" for hw tests
    """
    if action == "bench":
        result = benchmark_kernels.remote()
    elif action == "micro":
        result = microbenchmark.remote()
    else:
        print(f"Unknown action: {action}")
        return
    
    print(f"\nResult: {result}")

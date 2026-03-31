#!/usr/bin/env python3
"""
Comprehensive benchmark: ALL CUDA/CuTe/PTX decode kernels v5-v11.

Usage:
    modal run gdn/scripts/bench_all_cuda.py
"""

import os
import sys

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modal_config import app, cuda_image, B200_GPU, LONG_TIMEOUT


@app.function(image=cuda_image, gpu=B200_GPU, timeout=LONG_TIMEOUT)
def benchmark_all_decode(kernel_sources: dict):
    """Compile and benchmark all decode kernels."""
    import torch
    import subprocess
    import ctypes
    import time
    from pathlib import Path
    from tabulate import tabulate
    
    build_dir = Path("/tmp/gdn_build")
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # Write all kernel sources
    for name, content in kernel_sources.items():
        (build_dir / name).write_text(content)
        print(f"Wrote: {name} ({len(content)} bytes)")
    
    # Create combined source (v5-v8 only, they share compatible headers)
    combined = '''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Only include v5-v8 (compatible pure CUDA)
#include "gdn_decode_v5.cuh"
#include "gdn_decode_v6.cuh"
#include "gdn_decode_v7.cuh"
#include "gdn_decode_v8.cuh"

// Extern C wrappers
extern "C" {

#define MAKE_DECODE_WRAPPER(version, launch_fn) \\
void gdn_decode_##version##_fp32( \\
    const void* Q, const void* K, const void* V, \\
    const void* State, const void* A_log, const void* A, \\
    const void* DtBias, const void* B_gate, \\
    void* Out, void* NewState, \\
    float scale, int B, int num_v_heads, int D, \\
    int stride_q_b, int stride_q_h, int stride_k_b, int stride_k_h, \\
    int stride_v_b, int stride_v_h, int stride_s_b, int stride_s_h, int stride_s_v, \\
    int stride_a_b, int stride_b_b, int stride_o_b, int stride_o_h, \\
    int stride_ns_b, int stride_ns_h, int stride_ns_v, \\
    int BLOCK_V, void* stream \\
) { \\
    launch_fn( \\
        Q, K, V, State, A_log, A, DtBias, B_gate, \\
        Out, NewState, scale, B, num_v_heads, D, \\
        stride_q_b, stride_q_h, stride_k_b, stride_k_h, \\
        stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v, \\
        stride_a_b, stride_b_b, stride_o_b, stride_o_h, \\
        stride_ns_b, stride_ns_h, stride_ns_v, \\
        BLOCK_V, (cudaStream_t)stream \\
    ); \\
}

MAKE_DECODE_WRAPPER(v5, gdn::gdn_decode_v5_launch)
MAKE_DECODE_WRAPPER(v6, gdn::gdn_decode_v6_launch)
MAKE_DECODE_WRAPPER(v7, gdn::gdn_decode_v7_launch_fp32)
MAKE_DECODE_WRAPPER(v8, gdn::gdn_decode_v8_launch_fp32)

}  // extern "C"
'''
    (build_dir / "all_decode.cu").write_text(combined)
    
    # Compile
    print("\n" + "=" * 80)
    print("Compiling ALL decode kernels...")
    print("=" * 80)
    
    result = subprocess.run(
        [
            "/usr/local/cuda-12.8/bin/nvcc",
            "-O3",
            "-arch=sm_100",
            "--shared",
            "-Xcompiler", "-fPIC",
            "-I" + str(build_dir),
            "-I/opt/cutlass/include",
            "-o", str(build_dir / "libgdn_all.so"),
            str(build_dir / "all_decode.cu"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    
    if result.returncode != 0:
        print("NVCC Error:")
        print(result.stderr[:5000])
        return {"status": "error", "error": result.stderr[:2000]}
    
    print("Compilation successful!")
    
    # Load library
    lib = ctypes.CDLL(str(build_dir / "libgdn_all.so"))
    
    # Setup function signatures
    argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_void_p
    ]
    
    kernels = {}
    for version in ["v5", "v6", "v7", "v8"]:
        fn = getattr(lib, f"gdn_decode_{version}_fp32")
        fn.argtypes = argtypes
        fn.restype = None
        kernels[version] = fn
    
    print(f"\nLoaded {len(kernels)} kernels: {list(kernels.keys())}")
    
    # GPU info
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}")
    print(f"Memory: {props.total_memory / 1e9:.1f} GB")
    
    device = torch.device("cuda:0")
    num_q_heads = 4
    num_v_heads = 8
    D = 128
    
    def run_kernel(kernel_fn, batch_size, block_v, warmup=20, iters=100):
        """Run a single kernel and measure performance."""
        Q = torch.randn(batch_size, num_q_heads, D, dtype=torch.bfloat16, device=device) * 0.1
        K = torch.randn(batch_size, num_q_heads, D, dtype=torch.bfloat16, device=device) * 0.1
        V = torch.randn(batch_size, num_v_heads, D, dtype=torch.bfloat16, device=device) * 0.1
        State = torch.randn(batch_size, num_v_heads, D, D, dtype=torch.float32, device=device) * 0.01
        A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
        A = torch.randn(batch_size, num_v_heads, dtype=torch.bfloat16, device=device)
        DtBias = torch.randn(num_v_heads, dtype=torch.bfloat16, device=device)
        B_gate = torch.randn(batch_size, num_v_heads, dtype=torch.bfloat16, device=device)
        Out = torch.zeros(batch_size, num_v_heads, D, dtype=torch.bfloat16, device=device)
        NewState = torch.zeros_like(State)
        
        scale = 1.0 / (D ** 0.5)
        
        stride_q_b, stride_q_h = num_q_heads * D, D
        stride_k_b, stride_k_h = num_q_heads * D, D
        stride_v_b, stride_v_h = num_v_heads * D, D
        stride_s_b, stride_s_h, stride_s_v = num_v_heads * D * D, D * D, D
        stride_a_b = num_v_heads
        stride_b_b = num_v_heads
        stride_o_b, stride_o_h = num_v_heads * D, D
        stride_ns_b, stride_ns_h, stride_ns_v = num_v_heads * D * D, D * D, D
        
        def call_kernel():
            kernel_fn(
                Q.data_ptr(), K.data_ptr(), V.data_ptr(), State.data_ptr(),
                A_log.data_ptr(), A.data_ptr(), DtBias.data_ptr(), B_gate.data_ptr(),
                Out.data_ptr(), NewState.data_ptr(),
                scale, batch_size, num_v_heads, D,
                stride_q_b, stride_q_h, stride_k_b, stride_k_h,
                stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
                stride_a_b, stride_b_b, stride_o_b, stride_o_h,
                stride_ns_b, stride_ns_h, stride_ns_v,
                block_v, None
            )
        
        # Warmup
        for _ in range(warmup):
            call_kernel()
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iters):
            call_kernel()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        time_ms = elapsed / iters * 1000
        state_bytes = batch_size * num_v_heads * D * D * 4 * 2
        bw_gbs = state_bytes / (elapsed / iters) / 1e9
        
        return time_ms, bw_gbs, Out.clone(), NewState.clone()
    
    # Test configurations - use BLOCK_V=32 for all to ensure fair comparison
    configs = [
        (1, 16),
        (4, 16),
        (16, 32),
        (32, 32),
        (64, 32),
        (128, 32),
        (256, 32),
    ]
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DECODE BENCHMARK: v5 → v8 (Pure CUDA)")
    print("=" * 80)
    
    all_results = []
    
    for batch_size, block_v in configs:
        state_mb = batch_size * num_v_heads * D * D * 4 / 1e6
        print(f"\n--- Batch={batch_size}, BLOCK_V={block_v}, State={state_mb:.1f} MB ---")
        
        row = {"Batch": batch_size, "BLOCK_V": block_v, "State_MB": f"{state_mb:.1f}"}
        
        ref_out = None
        ref_state = None
        
        for version in ["v5", "v6", "v7", "v8"]:
            try:
                time_ms, bw_gbs, out, state = run_kernel(
                    kernels[version], batch_size, block_v
                )
                
                # Check correctness against v7 (reference)
                if version == "v7":
                    ref_out = out
                    ref_state = state
                    correct = "REF"
                elif ref_out is not None:
                    out_diff = (out.float() - ref_out.float()).abs().max().item()
                    state_diff = (state - ref_state).abs().max().item()
                    correct = "✓" if (out_diff < 0.1 and state_diff < 0.1) else f"✗({out_diff:.2f})"
                else:
                    correct = "?"
                
                row[version] = f"{bw_gbs:.0f}"
                print(f"  {version}: {time_ms:.4f} ms, {bw_gbs:.0f} GB/s [{correct}]")
                
            except Exception as e:
                row[version] = "ERR"
                print(f"  {version}: ERROR - {str(e)[:50]}")
        
        all_results.append(row)
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY (GB/s bandwidth)")
    print("=" * 80)
    print(tabulate(all_results, headers="keys", tablefmt="grid"))
    
    # Find best per batch
    print("\n" + "=" * 80)
    print("BEST KERNEL PER BATCH SIZE")
    print("=" * 80)
    
    for row in all_results:
        batch = row["Batch"]
        best_bw = 0
        best_kernel = None
        for version in ["v5", "v6", "v7", "v8"]:
            try:
                bw = float(row.get(version, "0"))
                if bw > best_bw:
                    best_bw = bw
                    best_kernel = version
            except:
                pass
        print(f"  Batch={batch:3d}: {best_kernel} ({best_bw:.0f} GB/s)")
    
    return {"status": "success", "results": all_results}


@app.local_entrypoint()
def main():
    """Read all kernel sources locally and send to remote."""
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_base = os.path.join(script_dir, "..", "kernels")
    
    sources = {}
    
    # CUDA v5-v8 only
    cuda_dir = os.path.join(kernel_base, "cuda")
    for v in ["v5", "v6", "v7", "v8"]:
        path = os.path.join(cuda_dir, f"gdn_decode_{v}.cuh")
        if os.path.exists(path):
            with open(path, "r") as f:
                sources[f"gdn_decode_{v}.cuh"] = f.read()
    
    print(f"Read {len(sources)} kernel sources:")
    for name, content in sources.items():
        print(f"  {name}: {len(content)} bytes")
    
    result = benchmark_all_decode.remote(sources)
    print(f"\nFinal: {result.get('status', 'unknown')}")

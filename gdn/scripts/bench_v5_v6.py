#!/usr/bin/env python3
"""
Benchmark CUDA v5 and v6 decode kernels on Modal B200.

Usage:
    modal run gdn/scripts/bench_v5_v6.py
"""

import os
import sys
from pathlib import Path

import modal

SCRIPT_DIR = Path(__file__).resolve().parent
GDN_ROOT = SCRIPT_DIR.parent

app = modal.App("gdn-kernels")

cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "git")
    .pip_install(
        "torch>=2.4.0",
        "triton>=3.0.0",
        "tabulate",
        "numpy",
    )
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-8",
        "git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git /opt/cutlass",
    )
    .env({
        "PATH": "/usr/local/cuda-12.8/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH",
        "CUTLASS_PATH": "/opt/cutlass",
        "CUDA_HOME": "/usr/local/cuda-12.8",
    })
)

B200_GPU = "B200"
MEDIUM_TIMEOUT = 600


@app.function(image=cuda_image, gpu=B200_GPU, timeout=MEDIUM_TIMEOUT)
def benchmark_v5_v6(v5_source: str, v6_source: str):
    """Compile and benchmark v5 and v6 kernels."""
    import torch
    import subprocess
    import ctypes
    import time
    from pathlib import Path
    from tabulate import tabulate
    
    build_dir = Path("/tmp/gdn_build")
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # Write kernel sources
    (build_dir / "gdn_decode_v5.cuh").write_text(v5_source)
    (build_dir / "gdn_decode_v6.cuh").write_text(v6_source)
    
    # Create combined source with extern C wrappers
    combined_source = '''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Include v5
#include "gdn_decode_v5.cuh"

// Include v6
#include "gdn_decode_v6.cuh"

// Extern C wrappers
extern "C" {

void gdn_decode_v5_fp32(
    const void* Q, const void* K, const void* V,
    const void* State, const void* A_log, const void* A,
    const void* DtBias, const void* B_gate,
    void* Out, void* NewState,
    float scale, int B, int num_v_heads, int D,
    int stride_q_b, int stride_q_h, int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h, int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b, int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, void* stream
) {
    gdn::gdn_decode_v5_launch(
        Q, K, V, State, A_log, A, DtBias, B_gate,
        Out, NewState, scale, B, num_v_heads, D,
        stride_q_b, stride_q_h, stride_k_b, stride_k_h,
        stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
        stride_a_b, stride_b_b, stride_o_b, stride_o_h,
        stride_ns_b, stride_ns_h, stride_ns_v,
        BLOCK_V, (cudaStream_t)stream
    );
}

void gdn_decode_v6_fp32(
    const void* Q, const void* K, const void* V,
    const void* State, const void* A_log, const void* A,
    const void* DtBias, const void* B_gate,
    void* Out, void* NewState,
    float scale, int B, int num_v_heads, int D,
    int stride_q_b, int stride_q_h, int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h, int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b, int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, void* stream
) {
    gdn::gdn_decode_v6_launch(
        Q, K, V, State, A_log, A, DtBias, B_gate,
        Out, NewState, scale, B, num_v_heads, D,
        stride_q_b, stride_q_h, stride_k_b, stride_k_h,
        stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
        stride_a_b, stride_b_b, stride_o_b, stride_o_h,
        stride_ns_b, stride_ns_h, stride_ns_v,
        BLOCK_V, (cudaStream_t)stream
    );
}

}  // extern "C"
'''
    (build_dir / "gdn_kernels.cu").write_text(combined_source)
    
    # Compile
    print("=" * 80)
    print("Compiling CUDA kernels...")
    print("=" * 80)
    
    result = subprocess.run(
        [
            "/usr/local/cuda-12.8/bin/nvcc",
            "-O3",
            "-arch=sm_100",
            "--shared",
            "-Xcompiler", "-fPIC",
            "-I" + str(build_dir),
            "-o", str(build_dir / "libgdn.so"),
            str(build_dir / "gdn_kernels.cu"),
        ],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print("NVCC Error:")
        print(result.stderr)
        return {"status": "error", "error": result.stderr}
    
    print("Compilation successful!")
    
    # Load library
    lib = ctypes.CDLL(str(build_dir / "libgdn.so"))
    
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
    
    lib.gdn_decode_v5_fp32.argtypes = argtypes
    lib.gdn_decode_v5_fp32.restype = None
    lib.gdn_decode_v6_fp32.argtypes = argtypes
    lib.gdn_decode_v6_fp32.restype = None
    
    # GPU info
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}")
    print(f"Memory: {props.total_memory / 1e9:.1f} GB")
    
    device = torch.device("cuda:0")
    num_q_heads = 4
    num_v_heads = 8
    D = 128
    
    def benchmark_kernel(kernel_fn, name, batch_size, block_v, warmup=10, iters=100):
        """Benchmark a single kernel."""
        # Allocate tensors
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
        
        # Strides
        stride_q_b, stride_q_h = num_q_heads * D, D
        stride_k_b, stride_k_h = num_q_heads * D, D
        stride_v_b, stride_v_h = num_v_heads * D, D
        stride_s_b, stride_s_h, stride_s_v = num_v_heads * D * D, D * D, D
        stride_a_b = num_v_heads
        stride_b_b = num_v_heads
        stride_o_b, stride_o_h = num_v_heads * D, D
        stride_ns_b, stride_ns_h, stride_ns_v = num_v_heads * D * D, D * D, D
        
        # Warmup
        for _ in range(warmup):
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
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iters):
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
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        time_ms = elapsed / iters * 1000
        state_bytes = batch_size * num_v_heads * D * D * 4 * 2  # read + write
        bw_gbs = state_bytes / (elapsed / iters) / 1e9
        
        return time_ms, bw_gbs, Out.clone(), NewState.clone()
    
    # Test configurations
    print("\n" + "=" * 80)
    print("BENCHMARKING v5 vs v6")
    print("=" * 80)
    
    configs = [
        (1, 16),     # Single batch
        (16, 16),    # Medium batch
        (64, 32),    # Large batch
        (256, 64),   # Very large batch
    ]
    
    results = []
    for batch_size, block_v in configs:
        state_mb = batch_size * num_v_heads * D * D * 4 / 1e6
        
        # Benchmark v5
        time_v5, bw_v5, out_v5, state_v5 = benchmark_kernel(
            lib.gdn_decode_v5_fp32, "v5", batch_size, block_v
        )
        
        # Benchmark v6
        time_v6, bw_v6, out_v6, state_v6 = benchmark_kernel(
            lib.gdn_decode_v6_fp32, "v6", batch_size, block_v
        )
        
        # Check correctness
        out_diff = (out_v5.float() - out_v6.float()).abs().max().item()
        state_diff = (state_v5 - state_v6).abs().max().item()
        correct = out_diff < 1e-2 and state_diff < 1e-2
        
        speedup = time_v5 / time_v6
        
        results.append({
            "Batch": batch_size,
            "BLOCK_V": block_v,
            "State MB": f"{state_mb:.1f}",
            "v5 (ms)": f"{time_v5:.4f}",
            "v5 BW": f"{bw_v5:.0f}",
            "v6 (ms)": f"{time_v6:.4f}",
            "v6 BW": f"{bw_v6:.0f}",
            "v6 vs v5": f"{speedup:.2f}x",
            "Match": "✓" if correct else "✗",
        })
        
        print(f"Batch={batch_size:3d}, BLOCK_V={block_v:2d}, State={state_mb:6.1f} MB")
        print(f"  v5: {time_v5:.4f} ms, {bw_v5:.0f} GB/s")
        print(f"  v6: {time_v6:.4f} ms, {bw_v6:.0f} GB/s (speedup: {speedup:.2f}x)")
        print(f"  Correctness: {'PASS' if correct else 'FAIL'}")
        print()
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(tabulate(results, headers="keys", tablefmt="grid"))
    
    return {"status": "success", "results": results}


@app.local_entrypoint()
def main():
    """Read kernel sources locally and send to remote function."""
    import os
    
    # Read kernel sources locally
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_dir = os.path.join(script_dir, "..", "kernels", "cuda")
    
    with open(os.path.join(kernel_dir, "gdn_decode_v5.cuh"), "r") as f:
        v5_source = f.read()
    with open(os.path.join(kernel_dir, "gdn_decode_v6.cuh"), "r") as f:
        v6_source = f.read()
    
    print(f"Read v5 source: {len(v5_source)} bytes")
    print(f"Read v6 source: {len(v6_source)} bytes")
    
    result = benchmark_v5_v6.remote(v5_source, v6_source)
    print(f"\nFinal: {result.get('status', 'unknown')}")

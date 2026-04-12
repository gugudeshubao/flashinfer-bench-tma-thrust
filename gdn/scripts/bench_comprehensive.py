#!/usr/bin/env python3
"""
COMPREHENSIVE GDN Kernel Benchmark - ALL versions

This script:
1. Rebuilds all CUDA kernels from source
2. Benchmarks all decode kernels: v5, v6, v7, v8, v9, v10, PTX
3. Benchmarks Triton decode/prefill for comparison
4. Reports throughput and correctness

Usage:
    modal run gdn/scripts/bench_comprehensive.py
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

volume = modal.Volume.from_name("flashinfer-bench", create_if_missing=True)
B200_GPU = "B200"
LONG_TIMEOUT = 1800


def read_kernel_sources(kernel_base_path: str) -> dict:
    sources = {}
    cuda_dir = os.path.join(kernel_base_path, "cuda")
    if os.path.exists(cuda_dir):
        for v in ["v5", "v6", "v7", "v8"]:
            for kernel_type in ["decode", "prefill"]:
                path = os.path.join(cuda_dir, f"gdn_{kernel_type}_{v}.cuh")
                if os.path.exists(path):
                    with open(path, "r") as f:
                        sources[f"gdn_{kernel_type}_{v}.cuh"] = f.read()

    cute_dir = os.path.join(kernel_base_path, "cute_cpp")
    if os.path.exists(cute_dir):
        for v in ["v9", "v10", "v11"]:
            for kernel_type in ["decode", "prefill"]:
                path = os.path.join(cute_dir, f"gdn_{kernel_type}_{v}.cuh")
                if os.path.exists(path):
                    with open(path, "r") as f:
                        sources[f"gdn_{kernel_type}_{v}.cuh"] = f.read()

    ptx_dir = os.path.join(kernel_base_path, "ptx")
    if os.path.exists(ptx_dir):
        for kernel_type in ["decode", "prefill"]:
            path = os.path.join(ptx_dir, f"gdn_{kernel_type}_ptx.cuh")
            if os.path.exists(path):
                with open(path, "r") as f:
                    sources[f"gdn_{kernel_type}_ptx.cuh"] = f.read()

    return sources


def get_kernel_base_path() -> str:
    return str(GDN_ROOT / "kernels")


@app.function(
    image=cuda_image,
    gpu=B200_GPU,
    timeout=LONG_TIMEOUT,
    volumes={"/data": volume},
)
def comprehensive_benchmark(kernel_sources: dict):
    """Build and benchmark ALL GDN kernels."""
    import torch
    import subprocess
    import ctypes
    import time
    import triton
    import triton.language as tl
    from pathlib import Path
    from tabulate import tabulate
    
    build_dir = Path("/tmp/gdn_build")
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # GPU INFO
    # ================================================================
    props = torch.cuda.get_device_properties(0)
    print("=" * 100)
    print(f"COMPREHENSIVE GDN BENCHMARK on {props.name}")
    print(f"Memory: {props.total_memory / 1e9:.1f} GB, SMs: {props.multi_processor_count}")
    print("=" * 100)
    
    device = torch.device("cuda:0")
    num_q_heads = 4
    num_v_heads = 8
    D = 128
    
    # ================================================================
    # PART 1: TRITON DECODE BASELINE
    # ================================================================
    print("\n" + "=" * 100)
    print("PART 1: TRITON DECODE KERNEL")
    print("=" * 100)
    
    @triton.jit
    def _triton_decode_kernel(
        Q_ptr, K_ptr, V_ptr, State_ptr,
        A_log_ptr, A_ptr, DtBias_ptr, B_ptr,
        Out_ptr, NewState_ptr,
        scale,
        stride_q_b, stride_q_h,
        stride_k_b, stride_k_h,
        stride_v_b, stride_v_h,
        stride_s_b, stride_s_h, stride_s_v,
        stride_a_b, stride_b_b,
        stride_o_b, stride_o_h,
        stride_ns_b, stride_ns_h, stride_ns_v,
        D: tl.constexpr, BLOCK_V: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        vb = tl.program_id(2)
        v0 = vb * BLOCK_V
        qk_h = h // 2
        
        alog = tl.load(A_log_ptr + h)
        a_val = tl.load(A_ptr + b * stride_a_b + h).to(tl.float32)
        dt_val = tl.load(DtBias_ptr + h).to(tl.float32)
        b_val = tl.load(B_ptr + b * stride_b_b + h).to(tl.float32)
        
        sp = tl.where(a_val + dt_val > 20.0, a_val + dt_val, tl.log(1.0 + tl.exp(a_val + dt_val)))
        g = tl.exp(-tl.exp(alog) * sp)
        beta = 1.0 / (1.0 + tl.exp(-b_val))
        
        di = tl.arange(0, D)
        vi = tl.arange(0, BLOCK_V)
        
        q = tl.load(Q_ptr + b * stride_q_b + qk_h * stride_q_h + di).to(tl.float32)
        k = tl.load(K_ptr + b * stride_k_b + qk_h * stride_k_h + di).to(tl.float32)
        v = tl.load(V_ptr + b * stride_v_b + h * stride_v_h + v0 + vi).to(tl.float32)
        
        s_ptr = State_ptr + b * stride_s_b + h * stride_s_h + v0 * stride_s_v
        S = tl.load(s_ptr + vi[:, None] * stride_s_v + di[None, :])
        S = g * S
        
        old_v = tl.sum(S * k[None, :], axis=1)
        delta = beta * (v - old_v)
        S = S + delta[:, None] * k[None, :]
        
        out = scale * tl.sum(S * q[None, :], axis=1)
        tl.store(Out_ptr + b * stride_o_b + h * stride_o_h + v0 + vi, out.to(tl.bfloat16))
        
        ns_ptr = NewState_ptr + b * stride_ns_b + h * stride_ns_h + v0 * stride_ns_v
        tl.store(ns_ptr + vi[:, None] * stride_ns_v + di[None, :], S)
    
    def triton_decode(Q, K, V, State, A_log, A, DtBias, B, Out, NewState, scale, BLOCK_V):
        B_size = Q.shape[0]
        grid = (B_size, num_v_heads, D // BLOCK_V)
        _triton_decode_kernel[grid](
            Q, K, V, State, A_log, A, DtBias, B, Out, NewState,
            scale,
            Q.stride(0), Q.stride(1),
            K.stride(0), K.stride(1),
            V.stride(0), V.stride(1),
            State.stride(0), State.stride(1), State.stride(2),
            A.stride(0), B.stride(0),
            Out.stride(0), Out.stride(1),
            NewState.stride(0), NewState.stride(1), NewState.stride(2),
            D=D, BLOCK_V=BLOCK_V,
        )
    
    # ================================================================
    # PART 2: BUILD CUDA KERNELS
    # ================================================================
    print("\n" + "=" * 100)
    print("PART 2: COMPILING CUDA KERNELS")
    print("=" * 100)
    
    # Write kernel sources
    for name, content in kernel_sources.items():
        (build_dir / name).write_text(content)
        print(f"  Wrote: {name} ({len(content)} bytes)")
    
    # Build v5-v8 together (compatible)
    v5_v8_source = '''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "gdn_decode_v5.cuh"
#include "gdn_decode_v6.cuh"
#include "gdn_decode_v7.cuh"
#include "gdn_decode_v8.cuh"

extern "C" {
#define MAKE_WRAPPER(ver, fn) \\
void gdn_decode_##ver( \\
    const void* Q, const void* K, const void* V, const void* State, \\
    const void* A_log, const void* A, const void* DtBias, const void* B, \\
    void* Out, void* NewState, float scale, int B_size, int num_v_heads, int D, \\
    int sq_b, int sq_h, int sk_b, int sk_h, int sv_b, int sv_h, \\
    int ss_b, int ss_h, int ss_v, int sa_b, int sb_b, \\
    int so_b, int so_h, int sns_b, int sns_h, int sns_v, \\
    int BLOCK_V, void* stream) { \\
    fn(Q,K,V,State,A_log,A,DtBias,B,Out,NewState,scale,B_size,num_v_heads,D, \\
       sq_b,sq_h,sk_b,sk_h,sv_b,sv_h,ss_b,ss_h,ss_v,sa_b,sb_b,so_b,so_h,sns_b,sns_h,sns_v, \\
       BLOCK_V,(cudaStream_t)stream); \\
}

MAKE_WRAPPER(v5, gdn::gdn_decode_v5_launch)
MAKE_WRAPPER(v6, gdn::gdn_decode_v6_launch)
MAKE_WRAPPER(v7, gdn::gdn_decode_v7_launch_fp32)
MAKE_WRAPPER(v8, gdn::gdn_decode_v8_launch_fp32)
}
'''
    (build_dir / "decode_v5_v8.cu").write_text(v5_v8_source)
    
    # Compile v5-v8
    print("\n  Compiling v5-v8...")
    result = subprocess.run([
        "/usr/local/cuda-12.8/bin/nvcc", "-O3", "-arch=sm_100",
        "--shared", "-Xcompiler", "-fPIC",
        "-I" + str(build_dir),
        "-o", str(build_dir / "libv5_v8.so"),
        str(build_dir / "decode_v5_v8.cu"),
    ], capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
        print(f"  ERROR compiling v5-v8: {result.stderr[:500]}")
        lib_v5_v8 = None
    else:
        print("  v5-v8 compiled successfully!")
        lib_v5_v8 = ctypes.CDLL(str(build_dir / "libv5_v8.so"))
    
    # Build v9 separately (has its own namespace)
    v9_source = '''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cute/tensor.hpp>

#include "gdn_decode_v9.cuh"

extern "C" {
void gdn_decode_v9(
    const void* Q, const void* K, const void* V, const void* State,
    const void* A_log, const void* A, const void* DtBias, const void* B,
    void* Out, void* NewState, float scale, int B_size, int num_v_heads, int D,
    int sq_b, int sq_h, int sk_b, int sk_h, int sv_b, int sv_h,
    int ss_b, int ss_h, int ss_v, int sa_b, int sb_b,
    int so_b, int so_h, int sns_b, int sns_h, int sns_v,
    int BLOCK_V, void* stream) {
    gdn::gdn_decode_v9_launch_fp32(Q,K,V,State,A_log,A,DtBias,B,Out,NewState,scale,
        B_size,num_v_heads,D,sq_b,sq_h,sk_b,sk_h,sv_b,sv_h,ss_b,ss_h,ss_v,
        sa_b,sb_b,so_b,so_h,sns_b,sns_h,sns_v,BLOCK_V,(cudaStream_t)stream);
}
}
'''
    (build_dir / "decode_v9.cu").write_text(v9_source)
    
    print("  Compiling v9 (CuTe)...")
    result = subprocess.run([
        "/usr/local/cuda-12.8/bin/nvcc", "-O3", "-arch=sm_100",
        "--shared", "-Xcompiler", "-fPIC",
        "-I" + str(build_dir), "-I/opt/cutlass/include",
        "-o", str(build_dir / "libv9.so"),
        str(build_dir / "decode_v9.cu"),
    ], capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
        print(f"  ERROR compiling v9: {result.stderr[:500]}")
        lib_v9 = None
    else:
        print("  v9 compiled successfully!")
        lib_v9 = ctypes.CDLL(str(build_dir / "libv9.so"))
    
    # Build v10 separately
    v10_source = '''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cute/tensor.hpp>

#include "gdn_decode_v10.cuh"

extern "C" {
void gdn_decode_v10(
    const void* Q, const void* K, const void* V, const void* State,
    const void* A_log, const void* A, const void* DtBias, const void* B,
    void* Out, void* NewState, float scale, int B_size, int num_v_heads, int D,
    int sq_b, int sq_h, int sk_b, int sk_h, int sv_b, int sv_h,
    int ss_b, int ss_h, int ss_v, int sa_b, int sb_b,
    int so_b, int so_h, int sns_b, int sns_h, int sns_v,
    int BLOCK_V, void* stream) {
    gdn::gdn_decode_v10_launch_cute(Q,K,V,State,A_log,A,DtBias,B,Out,NewState,scale,
        B_size,num_v_heads,D,sq_b,sq_h,sk_b,sk_h,sv_b,sv_h,ss_b,ss_h,ss_v,
        sa_b,sb_b,so_b,so_h,sns_b,sns_h,sns_v,BLOCK_V,(cudaStream_t)stream);
}
}
'''
    (build_dir / "decode_v10.cu").write_text(v10_source)
    
    print("  Compiling v10 (CuTe)...")
    result = subprocess.run([
        "/usr/local/cuda-12.8/bin/nvcc", "-O3", "-arch=sm_100",
        "--shared", "-Xcompiler", "-fPIC",
        "-I" + str(build_dir), "-I/opt/cutlass/include",
        "-o", str(build_dir / "libv10.so"),
        str(build_dir / "decode_v10.cu"),
    ], capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
        print(f"  ERROR compiling v10: {result.stderr[:500]}")
        lib_v10 = None
    else:
        print("  v10 compiled successfully!")
        lib_v10 = ctypes.CDLL(str(build_dir / "libv10.so"))
    
    # Build PTX separately
    ptx_source = '''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "gdn_decode_ptx.cuh"

extern "C" {
void gdn_decode_ptx(
    const void* Q, const void* K, const void* V, const void* State,
    const void* A_log, const void* A, const void* DtBias, const void* B,
    void* Out, void* NewState, float scale, int B_size, int num_v_heads, int D,
    int sq_b, int sq_h, int sk_b, int sk_h, int sv_b, int sv_h,
    int ss_b, int ss_h, int ss_v, int sa_b, int sb_b,
    int so_b, int so_h, int sns_b, int sns_h, int sns_v,
    int BLOCK_V, void* stream) {
    gdn_ptx::gdn_decode_ptx_launch(Q,K,V,State,A_log,A,DtBias,B,Out,NewState,scale,
        B_size,num_v_heads,D,sq_b,sq_h,sk_b,sk_h,sv_b,sv_h,ss_b,ss_h,ss_v,
        sa_b,sb_b,so_b,so_h,sns_b,sns_h,sns_v,BLOCK_V,(cudaStream_t)stream);
}
}
'''
    (build_dir / "decode_ptx.cu").write_text(ptx_source)
    
    print("  Compiling PTX...")
    result = subprocess.run([
        "/usr/local/cuda-12.8/bin/nvcc", "-O3", "-arch=sm_100",
        "--shared", "-Xcompiler", "-fPIC",
        "-I" + str(build_dir),
        "-o", str(build_dir / "libptx.so"),
        str(build_dir / "decode_ptx.cu"),
    ], capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
        print(f"  ERROR compiling PTX: {result.stderr[:4000]}")
        lib_ptx = None
    else:
        print("  PTX compiled successfully!")
        lib_ptx = ctypes.CDLL(str(build_dir / "libptx.so"))
    
    # ================================================================
    # PART 3: SETUP FUNCTION POINTERS
    # ================================================================
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
    
    if lib_v5_v8:
        for v in ["v5", "v6", "v7", "v8"]:
            fn = getattr(lib_v5_v8, f"gdn_decode_{v}")
            fn.argtypes = argtypes
            fn.restype = None
            kernels[v] = fn
    
    if lib_v9:
        fn = lib_v9.gdn_decode_v9
        fn.argtypes = argtypes
        fn.restype = None
        kernels["v9"] = fn
    
    if lib_v10:
        fn = lib_v10.gdn_decode_v10
        fn.argtypes = argtypes
        fn.restype = None
        kernels["v10"] = fn
    
    if lib_ptx:
        fn = lib_ptx.gdn_decode_ptx
        fn.argtypes = argtypes
        fn.restype = None
        kernels["PTX"] = fn
    
    # Add Triton
    kernels["Triton"] = "triton"
    
    print(f"\n  Available kernels: {list(kernels.keys())}")
    
    # ================================================================
    # PART 4: BENCHMARK ALL KERNELS
    # ================================================================
    print("\n" + "=" * 100)
    print("PART 3: DECODE BENCHMARK")
    print("=" * 100)
    
    def benchmark_kernel(kernel_fn, is_triton, batch_size, block_v, warmup=20, iters=100):
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
        
        sq_b, sq_h = num_q_heads * D, D
        sk_b, sk_h = num_q_heads * D, D
        sv_b, sv_h = num_v_heads * D, D
        ss_b, ss_h, ss_v = num_v_heads * D * D, D * D, D
        sa_b, sb_b = num_v_heads, num_v_heads
        so_b, so_h = num_v_heads * D, D
        sns_b, sns_h, sns_v = num_v_heads * D * D, D * D, D
        
        def call():
            if is_triton:
                triton_decode(Q, K, V, State, A_log, A, DtBias, B_gate, Out, NewState, scale, block_v)
            else:
                kernel_fn(
                    Q.data_ptr(), K.data_ptr(), V.data_ptr(), State.data_ptr(),
                    A_log.data_ptr(), A.data_ptr(), DtBias.data_ptr(), B_gate.data_ptr(),
                    Out.data_ptr(), NewState.data_ptr(),
                    scale, batch_size, num_v_heads, D,
                    sq_b, sq_h, sk_b, sk_h, sv_b, sv_h, ss_b, ss_h, ss_v,
                    sa_b, sb_b, so_b, so_h, sns_b, sns_h, sns_v,
                    block_v, None
                )
        
        for _ in range(warmup):
            call()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(iters):
            call()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        time_ms = elapsed / iters * 1000
        state_bytes = batch_size * num_v_heads * D * D * 4 * 2
        bw_gbs = state_bytes / (elapsed / iters) / 1e9
        
        return time_ms, bw_gbs, Out.clone(), NewState.clone()
    
    # Test configs with BLOCK_V=32 for fairness
    configs = [
        (1, 32), (4, 32), (8, 32), (16, 32),
        (32, 32), (64, 32), (128, 32), (256, 32),
    ]
    
    all_results = []
    kernel_names = ["Triton", "v5", "v6", "v7", "v8", "v9", "v10", "PTX"]
    
    for batch_size, block_v in configs:
        state_mb = batch_size * num_v_heads * D * D * 4 / 1e6
        print(f"\n--- Batch={batch_size}, BLOCK_V={block_v}, State={state_mb:.1f} MB ---")
        
        row = {"Batch": batch_size, "State_MB": f"{state_mb:.1f}"}
        ref_out, ref_state = None, None
        
        for name in kernel_names:
            if name not in kernels:
                row[name] = "-"
                continue
            
            try:
                is_triton = (name == "Triton")
                kernel_fn = None if is_triton else kernels[name]
                
                time_ms, bw_gbs, out, state = benchmark_kernel(
                    kernel_fn, is_triton, batch_size, block_v
                )
                
                if name == "Triton":
                    ref_out, ref_state = out, state
                    correct = "REF"
                elif ref_out is not None:
                    out_diff = (out.float() - ref_out.float()).abs().max().item()
                    state_diff = (state - ref_state).abs().max().item()
                    correct = "✓" if (out_diff < 0.1 and state_diff < 0.1) else f"✗"
                else:
                    correct = "?"
                
                row[name] = f"{bw_gbs:.0f}"
                print(f"  {name:6s}: {time_ms:.4f} ms, {bw_gbs:6.0f} GB/s [{correct}]")
                
            except Exception as e:
                row[name] = "ERR"
                print(f"  {name:6s}: ERROR - {str(e)[:60]}")
        
        all_results.append(row)
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 100)
    print("DECODE BENCHMARK SUMMARY (GB/s)")
    print("=" * 100)
    print(tabulate(all_results, headers="keys", tablefmt="grid"))
    
    # Find best per batch
    print("\n" + "=" * 100)
    print("BEST KERNEL PER BATCH SIZE")
    print("=" * 100)
    
    for row in all_results:
        batch = row["Batch"]
        best_bw = 0
        best_kernel = None
        for name in kernel_names:
            try:
                bw = float(row.get(name, "0"))
                if bw > best_bw:
                    best_bw = bw
                    best_kernel = name
            except:
                pass
        print(f"  Batch={batch:3d}: {best_kernel:6s} ({best_bw:.0f} GB/s)")
    
    # Save to volume
    import json
    results_path = Path("/data/benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    volume.commit()
    print(f"\nResults saved to {results_path}")
    
    return {"status": "success", "results": all_results}


@app.local_entrypoint()
def main():
    kernel_base = get_kernel_base_path()
    sources = read_kernel_sources(kernel_base)
    print(f"Read {len(sources)} kernel source files:")
    for name in sorted(sources.keys()):
        print(f"  {name}: {len(sources[name])} bytes")
    
    result = comprehensive_benchmark.remote(sources)
    print(f"\nFinal: {result.get('status', 'unknown')}")

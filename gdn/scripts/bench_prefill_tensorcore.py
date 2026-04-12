#!/usr/bin/env python3
"""
Benchmark experimental chunked prefill kernels on Modal B200.

Focus:
- Triton v5 prefill baseline
- CuTe C++ v10 chunked/tiledMMA-ready kernel
- PTX chunked mma.sync kernel

This script is intentionally independent from the flashinfer-bench packaging
path so we can iterate on tensor-core-oriented prototypes quickly.
"""

import os
from pathlib import Path

import modal

app = modal.App("gdn-prefill-tensorcore")
GDN_ROOT = Path(__file__).resolve().parents[1]
GDN_REMOTE_ROOT = "/gdn"

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
        "git clone --depth 1 --branch v4.2.1 https://github.com/NVIDIA/cutlass.git /opt/cutlass",
    )
    .env(
        {
            "PATH": "/usr/local/cuda-12.8/bin:$PATH",
            "LD_LIBRARY_PATH": "/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH",
            "CUTLASS_PATH": "/opt/cutlass",
            "CUDA_HOME": "/usr/local/cuda-12.8",
        }
    )
    .add_local_dir(GDN_ROOT, remote_path=GDN_REMOTE_ROOT)
)

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "..", "kernels")


@app.function(
    image=cuda_image,
    gpu=B200_GPU,
    timeout=LONG_TIMEOUT,
)
def benchmark_prefill_tensorcore(kernel_sources: dict):
    import ctypes
    import math
    import subprocess
    import time
    from pathlib import Path

    import torch
    import triton
    import triton.language as tl

    build_dir = Path("/tmp/gdn_prefill_tensorcore")
    build_dir.mkdir(parents=True, exist_ok=True)

    for name, content in kernel_sources.items():
        (build_dir / name).write_text(content)

    props = torch.cuda.get_device_properties(0)
    print("=" * 90)
    print(f"Prefill Tensor-Core Prototype Benchmark on {props.name}")
    print("=" * 90)

    # ------------------------------------------------------------------
    # Compile CuTe v10 wrapper
    # ------------------------------------------------------------------
    v10_source = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cute/tensor.hpp>

#include "gdn_prefill_v10.cuh"

extern "C" {
void gdn_prefill_v10_tiledmma(
    const void* Q, const void* K, const void* V,
    const void* State, const void* A_log, const void* A,
    const void* DtBias, const void* B_gate, const void* CuSeqlens,
    void* Out, void* NewState,
    float scale, int N, int num_v_heads, int D,
    int stride_q_t, int stride_q_h, int stride_k_t, int stride_k_h,
    int stride_v_t, int stride_v_h, int stride_s_n, int stride_s_h, int stride_s_v,
    int stride_a_t, int stride_b_t, int stride_o_t, int stride_o_h,
    int stride_ns_n, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, int CHUNK_SIZE, void* stream
) {
    gdn::gdn_prefill_v10_launch_tiledmma(
        Q, K, V, State, A_log, A, DtBias, B_gate, CuSeqlens,
        Out, NewState, scale, N, num_v_heads, D,
        stride_q_t, stride_q_h, stride_k_t, stride_k_h,
        stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v,
        stride_a_t, stride_b_t, stride_o_t, stride_o_h,
        stride_ns_n, stride_ns_h, stride_ns_v,
        BLOCK_V, CHUNK_SIZE, (cudaStream_t)stream
    );
}
}
'''
    (build_dir / "prefill_v10.cu").write_text(v10_source)

    v10_cmd = [
        "nvcc", "-O3", "-arch=sm_100a", "--shared", "-Xcompiler", "-fPIC",
        "-std=c++17",
        "-I", str(build_dir),
        "-I", "/opt/cutlass/include",
        "-o", str(build_dir / "libprefill_v10.so"),
        str(build_dir / "prefill_v10.cu"),
    ]
    v10_result = subprocess.run(v10_cmd, capture_output=True, text=True)
    if v10_result.returncode != 0:
        print("CuTe v10 compile failed:")
        print(v10_result.stderr[:2000])
        lib_v10 = None
    else:
        print("CuTe v10 compile: OK")
        lib_v10 = ctypes.CDLL(str(build_dir / "libprefill_v10.so"))

    # ------------------------------------------------------------------
    # Compile PTX mma wrapper
    # ------------------------------------------------------------------
    ptx_source = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "gdn_prefill_ptx.cuh"

extern "C" {
void gdn_prefill_ptx_mma(
    const void* Q, const void* K, const void* V,
    const void* State, const void* A_log, const void* A,
    const void* DtBias, const void* B_gate, const void* CuSeqlens,
    void* Out, void* NewState,
    float scale, int N, int num_v_heads, int D,
    int stride_q_t, int stride_q_h, int stride_k_t, int stride_k_h,
    int stride_v_t, int stride_v_h, int stride_s_n, int stride_s_h, int stride_s_v,
    int stride_a_t, int stride_b_t, int stride_o_t, int stride_o_h,
    int stride_ns_n, int stride_ns_h, int stride_ns_v,
    int CHUNK_SIZE, void* stream
) {
    gdn_ptx::gdn_prefill_ptx_mma_launch(
        Q, K, V, State, A_log, A, DtBias, B_gate, CuSeqlens,
        Out, NewState, scale, N, num_v_heads, D,
        stride_q_t, stride_q_h, stride_k_t, stride_k_h,
        stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v,
        stride_a_t, stride_b_t, stride_o_t, stride_o_h,
        stride_ns_n, stride_ns_h, stride_ns_v,
        CHUNK_SIZE, (cudaStream_t)stream
    );
}
}
'''
    (build_dir / "prefill_ptx.cu").write_text(ptx_source)

    ptx_cmd = [
        "nvcc", "-O3", "-arch=sm_100a", "--shared", "-Xcompiler", "-fPIC",
        "-I", str(build_dir),
        "-o", str(build_dir / "libprefill_ptx.so"),
        str(build_dir / "prefill_ptx.cu"),
    ]
    ptx_result = subprocess.run(ptx_cmd, capture_output=True, text=True)
    if ptx_result.returncode != 0:
        print("PTX mma compile failed:")
        print(ptx_result.stderr[:2000])
        lib_ptx = None
    else:
        print("PTX mma compile: OK")
        lib_ptx = ctypes.CDLL(str(build_dir / "libprefill_ptx.so"))

    compile_summary = {
        "cute_v10_compiled": lib_v10 is not None,
        "ptx_mma_compiled": lib_ptx is not None,
    }

    if not (lib_v10 or lib_ptx):
        return {"status": "compile_only", "summary": [], "compile": compile_summary}

    # ------------------------------------------------------------------
    # Triton baseline
    # ------------------------------------------------------------------
    @triton.jit
    def _prefill_kernel_v5(
        Q_ptr, K_ptr, V_ptr, State_ptr,
        A_log_ptr, A_ptr, DtBias_ptr, B_ptr,
        CuSeq_ptr, Out_ptr, NewState_ptr,
        scale,
        stride_q_t, stride_q_h, stride_k_t, stride_k_h,
        stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v,
        stride_a_t, stride_b_t, stride_o_t, stride_o_h,
        stride_ns_n, stride_ns_h, stride_ns_v,
        D: tl.constexpr, BLOCK_V: tl.constexpr,
    ):
        n = tl.program_id(0)
        h = tl.program_id(1)
        vb = tl.program_id(2)
        v0 = vb * BLOCK_V
        qk_h = h // 2

        t_start = tl.load(CuSeq_ptr + n).to(tl.int32)
        t_end = tl.load(CuSeq_ptr + n + 1).to(tl.int32)
        seq_len = t_end - t_start

        alog = tl.load(A_log_ptr + h)
        dt_val = tl.load(DtBias_ptr + h)

        vi = tl.arange(0, BLOCK_V)[:, None]
        ki = tl.arange(0, D)[None, :]
        s_ptr = State_ptr + n * stride_s_n + h * stride_s_h + v0 * stride_s_v
        S = tl.load(s_ptr + vi * stride_s_v + ki)

        di = tl.arange(0, D)
        vd = tl.arange(0, BLOCK_V)

        if seq_len <= 0:
            ns_ptr = NewState_ptr + n * stride_ns_n + h * stride_ns_h + v0 * stride_ns_v
            tl.store(ns_ptr + vi * stride_ns_v + ki, S)
            return

        t_curr = t_start
        a_curr = tl.load(A_ptr + t_curr * stride_a_t + h).to(tl.float32)
        b_curr = tl.load(B_ptr + t_curr * stride_b_t + h).to(tl.float32)
        k_curr = tl.load(K_ptr + t_curr * stride_k_t + qk_h * stride_k_h + di).to(tl.float32)
        v_curr = tl.load(V_ptr + t_curr * stride_v_t + h * stride_v_h + v0 + vd).to(tl.float32)
        q_curr = tl.load(Q_ptr + t_curr * stride_q_t + qk_h * stride_q_h + di).to(tl.float32)

        for i in range(seq_len):
            t = t_start + i
            t_next = tl.minimum(t + 1, t_end - 1)

            a_next = tl.load(A_ptr + t_next * stride_a_t + h).to(tl.float32)
            b_next = tl.load(B_ptr + t_next * stride_b_t + h).to(tl.float32)
            k_next = tl.load(K_ptr + t_next * stride_k_t + qk_h * stride_k_h + di).to(tl.float32)
            v_next = tl.load(V_ptr + t_next * stride_v_t + h * stride_v_h + v0 + vd).to(tl.float32)
            q_next = tl.load(Q_ptr + t_next * stride_q_t + qk_h * stride_q_h + di).to(tl.float32)

            x = a_curr + dt_val
            sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
            g = tl.exp(-tl.exp(alog) * sp)
            beta = tl.sigmoid(b_curr)

            S = g * S
            old_v = tl.sum(S * k_curr[None, :], axis=1)
            delta = beta * (v_curr - old_v)
            S = S + delta[:, None] * k_curr[None, :]
            ov = scale * tl.sum(S * q_curr[None, :], axis=1)
            tl.store(Out_ptr + t * stride_o_t + h * stride_o_h + v0 + vd, ov.to(tl.bfloat16))

            a_curr = a_next
            b_curr = b_next
            k_curr = k_next
            v_curr = v_next
            q_curr = q_next

        ns_ptr = NewState_ptr + n * stride_ns_n + h * stride_ns_h + v0 * stride_ns_v
        tl.store(ns_ptr + vi * stride_ns_v + ki, S)

    argtypes_v10 = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    argtypes_ptx = argtypes_v10[:-3] + [ctypes.c_int, ctypes.c_void_p]

    if lib_v10:
        lib_v10.gdn_prefill_v10_tiledmma.argtypes = argtypes_v10
        lib_v10.gdn_prefill_v10_tiledmma.restype = None
    if lib_ptx:
        lib_ptx.gdn_prefill_ptx_mma.argtypes = argtypes_ptx
        lib_ptx.gdn_prefill_ptx_mma.restype = None

    def make_inputs(num_seqs: int, seq_len: int):
        total_tokens = num_seqs * seq_len
        D = 128
        num_q_heads = 4
        num_v_heads = 8
        device = "cuda"

        q = (torch.randn(total_tokens, num_q_heads, D, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        k = torch.nn.functional.normalize(
            torch.randn(total_tokens, num_q_heads, D, device=device, dtype=torch.float32),
            dim=-1,
        ).to(torch.bfloat16)
        v = (torch.randn(total_tokens, num_v_heads, D, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        state = torch.randn(num_seqs, num_v_heads, D, D, device=device, dtype=torch.float32) * 0.01
        A_log = torch.randn(num_v_heads, device=device, dtype=torch.float32) * 0.1 - 1.0
        a = (torch.randn(total_tokens, num_v_heads, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        b = (torch.randn(total_tokens, num_v_heads, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        dt_bias = torch.randn(num_v_heads, device=device, dtype=torch.float32) * 0.1
        cu = torch.arange(0, total_tokens + 1, seq_len, dtype=torch.int32, device=device)
        scale = 1.0 / math.sqrt(D)
        return q, k, v, state, A_log, a, dt_bias, b, cu, scale

    def run_triton(q, k, v, state, A_log, a, dt_bias, b, cu, scale):
        N = cu.shape[0] - 1
        D = q.shape[-1]
        num_v_heads = v.shape[1]
        block_v = 16 if N <= 4 else 32
        v_blocks = D // block_v
        out = torch.empty(q.shape[0], num_v_heads, D, dtype=torch.bfloat16, device=q.device)
        new_state = torch.empty_like(state)
        _prefill_kernel_v5[(N, num_v_heads, v_blocks)](
            q.contiguous(), k.contiguous(), v.contiguous(), state.contiguous(),
            A_log, a.float().contiguous(), dt_bias, b.float().contiguous(),
            cu.contiguous(), out, new_state, float(scale),
            q.stride(0), q.stride(1), k.stride(0), k.stride(1),
            v.stride(0), v.stride(1), state.stride(0), state.stride(1), state.stride(2),
            a.stride(0), b.stride(0), out.stride(0), out.stride(1),
            new_state.stride(0), new_state.stride(1), new_state.stride(2),
            D=128, BLOCK_V=block_v, num_warps=4,
        )
        return out, new_state

    def run_v10(q, k, v, state, A_log, a, dt_bias, b, cu, scale, block_v=16, chunk_size=8):
        N = cu.shape[0] - 1
        D = q.shape[-1]
        num_v_heads = v.shape[1]
        out = torch.empty(q.shape[0], num_v_heads, D, dtype=torch.bfloat16, device=q.device)
        new_state = torch.empty_like(state)
        lib_v10.gdn_prefill_v10_tiledmma(
            q.data_ptr(), k.data_ptr(), v.data_ptr(),
            state.data_ptr(), A_log.data_ptr(), a.data_ptr(),
            dt_bias.data_ptr(), b.data_ptr(), cu.data_ptr(),
            out.data_ptr(), new_state.data_ptr(),
            ctypes.c_float(scale), N, num_v_heads, D,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1),
            v.stride(0), v.stride(1), state.stride(0), state.stride(1), state.stride(2),
            a.stride(0), b.stride(0), out.stride(0), out.stride(1),
            new_state.stride(0), new_state.stride(1), new_state.stride(2),
            block_v, chunk_size, ctypes.c_void_p(0),
        )
        torch.cuda.synchronize()
        return out, new_state

    def run_ptx(q, k, v, state, A_log, a, dt_bias, b, cu, scale, chunk_size=8):
        N = cu.shape[0] - 1
        D = q.shape[-1]
        num_v_heads = v.shape[1]
        out = torch.empty(q.shape[0], num_v_heads, D, dtype=torch.bfloat16, device=q.device)
        new_state = torch.empty_like(state)
        lib_ptx.gdn_prefill_ptx_mma(
            q.data_ptr(), k.data_ptr(), v.data_ptr(),
            state.data_ptr(), A_log.data_ptr(), a.data_ptr(),
            dt_bias.data_ptr(), b.data_ptr(), cu.data_ptr(),
            out.data_ptr(), new_state.data_ptr(),
            ctypes.c_float(scale), N, num_v_heads, D,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1),
            v.stride(0), v.stride(1), state.stride(0), state.stride(1), state.stride(2),
            a.stride(0), b.stride(0), out.stride(0), out.stride(1),
            new_state.stride(0), new_state.stride(1), new_state.stride(2),
            chunk_size, ctypes.c_void_p(0),
        )
        torch.cuda.synchronize()
        return out, new_state

    def benchmark(label, fn, inputs, warmup=5, iters=20):
        for _ in range(warmup):
            fn(*inputs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = state = None
        for _ in range(iters):
            out, state = fn(*inputs)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iters * 1000.0
        return elapsed, out, state

    configs = [(1, 256), (1, 1024), (4, 256), (16, 128)]
    summary = []

    for num_seqs, seq_len in configs:
        print(f"\n--- Config: N={num_seqs}, L={seq_len} ---")
        inputs = make_inputs(num_seqs, seq_len)
        q, k, v, state, A_log, a, dt_bias, b, cu, scale = inputs
        triton_ms, ref_out, ref_state = benchmark("triton", run_triton, inputs)
        row = {
            "N": num_seqs,
            "L": seq_len,
            "triton_ms": triton_ms,
        }
        print(f"  Triton v5: {triton_ms:.4f} ms [ref]")

        if lib_v10:
            try:
                v10_inputs = (q, k, v, state.clone(), A_log, a, dt_bias, b, cu, scale)
                v10_ms, out_v10, state_v10 = benchmark("cute_v10", run_v10, v10_inputs)
                out_diff = (out_v10.float() - ref_out.float()).abs().max().item()
                state_diff = (state_v10 - ref_state).abs().max().item()
                row["cute_v10_ms"] = v10_ms
                row["cute_v10_speedup_vs_triton"] = triton_ms / v10_ms
                row["cute_v10_out_diff"] = out_diff
                row["cute_v10_state_diff"] = state_diff
                print(
                    f"  CuTe v10: {v10_ms:.4f} ms | vs Triton {triton_ms / v10_ms:.2f}x | "
                    f"out_diff={out_diff:.2e} state_diff={state_diff:.2e}"
                )
            except Exception as exc:
                row["cute_v10_error"] = str(exc)
                print(f"  CuTe v10: ERROR - {exc}")

        if lib_ptx:
            try:
                ptx_inputs = (q, k, v, state.clone(), A_log, a, dt_bias, b, cu, scale)
                ptx_ms, out_ptx, state_ptx = benchmark("ptx_mma", run_ptx, ptx_inputs)
                out_diff = (out_ptx.float() - ref_out.float()).abs().max().item()
                state_diff = (state_ptx - ref_state).abs().max().item()
                row["ptx_mma_ms"] = ptx_ms
                row["ptx_mma_speedup_vs_triton"] = triton_ms / ptx_ms
                row["ptx_mma_out_diff"] = out_diff
                row["ptx_mma_state_diff"] = state_diff
                print(
                    f"  PTX mma:  {ptx_ms:.4f} ms | vs Triton {triton_ms / ptx_ms:.2f}x | "
                    f"out_diff={out_diff:.2e} state_diff={state_diff:.2e}"
                )
            except Exception as exc:
                row["ptx_mma_error"] = str(exc)
                print(f"  PTX mma: ERROR - {exc}")

        summary.append(row)

    return {"status": "ok", "summary": summary, "compile": compile_summary}


@app.function(
    image=cuda_image,
    gpu=B200_GPU,
    timeout=LONG_TIMEOUT,
)
def compile_prefill_tensorcore(kernel_sources: dict):
    import ctypes
    import subprocess
    from pathlib import Path

    build_dir = Path("/tmp/gdn_prefill_tensorcore_compile")
    build_dir.mkdir(parents=True, exist_ok=True)

    for name, content in kernel_sources.items():
        (build_dir / name).write_text(content)

    results = {}

    v10_source = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cute/tensor.hpp>

#include "gdn_prefill_v10.cuh"

extern "C" {
void gdn_prefill_v10_tiledmma(...) {}
}
'''
    (build_dir / "prefill_v10_compile.cu").write_text(v10_source)
    v10_cmd = [
        "nvcc", "-O3", "-arch=sm_100a", "--shared", "-Xcompiler", "-fPIC",
        "-std=c++17",
        "-I", str(build_dir), "-I", "/opt/cutlass/include",
        "-o", str(build_dir / "libprefill_v10_compile.so"),
        str(build_dir / "prefill_v10_compile.cu"),
    ]
    v10_result = subprocess.run(v10_cmd, capture_output=True, text=True)
    results["cute_v10"] = {
        "ok": v10_result.returncode == 0,
        "stderr": v10_result.stderr[:2000],
    }
    if v10_result.returncode == 0:
        ctypes.CDLL(str(build_dir / "libprefill_v10_compile.so"))

    ptx_source = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "gdn_prefill_ptx.cuh"

extern "C" {
void gdn_prefill_ptx_mma(...) {}
}
'''
    (build_dir / "prefill_ptx_compile.cu").write_text(ptx_source)
    ptx_cmd = [
        "nvcc", "-O3", "-arch=sm_100a", "--shared", "-Xcompiler", "-fPIC",
        "-I", str(build_dir),
        "-o", str(build_dir / "libprefill_ptx_compile.so"),
        str(build_dir / "prefill_ptx_compile.cu"),
    ]
    ptx_result = subprocess.run(ptx_cmd, capture_output=True, text=True)
    results["ptx_mma"] = {
        "ok": ptx_result.returncode == 0,
        "stderr": ptx_result.stderr[:2000],
    }
    if ptx_result.returncode == 0:
        ctypes.CDLL(str(build_dir / "libprefill_ptx_compile.so"))

    print(results)
    return results


@app.function(
    image=cuda_image,
    gpu=B200_GPU,
    timeout=LONG_TIMEOUT,
)
def instantiate_sm100_types():
    import subprocess
    from pathlib import Path

    build_dir = Path("/tmp/gdn_sm100_type_smoke")
    build_dir.mkdir(parents=True, exist_ok=True)

    source = r'''
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/pointer.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cutlass/bfloat16.h>

using namespace cute;

void smoke_types() {
    using Elem = cutlass::bfloat16_t;

    using Mma1 = decltype(make_tiled_mma(
        SM100_MMA_F16BF16_SS<Elem, Elem, float, 64, 8, UMMA::Major::K, UMMA::Major::MN>{}
    ));
    using Mma2 = decltype(make_tiled_mma(
        SM100_MMA_F16BF16_2x1SM_SS<Elem, Elem, float, 128, 16, UMMA::Major::K, UMMA::Major::MN>{}
    ));

    auto tmem_ptr = make_tmem_ptr<float>(0);
    auto tmem_layout = Layout<Shape<_16, _16>, Stride<_16, _1>>{};
    auto tmem_tensor = make_tensor(tmem_ptr, tmem_layout);

    [[maybe_unused]] Mma1 mma1{};
    [[maybe_unused]] Mma2 mma2{};
    [[maybe_unused]] auto tensor = tmem_tensor;
}
'''
    cu = build_dir / "sm100_type_smoke.cu"
    cu.write_text(source)

    cmd = [
        "nvcc", "-O3", "-arch=sm_100a", "--shared", "-Xcompiler", "-fPIC",
        "-std=c++17",
        "-I", "/opt/cutlass/include",
        "-o", str(build_dir / "libsm100_type_smoke.so"),
        str(cu),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    payload = {"ok": result.returncode == 0, "stderr": result.stderr[:4000]}
    print(payload)
    return payload


@app.function(
    image=cuda_image,
    gpu=B200_GPU,
    timeout=LONG_TIMEOUT,
)
def invoke_sm100_mma_smoke():
    import subprocess
    from pathlib import Path

    build_dir = Path("/tmp/gdn_sm100_mma_call")
    build_dir.mkdir(parents=True, exist_ok=True)

    source = r'''
#include <cute/arch/mma_sm100_umma.hpp>
#include <cutlass/bfloat16.h>

using namespace cute;

__global__ void smoke_kernel() {
    using Elem = cutlass::bfloat16_t;
    using Op = SM100_MMA_F16BF16_SS<Elem, Elem, float, 64, 8, UMMA::Major::K, UMMA::Major::MN>;

    uint64_t desc_a = 0;
    uint64_t desc_b = 0;
    uint32_t tmem_c = 0;
    uint32_t scale_c = 0;
    uint64_t idesc_e = 0;

    Op::fma(desc_a, desc_b, tmem_c, scale_c, idesc_e);
}
'''
    cu = build_dir / "sm100_mma_call.cu"
    cu.write_text(source)

    cmd = [
        "nvcc", "-O3", "-arch=sm_100a", "--shared", "-Xcompiler", "-fPIC",
        "-std=c++17",
        "-I", "/opt/cutlass/include",
        "-o", str(build_dir / "libsm100_mma_call.so"),
        str(cu),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    payload = {"ok": result.returncode == 0, "stderr": result.stderr[:4000]}
    print(payload)
    return payload


@app.function(
    image=cuda_image,
    gpu=B200_GPU,
    timeout=LONG_TIMEOUT,
)
def run_blackwell_example_gemm():
    import subprocess
    from pathlib import Path

    example_dir = Path("/opt/cutlass/examples/70_blackwell_gemm")
    source = example_dir / "70_blackwell_fp16_gemm.cu"
    binary = Path("/tmp/blackwell_fp16_gemm")

    compile_cmd = [
        "nvcc",
        "-O3",
        "-arch=sm_100a",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-I", "/opt/cutlass/include",
        "-I", "/opt/cutlass/tools/util/include",
        "-I", str(example_dir),
        "-I", "/opt/cutlass/examples/common",
        "-o", str(binary),
        str(source),
    ]
    compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
    payload = {
        "compile_ok": compile_result.returncode == 0,
        "compile_stderr": compile_result.stderr[:4000],
    }

    if compile_result.returncode != 0:
        print(payload)
        return payload

    runs = []
    configs = [
        ("prefill_tile_64x8x128", ["--m=64", "--n=8", "--k=128", "--iterations=20"]),
        ("prefill_tile_128x16x128", ["--m=128", "--n=16", "--k=128", "--iterations=20"]),
        ("small", ["--m=1024", "--n=1024", "--k=1024", "--iterations=5"]),
        ("medium", ["--m=2048", "--n=2048", "--k=2048", "--iterations=5"]),
    ]
    for name, args in configs:
        cmd = [str(binary)] + args
        result = subprocess.run(cmd, capture_output=True, text=True)
        runs.append(
            {
                "name": name,
                "ok": result.returncode == 0,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-2000:],
            }
        )
        print(f"\n== {name} ==")
        print(result.stdout[-2000:])
        if result.stderr:
            print(result.stderr[-1000:])

    payload["runs"] = runs
    return payload


@app.function(
    image=cuda_image,
    gpu=B200_GPU,
    timeout=LONG_TIMEOUT,
)
def run_repo_local_blackwell_microgemm():
    import subprocess
    from pathlib import Path

    build_dir = Path("/tmp/repo_local_blackwell_microgemm")
    build_dir.mkdir(parents=True, exist_ok=True)

    source = r'''
#include <iostream>
#include <string>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using ElementA = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::half_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = float;
using LayoutC = cutlass::layout::ColumnMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using MmaTileShape_MNK = Shape<_256,_128,_64>;
using ClusterShape_MNK = Shape<_2,_2,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

#endif

int main(int argc, char** argv) {
#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  std::cerr << "CUTLASS_ARCH_MMA_SM100_SUPPORTED is not enabled\n";
  return -1;
#else
  int m = 64;
  int n = 8;
  int k = 128;
  int iterations = 20;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.rfind("--m=", 0) == 0) m = std::stoi(arg.substr(4));
    if (arg.rfind("--n=", 0) == 0) n = std::stoi(arg.substr(4));
    if (arg.rfind("--k=", 0) == 0) k = std::stoi(arg.substr(4));
    if (arg.rfind("--iterations=", 0) == 0) iterations = std::stoi(arg.substr(13));
  }

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  cutlass::DeviceAllocation<ElementA> block_A(m * k);
  cutlass::DeviceAllocation<ElementB> block_B(k * n);
  cutlass::DeviceAllocation<ElementC> block_C(m * n);
  cutlass::DeviceAllocation<ElementC> block_D(m * n);

  cutlass::reference::device::BlockFillRandomUniform(block_A.get(), block_A.size(), 2025, ElementA(8), ElementA(-8), 0);
  cutlass::reference::device::BlockFillRandomUniform(block_B.get(), block_B.size(), 2026, ElementB(8), ElementB(-8), 0);
  cutlass::reference::device::BlockFillRandomUniform(block_C.get(), block_C.size(), 2027, ElementC(8), ElementC(-8), 0);

  Gemm gemm;
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {m, n, k, 1},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    {{1.0f, 0.0f}, block_C.get(), stride_C, block_D.get(), stride_D}
  };

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm.run());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int iter = 0; iter < iterations; ++iter) {
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm.run());
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  double runtime_ms = double(elapsed_ms) / double(iterations);
  uint64_t flop = uint64_t(2) * m * n * k;
  double gflops = double(flop) / double(1.0e6) / runtime_ms;

  std::cout << "Problem Size: " << m << "x" << n << "x" << k << "\n";
  std::cout << "Avg runtime: " << runtime_ms << " ms\n";
  std::cout << "GFLOPS: " << gflops << "\n";
  return 0;
#endif
}
'''

    cu = build_dir / "repo_local_blackwell_microgemm.cu"
    cu.write_text(source)
    binary = build_dir / "repo_local_blackwell_microgemm"

    compile_cmd = [
        "nvcc",
        "-O3",
        "-arch=sm_100a",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-I", "/opt/cutlass/include",
        "-I", "/opt/cutlass/tools/util/include",
        "-I", "/opt/cutlass/examples/common",
        "-o", str(binary),
        str(cu),
    ]
    compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
    payload = {
        "compile_ok": compile_result.returncode == 0,
        "compile_stderr": compile_result.stderr[:4000],
    }
    if compile_result.returncode != 0:
        print(payload)
        return payload

    runs = []
    configs = [
        ("prefill_tile_64x8x128", ["--m=64", "--n=8", "--k=128", "--iterations=20"]),
        ("prefill_tile_128x16x128", ["--m=128", "--n=16", "--k=128", "--iterations=20"]),
    ]
    for name, args in configs:
        result = subprocess.run([str(binary)] + args, capture_output=True, text=True)
        runs.append(
            {
                "name": name,
                "ok": result.returncode == 0,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-2000:],
            }
        )
        print(f"\n== {name} ==")
        print(result.stdout[-2000:])
        if result.stderr:
            print(result.stderr[-1000:])

    payload["runs"] = runs
    print(payload)
    return payload


@app.function(
    image=cuda_image,
    gpu=B200_GPU,
    timeout=LONG_TIMEOUT,
)
def run_repo_local_blackwell_microgemm_lib():
    import ctypes
    import subprocess
    from pathlib import Path

    import torch

    build_dir = Path("/tmp/repo_local_blackwell_microgemm_lib")
    build_dir.mkdir(parents=True, exist_ok=True)

    source = r'''
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"
#include <cuda_runtime.h>

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using ElementA = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::half_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = float;
using LayoutC = cutlass::layout::ColumnMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using MmaTileShape_MNK = Shape<_256,_128,_64>;
using ClusterShape_MNK = Shape<_2,_2,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

#endif

extern "C" int run_sm100_fp16_gemm(
    const void* a_ptr,
    const void* b_ptr,
    const void* c_ptr,
    void* d_ptr,
    int m,
    int n,
    int k
) {
#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  return -100;
#else
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  Gemm gemm;
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {m, n, k, 1},
    {static_cast<ElementA const*>(a_ptr), stride_A, static_cast<ElementB const*>(b_ptr), stride_B},
    {{1.0f, 0.0f}, static_cast<ElementC const*>(c_ptr), stride_C, static_cast<ElementC*>(d_ptr), stride_D}
  };

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status can = gemm.can_implement(arguments);
  if (can != cutlass::Status::kSuccess) {
    return static_cast<int>(can);
  }

  cutlass::Status init = gemm.initialize(arguments, workspace.get());
  if (init != cutlass::Status::kSuccess) {
    return static_cast<int>(init);
  }

  cutlass::Status run = gemm.run();
  if (run != cutlass::Status::kSuccess) {
    return static_cast<int>(run);
  }

  return 0;
#endif
}
'''

    cu = build_dir / "repo_local_blackwell_microgemm_lib.cu"
    cu.write_text(source)
    library = build_dir / "librepo_local_blackwell_microgemm.so"

    compile_cmd = [
        "nvcc",
        "-O3",
        "-arch=sm_100a",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--shared",
        "-Xcompiler", "-fPIC",
        "-I", "/opt/cutlass/include",
        "-I", "/opt/cutlass/tools/util/include",
        "-I", "/opt/cutlass/examples/common",
        "-o", str(library),
        str(cu),
    ]
    compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
    payload = {
        "compile_ok": compile_result.returncode == 0,
        "compile_stderr": compile_result.stderr[:4000],
    }
    if compile_result.returncode != 0:
        print(payload)
        return payload

    lib = ctypes.CDLL(str(library))
    fn = lib.run_sm100_fp16_gemm
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    fn.restype = ctypes.c_int

    def run_case(m: int, n: int, k: int):
        device = "cuda"
        a = torch.randn(m, k, device=device, dtype=torch.float16)
        b_storage = torch.randn(n, k, device=device, dtype=torch.float16)
        c_storage = torch.zeros(n, m, device=device, dtype=torch.float32)
        d_storage = torch.empty(n, m, device=device, dtype=torch.float32)

        b_ref = b_storage.transpose(0, 1).contiguous()
        ref = a.float() @ b_ref.float()

        # Warmup
        for _ in range(10):
            rc = fn(
                a.data_ptr(),
                b_storage.data_ptr(),
                c_storage.data_ptr(),
                d_storage.data_ptr(),
                m, n, k,
            )
            if rc != 0:
                raise RuntimeError(f"run_sm100_fp16_gemm returned {rc}")
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            rc = fn(
                a.data_ptr(),
                b_storage.data_ptr(),
                c_storage.data_ptr(),
                d_storage.data_ptr(),
                m, n, k,
            )
            if rc != 0:
                raise RuntimeError(f"run_sm100_fp16_gemm returned {rc}")
        stop.record()
        torch.cuda.synchronize()
        runtime_ms = start.elapsed_time(stop) / 50.0
        d_view = d_storage.transpose(0, 1)
        diff = (d_view - ref).abs().max().item()
        gflops = (2.0 * m * n * k) / (runtime_ms * 1e6)
        return runtime_ms, gflops, diff

    runs = []
    for name, (m, n, k) in {
        "prefill_tile_64x8x128": (64, 8, 128),
        "prefill_tile_128x16x128": (128, 16, 128),
    }.items():
        runtime_ms, gflops, diff = run_case(m, n, k)
        entry = {
            "name": name,
            "m": m,
            "n": n,
            "k": k,
            "runtime_ms": runtime_ms,
            "gflops": gflops,
            "max_abs_diff": diff,
        }
        runs.append(entry)
        print(entry)

    payload["runs"] = runs
    print(payload)
    return payload


@app.function(
    image=cuda_image,
    gpu=B200_GPU,
    timeout=LONG_TIMEOUT,
)
def run_chunked_prefill_prototype():
    import ctypes
    import math
    import subprocess
    from pathlib import Path

    import torch
    import torch.nn.functional as F

    build_dir = Path("/tmp/repo_local_blackwell_chunkproto")
    build_dir.mkdir(parents=True, exist_ok=True)

    source = r'''
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using ElementA = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::bfloat16_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = float;
using LayoutC = cutlass::layout::ColumnMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using MmaTileShape_MNK = Shape<_256,_128,_64>;
using ClusterShape_MNK = Shape<_2,_2,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

#endif

extern "C" int run_sm100_bf16_gemm(
    const void* a_ptr,
    const void* b_ptr,
    const void* c_ptr,
    void* d_ptr,
    int m,
    int n,
    int k
) {
#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  return -100;
#else
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  Gemm gemm;
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {m, n, k, 1},
    {static_cast<ElementA const*>(a_ptr), stride_A, static_cast<ElementB const*>(b_ptr), stride_B},
    {{1.0f, 0.0f}, static_cast<ElementC const*>(c_ptr), stride_C, static_cast<ElementC*>(d_ptr), stride_D}
  };

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status can = gemm.can_implement(arguments);
  if (can != cutlass::Status::kSuccess) {
    return static_cast<int>(can);
  }
  cutlass::Status init = gemm.initialize(arguments, workspace.get());
  if (init != cutlass::Status::kSuccess) {
    return static_cast<int>(init);
  }
  cutlass::Status run = gemm.run();
  if (run != cutlass::Status::kSuccess) {
    return static_cast<int>(run);
  }
  return 0;
#endif
}

__global__ void chunk_correction_kernel(
    const float* __restrict__ old_v_init,   // [V, C]
    const float* __restrict__ out_init,     // [V, C]
    const float* __restrict__ kk,           // [C, C]
    const float* __restrict__ kq,           // [C, C]
    const float* __restrict__ v,            // [C, V]
    const float* __restrict__ prefix,       // [C]
    const float* __restrict__ beta,         // [C]
    const float* __restrict__ state,        // [V, D]
    const float* __restrict__ k,            // [C, D]
    float scale,
    float* __restrict__ out,                // [V, C]
    float* __restrict__ final_state,        // [V, D]
    int chunk_size,
    int D
) {
    int vi = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi >= D) {
        return;
    }

    float deltas[16];

    for (int t = 0; t < chunk_size; ++t) {
        float old_v = old_v_init[vi * chunk_size + t];
        for (int j = 0; j < t; ++j) {
            float decay = prefix[t] / prefix[j];
            old_v += decay * kk[j * chunk_size + t] * deltas[j];
        }

        float delta_t = beta[t] * (v[t * D + vi] - old_v);
        deltas[t] = delta_t;

        float out_t = out_init[vi * chunk_size + t];
        for (int j = 0; j < t; ++j) {
            float decay = prefix[t] / prefix[j];
            out_t += scale * decay * kq[j * chunk_size + t] * deltas[j];
        }
        out_t += scale * kq[t * chunk_size + t] * delta_t;
        out[vi * chunk_size + t] = out_t;
    }

    float tail_decay = prefix[chunk_size - 1];
    for (int d = 0; d < D; ++d) {
        float value = tail_decay * state[vi * D + d];
        for (int j = 0; j < chunk_size; ++j) {
            float decay = tail_decay / prefix[j];
            value += decay * deltas[j] * k[j * D + d];
        }
        final_state[vi * D + d] = value;
    }
}

extern "C" void run_chunk_correction(
    const void* old_v_init,
    const void* out_init,
    const void* kk,
    const void* kq,
    const void* v,
    const void* prefix,
    const void* beta,
    const void* state,
    const void* k,
    float scale,
    void* out,
    void* final_state,
    int chunk_size,
    int D
) {
    dim3 block(128);
    dim3 grid((D + block.x - 1) / block.x);
    chunk_correction_kernel<<<grid, block>>>(
        static_cast<const float*>(old_v_init),
        static_cast<const float*>(out_init),
        static_cast<const float*>(kk),
        static_cast<const float*>(kq),
        static_cast<const float*>(v),
        static_cast<const float*>(prefix),
        static_cast<const float*>(beta),
        static_cast<const float*>(state),
        static_cast<const float*>(k),
        scale,
        static_cast<float*>(out),
        static_cast<float*>(final_state),
        chunk_size,
        D
    );
}
'''

    cu = build_dir / "chunked_prefill_proto_lib.cu"
    cu.write_text(source)
    library = build_dir / "libchunked_prefill_proto.so"

    compile_cmd = [
        "nvcc",
        "-O3",
        "-arch=sm_100a",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--shared",
        "-Xcompiler", "-fPIC",
        "-I", "/opt/cutlass/include",
        "-I", "/opt/cutlass/tools/util/include",
        "-I", "/opt/cutlass/examples/common",
        "-o", str(library),
        str(cu),
    ]
    compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
    payload = {
        "compile_ok": compile_result.returncode == 0,
        "compile_stderr": compile_result.stderr[:4000],
    }
    if compile_result.returncode != 0:
        print(payload)
        return payload

    lib = ctypes.CDLL(str(library))
    fn = lib.run_sm100_bf16_gemm
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    fn.restype = ctypes.c_int

    correction = lib.run_chunk_correction
    correction.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    correction.restype = None

    def gemm_state_cols(state_bf16: torch.Tensor, cols_bf16_storage: torch.Tensor) -> torch.Tensor:
        m, k = state_bf16.shape
        n = cols_bf16_storage.shape[0]
        c_storage = torch.zeros(n, m, device=state_bf16.device, dtype=torch.float32)
        d_storage = torch.empty(n, m, device=state_bf16.device, dtype=torch.float32)
        rc = fn(
            state_bf16.data_ptr(),
            cols_bf16_storage.data_ptr(),
            c_storage.data_ptr(),
            d_storage.data_ptr(),
            m, n, k,
        )
        if rc != 0:
            raise RuntimeError(f"run_sm100_bf16_gemm returned {rc}")
        return d_storage.transpose(0, 1).contiguous()

    def reference_chunk(q, k, v, state, g, beta, scale):
        C, D = q.shape
        S = state.clone()
        outs = []
        deltas = []
        for t in range(C):
            S = g[t] * S
            old_v = S @ k[t]
            delta = beta[t] * (v[t] - old_v)
            S = S + delta[:, None] * k[t][None, :]
            outs.append(scale * (S @ q[t]))
            deltas.append(delta)
        return torch.stack(outs), S, torch.stack(deltas)

    def chunked_proto(q, k, v, state, g, beta, scale):
        C, D = q.shape
        prefix = torch.cumprod(g, dim=0)

        k_init = (prefix[:, None] * k).to(torch.bfloat16).contiguous()     # [C, D]
        q_init = (prefix[:, None] * q).to(torch.bfloat16).contiguous()     # [C, D]
        state_bf16 = state.to(torch.bfloat16).contiguous()                 # [V, D]

        old_v_init = gemm_state_cols(state_bf16, k_init).float()           # [V, C]
        out_init = gemm_state_cols(state_bf16, q_init).float() * scale     # [V, C]

        kk = (k @ k.transpose(0, 1)).float()   # [C, C]
        kq = (k @ q.transpose(0, 1)).float()   # [C, C]

        deltas = []
        outs = []
        C_range = list(range(C))
        for t in C_range:
            old_v = old_v_init[:, t].clone()
            for j in range(t):
                decay = prefix[t] / prefix[j]
                old_v = old_v + decay * kk[j, t] * deltas[j]

            delta_t = beta[t] * (v[t] - old_v)
            deltas.append(delta_t)

            out_t = out_init[:, t].clone()
            for j in range(t):
                decay = prefix[t] / prefix[j]
                out_t = out_t + scale * decay * kq[j, t] * deltas[j]
            out_t = out_t + scale * kq[t, t] * delta_t
            outs.append(out_t)

        final_state = prefix[-1] * state
        for j in C_range:
            tail_decay = prefix[-1] / prefix[j]
            final_state = final_state + tail_decay * deltas[j][:, None] * k[j][None, :]

        return torch.stack(outs), final_state

    def chunked_proto_cuda(q, k, v, state, g, beta, scale):
        C, D = q.shape
        prefix = torch.cumprod(g, dim=0)

        k_init = (prefix[:, None] * k).to(torch.bfloat16).contiguous()     # [C, D]
        q_init = (prefix[:, None] * q).to(torch.bfloat16).contiguous()     # [C, D]
        state_bf16 = state.to(torch.bfloat16).contiguous()                 # [V, D]

        old_v_init = gemm_state_cols(state_bf16, k_init).float().contiguous()   # [V, C]
        out_init = (gemm_state_cols(state_bf16, q_init).float() * scale).contiguous()

        kk = (k @ k.transpose(0, 1)).float().contiguous()   # [C, C]
        kq = (k @ q.transpose(0, 1)).float().contiguous()   # [C, C]
        v_t = v.float().contiguous()                        # [C, V]
        prefix_c = prefix.float().contiguous()
        beta_c = beta.float().contiguous()
        state_c = state.float().contiguous()
        k_c = k.float().contiguous()

        out = torch.empty(D, C, device=q.device, dtype=torch.float32)
        final_state = torch.empty_like(state_c)

        correction(
            old_v_init.data_ptr(),
            out_init.data_ptr(),
            kk.data_ptr(),
            kq.data_ptr(),
            v_t.data_ptr(),
            prefix_c.data_ptr(),
            beta_c.data_ptr(),
            state_c.data_ptr(),
            k_c.data_ptr(),
            ctypes.c_float(scale),
            out.data_ptr(),
            final_state.data_ptr(),
            C,
            D,
        )
        torch.cuda.synchronize()

        return out.transpose(0, 1).contiguous(), final_state

    def reference_prefill_end_to_end(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size):
        total_tokens, num_q_heads, D = q.shape
        num_v_heads = v.shape[1]
        num_seqs = cu_seqlens.numel() - 1
        device = q.device

        x = a.float() + dt_bias.float()
        g_all = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
        beta_all = torch.sigmoid(b.float())

        out = torch.empty(total_tokens, num_v_heads, D, dtype=torch.bfloat16, device=device)
        new_state = state.clone().float()

        for seq_idx in range(num_seqs):
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())
            if end <= start:
                continue
            for h in range(num_v_heads):
                qk_h = h // 2
                state_h = new_state[seq_idx, h].clone()
                for t in range(start, end):
                    q_h = q[t, qk_h].float()
                    k_h = k[t, qk_h].float()
                    v_h = v[t, h].float()
                    g_h = g_all[t, h]
                    beta_h = beta_all[t, h]

                    state_h = g_h * state_h
                    old_v = state_h @ k_h
                    delta = beta_h * (v_h - old_v)
                    state_h = state_h + delta[:, None] * k_h[None, :]
                    out[t, h] = (scale * (state_h @ q_h)).to(torch.bfloat16)
                new_state[seq_idx, h] = state_h

        return out, new_state

    def chunked_prefill_end_to_end(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size):
        total_tokens, num_q_heads, D = q.shape
        num_v_heads = v.shape[1]
        num_seqs = cu_seqlens.numel() - 1
        device = q.device

        x = a.float() + dt_bias.float()
        g_all = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
        beta_all = torch.sigmoid(b.float())

        out = torch.empty(total_tokens, num_v_heads, D, dtype=torch.bfloat16, device=device)
        new_state = state.clone().float()

        for seq_idx in range(num_seqs):
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())
            if end <= start:
                continue
            for h in range(num_v_heads):
                qk_h = h // 2
                state_h = new_state[seq_idx, h].clone()
                for chunk_start in range(start, end, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, end)
                    q_chunk = q[chunk_start:chunk_end, qk_h].float().contiguous()
                    k_chunk = k[chunk_start:chunk_end, qk_h].float().contiguous()
                    v_chunk = v[chunk_start:chunk_end, h].float().contiguous()
                    g_chunk = g_all[chunk_start:chunk_end, h].float().contiguous()
                    beta_chunk = beta_all[chunk_start:chunk_end, h].float().contiguous()

                    out_chunk, state_h = chunked_proto_cuda(
                        q_chunk, k_chunk, v_chunk, state_h, g_chunk, beta_chunk, scale
                    )
                    out[chunk_start:chunk_end, h] = out_chunk.to(torch.bfloat16)
                new_state[seq_idx, h] = state_h

        return out, new_state

    def bench_cuda(label, fn, warmup=10, iters=50):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        torch.cuda.synchronize()
        return start.elapsed_time(stop) / iters

    device = "cuda"
    D = 128
    scale = 1.0 / math.sqrt(D)
    results = []

    for C in [8, 16]:
        torch.manual_seed(42 + C)
        q = torch.randn(C, D, device=device, dtype=torch.float32) * 0.1
        k = F.normalize(torch.randn(C, D, device=device, dtype=torch.float32), dim=-1)
        v = torch.randn(C, D, device=device, dtype=torch.float32) * 0.1
        state = torch.randn(D, D, device=device, dtype=torch.float32) * 0.01
        g = torch.rand(C, device=device, dtype=torch.float32) * 0.4 + 0.5
        beta = torch.rand(C, device=device, dtype=torch.float32) * 0.3 + 0.1

        ref_out, ref_state, _ = reference_chunk(q, k, v, state.clone(), g, beta, scale)
        proto_out, proto_state = chunked_proto(q, k, v, state.clone(), g, beta, scale)

        out_diff = (proto_out - ref_out).abs().max().item()
        state_diff = (proto_state - ref_state).abs().max().item()

        ref_ms = bench_cuda(
            "reference",
            lambda: reference_chunk(q, k, v, state.clone(), g, beta, scale),
            warmup=5,
            iters=20,
        )
        proto_ms = bench_cuda(
            "prototype",
            lambda: chunked_proto(q, k, v, state.clone(), g, beta, scale),
            warmup=5,
            iters=20,
        )
        proto_cuda_out, proto_cuda_state = chunked_proto_cuda(q, k, v, state.clone(), g, beta, scale)
        proto_cuda_out_diff = (proto_cuda_out - ref_out).abs().max().item()
        proto_cuda_state_diff = (proto_cuda_state - ref_state).abs().max().item()
        proto_cuda_ms = bench_cuda(
            "prototype_cuda",
            lambda: chunked_proto_cuda(q, k, v, state.clone(), g, beta, scale),
            warmup=5,
            iters=20,
        )

        entry = {
            "chunk_size": C,
            "out_diff": out_diff,
            "state_diff": state_diff,
            "reference_ms": ref_ms,
            "prototype_ms": proto_ms,
            "speedup_vs_reference": ref_ms / proto_ms,
            "prototype_cuda_out_diff": proto_cuda_out_diff,
            "prototype_cuda_state_diff": proto_cuda_state_diff,
            "prototype_cuda_ms": proto_cuda_ms,
            "prototype_cuda_speedup_vs_reference": ref_ms / proto_cuda_ms,
            "prototype_cuda_speedup_vs_python_proto": proto_ms / proto_cuda_ms,
        }
        results.append(entry)
        print(entry)

    payload["prototype"] = results

    end_to_end_results = []
    for cfg in [
        {"name": "N1_L64", "num_seqs": 1, "seq_len": 64, "chunk_size": 8},
        {"name": "N4_L32", "num_seqs": 4, "seq_len": 32, "chunk_size": 8},
    ]:
        num_seqs = cfg["num_seqs"]
        seq_len = cfg["seq_len"]
        total_tokens = num_seqs * seq_len
        chunk_size = cfg["chunk_size"]

        torch.manual_seed(123 + total_tokens)
        q = (torch.randn(total_tokens, 4, D, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        k = F.normalize(torch.randn(total_tokens, 4, D, device=device, dtype=torch.float32), dim=-1).to(torch.bfloat16)
        v = (torch.randn(total_tokens, 8, D, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        state = torch.randn(num_seqs, 8, D, D, device=device, dtype=torch.float32) * 0.01
        A_log = torch.randn(8, device=device, dtype=torch.float32) * 0.1 - 1.0
        a = torch.randn(total_tokens, 8, device=device, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(8, device=device, dtype=torch.float32) * 0.1
        b = torch.randn(total_tokens, 8, device=device, dtype=torch.float32) * 0.1
        cu_seqlens = torch.arange(0, total_tokens + 1, seq_len, device=device, dtype=torch.int32)

        ref_out, ref_state = reference_prefill_end_to_end(
            q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size
        )
        proto_out, proto_state = chunked_prefill_end_to_end(
            q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size
        )

        ref_ms = bench_cuda(
            "reference_e2e",
            lambda: reference_prefill_end_to_end(
                q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size
            ),
            warmup=2,
            iters=5,
        )
        proto_ms = bench_cuda(
            "proto_e2e",
            lambda: chunked_prefill_end_to_end(
                q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size
            ),
            warmup=2,
            iters=5,
        )

        entry = {
            "name": cfg["name"],
            "chunk_size": chunk_size,
            "out_diff": (proto_out.float() - ref_out.float()).abs().max().item(),
            "state_diff": (proto_state - ref_state).abs().max().item(),
            "reference_ms": ref_ms,
            "prototype_ms": proto_ms,
            "speedup_vs_reference": ref_ms / proto_ms,
        }
        end_to_end_results.append(entry)
        print(entry)

    payload["end_to_end"] = end_to_end_results
    return payload


@app.function(
    image=cuda_image,
    gpu=B200_GPU,
    timeout=LONG_TIMEOUT,
)
def run_chunked_prefill_module():
    import math
    import os
    import sys

    import torch
    import torch.nn.functional as F

    sys.path.insert(0, GDN_REMOTE_ROOT)
    from prefill.solution.cuda.chunked_proto import (
        chunked_prefill_end_to_end,
        chunked_prefill_end_to_end_batched,
        chunked_prefill_end_to_end_grouped,
        get_last_gemm_batched_path,
        kernel as chunked_kernel,
        recommend_chunk_size,
        chunked_prefill_end_to_end_uniform_batch,
    )
    from prefill.solution.triton.kernel import kernel as triton_prefill_kernel

    def reference_prefill_end_to_end(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        total_tokens, num_q_heads, D = q.shape
        num_v_heads = v.shape[1]
        num_seqs = cu_seqlens.numel() - 1
        device = q.device

        x = a.float() + dt_bias.float()
        g_all = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
        beta_all = torch.sigmoid(b.float())

        out = torch.empty(total_tokens, num_v_heads, D, dtype=torch.bfloat16, device=device)
        new_state = state.clone().float()

        for seq_idx in range(num_seqs):
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())
            if end <= start:
                continue
            for h in range(num_v_heads):
                qk_h = h // 2
                state_h = new_state[seq_idx, h].clone()
                for t in range(start, end):
                    q_h = q[t, qk_h].float()
                    k_h = k[t, qk_h].float()
                    v_h = v[t, h].float()
                    g_h = g_all[t, h]
                    beta_h = beta_all[t, h]

                    state_h = g_h * state_h
                    old_v = state_h @ k_h
                    delta = beta_h * (v_h - old_v)
                    state_h = state_h + delta[:, None] * k_h[None, :]
                    out[t, h] = (scale * (state_h @ q_h)).to(torch.bfloat16)
                new_state[seq_idx, h] = state_h

        return out, new_state

    def bench_cuda(fn, warmup=2, iters=5):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        torch.cuda.synchronize()
        return start.elapsed_time(stop) / iters

    device = "cuda"
    D = 128
    scale = 1.0 / math.sqrt(D)
    results = []

    for cfg in [
        {"name": "N1_L64", "lengths": [64], "chunk_size": 64},
        {"name": "N4_L32", "lengths": [32, 32, 32, 32], "chunk_size": 64},
        {"name": "grouped_varlen", "lengths": [32, 32, 64, 64], "chunk_size": 64},
    ]:
        lengths = cfg["lengths"]
        num_seqs = len(lengths)
        total_tokens = sum(lengths)
        chunk_size = cfg["chunk_size"]

        torch.manual_seed(123 + total_tokens)
        q = (torch.randn(total_tokens, 4, D, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        k = F.normalize(torch.randn(total_tokens, 4, D, device=device, dtype=torch.float32), dim=-1).to(torch.bfloat16)
        v = (torch.randn(total_tokens, 8, D, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        state = torch.randn(num_seqs, 8, D, D, device=device, dtype=torch.float32) * 0.01
        A_log = torch.randn(8, device=device, dtype=torch.float32) * 0.1 - 1.0
        a = torch.randn(total_tokens, 8, device=device, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(8, device=device, dtype=torch.float32) * 0.1
        b = torch.randn(total_tokens, 8, device=device, dtype=torch.float32) * 0.1
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.int32)

        ref_out, ref_state = reference_prefill_end_to_end(
            q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale
        )
        triton_out, triton_state = triton_prefill_kernel(
            q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale
        )
        proto_out, proto_state = chunked_prefill_end_to_end(
            q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
        )
        batched_out, batched_state = chunked_prefill_end_to_end_batched(
            q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
        )
        grouped_out, grouped_state = chunked_prefill_end_to_end_grouped(
            q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
        )
        auto_out, auto_state = chunked_kernel(
            q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=0
        )
        recommended_chunk_size = recommend_chunk_size(cu_seqlens)
        gemm_batched_path = get_last_gemm_batched_path()

        entry = {
            "name": cfg["name"],
            "chunk_size": chunk_size,
            "recommended_chunk_size": recommended_chunk_size,
            "out_diff": (proto_out.float() - ref_out.float()).abs().max().item(),
            "state_diff": (proto_state - ref_state).abs().max().item(),
            "batched_out_diff": (batched_out.float() - ref_out.float()).abs().max().item(),
            "batched_state_diff": (batched_state - ref_state).abs().max().item(),
            "grouped_out_diff": (grouped_out.float() - ref_out.float()).abs().max().item(),
            "grouped_state_diff": (grouped_state - ref_state).abs().max().item(),
            "auto_out_diff": (auto_out.float() - ref_out.float()).abs().max().item(),
            "auto_state_diff": (auto_state - ref_state).abs().max().item(),
            "gemm_batched_path": gemm_batched_path,
            "triton_out_diff": (triton_out.float() - ref_out.float()).abs().max().item(),
            "triton_state_diff": (triton_state - ref_state).abs().max().item(),
            "reference_ms": bench_cuda(
                lambda: reference_prefill_end_to_end(
                    q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale
                )
            ),
            "triton_ms": bench_cuda(
                lambda: triton_prefill_kernel(
                    q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale
                )
            ),
            "prototype_ms": bench_cuda(
                lambda: chunked_prefill_end_to_end(
                    q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
                )
            ),
            "batched_ms": bench_cuda(
                lambda: chunked_prefill_end_to_end_batched(
                    q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
                )
            ),
            "grouped_ms": bench_cuda(
                lambda: chunked_prefill_end_to_end_grouped(
                    q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
                )
            ),
            "auto_kernel_ms": bench_cuda(
                lambda: chunked_kernel(
                    q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=0
                )
            ),
        }
        entry["speedup_vs_reference"] = entry["reference_ms"] / entry["prototype_ms"]
        entry["speedup_vs_triton"] = entry["triton_ms"] / entry["prototype_ms"]
        entry["batched_speedup_vs_reference"] = entry["reference_ms"] / entry["batched_ms"]
        entry["batched_speedup_vs_triton"] = entry["triton_ms"] / entry["batched_ms"]
        entry["batched_speedup_vs_single_chunkproto"] = entry["prototype_ms"] / entry["batched_ms"]
        entry["grouped_speedup_vs_reference"] = entry["reference_ms"] / entry["grouped_ms"]
        entry["grouped_speedup_vs_triton"] = entry["triton_ms"] / entry["grouped_ms"]
        entry["grouped_speedup_vs_batched"] = entry["batched_ms"] / entry["grouped_ms"]
        entry["auto_speedup_vs_reference"] = entry["reference_ms"] / entry["auto_kernel_ms"]
        entry["auto_speedup_vs_triton"] = entry["triton_ms"] / entry["auto_kernel_ms"]
        results.append(entry)
        print(entry)

    return {"status": "ok", "results": results}


@app.function(
    image=cuda_image,
    gpu=B200_GPU,
    timeout=LONG_TIMEOUT,
)
def run_chunked_prefill_module_no_strided_batched():
    import math
    import os
    import sys

    import torch
    import torch.nn.functional as F

    os.environ["GDN_CHUNKPROTO_DISABLE_STRIDED_BATCHED"] = "1"

    sys.path.insert(0, GDN_REMOTE_ROOT)
    from prefill.solution.cuda.chunked_proto import (
        chunked_prefill_end_to_end,
        chunked_prefill_end_to_end_batched,
        chunked_prefill_end_to_end_grouped,
        get_last_gemm_batched_path,
        chunked_prefill_end_to_end_uniform_batch,
    )
    from prefill.solution.triton.kernel import kernel as triton_prefill_kernel

    def reference_prefill_end_to_end(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        total_tokens, num_q_heads, D = q.shape
        num_v_heads = v.shape[1]
        num_seqs = cu_seqlens.numel() - 1
        device = q.device

        x = a.float() + dt_bias.float()
        g_all = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
        beta_all = torch.sigmoid(b.float())

        out = torch.empty(total_tokens, num_v_heads, D, dtype=torch.bfloat16, device=device)
        new_state = state.clone().float()

        for seq_idx in range(num_seqs):
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())
            if end <= start:
                continue
            for h in range(num_v_heads):
                qk_h = h // 2
                state_h = new_state[seq_idx, h].clone()
                for t in range(start, end):
                    q_h = q[t, qk_h].float()
                    k_h = k[t, qk_h].float()
                    v_h = v[t, h].float()
                    g_h = g_all[t, h]
                    beta_h = beta_all[t, h]

                    state_h = g_h * state_h
                    old_v = state_h @ k_h
                    delta = beta_h * (v_h - old_v)
                    state_h = state_h + delta[:, None] * k_h[None, :]
                    out[t, h] = (scale * (state_h @ q_h)).to(torch.bfloat16)
                new_state[seq_idx, h] = state_h

        return out, new_state

    def bench_cuda(fn, warmup=2, iters=5):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        torch.cuda.synchronize()
        return start.elapsed_time(stop) / iters

    device = "cuda"
    D = 128
    scale = 1.0 / math.sqrt(D)
    results = []

    for cfg in [
        {"name": "N1_L64", "lengths": [64], "chunk_size": 64},
        {"name": "N4_L32", "lengths": [32, 32, 32, 32], "chunk_size": 64},
        {"name": "grouped_varlen", "lengths": [32, 32, 64, 64], "chunk_size": 64},
    ]:
        lengths = cfg["lengths"]
        num_seqs = len(lengths)
        total_tokens = sum(lengths)
        chunk_size = cfg["chunk_size"]

        torch.manual_seed(123 + total_tokens)
        q = (torch.randn(total_tokens, 4, D, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        k = F.normalize(torch.randn(total_tokens, 4, D, device=device, dtype=torch.float32), dim=-1).to(torch.bfloat16)
        v = (torch.randn(total_tokens, 8, D, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        state = torch.randn(num_seqs, 8, D, D, device=device, dtype=torch.float32) * 0.01
        A_log = torch.randn(8, device=device, dtype=torch.float32) * 0.1 - 1.0
        a = torch.randn(total_tokens, 8, device=device, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(8, device=device, dtype=torch.float32) * 0.1
        b = torch.randn(total_tokens, 8, device=device, dtype=torch.float32) * 0.1
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.int32)

        ref_out, ref_state = reference_prefill_end_to_end(
            q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale
        )
        triton_out, triton_state = triton_prefill_kernel(
            q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale
        )
        proto_out, proto_state = chunked_prefill_end_to_end(
            q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
        )
        batched_out, batched_state = chunked_prefill_end_to_end_batched(
            q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
        )
        grouped_out, grouped_state = chunked_prefill_end_to_end_grouped(
            q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
        )
        gemm_batched_path = get_last_gemm_batched_path()

        entry = {
            "name": cfg["name"],
            "chunk_size": chunk_size,
            "out_diff": (proto_out.float() - ref_out.float()).abs().max().item(),
            "state_diff": (proto_state - ref_state).abs().max().item(),
            "batched_out_diff": (batched_out.float() - ref_out.float()).abs().max().item(),
            "batched_state_diff": (batched_state - ref_state).abs().max().item(),
            "grouped_out_diff": (grouped_out.float() - ref_out.float()).abs().max().item(),
            "grouped_state_diff": (grouped_state - ref_state).abs().max().item(),
            "gemm_batched_path": gemm_batched_path,
            "triton_out_diff": (triton_out.float() - ref_out.float()).abs().max().item(),
            "triton_state_diff": (triton_state - ref_state).abs().max().item(),
            "reference_ms": bench_cuda(
                lambda: reference_prefill_end_to_end(
                    q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale
                )
            ),
            "triton_ms": bench_cuda(
                lambda: triton_prefill_kernel(
                    q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale
                )
            ),
            "prototype_ms": bench_cuda(
                lambda: chunked_prefill_end_to_end(
                    q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
                )
            ),
            "batched_ms": bench_cuda(
                lambda: chunked_prefill_end_to_end_batched(
                    q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
                )
            ),
            "grouped_ms": bench_cuda(
                lambda: chunked_prefill_end_to_end_grouped(
                    q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
                )
            ),
        }
        entry["speedup_vs_reference"] = entry["reference_ms"] / entry["prototype_ms"]
        entry["speedup_vs_triton"] = entry["triton_ms"] / entry["prototype_ms"]
        entry["batched_speedup_vs_reference"] = entry["reference_ms"] / entry["batched_ms"]
        entry["batched_speedup_vs_triton"] = entry["triton_ms"] / entry["batched_ms"]
        entry["batched_speedup_vs_single_chunkproto"] = entry["prototype_ms"] / entry["batched_ms"]
        entry["grouped_speedup_vs_reference"] = entry["reference_ms"] / entry["grouped_ms"]
        entry["grouped_speedup_vs_triton"] = entry["triton_ms"] / entry["grouped_ms"]
        entry["grouped_speedup_vs_batched"] = entry["batched_ms"] / entry["grouped_ms"]
        results.append(entry)
        print(entry)

    return {"status": "ok", "results": results}


@app.function(
    image=cuda_image,
    gpu=B200_GPU,
    timeout=LONG_TIMEOUT,
)
def run_chunked_prefill_sweep():
    import math
    import sys

    import torch
    import torch.nn.functional as F

    sys.path.insert(0, GDN_REMOTE_ROOT)
    from prefill.solution.cuda.chunked_proto import (
        chunked_prefill_end_to_end_batched,
        chunked_prefill_end_to_end_grouped,
        chunked_prefill_end_to_end_uniform_batch,
        kernel as chunked_kernel,
    )
    from prefill.solution.triton.kernel import kernel as triton_prefill_kernel

    def reference_prefill_end_to_end(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        total_tokens, _, D = q.shape
        num_v_heads = v.shape[1]
        num_seqs = cu_seqlens.numel() - 1
        device = q.device

        x = a.float() + dt_bias.float()
        g_all = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
        beta_all = torch.sigmoid(b.float())

        out = torch.empty(total_tokens, num_v_heads, D, dtype=torch.bfloat16, device=device)
        new_state = state.clone().float()

        for seq_idx in range(num_seqs):
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())
            if end <= start:
                continue
            for h in range(num_v_heads):
                qk_h = h // 2
                state_h = new_state[seq_idx, h].clone()
                for t in range(start, end):
                    q_h = q[t, qk_h].float()
                    k_h = k[t, qk_h].float()
                    v_h = v[t, h].float()
                    g_h = g_all[t, h]
                    beta_h = beta_all[t, h]

                    state_h = g_h * state_h
                    old_v = state_h @ k_h
                    delta = beta_h * (v_h - old_v)
                    state_h = state_h + delta[:, None] * k_h[None, :]
                    out[t, h] = (scale * (state_h @ q_h)).to(torch.bfloat16)
                new_state[seq_idx, h] = state_h

        return out, new_state

    def bench_cuda(fn, warmup=2, iters=5):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        torch.cuda.synchronize()
        return start.elapsed_time(stop) / iters

    device = "cuda"
    D = 128
    scale = 1.0 / math.sqrt(D)
    chunk_sizes = [4, 8, 16, 32, 64]
    configs = [
        {"name": "N1_L64", "lengths": [64]},
        {"name": "N1_L128", "lengths": [128]},
        {"name": "N4_L32", "lengths": [32, 32, 32, 32]},
        {"name": "N4_L64", "lengths": [64, 64, 64, 64]},
        {"name": "grouped_32_32_64_64", "lengths": [32, 32, 64, 64]},
        {"name": "grouped_32_64_128_256", "lengths": [32, 64, 128, 256]},
    ]

    results = []
    best_by_case = []

    for cfg in configs:
        lengths = cfg["lengths"]
        num_seqs = len(lengths)
        total_tokens = sum(lengths)
        uniform = all(length == lengths[0] for length in lengths)

        torch.manual_seed(321 + total_tokens)
        q = (torch.randn(total_tokens, 4, D, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        k = F.normalize(torch.randn(total_tokens, 4, D, device=device, dtype=torch.float32), dim=-1).to(torch.bfloat16)
        v = (torch.randn(total_tokens, 8, D, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        state = torch.randn(num_seqs, 8, D, D, device=device, dtype=torch.float32) * 0.01
        A_log = torch.randn(8, device=device, dtype=torch.float32) * 0.1 - 1.0
        a = torch.randn(total_tokens, 8, device=device, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(8, device=device, dtype=torch.float32) * 0.1
        b = torch.randn(total_tokens, 8, device=device, dtype=torch.float32) * 0.1
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.int32)

        ref_out, ref_state = reference_prefill_end_to_end(
            q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale
        )
        triton_out, triton_state = triton_prefill_kernel(
            q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale
        )
        triton_ms = bench_cuda(
            lambda: triton_prefill_kernel(
                q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale
            )
        )
        reference_ms = bench_cuda(
            lambda: reference_prefill_end_to_end(
                q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale
            )
        )

        case_entries = []
        for chunk_size in chunk_sizes:
            batched_out, batched_state = chunked_prefill_end_to_end_batched(
                q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
            )
            grouped_out, grouped_state = chunked_prefill_end_to_end_grouped(
                q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
            )
            kernel_out, kernel_state = chunked_kernel(
                q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
            )

            entry = {
                "name": cfg["name"],
                "lengths": lengths,
                "chunk_size": chunk_size,
                "uniform_case": uniform,
                "reference_ms": reference_ms,
                "triton_ms": triton_ms,
                "triton_out_diff": (triton_out.float() - ref_out.float()).abs().max().item(),
                "triton_state_diff": (triton_state - ref_state).abs().max().item(),
                "batched_out_diff": (batched_out.float() - ref_out.float()).abs().max().item(),
                "batched_state_diff": (batched_state - ref_state).abs().max().item(),
                "grouped_out_diff": (grouped_out.float() - ref_out.float()).abs().max().item(),
                "grouped_state_diff": (grouped_state - ref_state).abs().max().item(),
                "kernel_out_diff": (kernel_out.float() - ref_out.float()).abs().max().item(),
                "kernel_state_diff": (kernel_state - ref_state).abs().max().item(),
                "batched_ms": bench_cuda(
                    lambda: chunked_prefill_end_to_end_batched(
                        q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
                    )
                ),
                "grouped_ms": bench_cuda(
                    lambda: chunked_prefill_end_to_end_grouped(
                        q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
                    )
                ),
                "kernel_ms": bench_cuda(
                    lambda: chunked_kernel(
                        q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
                    )
                ),
            }

            if uniform:
                uniform_out, uniform_state = chunked_prefill_end_to_end_uniform_batch(
                    q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
                )
                entry["uniform_out_diff"] = (uniform_out.float() - ref_out.float()).abs().max().item()
                entry["uniform_state_diff"] = (uniform_state - ref_state).abs().max().item()
                entry["uniform_ms"] = bench_cuda(
                    lambda: chunked_prefill_end_to_end_uniform_batch(
                        q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale, chunk_size=chunk_size
                    )
                )
            else:
                entry["uniform_out_diff"] = None
                entry["uniform_state_diff"] = None
                entry["uniform_ms"] = None

            entry["kernel_speedup_vs_reference"] = reference_ms / entry["kernel_ms"]
            entry["kernel_speedup_vs_triton"] = triton_ms / entry["kernel_ms"]
            entry["grouped_speedup_vs_batched"] = entry["batched_ms"] / entry["grouped_ms"]
            case_entries.append(entry)
            results.append(entry)
            print(entry)

        best = min(case_entries, key=lambda item: item["kernel_ms"])
        best_by_case.append(
            {
                "name": cfg["name"],
                "lengths": lengths,
                "best_chunk_size": best["chunk_size"],
                "best_kernel_ms": best["kernel_ms"],
                "triton_ms": triton_ms,
                "reference_ms": reference_ms,
                "kernel_speedup_vs_reference": best["kernel_speedup_vs_reference"],
                "kernel_speedup_vs_triton": best["kernel_speedup_vs_triton"],
            }
        )
        print({"best_case": best_by_case[-1]})

    return {"status": "ok", "results": results, "best_by_case": best_by_case}


@app.function(
    image=cuda_image,
    gpu=B200_GPU,
    timeout=LONG_TIMEOUT,
)
def probe_cutlass_sm100():
    from pathlib import Path

    queries = {
        "sm100": r"SM100_",
        "tiled_mma": r"make_tiled_mma",
        "mma_atom": r"MMA_Atom",
        "tcgen05": r"tcgen05",
        "tmem": r"TMEM|tmem",
        "sm100_f16bf16": r"SM100_MMA_F16BF16",
        "make_tmem_copy": r"make_tmem_copy",
        "sm100_tmem_load": r"SM100_TMEM_LOAD",
        "umma_major": r"enum class Major|UMMA::Major::",
        "gemm_batched_mode": r"kBatched|GemmUniversalMode::kBatched",
        "gemm_grouped_mode": r"kGrouped|GemmUniversalMode::kGrouped",
        "gemm_problem_count": r"problem_count|batch_count",
        "gemm_batch_stride": r"batch_stride|lda_batch|ldb_batch|ldc_batch|ldd_batch",
    }

    search_roots = [Path("/opt/cutlass/include/cute"), Path("/opt/cutlass/include/cutlass")]
    extra_roots = [Path("/opt/cutlass/examples")]
    results = {}
    for name, pattern in queries.items():
        import re

        regex = re.compile(pattern)
        output = []
        for root in search_roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    text = path.read_text(errors="ignore")
                except Exception:
                    continue
                for lineno, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        output.append(f"{path}:{lineno}:{line.strip()}")
                        if len(output) >= 40:
                            break
                if len(output) >= 40:
                    break
            if len(output) >= 40:
                break
        results[name] = output[:40]
        print(f"\n== {name} ==")
        for line in output[:20]:
            print(line)

    example_queries = {
        "examples_sm100": r"sm100|blackwell|tcgen05|TMEM",
        "examples_gemm_adapter": r"GemmUniversalAdapter|device::Gemm|CollectiveBuilder",
    }
    for name, pattern in example_queries.items():
        import re

        regex = re.compile(pattern, re.IGNORECASE)
        output = []
        for root in extra_roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    text = path.read_text(errors="ignore")
                except Exception:
                    continue
                for lineno, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        output.append(f"{path}:{lineno}:{line.strip()}")
                        if len(output) >= 40:
                            break
                if len(output) >= 40:
                    break
            if len(output) >= 40:
                break
        results[name] = output[:40]
        print(f"\n== {name} ==")
        for line in output[:20]:
            print(line)

    snippets = [
        (Path("/opt/cutlass/include/cute/arch/mma_sm100_umma.hpp"), 1, 80),
        (Path("/opt/cutlass/include/cute/arch/mma_sm100_umma.hpp"), 80, 135),
        (Path("/opt/cutlass/include/cutlass/gemm/collective/builders/sm100_common.inl"), 315, 335),
        (Path("/opt/cutlass/include/cutlass/gemm/collective/builders/sm100_common.inl"), 385, 405),
        (Path("/opt/cutlass/include/cute/atom/copy_traits_sm100.hpp"), 268, 320),
        (Path("/opt/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h"), 1, 260),
        (Path("/opt/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h"), 260, 520),
        (Path("/opt/cutlass/include/cutlass/gemm/kernel/gemm_universal_decl.h"), 1, 260),
        (Path("/opt/cutlass/include/cutlass/gemm/kernel/gemm_universal_decl.h"), 260, 520),
        (Path("/opt/cutlass/include/cutlass/gemm/gemm_enumerated_types.h"), 1, 160),
        (Path("/opt/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp"), 1, 260),
        (Path("/opt/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp"), 260, 520),
        (Path("/opt/cutlass/examples/70_blackwell_gemm/70_blackwell_fp16_gemm.cu"), 1, 260),
        (Path("/opt/cutlass/examples/70_blackwell_gemm/70_blackwell_fp16_gemm.cu"), 260, 420),
    ]
    results["snippets"] = {}
    for path, start, end in snippets:
        key = f"{path.name}:{start}-{end}"
        lines = []
        if path.exists():
            text_lines = path.read_text(errors="ignore").splitlines()
            for lineno in range(start, min(end, len(text_lines)) + 1):
                lines.append(f"{lineno}: {text_lines[lineno - 1]}")
        results["snippets"][key] = lines
        print(f"\n== snippet {key} ==")
        for line in lines[:40]:
            print(line)

    example_dir = Path("/opt/cutlass/examples/70_blackwell_gemm")
    example_listing = []
    if example_dir.exists():
        for path in sorted(example_dir.rglob("*")):
            if path.is_file():
                example_listing.append(str(path))
    results["example70_files"] = example_listing[:40]
    print("\n== example70_files ==")
    for line in example_listing[:40]:
        print(line)

    return results


@app.local_entrypoint()
def main(mode: str = "bench"):
    kernel_base = get_kernel_base_path()
    sources = read_kernel_sources(kernel_base)
    needed = {
        name: sources[name]
        for name in ["gdn_prefill_v10.cuh", "gdn_prefill_ptx.cuh"]
        if name in sources
    }
    print(f"Using kernel sources: {sorted(needed.keys())}")
    if mode == "compile":
        result = compile_prefill_tensorcore.remote(needed)
    elif mode == "instantiate":
        result = instantiate_sm100_types.remote()
    elif mode == "example70":
        result = run_blackwell_example_gemm.remote()
    elif mode == "microgemm":
        result = run_repo_local_blackwell_microgemm.remote()
    elif mode == "microgemm_lib":
        result = run_repo_local_blackwell_microgemm_lib.remote()
    elif mode == "chunkproto":
        result = run_chunked_prefill_prototype.remote()
    elif mode == "chunkmodule":
        result = run_chunked_prefill_module.remote()
    elif mode == "chunkmodule_nobatched":
        result = run_chunked_prefill_module_no_strided_batched.remote()
    elif mode == "chunksweep":
        result = run_chunked_prefill_sweep.remote()
    elif mode == "mma":
        result = invoke_sm100_mma_smoke.remote()
    elif mode == "probe":
        result = probe_cutlass_sm100.remote()
    else:
        result = benchmark_prefill_tensorcore.remote(needed)
    print(result)

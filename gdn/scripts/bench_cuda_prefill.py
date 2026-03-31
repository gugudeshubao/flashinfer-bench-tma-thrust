"""
Benchmark: CuTe C++ v11 Prefill Kernel Performance Test

Compiles and runs the CUDA kernel on Modal B200.
Compares with Triton v5 baseline.
"""
import modal

app = modal.App("gdn-cuda-prefill-benchmark")

# Image with CUDA compilation support
cuda_image = modal.Image.from_registry(
    "nvidia/cuda:12.6.0-devel-ubuntu22.04",
    add_python="3.11"
).pip_install(
    "torch", "triton", "ninja", "pybind11"
).run_commands(
    # Install build essentials
    "apt-get update && apt-get install -y build-essential"
)


@app.function(image=cuda_image, gpu="B200", timeout=600)
def benchmark_cuda_kernels():
    """Benchmark CUDA prefill kernels vs Triton"""
    import torch
    import time
    import subprocess
    import os
    
    # Check GPU
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # For now, let's benchmark the Triton kernel as baseline
    # and prepare the CUDA compilation infrastructure
    
    D, num_q_heads, num_v_heads = 128, 4, 8
    device = "cuda"
    
    # Import Triton kernel (embedded inline to avoid Modal path issues)
    import triton
    import triton.language as tl
    import math
    
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

    def run_triton_v5(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        T, num_q_heads, D = q.shape
        num_v_heads = v.shape[1]
        N = cu_seqlens.shape[0] - 1
        device = q.device

        BLOCK_V = 16 if N <= 4 else 32
        V_BLOCKS = D // BLOCK_V

        if scale is None or scale == 0.0:
            scale = 1.0 / math.sqrt(D)

        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        a_c = a.contiguous()
        b_c = b.contiguous()
        cu = cu_seqlens.contiguous()

        S = state.contiguous() if state is not None else torch.zeros(N, num_v_heads, D, D, dtype=torch.float32, device=device)
        out = torch.empty(T, num_v_heads, D, dtype=torch.bfloat16, device=device)
        new_S = torch.empty_like(S)

        _prefill_kernel_v5[(N, num_v_heads, V_BLOCKS)](
            q_c, k_c, v_c, S, A_log, a_c, dt_bias, b_c, cu, out, new_S,
            float(scale),
            q_c.stride(0), q_c.stride(1), k_c.stride(0), k_c.stride(1),
            v_c.stride(0), v_c.stride(1), S.stride(0), S.stride(1), S.stride(2),
            a_c.stride(0), b_c.stride(0), out.stride(0), out.stride(1),
            new_S.stride(0), new_S.stride(1), new_S.stride(2),
            D=128, BLOCK_V=BLOCK_V, num_warps=4,
        )
        return out, new_S

    # Test configurations
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
    print("\n" + "="*80)
    print("Triton v5 Software Pipelining Benchmark (Baseline)")
    print("="*80)
    
    for N, seq_len in configs:
        T = N * seq_len
        
        # Generate inputs
        torch.manual_seed(42)
        q = torch.randn(T, num_q_heads, D, dtype=torch.bfloat16, device=device) * 0.1
        k = torch.randn(T, num_q_heads, D, dtype=torch.bfloat16, device=device) * 0.1
        v = torch.randn(T, num_v_heads, D, dtype=torch.bfloat16, device=device) * 0.1
        state = torch.zeros(N, num_v_heads, D, D, dtype=torch.float32, device=device)
        A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1 - 1.0
        a = torch.randn(T, num_v_heads, dtype=torch.float32, device=device) * 0.1
        dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1
        b = torch.randn(T, num_v_heads, dtype=torch.float32, device=device) * 0.1
        cu_seqlens = torch.arange(0, T + 1, seq_len, dtype=torch.int32, device=device)
        scale = 1.0 / (D ** 0.5)
        
        # Warmup
        for _ in range(3):
            run_triton_v5(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
        torch.cuda.synchronize()
        
        # Benchmark
        iters = 50
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            run_triton_v5(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / iters * 1000
        
        throughput = T / elapsed * 1000 / 1e6
        
        results.append({
            "N": N,
            "seq_len": seq_len,
            "T": T,
            "time_ms": elapsed,
            "throughput_Mtok_s": throughput,
        })
        
        print(f"N={N:2d}, L={seq_len:4d}, T={T:5d} | "
              f"{elapsed:.3f}ms | {throughput:.2f} M tok/s")
    
    # Summary
    print("\n" + "="*80)
    print("Performance Summary")
    print("="*80)
    
    # Best single-sequence
    single_seq = [r for r in results if r["N"] == 1]
    if single_seq:
        best_single = max(single_seq, key=lambda x: x["throughput_Mtok_s"])
        print(f"Best single-sequence: {best_single['throughput_Mtok_s']:.2f} M tok/s @ L={best_single['seq_len']}")
    
    # Best multi-batch
    multi_batch = [r for r in results if r["N"] > 1]
    if multi_batch:
        best_multi = max(multi_batch, key=lambda x: x["throughput_Mtok_s"])
        print(f"Best multi-batch: {best_multi['throughput_Mtok_s']:.2f} M tok/s @ N={best_multi['N']}, L={best_multi['seq_len']}")
    
    # Check NVCC availability
    print("\n" + "="*80)
    print("CUDA Compilation Check")
    print("="*80)
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        print(result.stdout)
        print("NVCC is available for CUDA kernel compilation")
    except Exception as e:
        print(f"NVCC not found: {e}")
    
    return results


@app.local_entrypoint()
def main():
    print("="*80)
    print("GDN Prefill Kernel Performance Benchmark")
    print("Hardware: NVIDIA B200")
    print("="*80)
    results = benchmark_cuda_kernels.remote()
    print("\nBenchmark completed!")

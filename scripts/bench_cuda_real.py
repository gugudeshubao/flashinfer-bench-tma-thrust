#!/usr/bin/env python3
"""
Benchmark REAL CUDA kernels (v7, v8) vs Triton (v5)

Usage:
    modal run scripts/bench_cuda_real.py
"""

import modal

app = modal.App("gdn-bench-cuda-real")

cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4.0", "numpy", "tabulate")
    .run_commands("pip install triton>=3.0.0")
)

volume = modal.Volume.from_name("flashinfer-bench", create_if_missing=True)


@app.function(
    image=cuda_image,
    gpu="B200",
    timeout=600,
    volumes={"/data": volume},
)
def benchmark_cuda_real():
    """Benchmark real CUDA v7/v8 kernels vs Triton v5."""
    import torch
    import triton
    import triton.language as tl
    import ctypes
    import math
    import numpy as np
    from pathlib import Path
    from tabulate import tabulate

    # ============================================================
    # GPU Info
    # ============================================================
    props = torch.cuda.get_device_properties(0)
    print("=" * 80)
    print(f"REAL CUDA vs Triton Benchmark on {props.name}")
    print("=" * 80)

    # ============================================================
    # Load CUDA Library
    # ============================================================
    lib_path = Path("/data/lib/libgdn_kernels.so")
    if not lib_path.exists():
        print(f"ERROR: {lib_path} not found. Run build_cuda.py first.")
        return {"status": "error", "error": "Library not found"}
    
    lib = ctypes.CDLL(str(lib_path))
    print(f"Loaded: {lib_path}")
    
    # Setup function signatures
    # void gdn_decode_v7_fp32(
    #     const void* Q, K, V, State, A_log, A, DtBias, B_gate,
    #     void* Out, NewState,
    #     float scale, int B, num_v_heads, D,
    #     int strides... (16 ints),
    #     int BLOCK_V, void* stream)
    
    lib.gdn_decode_v7_fp32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # Q, K, V, State
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # A_log, A, DtBias, B_gate
        ctypes.c_void_p, ctypes.c_void_p,  # Out, NewState
        ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # scale, B, num_v_heads, D
        # 16 stride ints
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_void_p  # BLOCK_V, stream
    ]
    lib.gdn_decode_v7_fp32.restype = None
    
    lib.gdn_decode_v8_fp32.argtypes = lib.gdn_decode_v7_fp32.argtypes
    lib.gdn_decode_v8_fp32.restype = None
    
    # CUDA Graph version for low-latency
    lib.gdn_decode_v7_graph_launch.argtypes = lib.gdn_decode_v7_fp32.argtypes
    lib.gdn_decode_v7_graph_launch.restype = None
    
    # v9 CuTe/Swizzle versions
    lib.gdn_decode_v9_fp32.argtypes = lib.gdn_decode_v7_fp32.argtypes
    lib.gdn_decode_v9_fp32.restype = None
    
    lib.gdn_decode_v9_tma.argtypes = lib.gdn_decode_v7_fp32.argtypes
    lib.gdn_decode_v9_tma.restype = None
    
    # v10 CuTe DSL + Library versions
    lib.gdn_decode_v10_cute.argtypes = lib.gdn_decode_v7_fp32.argtypes
    lib.gdn_decode_v10_cute.restype = None
    
    lib.gdn_decode_v10_tma.argtypes = lib.gdn_decode_v7_fp32.argtypes
    lib.gdn_decode_v10_tma.restype = None
    
    print("CUDA functions loaded: v7, v8, v9, v10 (cute, tma)")

    # ============================================================
    # Triton v5 Kernel (Baseline)
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

    def run_triton_v5(q, k, v, state, A_log, a, dt_bias, b_gate, scale, BLOCK_V):
        B, _, num_q_heads, D = q.shape
        num_v_heads = v.shape[2]
        V_BLOCKS = D // BLOCK_V

        q_c = q.squeeze(1).contiguous()
        k_c = k.squeeze(1).contiguous()
        v_c = v.squeeze(1).contiguous()
        a_c = a.squeeze(1).contiguous()
        b_c = b_gate.squeeze(1).contiguous()
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

    def run_cuda_v7(q, k, v, state, A_log, a, dt_bias, b_gate, scale, BLOCK_V):
        """Call real CUDA v7 kernel via ctypes."""
        B, _, num_q_heads, D = q.shape
        num_v_heads = v.shape[2]
        
        q_c = q.squeeze(1).contiguous()
        k_c = k.squeeze(1).contiguous()
        v_c = v.squeeze(1).contiguous()
        a_c = a.squeeze(1).contiguous()
        b_c = b_gate.squeeze(1).contiguous()
        S = state.contiguous()
        
        out = torch.empty(B, num_v_heads, D, dtype=torch.bfloat16, device='cuda')
        new_S = torch.empty_like(S)
        
        lib.gdn_decode_v7_fp32(
            q_c.data_ptr(), k_c.data_ptr(), v_c.data_ptr(), S.data_ptr(),
            A_log.data_ptr(), a_c.data_ptr(), dt_bias.data_ptr(), b_c.data_ptr(),
            out.data_ptr(), new_S.data_ptr(),
            ctypes.c_float(scale), B, num_v_heads, D,
            q_c.stride(0), q_c.stride(1),
            k_c.stride(0), k_c.stride(1),
            v_c.stride(0), v_c.stride(1),
            S.stride(0), S.stride(1), S.stride(2),
            a_c.stride(0), b_c.stride(0),
            out.stride(0), out.stride(1),
            new_S.stride(0), new_S.stride(1), new_S.stride(2),
            BLOCK_V, ctypes.c_void_p(0)  # null stream = default
        )
        torch.cuda.synchronize()
        return out.unsqueeze(1), new_S

    def run_cuda_v8(q, k, v, state, A_log, a, dt_bias, b_gate, scale, BLOCK_V):
        """Call real CUDA v8 kernel via ctypes."""
        B, _, num_q_heads, D = q.shape
        num_v_heads = v.shape[2]
        
        q_c = q.squeeze(1).contiguous()
        k_c = k.squeeze(1).contiguous()
        v_c = v.squeeze(1).contiguous()
        a_c = a.squeeze(1).contiguous()
        b_c = b_gate.squeeze(1).contiguous()
        S = state.contiguous()
        
        out = torch.empty(B, num_v_heads, D, dtype=torch.bfloat16, device='cuda')
        new_S = torch.empty_like(S)
        
        lib.gdn_decode_v8_fp32(
            q_c.data_ptr(), k_c.data_ptr(), v_c.data_ptr(), S.data_ptr(),
            A_log.data_ptr(), a_c.data_ptr(), dt_bias.data_ptr(), b_c.data_ptr(),
            out.data_ptr(), new_S.data_ptr(),
            ctypes.c_float(scale), B, num_v_heads, D,
            q_c.stride(0), q_c.stride(1),
            k_c.stride(0), k_c.stride(1),
            v_c.stride(0), v_c.stride(1),
            S.stride(0), S.stride(1), S.stride(2),
            a_c.stride(0), b_c.stride(0),
            out.stride(0), out.stride(1),
            new_S.stride(0), new_S.stride(1), new_S.stride(2),
            BLOCK_V, ctypes.c_void_p(0)
        )
        torch.cuda.synchronize()
        return out.unsqueeze(1), new_S

    def run_cuda_v7_graph(q, k, v, state, A_log, a, dt_bias, b_gate, scale, BLOCK_V):
        """Call CUDA v7 with CUDA Graph for low-latency launch."""
        B, _, num_q_heads, D = q.shape
        num_v_heads = v.shape[2]
        
        q_c = q.squeeze(1).contiguous()
        k_c = k.squeeze(1).contiguous()
        v_c = v.squeeze(1).contiguous()
        a_c = a.squeeze(1).contiguous()
        b_c = b_gate.squeeze(1).contiguous()
        S = state.contiguous()
        
        out = torch.empty(B, num_v_heads, D, dtype=torch.bfloat16, device='cuda')
        new_S = torch.empty_like(S)
        
        lib.gdn_decode_v7_graph_launch(
            q_c.data_ptr(), k_c.data_ptr(), v_c.data_ptr(), S.data_ptr(),
            A_log.data_ptr(), a_c.data_ptr(), dt_bias.data_ptr(), b_c.data_ptr(),
            out.data_ptr(), new_S.data_ptr(),
            ctypes.c_float(scale), B, num_v_heads, D,
            q_c.stride(0), q_c.stride(1),
            k_c.stride(0), k_c.stride(1),
            v_c.stride(0), v_c.stride(1),
            S.stride(0), S.stride(1), S.stride(2),
            a_c.stride(0), b_c.stride(0),
            out.stride(0), out.stride(1),
            new_S.stride(0), new_S.stride(1), new_S.stride(2),
            BLOCK_V, ctypes.c_void_p(0)
        )
        torch.cuda.synchronize()
        return out.unsqueeze(1), new_S

    def run_cuda_v9_fp32(q, k, v, state, A_log, a, dt_bias, b_gate, scale, BLOCK_V):
        """Call CUDA v9 with CuTe swizzled SMEM."""
        B, _, num_q_heads, D = q.shape
        num_v_heads = v.shape[2]
        
        q_c = q.squeeze(1).contiguous()
        k_c = k.squeeze(1).contiguous()
        v_c = v.squeeze(1).contiguous()
        a_c = a.squeeze(1).contiguous()
        b_c = b_gate.squeeze(1).contiguous()
        S = state.contiguous()
        
        out = torch.empty(B, num_v_heads, D, dtype=torch.bfloat16, device='cuda')
        new_S = torch.empty_like(S)
        
        lib.gdn_decode_v9_fp32(
            q_c.data_ptr(), k_c.data_ptr(), v_c.data_ptr(), S.data_ptr(),
            A_log.data_ptr(), a_c.data_ptr(), dt_bias.data_ptr(), b_c.data_ptr(),
            out.data_ptr(), new_S.data_ptr(),
            ctypes.c_float(scale), B, num_v_heads, D,
            q_c.stride(0), q_c.stride(1),
            k_c.stride(0), k_c.stride(1),
            v_c.stride(0), v_c.stride(1),
            S.stride(0), S.stride(1), S.stride(2),
            a_c.stride(0), b_c.stride(0),
            out.stride(0), out.stride(1),
            new_S.stride(0), new_S.stride(1), new_S.stride(2),
            BLOCK_V, ctypes.c_void_p(0)
        )
        torch.cuda.synchronize()
        return out.unsqueeze(1), new_S

    def run_cuda_v9_tma(q, k, v, state, A_log, a, dt_bias, b_gate, scale, BLOCK_V):
        """Call CUDA v9 with TMA swizzled loads."""
        B, _, num_q_heads, D = q.shape
        num_v_heads = v.shape[2]
        
        q_c = q.squeeze(1).contiguous()
        k_c = k.squeeze(1).contiguous()
        v_c = v.squeeze(1).contiguous()
        a_c = a.squeeze(1).contiguous()
        b_c = b_gate.squeeze(1).contiguous()
        S = state.contiguous()
        
        out = torch.empty(B, num_v_heads, D, dtype=torch.bfloat16, device='cuda')
        new_S = torch.empty_like(S)
        
        lib.gdn_decode_v9_tma(
            q_c.data_ptr(), k_c.data_ptr(), v_c.data_ptr(), S.data_ptr(),
            A_log.data_ptr(), a_c.data_ptr(), dt_bias.data_ptr(), b_c.data_ptr(),
            out.data_ptr(), new_S.data_ptr(),
            ctypes.c_float(scale), B, num_v_heads, D,
            q_c.stride(0), q_c.stride(1),
            k_c.stride(0), k_c.stride(1),
            v_c.stride(0), v_c.stride(1),
            S.stride(0), S.stride(1), S.stride(2),
            a_c.stride(0), b_c.stride(0),
            out.stride(0), out.stride(1),
            new_S.stride(0), new_S.stride(1), new_S.stride(2),
            BLOCK_V, ctypes.c_void_p(0)
        )
        torch.cuda.synchronize()
        return out.unsqueeze(1), new_S

    def run_cuda_v10_cute(q, k, v, state, A_log, a, dt_bias, b_gate, scale, BLOCK_V):
        """Call CUDA v10 with CuTe DSL."""
        B, _, num_q_heads, D = q.shape
        num_v_heads = v.shape[2]
        
        q_c = q.squeeze(1).contiguous()
        k_c = k.squeeze(1).contiguous()
        v_c = v.squeeze(1).contiguous()
        a_c = a.squeeze(1).contiguous()
        b_c = b_gate.squeeze(1).contiguous()
        S = state.contiguous()
        
        out = torch.empty(B, num_v_heads, D, dtype=torch.bfloat16, device='cuda')
        new_S = torch.empty_like(S)
        
        lib.gdn_decode_v10_cute(
            q_c.data_ptr(), k_c.data_ptr(), v_c.data_ptr(), S.data_ptr(),
            A_log.data_ptr(), a_c.data_ptr(), dt_bias.data_ptr(), b_c.data_ptr(),
            out.data_ptr(), new_S.data_ptr(),
            ctypes.c_float(scale), B, num_v_heads, D,
            q_c.stride(0), q_c.stride(1),
            k_c.stride(0), k_c.stride(1),
            v_c.stride(0), v_c.stride(1),
            S.stride(0), S.stride(1), S.stride(2),
            a_c.stride(0), b_c.stride(0),
            out.stride(0), out.stride(1),
            new_S.stride(0), new_S.stride(1), new_S.stride(2),
            BLOCK_V, ctypes.c_void_p(0)
        )
        torch.cuda.synchronize()
        return out.unsqueeze(1), new_S

    def run_cuda_v10_tma(q, k, v, state, A_log, a, dt_bias, b_gate, scale, BLOCK_V):
        """Call CUDA v10 with TMA async copy."""
        B, _, num_q_heads, D = q.shape
        num_v_heads = v.shape[2]
        
        q_c = q.squeeze(1).contiguous()
        k_c = k.squeeze(1).contiguous()
        v_c = v.squeeze(1).contiguous()
        a_c = a.squeeze(1).contiguous()
        b_c = b_gate.squeeze(1).contiguous()
        S = state.contiguous()
        
        out = torch.empty(B, num_v_heads, D, dtype=torch.bfloat16, device='cuda')
        new_S = torch.empty_like(S)
        
        lib.gdn_decode_v10_tma(
            q_c.data_ptr(), k_c.data_ptr(), v_c.data_ptr(), S.data_ptr(),
            A_log.data_ptr(), a_c.data_ptr(), dt_bias.data_ptr(), b_c.data_ptr(),
            out.data_ptr(), new_S.data_ptr(),
            ctypes.c_float(scale), B, num_v_heads, D,
            q_c.stride(0), q_c.stride(1),
            k_c.stride(0), k_c.stride(1),
            v_c.stride(0), v_c.stride(1),
            S.stride(0), S.stride(1), S.stride(2),
            a_c.stride(0), b_c.stride(0),
            out.stride(0), out.stride(1),
            new_S.stride(0), new_S.stride(1), new_S.stride(2),
            BLOCK_V, ctypes.c_void_p(0)
        )
        torch.cuda.synchronize()
        return out.unsqueeze(1), new_S

    # ============================================================
    # Correctness Validation
    # ============================================================
    print("\n" + "=" * 80)
    print("CORRECTNESS VALIDATION")
    print("=" * 80)
    
    def check_correctness(name, test_fn, ref_fn, batch, BLOCK_V, atol=1e-2, rtol=1e-2):
        """Compare kernel output against reference (Triton v5)."""
        torch.manual_seed(42)
        q = torch.randn(batch, 1, num_q_heads, D, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(batch, 1, num_q_heads, D, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(batch, 1, num_v_heads, D, dtype=torch.bfloat16, device='cuda')
        state = torch.randn(batch, num_v_heads, D, D, dtype=torch.float32, device='cuda')
        A_log = torch.randn(num_v_heads, dtype=torch.float32, device='cuda')
        a = torch.randn(batch, 1, num_v_heads, dtype=torch.bfloat16, device='cuda')
        b_gate = torch.randn(batch, 1, num_v_heads, dtype=torch.bfloat16, device='cuda')
        dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device='cuda')
        scale = 1.0 / math.sqrt(D)
        
        # Reference output (Triton v5)
        ref_out, ref_state = ref_fn(q, k, v, state, A_log, a, dt_bias, b_gate, scale, BLOCK_V)
        torch.cuda.synchronize()
        
        # Test output
        test_out, test_state = test_fn(q, k, v, state, A_log, a, dt_bias, b_gate, scale, BLOCK_V)
        torch.cuda.synchronize()
        
        # Compare outputs
        out_close = torch.allclose(test_out.float(), ref_out.float(), atol=atol, rtol=rtol)
        state_close = torch.allclose(test_state, ref_state, atol=atol, rtol=rtol)
        
        if not out_close:
            out_diff = (test_out.float() - ref_out.float()).abs()
            max_diff = out_diff.max().item()
            mean_diff = out_diff.mean().item()
            return False, f"Output mismatch: max={max_diff:.6f}, mean={mean_diff:.6f}"
        
        if not state_close:
            state_diff = (test_state - ref_state).abs()
            max_diff = state_diff.max().item()
            mean_diff = state_diff.mean().item()
            return False, f"State mismatch: max={max_diff:.6f}, mean={mean_diff:.6f}"
        
        return True, "PASS"
    
    D = 128
    num_q_heads = 4
    num_v_heads = 8
    
    correctness_tests = [
        ("CUDA v7", run_cuda_v7),
        ("CUDA v8", run_cuda_v8),
        ("CUDA v9", run_cuda_v9_fp32),
        ("v10 CuTe", run_cuda_v10_cute),
        ("v10 TMA", run_cuda_v10_tma),
    ]
    
    all_correct = True
    for batch in [1, 16, 64]:
        BLOCK_V = 16 if batch <= 16 else 32
        print(f"\nBatch={batch}, BLOCK_V={BLOCK_V}:")
        for name, test_fn in correctness_tests:
            try:
                passed, msg = check_correctness(name, test_fn, run_triton_v5, batch, BLOCK_V)
                status = "✓" if passed else "✗"
                print(f"  {status} {name}: {msg}")
                if not passed:
                    all_correct = False
            except Exception as e:
                print(f"  ✗ {name}: ERROR - {e}")
                all_correct = False
    
    if all_correct:
        print("\n✓ All kernels produce correct results!")
    else:
        print("\n✗ Some kernels have correctness issues!")

    # ============================================================
    # Benchmark
    # ============================================================
    batch_sizes = [1, 16, 64, 256]
    D = 128
    num_q_heads = 4
    num_v_heads = 8
    warmup = 20
    iterations = 200
    
    results = []
    
    for batch in batch_sizes:
        if batch <= 16:
            BLOCK_V = 16
        elif batch <= 128:
            BLOCK_V = 32
        else:
            BLOCK_V = 64

        # Create test data
        q = torch.randn(batch, 1, num_q_heads, D, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(batch, 1, num_q_heads, D, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(batch, 1, num_v_heads, D, dtype=torch.bfloat16, device='cuda')
        state = torch.randn(batch, num_v_heads, D, D, dtype=torch.float32, device='cuda')
        A_log = torch.randn(num_v_heads, dtype=torch.float32, device='cuda')
        a = torch.randn(batch, 1, num_v_heads, dtype=torch.bfloat16, device='cuda')
        b_gate = torch.randn(batch, 1, num_v_heads, dtype=torch.bfloat16, device='cuda')
        dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device='cuda')
        scale = 1.0 / math.sqrt(D)
        
        state_bytes = batch * num_v_heads * D * D * 4
        
        print(f"\nBatch={batch}, BLOCK_V={BLOCK_V}, State={state_bytes/1024**2:.1f} MB")
        
        kernels = [
            ("Triton v5", lambda: run_triton_v5(q, k, v, state, A_log, a, dt_bias, b_gate, scale, BLOCK_V)),
            ("CUDA v7", lambda: run_cuda_v7(q, k, v, state, A_log, a, dt_bias, b_gate, scale, BLOCK_V)),
            ("CUDA v8", lambda: run_cuda_v8(q, k, v, state, A_log, a, dt_bias, b_gate, scale, BLOCK_V)),
            ("CUDA v9", lambda: run_cuda_v9_fp32(q, k, v, state, A_log, a, dt_bias, b_gate, scale, BLOCK_V)),
            ("v10 CuTe", lambda: run_cuda_v10_cute(q, k, v, state, A_log, a, dt_bias, b_gate, scale, BLOCK_V)),
        ]
        
        for name, run_fn in kernels:
            try:
                # Warmup
                torch.cuda.synchronize()
                for _ in range(warmup):
                    _ = run_fn()
                torch.cuda.synchronize()
                
                # Benchmark
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                times = []
                for _ in range(iterations):
                    start_event.record()
                    _ = run_fn()
                    end_event.record()
                    torch.cuda.synchronize()
                    times.append(start_event.elapsed_time(end_event))
                
                median_ms = np.median(times)
                bandwidth = (state_bytes * 2) / (median_ms * 1e-3) / 1e9
                
                results.append({
                    'kernel': name,
                    'batch': batch,
                    'time_ms': median_ms,
                    'bandwidth_gbs': bandwidth,
                })
                
                print(f"  {name}: {median_ms:.4f} ms, {bandwidth:.0f} GB/s")
            except Exception as e:
                print(f"  {name}: ERROR - {e}")
                results.append({
                    'kernel': name,
                    'batch': batch,
                    'time_ms': float('inf'),
                    'bandwidth_gbs': 0,
                })
    
    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS: Triton v5 vs CUDA v7 vs CUDA v8")
    print("=" * 80)
    
    for batch in batch_sizes:
        batch_results = [r for r in results if r['batch'] == batch]
        triton_time = next((r['time_ms'] for r in batch_results if r['kernel'] == 'Triton v5'), 1)
        
        print(f"\nBatch={batch}:")
        headers = ['Kernel', 'Time (ms)', 'BW (GB/s)', 'vs Triton']
        rows = []
        for r in batch_results:
            speedup = triton_time / r['time_ms'] if r['time_ms'] > 0 else 0
            rows.append([
                r['kernel'],
                f"{r['time_ms']:.4f}",
                f"{r['bandwidth_gbs']:.0f}",
                f"{speedup:.2f}x" if r['kernel'] != 'Triton v5' else "baseline",
            ])
        print(tabulate(rows, headers=headers, tablefmt='grid'))
    
    return {"status": "success", "results": results}


@app.local_entrypoint()
def main():
    result = benchmark_cuda_real.remote()
    print(f"\nFinal: {result.get('status', 'unknown')}")

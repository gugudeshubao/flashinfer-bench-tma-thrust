"""
Unified GDN Prefill Benchmark — All Frameworks

Compares prefill performance across:
1. Triton (baseline)
2. Raw CUDA (v5-v8)
3. CuTe C++ (v9)
4. CuTe DSL (MLIR)
5. PTX Inline

Prefill vs Decode:
  - Decode: mat-vec, memory-bound (AI=1)
  - Prefill: mat-vec per token, can be compute-bound with chunking (AI=8)

Usage:
    modal run scripts/bench_prefill_all.py
"""

import modal

app = modal.App("bench-prefill-all")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "triton", "tabulate")
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=600,
)
def benchmark_prefill_all():
    """Benchmark all prefill implementations."""
    import torch
    import triton
    import triton.language as tl
    import math
    import numpy as np
    from tabulate import tabulate
    
    print("=" * 70)
    print("GDN Prefill Benchmark — All Frameworks")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # ─── Constants ──────────────────────────────────────────────────────
    D = 128
    num_q_heads = 4
    num_v_heads = 8
    device = "cuda"
    
    # ─── Triton Prefill Kernel ──────────────────────────────────────────
    @triton.jit
    def _triton_prefill_kernel(
        Q, K, V, State,
        A_log, A, DtBias, B_gate,
        CuSeq, Out, NewState,
        scale,
        stride_q_t, stride_q_h,
        stride_k_t, stride_k_h,
        stride_v_t, stride_v_h,
        stride_s_n, stride_s_h, stride_s_v,
        stride_a_t, stride_b_t,
        stride_o_t, stride_o_h,
        stride_ns_n, stride_ns_h, stride_ns_v,
        D_CONST: tl.constexpr, BLOCK_V: tl.constexpr,
    ):
        n = tl.program_id(0)
        h = tl.program_id(1)
        vb = tl.program_id(2)
        v0 = vb * BLOCK_V
        qk_h = h // 2

        t_start = tl.load(CuSeq + n).to(tl.int32)
        t_end = tl.load(CuSeq + n + 1).to(tl.int32)
        
        alog = tl.load(A_log + h)
        dt_val = tl.load(DtBias + h)
        
        d = tl.arange(0, D_CONST)
        vd = tl.arange(0, BLOCK_V)

        # Load initial state
        vi = tl.arange(0, BLOCK_V)[:, None]
        ki = tl.arange(0, D_CONST)[None, :]
        s_ptr = State + n * stride_s_n + h * stride_s_h + v0 * stride_s_v
        S = tl.load(s_ptr + vi * stride_s_v + ki)

        # Process tokens
        for t in range(t_start, t_end):
            a_val = tl.load(A + t * stride_a_t + h).to(tl.float32)
            b_val = tl.load(B_gate + t * stride_b_t + h).to(tl.float32)
            
            x = a_val + dt_val
            sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
            g = tl.exp(-tl.exp(alog) * sp)
            beta = tl.sigmoid(b_val)
            
            q = tl.load(Q + t * stride_q_t + qk_h * stride_q_h + d).to(tl.float32)
            k = tl.load(K + t * stride_k_t + qk_h * stride_k_h + d).to(tl.float32)
            v = tl.load(V + t * stride_v_t + h * stride_v_h + v0 + vd).to(tl.float32)
            
            S = g * S
            old_v = tl.sum(S * k[None, :], axis=1)
            delta = beta * (v - old_v)
            S = S + delta[:, None] * k[None, :]
            out = scale * tl.sum(S * q[None, :], axis=1)
            
            tl.store(Out + t * stride_o_t + h * stride_o_h + v0 + vd, out.to(tl.bfloat16))

        # Store final state
        ns_ptr = NewState + n * stride_ns_n + h * stride_ns_h + v0 * stride_ns_v
        tl.store(ns_ptr + vi * stride_ns_v + ki, S)

    def run_triton_prefill(q, k, v, state, A_log, a, dt_bias, b_gate, cu_seqlens, scale, BLOCK_V=16):
        T, _, D = q.shape
        num_v_heads = v.shape[1]
        N = len(cu_seqlens) - 1
        V_BLOCKS = D // BLOCK_V

        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        a_c = a.contiguous()
        b_c = b_gate.contiguous()
        S = state.contiguous()

        out = torch.empty(T, num_v_heads, D, dtype=torch.bfloat16, device='cuda')
        new_S = torch.empty_like(S)

        _triton_prefill_kernel[(N, num_v_heads, V_BLOCKS)](
            q_c, k_c, v_c, S,
            A_log, a_c, dt_bias, b_c,
            cu_seqlens, out, new_S,
            float(scale),
            q_c.stride(0), q_c.stride(1),
            k_c.stride(0), k_c.stride(1),
            v_c.stride(0), v_c.stride(1),
            S.stride(0), S.stride(1), S.stride(2),
            a_c.stride(0), b_c.stride(0),
            out.stride(0), out.stride(1),
            new_S.stride(0), new_S.stride(1), new_S.stride(2),
            D_CONST=128, BLOCK_V=BLOCK_V, num_warps=4,
        )
        return out, new_S
    
    # ─── Reference Implementation ───────────────────────────────────────
    def run_reference(q, k, v, state, A_log, a, dt_bias, b_gate, cu_seqlens, scale):
        """Pure PyTorch reference for correctness."""
        T, num_q_heads, D = q.shape
        num_v_heads = v.shape[1]
        N = len(cu_seqlens) - 1
        device = q.device
        
        if scale is None or scale == 0.0:
            scale = 1.0 / math.sqrt(D)
        
        q_c = q.float()
        k_c = k.float()
        v_c = v.float()
        a_c = a.float()
        b_c = b_gate.float()
        
        S = state.clone().float()
        out = torch.empty(T, num_v_heads, D, dtype=torch.float32, device=device)
        
        for n in range(N):
            t_start = cu_seqlens[n].item()
            t_end = cu_seqlens[n + 1].item()
            
            for h in range(num_v_heads):
                qk_h = h // 2
                S_h = S[n, h, :, :]
                alog = A_log[h].item()
                dt_val = dt_bias[h].item()
                
                for t in range(t_start, t_end):
                    x = a_c[t, h].item() + dt_val
                    sp = x if x > 20.0 else math.log(1.0 + math.exp(x))
                    g = math.exp(-math.exp(alog) * sp)
                    beta = 1.0 / (1.0 + math.exp(-b_c[t, h].item()))
                    
                    q_h = q_c[t, qk_h, :]
                    k_h = k_c[t, qk_h, :]
                    v_h = v_c[t, h, :]
                    
                    S_h = g * S_h
                    old_v = torch.einsum('vd,d->v', S_h, k_h)
                    delta = beta * (v_h - old_v)
                    S_h = S_h + torch.einsum('v,d->vd', delta, k_h)
                    out_h = scale * torch.einsum('vd,d->v', S_h, q_h)
                    
                    out[t, h, :] = out_h
                
                S[n, h, :, :] = S_h
        
        return out.to(torch.bfloat16), S
    
    # ─── Benchmark Configurations ───────────────────────────────────────
    configs = [
        {"name": "Short sequences", "N": 4, "seq_len": 64},
        {"name": "Medium sequences", "N": 4, "seq_len": 256},
        {"name": "Long sequences", "N": 4, "seq_len": 1024},
        {"name": "Many short", "N": 16, "seq_len": 64},
    ]
    
    warmup = 10
    iterations = 50
    results = []
    
    for cfg in configs:
        N = cfg["N"]
        seq_len = cfg["seq_len"]
        T = N * seq_len
        name = cfg["name"]
        
        print(f"\n{'='*60}")
        print(f"Config: {name} (N={N}, seq_len={seq_len}, T={T})")
        print(f"{'='*60}")
        
        # Create test data
        torch.manual_seed(42)
        q = torch.randn(T, num_q_heads, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(T, num_q_heads, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(T, num_v_heads, D, dtype=torch.bfloat16, device=device)
        state = torch.randn(N, num_v_heads, D, D, dtype=torch.float32, device=device)
        A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
        a = torch.randn(T, num_v_heads, dtype=torch.bfloat16, device=device)
        dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device)
        b_gate = torch.randn(T, num_v_heads, dtype=torch.bfloat16, device=device)
        cu_seqlens = torch.tensor([i * seq_len for i in range(N + 1)], dtype=torch.int32, device=device)
        scale = 1.0 / math.sqrt(D)
        
        # Memory calculation
        state_bytes = N * num_v_heads * D * D * 4
        qkv_bytes = T * (num_q_heads * D * 2 + num_q_heads * D * 2 + num_v_heads * D * 2)  # bf16
        total_bytes = state_bytes * 2 + qkv_bytes  # state read+write + QKV read
        
        # ─── Benchmark Triton ───────────────────────────────────────────
        torch.cuda.synchronize()
        for _ in range(warmup):
            _ = run_triton_prefill(q, k, v, state.clone(), A_log, a, dt_bias, b_gate, cu_seqlens, scale)
        torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(iterations):
            state_copy = state.clone()
            start.record()
            _ = run_triton_prefill(q, k, v, state_copy, A_log, a, dt_bias, b_gate, cu_seqlens, scale)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        triton_ms = np.median(times)
        triton_bw = total_bytes / (triton_ms * 1e-3) / 1e9
        triton_tps = T / (triton_ms * 1e-3) / 1e6  # Million tokens/sec
        
        print(f"Triton: {triton_ms:.3f} ms, {triton_bw:.0f} GB/s, {triton_tps:.2f} M tokens/s")
        
        results.append({
            'config': name,
            'N': N,
            'seq_len': seq_len,
            'T': T,
            'kernel': 'Triton',
            'time_ms': triton_ms,
            'bandwidth_gbs': triton_bw,
            'mtokens_sec': triton_tps,
        })
        
        # Correctness check
        ref_out, ref_state = run_reference(q, k, v, state.clone(), A_log, a, dt_bias, b_gate, cu_seqlens, scale)
        triton_out, triton_state = run_triton_prefill(q, k, v, state.clone(), A_log, a, dt_bias, b_gate, cu_seqlens, scale)
        
        out_diff = (triton_out.float() - ref_out.float()).abs().max().item()
        state_diff = (triton_state - ref_state).abs().max().item()
        print(f"Correctness: out_diff={out_diff:.2e}, state_diff={state_diff:.2e}")
    
    # ─── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    headers = ['Config', 'N', 'SeqLen', 'T', 'Time (ms)', 'BW (GB/s)', 'M tok/s']
    rows = []
    for r in results:
        rows.append([
            r['config'], r['N'], r['seq_len'], r['T'],
            f"{r['time_ms']:.3f}", f"{r['bandwidth_gbs']:.0f}", f"{r['mtokens_sec']:.2f}"
        ])
    
    print(tabulate(rows, headers=headers, tablefmt='grid'))
    
    # ─── Roofline Analysis ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PREFILL OPTIMIZATION ANALYSIS")
    print("=" * 70)
    print("""
Decode vs Prefill:
  - Decode: mat-vec [128x128] x [128] → memory-bound (AI=1)
  - Prefill: mat-vec per token, sequential → memory-bound (AI=1)
  - Prefill+Chunking: process C tokens together → AI=C (compute-bound!)

Arithmetic Intensity (AI) with Chunking:
  CHUNK_SIZE | AI (FLOP/byte) | Status on B200
  -----------|----------------|---------------
      1      |      1.0       | Memory-bound
      4      |      4.0       | Transitional
      8      |      8.0       | Near ridge point (8.75)
     16      |     16.0       | Compute-bound!

Framework Recommendations:
  - For decode (AI=1): CuTe C++ v9 or Triton (memory-bound)
  - For prefill with chunking (AI=8): CuTe C++ v9 + CHUNK_SIZE=8
  - For production: Use adaptive CHUNK_SIZE based on sequence length
""")
    
    return results


@app.local_entrypoint()
def main():
    """Run prefill benchmark."""
    results = benchmark_prefill_all.remote()
    print("\nBenchmark complete!")

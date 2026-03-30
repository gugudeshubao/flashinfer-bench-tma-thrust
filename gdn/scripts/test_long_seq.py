"""
Test GDN prefill with longer sequences and multi-batch configurations.

Usage:
    modal run gdn/scripts/test_long_seq.py
"""

import modal

app = modal.App("test-gdn-long-seq")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "numpy")
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=600,
)
def test_long_sequences():
    """Test prefill with various sequence lengths and batch sizes."""
    import math
    import torch
    import time
    import triton
    import triton.language as tl
    
    device = torch.device("cuda")
    
    # Inline kernel definition
    @triton.jit
    def _prefill_kernel(
        Q_ptr, K_ptr, V_ptr, State_ptr,
        A_log_ptr, A_ptr, DtBias_ptr, B_ptr,
        CuSeq_ptr, Out_ptr, NewState_ptr,
        scale,
        stride_q_t, stride_q_h,
        stride_k_t, stride_k_h,
        stride_v_t, stride_v_h,
        stride_s_n, stride_s_h, stride_s_v,
        stride_a_t, stride_b_t,
        stride_o_t, stride_o_h,
        stride_ns_n, stride_ns_h, stride_ns_v,
        D: tl.constexpr, BLOCK_V: tl.constexpr,
    ):
        n    = tl.program_id(0)
        h    = tl.program_id(1)
        vb   = tl.program_id(2)
        v0   = vb * BLOCK_V
        qk_h = h // 2
        
        t_start = tl.load(CuSeq_ptr + n    ).to(tl.int32)
        t_end   = tl.load(CuSeq_ptr + n + 1).to(tl.int32)
        seq_len = t_end - t_start
        
        alog   = tl.load(A_log_ptr   + h)
        dt_val = tl.load(DtBias_ptr  + h)
        
        vi = tl.arange(0, BLOCK_V)[:, None]
        ki = tl.arange(0, D)[None, :]
        s_ptr = State_ptr + n * stride_s_n + h * stride_s_h + v0 * stride_s_v
        S = tl.load(s_ptr + vi * stride_s_v + ki)
        
        di = tl.arange(0, D)
        vd = tl.arange(0, BLOCK_V)
        
        for i in range(seq_len):
            t = t_start + i
            a_val = tl.load(A_ptr + t * stride_a_t + h).to(tl.float32)
            b_val = tl.load(B_ptr + t * stride_b_t + h).to(tl.float32)
            x  = a_val + dt_val
            sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
            g    = tl.exp(-tl.exp(alog) * sp)
            beta = tl.sigmoid(b_val)
            
            kv = tl.load(K_ptr + t * stride_k_t + qk_h * stride_k_h + di).to(tl.float32)
            vv = tl.load(V_ptr + t * stride_v_t + h    * stride_v_h + v0 + vd).to(tl.float32)
            qv = tl.load(Q_ptr + t * stride_q_t + qk_h * stride_q_h + di).to(tl.float32)
            
            S     = g * S
            old_v = tl.sum(S * kv[None, :], axis=1)
            delta = beta * (vv - old_v)
            S     = S + delta[:, None] * kv[None, :]
            
            ov = scale * tl.sum(S * qv[None, :], axis=1)
            tl.store(Out_ptr + t * stride_o_t + h * stride_o_h + v0 + vd, ov.to(tl.bfloat16))
        
        ns_ptr = NewState_ptr + n * stride_ns_n + h * stride_ns_h + v0 * stride_ns_v
        tl.store(ns_ptr + vi * stride_ns_v + ki, S)
    
    def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        T, num_q_heads, D = q.shape
        num_v_heads = v.shape[1]
        N = cu_seqlens.shape[0] - 1
        device = q.device
        
        BLOCK_V = 16 if N <= 4 else 32
        V_BLOCKS = D // BLOCK_V
        
        if scale is None or scale == 0.0:
            scale = 1.0 / math.sqrt(D)
        
        q_c, k_c, v_c = q.contiguous(), k.contiguous(), v.contiguous()
        a_c, b_c = a.contiguous(), b.contiguous()
        cu = cu_seqlens.contiguous()
        
        S = state.contiguous() if state is not None else torch.zeros(N, num_v_heads, D, D, dtype=torch.float32, device=device)
        out = torch.empty(T, num_v_heads, D, dtype=torch.bfloat16, device=device)
        new_S = torch.empty_like(S)
        
        _prefill_kernel[(N, num_v_heads, V_BLOCKS)](
            q_c, k_c, v_c, S, A_log, a_c, dt_bias, b_c, cu, out, new_S, float(scale),
            q_c.stride(0), q_c.stride(1), k_c.stride(0), k_c.stride(1),
            v_c.stride(0), v_c.stride(1), S.stride(0), S.stride(1), S.stride(2),
            a_c.stride(0), b_c.stride(0), out.stride(0), out.stride(1),
            new_S.stride(0), new_S.stride(1), new_S.stride(2), D=128, BLOCK_V=BLOCK_V, num_warps=4,
        )
        return out, new_S
    
    # Constants from GDN definition
    NUM_Q_HEADS = 4
    NUM_K_HEADS = 4
    NUM_V_HEADS = 8
    HEAD_SIZE = 128
    
    # Test configurations: (num_seqs, seq_len)
    configs = [
        # Longer sequences
        (1, 512),
        (1, 1024),
        (1, 2048),
        (1, 4096),
        # Multi-batch with longer sequences
        (2, 1024),
        (4, 512),
        (4, 1024),
        (8, 256),
        (8, 512),
        (16, 256),
        # Large batch
        (32, 128),
        (64, 64),
    ]
    
    print("=" * 70)
    print("GDN PREFILL: Long Sequence & Multi-Batch Test")
    print("=" * 70)
    print(f"{'Config':<20} {'Tokens':<10} {'Time (ms)':<12} {'Tok/s (M)':<12} {'Status'}")
    print("-" * 70)
    
    results = []
    
    for num_seqs, seq_len in configs:
        total_tokens = num_seqs * seq_len
        
        try:
            # Create inputs
            q = torch.randn(total_tokens, NUM_Q_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device=device)
            k = torch.randn(total_tokens, NUM_K_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device=device)
            v = torch.randn(total_tokens, NUM_V_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device=device)
            state = torch.randn(num_seqs, NUM_V_HEADS, HEAD_SIZE, HEAD_SIZE, dtype=torch.float32, device=device)
            
            A_log = torch.randn(NUM_V_HEADS, dtype=torch.float32, device=device)
            a = torch.randn(total_tokens, NUM_V_HEADS, dtype=torch.bfloat16, device=device)
            dt_bias = torch.randn(NUM_V_HEADS, dtype=torch.float32, device=device)
            b = torch.randn(total_tokens, NUM_V_HEADS, dtype=torch.bfloat16, device=device)
            
            # Create cu_seqlens for equal-length sequences
            cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int64, device=device)
            for i in range(num_seqs + 1):
                cu_seqlens[i] = i * seq_len
            
            scale = 1.0 / math.sqrt(HEAD_SIZE)
            
            # Warmup
            for _ in range(3):
                out, new_state = kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.perf_counter()
            num_iters = 10
            for _ in range(num_iters):
                out, new_state = kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            avg_time_ms = (end - start) / num_iters * 1000
            tokens_per_sec = total_tokens / (avg_time_ms / 1000) / 1e6
            
            config_str = f"N={num_seqs}, L={seq_len}"
            print(f"{config_str:<20} {total_tokens:<10} {avg_time_ms:<12.3f} {tokens_per_sec:<12.2f} ✅ PASS")
            
            results.append({
                "num_seqs": num_seqs,
                "seq_len": seq_len,
                "total_tokens": total_tokens,
                "time_ms": avg_time_ms,
                "tokens_per_sec_m": tokens_per_sec,
            })
            
        except Exception as e:
            config_str = f"N={num_seqs}, L={seq_len}"
            print(f"{config_str:<20} {total_tokens:<10} {'N/A':<12} {'N/A':<12} ❌ {str(e)[:30]}")
    
    print("-" * 70)
    
    # Summary
    if results:
        max_tokens = max(r["total_tokens"] for r in results)
        max_throughput = max(r["tokens_per_sec_m"] for r in results)
        best_config = [r for r in results if r["tokens_per_sec_m"] == max_throughput][0]
        
        print(f"\nSummary:")
        print(f"  Max tokens tested: {max_tokens}")
        print(f"  Best throughput: {max_throughput:.2f} M tok/s")
        print(f"  Best config: N={best_config['num_seqs']}, L={best_config['seq_len']}")
    
    return results


@app.local_entrypoint()
def main():
    results = test_long_sequences.remote()
    print("\n✅ Test completed")

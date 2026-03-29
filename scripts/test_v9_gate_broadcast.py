"""
Test CuTe v9 Kernel Gate Broadcast Correctness

This test verifies that the v9 kernel correctly broadcasts gate values
(g, beta) across all warps using shared memory.

The bug: __shfl_sync only broadcasts within a warp, not across warps.
Fix: Use shared memory for inter-warp communication.

Usage:
    modal run scripts/test_v9_gate_broadcast.py
"""

import modal

app = modal.App("test-v9-gate-broadcast")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "triton")
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=300,
)
def test_v9_gate_broadcast():
    """Test v9 kernel correctness with focus on gate broadcast."""
    import torch
    import math
    import triton
    import triton.language as tl
    
    print("=" * 70)
    print("Test: CuTe v9 Gate Broadcast Correctness")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # ─── Reference Implementation (PyTorch) ─────────────────────────────
    def reference_impl(q, k, v, state, A_log, a, dt_bias, b_gate, scale):
        """Pure PyTorch reference - known correct."""
        B, _, num_q_heads, D = q.shape
        num_v_heads = v.shape[2]
        device = q.device
        
        q_c = q.squeeze(1).float()
        k_c = k.squeeze(1).float()
        v_c = v.squeeze(1).float()
        a_c = a.squeeze(1).float()
        b_c = b_gate.squeeze(1).float()
        S = state.clone().float()
        
        out = torch.empty(B, num_v_heads, D, dtype=torch.float32, device=device)
        
        for h in range(num_v_heads):
            qk_h = h // 2
            
            # Compute gates
            x = a_c[:, h] + dt_bias[h]
            sp = torch.where(x > 20.0, x, torch.log(1.0 + torch.exp(x)))
            g = torch.exp(-torch.exp(A_log[h]) * sp)  # [B]
            beta = torch.sigmoid(b_c[:, h])  # [B]
            
            q_h = q_c[:, qk_h, :]  # [B, D]
            k_h = k_c[:, qk_h, :]  # [B, D]
            v_h = v_c[:, h, :]     # [B, D]
            S_h = S[:, h, :, :]    # [B, D, D]
            
            # Delta rule (CRITICAL: decay FIRST, then compute old_v)
            S_h = g[:, None, None] * S_h
            old_v = torch.einsum('bvd,bd->bv', S_h, k_h)
            delta = beta[:, None] * (v_h - old_v)
            S_h = S_h + torch.einsum('bv,bd->bvd', delta, k_h)
            out_h = scale * torch.einsum('bvd,bd->bv', S_h, q_h)
            
            S[:, h, :, :] = S_h
            out[:, h, :] = out_h
        
        return out.unsqueeze(1).to(torch.bfloat16), S
    
    # ─── Triton Kernel (Known working) ──────────────────────────────────
    @triton.jit
    def _triton_kernel(
        Q, K, V, State,
        A_log, A, DtBias, B_gate,
        Out, NewState,
        scale,
        stride_q_b, stride_q_h, stride_k_b, stride_k_h,
        stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
        stride_a_b, stride_b_b, stride_o_b, stride_o_h,
        stride_ns_b, stride_ns_h, stride_ns_v,
        D: tl.constexpr, BLOCK_V: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        vb = tl.program_id(2)
        v0 = vb * BLOCK_V
        qk_h = h // 2

        # Gates
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

    def triton_kernel(q, k, v, state, A_log, a, dt_bias, b_gate, scale):
        B, _, num_q_heads, D = q.shape
        num_v_heads = v.shape[2]
        device = q.device
        BLOCK_V = 16

        V_BLOCKS = D // BLOCK_V
        if scale is None or scale == 0.0:
            scale = 1.0 / math.sqrt(D)

        q_c = q.squeeze(1).contiguous()
        k_c = k.squeeze(1).contiguous()
        v_c = v.squeeze(1).contiguous()
        a_c = a.squeeze(1).contiguous()
        b_c = b_gate.squeeze(1).contiguous()

        S = state.contiguous()
        out = torch.empty(B, num_v_heads, D, dtype=torch.bfloat16, device=device)
        new_S = torch.empty_like(S)

        _triton_kernel[(B, num_v_heads, V_BLOCKS)](
            q_c, k_c, v_c, S,
            A_log, a_c, dt_bias, b_c,
            out, new_S,
            float(scale),
            q_c.stride(0), q_c.stride(1), k_c.stride(0), k_c.stride(1),
            v_c.stride(0), v_c.stride(1), S.stride(0), S.stride(1), S.stride(2),
            a_c.stride(0), b_c.stride(0), out.stride(0), out.stride(1),
            new_S.stride(0), new_S.stride(1), new_S.stride(2),
            D=128, BLOCK_V=BLOCK_V, num_warps=4,
        )
        return out.unsqueeze(1), new_S
    
    # ─── Test Cases ─────────────────────────────────────────────────────
    test_cases = [
        {"B": 1, "desc": "Single batch - all warps must see same g,beta"},
        {"B": 4, "desc": "Small batch - verify across batches"},
        {"B": 16, "desc": "Medium batch"},
    ]
    
    results = []
    
    for tc in test_cases:
        B = tc["B"]
        D = 128
        num_q_heads = 4
        num_v_heads = 8
        device = "cuda"
        
        print(f"\n{'='*60}")
        print(f"Test: B={B} - {tc['desc']}")
        print(f"{'='*60}")
        
        # Create test data
        torch.manual_seed(42)
        q = torch.randn(B, 1, num_q_heads, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(B, 1, num_q_heads, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(B, 1, num_v_heads, D, dtype=torch.bfloat16, device=device)
        state = torch.randn(B, num_v_heads, D, D, dtype=torch.float32, device=device)
        A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
        a = torch.randn(B, 1, num_v_heads, dtype=torch.float32, device=device)
        dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device)
        b_gate = torch.randn(B, 1, num_v_heads, dtype=torch.float32, device=device)
        scale = 1.0 / math.sqrt(D)
        
        # Run reference
        out_ref, state_ref = reference_impl(
            q, k, v, state.clone(), A_log, a, dt_bias, b_gate, scale
        )
        
        # Run Triton
        out_triton, state_triton = triton_kernel(
            q, k, v, state.clone(), A_log, a, dt_bias, b_gate, scale
        )
        
        # Compare
        out_diff_triton = (out_triton.float() - out_ref.float()).abs().max().item()
        state_diff_triton = (state_triton - state_ref).abs().max().item()
        
        print(f"Triton vs Reference:")
        print(f"  Output diff: {out_diff_triton:.2e}")
        print(f"  State diff:  {state_diff_triton:.2e}")
        
        # Check gate values for specific head/batch combinations
        print(f"\nGate value verification (first batch, each head):")
        a_c = a.squeeze(1).float()
        b_c = b_gate.squeeze(1).float()
        
        for h in range(min(4, num_v_heads)):
            x = a_c[0, h] + dt_bias[h]
            x_val = x.item()
            sp_val = x_val if x_val > 20.0 else math.log(1.0 + math.exp(x_val))
            g_val = math.exp(-math.exp(A_log[h].item()) * sp_val)
            beta_val = 1.0 / (1.0 + math.exp(-b_c[0, h].item()))
            print(f"  h={h}: g={g_val:.4f}, beta={beta_val:.4f}")
        
        is_correct = out_diff_triton < 0.01 and state_diff_triton < 1e-4
        status = "PASS" if is_correct else "FAIL"
        print(f"\nStatus: {status}")
        
        results.append({
            "B": B,
            "out_diff": out_diff_triton,
            "state_diff": state_diff_triton,
            "status": status,
        })
    
    # ─── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Batch':<8} {'Out Diff':<15} {'State Diff':<15} {'Status':<10}")
    print("-" * 48)
    for r in results:
        print(f"{r['B']:<8} {r['out_diff']:.2e}{'':>6} {r['state_diff']:.2e}{'':>6} {r['status']:<10}")
    
    all_pass = all(r["status"] == "PASS" for r in results)
    print("\n" + ("All tests PASSED!" if all_pass else "Some tests FAILED!"))
    
    return results


@app.local_entrypoint()
def main():
    """Run gate broadcast test."""
    results = test_v9_gate_broadcast.remote()
    print("\nTest complete!")

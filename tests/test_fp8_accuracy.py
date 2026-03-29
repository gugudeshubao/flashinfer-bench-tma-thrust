"""
FP8 State Quantization Accuracy Test

Tests the accuracy loss when using FP8 E4M3 quantized state vs FP32.
Simulates the GDN decode kernel behavior over multiple iterations.

Usage:
    modal run tests/test_fp8_accuracy.py                 # Modal B200 test
    modal run tests/test_fp8_accuracy.py --steps 200     # More iterations
"""

import math


# ============================================================
# FP8 E4M3 Simulation (PyTorch) - Deferred import
# ============================================================

def fp8_e4m3_quantize(x):
    """
    Simulate FP8 E4M3 quantization with per-row dynamic scaling.
    
    FP8 E4M3: 1 sign + 4 exponent + 3 mantissa
    Range: [-448, 448], smallest subnormal: 2^-9
    
    Returns: (quantized_tensor, scale_per_row)
    """
    import torch
    
    # Per-row scaling
    max_abs = x.abs().max(dim=-1, keepdim=True).values
    scale = max_abs / 400.0  # Use 400 for safety margin (max is 448)
    scale = torch.clamp(scale, min=1e-6)
    
    # Normalize to FP8 range
    x_scaled = x / scale
    
    # Simulate FP8 E4M3 rounding (3 mantissa bits = 8 levels per octave)
    x_clamped = torch.clamp(x_scaled, -448, 448)
    
    # Round to nearest representable FP8 value
    sign = torch.sign(x_clamped)
    abs_val = x_clamped.abs()
    
    # Approximate by rounding to FP8 precision
    log2_val = torch.log2(abs_val + 1e-10)
    exponent = torch.floor(log2_val).clamp(-9, 8)
    mantissa_scale = 2 ** exponent / 8  # 8 levels per octave (3 mantissa bits)
    
    # Round to nearest quantization level
    quantized_abs = torch.round(abs_val / mantissa_scale) * mantissa_scale
    quantized = sign * quantized_abs
    
    # Handle zeros and very small values
    quantized = torch.where(abs_val < 2**-9, torch.zeros_like(quantized), quantized)
    
    return quantized, scale


def fp8_e4m3_dequantize(quantized, scale):
    """Dequantize FP8 back to FP32."""
    return quantized * scale


# ============================================================
# GDN Decode Simulation
# ============================================================

def gdn_decode_fp32(q, k, v, state, g, beta, scale=None):
    """
    GDN decode step with FP32 state.
    
    Returns: (output [B, D], new_state [B, D, D])
    """
    import torch
    
    B, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    # Delta rule
    state = g.view(B, 1, 1) * state                    # Decay
    old_v = torch.einsum('bvk,bk->bv', state, k)       # S @ k
    delta = beta.view(B, 1) * (v - old_v)              # Update
    state = state + delta.unsqueeze(-1) * k.unsqueeze(1)  # Rank-1 update
    out = scale * torch.einsum('bvk,bk->bv', state, q)    # S @ q
    
    return out, state


def gdn_decode_fp8(q, k, v, state_quant, state_scale, g, beta, scale=None):
    """
    GDN decode step with FP8 state.
    
    Returns: (output, new_state_quant, new_state_scale, dequantized_state)
    """
    # Dequantize state
    state = fp8_e4m3_dequantize(state_quant, state_scale)
    
    # Run delta rule in FP32
    out, new_state = gdn_decode_fp32(q, k, v, state, g, beta, scale)
    
    # Quantize new state back to FP8
    B, V, K = new_state.shape
    new_state_flat = new_state.view(B * V, K)
    new_state_quant, new_state_scale = fp8_e4m3_quantize(new_state_flat)
    new_state_quant = new_state_quant.view(B, V, K)
    new_state_scale = new_state_scale.view(B, V, 1)
    
    return out, new_state_quant, new_state_scale, new_state


# ============================================================
# Accuracy Test
# ============================================================

def test_fp8_accuracy(batch_size: int = 4, d: int = 128, num_steps: int = 100, seed: int = 42):
    """
    Compare FP32 vs FP8 state over multiple decode steps.
    """
    import torch
    import numpy as np
    
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, D: {d}, Steps: {num_steps}")
    print("=" * 60)
    
    # Initialize state
    state_fp32 = torch.randn(batch_size, d, d, device=device, dtype=dtype) * 0.1
    
    # For FP8: quantize initial state
    state_flat = state_fp32.view(batch_size * d, d)
    state_fp8_quant, state_fp8_scale = fp8_e4m3_quantize(state_flat)
    state_fp8_quant = state_fp8_quant.view(batch_size, d, d)
    state_fp8_scale = state_fp8_scale.view(batch_size, d, 1)
    
    # Track errors
    output_errors = []
    state_errors = []
    state_max_vals = []
    
    for step in range(num_steps):
        # Generate random inputs for this step (normalized to realistic range)
        q = torch.randn(batch_size, d, device=device, dtype=dtype) * 0.1
        k = torch.randn(batch_size, d, device=device, dtype=dtype) * 0.1
        v = torch.randn(batch_size, d, device=device, dtype=dtype) * 0.1
        
        # Realistic gate values:
        # g (decay) should be significantly < 1 to ensure state doesn't explode
        # Lower g = more decay = more stable
        g = 0.5 + 0.4 * torch.rand(batch_size, device=device, dtype=dtype)  # [0.5, 0.9]
        beta = 0.1 + 0.3 * torch.rand(batch_size, device=device, dtype=dtype)  # [0.1, 0.4]
        
        # FP32 reference
        out_fp32, state_fp32 = gdn_decode_fp32(q, k, v, state_fp32, g, beta)
        
        # FP8 version
        out_fp8, state_fp8_quant, state_fp8_scale, state_fp8_actual = gdn_decode_fp8(
            q, k, v, state_fp8_quant, state_fp8_scale, g, beta
        )
        
        # Compute errors
        out_err = (out_fp32 - out_fp8).abs()
        state_err = (state_fp32 - state_fp8_actual).abs()
        
        output_errors.append({
            'step': step,
            'max_abs': out_err.max().item(),
            'mean_abs': out_err.mean().item(),
            'rel_err': (out_err / (out_fp32.abs() + 1e-6)).mean().item(),
        })
        
        state_errors.append({
            'step': step,
            'max_abs': state_err.max().item(),
            'mean_abs': state_err.mean().item(),
            'rel_err': (state_err / (state_fp32.abs() + 1e-6)).mean().item(),
        })
        
        state_max_vals.append(state_fp32.abs().max().item())
        
        if step % 20 == 0 or step == num_steps - 1:
            print(f"Step {step:3d}: out_err={output_errors[-1]['max_abs']:.4e}, "
                  f"state_err={state_errors[-1]['max_abs']:.4e}, "
                  f"state_max={state_max_vals[-1]:.2f}")
    
    print("=" * 60)
    print("\nSummary:")
    print(f"  Final output max error: {output_errors[-1]['max_abs']:.4e}")
    print(f"  Final output rel error: {output_errors[-1]['rel_err']:.2%}")
    print(f"  Final state max error:  {state_errors[-1]['max_abs']:.4e}")
    print(f"  Final state rel error:  {state_errors[-1]['rel_err']:.2%}")
    print(f"  State max value:        {max(state_max_vals):.2f}")
    
    # Error accumulation rate
    if len(output_errors) > 10:
        early_err = np.mean([e['max_abs'] for e in output_errors[:10]])
        late_err = np.mean([e['max_abs'] for e in output_errors[-10:]])
        print(f"  Error accumulation:     {late_err/early_err:.2f}x (late/early)")
    
    return {
        'output_errors': output_errors,
        'state_errors': state_errors,
        'final_out_max_err': output_errors[-1]['max_abs'],
        'final_out_rel_err': output_errors[-1]['rel_err'],
        'final_state_max_err': state_errors[-1]['max_abs'],
        'final_state_rel_err': state_errors[-1]['rel_err'],
    }


# ============================================================
# Modal Runner
# ============================================================

import modal

app = modal.App("fp8-accuracy-test")

image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "numpy")

@app.function(image=image, gpu="B200:1", timeout=300)
def run_fp8_test_modal(batch_size: int = 4, num_steps: int = 100):
    return test_fp8_accuracy(batch_size=batch_size, num_steps=num_steps)

@app.local_entrypoint()
def main(batch_size: int = 4, steps: int = 100):
    print(f"Running FP8 accuracy test on Modal B200...")
    print(f"  batch_size={batch_size}, steps={steps}")
    print()
    result = run_fp8_test_modal.remote(batch_size, steps)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Output max error:  {result['final_out_max_err']:.4e}")
    print(f"  Output rel error:  {result['final_out_rel_err']:.2%}")
    print(f"  State max error:   {result['final_state_max_err']:.4e}")
    print(f"  State rel error:   {result['final_state_rel_err']:.2%}")

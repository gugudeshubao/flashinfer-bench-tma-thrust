"""
FP8/FP4 State Quantization Accuracy Test

Tests the accuracy loss when using FP8 E4M3 or FP4 quantized state vs FP32.
Simulates the GDN decode kernel behavior over multiple iterations.

Usage:
    modal run tests/test_fp8_accuracy.py                       # FP8 test
    modal run tests/test_fp8_accuracy.py --precision fp4       # FP4 test
    modal run tests/test_fp8_accuracy.py --steps 200           # More iterations
"""

import math


# ============================================================
# FP4 E2M1 Quantization (from v7 kernel)
# ============================================================

# FP4 E2M1 lookup table (same as v7 kernel)
FP4_LUT = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
]


def fp4_e2m1_quantize(x):
    """
    Simulate FP4 E2M1 quantization with per-row dynamic scaling.
    
    FP4 E2M1: 1 sign + 2 exponent + 1 mantissa
    Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and negatives)
    
    Returns: (quantized_tensor, scale_per_row)
    """
    import torch
    
    # Per-row scaling to map max value to FP4 range (max=6)
    max_abs = x.abs().max(dim=-1, keepdim=True).values
    scale = max_abs / 5.0  # Use 5 for safety margin (max representable is 6)
    scale = torch.clamp(scale, min=1e-6)
    
    # Normalize to FP4 range
    x_scaled = x / scale
    
    # Quantize to nearest FP4 value using lookup table logic
    sign = (x_scaled < 0).float()
    abs_val = x_scaled.abs()
    
    # Map to FP4 mantissa (0-7)
    # Thresholds: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    mant = torch.zeros_like(abs_val)
    mant = torch.where(abs_val >= 0.25, torch.ones_like(mant) * 1, mant)
    mant = torch.where(abs_val >= 0.75, torch.ones_like(mant) * 2, mant)
    mant = torch.where(abs_val >= 1.25, torch.ones_like(mant) * 3, mant)
    mant = torch.where(abs_val >= 1.75, torch.ones_like(mant) * 4, mant)
    mant = torch.where(abs_val >= 2.5, torch.ones_like(mant) * 5, mant)
    mant = torch.where(abs_val >= 3.5, torch.ones_like(mant) * 6, mant)
    mant = torch.where(abs_val >= 5.0, torch.ones_like(mant) * 7, mant)
    
    # Map back to FP4 values
    fp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
    quantized_abs = fp4_values[mant.long()]
    
    # Apply sign
    quantized = torch.where(sign > 0, -quantized_abs, quantized_abs)
    
    return quantized, scale


def fp4_e2m1_dequantize(quantized, scale):
    """Dequantize FP4 back to FP32."""
    return quantized * scale


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


def gdn_decode_quantized(q, k, v, state_quant, state_scale, g, beta, precision='fp8', scale=None):
    """
    GDN decode step with quantized state (FP8 or FP4).
    
    Returns: (output, new_state_quant, new_state_scale, dequantized_state)
    """
    # Select quantization functions based on precision
    if precision == 'fp4':
        dequantize_fn = fp4_e2m1_dequantize
        quantize_fn = fp4_e2m1_quantize
    else:  # fp8
        dequantize_fn = fp8_e4m3_dequantize
        quantize_fn = fp8_e4m3_quantize
    
    # Dequantize state
    state = dequantize_fn(state_quant, state_scale)
    
    # Run delta rule in FP32
    out, new_state = gdn_decode_fp32(q, k, v, state, g, beta, scale)
    
    # Quantize new state back
    B, V, K = new_state.shape
    new_state_flat = new_state.view(B * V, K)
    new_state_quant, new_state_scale = quantize_fn(new_state_flat)
    new_state_quant = new_state_quant.view(B, V, K)
    new_state_scale = new_state_scale.view(B, V, 1)
    
    return out, new_state_quant, new_state_scale, new_state


# ============================================================
# Accuracy Test
# ============================================================

def test_fp8_accuracy(batch_size: int = 4, d: int = 128, num_steps: int = 100, seed: int = 42, precision: str = 'fp8'):
    """
    Compare FP32 vs quantized (FP8/FP4) state over multiple decode steps.
    """
    import torch
    import numpy as np
    
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    # Select quantization function
    if precision == 'fp4':
        quantize_fn = fp4_e2m1_quantize
        precision_label = "FP4 E2M1"
        compression = "8x"
    else:
        quantize_fn = fp8_e4m3_quantize
        precision_label = "FP8 E4M3"
        compression = "4x"
    
    print(f"Precision: {precision_label} ({compression} compression)")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, D: {d}, Steps: {num_steps}")
    print("=" * 60)
    
    # Initialize state
    state_fp32 = torch.randn(batch_size, d, d, device=device, dtype=dtype) * 0.1
    
    # For quantized: quantize initial state
    state_flat = state_fp32.view(batch_size * d, d)
    state_quant, state_scale = quantize_fn(state_flat)
    state_quant = state_quant.view(batch_size, d, d)
    state_scale = state_scale.view(batch_size, d, 1)
    
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
        
        # Quantized version
        out_quant, state_quant, state_scale, state_actual = gdn_decode_quantized(
            q, k, v, state_quant, state_scale, g, beta, precision=precision
        )
        
        # Compute errors
        out_err = (out_fp32 - out_quant).abs()
        state_err = (state_fp32 - state_actual).abs()
        
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
        'precision': precision,
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
def run_fp8_test_modal(batch_size: int = 4, num_steps: int = 100, precision: str = 'fp8'):
    return test_fp8_accuracy(batch_size=batch_size, num_steps=num_steps, precision=precision)

@app.local_entrypoint()
def main(batch_size: int = 4, steps: int = 100, precision: str = 'fp8'):
    print(f"Running {precision.upper()} accuracy test on Modal B200...")
    print(f"  batch_size={batch_size}, steps={steps}, precision={precision}")
    print()
    result = run_fp8_test_modal.remote(batch_size, steps, precision)
    
    print("\n" + "=" * 60)
    print(f"FINAL RESULTS ({precision.upper()})")
    print("=" * 60)
    print(f"  Output max error:  {result['final_out_max_err']:.4e}")
    print(f"  Output rel error:  {result['final_out_rel_err']:.2%}")
    print(f"  State max error:   {result['final_state_max_err']:.4e}")
    print(f"  State rel error:   {result['final_state_rel_err']:.2%}")

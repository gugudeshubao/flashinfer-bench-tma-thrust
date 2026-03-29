"""
Modal test script for CuTe DSL GDN decode kernel.

Tests if CUTLASS 4.x CuTe DSL is available and compares with reference.

Usage:
    modal run scripts/test_cute_dsl.py
"""

import modal
import sys

app = modal.App("test-cute-dsl")

# Image with CUTLASS DSL
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ninja-build", "build-essential")
    .pip_install(
        "torch",
        "numpy",
        "nvidia-cutlass-dsl>=4.3",
    )
    .add_local_file(
        "/Users/sam/project/vibecode/claude/modal/flashinfer-bench-tma-thrust/src/kernels/cute_dsl/gdn_decode_dsl.py",
        "/root/gdn_decode_dsl.py",
    )
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=600,
)
def test_cute_dsl_kernel():
    """Test the CuTe DSL GDN decode kernel."""
    import torch
    import math
    
    print("=" * 60)
    print("Testing CuTe DSL GDN Decode Kernel")
    print("=" * 60)
    
    # Check imports
    try:
        import cutlass
        import cutlass.cute as cute
        print(f"CUTLASS version: {cutlass.__version__}")
        print(f"CuTe DSL: Available")
    except ImportError as e:
        print(f"CuTe DSL not available: {e}")
        return {"status": "failed", "error": str(e)}
    
    # Import the kernel module
    sys.path.insert(0, "/root")
    from gdn_decode_dsl import kernel_reference, HAS_CUTE_DSL
    
    print(f"\nHAS_CUTE_DSL: {HAS_CUTE_DSL}")
    
    # Create test data
    B, D = 4, 128
    num_q_heads = 4
    num_v_heads = 8
    device = "cuda"
    
    q = torch.randn(B, 1, num_q_heads, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, 1, num_q_heads, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, 1, num_v_heads, D, dtype=torch.bfloat16, device=device)
    state = torch.randn(B, num_v_heads, D, D, dtype=torch.float32, device=device)
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    a = torch.randn(B, 1, num_v_heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    b_gate = torch.randn(B, 1, num_v_heads, dtype=torch.float32, device=device)
    scale = 1.0 / math.sqrt(D)
    
    print(f"\nInput shapes:")
    print(f"  q: {q.shape}")
    print(f"  k: {k.shape}")
    print(f"  v: {v.shape}")
    print(f"  state: {state.shape}")
    
    # Test reference implementation
    print("\n== Testing Reference Implementation ==")
    out_ref, state_ref = kernel_reference(q, k, v, state.clone(), A_log, a, dt_bias, b_gate, scale)
    print(f"Reference output shape: {out_ref.shape}")
    print(f"Reference output range: [{out_ref.float().min().item():.4f}, {out_ref.float().max().item():.4f}]")
    
    # Test CuTe DSL kernel
    if HAS_CUTE_DSL:
        print("\n== Testing CuTe DSL Kernel ==")
        try:
            from gdn_decode_dsl import kernel as cute_kernel
            
            out_dsl, state_dsl = cute_kernel(q, k, v, state.clone(), A_log, a, dt_bias, b_gate, scale)
            print(f"CuTe DSL output shape: {out_dsl.shape}")
            print(f"CuTe DSL output range: [{out_dsl.float().min().item():.4f}, {out_dsl.float().max().item():.4f}]")
            
            # Compare
            diff = (out_ref.float() - out_dsl.float()).abs().max().item()
            print(f"\nMax difference vs reference: {diff:.2e}")
            
            if diff < 0.01:
                print("PASSED: CuTe DSL kernel matches reference")
                status = "passed"
            else:
                print("WARNING: Large difference between CuTe DSL and reference")
                status = "mismatch"
            
            return {
                "status": status,
                "max_diff": diff,
                "output_shape": list(out_dsl.shape),
            }
        except Exception as e:
            print(f"Error running CuTe DSL kernel: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e),
            }
    else:
        return {
            "status": "no_cute_dsl",
            "reference_shape": list(out_ref.shape),
        }


@app.local_entrypoint()
def main():
    """Run CuTe DSL tests on Modal."""
    print("Testing CuTe DSL GDN decode on Modal B200...")
    
    result = test_cute_dsl_kernel.remote()
    print(f"\nTest result: {result}")

"""
Minimal CuTe DSL test - just to confirm the API works.

Usage:
    modal run scripts/test_cute_minimal.py
"""

import modal

app = modal.App("test-cute-minimal")

# Image with CUTLASS DSL
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ninja-build", "build-essential")
    .pip_install(
        "torch",
        "numpy",
        "nvidia-cutlass-dsl>=4.3",
    )
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=600,
)
def test_minimal_kernel():
    """Test an absolute minimal CuTe DSL kernel."""
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    
    print("=" * 60)
    print("Testing Minimal CuTe DSL Kernel")
    print("=" * 60)
    print(f"CUTLASS version: {cutlass.__version__}")
    
    # Define the simplest possible kernel - just copy data
    @cute.kernel
    def copy_kernel(
        gInput: cute.Tensor,
        gOutput: cute.Tensor,
    ):
        """Simple copy kernel."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        
        idx = bidx * 128 + tidx
        
        # Simple copy
        gOutput[idx] = gInput[idx]
    
    @cute.jit
    def launch_copy(mInput, mOutput, num_blocks: int):
        """Launch copy kernel."""
        copy_kernel(mInput, mOutput).launch(
            grid=[num_blocks, 1, 1],
            block=[128, 1, 1],
        )
    
    # Test data
    N = 1024
    device = "cuda"
    
    A = torch.randn(N, dtype=torch.float32, device=device)
    B = torch.zeros(N, dtype=torch.float32, device=device)
    
    print(f"\nInput A[:5]: {A[:5]}")
    
    # Convert to CuTe tensors
    mA = from_dlpack(A).mark_layout_dynamic()
    mB = from_dlpack(B).mark_layout_dynamic()
    
    # Launch kernel
    num_blocks = N // 128
    
    try:
        launch_copy(mA, mB, num_blocks)
        torch.cuda.synchronize()
        
        print(f"Output B[:5]: {B[:5]}")
        
        # Check correctness
        diff = (A - B).abs().max().item()
        print(f"\nMax difference: {diff:.2e}")
        
        if diff < 1e-6:
            print("SUCCESS: Copy kernel works!")
            return {"status": "success", "max_diff": diff}
        else:
            print("ERROR: Copy kernel produced wrong results!")
            return {"status": "error", "max_diff": diff}
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


@app.function(
    image=image,
    gpu="B200:1",
    timeout=600,
)
def test_scale_kernel():
    """Test a scale kernel (multiply by scalar)."""
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    
    print("=" * 60)
    print("Testing Scale Kernel")
    print("=" * 60)
    
    # Define scale kernel
    @cute.kernel
    def scale_kernel(
        gInput: cute.Tensor,
        gOutput: cute.Tensor,
    ):
        """Scale kernel - multiply by 2.0."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        
        idx = bidx * 128 + tidx
        
        # Scale by 2.0
        val = gInput[idx]
        gOutput[idx] = val + val  # Use addition instead of multiplication
    
    @cute.jit
    def launch_scale(mInput, mOutput, num_blocks: int):
        """Launch scale kernel."""
        scale_kernel(mInput, mOutput).launch(
            grid=[num_blocks, 1, 1],
            block=[128, 1, 1],
        )
    
    # Test data
    N = 1024
    device = "cuda"
    
    A = torch.randn(N, dtype=torch.float32, device=device)
    B = torch.zeros(N, dtype=torch.float32, device=device)
    expected = A * 2.0
    
    print(f"\nInput A[:5]: {A[:5]}")
    print(f"Expected (A*2)[:5]: {expected[:5]}")
    
    # Convert to CuTe tensors
    mA = from_dlpack(A).mark_layout_dynamic()
    mB = from_dlpack(B).mark_layout_dynamic()
    
    # Launch kernel
    num_blocks = N // 128
    
    try:
        launch_scale(mA, mB, num_blocks)
        torch.cuda.synchronize()
        
        print(f"Output B[:5]: {B[:5]}")
        
        # Check correctness
        diff = (expected - B).abs().max().item()
        print(f"\nMax difference: {diff:.2e}")
        
        if diff < 1e-5:
            print("SUCCESS: Scale kernel works!")
            return {"status": "success", "max_diff": diff}
        else:
            print("ERROR: Scale kernel produced wrong results!")
            return {"status": "error", "max_diff": diff}
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


@app.local_entrypoint()
def main():
    """Run minimal CuTe DSL tests."""
    print("Testing minimal CuTe DSL kernels on Modal B200...")
    
    result1 = test_minimal_kernel.remote()
    print(f"\nCopy kernel result: {result1}")
    
    result2 = test_scale_kernel.remote()
    print(f"\nScale kernel result: {result2}")

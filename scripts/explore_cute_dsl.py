"""
Modal script to explore CuTe DSL API and create a simple kernel.

This explores the CuTe DSL API to understand how to build GDN decode kernel.

Usage:
    modal run scripts/explore_cute_dsl.py
"""

import modal

app = modal.App("explore-cute-dsl")

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
def explore_cute_api():
    """Explore CuTe DSL API."""
    import torch
    import cutlass
    import cutlass.cute as cute
    
    print("=" * 60)
    print("Exploring CuTe DSL API")
    print("=" * 60)
    
    # Print available attributes
    print("\n== cutlass module ==")
    print(f"cutlass.__version__ = {cutlass.__version__}")
    
    print("\n== cutlass.cute attributes ==")
    cute_attrs = [a for a in dir(cute) if not a.startswith('_')]
    print(f"Total: {len(cute_attrs)}")
    for attr in sorted(cute_attrs)[:30]:
        print(f"  {attr}")
    print("  ...")
    
    # Check for kernel decorator
    print("\n== Key decorators/functions ==")
    if hasattr(cute, 'kernel'):
        print("  cute.kernel: EXISTS")
    if hasattr(cute, 'jit'):
        print("  cute.jit: EXISTS")
    if hasattr(cute, 'Tensor'):
        print("  cute.Tensor: EXISTS")
    if hasattr(cute, 'make_tensor'):
        print("  cute.make_tensor: EXISTS")
    if hasattr(cute, 'make_layout'):
        print("  cute.make_layout: EXISTS")
    if hasattr(cute, 'compile'):
        print("  cute.compile: EXISTS")
        
    # Check runtime
    print("\n== cutlass.cute.runtime ==")
    from cutlass.cute import runtime
    runtime_attrs = [a for a in dir(runtime) if not a.startswith('_')]
    print(f"Total: {len(runtime_attrs)}")
    for attr in sorted(runtime_attrs):
        print(f"  {attr}")
    
    return {
        "status": "success",
        "cute_attrs": cute_attrs[:20],
    }


@app.function(
    image=image,
    gpu="B200:1",
    timeout=600,
)
def test_simple_cute_kernel():
    """Test a simple CuTe DSL kernel - explore API more."""
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass import Float32, Int32
    
    print("=" * 60)
    print("Testing Simple CuTe DSL Kernel")
    print("=" * 60)
    
    # Print more API details
    print("\n== cute.jit signature ==")
    import inspect
    print(inspect.signature(cute.jit))
    
    print("\n== cutlass types ==")
    print(f"cutlass.Float32 = {Float32}")
    print(f"cutlass.Int32 = {Int32}")
    
    # Look at the official examples pattern
    # From rmsnorm.py, the pattern is:
    # @cute.jit
    # def kernel_func(A, B, ...):
    #     tid = cute.threadIdx.x
    #     bid = cute.blockIdx.x
    #     ...
    
    # Try defining a kernel
    try:
        @cute.jit
        def simple_kernel(
            A,  # Input pointer/tensor
            B,  # Output pointer/tensor
            N: int,  # Scalar
        ):
            """Simple copy kernel."""
            tid = cute.threadIdx.x
            bid = cute.blockIdx.x
            idx = bid * 128 + tid
            
            if idx < N:
                B[idx] = A[idx]
        
        print("\n== Kernel defined successfully ==")
        print(f"simple_kernel = {simple_kernel}")
        print(f"type = {type(simple_kernel)}")
        
        # Try to get more info about the kernel
        if hasattr(simple_kernel, '__annotations__'):
            print(f"annotations = {simple_kernel.__annotations__}")
            
        return {
            "status": "success",
            "kernel_type": str(type(simple_kernel)),
        }
        
    except Exception as e:
        print(f"Error defining kernel: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
        }


@app.function(
    image=image,
    gpu="B200:1",
    timeout=600,
)
def test_cute_matmul():
    """Test a simple mat-vec multiply with CuTe DSL."""
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    
    print("=" * 60)
    print("Testing CuTe DSL Matrix Operations")
    print("=" * 60)
    
    # Create test data
    M, N = 128, 128
    device = "cuda"
    
    A = torch.randn(M, N, dtype=torch.float32, device=device)
    x = torch.randn(N, dtype=torch.float32, device=device)
    y_ref = A @ x  # Reference
    
    print(f"A shape: {A.shape}")
    print(f"x shape: {x.shape}")
    print(f"y_ref shape: {y_ref.shape}")
    
    # For now, just use PyTorch to verify the setup works
    # CuTe DSL GEMM requires more complex setup
    
    return {
        "status": "success",
        "y_ref_range": [y_ref.min().item(), y_ref.max().item()],
    }


@app.local_entrypoint()
def main():
    """Run exploration tests."""
    print("Exploring CuTe DSL on Modal B200...")
    
    # Explore API
    result1 = explore_cute_api.remote()
    print(f"\nAPI exploration result: {result1}")
    
    # Test simple kernel
    result2 = test_simple_cute_kernel.remote()
    print(f"\nSimple kernel result: {result2}")
    
    # Test matmul
    result3 = test_cute_matmul.remote()
    print(f"\nMatmul result: {result3}")

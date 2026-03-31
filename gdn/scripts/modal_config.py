"""
Shared Modal configuration for GDN kernel benchmarks.

All benchmark scripts should import from this module to use the same image.
This avoids rebuilding the image for each script.

Usage in benchmark scripts:
    from modal_config import app, cuda_image, volume, B200_GPU

    @app.function(image=cuda_image, gpu=B200_GPU, volumes={"/data": volume})
    def my_benchmark():
        ...
"""

import modal

# ============================================================
# SHARED APP - All scripts use the same app name for image caching
# ============================================================
app = modal.App("gdn-kernels")

# ============================================================
# SHARED CUDA IMAGE - Full toolchain for all kernels
# ============================================================
cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "git")
    .pip_install(
        "torch>=2.4.0",
        "triton>=3.0.0",
        "tabulate",
        "numpy",
    )
    .run_commands(
        # CUDA 12.8 for sm_100 (Blackwell/B200) support
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-8",
        # CUTLASS for CuTe headers
        "git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git /opt/cutlass",
    )
    .env({
        "PATH": "/usr/local/cuda-12.8/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH",
        "CUTLASS_PATH": "/opt/cutlass",
    })
)

# ============================================================
# LIGHTWEIGHT IMAGE - For Triton-only tests (faster to build)
# ============================================================
triton_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "triton>=3.0.0",
        "tabulate",
    )
)

# ============================================================
# SHARED VOLUME - For compiled kernels and results
# ============================================================
volume = modal.Volume.from_name("flashinfer-bench", create_if_missing=True)

# ============================================================
# GPU CONFIGURATION
# ============================================================
B200_GPU = "B200"

# ============================================================
# COMMON TIMEOUT SETTINGS
# ============================================================
SHORT_TIMEOUT = 300    # 5 minutes - quick tests
MEDIUM_TIMEOUT = 600   # 10 minutes - standard benchmarks  
LONG_TIMEOUT = 1800    # 30 minutes - comprehensive benchmarks
BUILD_TIMEOUT = 600    # 10 minutes - kernel compilation

# ============================================================
# KERNEL SOURCE READER UTILITY
# ============================================================
def read_kernel_sources(kernel_base_path: str) -> dict:
    """
    Read all kernel source files from the kernels directory.
    
    Args:
        kernel_base_path: Path to gdn/kernels directory
        
    Returns:
        Dict mapping filename to content
    """
    import os
    
    sources = {}
    
    # CUDA v5-v8
    cuda_dir = os.path.join(kernel_base_path, "cuda")
    if os.path.exists(cuda_dir):
        for v in ["v5", "v6", "v7", "v8"]:
            for kernel_type in ["decode", "prefill"]:
                path = os.path.join(cuda_dir, f"gdn_{kernel_type}_{v}.cuh")
                if os.path.exists(path):
                    with open(path, "r") as f:
                        sources[f"gdn_{kernel_type}_{v}.cuh"] = f.read()
    
    # CuTe v9-v11
    cute_dir = os.path.join(kernel_base_path, "cute_cpp")
    if os.path.exists(cute_dir):
        for v in ["v9", "v10", "v11"]:
            for kernel_type in ["decode", "prefill"]:
                path = os.path.join(cute_dir, f"gdn_{kernel_type}_{v}.cuh")
                if os.path.exists(path):
                    with open(path, "r") as f:
                        sources[f"gdn_{kernel_type}_{v}.cuh"] = f.read()
    
    # PTX
    ptx_dir = os.path.join(kernel_base_path, "ptx")
    if os.path.exists(ptx_dir):
        for kernel_type in ["decode", "prefill"]:
            path = os.path.join(ptx_dir, f"gdn_{kernel_type}_ptx.cuh")
            if os.path.exists(path):
                with open(path, "r") as f:
                    sources[f"gdn_{kernel_type}_ptx.cuh"] = f.read()
    
    return sources


def get_kernel_base_path() -> str:
    """Get the path to gdn/kernels directory."""
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "..", "kernels")

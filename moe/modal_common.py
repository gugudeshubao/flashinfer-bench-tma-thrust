from pathlib import Path

import modal


MOE_ROOT = Path(__file__).parent
TRACE_SET_PATH = "/data"
DEFINITION_NAME = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

# Keep a single shared image for all MoE setup / benchmark / probing tasks.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(
        "flashinfer-bench",
        "torch",
        "triton",
        "numpy",
        "huggingface-hub",
    )
    .run_commands(
        "if [ ! -d /opt/cutlass ]; then git clone --depth 1 https://github.com/NVIDIA/cutlass /opt/cutlass; fi"
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
        "TORCH_CUDA_ARCH_LIST": "10.0",
        "CUTLASS_PATH": "/opt/cutlass",
    })
    .add_local_dir(MOE_ROOT, remote_path="/root/moe")
)

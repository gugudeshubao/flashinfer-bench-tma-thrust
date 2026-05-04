"""
Inspect whether a custom CUDA extension can directly read torch.float8_e4m3fn.

Usage:
    modal run moe/scripts/inspect_float8_support.py
"""

import modal

from moe.modal_common import image

app = modal.App("tma-thrust-moe-inspect-float8-support")


CPP_SRC = r"""
#include <torch/extension.h>

torch::Tensor launch_probe(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("probe", &launch_probe, "Probe float8 tensor access");
}
"""


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>
#include <cuda_runtime.h>

__global__ void probe_kernel(const c10::Float8_e4m3fn* x, float* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = static_cast<float>(x[0]);
  }
}

torch::Tensor launch_probe(torch::Tensor x) {
  auto out = torch::zeros({1}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat32));
  probe_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<c10::Float8_e4m3fn>(),
      out.data_ptr<float>());
  return out;
}
"""


@app.function(image=image, gpu="B200:1", timeout=1800)
def inspect():
    from pathlib import Path

    import torch
    from torch.utils.cpp_extension import load_inline

    torch_root = Path(torch.__file__).resolve().parent
    print("torch:", torch.__version__)
    print("torch_root:", torch_root)

    matches = sorted(torch_root.glob("include/**/Float8*"))
    print("float8 headers:")
    for path in matches[:20]:
        print(" ", path.relative_to(torch_root))

    try:
        mod = load_inline(
            name="moe_float8_probe_ext",
            cpp_sources=[CPP_SRC],
            cuda_sources=[CUDA_SRC],
            functions=None,
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        x = torch.tensor([1.0], device="cuda", dtype=torch.float8_e4m3fn)
        y = mod.probe(x)
        print("compile: ok")
        print("probe:", y.cpu().tolist())
    except Exception as exc:
        print("compile/runtime failed:", repr(exc))


@app.local_entrypoint()
def main():
    inspect.remote()

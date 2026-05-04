"""
Probe whether a CUDA torch extension can use mm_out / addmm_out with preallocated output.

Usage:
    modal run moe/scripts/inspect_mm_out_support.py
"""

import modal

from moe.modal_common import image

app = modal.App("tma-thrust-moe-inspect-mm-out")


CPP_SRC = r"""
#include <torch/extension.h>

torch::Tensor mm_out_probe(torch::Tensor a, torch::Tensor b);
torch::Tensor addmm_out_probe(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mm_out_probe", &mm_out_probe);
  m.def("addmm_out_probe", &addmm_out_probe);
}
"""


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/addmm.h>

torch::Tensor mm_out_probe(torch::Tensor a, torch::Tensor b) {
  auto out = torch::empty({a.size(0), b.size(1)}, a.options());
  at::mm_out(out, a, b);
  return out;
}

torch::Tensor addmm_out_probe(torch::Tensor a, torch::Tensor b) {
  auto out = torch::zeros({a.size(0), b.size(1)}, a.options());
  at::addmm_out(out, out, a, b, 0.0, 1.0);
  return out;
}
"""


@app.function(image=image, gpu="B200:1", timeout=1800)
def inspect():
    import torch
    from torch.utils.cpp_extension import load_inline

    try:
        mod = load_inline(
            name="moe_mm_out_probe_ext",
            cpp_sources=[CPP_SRC],
            cuda_sources=[CUDA_SRC],
            functions=None,
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        a = torch.randn(4, 8, device="cuda", dtype=torch.float32)
        b = torch.randn(8, 16, device="cuda", dtype=torch.float32)
        y1 = mod.mm_out_probe(a, b)
        y2 = mod.addmm_out_probe(a, b)
        ref = a @ b
        print("compile: ok")
        print("mm_out max_abs_err:", (y1 - ref).abs().max().item())
        print("addmm_out max_abs_err:", (y2 - ref).abs().max().item())
    except Exception as exc:
        print("compile/runtime failed:", repr(exc))


@app.local_entrypoint()
def main():
    inspect.remote()

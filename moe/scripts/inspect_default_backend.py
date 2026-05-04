"""
Inspect which backend the current default moe/solution/triton/kernel.py can load.

Usage:
    modal run moe/scripts/inspect_default_backend.py
"""

import modal

from moe.modal_common import image

app = modal.App("tma-thrust-moe-inspect-default-backend")


@app.function(image=image, gpu="B200:1", timeout=1800)
def inspect():
    import importlib

    mod = importlib.import_module("moe.solution.triton.kernel")
    print("module:", mod.__file__)

    ptx = mod._load_ptx_torch_ext_module()
    print("ptx_torch_ext:", "loaded" if ptx is not None else "none")
    print("ptx_torch_failed:", getattr(mod, "_ptx_torch_ext_failed", None))

    cute = mod._load_torch_ext_module()
    print("cute_torch_ext:", "loaded" if cute is not None else "none")
    print("cute_torch_failed:", getattr(mod, "_torch_ext_failed", None))

    print("selected_example_seq_1:", "ptx" if ptx is not None else ("cute" if cute is not None else "python"))


@app.local_entrypoint()
def main():
    inspect.remote()

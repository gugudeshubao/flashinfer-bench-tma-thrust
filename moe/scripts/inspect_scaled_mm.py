"""
Inspect torch._scaled_mm / aten::_scaled_mm inside the shared Modal image.

Usage:
    modal run moe/scripts/inspect_scaled_mm.py
"""

import modal

from moe.modal_common import image

app = modal.App("tma-thrust-moe-inspect-scaled-mm")


@app.function(image=image)
def show_scaled_mm():
    from pathlib import Path

    import torch

    print("torch", torch.__version__)
    torch_root = Path(torch.__file__).resolve().parent
    ops_dir = torch_root / "include" / "ATen" / "ops"
    if ops_dir.exists():
        print("scaled_mm headers:")
        for path in sorted(ops_dir.glob("*scaled_mm*")):
            print(" ", path.name)

    try:
        op = torch.ops.aten._scaled_mm.default
        print("aten._scaled_mm.default:", op)
        print("schema:", op._schema)
    except Exception as exc:
        print("aten._scaled_mm.default unavailable:", repr(exc))

    try:
        print("dispatch table:")
        print(torch._C._dispatch_dump_table("aten::_scaled_mm"))
    except Exception as exc:
        print("dispatch table unavailable:", repr(exc))


@app.local_entrypoint()
def main():
    show_scaled_mm.remote()

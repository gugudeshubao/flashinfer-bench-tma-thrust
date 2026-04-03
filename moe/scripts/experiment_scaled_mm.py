"""
GPU-side probes for torch._scaled_mm / aten::_scaled_mm on MoE-like shapes.

Usage:
    modal run moe/scripts/experiment_scaled_mm.py
"""

import modal

from moe.modal_common import image

app = modal.App("tma-thrust-moe-experiment-scaled-mm")


@app.function(image=image, gpu="B200:1", timeout=1800)
def run_probe():
    import torch
    import torch._meta_registrations as mr

    H = 7168
    O = 4096
    BLK = 128
    HB = H // BLK
    OB = O // BLK

    def test_case(t: int):
        print(f"\n=== t={t} ===")
        a = torch.randn((t, H), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        b = torch.randn((H, O), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand((t, HB), device="cuda", dtype=torch.float32).contiguous()
        scale_b = torch.rand((HB, OB), device="cuda", dtype=torch.float32).contiguous()

        print("a", a.shape, a.stride(), a.dtype)
        print("b", b.shape, b.stride(), b.dtype)
        print("scale_a", scale_a.shape, scale_a.stride(), scale_a.dtype)
        print("scale_b", scale_b.shape, scale_b.stride(), scale_b.dtype)

        for name, fn in [
            ("torch._scaled_mm", lambda: torch._scaled_mm(
                a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32
            )),
            ("aten._scaled_mm.default", lambda: torch.ops.aten._scaled_mm.default(
                a, b, scale_a, scale_b, None, None, torch.float32, False
            )),
        ]:
            try:
                out = fn()
                print(name, "OK", out.shape, out.dtype, out.stride())
            except Exception as exc:
                print(name, "ERR", repr(exc))

        try:
            print(
                "ScalingType",
                {name: int(member.value) for name, member in mr.ScalingType.__members__.items()},
            )
            print(
                "SwizzleType",
                {name: int(member.value) for name, member in mr.SwizzleType.__members__.items()},
            )
        except Exception as exc:
            print("enum dump ERR", repr(exc))

        try:
            out = torch.ops.aten._scaled_mm_v2.default(
                a,
                b,
                [scale_a],
                [int(mr.ScalingType.BlockWise1x128.value)],
                [],
                [scale_b],
                [int(mr.ScalingType.BlockWise128x128.value)],
                [],
                None,
                torch.float32,
                [],
                False,
            )
            print("_scaled_mm_v2 blockwise OK", out.shape, out.dtype, out.stride())
        except Exception as exc:
            print("_scaled_mm_v2 blockwise ERR", repr(exc))

        for op_name in ["_scaled_mm_v2", "_scaled_mm_v2.default"]:
            try:
                op = eval(f"torch.ops.aten.{op_name}")
                print(op_name, "schema", op._schema)
            except Exception as exc:
                print(op_name, "unavailable", repr(exc))

    for t in [1, 7, 16]:
        test_case(t)


@app.local_entrypoint()
def main():
    run_probe.remote()

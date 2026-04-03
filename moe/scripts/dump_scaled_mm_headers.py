"""
Dump relevant pieces of PyTorch scaled_mm headers from the shared Modal image.

Usage:
    modal run moe/scripts/dump_scaled_mm_headers.py
"""

from pathlib import Path

import modal

from moe.modal_common import image

app = modal.App("tma-thrust-moe-dump-scaled-mm-headers")


def _print_matches(path: Path, patterns: tuple[str, ...], max_lines: int = 120):
    print(f"\n===== {path.name} =====")
    lines = path.read_text().splitlines()
    matched = 0
    for idx, line in enumerate(lines, start=1):
        if any(p in line for p in patterns):
            start = max(1, idx - 3)
            end = min(len(lines), idx + 8)
            for j in range(start, end + 1):
                print(f"{j:04d}: {lines[j - 1]}")
            print("-----")
            matched += end - start + 2
            if matched >= max_lines:
                break


@app.function(image=image)
def main_remote():
    import torch

    torch_root = Path(torch.__file__).resolve().parent
    ops_dir = torch_root / "include" / "ATen" / "ops"
    targets = [
        ops_dir / "_scaled_mm_v2.h",
        ops_dir / "_scaled_mm_v2_ops.h",
        ops_dir / "_scaled_mm_v2_native.h",
        ops_dir / "_scaled_mm.h",
        ops_dir / "_scaled_mm_ops.h",
    ]

    patterns = (
        "_scaled_mm_v2",
        "recipe",
        "swizzle",
        "contraction_dim",
        "BlockWise",
        "blockwise",
        "MX",
    )

    for path in targets:
        if path.exists():
            _print_matches(path, patterns)
        else:
            print(f"missing: {path}")


@app.local_entrypoint()
def main():
    main_remote.remote()

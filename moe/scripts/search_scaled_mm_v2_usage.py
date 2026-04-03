"""
Search the installed PyTorch package for scaled_mm_v2 usage examples.

Usage:
    modal run moe/scripts/search_scaled_mm_v2_usage.py
"""

from pathlib import Path

import modal

from moe.modal_common import image

app = modal.App("tma-thrust-moe-search-scaled-mm-v2")


@app.function(image=image)
def main_remote():
    import torch

    root = Path(torch.__file__).resolve().parent.parent
    patterns = ("_scaled_mm_v2", "recipe_a", "swizzle_a")
    shown = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".h", ".hpp", ".cpp", ".cu", ".cuh", ".txt", ".yaml"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        if any(p in text for p in patterns):
            print(f"\n===== {path} =====")
            lines = text.splitlines()
            for idx, line in enumerate(lines, start=1):
                if any(p in line for p in patterns):
                    start = max(1, idx - 2)
                    end = min(len(lines), idx + 4)
                    for j in range(start, end + 1):
                        print(f"{j:04d}: {lines[j - 1]}")
                    print("-----")
                    shown += 1
                    if shown >= 40:
                        return


@app.local_entrypoint()
def main():
    main_remote.remote()

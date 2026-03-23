"""
Setup Modal volume with GDN kernel definitions and workloads.

Usage:
    # Option A: Download from HuggingFace (requires internet access on Modal)
    modal run scripts/setup_volume.py

    # Option B: Upload local definitions + generate synthetic workloads
    modal run scripts/setup_volume.py --mode synthetic

This script:
1. Creates the 'flashinfer-trace' volume if it doesn't exist
2. Uploads GDN definition JSON files
3. Downloads real workloads from HuggingFace OR generates synthetic ones
"""

import json
import math
import uuid
from pathlib import Path

import modal

app = modal.App("tma-thrust-setup")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

REPO_ROOT = Path(__file__).parent.parent

# Definition files are in flashinfer_trace/definitions/gdn/
GDN_DECODE_DEF = REPO_ROOT / "flashinfer_trace" / "definitions" / "gdn" / "gdn_decode_qk4_v8_d128_k_last.json"
GDN_PREFILL_DEF = REPO_ROOT / "flashinfer_trace" / "definitions" / "gdn" / "gdn_prefill_qk4_v8_d128_k_last.json"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "numpy", "huggingface-hub")
)


def make_decode_workloads(batch_sizes=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)):
    """Generate synthetic GDN decode workloads for various batch sizes."""
    lines = []
    scale = 1.0 / math.sqrt(128)
    for bs in batch_sizes:
        w = {
            "definition": "gdn_decode_qk4_v8_d128_k_last",
            "solution": None,
            "workload": {
                "uuid": str(uuid.uuid4()),
                "axes": {"batch_size": bs},
                "inputs": {
                    "q":       {"type": "random"},
                    "k":       {"type": "random"},
                    "v":       {"type": "random"},
                    "state":   {"type": "random"},
                    "A_log":   {"type": "random"},
                    "a":       {"type": "random"},
                    "dt_bias": {"type": "random"},
                    "b":       {"type": "random"},
                    "scale":   {"type": "scalar", "value": scale},
                },
            },
            "evaluation": None,
        }
        lines.append(json.dumps(w))
    return "\n".join(lines)


def make_prefill_workloads():
    """Generate synthetic GDN prefill workloads (various seq_len / num_seqs combos)."""
    lines = []
    scale = 1.0 / math.sqrt(128)
    configs = [
        # (total_seq_len, num_seqs)  -- equal-length sequences
        (64,   1),
        (128,  1),
        (256,  1),
        (512,  1),
        (1024, 1),
        (128,  4),
        (256,  4),
        (512,  4),
        (1024, 4),
        (2048, 8),
        (4096, 8),
        (8192, 16),
    ]
    for total_len, num_seqs in configs:
        seq_len = total_len // num_seqs
        w = {
            "definition": "gdn_prefill_qk4_v8_d128_k_last",
            "solution": None,
            "workload": {
                "uuid": str(uuid.uuid4()),
                "axes": {
                    "total_seq_len": total_len,
                    "num_seqs": num_seqs,
                    "len_cu_seqlens": num_seqs + 1,
                },
                "inputs": {
                    "q":          {"type": "random"},
                    "k":          {"type": "random"},
                    "v":          {"type": "random"},
                    "state":      {"type": "random"},
                    "A_log":      {"type": "random"},
                    "a":          {"type": "random"},
                    "dt_bias":    {"type": "random"},
                    "b":          {"type": "random"},
                    "cu_seqlens": {"type": "random"},
                    "scale":      {"type": "scalar", "value": scale},
                },
            },
            "evaluation": None,
        }
        lines.append(json.dumps(w))
    return "\n".join(lines)


@app.function(
    image=image,
    volumes={TRACE_SET_PATH: trace_volume},
    timeout=600,
)
def setup_synthetic(decode_def_json: str, prefill_def_json: str):
    """Upload definitions and synthetic workloads to the Modal volume."""
    import os
    root = Path(TRACE_SET_PATH)

    # Create directory structure
    (root / "definitions" / "gdn").mkdir(parents=True, exist_ok=True)
    (root / "workloads" / "gdn").mkdir(parents=True, exist_ok=True)

    # Write definition files
    (root / "definitions" / "gdn" / "gdn_decode_qk4_v8_d128_k_last.json").write_text(decode_def_json)
    (root / "definitions" / "gdn" / "gdn_prefill_qk4_v8_d128_k_last.json").write_text(prefill_def_json)
    print("Wrote definition files.")

    # Write workloads
    decode_wl = make_decode_workloads()
    prefill_wl = make_prefill_workloads()
    (root / "workloads" / "gdn" / "gdn_decode_qk4_v8_d128_k_last.jsonl").write_text(decode_wl)
    (root / "workloads" / "gdn" / "gdn_prefill_qk4_v8_d128_k_last.jsonl").write_text(prefill_wl)
    print(f"Wrote {decode_wl.count(chr(10))+1} decode workloads.")
    print(f"Wrote {prefill_wl.count(chr(10))+1} prefill workloads.")

    trace_volume.commit()
    print("Volume committed.")

    # Verify
    for p in sorted(root.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(root)}")


@app.function(
    image=image,
    volumes={TRACE_SET_PATH: trace_volume},
    timeout=600,
)
def setup_from_hf():
    """Download contest dataset from HuggingFace into the Modal volume."""
    import subprocess
    root = Path(TRACE_SET_PATH)
    root.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset from HuggingFace...")
    subprocess.run([
        "python3", "-c",
        """
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='flashinfer-ai/mlsys26-contest',
    repo_type='dataset',
    local_dir='/data',
    ignore_patterns=['*.png','*.jpg','*.svg'],
)
print('Download complete.')
"""
    ], check=True)
    trace_volume.commit()
    print("HuggingFace dataset downloaded and committed to volume.")


@app.local_entrypoint()
def main(mode: str = "synthetic"):
    """
    Setup Modal volume for GDN benchmarks.
    --mode synthetic  (default): use synthetic workloads
    --mode hf:                   download from HuggingFace
    """
    if mode == "hf":
        print("Setting up from HuggingFace...")
        setup_from_hf.remote()
    else:
        print("Setting up with synthetic workloads...")
        # Read definitions locally and pass as strings
        decode_json = GDN_DECODE_DEF.read_text()
        prefill_json = GDN_PREFILL_DEF.read_text()
        setup_synthetic.remote(decode_json, prefill_json)

    print("Done! Volume 'flashinfer-trace' is ready.")

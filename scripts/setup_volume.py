"""
Setup Modal volume with GDN kernel definitions and workloads.

Usage:
    modal run scripts/setup_volume.py              # synthetic workloads
    modal run scripts/setup_volume.py --mode hf    # download from HuggingFace
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

GDN_DECODE_DEF = REPO_ROOT / "flashinfer_trace" / "definitions" / "gdn" / "gdn_decode_qk4_v8_d128_k_last.json"
GDN_PREFILL_DEF = REPO_ROOT / "flashinfer_trace" / "definitions" / "gdn" / "gdn_prefill_qk4_v8_d128_k_last.json"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "numpy", "huggingface-hub", "safetensors")
)


def make_decode_workloads(batch_sizes=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)):
    scale = 1.0 / math.sqrt(128)
    lines = []
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
    return lines


def make_prefill_workloads(root: Path):
    """Generate prefill workloads with proper cu_seqlens saved as safetensors."""
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file

    scale = 1.0 / math.sqrt(128)
    configs = [
        # (total_seq_len, num_seqs)
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

    tensors_dir = root / "tensors" / "gdn_prefill"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for total_len, num_seqs in configs:
        seq_len = total_len // num_seqs
        wid = str(uuid.uuid4())

        # Build cu_seqlens: [0, seq_len, 2*seq_len, ..., total_len]
        cu_seqlens = torch.arange(0, total_len + 1, seq_len, dtype=torch.int64)
        assert len(cu_seqlens) == num_seqs + 1
        assert cu_seqlens[-1].item() == total_len

        # Save cu_seqlens, zero state, and L2-normalized k to safetensors.
        # The GDN delta rule is stable only when ||k||^2 ≈ 1 per head vector.
        # With raw torch.randn, ||k||^2 ≈ head_size=128, causing the state to
        # grow ~128x per step → float32 overflow within ~30 steps from zero state.
        # L2-normalizing k (per head vector) ensures beta*||k||^2 = beta < 1.
        k_raw = torch.randn(total_len, 4, 128, dtype=torch.float32)
        k_norm = F.normalize(k_raw, dim=-1).to(torch.bfloat16)
        state_zero = torch.zeros(num_seqs, 8, 128, 128, dtype=torch.float32)

        fname = f"tensors_{wid}.safetensors"
        save_file(
            {"cu_seqlens": cu_seqlens, "k": k_norm, "state": state_zero},
            str(tensors_dir / fname),
        )
        rel_path = f"tensors/gdn_prefill/{fname}"

        w = {
            "definition": "gdn_prefill_qk4_v8_d128_k_last",
            "solution": None,
            "workload": {
                "uuid": wid,
                "axes": {
                    "total_seq_len": total_len,
                    "num_seqs": num_seqs,
                    "len_cu_seqlens": num_seqs + 1,
                },
                "inputs": {
                    "q":          {"type": "random"},
                    "k":          {"type": "safetensors", "path": rel_path, "tensor_key": "k"},
                    "v":          {"type": "random"},
                    "state":      {"type": "safetensors", "path": rel_path, "tensor_key": "state"},
                    "A_log":      {"type": "random"},
                    "a":          {"type": "random"},
                    "dt_bias":    {"type": "random"},
                    "b":          {"type": "random"},
                    "cu_seqlens": {"type": "safetensors", "path": rel_path, "tensor_key": "cu_seqlens"},
                    "scale":      {"type": "scalar", "value": scale},
                },
            },
            "evaluation": None,
        }
        lines.append(json.dumps(w))
    return lines


@app.function(
    image=image,
    volumes={TRACE_SET_PATH: trace_volume},
    timeout=600,
)
def setup_synthetic(decode_def_json: str, prefill_def_json: str):
    """Upload definitions and synthetic workloads to the Modal volume."""
    root = Path(TRACE_SET_PATH)
    (root / "definitions" / "gdn").mkdir(parents=True, exist_ok=True)
    (root / "workloads" / "gdn").mkdir(parents=True, exist_ok=True)

    (root / "definitions" / "gdn" / "gdn_decode_qk4_v8_d128_k_last.json").write_text(decode_def_json)
    (root / "definitions" / "gdn" / "gdn_prefill_qk4_v8_d128_k_last.json").write_text(prefill_def_json)
    print("Wrote definition files.")

    decode_lines = make_decode_workloads()
    (root / "workloads" / "gdn" / "gdn_decode_qk4_v8_d128_k_last.jsonl").write_text(
        "\n".join(decode_lines)
    )
    print(f"Wrote {len(decode_lines)} decode workloads.")

    prefill_lines = make_prefill_workloads(root)
    (root / "workloads" / "gdn" / "gdn_prefill_qk4_v8_d128_k_last.jsonl").write_text(
        "\n".join(prefill_lines)
    )
    print(f"Wrote {len(prefill_lines)} prefill workloads.")

    trace_volume.commit()
    print("Volume committed.")
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in (".json", ".jsonl"):
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
    print("HuggingFace dataset downloaded and committed.")


@app.local_entrypoint()
def main(mode: str = "synthetic"):
    """
    Setup Modal volume for GDN benchmarks.
    --mode synthetic (default) | hf
    """
    if mode == "hf":
        print("Setting up from HuggingFace...")
        setup_from_hf.remote()
    else:
        print("Setting up with synthetic workloads...")
        decode_json = GDN_DECODE_DEF.read_text()
        prefill_json = GDN_PREFILL_DEF.read_text()
        setup_synthetic.remote(decode_json, prefill_json)

    print("Done! Volume 'flashinfer-trace' is ready.")

"""
Setup Modal volume with MoE kernel definitions and workloads.

Usage:
    modal run moe/scripts/setup_moe_volume.py              # synthetic workloads
    modal run moe/scripts/setup_moe_volume.py --mode hf    # download from HuggingFace
"""

import json
import uuid
from pathlib import Path

import modal

app = modal.App("tma-thrust-moe-setup")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

MOE_ROOT = Path(__file__).parent.parent
MOE_DEF_FILE = MOE_ROOT / "trace_definitions" / "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "numpy", "huggingface-hub")
)

DEFINITION_NAME = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"


def make_moe_workloads(seq_lens=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)):
    """Generate synthetic MoE workloads with varying sequence lengths."""
    lines = []
    for seq_len in seq_lens:
        workload = {
            "definition": DEFINITION_NAME,
            "solution": None,
            "workload": {
                "uuid": str(uuid.uuid4()),
                "axes": {"seq_len": seq_len},
                "inputs": {
                    "routing_logits": {"type": "random"},
                    "routing_bias": {"type": "random"},
                    "hidden_states": {"type": "random"},
                    "hidden_states_scale": {"type": "random"},
                    "gemm1_weights": {"type": "random"},
                    "gemm1_weights_scale": {"type": "random"},
                    "gemm2_weights": {"type": "random"},
                    "gemm2_weights_scale": {"type": "random"},
                    "local_expert_offset": {"type": "scalar", "value": 0},
                    "routed_scaling_factor": {"type": "scalar", "value": 2.5},
                },
            },
            "evaluation": None,
        }
        lines.append(json.dumps(workload))
    return lines


@app.function(
    image=image,
    volumes={TRACE_SET_PATH: trace_volume},
    timeout=600,
)
def setup_synthetic(def_json: str):
    """Upload MoE definition and synthetic workloads to the Modal volume."""
    root = Path(TRACE_SET_PATH)
    (root / "definitions" / "moe").mkdir(parents=True, exist_ok=True)
    (root / "workloads" / "moe").mkdir(parents=True, exist_ok=True)

    # Write definition
    def_path = root / "definitions" / "moe" / f"{DEFINITION_NAME}.json"
    def_path.write_text(def_json)
    print(f"Wrote definition: {def_path.relative_to(root)}")

    # Write workloads
    workload_lines = make_moe_workloads()
    wl_path = root / "workloads" / "moe" / f"{DEFINITION_NAME}.jsonl"
    wl_path.write_text("\n".join(workload_lines))
    print(f"Wrote {len(workload_lines)} MoE workloads: {wl_path.relative_to(root)}")

    trace_volume.commit()
    print("Volume committed.")


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
def main(mode: str = "hf"):
    """
    Setup Modal volume for MoE benchmarks.
    --mode hf (default, recommended) | synthetic
    """
    if mode == "hf":
        print("Setting up from HuggingFace (recommended for official workloads)...")
        setup_from_hf.remote()
    else:
        print("Setting up with synthetic workloads...")
        def_json = MOE_DEF_FILE.read_text()
        setup_synthetic.remote(def_json)

    print("Done! Volume 'flashinfer-trace' is ready for MoE benchmarks.")

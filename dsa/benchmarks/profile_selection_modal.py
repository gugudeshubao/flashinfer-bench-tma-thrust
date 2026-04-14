"""Detailed Modal profiler for DSA selection path."""

from pathlib import Path

import modal


DSA_ROOT = Path(__file__).resolve().parents[1]

app = modal.App("tma-thrust-dsa-profile-selection")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "triton")
    .add_local_dir(DSA_ROOT, remote_path="/root/dsa")
)


@app.function(image=image, gpu="B200:1", timeout=3600)
def run_profile(iters: int = 20) -> dict:
    import math
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")

    from dsa.common.reference import build_causal_mask
    from dsa.prefill.solution.triton.kernel import _get_causal_bool_mask

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    def make_inputs(query_len: int, key_len: int, topk: int, causal: bool):
        batch_size = 1
        num_index_heads = 8
        index_dim = 32
        index_q = torch.randn(batch_size, query_len, num_index_heads, index_dim, device=device, dtype=dtype)
        index_k = torch.randn(batch_size, key_len, index_dim, device=device, dtype=dtype)
        index_weights = torch.randn(batch_size, query_len, num_index_heads, device=device, dtype=torch.float32)
        attn_mask = build_causal_mask(query_len, key_len, device=device) if causal else None
        return {
            "index_q": index_q,
            "index_k": index_k,
            "index_weights": index_weights,
            "attn_mask": attn_mask,
            "topk": topk,
            "index_dim": index_dim,
        }

    def _sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def profile_case(name: str, inputs: dict, causal: bool) -> dict:
        batch_size, query_len, num_index_heads, index_dim = inputs["index_q"].shape
        key_len = inputs["index_k"].shape[1]
        scale = 1.0 / math.sqrt(index_dim)

        weighted_ms = 0.0
        bmm_ms = 0.0
        mask_ms = 0.0
        topk_ms = 0.0
        total_ms = 0.0

        for _ in range(iters):
            _sync()
            t0 = time.perf_counter()

            q_flat = inputs["index_q"].float().reshape(batch_size * query_len, num_index_heads, index_dim).contiguous()
            w_flat = inputs["index_weights"].float().reshape(batch_size * query_len, 1, num_index_heads).contiguous()
            weighted_q = torch.bmm(w_flat, q_flat).reshape(batch_size, query_len, index_dim)

            _sync()
            t1 = time.perf_counter()

            scores = torch.bmm(weighted_q, inputs["index_k"].float().transpose(1, 2).contiguous())
            scores = scores * scale

            _sync()
            t2 = time.perf_counter()

            if causal:
                causal_invalid = _get_causal_bool_mask(query_len, key_len, inputs["index_q"].device)
                scores = scores.masked_fill(causal_invalid.unsqueeze(0), float("-inf"))
            elif inputs["attn_mask"] is not None:
                scores = scores + inputs["attn_mask"]

            _sync()
            t3 = time.perf_counter()

            _ = torch.topk(scores, k=min(inputs["topk"], key_len), dim=-1, largest=True, sorted=False)

            _sync()
            t4 = time.perf_counter()

            weighted_ms += (t1 - t0) * 1000.0
            bmm_ms += (t2 - t1) * 1000.0
            mask_ms += (t3 - t2) * 1000.0
            topk_ms += (t4 - t3) * 1000.0
            total_ms += (t4 - t0) * 1000.0

        return {
            "name": name,
            "weighted_ms": weighted_ms / iters,
            "bmm_ms": bmm_ms / iters,
            "mask_ms": mask_ms / iters,
            "topk_ms": topk_ms / iters,
            "total_ms": total_ms / iters,
        }

    torch.manual_seed(0)
    prefill_case = make_inputs(query_len=1024, key_len=1024, topk=128, causal=True)
    decode_case = make_inputs(query_len=1, key_len=8192, topk=128, causal=False)

    return {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "iters": iters,
        "prefill": profile_case("prefill_1024_128", prefill_case, causal=True),
        "decode": profile_case("decode_8192_128", decode_case, causal=False),
    }


def _print_case(case: dict) -> None:
    print(case["name"])
    print(
        f"  weighted={case['weighted_ms']:.3f}ms "
        f"bmm={case['bmm_ms']:.3f}ms "
        f"mask={case['mask_ms']:.3f}ms "
        f"topk={case['topk_ms']:.3f}ms "
        f"total={case['total_ms']:.3f}ms"
    )


@app.local_entrypoint()
def main(iters: int = 20) -> None:
    results = run_profile.remote(iters=iters)
    print(f"Device: {results['device']} ({results.get('gpu_name')})")
    print(f"Iters: {results['iters']}")
    print()
    _print_case(results["prefill"])
    print()
    _print_case(results["decode"])

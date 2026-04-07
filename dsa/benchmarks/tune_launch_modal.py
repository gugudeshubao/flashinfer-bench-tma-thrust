"""Modal launch-parameter sweep for DSA Triton kernels."""

from pathlib import Path

import modal


DSA_ROOT = Path(__file__).resolve().parents[1]

app = modal.App("tma-thrust-dsa-tune-launch")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "triton")
    .add_local_dir(DSA_ROOT, remote_path="/root/dsa")
)


def _cases():
    return [
        {"name": "prefill_1024", "query_len": 1024, "key_len": 1024, "topk": 128, "causal": True},
        {"name": "prefill_2048", "query_len": 2048, "key_len": 2048, "topk": 128, "causal": True},
        {"name": "decode_2048", "query_len": 1, "key_len": 2048, "topk": 64, "causal": False},
        {"name": "decode_8192", "query_len": 1, "key_len": 8192, "topk": 128, "causal": False},
    ]


@app.function(image=image, gpu="B200:1", timeout=3600)
def run_tuning(warmup: int = 3, iters: int = 10) -> dict:
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")

    from dsa.common.reference import build_causal_mask
    from dsa.decode.solution.triton.kernel import kernel as decode_triton
    from dsa.prefill.solution.triton.kernel import kernel as prefill_triton

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    def make_inputs(query_len: int, key_len: int, topk: int, causal: bool):
        batch_size = 1
        num_heads = 16
        qk_nope_head_dim = 32
        rope_dim = 16
        v_head_dim = 32
        kv_rank = 64
        num_index_heads = 8
        index_dim = 32

        q_nope = torch.randn(batch_size, query_len, num_heads, qk_nope_head_dim, device=device, dtype=dtype)
        q_pe = torch.randn(batch_size, query_len, num_heads, rope_dim, device=device, dtype=dtype)
        compressed_kv = torch.randn(batch_size, key_len, kv_rank, device=device, dtype=dtype)
        k_pe = torch.randn(batch_size, key_len, rope_dim, device=device, dtype=dtype)
        wkv_b = torch.randn(num_heads, qk_nope_head_dim + v_head_dim, kv_rank, device=device, dtype=dtype)
        index_q = torch.randn(batch_size, query_len, num_index_heads, index_dim, device=device, dtype=dtype)
        index_k = torch.randn(batch_size, key_len, index_dim, device=device, dtype=dtype)
        index_weights = torch.randn(batch_size, query_len, num_index_heads, device=device, dtype=torch.float32)
        attn_mask = build_causal_mask(query_len, key_len, device=device) if causal else None
        return (
            q_nope,
            q_pe,
            compressed_kv,
            k_pe,
            wkv_b,
            index_q,
            index_k,
            index_weights,
            topk,
            attn_mask,
        )

    def time_ms(fn, args, *, topk, attn_mask, causal, num_warps, num_stages):
        kwargs = {
            "backend": "triton",
            "causal_mask_hint": causal,
            "num_warps_override": num_warps,
            "num_stages_override": num_stages,
        }
        for _ in range(warmup):
            fn(*args, topk=topk, attn_mask=attn_mask, **kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn(*args, topk=topk, attn_mask=attn_mask, **kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0 / iters

    results = {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "warmup": warmup,
        "iters": iters,
        "cases": [],
    }

    torch.manual_seed(0)
    for case in _cases():
        inputs = make_inputs(case["query_len"], case["key_len"], case["topk"], case["causal"])
        kernel_fn = prefill_triton if case["query_len"] > 1 else decode_triton
        measurements = []
        for num_warps in [1, 2, 4]:
            for num_stages in [1, 2, 3]:
                if num_warps == 1 and case["query_len"] > 1:
                    continue
                ms = time_ms(
                    kernel_fn,
                    inputs[:8],
                    topk=inputs[8],
                    attn_mask=inputs[9],
                    causal=case["causal"],
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
                measurements.append(
                    {
                        "num_warps": num_warps,
                        "num_stages": num_stages,
                        "latency_ms": ms,
                    }
                )
        measurements.sort(key=lambda x: x["latency_ms"])
        results["cases"].append(
            {
                **case,
                "best": measurements[0],
                "measurements": measurements,
            }
        )

    return results


def _print_results(results: dict) -> None:
    print(f"Device: {results['device']} ({results.get('gpu_name')})")
    print(f"Warmup={results['warmup']}, iters={results['iters']}")
    for case in results["cases"]:
        best = case["best"]
        print(
            f"\n{case['name']}: q={case['query_len']} k={case['key_len']} topk={case['topk']} "
            f"best=({best['num_warps']} warps, {best['num_stages']} stages) "
            f"{best['latency_ms']:.3f}ms"
        )
        for row in case["measurements"]:
            print(
                f"  warps={row['num_warps']} stages={row['num_stages']} "
                f"latency={row['latency_ms']:.3f}ms"
            )


@app.local_entrypoint()
def main(warmup: int = 3, iters: int = 10) -> None:
    results = run_tuning.remote(warmup=warmup, iters=iters)
    _print_results(results)

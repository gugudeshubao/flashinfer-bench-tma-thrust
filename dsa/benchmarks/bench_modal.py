"""Modal benchmark runner for DSA baseline vs Triton solution."""

from pathlib import Path

import modal


DSA_ROOT = Path(__file__).resolve().parents[1]

app = modal.App("tma-thrust-dsa-bench")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "triton")
    .add_local_dir(DSA_ROOT, remote_path="/root/dsa")
)


def _default_cases():
    return {
        "prefill": [
            {"name": "p256", "query_len": 256, "key_len": 256, "topk": 64},
            {"name": "p512", "query_len": 512, "key_len": 512, "topk": 64},
            {"name": "p1024", "query_len": 1024, "key_len": 1024, "topk": 128},
            {"name": "p2048", "query_len": 2048, "key_len": 2048, "topk": 128},
            {"name": "p4096", "query_len": 4096, "key_len": 4096, "topk": 128},
        ],
        "decode": [
            {"name": "d2048", "query_len": 1, "key_len": 2048, "topk": 64},
            {"name": "d4096", "query_len": 1, "key_len": 4096, "topk": 64},
            {"name": "d8192", "query_len": 1, "key_len": 8192, "topk": 128},
        ],
    }


@app.function(image=image, gpu="B200:1", timeout=3600)
def run_benchmarks(warmup: int = 5, iters: int = 30) -> dict:
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")

    from dsa.common.reference import build_causal_mask
    from dsa.decode.baseline.python.kernel import kernel as decode_baseline
    from dsa.decode.solution.triton.kernel import kernel as decode_triton
    from dsa.prefill.baseline.python.kernel import kernel as prefill_baseline
    from dsa.prefill.solution.triton.kernel import kernel as prefill_triton, resolve_backend_mode

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    def make_inputs(query_len: int, key_len: int, topk: int):
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
        attn_mask = build_causal_mask(query_len, key_len, device=device) if query_len == key_len else None
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

    def time_ms(fn, args, *, topk: int, attn_mask, **kwargs):
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
        "prefill": [],
        "decode": [],
    }

    torch.manual_seed(0)
    for case in _default_cases()["prefill"]:
        inputs = make_inputs(case["query_len"], case["key_len"], case["topk"])
        backend_used = resolve_backend_mode(
            backend="auto",
            q_nope=inputs[0],
            q_pe=inputs[1],
            compressed_kv=inputs[2],
            topk=inputs[8],
            topk_indices=None,
        )
        baseline_ms = time_ms(prefill_baseline, inputs[:8], topk=inputs[8], attn_mask=inputs[9])
        triton_ms = time_ms(prefill_triton, inputs[:8], topk=inputs[8], attn_mask=inputs[9], backend="auto")
        results["prefill"].append(
            {
                "name": case["name"],
                "query_len": case["query_len"],
                "key_len": case["key_len"],
                "topk": case["topk"],
                "backend_used": backend_used,
                "baseline_ms": baseline_ms,
                "triton_ms": triton_ms,
                "speedup": baseline_ms / triton_ms,
            }
        )

    for case in _default_cases()["decode"]:
        inputs = make_inputs(case["query_len"], case["key_len"], case["topk"])
        backend_used = resolve_backend_mode(
            backend="auto",
            q_nope=inputs[0],
            q_pe=inputs[1],
            compressed_kv=inputs[2],
            topk=inputs[8],
            topk_indices=None,
        )
        baseline_ms = time_ms(decode_baseline, inputs[:8], topk=inputs[8], attn_mask=inputs[9])
        triton_ms = time_ms(decode_triton, inputs[:8], topk=inputs[8], attn_mask=inputs[9], backend="auto")
        results["decode"].append(
            {
                "name": case["name"],
                "query_len": case["query_len"],
                "key_len": case["key_len"],
                "topk": case["topk"],
                "backend_used": backend_used,
                "baseline_ms": baseline_ms,
                "triton_ms": triton_ms,
                "speedup": baseline_ms / triton_ms,
            }
        )

    return results


def _print_results(results: dict) -> None:
    print(f"Device: {results['device']} ({results.get('gpu_name')})")
    print(f"Warmup={results['warmup']}, iters={results['iters']}")

    print("\nPrefill")
    for row in results["prefill"]:
        print(
            f"  {row['name']}: q={row['query_len']} k={row['key_len']} topk={row['topk']} backend={row['backend_used']} "
            f"baseline={row['baseline_ms']:.3f}ms triton={row['triton_ms']:.3f}ms "
            f"speedup={row['speedup']:.3f}x"
        )

    print("\nDecode")
    for row in results["decode"]:
        print(
            f"  {row['name']}: q={row['query_len']} k={row['key_len']} topk={row['topk']} backend={row['backend_used']} "
            f"baseline={row['baseline_ms']:.3f}ms triton={row['triton_ms']:.3f}ms "
            f"speedup={row['speedup']:.3f}x"
        )


@app.local_entrypoint()
def main(warmup: int = 5, iters: int = 30) -> None:
    results = run_benchmarks.remote(warmup=warmup, iters=iters)
    _print_results(results)

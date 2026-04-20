"""Modal tuning sweep for DSA selection building blocks."""

from pathlib import Path

import modal


DSA_ROOT = Path(__file__).resolve().parents[1]

app = modal.App("tma-thrust-dsa-tune-selection")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "triton")
    .add_local_dir(DSA_ROOT, remote_path="/root/dsa")
)


@app.function(image=image, gpu="B200:1", timeout=3600)
def run_tuning(iters: int = 20) -> dict:
    import math
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")

    from dsa.prefill.solution.triton.kernel import _weighted_query_kernel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    def make_inputs(query_len: int, key_len: int, num_index_heads: int = 8, index_dim: int = 32):
        batch_size = 1
        index_q = torch.randn(batch_size, query_len, num_index_heads, index_dim, device=device, dtype=dtype)
        index_k = torch.randn(batch_size, key_len, index_dim, device=device, dtype=dtype)
        index_weights = torch.randn(batch_size, query_len, num_index_heads, device=device, dtype=torch.float32)
        return {
            "index_q": index_q,
            "index_k": index_k,
            "index_weights": index_weights,
            "num_index_heads": num_index_heads,
            "index_dim": index_dim,
        }

    def _sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def _time(fn):
        _sync()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        _sync()
        end = time.perf_counter()
        return (end - start) * 1000.0 / iters

    def profile_case(name: str, inputs: dict) -> dict:
        batch_size, query_len, num_index_heads, index_dim = inputs["index_q"].shape
        rows = batch_size * query_len
        q_flat_bf16 = inputs["index_q"].reshape(rows, num_index_heads, index_dim).contiguous()
        w_flat_bf16 = inputs["index_weights"].to(dtype=inputs["index_q"].dtype).reshape(rows, num_index_heads).contiguous()
        q_flat_f32 = inputs["index_q"].float().reshape(rows, num_index_heads, index_dim).contiguous()
        w_flat_f32 = inputs["index_weights"].float().reshape(rows, num_index_heads).contiguous()
        index_k_t = inputs["index_k"].float().transpose(1, 2).contiguous()
        scale = 1.0 / math.sqrt(index_dim)

        def weighted_bmm_f32():
            w = w_flat_f32.reshape(rows, 1, num_index_heads)
            _ = torch.bmm(w, q_flat_f32)

        def weighted_elemwise_f32():
            _ = (inputs["index_q"].float() * inputs["index_weights"].float().unsqueeze(-1)).sum(dim=2)

        def score_bmm_f32():
            weighted = (inputs["index_q"].float() * inputs["index_weights"].float().unsqueeze(-1)).sum(dim=2)
            _ = torch.bmm(weighted, index_k_t) * scale

        measurements = []

        measurements.append(
            {"name": "weighted_bmm_f32", "latency_ms": _time(weighted_bmm_f32)}
        )
        measurements.append(
            {"name": "weighted_elemwise_f32", "latency_ms": _time(weighted_elemwise_f32)}
        )
        measurements.append(
            {"name": "score_bmm_f32", "latency_ms": _time(score_bmm_f32)}
        )

        for block_h in [8, 16]:
            for block_d in [32, 64, 128]:
                if block_h < num_index_heads or block_d < index_dim:
                    continue
                for num_warps in [1, 2, 4]:
                    for num_stages in [1, 2]:
                        out = torch.empty(rows, index_dim, device=device, dtype=torch.float32)

                        def weighted_kernel():
                            _weighted_query_kernel[(rows,)](
                                q_flat_f32,
                                w_flat_f32,
                                out,
                                q_flat_f32.stride(0),
                                q_flat_f32.stride(1),
                                q_flat_f32.stride(2),
                                w_flat_f32.stride(0),
                                out.stride(0),
                                num_index_heads,
                                index_dim,
                                BLOCK_H=block_h,
                                BLOCK_D=block_d,
                                num_warps=num_warps,
                                num_stages=num_stages,
                            )

                        measurements.append(
                            {
                                "name": f"weighted_kernel_h{block_h}_d{block_d}_w{num_warps}_s{num_stages}",
                                "latency_ms": _time(weighted_kernel),
                            }
                        )

        measurements.sort(key=lambda x: x["latency_ms"])
        return {
            "name": name,
            "best": measurements[0],
            "measurements": measurements,
        }

    torch.manual_seed(0)
    prefill_case = make_inputs(query_len=1024, key_len=1024)
    decode_case = make_inputs(query_len=1, key_len=8192)

    return {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "iters": iters,
        "prefill": profile_case("prefill_1024", prefill_case),
        "decode": profile_case("decode_8192", decode_case),
    }


def _print_case(case: dict) -> None:
    print(f"\n{case['name']}: best={case['best']['name']} {case['best']['latency_ms']:.3f}ms")
    for row in case["measurements"]:
        print(f"  {row['name']}: {row['latency_ms']:.3f}ms")


@app.local_entrypoint()
def main(iters: int = 20) -> None:
    results = run_tuning.remote(iters=iters)
    print(f"Device: {results['device']} ({results.get('gpu_name')})")
    print(f"Iters: {results['iters']}")
    _print_case(results["prefill"])
    _print_case(results["decode"])

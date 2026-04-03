"""Modal smoke test for the first DSA baseline."""

from pathlib import Path

import modal


DSA_ROOT = Path(__file__).resolve().parents[1]

app = modal.App("tma-thrust-dsa-smoke")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "triton")
    .add_local_dir(DSA_ROOT, remote_path="/root/dsa")
)


@app.function(image=image, gpu="B200:1", timeout=1800)
def run_smoke() -> dict:
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")

    from dsa.common.reference import build_causal_mask
    from dsa.decode.baseline.python.kernel import kernel as decode_kernel
    from dsa.decode.solution.triton.kernel import kernel as decode_triton_kernel
    from dsa.prefill.baseline.python.kernel import kernel as prefill_kernel
    from dsa.prefill.solution.triton.kernel import kernel as prefill_triton_kernel
    from dsa.tests.test_correctness import (
        test_decode_matches_naive,
        test_full_topk_recovers_dense_prefill,
        test_prefill_matches_naive,
    )

    # CPU correctness path for deterministic smoke coverage.
    test_prefill_matches_naive()
    test_decode_matches_naive()
    test_full_topk_recovers_dense_prefill()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    def make_inputs(query_len: int, key_len: int):
        batch_size = 1
        num_heads = 16
        qk_nope_head_dim = 32
        rope_dim = 16
        v_head_dim = 32
        kv_rank = 64
        num_index_heads = 8
        index_dim = 32
        topk = min(64, key_len)

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

    def time_ms(fn, args, *, topk, attn_mask, warmup: int = 5, iters: int = 20, **kwargs) -> float:
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

    torch.manual_seed(0)
    prefill_inputs = make_inputs(query_len=256, key_len=256)
    decode_inputs = make_inputs(query_len=1, key_len=2048)
    prefill_ref = prefill_kernel(
        *prefill_inputs[:8],
        topk=prefill_inputs[8],
        attn_mask=prefill_inputs[9],
    )
    prefill_sol = prefill_triton_kernel(
        *prefill_inputs[:8],
        topk=prefill_inputs[8],
        attn_mask=prefill_inputs[9],
        backend="auto",
    )
    torch.testing.assert_close(prefill_sol.float(), prefill_ref.float(), atol=2e-2, rtol=2e-2)
    decode_ref = decode_kernel(
        *decode_inputs[:8],
        topk=decode_inputs[8],
        attn_mask=decode_inputs[9],
    )
    decode_sol = decode_triton_kernel(
        *decode_inputs[:8],
        topk=decode_inputs[8],
        attn_mask=decode_inputs[9],
        backend="auto",
    )
    torch.testing.assert_close(decode_sol.float(), decode_ref.float(), atol=2e-2, rtol=2e-2)

    prefill_baseline_latency_ms = time_ms(
        prefill_kernel,
        prefill_inputs[:8],
        topk=prefill_inputs[8],
        attn_mask=prefill_inputs[9],
    )
    prefill_triton_latency_ms = time_ms(
        prefill_triton_kernel,
        prefill_inputs[:8],
        topk=prefill_inputs[8],
        attn_mask=prefill_inputs[9],
        backend="auto",
    )
    decode_latency_ms = time_ms(
        decode_kernel,
        decode_inputs[:8],
        topk=decode_inputs[8],
        attn_mask=decode_inputs[9],
    )
    decode_triton_latency_ms = time_ms(
        decode_triton_kernel,
        decode_inputs[:8],
        topk=decode_inputs[8],
        attn_mask=decode_inputs[9],
        backend="auto",
    )

    return {
        "status": "passed",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "prefill_baseline_latency_ms": prefill_baseline_latency_ms,
        "prefill_triton_latency_ms": prefill_triton_latency_ms,
        "prefill_speedup": prefill_baseline_latency_ms / prefill_triton_latency_ms,
        "decode_baseline_latency_ms": decode_latency_ms,
        "decode_triton_latency_ms": decode_triton_latency_ms,
        "decode_speedup": decode_latency_ms / decode_triton_latency_ms,
    }


@app.local_entrypoint()
def main() -> None:
    result = run_smoke.remote()
    print(result)

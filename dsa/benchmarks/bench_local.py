"""Local benchmark for the first DSA baseline."""

import argparse
import sys
import time
from pathlib import Path

import torch


def _add_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_add_repo_root()

from dsa.common.reference import build_causal_mask
from dsa.decode.baseline.python.kernel import kernel as decode_kernel
from dsa.prefill.baseline.python.kernel import kernel as prefill_kernel


def _time_ms(fn, *args, warmup: int, iters: int, device: str, **kwargs) -> float:
    for _ in range(warmup):
        fn(*args, **kwargs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def _make_inputs(
    *,
    batch_size: int,
    query_len: int,
    key_len: int,
    num_heads: int,
    qk_nope_head_dim: int,
    rope_dim: int,
    v_head_dim: int,
    kv_rank: int,
    num_index_heads: int,
    index_dim: int,
    topk: int,
    device: str,
):
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DSA v1 baseline")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-seq-len", type=int, default=256)
    parser.add_argument("--decode-cache-len", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--qk-nope-head-dim", type=int, default=64)
    parser.add_argument("--qk-rope-head-dim", type=int, default=32)
    parser.add_argument("--v-head-dim", type=int, default=64)
    parser.add_argument("--kv-rank", type=int, default=128)
    parser.add_argument("--index-heads", type=int, default=8)
    parser.add_argument("--index-dim", type=int, default=32)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    prefill_inputs = _make_inputs(
        batch_size=args.batch_size,
        query_len=args.prefill_seq_len,
        key_len=args.prefill_seq_len,
        num_heads=args.num_heads,
        qk_nope_head_dim=args.qk_nope_head_dim,
        rope_dim=args.qk_rope_head_dim,
        v_head_dim=args.v_head_dim,
        kv_rank=args.kv_rank,
        num_index_heads=args.index_heads,
        index_dim=args.index_dim,
        topk=args.topk,
        device=device,
    )
    decode_inputs = _make_inputs(
        batch_size=args.batch_size,
        query_len=1,
        key_len=args.decode_cache_len,
        num_heads=args.num_heads,
        qk_nope_head_dim=args.qk_nope_head_dim,
        rope_dim=args.qk_rope_head_dim,
        v_head_dim=args.v_head_dim,
        kv_rank=args.kv_rank,
        num_index_heads=args.index_heads,
        index_dim=args.index_dim,
        topk=args.topk,
        device=device,
    )

    prefill_ms = _time_ms(
        prefill_kernel,
        *prefill_inputs[:8],
        topk=prefill_inputs[8],
        attn_mask=prefill_inputs[9],
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    decode_ms = _time_ms(
        decode_kernel,
        *decode_inputs[:8],
        topk=decode_inputs[8],
        attn_mask=decode_inputs[9],
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )

    print(f"Prefill latency: {prefill_ms:.3f} ms")
    print(f"Decode latency:  {decode_ms:.3f} ms")


if __name__ == "__main__":
    main()

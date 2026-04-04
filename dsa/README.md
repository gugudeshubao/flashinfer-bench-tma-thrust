# DSA (DeepSeek Sparse Attention)

First working version of DeepSeek V3.2 sparse attention under `dsa/`, scoped to a correctness-first MLA-core baseline.

## Scope

- Shared token-level indexer reference
- Prefill sparse attention baseline
- Prefill Triton attention core for selected-token softmax/value aggregation
- Decode sparse attention baseline
- Stable `solution/triton` entrypoints for iterative kernel replacement
- Local correctness tests and a small local benchmark

Current status:

- `prefill/solution/triton` and `decode/solution/triton` now run a real Triton sparse-attention core over MLA latent tensors.
- Both paths are correctness-validated on Modal B200.
- `decode` is now ahead of baseline on the current Modal shapes.
- `prefill` uses adaptive dispatch: short and medium shapes can fall back to the reference path, while long-sequence shapes switch to Triton and show large gains.

## Operator Boundary

The kernel boundary is MLA-core, not full transformer-hidden-state projection:

- `q_nope`: query non-RoPE component
- `q_pe`: query RoPE component
- `compressed_kv`: MLA latent KV cache / compressed KV tensor
- `k_pe` / `k_pe_cache`: RoPE key cache
- `wkv_b`: MLA expansion matrix used to recover `k_nope` and `v`
- `index_q`, `index_k`, `index_weights`: token-level sparse indexer inputs

This matches the real DSA split more closely than a generic dense-attention API and keeps prefill/decode aligned with DeepSeek V3.2.

## Directory Layout

```text
dsa/
├── common/
│   ├── config.py
│   ├── indexer.py
│   └── reference.py
├── prefill/
│   ├── baseline/python/kernel.py
│   ├── solution/triton/kernel.py
│   └── config.toml
├── decode/
│   ├── baseline/python/kernel.py
│   ├── solution/triton/kernel.py
│   └── config.toml
├── benchmarks/
│   ├── bench_local.py
│   ├── bench_modal.py
│   └── profile_modal.py
├── tests/
│   └── test_correctness.py
├── trace_definitions/
└── docs/
```

## Quick Start

```bash
python3 dsa/tests/test_correctness.py
python3 dsa/benchmarks/bench_local.py --prefill-seq-len 256 --decode-cache-len 2048
modal run dsa/tests/test_modal.py
modal run dsa/benchmarks/bench_modal.py --warmup 3 --iters 10
modal run dsa/benchmarks/profile_modal.py --iters 20
```

## Notes

- The indexer is implemented as weighted token-level top-k selection.
- Prefill applies dense attention with a sparse additive mask over selected tokens.
- `solution/triton` accepts `backend="auto" | "reference" | "triton"` for controlled benchmarking and debugging.
- Decode uses the DeepSeek MLA cache form directly:
  - `q_nope @ wkv_b[:, :qk_nope]` against `compressed_kv`
  - `q_pe @ k_pe_cache`
  - sparse softmax over selected positions
  - latent aggregation followed by value projection with `wkv_b[:, qk_nope:]`

## Current Perf Snapshot

- Modal B200 smoke:
  - prefill: `1.050x` of baseline on the 256/64 smoke shape
  - decode: `1.181x` over baseline on the 2048/64 smoke shape
- Modal B200 benchmark:
  - prefill: `1.131x`, `0.981x`, `1.011x`, `2.438x`, `5.083x` on 256, 512, 1024, 2048, 4096 token cases
  - decode: `1.176x`, `1.291x`, `1.043x` on 2048, 4096, 8192 cache cases
- Stage profile on B200:
  - forced-`prefill_1024_128` Triton path: prepare `0.099ms`, select `0.253ms`, kernel `0.277ms`, project `0.066ms`
  - `decode_8192_128`: prepare `0.065ms`, select `0.139ms`, kernel `0.141ms`, project `0.045ms`

## Next Steps

- Push forced large-shape prefill Triton farther by replacing Torch-side `select/topk+mask`
- Fuse more of the prefill metadata path into Triton or CUDA
- Push decode Triton farther past baseline once the prefill path is settled
- Add block-sparse / paged-cache metadata instead of token-index masks
- Add FP8 indexer and FP8 KV-cache paths
- Add FlashInfer benchmark integration once the interface is fully frozen

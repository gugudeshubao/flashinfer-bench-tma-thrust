# Performance Tracking

Hardware: **NVIDIA B200** (sm100, 180 GB HBM3e) via Modal
Benchmark config (full): `--warmup 3 --iters 100 --trials 5`
Benchmark config (correctness): `--warmup 0 --iters 1 --trials 1`

---

## Latest: v1 — Python Baseline (2026-03-24)

Both kernels pass all workloads with `abs_err = 0.0`.
This is the pure-PyTorch reference translation — no performance optimization yet.

### Decode — `gdn_decode_qk4_v8_d128_k_last`

Config: num_q_heads=4, num_v_heads=8, head_size=128, state=[B,8,128,128] k-last

| batch_size | solution (ms) | ref (ms) | speedup |
|------------|--------------|----------|---------|
| 1 | — | — | ~1.10x |
| 2 | — | — | ~1.10x |
| 4 | — | — | ~1.10x |
| 8 | — | — | ~1.10x |
| 16 | — | — | ~1.10x |
| 32 | — | — | ~1.10x |
| 64 | — | — | ~1.10x |
| 128 | — | — | ~1.10x |
| 256 | — | — | ~1.10x |
| 512 | — | — | ~1.10x |

> Full per-workload numbers pending `--warmup 3 --iters 100` run.
> **10/10 PASSED · avg speedup ≈ 1.10x**

---

### Prefill — `gdn_prefill_qk4_v8_d128_k_last`

Config: num_q_heads=4, num_v_heads=8, head_size=128, state=[N,8,128,128] k-last, varlen

Benchmark: `--warmup 0 --iters 1 --trials 1` (Python loop too slow for 100 iters on large workloads)

| total_seq_len | num_seqs | solution (ms) | ref (ms) | speedup |
|---------------|----------|--------------|----------|---------|
| 64 | 1 | 13.68 | 13.89 | 1.01x |
| 128 | 1 | 24.72 | 25.17 | 1.02x |
| 256 | 1 | 46.62 | 46.32 | 0.99x |
| 512 | 1 | 106.38 | 84.78 | 0.80x |
| 1024 | 1 | 179.56 | 187.14 | 1.04x |
| 128 | 4 | 21.98 | 23.61 | 1.07x |
| 256 | 4 | 46.85 | 46.96 | 1.00x |
| 512 | 4 | 96.03 | 95.39 | 0.99x |
| 1024 | 4 | 210.04 | 182.90 | 0.87x |
| 2048 | 8 | 354.94 | 351.94 | 0.99x |
| 4096 | 8 | 730.75 | 714.08 | 0.98x |
| 8192 | 16 | 1466.77 | 1468.93 | 1.00x |

**12/12 PASSED · avg speedup ≈ 0.98x**

---

## Version History

| Version | Date | Decode | Prefill | Notes |
|---------|------|--------|---------|-------|
| v1 | 2026-03-24 | 10/10 ✅ ~1.10x | 12/12 ✅ ~0.98x | PyTorch reference translation |

---

## Roofline Targets (B200)

See [ROOFLINE.md](ROOFLINE.md) for arithmetic intensity analysis.

Key numbers for B200:
- Peak BF16 GEMM: ~2.25 PFLOPS (with WGMMA/TMA)
- Memory bandwidth: ~8 TB/s (HBM3e)
- Ridge point: ~281 FLOP/byte

### Decode arithmetic intensity

Each token/batch: `O(B × H × K × V)` compute, `O(B × H × K × V)` state read/write.
→ Compute-to-memory ratio ≈ 1–2 FLOP/byte → **heavily memory-bound**.
Target: maximize HBM bandwidth utilization with coalesced access patterns.

### Prefill arithmetic intensity

Sequential scan, `O(T × H × K²)` compute total, state size `O(H × K²)` per sequence.
→ Compute bound grows with T; for T≥512 shifts toward compute-bound.
Target: chunk the sequence (e.g., chunk=64) to amortize state load, fuse with matmul.

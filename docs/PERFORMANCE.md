# Performance Tracking

Hardware: **NVIDIA B200** (sm100, 180 GB HBM3e) via Modal
Full benchmark config: `--warmup 3 --iters 100 --trials 5`

---

## Latest: v2 — Triton Kernel (2026-03-27)

Grid (B, H=8) for decode · Grid (N, H=8) for prefill.
128×128 state lives in registers; no Python loop overhead.

### Decode — `gdn_decode_qk4_v8_d128_k_last`

| batch | solution (ms) | ref (ms) | speedup |
|-------|--------------|----------|---------|
| 1 | 0.0468 | 1.2075 | 25.82x |
| 2 | 0.0467 | 2.1688 | 46.41x |
| 4 | 0.0464 | 4.1326 | 89.03x |
| 8 | 0.0464 | 8.0070 | 172.45x |
| 16 | 0.0481 | 15.874 | 330.01x |
| 32 | 0.0474 | 31.549 | 666.09x |
| 64 | 0.0526 | 62.924 | 1195.72x |
| 128 | 0.0717 | 125.32 | 1746.76x |
| 256 | 0.1052 | 249.68 | 2373.88x |
| 512 | 0.1746 | 499.26 | 2859.44x |

**10/10 PASSED · avg speedup 950.56x**

### Prefill — `gdn_prefill_qk4_v8_d128_k_last`

| total_seq_len | num_seqs | solution (ms) | ref (ms) | speedup |
|---------------|----------|--------------|----------|---------|
| 64 | 1 | 0.1461 | 11.476 | 78.55x |
| 128 | 1 | 0.2532 | 22.416 | 88.52x |
| 256 | 1 | 0.4679 | 44.484 | 95.06x |
| 512 | 1 | 0.8952 | 88.870 | 99.28x |
| 1024 | 1 | 1.7508 | 175.52 | 100.25x |
| 128 | 4 | 0.0960 | 22.781 | 237.22x |
| 256 | 4 | 0.1501 | 44.865 | 298.86x |
| 512 | 4 | 0.2587 | 88.756 | 343.08x |
| 1024 | 4 | 0.4823 | 176.87 | 366.72x |
| 2048 | 8 | 0.4819 | 351.62 | 729.60x |
| 4096 | 8 | 0.9093 | 705.47 | 775.84x |
| 8192 | 16 | 0.9945 | 1418.1 | 1425.86x |

**12/12 PASSED · avg speedup 386.57x**

---

## v1 — Python Baseline (2026-03-24)

Pure-PyTorch reference translation. Correctness only, no performance.

### Decode

**10/10 PASSED · avg speedup ~1.10x**

### Prefill

`--warmup 0 --iters 1 --trials 1` (Python loop too slow for 100 iters on large workloads)

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

**12/12 PASSED · avg speedup ~0.98x**

---

## Version History

| Version | Date | Decode avg | Prefill avg | Notes |
|---------|------|-----------|-------------|-------|
| v2 | 2026-03-27 | **950.56x** | **386.57x** | Triton: fused delta-rule, state in registers |
| v1 | 2026-03-24 | ~1.10x | ~0.98x | PyTorch reference translation |

---

## Roofline Targets (B200)

See [ROOFLINE.md](ROOFLINE.md) for arithmetic intensity analysis.

Key numbers for B200:
- Peak BF16 GEMM: ~2.25 PFLOPS (with WGMMA/TMA)
- Memory bandwidth: ~8 TB/s (HBM3e)
- Ridge point: ~281 FLOP/byte

### v2 decode analysis

Decode at batch=512: 0.175ms, state I/O = 512×8×128²×4B = 268MB
Effective BW = 268MB / 0.175ms ≈ **1.53 TB/s** (19% of peak 8TB/s).
Room to improve: better memory access patterns, persistent kernel, TMA.

### v2 prefill analysis

Prefill (8192,16): 0.995ms, 16×8 = 128 parallel (seq,head) programs.
Per program handles up to 512 tokens with [128,128] state in registers.
Register pressure: 128×128×4B = 64KB per program.
Next: chunked matmul with tl.dot() for tensor-core utilization.

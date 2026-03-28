# Performance Tracking

Hardware: **NVIDIA B200** (sm100, 180 GB HBM3e) via Modal
Full benchmark config: `--warmup 3 --iters 100 --trials 5`

---

## Latest: v3 — Triton V-Split (2026-03-27)

Grid (B, H=8, V_BLOCKS=4) for decode · Grid (N, H=8, V_BLOCKS=4) for prefill.
V dimension split across 4 programs (BLOCK_V=32), 4× more SM occupancy.
State slice [32, 128] per program vs [128, 128] in v2.

### Decode — `gdn_decode_qk4_v8_d128_k_last`

| batch | solution (ms) | ref (ms) | speedup |
|-------|--------------|----------|---------|
| 1 | 0.0484 | 1.2853 | 26.55x |
| 2 | 0.0484 | 2.2652 | 46.82x |
| 4 | 0.0487 | 4.3647 | 89.57x |
| 8 | 0.0476 | 8.7567 | 183.93x |
| 16 | 0.0517 | 16.799 | 325.02x |
| 32 | 0.0483 | 33.442 | 692.35x |
| 64 | 0.0489 | 66.486 | 1359.66x |
| 128 | 0.0605 | 131.52 | 2172.74x |
| 256 | 0.0832 | 264.34 | 3177.75x |
| 512 | 0.1304 | 531.90 | 4079.21x |

**10/10 PASSED · avg speedup 1215.36x**

### Prefill — `gdn_prefill_qk4_v8_d128_k_last`

| total_seq_len | num_seqs | solution (ms) | ref (ms) | speedup |
|---------------|----------|--------------|----------|---------|
| 64 | 1 | 0.1024 | 10.454 | 102.07x |
| 128 | 1 | 0.1684 | 20.669 | 122.73x |
| 256 | 1 | 0.3020 | 40.798 | 135.11x |
| 512 | 1 | 0.5707 | 81.099 | 142.10x |
| 1024 | 1 | 1.1104 | 165.13 | 148.72x |
| 128 | 4 | 0.0751 | 21.836 | 290.85x |
| 256 | 4 | 0.1112 | 41.252 | 371.04x |
| 512 | 4 | 0.1770 | 82.064 | 463.73x |
| 1024 | 4 | 0.3221 | 165.00 | 512.20x |
| 2048 | 8 | 0.3380 | 349.18 | 1033.11x |
| 4096 | 8 | 0.6165 | 652.61 | 1058.61x |
| 8192 | 16 | 0.7732 | 1324.2 | 1712.63x |

**12/12 PASSED · avg speedup 507.74x**

---

## v2 — Triton Kernel (2026-03-27)

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
| v3 | 2026-03-27 | **1215.36x** | **507.74x** | Triton V-split: BLOCK_V=32, 4× programs, 4× smaller state |
| v2 | 2026-03-27 | 950.56x | 386.57x | Triton: fused delta-rule, state in registers |
| v1 | 2026-03-24 | ~1.10x | ~0.98x | PyTorch reference translation |

---

## Roofline Targets (B200)

See [ROOFLINE.md](ROOFLINE.md) for arithmetic intensity analysis.

Key numbers for B200:
- Peak BF16 GEMM: ~2.25 PFLOPS (with WGMMA/TMA)
- Memory bandwidth: ~8 TB/s (HBM3e)
- Ridge point: ~281 FLOP/byte

### v3 decode analysis

Decode at batch=512: 0.130ms, state I/O = 512×8×128²×4B = 268MB
Effective BW = 268MB / 0.130ms ≈ **2.06 TB/s** (26% of peak 8TB/s).
V-split reduces per-program register pressure: 32×128×4B = 16KB vs 64KB in v2.
Room to improve: persistent kernel across batch, TMA bulk load/store.

### v3 prefill analysis

Prefill (8192,16): 0.773ms, 16×8×4 = 512 parallel programs (vs 128 in v2).
Register pressure per program: 32×128×4B = 16KB (vs 64KB in v2).
Single-seq workloads: 4 programs per head (vs 1), ~1.4× improvement observed.
Next: tl.dot() for tensor-core utilization in the inner token loop.

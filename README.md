# FlashInfer-GatedDelta: TMA Thrust Submission

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contest](https://img.shields.io/badge/MLSys-2026-red)](https://mlsys.org/)
[![GPU](https://img.shields.io/badge/NVIDIA-B200-green)](https://www.nvidia.com/)

**Team**: TMA Thrust  
**Track**: C — Gated Delta Net  
**Hardware**: NVIDIA B200 (Blackwell, sm_100) — 8 TB/s HBM3e, 2.25 PFLOPS BF16  
**Kernels**: `gdn_decode_qk4_v8_d128_k_last` · `gdn_prefill_qk4_v8_d128_k_last`

---

## Performance Summary (Verified Correct)

| Batch | Triton v5 | **CUDA v9** | Winner | Speedup |
|-------|-----------|-------------|--------|---------|
| **1** | 24 GB/s | **27 GB/s** | **v9** | **1.11x** |
| **16** | 386 GB/s | **405 GB/s** | **v9** | **1.05x** |
| 64 | **1,518 GB/s** | 1,302 GB/s | Triton | 1.17x |
| **256** | 2,834 GB/s | **7,585 GB/s** | **v9** | **2.68x** |

**Best: CUDA v9 (CuTe swizzle) achieves 7,600 GB/s (95% of B200 peak)**

All kernels verified for correctness against Triton v5 baseline.

> Full benchmark data → **[docs/PERFORMANCE.md](docs/PERFORMANCE.md)**  
> Optimization roadmap → **[docs/ROADMAP.md](docs/ROADMAP.md)**

---

## Quick Start

### 1. Setup Modal volume

```bash
modal run scripts/setup_volume.py
# or download from HuggingFace:
modal run scripts/setup_volume.py --mode hf
```

### 2. Run benchmarks

```bash
# Benchmark all versions (v5, v6, v7, v8)
modal run scripts/bench_all_versions.py --versions all --batches "1,16,64,256"

# Test specific version
modal run scripts/bench_all_versions.py --versions v7 --batches "256,512"

# Original benchmark (vs Python reference)
modal run benchmarks/bench_modal.py --kernel both
```

### 3. Build CUDA kernels

```bash
modal run scripts/build_cuda.py  # Compiles v5-v8 for sm_100
```

---

## Repository Structure

```
.
├── src/kernels/                              # Kernel implementations
│   ├── cuda/                                 # Basic CUDA (v5-v8)
│   │   ├── gdn_decode_v5.cuh                 # Baseline
│   │   ├── gdn_decode_v6.cuh                 # TMA async
│   │   ├── gdn_decode_v7.cuh                 # float4 + FP4
│   │   ├── gdn_decode_v8.cuh                 # Warp spec + FP8
│   │   └── gdn_prefill_v5-v8.cuh
│   ├── cute/                                 # CuTe DSL (v9-v10)
│   │   ├── gdn_decode_v9.cuh                 # SMEM swizzle
│   │   └── gdn_decode_v10.cuh                # Swizzle<3,3,3>
│   └── triton/                               # Triton baseline (symlinks)
├── gdn_decode_qk4_v8_d128_k_last/
│   ├── solution/triton/kernel.py             # Production Triton v5
│   └── baseline/triton/kernel.py             # Python reference
├── gdn_prefill_qk4_v8_d128_k_last/
│   ├── solution/triton/kernel.py
│   └── baseline/triton/kernel.py
├── scripts/
│   ├── bench_cuda_real.py                    # Correctness + benchmark
│   ├── build_cuda.py                         # CUDA compilation
│   └── setup_volume.py                       # Modal volume setup
├── benchmarks/bench_modal.py                 # Contest benchmark runner
└── docs/
    ├── PERFORMANCE.md                        # Performance tracking
    └── ROADMAP.md                            # Optimization history
```

---

## Kernel Versions

| Ver | Framework | Key Feature | Peak BW | Best For |
|-----|-----------|-------------|---------|----------|
| v5 | Triton | Auto-tuning | 1,518 GB/s | Batch=64 |
| v6 | CUDA | TMA async | ~1,500 GB/s | - |
| v7 | CUDA | float4 + FP4 | 7,578 GB/s | Batch=256 |
| v8 | CUDA | Warp spec + FP8 | 7,605 GB/s | Batch=256 |
| **v9** | **CuTe** | **SMEM swizzle** | **7,585 GB/s** | **Batch=1,16,256** |
| v10 | CuTe | Swizzle<3,3,3> | 7,602 GB/s | Batch=256 |

### When does CUDA help?

- **Small batch (1-16)**: v9 CuTe swizzle → **1.05-1.11x** over Triton
- **Medium batch (64)**: Triton wins (auto-tuning)
- **Large batch (256+)**: All CUDA → **2.68x** speedup

---

## Algorithm — Gated Delta Net

State layout: **k-last** `[B, H, V, K]` where H=8, V=K=128

```
g    = exp(-exp(A_log) * softplus(a + dt_bias))   # decay gate
beta = sigmoid(b)                                   # update gate

S     = g * S                             # apply decay
old_v = k @ S                             # [K] × [K,V] → [V]
new_v = beta * v + (1-beta) * old_v       # weighted merge
S     = S + outer(k, new_v - old_v)       # delta rule
o     = scale * q @ S                     # output
```

**GVA**: 4 Q-heads → 8 V-heads (`qk_h = h // 2`)

---

## Key Findings

### WGMMA Not Applicable for Decode

Blackwell WGMMA (Tensor Cores) requires matrix-matrix multiplication.  
GDN decode performs matrix-vector: `S@q` = [128×128] × [128] → [128]

**Decode**: Optimize memory bandwidth (achieved 95% of 8 TB/s peak).  
**Prefill**: Can use WGMMA with chunked algorithm (mat-mat possible).

### B200 Hardware Utilization

| Resource | Peak | GDN Decode | Utilization |
|----------|------|------------|-------------|
| HBM3e BW | 8 TB/s | 7.6 TB/s | **95%** |
| FP32 | 74.45 TFLOPS | 5.7 TFLOPS | 7.6% |
| BF16 Tensor | 2.25 PFLOPS | N/A | 0% (mat-vec) |

### Memory-Bound Analysis

| Batch | State Size | B200 Peak | Best Kernel | Achieved | Utilization |
|-------|------------|-----------|-------------|----------|-------------|
| 1 | 0.5 MB | 8,000 GB/s | CUDA v9 | 27 GB/s | 0.3% |
| 64 | 32 MB | 8,000 GB/s | Triton v5 | 1,518 GB/s | 19% |
| 256 | 128 MB | 8,000 GB/s | **CUDA v9** | **7,585 GB/s** | **95%** |

---

## Contest Info

- **Deadline**: April 24, 2026 (11:59 PM AoE)
- **Metric**: Arithmetic mean speedup over reference
- **Leaderboard**: https://bench.flashinfer.ai/
- **Reference**: https://mlsys26.flashinfer.ai/

# FlashInfer-GatedDelta: TMA Thrust Submission

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contest](https://img.shields.io/badge/MLSys-2026-red)](https://mlsys.org/)
[![GPU](https://img.shields.io/badge/NVIDIA-B200-green)](https://www.nvidia.com/)

**Team**: TMA Thrust
**Track**: C — Gated Delta Net
**Hardware**: NVIDIA B200 (sm100) via Modal
**Kernels**: `gdn_decode_qk4_v8_d128_k_last` · `gdn_prefill_qk4_v8_d128_k_last`

## 📊 Latest Performance

> Full numbers and version history → **[docs/PERFORMANCE.md](docs/PERFORMANCE.md)**

| Kernel | Workloads | Avg Speedup vs Ref | Version | Date |
|--------|-----------|-------------------|---------|------|
| decode | 10/10 ✅ | **~736x** | v5 CUDA | 2026-03-28 |
| prefill | 12/12 ✅ | **~457x** | v5 CUDA | 2026-03-28 |

## Quick Start

### 1. Setup Modal volume (first time)

```bash
modal run scripts/setup_volume.py
# or download from HuggingFace:
modal run scripts/setup_volume.py --mode hf
```

### 2. Run benchmarks on B200

```bash
# Correctness check (fast)
modal run benchmarks/bench_modal.py --kernel both --warmup 0 --iters 1 --trials 1

# Full benchmark
modal run benchmarks/bench_modal.py --kernel both

# CUDA v5 kernels
modal run benchmarks/bench_modal.py --kernel both --cuda

# Compare solution vs Python baseline
modal run benchmarks/bench_modal.py --kernel both --compare
```

## Repository Structure

```
.
├── src/kernels/                          # CUDA kernel sources
│   ├── gdn_decode_v5.cuh                 # 319 lines — decode CUDA kernel
│   └── gdn_prefill_v5.cuh                # 249 lines — prefill CUDA kernel
├── gdn_decode_qk4_v8_d128_k_last/
│   ├── solution/
│   │   ├── triton/kernel.py              # 135 lines — Triton v4
│   │   └── cuda/kernel.py                # 247 lines — CUDA wrapper + Triton fallback
│   └── baseline/triton/kernel.py         # Python reference
├── gdn_prefill_qk4_v8_d128_k_last/
│   ├── solution/
│   │   ├── triton/kernel.py              # 147 lines — Triton v4
│   │   └── cuda/kernel.py                # 255 lines — CUDA wrapper + Triton fallback
│   └── baseline/triton/kernel.py         # Python reference
├── flashinfer_trace/definitions/gdn/     # kernel definitions
├── scripts/setup_volume.py               # Modal volume setup
├── benchmarks/bench_modal.py             # benchmark runner
└── docs/
    ├── PERFORMANCE.md                    # performance tracking
    └── ROOFLINE.md                       # roofline analysis
```

## Kernel Implementation Summary

### Code Statistics

| Type | File | Lines | Description |
|------|------|-------|-------------|
| CUDA | `gdn_decode_v5.cuh` | 319 | Shared memory, warp shuffles |
| CUDA | `gdn_prefill_v5.cuh` | 249 | Sequential token scan |
| Triton | decode `kernel.py` | 135 | Adaptive BLOCK_V |
| Triton | prefill `kernel.py` | 147 | Adaptive BLOCK_V |
| Wrapper | decode `cuda/kernel.py` | 247 | JIT + Triton fallback |
| Wrapper | prefill `cuda/kernel.py` | 255 | JIT + Triton fallback |
| **Total** | | **1,352** | |

### Key Optimizations

- **Adaptive BLOCK_V**: 16/32/64 based on batch size for optimal SM occupancy
- **Shared memory**: State tiles [BLOCK_V × D], Q/K vectors
- **Warp-level reductions**: `__shfl_xor_sync` for fast dot products
- **GVA support**: 4 Q-heads → 8 V-heads mapping (`qk_h = h // 2`)
- **Triton fallback**: CUDA JIT blocked by flashinfer-bench sandbox

## Algorithm — Gated Delta Net

State layout: **k-last** `[B/N, H, V, K]`

```
g    = exp(-exp(A_log) * softplus(a + dt_bias))   # decay gate ∈ (0,1)
beta = sigmoid(b)                                   # update gate ∈ (0,1)

S     = g * S                             # apply decay
old_v = k @ S                             # [K] × [K,V] → [V]
new_v = beta * v + (1-beta) * old_v      # weighted merge
S     = S + outer(k, new_v - old_v)      # delta rule update
o     = scale * q @ S                    # output
```

**GVA**: `num_v_heads=8 > num_q_heads=4`, Q/K expanded by `repeat_interleave(2, dim=1)`.

## Optimization Roadmap

| Version | Status | Description | File |
|---------|--------|-------------|------|
| v1 | ✅ done | PyTorch baseline — correctness verified | `baseline/triton/kernel.py` |
| v2 | ✅ done | Triton: fused loop, full state tile [D,D] in registers | `kernel_v2.py` |
| v3 | ✅ done | Triton: V-split (BLOCK_V=32), 4× programs | `kernel_v3.py` |
| v4 | ✅ done | Triton: adaptive BLOCK_V based on batch/seq size | `solution/triton/kernel.py` |
| v5 | ✅ done | CUDA: shared memory, warp shuffles, Triton fallback | `solution/cuda/kernel.py` + `src/kernels/*.cuh` |
| v6 | ⏳ | CUDA: WGMMA + TMA on B200 (sm100) | planned |

## Contest Info

- **Deadline**: April 24, 2026 (11:59 PM AoE)
- **Metric**: Arithmetic mean speedup over reference implementation
- **Reference**: https://mlsys26.flashinfer.ai/

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
| decode | 10/10 ✅ | **950x** | v2 Triton | 2026-03-27 |
| prefill | 12/12 ✅ | **387x** | v2 Triton | 2026-03-27 |

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

# Compare solution vs Python baseline
modal run benchmarks/bench_modal.py --kernel both --compare
```

## Repository Structure

```
.
├── gdn_decode_qk4_v8_d128_k_last/
│   ├── solution/triton/kernel.py     # active solution (submit this)
│   └── baseline/triton/kernel.py    # Python reference (fixed, for comparison)
├── gdn_prefill_qk4_v8_d128_k_last/
│   ├── solution/triton/kernel.py     # active solution
│   └── baseline/triton/kernel.py    # Python reference
├── flashinfer_trace/definitions/gdn/ # kernel definitions
├── scripts/setup_volume.py           # Modal volume setup
├── benchmarks/bench_modal.py         # benchmark runner
└── docs/
    ├── PERFORMANCE.md                # ← performance tracking (updated each version)
    └── ROOFLINE.md                   # roofline analysis
```

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

| Version | Status | Description |
|---------|--------|-------------|
| v1 | ✅ done | PyTorch baseline — correctness verified, decode 10/10, prefill 12/12 |
| v2 | 🚧 | Triton: fused loop, batched matmul, no Python overhead |
| v3 | ⏳ | CUDA: WGMMA + TMA on B200 (sm100), chunked prefill |

## Contest Info

- **Deadline**: April 24, 2026 (11:59 PM AoE)
- **Metric**: Arithmetic mean speedup over reference implementation
- **Reference**: https://mlsys26.flashinfer.ai/

# FlashInfer-GatedDelta: TMA Thrust Submission

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contest](https://img.shields.io/badge/MLSys-2026-red)](https://mlsys.org/)
[![GPU](https://img.shields.io/badge/NVIDIA-B200-green)](https://www.nvidia.com/)

**Team**: TMA Thrust (Independent Researcher)
**Track**: C — Gated Delta Net (Track C)
**Hardware Target**: NVIDIA B200 (sm100) via Modal
**Kernels**: `gdn_decode_qk4_v8_d128_k_last` · `gdn_prefill_qk4_v8_d128_k_last`

## Quick Start

### 1. Setup Modal volume (first time)

```bash
# Create volume and upload synthetic workloads
modal run scripts/setup_volume.py

# Or download from HuggingFace
modal run scripts/setup_volume.py --mode hf
```

### 2. Run benchmarks on B200

```bash
# Both kernels
modal run benchmarks/bench_modal.py

# Single kernel
modal run benchmarks/bench_modal.py --kernel decode
modal run benchmarks/bench_modal.py --kernel prefill
```

## Repository Structure

```
.
├── gdn_decode_qk4_v8_d128_k_last/     # Decode kernel (Track C)
│   ├── config.toml
│   ├── solution/triton/kernel.py       # Implementation
│   └── scripts/pack_solution.py
├── gdn_prefill_qk4_v8_d128_k_last/    # Prefill kernel (Track C)
│   ├── config.toml
│   ├── solution/triton/kernel.py       # Implementation
│   └── scripts/pack_solution.py
├── flashinfer_trace/definitions/gdn/   # Kernel definitions (from contest)
├── scripts/
│   └── setup_volume.py                 # Modal volume setup
├── benchmarks/
│   └── bench_modal.py                  # Main benchmark runner
└── src/kernels/                        # Future CUDA/TMA kernels
```

## Algorithm — Gated Delta Net

State layout: **k-last** `[B, H, V, K]`

```
g    = exp(-exp(A_log) * softplus(a + dt_bias))   # decay gate
beta = sigmoid(b)                                   # update gate

S     = g * S                             # apply decay
old_v = k @ S                             # [K] × [K,V] → [V]
new_v = beta * v + (1-beta) * old_v      # weighted merge
S     = S + outer(k, new_v - old_v)      # delta rule update
o     = scale * q @ S                    # output projection
```

GVA (Grouped Value Attention): `num_v_heads=8 > num_q_heads=4`, Q/K expanded by `repeat_interleave(2)`.

## Development Roadmap

| Stage | Status | Description |
|-------|--------|-------------|
| v1 | ✅ | PyTorch baseline (correctness verified) |
| v2 | 🚧 | Triton kernel (batched matmul, no Python loop) |
| v3 | ⏳ | CUDA/WGMMA on B200 (TMA + Blackwell instructions) |

## Contest Info

- **Submission deadline**: April 24, 2026 (11:59 PM AoE)
- **Evaluation**: Biweekly on bare-metal B200, final on locked-clock B200
- **Metric**: Arithmetic mean speedup over reference implementation
- **Reference**: https://mlsys26.flashinfer.ai/

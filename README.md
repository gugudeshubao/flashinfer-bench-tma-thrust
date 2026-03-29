# FlashInfer Kernel Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contest](https://img.shields.io/badge/MLSys-2026-red)](https://mlsys.org/)
[![GPU](https://img.shields.io/badge/NVIDIA-B200-green)](https://www.nvidia.com/)

**Team**: TMA Thrust  
**Hardware**: NVIDIA B200 (Blackwell, sm_100) — 8 TB/s HBM3e, 2.25 PFLOPS BF16  

---

## Operators

| Operator | Description | Status |
|----------|-------------|--------|
| **[GDN](gdn/)** | Gated Delta Net attention | ✅ Production |
| **[MoE](moe/)** | FP8 Fused Mixture of Experts (DeepSeek-V3/R1) | 🚧 In Progress |

---

## GDN Performance Summary

| Kernel | Avg Speedup | Best Speedup | Status |
|--------|-------------|--------------|--------|
| Decode | 1127x | 3465x | ✅ ALL PASS |
| Prefill | 598x | 1886x | ✅ ALL PASS |

**Best: CuTe v9/v10 achieves 7,600 GB/s (95% of B200 peak)**

> Full benchmark data → **[gdn/docs/PERFORMANCE.md](gdn/docs/PERFORMANCE.md)**  
> Optimization roadmap → **[gdn/docs/ROADMAP.md](gdn/docs/ROADMAP.md)**

---

## Quick Start

### 1. Setup Modal volume

```bash
modal run scripts/setup_volume.py
```

### 2. Run GDN benchmarks

```bash
# Full benchmark (decode + prefill)
modal run gdn/benchmarks/bench_modal.py

# Decode only
modal run gdn/benchmarks/bench_modal.py --kernel decode

# Prefill only
modal run gdn/benchmarks/bench_modal.py --kernel prefill

# Correctness tests
modal run gdn/tests/test_correctness.py
```

---

## Repository Structure

```
.
├── gdn/                              # Gated Delta Net kernels
│   ├── decode/                       # Decode kernel (Triton solution)
│   ├── prefill/                      # Prefill kernel (Triton solution)
│   ├── kernels/                      # CUDA/CuTe/PTX implementations
│   │   ├── cuda/                     # Raw CUDA v5-v8
│   │   ├── cute_cpp/                 # CuTe C++ v9-v10
│   │   └── ptx/                      # PTX assembly
│   ├── scripts/                      # GDN-specific scripts
│   ├── benchmarks/                   # GDN benchmarks
│   ├── tests/                        # Correctness tests
│   ├── docs/                         # Documentation
│   └── README.md                     # GDN documentation
├── moe/                              # FP8 Fused MoE (DeepSeek-V3/R1)
│   ├── config.toml                   # Solution configuration
│   ├── solution/triton/              # Triton kernel implementation
│   ├── trace_definitions/            # Kernel definition JSON
│   ├── scripts/                      # MoE setup scripts
│   ├── benchmarks/                   # MoE benchmarks
│   └── README.md                     # MoE documentation
├── scripts/                          # Shared utility scripts
│   └── setup_volume.py               # Modal volume setup
├── CMakeLists.txt                    # CUDA build configuration
└── README.md                         # This file
```

---

## Kernel Versions (GDN)

| Ver | Framework | Key Feature | Peak BW | Best For |
|-----|-----------|-------------|---------|----------|
| v5 | Triton | Production baseline | 2,834 GB/s | All batches |
| v9 | CuTe C++ | SMEM swizzle | 7,585 GB/s | Batch=1,16,256 |
| v10 | CuTe C++ | BF16/FP8/FP4 state | 7,602 GB/s | Batch=256 |
| PTX | PTX Assembly | mma.sync, TMA | TBD | Prefill |

---

## Key Technical Findings

### Decode: Memory-Bound (Cannot Use Tensor Core)

- Operation: `S @ q` = [128×128] × [128] → [128] (matrix-vector)
- Arithmetic Intensity: AI = 1 FLOP/byte
- **Optimization**: Maximize HBM bandwidth via SMEM swizzle, state quantization

### Prefill: Can Use Tensor Core (with Chunking)

- Operation: `State @ Q_chunk` = [V×D] × [D×C] → [V×C] (matrix-matrix)
- Arithmetic Intensity: AI ≈ 8 FLOP/byte with CHUNK_SIZE=8
- **Optimization**: mma.sync.aligned.m16n8k16 for Tensor Core

### B200 Hardware Utilization

| Resource | Peak | GDN Decode (v10) | Utilization |
|----------|------|------------------|-------------|
| HBM3e BW | 8 TB/s | 7.6 TB/s | **95%** |
| BF16 Tensor | 2.25 PFLOPS | N/A | 0% (mat-vec) |

---

## Contest Info

- **Track C**: Gated Delta Net
- **Deadline**: April 24, 2026 (11:59 PM AoE)
- **Metric**: Arithmetic mean speedup over reference
- **Leaderboard**: https://bench.flashinfer.ai/

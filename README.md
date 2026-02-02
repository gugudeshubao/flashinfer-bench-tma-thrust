# FlashInfer-GatedDelta: TMA Thrust Submission

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contest](https://img.shields.io/badge/MLSys-2026-red)](https://mlsys.org/)
[![GPU](https://img.shields.io/badge/NVIDIA-B200-green)](https://www.nvidia.com/)

**Team**: TMA Thrust (Independent Researcher)  
**Track**: Gated Delta Net Optimization  
**Hardware Target**: NVIDIA Blackwell (B200) via Modal  

## Overview

This repository contains optimized CUDA kernels for **Gated DeltaNet** attention mechanism, submitted to the [MLSys 2026 FlashInfer-Bench Contest](https://mlsys.org/) (NVIDIA Track).

Gated DeltaNet combines:
- **Gating mechanism** (rapid memory erasure via decay factor α)
- **Delta rule** (selective memory update via β coefficients)

Our implementation targets the NVIDIA B200 architecture, utilizing Tensor Memory Accelerator (TMA) and Warp Group MMA (WGMMA) instructions for peak performance.

## Current Status

Following our FlashAttention roadmap:

- ✅ **Stage 0**: CPU/CUDA naive baseline completed
- ✅ **Stage 1**: Roofline analysis on Ampere (A100) completed
- 🚧 **Stage 2**: Hopper TMA/WGMMA migration (in progress)
- ⏳ **Stage 3**: Blackwell TCgen05.mma optimization (target)

## Repository Structure

```text
.
├── src/
│   ├── kernels/          # CUDA kernel implementations
│   │   ├── gated_delta_fwd.cu      # Forward kernel (TMA + WGMMA)
│   │   └── gated_delta_bwd.cu      # Backward kernel (future work)
│   ├── utils/            # Helper functions (memory management, timing)
│   └── third_party/      # FlashInfer headers (submodule)
├── benchmarks/
│   ├── bench_modal.py    # Modal cloud benchmarking script
│   └── sweep_configs/    # JSON configs for hyperparameter sweep
├── tests/
│   ├── test_correctness.py    # Numerical accuracy vs reference
│   └── test_roofline.py       # Memory/compute bound analysis
├── docs/
│   ├── ROOFLINE.md       # Roofline model documentation
│   └── TECHNICAL_REPORT.md    # 4-page contest submission (WIP)
└── scripts/
    ├── setup_modal.sh    # Environment setup for Modal B200
    └── build.sh          # NVCC compilation flags
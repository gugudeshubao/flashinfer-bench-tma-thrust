# MoE (Mixture of Experts) Kernels

FP8 Mixture of Experts kernel implementations for NVIDIA B200 (Blackwell, sm_100).

## Directory Structure

```
moe/
├── kernels/             # CUDA/Triton/PTX implementations
├── scripts/             # MoE-specific scripts
├── benchmarks/          # Modal benchmark runners
├── tests/               # Correctness tests
└── docs/                # Documentation
```

## Quick Start

```bash
# Setup Modal volume (shared)
modal run scripts/setup_volume.py

# Run MoE benchmarks
modal run moe/benchmarks/bench_modal.py

# Run correctness tests
modal run moe/tests/test_correctness.py
```

## Target Hardware

- **GPU**: NVIDIA B200 (Blackwell, sm_100)
- **Memory**: 8 TB/s HBM3e
- **Compute**: 2.25 PFLOPS BF16, FP8 Tensor Core

## Key Optimizations (TODO)

- [ ] FP8 quantization (E4M3/E5M2)
- [ ] Expert routing optimization
- [ ] Token-to-expert load balancing
- [ ] TMA bulk memory operations
- [ ] Tensor Core mma.sync for matmul

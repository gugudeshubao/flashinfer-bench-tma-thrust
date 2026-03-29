# GDN (Gated Delta Net) Kernels

CUDA/Triton/PTX implementations of Gated Delta Net attention kernels for NVIDIA B200 (Blackwell, sm_100).

## Directory Structure

```
gdn/
├── decode/              # Decode kernel (single-token generation)
│   ├── baseline/triton/ # Python reference
│   └── solution/triton/ # Optimized Triton kernel
├── prefill/             # Prefill kernel (sequence processing)
│   ├── baseline/triton/
│   └── solution/triton/
├── kernels/             # CUDA/CuTe/PTX implementations
│   ├── cuda/            # Raw CUDA v5-v8
│   ├── cute_cpp/        # CuTe C++ v9-v10
│   └── ptx/             # PTX assembly
├── scripts/             # GDN-specific scripts
│   ├── build_cuda.py    # CUDA compilation
│   ├── bench_*.py       # Various benchmarks
│   └── test_*.py        # Various tests
├── benchmarks/          # Modal benchmark runners
├── tests/               # Correctness tests
├── docs/                # Documentation
├── trace_definitions/   # FlashInfer trace definitions
└── gdn_kernels.cu       # CUDA compilation wrapper
```

## Quick Start

```bash
# Run decode benchmark
modal run gdn/benchmarks/bench_modal.py --kernel decode

# Run prefill benchmark
modal run gdn/benchmarks/bench_modal.py --kernel prefill

# Run correctness tests
modal run gdn/tests/test_correctness.py

# Build CUDA kernels
modal run gdn/scripts/build_cuda.py
```

## Performance (B200)

| Kernel | Avg Speedup | Best Speedup | Status |
|--------|-------------|--------------|--------|
| Decode | 1127x | 3465x | ✅ ALL PASS |
| Prefill | 598x | 1886x | ✅ ALL PASS |

See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for detailed benchmarks.

## Kernel Versions

| Version | Backend | Key Feature |
|---------|---------|-------------|
| v5 | Triton | Production baseline |
| v9 | CuTe C++ | SMEM swizzle, cp.async |
| v10 | CuTe C++ | BF16/FP8/FP4 state quantization |
| PTX | PTX Assembly | mma.sync.aligned, TMA |

See [docs/ROADMAP.md](docs/ROADMAP.md) for full version history.

# Kernel Sources

This directory contains raw kernel source experiments.

It is not the production source of truth by itself.
For current measured status, use:

- [docs/PERFORMANCE.md](../docs/PERFORMANCE.md)
- [benchmarks/bench_modal.py](../benchmarks/bench_modal.py)

## Directory Roles

```text
kernels/
├── cuda/       # Raw CUDA kernels
├── cute_cpp/   # CuTe C++ / CUTLASS kernels
├── cute_dsl/   # CuTe DSL experiments
├── cutile/     # cuTile experiments
├── ptx/        # Inline PTX experiments
└── triton/     # Older Triton-side experiments
```

## Current Interpretation

- Decode:
  raw kernel peak currently comes from the `v9/v10` family, especially `v10 CuTe` at large batch.
- Prefill:
  the source files in this directory are useful research material, but the current CuTe/tcgen chase path is mainly driven by
  [prefill/solution/cuda/chunked_proto.py](../prefill/solution/cuda/chunked_proto.py),
  not by a standalone kernel in `kernels/cute_cpp/`.

## What To Trust

- Trust `bench_modal.py` for production-facing decisions.
- Trust `scripts/bench_cuda_real.py` for decode raw kernel ranking.
- Trust `scripts/bench_prefill_tensorcore.py` for prefill CuTe/tcgen progress.

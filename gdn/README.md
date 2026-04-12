# GDN Kernels

Gated Delta Net kernels for NVIDIA B200.

This repo currently has:

- official production-facing benchmark paths
- raw decode kernel microbenchmarks
- CuTe / tcgen prefill research paths

The current performance source of truth is [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Current Status

### Decode

- Official engineering baseline: Triton
- Official workload result on B200: `473.38x` average speedup
- Packaged decode competition candidates:
  - `v10 CuTe`: `470.84x` average speedup on the official 54-workload suite
  - `v10 TMA`: `440.17x` average speedup on the official 54-workload suite
- Competition decode candidate: `v10 CuTe / v10 TMA`
- Current representative candidate metric: `0.0503 / 0.0505 ms`, `5333 / 5316 GB/s`, about `2.17x` vs Triton at `B=256`

### Prefill

- Official engineering baseline: Triton
- Official workload result on B200: `257.45x` average speedup
- Packaged CUDA wrapper: `180.12x` average speedup
- Best CuTe-based chase path: `prefill/solution/cuda/chunked_proto.py`
- Current representative CuTe chase metric: `0.395 ms` at `N=4,L=32`, about `186.63x` vs reference there, but still only `0.07x` vs Triton
- Important: the CuTe/tcgen prefill line is still slower than Triton on the official workload benchmark

## Terminology

- Official baseline / official reference:
  the reference implementation used by `flashinfer-bench`
- Triton baseline:
  our current engineering baseline inside this repo

Official workload `speedup` numbers are against the official reference, not against Triton.

## Main Paths

### Production Paths

- Decode Triton:
  [decode/solution/triton/kernel.py](decode/solution/triton/kernel.py)
- Prefill Triton:
  [prefill/solution/triton/kernel.py](prefill/solution/triton/kernel.py)
- Official benchmark harness:
  [benchmarks/bench_modal.py](benchmarks/bench_modal.py)

### Decode Research Paths

- Packaged CUDA wrapper:
  [decode/solution/cuda/kernel.py](decode/solution/cuda/kernel.py)
- Real decode kernel comparison:
  [scripts/bench_cuda_real.py](scripts/bench_cuda_real.py)

### Prefill Research Paths

- Packaged CUDA wrapper:
  [prefill/solution/cuda/kernel.py](prefill/solution/cuda/kernel.py)
- Current CuTe/tcgen chaser:
  [prefill/solution/cuda/chunked_proto.py](prefill/solution/cuda/chunked_proto.py)
- Prefill tensor-core benchmark harness:
  [scripts/bench_prefill_tensorcore.py](scripts/bench_prefill_tensorcore.py)

## Quick Start

```bash
# Official workload benchmarks
modal run benchmarks/bench_modal.py --kernel decode --warmup 1 --iters 5 --trials 1
modal run benchmarks/bench_modal.py --kernel decode --cuda --warmup 1 --iters 5 --trials 1
modal run benchmarks/bench_modal.py --kernel prefill --warmup 1 --iters 5 --trials 1
modal run benchmarks/bench_modal.py --kernel prefill --cuda --warmup 1 --iters 5 --trials 1

# Decode raw kernel comparison
modal run scripts/bench_cuda_real.py

# Triton prefill v4 vs v5
modal run scripts/bench_prefill_v5.py

# Current CuTe/tcgen prefill chaser
modal run scripts/bench_prefill_tensorcore.py --mode chunkmodule
```

## Directory Structure

```text
gdn/
├── benchmarks/
├── decode/
├── docs/
├── kernels/
├── prefill/
├── scripts/
├── tests/
└── trace_definitions/
```

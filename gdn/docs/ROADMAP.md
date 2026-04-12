# GDN Roadmap

Current performance source of truth lives in [PERFORMANCE.md](./PERFORMANCE.md).

## Current Reality

This repo is not a single unified production kernel stack.
It currently contains three layers of truth:

1. Official workload truth
   The authoritative production-facing benchmark path is [bench_modal.py](../benchmarks/bench_modal.py).
   This is the path that answers "what is the current shipped result?"

2. Operator microbenchmark truth
   Decode raw kernel truth is [bench_cuda_real.py](../scripts/bench_cuda_real.py).
   Prefill CuTe/tcgen truth is [bench_prefill_tensorcore.py](../scripts/bench_prefill_tensorcore.py).
   These scripts answer "which low-level kernel is currently best under a specific shape?"

3. Experimental algorithm truth
   The active CuTe/tcgen prefill chase path is [chunked_proto.py](../prefill/solution/cuda/chunked_proto.py).
   This is research code, not the current production winner.

## Current Status

### Decode

- Production engineering baseline: [decode Triton kernel](../decode/solution/triton/kernel.py)
- Official benchmark status on B200: competitive and stable
- Raw kernel peak at large batch: `v10 CuTe`, with `v10 TMA` effectively tied
- Packaged CUDA wrapper path: [decode CUDA wrapper](../decode/solution/cuda/kernel.py)
- Important nuance:
  The packaged CUDA wrapper is now a `v10` candidate path, while raw decode ranking still comes from [bench_cuda_real.py](../scripts/bench_cuda_real.py).

### Prefill

- Production engineering baseline: [prefill Triton kernel](../prefill/solution/triton/kernel.py)
- Official benchmark status on B200: still clearly the best production path
- Packaged CUDA wrapper path: [prefill CUDA wrapper](../prefill/solution/cuda/kernel.py)
- Current CuTe/tcgen chaser: [chunked_proto.py](../prefill/solution/cuda/chunked_proto.py)
- Important nuance:
  The CuTe/tcgen path has improved by orders of magnitude relative to its early prototype, but it is still not close enough to replace Triton in the official workload benchmark.

## Active Paths

### Official Production Paths

- Decode Triton:
  [decode/solution/triton/kernel.py](../decode/solution/triton/kernel.py)
- Prefill Triton:
  [prefill/solution/triton/kernel.py](../prefill/solution/triton/kernel.py)
- Official benchmark harness:
  [benchmarks/bench_modal.py](../benchmarks/bench_modal.py)

### Decode Research / Kernel Comparison Paths

- Packaged CUDA wrapper:
  [decode/solution/cuda/kernel.py](../decode/solution/cuda/kernel.py)
- Real decode kernel benchmark:
  [scripts/bench_cuda_real.py](../scripts/bench_cuda_real.py)
- Source build path for raw decode kernels:
  [scripts/build_cuda.py](../scripts/build_cuda.py)

### Prefill Research / CuTe Paths

- Packaged CUDA wrapper:
  [prefill/solution/cuda/kernel.py](../prefill/solution/cuda/kernel.py)
- Current chunked CuTe/tcgen prototype:
  [prefill/solution/cuda/chunked_proto.py](../prefill/solution/cuda/chunked_proto.py)
- Prefill tensor-core benchmark harness:
  [scripts/bench_prefill_tensorcore.py](../scripts/bench_prefill_tensorcore.py)

## What Has Been Learned

### Decode

- Decode remains fundamentally memory-bound.
- Triton is still a strong default production choice because its official workload average is effectively tied with the packaged CUDA path.
- The best raw decode kernels now sit in the `v9/v10` family, not in the packaged `v5` wrapper.
- Large-batch decode is the only place where non-Triton decode kernels clearly separate from Triton.

### Prefill

- Prefill Triton is still the production winner.
- The old standalone prefill CuTe/PTX prototypes are no longer the important chase path.
- The only CuTe/tcgen path that currently matters is the chunked prototype.
- The chunked prototype has improved by:
  - reducing Python orchestration
  - adding strided-batched CUTLASS fast paths
  - improving auto chunk selection
- Remaining gap is now mostly in correction-side launch structure, not in basic GEMM availability.

## Roadmap Priorities

### P0: Keep Source of Truth Clean

- Keep [PERFORMANCE.md](./PERFORMANCE.md) current after meaningful B200 reruns.
- Keep benchmark scripts self-contained enough to run on Modal without import-path surprises.
- Avoid further drift between official benchmark numbers and documentation.

### P1: Decode Packaging Alignment

- Decide whether packaged decode CUDA should remain the legacy `v5` wrapper or be upgraded to a `v9/v10`-based packaged path.
- If packaging stays on `v5`, document clearly that raw decode peak results come from separate microbenchmark kernels.
- If packaging upgrades, benchmark it again with [bench_modal.py](../benchmarks/bench_modal.py) before claiming any win.

### P2: Prefill CuTe Catch-Up

- Continue pushing [chunked_proto.py](../prefill/solution/cuda/chunked_proto.py).
- Main target is no longer "make GEMM exist"; that is already done.
- Main target is:
  - reducing correction launch overhead
  - reducing per-chunk orchestration boundaries
  - moving more of the chunk recurrence structure below the Python boundary

### P3: Promotion Criteria

The CuTe/tcgen prefill path should not be promoted into the official production path until all of these are true:

- It beats Triton on the official workload benchmark, not just on hand-picked microbenchmarks.
- It has stable correctness against the reference path.
- Its dispatch strategy is simple enough to explain and maintain.
- Its benchmark and build scripts are no more fragile than the Triton path.

## Current Decision Rules

- For production-facing numbers, trust [benchmarks/bench_modal.py](../benchmarks/bench_modal.py).
- For decode raw kernel choice, trust [scripts/bench_cuda_real.py](../scripts/bench_cuda_real.py).
- For prefill CuTe progress, trust [scripts/bench_prefill_tensorcore.py](../scripts/bench_prefill_tensorcore.py).

## Non-Goals Right Now

- Reconstructing a clean linear version history across `v5-v10`
- Claiming the CuTe/tcgen prefill line is production-ready
- Treating old `src/kernels/...` references as active source of truth

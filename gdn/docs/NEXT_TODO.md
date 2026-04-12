# Next TODO

This file tracks the next useful work from the repo's current state, not from the old version storyline.

## P0

### 1. Keep performance source of truth fresh

- Update [PERFORMANCE.md](./PERFORMANCE.md) whenever a meaningful B200 rerun changes the ranking.
- Always separate:
  - official workload results from [benchmarks/bench_modal.py](../benchmarks/bench_modal.py)
  - operator microbenchmarks from `scripts/*.py`

### 2. Finish decode packaging alignment

- File:
  [decode/solution/cuda/kernel.py](../decode/solution/cuda/kernel.py)
- Problem:
  packaged CUDA decode still wraps `v5`-style logic, while the current best raw decode kernels come from the `v9/v10` family.
- Goal:
  decide whether to:
  - keep packaged decode on legacy `v5`
  - or upgrade packaging to a `v9/v10`-backed path

### 3. Keep benchmark scripts self-contained on Modal

- Files:
  [scripts/bench_cuda_real.py](../scripts/bench_cuda_real.py)
  [scripts/bench_prefill_v5.py](../scripts/bench_prefill_v5.py)
  [scripts/bench_v5_v6.py](../scripts/bench_v5_v6.py)
- Goal:
  avoid import-path and local deserialization issues so each benchmark can be rerun independently without extra manual fixes

## P1

### 4. Continue reducing prefill chunked correction overhead

- File:
  [prefill/solution/cuda/chunked_proto.py](../prefill/solution/cuda/chunked_proto.py)
- Current status:
  GEMM-side batching is already in place.
  Auto chunk selection is already in place.
  The main remaining gap is still correction-side launch structure.
- Goal:
  reduce the number of independent correction-side launches or move more recurrence work below the Python boundary.

### 5. Add timing breakdown for chunked prefill

- File:
  [scripts/bench_prefill_tensorcore.py](../scripts/bench_prefill_tensorcore.py)
- Goal:
  split time into:
  - GEMM
  - small `kk/kq` products
  - correction kernel
  - orchestration overhead
- Why:
  this makes the next optimization target obvious instead of guessed.

### 6. Re-check auto chunk heuristic as the kernel changes

- Files:
  [prefill/solution/cuda/chunked_proto.py](../prefill/solution/cuda/chunked_proto.py)
  [scripts/bench_prefill_tensorcore.py](../scripts/bench_prefill_tensorcore.py)
- Current heuristic:
  - equal-length multi-sequence batch → `32`
  - otherwise → `64`
- Goal:
  keep validating that this remains correct after each structural optimization.

## P2

### 7. Promote only when official benchmark wins

- Files:
  [benchmarks/bench_modal.py](../benchmarks/bench_modal.py)
  [prefill/solution/cuda/chunked_proto.py](../prefill/solution/cuda/chunked_proto.py)
- Rule:
  the CuTe/tcgen prefill path should not replace Triton until it wins on the official workload benchmark, not only on hand-picked microbenchmarks.

### 8. Clean up stale path references

- Goal:
  keep docs and scripts pointing at actual current directories such as:
  - `kernels/cuda`
  - `kernels/cute_cpp`
  - `kernels/ptx`
- Avoid reintroducing old `src/kernels/...` references as if they were active.

### 9. Decide how much of the historical version story is worth preserving

- Some `v5-v10` labels are still useful for low-level kernels.
- They are not a reliable description of the production path anymore.
- Goal:
  preserve version labels only where they still help benchmark and implementation discussion.

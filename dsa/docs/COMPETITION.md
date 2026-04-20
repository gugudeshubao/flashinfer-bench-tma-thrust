# Competition Freeze

## Recommended Tag

- `dsa-comp-auto-v12`

## What This Tag Means

- `backend="auto"` is the intended competition path.
- Prefill uses adaptive dispatch:
  - shorter and medium cases can stay on the reference path
  - longer cases switch to the Triton MLA sparse kernel
- Decode uses the Triton MLA sparse kernel under `auto` on the benchmarked competition shapes.

## Validation Commands

```bash
modal run dsa/tests/test_modal.py
modal run dsa/benchmarks/bench_modal.py --warmup 2 --iters 5
modal run dsa/benchmarks/profile_modal.py --iters 20
modal run dsa/benchmarks/profile_selection_modal.py --iters 20
modal run dsa/benchmarks/tune_selection_modal.py --iters 20
```

## Current Modal B200 Snapshot

- Prefill benchmark:
  - `p256`: backend=`reference`, `1.067x`
  - `p512`: backend=`reference`, `1.023x`
  - `p1024`: backend=`reference`, `1.015x`
  - `p2048`: backend=`triton`, `2.666x`
  - `p4096`: backend=`triton`, `5.414x`
- Decode benchmark:
  - `d2048`: backend=`triton`, `1.219x`
  - `d4096`: backend=`triton`, `1.334x`
  - `d8192`: backend=`triton`, `1.247x`

## Remaining Risk

- The largest remaining hotspot is still prefill-side selection / metadata work on forced-Triton paths.
- Selection profiling shows `weighted_q` and `topk` dominate the remaining prefill selection cost.
- If a future tuning pass is needed, that should focus on moving prefill-side selection out of Torch.

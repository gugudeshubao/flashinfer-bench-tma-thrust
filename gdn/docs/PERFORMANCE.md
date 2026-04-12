# GDN Performance Snapshot

All numbers below were rerun on Modal B200 on 2026-04-12 unless noted otherwise.

## Terminology

There are two different meanings of "baseline" in this repo.

| Term | Meaning |
|------|---------|
| Official baseline / official reference | The reference implementation used by `flashinfer-bench` inside [`benchmarks/bench_modal.py`](../benchmarks/bench_modal.py). The `speedup` numbers in the official workload benchmark are measured against this reference latency. |
| Triton baseline | Our current engineering baseline: the Triton implementation we are using as the current strongest production-ready path for comparison inside this repo. |

Short version:

- If you ask "official baseline", that means the benchmark reference used by `flashinfer-bench`.
- If this document says "Triton baseline", it means our current internal baseline, not the official reference implementation.

## Hardware

| Item | Value |
|------|-------|
| GPU | NVIDIA B200 |
| CUDA | 12.8 toolchain in Modal images |
| FP32 compute | ~`80 TFLOPS` per GPU |
| FP16/BF16 Tensor compute | ~`2.5 PFLOPS` dense per GPU |
| HBM3e memory | ~`180 GB` per GPU |
| HBM3e bandwidth | ~`8 TB/s` per GPU |
| Head size | `D=128` |
| Decode shape | `qk4 / v8 / d128 / k-last` |
| Prefill shape | `qk4 / v8 / d128 / k-last` |

Notes:

- Memory capacity and bandwidth are derived from official NVIDIA DGX B200 system specs: `1,440 GB` total GPU memory and `64 TB/s` total HBM3e bandwidth over `8` GPUs.
- Compute figures are derived per GPU from official NVIDIA GB200 Grace Blackwell Superchip specs, which are published for `2` Blackwell GPUs together.
- NVIDIA sources:
  - https://www.nvidia.com/en-us/data-center/dgx-b200/
  - https://www.nvidia.com/en-us/data-center/gb200-nvl72/

## One-Glance Status

This is the current source of truth.

| Area | Triton Engineering Baseline | Competition Candidate | Best CuTe-Based Chaser | Current Judgment |
|------|------------------|---------------------|------------------------|------------------|
| Decode official workload benchmark | `473.38x` avg speedup | `v10 CuTe / v10 TMA` | official packaged candidate results: `470.84x / 440.17x` avg speedup; raw `B=256`: `0.0503 / 0.0505 ms`, `5333 / 5316 GB/s`, about `2.17x` vs Triton | If the goal is a non-Triton decode submission, `v10 CuTe / v10 TMA` is the right current candidate |
| Prefill official workload benchmark | `257.45x` avg speedup | CUDA wrapper v6: `180.12x` avg | `chunked_proto auto`: `0.395 ms` at `N=4,L=32`, about `0.07x` vs Triton there | Triton is still the only production winner; CuTe path is improving but not yet close enough |

## Competition Decode Candidate

If your target is the decode competition submission rather than the current packaged benchmark path, the current candidate should be treated as:

- `v10 CuTe`
- `v10 TMA`

They are your strongest measured non-Triton decode operators today.

PTX decode now compiles again after fixing the inline `selp` issue, but it is not competitive with the `v10` family.

### Current Competition Decode Parameters

| Candidate | Batch | Time | Bandwidth | vs Triton |
|-----------|------:|-----:|----------:|----------:|
| `v10 CuTe` | 1 | `0.0522 ms` | `20 GB/s` | `1.21x` |
| `v10 TMA` | 1 | `0.0521 ms` | `20 GB/s` | `1.21x` |
| `v10 CuTe` | 16 | `0.0535 ms` | `314 GB/s` | `1.15x` |
| `v10 TMA` | 16 | `0.0538 ms` | `312 GB/s` | `1.15x` |
| `v10 CuTe` | 64 | `0.0644 ms` | `1043 GB/s` | `0.97x` |
| `v10 TMA` | 64 | `0.0642 ms` | `1045 GB/s` | `0.97x` |
| `v10 CuTe` | 256 | `0.0503 ms` | `5333 GB/s` | `2.17x` |
| `v10 TMA` | 256 | `0.0505 ms` | `5316 GB/s` | `2.17x` |

### Official Workload Results For Decode Candidates

These are official 54-workload benchmark reruns using the packaged decode CUDA path switched to the `v10` family.

| Candidate | Workloads | Average Speedup vs Official Reference | Status |
|-----------|-----------|----------------------------------------|--------|
| Triton solution | 54 | `473.38x` | all `PASSED` |
| `v10 CuTe` packaged candidate | 54 | `470.84x` | all `PASSED` |
| `v10 TMA` packaged candidate | 54 | `440.17x` | all `PASSED` |

Current reading:

- `v10 CuTe` is essentially tied with the Triton official path on the official 54-workload suite.
- `v10 TMA` is clearly usable, but currently weaker than both Triton and `v10 CuTe` on the official workload average.
- So if you want to submit a non-Triton decode path today, `v10 CuTe` is the best current candidate.

## Official Benchmark Results

Important:

- These `speedup` values are against the official `flashinfer-bench` reference implementation.
- They are not speedups against Triton.
- `v10 CuTe / v10 TMA` is now runnable through the packaged official benchmark path, but it is selected explicitly via the decode CUDA backend choice.
- The current official decode choices exposed by [benchmarks/bench_modal.py](../benchmarks/bench_modal.py) are:
  - Triton solution
  - packaged `v10 CuTe` candidate
  - packaged `v10 TMA` candidate
- The `v9/v10` decode kernels are measured today through the raw-kernel microbenchmark path:
  [scripts/bench_cuda_real.py](../scripts/bench_cuda_real.py)
- The same applies to the CuTe/tcgen prefill chase path:
  it is measured through [scripts/bench_prefill_tensorcore.py](../scripts/bench_prefill_tensorcore.py), not through the official workload suite.

Commands:

```bash
modal run benchmarks/bench_modal.py --kernel decode --warmup 1 --iters 5 --trials 1
modal run benchmarks/bench_modal.py --kernel decode --cuda --cuda-backend cute --warmup 1 --iters 5 --trials 1
modal run benchmarks/bench_modal.py --kernel decode --cuda --cuda-backend tma --warmup 1 --iters 5 --trials 1
modal run benchmarks/bench_modal.py --kernel prefill --warmup 1 --iters 5 --trials 1
modal run benchmarks/bench_modal.py --kernel prefill --cuda --warmup 1 --iters 5 --trials 1
```

| Kernel | Path | Workloads | Result |
|--------|------|-----------|--------|
| Decode | Triton solution | 54 | `473.38x` avg speedup, all `PASSED` |
| Decode | packaged `v10 CuTe` candidate | 54 | `470.84x` avg speedup, all `PASSED` |
| Decode | packaged `v10 TMA` candidate | 54 | `440.17x` avg speedup, all `PASSED` |
| Prefill | Triton solution | 100 | `257.45x` avg speedup, all `PASSED` |
| Prefill | CUDA wrapper v6 | 100 | `180.12x` avg speedup, all `PASSED` |

## Speedup vs Official Reference

This table is the correct answer when you ask "compared to the official baseline, how fast are we now?"

| Area | Path | Speedup vs Official Reference | Included in Official Suite |
|------|------|-------------------------------|----------------------------|
| Decode | Triton solution | `473.38x` avg | Yes |
| Decode | `v10 CuTe` packaged candidate | `470.84x` avg | Yes |
| Decode | `v10 TMA` packaged candidate | `440.17x` avg | Yes |
| Decode | Raw `v10 CuTe / v10 TMA` operator family | representative raw speedup is `2.17x` vs Triton at `B=256` | No |
| Prefill | Triton solution | `257.45x` avg | Yes |
| Prefill | CUDA wrapper v6 | `180.12x` avg | Yes |
| Prefill | CuTe/tcgen `chunked_proto` | not reported in official suite; representative local gain is `186.63x` vs reference at `N=4,L=32` | No |

## Why There Are Two Decode `v10` Result Types

There are now two valid ways to describe decode `v10`:

1. Official workload result
   The packaged candidate is benchmarked through [benchmarks/bench_modal.py](../benchmarks/bench_modal.py), producing the 54-workload average speedup numbers.

2. Raw operator result
   The low-level kernel is benchmarked through [scripts/bench_cuda_real.py](../scripts/bench_cuda_real.py), producing per-batch latency and bandwidth numbers.

Both are useful:

- use the official workload result for competition-facing average speedup claims
- use the raw operator result to understand where `v10` actually wins

## Representative Throughput

The official workload benchmark is heterogeneous, so it does not have a single meaningful "throughput" scalar.
For throughput, use the operator-level microbenchmarks below.

### Decode Throughput

Decode is memory-bound, so bandwidth is the right throughput metric.

| Path | Representative Config | Throughput |
|------|------------------------|------------|
| Official reference | official workload suite | n/a as one scalar |
| Triton engineering baseline | `batch=256` | `2453 GB/s` |
| Best non-Triton raw operator | `v10 CuTe @ batch=256` | `5333 GB/s` |
| Best CuTe operator | `v10 CuTe @ batch=256` | `5333 GB/s` |
| Best TMA peer | `v10 TMA @ batch=256` | `5316 GB/s` |

### Decode Representative Acceleration

These are not official workload averages.
They are representative operator-level gains from the raw decode benchmark.

| Config | Triton | Best Candidate | Best Candidate vs Triton |
|--------|--------|-----------|---------------------|
| `batch=1` | `0.0630 ms` | `v9: 0.0517 ms` | `1.22x` |
| `batch=16` | `0.0617 ms` | `v10 CuTe: 0.0535 ms` | `1.15x` |
| `batch=64` | `0.0623 ms` | `v10 CuTe: 0.0644 ms` | `0.97x` |
| `batch=256` | `0.1094 ms` | `v10 CuTe: 0.0503 ms` | `2.17x` |

### Prefill Throughput

Prefill is easier to read in tokens per second.

Representative tokens/s below are computed from the current `chunkmodule` reruns:

- Throughput formula: `total_tokens / latency_ms / 1000`

| Path | Representative Config | Throughput |
|------|------------------------|------------|
| Official reference | official workload suite | n/a as one scalar |
| Triton engineering baseline | `N=4, L=32` | `~2.54 M tok/s` |
| Best CuTe-based chaser | `chunked_proto auto @ N=4, L=32` | `~0.19 M tok/s` |
| Best CuTe-based single-seq point | `chunked_proto batched @ N=1, L=64` | `~0.08 M tok/s` |

### Prefill Representative Acceleration

These are representative local benchmark gains, not official workload averages.

| Config | Reference | Triton | CuTe-Based Chaser | Chaser vs Reference | Chaser vs Triton |
|--------|-----------|--------|-------------------|---------------------|------------------|
| `N=1, L=64` | `59.10 ms` | `0.0573 ms` | `auto: 0.9564 ms` | `61.79x` | `0.06x` |
| `N=4, L=32` | `125.91 ms` | `0.0504 ms` | `auto: 0.6747 ms` | `186.63x` | `0.07x` |
| `lengths=[32,32,64,64]` | `180.51 ms` | `0.0498 ms` | `auto: 2.4804 ms` | `72.77x` | `0.02x` |

## Decode Operator Performance

Command:

```bash
modal run scripts/bench_cuda_real.py
```

Correctness:

- `CUDA v7`
- `CUDA v8`
- `CUDA v9`
- `v10 CuTe`
- `v10 TMA`

All passed against Triton reference in the current script.

### Current Decode Microbenchmark

| Batch | Triton v5 | CUDA v7 | CUDA v8 | CUDA v9 | v10 CuTe | v10 TMA | PTX | Winner |
|------:|----------:|--------:|--------:|--------:|---------:|--------:|-----|--------|
| 1 | `0.0630 ms` | `0.0540 ms` | `0.0547 ms` | `0.0517 ms` | `0.0522 ms` | `0.0521 ms` | see PTX section below | `v9` in the earlier raw table; `v10` family in the latest comprehensive rerun |
| 16 | `0.0617 ms` | `0.0594 ms` | `0.0620 ms` | `0.0535 ms` | `0.0535 ms` | `0.0538 ms` | see PTX section below | `v9 / v10 CuTe` |
| 64 | `0.0623 ms` | `0.0802 ms` | `0.0858 ms` | `0.0637 ms` | `0.0644 ms` | `0.0642 ms` | see PTX section below | Triton |
| 256 | `0.1094 ms` | `0.0515 ms` | `0.0508 ms` | `0.0507 ms` | `0.0503 ms` | `0.0505 ms` | see PTX section below | `v10 CuTe / v10 TMA` |

### Decode Interpretation

- If the target is the official workload average, Triton remains acceptable because it is already basically tied with the packaged CUDA wrapper.
- If the target is the best raw decode kernel at larger batch sizes, the current leader is the `v10` family, with `v10 CuTe` and `v10 TMA` effectively tied.
- PTX decode is now compilable again, but it is clearly behind Triton and well behind `v9/v10`.
- `v7 / v8` are no longer the best decode kernels except as historical stepping stones.

### Decode PTX Results

These are from the latest comprehensive decode rerun after fixing the PTX `selp` compile issue.

| Batch | PTX Time | PTX Bandwidth | PTX vs Triton |
|------:|---------:|--------------:|--------------:|
| 1 | `0.0124 ms` | `84 GB/s` | `1.50x` in that benchmark's local setup |
| 4 | `0.0124 ms` | `339 GB/s` | `1.49x` |
| 8 | `0.0145 ms` | `580 GB/s` | `1.21x` |
| 16 | `0.0206 ms` | `814 GB/s` | `0.86x` |
| 32 | `0.0329 ms` | `1020 GB/s` | `0.54x` |
| 64 | `0.0596 ms` | `1126 GB/s` | `0.30x` |
| 128 | `0.1168 ms` | `1149 GB/s` | `0.23x` |
| 256 | `0.2211 ms` | `1214 GB/s` | `0.22x` |

Current reading:

- PTX decode is not a serious competition candidate in the current repo state.
- It only looks good at very small batch in the comprehensive local benchmark.
- Once batch grows, `v10 CuTe / v10 TMA` and Triton are decisively better.

## Prefill Triton Internal Status

Command:

```bash
modal run scripts/bench_prefill_v5.py
```

This compares Triton `v4` against Triton `v5` directly by launching the two internal kernels.

| Config | v4 | v5 | v5/v4 |
|--------|----|----|--------|
| `N=1, L=256` | `0.210 ms` | `0.126 ms` | `1.66x` |
| `N=1, L=512` | `0.416 ms` | `0.249 ms` | `1.67x` |
| `N=1, L=1024` | `0.827 ms` | `0.493 ms` | `1.68x` |
| `N=4, L=256` | `0.222 ms` | `0.144 ms` | `1.54x` |
| `N=4, L=512` | `0.439 ms` | `0.285 ms` | `1.54x` |
| `N=8, L=256` | `0.267 ms` | `0.237 ms` | `1.13x` |
| `N=16, L=128` | `0.180 ms` | `0.178 ms` | `1.02x` |
| `N=32, L=64` | `0.147 ms` | `0.161 ms` | `0.91x` |

Summary:

- Average `v5/v4`: `1.394x`
- Best case: `1.679x` at `N=1, L=1024`
- Worst case: `0.914x` at `N=32, L=64`

Current reading:

- `v5` is better for long sequences and low batch.
- `v4` can still be competitive or better when the workload is dominated by many short sequences.
- The public wrapper currently keeps the fallback chain because both kernels still matter for robustness.

## Prefill CuTe / PTX Prototype Status

Command:

```bash
modal run scripts/bench_prefill_tensorcore.py --mode bench
```

This compares the older standalone prefill tensor-core prototypes against Triton.

| Config | Triton | CuTe v10 | CuTe v10 vs Triton | PTX mma | PTX vs Triton |
|--------|--------|----------|--------------------|---------|---------------|
| `N=1, L=256` | `0.1393 ms` | `0.5692 ms` | `0.24x` | `2.1616 ms` | `0.06x` |
| `N=1, L=1024` | `0.5283 ms` | `2.2025 ms` | `0.24x` | `8.5770 ms` | `0.06x` |
| `N=4, L=256` | `0.1545 ms` | `0.5752 ms` | `0.27x` | `2.2007 ms` | `0.07x` |
| `N=16, L=128` | `0.1878 ms` | `0.3969 ms` | `0.47x` | `2.4531 ms` | `0.08x` |

Current reading:

- The old standalone `CuTe v10` prefill prototype is not the current chaser anymore.
- The PTX standalone prototype is still far from usable as a competitive prefill implementation.

## Prefill Current CuTe Chaser

Command:

```bash
modal run scripts/bench_prefill_tensorcore.py --mode chunkmodule
```

This is the current CuTe/tcgen-based chase path:

- strided-batched CUTLASS fast path enabled
- auto chunk selection enabled
- current auto policy:
  - single-sequence or general varlen: `chunk=64`
  - equal-length multi-sequence: `chunk=32`

### Current Auto-Kernel Results

| Config | Triton | Auto kernel | Best manual path | Current note |
|--------|--------|-------------|------------------|--------------|
| `N=1, L=64` | `0.0422 ms` | `0.6795 ms` | `batched=0.6326 ms` | Auto chooses `64`; still much slower than Triton |
| `N=4, L=32` | `0.0286 ms` | `0.3954 ms` | `grouped=0.6909 ms`; auto wins by selecting `32` | Best current experimental point |
| `lengths=[32,32,64,64]` | `0.0465 ms` | `1.5054 ms` | `grouped=1.4012 ms` | Auto chooses `64`; still far from Triton |

### Current Chunk Search Result

The latest sweep over `chunk_size = [4, 8, 16, 32, 64]` shows:

| Workload | Best chunk size |
|----------|-----------------|
| `N=1, L=64` | `64` |
| `N=1, L=128` | `64` |
| `N=4, L=32` | `32` |
| `N=4, L=64` | `32` |
| `lengths=[32,32,64,64]` | `64` |
| `lengths=[32,64,128,256]` | `64` |

## Current Recommendations

| Area | Recommended path now | Reason |
|------|----------------------|--------|
| Decode official benchmark / packaging | Triton solution | Stable and already tied with packaged CUDA wrapper |
| Decode raw kernel peak at large batch | `v10 CuTe / v10 TMA` | Current best measured decode operators at `B=256` |
| Prefill official benchmark / packaging | Triton solution | Clear winner over CUDA wrapper and all current experimental paths |
| Prefill CuTe/tcgen research | `chunked_proto` auto kernel | Best current non-Triton chase path, but still not production-ready |

## Important Caveat

There are two different kinds of numbers in this document:

- Official workload averages from `benchmarks/bench_modal.py`
- Operator-level microbenchmarks from `scripts/*.py`

They are both useful, but they are not the same benchmark.
For production decisions, trust the official workload averages first.

## Benchmark Commands

```bash
# Official workload benchmarks
modal run benchmarks/bench_modal.py --kernel decode --warmup 1 --iters 5 --trials 1
modal run benchmarks/bench_modal.py --kernel decode --cuda --warmup 1 --iters 5 --trials 1
modal run benchmarks/bench_modal.py --kernel prefill --warmup 1 --iters 5 --trials 1
modal run benchmarks/bench_modal.py --kernel prefill --cuda --warmup 1 --iters 5 --trials 1

# Decode real CUDA kernels
modal run scripts/bench_cuda_real.py

# Triton prefill v4 vs v5
modal run scripts/bench_prefill_v5.py

# Older CuTe/PTX prefill prototypes
modal run scripts/bench_prefill_tensorcore.py --mode bench

# Current CuTe chunked prefill chaser
modal run scripts/bench_prefill_tensorcore.py --mode chunkmodule
```

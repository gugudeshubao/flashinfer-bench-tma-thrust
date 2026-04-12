# CuTe C++ Kernels

This directory contains CuTe C++ / CUTLASS kernels compiled with NVCC.

## Current Status

### Decode

- CuTe C++ is currently the strongest raw decode family at large batch.
- The current best measured raw decode operator on B200 is `v10 CuTe`, with `v10 TMA` effectively tied in the latest decode microbenchmark.
- Official packaged decode still uses Triton as the practical production baseline.

### Prefill

- The standalone CuTe C++ prefill kernels in this directory are no longer the most useful indicator of progress.
- The current CuTe/tcgen chase path is:
  [prefill/solution/cuda/chunked_proto.py](../../prefill/solution/cuda/chunked_proto.py)
- That path has improved substantially, but it is still slower than the Triton production path on the official workload benchmark.

## Practical Reading

- For decode:
  treat `gdn_decode_v9.cuh` and `gdn_decode_v10.cuh` as the important raw-kernel files.
- For prefill:
  treat `gdn_prefill_v9.cuh` and `gdn_prefill_v10.cuh` as historical/operator-building references, not as the current production chaser by themselves.

## Source of Truth

Use:

- [scripts/bench_cuda_real.py](../../scripts/bench_cuda_real.py) for decode kernel ranking
- [scripts/bench_prefill_tensorcore.py](../../scripts/bench_prefill_tensorcore.py) for prefill CuTe/tcgen progress
- [docs/PERFORMANCE.md](../../docs/PERFORMANCE.md) for the current summary

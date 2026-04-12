# Raw CUDA Kernels

This directory contains raw CUDA source kernels.

These files are useful for low-level decode/prefill experimentation, but they are not automatically the current production winner.

## Current Status

### Decode

- Packaged CUDA decode wrapper still centers on a legacy `v5`-style path:
  [decode/solution/cuda/kernel.py](../../decode/solution/cuda/kernel.py)
- Current best raw decode kernels measured on B200 come from the `v9/v10` family, not from the older raw CUDA `v5-v8` family.

### Prefill

- Packaged CUDA prefill wrapper is:
  [prefill/solution/cuda/kernel.py](../../prefill/solution/cuda/kernel.py)
- That wrapper is still behind the Triton production path on the official workload benchmark.

## Use This Directory For

- low-level CUDA implementation experiments
- historical comparison points
- manual kernel design ideas

## Do Not Assume

- that the highest-numbered raw CUDA file here is the current best kernel
- that numbers previously written in this README are still current
- that raw CUDA here beats Triton in the official benchmark

For current numbers, see:

- [docs/PERFORMANCE.md](../../docs/PERFORMANCE.md)

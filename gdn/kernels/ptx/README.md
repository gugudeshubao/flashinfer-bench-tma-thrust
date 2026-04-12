# PTX Kernels

This directory contains inline PTX experiments.

## Current Status

### Decode

- PTX ideas are still relevant as low-level optimization references.
- They are not the current production decode path.
- The current best measured raw decode results are coming from the `v9/v10` family measured through the decode microbenchmark, not from a standalone PTX path.

### Prefill

- The standalone PTX prefill prototype compiles and runs.
- It is still far behind Triton and also behind the current CuTe/tcgen chunked chase path.
- It should be viewed as a low-level building block or idea source, not as a current candidate for promotion.

## What PTX Still Helps With

- understanding tcgen / low-level instruction structure
- validating instruction availability and compile paths
- experimenting with copy, math, and memory primitives

## What PTX Is Not Right Now

- not the decode production winner
- not the prefill production winner
- not the current best CuTe/tcgen chase path

For current measured status, use:

- [docs/PERFORMANCE.md](../../docs/PERFORMANCE.md)

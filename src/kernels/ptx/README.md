# PTX Embedded CUDA Kernels

This directory contains GDN kernels implemented in CUDA C++ with **embedded PTX assembly instructions** for maximum control over low-level GPU operations.

## Files

| File | Description |
|------|-------------|
| `gdn_decode_ptx.cuh` | GDN decode kernel with PTX assembly |
| `gdn_prefill_ptx.cuh` | **NEW** GDN prefill kernel with chunking + PTX |

## Prefill Compute Density Optimization

### The Problem: Memory-Bound Prefill

```
Sequential processing (v5):
  - Process 1 token per iteration
  - State [D×D] loaded/stored per token
  - Arithmetic Intensity = 2*D*D FLOPs / 2*D*D bytes = 1 FLOP/byte
  - Result: Memory-bound on B200 (8 TB/s bandwidth)
```

### The Solution: Chunked Processing (v6)

```
Chunk processing (C tokens at once):
  - State loaded once, reused C times
  - Arithmetic Intensity = C * 2*D*D FLOPs / 2*D*D bytes = C FLOP/byte

  CHUNK_SIZE | Arithmetic Intensity | Bound
  -----------|----------------------|-------
      1      |     1 FLOP/byte      | Memory
      4      |     4 FLOP/byte      | Transitional
      8      |     8 FLOP/byte      | Compute!
     16      |    16 FLOP/byte      | Compute
```

### B200 Roofline Analysis

```
B200 Peak Performance:
  - FP32: 70 TFLOPS
  - Memory BW: 8 TB/s
  - Ridge Point: 70/8 = 8.75 FLOP/byte

With CHUNK_SIZE=8:
  - AI = 8 FLOP/byte ≈ Ridge point!
  - Approaches compute-bound territory
```

## What is PTX?

**PTX (Parallel Thread Execution)** is NVIDIA's low-level virtual machine instruction set. While CUDA C++ provides high-level abstractions, PTX allows direct control over:

- Memory operations with cache hints
- Warp shuffle instructions
- Fast math approximations
- Predicated execution
- Register allocation hints

## Files

- `gdn_decode_ptx.cuh` - GDN decode kernel with inline PTX assembly

## PTX Intrinsics Used

### 1. Warp Shuffle (shfl.sync)
```cpp
// Butterfly shuffle for warp-level reductions
__device__ float ptx_shfl_xor(float val, int lane_mask) {
    float result;
    asm volatile(
        "shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
        : "=f"(result)
        : "f"(val), "r"(lane_mask)
    );
    return result;
}
```

### 2. Fast Math (ex2, lg2, rcp)
```cpp
// exp(x) using PTX: 2^(x * log2(e))
__device__ float ptx_exp2(float x) {
    float result;
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

// log(x) using PTX: log2(x) * ln(2)
__device__ float ptx_log2(float x) {
    float result;
    asm volatile("lg2.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

// Fast reciprocal (1/x)
__device__ float ptx_rcp(float x) {
    float result;
    asm volatile("rcp.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}
```

### 3. Fused Multiply-Add (fma.rn)
```cpp
// a * b + c with single rounding
__device__ float ptx_fma(float a, float b, float c) {
    float result;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
}
```

### 4. Memory Operations with Cache Hints
```cpp
// Non-coherent load (bypasses L1 cache)
__device__ float ptx_ld_nc(const float* ptr) {
    float result;
    asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(result) : "l"(ptr));
    return result;
}

// Write-back store
__device__ void ptx_st_wb(float* ptr, float val) {
    asm volatile("st.global.wb.f32 [%0], %1;" :: "l"(ptr), "f"(val));
}
```

### 5. Predicated Execution (selp)
```cpp
// Branchless conditional: result = pred ? a : b
__device__ float ptx_selp(float a, float b, bool pred) {
    float result;
    asm volatile("selp.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "r"((int)pred));
    return result;
}
```

## Performance Benefits

| Optimization | PTX Instruction | Benefit |
|--------------|-----------------|---------|
| Fast exp/log | `ex2.approx`, `lg2.approx` | ~2-3x faster than libm |
| Warp shuffle | `shfl.sync.bfly` | No shared memory for warp reduce |
| FMA | `fma.rn.f32` | Single rounding, better precision |
| Cache bypass | `ld.global.nc` | Avoid polluting L1 for streaming |
| Branchless | `selp` | No divergence, better occupancy |

## When to Use PTX

Use PTX when:
1. **Maximum performance** - squeezing last 5-10% of performance
2. **Specific cache behavior** - L1/L2 bypass for streaming workloads
3. **Fast math approximations** - when ~1 ULP error is acceptable
4. **Warp-level primitives** - shuffle, vote, match operations

Avoid PTX when:
1. Code readability matters more than performance
2. Portability across GPU architectures is important
3. CUDA intrinsics (`__shfl_sync`, `__fma_rn`) are sufficient

## Comparison: PTX vs CUDA Intrinsics

| Operation | CUDA Intrinsic | PTX Assembly |
|-----------|----------------|--------------|
| Warp shuffle | `__shfl_xor_sync(mask, val, lane)` | `shfl.sync.bfly.b32` |
| FMA | `__fmaf_rn(a, b, c)` | `fma.rn.f32` |
| exp2 | `exp2f(x)` | `ex2.approx.f32` |
| log2 | `log2f(x)` | `lg2.approx.f32` |
| reciprocal | `__frcp_rn(x)` | `rcp.approx.f32` |

## References

- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Inline PTX Assembly](https://docs.nvidia.com/cuda/inline-ptx-assembly/)
- [CUDA Math Functions](https://docs.nvidia.com/cuda/cuda-math-api/)

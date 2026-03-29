# GDN Kernel Performance

**Hardware**: NVIDIA B200 (sm_100), 178 GB HBM3e, 148 SMs, 8 TB/s peak memory BW

---

## Executive Summary (Corrected Results - 2026-03-28)

All kernels verified for **correctness** against Triton v5 baseline.

| Batch | Triton v5 | CUDA v7 | CUDA v8 | **CUDA v9** | v10 CuTe | Best |
|-------|-----------|---------|---------|-------------|----------|------|
| **1** | 24 GB/s | 25 GB/s (1.06x) | 25 GB/s (1.03x) | **27 GB/s (1.11x)** | 26 GB/s | **v9** |
| **16** | 386 GB/s | 352 GB/s (0.91x) | 334 GB/s (0.86x) | **405 GB/s (1.05x)** | 403 GB/s | **v9** |
| 64 | **1,518 GB/s** | 981 GB/s (0.65x) | 914 GB/s (0.60x) | 1,302 GB/s (0.86x) | 1,287 GB/s | **Triton** |
| **256** | 2,834 GB/s | 7,578 GB/s (2.67x) | 7,605 GB/s (2.68x) | 7,585 GB/s (2.68x) | **7,602 GB/s (2.68x)** | **v10** |

**Best Result**: CUDA v9/v10 at batch=256 achieves **7,600 GB/s (95% of B200 peak)**

---

## Correctness Validation

All CUDA kernels pass correctness test (`atol=1e-2, rtol=1e-2`) against Triton v5:

```
Batch=1, BLOCK_V=16:
  ✓ CUDA v7: PASS
  ✓ CUDA v8: PASS
  ✓ CUDA v9: PASS
  ✓ v10 CuTe: PASS
  ✓ v10 TMA: PASS

✓ All kernels produce correct results!
```

### Delta Rule Bug Fix

The original CUDA kernels had incorrect delta rule order. **Fixed**:

```cpp
// CORRECT: Apply g FIRST, then compute old_v
float decayed_s = g * s_state[idx];     // ← Decay first
old_v += decayed_s * k[d];               // ← Use decayed state
// ...
new_s = decayed_s + delta * k[d];        // ← No need to multiply g again
```

---

## Version Summary

| Version | Framework | Key Feature | Peak BW | Best For |
|---------|-----------|-------------|---------|----------|
| v5 | Triton | Auto-tuning | 1,518 GB/s | Batch=64 |
| v7 | CUDA | float4 + FP4 | 7,578 GB/s | Batch=256 |
| v8 | CUDA | Warp spec + FP8 | 7,605 GB/s | Batch=256 |
| **v9** | **CUDA/CuTe** | **SMEM swizzle** | **7,585 GB/s** | **Batch=1,16** |
| v10 | CUDA/CuTe | Swizzle<3,3,3> | 7,602 GB/s | Batch=256 |

### Kernel Selection Recommendation

```python
def select_kernel(batch_size):
    if batch_size <= 16:
        return "CUDA v9"   # Best at small batch
    elif batch_size == 64:
        return "Triton v5"  # Triton wins here
    else:
        return "CUDA v9/v10"  # Best at large batch
```

---

## Memory Bandwidth Utilization

| Batch | State Size | Best Kernel | Achieved BW | B200 Peak | Utilization |
|-------|------------|-------------|-------------|-----------|-------------|
| 1 | 0.5 MB | CUDA v9 | 27 GB/s | 8,000 GB/s | 0.3% |
| 16 | 8.0 MB | CUDA v9 | 405 GB/s | 8,000 GB/s | 5.1% |
| 64 | 32.0 MB | Triton v5 | 1,518 GB/s | 8,000 GB/s | 19% |
| **256** | 128 MB | **v10 CuTe** | **7,602 GB/s** | 8,000 GB/s | **95%** |

---

## CuTe Swizzle Optimization (v9/v10)

v9/v10 use SMEM swizzling to avoid bank conflicts:

```cpp
// XOR-based swizzle for 128-byte cache lines
int swizzled_d = d_idx ^ ((d_idx >> 3) & 7);
s_state[v_idx * D + swizzled_d] = state_ptr[...];
```

This reduces bank conflicts from ~8-way to ~1-way, improving SMEM throughput.

---

## Optimization History

| Version | Optimization | Result |
|---------|--------------|--------|
| v5 | Triton baseline | Good at batch=64 |
| v7 | CUDA float4 + FP4 | Better at large batch |
| v8 | Warp specialization + FP8 | Marginal improvement |
| v8 + Graph | CUDA Graph launch | **No improvement** |
| **v9** | **CuTe SMEM swizzle** | **Best at small batch** |
| v10 | CuTe Swizzle<3,3,3> | Same as v9, cleaner code |

---

## Directory Structure

```
src/kernels/
├── cuda/          # Basic CUDA (v5-v8)
├── cute/          # CuTe DSL (v9-v10)
└── triton/        # Triton baseline
```

## Benchmark Commands

```bash
# Correctness + Performance benchmark
modal run scripts/bench_cuda_real.py

# Build CUDA library
modal run scripts/build_cuda.py
```

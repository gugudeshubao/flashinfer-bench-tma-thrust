# GDN Kernel Performance

## Hardware: NVIDIA B200 (Blackwell, sm_100)

| Specification | Value |
|---------------|-------|
| CUDA Cores | 16,896 |
| Tensor Cores | 528 (5th Gen) |
| Boost Clock | 1.98 GHz |
| SMs | 148 |
| HBM3e | 180 GB |
| **Memory BW** | **8 TB/s** |
| FP32 (CUDA) | 74.45 TFLOPS |
| BF16 Tensor | 2.25 PFLOPS |
| FP8 Tensor | 4.5 PFLOPS |
| TDP | 1,000 W |

---

## Benchmark Results (2026-03-28)

### Decode Performance (Triton v5 Baseline)

| Batch | Time (ms) | BW (GB/s) | State Size | B200 Util. |
|-------|-----------|-----------|------------|------------|
| 1 | 0.0479 | 23 | 0.5 MB | 0.3% |
| 4 | 0.0454 | 93 | 2.0 MB | 1.2% |
| 16 | 0.0456 | 375 | 8.0 MB | 4.7% |
| 64 | 0.0471 | 1,502 | 32.0 MB | 18.8% |
| **256** | 0.0939 | **2,798** | 128.0 MB | **35.0%** |
| 512 | 0.1512 | 3,465 | 256.0 MB | 43.3% |

### Prefill Performance (Triton v4 Baseline)

| Config | N | SeqLen | T | Time (ms) | Speedup vs Ref |
|--------|---|--------|---|-----------|----------------|
| Short | 4 | 64 | 256 | 0.1005 | 125x |
| Medium | 4 | 256 | 1024 | 0.2759 | 181x |
| Long | 4 | 1024 | 4096 | 0.9759 | 219x |
| **Many short** | 16 | 64 | 1024 | 0.0743 | **332x** |
| Many long | 16 | 1024 | 16384 | 0.3451 | 1088x |
| Max | 16 | 4096 | 65536 | 0.7894 | **1886x** |

---

## All Versions Comparison (v1-v11)

### Decode Bandwidth (GB/s)

| Batch | Triton v4 | Triton v5 | CUDA v5 | CUDA v6 | CUDA v7 | CUDA v8 | CuTe v9 | CuTe v10 |
|-------|-----------|-----------|---------|---------|---------|---------|---------|----------|
| **1** | 23 | 23 | 24 | 24 | 25 | 25 | **27** | 26 |
| **16** | 375 | 386 | 352 | 355 | 352 | 334 | **405** | 403 |
| **64** | 1,502 | **1,518** | 981 | 1,012 | 981 | 914 | 1,302 | 1,287 |
| **256** | 2,798 | 2,834 | 7,578 | 7,545 | 7,578 | 7,605 | 7,585 | **7,602** |

### Prefill Throughput (M tokens/s)

| N | SeqLen | Triton v4 | Triton v5 | v5/v4 Speedup |
|---|--------|-----------|-----------|---------------|
| 1 | 256 | 1.22 | **2.02** | 1.66x |
| 1 | 1024 | 1.24 | **2.08** | **1.68x** |
| 4 | 512 | 4.67 | **7.17** | 1.54x |
| 16 | 128 | 11.33 | 11.46 | 1.01x |
| 32 | 64 | **13.98** | 12.76 | 0.91x |

### Version Features

| Version | Framework | Key Features | Best Use Case |
|---------|-----------|--------------|---------------|
| v4 | Triton | V-Slice parallel | Baseline |
| **v5** | Triton | **Software pipelining** | Single-seq long ctx |
| v6 | CUDA | Medium batch opt | Batch 32-256 |
| v7 | CUDA | Small batch opt | Batch 1-16 |
| v8 | CUDA | Large batch opt | Batch 256+ |
| **v9** | CuTe C++ | SMEM swizzle + cp.async | All batches |
| **v10** | CuTe C++ | **FP8/BF16 state quant** | Memory-bound |
| v11 | CuTe C++ | Token-level pipeline | Single-seq prefill |
| PTX | PTX asm | mma.sync + fast-math | Max performance |

### Historical Best Records

| Metric | Version | Value |
|--------|---------|-------|
| **Decode Peak BW** | CuTe v10 @ B=256 | **7,602 GB/s (95% B200)** |
| **Prefill Peak** | Triton v5 @ N=32 | **14.13 M tok/s** |
| **Max Speedup** | Triton v5 @ N=16,L=4096 | **1886x vs ref** |
| **State Compression** | v10 FP8 | **4x (64KB→16KB)** |

---

## Optimization Targets

### Current Baseline (Triton v5)

```
Decode:  batch=256 → 2,798 GB/s (35% B200)
Prefill: N=16 → 167 GB/s (2% B200)
```

### Target with CuTe C++/PTX Optimization

```
Decode:  batch=256 → 7,600 GB/s (95% B200) ✓ Achieved
Prefill: N=16 → 1,000+ GB/s (12.5% B200) — TODO
```

---

## Framework Comparison

| Framework | Decode Peak | Prefill Peak | Pros | Cons |
|-----------|-------------|--------------|------|------|
| **Triton** | 1,518 GB/s | 167 GB/s | Easy, auto-tuning | Ceiling limited |
| **CuTe C++** | **7,602 GB/s** | TBD | Swizzle, TMA, Tensor Core | Complex |
| **PTX** | TBD | TBD | Ultimate control | Hard to maintain |

---

## Active Optimization Files

> **File Freeze Policy**: All future optimizations modify only these 6 files.

| Path | Type | Current Features |
|------|------|------------------|
| `src/kernels/cute_cpp/gdn_decode_v9.cuh` | CuTe C++ | SMEM swizzle, cp.async |
| `src/kernels/cute_cpp/gdn_decode_v10.cuh` | CuTe C++ | BF16/FP8/FP4 state quantization |
| `src/kernels/cute_cpp/gdn_prefill_v9.cuh` | CuTe C++ | Chunking, SMEM |
| `src/kernels/cute_cpp/gdn_prefill_v10.cuh` | CuTe C++ | TiledMMA structure |
| `src/kernels/ptx/gdn_decode_ptx.cuh` | PTX | ex2.approx, BF16/FP8/FP4 |
| `src/kernels/ptx/gdn_prefill_ptx.cuh` | PTX | **mma.sync.aligned, TMA** |

---

## Roofline Analysis

### Decode (Memory-Bound, AI=1)

```
B200: 8 TB/s memory, 74.45 TFLOPS
Ridge Point: 74.45 / 8 = 9.3 FLOP/byte

GDN Decode: [128×128] × [128] = 128×128×2 FLOP, 128×128×4 bytes
AI = 32K / 64K = 0.5 FLOP/byte → Memory-bound

Optimization: Maximize bandwidth utilization via SMEM swizzle
```

### Prefill (Can Be Compute-Bound with Chunking)

```
CHUNK_SIZE | AI (FLOP/byte) | Status on B200
-----------|----------------|---------------
    1      |      1.0       | Memory-bound
    8      |      8.0       | Near ridge (9.3)
   16      |     16.0       | Compute-bound!

Optimization: Use WGMMA/TiledMMA when AI > 9.3
```

---

## Benchmark Commands

```bash
# Full benchmark (decode + prefill)
modal run benchmarks/bench_modal.py

# Decode only
modal run benchmarks/bench_modal.py --kernel decode

# Prefill only  
modal run benchmarks/bench_modal.py --kernel prefill

# Compare with Python baseline
modal run benchmarks/bench_modal.py --compare

# Correctness tests
modal run tests/test_correctness.py

# Quantization accuracy benchmark
modal run benchmarks/bench_quantization_perf.py
```

---

## Latest Results (2026-03-28)

### Full Benchmark Suite

| Kernel | Workloads | Success | Avg Speedup | Peak Speedup |
|--------|-----------|---------|-------------|--------------|
| **Decode** | 54 | 100% ✅ | **470.32x** | 1273.67x |
| **Prefill** | 100 | 100% ✅ | **256.38x** | 874.59x |

### Prefill Implementation Details

The prefill kernel uses a **fallback chain** to handle Triton compilation issues with dynamic loops:

```
v5 Triton (software pipelining) → v4 Triton (simple) → PyTorch fallback
```

This ensures 100% correctness across all workloads while maximizing performance.

### Comparison with CUDA Backend

| Kernel | Backend | Success | Avg Speedup |
|--------|---------|---------|-------------|
| Decode | Triton | 100% | 470.32x |
| Prefill | Triton (fallback) | 100% | 256.38x |
| Prefill | CUDA | 100% | 203.61x |

**Triton with fallback outperforms pure CUDA by 1.26x on prefill.**

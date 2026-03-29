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
| 1 | 0.0455 | 23 | 0.5 MB | 0.3% |
| 4 | 0.0449 | 93 | 2.0 MB | 1.2% |
| 16 | 0.0447 | 375 | 8.0 MB | 4.7% |
| 64 | 0.0447 | 1,502 | 32.0 MB | 18.8% |
| **256** | 0.0959 | **2,798** | 128.0 MB | **35.0%** |

### Prefill Performance (Triton Baseline)

| Config | N | SeqLen | T | Time (ms) | BW (GB/s) | M tok/s |
|--------|---|--------|---|-----------|-----------|---------|
| Short | 4 | 64 | 256 | 0.085 | 62 | 3.02 |
| Medium | 4 | 256 | 1024 | 0.225 | 37 | 4.55 |
| Long | 4 | 1024 | 4096 | 0.786 | 27 | 5.21 |
| **Many short** | 16 | 64 | 1024 | 0.125 | **167** | **8.18** |

---

## Historical Best (Previous Benchmark)

| Batch | Triton v5 | CUDA v7 | CUDA v8 | **CuTe v9** | CuTe v10 | Best |
|-------|-----------|---------|---------|-------------|----------|------|
| **1** | 24 GB/s | 25 GB/s | 25 GB/s | **27 GB/s** | 26 GB/s | **v9** |
| **16** | 386 GB/s | 352 GB/s | 334 GB/s | **405 GB/s** | 403 GB/s | **v9** |
| 64 | **1,518 GB/s** | 981 GB/s | 914 GB/s | 1,302 GB/s | 1,287 GB/s | **Triton** |
| **256** | 2,834 GB/s | 7,578 GB/s | 7,605 GB/s | 7,585 GB/s | **7,602 GB/s** | **v10** |

**Peak**: CuTe v9/v10 at batch=256 achieves **7,600 GB/s (95% of B200 peak)**

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

> **File Freeze Policy**: All future optimizations modify only these 4 files.

| Path | Type | Current Features |
|------|------|------------------|
| `src/kernels/cute_cpp/gdn_decode_v9.cuh` | CuTe C++ | SMEM swizzle, Swizzle<3,3,3> |
| `src/kernels/cute_cpp/gdn_prefill_v9.cuh` | CuTe C++ | Chunking, SMEM |
| `src/kernels/ptx/gdn_decode_ptx.cuh` | PTX | ex2.approx, shfl.sync |
| `src/kernels/ptx/gdn_prefill_ptx.cuh` | PTX | Fast math assembly |

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
# Decode benchmark (Triton baseline)
modal run scripts/bench_all_versions.py --versions v5 --batches 1,4,16,64,256

# Prefill benchmark
modal run scripts/bench_prefill_all.py

# Build and benchmark CUDA kernels (requires volume setup)
modal run scripts/build_cuda.py
modal run scripts/bench_cuda_real.py
```

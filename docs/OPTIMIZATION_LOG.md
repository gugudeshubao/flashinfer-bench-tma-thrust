# GDN Kernel Optimization Log

> This document tracks our optimization iterations, decisions, and next steps.

---

## File Freeze Policy

**All future optimizations modify only these 4 files:**

| Path | Framework | Purpose |
|------|-----------|---------|
| `src/kernels/cute_cpp/gdn_decode_v9.cuh` | CuTe C++ | Decode kernel (primary) |
| `src/kernels/cute_cpp/gdn_prefill_v9.cuh` | CuTe C++ | Prefill kernel (primary) |
| `src/kernels/ptx/gdn_decode_ptx.cuh` | PTX | Decode kernel (fallback) |
| `src/kernels/ptx/gdn_prefill_ptx.cuh` | PTX | Prefill kernel (fallback) |

**Baseline Commit**: `188dc04` (2026-03-28)

---

## Iteration 0: Baseline Establishment (2026-03-28)

### Benchmark Results (Triton v5 on B200)

**Decode:**
| Batch | Time (ms) | BW (GB/s) | B200 Util. |
|-------|-----------|-----------|------------|
| 1 | 0.0455 | 23 | 0.3% |
| 4 | 0.0449 | 93 | 1.2% |
| 16 | 0.0447 | 375 | 4.7% |
| 64 | 0.0447 | 1,502 | 18.8% |
| **256** | 0.0959 | **2,798** | **35.0%** |

**Prefill:**
| Config | N | SeqLen | Time (ms) | BW (GB/s) | M tok/s |
|--------|---|--------|-----------|-----------|---------|
| Short | 4 | 64 | 0.085 | 62 | 3.02 |
| Medium | 4 | 256 | 0.225 | 37 | 4.55 |
| Long | 4 | 1024 | 0.786 | 27 | 5.21 |
| **Many** | 16 | 64 | 0.125 | **167** | **8.18** |

### Historical Best (CuTe C++ compiled)
- **Decode batch=256**: 7,602 GB/s (95% B200 peak)
- **Decode batch=1**: 27 GB/s (CuTe v9 best)

### Current Framework Status

| Framework | Decode | Prefill | Notes |
|-----------|--------|---------|-------|
| Triton v5 | 2,798 GB/s | 167 GB/s | Baseline |
| CuTe C++ v9 | 7,602 GB/s | TBD | Needs CUDA compile |
| PTX | TBD | TBD | Needs CUDA compile |

---

## Optimization Strategy

### Dual-Path Approach

```
┌─────────────────────────────────────────────────────────────┐
│  CuTe C++ (Primary)              PTX (Fallback)             │
│  ─────────────────              ──────────────              │
│  v9: SMEM Swizzle               decode_ptx: fast math       │
│  - Swizzle<3,3,3>               - ex2.approx, lg2.approx    │
│  - TiledMMA                     - shfl.sync.bfly            │
│       ↓                               ↓                     │
│  Phase 1: TMA Prefetch          Phase 1: cp.async.bulk      │
│       ↓                               ↓                     │
│  Phase 2: WGMMA (prefill)       Phase 2: tcgen05.mma        │
│       ↓                               ↓                     │
│  Phase 3: Multi-stage           Phase 3: Software pipeline  │
└─────────────────────────────────────────────────────────────┘
```

### Optimization Phases

| Phase | Focus | CuTe C++ | PTX | Target |
|-------|-------|----------|-----|--------|
| 1 | Memory latency | TMA async prefetch | cp.async.bulk | Hide latency |
| 2 | Compute density | WGMMA for prefill | tcgen05.mma | AI > 9.3 |
| 3 | Overlap | Multi-stage pipeline | Software pipeline | 100% util |
| 4 | Thread util | Warp specialization | Explicit scheduling | Reduce idle |

---

## Analysis: Why Triton Baseline is 35% B200?

### Decode Bottlenecks

1. **Small batch kernel launch overhead**
   - batch=1-16: Kernel launch ~45μs dominates
   - Solution: Persistent kernel or CUDA Graph

2. **SMEM bank conflicts**
   - State [128×128] access pattern causes conflicts
   - Solution: Swizzle (already in CuTe v9)

3. **Sequential token processing**
   - Each token depends on previous state
   - Solution: Cannot parallelize across tokens

### Prefill Bottlenecks

1. **Low arithmetic intensity (AI=1)**
   - Per-token mat-vec is memory-bound
   - Solution: Chunking (AI=8 with CHUNK_SIZE=8)

2. **No Tensor Core utilization**
   - Mat-vec [128×128]×[128] has N=1 (TC requires N≥16)
   - Solution: Chunk multiple tokens → [128×128]×[128×8]

---

## Next Steps (Iteration 1)

### Priority 1: Decode TMA Prefetch
- [ ] Add TMA async prefetch to `gdn_decode_v9.cuh`
- [ ] Add cp.async.bulk to `gdn_decode_ptx.cuh`
- [ ] Benchmark latency improvement

### Priority 2: Prefill Chunking + WGMMA
- [ ] Implement CHUNK_SIZE=8 in `gdn_prefill_v9.cuh`
- [ ] Add tcgen05.mma path in `gdn_prefill_ptx.cuh`
- [ ] Benchmark throughput improvement

### Priority 3: Correctness Validation
- [ ] Fix prefill correctness (current: nan/large diff)
- [ ] Verify delta rule implementation

---

## Commit History

| Commit | Date | Description |
|--------|------|-------------|
| `188dc04` | 2026-03-28 | feat: add CuTe C++ and PTX kernel implementations |
| (next) | TBD | Phase 1: TMA prefetch optimization |

---

## References

- [CuTe Documentation](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [B200 Architecture](https://developer.nvidia.com/blog/nvidia-blackwell-architecture-technical-deep-dive/)

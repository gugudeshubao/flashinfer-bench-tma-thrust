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

### Priority 1: Decode TMA Prefetch ✅ DONE
- [x] Add cp.async prefetch to `gdn_decode_v9.cuh`
  - Added `cp_async_ca()`, `cp_async_commit_group()`, `cp_async_wait_group<N>()`
  - State loading now uses async prefetch
- [x] Add cp.async to `gdn_decode_ptx.cuh`
  - Added `ptx_cp_async_ca()`, `ptx_cp_async_commit()`, `ptx_cp_async_wait<N>()`
  - State loading uses async prefetch
- [ ] Benchmark latency improvement (requires Modal B200)

### Priority 2: Prefill Chunking + WGMMA
- [ ] Implement CHUNK_SIZE=8 in `gdn_prefill_v9.cuh`
- [ ] Add tcgen05.mma path in `gdn_prefill_ptx.cuh`
- [ ] Benchmark throughput improvement

### Priority 3: Correctness Validation
- [ ] Fix prefill correctness (current: nan/large diff)
- [ ] Verify delta rule implementation

---

## Iteration 1: cp.async Prefetch (2026-03-28)

### Changes Made

**CuTe C++ (`gdn_decode_v9.cuh`):**
```cpp
// Added cp.async primitives
__device__ __forceinline__ void cp_async_ca(void* smem_ptr, const void* gmem_ptr);
__device__ __forceinline__ void cp_async_commit_group();
template<int N> __device__ __forceinline__ void cp_async_wait_group();

// State loading now uses async prefetch
for (int i = tid; i < BLOCK_V * V9_D; i += V9_THREADS) {
    cp_async_ca(&s_state[...], &state_ptr[...]);
}
cp_async_commit_group();
cp_async_wait_group<0>();
```

**PTX (`gdn_decode_ptx.cuh`):**
```cpp
// Added PTX cp.async primitives
__device__ __forceinline__ void ptx_cp_async_ca(void* smem_ptr, const void* gmem_ptr);
__device__ __forceinline__ void ptx_cp_async_commit();
template<int N> __device__ __forceinline__ void ptx_cp_async_wait();

// State loading uses async prefetch
for (int i = tid; i < BLOCK_V * D; i += num_threads) {
    ptx_cp_async_ca(&state_smem[i], &state_ptr[...]);
}
ptx_cp_async_commit();
ptx_cp_async_wait<0>();
```

### Expected Benefits
- **Memory latency hiding**: cp.async overlaps data transfer with compute
- **Reduced stalls**: GPU can execute other instructions while waiting for data
- **Better utilization**: Especially important for small batch (decode)

### Benchmark Status
- [ ] Pending Modal B200 benchmark to measure actual latency improvement

---

## Commit History

| Commit | Date | Description |
|--------|------|-------------|
| `188dc04` | 2026-03-28 | feat: add CuTe C++ and PTX kernel implementations |
| `a892d6c` | 2026-03-28 | docs: update performance benchmarks and create optimization log |
| (next) | 2026-03-28 | **Iteration 1: cp.async prefetch for decode kernels** |

---

## References

- [CuTe Documentation](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [B200 Architecture](https://developer.nvidia.com/blog/nvidia-blackwell-architecture-technical-deep-dive/)

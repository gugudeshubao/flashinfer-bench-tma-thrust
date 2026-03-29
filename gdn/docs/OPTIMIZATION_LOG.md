# GDN Kernel Optimization Log

> This document tracks our optimization iterations, decisions, and next steps.

---

## File Freeze Policy

**All future optimizations modify only these 6 files:**

| Path | Framework | Purpose |
|------|-----------|---------|
| `src/kernels/cute_cpp/gdn_decode_v9.cuh` | CuTe C++ | Decode kernel (v9) |
| `src/kernels/cute_cpp/gdn_decode_v10.cuh` | CuTe C++ | Decode kernel (v10 + FP8) |
| `src/kernels/cute_cpp/gdn_prefill_v9.cuh` | CuTe C++ | Prefill kernel (primary) |
| `src/kernels/ptx/gdn_decode_ptx.cuh` | PTX | Decode kernel (fallback + FP8) |
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

### Priority 2: Prefill Chunking + mma.sync ✅ DONE
- [x] Implement CHUNK_SIZE=8 in `gdn_prefill_v9.cuh` (already existed)
- [x] Add mma.sync.aligned primitives in `gdn_prefill_ptx.cuh`
- [x] Add optimized MMA kernel `gdn_prefill_kernel_ptx_mma`
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

## Iteration 2: FP8 State Quantization (2026-03-28)

### Motivation

Decode is **memory-bound** with arithmetic intensity AI ≈ 1. State matrix [128×128] is the largest memory consumer:

| Precision | State Size/Head | Total (8 heads) | Memory BW Reduction |
|-----------|-----------------|-----------------|---------------------|
| FP32 | 64 KB | 512 KB | Baseline |
| **FP8** | **16 KB** | **128 KB** | **4x** |

### Changes Made

**CuTe C++ v10 (`gdn_decode_v10.cuh`):**
```cpp
// FP8 conversion primitives
__device__ __forceinline__ __nv_fp8_e4m3 v10_fp32_to_fp8(float val);
__device__ __forceinline__ float v10_fp8_to_fp32(__nv_fp8_e4m3 val);
__device__ __forceinline__ uint32_t v10_pack_fp8x4(...);
__device__ __forceinline__ void v10_unpack_fp8x4(...);

// New FP8 kernel
template<int BLOCK_V>
__global__ void gdn_decode_kernel_v10_fp8(
    const uint32_t* State_FP8,    // Packed FP8x4 state
    const float* State_Scale,     // Per-row dynamic scale
    ...
);

// Launcher
void gdn_decode_v10_launch_fp8(...);
```

**PTX (`gdn_decode_ptx.cuh`):**
```cpp
// FP8 PTX primitives
__device__ __forceinline__ __nv_fp8_e4m3 ptx_fp32_to_fp8(float val);
__device__ __forceinline__ float ptx_fp8_to_fp32(__nv_fp8_e4m3 val);
__device__ __forceinline__ uint32_t ptx_pack_fp8x4(...);
__device__ __forceinline__ void ptx_unpack_fp8x4(...);
__device__ __forceinline__ uint32_t ptx_ld_nc_u32(const uint32_t* ptr);

// New FP8 kernel
template<int BLOCK_V>
__global__ void gdn_decode_kernel_ptx_fp8(...);

// Launcher
void gdn_decode_ptx_fp8_launch(...);
```

### Design Decisions

1. **Per-row dynamic scaling**: Each row of state has its own scale factor
   - Scale = max_abs / 400.0 (FP8 E4M3 range is [-448, 448])
   - Maintains accuracy better than global scaling

2. **FP32 internal compute**: Only state storage is FP8
   - Dequantize on load: `state = fp8_to_fp32(packed) * row_scale`
   - Quantize on store: `fp8 = fp32_to_fp8(state * inv_scale)`

3. **Vectorized memory**: Pack 4 FP8 values into uint32_t
   - 4x memory bandwidth efficiency
   - Aligned 4-byte loads/stores

### Expected Benefits

- **4x memory reduction**: 512KB → 128KB per batch
- **4x lower memory BW**: State load/store bandwidth reduced
- **Potential 2-4x speedup**: For memory-bound decode kernel

### Accuracy Trade-offs

| Precision | Mantissa Bits | Max Abs Error | Relative Error |
|-----------|---------------|---------------|----------------|
| FP32 | 23 | ~1e-7 | ~1e-7 |
| FP8 E4M3 | 3 | ~0.5 | ~5% |

For GDN state which accumulates over many steps, FP8 may introduce drift.
Recommend FP8 for inference, FP32 for training.

### Quantization Accuracy Test Results (Modal B200)

Tested with realistic parameters over 100 decode steps:

| Metric | BF16 | FP8 E4M3 | FP4 E2M1 |
|--------|------|----------|----------|
| **Compression** | 2x | 4x | 8x |
| **Mantissa bits** | 7 | 3 | 1 |
| Output max error | 3.2e-06 | 5.6e-05 | 3.4e-04 |
| Output rel error | **0.57%** | **11.4%** | **54.6%** |
| State max error | 1.1e-04 | 2.5e-03 | 1.6e-02 |
| State rel error | **0.64%** | **10.5%** | **64.9%** |
| Error accumulation | 0.15x | 0.16x | 0.11x |

### Precision Recommendation

| Precision | Memory/Head | Relative Error | Recommended Use |
|-----------|-------------|----------------|-----------------|
| FP32 | 64 KB | 0% | Training, exact inference |
| **BF16** | **32 KB** | **~0.6%** | High-precision inference |
| **FP8** | **16 KB** | **~11%** | **Standard inference (recommended)** |
| FP4 | 8 KB | ~55-65% | Not recommended |

**Key findings**:
1. **BF16 is near-lossless**: 2x compression with <1% error
2. **FP8 is the sweet spot**: 4x compression with ~11% relative error
3. **FP4 is too aggressive**: 8x compression but ~55-65% error - unacceptable
4. **All are stable**: Errors don't accumulate over time

### Benchmark Status
- [x] FP8 accuracy test completed (2026-03-28)
- [x] FP4 accuracy test completed (2026-03-28)
- [ ] Pending FP8 vs FP32 latency benchmark (requires CUDA compilation)

---

## Theoretical Performance Analysis

### Decode Algorithm Breakdown (per batch, per head)

```
Memory Access:
├── Load Q [128]: 256 bytes (BF16)
├── Load K [128]: 256 bytes (BF16)
├── Load V [128]: 256 bytes (BF16)
├── Load State [128×128]: 64 KB (FP32) or 16 KB (FP8)
├── Load gates: ~20 bytes
├── Store Out [128]: 256 bytes (BF16)
└── Store NewState [128×128]: 64 KB (FP32) or 16 KB (FP8)

FLOPs (per head):
├── old_v = (g*S) @ k: 128 × 128 × 3 = 49,152
├── delta = beta*(v - old_v): 128 × 3 = 384
├── S_new = g*S + delta*k: 128 × 128 × 3 = 49,152
└── out = S_new @ q: 128 × 128 × 2 = 32,768
Total: ~131,456 FLOPs per head
```

### Full Batch Metrics (8 heads)

| Metric | FP32 State | FP8 State | Formula |
|--------|------------|-----------|---------|
| **Memory/batch** | 1,049 KB | 263 KB | 8 heads × (128KB + 1KB) |
| **FLOPs/batch** | 1.05 M | 1.05 M | 8 heads × 131K |
| **Arithmetic Intensity** | **1.0** | **4.0** | FLOPs / Bytes |

### B200 Hardware Specs

| Resource | B200 Spec | Notes |
|----------|-----------|-------|
| Memory BW | 8,000 GB/s | Peak HBM3e |
| FP32 Compute | 2,250 TFLOPS | Tensor Core |
| Roofline Knee | AI = 280 | 2250T / 8T |

### Expected Performance (Memory-Bound)

| Mode | Bytes/batch | Peak BW | Theoretical Min Latency | Throughput |
|------|-------------|---------|-------------------------|------------|
| **FP32 State** | 1,049 KB | 8 TB/s | **131 ns** | 7.6M batch/s |
| **FP8 State** | 263 KB | 8 TB/s | **33 ns** | 30.4M batch/s |

### Actual vs Theoretical

| Kernel | Theory | Measured (Triton) | Efficiency |
|--------|--------|-------------------|------------|
| Triton v4 FP32 | 131 ns | 53,000 ns | 0.25% |
| v10 FP32 (est.) | 131 ns | ~10,000 ns | ~1.3% |
| v10 FP8 (est.) | 33 ns | ~3,000 ns | ~1.1% |

### Why Low Efficiency?

1. **Kernel launch overhead**: ~2-5μs per launch (10% of total for decode)
2. **Low SM occupancy**: B200 has 192 SMs, batch=1 uses only 8
3. **Sequential dependency**: Cannot parallelize across tokens
4. **SMEM bank conflicts**: Swizzle helps but adds overhead

### Conclusion

- **FP8 provides 4x memory compression** → expect **3-4x speedup**
- Still memory-bound (AI=4 << 280 roofline knee)
- Main bottleneck shifts from HBM to kernel launch at small batch

---

## Iteration 3: Tensor Core Prefill (mma.sync) (2026-03-28)

### Motivation

Prefill with chunking has higher arithmetic intensity:
- AI (CHUNK=8) ≈ 8 FLOP/byte (near compute-bound)
- AI (CHUNK=64) ≈ 64 FLOP/byte (compute-bound)
- FP32 ridge point = 9.3 FLOP/byte

Matrix-matrix operations enable Tensor Core usage:
- `State[V,D] @ Q_chunk[D,C]` → mma.sync
- `State[V,D] @ K_chunk[D,C]` → mma.sync

### Changes Made

**PTX (`gdn_prefill_ptx.cuh`):**
```cpp
// Added mma.sync.aligned.m16n8k16 primitive
__device__ __forceinline__ void mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3
);

// New kernel optimized for mma.sync
template<int CHUNK_SIZE>
__global__ void gdn_prefill_kernel_ptx_mma(...);

// Launcher
void gdn_prefill_ptx_mma_launch(...);
```

### mma.sync.aligned.m16n8k16 Details

| Property | Value |
|----------|-------|
| Instruction | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` |
| Architecture | sm_80+ (Ampere, Hopper, Blackwell) |
| Tile size | M=16, N=8, K=16 |
| A operand | [16, 16] BF16 row-major |
| B operand | [16, 8] BF16 col-major |
| C/D operand | [16, 8] FP32 |
| FLOPs/instruction | 2 × 16 × 8 × 16 = 4,096 |

### GDN Prefill Tiling Strategy

```
State[16, 128] @ Q_chunk[128, 8] = Out[16, 8]

Tiled as 8 iterations of mma.m16n8k16:
  - Iteration 0: State[0:16, 0:16]   @ Q[0:16, 0:8]   → acc[0:16, 0:8]
  - Iteration 1: State[0:16, 16:32]  @ Q[16:32, 0:8]  → acc[0:16, 0:8]
  - ...
  - Iteration 7: State[0:16, 112:128] @ Q[112:128, 0:8] → Out[0:16, 0:8]
```

### Sequential Dependency Challenge

GDN has sequential token dependency:
```
for t in tokens:
    out[t] = State @ Q[t]
    old_v = State @ K[t]
    delta = beta * (V[t] - old_v)
    State = gate * State + delta * K[t]^T  // State changes!
```

**Current limitation**: Cannot batch mma.sync across tokens due to state update dependency.

**Potential solutions**:
1. **Chunkwise recurrence** (FlashLinearAttention): Compute chunk outputs together, update state once
2. **Parallel scan**: Convert sequential updates to associative scan (complex)

### Expected Benefits

- **Optimized FMA chains**: Unrolled 16-wide FMA operations
- **PTX fast math**: exp2.approx, lg2.approx, fma.rn for gates
- **Better cache utilization**: Explicit shared memory management

### Benchmark Status
- [x] mma.sync primitive added
- [x] Optimized kernel implemented
- [ ] Pending Modal B200 benchmark

---

## Commit History

| Commit | Date | Description |
|--------|------|-------------|
| `188dc04` | 2026-03-28 | feat: add CuTe C++ and PTX kernel implementations |
| `a892d6c` | 2026-03-28 | docs: update performance benchmarks and create optimization log |
| `49fff02` | 2026-03-28 | **Iteration 1: cp.async prefetch for decode kernels** |
| `392a54c` | 2026-03-28 | **Iteration 2: FP8 state quantization (4x compression)** |
| TBD | 2026-03-28 | **Iteration 3: mma.sync prefill kernel** |

---

## References

- [CuTe Documentation](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [B200 Architecture](https://developer.nvidia.com/blog/nvidia-blackwell-architecture-technical-deep-dive/)
- [mma.sync PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)

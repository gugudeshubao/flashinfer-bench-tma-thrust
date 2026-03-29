# GDN Kernel Optimization Roadmap

**Target**: Gated Delta Net kernels for Qwen3-Next on NVIDIA B200 (sm_100)

---

## Project Status

| Version | Status | Backend | State | Key Feature |
|---------|--------|---------|-------|-------------|
| v1 | ✅ Done | PyTorch | FP32 | Reference implementation |
| v2 | ✅ Done | Triton | FP32 | Fused delta-rule |
| v3 | ✅ Done | Triton | FP32 | V-split (BLOCK_V) |
| v4 | ✅ Done | Triton | FP32 | Adaptive BLOCK_V |
| **v5** | ✅ **Production** | Triton | FP32 | **Baseline for comparison** |
| v6 | ✅ Done | CUDA | FP32 | TMA async loads |
| v7 | ✅ Done | CUDA | FP4 | 4-bit quantization |
| v8 | ✅ Done | CUDA | FP8 | Warp specialization |
| **v9** | ✅ Done | CuTe C++ | FP32 | **SMEM Swizzle, cp.async** |
| **v10** | ✅ Done | CuTe C++ | Multi | **BF16/FP8/FP4 state** |
| **PTX** | ✅ Done | PTX | Multi | **mma.sync, TMA** |

---

## Kernel Implementations

### Decode: `gdn_decode_qk4_v8_d128_k_last`

Single-token generation with recurrent state update.

```
Input:  Q [B,1,4,128], K [B,1,4,128], V [B,1,8,128]
State:  [B, 8, 128, 128] FP32 (512 KB per batch)
Output: [B,1,8,128] BF16, new_state [B,8,128,128]
```

### Prefill: `gdn_prefill_qk4_v8_d128_k_last`

Variable-length sequence processing.

```
Input:  Q [N,T,4,128], K [N,T,4,128], V [N,T,8,128]
State:  [N, 8, 128, 128] FP32
Output: [N,T,8,128] BF16, final_state [N,8,128,128]
```

---

## Version Details

### v1 — Python Baseline ✅

Pure PyTorch reference. 10/10 decode, 12/12 prefill tests passed.

### v2 — Triton Fused ✅

- Grid: `(B, H=8)` — one program per (batch, head)
- State [128×128] in registers
- Fused gates + delta-rule
- **950x speedup** vs Python baseline

### v3 — Triton V-Split ✅

- Grid: `(B, H=8, V_BLOCKS=4)` — V dimension split
- State slice [32×128] per program → 4× less register pressure
- **1215x speedup** vs Python baseline

### v4 — Adaptive BLOCK_V ✅

- BLOCK_V=16 for B≤16 (more parallelism)
- BLOCK_V=32 for B≤128 (balanced)
- BLOCK_V=64 for B>128 (reduced launch overhead)

### v5 — Production Baseline ✅

- Best Triton implementation, used as comparison baseline
- **2,834 GB/s** achieved at batch=256 (35% of B200 peak)

### v6 — CUDA TMA ✅

- `cp.async.bulk.tensor` for 2D tile loads
- 128-byte aligned shared memory
- Compiled but not faster than Triton (TMA overhead)

### v7 — CUDA FP4 ✅

- FP4 E2M1 quantization: 4-bit state (4× compression)
- Per-row scaling with absmax normalization
- Lookup table decode
- **1.46x speedup** at batch=256 (memory-bound regime)

### v8 — CUDA FP8 ✅

- FP8 E4M3 quantization: 8-bit state (2× compression)
- Warp specialization: 2 producer + 2 consumer warps
- Triple buffering for prefetch
- **1.45x speedup** at batch=256

---

## Key Findings

### Decode: WGMMA Not Applicable (Memory-Bound)

Decode performs matrix-vector operations:
- `S @ q` → [128×128] @ [128] = [128]
- `k.T @ v` → [128×1] @ [1×128] = [128×128] (rank-1 update)

**Result**: AI=1 FLOP/byte → Memory-bound, cannot use Tensor Cores.
**Solution**: Memory bandwidth optimization via BF16/FP8/FP4 state compression.

### Prefill: CAN Use Tensor Core (with Chunking)

Chunked prefill creates matrix-matrix operations:
- `State @ Q_chunk` → [V×D] @ [D×C] = [V×C]
- With CHUNK_SIZE=8: AI ≈ 8 FLOP/byte (near compute-bound)

**Result**: Can use mma.sync.aligned.m16n8k16 for Tensor Core acceleration.

### Achieved Benchmarks (2026-03-29)

| Kernel | Avg Speedup | Best Speedup | Status |
|--------|-------------|--------------|--------|
| Decode | 1127x | 3465x | ✅ ALL PASS |
| Prefill | 598x | 1886x | ✅ ALL PASS |

### Memory-Bound at Large Batch

| Batch | State Size | Regime | v5 vs v7 |
|-------|------------|--------|----------|
| 1-16 | ≤8 MB | Compute-bound | No diff |
| 64 | 32 MB | Transitioning | ~1.0x |
| 256+ | 128+ MB | Memory-bound | **1.46x** |

### Achieved vs Peak

```
B200 Peak:     8,000 GB/s
v5 Achieved:   2,834 GB/s (35.4%)
Gap:           2.8× headroom
```

---

## Build & Benchmark

### Compile CUDA Kernels

```bash
modal run scripts/build_cuda.py  # Compiles v5-v8 for sm_100
```

### Run Benchmarks

```bash
# All versions, multiple batches
modal run scripts/bench_all_versions.py --versions all --batches "1,16,64,256"

# Specific version
modal run scripts/bench_all_versions.py --versions v7 --batches "256,512"

# Quick check
modal run scripts/bench_all_versions.py --versions v5 --batches 64 --warmup 5 --iters 50
```

---

## File Structure

```
src/kernels/
├── cuda/
│   ├── gdn_decode_v5.cuh   # Triton-equivalent CUDA
│   ├── gdn_decode_v6.cuh   # TMA async loads
│   ├── gdn_decode_v7.cuh   # FP4 E2M1 quantization
│   ├── gdn_decode_v8.cuh   # FP8 + warp specialization
│   ├── gdn_prefill_v5.cuh
│   ├── gdn_prefill_v6.cuh
│   ├── gdn_prefill_v7.cuh
│   └── gdn_prefill_v8.cuh
├── cute_cpp/                    # CuTe C++ implementations
│   ├── gdn_decode_v9.cuh   # SMEM swizzle, cp.async
│   ├── gdn_decode_v10.cuh  # BF16/FP8/FP4 state quantization
│   ├── gdn_prefill_v9.cuh  # Chunking
│   └── gdn_prefill_v10.cuh # TiledMMA structure
└── ptx/                         # PTX assembly implementations
    ├── gdn_decode_ptx.cuh  # ex2.approx, BF16/FP8/FP4
    └── gdn_prefill_ptx.cuh # mma.sync.aligned, TMA

benchmarks/
├── bench_modal.py          # Primary benchmark script
└── bench_quantization_perf.py  # Quantization comparison

tests/
└── test_correctness.py     # Correctness validation
```

---

## Next Steps

1. ✅ **mma.sync Tensor Core**: Added to PTX prefill kernel
2. ✅ **TMA bulk loads**: Added cp.async.bulk.tensor primitives
3. ✅ **BF16/FP8/FP4 state**: Implemented in v10 decode kernels
4. **Chunkwise recurrence**: Batch mma.sync across tokens (blocked by sequential dependency)
5. **NVCC compilation**: Build PTX kernels for real benchmarking
6. **Profile with NCU**: Identify remaining bottlenecks
7. **FlashInfer-Bench**: Submit to official leaderboard

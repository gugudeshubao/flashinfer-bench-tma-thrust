# Roofline Analysis — GDN Kernels on B200

## Hardware: NVIDIA B200 (Blackwell, sm_100)

### Core Specifications

| Specification | Value |
|---------------|-------|
| Architecture | Blackwell (sm_100) |
| CUDA Cores | 16,896 |
| Tensor Cores | 528 (5th Gen) |
| Boost Clock | 1.98 GHz |
| SMs | 148 |
| Transistors | 208 billion |
| TDP | 1,000 W |
| Process | TSMC 4NP |

### Memory Specifications

| Resource | Value |
|----------|-------|
| HBM3e Capacity | 180-192 GB |
| HBM3e Bandwidth | **8 TB/s** |
| L2 Cache | 96 MB |
| Shared Memory / SM | 256 KB |

### Compute Performance

| Precision | Dense | Sparse (2:4) |
|-----------|-------|--------------|
| **FP4 Tensor** | 9 PFLOPS | 18 PFLOPS |
| **FP8 Tensor** | 4.5 PFLOPS | 9 PFLOPS |
| **FP16/BF16 Tensor** | **2.25 PFLOPS** | 4.5 PFLOPS |
| TF32 Tensor | 1.125 PFLOPS | 2.25 PFLOPS |
| FP32 (CUDA) | 74.45 TFLOPS | - |
| FP64 (CUDA) | 34 TFLOPS | - |
| FP64 Tensor | 40 TFLOPS | - |

### Ridge Points (Arithmetic Intensity)

| Precision | Peak Compute | Ridge Point |
|-----------|--------------|-------------|
| FP4 Tensor | 9 PFLOPS | 1,125 FLOP/byte |
| FP8 Tensor | 4.5 PFLOPS | 562 FLOP/byte |
| **BF16 Tensor** | 2.25 PFLOPS | **281 FLOP/byte** |
| TF32 Tensor | 1.125 PFLOPS | 140 FLOP/byte |
| **FP32 CUDA** | 74.45 TFLOPS | **9.3 FLOP/byte** |

---

## GEMM Performance Reference

### Theoretical Peak GEMM TFLOPS

| M×N×K | Precision | Peak TFLOPS | Notes |
|-------|-----------|-------------|-------|
| Large (4096³) | BF16 | 2,250 | Tensor Core saturated |
| Large (4096³) | FP8 | 4,500 | Tensor Core saturated |
| Medium (1024³) | BF16 | ~2,000 | ~90% efficiency |
| Small (256³) | BF16 | ~1,000 | Launch overhead |

### Matrix-Vector (GDN Decode)

| Shape | Operation | Peak | Achieved |
|-------|-----------|------|----------|
| [128×128] × [128] | FP32 mat-vec | 74 TFLOPS | ~7.6% util |
| [128×128] × [128] | BF16 mat-vec | N/A | Cannot use WGMMA |

**Note**: Matrix-vector cannot use Tensor Cores (WGMMA requires mat-mat).

---

## Decode — `gdn_decode_qk4_v8_d128_k_last`

**Shape**: q/k `[B,1,4,128]`, v `[B,1,8,128]`, state `[B,8,128,128]`

After GVA expansion q/k become `[B,8,128]`. One token per sequence.

### Compute (per batch element, per head)

| Op | FLOPs |
|----|-------|
| `g·S` (decay) | 2 × K×V = 32,768 |
| `old_v = k@S` | 2 × K×V = 32,768 |
| `new_v` interpolate | 3 × V = 384 |
| `S += outer(k, delta)` | 2 × K×V = 32,768 |
| `o = scale·q@S` | 2 × K×V = 32,768 |
| **Total / head** | **~131,072** |
| **Total / batch (8 heads)** | **~1.05M** |

Per token: `~1.05M FLOP × B`

### Memory (per batch element, per head)

| Tensor | Size (bytes, BF16/F32) |
|--------|----------------------|
| state read+write `[K,V]` f32 | 2 × 128×128×4 = 131,072 |
| q,k,v `[K or V]` bf16 | ~512+512+512 |
| **Total / head** | **~132 KB** |
| **Total / batch (8 heads)** | **~1.05 MB** |

### Roofline Analysis

```
Arithmetic Intensity = 1.05M FLOP / 1.05 MB = 1 FLOP/byte
FP32 Ridge Point = 74.45 TFLOPS / 8 TB/s = 9.3 FLOP/byte

AI (1) << Ridge (9.3) → MEMORY-BOUND
```

| Batch | Memory | Time @ 8TB/s | Achieved BW | Utilization |
|-------|--------|--------------|-------------|-------------|
| 1 | 1.05 MB | 0.13 µs | 27 GB/s | 0.3% |
| 64 | 67 MB | 8.4 µs | 1,518 GB/s | 19% |
| 256 | 268 MB | 33.5 µs | 7,600 GB/s | **95%** |

### Optimization strategy (decode)

1. Fuse all per-head ops into one kernel, single state read/write per launch
2. Tile over batch: grid = (B, H), block processes one (batch, head) pair
3. Keep state in registers/SMEM during the token update
4. SMEM swizzle to avoid bank conflicts (v9/v10)
5. Coalesced HBM access for state `[B, H, K, V]`

---

## Prefill — `gdn_prefill_qk4_v8_d128_k_last`

**Shape**: q/k `[T,4,128]`, v `[T,8,128]`, state `[N,8,128,128]`, varlen batched

### Compute per sequence (length L)

Each timestep: same as decode ≈ `8 × 131,072 = 1.05M FLOP`
Total for length-L sequence: `L × 1.05M FLOP`

For L=512: ~537M FLOP | For L=8192: ~8.6G FLOP

### Memory per sequence (length L)

State `[8,128,128]` f32 = 4MB read+write once (if chunked) or L times (if not chunked)

**Sequential scan (no chunking)**:
- State read/write per step: 2 × 4MB = 8MB
- Total: L × 8MB → L=512: 4GB → arithmetic intensity ≈ 1 FLOP/byte (still mem-bound)

**Chunked (chunk_size=C)**:
- Per chunk: C × 1.05M FLOP / (8MB state + C×tensor loads)
- At C=64: 67M / (8MB + ~1MB) ≈ **7.5 FLOP/byte** → approaching ridge point

### Roofline Analysis (Prefill)

```
Without chunking: AI = 1 FLOP/byte → Memory-bound
With chunking:    AI = 7.5 FLOP/byte → Near ridge point (9.3)

→ Chunked prefill CAN use Tensor Cores (WGMMA)!
```

| Mode | AI | Bottleneck | Can Use WGMMA? |
|------|-----|-----------|----------------|
| Sequential | 1 | Memory | No |
| Chunked (C=64) | 7.5 | Near ridge | **Yes** |
| Chunked (C=128) | ~12 | Compute | **Yes** |

### Optimization strategy (prefill)

1. **Chunked recurrence**: process C=64 tokens per SMEM tile, keep state in registers
2. **WGMMA**: Use Tensor Cores for S@Q matrix multiply (C tokens at once)
3. **TMA**: Async bulk loads for q,k,v tiles
4. **Vectorized loads**: 128-bit LDG for state matrix rows
5. **Future**: Flash-linear-attention style online algorithm for better cache reuse

---

## Summary: Decode vs Prefill

| Aspect | Decode | Prefill (Chunked) |
|--------|--------|-------------------|
| Arithmetic Intensity | 1 FLOP/byte | 7.5 FLOP/byte |
| Bottleneck | **Memory BW** | Near Compute |
| Can use WGMMA? | No (mat-vec) | **Yes** (mat-mat) |
| Primary Optimization | SMEM swizzle, BW | Chunking, WGMMA |
| Achieved Efficiency | 95% BW | TBD |

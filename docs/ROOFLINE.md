# Roofline Analysis — GDN Kernels on B200

## Hardware: NVIDIA B200 (sm100)

| Resource | Peak |
|----------|------|
| BF16 tensor core (WGMMA) | ~2.25 PFLOPS |
| FP32 | ~100 TFLOPS |
| HBM3e bandwidth | ~8 TB/s |
| Ridge point (BF16) | ~281 FLOP/byte |
| L2 cache | 96 MB |
| Shared memory / SM | 256 KB |

---

## Decode — `gdn_decode_qk4_v8_d128_k_last`

**Shape**: q/k `[B,1,4,128]`, v `[B,1,8,128]`, state `[B,8,128,128]`

After GVA expansion q/k become `[B,8,128]`. One token per sequence.

### Compute (per batch element, per head)

| Op | FLOPs |
|----|-------|
| `g·S` (decay) | 2 × K×V = 32768 |
| `old_v = k@S` | 2 × K×V = 32768 |
| `new_v` interpolate | 3 × V = 384 |
| `S += outer(k, delta)` | 2 × K×V = 32768 |
| `o = scale·q@S` | 2 × K×V = 32768 |
| **Total / head** | **~131072** |
| **Total / batch (8 heads)** | **~1.05M** |

Per token: `~1.05M FLOP × B`

### Memory (per batch element, per head)

| Tensor | Size (bytes, BF16/F32) |
|--------|----------------------|
| state read+write `[K,V]` f32 | 2 × 128×128×4 = 131072 |
| q,k,v `[K or V]` bf16 | ~512+512+512 |
| **Total / head** | **~132 KB** |
| **Total / batch (8 heads)** | **~1.05 MB** |

**Arithmetic intensity ≈ 1.05M / 1.05MB ≈ 1 FLOP/byte**
→ Extremely memory-bound. Target: maximize HBM bandwidth (~8 TB/s).

At 8 TB/s: theoretical min latency per token = 1.05 MB / 8e12 B/s ≈ **0.13 µs/token**
At batch=512 (large decode): 512 × 1.05MB state = 537 MB → 537MB / 8TB/s ≈ **67 µs**

### Optimization strategy (decode)

1. Fuse all per-head ops into one Triton kernel, single state read/write per launch
2. Tile over batch: grid = (B, H), block processes one (batch, head) pair
3. Keep state in registers/SMEM during the token update
4. Coalesced HBM access for state `[B, H, K, V]` → transpose to `[B, H, K, V]` access pattern

---

## Prefill — `gdn_prefill_qk4_v8_d128_k_last`

**Shape**: q/k `[T,4,128]`, v `[T,8,128]`, state `[N,8,128,128]`, varlen batched

### Compute per sequence (length L)

Each timestep: same as decode ≈ `8 × 131072 = 1.05M FLOP`
Total for length-L sequence: `L × 1.05M FLOP`

For L=512: ~537M FLOP | For L=8192: ~8.6G FLOP

### Memory per sequence (length L)

State `[8,128,128]` f32 = 4MB read+write once (if chunked) or L times (if not chunked)

**Sequential scan (no chunking)**:
- State read/write per step: 2 × 4MB = 8MB
- Total: L × 8MB → L=512: 4GB → arithmetic intensity ≈ 1 FLOP/byte (still mem-bound)

**Chunked (chunk_size=C)**:
- Per chunk: C × 1.05M FLOP / (8MB state + C×tensor loads)
- At C=64: 67M / (8MB + ~1MB) ≈ 7.5 FLOP/byte → approaching ridge point

### Optimization strategy (prefill)

1. **Chunked recurrence**: process C=64 tokens per SMEM tile, keep state in registers
2. **Triton kernel**: grid = (N, H), load state once per chunk from HBM
3. **Vectorized loads**: 128-bit LDG for state matrix rows
4. **Future**: Flash-linear-attention style online algorithm for better cache reuse

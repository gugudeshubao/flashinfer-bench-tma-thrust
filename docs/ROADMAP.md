# Optimization Roadmap

## Kernels

- `gdn_decode_qk4_v8_d128_k_last` — single-token decode, batch parallel
- `gdn_prefill_qk4_v8_d128_k_last` — variable-length prefill, sequential state recurrence

---

## v1 · Python Baseline ✅ (2026-03-24)

**Goal**: correctness pipeline end-to-end.

- Pure PyTorch, token-by-token Python loop
- Decode: 10/10 PASSED, ~1.10x speedup vs ref
- Prefill: 12/12 PASSED, ~0.98x speedup vs ref
- Infrastructure: Modal B200, flashinfer-bench, safetensors workloads, `--compare` mode

---

## v2 · Triton Kernel 🚧

**Goal**: eliminate Python loop overhead, fuse all per-head ops.

### Decode

- Grid: `(B, H)` — one warp-group per (batch, head)
- Load state tile `[K, V]` into SMEM (128×128 × 4B = 64 KB, fits in 256 KB SMEM)
- Compute decay, old_v, new_v, state update, output — all in registers
- Write state back; emit output
- Expected: eliminate per-token Python overhead, approach memory-bandwidth limit

### Prefill

- Grid: `(N, H)` — one warp-group per (sequence, head)
- Chunk size C=64: load chunk of k/v/g/beta, iterate state update in registers
- State: load from HBM once per chunk, write back after chunk
- Expected: ~10–50x over Python baseline for large T

---

## v3 · CUDA / WGMMA + TMA ⏳

**Goal**: peak B200 throughput with native Blackwell instructions.

### Decode

- WGMMA for `q@S` and `k@S` (128×128 matmul in tensor cores)
- TMA for asynchronous bulk state load/store
- Persistent kernel across batch: one SM handles multiple sequences

### Prefill

- TMA + double-buffering for state tiles
- WGMMA for chunk matmuls: `[C,K] × [K,V]` in tensor cores
- Fuse gate computation with state update

---

## Metrics to Track

After each version, update [PERFORMANCE.md](PERFORMANCE.md) with:
- Per-workload latency (ms) and speedup vs reference
- Speedup vs Python baseline (from `--compare` run)
- Hardware utilization estimate (% peak BW or FLOPS)

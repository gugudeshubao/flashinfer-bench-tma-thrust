# GDN Kernels

Gated Delta Net (GDN) kernel implementations organized by framework.

```
src/kernels/
├── cuda/          # Basic CUDA (v5-v8)
│   ├── gdn_decode_v5.cuh   # Baseline
│   ├── gdn_decode_v6.cuh   # TMA async
│   ├── gdn_decode_v7.cuh   # Vectorized + FP4
│   └── gdn_decode_v8.cuh   # Warp specialization + FP8
├── cute/          # CuTe DSL (v9-v10)
│   ├── gdn_decode_v9.cuh   # SMEM swizzle
│   └── gdn_decode_v10.cuh  # CuTe Swizzle<3,3,3>
└── triton/        # Triton (baseline)
    ├── gdn_decode_triton.py
    └── gdn_prefill_triton.py
```

## Version Summary

| Version | Framework | Key Feature | Correctness | Best At |
|---------|-----------|-------------|-------------|---------|
| v5 | CUDA | Baseline | ✓ | - |
| v6 | CUDA | TMA | ✓ | - |
| v7 | CUDA | float4 + FP4 | ✓ | batch=256 |
| v8 | CUDA | Warp spec + FP8 | ✓ | batch=256 |
| v9 | CuTe | SMEM swizzle | ✓ | batch=1,16 |
| v10 | CuTe | Layout algebra | ✓ | batch=256 |
| Triton | Triton | Auto-tune | ✓ (ref) | batch=64 |

## Performance Summary (B200)

| Batch | Triton | v7 | v8 | v9 | v10 |
|-------|--------|-----|-----|-----|-----|
| 1 | 24 | 25 (1.06x) | 25 (1.03x) | **27 (1.11x)** | 26 (1.10x) |
| 16 | 386 | 352 (0.91x) | 334 (0.86x) | **405 (1.05x)** | 403 (1.04x) |
| 64 | **1518** | 981 (0.65x) | 914 (0.60x) | 1302 (0.86x) | 1287 (0.85x) |
| 256 | 2834 | 7578 (2.67x) | **7605 (2.68x)** | 7585 (2.68x) | 7602 (2.68x) |

(Values in GB/s, B200 peak = 8,000 GB/s)

## Delta Rule Implementation

All versions implement the same algorithm:
```
S = g * S              # Decay first (CRITICAL!)
old_v = sum(S * k)     # Compute with decayed state
delta = beta * (v - old_v)
S = S + delta * k      # Update
out = scale * sum(S * q)
```

## Build

```bash
modal run scripts/build_cuda.py
```

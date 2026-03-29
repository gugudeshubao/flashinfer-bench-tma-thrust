# CUDA Kernels (v5-v8)

Basic CUDA implementations of GDN decode/prefill kernels.

## Files

| File | Version | Features |
|------|---------|----------|
| `gdn_decode_v5.cuh` | v5 | Baseline CUDA, atomicAdd reduction |
| `gdn_decode_v6.cuh` | v6 | TMA async loads (mbarrier) |
| `gdn_decode_v7.cuh` | v7 | Vectorized float4, FP4 quantization |
| `gdn_decode_v8.cuh` | v8 | Warp specialization, FP8 quantization |
| `gdn_prefill_v5.cuh` | v5 | Prefill baseline |
| `gdn_prefill_v6.cuh` | v6 | Prefill with TMA |
| `gdn_prefill_v7.cuh` | v7 | Prefill with FP4 |
| `gdn_prefill_v8.cuh` | v8 | Prefill with FP8 |

## Key Optimizations

- **v5**: Gate decay applied in-place (`state *= g`)
- **v6**: TMA `cp.async.bulk.tensor` for 2D state loading
- **v7**: `float4` vectorized loads, FP4 state quantization
- **v8**: Warp specialization (producer/consumer), FP8 quantization

## Performance (B200, batch=256)

All achieve ~7,600 GB/s (95% of peak)

# DSA Roadmap

## v1

- Freeze MLA-core sparse attention interface
- Implement token-level indexer reference
- Implement prefill/decode PyTorch baselines
- Add correctness tests

## v2

- Triton prefill kernel
- Shared metadata packing for top-k indices
- Better local benchmarks and profiling

## v3

- Triton decode kernel
- Paged cache metadata
- B200-specific tuning

## v4

- FP8 indexer path
- FP8 KV-cache path
- FlashInfer benchmark integration

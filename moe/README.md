# MoE (Mixture of Experts) Kernels

FP8 Block-Scale Fused MoE kernel for DeepSeek-V3/R1 on NVIDIA B200 (Blackwell, sm_100).

## Operator Specification

| Parameter | Value |
|-----------|-------|
| **Definition** | `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048` |
| **Model** | DeepSeek-V3/R1 |
| **Quantization** | FP8 (float8_e4m3fn) + block scale (block_size=128) |
| **Routing** | DeepSeek no-aux: sigmoid + group selection + top-k |
| **Experts** | 256 total, 32 local (EP=8) |
| **Hidden Size** | 7168 |
| **Intermediate Size** | 2048 |
| **Top-K** | 8 |
| **N_GROUP** | 8 |
| **TOPK_GROUP** | 4 |

## Pipeline

```
Routing (DeepSeek no-aux) → Token Permutation → GEMM1 (gate+up) → SwiGLU → GEMM2 (down) → Weighted Accumulation
```

## Directory Structure

```
moe/
├── config.toml                    # Solution configuration
├── solution/triton/kernel.py      # Triton kernel implementation
├── trace_definitions/             # Kernel definition JSON
├── scripts/                       # Setup scripts
│   └── setup_moe_volume.py       # Modal volume setup
├── benchmarks/                    # Benchmark runners
│   └── bench_modal.py            # Modal B200 benchmark
├── tests/                         # Correctness tests
└── docs/                          # Documentation
```

## Quick Start

```bash
# 1. Setup Modal volume (download official workloads from HuggingFace)
modal run moe/scripts/setup_moe_volume.py

# 2. Run MoE benchmark on Modal B200
modal run moe/benchmarks/bench_modal.py

# 3. Run with custom parameters
modal run moe/benchmarks/bench_modal.py --warmup 5 --iters 100 --trials 5
```

## Target Hardware

- **GPU**: NVIDIA B200 (Blackwell, sm_100)
- **Memory**: 8 TB/s HBM3e
- **Compute**: 2.25 PFLOPS BF16, 4.5 PFLOPS FP8 Tensor Core

## Evaluation Criteria

- **Correctness**: atol=1, rtol=0.3, required_matched_ratio=0.9
- **Performance**: Arithmetic mean speedup over reference implementation
- **Metric**: Speedup = reference_latency / solution_latency

## Optimization Roadmap

- [x] Correct baseline implementation (PyTorch dequant + matmul)
- [ ] Fused Triton FP8 GEMM with block-scale dequantization
- [ ] Triton grouped GEMM (batch experts together)
- [ ] Fused SwiGLU activation
- [ ] Token permutation optimization
- [ ] TMA bulk memory operations for weight loading
- [ ] FP8 Tensor Core (mma) on Blackwell

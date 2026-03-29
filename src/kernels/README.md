# GDN Kernels

Gated Delta Net (GDN) kernel implementations organized by framework.

## Directory Structure

```
src/kernels/
├── cuda/          # Raw CUDA (v5-v8) — 低级控制
├── cute/          # CuTe DSL (v9-v10) — 中级抽象
└── triton/        # Triton (baseline) — 高级 DSL
```

## Technology Stack Comparison

| 维度 | Raw CUDA | CuTe | Triton |
|------|----------|------|--------|
| **抽象级别** | 低 | 中 | 高 |
| **SMEM 控制** | 手动 | 声明式 | 自动 |
| **Bank Conflict** | 手动 swizzle | `Swizzle<B,M,S>` | 自动 |
| **Tensor Core** | 手动 PTX | WGMMA 抽象 | 自动 |
| **学习成本** | 高 | 中高 | 低 |
| **性能上限** | 最高 | 最高 | 略低 |
| **代码量** | 最多 | 中等 | 最少 |

## Version Summary

| Version | Framework | Abstraction | Key Feature | Best At |
|---------|-----------|-------------|-------------|---------|
| v5 | Raw CUDA | Low | Baseline | - |
| v6 | Raw CUDA | Low | TMA async | - |
| v7 | Raw CUDA | Low | float4 + FP4 | batch=256 |
| v8 | Raw CUDA | Low | Warp specialization | batch=256 |
| **v9** | **CuTe** | **Medium** | **SMEM swizzle** | **batch=1,16** |
| v10 | CuTe | Medium | `Swizzle<3,3,3>` | batch=256 |
| Triton | Triton | High | Auto-tuning | batch=64 |

## Performance Summary (B200, 8 TB/s peak)

| Batch | Triton | CUDA v7 | CUDA v8 | CuTe v9 | CuTe v10 |
|-------|--------|---------|---------|---------|----------|
| 1 | 24 GB/s | 25 (1.06x) | 25 (1.03x) | **27 (1.11x)** | 26 |
| 16 | 386 GB/s | 352 (0.91x) | 334 (0.86x) | **405 (1.05x)** | 403 |
| 64 | **1,518 GB/s** | 981 (0.65x) | 914 (0.60x) | 1,302 (0.86x) | 1,287 |
| 256 | 2,834 GB/s | 7,578 (2.67x) | **7,605 (2.68x)** | 7,585 | 7,602 |

**Key Insights:**
- Raw CUDA 和 CuTe 性能相当 (95% 带宽利用率)
- CuTe 代码更简洁，Swizzle 由库处理
- Triton 在 batch=64 胜出 (auto-tuning 优势)

## Why Can't We Use Tensor Core for Decode?

```
GDN Decode: S @ q = [128×128] × [128] → [128]
                                  ↑
                            矩阵-向量 (N=1)
                            Tensor Core 要求 N≥16
```

- **Decode**: Memory-bound (AI=1 FLOP/byte) → 优化带宽
- **Prefill**: Can be compute-bound (AI=7.5 with chunking) → 可用 WGMMA

## Delta Rule (All Versions)

```cpp
// CRITICAL: Apply g FIRST, then compute old_v
float decayed_s = g * state[idx];      // Decay first!
old_v += decayed_s * k[d];              // Use decayed state
// ...
new_s = decayed_s + delta * k[d];       // Update
```

## Build

```bash
modal run scripts/build_cuda.py
```

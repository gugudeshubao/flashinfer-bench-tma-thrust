# GDN Kernels

Gated Delta Net (GDN) kernel implementations organized by framework.

## Directory Structure

```
src/kernels/
├── cuda/          # Raw CUDA C++ (v5-v8) — 低级控制，手动优化
├── cute/          # CuTe C++ (v9-v10) — CUTLASS 3.x 张量抽象
├── cute_dsl/      # CuTe DSL Python (规划) — CUTLASS 4.0，FlashAttention-4 风格
├── cutile/        # cuTile Python (v11 规划) — CUDA 13.1，对标 Triton
└── triton/        # Triton (baseline) — OpenAI 高级 DSL
```

## Technology Stack Comparison

| 维度 | Raw CUDA | CuTe C++ | CuTe DSL | cuTile | Triton |
|------|----------|----------|----------|--------|--------|
| **语言** | C++ | C++ | **Python** | Python | Python |
| **抽象级别** | 低 | 中 | 中 | 高 | 高 |
| **编译时间** | 秒 | 分钟 | **秒** | 秒 | 秒 |
| **SMEM 控制** | 手动 | 声明式 | 声明式 | 自动 | 自动 |
| **Tensor Core** | 手动 PTX | `TiledMMA` | `TiledMMA` | 自动 | 自动 |
| **学习成本** | 高 | 中高 | 中 | 低 | 低 |
| **性能上限** | 最高 | 最高 | **最高** | TBD | 略低 |
| **典型应用** | v7/v8 | v9/v10 | **FlashAttn-4** | v11 | v5 |

## Version Summary

| Version | Framework | Language | Key Feature | Status |
|---------|-----------|----------|-------------|--------|
| v5 | Raw CUDA | C++ | Baseline | ✅ |
| v6 | Raw CUDA | C++ | TMA async | ✅ |
| v7 | Raw CUDA | C++ | float4 + FP4 | ✅ |
| v8 | Raw CUDA | C++ | Warp specialization | ✅ |
| **v9** | **CuTe** | C++ | **SMEM swizzle** | ✅ |
| v10 | CuTe | C++ | TiledMMA | ✅ |
| v11 | cuTile | Python | Tile-based | 🚧 规划 |
| Triton | Triton | Python | Auto-tuning | ✅ |

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
- **Prefill**: Can be compute-bound (AI=7.5 with chunking) → 可用 tcgen05.mma (Blackwell)

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

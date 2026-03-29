# GDN Kernels

Gated Delta Net (GDN) kernel implementations organized by framework.

## Directory Structure

```
src/kernels/
├── cuda/          # Raw CUDA C++ (v5-v8) — 低级控制，手动优化
├── cute_cpp/      # CuTe C++ (v9-v10) — CUTLASS 3.x 模板，NVCC 编译
├── cute_dsl/      # CuTe DSL (MLIR) — CUTLASS 4.0+，Python → MLIR → PTX
├── cutile/        # cuTile Python — CUDA 13.1，NVIDIA Tile-based API
├── ptx/           # PTX Inline Assembly — 极致底层控制，内嵌汇编
└── triton/        # Triton (baseline) — OpenAI 高级 DSL
```

## CuTe C++ vs CuTe DSL (重要区分!)

| 方面 | CuTe C++ (cute_cpp/) | CuTe DSL (cute_dsl/) |
|------|----------------------|----------------------|
| **语言** | C++ 模板 | Python |
| **编译** | NVCC → PTX | **MLIR → LLVM → PTX** |
| **库版本** | CUTLASS 3.x | CUTLASS 4.0+ |
| **开发效率** | 中 (需懂模板元编程) | 高 (Python 语法) |
| **编译时间** | 分钟级 (AOT) | 秒级 (JIT) |
| **性能** | 100% | ~100% (理论) |
| **典型应用** | 通用 CUDA | **FlashAttention-4** |

```
CuTe C++:   C++ Templates  →  NVCC  →  PTX  →  SASS
                ↑
           手动控制 Layout/Swizzle

CuTe DSL:   Python DSL  →  MLIR  →  LLVM  →  PTX  →  SASS
                              ↑
                        自动优化 Pass
```

## Technology Stack Comparison

| 维度 | Raw CUDA | PTX Inline | CuTe C++ | CuTe DSL | cuTile | Triton |
|------|----------|------------|----------|----------|--------|--------|
| **语言** | C++ | C++ + ASM | C++ | **Python** | Python | Python |
| **编译器** | NVCC | NVCC | NVCC | **MLIR** | cuTile JIT | Triton |
| **抽象级别** | 低 | **最低** | 中 | 中 | 高 | 高 |
| **SMEM 控制** | 手动 | 手动 | 声明式 | 声明式 | 自动 | 自动 |
| **Tensor Core** | 手动 PTX | 手动 PTX | `TiledMMA` | `TiledMMA` | 自动 | 自动 |
| **性能上限** | 最高 | **最高** | 最高 | **最高** | 受限* | 略低 |
| **典型应用** | v7/v8 | 极致优化 | v9/v10 | **FlashAttn-4** | N/A | v5 |

*cuTile 受限于 tile-based 索引，不适合 4D strided 访问模式

## Version Summary

| Version | Framework | Language | Compiler | Key Feature | Status |
|---------|-----------|----------|----------|-------------|--------|
| v5 | Raw CUDA | C++ | NVCC | Baseline | ✅ |
| v6 | Raw CUDA | C++ | NVCC | TMA async | ✅ |
| v7 | Raw CUDA | C++ | NVCC | float4 + FP4 | ✅ |
| v8 | Raw CUDA | C++ | NVCC | Warp specialization | ✅ |
| **v9** | **CuTe C++** | C++ | NVCC | **SMEM swizzle** | ✅ |
| v10 | CuTe C++ | C++ | NVCC | TiledMMA | ✅ |
| PTX | PTX Inline | C++ + ASM | NVCC | Inline assembly | ✅ |
| DSL | **CuTe DSL** | Python | **MLIR** | FlashAttn-4 style | ✅ |
| cuTile | cuTile | Python | cuTile JIT | Tile-based | ⚠️ 受限 |
| Triton | Triton | Python | Triton | Auto-tuning | ✅ |

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
- **Prefill**: Can be compute-bound (AI=8 with chunking) → 可用 tcgen05.mma (Blackwell)

## Prefill Compute Density Optimization (v6)

The key insight: **Chunking increases arithmetic intensity!**

```
                     Sequential (v5)          Chunked (v6, C=8)
                     ───────────────          ─────────────────
Tokens processed     1 per iteration          8 per iteration
State access         Load once per token      Load once per 8 tokens
Arithmetic Intensity 1 FLOP/byte              8 FLOP/byte
Bound                Memory-bound             Compute-bound!
```

### Roofline Analysis (B200)

```
B200 Specs:
  - FP32: 70 TFLOPS
  - Memory BW: 8 TB/s
  - Ridge Point: 70/8 = 8.75 FLOP/byte

  CHUNK_SIZE | AI (FLOP/byte) | Status
  -----------|----------------|--------
      1      |      1.0       | Memory-bound ❌
      4      |      4.0       | Transitional
      8      |      8.0       | Near ridge ≈
     16      |     16.0       | Compute-bound ✅
```

### Files

| Directory | Decode | Prefill | Notes |
|-----------|--------|---------|-------|
| `cuda/` | v5-v8 | v5-v8 | 完整版本 |
| `cute_cpp/` | v9-v10 | **v9** | SMEM swizzle + chunking |
| `cute_dsl/` | ✅ | **✅** | MLIR-based |
| `ptx/` | ✅ | ✅ | Fast math assembly |
| `triton/` | ✅ | ✅ | Baseline |
| `cutile/` | ✅ | ⚠️ | 4D限制，不推荐 |

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

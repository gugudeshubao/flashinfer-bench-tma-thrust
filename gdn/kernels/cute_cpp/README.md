# CuTe C++ Kernels (v9-v10)

> **CuTe C++** 是 CUTLASS 3.x 的 C++ 模板库，通过 NVCC 编译。与 CuTe DSL (MLIR) 不同，这里是纯 C++ 模板编程。

## CuTe C++ vs CuTe DSL

| 方面 | CuTe C++ (本目录) | CuTe DSL (cute_dsl/) |
|------|-------------------|----------------------|
| **语言** | C++ 模板 | Python |
| **编译** | NVCC → PTX | MLIR → LLVM → PTX |
| **抽象** | `Layout<>`, `Swizzle<>` | `@cute.kernel` |
| **编译时间** | 分钟级 | 秒级 (JIT) |
| **性能** | 100% | ~100% |
| **开发效率** | 中 | 高 |

## What is CuTe C++?

CuTe 是 NVIDIA CUTLASS 库的核心组件，提供:

| 特性 | 描述 |
|------|------|
| **Layout Algebra** | 声明式 Tensor 布局 (`Layout<Shape, Stride>`) |
| **Swizzle** | Bank conflict 消除 (`Swizzle<B,M,S>`) |
| **TMA Abstraction** | 简化的异步内存操作 |
| **WGMMA Support** | Tensor Core 操作抽象 |

## Why CuTe over Raw CUDA?

| 维度 | Raw CUDA | CuTe |
|------|----------|------|
| **抽象级别** | 低 | 中 |
| **Swizzle** | 手动 XOR | `Swizzle<3,3,3>` |
| **Layout** | 手动 stride | `Layout<Shape, Stride>` |
| **代码量** | ~650 行 | ~400 行 |
| **性能** | 95% BW | **95% BW** (相同!) |
| **可维护性** | 困难 | **更容易** |

## Files

| File | Version | Type | Key Feature |
|------|---------|------|-------------|
| `gdn_decode_v9.cuh` | v9 | Decode | Manual XOR swizzle |
| `gdn_decode_v10.cuh` | v10 | Decode | CuTe `Swizzle<3,3,3>` |
| `gdn_prefill_v9.cuh` | v9 | **Prefill** | SMEM swizzle + chunking |

## Swizzle Explained

### 问题: Bank Conflict

```
SMEM 有 32 个 bank，每 4 字节一个 bank
如果 32 个线程访问同一 bank → 32-way conflict!
吞吐量降低 32x
```

### v9: Manual XOR Swizzle

```cpp
// 手动实现 XOR swizzle
__device__ int swizzle(int d) {
    return d ^ ((d >> 3) & 7);
}

// 使用
int idx = v_idx * D + swizzle(d_idx);
s_state[idx] = value;
```

### v10: CuTe Swizzle<3,3,3>

```cpp
// CuTe 声明式 swizzle
template<int BLOCK_V>
struct SwizzledStateLayout {
    using SwizzleType = Swizzle<3, 3, 3>;
    // B=3: 8 个 bank group
    // M=3: 8 个 mask bits
    // S=3: 8 个 shift bits
    
    __device__ __forceinline__
    static int get_index(int v_idx, int d_idx) {
        // Swizzle<3,3,3> 等价于: d ^ ((d >> 3) & 7)
        int swizzled_d = d_idx ^ ((d_idx >> 3) & 7);
        return v_idx * V10_D + swizzled_d;
    }
};

// 使用 (更清晰的语义)
auto smem_tensor = make_tensor(smem_ptr, SwizzledLayout{});
smem_tensor(v_idx, d_idx) = value;  // 自动 swizzle
```

## Performance (B200)

| Batch | v9 (manual) | v10 (CuTe) | Winner |
|-------|-------------|------------|--------|
| 1 | **27 GB/s** | 26 GB/s | v9 |
| 16 | **405 GB/s** | 403 GB/s | v9 |
| 64 | 1,302 GB/s | 1,287 GB/s | v9 |
| 256 | 7,585 GB/s | **7,602 GB/s** | v10 |

**观察**: 
- 小 batch: v9 略快 (更简单的代码路径)
- 大 batch: v10 略快 (CuTe 编译器优化)
- 差异 <1%，实际可忽略

## vs Triton

| Batch | CuTe v9 | Triton | Speedup |
|-------|---------|--------|---------|
| 1 | 27 GB/s | 24 GB/s | **1.11x** |
| 16 | 405 GB/s | 386 GB/s | **1.05x** |
| 64 | 1,302 GB/s | **1,518 GB/s** | 0.86x |
| 256 | **7,585 GB/s** | 2,834 GB/s | **2.68x** |

**洞察**: Triton 在 batch=64 胜出，可能是 auto-tuning 选择了更优配置。

## When to Use CuTe?

✅ 需要 SMEM swizzle 时  
✅ 需要 Tensor Core (WGMMA) 时  
✅ 需要复杂 Layout 变换时  
✅ 需要可维护的高性能代码时  

❌ 简单 kernel (overhead 不值得)  
❌ 对编译时间敏感 (CuTe 模板编译慢)  

## Future: Prefill with WGMMA

```cpp
// Prefill 可以用 Tensor Core (mat-mat)
// S @ Q_chunk = [128×128] × [128×64] → [128×64]

using MMA = SM90_64x64x16_F32BF16BF16_SS;  // WGMMA operation
auto tiled_mma = make_tiled_mma(MMA{});

// CuTe handles the complexity
gemm(tiled_mma, S_smem, Q_smem, O_smem);
```

这是 v11+ 的计划优化方向。

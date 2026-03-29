# GDN Kernel 优化深度解析：为什么 Decode 无法使用 Tensor Core？

> 本文基于 NVIDIA B200 (Blackwell) 上 Gated Delta Net 内核的优化实践，解析 Decode 与 Prefill 阶段的性能瓶颈差异。

## TL;DR

| 阶段 | 操作类型 | 能否用 Tensor Core | 瓶颈 | 最佳策略 |
|------|---------|-------------------|------|----------|
| **Decode** | 矩阵-向量 | ❌ 不能 | 内存带宽 | SMEM Swizzle |
| **Prefill** | 矩阵-矩阵 | ✅ 可以 | 接近算力 | Chunked WGMMA |

---

## 1. 背景：什么是 Gated Delta Net (GDN)？

GDN 是一种线性注意力变体，用于替代标准 Transformer 的 Softmax Attention。其核心是一个**递归状态更新**：

```python
# 每个 token 的计算
g = exp(-exp(A_log) * softplus(a))   # 衰减门
beta = sigmoid(b)                     # 更新门

S = g * S                             # 状态衰减
old_v = k @ S                         # 旧值 [K] × [K,V] → [V]
new_v = beta * v + (1-beta) * old_v   # 加权混合
S = S + outer(k, new_v - old_v)       # Delta Rule 更新
o = scale * q @ S                     # 输出 [K] × [K,V] → [V]
```

**关键参数**: State `S` 形状为 `[V, K] = [128, 128]`，每个 head 独立维护。

---

## 2. 硬件：NVIDIA B200 (Blackwell) 规格

| 资源 | 峰值性能 |
|------|---------|
| **BF16 Tensor Core (WGMMA)** | **2.25 PFLOPS** |
| **FP32 CUDA Core** | 74.45 TFLOPS |
| **HBM3e 带宽** | **8 TB/s** |
| L2 Cache | 96 MB |
| Shared Memory / SM | 256 KB |

### Ridge Point（转折点）

```
Ridge Point = Peak Compute / Peak Bandwidth

BF16 Tensor: 2.25 PFLOPS / 8 TB/s = 281 FLOP/byte
FP32 CUDA:   74.45 TFLOPS / 8 TB/s = 9.3 FLOP/byte
```

> **解读**：如果你的算法 Arithmetic Intensity (AI) < Ridge Point，则为**内存瓶颈**；反之为**算力瓶颈**。

---

## 3. Decode 阶段：为什么无法使用 Tensor Core？

### 3.1 操作分析

Decode 阶段每次只处理 **1 个 token**：

```
输入: q, k, v 都是 [128] 向量
状态: S 是 [128×128] 矩阵

核心操作:
  old_v = S @ k    → [128×128] × [128] → [128]  ← 矩阵-向量！
  o = S @ q        → [128×128] × [128] → [128]  ← 矩阵-向量！
```

### 3.2 Tensor Core 的限制

NVIDIA Tensor Core (包括 Blackwell 的 WGMMA) 设计用于**矩阵-矩阵乘法 (GEMM)**：

```
WGMMA 操作: C = A @ B
  A: [M, K]
  B: [K, N]
  C: [M, N]

最小 tile 要求: M, N, K ≥ 16 (BF16)
```

**问题**：GDN Decode 的 `S @ q` 中，`q` 是向量 (N=1)，无法填满 Tensor Core tile。

```
GDN Decode: [128×128] × [128×1] → [128×1]
                           ↑
                        N=1，不满足 N≥16
```

### 3.3 Roofline 分析

```
每 token 计算量: ~1.05M FLOPs (8 heads × 131K)
每 token 内存量: ~1.05 MB (State 读写)

Arithmetic Intensity = 1.05M / 1.05MB = 1 FLOP/byte
```

**对比 Ridge Point**:
```
AI (1) << FP32 Ridge (9.3) << BF16 Ridge (281)

→ 完全内存瓶颈，即使能用 Tensor Core 也没有意义！
```

### 3.4 实测结果

| Batch | 内存带宽利用率 | FP32 算力利用率 |
|-------|--------------|----------------|
| 1 | 0.3% | 0.03% |
| 64 | 19% | 1.5% |
| **256** | **95%** | 7.6% |

> **结论**：Decode 阶段应该优化**内存带宽**，而非算力。我们的 CuTe v9 内核通过 SMEM Swizzle 达到了 **95% 带宽利用率**。

---

## 4. Prefill/Encoder 阶段：可以使用 Tensor Core！

### 4.1 关键差异：批量处理多个 Token

Prefill 阶段处理整个输入序列 (L tokens)：

```python
# 如果直接循环，仍然是 mat-vec:
for t in range(L):
    o[t] = S @ q[t]  # 逐个 token，N=1

# 但是，可以用 Chunking 转换成 mat-mat:
Q_chunk = Q[0:C]         # [C, K] = [64, 128]
O_chunk = S @ Q_chunk.T  # [V,K] × [K,C] → [V,C]
                         #          ↑
                         #       C=64，可用 Tensor Core！
```

### 4.2 Chunked Prefill 算法

```python
def prefill_chunked(Q, K, V, S, chunk_size=64):
    L = Q.shape[0]
    O = zeros_like(V)
    
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        C = end - start
        
        # ===== 可并行部分 (WGMMA) =====
        Q_chunk = Q[start:end]  # [C, K]
        # 计算 chunk 内所有 output
        O_chunk = S @ Q_chunk.T  # [V,K] × [K,C] → [V,C] ✅ mat-mat!
        
        # ===== 顺序部分 (无法并行) =====
        for t in range(start, end):
            # State 更新必须顺序执行
            S = g[t] * S + outer(k[t], delta[t])
        
        O[start:end] = O_chunk.T
    
    return O, S
```

### 4.3 Arithmetic Intensity 提升

| 模式 | AI | 瓶颈 | 能否用 WGMMA |
|------|-----|-----|-------------|
| 无 Chunking (逐 token) | 1 FLOP/byte | 内存 | ❌ |
| Chunk=16 | ~2.5 FLOP/byte | 内存 | ⚠️ 边缘 |
| **Chunk=64** | **~7.5 FLOP/byte** | **接近转折** | ✅ |
| Chunk=128 | ~12 FLOP/byte | 算力 | ✅ |

### 4.4 为什么 State 更新仍需顺序？

```
S_1 = g_1 * S_0 + outer(k_1, delta_1)
S_2 = g_2 * S_1 + outer(k_2, delta_2)  ← 依赖 S_1
S_3 = g_3 * S_2 + outer(k_3, delta_3)  ← 依赖 S_2
...
```

**递归依赖**：每个 State 依赖前一个，无法完全并行。

但通过 Chunking，我们将并行粒度从**单 token** 提升到 **64 tokens**，显著提高了 Tensor Core 利用率。

---

## 5. 优化策略总结

### 5.1 Decode 优化路径

```
目标: 最大化内存带宽 (8 TB/s)

已实现:
✅ CuTe SMEM Swizzle (消除 Bank Conflict) → 95% 带宽利用
✅ Coalesced Memory Access
✅ float4 向量化加载

无需实现:
❌ Tensor Core (无法使用)
❌ 更复杂的量化 (AI=1, 算力不是瓶颈)
```

### 5.2 CUDA vs CuTe vs CuTile：我们的演进路径

我们实现了多个版本的 GDN Decode 内核，使用不同的抽象层级：

| 版本 | 技术栈 | 抽象级别 | 核心优化 | 带宽利用率 |
|------|--------|---------|---------|-----------|
| v5 | Triton | 高级 DSL | Auto-tuning | 35% (baseline) |
| v7 | Raw CUDA | 低级 | float4 向量化 | 95% |
| v8 | Raw CUDA | 低级 | Warp Specialization | 95% |
| **v9** | **CuTe** | **中级 DSL** | **SMEM Swizzle** | **95%** |
| v10 | CuTe | 中级 DSL | Swizzle<3,3,3> | 95% |

#### Raw CUDA (v7/v8)

直接使用 CUDA C++ 编写，手动管理所有细节：

```cpp
// v7: 手动 float4 向量化加载
float4* state_f4 = reinterpret_cast<float4*>(&state[idx]);
float4 s = state_f4[d / 4];

// v8: 手动 Warp Specialization
if (warp_id < 2) {
    // Compute warps: 执行计算
} else {
    // Memory warp: 异步预取下一个 state
}
```

**优点**: 完全控制，性能可预测  
**缺点**: 代码冗长，容易出错，难以维护

#### CuTe (v9/v10)

NVIDIA CUTLASS 库提供的 C++ 模板 DSL，抽象 Tensor 布局和操作：

```cpp
// CuTe: 声明式定义 SMEM 布局
using SmemLayout = Layout<Shape<_128, _128>, 
                          Stride<_128, _1>>;

// CuTe Swizzle: 消除 Bank Conflict
using SwizzledLayout = decltype(
    composition(Swizzle<3,3,3>{}, SmemLayout{})
);

// CuTe: 自动生成正确的索引
auto smem_tensor = make_tensor(smem_ptr, SwizzledLayout{});
smem_tensor(v_idx, d_idx) = value;  // 自动应用 swizzle
```

**优点**: 高级抽象，自动处理复杂布局，代码简洁  
**缺点**: 学习曲线陡峭，编译时间长

#### Swizzle 原理详解

Bank Conflict 问题：
```
SMEM 有 32 个 bank，每 4 字节一个 bank
连续地址: addr % 32 决定 bank
如果 32 个线程访问同一 bank → 32-way conflict!
```

Swizzle 解决方案 (XOR-based):
```cpp
// Swizzle<B, M, S> = Swizzle<3, 3, 3>
// B=3: 8 个 bank group
// M=3: 8 个 bank 内偏移  
// S=3: 8 个 swizzle pattern

// 物理索引 = 逻辑索引 XOR (逻辑索引 >> 3) & 7
int swizzled_idx = logical_idx ^ ((logical_idx >> 3) & 7);
```

效果：将 8-way bank conflict 降为 1-way，SMEM 吞吐量提升 8x。

#### 为什么选择 CuTe？

| 维度 | Raw CUDA | CuTe | Triton |
|------|----------|------|--------|
| 抽象级别 | 低 | 中 | 高 |
| SMEM 控制 | 手动 | 声明式 | 自动 |
| Bank Conflict | 手动处理 | Swizzle 抽象 | 自动 |
| Tensor Core | 手动 PTX | WGMMA 抽象 | 自动 |
| 学习成本 | 高 | 中高 | 低 |
| 性能上限 | 最高 | 最高 | 略低 |

> **我们的选择**: v9/v10 使用 CuTe，获得接近 Raw CUDA 的性能，同时保持代码可维护性。

### 5.3 Prefill 优化路径

```
目标: 利用 Tensor Core (2.25 PFLOPS BF16)

推荐实现:
✅ Chunked Recurrence (chunk_size=64)
✅ WGMMA for S @ Q_chunk
✅ TMA for async bulk loads
✅ State in registers/SMEM

挑战:
⚠️ State 更新仍为顺序
⚠️ 需要精细的 SMEM 管理
```

---

## 6. 性能对比

### Decode 各版本对比 (Batch=256)

| 版本 | 技术栈 | 带宽 | vs Triton | 关键优化 |
|------|--------|------|-----------|---------|
| Triton v5 | Triton | 2,834 GB/s | 1.0x | Auto-tuning |
| CUDA v7 | Raw CUDA | 7,578 GB/s | 2.67x | float4 向量化 |
| CUDA v8 | Raw CUDA | 7,605 GB/s | 2.68x | Warp Specialization |
| **CuTe v9** | **CuTe** | **7,585 GB/s** | **2.68x** | **SMEM Swizzle** |
| CuTe v10 | CuTe | 7,602 GB/s | 2.68x | Swizzle<3,3,3> |

**观察**：
- Raw CUDA 和 CuTe 性能相当（都达到 95% 带宽利用率）
- CuTe 代码更简洁，Swizzle 逻辑由库处理
- Triton 在大 batch 时差距明显（35% vs 95%）

### 不同 Batch Size 表现

| Batch | Triton v5 | CuTe v9 | 胜者 | 原因 |
|-------|-----------|---------|------|------|
| 1 | 24 GB/s | 27 GB/s | CuTe | Launch 开销更小 |
| 16 | 386 GB/s | 405 GB/s | CuTe | 更好的 SMEM 利用 |
| 64 | **1,518 GB/s** | 1,302 GB/s | **Triton** | Auto-tuning 优势 |
| 256 | 2,834 GB/s | **7,585 GB/s** | **CuTe** | Swizzle 消除 conflict |

**洞察**：Triton 在 batch=64 时胜出，可能是其 auto-tuning 选择了更优的 tile 配置。

### Prefill (规划中)

| 方案 | 预期性能 |
|------|---------|
| 当前 Triton | Baseline |
| Chunked + WGMMA | **2-4x** (TBD) |

---

## 7. 结论

| 阶段 | 本质 | 优化重点 | Tensor Core |
|------|------|---------|-------------|
| **Decode** | 矩阵-向量 × L次 | 内存带宽 | ❌ 无法使用 |
| **Prefill** | 矩阵-矩阵 (chunked) | Tensor Core + 带宽 | ✅ 可利用 |

> **核心洞察**：不是所有算子都应该追求 Tensor Core。对于 Memory-Bound 的 Decode，优化带宽才是正道；对于可转换为 GEMM 的 Prefill，Tensor Core 是关键加速手段。

---

## 参考

- [FlashInfer GDN Implementation](https://github.com/flashinfer-ai/flashinfer)
- [NVIDIA Blackwell Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [CuTe Documentation](https://github.com/NVIDIA/cutlass/tree/main/include/cute)

---

*作者注：本文数据基于 NVIDIA B200 GPU (Modal Cloud) 实测，代码开源于 [flashinfer-bench-tma-thrust](https://github.com/xxx/flashinfer-bench-tma-thrust)。*

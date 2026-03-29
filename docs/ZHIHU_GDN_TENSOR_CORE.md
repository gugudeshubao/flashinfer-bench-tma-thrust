# GDN Kernel 优化深度解析：为什么 Decode 无法使用 Tensor Core？

> 本文基于 NVIDIA B200 (Blackwell) 上 Gated Delta Net 内核的优化实践，解析 Decode 与 Prefill 阶段的性能瓶颈差异，以及 Raw CUDA、CuTe、CuTile、Triton 四种技术栈的对比。

---

## 术语表 (Glossary)

| 术语 | 英文全称 | 中文解释 |
|------|---------|---------|
| **Tensor Core** | Tensor Core | NVIDIA GPU 上专门用于矩阵乘法的硬件单元 |
| **tcgen05.mma** | Tensor Core Gen 05 Matrix Multiply Accumulate | Blackwell 架构的 Tensor Core 指令 |
| **WGMMA** | Warpgroup Matrix Multiply Accumulate | Hopper 架构的 Tensor Core 指令 |
| **SMEM** | Shared Memory | GPU 上 SM 内线程共享的高速缓存 |
| **HBM** | High Bandwidth Memory | GPU 的高带宽显存 |
| **Swizzle** | Swizzle | 地址重映射技术，用于消除 Bank Conflict |
| **Bank Conflict** | Bank Conflict | 多线程同时访问同一 SMEM bank 导致的性能下降 |
| **Roofline** | Roofline Model | 分析算法受限于算力还是带宽的性能模型 |
| **Ridge Point** | Ridge Point | Roofline 模型中算力和带宽的转折点 |
| **AI** | Arithmetic Intensity | 算术强度，每字节内存访问的浮点操作数 |
| **TMA** | Tensor Memory Accelerator | Hopper+ 架构的异步内存加载单元 |
| **CuTe** | CUTLASS Tensor | NVIDIA CUTLASS 库的张量抽象层 |
| **CuTile** | CUTLASS Tile | CuTe 的 Tile 迭代抽象 |
| **GEMM** | General Matrix Multiply | 通用矩阵乘法 |
| **GEMV** | General Matrix Vector Multiply | 矩阵-向量乘法 |
| **Warp** | Warp | GPU 上 32 个线程组成的执行单元 |
| **Lane** | Lane | Warp 内的单个线程 |
| **PTX** | Parallel Thread Execution | NVIDIA GPU 的虚拟指令集 |

---

## TL;DR

| 阶段 | 操作类型 | 能否用 Tensor Core | 瓶颈 | 最佳策略 |
|------|---------|-------------------|------|----------|
| **Decode** | 矩阵-向量 | ❌ 不能 | 内存带宽 | CuTe SMEM Swizzle |
| **Prefill** | 矩阵-矩阵 | ✅ 可以 | 接近算力 | Chunked tcgen05.mma |

---

## 1. 背景：什么是 Gated Delta Net (GDN)？

GDN (Gated Delta Net，门控增量网络) 是一种**线性注意力变体**，用于替代标准 Transformer 的 Softmax Attention。相比 Softmax Attention 的 O(L²) 复杂度，GDN 使用递归状态实现 O(L) 复杂度。

其核心是一个**递归状态更新** (Recurrent State Update)：

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

## 2. 硬件：NVIDIA B200 (Blackwell, SM100) 规格

### 2.1 核心规格

| 规格 | 值 |
|------|-----|
| 架构 | Blackwell (SM100) |
| CUDA Cores | 16,896 |
| Tensor Cores | 528 (5th Gen) |
| HBM3e | 180 GB @ **8 TB/s** |
| L2 Cache | 96 MB |
| SMEM / SM | 256 KB |
| TDP | 1,000 W |

### 2.2 Tensor Core 指令演进

| 架构 | 指令 | 相对性能 |
|------|------|---------|
| Ampere (A100, sm_80) | `mma.sync` | 1.0x |
| Hopper (H100, sm_90) | `wgmma` | ~2x |
| **Blackwell (B200, sm_100)** | **`tcgen05.mma`** | **2-4x vs Hopper** |

**注意**: B200 使用 `tcgen05.mma`，**不是** `wgmma`！

### 2.3 tcgen05.mma 指令集

| 指令 | 吞吐量 | 数据类型 |
|------|--------|---------|
| `tcgen05.mma.kind::tf32` | 2x Hopper | TF32 × TF32 |
| `tcgen05.mma.kind::f16` | 2x Hopper | FP16/BF16 |
| `tcgen05.mma.kind::i8` | 2x Hopper | INT8 |
| `tcgen05.mma.kind::f8f6f4` | 2x Hopper | FP4/FP6/FP8 混合 |
| `tcgen05.mma.kind::mxf4` | **4x Hopper** | MX FP4 (block scaled) |

### 2.4 计算性能

| 精度 | Dense | Sparse (2:4) |
|------|-------|--------------|
| FP4 Tensor | 9 PFLOPS | 18 PFLOPS |
| FP8 Tensor | 4.5 PFLOPS | 9 PFLOPS |
| **BF16 Tensor** | **2.25 PFLOPS** | 4.5 PFLOPS |
| FP32 CUDA | 74.45 TFLOPS | - |

### 2.5 Ridge Point（转折点）

Ridge Point 是 Roofline 模型中的关键概念：当算法的**算术强度 (Arithmetic Intensity, AI)** 等于 Ridge Point 时，算法刚好平衡算力和带宽。

```
Ridge Point = Peak Compute / Peak Bandwidth
            = 峰值算力 / 峰值带宽

BF16 Tensor: 2.25 PFLOPS / 8 TB/s = 281 FLOP/byte
FP32 CUDA:   74.45 TFLOPS / 8 TB/s = 9.3 FLOP/byte
```

> **解读**：
> - 如果 AI < Ridge Point → **内存瓶颈** (Memory-Bound)，优化带宽
> - 如果 AI > Ridge Point → **算力瓶颈** (Compute-Bound)，优化计算

---

## 3. Decode 阶段：为什么无法使用 Tensor Core？

> **Decode (解码阶段)**：在自回归生成 (如 GPT) 中，每次生成一个新 token 的阶段。特点是每次只处理 1 个 token。

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

NVIDIA Tensor Core (包括 Blackwell 的 `tcgen05.mma`) 设计用于**矩阵-矩阵乘法 (GEMM)**：

```
tcgen05.mma 操作: D = A @ B + C
  A: [M, K]
  B: [K, N]
  D: [M, N]

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

> **Prefill (预填充阶段)**：处理用户输入的完整 prompt 的阶段。特点是一次处理 L 个 token (L = prompt 长度)。
> 
> **Encoder (编码器)**：在 BERT 等模型中，处理整个输入序列的阶段，与 Prefill 类似。

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
                         #       C=64，可用 tcgen05.mma！
```

### 4.2 Chunked Prefill 算法

> **Chunked Prefill (分块预填充)**：将长度为 L 的序列分成多个大小为 C 的 chunk，在 chunk 内部使用 GEMM (矩阵乘法)，chunk 之间传递状态。这样可以将 GEMV 转换为 GEMM，从而利用 Tensor Core。

```python
def prefill_chunked(Q, K, V, S, chunk_size=64):
    L = Q.shape[0]
    O = zeros_like(V)
    
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        C = end - start
        
        # ===== 可并行部分 (tcgen05.mma) =====
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

| 模式 | AI | 瓶颈 | 能否用 tcgen05.mma |
|------|-----|-----|-------------------|
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

## 5. 技术栈深度对比：Raw CUDA vs CuTe vs CuTile vs Triton

### 5.1 技术栈概览

| 技术栈 | 抽象级别 | 语言 | 核心特点 |
|--------|---------|------|---------|
| **Raw CUDA** | 低 | C++ | 完全控制，手动管理一切 |
| **CuTe** | 中 | C++ | Layout algebra + Swizzle 抽象 |
| **CuTile** | 中 | C++ | CuTe 的 Tile 迭代器抽象 |
| **Triton** | 高 | Python | Auto-tuning，跨平台 |

### 5.2 我们的版本演进

| 版本 | 技术栈 | 核心优化 | 代码行数 | 带宽利用率 |
|------|--------|---------|---------|-----------|
| v5 | Triton | Auto-tuning | ~200 | 35% |
| v7 | Raw CUDA | float4 向量化 | ~650 | 95% |
| v8 | Raw CUDA | Warp Specialization | ~650 | 95% |
| **v9** | **CuTe** | **SMEM Swizzle** | ~400 | **95%** |
| v10 | CuTe/CuTile | Layout Algebra | ~350 | 95% |

### 5.3 Raw CUDA 实现 (v7/v8)

直接使用 CUDA C++ 编写，手动管理所有细节：

```cpp
// ============================================
// v7: 手动 float4 向量化加载
// ============================================
__global__ void gdn_decode_v7(float* state, float* q, float* k, ...) {
    // 手动计算索引
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int head_id = blockIdx.y;
    
    // 手动向量化加载 (4x 带宽效率)
    float4* state_f4 = reinterpret_cast<float4*>(&state[state_offset]);
    float4 s = state_f4[tid];
    
    // 解包并计算 (手动展开)
    float old_v = 0.0f;
    old_v += s.x * k[tid * 4 + 0];
    old_v += s.y * k[tid * 4 + 1];
    old_v += s.z * k[tid * 4 + 2];
    old_v += s.w * k[tid * 4 + 3];
    
    // 手动 warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        old_v += __shfl_xor_sync(0xffffffff, old_v, offset);
    }
    
    // ... 后续 delta rule 更新
}

// ============================================
// v8: 手动 Warp Specialization
// ============================================
__global__ void gdn_decode_v8(...) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    __shared__ float smem_state[128 * 128];
    __shared__ float smem_k[128];
    
    if (warp_id < 2) {
        // ===== Compute Warps =====
        // 执行矩阵-向量乘法
        float sum = 0.0f;
        for (int d = lane_id; d < 128; d += 32) {
            sum += smem_state[v_idx * 128 + d] * smem_k[d];
        }
        // warp reduction...
    } else {
        // ===== Memory Warp =====
        // 异步预取下一个 batch 的 state
        for (int i = lane_id; i < 128; i += 32) {
            smem_state_next[i] = state_next[i];
        }
    }
    __syncthreads();
}
```

**优点**: 
- 完全控制寄存器/SMEM 分配
- 性能可预测，无黑盒
- 可使用所有 PTX 指令

**缺点**: 
- 代码冗长 (~650 行/kernel)
- 容易出错 (Bank Conflict、Race Condition)
- 难以维护和修改

### 5.4 CuTe 实现 (v9)

CuTe 是 NVIDIA CUTLASS 库的核心，提供 Layout Algebra：

```cpp
#include <cute/tensor.hpp>
#include <cute/swizzle.hpp>

using namespace cute;

// ============================================
// CuTe: 声明式定义 SMEM 布局
// ============================================

// 定义 State 的逻辑布局: [V, K] = [128, 128]
using StateShape  = Shape<_128, _128>;
using StateStride = Stride<_128, _1>;  // Row-major
using StateLayout = Layout<StateShape, StateStride>;

// 定义 Swizzle 消除 Bank Conflict
// Swizzle<B, M, S>: B=bits used, M=base, S=shift
using StateSwizzle = Swizzle<3, 3, 3>;

// 组合 Layout + Swizzle
using SwizzledStateLayout = decltype(
    composition(StateSwizzle{}, StateLayout{})
);

// ============================================
// v9 Kernel: 使用 CuTe 抽象
// ============================================
__global__ void gdn_decode_v9(float* state_ptr, float* q, float* k, ...) {
    // 创建 SMEM tensor (自动应用 swizzle)
    __shared__ float smem_state_raw[128 * 128];
    auto smem_state = make_tensor(
        make_smem_ptr(smem_state_raw), 
        SwizzledStateLayout{}
    );
    
    // 创建 Global tensor
    auto gstate = make_tensor(
        make_gmem_ptr(state_ptr + batch_offset),
        StateLayout{}
    );
    
    // CuTe copy: 自动处理 swizzle 索引转换
    // 从 Global 到 SMEM
    copy(gstate, smem_state);
    __syncthreads();
    
    // 计算: 访问时自动应用 swizzle
    int lane_id = threadIdx.x % 32;
    float old_v = 0.0f;
    for (int d = lane_id; d < 128; d += 32) {
        // smem_state(v_idx, d) 自动转换为 swizzled 地址
        old_v += smem_state(v_idx, d) * k[d];
    }
    
    // ... 后续计算
}
```

**关键概念**:

| CuTe 概念 | 作用 | 示例 |
|----------|------|------|
| `Shape` | Tensor 形状 | `Shape<_128, _128>` |
| `Stride` | 内存步长 | `Stride<_128, _1>` (row-major) |
| `Layout` | Shape + Stride | `Layout<Shape, Stride>` |
| `Swizzle` | Bank conflict 消除 | `Swizzle<3,3,3>` |
| `make_tensor` | 创建 tensor 视图 | `make_tensor(ptr, layout)` |

### 5.5 CuTile 实现 (v10)

CuTile 是 CuTe 的高级抽象，提供 Tile 迭代器：

```cpp
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

using namespace cute;

// ============================================
// CuTile: Tile 迭代和分区抽象
// ============================================

// 定义 Tile 形状
using BlockShape = Shape<_64, _64, _32>;  // M=64, N=64, K=32

// 定义 TiledCopy: 如何将 Global 数据拷贝到 SMEM
using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, float>;
using GmemTiledCopy = decltype(
    make_tiled_copy(
        GmemCopyAtom{},
        Layout<Shape<_32, _4>>{},    // Thread layout
        Layout<Shape<_4, _1>>{}      // Value layout per thread
    )
);

// 定义 TiledMMA: 如何执行矩阵乘法 (for Prefill)
using MmaAtom = MMA_Atom<SM100_64x64x16_F32BF16BF16_SS>;  // Blackwell tcgen05.mma
using TiledMma = TiledMMA<MmaAtom, Layout<Shape<_2, _2, _1>>>;

// ============================================
// v10 Kernel: 使用 CuTile 抽象
// ============================================
template<int BLOCK_V>
struct SwizzledStateLayout {
    // CuTile 风格的 Layout 计算
    using SwizzleType = Swizzle<3, 3, 3>;
    
    static constexpr int D = 128;
    
    __device__ __forceinline__
    static int get_index(int v_idx, int d_idx) {
        // Swizzle<3,3,3>: d ^ ((d >> 3) & 7)
        int swizzled_d = d_idx ^ ((d_idx >> 3) & 7);
        return v_idx * D + swizzled_d;
    }
};

__global__ void gdn_decode_v10(...) {
    // 使用 CuTile 的 Swizzle 计算
    using SL = SwizzledStateLayout<BLOCK_V>;
    
    __shared__ float smem_state[BLOCK_V * 128];
    
    // 加载到 SMEM (with swizzle)
    for (int v = threadIdx.x; v < BLOCK_V; v += blockDim.x) {
        for (int d = 0; d < 128; d++) {
            int smem_idx = SL::get_index(v, d);
            smem_state[smem_idx] = state[global_idx(v, d)];
        }
    }
    __syncthreads();
    
    // 计算 (with swizzle)
    float old_v = 0.0f;
    for (int d = lane_id; d < 128; d += 32) {
        int smem_idx = SL::get_index(v_idx, d);
        old_v += smem_state[smem_idx] * smem_k[d];
    }
    
    // ...
}

// ============================================
// Prefill with tcgen05.mma (规划中)
// ============================================
__global__ void gdn_prefill_v11(...) {
    // 创建 TiledMMA
    TiledMma tiled_mma;
    
    // 分区 tensor
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCrA = thr_mma.partition_fragment_A(sA);  // S state
    auto tCrB = thr_mma.partition_fragment_B(sB);  // Q chunk
    auto tCrC = thr_mma.partition_fragment_C(sC);  // Output
    
    // 执行 tcgen05.mma
    gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
}
```

**CuTe vs CuTile**:

| 维度 | CuTe | CuTile |
|------|------|--------|
| 抽象级别 | Layout + Swizzle | + Tile 迭代器 |
| 主要用途 | 内存布局 | Tile 级并行 |
| Tensor Core | 需手动 | TiledMMA 抽象 |
| 学习曲线 | 陡峭 | 更陡峭 |
| 适用场景 | 内存密集型 | 计算密集型 |

### 5.6 Swizzle 原理详解

**Bank Conflict 问题**:
```
SMEM 有 32 个 bank，每 4 字节一个 bank
连续地址: addr % 32 决定 bank
如果 32 个线程访问同一 bank → 32-way conflict!
吞吐量降低 32x!
```

**Swizzle 解决方案 (XOR-based)**:
```cpp
// Swizzle<B, M, S> = Swizzle<3, 3, 3>
// B=3: 8 个 bank group (2^3 = 8)
// M=3: 8 个 mask bits
// S=3: 8 个 shift bits

// 物理索引 = 逻辑索引 XOR (逻辑索引 >> 3) & 7
int swizzled_idx = logical_idx ^ ((logical_idx >> 3) & 7);

// 示例:
// logical  = 0  1  2  3  4  5  6  7  8  9  ...
// shift    = 0  0  0  0  0  0  0  0  1  1  ...
// mask     = 0  0  0  0  0  0  0  0  1  1  ...
// physical = 0  1  2  3  4  5  6  7  9  8  ...
//                                    ↑  ↑
//                              XOR 交换了 8 和 9
```

**效果**: 将 8-way bank conflict 降为 1-way，SMEM 吞吐量提升 **8x**。

### 5.7 技术栈对比总结

| 维度 | Raw CUDA | CuTe | CuTile | Triton |
|------|----------|------|--------|--------|
| **语言** | C++ | C++ | C++ | Python |
| **抽象级别** | 低 | 中 | 中高 | 高 |
| **SMEM 控制** | 手动 | 声明式 | 声明式 | 自动 |
| **Bank Conflict** | 手动 XOR | `Swizzle<B,M,S>` | `Swizzle<B,M,S>` | 自动 |
| **Tensor Core** | 手动 PTX | 手动 | `TiledMMA` | 自动 |
| **学习成本** | 高 | 中高 | 高 | 低 |
| **代码量** | ~650 行 | ~400 行 | ~350 行 | ~200 行 |
| **性能上限** | 最高 | 最高 | 最高 | 略低 |
| **我们的选择** | v7/v8 | **v9** | v10 | v5 (baseline) |

> **结论**: CuTe 是最佳平衡点——获得接近 Raw CUDA 的性能，同时保持代码可维护性。

---

## 6. 性能对比

### 6.1 Decode 各版本对比 (Batch=256, B200)

| 版本 | 技术栈 | 带宽 | vs Triton | 关键优化 |
|------|--------|------|-----------|---------|
| Triton v5 | Triton | 2,834 GB/s | 1.0x | Auto-tuning |
| CUDA v7 | Raw CUDA | 7,578 GB/s | 2.67x | float4 向量化 |
| CUDA v8 | Raw CUDA | 7,605 GB/s | 2.68x | Warp Specialization |
| **CuTe v9** | **CuTe** | **7,585 GB/s** | **2.68x** | **SMEM Swizzle** |
| CuTile v10 | CuTile | 7,602 GB/s | 2.68x | Layout Algebra |

**观察**:
- Raw CUDA、CuTe、CuTile 性能相当（都达到 **95% 带宽利用率**）
- CuTe/CuTile 代码更简洁，Swizzle 逻辑由库处理
- Triton 在大 batch 时差距明显（35% vs 95%）

### 6.2 不同 Batch Size 表现

| Batch | Triton v5 | CuTe v9 | 胜者 | 原因 |
|-------|-----------|---------|------|------|
| 1 | 24 GB/s | **27 GB/s** | CuTe | Launch 开销更小 |
| 16 | 386 GB/s | **405 GB/s** | CuTe | 更好的 SMEM 利用 |
| 64 | **1,518 GB/s** | 1,302 GB/s | **Triton** | Auto-tuning 优势 |
| 256 | 2,834 GB/s | **7,585 GB/s** | **CuTe** | Swizzle 消除 conflict |

**洞察**: Triton 在 batch=64 时胜出，可能是其 auto-tuning 选择了更优的 tile 配置。

### 6.3 Prefill 优化路径 (规划中)

```
目标: 利用 tcgen05.mma (2.25 PFLOPS BF16, 2x Hopper)

推荐实现:
✅ Chunked Recurrence (chunk_size=64)
✅ tcgen05.mma for S @ Q_chunk
✅ TMA for async bulk loads
✅ CuTile TiledMMA abstraction

挑战:
⚠️ State 更新仍为顺序
⚠️ 需要精细的 SMEM 管理
```

| 方案 | 预期性能 |
|------|---------|
| 当前 Triton | Baseline |
| Chunked + tcgen05.mma | **2-4x** (TBD) |

---

## 7. 结论

| 阶段 | 本质 | 优化重点 | Tensor Core |
|------|------|---------|-------------|
| **Decode** | 矩阵-向量 × L次 | 内存带宽 | ❌ 无法使用 |
| **Prefill** | 矩阵-矩阵 (chunked) | tcgen05.mma | ✅ 可利用 |

> **核心洞察**：不是所有算子都应该追求 Tensor Core。对于 Memory-Bound 的 Decode，优化带宽才是正道；对于可转换为 GEMM 的 Prefill，Tensor Core (tcgen05.mma) 是关键加速手段。

### 技术选型建议

| 场景 | 推荐技术栈 | 原因 |
|------|-----------|------|
| 快速原型 | Triton | Python，学习成本低 |
| 内存密集型 | **CuTe** | Swizzle 抽象，代码简洁 |
| 计算密集型 | CuTile | TiledMMA 支持 tcgen05.mma |
| 极致性能 | Raw CUDA | 完全控制 |

---

## 参考

- [FlashInfer GDN Implementation](https://github.com/flashinfer-ai/flashinfer)
- [NVIDIA Blackwell Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [CUTLASS tcgen05.mma Documentation](https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/blackwell_functionality.html)
- [CuTe Documentation](https://github.com/NVIDIA/cutlass/tree/main/include/cute)

---

*作者注：本文数据基于 NVIDIA B200 GPU (Modal Cloud) 实测，代码开源于 [flashinfer-bench-tma-thrust](https://github.com/xxx/flashinfer-bench-tma-thrust)。*

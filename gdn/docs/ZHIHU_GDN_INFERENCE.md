# 深入理解 Qwen3.5 的 Gated DeltaNet：从云端加速到端侧部署

> 本文从推理优化的视角，深入分析 Qwen3.5 采用的 Gated DeltaNet (GDN) 架构，解释为什么这种看似"慢"的算法实际上是长上下文和端侧部署的最优解。

---

## 一、Qwen3.5 的混合架构设计

Qwen3.5 (Qwen3-Next) 采用了一种创新的**混合注意力架构**，将传统 Softmax Attention 与 Gated DeltaNet 按 **1:3 的比例**混合使用：

```
┌─────────────────────────────────────────────────────────┐
│  Qwen3.5 Layer Structure (每4层为一组)                   │
├─────────────────────────────────────────────────────────┤
│  Layer 1:  Softmax Attention  (完整 KV Cache)           │
│  Layer 2:  Gated DeltaNet     (固定状态矩阵)            │
│  Layer 3:  Gated DeltaNet     (固定状态矩阵)            │
│  Layer 4:  Gated DeltaNet     (固定状态矩阵)            │
│  ... 重复 ...                                           │
└─────────────────────────────────────────────────────────┘
```

### 为什么是 3:1？

这个比例经过精心设计，平衡了**模型质量**和**内存效率**：

| 层类型 | 优势 | 劣势 |
|--------|------|------|
| Softmax Attention | 建模能力最强 | KV Cache 随上下文线性增长 |
| Gated DeltaNet | 固定内存，O(1) | 建模能力略弱 |

通过保留 1/4 的 Attention 层确保模型质量，同时用 3/4 的 GDN 层大幅降低内存开销。

---

## 二、GDN 的核心原理：Delta Rule

GDN 的核心是**增量学习规则 (Delta Rule)**，源自联想记忆理论。状态矩阵 $S \in \mathbb{R}^{d \times d}$ 充当一个可学习的键值存储：

```python
# GDN 单步更新 (简化版)
def gdn_step(S, q, k, v, g, beta):
    # 1. 检索：用 key 查询当前记忆
    old_v = S @ k                    # 状态矩阵预测的 value
    
    # 2. 计算误差并更新
    delta = beta * (v - old_v)       # 预测误差 × 更新门控
    S_new = g * S + outer(delta, k)  # 衰减旧记忆 + 写入新关联
    
    # 3. 输出
    output = scale * S_new @ q       # 用 query 读取
    return output, S_new
```

**关键洞察**：这个更新是**严格顺序依赖**的！

```
S₀ → old_v₀ → delta₀ → S₁ → old_v₁ → delta₁ → S₂ → ...
     ↑                      ↑
     无法跳过               必须等待 S₁
```

这意味着**无法使用 Parallel Scan** 进行 token 级并行——每个 token 的输出依赖于前一个 token 更新后的状态。

---

## 三、云端推理加速策略

既然 GDN 有顺序依赖，Qwen3.5 是如何实现高效推理的？

### 策略 1：Chunkwise Parallel（训练 & Prefill）

虽然 token 级无法并行，但可以在 **chunk 级别**实现并行：

```
传统方式：逐 token 更新 (O(L) 顺序步骤)
token₁ → token₂ → token₃ → ... → tokenₗ

Chunkwise Parallel：分块处理 (O(L/C) 顺序步骤)
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Chunk 1 │──▶│ Chunk 2 │──▶│ Chunk 3 │
│ 64 tok  │   │ 64 tok  │   │ 64 tok  │
└─────────┘   └─────────┘   └─────────┘
    ↑              ↑            ↑
 chunk 内       chunk 内      chunk 内
 矩阵乘法       矩阵乘法      矩阵乘法
(Tensor Core)  (Tensor Core) (Tensor Core)
```

**核心思想**：
- **Chunk 内部**：转化为密集矩阵乘法，可以使用 Tensor Core 加速
- **Chunk 之间**：顺序传递状态，但只有 L/C 步

这使得 **Prefill 阶段**（处理用户输入 prompt）可以高效利用 GPU 的计算能力。

### 策略 2：Multi-Batch 并行（Decode）

Decode 阶段（逐 token 生成）无法 chunk 化，但可以**批量处理多个请求**：

我们在 NVIDIA B200 上的实测数据：

| 配置 | 总 Tokens | 耗时 (ms) | 吞吐 (M tok/s) |
|------|-----------|-----------|----------------|
| N=1, L=4096 | 4096 | 3.31 | **1.24** |
| N=4, L=1024 | 4096 | 0.87 | **4.70** |
| N=16, L=256 | 4096 | 0.36 | **11.38** |
| **N=32, L=128** | **4096** | **0.29** | **14.13** |
| N=64, L=64 | 4096 | 0.30 | 13.70 |

**关键发现**：
1. **单序列吞吐恒定**：无论 L=512 还是 L=4096，都是 ~1.2 M tok/s（顺序依赖的体现）
2. **批量线性扩展**：N=32 时达到 14.1 M tok/s，是单序列的 **11.7 倍**
3. **存在饱和点**：N>32 后吞吐不再增长（计算资源饱和）

### 策略 3：内存带宽优化

GDN Decode 是**内存受限**的（Arithmetic Intensity ~0.87 FLOP/B），优化重点是减少内存访问：

- **TMA Double-Buffering**：预取下一个 chunk 的数据，隐藏内存延迟
- **cp.async Prefetch**：异步加载状态矩阵
- **FP8/BF16 量化**：减少状态矩阵的内存占用

---

## 四、为什么云端还要用 GDN？

读到这里你可能会问：既然 GDN 这么"慢"，为什么云端还要用？

答案是：**GDN 的优势不是速度，而是内存！**

### 内存对比（100K 上下文）

```python
# 标准 Attention 的 KV Cache
kv_cache = num_layers * context_len * head_dim * 2  # K + V
         = 32 * 100,000 * 128 * 2 * 2 bytes
         = 1.6 GB  # 每个请求！

# GDN 的状态矩阵
gdn_state = num_layers * head_dim * head_dim * 4 bytes
          = 32 * 128 * 128 * 4
          = 2 MB  # 恒定！
```

**100K 上下文时，GDN 内存占用仅为 Attention 的 1/800！**

### 实际部署影响

```python
# 假设 GPU 有 80GB 显存，模型占用 30GB

# 纯 Attention 模型 (100K 上下文)
可用内存 = 80 - 30 = 50 GB
KV Cache/请求 = 1.6 GB
最大并发 = 50 / 1.6 ≈ 31 个请求

# Qwen3.5 混合模型 (3:1 GDN:Attention)
Attention 层 KV = 8 层 * 100K * 128 * 2 * 2B = 400 MB
GDN 层状态 = 24 层 * 128 * 128 * 4B = 1.5 MB
总计/请求 ≈ 402 MB
最大并发 = 50 GB / 402 MB ≈ 124 个请求  # 4倍提升！
```

---

## 五、端侧部署：GDN 的主场

云端的优势已经很明显，但 GDN 真正的主场是**端侧设备**。

### 端侧设备的特点

| 特性 | 云端 GPU | 端侧 NPU/DSP |
|------|----------|--------------|
| 总内存 | 80-192 GB | 4-16 GB |
| 内存带宽 | 2-8 TB/s | 50-200 GB/s |
| 片上 SRAM | 50 MB (L2 Cache) | 1-8 MB |
| 典型 Batch | 16-128 | **1** |

### GDN 在端侧的独特优势

#### 1. 状态矩阵可驻留片上 SRAM

```
┌─────────────────────────────────────────────────────────┐
│  端侧 NPU 内存架构                                       │
│                                                         │
│  ┌─────────────┐                                        │
│  │  片上 SRAM  │ ← GDN 状态 (64KB/head) 完全装下！       │
│  │   2-4 MB    │   无需访问外部 DRAM                     │
│  └──────┬──────┘                                        │
│         │                                               │
│         │  ← Attention KV Cache (MB-GB级) 必须走这里    │
│         ▼                                               │
│  ┌─────────────┐                                        │
│  │   DRAM      │   访问速度慢 10-100x                   │
│  │   8-16 GB   │   功耗高                                │
│  └─────────────┘                                        │
└─────────────────────────────────────────────────────────┘
```

**FPGA 研究验证**：将 GDN 状态持久化在 BRAM 中，相比 H100 GPU：
- 延迟降低 **4.5 倍**
- 能效提升 **60 倍**

#### 2. 无限上下文，恒定内存

```python
# 机器人场景：连续对话 8 小时

# Attention 模型
context_tokens = 8 * 60 * 60 * 10  # 假设每秒 10 tokens
               = 288,000 tokens
kv_cache = 288K * 128 * 2 * 2B * 32 layers
         = 4.7 GB  # 超出端侧设备内存！
         
# 必须截断上下文，丢失早期记忆

# GDN 模型
state_memory = 32 * 128 * 128 * 4B
             = 2 MB  # 恒定！
             
# 完整保留所有交互历史
```

#### 3. 可预测的延迟

| 模型 | 延迟特性 | 原因 |
|------|----------|------|
| Attention | **不可预测**，随上下文增长 | KV Cache 查询复杂度 O(L) |
| GDN | **恒定**，与上下文无关 | 状态矩阵固定大小 |

对于实时性要求高的机器人控制，可预测的延迟至关重要。

#### 4. 更低功耗

```
功耗 ∝ DRAM 访问次数

Attention: 每 token 读取 O(L) 的 KV Cache → 高功耗
GDN: 每 token 读写 O(d²) 的状态 (可驻留 SRAM) → 低功耗
```

---

## 六、总结：不同场景的最优选择

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| **云端高并发** | Attention + GQA | Tensor Core 利用率高 |
| **云端长上下文** | **混合架构 (Qwen3.5)** | 内存效率 + 质量平衡 |
| **端侧 Batch=1** | **GDN 主导** | 状态驻留 SRAM，功耗低 |
| **机器人/IoT** | **GDN 主导** | 无限上下文，低延迟 |

### 核心洞察

> **GDN 不是为了更快，而是为了用更少的内存做到同样的事。**

在内存受限的场景（长上下文、端侧设备）中，GDN 的"慢"是可接受的代价，换来的是：

1. **100K+ 上下文**而不 OOM
2. **更高的并发**（云端）
3. **更低的功耗**（端侧）
4. **可预测的延迟**（实时系统）

这就是为什么 Qwen3.5、Kimi 等前沿模型都选择了这条路。

---

## 附录：我们的优化实践

在 FlashInfer GDN 内核优化中，我们针对 NVIDIA B200 GPU 进行了多轮迭代：

| 迭代 | 优化技术 | 效果 |
|------|----------|------|
| Baseline | Triton v5 | 1x |
| Iter 1 | cp.async prefetch | 减少内存延迟 |
| Iter 4 | TMA double-buffering | 计算/访存重叠 |
| Iter 5 | Warp-cooperative reduction | 8x 并行 dot product |
| Iter 6 | Multi-batch analysis | 确认 N=32 最优 |

完整优化日志和代码：[GitHub 链接]

---

*欢迎关注后续文章，我们将深入探讨 Chunkwise Parallel 算法的实现细节，以及如何在不同硬件平台上优化 GDN 推理。*

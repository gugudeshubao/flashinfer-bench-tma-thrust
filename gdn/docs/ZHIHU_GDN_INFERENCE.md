# 深入理解 Qwen3.5 的 Gated DeltaNet：为什么"慢"的算法反而是最优解？

> 本文从推理优化的视角，深入分析 Qwen3.5 采用的 Gated DeltaNet (GDN) 架构。你会发现，这种看似"低效"的顺序计算，实际上解决了大模型部署中最棘手的问题。

---

## 一、一个反直觉的架构选择

当我们第一次看到 Qwen3.5 的架构时，会感到困惑：

```
Qwen3.5 每 4 层的结构：
  Layer 1:  Softmax Attention
  Layer 2:  Gated DeltaNet (GDN)
  Layer 3:  Gated DeltaNet (GDN)  
  Layer 4:  Gated DeltaNet (GDN)
```

**3/4 的层使用了 GDN，而不是久经验证的 Softmax Attention？**

更让人困惑的是，GDN 有一个"致命缺陷"——它是**严格顺序依赖**的，无法像 Attention 那样在序列维度上并行计算。

那么问题来了：
- 为什么要用这种"慢"的算法？
- Qwen3.5 是如何让它变快的？
- 这种设计到底解决了什么问题？

让我们一层层揭开答案。

---

## 二、GDN 的核心机制：Delta Rule

要理解 GDN 的价值，首先要理解它的工作原理。

### 2.1 状态矩阵：一个可学习的键值存储

GDN 维护一个 $S \in \mathbb{R}^{128 \times 128}$ 的**状态矩阵**，可以理解为一个"压缩的记忆"：

```python
# 传统 Attention：显式存储所有历史 KV
kv_cache = [(k₁,v₁), (k₂,v₂), ..., (kₗ,vₗ)]  # O(L) 内存

# GDN：压缩到固定大小的矩阵
state = S  # 128×128，始终 64KB
```

### 2.2 增量更新规则

GDN 的更新遵循**增量学习 (Delta Rule)**，源自联想记忆理论：

```python
def gdn_update(S, q, k, v, g, beta):
    """GDN 单 token 更新"""
    
    # Step 1: 检索 —— 用 key 查询当前记忆预测的 value
    old_v = S @ k
    
    # Step 2: 计算误差 —— 真实 value 与预测的差异
    error = v - old_v
    
    # Step 3: 增量更新 —— 只修正误差部分
    delta = beta * error
    S_new = g * S + outer(delta, k)
    
    # Step 4: 输出 —— 用 query 读取更新后的状态
    output = S_new @ q
    
    return output, S_new
```

**关键洞察**：每个 token 的更新依赖于**前一个 token 更新后的状态**：

```
S₀ ──▶ S₁ ──▶ S₂ ──▶ S₃ ──▶ ...
   ↑      ↑      ↑      ↑
 old_v₀ old_v₁ old_v₂ old_v₃  (必须顺序计算)
```

这就是为什么 GDN **无法使用 Parallel Scan** 进行 token 级并行——`old_v = S @ k` 创造了不可绕过的数据依赖。

---

## 三、顺序依赖的代价：实测数据

我们在 NVIDIA B200 GPU 上对 GDN Prefill 进行了详细测试：

### 3.1 单序列：吞吐与长度无关

| 序列长度 | 总 Tokens | 耗时 (ms) | 吞吐 (M tok/s) |
|----------|-----------|-----------|----------------|
| 512 | 512 | 0.42 | 1.21 |
| 1024 | 1024 | 0.84 | 1.23 |
| 2048 | 2048 | 1.66 | 1.23 |
| 4096 | 4096 | 3.31 | **1.24** |

**发现**：无论序列多长，单序列吞吐都是 ~1.2 M tok/s。时间与长度**严格线性**——这正是顺序依赖的特征。

### 3.2 Tensor Core 无法救场

GDN 的计算强度 (Arithmetic Intensity) 约为 0.87 FLOP/B，而 H100 的 ridge point 是 25.6 FLOP/B。

```
GDN Decode 位于 Roofline 模型的哪里？

                    ┌─────────── Compute Bound ───────────
                    │
Throughput          │                    ★ Attention (batched)
    │               │              
    │         ┌─────┤           
    │         │     │     
    │    ★ GDN│     │  ← Memory Bound (带宽受限)
    │         │     │
    └─────────┴─────┴──────────────────────────────────▶
              ↑                                   Arithmetic
           0.87                                   Intensity
```

**结论**：GDN 是**内存受限**的，Tensor Core 的算力根本用不上。

---

## 四、Qwen3.5 的加速策略

既然单序列这么"慢"，Qwen3.5 是如何让 GDN 变得实用的？

### 4.1 策略一：Chunkwise Parallel（Prefill 阶段）

虽然 token 级无法并行，但可以在 **chunk 级别**实现并行：

```
传统方式：逐 token 顺序处理
t₁ → t₂ → t₃ → t₄ → t₅ → t₆ → ... → tₗ  (L 步)

Chunkwise Parallel：分块 + 块间顺序
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Chunk 1  │ ─▶ │ Chunk 2  │ ─▶ │ Chunk 3  │  (L/C 步)
│ t₁~t₆₄   │    │ t₆₅~t₁₂₈ │    │ t₁₂₉~... │
│ (matmul) │    │ (matmul) │    │ (matmul) │
└──────────┘    └──────────┘    └──────────┘
     ↑               ↑               ↑
  Tensor Core    Tensor Core     Tensor Core
  块内并行        块内并行         块内并行
```

**核心思想**：
- **块内**：转化为密集矩阵乘法，Tensor Core 可以发挥作用
- **块间**：顺序传递状态，但只需 L/C 步（C=64 时，减少 64 倍顺序步骤）

这使得 **Prefill**（处理用户输入）可以高效利用 GPU 算力。

### 4.2 策略二：Multi-Batch（Decode 阶段）

Decode（逐 token 生成）无法 chunk 化，但可以**并行处理多个请求**：

| Batch Size | 序列长度 | 吞吐 (M tok/s) | 相对加速 |
|------------|----------|----------------|----------|
| N=1 | 4096 | 1.24 | 1x |
| N=4 | 1024 | 4.70 | 3.8x |
| N=8 | 512 | 7.65 | 6.2x |
| N=16 | 256 | 11.38 | 9.2x |
| **N=32** | **128** | **14.13** | **11.4x** |
| N=64 | 64 | 13.70 | 11.1x |

**发现**：
1. 吞吐随 batch 线性增长，直到 N=32
2. N=32 时达到峰值 14.1 M tok/s，是单序列的 **11.4 倍**
3. 超过 N=32 后饱和（GPU 资源打满）

**生产环境的启示**：Qwen3.5 服务应该将多个用户请求 batch 在一起处理。

### 4.3 策略三：内存带宽优化

既然 GDN 是内存受限的，优化重点应该放在减少内存访问上：

| 优化技术 | 原理 | 效果 |
|----------|------|------|
| **TMA Double-Buffering** | 预取下一 chunk，隐藏延迟 | 计算/访存重叠 |
| **cp.async Prefetch** | 异步加载状态矩阵 | 减少等待时间 |
| **FP8/BF16 量化** | 减少状态矩阵位宽 | 带宽需求减半 |

---

## 五、回到核心问题：为什么要用 GDN？

讲了这么多优化，但还没回答最根本的问题：**GDN 到底解决了什么问题？**

答案是：**内存，内存，还是内存！**

### 5.1 KV Cache 的内存危机

标准 Attention 需要为每一层保存完整的 KV Cache：

```python
# 100K 上下文，32 层模型
kv_cache_per_layer = context_len * head_dim * 2  # K + V
                   = 100,000 * 128 * 2 * 2 bytes
                   = 51.2 MB

total_kv_cache = 32 layers * 51.2 MB = 1.6 GB  # 每个请求！
```

### 5.2 GDN 的内存优势

GDN 用固定大小的状态矩阵替代增长的 KV Cache：

```python
# 任意上下文长度，32 层模型
state_per_layer = head_dim * head_dim * 4 bytes
                = 128 * 128 * 4 = 64 KB

total_state = 32 layers * 64 KB = 2 MB  # 恒定！
```

**对比**：100K 上下文时，**GDN 内存是 Attention 的 1/800！**

### 5.3 对部署的实际影响

```python
# 假设 80GB 显存，模型占用 30GB，100K 上下文

# 纯 Attention 模型
可用内存 = 50 GB
每请求 KV Cache = 1.6 GB
最大并发 = 50 / 1.6 ≈ 31 请求

# Qwen3.5 混合模型 (1:3 Attention:GDN)
Attention 层 (8层) = 8 * 51.2 MB = 410 MB
GDN 层 (24层) = 24 * 64 KB = 1.5 MB
每请求总计 ≈ 412 MB
最大并发 = 50 GB / 412 MB ≈ 121 请求  # 4倍并发能力！
```

### 5.4 混合架构的精妙平衡

| 层类型 | 占比 | 角色 |
|--------|------|------|
| Softmax Attention | 25% | 保证模型质量，处理需要精确全局注意的场景 |
| Gated DeltaNet | 75% | 扛住内存压力，处理大部分上下文信息 |

这就是 **3:1 比例**的设计哲学：用少量 Attention 保质量，用大量 GDN 省内存。

---

## 六、超越云端：GDN 在端侧的独特优势

最后，值得一提的是，GDN 的特性使它在**端侧设备**上有着独特优势：

| 特性 | 云端 GPU | 端侧 NPU |
|------|----------|----------|
| 内存 | 80+ GB | 4-16 GB |
| Batch 大小 | 16-128 | **通常 =1** |
| 片上 SRAM | Cache 不可控 | 可编程 SRAM |

对于 Batch=1 的端侧场景：

1. **GDN 状态 (64KB/head) 可以完全驻留在片上 SRAM**，避免反复访问外部内存
2. **支持无限上下文**而不会 OOM（机器人 8 小时对话？没问题）
3. **延迟恒定可预测**，适合实时控制系统

有研究表明，在 FPGA 上将 GDN 状态持久化到 BRAM，相比 H100 可实现 **4.5x 延迟降低**和 **60x 能效提升**。

---

## 七、总结

回到开头的问题：**为什么 Qwen3.5 要用这种"慢"的算法？**

答案现在清晰了：

> **GDN 不是为了更快，而是为了用更少的内存做到同样的事。**

| 维度 | Attention | GDN |
|------|-----------|-----|
| 单序列速度 | 快 | 慢（顺序依赖） |
| 内存占用 | O(L) 增长 | O(1) 恒定 |
| 长上下文 | 内存瓶颈 | 无压力 |
| 并发能力 | 受限于 KV Cache | 高并发 |

在 **长上下文** 和 **高并发** 成为刚需的今天，GDN 的"慢"是可以接受的代价，换来的是：

- 支持 100K+ 上下文而不 OOM
- 4 倍以上的服务并发能力
- 端侧设备的可部署性

这就是为什么 Qwen3.5、Kimi 等前沿模型都选择了这条路——**不是因为 GDN 更快，而是因为它让不可能成为可能。**

---

*本文基于我们在 FlashInfer GDN 内核优化项目中的实践经验，完整优化日志和代码见 [GitHub](https://github.com/flashinfer-ai)。*

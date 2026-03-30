# Chunkwise Parallel 详解：如何将 O(L) 顺序计算变成 O(L/C)

> 本文深入解析 Gated DeltaNet 的 Chunkwise Parallel 算法，从数学推导到实现细节，帮你理解这个让"无法并行"的算法变得高效的关键技术。

---

## 一、问题背景：为什么 GDN 无法直接并行？

### 1.1 GDN 的递推公式

Gated DeltaNet 的核心是一个状态递推：

$$
S_t = g_t \cdot S_{t-1} + \Delta_t \cdot k_t^\top
$$

其中：
- $S_t \in \mathbb{R}^{d \times d}$ 是状态矩阵
- $g_t \in \mathbb{R}$ 是衰减门控
- $\Delta_t = \beta_t (v_t - S_{t-1} k_t)$ 是增量更新
- $k_t, v_t \in \mathbb{R}^d$ 是 key 和 value 向量

### 1.2 顺序依赖链

关键问题在于 $\Delta_t$ 的计算：

```python
old_v = S_{t-1} @ k_t     # 必须知道 S_{t-1}
delta = beta * (v - old_v) # 依赖 old_v
S_t = g * S_{t-1} + outer(delta, k)  # 更新状态
```

这形成了严格的依赖链：

```
S₀ → old_v₀ → Δ₀ → S₁ → old_v₁ → Δ₁ → S₂ → ...
```

**每个 token 必须等待前一个 token 完成状态更新**，导致 O(L) 的顺序步骤。

### 1.3 Parallel Scan 为什么不行？

理论上，形如 $S_t = A_t S_{t-1} + B_t$ 的递推可以用 Parallel Scan 并行化。但 GDN 的问题是：

$$
\Delta_t = \beta_t (v_t - \underbrace{S_{t-1} k_t}_{\text{依赖状态}})
$$

$\Delta_t$ **本身就依赖 $S_{t-1}$**，不是预先可知的常数。这打破了 Parallel Scan 的前提条件。

---

## 二、Chunkwise Parallel 的核心思想

### 2.1 分而治之

既然 token 级别无法并行，我们退而求其次：**chunk 级别并行**。

```
原始序列 (L=256 tokens):
t₁ t₂ t₃ ... t₆₄ | t₆₅ t₆₆ ... t₁₂₈ | t₁₂₉ ... t₁₉₂ | t₁₉₃ ... t₂₅₆
└────Chunk 1────┘ └────Chunk 2─────┘ └───Chunk 3───┘ └───Chunk 4───┘

处理方式:
1. Chunk 1 内部：顺序处理 64 个 token → 得到 S₆₄
2. Chunk 2 内部：以 S₆₄ 为初始状态，顺序处理 → 得到 S₁₂₈
3. ...依此类推

顺序步骤: O(L) → O(L/C) = O(4) when C=64
```

但这只是最朴素的分块，还没有利用并行。真正的 Chunkwise Parallel 更精妙。

### 2.2 关键洞察：Chunk 内部可以矩阵化

在一个 chunk 内部，虽然 token 间有依赖，但我们可以把整个 chunk 的计算**重新表述为矩阵运算**。

设 chunk 大小为 $C$，chunk 内的输入为：
- $Q \in \mathbb{R}^{C \times d}$ (queries)
- $K \in \mathbb{R}^{C \times d}$ (keys)  
- $V \in \mathbb{R}^{C \times d}$ (values)
- $G \in \mathbb{R}^{C}$ (gates)
- $B \in \mathbb{R}^{C}$ (betas)

目标是计算：
- $O \in \mathbb{R}^{C \times d}$ (outputs)
- $S_{out} \in \mathbb{R}^{d \times d}$ (chunk 结束时的状态)

---

## 三、数学推导：Chunk 内的矩阵形式

### 3.1 展开递推

设 chunk 的初始状态为 $S_0$，我们逐步展开每个 token 的状态更新：

**Token 1:**
$$
\begin{align}
\text{old\_v}_1 &= S_0 k_1 \\
\Delta_1 &= \beta_1 (v_1 - \text{old\_v}_1) \\
S_1 &= g_1 S_0 + \Delta_1 k_1^\top
\end{align}
$$

**Token 2:**
$$
\begin{align}
\text{old\_v}_2 &= S_1 k_2 = (g_1 S_0 + \Delta_1 k_1^\top) k_2 \\
&= g_1 S_0 k_2 + \Delta_1 (k_1^\top k_2) \\
\Delta_2 &= \beta_2 (v_2 - \text{old\_v}_2) \\
S_2 &= g_2 S_1 + \Delta_2 k_2^\top \\
&= g_2 g_1 S_0 + g_2 \Delta_1 k_1^\top + \Delta_2 k_2^\top
\end{align}
$$

### 3.2 归纳出模式

对于 token $t$ 在 chunk 内，我们可以写出：

$$
S_t = \underbrace{\left(\prod_{i=1}^{t} g_i\right)}_{\gamma_t} S_0 + \sum_{j=1}^{t} \underbrace{\left(\prod_{i=j+1}^{t} g_i\right)}_{\gamma_{j:t}} \Delta_j k_j^\top
$$

其中：
- $\gamma_t = \prod_{i=1}^{t} g_i$ 是累积衰减因子
- $\gamma_{j:t} = \prod_{i=j+1}^{t} g_i$ 是从 token $j$ 到 $t$ 的衰减

### 3.3 输出的矩阵形式

输出 $o_t = S_t q_t$ 可以分解为两部分：

$$
o_t = \underbrace{\gamma_t (S_0 q_t)}_{\text{来自初始状态}} + \underbrace{\sum_{j=1}^{t} \gamma_{j:t} \Delta_j (k_j^\top q_t)}_{\text{来自 chunk 内更新}}
$$

**关键观察**：第二项可以写成**因果注意力**的形式！

定义：
- $A_{tj} = \gamma_{j:t} (k_j^\top q_t)$ 为"衰减注意力分数"
- 这是一个**下三角矩阵**（因果性：$t$ 只能看到 $j \leq t$）

则：
$$
O = \underbrace{\Gamma \cdot (S_0 Q^\top)^\top}_{\text{状态贡献}} + \underbrace{\text{CausalAttn}(\Delta, K, Q, G)}_{\text{Chunk 内交互}}
$$

### 3.4 转化为 Tensor Core 友好的形式

上述公式中的 CausalAttn 可以进一步转化为矩阵乘法：

```python
# 伪代码：Chunk 内计算
def chunk_forward(Q, K, V, G, B, S_init):
    C, d = Q.shape
    
    # Step 1: 计算累积衰减矩阵 Γ (下三角)
    # Γ[i,j] = prod(g[j+1:i+1]) for j < i, else 0
    Gamma = compute_causal_decay_matrix(G)  # [C, C]
    
    # Step 2: 计算"原始"注意力分数
    # A = Q @ K.T  → [C, C]
    A_raw = Q @ K.T
    
    # Step 3: 应用因果掩码和衰减
    A = Gamma * A_raw  # element-wise，下三角
    
    # Step 4: 计算 chunk 内的 delta 贡献
    # 这里需要迭代或矩阵分解来处理 Δ 的依赖
    # ... (见下一节详细算法)
    
    # Step 5: 加上初始状态的贡献
    gamma_vec = cumprod(G)  # [C]
    state_contrib = gamma_vec[:, None] * (S_init @ Q.T).T  # [C, d]
    
    # Step 6: 合并输出
    O = state_contrib + intra_chunk_output
    
    # Step 7: 计算最终状态
    S_out = gamma_vec[-1] * S_init + sum_of_updates
    
    return O, S_out
```

---

## 四、完整算法：两阶段 Chunkwise Parallel

### 4.1 阶段一：Intra-Chunk（Chunk 内并行）

每个 chunk 独立计算，可以**并行处理所有 chunk**：

```python
def intra_chunk_parallel(chunks, S_init_per_chunk):
    """
    输入:
      chunks: List of (Q, K, V, G, B), 每个 shape [C, d]
      S_init_per_chunk: 每个 chunk 的初始状态 [num_chunks, d, d]
    
    输出:
      outputs: [num_chunks, C, d]
      S_finals: [num_chunks, d, d]
    """
    outputs = []
    S_finals = []
    
    # 这个循环可以完全并行！
    for chunk_idx, (Q, K, V, G, B) in enumerate(chunks):
        S_init = S_init_per_chunk[chunk_idx]
        
        # Chunk 内计算（矩阵运算，Tensor Core）
        O, S_final = compute_chunk(Q, K, V, G, B, S_init)
        
        outputs.append(O)
        S_finals.append(S_final)
    
    return stack(outputs), stack(S_finals)
```

### 4.2 阶段二：Inter-Chunk（Chunk 间传递）

Chunk 间必须顺序传递状态，但只有 $L/C$ 步：

```python
def inter_chunk_sequential(S_finals, initial_state):
    """
    输入:
      S_finals: 每个 chunk 的"局部最终状态" [num_chunks, d, d]
      initial_state: 序列的初始状态 [d, d]
    
    输出:
      S_corrected: 修正后的最终状态 [num_chunks, d, d]
    """
    num_chunks = len(S_finals)
    S_prev = initial_state
    S_corrected = []
    
    # O(L/C) 顺序步骤
    for i in range(num_chunks):
        # 修正：考虑之前 chunk 的状态传递
        S_curr = propagate_state(S_prev, S_finals[i])
        S_corrected.append(S_curr)
        S_prev = S_curr
    
    return stack(S_corrected)
```

### 4.3 整体流程图

```
输入序列 (L tokens)
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: 分块                                           │
│  [t₁...t_C] [t_{C+1}...t_{2C}] ... [t_{L-C+1}...t_L]   │
│    Chunk 1      Chunk 2      ...      Chunk N          │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: Intra-Chunk 并行 (Tensor Core)                 │
│                                                         │
│  ┌─────────┐ ┌─────────┐     ┌─────────┐               │
│  │ Chunk 1 │ │ Chunk 2 │ ... │ Chunk N │  ← 并行！     │
│  │ MatMul  │ │ MatMul  │     │ MatMul  │               │
│  └────┬────┘ └────┬────┘     └────┬────┘               │
│       │           │               │                     │
│    (O₁,S₁)     (O₂,S₂)        (O_N,S_N)                │
└───────┼───────────┼───────────────┼─────────────────────┘
        │           │               │
        ▼           ▼               ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: Inter-Chunk 顺序传递 (O(L/C) 步)               │
│                                                         │
│  S_init → S₁' → S₂' → ... → S_N'                       │
│           │      │           │                          │
│           ▼      ▼           ▼                          │
│          修正   修正        修正                        │
│          O₁     O₂          O_N                        │
└─────────────────────────────────────────────────────────┘
        │
        ▼
   最终输出 O
```

---

## 五、复杂度分析

### 5.1 时间复杂度

| 操作 | 朴素递推 | Chunkwise Parallel |
|------|----------|-------------------|
| 顺序步骤 | O(L) | O(L/C) |
| 并行计算 | - | O(C²d) per chunk |
| **总计** | O(Ld²) sequential | O(L/C) sequential + O(LC·d) parallel |

当 $C = 64$，$L = 4096$ 时：
- 朴素方法：4096 顺序步骤
- Chunkwise：64 顺序步骤 + 并行矩阵乘法

**加速比：64x 顺序步骤减少！**

### 5.2 空间复杂度

| 存储项 | 大小 | 说明 |
|--------|------|------|
| Chunk 状态 | O(L/C · d²) | 每个 chunk 一个 d×d 矩阵 |
| 注意力矩阵 | O(C²) per chunk | 下三角因果掩码 |
| **总计** | O(Ld²/C + C²) | 远小于 O(Ld²) |

### 5.3 Tensor Core 利用率

Intra-Chunk 阶段的核心操作是：
- $A = Q K^\top$：[C, d] × [d, C] = [C, C] matmul
- $O = A V$：[C, C] × [C, d] = [C, d] matmul

当 $C = 64$, $d = 128$ 时：
- 矩阵大小完美适配 Tensor Core (16×16 或 8×16 tiles)
- 可以达到接近峰值的 Tensor Core 吞吐

---

## 六、实现细节与技巧

### 6.1 处理 Delta 的依赖

最棘手的部分是 $\Delta_t$ 依赖 $S_{t-1}$。有两种处理方式：

**方式 A：迭代求解（精确）**

```python
def compute_deltas_iterative(K, V, G, B, S_init):
    """Chunk 内迭代计算 Δ"""
    C, d = K.shape
    S = S_init.clone()
    deltas = []
    
    for t in range(C):
        old_v = S @ K[t]
        delta = B[t] * (V[t] - old_v)
        deltas.append(delta)
        S = G[t] * S + outer(delta, K[t])
    
    return stack(deltas)  # [C, d]
```

这仍然是 O(C) 顺序，但 C << L，所以可以接受。

**方式 B：矩阵求解（近似/精确）**

对于某些特殊情况，可以将 delta 的计算转化为线性系统求解，但这超出了本文范围。

### 6.2 数值稳定性

累积衰减 $\gamma_t = \prod_{i=1}^{t} g_i$ 可能非常小（当 $g < 1$ 时）。建议：

```python
# 使用 log-space 计算
log_gamma = cumsum(log(G))
gamma = exp(log_gamma)

# 或使用 float32 累积
gamma = cumprod(G.float()).to(G.dtype)
```

### 6.3 Memory-Efficient 实现

为了避免 O(C²) 的注意力矩阵，可以使用 **Flash Attention 风格**的分块：

```python
def memory_efficient_chunk(Q, K, V, G, B, S_init, BLOCK=16):
    """分块计算，避免 O(C²) 显存"""
    C, d = Q.shape
    O = zeros(C, d)
    
    for i in range(0, C, BLOCK):
        q_block = Q[i:i+BLOCK]  # [BLOCK, d]
        
        # 只计算需要的注意力块
        for j in range(0, i + BLOCK, BLOCK):
            k_block = K[j:j+BLOCK]
            v_block = V[j:j+BLOCK]
            
            # 局部注意力 + 因果掩码
            a_block = q_block @ k_block.T  # [BLOCK, BLOCK]
            # ... apply mask and accumulate
        
        O[i:i+BLOCK] = ...
    
    return O
```

---

## 七、与其他方法的对比

### 7.1 vs Parallel Scan

| 特性 | Parallel Scan | Chunkwise Parallel |
|------|--------------|-------------------|
| 适用条件 | $S_t = A_t S_{t-1} + B_t$, A,B 预知 | 状态依赖的更新 |
| GDN 可用 | ❌ (Δ 依赖 S) | ✅ |
| 并行度 | O(log L) depth | O(L/C) sequential + full parallel |
| Tensor Core | 有限 | 充分利用 |

### 7.2 vs Flash Attention

| 特性 | Flash Attention | GDN Chunkwise |
|------|----------------|---------------|
| 算法类型 | 标准 Attention | 递归 + Delta Rule |
| 内存 | O(L) → O(1) | O(Ld²) → O(Ld²/C) |
| IO 优化 | Tiling + Recomputation | Tiling + State Propagation |
| 复杂度 | O(L²d) | O(LCd + Ld²/C) |

### 7.3 vs 纯递推

| 特性 | 纯递推 | Chunkwise Parallel |
|------|--------|-------------------|
| 顺序步骤 | O(L) | O(L/C) |
| Tensor Core | ❌ | ✅ |
| 吞吐 | ~1 M tok/s | ~10+ M tok/s |
| 实现复杂度 | 低 | 中 |

---

## 八、实际效果

### 8.1 Prefill 性能对比

在 NVIDIA B200 GPU 上的 GDN Prefill 测试：

| 方法 | 序列长度 | 耗时 (ms) | 吞吐 (M tok/s) |
|------|----------|-----------|----------------|
| 纯递推 | 4096 | 3.31 | 1.24 |
| Chunkwise (C=64) | 4096 | ~0.5 | ~8.2 |
| Chunkwise + Tensor Core | 4096 | ~0.3 | ~13.6 |

**加速比：6-10x！**

### 8.2 Chunk Size 的选择

| Chunk Size | 优点 | 缺点 |
|------------|------|------|
| C=32 | 更少顺序步骤 | 矩阵太小，Tensor Core 利用率低 |
| **C=64** | **平衡** | **推荐** |
| C=128 | Tensor Core 利用率高 | 更多顺序步骤 |
| C=256 | 最高并行度 | 内存压力大，顺序步骤多 |

通常 **C=64 或 C=128** 是最优选择。

---

## 九、总结

### 核心思想

> **Chunkwise Parallel 的本质是将"无法并行"的 token 级递推，转化为"可以高效并行"的 chunk 级矩阵运算。**

### 三个关键步骤

1. **分块**：将长度 L 的序列分成 L/C 个 chunk
2. **Intra-Chunk 并行**：每个 chunk 内部用矩阵乘法（Tensor Core）
3. **Inter-Chunk 顺序**：chunk 之间顺序传递状态（O(L/C) 步）

### 适用场景

- **Training / Prefill**：处理长序列，需要高吞吐
- **不适用于 Decode**：Decode 每次只处理 1 个 token，无法 chunk

### 实现建议

- Chunk size 选择 64 或 128
- 使用 Flash Attention 风格的分块避免 O(C²) 显存
- 在 log-space 计算累积衰减保证数值稳定

---

*本文是 GDN 系列文章的第二篇。上一篇：[深入理解 Qwen3.5 的 Gated DeltaNet](./ZHIHU_GDN_INFERENCE.md)*

> 人性的偏见是你进步的最大阻碍，没有什么是不可尝试的。

# MoE 里最核心也最容易写错的部分：top-k、counting sort、token 重排怎么实现

前两篇我主要讲了两件事：

1. MoE 算子本质上到底在做什么  
2. 我会怎么从 reference 版，一步步走到真正可用的 GPU 实现

这一篇我想专门讲 MoE 里最核心、但也最容易写错的一段：

> top-k、route metadata、counting sort、token 重排，到底该怎么实现？

如果把 MoE 比作“router + 多个 expert FFN”，那很多人会以为 expert FFN 才是最核心的部分。

但我自己真正开始实现以后，越来越强烈的一个感觉是：

> MoE 里最难、最值钱、最决定整体表现的，不是 FFN，而是路由之后的数据重排。

因为 FFN 至少还是规则的 GEMM；
而 route、bucket、permute、combine 这些步骤，本质上是**稀疏、动态、不规则**的数据流问题。

这一篇我就专门讲这一层。

---

## 1. 为什么我会把这部分当成 MoE 的核心

如果只看数学，MoE 很简单：

- 先算 router logits
- 取 top-k experts
- 去对应 expert 做 FFN
- 把结果加权求和

但在算子实现上，真正麻烦的是：

- 每个 token 去的 expert 不同
- 每个 expert 收到的 token 数不同
- 一个 token 可能去多个 expert
- expert 计算前必须先把 token 按 expert 分组
- expert 计算后还要再还原回原 token 顺序

也就是说，MoE 的实现本质上不是：

> “我做几个 FFN 就行了”

而是：

> “我怎么把一堆稀疏路由关系变成 GPU 喜欢的连续批处理输入”

这一步做不好，后面的 CUDA / Triton / CuTe 再漂亮，也只是把混乱变成更快的混乱。

---

## 2. 先明确输入和输出

我先固定一组最常见的 forward 输入：

- `x [T, H]`
- `router_logits [T, E]`
- `topk = K`

这里：

- `T` = token 数
- `H` = hidden size
- `E` = expert 数
- `K` = 每个 token 去几个 expert

经过 routing 之后，我会得到：

- `topk_idx [T, K]`
- `topk_weight [T, K]`

然后我要做的是：

- 把同一个 expert 的 token 聚到一起
- 形成连续输入
- 让 expert FFN 能按连续区间批处理

最后再把结果放回原 token 的位置。

---

## 3. 第一件事：top-k 本身怎么理解

这一层先不要想 GPU 优化，先想逻辑。

对于第 `t` 个 token，我先有一行 logits：

\[
l_t \in \mathbb{R}^{E}
\]

然后取 top-k：

- `topk_idx[t, :]`
- `topk_vals[t, :]`

再决定 gating weight。

这里有一个我觉得必须先说清楚的点：

### 3.1 gating weight 的定义不唯一

常见做法有两种：

#### 方式 1：先 full softmax，再取 top-k

\[
p = \text{softmax}(l)
\]

然后取 top-k 对应的概率。

#### 方式 2：先取 top-k，再在 top-k 内归一化

\[
p_{topk} = \text{softmax}(l_{topk})
\]

或者更一般一点：

- 先拿到 top-k score
- 再做 normalize

这两种不是完全一样的。

如果模型定义没弄清楚，这里就很容易在数值上和 reference 对不上。

所以我现在做 MoE，第一件事就是先把这一点钉死：

> 到底是 full softmax 后取 top-k，还是 top-k 内再归一化。

不然你后面 dispatch 写得再漂亮，结果也会错。

---

## 4. 第二件事：为什么要把 `[T, K]` 展平成一维路由表

假设我已经有：

- `topk_idx [T, K]`
- `topk_weight [T, K]`

这时候很多人会下意识在脑子里继续保留二维结构。

但如果我要做高效 dispatch，我通常第一步就是把它展平成 `N = T*K` 条路由记录。

也就是：

- `flat_token_ids [N]`
- `flat_expert_ids [N]`
- `flat_weights [N]`

我为什么一定会这样做？

因为后面所有和“按 expert 分桶”有关的操作，本质上都更适合在一维路由记录上做。

比如：

- histogram
- prefix sum
- counting sort
- grouped gather

这些操作放在 `[T, K]` 上做会很绕，但放在 `[N]` 上就顺了。

所以我几乎把这个看成 MoE dispatch 的标准中间表示。

---

## 5. 第三件事：counting sort 为什么比通用 sort 更合适

这是我觉得特别值得强调的点。

很多人一想到“按 expert 重排”，第一反应就是：

- sort by expert id

逻辑上没错，但工程上通常不是最优。

因为 `expert_id` 的取值范围其实很小：

- 就是 `[0, E)`

比如 `E = 8 / 16 / 32 / 64 / 256`

这非常适合做：

- histogram
- prefix sum
- counting sort

而不是通用 radix sort。

我更喜欢把这一步理解成：

> 不是“排序一个大数组”，而是“把有限类别的记录分桶”。

这样实现会更贴近问题本质。

---

## 6. counting sort 的逻辑到底是什么

如果我已经有：

- `flat_expert_ids [N]`
- `flat_token_ids [N]`
- `flat_weights [N]`

那 counting sort 大致分三步。

### 6.1 统计每个 expert 的数量

得到：

- `expert_counts [E]`

这一步本质上是 histogram。

### 6.2 做 prefix sum

得到：

- `expert_offsets [E+1]`

这样 expert `e` 对应的输出区间就是：

- `[expert_offsets[e], expert_offsets[e+1])`

### 6.3 把记录写到对应区间

再根据 `expert_id`，把 token id / weight 写入该 expert 的连续区域。

最终得到：

- `sorted_token_ids [N]`
- `sorted_weights [N]`

于是：

- 同一 expert 的 token 会在内存里连续

这就是后面 grouped FFN 的输入基础。

---

## 7. 这一步到底解决了什么问题

我觉得很多人第一次看 counting sort，会觉得：

“只是换了个顺序，为什么这么重要？”

原因非常简单：

GPU 喜欢连续、规则、成批的数据。

而 router 给我的原始结构恰恰是：

- 稀疏
- 离散
- 不规则
- token 去向混杂

counting sort 做的事情，本质上就是把“不规则的路由关系”变成“按 expert 连续排列的批处理输入”。

从这个角度说，我甚至觉得：

> counting sort 不是一个小技巧，而是 MoE 从“逻辑正确”走向“工程可实现”的关键桥梁。

---

## 8. 如果不做这一步，会发生什么

假设我不做 expert 分桶，那 expert 计算就很容易退化成：

- 每个 token 单独找 expert
- 每个 expert 调一次小算子
- gather/scatter 极度零散

后果通常是：

- launch overhead 爆炸
- GPU SM 利用率很差
- GEMM 规模碎得不成样子
- 很多优化根本无从谈起

所以我现在看 MoE，会把“是否把 token 按 expert 聚成连续区间”当成一个分水岭：

- 聚起来了，后面才有 grouped GEMM 和低层优化的空间
- 没聚起来，后面基本什么都很难做好

---

## 9. 重排之后，expert 输入长什么样

重排之后，我一般会得到：

- `expert_input [N, H]`

它是从原始 `x [T, H]` 按 `sorted_token_ids` gather 出来的。

于是 expert `e` 的输入区间就是：

- `expert_input[start:end]`

其中：

- `start = expert_offsets[e]`
- `end = expert_offsets[e+1]`

这时对于每个 expert，我终于能比较自然地做：

```python
inp = expert_input[start:end]   # [Ne, H]
h = inp @ W1[e]
h = act(h)
out = h @ W2[e]
```

这时问题就从“稀疏路由”变回了“变长分组 GEMM”。

这已经是一个更容易处理的问题了。

---

## 10. Combine 为什么也很关键

dispatch 很重要，但 combine 一样重要。

因为一个 token 可能去多个 expert，所以最后输出不是简单回填，而是：

\[
y_t = \sum_j g_{t,j}\cdot o_{t,j}
\]

工程上常见实现是：

- 根据 `sorted_token_ids` 把 expert 输出 scatter 回去
- 再乘上对应的 weight
- 最后做 scatter-add

如果 `K=1`，这一步还比较简单。

如果 `K=2` 或更大，就会遇到两个问题：

### 10.1 同一 token 可能被多次回写

所以会有：

- 冲突
- 原子加
- 或者分阶段 reduce

### 10.2 combine 可能本身就变成瓶颈

特别是在小 batch / 小 expert token 数时，

- compute 很少
- scatter-add 反而显得重

所以我自己会把 combine 和 dispatch 看成一对：

> dispatch 决定输入是否连续，combine 决定输出是否便宜。

两边都得认真做。

---

## 11. 训练版和推理版在这里的差别

如果我写的是推理版，通常更偏向：

- dropless
- 动态 token 数
- 小 batch latency 优先

这时：

- counting sort
- 动态分桶
- grouped GEMM

往往是更自然的路线。

如果我写的是训练版，尤其大 batch，多 expert parallel 的情况，通常更偏向：

- capacity
- padding
- `[E, C, H]` 这种更规则的布局

因为这样更适合 batched GEMM 和通信。

所以我现在越来越不喜欢问：

“MoE 正确实现应该长什么样？”

而更喜欢问：

> 我现在做的是推理还是训练，我到底在为哪种 workload 优化？

因为这个问题会直接决定 dispatch 的形态。

---

## 12. 我最容易踩的几个坑

这一部分我觉得最值得写。

### 12.1 top-k 权重归一化没对齐

前面说过：

- full softmax 后取 top-k
- top-k 后再归一化

这两个不一样。

如果 reference 和优化版没统一，后面全都会乱。

### 12.2 dispatch 做对了，但没保留逆映射

如果我只顾着把 token 按 expert 排好了，却没保留：

- 原 token id
- 对应权重
- 可选 route 索引

那后面 combine 和 backward 都会变得非常麻烦。

### 12.3 过早上通用 sort

`expert_id` 是有限小整数，用 counting sort 几乎总是更合理。

### 12.4 expert 为 0 token 的边界没处理好

这个特别常见。

某些 expert 没收到任何 token 时：

- offset 逻辑很容易写错
- 后面 slice / GEMM / scatter 很容易越界或做无意义操作

### 12.5 combine 冲突被低估

尤其 `K=2` 时，很容易在最后的 `scatter-add` 上踩坑。

---

## 13. 如果让我总结 dispatch 这一段的真正难点

我现在会这样概括：

> MoE dispatch 的难点，不在于“会不会写 top-k”，而在于“能不能把 top-k 的结果组织成 GPU 喜欢的连续批处理数据”。

也就是说：

- top-k 本身只是开始
- 真正的难点是后面的 metadata、prefix sum、分桶、重排、还原

如果这一步做对了，后面很多优化才有意义。

如果这一步没做对，后面所有低层实现都会很痛苦。

---

## 14. 我现在最推荐的写法顺序

如果我今天要重新手写一遍 MoE dispatch，我会严格按下面顺序来：

1. 先有 `topk_idx / topk_weights`
2. 再展平成 `flat_*`
3. 做 `expert_counts`
4. 做 `expert_offsets`
5. 做 counting sort
6. gather 成连续 expert_input
7. expert FFN
8. scatter-add combine

这个顺序最大的好处是：

- 每一步都能单独验证
- 每一步都能和 reference 对比
- 出错时很容易定位

我现在越来越不愿意一开始就写“大而全的 fused kernel”，就是因为那样 debug 成本太高。

---

## 15. 这一篇最后，我想讲一句最实际的话

在我看来，MoE 算子里最容易写错、也最决定成败的一段，不是 expert FFN，而是：

> 路由结果如何被变成连续、可批处理、可还原的数据布局。

这一步如果做顺了，后面 CUDA / Triton / CuTe / PTX 才值得上。

这一步如果做不顺，后面所有低层优化都会变成在错误基础上堆复杂度。

所以如果你也在写 MoE，我最建议你优先花时间的部分不是：

- 怎么写更酷的 kernel

而是：

- 怎么把 `top-k -> bucket -> permute -> combine` 这条链路彻底收拾干净

---

如果这篇算第三篇，那第四篇我最想继续写的是：

**MoE 里的 expert FFN 部分，怎么从逐 expert 小 GEMM，优化成真正可用的 grouped / batched GPU 计算。**

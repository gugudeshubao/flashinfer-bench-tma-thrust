> 人性的偏见是你进步的最大阻碍，没有什么是不可尝试的。

# MoE 算子到底在做什么？我为什么觉得它比普通 FFN 难得多

如果把 Transformer 里的普通 FFN 看成“一家大工厂”，那 MoE（Mixture of Experts）更像是“一个调度中心 + 多家小工厂”。

每个 token 先经过一个 router，决定它应该去哪些 expert；然后算子把 token 按 expert 重新分组，每个 expert 各自做 FFN，最后再把结果按权重加回来。

从数学上看，MoE 很简单；但从算子实现上看，它比普通 FFN 难得多。

这篇文章我只想回答三个问题：

1. MoE 前向到底在算什么
2. 它在工程实现上到底难在哪
3. 一个可落地的 MoE 算子应该怎么写

## 一句话先讲本质

我现在对 MoE 算子的理解，可以浓缩成一句话：

> 先用 router 为每个 token 选少量 expert，再把 token 按 expert 分组重排，做分组 FFN，最后按路由关系加权还原。

所以在实现 MoE 的时候，我最关注的其实不是 FFN，而是三件事：

- 路由元数据生成
- token 重排 / 分桶
- 输出还原 / 加权累加

## 1. 数学形式并不复杂

设输入是：

- `x [T, H]`
- `T` 是 token 数
- `H` 是 hidden size

设有：

- `E` 个 expert
- 每个 token 只去 `K` 个 expert，通常是 top-1 或 top-2
- 每个 expert 都是一套独立 FFN

router 先产生 logits：

\[
\text{router\_logits} = x W_g,\quad [T, E]
\]

然后为每个 token 选出 top-k：

- `topk_idx[t, :]`
- `topk_weight[t, :]`

最后输出是：

\[
y_t = \sum_{j=1}^{K} g_{t,j}\cdot \text{Expert}_{e_{t,j}}(x_t)
\]

这就是 MoE 的前向。

## 2. 从算子视角看，前向到底分几步

我自己在实现时，通常把 MoE 前向拆成 5 步。

### Step 1: Router / Gating

输入 `x [T, H]`，得到：

- `logits [T, E]`

然后做 top-k，输出：

- `topk_ids [T, K]`
- `topk_weights [T, K]`

### Step 2: Dispatch / 重排

把 `[T, K]` 的路由结果展开成 `T*K` 条记录：

- `flat_token_ids`
- `flat_expert_ids`
- `flat_weights`

然后按 `expert_id` 分桶，让同一个 expert 的 token 变成连续区间。

### Step 3: Expert FFN

对每个 expert 单独做 FFN：

- 输入 `[N_e, H]`
- 输出 `[N_e, H]`

其中 `N_e` 是第 `e` 个 expert 收到的 token 数。

### Step 4: Combine / 还原

根据 dispatch 时保存的映射关系，把 expert 输出 scatter 回原 token 位置，并乘上对应 gate weight。

### Step 5: 得到最终输出

最终输出 shape 还是 `[T, H]`。

## 3. 为什么它比普通 FFN 难这么多

普通 FFN 非常规整：

- 一个大 GEMM
- 一个激活
- 一个大 GEMM

MoE 则会变成：

- router GEMM
- top-k
- histogram / prefix-sum / sort / bucket
- 多个变长 expert GEMM
- scatter / gather / reduce

也就是说，MoE 不是“一个大矩阵乘法问题”，而是“稀疏路由 + 不规则数据搬运 + 分组计算”问题。

这也是为什么我第一次写 MoE 的时候，明明数学写对了，性能却完全不对劲。

## 4. 最朴素的实现为什么几乎没法跑

最直观的版本是：

```python
for t in range(T):
    for j in range(K):
        e = topk_ids[t, j]
        w = topk_weights[t, j]
        out = expert_forward(e, x[t])
        y[t] += w * out
```

这个版本逻辑完全正确，但工程上几乎不可用：

- expert 调用太碎
- GPU 利用率极低
- kernel launch 太多
- 数据读写极度不连续

所以我在真正实现时，一定会先想办法把“发给同一个 expert 的 token”聚在一起算。

## 5. 工程上最标准的做法：按 expert 分桶

这是我最推荐的入门实现路线。

### 5.1 展平路由

把 `[T, K]` 展平成 `N = T*K` 条路由记录。

### 5.2 做 histogram 和 prefix-sum

统计每个 expert 收到多少 token：

- `expert_counts [E]`

再得到：

- `expert_offsets [E+1]`

于是 expert `e` 的 token 区间就变成：

- `[expert_offsets[e], expert_offsets[e+1])`

### 5.3 按 expert 重排

把同一个 expert 的 token 放到连续区域里。

### 5.4 Gather 成连续输入

根据 `sorted_token_ids` 从原始 `x [T, H]` gather，得到：

- `expert_input [N, H]`

### 5.5 做 expert FFN

对每个 expert 的连续切片做 FFN。

### 5.6 Scatter-add 回原 token

最后按 `(token_id, weight)` 把 expert 输出加回去。

这就是我理解里一个真实可用的 MoE 前向算子的逻辑真身。

## 6. 真正要关心的瓶颈在哪里

很多人会下意识觉得：

“MoE 的瓶颈一定是 expert FFN 的 GEMM。”

但我自己在实际实现和测试里，尤其是推理、小 batch、短序列场景下，往往先碰到的不是这个。

更常见的情况是：

- router 元数据生成先变成瓶颈
- token 重排 / gather / scatter 先变成瓶颈
- 小 expert batch 导致 GEMM 很碎
- Python 层调度和 kernel launch 反而吃掉很多时间

基于我这边做 MoE 实现和实测时一个非常强的经验是：

> MoE 的第一优先级不是“先把所有东西都 fused”，而是先把 dispatch 和 combine 做对、做顺，然后再逐步把热点段落下沉到 CUDA / CuTe / PTX。

## 7. 一个来自实际测试的提醒

如果我已经开始写高性能 MoE 算子，这里有一个非常现实的结论：

> 不是所有“更低层”的优化，都会在所有 workload 上更快。

我这边的实际测试里，出现过很多这种情况：

- 小 workload 上，低层 CUDA/CuTe 方案更快
- 中等 workload 上，优势缩小，甚至只剩持平
- 某些看起来很高级的缓存或融合，在部分规模上反而变慢

所以我现在更认同一个更成熟的优化流程：

1. 先做 reference，确保数值正确
2. 再做稳定基线
3. 用小 workload smoke test 看低层优化有没有收益
4. 最后再做全 workload sweep

这比“每改一点就直接跑全量 benchmark”更稳，也更接近真实工程实践。

## 8. 推理和训练，关注点是不一样的

### 推理更关注

- latency
- 小 batch / 变长输入
- dropless 路由
- launch overhead
- token 重排开销

### 训练更关注

- 吞吐
- capacity / padding
- all-to-all 通信
- backward 效率
- load balancing

如果我现在写的是单卡推理 MoE op，那优先级应该是：

- 先把本地路由和分桶做好
- 再考虑 grouped GEMM 和 fused kernel
- 最后再谈分布式通信

## 9. 三种实现层次

### 第一层：Reference 版

用纯 PyTorch 写一个慢但绝对正确的版本。

它的作用不是快，而是做数值标准。

### 第二层：按 expert 分桶版

把路由、重排、expert 计算、combine 拆开实现。

这是最适合第一次写 MoE op 的版本。

### 第三层：低层优化版

等逻辑稳定后，再逐步做：

- grouped GEMM
- batched GEMM
- fused routing metadata
- CUDA/C++ / Triton / CuTe kernel
- 减少中间张量
- 减少 launch overhead

如果我一开始就想“一步到位做全融合”，大概率会在 debug 上消耗远超预期的时间。

## 10. 一条真正能落地的实现路线

如果我是第一次写 MoE 算子，我会强烈建议自己按下面顺序来：

1. 写一个完全正确的 reference
2. 写按 expert 分桶的非 fused 版本
3. 把 per-expert GEMM 升级成 grouped / batched
4. 再去考虑 fuse 和低层 CUDA/Triton/CuTe 实现

这个顺序非常重要。

因为 MoE 真正难的是“不规则调度逻辑”，不是 GEMM 本身。

## 11. 如果我做的是 DeepSeek / FP8 block-scale 这一类 MoE

还要额外注意两件事：

- 量化 scale 的布局是否适合后续 GEMM 访问
- 不要为了“看起来更高级”的缓存或融合，破坏 correctness 或让中大 workload 退化

低层实现确实可能在小 workload 上带来明显收益，但不是所有低层改法都会带来全局收益。

所以我现在越来越认同一个现实的工作流：

- 先做 reference
- 再做稳定基线
- 再做 smoke test
- 最后做全 workload sweep

## 12. 最后的总结

MoE 算子实现最容易让人误判的一点是：

> 它看上去像“多个 FFN”，但真正决定性能的，常常是“token 如何被分桶和还原”。

所以如果我正在实现 MoE，不应该先执着于怎么把 FFN 算得更花哨，而应该先问自己：

- 路由元数据是不是最小化了
- token 重排是不是连续了
- expert 输入是不是成批了
- combine 是否避免了无意义的冲突
- 小 workload 的 launch overhead 有没有被控制住

把这些做好以后，低层优化才真正值得上。

---

如果这篇算是第一篇，那第二篇最自然的承接应该就是：

**MoE 算子怎么从 reference 版，一步步优化到真正可用的 GPU 实现。**

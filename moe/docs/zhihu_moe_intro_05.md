> 人性的偏见是你进步的最大阻碍，没有什么是不可尝试的。

# MoE 的最后一公里：combine / scatter-add 为什么经常比我想象中更难优化

前面几篇我已经分别讲了：

1. MoE 算子本质上在做什么  
2. 我怎么从 reference 版走到可用的 GPU 实现  
3. top-k、counting sort、token 重排为什么是核心  
4. expert FFN 为什么会变成“小 GEMM 地狱”  

这一篇我想讲 MoE 里最后一公里、也最容易被低估的一段：

> combine / scatter-add 到底为什么会这么麻烦？

很多人第一次写 MoE，会把注意力放在：

- router
- top-k
- expert FFN

但真到后面会发现：

> dispatch 和 expert FFN 都做完了，最后把结果“加回去”这一步，依然可能成为性能和正确性的双重麻烦。

这篇文章我就只讲这一段。

---

## 1. combine 这一步到底在做什么

假设我已经完成了前面的步骤：

- router 选出了 top-k experts
- token 已经按 expert 分桶
- 每个 expert 的 FFN 也算完了

那么现在我手上有的通常是：

- `expert_output [N, H]`
- `sorted_token_ids [N]`
- `sorted_weights [N]`

其中：

- `N = T * K`
- 每一条记录对应一条路由 `(token, expert)`

最终我想得到的是：

\[
y_t = \sum_j g_{t,j}\cdot o_{t,j}
\]

也就是：

- 同一个 token 可能有多个 expert 输出
- 我要把它们都加回 `y[t]`

这一步在代码上通常长这样：

```python
for pos in range(N):
    t = sorted_token_ids[pos]
    w = sorted_weights[pos]
    y[t] += w * expert_output[pos]
```

逻辑很简单。

但在 GPU 上，这一步很可能并不便宜。

---

## 2. 为什么 combine 比看起来难

从数学上看，combine 只是一个加权求和。

但从实现上看，它有三个天然的难点：

### 2.1 写回目标不连续

`expert_output[pos]` 是连续的，
但它要写回的 `y[token_id]` 并不一定连续。

也就是说：

- 读是连续的
- 写往往是不规则的

这会让内存访问变得很不友好。

### 2.2 同一个 token 可能被多次写

如果 `top-k > 1`，那一个 token 往往会从多个 expert 回来。

比如：

- token 17 去了 expert 3 和 expert 9

那么 combine 时就会有两次写回：

\[
y_{17} += g_{17,1} \cdot o_{17,1}
\]
\[
y_{17} += g_{17,2} \cdot o_{17,2}
\]

这在 GPU 上就意味着：

- 写冲突
- 原子加
- 或者额外的归约步骤

### 2.3 它经常发生在“小而碎”的场景里

尤其在小 batch / 小 seq 的推理场景下：

- 每个 expert 的 token 数可能不大
- expert FFN 已经不算很重了
- combine 的 scatter-add 反而显得突出

这也是为什么我后来越来越重视 combine，
而不是把它当成一个“最后随便加一下”的小步骤。

---

## 3. `K=1` 和 `K=2` 的 combine 根本不是同一个问题

这是我很想单独强调的一点。

### `K=1`

如果每个 token 只去一个 expert，那 combine 非常简单：

- 没有多 expert 累加
- 没有同 token 的写冲突
- 很多时候甚至可以直接 scatter 写回

### `K=2`

一旦 `K=2`，问题立刻变复杂：

- 同一个 token 会回写两次
- 这两次可能来自两个不同线程块
- 需要处理写冲突

所以我现在看 MoE combine，会先问：

> 当前优化面对的是 top-1 还是 top-2？

这两种情况在实现复杂度上差得非常大。

---

## 4. 最直接的做法：atomic add

最容易想到的方法就是：

```cpp
atomicAdd(&y[token_id, h], weight * expert_output[pos, h]);
```

这个做法的优点很明显：

- 逻辑简单
- 容易写对
- 很适合作为第一版可用实现

但问题也一样明显：

- 写冲突严重时会拖慢
- 如果很多 route 指向同一个 token，原子加会堆在一起
- 对大 hidden size 来说，atomic 写的压力也不小

所以我会把 atomic combine 看成：

> 一个很适合 reference 或第一版 GPU 实现的办法，但通常不是最后的高性能答案。

---

## 5. 第二种思路：先按 token 再分桶

如果我不想在 combine 时大量 atomic add，一个自然的思路就是：

> 先按 token 把结果重新分组，再做归约。

也就是说：

- 前面 dispatch 是按 expert 分组
- combine 前，我再把 expert_output 根据 token 分组

这样就能得到：

- 同一 token 对应的多个 route 是连续的

然后可以直接做：

- segment reduce
- block 内归约
- 最后一次性写回

这个思路的优点是：

- 减少写冲突
- 比 atomic add 更规整

缺点是：

- 你又引入了一次排序 / 分桶 / 重排
- 额外开销不一定划算

所以我通常会这样判断：

- 如果 `K` 很小、冲突也不严重，atomic add 可能更划算
- 如果 `K` 更大，或者 token 冲突非常明显，再考虑 token-side reduce

---

## 6. 第三种思路：把 combine 融到前一步

还有一种更“算子设计味”的思路：

> 不把 `expert_output [N, H]` 完整写出来，而是在 expert 计算后直接按 token 累加回去。

这相当于把：

- second GEMM
- weight multiply
- combine

尽量融合起来。

这条路理论上很诱人，因为它可以减少：

- 中间张量
- 全量写回
- 单独的 combine kernel

但我自己对这条路一直比较谨慎。

原因是：

- debug 会更复杂
- kernel 结构会更臃肿
- 一旦 combine 逻辑、weight 逻辑、route 索引有任何一个错，排查成本很高

所以我通常只会在：

- reference 足够稳定
- dispatch 足够稳定
- expert FFN 足够稳定

之后，才会认真考虑这种融合。

---

## 7. 为什么我觉得 combine 特别容易被低估

因为它看起来“太简单了”。

大家容易觉得：

- 路由复杂
- GEMM 很大
- combine 不就是一行加法吗

但在 GPU 上，事情恰恰反过来。

很多时候：

- 复杂的 GEMM 反而很规整
- 简单的 scatter-add 反而很不规整

这就是 GPU 编程里很典型的一种反直觉：

> 数学上简单，不代表工程上便宜。

而 combine 正是这种问题的典型代表。

---

## 8. 我在实际实现里会怎么选 combine 路线

如果我是第一次把 MoE 跑通，我会这样选：

### 第一阶段

先用：

- 直接 scatter-add
- 必要时 atomic add

目标只是：

- 正确
- 好验证

### 第二阶段

如果 profiling 发现 combine 真成了热点，再考虑：

- token-side reduce
- segment reduce
- 按 token 分桶

### 第三阶段

只有在我已经确认：

- combine 的确是主瓶颈
- 额外排序/重排是值得的

时，我才会去做更激进的融合。

这套节奏不是因为我不想快，而是因为我知道：

> combine 这一步一旦写复杂，debug 会非常痛苦。

---

## 9. combine 和 dispatch 是镜像关系

这是我后来越来越强烈的一个体会。

dispatch 做的是：

- `token -> expert`

combine 做的是：

- `expert -> token`

从工程上看，它们其实是一对镜像问题。

dispatch 解决的是：

- 输入如何变成连续的 expert batch

combine 解决的是：

- expert 输出如何重新还原回 token 空间

如果 dispatch 很乱，expert 计算前就乱。
如果 combine 很乱，expert 计算后又乱。

所以我现在会把这两件事放在一起看：

> 一个 MoE 实现到底成熟不成熟，不只是看 FFN 算得快不快，而是看 dispatch 和 combine 这两头是不是都被收拾干净了。

---

## 10. 我踩过的 combine 坑

这一部分我也想讲得更实在一点。

### 10.1 以为 `K=2` 只是多加一次

不是的。

`K=2` 的问题不是多算一次，而是：

- 多了一层写冲突
- 多了一层累加语义
- 多了一层还原逻辑

### 10.2 以为 atomic add 一定很差

也不一定。

在某些小场景下，atomic add 反而是最划算的。

因为你为了避免 atomic add 引入的额外排序 / reduce，本身也会花很多钱。

### 10.3 以为 combine 只是“顺手写一下”

如果我这样想，后面一般都会被它教育。

因为 combine 很容易变成：

- 小 workload 下的明显固定开销
- 多 expert 路由下的冲突热点
- 数值问题和索引问题的集中爆发点

---

## 11. 我现在对 combine 的判断标准

如果我要决定 combine 是否值得继续优化，我现在只看三件事：

### 11.1 它是不是已经成了明显热点

先证明它是瓶颈，再去优化它。

### 11.2 它的冲突是否严重

如果 `K=1` 或冲突很轻，atomic add 可能已经够用了。

### 11.3 为了优化 combine，我要不要再引入一次全局重排

如果要，那我会非常谨慎。

因为很多时候，额外一次重排带来的成本，比 atomic add 还贵。

---

## 12. 如果我总结这一篇，只想留一句话

我现在会这样总结 combine 这一步：

> MoE 的最后一公里，往往不是“把结果加回去”这么简单，而是“如何在不规则写回、写冲突和额外重排之间找到真正划算的平衡”。

这就是 combine 难的地方。

不是它数学复杂，而是它工程上很容易又碎、又乱、又冲突。

---

## 13. 这篇文章的最后，我最想引出的问题

到这里为止，MoE 前向的三块核心已经都讲过了：

1. route / top-k / dispatch  
2. expert FFN / grouped GEMM  
3. combine / scatter-add

如果再往后写，我最想继续写的就是：

**MoE 算子到底该怎么做 benchmark，为什么很多优化在小 workload 上成立，但完整 sweep 里却会退化。**

因为我自己做这件事时，越来越觉得：

> 真正决定一个 MoE 实现值不值得保留的，不是某一个点快不快，而是它在完整 workload 分布上的表现是不是稳定。**

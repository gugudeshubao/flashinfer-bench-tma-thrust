> 人性的偏见是你进步的最大阻碍，没有什么是不可尝试的。

# MoE 的 expert FFN 怎么从“小 GEMM 地狱”，走到真正可用的 grouped / batched 计算

前面几篇我已经把 MoE 的整体结构、dispatch、counting sort、token 重排这些部分讲清楚了。

这一篇我只讲一件事：

> 当 token 已经按 expert 分好桶之后，expert FFN 这部分到底该怎么做，才不会掉进“小 GEMM 地狱”。

因为很多人第一次写 MoE，会觉得最难的是路由；等路由写完之后，又会发现另一个现实：

> expert FFN 看上去只是普通 GEMM，但一旦每个 expert 收到的 token 数不一样，它就会变成一堆很碎的小矩阵乘法。

这就是 MoE 里第二个很硬的工程问题。

这一篇我想讲清楚：

1. 为什么 expert FFN 很容易碎掉  
2. 我为什么不建议一开始就做“全局大融合”  
3. 从逐 expert 小 GEMM，到 grouped / batched 的实现路线  
4. 什么情况下 grouped 更合适，什么情况下 batched 更合适  
5. 我自己做这件事时，最看重的判断标准是什么  

---

## 1. dispatch 做完以后，expert FFN 的问题才刚开始

假设前一阶段我已经做完：

- `top-k`
- counting sort
- token 重排

这时候我终于得到了：

- expert `e` 的 token 区间是连续的
- 它的输入是 `expert_input[start:end]`

设：

- `N_e = end - start`

那么 expert `e` 要做的事情就是：

\[
h_e = X_e W_1[e]
\]
\[
o_e = \phi(h_e) W_2[e]
\]

看起来很普通。

问题是：

- 每个 expert 的 `N_e` 不一样
- 有的 expert 收到很多 token
- 有的 expert 只收到几个 token
- 有的 expert 甚至一个 token 都没有

所以这一层从 dense FFN 里的：

- 一个大 GEMM

变成了：

- `E` 个大小不同的 GEMM

这就是我说的“小 GEMM 地狱”。

---

## 2. 为什么小 GEMM 会这么难受

在 GPU 上，大 GEMM 通常很好做：

- tile 很规整
- Tensor Core 容易吃满
- launch overhead 可以摊薄
- 内存访问比较连续

但小 GEMM 的问题是：

- 每次算得少
- launch overhead 占比高
- Tensor Core 利用率低
- 甚至有时候读写比计算还贵

而 MoE 里最麻烦的是：

> 这些小 GEMM 不是固定 shape，而是一组变长、动态、每个 batch 都可能变化的小 GEMM。

这就比普通“小矩阵乘法”还更棘手。

---

## 3. 我最开始会怎么写 expert FFN

如果只是为了先把东西跑通，我通常会写成：

```python
for e in range(E):
    start = expert_offsets[e]
    end = expert_offsets[e + 1]
    if start == end:
        continue

    inp = expert_input[start:end]      # [Ne, H]
    h = inp @ W1[e]                    # [Ne, I]
    h = act(h)
    out = h @ W2[e]                    # [Ne, H]
    expert_output[start:end] = out
```

这个版本的优点是：

- 逻辑最清楚
- 最容易和 reference 对齐
- 出错时最好 debug

它的缺点也非常明显：

- 一个 expert 一个 GEMM
- launch 次数很多
- 小 `N_e` 时效率极低

所以这通常只能是“第一版可用实现”，而不会是最终高性能实现。

---

## 4. 我为什么不建议一开始就做全局大融合

这是我很想强调的一点。

很多人看到小 GEMM 碎，就会立刻想到：

- 把两个 GEMM 融合
- 把激活融合
- 把 combine 融合
- 甚至把整个 expert FFN + combine 一次做完

听起来很对，但我通常不会这么起步。

原因有两个：

### 4.1 太难 debug

如果 dispatch、expert 计算、combine 全堆在一个 kernel 里：

- 数值一旦不对，定位会非常困难
- 很难知道问题出在路由、索引、激活还是累加

### 4.2 你未必真的知道瓶颈在哪

我在实际优化里很强的一个经验是：

> 很多时候我以为是 expert GEMM 太碎，结果真正先炸的是 dispatch 或 combine。

所以如果一开始就做大融合，经常会出现：

- 复杂度上去了
- 性能却没有真正提升

这就是为什么我更喜欢先走下面这条路线：

1. 逐 expert 小 GEMM版  
2. grouped GEMM / batched GEMM  
3. 再考虑融合

---

## 5. 第一种真正值得做的升级：Grouped GEMM

如果我已经确认：

- dispatch 没问题
- expert 输入已经连续
- 性能瓶颈确实在 expert 计算

那我通常最先考虑的是 grouped GEMM。

### 5.1 它的核心思想

不是每个 expert 单独 launch 一次 GEMM，而是：

> 把多个 expert 的 GEMM 描述打包成一个 grouped call，一次性交给底层实现。

也就是：

- expert 0: `X_0 [N0, H] @ W1[0] [H, I]`
- expert 1: `X_1 [N1, H] @ W1[1] [H, I]`
- ...

我不想自己写：

```python
for e in experts:
    gemm(...)
```

而是希望交给 grouped GEMM 一次做。

### 5.2 它的优点

- 减少 launch overhead
- 更容易让多个小 GEMM 合并成一次大任务
- 比逐 expert loop 更接近高性能实现

### 5.3 它的缺点

- 仍然是变长问题
- 实现和调试难度比逐 expert 版高
- 不同后端对 grouped GEMM 支持不一样

所以我会把 grouped GEMM 视为：

> 从“逻辑正确”走向“性能可用”的第一步。

---

## 6. 第二种常见路线：Batched GEMM + padding

另一种很常见的方法，是把每个 expert 的输入 pad 到统一容量：

- `capacity = C`

然后把 expert 输入组织成：

- `[E, C, H]`

这样两个 GEMM 就都变成规则的 batched GEMM：

\[
[E, C, H] \times [E, H, I] \rightarrow [E, C, I]
\]

\[
[E, C, I] \times [E, I, H] \rightarrow [E, C, H]
\]

### 6.1 为什么这个思路很吸引人

因为它特别规整。

只要 shape 规则了，后面很多优化都会变得容易：

- batched GEMM
- Tensor Core
- 融合激活
- 规则 shared memory tile

### 6.2 它的问题也很现实

padding 不是免费的。

如果很多 expert 实际只收到很少 token，但我又把它们 pad 到统一容量，那就会浪费大量算力和带宽。

所以我通常会这样理解：

- grouped GEMM 更适合 dropless / 动态推理
- batched GEMM + padding 更适合训练 / capacity-based 实现

不是哪一个绝对更强，而是看场景。

---

## 7. grouped 和 batched，我会怎么选

如果是推理，我更倾向先做：

- dropless
- counting sort
- grouped GEMM

因为推理里：

- token 数动态
- batch 往往不大
- 我更关心 latency

这时为了追求规则 shape 而强行 pad，未必划算。

如果是训练，我会更认真考虑：

- capacity
- padding
- batched GEMM

因为训练里：

- batch 更大
- 吞吐更重要
- 规则 shape 的收益更大

所以我现在不会简单问：

> grouped GEMM 和 batched GEMM 谁更强？

我会问：

> 我当前是在优化推理还是训练，我当前的 token 分布到底长什么样？

---

## 8. 为什么激活往往是一个被低估的点

在 expert FFN 里，两层 GEMM 中间通常会有一个激活：

- GELU
- SiLU
- SwiGLU

很多人会觉得激活相比 GEMM 不重要。

但在 MoE 里不一定。

因为当 expert batch 很碎的时候：

- GEMM 本身不一定大
- 激活这一步也会变成碎片化的小 kernel
- launch overhead 可能同样明显

这也是为什么我在很多低层尝试里，会先把：

- `SwiGLU`

单独拿出来做 CUDA / PTX / CuTe 版本。

不是因为它数学复杂，而是因为：

> 在小 batch、不规则 expert 输入场景下，任何一段碎片化的小计算都可能值得单独优化。

---

## 9. 我怎么判断 expert 计算该不该继续往低层推

不是每一个版本都值得下沉到 CUDA/CuTe。

我通常会先看三个信号：

### 9.1 小 smoke test 里有没有收益

如果小 workload 上一点优势都没有，那后面继续做很可能就是浪费时间。

### 9.2 中段 workload 会不会退化

很多版本小输入快，大一点马上掉回去。

这种时候我通常不会直接上主线。

### 9.3 完整 sweep 的平均值有没有意义

如果完整 benchmark 里平均没有明显优势，那我不会因为个别点漂亮就保留它。

这套标准我自己踩坑之后才越来越坚定。

---

## 10. 我踩过的典型坑

这一部分我觉得最真实。

### 10.1 以为“把 FFN 下沉到 CUDA”就一定会赢

不一定。

有些低层实现只是把 Python 层问题换成了：

- 编译开销
- 调试复杂度
- shape 适配问题
- kernel 过于碎片化

### 10.2 过早做缓存

我也试过给 expert 权重、中间布局、甚至路由元数据做缓存。

结论是：

- 有些缓存很值
- 但很多缓存会带来新的复杂度和边界问题

尤其在 MoE 这种动态输入里，缓存键如果没设计好，非常容易踩坑。

### 10.3 忘记区分“值得优化的段落”和“只是看起来复杂的段落”

不是所有复杂代码都值得优化。

有些地方看起来复杂，但不一定是热点；
有些地方代码很短，却是最关键的性能瓶颈。

这个只能靠：

- smoke test
- 完整 sweep
- 反复验证

慢慢建立直觉。

---

## 11. 如果让我给 expert FFN 部分一个实现建议

我现在会给出这样一个非常务实的建议：

### 第一步：逐 expert 小 GEMM版

先把：

- dispatch
- expert_input
- expert_output
- combine

这整条链路做对。

### 第二步：Grouped GEMM

当我确认 expert FFN 真的是瓶颈，再把逐 expert loop 升级成 grouped GEMM。

### 第三步：必要时再考虑 batched / fused

如果我的场景真的适合规则 shape，再考虑 padding 和 batched GEMM。

如果激活或 combine 真的已经成了热点，再把它们下沉到 CUDA/CuTe/PTX。

这个顺序对我来说非常重要。

---

## 12. 我现在怎么总结 expert FFN 这一层

如果让我用一句话总结这一层的本质，我会说：

> expert FFN 的难点，不在于“它是不是两个 GEMM”，而在于“它是不是一堆大小不一、极容易碎掉的小 GEMM”。

所以真正的优化目标不是：

- 把一个 FFN 写得更极致

而是：

- 把很多碎小的 FFN 组织成 GPU 能高效处理的形态

这就是为什么 grouped / batched 这些思路在 MoE 里这么重要。

---

## 13. 这篇文章的结尾，我想留一个更实际的问题

当我把 dispatch 和 expert FFN 都讲完以后，下一个自然的问题就是：

> combine 这一步到底该怎么写，尤其在 `top-k > 1` 的时候，怎么避免 scatter-add 变成瓶颈？

因为：

- dispatch 解决了输入怎么组织
- expert FFN 解决了中间怎么计算
- combine 解决的是输出怎么还原

这三件事里，combine 往往最容易被低估。

---

如果这篇算第四篇，那第五篇我最想继续写的是：

**MoE 的最后一公里：combine / scatter-add 为什么经常比你想象中更难优化。**

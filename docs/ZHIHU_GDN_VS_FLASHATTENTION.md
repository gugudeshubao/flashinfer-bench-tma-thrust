# 为什么 FlashAttention 越做越快，而 Gated DeltaNet 更像一道系统题？

如果只看论文里的复杂度，很多人第一次接触 Gated DeltaNet（GDN）这类结构时，都会有一个直觉：

既然它不像 softmax attention 那样显式构造完整 attention matrix，复杂度也更线性，那它是不是天然就比 attention 更适合长序列，也更容易在 GPU 上做快？

但真正做过 kernel 之后，你很快就会发现，事情并不是这样。

**FlashAttention 的优化路径，和 GDN 的优化路径，几乎是两种完全不同的世界。**

FlashAttention 越做越快，很大程度上是因为它一直在把问题往 GPU 最擅长的方向推：更厚的矩阵块、更高的 Tensor Core 利用率、更少的中间访存、更顺的流水。

而 GDN / Delta-rule 这一类结构，看起来复杂度更漂亮，但真到硬件上，核心矛盾却不是“算得不够快”，而是：

- 时间维存在状态递推依赖
- 每一步都要读写 state
- 低精度下还很容易不稳定

所以如果要用一句话概括两者的差别，那就是：

**FlashAttention 的核心问题是“怎么把矩阵乘做得更好”，而 GDN 的核心问题是“怎么把状态系统做得不慢、不炸、还不浪费带宽”。**

这篇文章就专门讲这件事。

## 一、FlashAttention 和 GDN，表面都在做序列建模，底层却不是一类问题

FlashAttention 本质上还是 attention。

也就是说，它的底层计算主轴依然是：

- `QK^T`
- `softmax`
- `PV`

它做的不是改掉 attention，而是**重排 attention 的执行顺序和中间存储方式**，让原本需要落到 HBM 的大中间矩阵，尽量在片上完成。

换句话说，FlashAttention 的本质是：

**在不改变 attention 数学定义的前提下，把它实现成一个 IO-aware 的分块矩阵算法。**

所以它虽然优化难，但优化方向很统一：

- 分块
- 融合
- 流水
- Tensor Core
- 片上归约

而 GDN 不一样。

GDN / Delta-rule 这类结构，本质上不是先算一个 token-token 的交互矩阵，再做归一化聚合；它是**沿着时间维递推一个状态 `state`**，每个 token 都要基于当前输入和历史 state 做一次更新。

你可以把它理解成：

- FlashAttention 的核心对象是一个“块化矩阵问题”
- GDN 的核心对象是一个“递推状态问题”

这两个问题对 GPU 的友好程度，天然就不一样。

## 二、为什么 FlashAttention 更像 GPU 的“顺风局”

FlashAttention 的成功，不只是因为它减少了 HBM 读写，更重要的是：**它做的事情高度符合 GPU 的执行偏好。**

GPU 最喜欢什么？

- 大块并行
- 规则张量
- 高算术强度
- 能持续喂给 Tensor Core 的矩阵乘

而 FlashAttention 虽然也面临 softmax 的数值稳定和片上存储限制，但它的核心计算依旧能被组织成高度规则的 tile：

- `Q`、`K`、`V` 分块
- tile 内做矩阵乘
- 在线维护 row max / row sum
- 最终把输出写回

也就是说，FlashAttention 的很多优化，本质上都在增强一件事：

**让算子更像 GEMM。**

这是一条非常强的主线，因为现代 GPU 的硬件和软件栈，本来就是围绕 GEMM 优化出来的。

所以你会发现，FlashAttention 的演进路线非常统一：

- 更好的 tiling
- 更深的 pipeline
- 更高的 Tensor Core 吞吐
- 更少的非必要访存
- 更强的寄存器 / shared memory 复用

它难，但它的难是“往 GPU 甜点区再推进一点”的难。

## 三、为什么 GDN 不像 FlashAttention 那样容易一路吃到硬件红利

GDN 最大的问题在于：**它并不是一个天然能被重写成厚重矩阵乘的问题。**

它每个 token 都要做的事情，大致是：

1. 读取当前 token 的 `q, k, v`
2. 根据 gate 参数算衰减和更新比例
3. 读取旧 state
4. 更新 state
5. 再由新 state 生成输出

看起来只是多了个状态，但这件事会连锁带来三个系统后果。

### 1. 时间维并行性弱

FlashAttention 在 prefill 时可以把一整段序列切成 tile，然后并行处理大量 token-token 交互。

但 GDN 的 decode 本质上是：

- 当前 token 依赖上一个 token 的 state
- 下一个 token 又依赖当前 token 刚写回去的 state

于是时间维天然带依赖，没法像 attention 那样直接铺开。

这意味着 GDN 的 decode 会天然更敏感于：

- 单 token latency
- kernel launch overhead
- 小 batch 下 occupancy 不足

### 2. 中心资源从“算力”变成“状态流量”

FlashAttention 的核心矛盾通常是：如何减少大中间矩阵的显存往返，并让 tile 内计算足够厚。

而 GDN 的核心矛盾更像是：**每一步都要去碰一个不小的 state。**

以这次比赛的 workload 为例，state 典型形状类似：

```python
state = [B, 8, 128, 128]
```

decode 每来一个 token，就要对这个 state 做读、算、写。问题在于，这一步的计算量并没有厚到足以压过访存，于是实际瓶颈很容易变成 memory bandwidth。

也就是说：

- FlashAttention 的优化常常是在提升 compute efficiency
- GDN 的优化很多时候是在提升 state traffic efficiency

这是两套完全不同的思维方式。

### 3. GDN 对低精度的敏感程度更高

FlashAttention 当然也关心数值稳定，尤其是 softmax 的缩放、max-trick、混合精度累积等问题。

但 GDN 的数值稳定性更“系统性”。

因为它不是一次性算完，而是在递推 state。只要 state update 的尺度失控，就会出现：

- state 爆炸
- 误差逐步累积
- 长序列下越来越偏
- 低精度压缩后误差被反复放大

这意味着 GDN 不是简单地“把算子换成 FP8 就完事”，而是要先回答：

- 这个状态系统在低精度下还能稳定吗？
- scale 怎么选？
- 哪些量能压，哪些量必须高精度累积？

所以 GDN 的低精度优化，不像 FlashAttention 那样主要是在追求更高 Tensor Core 吞吐，而更像是在平衡“少搬字节”和“别把 state 搞炸”。

## 四、如果从 decode / prefill 分开看，两者差异会更明显

### 1. Decode：GDN 和 FlashAttention 几乎不是一个优化题

FlashAttention 的 decode，虽然也会面临 KV cache 读写、batch 小、memory-bound 等问题，但 attention 至少仍然保持着“查询当前 token 与历史 KV 交互”的结构。很多优化仍然可以围绕：

- KV cache layout
- paged attention
- 分块加载
- 更高效的 score / value 聚合

来做。

而 GDN 的 decode 是另一回事。它不是在长历史上做一次聚合，而是在维护一个持续演化的 state。对于类似 `gdn_decode_qk4_v8_d128_k_last` 这种 workload，核心操作更接近：

```text
[128 x 128] × [128]
```

也就是 mat-vec，而不是 mat-mat。

这意味着：

- 很难有效使用 Tensor Core
- 算术强度偏低
- 状态读写成为主导
- 单步 fusion 比“追求大算力峰值”更重要

所以在 decode 阶段，FlashAttention 和 GDN 最大的区别可以概括成：

**FlashAttention 更像“怎么高效读 KV 并完成在线聚合”，GDN 更像“怎么高效更新一个大状态并立刻用它出结果”。**

前者还是 attention 工程，后者已经更接近递推系统工程。

### 2. Prefill：FlashAttention 继续吃 GEMM 红利，GDN 需要先把自己改造成块并行

FlashAttention 的 prefill 基本就是它最擅长的舞台。

因为 prefill 本来就是整段 prompt 一次处理，天然适合：

- 大 tile
- 高并发
- Tensor Core
- 片上 softmax 归约

也就是说，FlashAttention 的 prefill 是“原问题就很适合 GPU，只是原始实现没写好”。

但 GDN 的 prefill 不是这样。

如果你直接按 token 顺序递推，那它依然会：

- 并行度不足
- state 重复读写过多
- 算术强度偏低

所以 GDN 的 prefill 要先做一件 FlashAttention 不需要做的事：

**先把递推问题改写成 chunkwise recurrence 或 scan 风格问题。**

只有这样，chunk 内部的一部分计算才有机会矩阵化，才有机会进入 Tensor Core 的甜点区。

所以：

- FlashAttention 的 prefill 是“天然矩阵化，再把实现做对”
- GDN 的 prefill 是“先设法改造成块并行，再谈怎么把块内做快”

这就是为什么两者虽然都在处理长序列，但工程难点根本不是一个方向。

## 五、为什么 FlashAttention 的优化成果更容易被复用，而 GDN 的优化更依赖具体 workload

FlashAttention 这些年一个很大的优势，是它的优化成果有很强的普适性。

原因在于 attention 的基本形式相对稳定：

- 还是 `QK^T`
- 还是 softmax
- 还是 `PV`

输入 shape 变化当然会影响调优，但整体优化范式不会变。

而 GDN 的实现更依赖具体设计细节：

- state 的形状是多大
- gate 的公式是什么
- head 之间怎么映射
- prefill 是否支持 chunk 化
- state 是按什么精度存
- correctness 要求对长期递推有多严格

这会导致一个现象：

**FlashAttention 的很多优化像“通用基础设施”，而 GDN 的很多优化更像“强绑定具体状态方程的定制工程”。**

这也解释了为什么 attention 生态里更容易形成稳定、成熟、广泛复用的高性能内核，而 GDN 这类结构往往需要更深的模型-系统协同设计。

## 六、如果从 Roofline 的角度看，两者的优化目标也不一样

FlashAttention 的一个典型目标，是通过更好的 tiling 和片上复用，把算术强度不断拉高，让 kernel 尽量靠近算力上限。

也就是说，它经常是在做这样的事：

- 降低 HBM traffic
- 提高 tile 内复用
- 把 bottleneck 从 IO 推向 compute

而 GDN 尤其是 decode，经常很难把自己推到 compute-bound。

对于这次 workload，decode 的算术强度大约只有：

```text
AI ≈ 1 FLOP / byte
```

这意味着它离 compute roof 很远，更像是一个先天 memory-bound 的问题。

所以两者的 roofline 目标也不同：

- FlashAttention：尽量向 compute roof 靠
- GDN decode：尽量把 memory roof 吃满
- GDN prefill：先靠 chunking 把自己从低 AI 拉起来，再谈算力利用率

这就是为什么 FlashAttention 的性能讨论经常围绕 TFLOPS，而 GDN decode 的性能讨论更自然地围绕 GB/s。

## 七、两者真正的分水岭：一个在优化“算子表达”，一个在优化“状态机制”

我觉得 GDN 和 FlashAttention 最本质的区别，不是一个叫 attention，一个叫 linear attention，而是：

**FlashAttention 优化的对象，主要还是一个算子表达问题；GDN 优化的对象，已经是一个状态机制问题。**

FlashAttention 做得再深，核心依然是在想：

- 怎么让 attention 这个算子表达更贴近 GPU

而 GDN 必须同时想：

- 状态方程如何稳定
- 状态存储如何压缩
- 状态递推如何并行
- 状态更新如何减少读写

这就是为什么 GDN 的优化往往必须同时牵涉：

- 算法改写
- 内存布局
- kernel fusion
- 量化策略
- 数值稳定性设计

它比 FlashAttention 更像一个“模型结构与系统实现共同决定性能上限”的问题。

## 八、如果要给工程团队一个最简洁的判断标准

如果你的目标是给团队一个很短的判断标准，我会这样说：

### FlashAttention 的核心问题

- 本质仍是 attention
- 核心目标是减少 IO、提高 tile 复用、吃满 Tensor Core
- 优化主线非常统一：分块、融合、流水、矩阵化

### GDN 的核心问题

- 本质是状态递推
- 核心目标是降低 state traffic、缓解时间依赖、守住数值稳定
- 优化主线不是单一算力问题，而是并行性、带宽、稳定性三者平衡

这也是为什么我会说：

**FlashAttention 更像“把硬件甜点区吃到极致”，GDN 更像“在系统约束下艰难地把理论优势兑现出来”。**

## 九、结论：GDN 不是 FlashAttention 的简单替身，它难在另一个维度

最后回到最核心的问题。

为什么 FlashAttention 越做越快，而 GDN 往往让人觉得“理论很好，工程上却很难轻松赢”？

因为两者压根不是同一种优化对象。

FlashAttention 的优势在于：它虽然复杂，但始终是在 GPU 最喜欢的矩阵块世界里做文章，所以硬件、编译器、库生态、工程经验都在帮它。

而 GDN 的难点在于：它把问题从“矩阵交互”换成了“状态递推”，于是你必须同时处理：

- 时间维依赖
- 大状态读写
- 长程数值稳定

所以一句话总结就是：

**FlashAttention 难在把 attention 做到足够极致，GDN 难在它根本不是一个天然适合 GPU 的 attention 问题。**

这也解释了为什么在很多场景里，GDN 的论文优势不会自动变成工程优势。它当然可能在长序列上更有潜力，但前提是你真的把状态系统这件事做对了。

而这，恰恰比“再写一个更快的 attention kernel”更难。

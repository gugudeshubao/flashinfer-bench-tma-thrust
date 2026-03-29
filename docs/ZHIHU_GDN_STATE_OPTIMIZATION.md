# Gated DeltaNet 真正难优化的，不是 FLOPs，而是状态：从模型推理问题到算子落地方案

这次做 Gated DeltaNet（GDN）相关比赛，我最大的感受不是“这个算子算得多复杂”，而是：**它的难点根本不只是计算量，而是状态。**

如果只从论文公式出发，GDN / Delta-rule 这类结构看起来很美：它不像标准 softmax attention 那样显式构造完整 attention matrix，理论上复杂度更线性，也更适合长序列。但一旦真的把它放到 GPU 上跑，问题马上就从“公式优不优雅”变成“系统能不能扛住”。

尤其结合这次比赛给出的 workload：

- `gdn_decode_qk4_v8_d128_k_last`
- `gdn_prefill_qk4_v8_d128_k_last`

以及实际 benchmark 里暴露出来的 `State Size`、`BW (GB/s)`、小 batch 与大 batch 的性能差异，可以看到一件非常明确的事：

**GDN 算子优化的核心瓶颈，不是单纯的 FLOPs，而是“状态递推难并行、state 读写吃带宽、低精度又容易不稳定”这三件事同时成立。**

这篇文章想做的事情很简单：先把这个问题讲清楚，再把它一步一步过渡到真正能落地的算子优化方案。

## 一、先从模型推理说起：为什么 attention 好优化，而 GDN 难很多

做大模型推理时，大家通常会把计算分成两个阶段：

- `prefill`：处理整段输入 prompt
- `decode`：自回归地一个 token 一个 token 往后生成

标准 attention 在这两个阶段虽然瓶颈不同，但整体有一个非常大的工程优势：**它很容易被改写成大块矩阵乘。**

而 GPU 最擅长的，恰恰就是大矩阵乘。

也就是说，attention 的很多高性能实现，本质上都是在把问题不断往 GEMM 靠：

- 更大的 tile
- 更少的中间访存
- 更好的 Tensor Core 利用率
- 更强的流水与并行

但 GDN 不一样。

GDN / Delta-rule 这类结构，本质上不是“先算一个分数矩阵，再做归一化聚合”，而是**按 token 维护并更新一个状态 `state`**。每来一个 token，大致都要做下面这些事情：

1. 读入当前 token 的 `q, k, v`
2. 结合门控量，例如 `a / b / dt_bias / A_log`
3. 更新隐状态 `state`
4. 再由 `state` 生成当前输出

这意味着它在推理阶段面临的不是单一问题，而是三类问题同时存在：

- 时间维有递推依赖
- 每一步都要动一个不小的状态张量
- 状态更新本身还很怕数值不稳定

从这里开始，GDN 的优化方向就已经和 attention 分叉了。

## 二、第一层瓶颈：状态递推天然带来串行依赖

GDN 最根本的结构特点，是它的计算更像 RNN 或某些 SSM，而不是标准 attention。

原因很简单：**当前 token 的结果依赖上一步留下来的 `state`。**

于是第一个瓶颈就出来了：

**decode 阶段的时间维并行性很弱。**

在标准 attention 里，prefill 往往能很自然地把长序列变成矩阵计算；但在 GDN 里，decode 时每一步都必须先拿到上一步的 state，才能做下一步更新。这会直接带来几个后果：

- 单 token latency 变得非常关键
- kernel launch overhead 会被明显放大
- 小 batch 场景下很难把 GPU 吃满
- 很多优化不能靠“把矩阵做大”来解决

这也是为什么 GDN 的 decode 优化，往往不像 FlashAttention 那样主要围绕“块内并行 + Tensor Core 吃满”展开，而更像是在解决“如何把一次很细的状态更新做得足够快”。

换句话说，**GDN 的第一瓶颈不是算子太复杂，而是算子太递推。**

## 三、第二层瓶颈：真正最重的往往不是算力，而是 state 的读写

很多人分析这类 kernel 时，第一反应还是数 FLOPs。但从这次 workload 的数据看，GDN 更值得先看的其实是状态流量。

在比赛 benchmark 里，decode 单 token 会直接给出：

- `State Size`
- `BW (GB/s)`

这两个指标已经非常说明问题了。因为它们都在暗示同一件事：**这个算子很可能是 memory-bound，而不是 compute-bound。**

以这次 workload 为例，状态张量是按 head 维护的，典型形状类似：

```python
state = [B, 8, 128, 128]
```

如果用 `float32` 保存，每个 batch item 的 state 就已经不小了。decode 每生成一个 token，基本都要：

1. 从 HBM 或 cache 把 state 读出来
2. 做一次衰减、混合、外积更新
3. 再把新 state 写回去

问题在于，这一步的计算量并没有大到足以掩盖访存成本。

对于 decode，一个 batch element 的计算量大约是：

- 每个 head 约 `131K FLOPs`
- 8 个 head 合计约 `1.05M FLOPs`

而对应的状态读写流量大约也是：

- 每个 batch element 约 `1.05 MB`

于是算术强度大致只有：

```text
AI ≈ 1 FLOP / byte
```

而 B200 上 FP32 的 ridge point 大约是：

```text
74.45 TFLOPS / 8 TB/s ≈ 9.3 FLOP / byte
```

也就是说：

```text
1 << 9.3
```

这已经足够说明 decode 是典型的 memory-bound。

这和实际 benchmark 现象也是一致的。比如 batch 从 1 增长到 256 时，我们看到的不是“算力利用率暴涨”，而是**带宽利用率不断逼近上限**。在大 batch 下，优化过的 CUDA / CuTe 实现可以把带宽利用率推到接近 B200 峰值的 95%，这本身就说明 decode 的主战场是带宽，不是 FLOPs。

所以 GDN 的第二个瓶颈可以概括成一句话：

**不是算不动，而是搬不动。**

## 四、第三层瓶颈：数值稳定性不是附属问题，而是性能前提

如果只有串行依赖和访存压力，那问题还只是“难优化”；真正让 GDN 更棘手的，是它还同时背着数值稳定性约束。

这次数据准备脚本里有一段非常关键的注释，核心意思是：

- 如果直接用 `torch.randn` 生成 `k`
- 那么 `||k||^2` 的量级会接近 head size，也就是大约 128
- 在 delta rule 更新里，state 可能每步被放大约 128 倍
- 从零状态出发，几十步内连 `float32` 都可能 overflow
- 所以必须对 `k` 做 L2 normalize

这个现象说明什么？

说明 GDN 的状态递推不是“只要公式对就行”，而是**数值尺度必须从一开始就被控制住**。否则你连 baseline 都跑不稳，更别说再往低精度压。

这会直接影响算子优化的边界：

- 不是所有中间量都能放心降到 FP16 / BF16
- FP8 / FP4 压缩虽然很诱人，但必须和缩放策略、累积策略一起设计
- recurrent 版本和 chunked 版本即使数学上等价，数值上也可能不等价
- 如果 gate 参数范围控制不好，state 可能爆炸或快速下溢

所以对 GDN 来说，**稳定性不是“最后顺手修一下”的问题，而是决定低精度和高性能方案能否成立的根本前提。**

## 五、为什么论文里的“线性复杂度优势”，不会自动变成 GPU 上的优势

这其实是我觉得最值得反复强调的一点。

GDN / DeltaNet 这类结构从论文上看有很强的吸引力：

- 不显式构造完整 attention matrix
- 时间和空间复杂度更接近线性
- 理论上更适合超长上下文

但工程实现时，现实往往是另一套逻辑：

- 时间维递推依赖导致并行性弱
- 状态读写可能压倒计算本身
- kernel 更像“细粒度流式更新”，不像“厚重 GEMM”
- 低精度压缩受到稳定性强约束

所以，**渐进复杂度更好，不等于 GPU 上一定更快。**

GPU 不是根据大 O 记法工作的，GPU 关心的是：

- 有没有足够厚的并行
- 有没有足够高的算术强度
- 访存是不是连续
- 中间结果能不能留在寄存器或共享内存里
- kernel launch 是否被摊薄

而 GDN 恰好在这些点上都不天然占优。

这也是为什么我更愿意把 GDN kernel optimization 看成一道系统题，而不是一道纯算法题。

## 六、Decode 和 Prefill 的难点并不一样

虽然 decode 和 prefill 都属于同一个模型，但它们的优化抓手并不相同。

### 1. Decode：重点是单步状态更新的极致优化

`gdn_decode_qk4_v8_d128_k_last` 的核心特征是：一次只处理一个 token。

此时 q/k 是单 token 向量，经过 GVA 扩展后，本质计算仍然更接近：

```text
[128 x 128] × [128]
```

也就是矩阵向量乘，而不是矩阵矩阵乘。

这直接带来两个结论：

- Tensor Core 价值非常有限，甚至很多情况下根本用不上
- 主要矛盾落在状态读写、访存模式、launch overhead 和 occupancy 上

因此 decode 阶段真正该优化的是：

- 一次 kernel 内尽可能完成完整的 state update
- 尽量减少中间张量回写
- 用更好的状态布局提高 coalesced access
- 用 shared memory 做短时缓存并消除 bank conflict
- 让小 batch 下的开销尽可能低，让大 batch 下的带宽尽可能满

换句话说，decode 的目标不是“做成大矩阵乘”，而是**把单步状态更新做成极致 memory-efficient 的流式 kernel。**

### 2. Prefill：重点是把递推改造成块并行

`gdn_prefill_qk4_v8_d128_k_last` 的问题则完全不同。

prefill 面对的是整段输入序列，而且还是 variable-length + packed layout 的输入。这时候如果还按 token 一个个串行更新，就会出现两个问题：

- 并行度不够
- state 会被重复读写太多次

因此 prefill 的关键不再是“单步做到多快”，而是**能否把 recurrence 改写成 chunkwise scan / parallel prefix 风格的实现**。

只要能把连续的多个 token 组织成一个 chunk，那么 chunk 内部的一部分计算就有机会从 mat-vec 变成 mat-mat，从而进入 Tensor Core 的甜点区。这也是为什么 prefill 相比 decode，更有机会向高算术强度靠近。

但 prefill 也有自己的系统难点：

- 变长序列导致负载不均衡
- `cu_seqlens` 访问需要很小心
- packed 布局下索引与访存更复杂
- 块间 state 传递不能太贵

所以 prefill 更像一道并行算法设计题，而 decode 更像一道单步 kernel 工程题。

## 七、把问题收束成三个核心瓶颈

如果把上面所有问题再压缩一层，GDN 算子优化的核心瓶颈其实就是三件事：

### 1. 状态递推的串行依赖

- decode 不能像 attention prefill 那样大规模展开并行
- 单 token latency 敏感
- prefill 也必须借助 scan 或 chunk 技术才能并行

### 2. 大状态张量的读写带宽

- 每一步都要读写 state
- 很多场景天然 memory-bound
- 精度压缩会直接影响性能上限

### 3. 低精度下的数值稳定性

- state 容易爆炸或误差累积
- 需要规范化、重参数化和混合精度累积
- 否则“更快”的实现未必能通过正确性

这三点叠加在一起，才构成了 GDN 真正的优化难度。

## 八、真正能落地的算子优化方案，应该怎么做

如果文章只停在“GDN 很难、GDN 很受限”，其实价值不大。真正重要的是：面对这三个瓶颈，算子层面到底有哪些可执行的突破口。

我会把落地方案分成 decode、prefill 和稳定性三条线来讲。

### A. Decode 的落地优化方案：围绕 state update 做极致带宽优化

decode 的目标不是把 FLOPs 做大，而是把**每次状态更新搬得更少、搬得更快、搬得更稳**。

#### 方法 1：融合完整单步更新，避免 state 多次往返 HBM

decode 的一整步通常包含：

- 状态衰减 `g * S`
- `old_v = k @ S`
- `new_v` 的门控混合
- `S += outer(k, delta)`
- `o = q @ S`

如果这些步骤拆成多个 kernel，问题会很严重：

- 每一步都要重新读 state
- 中间结果频繁落回 HBM
- launch overhead 被放大

最基本、也最有效的办法就是：**把单步状态更新尽量 fuse 到一个 kernel 里完成**。一次读入 state，在寄存器或 shared memory 中完成衰减、混合、更新、输出，再一次写回。

这通常是 decode 优化的第一性原则。

#### 方法 2：重排 state layout，保证连续访存和 coalesced access

既然 decode 主要被 state 读写卡住，那 state 的内存布局就非常关键。

一个好的布局至少要满足：

- 同一 warp 访问的是连续地址
- head 维和 `K/V` 维的映射符合线程组织
- 向量化 load/store 足够自然

如果布局不合适，GPU 即使理论带宽很高，也会在非连续访存和 transaction 放大里白白损失掉。

对这类 `[B, H, K, V]` 的状态张量，通常需要围绕 kernel 的 thread mapping 反推 layout，而不是先拍脑袋定布局，再让 kernel 去硬适配。

#### 方法 3：shared memory staging + swizzle，减少 bank conflict

当 state tile 被搬进 shared memory 做局部更新时，bank conflict 会非常容易出现。特别是 `128 x 128` 这种规则矩阵，如果线程映射和地址映射刚好撞上，很容易把 SMEM 吞吐打穿。

这时候一个非常实用的手段就是：

- 把 state tile 分块搬进 SMEM
- 对 SMEM 索引做 swizzle
- 让 warp 内访问尽量打散到不同 bank

在这次实现里，小 batch 场景下，SMEM swizzle 对性能改善是实打实存在的，本质上就是减少 bank conflict，提升 tile 内部更新吞吐。

#### 方法 4：persistent kernel / warp specialization，压缩 launch overhead

decode 的另一个问题是“步子太碎”。

如果每个 token 都是一次很小的 kernel launch，那 launch overhead 在小 batch 时会异常明显。此时可以考虑两类办法：

- `persistent kernel`：让 kernel 常驻，持续消费 token/update task
- `warp specialization`：不同 warp 分工处理 load、compute、store，减少同步和切换开销

这类方法不一定总能带来决定性收益，但在 decode 这种高频、小粒度、强依赖的工作负载下，往往值得尝试。

#### 方法 5：压缩 state 精度，但保留高精度累积

既然 decode 是 memory-bound，那么降低 state 流量往往直接对应性能收益。最直接的办法就是降低 state 存储精度，例如：

- FP16/BF16 state
- FP8 state
- 甚至 FP4 / block-scaled FP4 state

但这里不能只看流量，必须同时看稳定性。更可靠的路径通常不是“全低精度”，而是：

- state 用低精度存储
- 关键累积和更新仍使用 FP32 或至少更高精度
- 为每个 tile / block / head 维护单独 scale

这类 mixed-precision state update，往往是 decode 真正有价值的优化方向，因为它同时对准了“带宽瓶颈”和“稳定性约束”。

### B. Prefill 的落地优化方案：把递推改写成块并行

prefill 的重点不是省一次 launch，而是想办法把长序列递推改造成更厚的并行计算。

#### 方法 1：chunkwise recurrence，把 token-by-token 变成 block-by-block

这是 prefill 最核心的思路。

把序列按 chunk 切开，例如每 64 或 128 个 token 为一块。这样做的价值有两个：

- chunk 内部可以复用同一份 state tile
- 某些 mat-vec 型计算可以转成 mat-mat

当 chunk size 足够合适时，算术强度会明显提高，prefill 就不再像原始串行版本那样完全被带宽卡死。

#### 方法 2：chunk 内用 Tensor Core，chunk 间传递 state

对于 prefill，真正有机会用上 Tensor Core 的位置，不是在“每一步单 token 更新”上，而是在**chunk 内把多个 token 合起来处理**的地方。

例如一段 chunk 内的 `Q_chunk`、`K_chunk`、`V_chunk` 可以组织成矩阵，而 state 到 chunk 输出的部分计算也更接近 mat-mat。这样一来：

- Tensor Core 才有足够大的 tile 可以吃
- TMA / async copy 才更有价值
- 流水线收益也更明显

需要注意的是，chunk 之间仍然有 state 依赖，所以 prefill 并不是“彻底并行”，而是“块内并行、块间递推”。

#### 方法 3：针对 varlen 输入做 length-aware 调度

比赛 workload 里的 prefill 是变长序列，且依赖 `cu_seqlens`。这会带来一个典型问题：不同序列长度差异大时，很容易出现有些 block 很忙、有些 block 很闲。

更好的调度方式通常包括：

- 按序列长度分桶
- 尽量让一个 thread block 处理长度相近的序列块
- 减少 tail block 中的 padding 浪费
- 减少 packed layout 下的分支发散

这部分优化不像 Tensor Core 那样“看起来很酷”，但在真实系统里非常重要，因为它直接决定了 chunked 算法能不能稳定发挥。

#### 方法 4：最小化块间 state 传递和中间写回

prefill 的另一个容易被忽略的问题，是块间 state 传递。

如果每个 chunk 做完都把完整 state 写回 HBM，再让下一个 chunk 重新读回来，那 chunking 的收益会被冲掉一大半。因此实现上最好做到：

- 一个 chunk 内尽量长时间驻留 state tile
- 块边界只传必要状态
- 尽量减少完整 state 的反复回写

本质上，prefill 的高性能不是“把循环拆成块”这么简单，而是“让 state 在块内尽可能活得久，在块间尽可能传得少”。

### C. 稳定性的落地优化方案：不给低精度留下炸掉的机会

如果没有稳定性控制，前面所有优化最终都可能变成“benchmark 跑得快，但结果不对”。

#### 方法 1：对 `k` 或中间量做规范化

这是最直接也最必要的一步。

既然原始 `k` 的范数会直接影响状态更新尺度，那么对 `k` 做 L2 normalize，或者采用某种等价的 scale 控制，本质上就是在从源头限制 state 爆炸风险。

这不是“为了漂亮”，而是为了让递推系统处于可控区间。

#### 方法 2：gate 参数重参数化，限制更新幅度

对于 `a / b / dt_bias / A_log` 这类门控参数，很多时候不能直接裸用，而是需要通过：

- `sigmoid`
- `softplus`
- `exp(-exp())`

这类方式把参数映射到更稳定的范围内。

这一步的意义在于：把“模型学到的自由度”和“递推系统可稳定执行的范围”对齐起来，避免 kernel 端面对过大的动态范围。

#### 方法 3：低精度存储，高精度累积

对于 GDN 来说，一个很现实的折中是：

- 用低精度减少 state 流量
- 用高精度守住递推累积质量

例如：

- `q/k/v` 用 BF16 或 FP8
- state 存储用 FP8 / FP16
- `old_v`、`delta`、state update accumulation 保持 FP32

这个方案不一定是绝对最快的，但通常是工程上风险最低、最容易先跑通 correctness 的路径。

#### 方法 4：为量化状态引入分块 scale

如果进一步把 state 压到 FP8 或 FP4，那么最怕的不是“平均误差大一点”，而是不同 head、不同 tile、不同时间段的动态范围差异太大。

因此更合理的方案通常是：

- 以 head 或 tile 为单位做 block scaling
- 将 scale 和量化后的 state 一起维护
- 在读入 tile 时反量化，在更新后再回量化

这类方案虽然实现复杂度更高，但它能让“低字节流量”和“数值可控”同时成立，是 GDN 真正走向更激进压缩时绕不过去的一步。

## 九、如果我要给这类比赛定一个优化优先级

如果是站在比赛或者工程实现的角度，我会把优先级排成下面这样：

### 对 decode

1. 先做完整单步 fusion，保证只读写一次 state
2. 再做 state layout 调整和向量化访存
3. 再做 shared memory swizzle，减少 bank conflict
4. 再看 persistent kernel / warp specialization 能否改善小 batch 延迟
5. 最后做低精度 state 压缩，并配套 mixed precision 累积

### 对 prefill

1. 先把顺序递推改写成 chunkwise recurrence
2. 再让 chunk 内计算尽可能矩阵化，争取 Tensor Core 利用率
3. 再处理 varlen 调度、packed layout 和负载均衡
4. 最后优化块间 state 传递和量化存储

### 对 correctness

1. 先控制 `k` 和 gate 的数值范围
2. 再验证 recurrent 与 chunked 实现的一致性
3. 再验证低精度压缩下长期递推误差是否可接受

这个顺序的核心逻辑是：**先解决主瓶颈，再做锦上添花。**

GDN 最忌讳的一件事，就是还没把 state traffic 和稳定性理顺，就急着去追求某个局部算力峰值。那样通常只会得到一版“看起来很先进，但整体不划算”的 kernel。

## 十、结论：GDN 的核心矛盾，不在算得慢，而在状态太重

回到最开始的问题。

如果问我，Gated DeltaNet 这类算子和标准 attention 最大的不同是什么，我会说：

**attention 的核心资源是矩阵乘，GDN 的核心资源是状态。**

也正因为如此，GDN 优化的重点从来不只是“把 FLOPs 做高”，而是要同时解决三件事：

- 状态递推带来的串行依赖
- state 读写带来的带宽瓶颈
- 低精度实现面临的稳定性约束

所以一句话总结就是：

**GDN 真正难的，不是“算得不够快”，而是“状态递推难并行、state 读写吃带宽、低精度又容易不稳定”这三件事同时成立。**

而真正可落地的优化路径，也因此非常明确：

- 对 decode，核心是做极致 memory-efficient 的单步状态更新 kernel
- 对 prefill，核心是把递推改写成块并行
- 对低精度，核心是让压缩和稳定性一起成立

这才是 GDN 从论文优势走向工程优势时，真正必须跨过去的那道坎。

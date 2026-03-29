# GDN 文章论点与源码证据对照

这份文档把前面两篇文章里的核心判断，尽量挂到当前仓库的源码、脚本和配置上。

目标不是证明“文章一定完整”，而是说明：**文章里的主判断，和这份仓库当前的实现状态是对得上的。**

## 1. GDN 的核心对象是 `state`，不是 attention matrix

源码里最直接的证据在算法定义本身。

- `README.md:117-128`
- `README.md:120-127`

这里把 GDN 的主流程直接写成：

- `S = g * S`
- `old_v = k @ S`
- `S = S + outer(k, new_v - old_v)`
- `o = q @ S`

这说明实现中心确实不是显式 attention matrix，而是递推状态 `S`。

## 2. Decode 的主瓶颈更接近状态读写带宽，而不是 Tensor Core 算力

仓库最明确的证据来自 README 中已经写死的性能解读。

- `README.md:136-142`
- `README.md:146-158`

这里直接给出：

- decode 是 matrix-vector，而不是 matrix-matrix
- decode 优化目标是 memory bandwidth
- batch=256 时达到约 `7.6 TB/s`，接近 B200 `8 TB/s` 峰值的 `95%`
- 对应 FP32 利用率只有 `7.6%`

这和“decode 主要受 state traffic 和带宽限制”的文章判断完全一致。

## 3. Decode 路线明显比 Prefill 更成熟

从版本分布和 benchmark 入口就能看出来，decode 是主战场。

- `README.md:96-111`
- `scripts/bench_cuda_real.py:1-6`
- `scripts/bench_cuda_real.py:66-101`
- `scripts/build_cuda.py:141-327`

这几处能看到：

- decode 有 v7、v8、v9、v10 多个真实 CUDA/CuTe 版本
- benchmark 脚本显式加载并对比 v7/v8/v9/v10
- build 脚本也为 decode 暴露了多套 wrapper 和 graph/tma 入口

相对地，prefill 当前 checked-in 的实现没有同等成熟的版本演进和 benchmark 支撑。

## 4. Prefill 当前实现仍然是逐 token 递推，不是真正的 chunked scan / Tensor Core 方案

这点在 prefill v7 kernel 里非常清楚。

- `src/kernels/cuda/gdn_prefill_v7.cuh:155-166`
- `src/kernels/cuda/gdn_prefill_v7.cuh:165-260`

当前实现做了：

- 初始 state 读入
- Q/K 双缓冲
- `for (tok = 0; tok < seq_len; tok++)` 的顺序处理
- 每一步做 gate、`old_v`、rank-1 update、output

这说明 prefill 目前仍然是“token-by-token recurrence + 工程优化”，并没有真正走到：

- chunkwise recurrence
- parallel prefix / scan
- chunk 内矩阵化
- Tensor Core 主导的 prefill

这正是文章里“prefill 更像并行算法题，目前仓库还没完全跨过去”的直接源码证据。

## 5. 数值稳定性是硬约束，不是附属问题

这点在 workload 生成脚本里写得非常直接。

- `scripts/setup_volume.py:96-103`

脚本明确说明：

- 原始 `torch.randn` 的 `k` 会导致 `||k||^2 ≈ 128`
- 状态会大约 `128x / step` 地增长
- 从零状态出发，几十步内 float32 也可能 overflow
- 因此必须对 `k` 做 L2 normalize

这几乎就是“GDN 的低精度和高性能实现，建立在先把递推系统稳定住”的源码版结论。

## 6. Decode 的优化路径确实是围绕 state update 做融合、访存和 swizzle

decode v5/v8/v9/v10 代码都指向同一个方向：不是把问题做成更大的 GEMM，而是把单步 state update 做得更高效。

代表性证据：

- `src/kernels/cuda/gdn_decode_v5.cuh:104-119`
- `src/kernels/cuda/gdn_decode_v5.cuh:155-257`
- `src/kernels/cuda/gdn_decode_v8.cuh:1-22`
- `src/kernels/cuda/gdn_decode_v8.cuh:200-260`
- `src/kernels/cute/gdn_decode_v9.cuh:153-235`
- `src/kernels/cute/gdn_decode_v10.cuh:138-176`

这些代码里出现的主关键词是：

- fused gate
- state tile load/store
- shared memory
- swizzle
- warp specialization
- cp.async
- low-precision state

这和文章里总结的 decode 落地路线一致：

- 先把完整单步更新 fuse 起来
- 再优化 state layout 和 coalesced access
- 再做 SMEM swizzle / launch overhead / 低精度 state

## 7. 仓库里存在“两套构建真相”，说明当前更多是实验场而不是完全收敛的产品形态

真正的运行链路和静态构建链路已经分叉了。

### 实际在用的链路

- `README.md:55-59`
- `scripts/build_cuda.py:1-31`
- `scripts/bench_cuda_real.py:48-56`

这条链路是：

- 用 Modal 准备环境
- 用 `scripts/build_cuda.py` 动态拼接 kernel 源码
- 生成 `/data/lib/libgdn_kernels.so`
- 再由 benchmark 脚本通过 `ctypes` 加载

### 已经过期或未同步的链路

- `CMakeLists.txt:35-36`
- `CMakeLists.txt:51-60`
- `src/gdn_kernels.cu:11-20`

这里有几个明显问题：

- include path 指向 `src/kernels`，但 `src/gdn_kernels.cu` 直接 include 裸文件名
- install 路径指向不存在的 `src/kernels/gdn_decode_v7.cuh`
- pybind 模块引用了不存在的 `src/gdn_bindings.cu`

这说明当前仓库的 source of truth 更偏向 `scripts/build_cuda.py`，而不是 CMake。

## 8. 比赛 `solution/cuda` 包装层还停在旧版本，和最佳 kernel 已经脱节

decode 和 prefill 的 solution wrapper 仍然指向 v5。

- `gdn_decode_qk4_v8_d128_k_last/solution/cuda/kernel.py:1-5`
- `gdn_decode_qk4_v8_d128_k_last/solution/cuda/kernel.py:19`
- `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.py:1-5`
- `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.py:18`

这说明：

- 仓库里“最佳 decode kernel”的实验进展
- 比赛提交接口里默认使用的 kernel

已经不是同一个层面的东西。

这也进一步证明：当前仓库更像“高强度实验仓库”，而不是所有入口都已经统一到最新最佳版本。

## 9. 当前 correctness 主要靠 benchmark 对拍，不靠正式回归测试

形式上的测试文件基本是空的。

- `tests/test_correctness.py`

而实际 correctness 更像是通过：

- `README.md:25`
- `scripts/bench_cuda_real.py:103-120`

这类 benchmark / baseline 对拍方式来保障。

这解释了为什么仓库对性能版本迭代很快，但回归测试体系还没有完全跟上。

## 10. 当前源码里还有实验痕迹，说明文章谈“系统题”不是夸张，而是现实

最典型的一处是 v9 decode。

- `src/kernels/cute/gdn_decode_v9.cuh:194-209`

这里只有 `warp 0, lane 0` 计算 `g` 和 `beta`，随后立刻用 `__shfl_sync` 广播。`__shfl_sync` 只能在 warp 内广播，这个写法对其他 warps 并不安全，属于非常值得复核的 correctness 风险点。

而 v10 对应位置则改成了每个线程自己计算 gate：

- `src/kernels/cute/gdn_decode_v10.cuh:124-136`
- `src/kernels/cute/gdn_decode_v10.cuh:279-289`

这说明仓库确实处在“性能探索 + 正确性修正并行推进”的状态里，而不是一个已经完全冻结的最终版代码库。

## 总结

如果把这份源码扫一遍，再回头看前面文章的主结论，会发现它们是互相支持的：

1. GDN 的中心对象确实是 `state`，不是 attention matrix。
2. decode 的主要矛盾确实是 state traffic 和带宽，而不是 Tensor Core 算力。
3. prefill 当前还没有真正跨到 chunked scan / Tensor Core 主导的阶段。
4. 数值稳定性确实是前置条件，不是附属问题。
5. 当前仓库整体上是一个“比赛 kernel 实验场”，不是完全收敛的产品化实现。

因此，前面文章里那句总结是有源码支撑的：

**GDN 真正难的，不是“算得不够快”，而是“状态递推难并行、state 读写吃带宽、低精度又容易不稳定”这三件事同时成立。**

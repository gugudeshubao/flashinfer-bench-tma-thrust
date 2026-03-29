# Next TODO

这份 TODO 按我目前扫源码后的优先级来列，目标是先把 source of truth、correctness 和主路径统一，再继续冲性能。

## P0

### 1. 修复并确认 v9 decode 的 gate 广播正确性

- 文件：`src/kernels/cute/gdn_decode_v9.cuh`
- 问题：`g` / `beta` 目前只在 `warp 0, lane 0` 计算，然后用 `__shfl_sync` 广播；这对 warp 1-3 不安全。
- 方案：
- 用 shared memory 广播给全 block。
- 或直接改成每个线程独立计算 gate，和 v10 保持一致。
- 产出：
- 增加一个最小 correctness regression case，专门对比 v9 和 Triton baseline。

### 2. 明确唯一构建真相，统一 CMake 和 Modal build

- 文件：`CMakeLists.txt`
- 文件：`src/gdn_kernels.cu`
- 文件：`scripts/build_cuda.py`
- 问题：当前 CMake 路线和 Modal build 路线已经分叉，前者存在过期路径和缺失文件引用。
- 方案：
- 要么正式废弃 CMake，并在 README 里写清楚只支持 `scripts/build_cuda.py`。
- 要么修好 CMake include path、install path、bindings path，让它可用。
- 产出：
- 一个明确的 source of truth。
- 一条本地可复现的统一构建命令。

### 3. 补正式 correctness 测试，不再只靠 benchmark 对拍

- 文件：`tests/test_correctness.py`
- 问题：当前测试文件为空，回归主要靠 benchmark 脚本。
- 方案：
- 补 decode/prefill 的最小 correctness tests。
- 至少覆盖 v5、v7、v9、v10 和 Triton baseline。
- 加入小 batch、长序列、不同 `BLOCK_V` 的 case。
- 产出：
- `pytest` 可跑的 correctness 基线。

## P1

### 4. 把比赛 `solution/cuda` 包装层从 v5 升级到当前最佳版本

- 文件：`gdn_decode_qk4_v8_d128_k_last/solution/cuda/kernel.py`
- 文件：`gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.py`
- 问题：这两层包装还停在 v5，和当前 benchmark/最佳实现脱节。
- 方案：
- decode 默认接到 v9 或 v10。
- prefill 明确当前默认版本，并把 fallback 策略说清楚。
- 修正路径引用，避免继续指向旧的 `src/kernels/gdn_*_v5.cuh`。
- 产出：
- 比赛提交入口和仓库最佳实现一致。

### 5. 给 decode 建立稳定的 kernel selection 策略

- 文件：`scripts/bench_cuda_real.py`
- 文件：`README.md`
- 目标：
- 把 batch-size 到 kernel version 的推荐策略固化成代码和文档。
- 明确哪些 batch 用 Triton，哪些 batch 用 v9/v10。
- 产出：
- 一个统一的 runtime dispatch 策略。

### 6. 给 README 和 docs 做一次同步清理

- 文件：`README.md`
- 文件：`docs/PERFORMANCE.md`
- 文件：`docs/ROADMAP.md`
- 问题：README 里有些描述和真实构建/入口已经不完全一致。
- 方案：
- 清理过期版本说明。
- 明确 decode/prefill 的成熟度差异。
- 标注 CMake 是否支持、`solution/cuda` 是否已对齐最佳版本。

## P2

### 7. 真正推进 prefill 的 chunked recurrence 原型

- 文件：`src/kernels/cuda/gdn_prefill_v7.cuh`
- 问题：当前 prefill 还是逐 token recurrence，只做了双缓冲和向量化，没有进入 chunked scan 阶段。
- 方案：
- 先做最小 chunked prefill 原型，不追求一开始就最优。
- 明确 chunk 内哪些部分可以矩阵化。
- 先验证 correctness 和 AI 提升，再谈 Tensor Core。
- 产出：
- 一个和当前 sequential prefill 并行存在的 `prefill_chunked` 原型版本。

### 8. 为 prefill 建立独立的 roofline/perf 报表

- 文件：`docs/ROOFLINE.md`
- 文件：`scripts/bench_*`
- 问题：当前 decode 的性能叙事更清楚，prefill 还缺少独立的版本对比和 roofline 结果。
- 方案：
- 单独统计 prefill 的 seq_len、num_seqs、state traffic、AI、GB/s、TFLOPS。
- 产出：
- 一份可支持文章和优化决策的 prefill 性能表。

### 9. 评估低精度 state 的稳定边界

- 文件：`src/kernels/cuda/gdn_decode_v7.cuh`
- 文件：`src/kernels/cuda/gdn_decode_v8.cuh`
- 文件：`scripts/setup_volume.py`
- 目标：
- 系统测试 FP32 / FP16 / FP8 / FP4 state 在不同长度和不同输入分布下的误差积累。
- 明确“低精度存储 + 高精度累积”的实际边界。
- 产出：
- 一张误差 vs 带宽收益表。

## P3

### 10. 给文档补一份“源码到文章”的对照索引

- 文件：`docs/GDN_SOURCE_ARGUMENT_MAP.md`
- 目标：
- 后续写知乎、技术分享、汇报时，能快速从论点跳回源码证据。

### 11. 统一命名和目录语义

- 目标：
- 让 `v5-v10`、`solution/cuda`、`scripts/build_cuda.py`、`src/gdn_kernels.cu` 的角色更清晰。
- 减少“版本在、入口在，但不是同一条运行链”的混乱。

### 12. 决定仓库最终形态

- 方案 A：
- 保持“比赛实验仓库”定位，允许多条链路并存。
- 方案 B：
- 收敛成“可复现、可构建、可 benchmark、可测试”的统一工程仓库。

- 建议：
- 如果后续还要继续写文章、做展示、给别人复现，优先走方案 B。

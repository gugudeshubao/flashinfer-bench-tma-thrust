# 为什么 Gated Delta Net 的低精度量化比你想象的更难？

> 实测 BF16/FP8/FP4 在 GDN State 上的精度损失，以及背后的数值原因分析

---

## 一、背景：我在做什么

最近在优化 **Gated Delta Net (GDN)** 的 CUDA kernel，目标是在 NVIDIA B200 上跑到接近硬件极限。

GDN 是一种带**递推状态**的注意力变体，核心公式：

```
State_{t+1} = g * State_t + β * (v - old_v) ⊗ k
Output_t = State_{t+1} @ q
```

其中 State 是一个 `[128 × 128]` 的矩阵，每个 token 都要更新一次。

**问题来了**：State 占用的内存是瓶颈（64KB/head），能不能用 FP8 甚至 FP4 压缩？

---

## 二、实测结果：精度 vs 压缩率

我在 Modal B200 上跑了 100 步 decode，对比 FP32 baseline：

| 精度 | 压缩率 | 输出相对误差 | State 相对误差 | 误差累积 |
|------|--------|-------------|---------------|----------|
| **BF16** | 2x | **0.57%** | 0.64% | 0.15x |
| **FP8 E4M3** | 4x | **11.4%** | 10.5% | 0.16x |
| **FP4 E2M1** | 8x | **54.6%** | 64.9% | 0.11x |

**关键发现**：
- BF16 几乎无损（<1%）
- FP8 有 ~11% 误差，但**误差不累积**
- FP4 误差高达 55%，基本不可用

测试命令：
```bash
modal run tests/test_quantization_accuracy.py --precision bf16  # 0.6%
modal run tests/test_quantization_accuracy.py --precision fp8   # 11%
modal run tests/test_quantization_accuracy.py --precision fp4   # 55%
```

---

## 三、为什么 GDN 比普通算子更难低精度？

### 普通 GEMM / Attention

误差是**单层、单次**的：
- 输入低精度 → 计算 → 输出
- 误差停留在这一层，不会传播

### GDN / Delta-rule / Recurrent 系统

误差是**递推累积**的：
- 每个 token 更新 State
- 当前输出依赖当前 State
- 下一个 token 依赖刚更新的 State

```
State_0 → [误差 ε₀] → State_1 → [误差 ε₁] → State_2 → ...
                ↓                    ↓
            Output_0             Output_1
```

**误差会进入 State，被后续步骤反复使用。**

---

## 四、5 个具体的数值问题

### 1. 递推误差累积（最致命）

某一步 State 误差 `ε_t`，不会消失，而是：
- 带着误差进入 `State_{t+1}`
- 再乘 gate、加新项
- 继续传播

**长序列表现明显恶化。**

### 2. 门控量对误差极其敏感

GDN 里的 `a`, `b`, `A_log`, `dt_bias` 这些 gate 参数：
- 低精度量化稍偏一点
- 直接改变系统动力学
- 本来该衰减的没衰减，本来该保留的被抹掉

**不是"噪声大了"，而是"状态演化规律都变了"。**

### 3. State 同时包含大量级和小增量

GDN 的 State 里可能同时存在：
- 比较大的主量级
- 比较小但重要的修正项

FP8/FP4 容易：
- 大量级占据表示能力
- 小更新被吞掉

### 4. 不同张量量化难度不同

| 张量 | 量化难度 | 原因 |
|------|---------|------|
| Q/K/V | 较易 | 单次使用，不累积 |
| Gate 参数 | 敏感 | 影响系统动力学 |
| **State** | **最难** | 递推记忆本体，误差跨步传播 |

### 5. Decode 比 Prefill 更难

- **Decode**：长期递推，State 不断更新，对误差更敏感
- **Prefill**：有并行化，数值路径不同

---

## 五、为什么我的测试显示"误差不累积"？

你可能注意到表里 "误差累积" 列都是 < 1x（0.11x ~ 0.16x）。

这是因为：
1. **Gate g < 1**：State 自带衰减，旧误差会被逐步遗忘
2. **Per-row 动态缩放**：每行独立计算 scale，限制了误差放大
3. **测试用了稳定的输入分布**：`g ∈ [0.5, 0.9]`, `β ∈ [0.1, 0.4]`

**但这不代表实际模型也这样！** 真实模型的 gate 分布可能更极端。

---

## 六、工程建议：怎么做 GDN 低精度

### 推荐方案：混合精度

```
存储: FP8 (4x 压缩)
计算: FP32 (累积精度)
Gate: 保持高精度
```

**核心原则：压存储，不压核心递推累积。**

### 可行性分级

| 难度 | 方案 |
|------|------|
| ✅ 容易 | BF16 compute + FP32 accumulate |
| ⚠️ 中等 | State 用 FP8 存储，FP32 计算 |
| ❌ 很难 | State 全程 FP8 递推 |
| ❌ 极难 | State + Gate 全 FP4 |

### 我的实现

```cpp
// 加载时 dequantize
float state = fp8_to_fp32(packed) * row_scale;

// FP32 计算 delta rule
state = g * state + delta * k;

// 存储时 quantize
fp8 = fp32_to_fp8(state / new_scale);
```

---

## 七、结论

**一句话总结**：

> GDN 比普通算子更难做 FP8/FP4，因为它是带递推 State 的动态系统，低精度误差会跨 token 累积。State 和 Gate 对量化误差极其敏感。

**实测数据**：
- BF16：0.6% 误差，推荐用于高精度推理
- FP8：11% 误差，可用于一般推理（需混合精度）
- FP4：55% 误差，不推荐

**工程建议**：
- 压存储，不压计算
- State 存低精度，算高精度
- Gate 保持高精度

---

## 代码

完整实现见：
- `src/kernels/cute_cpp/gdn_decode_v10.cuh` - FP8/FP4 kernel
- `src/kernels/ptx/gdn_decode_ptx.cuh` - PTX 实现
- `tests/test_quantization_accuracy.py` - 精度测试

---

*如果这篇文章对你有帮助，欢迎点赞收藏。有问题可以评论区讨论。*

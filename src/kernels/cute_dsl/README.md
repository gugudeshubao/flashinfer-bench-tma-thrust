# CuTe DSL Python Kernels

> **CuTe DSL** 是 CUTLASS 4.0 新增的 Python 原生接口，让开发者用 Python 编写 GPU kernel，同时保持 C++ 级别的性能。

## 技术栈

| 特性 | 描述 |
|------|------|
| **库** | CUTLASS 4.4.2 |
| **语言** | Python |
| **编译** | JIT (秒级) |
| **性能** | ~100% C++ 性能 |
| **GPU** | B200 (sm100) |

## 当前实现

### gdn_decode_dsl.py

简化版 GDN decode kernel，演示 CuTe DSL 基本模式：

- **功能**: 计算 `out = scale * State @ Q` (矩阵-向量乘法)
- **状态**: ✅ 已测试通过 (Modal B200)
- **精度**: Max diff vs PyTorch < 0.01 (bf16 precision)

```python
@cute.kernel
def _gdn_state_matmul_kernel(
    gState: cute.Tensor,   # [B * 8 * D * D]
    gQ: cute.Tensor,       # [B * 4 * D]
    gOut: cute.Tensor,     # [B * 8 * D]
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    
    # ... compute State @ Q
    gOut[out_idx] = acc
```

### 完整 GDN Delta Rule

完整的 GDN kernel 需要：
1. 门控计算 (softplus, sigmoid) 
2. 状态衰减和 rank-1 更新
3. Tensor Core MMA 优化

对于生产环境，建议使用 **Triton kernel** (`gdn_decode_triton.py`)。

## 使用方法

```bash
# 安装 CUTLASS DSL
pip install nvidia-cutlass-dsl>=4.3

# 测试 (Modal B200)
modal run scripts/test_cute_dsl.py
```

## API 模式

```python
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# 1. 定义 kernel
@cute.kernel
def my_kernel(gInput: cute.Tensor, gOutput: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    idx = bidx * 128 + tidx
    gOutput[idx] = gInput[idx]

# 2. 定义 host function
@cute.jit
def launch(mInput, mOutput, num_blocks: int):
    my_kernel(mInput, mOutput).launch(
        grid=[num_blocks, 1, 1],
        block=[128, 1, 1],
    )

# 3. 调用
mInput = from_dlpack(torch_tensor).mark_layout_dynamic()
mOutput = from_dlpack(output_tensor).mark_layout_dynamic()
launch(mInput, mOutput, num_blocks)
```

## 状态

✅ **已验证** - CuTe DSL 4.4.2 在 Modal B200 上可用

## 参考

- [CUTLASS 4.0 CuTe DSL](https://github.com/NVIDIA/cutlass)
- [FlashAttention-4 Paper](https://arxiv.org/abs/2603.05451)
- [NVIDIA Blog: CuTe DSL](https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/)

# CuTe DSL Python Kernels

> **CuTe DSL** 是 CUTLASS 4.0 新增的 Python 原生接口，让开发者用 Python 编写 GPU kernel，同时保持 C++ 级别的性能。

## 技术栈

| 特性 | 描述 |
|------|------|
| **库** | CUTLASS 4.0+ |
| **语言** | Python |
| **编译** | JIT (秒级) |
| **性能** | ~100% C++ 性能 |

## 典型应用

- **FlashAttention-4**: 完全使用 CuTe DSL 实现
- 编译时间从分钟级降到秒级 (20-30x 更快)

## API 示例

```python
import cute
from cutlass import Float16, Float32

@cute.jit
def gdn_decode_dsl(state, q, k, v, out):
    # Layout 定义 (与 C++ CuTe 一致)
    state_layout = cute.make_layout(
        cute.make_shape(V, D),
        cute.make_stride(D, 1)
    )
    
    # SMEM 分配
    smem_state = cute.SharedMemory(state_layout, dtype=Float32)
    
    # TiledMMA (tcgen05.mma for Blackwell)
    mma_atom = cute.tcgen05.MmaF16BF16Op(
        io_dtype=Float16,
        acc_dtype=Float32,
        mma_inst_shape_mnk=(128, 128, 64),
    )
    tiled_mma = cute.make_tiled_mma(mma_atom)
    
    # 计算
    # ...
```

## 状态

🚧 **规划中** - 等待 CUTLASS 4.0 稳定版本

## 参考

- [CUTLASS 4.0 CuTe DSL](https://github.com/NVIDIA/cutlass)
- [FlashAttention-4 Paper](https://arxiv.org/abs/2603.05451)
- [NVIDIA Blog: CuTe DSL](https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/)

# cuTile Python Kernels (v11 规划)

> **cuTile** 是 NVIDIA 在 **CUDA 13.1 (2025年12月)** 发布的全新 Python GPU 编程模型，直接对标 Triton！

## 技术栈

| 特性 | 描述 |
|------|------|
| **库** | CUDA 13.1+ |
| **语言** | Python |
| **抽象级别** | 高 (Tile-based) |
| **发布** | 2025.12 |

## cuTile vs Triton

| 特性 | cuTile | Triton |
|------|--------|--------|
| 发布方 | NVIDIA (官方) | OpenAI |
| 支持硬件 | NVIDIA only | NVIDIA + AMD |
| Tensor Core | 自动 | 自动 |
| TMA 支持 | 原生 | 需要 tl.experimental |
| 开源 | ❌ | ✅ |

## API 示例

```python
import cuda.tile as ct

@ct.kernel
def gdn_decode_v11(state, q, k, v, out, 
                   D: ct.Constant[int] = 128,
                   V: ct.Constant[int] = 128):
    # 获取 block ID
    batch_id = ct.bid(0)
    head_id = ct.bid(1)
    
    # 加载 tiles (自动处理内存传输)
    state_tile = ct.load(state, index=(batch_id, head_id), shape=(V, D))
    q_tile = ct.load(q, index=(batch_id, head_id), shape=(D,))
    k_tile = ct.load(k, index=(batch_id, head_id), shape=(D,))
    
    # 计算 (自动向量化，无需手动 float4)
    o_tile = ct.sum(state_tile * q_tile, axis=1)  # [V,D] * [D] -> [V]
    
    # 更新 state (delta rule)
    v_tile = ct.load(v, index=(batch_id, head_id), shape=(V,))
    delta = v_tile - o_tile
    state_tile = state_tile + ct.outer(k_tile, delta)
    
    # 写回
    ct.store(state, index=(batch_id, head_id), tile=state_tile)
    ct.store(out, index=(batch_id, head_id), tile=o_tile)
```

## 核心特点

- **纯 Python 语法**，无需 C++
- **Tile-based 抽象**，自动管理线程/block
- 自动利用 **Tensor Core** 和 **TMA**
- 编译器自动优化，无需手动调优

## 状态

🚧 **规划中** - 需要 CUDA 13.1+

## 参考

- [NVIDIA Blog: cuTile Python](https://developer.nvidia.com/blog/simplify-gpu-programming-with-nvidia-cuda-tile-in-python/)
- [CUDA 13.1 Release Notes](https://developer.nvidia.com/cuda-toolkit)

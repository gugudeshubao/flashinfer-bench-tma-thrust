# Triton Kernels

High-level DSL baseline for performance comparison.

## What is Triton?

Triton 是 OpenAI 开发的 GPU 编程 DSL，特点:

| 特性 | 描述 |
|------|------|
| **Python DSL** | 用 Python 写 GPU kernel |
| **Auto-tuning** | 自动选择 tile size、warp 数量等 |
| **Auto SMEM** | 自动管理 shared memory |
| **Portable** | 支持 NVIDIA/AMD GPU |

## Files

| File | Description |
|------|-------------|
| `gdn_decode_triton.py` | Triton decode kernel (symlink) |
| `gdn_prefill_triton.py` | Triton prefill kernel (symlink) |

## Why Triton?

| 优点 | 缺点 |
|------|------|
| Python 语法，学习成本低 | 性能上限略低 |
| 自动优化 tile/warp | 难以精细控制 SMEM |
| 跨平台 (NVIDIA/AMD) | Swizzle 需手动实现 |
| 快速迭代 | JIT 编译开销 |

## vs Raw CUDA / CuTe

| 维度 | Triton | CuTe | Raw CUDA |
|------|--------|------|----------|
| **语言** | Python | C++ | C++ |
| **抽象级别** | 高 | 中 | 低 |
| **学习成本** | 低 | 中高 | 高 |
| **SMEM 控制** | 自动 | 声明式 | 手动 |
| **Bank Conflict** | 自动处理 | Swizzle 抽象 | 手动 XOR |
| **Tensor Core** | 自动 | WGMMA 抽象 | 手动 PTX |
| **性能上限** | 略低 | 最高 | 最高 |

## Implementation

```python
@triton.jit
def _decode_kernel_v5(Q, K, V, State, ...):
    # Triton 自动处理 tile 和 SMEM
    q = tl.load(Q + offsets)
    k = tl.load(K + offsets)
    v = tl.load(V + offsets)
    S = tl.load(State + state_offsets)
    
    # Delta rule (same algorithm)
    S = g * S              # Decay first
    old_v = tl.sum(S * k)  # Then compute old_v
    delta = beta * (v - old_v)
    S = S + delta * k
    
    # Output
    out = scale * tl.sum(S * q)
    tl.store(Out + ..., out)
```

## Performance (B200)

| Batch | Triton v5 | CuTe v9 | Winner | Why |
|-------|-----------|---------|--------|-----|
| 1 | 24 GB/s | **27 GB/s** | CuTe | Launch overhead |
| 16 | 386 GB/s | **405 GB/s** | CuTe | Better SMEM |
| 64 | **1,518 GB/s** | 1,302 GB/s | **Triton** | Auto-tuning |
| 256 | 2,834 GB/s | **7,585 GB/s** | CuTe | Swizzle |

### Why Triton Wins at batch=64?

1. **Auto-tuning**: Triton 自动尝试多种 tile 配置
2. **Optimal BLOCK_V**: 可能选择了不同于我们固定配置的值
3. **L2 Cache**: batch=64 刚好适合 L2 cache 大小

### Why Triton Loses at batch=256?

1. **Bank Conflict**: 没有 SMEM swizzle
2. **Suboptimal Tile**: Auto-tuning 不一定找到全局最优
3. **Less Control**: 无法精细优化内存访问模式

## When to Use Triton?

✅ 快速原型开发  
✅ 需要跨平台支持  
✅ 中小 batch size  
✅ 不需要极致性能  

❌ 大 batch 高吞吐场景  
❌ 需要精细 SMEM 控制  
❌ 需要 Tensor Core 优化  

## Recommendation

```python
def select_kernel(batch_size):
    if batch_size <= 16:
        return "CuTe v9"   # Best at small batch
    elif batch_size <= 64:
        return "Triton v5"  # Auto-tuning wins
    else:
        return "CuTe v9/v10"  # Swizzle wins
```

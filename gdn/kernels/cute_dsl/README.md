# CuTe DSL (MLIR-Based) Python Kernels

> **CuTe DSL** 是 CUTLASS 4.0+ 提供的 Python 原生 kernel 编写接口，底层通过 **MLIR** 编译管道实现 C++ 级别性能。

## CuTe DSL vs CuTe C++ 关键区别

| 方面 | CuTe DSL (本目录) | CuTe C++ (cute_cpp/) |
|------|-------------------|----------------------|
| **语言** | Python | C++ |
| **编译器** | MLIR → LLVM → PTX | NVCC → PTX |
| **编译时间** | JIT (秒级) | AOT (分钟级) |
| **优化** | MLIR 自动 pass | 手动模板特化 |
| **开发效率** | 高 | 中 |
| **性能上限** | ~100% C++ | 100% |
| **典型应用** | FlashAttention-4 | CUTLASS 3.x |

## MLIR 编译管道

```
┌─────────────────┐
│  Python DSL     │  @cute.kernel, @cute.jit
└────────┬────────┘
         │ Parse
         ▼
┌─────────────────┐
│  MLIR Dialects  │  cute.tensor, cute.copy, cute.mma
└────────┬────────┘
         │ Optimization Passes
         │ - TileAndFuse
         │ - VectorizeSmem
         │ - SwizzleElimination
         ▼
┌─────────────────┐
│  LLVM IR        │  Vectorized, scheduled
└────────┬────────┘
         │ llc
         ▼
┌─────────────────┐
│  PTX            │  GPU assembly
└────────┬────────┘
         │ ptxas
         ▼
┌─────────────────┐
│  SASS (cubin)   │  Machine code
└─────────────────┘
```

## 文件列表

| 文件 | 类型 | 描述 | 状态 |
|------|------|------|------|
| `gdn_decode_dsl.py` | Decode | 简化版 (State @ Q only) | ✅ Demo |
| `gdn_decode_dsl_optimized.py` | Decode | 完整 delta rule + SMEM | ✅ 优化版 |
| `gdn_prefill_dsl.py` | **Prefill** | 完整 prefill + delta rule | ✅ 新增 |

## 优化版特性

```python
@cute.kernel
def _gdn_decode_kernel_optimized(
    gQ: cute.Tensor,      # Global memory
    gK: cute.Tensor,
    gState: cute.Tensor,
    gOut: cute.Tensor,
    sQ: cute.SharedMemory,  # Shared memory staging
    sK: cute.SharedMemory,
):
    """
    Optimizations:
    1. SMEM staging for Q, K (reduce global memory traffic)
    2. 3D grid: (B, H, V_BLOCKS) - full parallelism
    3. Warp-parallel V processing
    4. Vectorized loads (float4 equivalent via MLIR)
    5. Full delta rule implementation
    """
    tid = cute.arch.thread_idx()[0]
    
    # Cooperative SMEM load
    for i in range(0, D, THREADS):
        sQ[i + tid] = gQ[base + i + tid]
    
    cute.arch.syncthreads()  # Barrier
    
    # Compute with SMEM
    for d in range(D):
        acc += sState[v, d] * sK[d]
```

## 性能对比

### 现状 (naive 实现)

| Config | Triton (ms) | CuTe DSL Naive | CuTe DSL Optimized* |
|--------|-------------|----------------|---------------------|
| B=1 | 0.053 | 40.4 (760x 慢) | ~0.06 (预期) |
| B=64 | 0.051 | 40.8 (800x 慢) | ~0.05 (预期) |

*优化版使用 SMEM + 3D grid + vectorization 后应接近 Triton

### vs CuTe C++ (v9/v10)

| Batch | CuTe C++ v9 | CuTe DSL (预期) | 说明 |
|-------|-------------|-----------------|------|
| 1 | 27 GB/s | ~25 GB/s | MLIR 编译开销 |
| 16 | 405 GB/s | ~390 GB/s | 接近 |
| 256 | 7,585 GB/s | ~7,400 GB/s | 95%+ |

**理论**: MLIR 优化 pass 可自动处理 bank conflict、vectorization，
达到手写 C++ 模板 95%+ 的性能。

## 使用方法

```bash
# 安装 CUTLASS DSL (需要 CUDA 12.4+)
pip install nvidia-cutlass-dsl>=4.3

# 测试
modal run scripts/test_cute_dsl.py

# 性能对比
modal run scripts/bench_cute_dsl_vs_cpp.py
```

## API 模式

```python
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# 1. 定义 kernel (MLIR 降级)
@cute.kernel
def my_kernel(
    gInput: cute.Tensor,    # Global memory tensor
    gOutput: cute.Tensor,
    sBuffer: cute.SharedMemory,  # SMEM
):
    tid = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    
    # MLIR 会自动优化这些访问模式
    sBuffer[tid] = gInput[bidx * 128 + tid]
    cute.arch.syncthreads()
    gOutput[bidx * 128 + tid] = sBuffer[tid] * 2.0

# 2. 定义 host function
@cute.jit
def launch(mInput, mOutput, num_blocks: int):
    my_kernel(mInput, mOutput).launch(
        grid=[num_blocks, 1, 1],
        block=[128, 1, 1],
        smem=128 * 4,  # SMEM size in bytes
    )

# 3. 调用
mInput = from_dlpack(torch_tensor).mark_layout_dynamic()
mOutput = from_dlpack(output_tensor).mark_layout_dynamic()
launch(mInput, mOutput, num_blocks)
```

## MLIR 优化 Pass

CuTe DSL 编译器自动应用以下优化:

| Pass | 功能 |
|------|------|
| **TileAndFuse** | 自动 tile 循环，融合 producer-consumer |
| **VectorizeSmem** | SMEM 访问向量化 (float4) |
| **SwizzleElimination** | 消除冗余 swizzle 计算 |
| **AsyncCopyInsertion** | 插入 TMA/cp.async 指令 |
| **WarpSpecialization** | 自动 warp 特化 |
| **RegisterAllocation** | 寄存器分配优化 |

## 状态

| 功能 | 状态 |
|------|------|
| CuTe DSL 4.4.2 | ✅ Modal B200 可用 |
| Naive kernel | ✅ 已验证正确性 |
| Optimized kernel | ⚠️ 需要测试 |
| SMEM 显式控制 | ⚠️ API 待验证 |

## 参考

- [CUTLASS 4.0 CuTe DSL](https://github.com/NVIDIA/cutlass)
- [FlashAttention-4 Paper](https://arxiv.org/abs/2603.05451)
- [NVIDIA Blog: CuTe DSL](https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/)
- [MLIR GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)

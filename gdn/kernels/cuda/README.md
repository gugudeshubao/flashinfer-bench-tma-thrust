# Raw CUDA Kernels (v5-v8)

Low-level CUDA C++ implementations with full control over memory and compute.

## Why Raw CUDA?

| 优点 | 缺点 |
|------|------|
| 完全控制寄存器/SMEM | 代码冗长 |
| 性能可预测 | 容易出错 |
| 可使用所有 PTX 指令 | 维护成本高 |
| 最高性能上限 | 学习曲线陡峭 |

## Version Evolution

| Version | Key Optimization | Code Example |
|---------|-----------------|--------------|
| v5 | Baseline | `atomicAdd` reduction |
| v6 | TMA async | `cp.async.bulk.tensor` |
| v7 | Vectorized | `float4` loads |
| v8 | Warp spec | Producer/consumer warps |

## Files

| File | Features |
|------|----------|
| `gdn_decode_v5.cuh` | Baseline, atomicAdd reduction |
| `gdn_decode_v6.cuh` | TMA async loads (mbarrier) |
| `gdn_decode_v7.cuh` | Vectorized float4, FP4 quantization |
| `gdn_decode_v8.cuh` | Warp specialization, FP8 quantization |
| `gdn_prefill_v5-v8.cuh` | Prefill kernel variants |

## Code Examples

### v7: Manual float4 Vectorization

```cpp
// 手动向量化加载 (4x 带宽效率)
float4* state_f4 = reinterpret_cast<float4*>(&state[idx]);
float4 s = state_f4[d / 4];

// 解包并计算
old_v += s.x * k[d+0];
old_v += s.y * k[d+1];
old_v += s.z * k[d+2];
old_v += s.w * k[d+3];
```

### v8: Warp Specialization

```cpp
// Warp 角色分工
int warp_id = threadIdx.x / 32;

if (warp_id < 2) {
    // Compute warps: 执行矩阵-向量乘法
    compute_matvec();
} else {
    // Memory warp: 异步预取下一个 state
    prefetch_next_state();
}
__syncthreads();  // 同步
```

## Performance (B200, batch=256)

| Version | Bandwidth | vs Triton | Key Feature |
|---------|-----------|-----------|-------------|
| v5 | ~7,500 GB/s | 2.65x | Baseline |
| v6 | ~7,500 GB/s | 2.65x | TMA |
| v7 | 7,578 GB/s | 2.67x | float4 |
| v8 | **7,605 GB/s** | **2.68x** | Warp spec |

All versions achieve **~95% of B200 peak bandwidth** (8 TB/s).

## vs CuTe

| Aspect | Raw CUDA | CuTe |
|--------|----------|------|
| Swizzle | 手动 XOR 计算 | `Swizzle<3,3,3>` |
| Layout | 手动 stride 计算 | `Layout<Shape, Stride>` |
| TMA | 手动 mbarrier | `copy_async()` |
| 代码量 | ~650 行/kernel | ~400 行/kernel |
| 性能 | **95% BW** | **95% BW** |

**结论**: 性能相同，但 CuTe 代码更简洁、易维护。

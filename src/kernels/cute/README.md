# CuTe Kernels (v9-v10)

NVIDIA CuTe (CUTLASS Tile) based implementations.

## Files

| File | Version | Features |
|------|---------|----------|
| `gdn_decode_v9.cuh` | v9 | SMEM swizzle, cp.async |
| `gdn_decode_v10.cuh` | v10 | CuTe `Swizzle<3,3,3>` layout algebra |

## Key Features

### v9 - Manual Swizzle
```cpp
// XOR-based swizzle for bank conflict avoidance
int swizzled_d = d ^ ((d >> 3) & 7);
s_state[v_idx * D + swizzled_d] = ...;
```

### v10 - CuTe Layout Algebra
```cpp
template<int BLOCK_V>
struct SwizzledStateLayout {
    using SwizzleType = Swizzle<3, 3, 3>;  // B=3, M=3, S=3
    
    __device__ __forceinline__ 
    static int get_index(int v_idx, int d_idx) {
        int swizzled_d = d_idx ^ ((d_idx >> 3) & 7);
        return v_idx * V10_D + swizzled_d;
    }
};
```

## Performance (B200)

| Batch | v9 | v10 CuTe |
|-------|-----|----------|
| 1 | 27 GB/s (1.11x) | 26 GB/s (1.10x) |
| 16 | 405 GB/s (1.05x) | 403 GB/s (1.04x) |
| 256 | 7,585 GB/s (2.68x) | 7,602 GB/s (2.68x) |

v9 is slightly faster at small batch due to simpler code.

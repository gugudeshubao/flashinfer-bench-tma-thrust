# cuTile GDN Decode Kernel

> **cuTile** is NVIDIA's official tile-based Python GPU programming model, released in **CUDA 13.1**.

## Status: ✅ Working (Correctness Verified)

| Test | Max Diff | Status |
|------|----------|--------|
| Vector Add | 0.00e+00 | ✓ |
| Matrix-Vector | 1.9e-06 | ✓ |
| GDN Decode (Full Delta Rule) | out: 3.81e-06, state: 1.14e-05 | ✓ |

## Performance Benchmark (cuTile vs Triton)

| Batch | Triton (ms) | cuTile (ms) | Speedup | Triton BW |
|-------|-------------|-------------|---------|-----------|
| 1 | 0.052 | 1.835 | 0.03x | 20 GB/s |
| 4 | 0.057 | 6.611 | 0.01x | 74 GB/s |
| 16 | 0.043 | 25.98 | 0.00x | 395 GB/s |
| 64 | 0.043 | N/A | N/A | 1577 GB/s |

**Key Finding**: cuTile is currently **~30-600x slower** than Triton due to:
1. **Per-slice processing**: B×H=32 Python loop iterations
2. **Multiple kernel launches**: 32 launches vs 1 for Triton
3. **CuPy/PyTorch conversion overhead**: Memory copies per slice

## cuTile Limitation

cuTile's `ct.load` uses **tile-based indexing** which doesn't support arbitrary strided 4D access:

```python
# Triton: element-based pointer arithmetic
s_ptr = State + b * stride_s_b + h * stride_s_h + v0 * stride_s_v
S = tl.load(s_ptr + vi * stride_s_v + ki)  # Direct offset!

# cuTile: tile-based indexing (must pass 2D slice)
S_tile = ct.load(state_2d_slice, index=(pid, 0), shape=(tile_v, D))
```

**Implication**: Cannot efficiently batch 4D state access in a single kernel launch.

## Key Implementation Insight

**cuTile uses TILE-BASED indexing, not element indexing!**

```python
# For a 2D array of shape (V, D):
# ct.load(arr, index=(tile_row, tile_col), shape=(h, w)) loads:
#   - rows from tile_row * h to tile_row * h + h
#   - cols from tile_col * w to tile_col * w + w

# Example: load rows 32-48 (tile_row=2, h=16)
S_tile = ct.load(state, index=(2, 0), shape=(16, 128))
```

## Files

- `gdn_decode_cutile.py` - Full GDN decode kernel with delta rule

## Algorithm (Delta Rule)

```
1. S = g * S                   # decay state
2. old_v = S @ k               # mat-vec [V,D] × [D] → [V]
3. delta = beta * (v - old_v)  # compute delta
4. S = S + outer(delta, k)     # rank-1 update
5. out = scale * S @ q         # output
```

## cuTile vs Triton

| Feature | cuTile | Triton |
|---------|--------|--------|
| Publisher | NVIDIA (official) | OpenAI |
| Hardware | NVIDIA only | NVIDIA + AMD |
| Tensor Core | Auto | Auto |
| TMA Support | Native | tl.experimental |
| Open Source | ❌ | ✅ |

## API Example

```python
import cuda.tile as ct

@ct.kernel
def gdn_kernel(
    state,    # [D, D] 2D array
    q_vec,    # [D] 1D array
    out,      # [D] 1D array
    D_size: ct.Constant[int],
    tile_v: ct.Constant[int],
):
    pid = ct.bid(0)  # Tile index
    
    # Load tiles (tile-based indexing)
    S_tile = ct.load(state, index=(pid, 0), shape=(tile_v, D_size))
    q_tile = ct.load(q_vec, index=(0,), shape=(D_size,))
    
    # Compute (auto-vectorized)
    out_tile = ct.sum(S_tile * q_tile, axis=1)  # [tile_v]
    
    # Store
    ct.store(out, index=(pid,), tile=out_tile)
```

## Usage

```bash
# Run tests
modal run scripts/test_cutile.py
```

## Requirements

- CUDA 13.0+ (available on Modal B200)
- `pip install cuda-tile[tileiras] cupy-cuda13x`

## References

- [NVIDIA Blog: cuTile Python](https://developer.nvidia.com/blog/simplify-gpu-programming-with-nvidia-cuda-tile-in-python/)
- [CUDA 13.1 Release Notes](https://developer.nvidia.com/cuda-toolkit)

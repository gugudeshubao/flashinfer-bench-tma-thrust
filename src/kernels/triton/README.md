# Triton Kernels

Triton-based implementations (baseline for comparison).

## Files

| File | Description |
|------|-------------|
| `gdn_decode_triton.py` | Triton decode kernel (symlink to solution) |
| `gdn_prefill_triton.py` | Triton prefill kernel (symlink to solution) |

## Implementation

```python
@triton.jit
def _decode_kernel_v5(Q, K, V, State, ...):
    # Load Q, K, V
    q = tl.load(Q + ...)
    k = tl.load(K + ...)
    v = tl.load(V + ...)
    S = tl.load(State + ...)
    
    # Delta rule
    S = g * S              # Decay first
    old_v = tl.sum(S * k)  # Then compute old_v
    delta = beta * (v - old_v)
    S = S + delta * k
    
    # Output
    out = scale * tl.sum(S * q)
```

## Performance (B200)

| Batch | Triton v5 | vs CUDA |
|-------|-----------|---------|
| 1 | 24 GB/s | baseline |
| 16 | 386 GB/s | baseline |
| 64 | 1,518 GB/s | **wins** |
| 256 | 2,834 GB/s | CUDA 2.68x |

Triton auto-tuning wins at batch=64 due to optimal tile sizes.

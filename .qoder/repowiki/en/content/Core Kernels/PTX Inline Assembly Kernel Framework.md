# PTX Inline Assembly Kernel Framework

<cite>
**Referenced Files in This Document**
- [README.md](file://README.md)
- [gdn_decode_ptx.cuh](file://src/kernels/ptx/gdn_decode_ptx.cuh)
- [gdn_prefill_ptx.cuh](file://src/kernels/ptx/gdn_prefill_ptx.cuh)
- [README.md](file://src/kernels/ptx/README.md)
- [gdn_kernels.cu](file://src/gdn_kernels.cu)
- [CMakeLists.txt](file://CMakeLists.txt)
- [bench_all_versions.py](file://scripts/bench_all_versions.py)
- [build_cuda.py](file://scripts/build_cuda.py)
- [bench_modal.py](file://benchmarks/bench_modal.py)
- [ROADMAP.md](file://docs/ROADMAP.md)
- [OPTIMIZATION_LOG.md](file://docs/OPTIMIZATION_LOG.md)
- [config.toml](file://gdn_decode_qk4_v8_d128_k_last/config.toml)
- [test_fp8_accuracy.py](file://tests/test_fp8_accuracy.py)
- [gdn_decode_v8.cuh](file://src/kernels/cuda/gdn_decode_v8.cuh)
- [gdn_decode_v10.cuh](file://src/kernels/cute_cpp/gdn_decode_v10.cuh)
</cite>

## Update Summary
**Changes Made**
- Added comprehensive FP8 state quantization support documentation
- Updated Inline Assembly Primitives section with FP8 conversion primitives
- Enhanced Memory Operations section with vectorized FP8 memory operations
- Added per-row dynamic scaling implementation details
- Updated Performance Optimization Strategies with FP8 compression benefits
- Enhanced Implementation Details with FP8 kernel variants

## Table of Contents
1. [Introduction](#introduction)
2. [Framework Architecture](#framework-architecture)
3. [PTX Kernel Implementations](#ptx-kernel-implementations)
4. [Inline Assembly Primitives](#inline-assembly-primitives)
5. [Performance Optimization Strategies](#performance-optimization-strategies)
6. [Build System and Integration](#build-system-and-integration)
7. [Benchmarking and Evaluation](#benchmarking-and-evaluation)
8. [Optimization Roadmap](#optimization-roadmap)
9. [Implementation Details](#implementation-details)
10. [Conclusion](#conclusion)

## Introduction

The PTX Inline Assembly Kernel Framework represents the lowest-level optimization layer in the Gated Delta Net (GDN) kernel ecosystem. This framework leverages NVIDIA's Parallel Thread Execution (PTX) instruction set to achieve maximum control over GPU operations, enabling fine-grained optimizations that are not possible with high-level CUDA abstractions.

The framework serves as a critical fallback mechanism and performance optimization layer, particularly for scenarios where maximum performance is paramount and every micro-optimization counts. It provides direct access to warp-level primitives, fast mathematical functions, memory operations with cache hints, and predicated execution capabilities.

**Updated** Added FP8 state quantization support for 4x memory compression while maintaining computational accuracy through per-row dynamic scaling.

## Framework Architecture

The PTX framework operates within a multi-layered kernel optimization hierarchy, positioned as the most granular level of control:

```mermaid
graph TB
subgraph "Kernel Optimization Hierarchy"
Triton[Triton Baseline] --> CuTe[CuTe C++]
CuTe --> PTX[PTX Inline Assembly]
PTX --> Hardware[GPU Hardware]
subgraph "Framework Layers"
PTX --> Primitives[Inline Assembly Primitives]
PTX --> Math[Fast Math Functions]
PTX --> Memory[Memory Operations]
PTX --> Warp[Warp-Level Operations]
end
subgraph "Optimization Targets"
Decode[Decode Kernel]
Prefill[Prefill Kernel]
Async[Async Prefetch]
Chunking[Chunked Processing]
Quantization[FP8 State Quantization]
end
PTX --> Decode
PTX --> Prefill
PTX --> Async
PTX --> Chunking
PTX --> Quantization
end
```

**Diagram sources**
- [gdn_decode_ptx.cuh:1-491](file://src/kernels/ptx/gdn_decode_ptx.cuh#L1-L491)
- [gdn_prefill_ptx.cuh:1-358](file://src/kernels/ptx/gdn_prefill_ptx.cuh#L1-L358)

The architecture emphasizes three core principles:
- **Maximum Performance**: Direct hardware control through PTX assembly
- **Fallback Capability**: Provides optimized implementation when higher layers cannot achieve desired performance
- **Complementary Optimization**: Works alongside CuTe C++ implementations for comprehensive coverage
- **Memory Efficiency**: FP8 quantization reduces memory bandwidth by 4x while maintaining accuracy

**Section sources**
- [README.md:1-168](file://README.md#L1-L168)
- [ROADMAP.md:1-180](file://docs/ROADMAP.md#L1-L180)

## PTX Kernel Implementations

### Decode Kernel Implementation

The PTX decode kernel implements the core GDN operation with embedded assembly optimizations:

```mermaid
flowchart TD
Start([Kernel Entry]) --> LoadGates[Load Gate Parameters]
LoadGates --> LoadQK[Load Q and K with Vectorized Access]
LoadQK --> LoadV[Load V Slice]
LoadV --> LoadState[Load FP8 State with Dynamic Scaling]
LoadState --> AsyncPrefetch[Async State Prefetch]
AsyncPrefetch --> ApplyDecay[Apply Gate Decay]
ApplyDecay --> ComputeOldV[Compute old_v = S @ k]
ComputeOldV --> RankUpdate[Rank-1 Update: S += δ * k^T]
RankUpdate --> ComputeOutput[Compute output = scale * S @ q]
ComputeOutput --> StoreOutput[Store Results]
StoreOutput --> QuantizeState[Quantize New State to FP8]
QuantizeState --> StoreState[Store FP8 State + Scale]
StoreState --> End([Kernel Exit])
```

**Diagram sources**
- [gdn_decode_ptx.cuh:248-413](file://src/kernels/ptx/gdn_decode_ptx.cuh#L248-L413)

### Prefill Kernel Implementation

The PTX prefill kernel extends the decode concept to handle variable-length sequences with chunked processing:

```mermaid
sequenceDiagram
participant Seq as Sequence Processing
participant Chunk as Chunk Processing
participant Gate as Gate Computation
participant State as State Management
participant Math as Mathematical Ops
Seq->>Chunk : Process Tokens in Chunks
Chunk->>Gate : Compute Gates (exp, sigmoid)
Gate->>Math : Fast Math Operations
Math-->>Gate : Optimized Results
Gate->>State : Apply Gate Decay
State->>State : Rank-1 Updates
State->>Math : Dot Products with FMA
Math-->>State : Optimized Results
State-->>Chunk : Updated State
Chunk-->>Seq : Process Next Chunk
```

**Diagram sources**
- [gdn_prefill_ptx.cuh:121-301](file://src/kernels/ptx/gdn_prefill_ptx.cuh#L121-L301)

**Section sources**
- [gdn_decode_ptx.cuh:248-491](file://src/kernels/ptx/gdn_decode_ptx.cuh#L248-L491)
- [gdn_prefill_ptx.cuh:121-358](file://src/kernels/ptx/gdn_prefill_ptx.cuh#L121-L358)

## Inline Assembly Primitives

### Mathematical Operations

The framework provides optimized mathematical functions through PTX assembly:

| Operation | PTX Instruction | Performance Benefit |
|-----------|----------------|---------------------|
| Fast Exponential | `ex2.approx.f32` | ~2-3x faster than libm |
| Fast Logarithm | `lg2.approx.f32` | ~2-3x faster than libm |
| Fast Reciprocal | `rcp.approx.f32` | ~2-3x faster than libm |
| Fused Multiply-Add | `fma.rn.f32` | Single rounding, better precision |

### FP8 State Quantization Primitives

**Updated** New FP8 conversion and memory operations for state quantization:

```mermaid
classDiagram
class FP8Primitives {
+ptx_fp32_to_fp8(val) __nv_fp8_e4m3
+ptx_fp8_to_fp32(val) float
+ptx_pack_fp8x4(a,b,c,d) uint32_t
+ptx_unpack_fp8x4(packed) (__nv_fp8_e4m3[4])
+ptx_ld_nc_u32(ptr) uint32_t
}
class QuantizationOps {
+Per-Row Dynamic Scaling
+FP8 E4M3 Format
+4x Memory Compression
+Range [-448, 448]
}
FP8Primitives --> QuantizationOps : "implements"
```

**Diagram sources**
- [gdn_decode_ptx.cuh:198-242](file://src/kernels/ptx/gdn_decode_ptx.cuh#L198-L242)

### Memory Operations

Advanced memory access patterns with cache control and FP8 vectorized operations:

```mermaid
classDiagram
class MemoryPrimitives {
+ptx_ld_nc(ptr) float
+ptx_st_wb(ptr, val) void
+ptx_ld_nc_v4(dst, ptr) void
+ptx_st_wb_v4(ptr, val) void
+ptx_ld_nc_u32(ptr) uint32_t
+ptx_cp_async_ca(smem, gmem) void
+ptx_cp_async_cg(smem, gmem) void
}
class CacheControl {
+NonCoherentLoad : bypass L1 cache
+WriteBackStore : optimize cache policy
+VectorizedAccess : 4-float operations
+FP8 Vectorized : 4x FP8 operations
+AsyncPrefetch : overlapping transfers
}
MemoryPrimitives --> CacheControl : "implements"
```

**Diagram sources**
- [gdn_decode_ptx.cuh:98-174](file://src/kernels/ptx/gdn_decode_ptx.cuh#L98-L174)

### Warp-Level Operations

Warp shuffle operations for efficient parallel reductions:

```mermaid
flowchart LR
Input[Thread Input] --> Shuffle1[Shuffle XOR 16]
Shuffle1 --> Shuffle2[Shuffle XOR 8]
Shuffle2 --> Shuffle3[Shuffle XOR 4]
Shuffle3 --> Shuffle4[Shuffle XOR 2]
Shuffle4 --> Shuffle5[Shuffle XOR 1]
Shuffle5 --> Output[Reduction Result]
```

**Diagram sources**
- [gdn_decode_ptx.cuh:227-235](file://src/kernels/ptx/gdn_decode_ptx.cuh#L227-L235)

**Section sources**
- [gdn_decode_ptx.cuh:32-190](file://src/kernels/ptx/gdn_decode_ptx.cuh#L32-L190)
- [gdn_prefill_ptx.cuh:34-80](file://src/kernels/ptx/gdn_prefill_ptx.cuh#L34-L80)

## Performance Optimization Strategies

### Compute Density Enhancement

The framework employs several strategies to increase arithmetic intensity:

| Strategy | Implementation | Impact |
|----------|----------------|--------|
| Chunked Processing | Process multiple tokens per iteration | AI increases from 1 to 8 FLOP/byte |
| Vectorized Loads | 4-float memory operations | 4x memory throughput |
| FMA Chains | Single instruction for multiply-add | Reduced instruction count |
| Async Prefetch | Overlap computation with memory | Hidden latency |
| **FP8 Quantization** | **4x memory compression** | **Reduced bandwidth usage** |

### Memory Bandwidth Optimization

Advanced cache management techniques with FP8 memory efficiency:

```mermaid
graph LR
subgraph "Memory Hierarchy"
Global[Global Memory] --> NC[L2 Bypass Load]
NC --> SMEM[Shared Memory]
SMEM --> Registers[Registers]
end
subgraph "Optimization Techniques"
NC -.-> Bypass[Bypass L1 Cache]
SMEM -.-> BankConflict[Bank Conflict Avoidance]
Registers -.-> Coalesced[Coalesced Access]
end
Global -.-> WB[Write-Back Stores]
WB --> Global
subgraph "FP8 Memory Benefits"
FP8[FP8 State Storage] --> Compressed[4x Memory Usage]
Compressed --> Bandwidth[Reduced Bandwidth]
Bandwidth --> Performance[Improved Performance]
end
Global -.-> FP8
```

**Diagram sources**
- [gdn_decode_ptx.cuh:113-174](file://src/kernels/ptx/gdn_decode_ptx.cuh#L113-L174)

**Section sources**
- [README.md:14-51](file://README.md#L14-L51)
- [OPTIMIZATION_LOG.md:116-131](file://docs/OPTIMIZATION_LOG.md#L116-L131)

## Build System and Integration

### Compilation Architecture

The build system supports multiple optimization targets and deployment scenarios:

```mermaid
flowchart TD
Source[Kernels Source] --> Combine[Source Combination]
Combine --> Compile[NVCC Compilation]
Compile --> Optimize[Optimization Flags]
Optimize --> Link[Link Libraries]
Link --> Package[Package Library]
subgraph "Compilation Targets"
PTX[PTX Assembly]
CuTe[CuTe Templates]
Triton[Triton Baseline]
FP8[FP8 Quantization]
end
Source --> PTX
Source --> CuTe
Source --> Triton
Source --> FP8
```

**Diagram sources**
- [build_cuda.py:332-373](file://scripts/build_cuda.py#L332-L373)

### Integration Patterns

The framework integrates seamlessly with the broader kernel ecosystem:

| Integration Point | Purpose | Implementation |
|------------------|---------|----------------|
| C++ Wrapper | Python FFI Access | Extern C functions |
| CUDA Runtime | GPU Execution | Standard CUDA launch |
| Memory Management | Buffer Allocation | Unified memory model |
| Stream Support | Asynchronous Execution | CUDA streams |
| **FP8 Support** | **Quantized State Storage** | **Dynamic scaling + packing** |

**Section sources**
- [CMakeLists.txt:1-68](file://CMakeLists.txt#L1-L68)
- [gdn_kernels.cu:25-170](file://src/gdn_kernels.cu#L25-L170)

## Benchmarking and Evaluation

### Performance Metrics

The framework provides comprehensive performance evaluation capabilities:

| Metric | Measurement | Significance |
|--------|-------------|--------------|
| Throughput | GB/s achieved | Memory bandwidth utilization |
| Latency | ms per operation | Kernel launch overhead |
| Utilization | % of peak | Hardware resource usage |
| Speedup | vs baseline | Optimization effectiveness |
| **Memory Usage** | **GB allocated** | **FP8 compression benefits** |

### Benchmarking Infrastructure

Automated benchmarking supports multiple scenarios:

```mermaid
sequenceDiagram
participant Script as Benchmark Script
participant Kernel as PTX Kernel
participant GPU as GPU Device
participant Metrics as Performance Metrics
Script->>Kernel : Configure Parameters
Kernel->>GPU : Launch Kernel
GPU->>Kernel : Execute PTX Instructions
Kernel->>Metrics : Collect Statistics
Metrics->>Script : Report Results
Script->>Script : Analyze Performance
```

**Diagram sources**
- [bench_all_versions.py:38-444](file://scripts/bench_all_versions.py#L38-L444)

**Section sources**
- [bench_all_versions.py:38-444](file://scripts/bench_all_versions.py#L38-L444)
- [bench_modal.py:115-330](file://benchmarks/bench_modal.py#L115-L330)

## Optimization Roadmap

### Current Status and Future Directions

The PTX framework represents a crucial component in the optimization journey:

```mermaid
timeline
title PTX Framework Evolution
2026-03-28 : Initial Implementation
: Basic PTX primitives
: Decode kernel support
2026-03-28 : Async Prefetch Integration
: cp.async optimization
: Memory latency reduction
2026-04-24 : Performance Validation
: Modal B200 benchmarking
: Optimization verification
2026-05-24 : Advanced Features
: Enhanced chunking
: Multi-warp optimizations
2026-06-15 : **FP8 State Quantization**
: **4x memory compression**
: **Per-row dynamic scaling**
: **Vectorized FP8 operations**
```

### Strategic Priorities

The framework development follows a structured approach:

1. **Foundation Stability**: Ensure reliable PTX assembly implementation
2. **Performance Validation**: Comprehensive benchmarking across scenarios
3. **Integration Enhancement**: Seamless cooperation with higher-level frameworks
4. **Advanced Optimizations**: Explore additional PTX instruction opportunities
5. **Memory Efficiency**: FP8 quantization for reduced bandwidth usage

**Section sources**
- [ROADMAP.md:1-180](file://docs/ROADMAP.md#L1-L180)
- [OPTIMIZATION_LOG.md:1-197](file://docs/OPTIMIZATION_LOG.md#L1-L197)

## Implementation Details

### Template-Based Design

The framework utilizes C++ templates for compile-time optimization:

```mermaid
classDiagram
class PTXKernelTemplate {
<<template>>
+BLOCK_V : int
+decode_kernel_ptx() kernel
+decode_kernel_ptx_fp8() kernel
+prefill_kernel_ptx() kernel
}
class PrimitiveTemplates {
<<template>>
+ptx_cp_async_wait~N~() void
+ptx_fma~a,b,c~() float
+ptx_shfl_xor~val,lane~() float
+ptx_fp32_to_fp8~val~() __nv_fp8_e4m3
+ptx_pack_fp8x4~a,b,c,d~() uint32_t
}
PTXKernelTemplate --> PrimitiveTemplates : "uses"
```

**Diagram sources**
- [gdn_decode_ptx.cuh:248-488](file://src/kernels/ptx/gdn_decode_ptx.cuh#L248-L488)

### Memory Layout Optimization

Sophisticated memory management for optimal performance with FP8 support:

| Memory Region | Purpose | Allocation Strategy |
|---------------|---------|-------------------|
| Q/K Buffers | Query and Key data | Vectorized loads |
| V Slices | Value projections | Coalesced access |
| State Tiles | FP32 recurrent state | Async prefetch |
| **FP8 State** | **Quantized state tiles** | **Vectorized FP8 loads** |
| **Scale Arrays** | **Per-row scaling factors** | **Vectorized loads** |
| Scratch Space | Temporary computations | Shared memory |

**Updated** Added FP8 state and per-row scale memory layouts for quantized state storage.

### FP8 State Quantization Implementation

**New** Detailed FP8 quantization process for memory-efficient state storage:

```mermaid
flowchart TD
Start([FP8 Quantization]) --> ComputeMax[Compute Max Absolute Value]
ComputeMax --> CalculateScale[Calculate Per-Row Scale]
CalculateScale --> Normalize[Normalize by Scale]
Normalize --> Quantize[Quantize to FP8 E4M3]
Quantize --> Pack[Pack 4 FP8 Values]
Pack --> Store[Store Packed Data]
Store --> End([Complete])
```

**Diagram sources**
- [gdn_decode_ptx.cuh:636-669](file://src/kernels/ptx/gdn_decode_ptx.cuh#L636-L669)

**Section sources**
- [gdn_decode_ptx.cuh:285-292](file://src/kernels/ptx/gdn_decode_ptx.cuh#L285-L292)
- [gdn_prefill_ptx.cuh:167-177](file://src/kernels/ptx/gdn_prefill_ptx.cuh#L167-L177)

## Conclusion

The PTX Inline Assembly Kernel Framework stands as the pinnacle of optimization within the GDN kernel ecosystem. By providing direct control over GPU operations through PTX assembly, it enables unprecedented performance gains while serving as a critical fallback mechanism for extreme optimization scenarios.

**Updated** The framework now includes comprehensive FP8 state quantization support, providing 4x memory compression through per-row dynamic scaling and vectorized FP8 operations. This enhancement maintains computational accuracy while significantly reducing memory bandwidth requirements, making it particularly valuable for memory-bound scenarios where every optimization counts toward maximizing hardware utilization.

The framework's strength lies in its comprehensive approach to GPU optimization, combining advanced mathematical operations, sophisticated memory management, warp-level parallelism, and efficient state quantization. Its integration with the broader kernel ecosystem ensures that performance optimizations are systematically applied across all layers, from high-level Triton implementations to the most granular PTX assembly optimizations.

As the framework continues to evolve, it maintains its position as the essential foundation for achieving peak performance in GDN kernel implementations, particularly in memory-bound scenarios where FP8 quantization provides substantial bandwidth savings while preserving numerical accuracy through careful dynamic scaling strategies.
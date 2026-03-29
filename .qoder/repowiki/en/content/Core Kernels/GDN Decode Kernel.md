# GDN Decode Kernel

<cite>
**Referenced Files in This Document**
- [kernel.py](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py)
- [kernel.py](file://gdn_decode_qk4_v8_d128_k_last/baseline/triton/kernel.py)
- [kernel_v2.py](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel_v2.py)
- [kernel_v3.py](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel_v3.py)
- [config.toml](file://gdn_decode_qk4_v8_d128_k_last/config.toml)
- [gdn_decode_qk4_v8_d128_k_last.json](file://flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json)
- [OPTIMIZATION_LOG.md](file://docs/OPTIMIZATION_LOG.md)
- [ROADMAP.md](file://docs/ROADMAP.md)
- [ROOFLINE.md](file://docs/ROOFLINE.md)
- [bench_modal.py](file://benchmarks/bench_modal.py)
- [gdn_decode_ptx.cuh](file://src/kernels/ptx/gdn_decode_ptx.cuh)
- [gdn_decode_v7.cuh](file://src/kernels/cuda/gdn_decode_v7.cuh)
- [gdn_decode_dsl.py](file://src/kernels/cute_dsl/gdn_decode_dsl.py)
- [gdn_decode_dsl_optimized.py](file://src/kernels/cute_dsl/gdn_decode_dsl_optimized.py)
- [gdn_prefill_dsl.py](file://src/kernels/cute_dsl/gdn_prefill_dsl.py)
- [README.md](file://src/kernels/cute_dsl/README.md)
- [gdn_decode_v10.cuh](file://src/kernels/cute_cpp/gdn_decode_v10.cuh)
- [README.md](file://src/kernels/cute_cpp/README.md)
- [bench_cute_dsl_vs_cpp.py](file://scripts/bench_cute_dsl_vs_cpp.py)
- [bench_cute_vs_triton.py](file://scripts/bench_cute_vs_triton.py)
</cite>

## Update Summary
**Changes Made**
- Enhanced kernel documentation with new cp.async prefetch capabilities across all kernel implementations
- Updated performance characteristics to include memory latency hiding benefits through async memory loading
- Expanded optimization strategies to cover the new async memory loading approach in PTX, CUDA, and CuTe implementations
- Added detailed coverage of cp.async primitive implementations and their performance implications
- Updated kernel comparison matrix to reflect the latest optimization techniques including async prefetch

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Kernel Technology Stack Comparison](#kernel-technology-stack-comparison)
7. [Dependency Analysis](#dependency-analysis)
8. [Performance Considerations](#performance-considerations)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Conclusion](#conclusion)

## Introduction
This document explains the GDN Decode Kernel implementation for single-token autoregressive inference, covering the mathematical foundation of the gated delta net (GDN) mechanism, the V-dimension splitting strategy that distributes the output dimension across four parallel programs for improved SM occupancy, and the algorithmic steps: decay gate computation using sigmoid activation, update gate calculation with exponential functions, state evolution through delta rule rank-1 updates, and output projection with scaling factors. It also documents the grouped value attention (GVA) mechanism where two V-heads share each Q/K head, and provides concrete examples from the Triton kernel showing memory access patterns, register blocking strategies, and thread block organization. The document now includes comprehensive coverage of PTX inline assembly kernels with embedded PTX primitives, CuTe DSL implementations, and enhanced kernel implementations featuring cp.async prefetch capabilities for memory latency hiding, comparing them against the baseline Triton solution to highlight performance optimizations achieved through different compilation technologies and memory coalescing strategies.

## Project Structure
The repository organizes the GDN decode kernel under a dedicated directory with separate solution and baseline implementations, a configuration file, and a trace definition that captures the operation's semantics and shapes. The implementation now includes multiple kernel technologies: Triton (with adaptive blocking), PTX inline assembly, CUDA C++ with TMA, and CuTe DSL variants, all enhanced with cp.async prefetch capabilities.

```mermaid
graph TB
A["Root"] --> B["gdn_decode_qk4_v8_d128_k_last/"]
B --> C["solution/triton/kernel.py"]
B --> C2["solution/triton/kernel_v2.py"]
B --> C3["solution/triton/kernel_v3.py"]
B --> D["baseline/triton/kernel.py"]
B --> E["config.toml"]
B --> F["scripts/pack_solution.py"]
A --> G["flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json"]
A --> H["docs/OPTIMIZATION_LOG.md"]
A --> I["docs/ROADMAP.md"]
A --> J["benchmarks/bench_modal.py"]
A --> K["src/kernels/"]
K --> L["ptx/gdn_decode_ptx.cuh"]
K --> M["cuda/gdn_decode_v7.cuh"]
J --> N["cute_dsl/"]
N --> O["gdn_decode_dsl.py"]
N --> P["gdn_decode_dsl_optimized.py"]
N --> Q["gdn_prefill_dsl.py"]
J --> R["cute_cpp/"]
R --> S["gdn_decode_v10.cuh"]
A --> T["scripts/"]
T --> U["bench_cute_dsl_vs_cpp.py"]
T --> V["bench_cute_vs_triton.py"]
```

**Diagram sources**
- [kernel.py:1-136](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L1-L136)
- [kernel_v2.py:1-122](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel_v2.py#L1-L122)
- [kernel_v3.py:1-130](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel_v3.py#L1-L130)
- [kernel.py:1-101](file://gdn_decode_qk4_v8_d128_k_last/baseline/triton/kernel.py#L1-L101)
- [config.toml:1-10](file://gdn_decode_qk4_v8_d128_k_last/config.toml#L1-L10)
- [gdn_decode_qk4_v8_d128_k_last.json:1-153](file://flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json#L1-L153)
- [OPTIMIZATION_LOG.md:138-179](file://docs/OPTIMIZATION_LOG.md#L138-L179)
- [ROADMAP.md:70-180](file://docs/ROADMAP.md#L70-L180)
- [bench_modal.py:1-330](file://benchmarks/bench_modal.py#L1-L330)
- [gdn_decode_ptx.cuh:1-543](file://src/kernels/ptx/gdn_decode_ptx.cuh#L1-L543)
- [gdn_decode_v7.cuh:160-359](file://src/kernels/cuda/gdn_decode_v7.cuh#L160-L359)
- [gdn_decode_dsl.py:1-442](file://src/kernels/cute_dsl/gdn_decode_dsl_optimized.py#L1-L442)
- [gdn_decode_v10.cuh:1-200](file://src/kernels/cute_cpp/gdn_decode_v10.cuh#L1-L200)

**Section sources**
- [config.toml:1-10](file://gdn_decode_qk4_v8_d128_k_last/config.toml#L1-L10)
- [gdn_decode_qk4_v8_d128_k_last.json:1-153](file://flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json#L1-L153)

## Core Components
- **Triton solution kernel**: Implements the GDN decode forward pass with autotuning, register blocking over V-dimension, and k-last state layout. Now includes adaptive BLOCK_V based on batch size for optimal SM occupancy.
- **Triton v2 kernel**: Fused delta-rule implementation with full state tile in registers, eliminating Python loop overhead and single HBM read/write per kernel launch.
- **Triton v3 kernel**: V-dimension splitting strategy with fixed BLOCK_V=32 across 4 parallel programs for improved SM occupancy.
- **Baseline Python kernel**: Reference implementation using PyTorch operations and GVA expansion.
- **PTX inline assembly kernels**: CUDA C++ kernels with embedded PTX assembly instructions for maximum control over low-level GPU operations, including warp shuffle, fast math, memory operations, and cp.async prefetch for memory latency hiding.
- **CUDA C++ kernels**: Enhanced with TMA (Tensor Memory Access) and cp.async bulk operations for 2D tile loads with memory latency hiding.
- **CuTe DSL kernels**: Python-based kernels using CUTLASS 4.0+ DSL with MLIR compilation pipeline, offering automatic optimization passes including async copy insertion.
- **CuTe C++ kernels**: Traditional C++ template-based implementations with manual optimization and NVCC compilation, now featuring cp.async prefetch capabilities.
- **Configuration**: Defines the solution metadata and build specification.
- **Trace definition**: Documents axes, constraints, inputs/outputs, and a reference implementation.
- **Optimization logs**: Detailed records of cp.async prefetch implementation across all kernel technologies.
- **Roadmap**: Strategic direction for kernel optimization including cp.async prefetch integration.
- **Benchmark runner**: Orchestrates benchmarking on Modal B200 and compares solution vs baseline across multiple kernel technologies.

**Section sources**
- [kernel.py:1-136](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L1-L136)
- [kernel_v2.py:1-122](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel_v2.py#L1-L122)
- [kernel_v3.py:1-130](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel_v3.py#L1-L130)
- [kernel.py:1-101](file://gdn_decode_qk4_v8_d128_k_last/baseline/triton/kernel.py#L1-L101)
- [gdn_decode_ptx.cuh:1-543](file://src/kernels/ptx/gdn_decode_ptx.cuh#L1-L543)
- [gdn_decode_v7.cuh:160-359](file://src/kernels/cuda/gdn_decode_v7.cuh#L160-L359)
- [gdn_decode_dsl.py:1-442](file://src/kernels/cute_dsl/gdn_decode_dsl_optimized.py#L1-L442)
- [gdn_decode_v10.cuh:1-200](file://src/kernels/cute_cpp/gdn_decode_v10.cuh#L1-L200)
- [config.toml:1-10](file://gdn_decode_qk4_v8_d128_k_last/config.toml#L1-L10)
- [gdn_decode_qk4_v8_d128_k_last.json:1-153](file://flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json#L1-L153)
- [OPTIMIZATION_LOG.md:138-179](file://docs/OPTIMIZATION_LOG.md#L138-L179)
- [ROADMAP.md:70-180](file://docs/ROADMAP.md#L70-L180)
- [bench_modal.py:1-330](file://benchmarks/bench_modal.py#L1-L330)

## Architecture Overview
The GDN decode kernel performs single-token generation with recurrent state updates. The solution kernel is organized as a Triton program with a grid of (B, H=8, V_BLOCKS) where each program handles a V-tile of size BLOCK_V and a single head. The kernel computes decay and update gates per head, applies a decay to the state, computes the old value, interpolates the new value, updates the state via a rank-1 delta rule, and produces the output by projecting with Q. The architecture now supports multiple kernel technologies, each with distinct compilation strategies and optimization approaches including cp.async prefetch for memory latency hiding.

```mermaid
sequenceDiagram
participant Host as "Host"
participant Kernel as "Multi-Tech Kernel"
participant SMem as "Shared Memory"
participant HBM as "HBM"
Host->>Kernel : Launch with grid (B, H=8, V_BLOCKS)
Kernel->>Kernel : Load per-head gates (a, dt_bias, A_log, b)
Kernel->>HBM : Load Q[K], K[K], V[V-tile]
Kernel->>HBM : Load State[B,H,V,K] slice [BLOCK_V x D]
Note over Kernel,HBM : cp.async prefetch begins
Kernel->>Kernel : Compute decay gate g and update gate beta
Kernel->>Kernel : Apply decay : S = g * S
Kernel->>Kernel : old_v = reduce(S * K)
Kernel->>Kernel : new_v = beta * V + (1 - beta) * old_v
Kernel->>Kernel : S = S + (new_v - old_v)^T @ K
Kernel->>Kernel : out = scale * (Q @ S)
Kernel->>HBM : Store Out[B,H,V-tile]
Kernel->>HBM : Store NewState[B,H,V,K]
```

**Diagram sources**
- [kernel.py:38-98](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L38-L98)
- [gdn_decode_ptx.cuh:205-378](file://src/kernels/ptx/gdn_decode_ptx.cuh#L205-L378)
- [gdn_decode_v7.cuh:204-359](file://src/kernels/cuda/gdn_decode_v7.cuh#L204-L359)
- [gdn_decode_dsl_optimized.py:54-286](file://src/kernels/cute_dsl/gdn_decode_dsl_optimized.py#L54-L286)

## Detailed Component Analysis

### Mathematical Foundation: Gated Delta Net (GDN)
- **Decay gate computation**: The decay gate g is computed per head using an exponential of the softplus of (a + dt_bias), modulated by A_log. This stabilizes and scales the decay rate.
- **Update gate calculation**: The update gate beta is computed via sigmoid of b, controlling the interpolation between old and new values.
- **State evolution**: The state S evolves by applying the decay gate, computing old_v as the projection of S onto K, interpolating new_v, and updating S via a rank-1 update using K and the delta (new_v - old_v).
- **Output projection**: The output is produced by projecting S with Q, scaled by a normalization factor.

These steps are implemented across multiple kernel technologies, with the Triton version fusing operations and using register blocking for improved throughput, while PTX kernels leverage inline assembly for maximum performance and CuTe DSL provides automatic optimization through MLIR passes.

**Section sources**
- [kernel.py:61-91](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L61-L91)
- [kernel.py:55-94](file://gdn_decode_qk4_v8_d128_k_last/baseline/triton/kernel.py#L55-L94)
- [gdn_decode_ptx.cuh:250-345](file://src/kernels/ptx/gdn_decode_ptx.cuh#L250-L345)
- [gdn_decode_dsl_optimized.py:107-185](file://src/kernels/cute_dsl/gdn_decode_dsl_optimized.py#L107-L185)

### V-Dimension Splitting Strategy and Parallel Programs
- **Grid organization**: The kernel uses a grid of (B, H=8, V_BLOCKS) where each program handles a V-tile of size BLOCK_V and a single head. This splits the V dimension across four parallel programs for improved SM occupancy.
- **Register blocking**: BLOCK_V is autotuned across {16, 32, 64, 128} with varying num_warps to balance register pressure and occupancy. The solution sets BLOCK_V=32 and num_warps=4 for a fixed configuration in the wrapper.
- **Independence**: Each V-slice is independent, enabling correctness when executed in parallel.

```mermaid
flowchart TD
Start(["Launch grid (B, H=8, V_BLOCKS)"]) --> SelectTile["Program selects V-tile [v0:v0+BLOCK_V]"]
SelectTile --> LoadGates["Load per-head gates (a, dt_bias, A_log, b)"]
LoadGates --> LoadQKV["Load Q[K], K[K], V[V-tile]"]
LoadQKV --> LoadState["Load State slice [BLOCK_V x D]"]
LoadState --> StartPrefetch["Start cp.async prefetch"]
StartPrefetch --> Decay["Apply decay: S = g * S"]
Decay --> OldV["Compute old_v = reduce(S * K)"]
OldV --> Interpolate["Interpolate new_v = beta * V + (1 - beta) * old_v"]
Interpolate --> Rank1["Rank-1 update: S += (new_v - old_v)^T @ K"]
Rank1 --> Output["Output = scale * (Q @ S)"]
Output --> StoreOut["Store Out[B,H,V-tile]"]
StoreOut --> StoreState["Store NewState[B,H,V,K]"]
StoreState --> End(["Done"])
```

**Diagram sources**
- [kernel.py:55-97](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L55-L97)
- [gdn_decode_ptx.cuh:288-378](file://src/kernels/ptx/gdn_decode_ptx.cuh#L288-L378)
- [gdn_decode_v7.cuh:283-359](file://src/kernels/cuda/gdn_decode_v7.cuh#L283-L359)
- [gdn_decode_dsl_optimized.py:261-286](file://src/kernels/cute_dsl/gdn_decode_dsl_optimized.py#L261-L286)

**Section sources**
- [kernel.py:5-13](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L5-L13)
- [kernel_v3.py:5-15](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel_v3.py#L5-L15)
- [kernel_v2.py:5-15](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel_v2.py#L5-L15)
- [kernel.py:105-136](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L105-L136)
- [gdn_decode_ptx.cuh:205-249](file://src/kernels/ptx/gdn_decode_ptx.cuh#L205-L249)
- [gdn_decode_dsl_optimized.py:77-106](file://src/kernels/cute_dsl/gdn_decode_dsl_optimized.py#L77-L106)

### Grouped Value Attention (GVA) Mechanism
- **Head configuration**: num_q_heads=4, num_k_heads=4, num_v_heads=8. Two V-heads share each Q/K head (qk_h = h // 2).
- **Expansion**: The kernel derives the Q/K head index for each V-head and loads the corresponding Q/K slices accordingly.

This ensures that the attention computation aligns with the GVA topology while maintaining efficient memory access patterns across all kernel implementations.

**Section sources**
- [kernel.py:12-13](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L12-L13)
- [kernel_v3.py:44](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel_v3.py#L44)
- [kernel_v2.py:42](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel_v2.py#L42)
- [gdn_decode_ptx.cuh:235](file://src/kernels/ptx/gdn_decode_ptx.cuh#L235)
- [gdn_decode_dsl_optimized.py:86](file://src/kernels/cute_dsl/gdn_decode_dsl_optimized.py#L86)

### Algorithm Steps in Detail
- **Gates**:
  - Decay gate g: computed from A_log and softplus(a + dt_bias).
  - Update gate beta: computed from sigmoid(b).
- **State evolution**:
  - Decay: S = g * S.
  - old_v: matrix-vector multiply of S and K.
  - new_v: interpolate between beta * V and (1 - beta) * old_v.
  - Rank-1 update: S += (new_v - old_v)^T @ K.
- **Output projection**:
  - out = scale * (Q @ S).

These steps are fused within a single kernel per (B, H, V-tile) across all implementations to minimize synchronization overhead and maximize throughput.

**Section sources**
- [kernel.py:61-91](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L61-L91)
- [gdn_decode_ptx.cuh:309-363](file://src/kernels/ptx/gdn_decode_ptx.cuh#L309-L363)
- [gdn_decode_dsl_optimized.py:134-185](file://src/kernels/cute_dsl/gdn_decode_dsl_optimized.py#L134-L185)

### Memory Access Patterns and Thread Block Organization
- **State layout**: k-last [B, H, V=128, K=128] float32. The kernel loads a [BLOCK_V x D] slice of the state and stores the updated slice back.
- **Access patterns**:
  - Coalesced loads for Q[K], K[K], V[V-tile] along contiguous dimensions.
  - Coalesced stores for Out[B,H,V-tile] and NewState[B,H,V,K].
- **Thread block organization**:
  - Grid: (B, H=8, V_BLOCKS) with BLOCK_V tiles over V.
  - Registers: per-program scalars for gates and per-thread vectors for Q, K, V, and partial reductions.

These patterns enable efficient HBM bandwidth utilization and register reuse across all kernel implementations.

**Section sources**
- [kernel.py:46-50](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L46-L50)
- [kernel_v3.py:66-70](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel_v3.py#L66-L70)
- [gdn_decode_ptx.cuh:242-250](file://src/kernels/ptx/gdn_decode_ptx.cuh#L242-L250)
- [gdn_decode_dsl_optimized.py:220-235](file://src/kernels/cute_dsl/gdn_decode_dsl_optimized.py#L220-L235)

### k-Last State Layout [B, H, V=128, K=128]
- The state is stored in k-last layout [B, H, V, K] to support efficient coalesced memory access patterns during the decode phase.
- The kernel reads and writes state slices aligned with the V-tile, enabling persistent state across tokens with minimal overhead.

**Section sources**
- [gdn_decode_qk4_v8_d128_k_last.json:80-89](file://flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json#L80-L89)
- [kernel.py:13](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L13)
- [gdn_decode_ptx.cuh:14](file://src/kernels/ptx/gdn_decode_ptx.cuh#L14)
- [gdn_decode_dsl_optimized.py:98](file://src/kernels/cute_dsl/gdn_decode_dsl_optimized.py#L98)

### cp.Async Prefetch Implementation Across Technologies
**Enhanced** All kernel implementations now feature cp.async prefetch capabilities for memory latency hiding:

- **PTX Inline Assembly**: Implements `ptx_cp_async_ca` and `ptx_cp_async_cg` functions for async memory copying with commit/wait operations.
- **CUDA C++**: Uses `cp.async.bulk.tensor.2d` for coalesced 2D tile loads with mbarrier completion.
- **CuTe DSL**: Automatic insertion of async copy operations through MLIR optimization passes.
- **Triton**: While Triton doesn't directly expose cp.async primitives, the memory access patterns are optimized for latency hiding through coalesced access and register blocking.

**Section sources**
- [gdn_decode_ptx.cuh:112-176](file://src/kernels/ptx/gdn_decode_ptx.cuh#L112-L176)
- [gdn_decode_v7.cuh:163-186](file://src/kernels/cuda/gdn_decode_v7.cuh#L163-L186)
- [OPTIMIZATION_LOG.md:138-179](file://docs/OPTIMIZATION_LOG.md#L138-L179)

### Comparison Against Baseline
- **Baseline (Python)**: Uses PyTorch operations with explicit GVA expansion and k-first layout conversion. It demonstrates the algorithmic steps and serves as a correctness reference.
- **Solution (Triton)**: Fuses all per-head operations into a single kernel, uses register blocking over V, and maintains k-last state layout. This reduces kernel launch overhead, improves memory coalescing, and leverages Triton JIT compilation for performance.
- **PTX Inline Assembly**: Provides maximum performance through embedded PTX instructions for warp shuffle, fast math, memory operations with cache hints, and cp.async prefetch for memory latency hiding.
- **CUDA C++**: Enhanced with TMA and cp.async bulk operations for 2D tile loads with memory latency hiding, achieving significant performance improvements.
- **CuTe DSL**: Offers automatic optimization through MLIR compilation pipeline with vectorization, shared memory optimization, warp specialization, and async copy insertion.

Benchmarking on Modal B200 compares the solution against the baseline and reports latency, reference latency, speedup, and correctness metrics across all kernel technologies.

**Section sources**
- [kernel.py:1-101](file://gdn_decode_qk4_v8_d128_k_last/baseline/triton/kernel.py#L1-L101)
- [bench_modal.py:202-307](file://benchmarks/bench_modal.py#L202-L307)
- [gdn_decode_ptx.cuh:400-453](file://src/kernels/ptx/gdn_decode_ptx.cuh#L400-L453)
- [gdn_decode_dsl_optimized.py:289-377](file://src/kernels/cute_dsl/gdn_decode_dsl_optimized.py#L289-L377)

## Kernel Technology Stack Comparison

### PTX Inline Assembly Kernels
PTX kernels provide the highest level of hardware control through embedded assembly instructions:

**Key Features:**
- **Warp Shuffle Operations**: `shfl.sync.bfly.b32` for warp-level reductions without shared memory
- **Fast Math Approximations**: `ex2.approx.f32`, `lg2.approx.f32`, `rcp.approx.f32` for 2-3x speedup
- **Fused Multiply-Add**: `fma.rn.f32` with single rounding for better precision
- **Cache Control**: `ld.global.nc`, `st.global.wb` for L1/L2 bypass and write-back optimization
- **Predicated Execution**: `selp.f32` for branchless conditional operations
- **cp.async Prefetch**: `ptx_cp_async_ca`, `ptx_cp_async_cg` for memory latency hiding

**Performance Benefits:**
- Maximum performance extraction (~100% of theoretical limits)
- Custom cache behavior for streaming workloads
- Warp-level primitives for efficient reductions
- Branchless execution for better occupancy
- Memory latency hiding through async prefetch

**Section sources**
- [README.md:52-179](file://src/kernels/ptx/README.md#L52-L179)
- [gdn_decode_ptx.cuh:31-147](file://src/kernels/ptx/gdn_decode_ptx.cuh#L31-L147)
- [gdn_decode_ptx.cuh:112-176](file://src/kernels/ptx/gdn_decode_ptx.cuh#L112-L176)
- [gdn_prefill_ptx.cuh:34-80](file://src/kernels/ptx/gdn_prefill_ptx.cuh#L34-L80)

### CUDA C++ Kernels with TMA and cp.async
CUDA C++ kernels now feature advanced memory management with TMA and cp.async prefetch:

**Key Features:**
- **TMA (Tensor Memory Access)**: `cp.async.bulk.tensor.2d` for coalesced 2D tile loads
- **Memory Barriers**: `mbarrier::complete_tx::bytes` for synchronization
- **cp.async Bulk Operations**: `cp.async.ca.shared.global` for async prefetch
- **Vectorized Operations**: `float4` loads/stores for 16-byte aligned access
- **Warp Specialization**: Producer/consumer warps for optimal pipeline utilization

**Performance Benefits:**
- Significant memory bandwidth utilization through TMA
- Memory latency hiding with async prefetch
- Reduced synchronization overhead
- Improved memory coalescing patterns

**Section sources**
- [README.md:141-179](file://src/kernels/ptx/README.md#L141-L179)
- [gdn_decode_v7.cuh:163-186](file://src/kernels/cuda/gdn_decode_v7.cuh#L163-L186)
- [gdn_decode_v7.cuh:283-359](file://src/kernels/cuda/gdn_decode_v7.cuh#L283-L359)
- [OPTIMIZATION_LOG.md:138-179](file://docs/OPTIMIZATION_LOG.md#L138-L179)

### CuTe DSL Kernels
CuTe DSL provides high-level Python interface with automatic MLIR optimization:

**Compilation Pipeline:**
```
Python DSL → MLIR Dialects → LLVM IR → PTX → SASS
```

**Automatic Optimizations:**
- **TileAndFuse**: Loop fusion and tiling
- **VectorizeSmem**: Shared memory vectorization (float4 equivalent)
- **SwizzleElimination**: Bank conflict elimination
- **AsyncCopyInsertion**: TMA/cp.async instruction insertion
- **WarpSpecialization**: Automatic warp specialization
- **RegisterAllocation**: Optimized register scheduling

**Kernel Variants:**
- **Simplified DSL**: Basic State @ Q computation for demonstration
- **Optimized DSL**: Full delta rule with SMEM staging and vectorization
- **Prefill DSL**: Chunk-based processing for compute density optimization

**Performance Characteristics:**
- Development efficiency: High (Python-based)
- Optimization level: ~95-100% of hand-optimized C++
- Compilation time: Seconds (JIT compilation)
- Typical performance: 25-40 GB/s on B200
- Memory latency hiding: Through automatic async copy insertion

**Section sources**
- [README.md:1-188](file://src/kernels/cute_dsl/README.md#L1-L188)
- [gdn_decode_dsl.py:1-442](file://src/kernels/cute_dsl/gdn_decode_dsl_optimized.py#L1-L442)
- [gdn_decode_dsl_optimized.py:1-442](file://src/kernels/cute_dsl/gdn_decode_dsl_optimized.py#L1-L442)
- [gdn_prefill_dsl.py:1-323](file://src/kernels/cute_dsl/gdn_prefill_dsl.py#L1-L323)

### CuTe C++ Kernels
Traditional C++ template-based approach with manual optimization:

**Key Features:**
- **Layout Algebra**: `Layout<Shape, Stride>` for declarative tensor layouts
- **Swizzle System**: `Swizzle<3,3,3>` for bank conflict elimination
- **TMA Abstraction**: Simplified asynchronous memory operations
- **WGMMA Support**: Tensor Core operation abstraction
- **cp.async Prefetch**: Manual implementation for memory latency hiding

**Version Evolution:**
- **v9**: Manual XOR swizzle implementation
- **v10**: CuTe `Swizzle<3,3,3>` with automatic optimization and cp.async prefetch
- **Performance**: 25-40 GB/s on B200, with v10 showing slight improvements

**Development Trade-offs:**
- **Compilation Time**: Minutes (AOT compilation)
- **Code Complexity**: Moderate (C++ templates)
- **Maintenance**: Better than raw CUDA
- **Performance**: ~100% of optimized hand-written code
- **Memory Latency Hiding**: Through cp.async prefetch implementation

**Section sources**
- [README.md:1-142](file://src/kernels/cute_cpp/README.md#L1-L142)
- [gdn_decode_v10.cuh:1-200](file://src/kernels/cute_cpp/gdn_decode_v10.cuh#L1-L200)

### Triton Kernels
High-level Python kernel with JIT compilation:

**Key Features:**
- **Auto-tuning**: Automatic selection of optimal BLOCK_V and num_warps
- **Memory coalescing**: Built-in support for coalesced access patterns
- **Broadcasting**: Automatic broadcasting for gate computation
- **Integration**: Seamless PyTorch integration
- **Adaptive Blocking**: BLOCK_V varies based on batch size for optimal occupancy

**Performance Characteristics:**
- **Adaptive BLOCK_V**: 16 for B≤16, 32 for B≤128, 64 for B>128
- **Typical Performance**: 24-40 GB/s on B200
- **Development Efficiency**: Very high (Python-based)
- **Memory Latency Hiding**: Through coalesced access patterns and register blocking

**Section sources**
- [kernel.py:85-136](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L85-L136)
- [kernel_v2.py:82-122](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel_v2.py#L82-L122)
- [kernel_v3.py:86-130](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel_v3.py#L86-L130)

## Dependency Analysis
The solution kernel depends on Triton for JIT compilation and CUDA execution, and on PyTorch for tensor creation and shape handling in the wrapper. The baseline kernel depends on PyTorch for all computations. The PTX implementation requires CUDA runtime and supports embedded assembly instructions including cp.async prefetch. The CUDA C++ implementation requires TMA support and modern CUDA toolkit. The CuTe DSL implementation requires CUTLASS 4.0+ with MLIR support. The configuration ties the solution to the trace definition and build specification.

```mermaid
graph TB
A["solution/triton/kernel.py"] --> B["Triton JIT/CUDA"]
A --> C["PyTorch (wrapper)"]
D["baseline/triton/kernel.py"] --> E["PyTorch"]
F["config.toml"] --> G["BuildSpec"]
G --> A
H["gdn_decode_qk4_v8_d128_k_last.json"] --> A
H --> D
I["gdn_decode_ptx.cuh"] --> J["CUDA Runtime"]
I --> K["Embedded PTX Assembly"]
I --> L["cp.async Prefetch"]
M["gdn_decode_v7.cuh"] --> N["CUDA Toolkit + TMA"]
M --> O["cp.async Bulk Operations"]
P["gdn_decode_dsl.py"] --> Q["CuTe DSL (MLIR)"]
R["gdn_decode_dsl_optimized.py"] --> Q
S["gdn_decode_v10.cuh"] --> T["CuTe C++ (NVCC)"]
S --> U["Manual cp.async Implementation"]
```

**Diagram sources**
- [kernel.py:16-20](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L16-L20)
- [kernel.py:23-24](file://gdn_decode_qk4_v8_d128_k_last/baseline/triton/kernel.py#L23-L24)
- [config.toml:6-9](file://gdn_decode_qk4_v8_d128_k_last/config.toml#L6-L9)
- [gdn_decode_qk4_v8_d128_k_last.json:1-153](file://flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json#L1-L153)
- [gdn_decode_ptx.cuh:20-24](file://src/kernels/ptx/gdn_decode_ptx.cuh#L20-L24)
- [gdn_decode_v7.cuh:163-186](file://src/kernels/cuda/gdn_decode_v7.cuh#L163-L186)
- [gdn_decode_dsl.py:22-31](file://src/kernels/cute_dsl/gdn_decode_dsl.py#L22-L31)
- [gdn_decode_dsl_optimized.py:26-35](file://src/kernels/cute_dsl/gdn_decode_dsl_optimized.py#L26-L35)
- [gdn_decode_v10.cuh:1-200](file://src/kernels/cute_cpp/gdn_decode_v10.cuh#L1-L200)

**Section sources**
- [config.toml:1-10](file://gdn_decode_qk4_v8_d128_k_last/config.toml#L1-L10)
- [gdn_decode_qk4_v8_d128_k_last.json:1-153](file://flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json#L1-L153)

## Performance Considerations
- **Arithmetic intensity**: The decode kernel is extremely memory-bound with an estimated arithmetic intensity of approximately 1 FLOP/byte. Optimization focuses on maximizing HBM bandwidth utilization.
- **Optimization strategy**:
  - Fuse all per-head operations into a single kernel.
  - Tile over batch with grid (B, H) and register-blocking over V.
  - Keep state in registers/SMEM during the token update.
  - Coalesce HBM access for state [B, H, K, V] by using k-last layout and aligned access patterns.
  - Leverage technology-specific optimizations: PTX fast math, CuTe automatic optimization, Triton auto-tuning.
  - **Enhanced**: Implement cp.async prefetch for memory latency hiding across all kernel technologies.
- **Memory latency hiding**: All implementations now feature mechanisms to overlap memory transfers with computation, reducing stall time and improving overall throughput.

These strategies are validated by roofline analysis and implemented across all kernel implementations, with PTX kernels achieving the highest performance through embedded assembly and CuTe DSL providing near-optimal performance with automatic optimization.

**Section sources**
- [ROOFLINE.md:16-59](file://docs/ROOFLINE.md#L16-L59)
- [kernel.py:5-13](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L5-L13)
- [gdn_decode_ptx.cuh:112-176](file://src/kernels/ptx/gdn_decode_ptx.cuh#L112-L176)
- [gdn_decode_v7.cuh:163-186](file://src/kernels/cuda/gdn_decode_v7.cuh#L163-L186)
- [README.md:141-179](file://src/kernels/ptx/README.md#L141-L179)
- [README.md:93-117](file://src/kernels/cute_cpp/README.md#L93-L117)

## Troubleshooting Guide
- **Incorrect shapes or strides**: Ensure inputs match the documented shapes and that tensors are contiguous where required by the kernel wrapper.
- **State initialization**: If state is None, the kernel initializes zeros; otherwise, ensure the state layout is k-last [B, H, V, K].
- **Scaling factor**: If scale is None or zero, the kernel defaults to 1/sqrt(D).
- **GVA mismatch**: Verify that num_v_heads equals twice num_q_heads for the intended GVA sharing scheme.
- **PTX compatibility**: Ensure GPU architecture supports the PTX instructions used (requires compute capability 8.0+).
- **CuTe DSL installation**: Verify CUTLASS 4.0+ is installed with proper MLIR support.
- **Memory alignment**: PTX kernels require proper memory alignment for vectorized operations.
- **cp.async prefetch issues**: Ensure proper commit/wait sequences are used in CUDA/PTX implementations.
- **TMA compatibility**: Verify TMA support is available for CUDA kernels using cp.async bulk operations.

**Section sources**
- [gdn_decode_qk4_v8_d128_k_last.json:44-48](file://flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json#L44-L48)
- [kernel.py:108-109](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L108-L109)
- [kernel.py:117-123](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L117-L123)
- [README.md:151-163](file://src/kernels/ptx/README.md#L151-L163)
- [README.md:118-127](file://src/kernels/cute_cpp/README.md#L118-L127)

## Conclusion
The GDN Decode Kernel achieves significant performance improvements over the baseline Python implementation by fusing operations, using register blocking over the V-dimension, and leveraging multiple compilation technologies. The Triton solution provides excellent balance of performance and development efficiency, while PTX inline assembly kernels deliver maximum performance through embedded assembly instructions and cp.async prefetch for memory latency hiding. CUDA C++ kernels now feature TMA and advanced cp.async bulk operations for coalesced 2D tile loads. CuTe DSL offers automatic optimization with high development efficiency, and CuTe C++ provides traditional template-based optimization with manual cp.async prefetch implementation. The k-last state layout and GVA mechanism enable efficient state persistence and head sharing across all implementations, while autotuning identifies optimal configurations for BLOCK_V and num_warps. The enhanced cp.async prefetch capabilities across all kernel technologies provide substantial memory latency hiding benefits, with PTX kernels achieving the highest performance through embedded assembly and CuTe DSL providing near-optimal performance with significantly reduced development effort. The expanded technology stack now provides flexible deployment options depending on specific requirements for performance, development efficiency, and maintenance considerations.
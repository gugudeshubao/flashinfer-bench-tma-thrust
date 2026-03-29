# GDN Prefill Kernel

<cite>
**Referenced Files in This Document**
- [gdn_prefill_qk4_v8_d128_k_last/config.toml](file://gdn_prefill_qk4_v8_d128_k_last/config.toml)
- [gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py](file://gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py)
- [gdn_prefill_qk4_v8_d128_k_last/baseline/triton/kernel.py](file://gdn_prefill_qk4_v8_d128_k_last/baseline/triton/kernel.py)
- [gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py)
- [gdn_decode_qk4_v8_d128_k_last/baseline/triton/kernel.py](file://gdn_decode_qk4_v8_d128_k_last/baseline/triton/kernel.py)
- [flashinfer_trace/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json](file://flashinfer_trace/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json)
- [flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json](file://flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json)
- [benchmarks/bench_modal.py](file://benchmarks/bench_modal.py)
- [docs/PERFORMANCE.md](file://docs/PERFORMANCE.md)
- [scripts/debug_prefill.py](file://scripts/debug_prefill.py)
- [scripts/debug_prefill2.py](file://scripts/debug_prefill2.py)
- [src/kernels/ptx/gdn_prefill_ptx.cuh](file://src/kernels/ptx/gdn_prefill_ptx.cuh)
- [src/kernels/cuda/gdn_prefill_v6_chunked.cuh](file://src/kernels/cuda/gdn_prefill_v6_chunked.cuh)
- [src/kernels/ptx/README.md](file://src/kernels/ptx/README.md)
- [scripts/bench_prefill_all.py](file://scripts/bench_prefill_all.py)
- [src/kernels/cuda/gdn_prefill_v7.cuh](file://src/kernels/cuda/gdn_prefill_v7.cuh)
- [src/kernels/cuda/gdn_decode_v6.cuh](file://src/kernels/cuda/gdn_decode_v6.cuh)
- [src/kernels/cuda/gdn_decode_v7.cuh](file://src/kernels/cuda/gdn_decode_v7.cuh)
- [src/kernels/cute_cpp/gdn_prefill_v10.cuh](file://src/kernels/cute_cpp/gdn_prefill_v10.cuh)
</cite>

## Update Summary
**Changes Made**
- Added comprehensive TMA (Tensor Memory Accelerator) primitives and mma.sync Tensor Core support for Blackwell architecture
- Enhanced with 5 new TMA functions (ptx_mbarrier_init, ptx_mbarrier_arrive_tx, ptx_mbarrier_wait, ptx_tma_load_2d) and mma.sync.aligned.m16n8k16 primitives with BF16 matrix multiplication capabilities
- Integrated advanced PTX assembly optimizations including fast math operations, FMA fusion, and predicated execution
- Added new TiledMMA kernel (v10) specifically designed for Blackwell architecture with tcgen05.mma support
- Expanded performance analysis with roofline modeling and arithmetic intensity calculations for Tensor Core utilization

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Dependency Analysis](#dependency-analysis)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Conclusion](#conclusion)
10. [Appendices](#appendices)

## Introduction
This document provides a comprehensive technical and practical guide to the GDN Prefill Kernel implementation, focusing on the latest advances in TMA (Tensor Memory Accelerator) primitives and Tensor Core optimization for NVIDIA Blackwell architecture. The implementation now features comprehensive PTX assembly optimization with embedded assembly instructions, mma.sync Tensor Core support for BF16 matrix multiplication, and advanced TMA primitives for asynchronous memory operations. The document explains the mathematical formulation for batched sequence processing during initial token generation, contrasts it with the decode kernel's single-token approach, documents the sequential token processing algorithm for variable-length sequences, and details the chunked processing strategy and memory bandwidth optimization for prefill phases.

**Updated** The kernel now supports Blackwell architecture (sm_100) with tcgen05.mma tensor core operations, enabling matrix-matrix multiplications for compute-bound prefill processing with arithmetic intensities reaching up to 64 FLOP/byte.

## Project Structure
The repository organizes GDN kernels by stage (prefill/decode) with multiple implementation variants, including the new TMA-optimized versions. Each kernel directory contains:
- A configuration file specifying the solution metadata and build entry point.
- Multiple solution implementations (Triton, CUDA, PTX) with varying optimization levels.
- A baseline Python reference implementation.
- A definition JSON describing inputs, outputs, and axes for the benchmarking framework.
- Scripts for packaging, benchmarking, and performance analysis.

```mermaid
graph TB
subgraph "Kernel Definitions"
PrefillDef["gdn_prefill_qk4_v8_d128_k_last.json"]
DecodeDef["gdn_decode_qk4_v8_d128_k_last.json"]
end
subgraph "Prefill Kernel Implementations"
PrefillCfg["gdn_prefill_qk4_v8_d128_k_last/config.toml"]
PrefillSol["solution/triton/kernel.py"]
PrefillBase["baseline/triton/kernel.py"]
PrefillPTX["PTX/gdn_prefill_ptx.cuh<br/>(NEW: TMA + Tensor Core)"]
PrefillV6["CUDA/gdn_prefill_v6_chunked.cuh<br/>(NEW: Chunked)"]
PrefillV10["CUDA/gdn_prefill_v10.cuh<br/>(NEW: TiledMMA + Tensor Core)"]
end
subgraph "Decode Kernel"
DecodeCfg["gdn_decode_qk4_v8_d128_k_last/config.toml"]
DecodeSol["solution/triton/kernel.py"]
DecodeBase["baseline/triton/kernel.py"]
DecodeV6["CUDA/gdn_decode_v6.cuh<br/>(NEW: TMA)"]
DecodeV7["CUDA/gdn_decode_v7.cuh<br/>(NEW: Advanced TMA + Quantization)"]
end
Bench["benchmarks/bench_modal.py"]
Perf["docs/PERFORMANCE.md"]
Debug1["scripts/debug_prefill.py"]
Debug2["scripts/debug_prefill2.py"]
BenchAll["scripts/bench_prefill_all.py<br/>(NEW: Unified Benchmark)"]
PTXReadme["PTX/README.md<br/>(NEW: Optimization Guide)"]
PrefillDef --> PrefillSol
PrefillDef --> PrefillBase
PrefillDef --> PrefillPTX
PrefillDef --> PrefillV6
PrefillDef --> PrefillV10
DecodeDef --> DecodeSol
DecodeDef --> DecodeBase
DecodeDef --> DecodeV6
DecodeDef --> DecodeV7
BenchAll --> PrefillSol
BenchAll --> PrefillPTX
BenchAll --> PrefillV6
BenchAll --> PrefillV10
BenchAll --> DecodeSol
BenchAll --> DecodeV6
BenchAll --> DecodeV7
Perf --> BenchAll
PTXReadme --> PrefillPTX
```

**Diagram sources**
- [gdn_prefill_qk4_v8_d128_k_last/config.toml:1-10](file://gdn_prefill_qk4_v8_d128_k_last/config.toml#L1-L10)
- [gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py:1-148](file://gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py#L1-L148)
- [gdn_prefill_qk4_v8_d128_k_last/baseline/triton/kernel.py:1-99](file://gdn_prefill_qk4_v8_d128_k_last/baseline/triton/kernel.py#L1-L99)
- [src/kernels/ptx/gdn_prefill_ptx.cuh:1-708](file://src/kernels/ptx/gdn_prefill_ptx.cuh#L1-L708)
- [src/kernels/cuda/gdn_prefill_v6_chunked.cuh:1-285](file://src/kernels/cuda/gdn_prefill_v6_chunked.cuh#L1-L285)
- [src/kernels/cute_cpp/gdn_prefill_v10.cuh:1-390](file://src/kernels/cute_cpp/gdn_prefill_v10.cuh#L1-L390)
- [scripts/bench_prefill_all.py:1-331](file://scripts/bench_prefill_all.py#L1-L331)
- [src/kernels/ptx/README.md:1-179](file://src/kernels/ptx/README.md#L1-L179)
- [src/kernels/cuda/gdn_prefill_v7.cuh:1-549](file://src/kernels/cuda/gdn_prefill_v7.cuh#L1-L549)

**Section sources**
- [gdn_prefill_qk4_v8_d128_k_last/config.toml:1-10](file://gdn_prefill_qk4_v8_d128_k_last/config.toml#L1-L10)
- [gdn_decode_qk4_v8_d128_k_last/config.toml:1-10](file://gdn_decode_qk4_v8_d128_k_last/config.toml#L1-L10)
- [flashinfer_trace/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json:1-156](file://flashinfer_trace/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json#L1-L156)
- [flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json:1-153](file://flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json#L1-L153)

## Core Components
- **Prefill kernel (solution)**: Triton JIT kernel implementing batched sequential token processing with V-dimension splitting across programs for improved occupancy and register pressure management.
- **Prefill kernel (baseline)**: Python reference implementation performing the same GDN delta-rule update and output computation in a straightforward loop over sequences and tokens.
- **Prefill kernel (PTX)**: **NEW** CUDA kernel with embedded PTX assembly featuring chunked processing, fast math operations, mma.sync Tensor Core support, and TMA primitives for maximum performance on Blackwell architecture.
- **Prefill kernel (CUDA v6)**: **NEW** CUDA kernel implementing chunked processing with shared memory optimization and register blocking.
- **Prefill kernel (CUDA v10)**: **NEW** TiledMMA kernel specifically designed for Blackwell architecture with tcgen05.mma tensor core operations and optimized memory access patterns.
- **Decode kernel (solution)**: Triton JIT kernel implementing single-token generation with autotuned tile sizes and V-dimension splitting for optimal performance.
- **Decode kernel (baseline)**: Python reference implementation for correctness verification of single-token decode.
- **Decode kernel (CUDA v6)**: **NEW** TMA-optimized decode kernel with asynchronous memory operations and barrier synchronization.
- **Decode kernel (CUDA v7)**: **NEW** Advanced TMA kernel with quantized state storage, vectorized loads, and double buffering.
- **Definition JSONs**: Specify input/output shapes, dtypes, and axes for the benchmarking framework.
- **Benchmarking harness**: **NEW** Unified framework running all prefill implementations across workloads and reporting comprehensive performance metrics.

Key implementation highlights:
- Grouped Value Attention (GVA): num_q_heads=4, num_v_heads=8; qk_head = v_head // 2.
- State layout: k-last [N, H, V=128, K=128] for prefill; [B, H, V=128, K=128] for decode.
- Head size D=128; BLOCK_V=32; V_BLOCKS=D//BLOCK_V=4.
- Scale defaults to 1/sqrt(D) when not provided.
- **NEW**: Chunked processing with configurable CHUNK_SIZE (4, 8, 16) for increased arithmetic intensity.
- **NEW**: TMA primitives for asynchronous memory operations with mbarrier synchronization.
- **NEW**: Tensor Core support with mma.sync.aligned.m16n8k16 BF16 matrix multiplication.
- **NEW**: TiledMMA structure enabling matrix-matrix operations for compute-bound prefill.

**Section sources**
- [gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py:1-148](file://gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py#L1-L148)
- [gdn_prefill_qk4_v8_d128_k_last/baseline/triton/kernel.py:1-99](file://gdn_prefill_qk4_v8_d128_k_last/baseline/triton/kernel.py#L1-L99)
- [src/kernels/ptx/gdn_prefill_ptx.cuh:1-708](file://src/kernels/ptx/gdn_prefill_ptx.cuh#L1-L708)
- [src/kernels/cuda/gdn_prefill_v6_chunked.cuh:1-285](file://src/kernels/cuda/gdn_prefill_v6_chunked.cuh#L1-L285)
- [src/kernels/cute_cpp/gdn_prefill_v10.cuh:1-390](file://src/kernels/cute_cpp/gdn_prefill_v10.cuh#L1-L390)
- [src/kernels/cuda/gdn_decode_v6.cuh:1-310](file://src/kernels/cuda/gdn_decode_v6.cuh#L1-L310)
- [src/kernels/cuda/gdn_decode_v7.cuh:1-634](file://src/kernels/cuda/gdn_decode_v7.cuh#L1-L634)
- [flashinfer_trace/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json:1-156](file://flashinfer_trace/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json#L1-L156)
- [flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json:1-153](file://flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json#L1-L153)

## Architecture Overview
The GDN prefill kernel orchestrates batched sequential processing over variable-length sequences with multiple optimization strategies. The grid is partitioned by sequence, head, and V-tile, enabling independent processing of V-slices and efficient register usage. The new TMA-optimized implementations add asynchronous memory operations with barrier synchronization, while the Tensor Core variants leverage mma.sync operations for compute-bound performance.

```mermaid
graph TB
subgraph "Prefill Kernel Variants"
subgraph "Triton Implementation"
PGrid["Grid: (N, H=8, V_BLOCKS=4)"]
PState["State: k-last [N,H,V=128,K=128]"]
PSeq["Sequential Token Loop<br/>for i in range(seq_len)"]
PVSplit["V Split: BLOCK_V=32 per program"]
PGates["Gate Computation<br/>g, beta per token/head"]
PDelta["Delta Rule Update<br/>rank-1 update on S"]
POut["Output per token<br/>scale * S @ q"]
end
subgraph "PTX Implementation (NEW)"
PTXGrid["Grid: (N, H=8, V_BLOCKS)"]
PTXState["State: k-last [N,H,V=128,K=128]"]
PTXChunk["Chunked Processing<br/>C tokens per iteration"]
PTXPTX["PTX Assembly<br/>Fast math, FMA, cache hints"]
PTXTMA["TMA Primitives<br/>mbarrier, tma_load_2d"]
PTXMMA["Tensor Core<br/>mma.sync.aligned.m16n8k16"]
PTXOpt["Optimized Memory<br/>Non-coherent loads, predication"]
PTXOut["Output per token<br/>scale * S @ q"]
end
subgraph "CUDA v6 Implementation (NEW)"
V6Grid["Grid: (N, H=8, V_BLOCKS)"]
V6State["State: k-last [N,H,V=128,K=128]"]
V6Chunk["Chunked Processing<br/>Shared memory buffers"]
V6SMEM["Shared Memory<br/>Q/K prefetch, state reuse"]
V6Opt["Register Blocking<br/>Vectorized operations"]
V6Out["Output per token<br/>scale * S @ q"]
end
subgraph "CUDA v10 Implementation (NEW)"
V10Grid["Grid: (N, H=8, V_BLOCKS)"]
V10State["State: k-last [N,H,V=128,K=128]"]
V10Chunk["Chunked Processing<br/>TiledMMA structure"]
V10TMA["TiledMMA Ready<br/>Matrix-matrix ops"]
V10Tensor["Tensor Core<br/>tcgen05.mma (sm_100)"]
V10Opt["Bank Conflict Avoidance<br/>Swizzled access patterns"]
V10Out["Output per token<br/>scale * S @ q"]
end
end
subgraph "Decode Kernel"
DGrid["Grid: (B, H=8, V_BLOCKS)"]
DState["State: k-last [B,H,V=128,K=128]"]
DSingle["Single Token"]
DVSplit["V Split: BLOCK_V autotuned"]
DGates["Gate Computation<br/>g, beta per head"]
DDelta["Delta Rule Update<br/>rank-1 update on S"]
DOut["Output<br/>scale * S @ q"]
end
PGrid --> PState --> PSeq --> PDelta --> POut
PTXGrid --> PTXState --> PTXChunk --> PTXPTX --> PTXTMA --> PTXMMA --> PTXOpt --> PTXOut
V6Grid --> V6State --> V6Chunk --> V6SMEM --> V6Opt --> V6Out
V10Grid --> V10State --> V10Chunk --> V10TMA --> V10Tensor --> V10Opt --> V10Out
DGrid --> DState --> DSingle --> DDelta --> DOut
```

**Diagram sources**
- [gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py:24-96](file://gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py#L24-L96)
- [src/kernels/ptx/gdn_prefill_ptx.cuh:121-301](file://src/kernels/ptx/gdn_prefill_ptx.cuh#L121-L301)
- [src/kernels/cuda/gdn_prefill_v6_chunked.cuh:56-228](file://src/kernels/cuda/gdn_prefill_v6_chunked.cuh#L56-L228)
- [src/kernels/cute_cpp/gdn_prefill_v10.cuh:93-310](file://src/kernels/cute_cpp/gdn_prefill_v10.cuh#L93-L310)
- [gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py:37-97](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L37-L97)

## Detailed Component Analysis

### Mathematical Formulation: Batched Prefill vs Decode
- Gates:
  - Decay gate: g = exp(-exp(A_log) * softplus(a + dt_bias))
  - Update gate: beta = sigmoid(b)
- State update (delta rule):
  - S ← g ⊙ S
  - old_v = k^T @ S
  - new_v = beta * v + (1 - beta) * old_v
  - S ← S + k^T ⊗ (new_v - old_v)
- Output:
  - o = scale * q^T @ S

Batched prefill processes multiple tokens sequentially per sequence, while decode processes a single token per head per batch. The prefill kernel splits the V dimension across programs to reduce register pressure and increase occupancy. **NEW**: The PTX implementation adds chunked processing where C tokens are processed together, dramatically increasing arithmetic intensity from 1 FLOP/byte to C FLOP/byte. **NEW**: Tensor Core support enables matrix-matrix operations for compute-bound performance with tcgen05.mma on Blackwell architecture.

**Section sources**
- [gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py:65-96](file://gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py#L65-L96)
- [gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py:61-97](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L61-L97)
- [gdn_prefill_qk4_v8_d128_k_last/baseline/triton/kernel.py:78-94](file://gdn_prefill_qk4_v8_d128_k_last/baseline/triton/kernel.py#L78-L94)
- [gdn_decode_qk4_v8_d128_k_last/baseline/triton/kernel.py:79-98](file://gdn_decode_qk4_v8_d128_k_last/baseline/triton/kernel.py#L79-L98)

### Sequential Token Processing Algorithm (Variable-Length Sequences)
The prefill kernel iterates over tokens within each sequence using cumulative sequence lengths. For each token:
- Loads per-token gates g and beta.
- Loads k, v, q slices aligned to the current head mapping.
- Applies the delta rule to update the state slice S.
- Computes output for the current token.

**NEW**: The PTX implementation introduces chunked processing where multiple tokens (C tokens) are processed in each iteration, sharing state across tokens in the chunk for better memory bandwidth utilization. **NEW**: TMA primitives enable asynchronous memory operations with mbarrier synchronization for improved pipeline efficiency.

```mermaid
flowchart TD
Start(["Start Prefill"]) --> LoadBounds["Load cu_seqlens[n], cu_seqlens[n+1]"]
LoadBounds --> SeqLen["Compute seq_len = end - start"]
SeqLen --> NumChunks["Compute num_chunks = ceil(seq_len/CHUNK_SIZE)"]
NumChunks --> LoopChunks{"chunk = 0 to num_chunks-1"}
LoopChunks --> |Yes| ChunkStart["chunk_start = chunk * CHUNK_SIZE"]
ChunkStart --> ChunkEnd["chunk_end = min(chunk_start + CHUNK_SIZE, seq_len)"]
ChunkEnd --> ActualSize["actual_chunk_size = chunk_end - chunk_start"]
ActualSize --> LoadChunk["Load Q, K, V, gates for C tokens"]
LoadChunk --> TMAOps["TMA Asynchronous Loading<br/>mbarrier synchronization"]
TMAOps --> ProcessChunk["Process C tokens with shared state<br/>mma.sync Tensor Core ops"]
ProcessChunk --> StoreChunk["Store outputs for C tokens"]
StoreChunk --> NextChunk["chunk = chunk + 1"]
NextChunk --> LoopChunks
LoopChunks --> |No| FinalState["Store new state slice S[n,h,v_slice]"]
FinalState --> End(["End Prefill"])
```

**Diagram sources**
- [src/kernels/ptx/gdn_prefill_ptx.cuh:188-291](file://src/kernels/ptx/gdn_prefill_ptx.cuh#L188-L291)
- [src/kernels/cuda/gdn_prefill_v6_chunked.cuh:123-218](file://src/kernels/cuda/gdn_prefill_v6_chunked.cuh#L123-L218)
- [src/kernels/cuda/gdn_decode_v6.cuh:71-87](file://src/kernels/cuda/gdn_decode_v6.cuh#L71-L87)

**Section sources**
- [gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py:47-96](file://gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py#L47-L96)
- [src/kernels/ptx/gdn_prefill_ptx.cuh:188-291](file://src/kernels/ptx/gdn_prefill_ptx.cuh#L188-L291)
- [src/kernels/cuda/gdn_prefill_v6_chunked.cuh:123-218](file://src/kernels/cuda/gdn_prefill_v6_chunked.cuh#L123-L218)

### Chunked Processing Strategy and Memory Bandwidth Optimization
**NEW**: The chunked processing strategy represents a fundamental shift in prefill kernel optimization:

- **Arithmetic Intensity Analysis**:
  - Single-token processing: AI = 1 FLOP/byte (memory-bound)
  - Chunked processing (C tokens): AI = C FLOP/byte (compute-bound)
  - With CHUNK_SIZE=8: AI = 8 FLOP/byte approaching B200 ridge point (70 TFLOPS/8 TB/s = 8.75)
  - With CHUNK_SIZE=64: AI = 64 FLOP/byte (compute-bound!) for Tensor Core utilization

- **Memory Access Optimization**:
  - State loaded once per chunk, reused C times
  - Shared memory buffers for Q, K, V, and intermediate computations
  - Vectorized loads using float4 for 128B aligned access patterns
  - Non-coherent loads (ld.global.nc) to bypass L1 cache for streaming workloads
  - **NEW**: TMA asynchronous loading with mbarrier synchronization for improved pipeline efficiency

- **PTX Assembly Optimizations**:
  - Fast math: ex2.approx, lg2.approx, rcp.approx for 2-3x speedup
  - FMA fusion: fma.rn.f32 for single rounding with better precision
  - Predicated execution: selp.f32 for branchless conditionals
  - Warp shuffle: shfl.sync.bfly for warp-level reductions without shared memory

- **Tensor Core Integration**:
  - **NEW**: mma.sync.aligned.m16n8k16 BF16 matrix multiplication for compute-bound operations
  - **NEW**: tcgen05.mma tensor core operations on Blackwell (sm_100) architecture
  - **NEW**: Matrix-matrix operations enabling chunked processing with 16x8x16 tile sizes

**Section sources**
- [src/kernels/ptx/gdn_prefill_ptx.cuh:10-19](file://src/kernels/ptx/gdn_prefill_ptx.cuh#L10-L19)
- [src/kernels/ptx/README.md:12-50](file://src/kernels/ptx/README.md#L12-L50)
- [src/kernels/cuda/gdn_prefill_v6_chunked.cuh:40-55](file://src/kernels/cuda/gdn_prefill_v6_chunked.cuh#L40-L55)
- [src/kernels/cute_cpp/gdn_prefill_v10.cuh:9-26](file://src/kernels/cute_cpp/gdn_prefill_v10.cuh#L9-L26)

### State Management Differences: Prefill vs Decode
- Initialization:
  - Prefill: state can be provided as k-last [N, H, V, K]; if absent, initialized to zeros.
  - Decode: state can be provided as k-last [B, H, V, K]; if absent, initialized to zeros.
- Layout:
  - Both use k-last layout [*, H, V, K] internally for state.
- Final state:
  - Prefill: returns new_state [N, H, V, K] for each sequence.
  - Decode: returns new_state [B, H, V, K] for each batch item.
- Head mapping:
  - GVA: qk_h = h // 2; ensures 2 v-heads per qk-head.

**Section sources**
- [gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py:118-124](file://gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py#L118-L124)
- [gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py:117-123](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L117-L123)
- [flashinfer_trace/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json:79-89](file://flashinfer_trace/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json#L79-L89)
- [flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json:80-90](file://flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json#L80-L90)

### Triton Implementation Details: Loop Unrolling and Memory Access
- Loop unrolling:
  - The sequential token loop is explicit and unrolled per-token; the kernel schedules multiple programs to handle V-slices independently, effectively unrolling across V-tiles.
- Memory access optimization:
  - Uses tl.arange for vectorized indices (di, vd) to enable coalesced reads/writes.
  - Loads k, v, q with stride-aware indexing aligned to head groups.
  - Stores outputs and final state slices with stride offsets.
- Grid configuration:
  - Prefill: (N, H=8, V_BLOCKS=4); BLOCK_V=32; num_warps=4.
  - Decode: autotunes BLOCK_V across {16,32,64,128} with num_warps={2,4,8}; num_stages=2.

**Section sources**
- [gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py:62-96](file://gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py#L62-L96)
- [gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py:23-36](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L23-L36)
- [gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py:105-141](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L105-L141)

### Relationship Between Q/K/V Dimensions and Sequence Length Handling
- Q/K/V shapes:
  - Q: [T, 4, 128], K: [T, 4, 128], V: [T, 8, 128] for prefill.
  - Q/K: [B, 1, 4, 128], V: [B, 1, 8, 128] for decode.
- Head mapping:
  - num_q_heads=4, num_v_heads=8; qk_h = h // 2.
- Sequence lengths:
  - cu_seqlens defines variable-length batches; the kernel computes t_start/t_end per sequence and iterates over tokens within bounds.

**Section sources**
- [flashinfer_trace/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json:51-132](file://flashinfer_trace/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json#L51-L132)
- [flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json:49-127](file://flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json#L49-L127)
- [gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py:47-50](file://gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py#L47-L50)

### Comparison Against Baseline: Throughput Improvements
**NEW**: Comprehensive benchmarking framework comparing all prefill implementations:

- **Unified Benchmarking**: `bench_prefill_all.py` compares Triton, PTX, CUDA v6, and CUDA v10 implementations across multiple configurations
- **Performance Metrics**: Time (ms), Bandwidth (GB/s), Tokens per second (M tokens/s)
- **Roofline Analysis**: Demonstrates how chunking increases arithmetic intensity from 1 to 16 FLOP/byte
- **Tensor Core Analysis**: Shows compute-bound operation with 64 FLOP/byte arithmetic intensity on Blackwell
- **Framework Recommendations**: 
  - Decode (AI=1): Triton or advanced CUDA variants
  - Prefill with chunking (AI=8): PTX or CUDA v6 with CHUNK_SIZE=8
  - Prefill with Tensor Core (AI=64): CUDA v10 with tcgen05.mma on Blackwell
  - Production: Adaptive chunk sizing based on sequence length and hardware capabilities

```mermaid
sequenceDiagram
participant Runner as "Unified Benchmark Runner"
participant Triton as "Triton Prefill"
participant PTX as "PTX Prefill (NEW)"
participant V6 as "CUDA v6 Prefill (NEW)"
participant V10 as "CUDA v10 Prefill (NEW)"
participant Base as "Baseline Kernel"
participant Perf as "Performance Analysis"
Runner->>Triton : "Run prefill with Triton"
Triton-->>Runner : "Latency ms, BW GB/s"
Runner->>PTX : "Run prefill with PTX"
PTX-->>Runner : "Latency ms, BW GB/s"
Runner->>V6 : "Run prefill with CUDA v6"
V6-->>Runner : "Latency ms, BW GB/s"
Runner->>V10 : "Run prefill with CUDA v10"
V10-->>Runner : "Latency ms, BW GB/s"
Runner->>Base : "Run baseline reference"
Base-->>Runner : "Reference output"
Runner->>Perf : "Analyze results"
Perf-->>Runner : "Roofline analysis, recommendations"
```

**Diagram sources**
- [scripts/bench_prefill_all.py:34-331](file://scripts/bench_prefill_all.py#L34-L331)

**Section sources**
- [benchmarks/bench_modal.py:202-239](file://benchmarks/bench_modal.py#L202-L239)
- [scripts/bench_prefill_all.py:302-321](file://scripts/bench_prefill_all.py#L302-L321)
- [docs/PERFORMANCE.md:31-48](file://docs/PERFORMANCE.md#L31-L48)

## Dependency Analysis
The kernels depend on the benchmarking framework and definition JSONs for input/output specification and workload generation. The solution and baseline implementations are decoupled from each other and can be evaluated independently. **NEW**: The unified benchmarking framework coordinates all implementations for comprehensive performance analysis, including TMA and Tensor Core variants.

```mermaid
graph TB
DefPrefill["Prefill Definition JSON"]
DefDecode["Decode Definition JSON"]
Bench["Unified Benchmark Runner"]
PrefillSol["Triton Solution"]
PrefillPTX["PTX Solution (NEW)"]
PrefillV6["CUDA v6 Solution (NEW)"]
PrefillV10["CUDA v10 Solution (NEW)"]
PrefillBase["Prefill Baseline"]
DecodeSol["Decode Solution"]
DecodeBase["Decode Baseline"]
DecodeV6["Decode v6 (NEW)"]
DecodeV7["Decode v7 (NEW)"]
PTXReadme["PTX Optimization Guide"]
BenchAll["bench_prefill_all.py"]
DefPrefill --> Bench
DefDecode --> Bench
BenchAll --> PrefillSol
BenchAll --> PrefillPTX
BenchAll --> PrefillV6
BenchAll --> PrefillV10
BenchAll --> PrefillBase
BenchAll --> DecodeSol
BenchAll --> DecodeV6
BenchAll --> DecodeV7
PTXReadme --> PrefillPTX
```

**Diagram sources**
- [flashinfer_trace/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json:1-156](file://flashinfer_trace/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json#L1-L156)
- [flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json:1-153](file://flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json#L1-L153)
- [scripts/bench_prefill_all.py:1-331](file://scripts/bench_prefill_all.py#L1-L331)
- [src/kernels/ptx/README.md:1-179](file://src/kernels/ptx/README.md#L1-L179)

**Section sources**
- [scripts/bench_prefill_all.py:1-331](file://scripts/bench_prefill_all.py#L1-L331)
- [flashinfer_trace/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json:1-156](file://flashinfer_trace/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json#L1-L156)
- [flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json:1-153](file://flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json#L1-L153)

## Performance Considerations
- **Register pressure**: V-split with BLOCK_V=32 reduces per-program state from [128,128] to [32,128], improving occupancy.
- **Arithmetic intensity**: **NEW** Chunked processing dramatically increases AI from 1 to 16 FLOP/byte, moving from memory-bound to compute-bound operation. **NEW** Tensor Core utilization can achieve AI up to 64 FLOP/byte on Blackwell.
- **Memory bandwidth**: **NEW** PTX implementation uses non-coherent loads and predicated execution to minimize memory stalls; shared memory optimization in CUDA v6. **NEW** TMA asynchronous loading with mbarrier synchronization for improved pipeline efficiency.
- **Roofline analysis**: **NEW** Demonstrates B200 peak performance (70 TFLOPS, 8 TB/s) and optimal operating points for different chunk sizes and Tensor Core utilization.
- **Framework selection**: **NEW** Recommendations based on batch size, sequence length, and hardware capabilities for optimal performance.
- **Tensor Core utilization**: **NEW** tcgen05.mma operations on Blackwell architecture enable compute-bound prefill processing with BF16 precision.

**Section sources**
- [src/kernels/ptx/README.md:39-50](file://src/kernels/ptx/README.md#L39-L50)
- [scripts/bench_prefill_all.py:309-321](file://scripts/bench_prefill_all.py#L309-L321)
- [docs/PERFORMANCE.md:145-158](file://docs/PERFORMANCE.md#L145-L158)
- [src/kernels/cute_cpp/gdn_prefill_v10.cuh:15-26](file://src/kernels/cute_cpp/gdn_prefill_v10.cuh#L15-L26)

## Troubleshooting Guide
- **Correctness verification**:
  - Use debug scripts to compare solution outputs against the baseline reference implementation.
  - Scripts validate outputs and new_state tensors for numerical agreement.
- **Benchmarking**:
  - **NEW**: Unified benchmarking framework provides comprehensive performance analysis across all implementations.
  - Side-by-side comparison available for Triton, PTX, CUDA v6, and CUDA v10 implementations.
  - **NEW**: Roofline analysis helps identify optimal chunk sizes and configurations.
  - **NEW**: Tensor Core performance analysis for Blackwell architecture.
- **PTX-specific debugging**:
  - **NEW**: PTX assembly requires careful attention to register usage and memory alignment.
  - Use `ptxas` compiler flags for detailed assembly analysis.
  - **NEW**: TMA primitive debugging with mbarrier synchronization validation.
- **Tensor Core debugging**:
  - **NEW**: Verify tcgen05.mma instruction availability on target architecture.
  - **NEW**: Check chunk size compatibility with mma.sync tile sizes (16x8x16).

**Section sources**
- [scripts/debug_prefill.py:159-166](file://scripts/debug_prefill.py#L159-L166)
- [scripts/debug_prefill2.py:159-178](file://scripts/debug_prefill2.py#L159-L178)
- [scripts/bench_prefill_all.py:276-282](file://scripts/bench_prefill_all.py#L276-L282)
- [src/kernels/ptx/README.md:151-163](file://src/kernels/ptx/README.md#L151-L163)

## Conclusion
The GDN Prefill Kernel has evolved significantly with the introduction of TMA (Tensor Memory Accelerator) primitives, Tensor Core support, and comprehensive PTX assembly optimization. The new PTX implementation achieves substantial performance improvements through embedded assembly instructions, while the CUDA v6 variant demonstrates the effectiveness of shared memory optimization. **NEW**: The CUDA v10 implementation leverages TiledMMA structure and tcgen05.mma tensor core operations for compute-bound prefill processing on Blackwell architecture. The unified benchmarking framework provides comprehensive analysis of different approaches, enabling informed decisions about kernel selection based on workload characteristics and hardware capabilities. These advances represent a paradigm shift from memory-bound to compute-bound prefill processing, with arithmetic intensities reaching up to 64 FLOP/byte on modern Blackwell hardware.

## Appendices

### Appendix A: PTX Assembly Instructions and Optimizations
**NEW**: Detailed breakdown of PTX assembly optimizations used in the prefill kernel:

- **Fast Math Operations**:
  - `ex2.approx.f32`: Fast exponential approximation (2-3x faster)
  - `lg2.approx.f32`: Fast logarithm base-2 approximation
  - `rcp.approx.f32`: Fast reciprocal approximation
  
- **Fused Operations**:
  - `fma.rn.f32`: Fused multiply-add with single rounding
  - Eliminates intermediate rounding errors and improves performance
  
- **Memory Operations**:
  - `ld.global.nc.f32`: Non-coherent load bypassing L1 cache
  - `st.global.wb.f32`: Write-back store for efficient caching
  
- **Predicated Execution**:
  - `selp.f32`: Branchless conditional selection
  - Reduces warp divergence and improves occupancy

**Section sources**
- [src/kernels/ptx/gdn_prefill_ptx.cuh:34-108](file://src/kernels/ptx/gdn_prefill_ptx.cuh#L34-L108)
- [src/kernels/ptx/README.md:82-139](file://src/kernels/ptx/README.md#L82-L139)

### Appendix B: TMA Primitives and Barrier Synchronization
**NEW**: Comprehensive TMA (Tensor Memory Accelerator) primitives for asynchronous memory operations:

- **mbarrier Management**:
  - `ptx_mbarrier_init`: Initialize barrier with expected arrival count
  - `ptx_mbarrier_arrive_tx`: Announce expected bytes for TMA transaction
  - `ptx_mbarrier_wait`: Wait on barrier with parity checking
  
- **TMA Memory Operations**:
  - `ptx_tma_load_2d`: Bulk async 2D tensor load from global to shared memory
  - Supports complete_tx::bytes synchronization model
  
- **CP.Async Operations**:
  - `ptx_cp_async_16`: Element-wise async copy with 16-byte granularity
  - `ptx_cp_async_commit`: Commit async copy group
  - `ptx_cp_async_wait_all`: Wait for all async copies to complete

**Section sources**
- [src/kernels/ptx/gdn_prefill_ptx.cuh:139-188](file://src/kernels/ptx/gdn_prefill_ptx.cuh#L139-L188)
- [src/kernels/cuda/gdn_decode_v6.cuh:46-87](file://src/kernels/cuda/gdn_decode_v6.cuh#L46-L87)
- [src/kernels/cuda/gdn_decode_v7.cuh:142-186](file://src/kernels/cuda/gdn_decode_v7.cuh#L142-L186)

### Appendix C: Tensor Core Integration and mma.sync Operations
**NEW**: Detailed Tensor Core support for Blackwell architecture:

- **mma.sync Operations**:
  - `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`: 16x8x16 BF16 matrix multiply
  - Each warp computes one 16x8 output tile
  - Thread mapping: 32 threads per warp, 4 BF16 elements per thread
  
- **tcgen05.mma Performance**:
  - FP4: 9 PFLOPS dense, 18 PFLOPS sparse
  - FP8: 4.5 PFLOPS dense, 9 PFLOPS sparse  
  - BF16: 2.25 PFLOPS dense, 4.5 PFLOPS sparse
  
- **Chunked Processing Integration**:
  - Enables matrix-matrix operations for compute-bound prefill
  - CHUNK_SIZE determines output tile dimensions [V, C]
  - Requires BLOCK_V=16 for m16n8k16 tile compatibility

**Section sources**
- [src/kernels/ptx/gdn_prefill_ptx.cuh:105-132](file://src/kernels/ptx/gdn_prefill_ptx.cuh#L105-L132)
- [src/kernels/cute_cpp/gdn_prefill_v10.cuh:9-26](file://src/kernels/cute_cpp/gdn_prefill_v10.cuh#L9-L26)

### Appendix D: Chunked Processing Configuration
**NEW**: Optimal chunk size selection based on hardware characteristics:

- **CHUNK_SIZE = 4**: Good balance between memory usage and compute utilization
- **CHUNK_SIZE = 8**: Targets B200 ridge point (8.75 FLOP/byte)
- **CHUNK_SIZE = 16**: Maximum compute-bound operation, requires sufficient memory bandwidth
- **CHUNK_SIZE = 64**: **NEW** Tensor Core optimal for tcgen05.mma operations

```mermaid
graph LR
A["Sequence Length"] --> B{"CHUNK_SIZE Selection"}
B --> |"Length < 16"| C["CHUNK_SIZE = 4"]
B --> |"16 ≤ Length < 64"| D["CHUNK_SIZE = 8"]
B --> |"Length ≥ 64"| E["CHUNK_SIZE = 16 or 64 (Tensor Core)"]
C --> F["Memory Usage: Low"]
D --> G["Memory Usage: Medium"]
E --> H["Memory Usage: High"]
```

**Section sources**
- [src/kernels/ptx/README.md:31-37](file://src/kernels/ptx/README.md#L31-L37)
- [src/kernels/cuda/gdn_prefill_v6_chunked.cuh:40-55](file://src/kernels/cuda/gdn_prefill_v6_chunked.cuh#L40-L55)
- [src/kernels/cute_cpp/gdn_prefill_v10.cuh:22-26](file://src/kernels/cute_cpp/gdn_prefill_v10.cuh#L22-L26)

### Appendix E: Triton Kernel Entry Points and Grid Configuration
- **Prefill solution entry point**: kernel function with grid (N, H=8, V_BLOCKS=4), BLOCK_V=32, num_warps=4.
- **Decode solution entry point**: kernel function with autotuned grid and BLOCK_V across {16,32,64,128}, num_warps={2,4,8}, num_stages=2.
- **PTX implementation**: Template-based kernel with compile-time CHUNK_SIZE selection and TMA support.
- **CUDA v6 implementation**: Template-based kernel with shared memory optimization and chunked processing.
- **CUDA v10 implementation**: Template-based kernel with TiledMMA structure and Tensor Core integration.

**Section sources**
- [gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py:126-148](file://gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py#L126-L148)
- [gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py:125-141](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L125-L141)
- [src/kernels/ptx/gdn_prefill_ptx.cuh:121-355](file://src/kernels/ptx/gdn_prefill_ptx.cuh#L121-L355)
- [src/kernels/cuda/gdn_prefill_v6_chunked.cuh:230-282](file://src/kernels/cuda/gdn_prefill_v6_chunked.cuh#L230-L282)
- [src/kernels/cute_cpp/gdn_prefill_v10.cuh:93-390](file://src/kernels/cute_cpp/gdn_prefill_v10.cuh#L93-L390)

### Appendix F: Definition JSON Inputs/Outputs Summary
- **Prefill inputs**: q[T,4,128], k[T,4,128], v[T,8,128], state[N,8,128,128], A_log[8], a[T,8], dt_bias[8], b[T,8], cu_seqlens[num_seqs+1], scale.
- **Prefill outputs**: output[T,8,128], new_state[N,8,128,128].
- **Decode inputs**: q[B,1,4,128], k[B,1,4,128], v[B,1,8,128], state[B,8,128,128], A_log[8], a[B,1,8], dt_bias[8], b[B,1,8], scale.
- **Decode outputs**: output[B,1,8,128], new_state[B,8,128,128].

**Section sources**
- [flashinfer_trace/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json:51-153](file://flashinfer_trace/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json#L51-L153)
- [flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json:49-151](file://flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json#L49-L151)

### Appendix G: Unified Benchmarking Framework
**NEW**: Complete benchmarking infrastructure for comprehensive performance analysis:

- **Multi-framework support**: Tests Triton, PTX, CUDA v6, and CUDA v10 implementations
- **Configurable workloads**: Short, medium, long sequences with varying batch sizes
- **Performance metrics**: Latency, bandwidth, tokens per second
- **Roofline analysis**: Arithmetic intensity calculations and hardware utilization
- **Tensor Core analysis**: Performance evaluation for Blackwell architecture
- **Framework recommendations**: Optimal kernel selection based on workload characteristics and hardware capabilities

**Section sources**
- [scripts/bench_prefill_all.py:1-331](file://scripts/bench_prefill_all.py#L1-L331)
- [src/kernels/ptx/README.md:303-321](file://src/kernels/ptx/README.md#L303-L321)
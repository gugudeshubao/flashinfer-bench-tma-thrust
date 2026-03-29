# Benchmarking Framework

<cite>
**Referenced Files in This Document**
- [README.md](file://README.md)
- [bench_all_versions.py](file://scripts/bench_all_versions.py)
- [bench_cuda_real.py](file://scripts/bench_cuda_real.py)
- [bench_modal.py](file://benchmarks/bench_modal.py)
- [build_cuda.py](file://scripts/build_cuda.py)
- [setup_volume.py](file://scripts/setup_volume.py)
- [gdn_decode_qk4_v8_d128_k_last.json](file://flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json)
- [gdn_prefill_qk4_v8_d128_k_last.json](file://flashinfer_trace/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json)
- [gdn_decode_v5.cuh](file://src/kernels/cuda/gdn_decode_v5.cuh)
- [gdn_decode_v6.cuh](file://src/kernels/cuda/gdn_decode_v6.cuh)
- [gdn_decode_v7.cuh](file://src/kernels/cuda/gdn_decode_v7.cuh)
- [gdn_decode_v8.cuh](file://src/kernels/cuda/gdn_decode_v8.cuh)
- [gdn_decode_v9.cuh](file://src/kernels/cute/gdn_decode_v9.cuh)
- [gdn_decode_v10.cuh](file://src/kernels/cute/gdn_decode_v10.cuh)
- [PERFORMANCE.md](file://docs/PERFORMANCE.md)
- [debug_prefill.py](file://scripts/debug_prefill.py)
- [debug_prefill2.py](file://scripts/debug_prefill2.py)
</cite>

## Update Summary
**Changes Made**
- Replaced old bench_modal.py approach with new unified benchmarking system
- Added comprehensive scripts/bench_all_versions.py for multi-version benchmarking (v5-v8)
- Added scripts/bench_cuda_real.py for real CUDA kernel benchmarking (v7-v10)
- Expanded kernel support to include v9 and v10 with advanced features like CuTe DSL and TMA
- Enhanced batch size testing across 1, 16, 64, 256 with adaptive BLOCK_V sizing
- Added comprehensive correctness validation framework
- Integrated CUDA library compilation and testing infrastructure

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
This document explains the comprehensive benchmarking framework and execution system for the Gated Delta Net (GDN) kernels across multiple kernel versions (v5-v10) on the Modal cloud platform. The framework has evolved from a simple Triton-based benchmark to a unified system supporting CUDA kernels with advanced features like Tensor Memory Accelerator (TMA), CuTe DSL, and various precision optimizations. It covers cloud integration for GPU execution (NVIDIA B200), volume setup, workload provisioning, and comprehensive benchmark orchestration with correctness validation.

The framework now supports extensive kernel version testing, adaptive batch size optimization, and real CUDA library benchmarking with performance validation against Triton baselines. It provides detailed performance analysis across different hardware configurations and kernel implementations.

## Project Structure
The repository organizes the benchmarking stack into:
- **Unified benchmarking scripts**: [bench_all_versions.py](file://scripts/bench_all_versions.py), [bench_cuda_real.py](file://scripts/bench_cuda_real.py) replacing old bench_modal.py
- **CUDA kernel compilation**: [build_cuda.py](file://scripts/build_cuda.py) for compiling v5-v10 kernels
- **Volume setup**: [setup_volume.py](file://scripts/setup_volume.py) for creating synthetic or HF datasets
- **Kernel implementations**: Multi-version support (v5-v10) with CUDA and CuTe implementations
- **Workload definitions**: JSON specification files under [flashinfer_trace/definitions/gdn](file://flashinfer_trace/definitions/gdn)
- **Documentation**: Performance tracking and optimization guides
- **Debugging utilities**: Scripts for correctness validation and framework evaluation

```mermaid
graph TB
subgraph "Unified Benchmarking Scripts"
BAV["scripts/bench_all_versions.py"]
BCR["scripts/bench_cuda_real.py"]
END
subgraph "CUDA Infrastructure"
BC["scripts/build_cuda.py"]
SV["scripts/setup_volume.py"]
LIB["/data/lib/libgdn_kernels.so"]
END
subgraph "Multi-Version Kernels"
CUDA5["src/kernels/cuda/gdn_decode_v5.cuh"]
CUDA6["src/kernels/cuda/gdn_decode_v6.cuh"]
CUDA7["src/kernels/cuda/gdn_decode_v7.cuh"]
CUDA8["src/kernels/cuda/gdn_decode_v8.cuh"]
CUTE9["src/kernels/cute/gdn_decode_v9.cuh"]
CUTE10["src/kernels/cute/gdn_decode_v10.cuh"]
END
subgraph "Workload Definitions"
DEF_DEC["gdn_decode_* JSON"]
DEF_PREF["gdn_prefill_* JSON"]
END
subgraph "Documentation"
PERF["docs/PERFORMANCE.md"]
ROAD["docs/ROADMAP.md"]
END
BAV --> CUDA5
BAV --> CUDA6
BAV --> CUDA7
BAV --> CUDA8
BCR --> LIB
BC --> LIB
SV --> DEF_DEC
SV --> DEF_PREF
CUDA5 --> LIB
CUDA6 --> LIB
CUDA7 --> LIB
CUDA8 --> LIB
CUTE9 --> LIB
CUTE10 --> LIB
```

**Diagram sources**
- [bench_all_versions.py:1-444](file://scripts/bench_all_versions.py#L1-L444)
- [bench_cuda_real.py:1-604](file://scripts/bench_cuda_real.py#L1-L604)
- [build_cuda.py:1-436](file://scripts/build_cuda.py#L1-L436)
- [setup_volume.py:1-220](file://scripts/setup_volume.py#L1-L220)
- [gdn_decode_v5.cuh:1-320](file://src/kernels/cuda/gdn_decode_v5.cuh#L1-L320)
- [gdn_decode_v7.cuh:1-634](file://src/kernels/cuda/gdn_decode_v7.cuh#L1-L634)
- [gdn_decode_v9.cuh:1-200](file://src/kernels/cute/gdn_decode_v9.cuh#L1-L200)
- [gdn_decode_v10.cuh:1-200](file://src/kernels/cute/gdn_decode_v10.cuh#L1-L200)

**Section sources**
- [README.md:63-92](file://README.md#L63-L92)
- [bench_all_versions.py:1-444](file://scripts/bench_all_versions.py#L1-L444)
- [bench_cuda_real.py:1-604](file://scripts/bench_cuda_real.py#L1-L604)
- [build_cuda.py:1-436](file://scripts/build_cuda.py#L1-L436)

## Core Components
- **Unified benchmarking system**: Two main scripts for different benchmarking scenarios - bench_all_versions.py for multi-version testing and bench_cuda_real.py for real CUDA kernel validation
- **CUDA kernel compilation**: Automated compilation of v5-v10 kernels with nvcc for B200 (sm_100) architecture
- **Volume management**: Synthetic dataset generation and HuggingFace dataset download for comprehensive testing
- **Multi-version kernel support**: Complete coverage from Triton v5 baseline through CuTe v10 advanced implementations
- **Adaptive batch optimization**: Intelligent BLOCK_V sizing based on batch size for optimal performance
- **Correctness validation**: Comprehensive verification framework comparing CUDA kernels against Triton baseline

Key responsibilities:
- **Multi-version benchmarking**: [benchmark_versions function:38-404](file://scripts/bench_all_versions.py#L38-L404)
- **Real CUDA validation**: [benchmark_cuda_real function:28-597](file://scripts/bench_cuda_real.py#L28-L597)
- **CUDA compilation**: [build_cuda_kernels function:69-373](file://scripts/build_cuda.py#L69-L373)
- **Volume setup**: [setup_synthetic function:146-169](file://scripts/setup_volume.py#L146-L169)
- **Kernel version support**: [v5-v10 kernel implementations:1-320](file://src/kernels/cuda/gdn_decode_v5.cuh#L1-L320)

**Section sources**
- [bench_all_versions.py:38-404](file://scripts/bench_all_versions.py#L38-L404)
- [bench_cuda_real.py:28-597](file://scripts/bench_cuda_real.py#L28-L597)
- [build_cuda.py:69-373](file://scripts/build_cuda.py#L69-L373)
- [setup_volume.py:146-169](file://scripts/setup_volume.py#L146-L169)

## Architecture Overview
The system now features a unified benchmarking architecture supporting multiple kernel versions with comprehensive validation and performance analysis. It includes automated CUDA compilation, real kernel benchmarking, and multi-version comparison capabilities.

```mermaid
sequenceDiagram
participant CLI as "User CLI"
participant BAV as "bench_all_versions.py"
participant BCR as "bench_cuda_real.py"
participant BC as "build_cuda.py"
participant CUDA as "CUDA Library"
participant FS as "Modal Volume"
CLI->>BAV : "modal run scripts/bench_all_versions.py --versions all"
BAV->>FS : "Load test data and configurations"
BAV->>BAV : "Run v5-v8 benchmarks with adaptive BLOCK_V"
BAV->>CLI : "Print version comparison results"
CLI->>BC : "modal run scripts/build_cuda.py"
BC->>FS : "Compile v5-v10 kernels with nvcc"
BC->>FS : "Generate libgdn_kernels.so"
BC->>CLI : "Verify library exports"
CLI->>BCR : "modal run scripts/bench_cuda_real.py"
BCR->>FS : "Load compiled CUDA library"
BCR->>BCR : "Validate correctness vs Triton v5"
BCR->>BCR : "Benchmark v7-v10 kernels"
BCR->>CLI : "Print CUDA vs Triton comparison"
```

**Diagram sources**
- [bench_all_versions.py:407-444](file://scripts/bench_all_versions.py#L407-L444)
- [bench_cuda_real.py:600-604](file://scripts/bench_cuda_real.py#L600-L604)
- [build_cuda.py:416-436](file://scripts/build_cuda.py#L416-L436)

## Detailed Component Analysis

### Unified Benchmarking System
The new benchmarking system replaces the old bench_modal.py approach with two specialized scripts:

**bench_all_versions.py**: Comprehensive multi-version benchmarking supporting v5-v8 with adaptive batch optimization
- Tests kernel versions v5, v6, v7, v8 with configurable batch sizes (1, 16, 64, 256)
- Implements intelligent BLOCK_V sizing based on batch size for optimal performance
- Provides bandwidth calculations and performance comparisons across versions
- Supports both synthetic and real kernel execution

**bench_cuda_real.py**: Real CUDA kernel validation and benchmarking
- Validates correctness of compiled CUDA kernels against Triton v5 baseline
- Benchmarks real CUDA v7, v8, v9, v10 kernels with comprehensive performance analysis
- Includes CUDA Graph optimization for low-latency launches
- Supports both FP32 and quantized precision modes (FP4, FP8)

**Section sources**
- [bench_all_versions.py:1-444](file://scripts/bench_all_versions.py#L1-L444)
- [bench_cuda_real.py:1-604](file://scripts/bench_cuda_real.py#L1-L604)

### CUDA Kernel Compilation Infrastructure
The build system automates compilation of all kernel versions with proper dependencies and optimizations:

- **CUDA 12.8 support**: Full B200 (sm_100) compatibility with modern CUDA features
- **CUTLASS integration**: CuTe DSL support through CUTLASS headers for advanced kernel optimization
- **Combined compilation**: Single shared library containing all kernel versions
- **External C wrappers**: ctypes-compatible function exports for Python integration
- **CUDA Graph support**: Low-latency kernel launching for small batches

**Section sources**
- [build_cuda.py:16-34](file://scripts/build_cuda.py#L16-L34)
- [build_cuda.py:332-373](file://scripts/build_cuda.py#L332-L373)
- [build_cuda.py:110-330](file://scripts/build_cuda.py#L110-L330)

### Multi-Version Kernel Support
The framework now supports a complete evolution of GDN kernels:

**v5 (Baseline)**: Triton implementation with auto-tuning and basic optimizations
**v6 (TMA)**: CUDA implementation with Tensor Memory Accelerator for async state loading
**v7 (Quantization)**: Advanced CUDA with FP4/FP8 quantization and vectorized loads
**v8 (Warp Specialization)**: Maximum performance with warp specialization and FP8 optimization
**v9 (CuTe Swizzle)**: CuTe DSL with SMEM swizzling for optimal memory access patterns
**v10 (Advanced CuTe)**: Latest CuTe optimizations with TMA async copy capabilities

**Section sources**
- [gdn_decode_v5.cuh:1-320](file://src/kernels/cuda/gdn_decode_v5.cuh#L1-L320)
- [gdn_decode_v6.cuh:1-310](file://src/kernels/cuda/gdn_decode_v6.cuh#L1-L310)
- [gdn_decode_v7.cuh:1-634](file://src/kernels/cuda/gdn_decode_v7.cuh#L1-L634)
- [gdn_decode_v8.cuh:1-653](file://src/kernels/cuda/gdn_decode_v8.cuh#L1-L653)
- [gdn_decode_v9.cuh:1-200](file://src/kernels/cute/gdn_decode_v9.cuh#L1-L200)
- [gdn_decode_v10.cuh:1-200](file://src/kernels/cute/gdn_decode_v10.cuh#L1-L200)

### Adaptive Batch Optimization
The benchmarking system implements intelligent batch size optimization:

- **Batch 1**: BLOCK_V = 16 for optimal occupancy with small batches
- **Batch 1-16**: BLOCK_V = 16 for balanced memory and compute utilization
- **Batch 17-128**: BLOCK_V = 32 for increased parallelism
- **Batch 129+**: BLOCK_V = 64 for maximum throughput

This adaptive approach ensures optimal performance across the entire batch size spectrum.

**Section sources**
- [bench_all_versions.py:266-274](file://scripts/bench_all_versions.py#L266-L274)
- [bench_cuda_real.py:505-512](file://scripts/bench_cuda_real.py#L505-L512)

### Comprehensive Correctness Validation
The framework includes robust correctness validation:

- **Reference baseline**: Triton v5 implementation as the authoritative reference
- **Numerical tolerance**: Configurable absolute (1e-2) and relative (1e-2) tolerances
- **State validation**: Verification of internal state consistency alongside output accuracy
- **Multi-batch testing**: Validation across batch sizes 1, 16, 64 for comprehensive coverage
- **Error reporting**: Detailed error messages with maximum and mean differences

**Section sources**
- [bench_cuda_real.py:422-460](file://scripts/bench_cuda_real.py#L422-L460)
- [bench_cuda_real.py:474-492](file://scripts/bench_cuda_real.py#L474-L492)

### Volume Management and Dataset Generation
Enhanced volume management supports both synthetic and real-world datasets:

- **Synthetic workloads**: Automated generation of decode and prefill workloads
- **HF dataset integration**: Direct download from HuggingFace for contest data
- **Tensor optimization**: L2-normalization of k vectors to prevent state overflow
- **Safetensors integration**: Efficient storage and loading of auxiliary tensors

**Section sources**
- [setup_volume.py:32-57](file://scripts/setup_volume.py#L32-L57)
- [setup_volume.py:60-138](file://scripts/setup_volume.py#L60-L138)
- [setup_volume.py:180-202](file://scripts/setup_volume.py#L180-L202)

### Performance Measurement and Analysis
The benchmarking system provides comprehensive performance analysis:

- **Latency measurement**: Median timing with warmup and iteration controls
- **Bandwidth calculation**: Throughput analysis based on state memory access patterns
- **Version comparison**: Side-by-side performance analysis across kernel versions
- **Statistical validation**: Multiple iterations for reliable performance metrics
- **Resource utilization**: GPU properties and memory bandwidth analysis

**Section sources**
- [bench_all_versions.py:325-345](file://scripts/bench_all_versions.py#L325-L345)
- [bench_cuda_real.py:547-574](file://scripts/bench_cuda_real.py#L547-L574)

## Dependency Analysis
The unified benchmarking system introduces several key dependencies:

- **Modal runtime**: Cloud execution platform with B200 GPU support
- **CUDA 12.8**: Full B200 (sm_100) compatibility for advanced kernel features
- **CUTLASS**: CuTe DSL support for advanced kernel optimization
- **Triton**: Baseline implementation for correctness validation
- **PyTorch**: CUDA operations and tensor management
- **ctypes**: Python-C integration for CUDA library access
- **Tabulate**: Formatted result presentation

```mermaid
graph LR
BAV["bench_all_versions.py"] --> MODAL["Modal Runtime"]
BCR["bench_cuda_real.py"] --> MODAL
BC["build_cuda.py"] --> CUDA["CUDA 12.8"]
BC --> CUTLASS["CUTLASS Headers"]
BC --> NVCC["nvcc Compiler"]
BAV --> TRITON["Triton v5"]
BCR --> TRITON
BCR --> CTYPES["ctypes Library"]
BCR --> PYTORCH["PyTorch"]
BAV --> TABULATE["tabulate"]
BCR --> TABULATE
```

**Diagram sources**
- [bench_all_versions.py:17-27](file://scripts/bench_all_versions.py#L17-L27)
- [bench_cuda_real.py:9-17](file://scripts/bench_cuda_real.py#L9-L17)
- [build_cuda.py:18-34](file://scripts/build_cuda.py#L18-L34)

**Section sources**
- [bench_all_versions.py:17-27](file://scripts/bench_all_versions.py#L17-L27)
- [bench_cuda_real.py:9-17](file://scripts/bench_cuda_real.py#L9-L17)
- [build_cuda.py:18-34](file://scripts/build_cuda.py#L18-L34)

## Performance Considerations
The unified benchmarking system addresses several critical performance aspects:

- **Memory-bound optimization**: All kernels are highly memory-bound with focus on bandwidth utilization
- **Adaptive BLOCK_V sizing**: Intelligent grid configuration based on batch size for optimal occupancy
- **Precision trade-offs**: FP4/FP8 quantization provides significant bandwidth savings with controlled accuracy loss
- **TMA utilization**: Tensor Memory Accelerator enables efficient async state loading in CUDA kernels
- **Warp specialization**: v8 and later versions utilize specialized warp configurations for maximum throughput
- **CuTe optimization**: Advanced DSL and swizzling techniques optimize memory access patterns
- **CUDA Graph caching**: Low-latency kernel launching for repeated small-batch operations

**Section sources**
- [PERFORMANCE.md:1-158](file://docs/PERFORMANCE.md#L1-L158)
- [README.md:96-112](file://README.md#L96-L112)
- [gdn_decode_v7.cuh:1-200](file://src/kernels/cuda/gdn_decode_v7.cuh#L1-L200)
- [gdn_decode_v8.cuh:1-200](file://src/kernels/cuda/gdn_decode_v8.cuh#L1-L200)

## Troubleshooting Guide
Common issues and remedies in the unified benchmarking system:

- **Library not found**: Ensure CUDA library is built and available at `/data/lib/libgdn_kernels.so`
- **CUDA compilation errors**: Verify CUDA 12.8 installation and sm_100 architecture compatibility
- **Version not supported**: Check that requested kernel version is included in compiled library
- **Batch size limitations**: Some versions may not support very large batch sizes due to memory constraints
- **Precision issues**: Quantized kernels (FP4/FP8) may have different numerical behavior than FP32
- **TMA compatibility**: TMA features require compatible CUDA runtime and driver versions
- **Memory overflow**: Large batch sizes may exceed GPU memory limits, requiring reduced batch sizes

**Section sources**
- [build_cuda.py:50-56](file://scripts/build_cuda.py#L50-L56)
- [bench_cuda_real.py:51-56](file://scripts/bench_cuda_real.py#L51-L56)
- [bench_all_versions.py:295-314](file://scripts/bench_all_versions.py#L295-L314)

## Conclusion
The unified benchmarking framework represents a significant advancement in GDN kernel evaluation, providing comprehensive multi-version testing, real CUDA validation, and adaptive optimization capabilities. The system successfully bridges the gap between Triton baselines and production CUDA implementations, offering detailed performance analysis across the complete kernel evolution from v5 to v10.

Key achievements include:
- **Complete kernel coverage**: Support for all versions (v5-v10) with proper compilation and validation
- **Adaptive optimization**: Intelligent batch size and BLOCK_V sizing for optimal performance
- **Comprehensive validation**: Robust correctness checking against Triton baselines
- **Production readiness**: Real CUDA library compilation with external C interfaces
- **Scalable architecture**: Modular design supporting future kernel version additions

The framework enables precise performance characterization across different hardware configurations and provides actionable insights for kernel optimization and deployment decisions.

## Appendices

### Appendix A: Execution Commands
- **Multi-version benchmarking**: [scripts/bench_all_versions.py:6-8](file://scripts/bench_all_versions.py#L6-L8)
- **Real CUDA validation**: [scripts/bench_cuda_real.py:5-7](file://scripts/bench_cuda_real.py#L5-L7)
- **CUDA compilation**: [scripts/build_cuda.py:6-10](file://scripts/build_cuda.py#L6-L10)
- **Volume setup**: [scripts/setup_volume.py:5-7](file://scripts/setup_volume.py#L5-L7)

### Appendix B: Configuration Options
- **Multi-version testing**: `--versions` (v5,v6,v7,v8 or 'all'), `--batches` (1,16,64,256)
- **Benchmark parameters**: `--warmup` (default: 20), `--iters` (default: 200)
- **CUDA compilation**: Automatic nvcc compilation with -O3 and --use_fast_math flags
- **Kernel selection**: Automatic BLOCK_V sizing based on batch size
- **Library export**: ctypes-compatible function exports for Python integration

**Section sources**
- [bench_all_versions.py:10-15](file://scripts/bench_all_versions.py#L10-L15)
- [bench_all_versions.py:408-413](file://scripts/bench_all_versions.py#L408-L413)
- [build_cuda.py:335-347](file://scripts/build_cuda.py#L335-L347)
- [bench_cuda_real.py:408-413](file://scripts/bench_cuda_real.py#L408-L413)
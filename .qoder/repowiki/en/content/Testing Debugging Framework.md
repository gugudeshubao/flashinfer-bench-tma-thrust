# Testing Debugging Framework

<cite>
**Referenced Files in This Document**
- [README.md](file://README.md)
- [tests/test_correctness.py](file://tests/test_correctness.py)
- [tests/test_quantization_accuracy.py](file://tests/test_quantization_accuracy.py)
- [scripts/debug_prefill.py](file://scripts/debug_prefill.py)
- [scripts/debug_prefill2.py](file://scripts/debug_prefill2.py)
- [benchmarks/bench_modal.py](file://benchmarks/bench_modal.py)
- [scripts/setup_volume.py](file://scripts/setup_volume.py)
- [scripts/bench_all_versions.py](file://scripts/bench_all_versions.py)
- [src/kernels/cute_cpp/gdn_decode_v10.cuh](file://src/kernels/cute_cpp/gdn_decode_v10.cuh)
- [src/kernels/cuda/gdn_decode_v8.cuh](file://src/kernels/cuda/gdn_decode_v8.cuh)
- [src/kernels/ptx/gdn_decode_ptx.cuh](file://src/kernels/ptx/gdn_decode_ptx.cuh)
- [gdn_decode_qk4_v8_d128_k_last/baseline/triton/kernel.py](file://gdn_decode_qk4_v8_d128_k_last/baseline/triton/kernel.py)
- [gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py)
- [gdn_prefill_qk4_v8_d128_k_last/baseline/triton/kernel.py](file://gdn_prefill_qk4_v8_d128_k_last/baseline/triton/kernel.py)
- [gdn_decode_qk4_v8_d128_k_last/config.toml](file://gdn_decode_qk4_v8_d128_k_last/config.toml)
- [gdn_prefill_qk4_v8_d128_k_last/config.toml](file://gdn_prefill_qk4_v8_d128_k_last/config.toml)
</cite>

## Update Summary
**Changes Made**
- Enhanced verification procedures for kernel correctness across multiple framework versions
- Improved comprehensive validation against Triton baselines
- Added robust debugging tools with direct framework integration
- Strengthened multi-version benchmarking capabilities
- Expanded volume management with persistent trace datasets
- **Refactored test infrastructure from FP8-only testing to comprehensive multi-precision quantization accuracy testing supporting BF16, FP8, and FP4 modes**
- **Integrated Modal-based testing infrastructure for multi-precision validation**
- **Enhanced kernel implementations with native FP8/FP4/BF16 support in CUDA and CuTe kernels**
- **Added comprehensive quantization accuracy testing framework with statistical analysis**
- **Renamed test_fp8_accuracy.py to test_quantization_accuracy.py to reflect expanded capabilities**

## Table of Contents
1. [Introduction](#introduction)
2. [Framework Architecture](#framework-architecture)
3. [Testing Infrastructure](#testing-infrastructure)
4. [Debugging Tools](#debugging-tools)
5. [Benchmarking System](#benchmarking-system)
6. [Volume Management](#volume-management)
7. [Kernel Implementation Analysis](#kernel-implementation-analysis)
8. [Performance Validation](#performance-validation)
9. [Multi-Precision Quantization Testing](#multi-precision-quantization-testing)
10. [Troubleshooting Guide](#troubleshooting-guide)
11. [Best Practices](#best-practices)

## Introduction

The Testing Debugging Framework is a comprehensive system designed for validating and benchmarking Gated Delta Net (GDN) kernel implementations across multiple versions and architectures. This framework provides automated testing, debugging capabilities, and performance benchmarking specifically tailored for NVIDIA Blackwell B200 hardware with TMA Thrust submission optimizations.

The framework supports multiple kernel versions (v5 through v10) with different optimization strategies including Triton-based kernels, CUDA implementations, and CuTe DSL layouts. It includes sophisticated correctness testing, performance benchmarking, and debugging tools that enable developers to validate kernel implementations against reference baselines with enhanced verification procedures.

**Updated** Enhanced with improved verification procedures for kernel correctness across multiple framework versions, comprehensive validation against Triton baselines, and **comprehensive multi-precision quantization accuracy testing capabilities supporting BF16, FP8, and FP4 formats** with integrated Modal-based testing infrastructure.

## Framework Architecture

The testing framework follows a modular architecture with distinct components for testing, debugging, benchmarking, and volume management:

```mermaid
graph TB
subgraph "Core Framework"
A[Test Scripts] --> B[Benchmark Runner]
B --> C[Volume Manager]
C --> D[Trace Sets]
D --> E[Evaluation Engine]
end
subgraph "Testing Layer"
F[Correctness Tests] --> G[Gate Broadcast Tests]
G --> H[Multi-batch Validation]
F --> I[Multi-Precision Quantization Tests]
end
subgraph "Debugging Layer"
J[Reference Comparison] --> K[Framework Evaluation]
K --> L[Error Analysis]
end
subgraph "Kernel Implementations"
M[Triton Kernels] --> N[CUDA Kernels]
N --> O[CuTe DSL Kernels]
M --> P[PTX Assembly]
end
F --> A
J --> B
M --> E
N --> E
O --> E
P --> E
I --> A
```

**Diagram sources**
- [tests/test_correctness.py:1-363](file://tests/test_correctness.py#L1-L363)
- [tests/test_quantization_accuracy.py:1-361](file://tests/test_quantization_accuracy.py#L1-L361)
- [scripts/debug_prefill.py:1-306](file://scripts/debug_prefill.py#L1-L306)
- [benchmarks/bench_modal.py:1-330](file://benchmarks/bench_modal.py#L1-L330)

The architecture consists of four primary layers:

1. **Testing Infrastructure**: Automated correctness validation and regression testing with enhanced verification procedures, including **multi-precision quantization testing with BF16, FP8, and FP4 formats**
2. **Debugging Tools**: Interactive debugging and comparison utilities with direct framework integration
3. **Benchmarking System**: Performance measurement and comparison frameworks with comprehensive validation
4. **Volume Management**: Persistent storage and trace set management with comprehensive dataset support

## Testing Infrastructure

The testing infrastructure provides comprehensive validation of GDN kernel implementations through multiple test scenarios with enhanced verification procedures:

### Enhanced Correctness Testing Framework

The primary testing mechanism validates kernel implementations against PyTorch reference implementations with improved verification:

```mermaid
sequenceDiagram
participant Test as Test Script
participant Ref as Reference Impl
participant Triton as Triton Kernel
participant CUDA as CUDA Kernel
participant Results as Validation Results
Test->>Ref : Generate reference outputs
Test->>Triton : Execute kernel implementation
Test->>CUDA : Execute optimized implementation
Ref-->>Results : Reference outputs
Triton-->>Results : Triton outputs
CUDA-->>Results : CUDA outputs
Results->>Results : Compare differences with tolerance thresholds
Results-->>Test : Enhanced validation status
```

**Diagram sources**
- [tests/test_correctness.py:29-277](file://tests/test_correctness.py#L29-L277)

### Advanced Test Categories

The framework implements several enhanced test categories:

1. **Reference vs Implementation**: Direct comparison between PyTorch reference and kernel implementations with configurable tolerance thresholds
2. **Gate Broadcast Verification**: Ensures proper broadcasting of gate values across thread blocks with comprehensive statistical analysis
3. **Multi-batch Validation**: Tests across different batch sizes (1, 4, 16, 64) with adaptive BLOCK_V selection
4. **Block Size Testing**: Validates different BLOCK_V configurations (16, 32, 64) with performance impact analysis
5. **Framework Integration Testing**: Direct integration with flashinfer-bench evaluation system for comprehensive validation
6. **Multi-Precision Quantization Testing**: **Comprehensive validation of quantization accuracy across BF16, FP8, and FP4 formats with Modal-based testing**

**Section sources**
- [tests/test_correctness.py:186-277](file://tests/test_correctness.py#L186-L277)

## Debugging Tools

The debugging framework provides interactive tools for kernel validation and comparison with enhanced verification procedures:

### Interactive Debug Scripts

Two primary debugging scripts offer different approaches to kernel validation with comprehensive comparison capabilities:

```mermaid
flowchart TD
A[Debug Script] --> B{Execution Mode}
B --> |Direct Comparison| C[Reference vs Kernel]
B --> |Framework Integration| D[FlashInfer Framework]
C --> E[Manual Input Generation]
C --> F[Output Statistics]
C --> G[Difference Analysis]
D --> H[Trace Set Loading]
D --> I[Solution Building]
D --> J[Evaluation Comparison]
E --> K[Statistical Analysis]
F --> K
G --> K
H --> K
I --> K
J --> K
```

**Diagram sources**
- [scripts/debug_prefill.py:14-306](file://scripts/debug_prefill.py#L14-L306)
- [scripts/debug_prefill2.py:23-184](file://scripts/debug_prefill2.py#L23-L184)

### Enhanced Debug Capabilities

The debugging tools provide:

1. **Reference Implementation**: PyTorch-based reference for ground truth validation with comprehensive statistical analysis
2. **Framework Integration**: Direct integration with flashinfer-bench evaluation system for comprehensive validation
3. **Statistical Analysis**: Comprehensive output statistics and difference metrics with configurable tolerance thresholds
4. **Trace Set Validation**: Integration with persistent trace datasets for reproducible testing
5. **Direct Framework Comparison**: Ability to compare custom solutions against baseline implementations within the framework

**Section sources**
- [scripts/debug_prefill.py:14-306](file://scripts/debug_prefill.py#L14-L306)
- [scripts/debug_prefill2.py:23-184](file://scripts/debug_prefill2.py#L23-L184)

## Benchmarking System

The benchmarking system provides comprehensive performance evaluation across multiple kernel versions and configurations with enhanced validation procedures:

### Multi-Version Benchmarking

The framework supports benchmarking across all GDN kernel versions with comprehensive validation:

```mermaid
graph LR
subgraph "Kernel Versions"
A[v5 - Baseline] --> B[v6 - TMA Simulation]
B --> C[v7 - FP4 Simulation]
C --> D[v8 - FP8 Simulation]
D --> E[v9 - CuTe Swizzle]
E --> F[v10 - TMA/CuTe]
end
subgraph "Benchmark Parameters"
G[Batch Sizes] --> H[1, 4, 16, 64, 256]
I[Block Sizes] --> J[16, 32, 64]
K[Precision] --> L[FP32, FP16, BF16, FP8, FP4]
end
A --> G
B --> H
C --> I
D --> J
E --> K
F --> L
```

**Diagram sources**
- [scripts/bench_all_versions.py:32-444](file://scripts/bench_all_versions.py#L32-L444)

### Enhanced Benchmark Configuration

The benchmarking system supports:

1. **Adaptive Block Sizes**: Automatically selects optimal BLOCK_V based on batch size with performance optimization
2. **Multiple Precision Modes**: Supports FP32, FP16, and simulated BF16/FP4/FP8 compression with bandwidth utilization analysis
3. **Performance Metrics**: Measures execution time, bandwidth utilization, and state memory usage with comprehensive reporting
4. **Statistical Analysis**: Provides median timing and bandwidth calculations with confidence intervals
5. **Framework Integration**: Direct integration with flashinfer-bench for comprehensive validation against baselines

**Section sources**
- [scripts/bench_all_versions.py:260-404](file://scripts/bench_all_versions.py#L260-L404)

## Volume Management

The volume management system handles persistent storage and trace set organization with comprehensive dataset support:

### Enhanced Trace Set Structure

```mermaid
graph TB
A[Trace Volume] --> B[Definitions]
A --> C[Workloads]
A --> D[Solutions]
A --> E[Traces]
B --> F[gdn_decode_qk4_v8_d128_k_last.json]
B --> G[gdn_prefill_qk4_v8_d128_k_last.json]
C --> H[gdn_decode_workloads.jsonl]
C --> I[gdn_prefill_workloads.jsonl]
D --> J[Solution Configurations]
E --> K[Benchmark Results]
```

**Diagram sources**
- [scripts/setup_volume.py:141-173](file://scripts/setup_volume.py#L141-L173)

### Advanced Volume Operations

The system provides:

1. **Synthetic Workload Generation**: Creates realistic test workloads with proper tensor distributions and L2 normalization for stability
2. **HuggingFace Integration**: Downloads official contest datasets for comprehensive validation
3. **Persistent Storage**: Maintains trace sets across benchmark runs with automatic commit
4. **Volume Commit**: Ensures data persistence and availability with comprehensive dataset management
5. **Comprehensive Dataset Support**: Supports both synthetic and real-world datasets for thorough validation

**Section sources**
- [scripts/setup_volume.py:32-138](file://scripts/setup_volume.py#L32-L138)

## Kernel Implementation Analysis

The framework supports multiple kernel implementation strategies, each optimized for different use cases with comprehensive validation:

### Triton Kernel Architecture

The Triton-based kernels provide flexible, auto-tuned implementations with enhanced verification:

```mermaid
classDiagram
class TritonKernel {
+float32 scale
+int D
+int BLOCK_V
+adaptive_block_selection()
+gate_computation()
+delta_rule_update()
+output_generation()
}
class ReferenceKernel {
+pytorch_reference()
+manual_implementation()
+einsum_operations()
+broadcast_gates()
}
class OptimizedKernel {
+vectorized_loads()
+shared_memory_optimization()
+warp_shuffles()
+bank_conflict_avoidance()
}
TritonKernel --> ReferenceKernel : "validates against"
TritonKernel --> OptimizedKernel : "implements optimizations"
```

**Diagram sources**
- [gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py:23-136](file://gdn_decode_qk4_v8_d128_k_last/solution/triton/kernel.py#L23-L136)
- [gdn_decode_qk4_v8_d128_k_last/baseline/triton/kernel.py:27-101](file://gdn_decode_qk4_v8_d128_k_last/baseline/triton/kernel.py#L27-L101)

### Enhanced CUDA Kernel Optimizations

The CUDA implementations leverage advanced GPU optimization techniques with comprehensive validation:

**Section sources**
- [src/kernels/cute_cpp/gdn_decode_v10.cuh:67-218](file://src/kernels/cute_cpp/gdn_decode_v10.cuh#L67-L218)

### PTX Assembly Integration

The PTX assembly kernels provide low-level optimization for FP8 quantization operations:

**Section sources**
- [src/kernels/ptx/gdn_decode_ptx.cuh:221-258](file://src/kernels/ptx/gdn_decode_ptx.cuh#L221-L258)

## Performance Validation

The framework provides comprehensive performance validation across different hardware configurations and kernel versions with enhanced verification procedures:

### Enhanced Performance Metrics

The system tracks multiple performance indicators with comprehensive validation:

1. **Throughput Measurements**: Bandwidth utilization in GB/s with peak utilization analysis
2. **Latency Analysis**: Execution time in milliseconds with statistical analysis
3. **Memory Usage**: State memory footprint calculations with compression ratio analysis
4. **Speedup Analysis**: Comparative performance against baseline implementations with confidence intervals
5. **Framework Validation**: Direct comparison against Triton baselines with comprehensive error analysis

### Hardware-Specific Optimizations

The framework accounts for B200-specific optimizations with comprehensive validation:

```mermaid
graph LR
A[Hardware Constraints] --> B[Memory Bandwidth]
A --> C[Compute Capability]
A --> D[Architecture Features]
B --> E[95% Peak Utilization]
C --> F[TMA Thrust Support]
D --> G[Smem Swizzling]
E --> H[Optimized Kernels]
F --> H
G --> H
```

**Diagram sources**
- [README.md:144-151](file://README.md#L144-L151)

**Section sources**
- [README.md:14-28](file://README.md#L14-L28)

## Multi-Precision Quantization Testing

**New Section** The framework now includes comprehensive multi-precision quantization testing capabilities with detailed simulation and validation procedures across BF16, FP8, and FP4 formats.

### Quantization Testing Architecture

The multi-precision accuracy testing framework provides detailed validation of quantization accuracy across multiple iterations and precision formats:

```mermaid
sequenceDiagram
participant Test as Multi-Precision Test
participant Quant as Quantization Module
participant GDN as GDN Decode
participant Stats as Statistical Analysis
Test->>Quant : Initialize quantization (BF16/FP8/FP4)
Test->>GDN : Run decode with FP32 state
Test->>GDN : Run decode with quantized state
Quant->>GDN : Dequantize state to FP32
GDN->>Stats : Compare outputs and states
Stats->>Stats : Calculate error metrics
Stats-->>Test : Report accuracy results
```

**Diagram sources**
- [tests/test_quantization_accuracy.py:152-179](file://tests/test_quantization_accuracy.py#L152-L179)

### Quantization Methods

The framework supports three primary quantization formats:

1. **BF16 Quantization**:
   - 1 sign bit, 8 exponent bits, 7 mantissa bits
   - Same range as FP32, but lower precision (~0.8% relative error)
   - 2x memory compression ratio
   - No scaling needed (same range as FP32)
   - **Native support in CUDA v8/v10 kernels with vectorized memory operations**

2. **FP8 E4M3 Quantization**:
   - 1 sign bit, 4 exponent bits, 3 mantissa bits
   - Range: [-448, 448], smallest subnormal: 2^-9
   - 4x memory compression ratio
   - Simulated using PyTorch with per-row dynamic scaling
   - **Native FP8 support in CUDA v8/v10 kernels with dedicated FP8 packing/unpacking functions**

3. **FP4 E2M1 Quantization**:
   - 1 sign bit, 2 exponent bits, 1 mantissa bit
   - Lookup table approach with 8 representable values
   - Range: [-6, 6], 8x memory compression ratio
   - Directly compatible with v7 kernel implementation
   - **Enhanced with CUDA v10 support including lookup tables and vectorized operations**
   - **Native FP4 support in CUDA v10 kernels with constant memory lookup tables**

### Test Methodology

The multi-precision accuracy testing framework implements comprehensive validation procedures:

```mermaid
flowchart TD
A[Multi-Precision Test] --> B[Initialize State]
B --> C[Generate Inputs]
C --> D[Run FP32 Reference]
D --> E[Run Quantized Version]
E --> F[Calculate Errors]
F --> G[Track Metrics]
G --> H[Analyze Accumulation]
H --> I[Compare Precision Formats]
I --> J[Report Results]
```

**Diagram sources**
- [tests/test_quantization_accuracy.py:186-293](file://tests/test_quantization_accuracy.py#L186-L293)

### Key Features

1. **Multi-step Validation**: Tests accuracy over 100+ decode iterations to detect error accumulation
2. **Statistical Analysis**: Tracks maximum absolute errors, mean absolute errors, and relative errors
3. **Error Accumulation Detection**: Monitors whether quantization errors grow over time
4. **Modal Integration**: Uses Modal platform for consistent B200 GPU testing environment
5. **Configurable Parameters**: Adjustable batch sizes, dimensions, and iteration counts
6. **Realistic Input Generation**: Creates normalized random inputs with realistic gate values
7. **Precision Comparison**: Direct comparison between BF16, FP8, and FP4 formats

### Quantization Implementation Details

The framework provides detailed quantization implementations:

**BF16 Quantization**:
- Direct conversion to bfloat16 format using CUDA native types
- No scaling required (same range as FP32)
- Minimal precision loss (~0.8% relative error)
- **Vectorized memory operations for 2x compression efficiency**

**FP8 E4M3 Quantization**:
- Per-row dynamic scaling with safety margins
- Rounding to nearest representable FP8 value using CUDA native FP8 types
- Proper handling of zeros and subnormal values
- 3 mantissa bits providing 8 levels per octave
- **Dedicated packing/unpacking functions for vectorized memory access**

**FP4 E2M1 Quantization**:
- Lookup table approach for consistent quantization
- Direct mapping to 8 representable values using constant memory
- Compatible with existing v7 kernel implementation
- **Enhanced with CUDA v10 support including lookup tables and vectorized operations**
- Optimized for memory bandwidth efficiency with 8 FP4 values packed into 32-bit integers

**Section sources**
- [tests/test_quantization_accuracy.py:16-124](file://tests/test_quantization_accuracy.py#L16-L124)
- [tests/test_quantization_accuracy.py:186-293](file://tests/test_quantization_accuracy.py#L186-L293)
- [src/kernels/cute_cpp/gdn_decode_v10.cuh:90-158](file://src/kernels/cute_cpp/gdn_decode_v10.cuh#L90-L158)

## Troubleshooting Guide

Common issues and their resolutions with enhanced diagnostic capabilities:

### Compilation Issues

1. **Missing Dependencies**: Ensure all required packages are installed with version compatibility checks
2. **CUDA Version Compatibility**: Verify CUDA 12.8+ compatibility with framework integration
3. **Triton Installation**: Confirm Triton 3.0+ installation with validation procedures
4. **Modal Environment**: Ensure Modal CLI is properly configured for cloud testing

### Runtime Errors

1. **Memory Allocation Failures**: Check available GPU memory with utilization monitoring
2. **Kernel Launch Failures**: Verify grid/block dimensions with validation procedures
3. **Data Type Mismatches**: Ensure proper tensor dtype conversions with comprehensive type checking
4. **Quantization Accuracy Errors**: Verify proper scaling factors and quantization ranges

### Performance Issues

1. **Low Bandwidth Utilization**: Check BLOCK_V selection with adaptive optimization analysis
2. **Memory Access Patterns**: Verify coalesced memory access with validation tools
3. **Occupancy Problems**: Adjust thread block sizes with performance impact analysis
4. **Quantization Accuracy Loss**: Monitor error accumulation rates and adjust precision

### Enhanced Diagnostic Procedures

1. **Framework Integration**: Use flashinfer-bench evaluation system for comprehensive validation
2. **Statistical Analysis**: Leverage comprehensive statistical comparisons with tolerance thresholds
3. **Trace Set Validation**: Utilize persistent trace datasets for reproducible testing
4. **Direct Comparison**: Compare custom solutions against baseline implementations with validation tools
5. **Multi-Precision Testing**: Use dedicated quantization tests to validate effects across BF16, FP8, and FP4 formats
6. **Modal Platform**: Leverage Modal for consistent cloud-based testing environment

**Section sources**
- [benchmarks/bench_modal.py:115-120](file://benchmarks/bench_modal.py#L115-L120)
- [scripts/setup_volume.py:141-145](file://scripts/setup_volume.py#L141-L145)
- [tests/test_quantization_accuracy.py:306-323](file://tests/test_quantization_accuracy.py#L306-L323)

## Best Practices

### Enhanced Testing Strategy

1. **Multi-Level Validation**: Combine unit tests with integration tests and framework validation
2. **Regression Testing**: Maintain test suites for all kernel versions with comprehensive coverage
3. **Performance Baselines**: Establish performance baselines for each version with validation procedures
4. **Framework Integration**: Leverage flashinfer-bench for comprehensive validation against baselines
5. **Statistical Analysis**: Use comprehensive statistical comparisons with configurable tolerance thresholds
6. **Multi-Precision Testing**: Include BF16/FP8/FP4 accuracy tests in all performance validation cycles
7. **Error Accumulation Monitoring**: Regularly test quantization accuracy over extended operation sequences
8. **Precision Comparison**: Always test quantization effects across multiple precision formats

### Advanced Debugging Approach

1. **Incremental Testing**: Test simpler cases before complex scenarios with validation procedures
2. **Statistical Analysis**: Use comprehensive statistical comparisons with error threshold analysis
3. **Trace Set Validation**: Leverage persistent trace datasets for reproducible testing
4. **Framework Integration**: Use direct framework comparison for comprehensive validation
5. **Modal Testing**: Utilize Modal platform for consistent cloud-based testing environment

### Enhanced Benchmarking Guidelines

1. **Consistent Parameters**: Use standardized test parameters with validation procedures
2. **Multiple Runs**: Execute multiple iterations for reliable metrics with statistical analysis
3. **Hardware Characterization**: Account for specific hardware capabilities with comprehensive testing
4. **Framework Validation**: Use flashinfer-bench for comprehensive validation against baselines
5. **Performance Analysis**: Include comprehensive performance analysis with utilization metrics
6. **Multi-Precision Validation**: Always include BF16/FP8/FP4 accuracy testing in performance validation
7. **Error Tracking**: Monitor both immediate accuracy and long-term error accumulation

### Multi-Precision Quantization Best Practices

1. **Proper Scaling**: Use per-row dynamic scaling with appropriate safety margins
2. **Error Monitoring**: Track both absolute and relative errors during quantization
3. **Iteration Testing**: Test quantization accuracy over multiple decode steps
4. **Modal Validation**: Use Modal platform for consistent B200 GPU testing conditions
5. **Parameter Tuning**: Adjust quantization parameters based on accuracy requirements
6. **Fallback Strategies**: Implement graceful degradation when quantization accuracy is insufficient
7. **Precision Selection**: Choose appropriate precision format based on accuracy and performance trade-offs
8. **Native Support**: Leverage native CUDA FP8/FP4/BF16 support for optimal performance

**Section sources**
- [tests/test_quantization_accuracy.py:186-293](file://tests/test_quantization_accuracy.py#L186-L293)
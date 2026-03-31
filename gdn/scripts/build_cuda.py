#!/usr/bin/env python3
"""
Build GDN CUDA kernels on Modal B200.

Usage:
    modal run scripts/build_cuda.py

This compiles the CUDA kernels with nvcc and uploads
the shared library to the Modal volume.
"""

import os
import sys
import pathlib

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modal_config import app, cuda_image, volume, B200_GPU, BUILD_TIMEOUT, SHORT_TIMEOUT

PROJECT_ROOT = pathlib.Path(__file__).parent.parent

def get_kernel_sources():
    """Read all kernel source files from subdirectories."""
    sources = {}
    kernels_dir = PROJECT_ROOT / "src" / "kernels"
    
    # Read CUDA kernels (v5, v6, v7, v8)
    cuda_dir = kernels_dir / "cuda"
    for cuh in cuda_dir.glob("gdn_*.cuh"):
        sources[cuh.name] = cuh.read_text()
    
    # Read CuTe kernels (v9, v10)
    cute_dir = kernels_dir / "cute"
    for cuh in cute_dir.glob("gdn_*.cuh"):
        sources[cuh.name] = cuh.read_text()
    
    return sources

# Pre-read kernel sources (executed locally)
KERNEL_SOURCES = get_kernel_sources()


@app.function(
    image=cuda_image,
    gpu=B200_GPU,
    timeout=BUILD_TIMEOUT,
    volumes={"/data": volume},
)
def build_cuda_kernels(kernel_sources: dict = None):
    """Compile GDN CUDA kernels for B200 (sm_100)."""
    import subprocess
    from pathlib import Path
    
    if kernel_sources is None:
        kernel_sources = {}
    
    build_dir = Path("/tmp/gdn_build")
    output_dir = Path("/data/lib")
    
    build_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write kernel sources to build directory
    for name, content in kernel_sources.items():
        (build_dir / name).write_text(content)
        print(f"Wrote: {name}")
    
    # Check CUDA version
    result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    print("\nNVCC Version:")
    print(result.stdout)
    
    # Create combined source file
    combined_cu = build_dir / "gdn_kernels.cu"
    with open(combined_cu, "w") as f:
        f.write('#include <cuda_runtime.h>\n')
        f.write('#include <cuda_bf16.h>\n')
        f.write('#include <cuda_fp16.h>\n')
        f.write('#include <cuda_fp8.h>\n\n')
        
        # Include all headers (v5-v10)
        for version in ["v5", "v6", "v7", "v8", "v9", "v10"]:
            for name in sorted(kernel_sources.keys()):
                if f"_{version}.cuh" in name:
                    print(f"Including: {name}")
                    f.write(f"// ====== {name} ======\n")
                    f.write(kernel_sources[name])
                    f.write("\n\n")
        
        # Add extern "C" wrappers for ctypes access
        f.write('''
// ============================================================
// Extern C wrappers for Python ctypes access
// ============================================================

extern "C" {

void gdn_decode_v7_fp32(
    const void* Q, const void* K, const void* V,
    const void* State, const void* A_log, const void* A,
    const void* DtBias, const void* B_gate,
    void* Out, void* NewState,
    float scale, int B, int num_v_heads, int D,
    int stride_q_b, int stride_q_h, int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h, int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b, int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, void* stream
) {
    gdn::gdn_decode_v7_launch_fp32(
        Q, K, V, State, A_log, A, DtBias, B_gate,
        Out, NewState, scale, B, num_v_heads, D,
        stride_q_b, stride_q_h, stride_k_b, stride_k_h,
        stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
        stride_a_b, stride_b_b, stride_o_b, stride_o_h,
        stride_ns_b, stride_ns_h, stride_ns_v,
        BLOCK_V, (cudaStream_t)stream
    );
}

void gdn_decode_v8_fp32(
    const void* Q, const void* K, const void* V,
    const void* State, const void* A_log, const void* A,
    const void* DtBias, const void* B_gate,
    void* Out, void* NewState,
    float scale, int B, int num_v_heads, int D,
    int stride_q_b, int stride_q_h, int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h, int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b, int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, void* stream
) {
    gdn::gdn_decode_v8_launch_fp32(
        Q, K, V, State, A_log, A, DtBias, B_gate,
        Out, NewState, scale, B, num_v_heads, D,
        stride_q_b, stride_q_h, stride_k_b, stride_k_h,
        stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
        stride_a_b, stride_b_b, stride_o_b, stride_o_h,
        stride_ns_b, stride_ns_h, stride_ns_v,
        BLOCK_V, (cudaStream_t)stream
    );
}

// ============================================================
// CUDA Graph wrapper for low-latency small batch
// ============================================================

static cudaGraph_t g_graph_v7 = nullptr;
static cudaGraphExec_t g_graph_exec_v7 = nullptr;
static int g_graph_batch_v7 = -1;
static int g_graph_block_v7 = -1;

// Cached kernel node params for graph update
static cudaKernelNodeParams g_node_params_v7;
static cudaGraphNode_t g_kernel_node_v7 = nullptr;

void gdn_decode_v7_graph_launch(
    const void* Q, const void* K, const void* V,
    const void* State, const void* A_log, const void* A,
    const void* DtBias, const void* B_gate,
    void* Out, void* NewState,
    float scale, int B, int num_v_heads, int D,
    int stride_q_b, int stride_q_h, int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h, int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b, int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    
    // If graph doesn't exist or config changed, create new graph
    if (g_graph_exec_v7 == nullptr || g_graph_batch_v7 != B || g_graph_block_v7 != BLOCK_V) {
        // Destroy old graph
        if (g_graph_exec_v7 != nullptr) {
            cudaGraphExecDestroy(g_graph_exec_v7);
            cudaGraphDestroy(g_graph_v7);
            g_graph_exec_v7 = nullptr;
            g_graph_v7 = nullptr;
        }
        
        // Capture new graph
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        
        gdn::gdn_decode_v7_launch_fp32(
            Q, K, V, State, A_log, A, DtBias, B_gate,
            Out, NewState, scale, B, num_v_heads, D,
            stride_q_b, stride_q_h, stride_k_b, stride_k_h,
            stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
            stride_a_b, stride_b_b, stride_o_b, stride_o_h,
            stride_ns_b, stride_ns_h, stride_ns_v,
            BLOCK_V, s
        );
        
        cudaStreamEndCapture(s, &g_graph_v7);
        cudaGraphInstantiate(&g_graph_exec_v7, g_graph_v7, nullptr, nullptr, 0);
        
        g_graph_batch_v7 = B;
        g_graph_block_v7 = BLOCK_V;
    }
    
    // Launch cached graph (~1μs instead of ~5-10μs)
    cudaGraphLaunch(g_graph_exec_v7, s);
}

void gdn_decode_v7_graph_destroy() {
    if (g_graph_exec_v7 != nullptr) {
        cudaGraphExecDestroy(g_graph_exec_v7);
        cudaGraphDestroy(g_graph_v7);
        g_graph_exec_v7 = nullptr;
        g_graph_v7 = nullptr;
        g_graph_batch_v7 = -1;
    }
}

// v9 CuTe/TMA wrapper
void gdn_decode_v9_fp32(
    const void* Q, const void* K, const void* V,
    const void* State, const void* A_log, const void* A,
    const void* DtBias, const void* B_gate,
    void* Out, void* NewState,
    float scale, int B, int num_v_heads, int D,
    int stride_q_b, int stride_q_h, int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h, int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b, int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, void* stream
) {
    gdn::gdn_decode_v9_launch_fp32(
        Q, K, V, State, A_log, A, DtBias, B_gate,
        Out, NewState, scale, B, num_v_heads, D,
        stride_q_b, stride_q_h, stride_k_b, stride_k_h,
        stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
        stride_a_b, stride_b_b, stride_o_b, stride_o_h,
        stride_ns_b, stride_ns_h, stride_ns_v,
        BLOCK_V, (cudaStream_t)stream
    );
}

void gdn_decode_v9_tma(
    const void* Q, const void* K, const void* V,
    const void* State, const void* A_log, const void* A,
    const void* DtBias, const void* B_gate,
    void* Out, void* NewState,
    float scale, int B, int num_v_heads, int D,
    int stride_q_b, int stride_q_h, int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h, int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b, int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, void* stream
) {
    gdn::gdn_decode_v9_launch_tma(
        Q, K, V, State, A_log, A, DtBias, B_gate,
        Out, NewState, scale, B, num_v_heads, D,
        stride_q_b, stride_q_h, stride_k_b, stride_k_h,
        stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
        stride_a_b, stride_b_b, stride_o_b, stride_o_h,
        stride_ns_b, stride_ns_h, stride_ns_v,
        BLOCK_V, (cudaStream_t)stream
    );
}

// v10 CuTe DSL wrapper
void gdn_decode_v10_cute(
    const void* Q, const void* K, const void* V,
    const void* State, const void* A_log, const void* A,
    const void* DtBias, const void* B_gate,
    void* Out, void* NewState,
    float scale, int B, int num_v_heads, int D,
    int stride_q_b, int stride_q_h, int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h, int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b, int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, void* stream
) {
    gdn::gdn_decode_v10_launch_cute(
        Q, K, V, State, A_log, A, DtBias, B_gate,
        Out, NewState, scale, B, num_v_heads, D,
        stride_q_b, stride_q_h, stride_k_b, stride_k_h,
        stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
        stride_a_b, stride_b_b, stride_o_b, stride_o_h,
        stride_ns_b, stride_ns_h, stride_ns_v,
        BLOCK_V, (cudaStream_t)stream
    );
}

void gdn_decode_v10_tma(
    const void* Q, const void* K, const void* V,
    const void* State, const void* A_log, const void* A,
    const void* DtBias, const void* B_gate,
    void* Out, void* NewState,
    float scale, int B, int num_v_heads, int D,
    int stride_q_b, int stride_q_h, int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h, int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b, int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, void* stream
) {
    gdn::gdn_decode_v10_launch_tma(
        Q, K, V, State, A_log, A, DtBias, B_gate,
        Out, NewState, scale, B, num_v_heads, D,
        stride_q_b, stride_q_h, stride_k_b, stride_k_h,
        stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
        stride_a_b, stride_b_b, stride_o_b, stride_o_h,
        stride_ns_b, stride_ns_h, stride_ns_v,
        BLOCK_V, (cudaStream_t)stream
    );
}

}  // extern "C"
''')
    
    # Compile with nvcc
    output_so = output_dir / "libgdn_kernels.so"
    
    compile_cmd = [
        "nvcc",
        "-O3",
        "--use_fast_math",
        "-arch=sm_100",  # B200
        "-Xcompiler", "-fPIC",
        "-shared",
        "-std=c++17",  # CuTe requires C++17
        "-o", str(output_so),
        str(combined_cu),
        "-I", str(build_dir),
        "-I", "/opt/cutlass/include",  # CUTLASS headers for CuTe
    ]
    
    print(f"\nCompiling: {' '.join(compile_cmd)}")
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("COMPILE ERROR:")
        print(result.stderr)
        return {"status": "error", "error": result.stderr}
    
    print(f"\nSuccess! Output: {output_so}")
    print(f"Size: {output_so.stat().st_size / 1024:.1f} KB")
    
    # Verify the shared library
    result = subprocess.run(["nm", "-D", str(output_so)], capture_output=True, text=True)
    print("\nExported symbols:")
    for line in result.stdout.split("\n"):
        if "gdn_" in line:
            print(f"  {line.split()[-1]}")
    
    volume.commit()
    
    return {
        "status": "success",
        "output": str(output_so),
        "size_kb": output_so.stat().st_size / 1024,
    }


@app.function(
    image=cuda_image,
    gpu=B200_GPU,
    timeout=SHORT_TIMEOUT,
    volumes={"/data": volume},
)
def test_cuda_library():
    """Test loading the compiled CUDA library."""
    import ctypes
    from pathlib import Path
    
    lib_path = Path("/data/lib/libgdn_kernels.so")
    
    if not lib_path.exists():
        return {"status": "error", "error": "Library not found. Run build first."}
    
    try:
        lib = ctypes.CDLL(str(lib_path))
        print(f"Loaded: {lib_path}")
        
        # Check for exported functions
        functions = [
            "gdn_decode_v7_fp32",
            "gdn_decode_v7_fp4",
            "gdn_prefill_v7_fp32",
            "gdn_prefill_v7_fp4",
        ]
        
        found = []
        for func in functions:
            if hasattr(lib, func):
                found.append(func)
                print(f"  Found: {func}")
        
        return {"status": "success", "functions": found}
        
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.local_entrypoint()
def main(action: str = "build"):
    """
    Build or test CUDA kernels.
    
    Args:
        action: "build" to compile, "test" to verify
    """
    if action == "build":
        print(f"Found {len(KERNEL_SOURCES)} kernel files:")
        for name in sorted(KERNEL_SOURCES.keys()):
            print(f"  {name}")
        result = build_cuda_kernels.remote(kernel_sources=KERNEL_SOURCES)
    elif action == "test":
        result = test_cuda_library.remote()
    else:
        print(f"Unknown action: {action}")
        return
    
    print(f"\nResult: {result}")

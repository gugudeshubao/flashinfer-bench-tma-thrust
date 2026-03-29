/*
 * GDN Kernels — Combined CUDA source for compilation
 *
 * This file includes all GDN kernel headers and provides
 * C-linkage exports for Python FFI.
 *
 * Build: cmake -B build && cmake --build build
 * Result: libgdn_kernels.so
 */

#include "gdn_decode_v7.cuh"
#include "gdn_prefill_v7.cuh"
#include "gdn_decode_v8.cuh"
#include "gdn_prefill_v8.cuh"

// Also include v5/v6 for backward compatibility
#include "gdn_decode_v5.cuh"
#include "gdn_prefill_v5.cuh"
#include "gdn_decode_v6.cuh"
#include "gdn_prefill_v6.cuh"

// ============================================================
// C-linkage exports for Python ctypes/FFI
// ============================================================

extern "C" {

// v7 FP32 launchers
void gdn_decode_v7_fp32(
    const void* Q, const void* K, const void* V, const void* State,
    const void* A_log, const void* A, const void* DtBias, const void* B_gate,
    void* Out, void* NewState,
    float scale, int B, int num_v_heads, int D,
    int stride_q_b, int stride_q_h, int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h, int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b, int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, cudaStream_t stream
) {
    gdn::gdn_decode_v7_launch_fp32(
        Q, K, V, State, A_log, A, DtBias, B_gate, Out, NewState,
        scale, B, num_v_heads, D,
        stride_q_b, stride_q_h, stride_k_b, stride_k_h,
        stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
        stride_a_b, stride_b_b, stride_o_b, stride_o_h,
        stride_ns_b, stride_ns_h, stride_ns_v,
        BLOCK_V, stream
    );
}

void gdn_prefill_v7_fp32(
    const void* Q, const void* K, const void* V, const void* State,
    const void* A_log, const void* A, const void* DtBias, const void* B_gate,
    const void* CuSeqlens,
    void* Out, void* NewState,
    float scale, int N, int num_v_heads, int D,
    int stride_q_t, int stride_q_h, int stride_k_t, int stride_k_h,
    int stride_v_t, int stride_v_h, int stride_s_n, int stride_s_h, int stride_s_v,
    int stride_a_t, int stride_b_t, int stride_o_t, int stride_o_h,
    int stride_ns_n, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, cudaStream_t stream
) {
    gdn::gdn_prefill_v7_launch_fp32(
        Q, K, V, State, A_log, A, DtBias, B_gate, CuSeqlens, Out, NewState,
        scale, N, num_v_heads, D,
        stride_q_t, stride_q_h, stride_k_t, stride_k_h,
        stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v,
        stride_a_t, stride_b_t, stride_o_t, stride_o_h,
        stride_ns_n, stride_ns_h, stride_ns_v,
        BLOCK_V, stream
    );
}

// v7 FP4 launchers
void gdn_decode_v7_fp4(
    const void* Q, const void* K, const void* V,
    const void* State_FP4, const void* State_Scale,
    const void* A_log, const void* A, const void* DtBias, const void* B_gate,
    void* Out, void* NewState_FP4, void* NewState_Scale,
    float scale, int B, int num_v_heads, int D,
    int stride_q_b, int stride_q_h, int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h, int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b, int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, cudaStream_t stream
) {
    gdn::gdn_decode_v7_launch_fp4(
        Q, K, V, State_FP4, State_Scale, A_log, A, DtBias, B_gate,
        Out, NewState_FP4, NewState_Scale,
        scale, B, num_v_heads, D,
        stride_q_b, stride_q_h, stride_k_b, stride_k_h,
        stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
        stride_a_b, stride_b_b, stride_o_b, stride_o_h,
        stride_ns_b, stride_ns_h, stride_ns_v,
        BLOCK_V, stream
    );
}

void gdn_prefill_v7_fp4(
    const void* Q, const void* K, const void* V,
    const void* State_FP4, const void* State_Scale,
    const void* A_log, const void* A, const void* DtBias, const void* B_gate,
    const void* CuSeqlens,
    void* Out, void* NewState_FP4, void* NewState_Scale,
    float scale, int N, int num_v_heads, int D,
    int stride_q_t, int stride_q_h, int stride_k_t, int stride_k_h,
    int stride_v_t, int stride_v_h, int stride_s_n, int stride_s_h, int stride_s_v,
    int stride_a_t, int stride_b_t, int stride_o_t, int stride_o_h,
    int stride_ns_n, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, cudaStream_t stream
) {
    gdn::gdn_prefill_v7_launch_fp4(
        Q, K, V, State_FP4, State_Scale, A_log, A, DtBias, B_gate, CuSeqlens,
        Out, NewState_FP4, NewState_Scale,
        scale, N, num_v_heads, D,
        stride_q_t, stride_q_h, stride_k_t, stride_k_h,
        stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v,
        stride_a_t, stride_b_t, stride_o_t, stride_o_h,
        stride_ns_n, stride_ns_h, stride_ns_v,
        BLOCK_V, stream
    );
}

// v5 launchers (backward compatibility)
void gdn_decode_v5(
    const void* Q, const void* K, const void* V, const void* State,
    const void* A_log, const void* A, const void* DtBias, const void* B_gate,
    void* Out, void* NewState,
    float scale, int B, int num_v_heads, int D,
    int stride_q_b, int stride_q_h, int stride_k_b, int stride_k_h,
    int stride_v_b, int stride_v_h, int stride_s_b, int stride_s_h, int stride_s_v,
    int stride_a_b, int stride_b_b, int stride_o_b, int stride_o_h,
    int stride_ns_b, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, cudaStream_t stream
) {
    gdn::gdn_decode_v5_launch(
        Q, K, V, State, A_log, A, DtBias, B_gate, Out, NewState,
        scale, B, num_v_heads, D,
        stride_q_b, stride_q_h, stride_k_b, stride_k_h,
        stride_v_b, stride_v_h, stride_s_b, stride_s_h, stride_s_v,
        stride_a_b, stride_b_b, stride_o_b, stride_o_h,
        stride_ns_b, stride_ns_h, stride_ns_v,
        BLOCK_V, stream
    );
}

void gdn_prefill_v5(
    const void* Q, const void* K, const void* V, const void* State,
    const void* A_log, const void* A, const void* DtBias, const void* B_gate,
    const void* CuSeqlens,
    void* Out, void* NewState,
    float scale, int N, int num_v_heads, int D,
    int stride_q_t, int stride_q_h, int stride_k_t, int stride_k_h,
    int stride_v_t, int stride_v_h, int stride_s_n, int stride_s_h, int stride_s_v,
    int stride_a_t, int stride_b_t, int stride_o_t, int stride_o_h,
    int stride_ns_n, int stride_ns_h, int stride_ns_v,
    int BLOCK_V, cudaStream_t stream
) {
    gdn::gdn_prefill_v5_launch(
        Q, K, V, State, A_log, A, DtBias, B_gate, CuSeqlens, Out, NewState,
        scale, N, num_v_heads, D,
        stride_q_t, stride_q_h, stride_k_t, stride_k_h,
        stride_v_t, stride_v_h, stride_s_n, stride_s_h, stride_s_v,
        stride_a_t, stride_b_t, stride_o_t, stride_o_h,
        stride_ns_n, stride_ns_h, stride_ns_v,
        BLOCK_V, stream
    );
}

}  // extern "C"

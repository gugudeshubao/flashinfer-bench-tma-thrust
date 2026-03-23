"""
Debug script: test the benchmark framework evaluation directly (no subprocess).
"""
import modal

app = modal.App("tma-thrust-debug2")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy", "safetensors")
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=300,
    volumes={TRACE_SET_PATH: trace_volume},
)
def debug_framework():
    import math
    import torch
    import torch.nn.functional as F
    from pathlib import Path

    from flashinfer_bench.data import TraceSet, BuildSpec, SourceFile, Solution
    from flashinfer_bench.compile import BuilderRegistry
    from flashinfer_bench.bench.config import BenchmarkConfig
    from flashinfer_bench.bench.evaluators.default import DefaultEvaluator
    from flashinfer_bench.bench.evaluators.utils import normalize_result
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    # Load from volume
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    print("Definitions:", list(trace_set.definitions.keys()))
    print("Workload keys:", list(trace_set.workloads.keys()))

    if "gdn_prefill_qk4_v8_d128_k_last" not in trace_set.definitions:
        print("ERROR: definition not found")
        return

    definition = trace_set.definitions["gdn_prefill_qk4_v8_d128_k_last"]
    workloads = trace_set.workloads.get("gdn_prefill_qk4_v8_d128_k_last", [])
    print(f"Found {len(workloads)} workloads")

    # Use first (smallest) workload
    wl_trace = workloads[0]
    workload = wl_trace.workload
    print(f"Testing workload {workload.uuid}")
    print(f"  axes: {workload.axes}")
    print(f"  inputs: {list(workload.inputs.keys())}")

    # Build solution
    kernel_module = '''
import math
import torch
import torch.nn.functional as F

def _matmul(a, b):
    return a.float() @ b.float()

def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_k_heads = k.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)
    x = a.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b.float())
    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)
    output = torch.zeros((total_seq_len, num_sab_heads, head_size), dtype=torch.bfloat16, device=device)
    new_state = torch.zeros((num_seqs, num_sab_heads, head_size, head_size), dtype=torch.float32, device=device)
    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = seq_end - seq_start
        if seq_len <= 0:
            continue
        if state is not None:
            state_HKV = state[seq_idx].clone().float().transpose(-1, -2)
        else:
            state_HKV = torch.zeros((num_sab_heads, head_size, head_size), dtype=torch.float32, device=device)
        for i in range(seq_len):
            t = seq_start + i
            q_H1K = q_exp[t].unsqueeze(1).float()
            k_H1K = k_exp[t].unsqueeze(1).float()
            v_H1V = v[t].unsqueeze(1).float()
            g_H11 = g[t].unsqueeze(1).unsqueeze(2)
            beta_H11 = beta[t].unsqueeze(1).unsqueeze(2)
            old_state_HKV = g_H11 * state_HKV
            old_v_H1V = _matmul(k_H1K, old_state_HKV)
            new_v_H1V = beta_H11 * v_H1V + (1 - beta_H11) * old_v_H1V
            state_remove = torch.einsum("hkl,hlv->hkv", k_H1K.transpose(-1, -2), old_v_H1V)
            state_update = torch.einsum("hkl,hlv->hkv", k_H1K.transpose(-1, -2), new_v_H1V)
            state_HKV = old_state_HKV - state_remove + state_update
            o_H1V = scale * _matmul(q_H1K, state_HKV)
            output[t] = o_H1V.squeeze(1).to(torch.bfloat16)
        new_state[seq_idx] = state_HKV.transpose(-1, -2)
    return output, new_state
'''

    solution = Solution(
        name="debug-prefill",
        definition="gdn_prefill_qk4_v8_d128_k_last",
        author="debug",
        spec=BuildSpec(
            language="triton",
            target_hardware=["cuda"],
            entry_point="kernel.py::kernel",
            dependencies=[],
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="kernel.py", content=kernel_module)],
    )

    cfg = BenchmarkConfig(warmup_runs=0, iterations=1, num_trials=1)

    print("\n--- Building baseline (reference) ---")
    baseline = DefaultEvaluator.build_baseline(
        definition=definition,
        workload=workload,
        cfg=cfg,
        device="cuda:0",
        trace_set_root=trace_set.root,
    )
    print(f"Baseline inputs count: {len(baseline.inputs[0])}")
    print(f"Baseline input types: {[type(x).__name__ if not hasattr(x, 'shape') else str(x.shape) for x in baseline.inputs[0]]}")
    ref_out = baseline.outputs[0]
    print(f"Ref outputs: {[str(t.shape) for t in ref_out]}")
    print(f"Ref output[0] (bfloat16): has_nan={ref_out[0].isnan().any()}, has_inf={ref_out[0].isinf().any()}")
    print(f"  min={ref_out[0].float().min():.4f} max={ref_out[0].float().max():.4f}")
    print(f"Ref output[1] (new_state): has_nan={ref_out[1].isnan().any()}, has_inf={ref_out[1].isinf().any()}")

    print("\n--- Building solution runnable ---")
    registry = BuilderRegistry.get_instance()
    sol_runnable = registry.build(definition, solution)

    print("\n--- Running solution with baseline inputs ---")
    inp = baseline.inputs[0]
    print(f"Scale value: {inp[-1]}")
    print(f"cu_seqlens: {inp[8]}")

    with torch.no_grad():
        result = sol_runnable(*inp)

    out = normalize_result(definition, result, "cuda:0")
    print(f"Sol output[0]: has_nan={out[0].isnan().any()}, has_inf={out[0].isinf().any()}")
    print(f"  min={out[0].float().min():.4f} max={out[0].float().max():.4f}")
    print(f"Sol output[1]: has_nan={out[1].isnan().any()}, has_inf={out[1].isinf().any()}")

    diff0 = (ref_out[0].float() - out[0].float()).abs()
    diff1 = (ref_out[1].float() - out[1].float()).abs()
    print(f"\nDiff output: max_abs={diff0.max():.6e}")
    print(f"Diff new_state: max_abs={diff1.max():.6e}")

    print("\n--- Running DefaultEvaluator.check_correctness directly ---")
    correctness, ev = DefaultEvaluator.check_correctness(
        definition=definition,
        sol_runnable=sol_runnable,
        inputs=baseline.inputs,
        ref_outputs=baseline.outputs,
        cfg=cfg,
        log_path="/tmp/debug_log.txt",
        device="cuda:0",
    )
    print(f"Correctness result: {correctness}")
    print(f"Evaluation: {ev}")
    if correctness:
        print(f"  max_abs={correctness.max_absolute_error:.6e}")
        print(f"  max_rel={correctness.max_relative_error:.6e}")


@app.local_entrypoint()
def main():
    debug_framework.remote()

"""
Debug script: run reference vs our kernel directly on Modal B200 and compare outputs.
"""
import modal

app = modal.App("tma-thrust-debug")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


@app.function(image=image, gpu="B200:1", timeout=300)
def debug_comparison():
    import math
    import torch
    import torch.nn.functional as F

    torch.manual_seed(42)
    device = "cuda:0"

    # Simple test case: total_seq_len=8, num_seqs=1, seq_len=8
    total_seq_len = 8
    num_seqs = 1
    num_q_heads = 4
    num_v_heads = 8
    head_size = 128
    scale = 1.0 / math.sqrt(head_size)

    q = torch.randn(total_seq_len, num_q_heads, head_size, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_seq_len, num_q_heads, head_size, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_seq_len, num_v_heads, head_size, dtype=torch.bfloat16, device=device)
    state = torch.randn(num_seqs, num_v_heads, head_size, head_size, dtype=torch.float32, device=device)
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    a = torch.randn(total_seq_len, num_v_heads, dtype=torch.bfloat16, device=device)
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    b = torch.randn(total_seq_len, num_v_heads, dtype=torch.bfloat16, device=device)
    cu_seqlens = torch.tensor([0, total_seq_len], dtype=torch.int64, device=device)

    inputs = [q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale]

    print("Input dtypes:", [x.dtype if hasattr(x, 'dtype') else type(x) for x in inputs])
    print("q stats: min={:.4f} max={:.4f}".format(q.float().min().item(), q.float().max().item()))
    print("state stats: min={:.4f} max={:.4f}".format(state.min().item(), state.max().item()))

    # --- Reference ---
    def _matmul(a, b):
        return a.float() @ b.float()

    def reference(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
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
                state_remove = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), old_v_H1V)
                state_update = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), new_v_H1V)
                state_HKV = old_state_HKV - state_remove + state_update
                o_H1V = scale * _matmul(q_H1K, state_HKV)
                output[t] = o_H1V.squeeze(1).to(torch.bfloat16)
            new_state[seq_idx] = state_HKV.transpose(-1, -2)
        return output, new_state

    out_ref, state_ref = reference(*inputs)
    print("\nReference output stats:")
    print("  output: min={:.4f} max={:.4f} has_nan={}".format(
        out_ref.float().min().item(), out_ref.float().max().item(), out_ref.isnan().any().item()
    ))
    print("  new_state: min={:.4f} max={:.4f}".format(
        state_ref.min().item(), state_ref.max().item()
    ))

    # --- Our kernel (loaded from source) ---
    # Inline the kernel here (simulating what framework does)
    kernel_src = open("/dev/stdin", "r")  # won't work, use inline

    def our_kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Exact copy of kernel.py
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
                state_remove = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), old_v_H1V)
                state_update = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), new_v_H1V)
                state_HKV = old_state_HKV - state_remove + state_update
                o_H1V = scale * _matmul(q_H1K, state_HKV)
                output[t] = o_H1V.squeeze(1).to(torch.bfloat16)
            new_state[seq_idx] = state_HKV.transpose(-1, -2)
        return output, new_state

    out_ours, state_ours = our_kernel(*inputs)
    print("\nOur kernel output stats:")
    print("  output: min={:.4f} max={:.4f} has_nan={}".format(
        out_ours.float().min().item(), out_ours.float().max().item(), out_ours.isnan().any().item()
    ))

    # Compare
    diff = (out_ref.float() - out_ours.float()).abs()
    print("\nDifference:")
    print("  max_abs={:.6e}  max_rel={:.6e}".format(
        diff.max().item(),
        (diff / (out_ref.float().abs() + 1e-8)).max().item()
    ))
    print("  outputs identical:", torch.allclose(out_ref.float(), out_ours.float(), atol=1e-5))

    # Now test using the benchmark framework directly
    print("\n--- Testing via flashinfer_bench framework ---")
    from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
    from flashinfer_bench.data import BuildSpec, SourceFile

    import inspect
    import textwrap

    kernel_src_str = inspect.getsource(our_kernel)
    # Reformat as module-level function
    kernel_module = textwrap.dedent("""
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
            state_remove = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), old_v_H1V)
            state_update = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), new_v_H1V)
            state_HKV = old_state_HKV - state_remove + state_update
            o_H1V = scale * _matmul(q_H1K, state_HKV)
            output[t] = o_H1V.squeeze(1).to(torch.bfloat16)
        new_state[seq_idx] = state_HKV.transpose(-1, -2)
    return output, new_state
""").strip()

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

    from flashinfer_bench.data import TraceSet
    trace_set = TraceSet.from_path("/data")

    if "gdn_prefill_qk4_v8_d128_k_last" not in trace_set.definitions:
        print("ERROR: definition not found in volume!")
        return

    definition = trace_set.definitions["gdn_prefill_qk4_v8_d128_k_last"]
    workloads = trace_set.workloads.get("gdn_prefill_qk4_v8_d128_k_last", [])
    print(f"Found {len(workloads)} workloads")

    # Run with just the first workload
    if workloads:
        from flashinfer_bench.bench.config import BenchmarkConfig
        from flashinfer_bench.bench.evaluators.default import DefaultEvaluator

        workload_trace = workloads[0]
        workload = workload_trace.workload
        print(f"Testing workload {workload.uuid}")
        print(f"  axes: {workload.axes}")

        cfg = BenchmarkConfig(warmup_runs=0, iterations=1, num_trials=1)

        # Build baseline (reference)
        baseline = DefaultEvaluator.build_baseline(
            definition=definition,
            workload=workload,
            cfg=cfg,
            device="cuda:0",
            trace_set_root=trace_set.root,
        )
        print(f"Baseline built with {len(baseline.inputs)} trial(s)")
        print(f"Ref output shapes: {[o.shape for o in baseline.outputs[0]]}")
        print(f"Ref output dtypes: {[o.dtype for o in baseline.outputs[0]]}")
        print(f"Ref output[0] stats: min={baseline.outputs[0][0].float().min():.4f} max={baseline.outputs[0][0].float().max():.4f} has_nan={baseline.outputs[0][0].isnan().any()}")

        # Run our solution
        from flashinfer_bench.compile import BuilderRegistry
        registry = BuilderRegistry.get_instance()
        sol_runnable = registry.build(definition, solution)

        inp = baseline.inputs[0]
        with torch.no_grad():
            result = sol_runnable(*inp)

        from flashinfer_bench.bench.evaluators.utils import normalize_result
        out = normalize_result(definition, result, "cuda:0")
        print(f"Sol output shapes: {[o.shape for o in out]}")
        print(f"Sol output dtypes: {[o.dtype for o in out]}")
        print(f"Sol output[0] stats: min={out[0].float().min():.4f} max={out[0].float().max():.4f} has_nan={out[0].isnan().any()}")

        diff = (baseline.outputs[0][0].float() - out[0].float()).abs()
        print(f"\nDiff output: max_abs={diff.max():.6e}")

        diff_state = (baseline.outputs[0][1].float() - out[1].float()).abs()
        print(f"Diff state: max_abs={diff_state.max():.6e}")


@app.local_entrypoint()
def main():
    debug_comparison.remote()

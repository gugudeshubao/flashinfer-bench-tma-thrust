"""
Modal benchmark runner for GDN kernels (decode + prefill).

Usage:
    modal run gdn/benchmarks/bench_modal.py              # run both (solution only)
    modal run gdn/benchmarks/bench_modal.py --kernel decode
    modal run gdn/benchmarks/bench_modal.py --kernel prefill
    modal run gdn/benchmarks/bench_modal.py --compare     # solution vs Python baseline
    modal run gdn/benchmarks/bench_modal.py --kernel decode --warmup 5 --iters 100

Setup (one-time):
    modal run scripts/setup_volume.py
"""

import json
from pathlib import Path

# GDN directory is the parent of benchmarks/
GDN_ROOT = Path(__file__).parent.parent

import modal

app = modal.App("tma-thrust-gdn-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "git", "ninja-build", "build-essential")
    .pip_install("torch>=2.4.0", "triton>=3.0.0", "numpy", "tabulate")
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-8",
        "git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git /opt/cutlass",
    )
    .env(
        {
            "PATH": "/usr/local/cuda-12.8/bin:$PATH",
            "LD_LIBRARY_PATH": "/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH",
            "CUTLASS_PATH": "/opt/cutlass",
            "CUDA_HOME": "/usr/local/cuda-12.8",
        }
    )
    .pip_install("flashinfer-bench", "ninja", "numba")
)


KERNEL_NAMES = {
    "decode": "decode",  # gdn/decode/
    "prefill": "prefill",  # gdn/prefill/
}


KERNEL_CONFIGS = {
    "decode": {
        "solution": {
            "name": "tma-thrust-gdn-decode-v1",
            "subdir": "solution/triton",
        },
        "cuda": {
            "name": "tma-thrust-gdn-decode-cuda-v10",
            "subdir": "solution/cuda",
        },
        "baseline": {
            "name": "tma-thrust-gdn-decode-baseline",
            "subdir": "baseline/triton",
        },
        "definition": "gdn_decode_qk4_v8_d128_k_last",
        "author": "tma-thrust",
        "language": "triton",
        "entry_point": "kernel.py::kernel",
        "destination_passing_style": False,
    },
    "prefill": {
        "solution": {
            "name": "tma-thrust-gdn-prefill-v1",
            "subdir": "solution/triton",
        },
        "cuda": {
            "name": "tma-thrust-gdn-prefill-cuda-v6",
            "subdir": "solution/cuda",
        },
        "baseline": {
            "name": "tma-thrust-gdn-prefill-baseline",
            "subdir": "baseline/triton",
        },
        "definition": "gdn_prefill_qk4_v8_d128_k_last",
        "author": "tma-thrust",
        "language": "triton",
        "entry_point": "kernel.py::kernel",
        "destination_passing_style": False,
    },
}


def build_solution_dict(kernel_dir_name: str, variant: str = "solution", cuda_backend: str = "cute") -> dict:
    """Build Solution JSON dict locally (no flashinfer_bench needed).

    variant: 'solution' (optimized) or 'baseline' (Python reference)
    """
    cfg = KERNEL_CONFIGS[kernel_dir_name]
    kernel_dir = GDN_ROOT / kernel_dir_name
    source_dir = kernel_dir / cfg[variant]["subdir"]

    sources = []
    for py_file in sorted(source_dir.glob("*.py")):
        sources.append({"path": py_file.name, "content": py_file.read_text()})

    if kernel_dir_name == "decode" and variant == "cuda":
        backend = cuda_backend.lower()
        if backend not in {"cute", "tma"}:
            raise ValueError(f"Unsupported decode cuda backend: {cuda_backend}")
        sources = [s for s in sources if s["path"] != "backend.py"]
        sources.append({"path": "backend.py", "content": f'BACKEND = "{backend}"\n'})

        v10_header = GDN_ROOT / "kernels" / "cute_cpp" / "gdn_decode_v10.cuh"
        if not v10_header.exists():
            raise FileNotFoundError(f"Missing decode v10 header: {v10_header}")
        sources.append({"path": "gdn_decode_v10.cuh", "content": v10_header.read_text()})

    entry_file = cfg["entry_point"].split("::")[0]
    assert any(s["path"] == entry_file for s in sources), \
        f"Entry file {entry_file!r} not found in {[s['path'] for s in sources]}"

    solution_name = cfg[variant]["name"]
    if kernel_dir_name == "decode" and variant == "cuda":
        solution_name = f"{solution_name}-{cuda_backend.lower()}"

    return {
        "name": solution_name,
        "definition": cfg["definition"],
        "author": cfg["author"],
        "spec": {
            "language": cfg["language"],
            "target_hardware": ["cuda"],
            "entry_point": cfg["entry_point"],
            "dependencies": [],
            "destination_passing_style": cfg["destination_passing_style"],
        },
        "sources": sources,
    }


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={TRACE_SET_PATH: trace_volume},
)
def run_benchmark(solution_dict: dict, config_dict: dict = None) -> dict:
    """Build solution from dict and run benchmark on Modal B200."""
    from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

    solution = Solution.model_validate(solution_dict)

    if config_dict is None:
        bench_config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
    else:
        bench_config = BenchmarkConfig(**config_dict)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        avail = list(trace_set.definitions.keys())
        raise ValueError(f"Definition '{solution.definition}' not found. Available: {avail}")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads for '{solution.definition}'")

    print(f"Running {solution.name} ({solution.definition}): {len(workloads)} workloads")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, bench_config)
    result_trace_set = benchmark.run_all(dump_traces=True)
    traces = result_trace_set.traces.get(definition.name, [])

    results = {}
    debug_failures = 0
    for trace in traces:
        if trace.evaluation:
            entry = {
                "solution_name": solution.name,
                "status": trace.evaluation.status.value,
            }
            if trace.evaluation.performance:
                p = trace.evaluation.performance
                entry["latency_ms"] = p.latency_ms
                entry["reference_latency_ms"] = p.reference_latency_ms
                entry["speedup_factor"] = p.speedup_factor
            if trace.evaluation.correctness:
                c = trace.evaluation.correctness
                entry["max_abs_error"] = c.max_absolute_error
                entry["max_rel_error"] = c.max_relative_error

            if entry["status"] != "PASSED" and debug_failures < 3:
                debug_failures += 1
                print("\n[debug] Non-passed evaluation details")
                print(f"  workload={trace.workload.uuid}")
                try:
                    print(json.dumps(trace.evaluation.model_dump(exclude_none=True), indent=2, default=str))
                except Exception:
                    print(repr(trace.evaluation))
            results[trace.workload.uuid] = entry

    return {solution.definition: results}


def print_results(results: dict):
    for def_name, workloads in results.items():
        print(f"\n{'='*60}")
        print(f"Definition: {def_name}")
        print(f"{'='*60}")
        if not workloads:
            print("  (no results)")
            continue
        speedups = []
        for wid, r in workloads.items():
            sol_name = r.get("solution_name", "?")
            status = r.get("status", "?")
            lat = r.get("latency_ms")
            ref_lat = r.get("reference_latency_ms")
            spdup = r.get("speedup_factor")
            abs_err = r.get("max_abs_error")
            parts = [f"  {wid[:8]}... | {sol_name} | {status}"]
            if lat is not None:
                parts.append(f"{lat:.4f}ms")
            if ref_lat is not None:
                parts.append(f"ref={ref_lat:.4f}ms")
            if spdup is not None:
                parts.append(f"speedup={spdup:.2f}x")
                speedups.append(spdup)
            if abs_err is not None:
                parts.append(f"abs_err={abs_err:.2e}")
            print(" | ".join(parts))
        if speedups:
            avg = sum(speedups) / len(speedups)
            print(f"\n  Average speedup: {avg:.2f}x  (N={len(speedups)})")


def print_comparison(sol_results: dict, base_results: dict):
    """Print side-by-side comparison of solution vs baseline."""
    for def_name in sol_results:
        sol_wl = sol_results.get(def_name, {})
        base_wl = base_results.get(def_name, {})
        all_wids = sorted(set(sol_wl) | set(base_wl))

        print(f"\n{'='*72}")
        print(f"Comparison: {def_name}")
        print(f"{'='*72}")
        print(f"  {'workload':10s}  {'baseline':>12s}  {'solution':>12s}  {'gain':>8s}  status")
        print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*8}  ------")

        gains = []
        for wid in all_wids:
            b = base_wl.get(wid, {})
            s = sol_wl.get(wid, {})
            b_lat = b.get("latency_ms")
            s_lat = s.get("latency_ms")
            s_status = s.get("status", "?")
            b_status = b.get("status", "?")

            b_str = f"{b_lat:.2f}ms" if b_lat is not None else "n/a"
            s_str = f"{s_lat:.2f}ms" if s_lat is not None else "n/a"

            if b_lat and s_lat:
                gain = b_lat / s_lat
                gain_str = f"{gain:.2f}x"
                gains.append(gain)
            else:
                gain_str = "n/a"

            print(f"  {wid[:10]}  {b_str:>12s}  {s_str:>12s}  {gain_str:>8s}  {s_status}")

        if gains:
            avg = sum(gains) / len(gains)
            print(f"\n  Average gain over Python baseline: {avg:.2f}x  (N={len(gains)})")


@app.local_entrypoint()
def main(
    kernel: str = "both",
    warmup: int = 3,
    iters: int = 100,
    trials: int = 5,
    compare: bool = False,
    cuda: bool = False,
    cuda_backend: str = "cute",
):
    """
    Run GDN kernel benchmarks on Modal B200.

    --kernel:  decode | prefill | both  (default: both)
    --compare: also run Python baseline and show side-by-side latency comparison
    --cuda:    run packaged CUDA kernel instead of Triton solution
    --cuda-backend: for decode CUDA packaging, choose `cute` or `tma`
    """
    config_dict = {"warmup_runs": warmup, "iterations": iters, "num_trials": trials}

    if kernel == "both":
        targets = list(KERNEL_NAMES.keys())
    elif kernel in KERNEL_NAMES:
        targets = [kernel]
    else:
        print(f"Unknown kernel '{kernel}'. Use: decode | prefill | both")
        sys.exit(1)

    # Determine variant
    variant = "cuda" if cuda else "solution"

    # Spawn solution jobs
    sol_futures = {}
    for k in targets:
        dir_name = KERNEL_NAMES[k]
        use_variant = variant
        cfg = KERNEL_CONFIGS[dir_name]
        
        # Check if variant exists
        if use_variant not in cfg:
            print(f"Warning: {use_variant} not available for {k}, using solution")
            use_variant = "solution"
        
        print(f"Packing {k} {use_variant} ({dir_name})...")
        sol_dict = build_solution_dict(dir_name, variant=use_variant, cuda_backend=cuda_backend)
        print(f"  -> {sol_dict['name']}")
        sol_futures[k] = run_benchmark.spawn(sol_dict, config_dict)

    # Optionally spawn baseline jobs in parallel
    base_futures = {}
    if compare:
        for k in targets:
            dir_name = KERNEL_NAMES[k]
            print(f"Packing {k} baseline ({dir_name})...")
            base_dict = build_solution_dict(dir_name, variant="baseline")
            print(f"  -> {base_dict['name']}")
            base_futures[k] = run_benchmark.spawn(base_dict, config_dict)

    # Collect solution results
    sol_all = {}
    for k, fut in sol_futures.items():
        print(f"Waiting for {k} solution results...")
        sol_all.update(fut.get())

    print("\n" + "=" * 60)
    print("SOLUTION RESULTS")
    print_results(sol_all)

    # Collect and compare baseline results
    if compare:
        base_all = {}
        for k, fut in base_futures.items():
            print(f"Waiting for {k} baseline results...")
            base_all.update(fut.get())

        print("\n" + "=" * 60)
        print("BASELINE RESULTS")
        print_results(base_all)

        print("\n" + "=" * 60)
        print("SOLUTION vs BASELINE COMPARISON")
        print_comparison(sol_all, base_all)

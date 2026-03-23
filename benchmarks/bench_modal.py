"""
Modal benchmark runner for GDN kernels (decode + prefill).

Usage:
    modal run benchmarks/bench_modal.py              # run both
    modal run benchmarks/bench_modal.py --kernel decode
    modal run benchmarks/bench_modal.py --kernel prefill
    modal run benchmarks/bench_modal.py --kernel decode --warmup 5 --iters 100

Setup (one-time):
    modal volume create flashinfer-trace
    # Then upload the dataset:
    # modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("tma-thrust-gdn-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


def pack_kernel(kernel_name: str) -> Solution:
    """Pack a kernel solution from its subfolder."""
    kernel_dir = REPO_ROOT / kernel_name
    if not kernel_dir.exists():
        raise ValueError(f"Kernel directory not found: {kernel_dir}")
    sys.path.insert(0, str(kernel_dir))

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    from flashinfer_bench import BuildSpec
    from flashinfer_bench.agents import pack_solution_from_files

    config_path = kernel_dir / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    sol = config["solution"]
    bld = config["build"]
    source_dir = kernel_dir / "solution" / "triton"
    dps = bld.get("destination_passing_style", True)

    spec = BuildSpec(
        language=bld["language"],
        target_hardware=["cuda"],
        entry_point=bld["entry_point"],
        destination_passing_style=dps,
    )

    return pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=sol["name"],
        definition=sol["definition"],
        author=sol["author"],
    )


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={TRACE_SET_PATH: trace_volume},
)
def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark on Modal B200 and return results."""
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads for '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)
    traces = result_trace_set.traces.get(definition.name, [])

    results = {}
    for trace in traces:
        if trace.evaluation:
            entry = {"status": trace.evaluation.status.value}
            if trace.evaluation.performance:
                p = trace.evaluation.performance
                entry["latency_ms"] = p.latency_ms
                entry["reference_latency_ms"] = p.reference_latency_ms
                entry["speedup_factor"] = p.speedup_factor
            if trace.evaluation.correctness:
                c = trace.evaluation.correctness
                entry["max_abs_error"] = c.max_absolute_error
                entry["max_rel_error"] = c.max_relative_error
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
            status = r.get("status", "?")
            lat = r.get("latency_ms")
            ref_lat = r.get("reference_latency_ms")
            spdup = r.get("speedup_factor")
            abs_err = r.get("max_abs_error")
            parts = [f"  {wid[:8]}... | {status}"]
            if lat is not None:
                parts.append(f"{lat:.3f}ms")
            if ref_lat is not None:
                parts.append(f"ref={ref_lat:.3f}ms")
            if spdup is not None:
                parts.append(f"speedup={spdup:.2f}x")
                speedups.append(spdup)
            if abs_err is not None:
                parts.append(f"abs_err={abs_err:.2e}")
            print(" | ".join(parts))
        if speedups:
            avg = sum(speedups) / len(speedups)
            print(f"\n  Average speedup: {avg:.2f}x  (N={len(speedups)})")


KERNEL_NAMES = {
    "decode": "gdn_decode_qk4_v8_d128_k_last",
    "prefill": "gdn_prefill_qk4_v8_d128_k_last",
}


@app.local_entrypoint()
def main(
    kernel: str = "both",
    warmup: int = 3,
    iters: int = 100,
    trials: int = 5,
):
    """
    Run GDN kernel benchmarks on Modal B200.

    --kernel: decode | prefill | both  (default: both)
    """
    config = BenchmarkConfig(warmup_runs=warmup, iterations=iters, num_trials=trials)

    if kernel == "both":
        targets = list(KERNEL_NAMES.keys())
    elif kernel in KERNEL_NAMES:
        targets = [kernel]
    else:
        print(f"Unknown kernel '{kernel}'. Choose from: both, decode, prefill")
        sys.exit(1)

    all_results = {}
    futures = {}

    for k in targets:
        dir_name = KERNEL_NAMES[k]
        print(f"\nPacking {k} kernel ({dir_name})...")
        solution = pack_kernel(dir_name)
        print(f"  -> {solution.name}")
        futures[k] = run_benchmark.spawn(solution, config)

    for k, fut in futures.items():
        print(f"\nWaiting for {k} results...")
        result = fut.get()
        all_results.update(result)

    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print_results(all_results)

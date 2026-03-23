"""
Modal benchmark runner for GDN kernels (decode + prefill).

Usage:
    modal run benchmarks/bench_modal.py              # run both
    modal run benchmarks/bench_modal.py --kernel decode
    modal run benchmarks/bench_modal.py --kernel prefill
    modal run benchmarks/bench_modal.py --kernel decode --warmup 5 --iters 100

Setup (one-time):
    modal run scripts/setup_volume.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

import modal

app = modal.App("tma-thrust-gdn-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


KERNEL_NAMES = {
    "decode": "gdn_decode_qk4_v8_d128_k_last",
    "prefill": "gdn_prefill_qk4_v8_d128_k_last",
}


# Per-kernel static config (avoids TOML parsing dependency on local Python 3.9)
KERNEL_CONFIGS = {
    "gdn_decode_qk4_v8_d128_k_last": {
        "name": "tma-thrust-gdn-decode-v1",
        "definition": "gdn_decode_qk4_v8_d128_k_last",
        "author": "tma-thrust",
        "language": "triton",
        "entry_point": "kernel.py::kernel",
        "destination_passing_style": False,
    },
    "gdn_prefill_qk4_v8_d128_k_last": {
        "name": "tma-thrust-gdn-prefill-v1",
        "definition": "gdn_prefill_qk4_v8_d128_k_last",
        "author": "tma-thrust",
        "language": "triton",
        "entry_point": "kernel.py::kernel",
        "destination_passing_style": False,
    },
}


def build_solution_dict(kernel_dir_name: str) -> dict:
    """Build Solution JSON dict locally (no flashinfer_bench needed)."""
    cfg = KERNEL_CONFIGS[kernel_dir_name]
    kernel_dir = REPO_ROOT / kernel_dir_name
    source_dir = kernel_dir / "solution" / "triton"

    sources = []
    for py_file in sorted(source_dir.glob("*.py")):
        sources.append({"path": py_file.name, "content": py_file.read_text()})

    entry_file = cfg["entry_point"].split("::")[0]
    assert any(s["path"] == entry_file for s in sources), \
        f"Entry file {entry_file!r} not found in {[s['path'] for s in sources]}"

    return {
        "name": cfg["name"],
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

    # Deserialize
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

    print(f"Running {solution.definition}: {len(workloads)} workloads")

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
    config_dict = {"warmup_runs": warmup, "iterations": iters, "num_trials": trials}

    if kernel == "both":
        targets = list(KERNEL_NAMES.keys())
    elif kernel in KERNEL_NAMES:
        targets = [kernel]
    else:
        print(f"Unknown kernel '{kernel}'. Use: decode | prefill | both")
        sys.exit(1)

    futures = {}
    for k in targets:
        dir_name = KERNEL_NAMES[k]
        print(f"Packing {k} kernel ({dir_name})...")
        sol_dict = build_solution_dict(dir_name)
        print(f"  -> {sol_dict['name']}")
        futures[k] = run_benchmark.spawn(sol_dict, config_dict)

    all_results = {}
    for k, fut in futures.items():
        print(f"Waiting for {k} results...")
        result = fut.get()
        all_results.update(result)

    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print_results(all_results)

"""
Modal benchmark runner for FP8 MoE kernel.

Usage:
    modal run moe/benchmarks/bench_modal.py
    modal run moe/benchmarks/bench_modal.py --warmup 5 --iters 100

Setup (one-time):
    modal run scripts/setup_volume.py --mode hf
"""

import json
import sys
from pathlib import Path

MOE_ROOT = Path(__file__).parent.parent

import modal

app = modal.App("tma-thrust-moe-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)

DEFINITION_NAME = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"

VARIANTS = {
    "triton": {
        "name": "tma-thrust-moe-v1",
        "subdir": "solution/triton",
    },
    "cuda": {
        "name": "tma-thrust-moe-cuda-v1",
        "subdir": "solution/cuda",
    },
    "scaled_mm": {
        "name": "tma-thrust-moe-scaled-mm-v1",
        "subdir": "solution/scaled_mm",
    },
    "v2": {
        "name": "tma-thrust-moe-v2",
        "subdir": "solution/v2",
    },
    "v3": {
        "name": "tma-thrust-moe-v3",
        "subdir": "solution/v3",
    },
    "v4": {
        "name": "tma-thrust-moe-v4",
        "subdir": "solution/v4",
    },
}

DEFAULT_VARIANT = "triton"

def _get_kernel_config(variant: str = DEFAULT_VARIANT) -> dict:
    v = VARIANTS[variant]
    return {
        "solution": v,
        "definition": DEFINITION_NAME,
        "author": "tma-thrust",
        "language": "triton",
        "entry_point": "kernel.py::kernel",
        "destination_passing_style": False,
    }


def build_solution_dict(variant: str = DEFAULT_VARIANT) -> dict:
    """Build Solution JSON dict from local source files."""
    cfg = _get_kernel_config(variant)
    source_dir = MOE_ROOT / cfg["solution"]["subdir"]

    sources = []
    for py_file in sorted(source_dir.glob("*.py")):
        sources.append({"path": py_file.name, "content": py_file.read_text()})

    entry_file = cfg["entry_point"].split("::")[0]
    assert any(s["path"] == entry_file for s in sources), \
        f"Entry file {entry_file!r} not found in {[s['path'] for s in sources]}"

    return {
        "name": cfg["solution"]["name"],
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
    for trace in traces:
        if trace.evaluation:
            entry = {
                "solution_name": solution.name,
                "status": trace.evaluation.status.value,
            }
            if trace.evaluation.performance:
                perf = trace.evaluation.performance
                entry["latency_ms"] = perf.latency_ms
                entry["reference_latency_ms"] = perf.reference_latency_ms
                entry["speedup_factor"] = perf.speedup_factor
            if trace.evaluation.correctness:
                corr = trace.evaluation.correctness
                entry["max_abs_error"] = corr.max_absolute_error
                entry["max_rel_error"] = corr.max_relative_error
            results[trace.workload.uuid] = entry

    return {solution.definition: results}


def print_results(results: dict):
    """Print benchmark results in a formatted table."""
    for def_name, workloads in results.items():
        print(f"\n{'='*60}")
        print(f"Definition: {def_name}")
        print(f"{'='*60}")
        if not workloads:
            print("  (no results)")
            continue
        speedups = []
        for wid, result in workloads.items():
            sol_name = result.get("solution_name", "?")
            status = result.get("status", "?")
            lat = result.get("latency_ms")
            ref_lat = result.get("reference_latency_ms")
            spdup = result.get("speedup_factor")
            abs_err = result.get("max_abs_error")
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


@app.local_entrypoint()
def main(
    warmup: int = 3,
    iters: int = 100,
    trials: int = 5,
    variant: str = DEFAULT_VARIANT,
):
    """
    Run MoE kernel benchmark on Modal B200.

    --warmup:  number of warmup runs
    --iters:   number of benchmark iterations
    --trials:  number of trials
    --variant: solution variant (triton, cuda)
    """
    if variant not in VARIANTS:
        print(f"Unknown variant '{variant}'. Available: {list(VARIANTS.keys())}")
        sys.exit(1)

    config_dict = {"warmup_runs": warmup, "iterations": iters, "num_trials": trials}

    print(f"Packing MoE solution (variant={variant})...")
    sol_dict = build_solution_dict(variant)
    print(f"  -> {sol_dict['name']}")

    print("\nRunning benchmark on Modal B200...")
    result = run_benchmark.remote(sol_dict, config_dict)

    print("\n" + "=" * 60)
    print("MoE BENCHMARK RESULTS")
    print_results(result)

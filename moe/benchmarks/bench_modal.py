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

import modal

from moe.modal_common import DEFINITION_NAME, MOE_ROOT, TRACE_SET_PATH, image, trace_volume

app = modal.App("tma-thrust-moe-bench")

VARIANTS = {
    "triton": {
        "name": "tma-thrust-moe-v1",
        "subdir": "solution/triton",
        "language": "triton",
    },
    "cuda": {
        "name": "tma-thrust-moe-cuda-v1",
        "subdir": "solution/cuda",
        "language": "python",
    },
    "cuda_ptx": {
        "name": "tma-thrust-moe-cuda-ptx-v1",
        "subdir": "solution/cuda_ptx",
        "language": "python",
    },
    "cute_cpp": {
        "name": "tma-thrust-moe-cute-cpp-v1",
        "subdir": "solution/cute_cpp",
        "language": "python",
    },
    "cute_cpp_torch": {
        "name": "tma-thrust-moe-cute-cpp-torch-v1",
        "subdir": "solution/cute_cpp_torch",
        "language": "cuda",
        "binding": "torch",
        "entry_point": "kernel.cu::kernel",
        "dependencies": [],
    },
    "scaled_mm": {
        "name": "tma-thrust-moe-scaled-mm-v1",
        "subdir": "solution/scaled_mm",
        "language": "python",
    },
    "v2": {
        "name": "tma-thrust-moe-v2",
        "subdir": "solution/v2",
        "language": "python",
    },
    "v3": {
        "name": "tma-thrust-moe-v3",
        "subdir": "solution/v3",
        "language": "python",
    },
    "v4": {
        "name": "tma-thrust-moe-v4",
        "subdir": "solution/v4",
        "language": "python",
    },
    "v5": {
        "name": "tma-thrust-moe-v5",
        "subdir": "solution/v5",
        "language": "python",
    },
    "v6": {
        "name": "tma-thrust-moe-v6",
        "subdir": "solution/v6",
        "language": "python",
    },
    "v7": {
        "name": "tma-thrust-moe-v7",
        "subdir": "solution/v7",
        "language": "python",
    },
    "v8": {
        "name": "tma-thrust-moe-v8",
        "subdir": "solution/v8",
        "language": "python",
    },
    "v9": {
        "name": "tma-thrust-moe-v9",
        "subdir": "solution/v9",
        "language": "python",
    },
    "v10": {
        "name": "tma-thrust-moe-v10",
        "subdir": "solution/v10",
        "language": "python",
    },
    "v11": {
        "name": "tma-thrust-moe-v11",
        "subdir": "solution/v11",
        "language": "python",
    },
    "v12": {
        "name": "tma-thrust-moe-v12",
        "subdir": "solution/v12",
        "language": "python",
    },
    "v13": {
        "name": "tma-thrust-moe-v13",
        "subdir": "solution/v13",
        "language": "python",
    },
    "v14": {
        "name": "tma-thrust-moe-v14",
        "subdir": "solution/v14",
        "language": "python",
    },
    "v15": {
        "name": "tma-thrust-moe-v15",
        "subdir": "solution/v15",
        "language": "python",
    },
    "v16": {
        "name": "tma-thrust-moe-v16",
        "subdir": "solution/v16",
        "language": "python",
    },
    "v17": {
        "name": "tma-thrust-moe-v17",
        "subdir": "solution/v17",
        "language": "python",
    },
}

DEFAULT_VARIANT = "triton"
SOURCE_SUFFIXES = {".py", ".cu", ".cuh", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".ptx"}


def _get_kernel_config(variant: str = DEFAULT_VARIANT) -> dict:
    v = VARIANTS[variant]
    return {
        "solution": v,
        "definition": DEFINITION_NAME,
        "author": "tma-thrust",
        "language": v.get("language", "triton"),
        "entry_point": v.get("entry_point", "kernel.py::kernel"),
        "destination_passing_style": v.get("destination_passing_style", False),
        "binding": v.get("binding"),
        "dependencies": v.get("dependencies", []),
    }


def build_solution_dict(variant: str = DEFAULT_VARIANT) -> dict:
    """Build Solution JSON dict from local source files."""
    cfg = _get_kernel_config(variant)
    source_dir = MOE_ROOT / cfg["solution"]["subdir"]

    sources = []
    for src_file in sorted(p for p in source_dir.rglob("*") if p.is_file() and p.suffix in SOURCE_SUFFIXES):
        rel_path = src_file.relative_to(source_dir).as_posix()
        sources.append({"path": rel_path, "content": src_file.read_text()})

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
            "dependencies": cfg["dependencies"],
            "destination_passing_style": cfg["destination_passing_style"],
            "binding": cfg["binding"],
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
    from flashinfer_bench import BenchmarkConfig, TraceSet

    workload_limit = None
    workload_offset = 0
    if config_dict is None:
        bench_config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
    else:
        runtime_cfg = dict(config_dict)
        workload_limit = runtime_cfg.pop("workload_limit", None)
        workload_offset = int(runtime_cfg.pop("workload_offset", 0) or 0)
        bench_config = BenchmarkConfig(**runtime_cfg)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    results = _run_solution_dicts(
        trace_set, [solution_dict], bench_config, workload_limit, workload_offset
    )
    return next(iter(results.values()))


def _run_solution_dicts(trace_set, solution_dicts, bench_config, workload_limit, workload_offset):
    from flashinfer_bench import Benchmark, Solution, TraceSet

    all_results = {}

    for solution_dict in solution_dicts:
        solution = Solution.model_validate(solution_dict)

        if solution.definition not in trace_set.definitions:
            avail = list(trace_set.definitions.keys())
            raise ValueError(f"Definition '{solution.definition}' not found. Available: {avail}")

        definition = trace_set.definitions[solution.definition]
        workloads = trace_set.workloads.get(solution.definition, [])

        if not workloads:
            raise ValueError(f"No workloads for '{solution.definition}'")

        if workload_offset > 0:
            workloads = workloads[workload_offset:]
        if workload_limit is not None:
            workloads = workloads[: int(workload_limit)]

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
                if trace.evaluation.log and trace.evaluation.status.value != "PASSED":
                    entry["log_excerpt"] = trace.evaluation.log[-3000:]
                results[trace.workload.uuid] = entry

        all_results[solution.name] = {solution.definition: results}

    return all_results


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={TRACE_SET_PATH: trace_volume},
)
def run_benchmark_batch(solution_dicts: list[dict], config_dict: dict = None) -> dict:
    """Run multiple solution variants in one Modal submission."""
    from flashinfer_bench import BenchmarkConfig, TraceSet

    workload_limit = None
    workload_offset = 0
    if config_dict is None:
        bench_config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
    else:
        runtime_cfg = dict(config_dict)
        workload_limit = runtime_cfg.pop("workload_limit", None)
        workload_offset = int(runtime_cfg.pop("workload_offset", 0) or 0)
        bench_config = BenchmarkConfig(**runtime_cfg)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    return _run_solution_dicts(trace_set, solution_dicts, bench_config, workload_limit, workload_offset)


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
            log_excerpt = result.get("log_excerpt")
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
            if log_excerpt:
                print(log_excerpt)
        if speedups:
            avg = sum(speedups) / len(speedups)
            print(f"\n  Average speedup: {avg:.2f}x  (N={len(speedups)})")


def print_batched_results(results_by_solution: dict):
    for solution_name, result in results_by_solution.items():
        print("\n" + "#" * 60)
        print(f"Solution: {solution_name}")
        print("#" * 60)
        print_results(result)


@app.local_entrypoint()
def main(
    warmup: int = 3,
    iters: int = 100,
    trials: int = 5,
    variant: str = DEFAULT_VARIANT,
    workload_limit: int = 0,
    workload_offset: int = 0,
):
    """
    Run MoE kernel benchmark on Modal B200.

    --warmup:  number of warmup runs
    --iters:   number of benchmark iterations
    --trials:  number of trials
    --variant: solution variant (triton, cuda)
    --workload-limit: run only the first N workloads for cheaper smoke tests (0 = all)
    --workload-offset: skip the first N workloads before applying the limit
    """
    variant_names = [v.strip() for v in variant.split(",") if v.strip()]
    unknown = [v for v in variant_names if v not in VARIANTS]
    if unknown:
        print(f"Unknown variant(s) {unknown}. Available: {list(VARIANTS.keys())}")
        sys.exit(1)

    config_dict = {
        "warmup_runs": warmup,
        "iterations": iters,
        "num_trials": trials,
        "workload_limit": workload_limit if workload_limit > 0 else None,
        "workload_offset": workload_offset if workload_offset > 0 else 0,
    }

    print(f"Packing MoE solution(s): {variant_names}")
    sol_dicts = [build_solution_dict(v) for v in variant_names]
    for sol_dict in sol_dicts:
        print(f"  -> {sol_dict['name']}")

    print("\nRunning benchmark on Modal B200...")
    if len(sol_dicts) == 1:
        result = run_benchmark.remote(sol_dicts[0], config_dict)
        print("\n" + "=" * 60)
        print("MoE BENCHMARK RESULTS")
        print_results(result)
    else:
        results = run_benchmark_batch.remote(sol_dicts, config_dict)
        print("\n" + "=" * 60)
        print("MoE BENCHMARK RESULTS")
        print_batched_results(results)

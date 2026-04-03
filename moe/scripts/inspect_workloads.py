"""
Inspect MoE workloads stored in the shared Modal trace volume.

Usage:
    modal run moe/scripts/inspect_workloads.py
"""

from pathlib import Path

import modal

from moe.modal_common import DEFINITION_NAME, TRACE_SET_PATH, image, trace_volume

app = modal.App("tma-thrust-moe-inspect")


@app.function(image=image, volumes={TRACE_SET_PATH: trace_volume})
def show_workloads():
    from flashinfer_bench import TraceSet

    trace_set = TraceSet.from_path(Path(TRACE_SET_PATH))
    workloads = trace_set.workloads.get(DEFINITION_NAME, [])
    print(f"definition={DEFINITION_NAME} count={len(workloads)}")
    for idx, workload in enumerate(workloads):
        data = workload.model_dump()
        workload_meta = data.get("workload", {})
        axes = dict(workload_meta.get("axes", {}) or {})
        uuid = workload_meta.get("uuid") or data.get("uuid") or "unknown"
        seq_len = axes.get("seq_len")
        print(f"{idx:02d} {uuid} seq_len={seq_len} axes={axes}")


@app.local_entrypoint()
def main():
    show_workloads.remote()

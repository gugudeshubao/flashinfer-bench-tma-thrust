"""Modal stage profiler for DSA Triton paths."""

from pathlib import Path

import modal


DSA_ROOT = Path(__file__).resolve().parents[1]

app = modal.App("tma-thrust-dsa-profile")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "triton")
    .add_local_dir(DSA_ROOT, remote_path="/root/dsa")
)


@app.function(image=image, gpu="B200:1", timeout=3600)
def run_profile(iters: int = 20) -> dict:
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")

    from dsa.common.reference import _normalize_attn_mask, build_causal_mask
    from dsa.prefill.solution.triton.kernel import (
        _launch_triton_latent,
        _prepare_query_and_value,
        _project_output,
        _select_sparse_metadata,
        kernel as prefill_solution_kernel,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    def make_inputs(query_len: int, key_len: int, topk: int):
        batch_size = 1
        num_heads = 16
        qk_nope_head_dim = 32
        rope_dim = 16
        v_head_dim = 32
        kv_rank = 64
        num_index_heads = 8
        index_dim = 32

        q_nope = torch.randn(batch_size, query_len, num_heads, qk_nope_head_dim, device=device, dtype=dtype)
        q_pe = torch.randn(batch_size, query_len, num_heads, rope_dim, device=device, dtype=dtype)
        compressed_kv = torch.randn(batch_size, key_len, kv_rank, device=device, dtype=dtype)
        k_pe = torch.randn(batch_size, key_len, rope_dim, device=device, dtype=dtype)
        wkv_b = torch.randn(num_heads, qk_nope_head_dim + v_head_dim, kv_rank, device=device, dtype=dtype)
        index_q = torch.randn(batch_size, query_len, num_index_heads, index_dim, device=device, dtype=dtype)
        index_k = torch.randn(batch_size, key_len, index_dim, device=device, dtype=dtype)
        index_weights = torch.randn(batch_size, query_len, num_index_heads, device=device, dtype=torch.float32)
        attn_mask = build_causal_mask(query_len, key_len, device=device) if query_len == key_len else None
        return {
            "q_nope": q_nope,
            "q_pe": q_pe,
            "compressed_kv": compressed_kv,
            "k_pe": k_pe,
            "wkv_b": wkv_b,
            "index_q": index_q,
            "index_k": index_k,
            "index_weights": index_weights,
            "topk": topk,
            "attn_mask": attn_mask,
            "qk_nope_head_dim": qk_nope_head_dim,
            "rope_dim": rope_dim,
        }

    def _sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def profile_case(name: str, inputs: dict) -> dict:
        attn_mask_f = _normalize_attn_mask(
            inputs["attn_mask"],
            batch_size=inputs["q_nope"].shape[0],
            query_len=inputs["q_nope"].shape[1],
            key_len=inputs["compressed_kv"].shape[1],
            device=inputs["q_nope"].device,
        )
        q_nope_proj, q_rope, value_proj = _prepare_query_and_value(
            q_nope=inputs["q_nope"],
            q_pe=inputs["q_pe"],
            wkv_b=inputs["wkv_b"],
            qk_nope_head_dim=inputs["qk_nope_head_dim"],
        )
        _index_scores, topk_indices, _sparse_mask, selected_mask = _select_sparse_metadata(
            index_q=inputs["index_q"],
            index_k=inputs["index_k"],
            index_weights=inputs["index_weights"],
            topk=inputs["topk"],
            topk_indices=None,
            attn_mask_f=attn_mask_f,
            index_scale=None,
            key_len=inputs["compressed_kv"].shape[1],
            need_sparse_mask=False,
            causal_mask=inputs["attn_mask"] is not None and inputs["q_nope"].shape[1] == inputs["compressed_kv"].shape[1],
            approximate_scores=inputs["q_nope"].shape[1] > 1,
        )
        latent = _launch_triton_latent(
            q_nope_proj=q_nope_proj,
            q_rope=q_rope,
            compressed_kv=inputs["compressed_kv"],
            k_pe=inputs["k_pe"],
            topk_indices=topk_indices,
            selected_mask=selected_mask,
            scale=float((inputs["qk_nope_head_dim"] + inputs["rope_dim"]) ** -0.5),
            causal_mask=inputs["attn_mask"] is not None and inputs["q_nope"].shape[1] == inputs["compressed_kv"].shape[1],
            num_warps_override=None,
            num_stages_override=None,
        )
        _ = _project_output(latent, value_proj, inputs["q_nope"].dtype)
        _sync()

        prepare_ms = 0.0
        select_ms = 0.0
        kernel_ms = 0.0
        project_ms = 0.0
        total_ms = 0.0

        for _ in range(iters):
            attn_mask_f = _normalize_attn_mask(
                inputs["attn_mask"],
                batch_size=inputs["q_nope"].shape[0],
                query_len=inputs["q_nope"].shape[1],
                key_len=inputs["compressed_kv"].shape[1],
                device=inputs["q_nope"].device,
            )

            _sync()
            t0 = time.perf_counter()
            q_nope_proj, q_rope, value_proj = _prepare_query_and_value(
                q_nope=inputs["q_nope"],
                q_pe=inputs["q_pe"],
                wkv_b=inputs["wkv_b"],
                qk_nope_head_dim=inputs["qk_nope_head_dim"],
            )
            _sync()
            t1 = time.perf_counter()

            index_scores, topk_indices, _sparse_mask, selected_mask = _select_sparse_metadata(
                index_q=inputs["index_q"],
                index_k=inputs["index_k"],
                index_weights=inputs["index_weights"],
                topk=inputs["topk"],
                topk_indices=None,
                attn_mask_f=attn_mask_f,
                index_scale=None,
                key_len=inputs["compressed_kv"].shape[1],
                need_sparse_mask=False,
                causal_mask=inputs["attn_mask"] is not None and inputs["q_nope"].shape[1] == inputs["compressed_kv"].shape[1],
                approximate_scores=inputs["q_nope"].shape[1] > 1,
            )
            _sync()
            t2 = time.perf_counter()

            latent = _launch_triton_latent(
                q_nope_proj=q_nope_proj,
                q_rope=q_rope,
                compressed_kv=inputs["compressed_kv"],
                k_pe=inputs["k_pe"],
                topk_indices=topk_indices,
                selected_mask=selected_mask,
                scale=float((inputs["qk_nope_head_dim"] + inputs["rope_dim"]) ** -0.5),
                causal_mask=inputs["attn_mask"] is not None and inputs["q_nope"].shape[1] == inputs["compressed_kv"].shape[1],
                num_warps_override=None,
                num_stages_override=None,
            )
            _sync()
            t3 = time.perf_counter()

            _ = _project_output(latent, value_proj, inputs["q_nope"].dtype)
            _sync()
            t4 = time.perf_counter()

            prepare_ms += (t1 - t0) * 1000.0
            select_ms += (t2 - t1) * 1000.0
            kernel_ms += (t3 - t2) * 1000.0
            project_ms += (t4 - t3) * 1000.0
            total_ms += (t4 - t0) * 1000.0

        return {
            "name": name,
            "prepare_ms": prepare_ms / iters,
            "select_ms": select_ms / iters,
            "kernel_ms": kernel_ms / iters,
            "project_ms": project_ms / iters,
            "total_ms": total_ms / iters,
        }

    def confirm_forced_triton(inputs: dict) -> None:
        _ = prefill_solution_kernel(
            inputs["q_nope"],
            inputs["q_pe"],
            inputs["compressed_kv"],
            inputs["k_pe"],
            inputs["wkv_b"],
            inputs["index_q"],
            inputs["index_k"],
            inputs["index_weights"],
            topk=inputs["topk"],
            attn_mask=inputs["attn_mask"],
            backend="triton",
            causal_mask_hint=inputs["attn_mask"] is not None and inputs["q_nope"].shape[1] == inputs["compressed_kv"].shape[1],
        )
        _sync()

    torch.manual_seed(0)
    prefill_case = make_inputs(query_len=1024, key_len=1024, topk=128)
    decode_case = make_inputs(query_len=1, key_len=8192, topk=128)
    confirm_forced_triton(prefill_case)
    confirm_forced_triton(decode_case)

    return {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "iters": iters,
        "prefill": profile_case("prefill_1024_128", prefill_case),
        "decode": profile_case("decode_8192_128", decode_case),
    }


def _print_case(case: dict) -> None:
    print(case["name"])
    print(
        f"  prepare={case['prepare_ms']:.3f}ms "
        f"select={case['select_ms']:.3f}ms "
        f"kernel={case['kernel_ms']:.3f}ms "
        f"project={case['project_ms']:.3f}ms "
        f"total={case['total_ms']:.3f}ms"
    )


@app.local_entrypoint()
def main(iters: int = 20) -> None:
    results = run_profile.remote(iters=iters)
    print(f"Device: {results['device']} ({results.get('gpu_name')})")
    print(f"Iters: {results['iters']}")
    print()
    _print_case(results["prefill"])
    print()
    _print_case(results["decode"])

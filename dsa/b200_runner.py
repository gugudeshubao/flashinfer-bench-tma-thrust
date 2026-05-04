"""
统一的 Modal B200 入口文件。

后续所有"在 B200 上跑一段实验"的需求，都通过往本文件追加 @app.function 来扩展，
而不是再开新文件。这样所有 entry 共享同一个 modal.App，按需选择 image。

当前提供的 entry：
  [默认 image：debian_slim + pip torch，首次 build ~3-5 min]
  1. gpu_info        —— 仅打印 GPU 元信息（最快、~5 秒，链路验证用）
  2. smoke           —— PyTorch + CUDA 烟测（BF16 matmul + bandwidth）
  3. bench_gemm      —— BF16 GEMM 多 shape benchmark (1K~16K square)
  4. bench_bf16      —— B200 BF16 算力 vs 标称 2382 TFLOPS 的利用率扫描
  5. bench_bandwidth —— HBM3e 真实带宽 vs 标称 7672 GB/s 的尺寸扫描
  6. shell_cmd       —— 在 B200 容器跑任意 shell 命令（探查用）

  [CUDA-devel image：nvidia/cuda:13.x-devel + cutlass-dsl，首次 build ~13 min，之后缓存]
  7. cuda_env_check         —— 验证 image_with_cuda 里 nvcc/cuobjdump/cutlass-dsl 都装好
  8. dump_cute_dsl_cubin    —— 在 B200 (sm_100) 上跑 CuTe DSL dense_gemm + cuobjdump 反汇编
                                （受阻：4.4.1 wheel 不带 example，需 git clone 才能用）
  9. disasm_cublas          —— ★ 反汇编 PyTorch wheel 自带的 libcublas/libcublasLt/libcudnn
                                so 文件，统计 sm_100 cubin 里 UMMA/UTCMMA/UTMA/HMMA/QMMA
                                指令次数。**不依赖 cutlass-dsl**，直接看 NVIDIA 自家成熟库
                                到底用没用 tcgen05 SASS（V6 真相最硬证据来源）
 10. ptxas_tcgen05_check    —— ★ 直接喂 ptxas 8 条最小 tcgen05 PTX × 6 个 arch
                                (sm_100/100a/110/110a/120/120a) = 48 case，验证 ptxas 13.0
                                自身到底支不支持 tcgen05 codegen（一刀切证据）。
                                ptxas 是 CPU 工具：在 B200 容器跑 sm_120 的结果 = 5090
                                上的结果；跑 sm_110 的结果 = Thor 上的结果（同 CUDA 版本）。

调用方式（必须从仓库根 flashinfer-bench-tma-thrust/ 执行）:

  modal run dsa/b200_runner.py::gpu_info
  modal run dsa/b200_runner.py::smoke
  modal run dsa/b200_runner.py::bench_gemm --warmup 3 --iters 20
  modal run dsa/b200_runner.py::bench_bf16
  modal run dsa/b200_runner.py::bench_bandwidth
  modal run dsa/b200_runner.py::shell_cmd --cmd "nvidia-smi -q | head -40"
  modal run dsa/b200_runner.py::cuda_env_check
  modal run dsa/b200_runner.py::dump_cute_dsl_cubin
  modal run dsa/b200_runner.py::disasm_cublas
  modal run dsa/b200_runner.py::ptxas_tcgen05_check
"""

from pathlib import Path

import modal

DSA_ROOT = Path(__file__).resolve().parent

app = modal.App("tma-thrust-dsa-b200-runner")

# ---------------------------------------------------------------------------
# image（默认）：与 dsa/ 下其他 modal 脚本保持一致的最小 image。
# debian + python3.12 + torch + numpy + triton。首次构建 ~3-5 min，之后缓存。
# 适用于：纯 PyTorch / Triton 路径的 benchmark 与 profile。
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "triton")
    .add_local_dir(DSA_ROOT, remote_path="/root/dsa")
)

# ---------------------------------------------------------------------------
# image_with_cuda（重型）：基于 nvidia/cuda:13.0.2-devel，含完整 nvcc/cuobjdump/ptxas。
# 仅供需要"自编 CUDA kernel"或"反汇编 CuTe DSL cubin"的 entry 使用。
# 首次构建 ~10-15 min（拉 5GB+ 镜像 + pip 装 torch + cutlass-dsl），之后 Modal 缓存。
# ---------------------------------------------------------------------------
image_with_cuda = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "torch",
        "numpy",
        "triton",
        "nvidia-cutlass-dsl==4.4.1",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        }
    )
    .add_local_dir(DSA_ROOT, remote_path="/root/dsa")
)


# ---------------------------------------------------------------------------
# 1) gpu_info —— 最快的链路验证（不依赖 PyTorch，仅打印硬件元信息）
# ---------------------------------------------------------------------------
@app.function(image=image, gpu="B200:1", timeout=300)
def gpu_info() -> dict:
    """打印 B200 的 nvidia-smi 元信息 + driver / CUDA runtime 版本。"""
    import platform
    import shutil
    import subprocess

    info = {
        "host_python": platform.python_version(),
        "host_uname": platform.uname()._asdict(),
    }

    # nvidia-smi 三段式：列表、详细 query、版本
    if shutil.which("nvidia-smi"):
        info["nvidia_smi_L"] = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True
        ).stdout.strip()

        info["nvidia_smi_query"] = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,memory.total,memory.free,"
                "clocks.max.sm,clocks.max.mem,driver_version,vbios_version",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()

        info["nvidia_smi_version"] = subprocess.run(
            ["nvidia-smi", "--version"], capture_output=True, text=True
        ).stdout.strip()

    # PyTorch 视角（**所有值强制转纯 Python 类型，避免本地反序列化时依赖 torch**）
    try:
        import torch

        info["torch_version"] = str(torch.__version__)
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["torch_cuda_device_count"] = int(torch.cuda.device_count())
            info["torch_cuda_device_0_name"] = str(torch.cuda.get_device_name(0))
            cap = torch.cuda.get_device_capability(0)
            info["torch_cuda_device_0_capability"] = f"{cap[0]}.{cap[1]}"
            info["torch_cuda_runtime_version"] = str(torch.version.cuda)
    except Exception as exc:  # pragma: no cover
        info["torch_error"] = repr(exc)

    print("=" * 72)
    print("Modal B200 runner :: gpu_info")
    print("=" * 72)
    for key, value in info.items():
        print(f"\n[{key}]")
        print(value)
    print("=" * 72)

    return info


# ---------------------------------------------------------------------------
# 2) smoke —— PyTorch + CUDA 烟测（确认 BF16 matmul + HBM 拷贝都能跑）
# ---------------------------------------------------------------------------
@app.function(image=image, gpu="B200:1", timeout=600)
def smoke() -> dict:
    """烟测：BF16 matmul 算力 + HBM 拷贝带宽，验证 B200 + PyTorch 链路。"""
    import time

    import torch

    assert torch.cuda.is_available(), "CUDA not available on B200 container"
    device = "cuda"
    dtype = torch.bfloat16

    cap = torch.cuda.get_device_capability(0)
    result: dict = {
        "device_name": str(torch.cuda.get_device_name(0)),
        "device_capability": f"{cap[0]}.{cap[1]}",
        "torch_version": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
    }

    # ---- 4Kx4K BF16 matmul（warmup + 计时）-----------------------------
    n = 4096
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)

    # warmup
    for _ in range(3):
        _ = a @ b
    torch.cuda.synchronize()

    iters = 20
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        c = a @ b
    end.record()
    torch.cuda.synchronize()
    matmul_ms = start.elapsed_time(end) / iters
    flops = 2.0 * n * n * n
    matmul_tflops = flops / (matmul_ms / 1e3) / 1e12
    result["matmul_4k_bf16_ms"] = round(matmul_ms, 4)
    result["matmul_4k_bf16_tflops"] = round(matmul_tflops, 1)

    # ---- HBM 拷贝带宽（device-to-device copy）---------------------------
    nbytes_gb = 1.0
    elems = int(nbytes_gb * 1024**3 // 2)  # bf16 = 2 bytes
    src = torch.empty(elems, device=device, dtype=dtype)
    dst = torch.empty(elems, device=device, dtype=dtype)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize()
    copy_ms = start.elapsed_time(end) / iters
    copy_gbps = (nbytes_gb * 2) / (copy_ms / 1e3)  # 读+写
    result["d2d_copy_1gb_ms"] = round(copy_ms, 4)
    result["d2d_copy_gbps"] = round(copy_gbps, 1)

    # 数值 sanity check
    result["matmul_finite"] = bool(torch.isfinite(c).all().item())

    print("=" * 72)
    print("Modal B200 runner :: smoke")
    print("=" * 72)
    for key, value in result.items():
        print(f"  {key:32s} = {value}")
    print("=" * 72)
    return result


# ---------------------------------------------------------------------------
# 3) bench_gemm —— 一组 BF16 GEMM benchmark（4K/8K/16K square）
# ---------------------------------------------------------------------------
@app.function(image=image, gpu="B200:1", timeout=1800)
def bench_gemm(warmup: int = 3, iters: int = 20) -> dict:
    """BF16 GEMM 多 shape benchmark。返回每个 shape 的 TFLOPS。"""
    import torch

    assert torch.cuda.is_available()
    device = "cuda"
    dtype = torch.bfloat16

    shapes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (16384, 16384, 16384),
    ]
    results = []

    for m, n, k in shapes:
        try:
            a = torch.randn(m, k, device=device, dtype=dtype)
            b = torch.randn(k, n, device=device, dtype=dtype)

            for _ in range(warmup):
                _ = a @ b
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                c = a @ b
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / iters
            flops = 2.0 * m * n * k
            tflops = flops / (ms / 1e3) / 1e12
            entry = {
                "shape": f"{m}x{n}x{k}",
                "ms": round(ms, 4),
                "tflops": round(tflops, 1),
                "finite": bool(torch.isfinite(c).all().item()),
            }
            del a, b, c
            torch.cuda.empty_cache()
        except RuntimeError as exc:  # OOM 等
            entry = {"shape": f"{m}x{n}x{k}", "error": repr(exc)}
        results.append(entry)
        print(entry)

    return {"device": str(torch.cuda.get_device_name(0)), "results": results}


# ---------------------------------------------------------------------------
# 3.5) bench_bf16 —— B200 BF16 算力 vs 标称 2382 TFLOPS 的利用率扫描
# ---------------------------------------------------------------------------
@app.function(image=image, gpu="B200:1", timeout=1800)
def bench_bf16(warmup: int = 5, iters: int = 30) -> dict:
    """B200 BF16 GEMM 算力实测，按 shape 报 TFLOPS / 标称利用率 / vs HBM roofline。

    标称：sm_100 BF16 Tensor Core peak ≈ 2382 TFLOPS（148 SM × 4096 FMA/clk × 2 × 1.965 GHz）
    HBM roofline：7672 GB/s（理论），arithmetic intensity = 2*M*N*K / (2*(M*K+K*N+M*N)) ops/byte

    对每个 shape 输出：
      - achieved_tflops
      - peak_utilization (%) = achieved / 2382
      - is_compute_bound (M*N 大于一定阈值时)
    """
    import torch

    assert torch.cuda.is_available()
    device = "cuda"
    dtype = torch.bfloat16

    PEAK_BF16_TFLOPS = 2382.0  # B200 sm_100 标称

    # 覆盖典型 LLM 矩阵尺寸（含非方阵 GEMM-N 形态：训练 fwd / decode-1 / prefill）
    shapes = [
        # (M, N, K, label)
        (1024, 1024, 1024, "1K square"),
        (2048, 2048, 2048, "2K square"),
        (4096, 4096, 4096, "4K square (compute-bound 起点)"),
        (8192, 8192, 8192, "8K square (FA4 sweet spot)"),
        (16384, 16384, 16384, "16K square (long-ctx prefill)"),
        # 非方阵：模拟训练前向 (B*S, hidden, hidden)
        (8192, 14336, 4096, "Llama-70B-MLP fwd-up"),
        (8192, 4096, 14336, "Llama-70B-MLP fwd-down"),
        # decode-1：极端 skinny GEMM（memory-bound）
        (1, 4096, 4096, "decode-1 4K (HBM-bound)"),
        (1, 14336, 4096, "decode-1 70B-MLP (HBM-bound)"),
    ]
    results = []

    for m, n, k, label in shapes:
        try:
            a = torch.randn(m, k, device=device, dtype=dtype)
            b = torch.randn(k, n, device=device, dtype=dtype)

            for _ in range(warmup):
                _ = a @ b
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                c = a @ b
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / iters

            flops = 2.0 * m * n * k
            tflops = flops / (ms / 1e3) / 1e12
            util = tflops / PEAK_BF16_TFLOPS * 100

            # AI = ops / bytes_moved
            bytes_moved = 2 * (m * k + k * n + m * n)  # bf16=2B
            ai = flops / bytes_moved

            entry = {
                "label": label,
                "shape": f"{m}x{n}x{k}",
                "ms": round(ms, 4),
                "tflops": round(tflops, 1),
                "peak_util_pct": round(util, 1),
                "arith_intensity": round(ai, 2),
                "finite": bool(torch.isfinite(c).all().item()),
            }
            del a, b, c
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            entry = {"label": label, "shape": f"{m}x{n}x{k}", "error": repr(exc)}
        results.append(entry)
        print(entry)

    return {
        "device": str(torch.cuda.get_device_name(0)),
        "peak_bf16_tflops_nominal": PEAK_BF16_TFLOPS,
        "results": results,
    }


# ---------------------------------------------------------------------------
# 3.6) bench_bandwidth —— HBM3e 真实带宽 vs 标称 7672 GB/s 的尺寸扫描
# ---------------------------------------------------------------------------
@app.function(image=image, gpu="B200:1", timeout=1800)
def bench_bandwidth(warmup: int = 5, iters: int = 30) -> dict:
    """B200 HBM3e 真实带宽扫描：device-to-device copy + memset + add 三种模式。

    标称：7672 GB/s（bus_width 7680 bit × mem_clk 3996 MHz × 2 / 8）

    对不同尺寸（1MB ~ 16GB）跑：
      - copy: dst.copy_(src)            读 + 写
      - memset: tensor.zero_()          仅写
      - axpy: y.add_(x, alpha=2.0)      读 x + 读写 y
    """
    import torch

    assert torch.cuda.is_available()
    device = "cuda"
    dtype = torch.bfloat16
    elem_size = 2  # bf16

    PEAK_HBM_GBPS = 7672.0

    sizes_mb = [1, 16, 256, 1024, 4096, 16384]  # 1MB → 16GB
    results = []

    def time_op(op, expected_bytes):
        for _ in range(warmup):
            op()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            op()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
        gbps = expected_bytes / (ms / 1e3) / 1e9
        return ms, gbps

    for size_mb in sizes_mb:
        try:
            elems = size_mb * 1024 * 1024 // elem_size
            x = torch.empty(elems, device=device, dtype=dtype)
            y = torch.empty(elems, device=device, dtype=dtype)
            entry = {"size_mb": size_mb}

            # copy: 读 src + 写 dst = 2× size
            ms, gbps = time_op(lambda: y.copy_(x), elems * elem_size * 2)
            entry["copy_ms"] = round(ms, 4)
            entry["copy_gbps"] = round(gbps, 1)
            entry["copy_util_pct"] = round(gbps / PEAK_HBM_GBPS * 100, 1)

            # memset: 仅写
            ms, gbps = time_op(lambda: x.zero_(), elems * elem_size)
            entry["memset_gbps"] = round(gbps, 1)

            # axpy y = y + 2*x: 读 x + 读 y + 写 y = 3× size
            ms, gbps = time_op(lambda: y.add_(x, alpha=2.0), elems * elem_size * 3)
            entry["axpy_gbps"] = round(gbps, 1)
            entry["axpy_util_pct"] = round(gbps / PEAK_HBM_GBPS * 100, 1)

            del x, y
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            entry = {"size_mb": size_mb, "error": repr(exc)}
        results.append(entry)
        print(entry)

    return {
        "device": str(torch.cuda.get_device_name(0)),
        "peak_hbm_gbps_nominal": PEAK_HBM_GBPS,
        "results": results,
    }

# ---------------------------------------------------------------------------
# 7) disasm_cublas —— 反汇编 PyTorch wheel 自带的 NVIDIA 库（不依赖 cutlass）
# ---------------------------------------------------------------------------
@app.function(image=image_with_cuda, gpu="B200:1", timeout=1800)
def disasm_cublas(libs: str = "auto") -> dict:
    """反汇编 NVIDIA 自家库的 sm_100 cubin，统计 tcgen05 / TMA / 普通 mma 指令出现次数。

    回答的核心问题：**NVIDIA 自家成熟生产库（cuBLAS/cuBLASLt/cuDNN）在 sm_100 上
    到底用没用 tcgen05 SASS？** 这是 V6 真相最直接的证据来源——
    比通过 PyTorch 黑箱测利用率（67.8%）更硬。

    SASS 关键字符串识别：
      - UTCMMA / UMMA  : tcgen05 mma SASS（5th gen tensor core，sm_100/110 旗舰）
      - UTCMOV         : tcgen05 ldsm/stsm
      - UTMA           : TMA SASS
      - HMMA           : 普通 sm_90/100/120 hopper-style mma
      - QMMA           : FP4/FP8 mma
      - DGMMA / IGMMA  : sm_90 hopper warpgroup mma 变体（参考）

    Args:
        libs: 逗号分隔的 .so 路径或文件名关键字，'auto' = 默认扫描 cublas/cublasLt/cudnn。
    """
    import shutil
    import subprocess
    from pathlib import Path

    cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"

    # 1. 找候选 .so 文件
    if libs == "auto":
        # PyTorch wheel 自带 NVIDIA libs 的标准位置
        roots = [
            Path("/usr/local/lib/python3.12/site-packages/nvidia_cublas/lib"),
            Path("/usr/local/lib/python3.12/site-packages/nvidia_cudnn/lib"),
            Path("/usr/local/lib/python3.12/site-packages/torch/lib"),
            Path("/usr/local/cuda/lib64"),
        ]
        keywords = ("libcublas", "libcudnn")
        candidates = []
        for r in roots:
            if not r.exists():
                continue
            for p in r.iterdir():
                if p.is_file() and any(k in p.name for k in keywords):
                    # 跳过 symlink，留实际 .so 文件
                    if p.is_symlink():
                        continue
                    candidates.append(p)
    else:
        candidates = [Path(p.strip()) for p in libs.split(",") if p.strip()]

    print(f"[disasm_cublas] {len(candidates)} candidate libs:")
    for c in candidates:
        print(f"  {c} ({c.stat().st_size / 1e6:.1f} MB)" if c.exists() else f"  {c} (MISSING)")

    # 2. 对每个 lib：list-elf 看包含哪些 sm 架构 → 提取 sm_100 cubin → dump-sass → 统计
    results = []
    SASS_KEYS = ("UMMA", "UTCMMA", "UTCMOV", "UTMA", "HMMA", "QMMA", "DGMMA", "IGMMA")

    for lib in candidates:
        if not lib.exists():
            results.append({"lib": str(lib), "error": "FILE_NOT_FOUND"})
            continue

        entry: dict = {
            "lib": str(lib),
            "size_mb": round(lib.stat().st_size / 1e6, 1),
        }

        # 2a. list-elf：看包含哪些 sm 架构
        try:
            elf_list = subprocess.run(
                [cuobjdump, "--list-elf", str(lib)],
                capture_output=True, text=True, timeout=120,
            ).stdout
            # 抽出形如 "arch = sm_100" 或 "...sm_100..." 的行
            import re
            arches = sorted(set(re.findall(r"sm_\d+\w*", elf_list)))
            entry["arches"] = arches
            entry["elf_count"] = elf_list.count("ELF")
        except Exception as exc:
            entry["list_elf_error"] = repr(exc)
            results.append(entry)
            continue

        # 2b. 对所有 sm_100 / sm_100a 单独 dump-sass
        sm100_arches = [a for a in arches if a.startswith("sm_100")]
        per_arch = {}
        for arch in sm100_arches:
            try:
                # cuobjdump 可以按 -arch 过滤
                sass = subprocess.run(
                    [cuobjdump, "--dump-sass", "-arch", arch, str(lib)],
                    capture_output=True, text=True, timeout=300,
                ).stdout
                counts = {k: sass.count(k) for k in SASS_KEYS}
                # 总指令行数估算（每行一条 SASS）
                total_lines = sass.count("/*")
                # 抓出现的 unique kernel 数
                kernels = sass.count("Function :")
                per_arch[arch] = {
                    "sass_size_kb": round(len(sass) / 1024, 1),
                    "total_instr_lines": total_lines,
                    "kernel_count": kernels,
                    "instr_counts": counts,
                }
            except Exception as exc:
                per_arch[arch] = {"error": repr(exc)}
        entry["sm100_dump"] = per_arch
        results.append(entry)
        print(f"  {lib.name}: arches={arches}, sm_100_dump={per_arch}")

    # 3. 汇总：跨所有 lib 的 sm_100 SASS 总指令计数
    grand_total = {k: 0 for k in SASS_KEYS}
    for r in results:
        for arch, d in r.get("sm100_dump", {}).items():
            for k in SASS_KEYS:
                grand_total[k] += d.get("instr_counts", {}).get(k, 0)

    print("=" * 72)
    print("Modal B200 runner :: disasm_cublas (sm_100 grand total)")
    print("=" * 72)
    for k in SASS_KEYS:
        print(f"  {k:8s} = {grand_total[k]}")
    print("=" * 72)

    return {
        "candidates": [str(c) for c in candidates],
        "per_lib": results,
        "sm100_grand_total": grand_total,
    }


# ---------------------------------------------------------------------------
# 8) disasm_fa4 —— 反汇编 FlashAttention-4 的 cubin，验证 L4 PyPI 路径在 sm_100a
#    上是否真能 emit UTCMMA / UMMA SASS（FA4 反例硬证据，回答主文档 §0.5 的核心断言）
# ---------------------------------------------------------------------------
@app.function(image=image_with_cuda, gpu="B200:1", timeout=1800)
def disasm_fa4() -> dict:
    """反汇编 flash-attn-4 (CuTeDSL 实现) 的 cubin，统计 sm_100a 上的 tcgen05 SASS 出现次数。

    回答的核心问题：**FlashAttention-4（Tri Dao 团队，2025，针对 B200，走 CuTeDSL 路径）
    在 sm_100a 上是否真的 emit 出 UTCMMA / UMMA SASS？**

    这是主文档 §0.5 "tcgen05 可用性 5 层模型" 中 L4 PyPI 工具链断言的硬证据兜底。

    背景：
    - V7 实测（ptxas_tcgen05_check）证实 L5 公开 ptxas 13.0.88 对 6 arch × 8 instr 全拒/假接受
    - 但 FA4 走的是 L4 路径（pip install flash-attn-4，基于 nvidia-cutlass-dsl）
    - L4 路径不经过 ptxas，而是 CuTeDSL → MLIR → LLVM NVPTX backend → 直接 emit cubin
    - 如果 LLVM NVPTX backend 真的 ship 了 sm_100a tcgen05 codegen，FA4 cubin 里应该有 UTCMMA SASS

    SASS 关键字符串识别（同 disasm_cublas）：
      - UTCMMA / UMMA  : tcgen05 mma SASS（5th gen tensor core，sm_100/110）
      - UTCMOV         : tcgen05 ldsm/stsm
      - UTMA           : TMA SASS
      - HMMA           : 普通 sm_90/100/120 hopper-style mma
      - QMMA           : FP4/FP8 mma

    成功路径（预期）：sm_100a SASS 里 UTCMMA + UMMA > 0 → 验证主文档 §0.5 L4 断言
    失败路径（兜底）：sm_100a SASS 里 UTCMMA + UMMA = 0 → 主文档 §0.5 需进一步软化
    """
    import shutil
    import subprocess
    from pathlib import Path

    cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    pip = shutil.which("pip") or "/usr/local/bin/pip"

    # 1. pip install flash-attn-4 (PyPI 包，CuTeDSL 纯 JIT 实现)
    # 注意：截至 2026-05，flash-attn-4 在 PyPI 上只有 beta 版本（4.0.0b3-b11），无稳定版
    # 关键：flash-attn-4 是纯 Python wheel（324 KB，无静态 cubin）
    #       cubin 必须在 GPU 上首次调用 kernel 时由 nvidia-cutlass-dsl JIT 生成
    print("[disasm_fa4] installing flash-attn-4==4.0.0b11 (latest beta) ...")
    install_proc = subprocess.run(
        [pip, "install", "--no-cache-dir", "flash-attn-4==4.0.0b11"],
        capture_output=True, text=True, timeout=1500,
    )
    install_log = (install_proc.stdout or "")[-2000:] + "\n--- STDERR ---\n" + (install_proc.stderr or "")[-2000:]
    print(install_log[-500:])
    if install_proc.returncode != 0:
        return {
            "error": "pip install flash-attn-4 failed",
            "rc": install_proc.returncode,
            "log_tail": install_log,
        }

    # 2. 在 B200 上真跑一次 FA4 forward，触发 JIT compile 产生 cubin
    print("[disasm_fa4] triggering JIT compile by running FA4 forward on B200 ...")
    import os
    # 设置 cutlass-dsl JIT 缓存目录到一个我们能找到的位置
    jit_cache_dir = Path("/tmp/fa4_jit_cache")
    jit_cache_dir.mkdir(exist_ok=True)
    os.environ["CUTLASS_DSL_CACHE_DIR"] = str(jit_cache_dir)
    os.environ["TRITON_CACHE_DIR"] = str(jit_cache_dir / "triton")  # 兜底
    os.environ["TVM_FFI_CACHE_DIR"] = str(jit_cache_dir / "tvm")    # 兜底

    try:
        import torch
        # 检查 GPU 可见
        if not torch.cuda.is_available():
            return {"error": "torch.cuda.is_available() = False", "install_log_tail": install_log}
        device = torch.device("cuda:0")
        print(f"[disasm_fa4] GPU: {torch.cuda.get_device_name(0)}, capability: {torch.cuda.get_device_capability(0)}")

        # 真跑 FA4 forward
        # 注意：flash-attn-4 的导入路径是 flash_attn.cute（不是 flash_attn_4）
        # 这是 Tri Dao 团队的设计，FA4 wheel 内容塞进 flash_attn.cute 子模块
        from flash_attn.cute import flash_attn_func  # type: ignore[import-not-found]
        # 标准 FA4 输入：(batch, seqlen, nheads, headdim)
        B, S, H, D = 1, 512, 8, 128
        q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
        out = flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
        if isinstance(out, tuple):
            out = out[0]
        print(f"[disasm_fa4] FA4 forward OK, out.shape={tuple(out.shape)}, dtype={out.dtype}")
    except ImportError as exc:
        # 接口名可能不一致，尝试其他常见入口
        print(f"[disasm_fa4] import flash_attn.cute.flash_attn_func failed: {exc}; trying alternatives ...")
        try:
            import flash_attn  # type: ignore[import-not-found]
            print(f"[disasm_fa4] flash_attn module dir: {dir(flash_attn)}")
            try:
                import flash_attn.cute  # type: ignore[import-not-found]
                print(f"[disasm_fa4] flash_attn.cute module dir: {dir(flash_attn.cute)}")
            except Exception as exc3:
                print(f"[disasm_fa4] flash_attn.cute import failed: {exc3!r}")
        except Exception as exc2:
            return {"error": f"FA4 import failed: {exc2!r}", "install_log_tail": install_log}
    except Exception as exc:
        # 即使 forward 跑挂了，JIT cubin 可能已经生成在缓存里，继续 dump
        print(f"[disasm_fa4] FA4 forward failed (may still have JIT cache): {exc!r}")

    # 3. 扫描 JIT 缓存目录 + flash_attn_4 包目录 + cutlass_dsl 包目录，找所有 .cubin / .fatbin / .so
    candidates = []
    scan_roots = [
        jit_cache_dir,
        Path("/tmp"),  # cutlass-dsl 也可能 dump 到 /tmp
        Path("/root/.cache"),
        Path("/root/.cutlass_dsl_cache"),
        Path.home() / ".cache",
        Path("/usr/local/lib/python3.12/site-packages"),
    ]
    target_keywords_pkg = ("flash_attn_4", "nvidia_cutlass_dsl", "quack_kernels", "apache_tvm_ffi")
    seen = set()
    for root in scan_roots:
        if not root.exists():
            continue
        # 包目录只扫指定的子目录（site-packages 太大）
        if "site-packages" in str(root):
            for sub in root.iterdir():
                if not sub.is_dir():
                    continue
                if not any(k in sub.name.lower() for k in target_keywords_pkg):
                    continue
                for path in sub.rglob("*"):
                    if path.is_symlink() or not path.is_file():
                        continue
                    if path.suffix.lower() in (".so", ".cubin", ".fatbin"):
                        if path.resolve() not in seen:
                            candidates.append(path)
                            seen.add(path.resolve())
        else:
            for path in root.rglob("*"):
                if path.is_symlink() or not path.is_file():
                    continue
                if path.suffix.lower() in (".so", ".cubin", ".fatbin"):
                    if path.resolve() not in seen:
                        candidates.append(path)
                        seen.add(path.resolve())

    print(f"[disasm_fa4] found {len(candidates)} candidate binaries (.so/.cubin/.fatbin):")
    for c in candidates[:80]:
        try:
            print(f"  {c} ({c.stat().st_size / 1e6:.2f} MB)")
        except Exception:
            pass

    if not candidates:
        return {
            "error": "no FA4 binary or JIT cubin found",
            "install_log_tail": install_log,
        }

    # 3. 对每个 binary：list-elf 看包含哪些 sm 架构 → dump-sass sm_100/sm_100a → 统计
    SASS_KEYS = ("UMMA", "UTCMMA", "UTCMOV", "UTMA", "HMMA", "QMMA", "DGMMA", "IGMMA")
    results = []

    for bin_path in candidates:
        entry: dict = {
            "path": str(bin_path),
            "size_mb": round(bin_path.stat().st_size / 1e6, 2),
        }

        # 3a. list-elf
        try:
            elf_list = subprocess.run(
                [cuobjdump, "--list-elf", str(bin_path)],
                capture_output=True, text=True, timeout=120,
            ).stdout
            import re
            arches = sorted(set(re.findall(r"sm_\d+\w*", elf_list)))
            entry["arches"] = arches
            entry["elf_count"] = elf_list.count("ELF")
        except Exception as exc:
            entry["list_elf_error"] = repr(exc)
            results.append(entry)
            continue

        if not entry.get("arches"):
            # 不是 cubin/fatbin，跳过
            results.append(entry)
            continue

        # 3b. 对所有 sm_100 / sm_100a / sm_110 / sm_110a / sm_120 / sm_120a 单独 dump-sass
        per_arch = {}
        target_arches = [a for a in arches
                         if a.startswith(("sm_100", "sm_110", "sm_120"))]
        for arch in target_arches:
            try:
                sass = subprocess.run(
                    [cuobjdump, "--dump-sass", "-arch", arch, str(bin_path)],
                    capture_output=True, text=True, timeout=300,
                ).stdout
                counts = {k: sass.count(k) for k in SASS_KEYS}
                kernels = sass.count("Function :")
                per_arch[arch] = {
                    "sass_size_kb": round(len(sass) / 1024, 1),
                    "kernel_count": kernels,
                    "instr_counts": counts,
                }
            except Exception as exc:
                per_arch[arch] = {"error": repr(exc)}
        entry["per_arch_dump"] = per_arch
        results.append(entry)
        print(f"  {bin_path.name}: arches={arches}")
        for arch, d in per_arch.items():
            print(f"    {arch}: {d.get('instr_counts', d)}")

    # 4. 汇总：跨所有 binary 的 sm_100a / sm_100 / sm_110 / sm_120 SASS 总指令计数
    grand_total: dict = {}
    for r in results:
        for arch, d in r.get("per_arch_dump", {}).items():
            if arch not in grand_total:
                grand_total[arch] = {k: 0 for k in SASS_KEYS}
            for k in SASS_KEYS:
                grand_total[arch][k] += d.get("instr_counts", {}).get(k, 0)

    print("=" * 72)
    print("Modal B200 runner :: disasm_fa4 (per-arch grand total)")
    print("=" * 72)
    for arch, counts in sorted(grand_total.items()):
        utcmma_total = counts.get("UMMA", 0) + counts.get("UTCMMA", 0)
        utma_total = counts.get("UTMA", 0)
        hmma_total = counts.get("HMMA", 0)
        verdict = "✅ tcgen05 SASS PRESENT" if utcmma_total > 0 else "❌ tcgen05 SASS ABSENT"
        print(f"  {arch}: UMMA+UTCMMA={utcmma_total}, UTMA={utma_total}, HMMA={hmma_total}  {verdict}")
        print(f"      full: {counts}")
    print("=" * 72)

    return {
        "candidates": [str(c) for c in candidates],
        "per_binary": results,
        "per_arch_grand_total": grand_total,
    }

# ---------------------------------------------------------------------------
# 9) ptxas_tcgen05_check —— 直接喂 ptxas 含 tcgen05.mma 的 PTX，看能不能编出来
# ---------------------------------------------------------------------------
@app.function(image=image_with_cuda, gpu="B200:1", timeout=600)
def ptxas_tcgen05_check() -> dict:
    """写最小 PTX 含 tcgen05 系列指令，逐个喂 ptxas，验证 codegen 是否真支持。

    回答的核心问题：**ptxas 13.0 自身能不能为 sm_100a / sm_110a 接受 tcgen05.mma 指令？**
    如果连 ptxas 自身都拒绝（PARSE_ERROR / unrecognized instruction），那 cuBLAS / CuTe DSL
    都不可能输出 tcgen05 SASS——一刀切证据。

    测试 8 条典型 tcgen05 指令 × 6 个目标架构（sm_100/100a/110/110a/120/120a）= 48 个 case。

    ★ 一次回答 3 张卡的真相：
    - sm_100/100a → B200 真相
    - sm_110/110a → Thor 真相（ptxas 是 CPU 工具，与 GPU 在场无关，CUDA 版本对齐即可）
    - sm_120/120a → 5090 真相（消费 Blackwell 架构层面就不带 tcgen05/TMEM/MX）
    """
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    ptxas = shutil.which("ptxas") or "/usr/local/cuda/bin/ptxas"
    cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"

    # 8 条 PTX 片段（每条都是完整可独立编译的 .ptx 文件，仅核心指令换）
    INSTRUCTIONS = {
        "tcgen05_alloc": "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r0], 32;",
        "tcgen05_relinquish": "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;",
        "tcgen05_dealloc": "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r0, 32;",
        "tcgen05_mma_f16": (
            "tcgen05.mma.cta_group::1.kind::f16 "
            "[%r0], %rd1, %rd2, %r3, p;"
        ),
        "tcgen05_mma_f8f6f4": (
            "tcgen05.mma.cta_group::1.kind::mxf8f6f4 "
            "[%r0], %rd1, %rd2, %r3, p;"
        ),
        "tcgen05_commit": (
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%rd0];"
        ),
        "tcgen05_wait_ld": "tcgen05.wait::ld.sync.aligned;",
        "tcgen05_ld_16x256b": (
            "tcgen05.ld.sync.aligned.16x256b.x1.b32 {%r0}, [%r1];"
        ),
    }

    # 标准 PTX 模板
    PTX_TEMPLATE = """//
// Generated by ptxas_tcgen05_check
//
.version {ver}
.target {arch}
.address_size 64

.visible .entry test_kernel(
    .param .u64 test_kernel_param_0
)
{{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .reg .pred p;

    ld.param.u64 %rd0, [test_kernel_param_0];
    cvta.to.global.u64 %rd1, %rd0;
    mov.b64 %rd2, %rd1;
    mov.b32 %r0, 0;
    mov.b32 %r1, 0;
    mov.b32 %r3, 0;
    setp.eq.s32 p, %r0, 0;

    {body}

    ret;
}}
"""

    # 测试 6 个 target arch × 各自合适的 PTX version：
    #   sm_100/sm_100a   B200 datacenter Blackwell（理论支持 tcgen05 全集）
    #   sm_110/sm_110a   Thor datacenter Blackwell（理论支持 tcgen05 全集，但 ptxas 13.0 codegen 缺失）
    #   sm_120/sm_120a   5090 consumer Blackwell（架构层不支持 tcgen05/TMEM/MX，应全部拒绝）
    # 注意：ptxas 是纯 CPU 工具，结果与目标 GPU 物理在场无关。
    # 所以在 B200 容器里 ptxas -arch sm_120 的结果 = 5090 上的结果；
    # 在 B200 容器里 ptxas -arch sm_110 的结果 = Thor 上的结果（同版本 CUDA 13.0.2）。
    TARGETS = [
        ("sm_100", "8.7"),
        ("sm_100a", "8.7"),
        ("sm_110", "8.8"),
        ("sm_110a", "8.8"),
        ("sm_120", "8.7"),
        ("sm_120a", "8.7"),
    ]

    workdir = Path(tempfile.mkdtemp(prefix="ptxas_tcgen05_"))
    print(f"[ptxas_tcgen05_check] workdir = {workdir}")

    # ptxas 版本
    pv = subprocess.run([ptxas, "--version"], capture_output=True, text=True)
    ptxas_version = (pv.stdout or pv.stderr).strip()
    print(f"[ptxas_tcgen05_check] ptxas version:\n{ptxas_version}")

    matrix = []
    sass_dumps = {}

    for arch, ver in TARGETS:
        for instr_name, instr_body in INSTRUCTIONS.items():
            ptx_text = PTX_TEMPLATE.format(ver=ver, arch=arch, body=instr_body)
            ptx_path = workdir / f"{arch}_{instr_name}.ptx"
            cubin_path = workdir / f"{arch}_{instr_name}.cubin"
            ptx_path.write_text(ptx_text)

            proc = subprocess.run(
                [ptxas, "-arch", arch, "-o", str(cubin_path), str(ptx_path)],
                capture_output=True, text=True, timeout=60,
            )
            ok = proc.returncode == 0 and cubin_path.exists()
            entry = {
                "arch": arch,
                "ptx_version": ver,
                "instr": instr_name,
                "accepted": ok,
                "stderr": (proc.stderr or "")[:300],
            }

            # 如果编译通过，立即反汇编看 SASS 里到底有什么
            if ok:
                try:
                    sass = subprocess.run(
                        [cuobjdump, "--dump-sass", str(cubin_path)],
                        capture_output=True, text=True, timeout=60,
                    ).stdout
                    keys = ("UMMA", "UTCMMA", "UTCMOV", "UTMA", "HMMA", "QMMA")
                    entry["sass_counts"] = {k: sass.count(k) for k in keys}
                    # 把 SASS 内容也存一份（截前 1500 字符），后面汇报用
                    sass_dumps[f"{arch}_{instr_name}"] = sass[:1500]
                except Exception as exc:
                    entry["dump_error"] = repr(exc)

            matrix.append(entry)
            status = "✅" if ok else "❌"
            print(f"  {status} {arch:8s} | {instr_name:24s} | rc={proc.returncode}")

    # 汇总：每个 arch 接受了几条指令、出了多少 UMMA/UTCMMA SASS
    summary = {}
    for arch, _ in TARGETS:
        accepted = [m for m in matrix if m["arch"] == arch and m["accepted"]]
        rejected = [m for m in matrix if m["arch"] == arch and not m["accepted"]]
        umma_total = sum(
            m.get("sass_counts", {}).get("UMMA", 0)
            + m.get("sass_counts", {}).get("UTCMMA", 0)
            for m in accepted
        )
        summary[arch] = {
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "rejected_instrs": [m["instr"] for m in rejected],
            "umma_utcmma_total": umma_total,
        }

    print("=" * 72)
    print("Modal B200 runner :: ptxas_tcgen05_check (per-arch summary)")
    print("=" * 72)
    for arch, s in summary.items():
        print(f"  {arch:8s}: accepted={s['accepted_count']}/8, "
              f"UMMA+UTCMMA={s['umma_utcmma_total']}, "
              f"rejected={s['rejected_instrs']}")
    print("=" * 72)

    return {
        "ptxas_version": ptxas_version.split("\n")[0],
        "matrix": matrix,
        "summary": summary,
        "sass_samples": {k: v[:500] for k, v in list(sass_dumps.items())[:4]},
    }


# ---------------------------------------------------------------------------
# 10) disasm_sanity_check —— 反汇编可信度自检（response to 用户 "你得确定你的 cubin
#     反汇编的方法是可信的" 的质疑）
# ---------------------------------------------------------------------------
@app.function(image=image_with_cuda, gpu="B200:1", timeout=900)
def disasm_sanity_check() -> dict:
    """反汇编方法学闭环验证：跑 3 个 sub-experiment，量化 cuobjdump --dump-sass +
    sass.count(key) 的可信度。

    A. **已知真值 kernel**：写一个最小 BF16 mma.sync.m16n8k16 PTX kernel，nvcc 编 sm_90
       和 sm_100a，反汇编后**预期 emit HMMA**。验证 cuobjdump --dump-sass + 精确正则
       提取 mnemonic 直方图能正确识别已知真值。

    B. **sm_100a 假接受 5 case 完整 SASS dump**：复跑 ptxas_tcgen05_check 中接受的
       5 个 sm_100a case，但这次输出**完整 mnemonic 直方图**（top 30）+ 完整 SASS
       前 200 行，给读者看 sm_100a 假接受 cubin 里到底是什么指令（不只是 0 UMMA 的
       结论）。

    C. **cuobjdump --list-elf vs --dump-sass -arch X 对齐验证**：对 cuBLAS .so
       跑 list-elf 看实际 cubin slot 列表，再对每个 -arch 跑 dump-sass，对比 SASS
       字节数 + mnemonic 总数，验证 -arch 过滤是否完整覆盖。

    交付：3 个实验的 raw 数据 + 方法学结论。
    """
    import re
    import shutil
    import subprocess
    import tempfile
    from collections import Counter
    from pathlib import Path

    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    ptxas = shutil.which("ptxas") or "/usr/local/cuda/bin/ptxas"
    cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"

    # 核心：精确 mnemonic 提取正则（替代 sass.count(key) 的不精确做法）
    # SASS 行格式：    /*0xoffset*/   MNEMONIC[.MOD][.MOD] args ;
    # 其中 MNEMONIC 必须以大写字母开头，全大写 + 数字 + 下划线
    MNEMONIC_PAT = re.compile(r"^\s*/\*[0-9a-fA-F]+\*/\s+([A-Z][A-Z0-9_]*)", re.M)

    def extract_mnemonics(sass: str) -> Counter:
        """从 SASS dump 文本里精确提取每条指令的 mnemonic（不含 modifier）。"""
        return Counter(MNEMONIC_PAT.findall(sass))

    def count_substring(sass: str, keys: tuple) -> dict:
        """旧的子串 count 法（与之前 disasm_cublas / disasm_fa4 / ptxas_tcgen05_check 一致）。"""
        return {k: sass.count(k) for k in keys}

    SASS_KEYS = ("UMMA", "UTCMMA", "UTCMOV", "UTMA", "HMMA", "QMMA", "DGMMA", "IGMMA")

    workdir = Path(tempfile.mkdtemp(prefix="disasm_sanity_"))
    print(f"[disasm_sanity] workdir = {workdir}")
    print(f"[disasm_sanity] nvcc: {nvcc}")
    print(f"[disasm_sanity] ptxas: {ptxas}")
    print(f"[disasm_sanity] cuobjdump: {cuobjdump}")

    # ptxas 版本
    pv = subprocess.run([ptxas, "--version"], capture_output=True, text=True)
    print(f"[disasm_sanity] ptxas version:\n{pv.stdout or pv.stderr}")

    results: dict = {"experiment_A": {}, "experiment_B": {}, "experiment_C": {}}

    # =========================================================================
    # 实验 A：已知真值 kernel —— 最小 BF16 mma.sync PTX kernel
    # =========================================================================
    # 目的：写一个明确含 mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 的 kernel，
    # nvcc 编 sm_90 / sm_100a，反汇编预期 emit HMMA（或 sm_100 上的 HMMA 变体）。
    # 如果 cuobjdump 能正确 dump 出 HMMA，说明反汇编工具链可信。
    print("\n" + "=" * 72)
    print("EXPERIMENT A: 已知真值 BF16 mma.sync kernel → 应 emit HMMA")
    print("=" * 72)

    # CUDA C++ 源码（直接用 inline PTX，确保 emit mma.sync）
    cu_src = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

extern "C" __global__ void bf16_mma_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C
) {
    // 最小 m16n8k16 BF16 mma：每个 warp 一条 mma.sync
    int lane = threadIdx.x & 31;

    // A: 4 个 bf16 reg per thread, B: 2 个 bf16 reg per thread, D: 4 个 fp32 reg
    unsigned a0 = ((const unsigned*)A)[lane];
    unsigned a1 = ((const unsigned*)A)[lane + 32];
    unsigned a2 = ((const unsigned*)A)[lane + 64];
    unsigned a3 = ((const unsigned*)A)[lane + 96];
    unsigned b0 = ((const unsigned*)B)[lane];
    unsigned b1 = ((const unsigned*)B)[lane + 32];

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    // 这条 PTX 必然在 SASS 里 emit HMMA（sm_80+）
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );

    C[lane] = d0;
    C[lane + 32] = d1;
    C[lane + 64] = d2;
    C[lane + 96] = d3;
}
"""
    cu_path = workdir / "bf16_mma.cu"
    cu_path.write_text(cu_src)

    # 编译 4 个 arch（sm_90 是已知会 emit HMMA 的基准；sm_100/100a 看是否一致）
    arch_targets_A = [
        ("sm_90", "compute_90,sm_90"),
        ("sm_100", "compute_100,sm_100"),
        ("sm_100a", "compute_100a,sm_100a"),
        ("sm_120", "compute_120,sm_120"),
    ]
    exp_A = {}
    for arch_name, gencode in arch_targets_A:
        cubin_path = workdir / f"bf16_mma_{arch_name}.cubin"
        compile_proc = subprocess.run(
            [nvcc, "-arch", arch_name, "-cubin", "-o", str(cubin_path), str(cu_path)],
            capture_output=True, text=True, timeout=120,
        )
        case = {
            "arch": arch_name,
            "compile_ok": compile_proc.returncode == 0 and cubin_path.exists(),
            "compile_stderr_tail": (compile_proc.stderr or "")[-400:],
        }
        if case["compile_ok"]:
            sass = subprocess.run(
                [cuobjdump, "--dump-sass", str(cubin_path)],
                capture_output=True, text=True, timeout=60,
            ).stdout
            mnemonic_counter = extract_mnemonics(sass)
            substring_counts = count_substring(sass, SASS_KEYS)
            case["sass_size_bytes"] = len(sass)
            case["sass_total_instructions"] = sum(mnemonic_counter.values())
            case["mnemonic_top20"] = dict(mnemonic_counter.most_common(20))
            case["substring_count_legacy"] = substring_counts
            case["mnemonic_for_keys"] = {k: mnemonic_counter.get(k, 0) for k in SASS_KEYS}
            # 关键诊断：legacy count 法 vs 精确 mnemonic 法的差异
            case["substring_minus_mnemonic"] = {
                k: substring_counts[k] - mnemonic_counter.get(k, 0) for k in SASS_KEYS
            }
            print(f"  [A] {arch_name:8s}: compile=OK, SASS={len(sass)/1024:.1f} KB, "
                  f"total_instr={sum(mnemonic_counter.values())}, "
                  f"HMMA={mnemonic_counter.get('HMMA', 0)} (legacy count={substring_counts['HMMA']}), "
                  f"UMMA={mnemonic_counter.get('UMMA', 0)} (legacy count={substring_counts['UMMA']})")
        else:
            print(f"  [A] {arch_name:8s}: compile FAILED, stderr_tail={case['compile_stderr_tail'][-200:]}")
        exp_A[arch_name] = case

    results["experiment_A"] = exp_A

    # =========================================================================
    # 实验 B：sm_100a 假接受 5 case 完整 SASS dump + mnemonic 直方图
    # =========================================================================
    # 目的：之前 ptxas_tcgen05_check 报告 sm_100a 接受 5 case 但 0 UMMA + 0 UTCMMA。
    # 这次给读者看假接受 cubin 里到底是什么指令（top 30 mnemonic + 前 100 行 SASS）。
    print("\n" + "=" * 72)
    print("EXPERIMENT B: sm_100a 假接受 5 case 完整 SASS dump")
    print("=" * 72)

    INSTRUCTIONS_B = {
        "tcgen05_alloc": "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r0], 32;",
        "tcgen05_relinquish": "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;",
        "tcgen05_dealloc": "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r0, 32;",
        "tcgen05_mma_f16": (
            "tcgen05.mma.cta_group::1.kind::f16 "
            "[%r0], %rd1, %rd2, %r3, p;"
        ),
        "tcgen05_wait_ld": "tcgen05.wait::ld.sync.aligned;",
    }
    PTX_TEMPLATE_B = """//
.version 8.7
.target sm_100a
.address_size 64

.visible .entry test_kernel(
    .param .u64 test_kernel_param_0
)
{{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .reg .pred p;

    ld.param.u64 %rd0, [test_kernel_param_0];
    cvta.to.global.u64 %rd1, %rd0;
    mov.b64 %rd2, %rd1;
    mov.b32 %r0, 0;
    mov.b32 %r1, 0;
    mov.b32 %r3, 0;
    setp.eq.s32 p, %r0, 0;

    {body}

    ret;
}}
"""

    exp_B = {}
    for instr_name, instr_body in INSTRUCTIONS_B.items():
        ptx_text = PTX_TEMPLATE_B.format(body=instr_body)
        ptx_path = workdir / f"sm100a_{instr_name}.ptx"
        cubin_path = workdir / f"sm100a_{instr_name}.cubin"
        ptx_path.write_text(ptx_text)

        ptxas_proc = subprocess.run(
            [ptxas, "-arch", "sm_100a", "-o", str(cubin_path), str(ptx_path)],
            capture_output=True, text=True, timeout=60,
        )
        case = {
            "instr": instr_name,
            "ptxas_ok": ptxas_proc.returncode == 0 and cubin_path.exists(),
            "ptxas_stderr_tail": (ptxas_proc.stderr or "")[-300:],
        }
        if case["ptxas_ok"]:
            sass = subprocess.run(
                [cuobjdump, "--dump-sass", str(cubin_path)],
                capture_output=True, text=True, timeout=60,
            ).stdout
            mnemonic_counter = extract_mnemonics(sass)
            substring_counts = count_substring(sass, SASS_KEYS)
            case["sass_size_bytes"] = len(sass)
            case["sass_total_instructions"] = sum(mnemonic_counter.values())
            case["mnemonic_full_histogram"] = dict(mnemonic_counter.most_common())
            case["substring_count_legacy"] = substring_counts
            case["sass_first_2000_chars"] = sass[:2000]  # 给读者看实际 SASS 长什么样
            print(f"  [B] {instr_name:24s}: ptxas=OK, SASS={len(sass)} bytes, "
                  f"total_instr={sum(mnemonic_counter.values())}, "
                  f"top5_mnemonic={dict(mnemonic_counter.most_common(5))}")
        else:
            print(f"  [B] {instr_name:24s}: ptxas REJECTED, stderr_tail={case['ptxas_stderr_tail'][-150:]}")
        exp_B[instr_name] = case

    results["experiment_B"] = exp_B

    # =========================================================================
    # 实验 C：cuobjdump --list-elf vs --dump-sass -arch X 对齐验证
    # =========================================================================
    print("\n" + "=" * 72)
    print("EXPERIMENT C: cuobjdump --list-elf vs --dump-sass -arch 对齐验证")
    print("=" * 72)

    # 找 PyTorch 装的 cuBLAS .so（之前 disasm_cublas 用的同一个文件）
    candidate_libs = []
    site_pkgs = Path("/usr/local/lib/python3.12/site-packages")
    for sub in ("nvidia/cublas/lib", "nvidia/cudnn/lib"):
        d = site_pkgs / sub
        if d.exists():
            for so in d.glob("*.so*"):
                if so.is_file() and not so.is_symlink():
                    candidate_libs.append(so)

    print(f"  [C] found {len(candidate_libs)} cuBLAS/cuDNN libraries")
    exp_C = {}
    for lib in candidate_libs[:3]:  # 只测前 3 个最大的
        # list-elf 看实际 cubin slot
        list_elf = subprocess.run(
            [cuobjdump, "--list-elf", str(lib)],
            capture_output=True, text=True, timeout=120,
        )
        list_elf_text = list_elf.stdout or ""
        # 提取所有 sm_XX
        archs_in_elf = sorted(set(re.findall(r"sm_(\d+a?)", list_elf_text)))

        # 对每个 arch 跑 dump-sass，记录字节数 + mnemonic 总数
        per_arch_dump = {}
        for arch_suffix in archs_in_elf[:6]:  # 只测前 6 个 arch（避免超时）
            arch = f"sm_{arch_suffix}"
            sass_proc = subprocess.run(
                [cuobjdump, "--dump-sass", "-arch", arch, str(lib)],
                capture_output=True, text=True, timeout=600,
            )
            sass = sass_proc.stdout or ""
            mnemonic_counter = extract_mnemonics(sass)
            per_arch_dump[arch] = {
                "sass_size_bytes": len(sass),
                "sass_total_instructions": sum(mnemonic_counter.values()),
                "mnemonic_top10": dict(mnemonic_counter.most_common(10)),
                "umma_exact": mnemonic_counter.get("UMMA", 0),
                "utcmma_exact": mnemonic_counter.get("UTCMMA", 0),
                "hmma_exact": mnemonic_counter.get("HMMA", 0),
                "umma_substring": sass.count("UMMA"),
                "utcmma_substring": sass.count("UTCMMA"),
                "hmma_substring": sass.count("HMMA"),
            }
            print(f"  [C] {lib.name} -arch {arch}: SASS={len(sass)/1024/1024:.2f} MB, "
                  f"total_instr={sum(mnemonic_counter.values())}, "
                  f"HMMA exact={mnemonic_counter.get('HMMA', 0)} / substring={sass.count('HMMA')}, "
                  f"UMMA exact={mnemonic_counter.get('UMMA', 0)} / substring={sass.count('UMMA')}")

        exp_C[lib.name] = {
            "lib_path": str(lib),
            "lib_size_mb": lib.stat().st_size / 1e6,
            "list_elf_archs": archs_in_elf,
            "list_elf_first_2000_chars": list_elf_text[:2000],
            "per_arch_dump": per_arch_dump,
        }

    results["experiment_C"] = exp_C

    # =========================================================================
    # 综合方法学结论
    # =========================================================================
    print("\n" + "=" * 72)
    print("METHODOLOGY VERDICT")
    print("=" * 72)

    # 实验 A: 是否 emit HMMA
    A_summary = []
    for arch, case in exp_A.items():
        if case.get("compile_ok"):
            hmma = case.get("mnemonic_for_keys", {}).get("HMMA", 0)
            A_summary.append(f"{arch}: HMMA={hmma}")
    print(f"  [A] BF16 mma.sync 已知真值: {' | '.join(A_summary)}")
    print(f"      预期: sm_90/100/100a/120 都应 HMMA > 0 (mma.sync 是 sm_80+ 标配)")

    # 实验 B: sm_100a 假接受 cubin 真实指令
    B_with_sass = {n: c for n, c in exp_B.items() if c.get("ptxas_ok")}
    print(f"  [B] sm_100a ptxas 接受的 {len(B_with_sass)} 个 case，全部反汇编无 UMMA/UTCMMA：")
    for n, c in B_with_sass.items():
        top = c.get("mnemonic_full_histogram", {})
        umma = top.get("UMMA", 0); utcmma = top.get("UTCMMA", 0)
        print(f"      {n}: UMMA={umma}, UTCMMA={utcmma}, top mnemonic={dict(list(top.items())[:5])}")

    # 实验 C: substring vs exact 是否一致
    print(f"  [C] cuBLAS .so substring count vs exact mnemonic count 一致性：")
    pollution_found = False
    for lib_name, lib_data in exp_C.items():
        for arch, d in lib_data.get("per_arch_dump", {}).items():
            for key in ("HMMA", "UMMA", "UTCMMA"):
                exact = d.get(f"{key.lower()}_exact", 0)
                substr = d.get(f"{key.lower()}_substring", 0)
                if exact != substr:
                    print(f"      ⚠️ {lib_name} {arch} {key}: exact={exact}, substr={substr}, diff={substr - exact}")
                    pollution_found = True
    if not pollution_found:
        print(f"      ✅ substring count == exact mnemonic count for all (HMMA/UMMA/UTCMMA)")

    print("=" * 72)

    return results

# ---------------------------------------------------------------------------
# 11) bench_nvfp4_gemm —— B200 NVFP4 数据通路端到端验证（response to "FA4 跑通
#     不能证明 NVFP4 可用"的开口）
# ---------------------------------------------------------------------------
@app.function(image=image_with_cuda, gpu="B200:1", timeout=1800)
def bench_nvfp4_gemm() -> dict:
    """B200 NVFP4 数据通路验证：4 条 fallback 路径同时跑，至少 1 条能给定性结论。

    回答的核心问题：**B200 (sm_100) 的 NVFP4 数据通路到底是不是真的可用？**

    背景：之前 §7.11 FA4 forward 跑通用的是 bf16，不是 fp4。所以 "B200 tcgen05 硬件可用"
    已证，但 "B200 NVFP4 数据通路可用" 仍是开口。本 entry 用 4 条 fallback 路径补齐。

    路径设计：
    - P1: PyTorch _scaled_mm fp8（fp8 是 fp4 的前置依赖，必先通）
    - P2: PyTorch fp4 dtype (torch.float4_e2m1fn_x2 if available)
    - P3: ptxas 喂最小 NVFP4 mma PTX (mma.m16n8k64.f32.e2m1.e2m1.f32) 编 sm_100a / sm_120a
    - P4: 反汇编 cuBLASLt/cuDNN sm_100 cubin 搜 NVFP4 相关 SASS（QMMA / mxf8f6f4 等）

    输出：4 条路径独立结论 + 综合 verdict。
    """
    import re
    import shutil
    import subprocess
    import tempfile
    import time
    from collections import Counter
    from pathlib import Path

    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    ptxas = shutil.which("ptxas") or "/usr/local/cuda/bin/ptxas"
    cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"

    MNEMONIC_PAT = re.compile(r"^\s*/\*[0-9a-fA-F]+\*/\s+([A-Z][A-Z0-9_]*)", re.M)

    workdir = Path(tempfile.mkdtemp(prefix="bench_nvfp4_"))
    print(f"[bench_nvfp4] workdir = {workdir}")

    results: dict = {"P1": {}, "P2": {}, "P3": {}, "P4": {}}

    # =========================================================================
    # 准备：torch + GPU 信息
    # =========================================================================
    import torch
    print(f"[bench_nvfp4] torch={torch.__version__}, cuda={torch.version.cuda}, "
          f"cudnn={torch.backends.cudnn.version()}")
    if not torch.cuda.is_available():
        return {"error": "torch.cuda.is_available() = False"}
    device = torch.device("cuda:0")
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"[bench_nvfp4] GPU: {name}, capability: {cap}")

    # 探测 torch 上有什么 fp4/fp8 dtype
    torch_dtypes = {}
    for attr in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz",
                 "float4_e2m1fn_x2", "float4_e2m1", "float6_e2m3fn", "float6_e3m2fn"):
        torch_dtypes[attr] = hasattr(torch, attr)
    print(f"[bench_nvfp4] torch low-precision dtype availability: {torch_dtypes}")

    # =========================================================================
    # P1: PyTorch _scaled_mm fp8（前置依赖 — 如果 fp8 跑不通，fp4 更不可能跑通）
    # =========================================================================
    print("\n" + "=" * 72)
    print("PATH P1: PyTorch _scaled_mm fp8 GEMM (sanity check, fp4 前置依赖)")
    print("=" * 72)
    p1: dict = {}
    try:
        if not (torch_dtypes.get("float8_e4m3fn") and hasattr(torch, "_scaled_mm")):
            p1["status"] = "skipped"
            p1["reason"] = "torch.float8_e4m3fn or torch._scaled_mm missing"
        else:
            # 测试 fp8 GEMM (4096x4096x4096)
            M, N, K = 4096, 4096, 4096
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
            b = torch.randn(N, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn).t()
            scale_a = torch.tensor(1.0, dtype=torch.float32, device=device)
            scale_b = torch.tensor(1.0, dtype=torch.float32, device=device)

            # warmup
            for _ in range(3):
                out = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
            torch.cuda.synchronize()

            iters = 20
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                out = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
            end.record()
            torch.cuda.synchronize()
            ms_per_iter = start.elapsed_time(end) / iters
            tflops = (2.0 * M * N * K) / (ms_per_iter * 1e-3) / 1e12

            p1["status"] = "OK"
            p1["shape"] = f"{M}x{N}x{K}"
            p1["dtype"] = "float8_e4m3fn"
            p1["ms_per_iter"] = ms_per_iter
            p1["tflops"] = tflops
            p1["out_dtype"] = str(out.dtype)
            p1["out_shape"] = list(out.shape)
            # B200 fp8 标称 ~4845 TFLOPS（dense, 2x bf16）
            p1["nominal_fp8_tflops"] = 4845
            p1["utilization_pct"] = tflops / 4845 * 100
            print(f"  [P1] fp8 GEMM 4Kx4Kx4K: {ms_per_iter:.3f} ms/iter, {tflops:.0f} TFLOPS "
                  f"({p1['utilization_pct']:.1f}% of B200 fp8 nominal 4845 TFLOPS)")
    except Exception as exc:
        p1["status"] = "FAILED"
        p1["error"] = repr(exc)
        print(f"  [P1] FAILED: {exc!r}")
    results["P1"] = p1

    # =========================================================================
    # P2: PyTorch fp4 dtype (torch.float4_e2m1fn_x2 if available)
    # =========================================================================
    print("\n" + "=" * 72)
    print("PATH P2: PyTorch fp4 dtype (torch.float4_e2m1fn_x2 if available)")
    print("=" * 72)
    p2: dict = {}
    try:
        # 探测 torch 是否有 fp4 dtype
        fp4_dtype = None
        for attr in ("float4_e2m1fn_x2", "float4_e2m1"):
            if hasattr(torch, attr):
                fp4_dtype = getattr(torch, attr)
                p2["fp4_dtype_name"] = attr
                break
        if fp4_dtype is None:
            p2["status"] = "skipped"
            p2["reason"] = f"no fp4 dtype found in torch {torch.__version__}; tried float4_e2m1fn_x2 / float4_e2m1"
            print(f"  [P2] {p2['reason']}")
        else:
            # NVFP4 GEMM API（来自 PyTorch 2.11 错误信息暴露的精确形状）：
            # - For Blockwise 1x16 scaling: a/b should be float4 (packed 2x),
            #   scales should be float8_e4m3fn,
            #   scale_a should have M*K/16 elements, scale_b should have N*K/16 elements,
            #   both contiguous.
            # 4096x4096x4096 → scale 元素数 = 4096*4096/16 = 1048576
            print(f"  [P2] found {p2['fp4_dtype_name']}, attempting NVFP4 GEMM with block 1x16 scaling ...")
            M, N, K = 4096, 4096, 4096
            try:
                # fp4 packed 2x：每 byte 2 个 fp4 → K // 2 列 uint8
                a_uint8 = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=device)
                b_uint8 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
                a = a_uint8.view(fp4_dtype)
                b = b_uint8.view(fp4_dtype).t()
                # NVFP4 scaling：每 16 个 fp4 共享一个 e4m3 scale
                scale_elems_a = M * K // 16  # 1048576
                scale_elems_b = N * K // 16  # 1048576
                # e4m3 scale 用 e4m3fn dtype，构造为 [scale_elems] 1D tensor 即可
                scale_a_uint8 = torch.randint(64, 192, (scale_elems_a,), dtype=torch.uint8, device=device)
                scale_b_uint8 = torch.randint(64, 192, (scale_elems_b,), dtype=torch.uint8, device=device)
                scale_a = scale_a_uint8.view(torch.float8_e4m3fn).contiguous()
                scale_b = scale_b_uint8.view(torch.float8_e4m3fn).contiguous()
                p2["scale_elems_a"] = scale_elems_a
                p2["scale_elems_b"] = scale_elems_b
                # warmup
                for _ in range(3):
                    out = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
                torch.cuda.synchronize()
                # bench
                iters = 50
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iters):
                    out = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
                end.record()
                torch.cuda.synchronize()
                ms_per_iter = start.elapsed_time(end) / iters
                tflops = (2.0 * M * N * K) / (ms_per_iter * 1e-3) / 1e12
                p2["status"] = "OK"
                p2["shape"] = f"{M}x{N}x{K}"
                p2["dtype"] = "float4_e2m1fn_x2"
                p2["scaling"] = "block 1x16, e4m3 scales"
                p2["ms_per_iter"] = ms_per_iter
                p2["tflops"] = tflops
                p2["out_dtype"] = str(out.dtype)
                p2["out_shape"] = list(out.shape)
                # B200 NVFP4 标称 ~9690 TFLOPS（dense, 2x fp8）
                p2["nominal_nvfp4_tflops"] = 9690
                p2["utilization_pct"] = tflops / 9690 * 100
                print(f"  [P2] ★★★ NVFP4 GEMM 4Kx4Kx4K block 1x16 scaling: "
                      f"{ms_per_iter:.3f} ms/iter, {tflops:.0f} TFLOPS "
                      f"({p2['utilization_pct']:.1f}% of B200 NVFP4 nominal 9690 TFLOPS)")
                print(f"  [P2] out {out.shape} {out.dtype}")

                # 多 shape 扫描
                p2["shapes"] = []
                for M_, N_, K_ in [(2048, 2048, 2048), (8192, 8192, 8192), (16384, 16384, 16384)]:
                    try:
                        a2_u = torch.randint(0, 256, (M_, K_ // 2), dtype=torch.uint8, device=device)
                        b2_u = torch.randint(0, 256, (N_, K_ // 2), dtype=torch.uint8, device=device)
                        a2 = a2_u.view(fp4_dtype)
                        b2 = b2_u.view(fp4_dtype).t()
                        sa_u = torch.randint(64, 192, (M_ * K_ // 16,), dtype=torch.uint8, device=device)
                        sb_u = torch.randint(64, 192, (N_ * K_ // 16,), dtype=torch.uint8, device=device)
                        sa = sa_u.view(torch.float8_e4m3fn).contiguous()
                        sb = sb_u.view(torch.float8_e4m3fn).contiguous()
                        for _ in range(2):
                            o2 = torch._scaled_mm(a2, b2, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
                        torch.cuda.synchronize()
                        iters2 = 30
                        s2 = torch.cuda.Event(enable_timing=True); e2 = torch.cuda.Event(enable_timing=True)
                        s2.record()
                        for _ in range(iters2):
                            o2 = torch._scaled_mm(a2, b2, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
                        e2.record()
                        torch.cuda.synchronize()
                        ms2 = s2.elapsed_time(e2) / iters2
                        tflops2 = (2.0 * M_ * N_ * K_) / (ms2 * 1e-3) / 1e12
                        util2 = tflops2 / 9690 * 100
                        p2["shapes"].append({
                            "shape": f"{M_}x{N_}x{K_}",
                            "ms_per_iter": ms2,
                            "tflops": tflops2,
                            "utilization_pct": util2,
                        })
                        print(f"  [P2] NVFP4 {M_}x{N_}x{K_}: {ms2:.3f} ms, {tflops2:.0f} TFLOPS ({util2:.1f}% of 9690)")
                    except Exception as exc_shape:
                        p2["shapes"].append({"shape": f"{M_}x{N_}x{K_}", "error": repr(exc_shape)[:200]})
                        print(f"  [P2] NVFP4 {M_}x{N_}x{K_}: FAILED {exc_shape!r}")
            except Exception as exc:
                p2["status"] = "FAILED"
                p2["error"] = repr(exc)
                print(f"  [P2] NVFP4 GEMM call FAILED: {exc!r}")
    except Exception as exc:
        p2["status"] = "FAILED"
        p2["error"] = repr(exc)
        print(f"  [P2] FAILED: {exc!r}")
    results["P2"] = p2

    # =========================================================================
    # P3: ptxas 喂最小 NVFP4 mma PTX，看 sm_100a / sm_120a 是否接受
    # =========================================================================
    # 关键 PTX: mma.m16n8k64.row.col.f32.e2m1.e2m1.f32 (NVFP4 mma, PTX ISA 8.7+)
    # 这是非 tcgen05 的"普通"NVFP4 mma 指令，应该不需要 TMEM
    print("\n" + "=" * 72)
    print("PATH P3: ptxas 最小 NVFP4 mma PTX 接受性测试")
    print("=" * 72)
    NVFP4_PTX_TEMPLATE = """//
.version {ver}
.target {arch}
.address_size 64

.visible .entry test_kernel(
    .param .u64 test_kernel_param_0
)
{{
    .reg .b32 %r<32>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<32>;

    ld.param.u64 %rd0, [test_kernel_param_0];
    cvta.to.global.u64 %rd1, %rd0;

    mov.b32 %r0, 0;
    mov.b32 %r1, 0;
    mov.b32 %r2, 0;
    mov.b32 %r3, 0;
    mov.b32 %r4, 0;
    mov.b32 %r5, 0;
    mov.b32 %r6, 0;
    mov.b32 %r7, 0;
    mov.f32 %f0, 0f00000000;
    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;

    {body}

    st.global.f32 [%rd1+0], %f0;
    st.global.f32 [%rd1+4], %f1;
    st.global.f32 [%rd1+8], %f2;
    st.global.f32 [%rd1+12], %f3;

    ret;
}}
"""
    NVFP4_INSTRUCTIONS = {
        # 标准 NVFP4 mma（非 tcgen05），PTX ISA 8.7+ 引入
        "mma_m16n8k64_e2m1": (
            "mma.sync.aligned.m16n8k64.row.col.f32.e2m1.e2m1.f32 "
            "{%f0, %f1, %f2, %f3}, "
            "{%r0, %r1, %r2, %r3}, "
            "{%r4, %r5}, "
            "{%f0, %f1, %f2, %f3};"
        ),
        # MX-FP8 mma (sm_100a) — 用 e4m3 替代 e2m1
        "mma_m16n8k32_e4m3": (
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%f0, %f1, %f2, %f3}, "
            "{%r0, %r1, %r2, %r3}, "
            "{%r4, %r5}, "
            "{%f0, %f1, %f2, %f3};"
        ),
        # 对照组：常规 BF16 mma（已知 sm_80+ 都支持）
        "mma_m16n8k16_bf16": (
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%f0, %f1, %f2, %f3}, "
            "{%r0, %r1, %r2, %r3}, "
            "{%r4, %r5}, "
            "{%f0, %f1, %f2, %f3};"
        ),
    }
    p3_targets = [("sm_100", "8.7"), ("sm_100a", "8.7"), ("sm_120", "8.7"), ("sm_120a", "8.7")]
    p3: dict = {"matrix": [], "summary": {}}
    for arch, ver in p3_targets:
        for instr_name, instr_body in NVFP4_INSTRUCTIONS.items():
            ptx_text = NVFP4_PTX_TEMPLATE.format(ver=ver, arch=arch, body=instr_body)
            ptx_path = workdir / f"nvfp4_{arch}_{instr_name}.ptx"
            cubin_path = workdir / f"nvfp4_{arch}_{instr_name}.cubin"
            ptx_path.write_text(ptx_text)
            proc = subprocess.run(
                [ptxas, "-arch", arch, "-o", str(cubin_path), str(ptx_path)],
                capture_output=True, text=True, timeout=60,
            )
            ok = proc.returncode == 0 and cubin_path.exists()
            entry = {
                "arch": arch,
                "ptx_version": ver,
                "instr": instr_name,
                "accepted": ok,
                "stderr_tail": (proc.stderr or "")[-300:],
            }
            if ok:
                sass = subprocess.run(
                    [cuobjdump, "--dump-sass", str(cubin_path)],
                    capture_output=True, text=True, timeout=60,
                ).stdout
                mnemonic_counter = Counter(MNEMONIC_PAT.findall(sass))
                entry["sass_size"] = len(sass)
                entry["sass_total_instructions"] = sum(mnemonic_counter.values())
                entry["mnemonic_top10"] = dict(mnemonic_counter.most_common(10))
                # 关键：看是否有 mma 类 SASS
                entry["mma_class_sass"] = {
                    k: v for k, v in mnemonic_counter.items()
                    if "MMA" in k or "TCMOV" in k
                }
            p3["matrix"].append(entry)
            mma_info = ""
            if ok:
                mma_info = f" mma_sass={entry['mma_class_sass']}"
            status = "✅" if ok else "❌"
            print(f"  [P3] {status} {arch:8s} | {instr_name:24s} | rc={proc.returncode}{mma_info}")

    # P3 汇总：每个 arch 接受了多少 NVFP4-related 指令
    for arch, _ in p3_targets:
        accepted = [m for m in p3["matrix"] if m["arch"] == arch and m["accepted"]]
        rejected = [m for m in p3["matrix"] if m["arch"] == arch and not m["accepted"]]
        # 哪些 case 真的 emit 出 mma 类 SASS（非 0）
        emit_mma = [m for m in accepted if any(v > 0 for v in m.get("mma_class_sass", {}).values())]
        p3["summary"][arch] = {
            "accepted": [m["instr"] for m in accepted],
            "rejected": [m["instr"] for m in rejected],
            "emit_mma_sass": [m["instr"] for m in emit_mma],
        }
    print("  [P3] 汇总：")
    for arch, s in p3["summary"].items():
        print(f"    {arch:8s}: accepted={s['accepted']}, "
              f"emit_mma_sass={s['emit_mma_sass']}, rejected={s['rejected']}")
    results["P3"] = p3

    # =========================================================================
    # P4: 反汇编 cuBLASLt / cuDNN sm_100 cubin 搜 NVFP4 相关 SASS
    # =========================================================================
    # NVFP4 在 SASS 层面可能表现为：
    #   - QMMA.16864.F32 (quarter-precision mma, NVFP4 m16n8k64)
    #   - HMMA.kind::mxf8f6f4 (如果 ptxas 走 mma.kind path)
    #   - UTCMMA.kind::mxf8f6f4 (tcgen05 path)
    # 找 cuBLASLt 13.x sm_100 cubin 看是否有这些 SASS
    print("\n" + "=" * 72)
    print("PATH P4: 反汇编 cuBLASLt/cuDNN sm_100 cubin 搜 NVFP4 SASS")
    print("=" * 72)

    candidate_libs = []
    site_pkgs = Path("/usr/local/lib/python3.12/site-packages")
    for sub in ("nvidia/cublas/lib", "nvidia/cudnn/lib"):
        d = site_pkgs / sub
        if d.exists():
            for so in d.glob("*.so*"):
                if so.is_file() and not so.is_symlink():
                    candidate_libs.append(so)
    # 按文件大小降序，优先扫大文件
    candidate_libs.sort(key=lambda p: p.stat().st_size, reverse=True)

    p4: dict = {"libs_scanned": [], "nvfp4_sass_found": []}
    NVFP4_SASS_PATTERNS = [
        "QMMA",                # NVFP4 mma SASS（推测命名）
        "mxf8f6f4",            # mma.kind 标识
        "UTCMMA",              # tcgen05 mma
        "UMMA",                # tcgen05 utility mma
        "e2m1",                # NVFP4 mantissa marker
        "F4",                  # 通用 FP4 marker
    ]
    # 对前 2 个最大的 lib 跑 -arch sm_100 dump
    for lib in candidate_libs[:2]:
        print(f"  [P4] scanning {lib.name} ({lib.stat().st_size / 1e6:.1f} MB) ...")
        t0 = time.time()
        sass_proc = subprocess.run(
            [cuobjdump, "--dump-sass", "-arch", "sm_100", str(lib)],
            capture_output=True, text=True, timeout=900,
        )
        sass = sass_proc.stdout or ""
        elapsed = time.time() - t0
        mnemonic_counter = Counter(MNEMONIC_PAT.findall(sass))
        # 搜 NVFP4 相关字符串
        pattern_counts = {}
        for pat in NVFP4_SASS_PATTERNS:
            pattern_counts[pat] = sass.count(pat)
        # 找 mma 类 mnemonic（任何含 MMA 的 mnemonic）
        mma_mnemonics = {k: v for k, v in mnemonic_counter.items() if "MMA" in k or "MFP4" in k or "TCMOV" in k}
        lib_result = {
            "lib_name": lib.name,
            "lib_size_mb": lib.stat().st_size / 1e6,
            "dump_elapsed_sec": elapsed,
            "sass_size_mb": len(sass) / 1e6,
            "sass_total_instructions": sum(mnemonic_counter.values()),
            "mma_mnemonics": mma_mnemonics,
            "pattern_counts": pattern_counts,
            "top10_mnemonic": dict(mnemonic_counter.most_common(10)),
        }
        p4["libs_scanned"].append(lib_result)
        if any(v > 0 for v in pattern_counts.values()) or mma_mnemonics:
            p4["nvfp4_sass_found"].append(lib.name)
        print(f"  [P4] {lib.name} -arch sm_100: SASS={len(sass)/1e6:.1f}MB, "
              f"total_instr={sum(mnemonic_counter.values())}, "
              f"mma_mnemonics={mma_mnemonics}, "
              f"pattern_counts={pattern_counts}")
    results["P4"] = p4

    # =========================================================================
    # 综合 verdict
    # =========================================================================
    print("\n" + "=" * 72)
    print("COMPREHENSIVE VERDICT: B200 NVFP4 数据通路可用性")
    print("=" * 72)
    print(f"  [P1] PyTorch fp8 _scaled_mm: status={p1.get('status')}, "
          f"tflops={p1.get('tflops', 'N/A')}")
    print(f"  [P2] PyTorch fp4 dtype: status={p2.get('status')}, "
          f"reason={p2.get('reason', p2.get('error', 'OK'))[:100]}")
    print(f"  [P3] ptxas NVFP4 PTX:")
    for arch, s in p3["summary"].items():
        print(f"        {arch}: accepted={s['accepted']}, emit_mma={s['emit_mma_sass']}")
    print(f"  [P4] cuBLAS/cuDNN NVFP4 SASS found: {p4['nvfp4_sass_found']}")
    print("=" * 72)

    return results

# ---------------------------------------------------------------------------
# local entry：默认跑 gpu_info（最轻量）
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(action: str = "gpu_info"):
    """本地默认入口：modal run dsa/b200_runner.py 即等价于 ::gpu_info。"""
    if action == "gpu_info":
        gpu_info.remote()
    elif action == "smoke":
        smoke.remote()
    elif action == "bench":
        bench_gemm.remote()
    elif action == "bench_bf16":
        bench_bf16.remote()
    elif action == "bench_bandwidth":
        bench_bandwidth.remote()
    elif action == "cuda_env_check":
        cuda_env_check.remote()
    elif action == "dump_cute_dsl":
        dump_cute_dsl_cubin.remote()
    elif action == "disasm_fa4":
        disasm_fa4.remote()
    elif action == "disasm_sanity_check":
        disasm_sanity_check.remote()
    elif action == "bench_nvfp4_gemm":
        bench_nvfp4_gemm.remote()
    else:
        raise ValueError(
            f"Unknown action: {action} "
            "(expect: gpu_info|smoke|bench|bench_bf16|bench_bandwidth|"
            "cuda_env_check|dump_cute_dsl|disasm_fa4|disasm_sanity_check|"
            "bench_nvfp4_gemm)"
        )

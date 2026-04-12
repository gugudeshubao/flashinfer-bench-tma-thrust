import hashlib
from pathlib import Path

_module = None
_failed = False


def get_module():
    global _module, _failed
    if _module is not None:
        return _module
    if _failed:
        return None

    try:
        from torch.utils.cpp_extension import load

        src = Path(__file__).with_name("kernel.cu")
        digest = hashlib.sha1(src.read_bytes()).hexdigest()[:10]
        include_paths = ["/opt/cutlass/include", "/opt/cutlass/tools/util/include", str(src.parent)]
        _module = load(
            name=f"moe_cuda_ptx_torch_runtime_ext_{digest}",
            sources=[str(src)],
            extra_include_paths=include_paths,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        return _module
    except Exception:
        _failed = True
        return None

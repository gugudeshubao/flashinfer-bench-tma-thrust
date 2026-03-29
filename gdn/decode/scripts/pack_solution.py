"""
Pack solution for GDN decode kernel.
Reads config.toml from parent directory and packs solution/triton/kernel.py.
"""
import sys
from pathlib import Path

KERNEL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(KERNEL_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files


def pack_solution(output_path: Path = None) -> Path:
    config_path = KERNEL_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    sol = config["solution"]
    bld = config["build"]

    source_dir = KERNEL_ROOT / "solution" / "triton"
    dps = bld.get("destination_passing_style", True)

    spec = BuildSpec(
        language=bld["language"],
        target_hardware=["cuda"],
        entry_point=bld["entry_point"],
        destination_passing_style=dps,
    )

    solution = pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=sol["name"],
        definition=sol["definition"],
        author=sol["author"],
    )

    if output_path is None:
        output_path = KERNEL_ROOT / "solution.json"

    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"Packed: {output_path}")
    print(f"  definition: {solution.definition}")
    return output_path


if __name__ == "__main__":
    pack_solution()

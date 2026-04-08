"""
FlashInfer-Bench NCU Profiler Runner (Modal Backend).

Profiles a solution on a specific workload using NVIDIA Nsight Compute (NCU)
on a Modal cloud GPU. Reuses pack_solution() to build Solution objects and
the flashinfer-bench NCU agent API.

Usage:
    modal run scripts/run_modal_profile.py -- --list
    modal run scripts/run_modal_profile.py -- 0
    modal run scripts/run_modal_profile.py -- 0 --set full --page raw
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

# Read config locally; on Modal remote container config.toml doesn't exist
_config_path = PROJECT_ROOT / "config.toml"
if _config_path.exists():
    with open(_config_path, "rb") as _f:
        _config = tomllib.load(_f)
    _GPU_TYPE = _config.get("build", {}).get("gpu", "b200").upper()
else:
    _GPU_TYPE = "B200"

import modal
from flashinfer_bench import Solution, Workload

app = modal.App("flashinfer-bench-ncu-profile")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

_pip_deps = ["torch", "numpy", "cupti-python", "triton", "tilelang", "apache-tvm-ffi"]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(*_pip_deps)
    .pip_install(
        "flashinfer-bench @ https://github.com/flashinfer-ai/flashinfer-bench/archive/refs/heads/main.tar.gz"
    )
    .pip_install("torch-c-dlpack-ext")
)


# ---------------------------------------------------------------------------
# Modal remote function
# ---------------------------------------------------------------------------

@app.function(image=image, gpu=f"{_GPU_TYPE}:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_ncu_remote(solution: Solution, workload: Workload, ncu_kwargs: dict) -> str:
    """Run NCU profiling inside Modal container and return output."""
    from flashinfer_bench.agents import flashinfer_bench_run_ncu

    return flashinfer_bench_run_ncu(
        solution, workload,
        trace_set_path=TRACE_SET_PATH,
        **ncu_kwargs,
    )


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _load_config():
    """Load operator name from config.toml."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        sys.exit("Error: config.toml not found")
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _load_workloads_jsonl():
    """Load workloads from docs/workloads.jsonl (local file, no volume needed)."""
    jsonl_path = PROJECT_ROOT / "docs" / "workloads.jsonl"
    if not jsonl_path.exists():
        sys.exit(f"Error: {jsonl_path} not found")
    workloads = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                workloads.append(json.loads(line))
    return workloads


def _list_workloads():
    """Print workload table with indices."""
    config = _load_config()
    operator = config["solution"]["definition"]
    workloads = _load_workloads_jsonl()
    print(f"Workloads for '{operator}' ({len(workloads)} total):\n")
    print(f"{'Index':<7} {'UUID':<12} {'Axes'}")
    print(f"{'-----':<7} {'----':<12} {'----'}")
    for i, entry in enumerate(workloads):
        wl = entry["workload"]
        uuid_prefix = wl["uuid"][:8]
        axes_str = ", ".join(f"{k}={v}" for k, v in sorted(wl["axes"].items()))
        print(f"{i:<7} {uuid_prefix:<12} {axes_str}")


def _list_ncu_options():
    """List available NCU sets and sections."""
    import shutil as _shutil
    if _shutil.which("ncu"):
        from flashinfer_bench.agents import flashinfer_bench_list_ncu_options
        print(flashinfer_bench_list_ncu_options())
    else:
        print("NCU not available locally. Common section sets:\n")
        print("  basic     — Minimal: launch config, duration, memory throughput. Fast.")
        print("  detailed  — Key metrics: compute/memory utilization, occupancy, warp stalls. (default)")
        print("  full      — All available metrics. Slow but exhaustive.")
        print("\nOutput pages: details (default), source, raw")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    workload_index: int = -1,
    list: bool = False,
    ncu_options: bool = False,
    set: str = "detailed",
    page: str = "details",
    kernel_name: str = "",
    timeout: int = 60,
    max_lines: int = -1,
):
    """NCU profiler — profile a solution on a specific workload via Modal."""
    import os

    if list:
        _list_workloads()
        return

    if ncu_options:
        _list_ncu_options()
        return

    if workload_index < 0:
        print("Usage: modal run scripts/run_modal_profile.py -- <index>")
        print("       modal run scripts/run_modal_profile.py -- --list")
        print("       modal run scripts/run_modal_profile.py -- --ncu-options")
        sys.exit(1)

    # Isolate Triton JIT cache to this project
    os.environ.setdefault("TRITON_CACHE_DIR", str(PROJECT_ROOT / ".triton_cache"))

    # Load workloads
    workloads = _load_workloads_jsonl()
    if workload_index >= len(workloads):
        print(f"Error: Workload index {workload_index} out of range [0, {len(workloads) - 1}]",
              file=sys.stderr)
        sys.exit(1)

    wl_entry = workloads[workload_index]
    workload = Workload.model_validate(wl_entry["workload"])

    # Pack solution
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text())

    axes_str = ", ".join(f"{k}={v}" for k, v in sorted(workload.axes.items()))
    print(f"\nProfiling workload {workload_index}: {workload.uuid[:8]}...")
    print(f"  Axes: {axes_str}")
    print(f"  Set: {set}, Page: {page}")
    print(f"  GPU: {_GPU_TYPE}")
    if kernel_name:
        print(f"  Kernel filter: {kernel_name}")
    print()

    # Build NCU kwargs
    ncu_kwargs = {"set": set, "page": page, "timeout": timeout}
    if kernel_name:
        ncu_kwargs["kernel_name"] = kernel_name
    if max_lines > 0:
        ncu_kwargs["max_lines"] = max_lines

    # Run on Modal
    result = run_ncu_remote.remote(solution, workload, ncu_kwargs)
    print(result)

    # Auto-save profile output
    profiles_dir = PROJECT_ROOT / "profiles"
    profiles_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_data = {
        "timestamp": datetime.now().isoformat(),
        "workload_index": workload_index,
        "workload_uuid": workload.uuid,
        "axes": dict(workload.axes),
        "ncu_set": set,
        "ncu_page": page,
        "kernel_filter": kernel_name or None,
        "backend": f"modal-{_GPU_TYPE.lower()}",
        "output": result,
    }
    profile_path = profiles_dir / f"w{workload_index}_{timestamp}.json"
    profile_path.write_text(json.dumps(profile_data, indent=2))
    print(f"\nProfile saved to: {profile_path}", file=sys.stderr)

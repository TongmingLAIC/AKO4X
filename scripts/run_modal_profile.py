"""
FlashInfer-Bench Profiler Runner (Modal Backend).

Profiles a solution on specific workloads using NVIDIA Nsight Compute (NCU)
or lightweight torch.profiler for kernel launch analysis on Modal cloud GPU.

Usage:
    modal run scripts/run_modal_profile.py --list
    modal run scripts/run_modal_profile.py --index 0
    modal run scripts/run_modal_profile.py --index 0 --set full --page raw
    modal run scripts/run_modal_profile.py --index 0,3,5 --launches
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
# Modal remote functions
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


@app.function(image=image, gpu=f"{_GPU_TYPE}:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_launches_remote(solution: Solution, workload: Workload) -> str:
    """Run torch.profiler inside Modal container and return kernel launch summary."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.evaluators.utils import allocate_outputs
    from flashinfer_bench.bench.utils import gen_inputs
    from flashinfer_bench.compile import BuilderRegistry

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[solution.definition]

    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, solution)

    device = "cuda:0"
    inputs = gen_inputs(definition, workload, device)
    outputs = allocate_outputs(definition, inputs, device)

    # Warmup
    with torch.no_grad():
        runnable.call_destination_passing(*inputs, *outputs)
    torch.cuda.synchronize()

    # Regenerate for profiling run
    inputs = gen_inputs(definition, workload, device)
    outputs = allocate_outputs(definition, inputs, device)

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with torch.no_grad():
            runnable.call_destination_passing(*inputs, *outputs)
        torch.cuda.synchronize()

    events = prof.key_averages()
    cuda_events = [e for e in events if e.self_cuda_time_total > 0]
    total_launches = sum(e.count for e in cuda_events)
    total_cuda_us = sum(e.self_cuda_time_total for e in cuda_events)

    lines = []
    lines.append(f"Total CUDA kernel launches: {total_launches}")
    lines.append(f"Total CUDA time: {total_cuda_us / 1000:.3f} ms")
    lines.append("")
    lines.append(events.table(sort_by="cuda_time_total", row_limit=50))

    runnable.cleanup()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _load_config():
    """Load config from config.toml."""
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


def _filter_workloads(entries, indices=None, group_values=None, exclude_group_values=None):
    """Filter workloads by index and group axis. Returns list of (original_index, entry)."""
    from scripts.bench_utils import find_group_axis
    group_axis = find_group_axis() if (group_values or exclude_group_values) else ""

    indexed = list(enumerate(entries))

    if indices is not None:
        index_set = set(indices)
        indexed = [(i, e) for i, e in indexed if i in index_set]

    if group_values and group_axis:
        group_set = set(group_values)
        indexed = [(i, e) for i, e in indexed if e["workload"]["axes"].get(group_axis) in group_set]

    if exclude_group_values and group_axis:
        exclude_set = set(exclude_group_values)
        indexed = [(i, e) for i, e in indexed if e["workload"]["axes"].get(group_axis) not in exclude_set]

    return indexed


def _list_workloads(entries, indices=None, group_values=None, exclude_group_values=None):
    """Print workload table with indices, optionally filtered."""
    indexed = _filter_workloads(entries, indices, group_values, exclude_group_values)
    total = len(entries)
    shown = len(indexed)
    label = f" (filtered {shown}/{total})" if shown < total else ""
    print(f"Workloads ({total} total{label}):\n")
    print(f"{'Index':<7} {'UUID':<12} {'Axes'}")
    print(f"{'-----':<7} {'----':<12} {'----'}")
    for i, entry in indexed:
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
    index: str = None,
    list: bool = False,
    launches: bool = False,
    ncu_options: bool = False,
    set: str = "detailed",
    page: str = "details",
    kernel_name: str = "",
    sections: str = "",
    timeout: int = 60,
    max_lines: int = -1,
    group: str = None,
    exclude_group: str = None,
):
    """Profiler — NCU or kernel launch summary for specific workloads via Modal."""
    import os

    from scripts.bench_utils import parse_int_filter

    # Parse filter values
    indices = parse_int_filter(index) if index else None
    group_values = parse_int_filter(group) if group else None
    exclude_group_values = parse_int_filter(exclude_group) if exclude_group else None

    if ncu_options:
        _list_ncu_options()
        return

    if list:
        entries = _load_workloads_jsonl()
        _list_workloads(entries, indices, group_values, exclude_group_values)
        return

    if index is None:
        print("Usage: modal run scripts/run_modal_profile.py --index <N>")
        print("       modal run scripts/run_modal_profile.py --index <N> --launches")
        print("       modal run scripts/run_modal_profile.py --list")
        print("       modal run scripts/run_modal_profile.py --ncu-options")
        sys.exit(1)

    os.environ.setdefault("TRITON_CACHE_DIR", str(PROJECT_ROOT / ".triton_cache"))

    # Load and filter workloads
    entries = _load_workloads_jsonl()
    indexed = _filter_workloads(entries, indices, group_values, exclude_group_values)
    if not indexed:
        print("Error: No workloads match the specified filters.", file=sys.stderr)
        sys.exit(1)

    # Pack solution
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text())

    for idx, wl_entry in indexed:
        workload = Workload.model_validate(wl_entry["workload"])
        axes_str = ", ".join(f"{k}={v}" for k, v in sorted(workload.axes.items()))

        if launches:
            print(f"\n{'=' * 60}")
            print(f"Workload {idx}: {workload.uuid[:8]}  ({axes_str})")
            print(f"{'=' * 60}")

            result = run_launches_remote.remote(solution, workload)
            print(result)
        else:
            print(f"\nProfiling workload {idx}: {workload.uuid[:8]}...")
            print(f"  Axes: {axes_str}")
            print(f"  Set: {set}, Page: {page}")
            print(f"  GPU: {_GPU_TYPE}")
            if kernel_name:
                print(f"  Kernel filter: {kernel_name}")
            print()

            ncu_kwargs = {"set": set, "page": page, "timeout": timeout}
            if kernel_name:
                ncu_kwargs["kernel_name"] = kernel_name
            if sections:
                ncu_kwargs["sections"] = [s.strip() for s in sections.split(",")]
            if max_lines > 0:
                ncu_kwargs["max_lines"] = max_lines

            result = run_ncu_remote.remote(solution, workload, ncu_kwargs)
            print(result)

            # Auto-save NCU profile output
            profiles_dir = PROJECT_ROOT / "profiles"
            profiles_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            profile_data = {
                "timestamp": datetime.now().isoformat(),
                "workload_index": idx,
                "workload_uuid": workload.uuid,
                "axes": dict(workload.axes),
                "ncu_set": set,
                "ncu_page": page,
                "kernel_filter": kernel_name or None,
                "backend": f"modal-{_GPU_TYPE.lower()}",
                "output": result,
            }
            profile_path = profiles_dir / f"w{idx}_{timestamp}.json"
            profile_path.write_text(json.dumps(profile_data, indent=2))
            print(f"\nProfile saved to: {profile_path}", file=sys.stderr)

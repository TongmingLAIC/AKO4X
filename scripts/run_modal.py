"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA GPUs via Modal, with trajectory tracking.
GPU type is read from config.toml (set at environment creation time).
Caches reference baseline on first run for stable, efficient subsequent runs.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

# Read config locally; on Modal remote container config.toml doesn't exist
# (image & GPU are already determined at deploy time, so defaults are fine)
_config_path = PROJECT_ROOT / "config.toml"
if _config_path.exists():
    with open(_config_path, "rb") as _f:
        _config = tomllib.load(_f)
    _GPU_TYPE = _config.get("build", {}).get("gpu", "b200").upper()
else:
    _GPU_TYPE = "B200"

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

# Universal image: all language deps pre-installed
_pip_deps = ["torch", "numpy", "cupti-python", "triton", "tilelang", "apache-tvm-ffi"]

_base_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .env({"CUDA_HOME": "/usr/local/cuda"})
)

image = (
    _base_image
    .pip_install(*_pip_deps)
    .pip_install("flashinfer-bench @ https://github.com/flashinfer-ai/flashinfer-bench/archive/refs/heads/main.tar.gz")
    .pip_install("flashinfer-python @ https://github.com/flashinfer-ai/flashinfer/archive/refs/heads/main.tar.gz")
    .pip_install("torch-c-dlpack-ext")
    .run_commands("apt-get update && apt-get install -y git clang && pip install wheel && pip install git+https://github.com/deepseek-ai/DeepGEMM.git@main --no-build-isolation")
    .add_local_file(str(PROJECT_ROOT / "scripts" / "bench_utils.py"), "/root/project/scripts/bench_utils.py")
    .add_local_file(str(PROJECT_ROOT / "config.toml"), "/root/project/config.toml")
)


@app.function(image=image, gpu=f"{_GPU_TYPE}:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(
    solution: Solution,
    config: BenchmarkConfig = None,
    max_workloads: int = 0,
    group_axis: str = "",
    group_values: list = None,
    exclude_group_values: list = None,
    workload_indices: list = None,
) -> dict:
    """Run benchmark on Modal GPU and return results."""
    from flashinfer_bench.bench.benchmark import Benchmark as _Benchmark

    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5, timeout_seconds=300)

    # Import bench_utils from the file copied into the image
    import sys as _sys
    _sys.path.insert(0, "/root/project/scripts")
    from bench_utils import build_bench_trace_set, extract_results

    bench_trace_set = build_bench_trace_set(
        TRACE_SET_PATH, solution,
        max_workloads=max_workloads, group_axis=group_axis, group_values=group_values,
        exclude_group_values=exclude_group_values, workload_indices=workload_indices,
    )
    benchmark = _Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)
    return extract_results(result_trace_set, solution.definition)


@app.local_entrypoint()
def main(label: str = None, force_baseline: bool = False, quiet: bool = False,
         first: int = 0, group: str = None, exclude_group: str = None, index: str = None,
         variance_check: int = 0):
    """Pack solution and run benchmark on Modal GPU, with trajectory tracking.

    --variance-check N: run the same unchanged solution N times to measure
    across-run noise (~N times bench cost). Result saved to trajectory/noise-floor.json.
    """
    from functools import partial

    from scripts.bench_utils import find_group_axis, parse_int_filter, run_and_report, run_variance_check
    from scripts.pack_solution import pack_solution

    # Parse group/exclude/index filters
    group_axis = ""
    group_values = None
    exclude_group_values = None
    workload_indices = None

    if group or exclude_group:
        group_axis = find_group_axis()
        if not group_axis:
            import sys as _sys
            print("ERROR: --group/--exclude-group requires a variable axis in definition.json, but none found.",
                  file=_sys.stderr)
            _sys.exit(1)
    if group:
        group_values = parse_int_filter(group)
    if exclude_group:
        exclude_group_values = parse_int_filter(exclude_group)
    if index:
        workload_indices = parse_int_filter(index)

    if not quiet:
        print("Packing solution from source files...")
    solution_path = pack_solution(quiet=quiet)

    if not quiet:
        print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    if not quiet:
        print(f"Loaded: {solution.name} ({solution.definition})")

    run_fn = partial(run_benchmark.remote, max_workloads=first,
                     group_axis=group_axis, group_values=group_values,
                     exclude_group_values=exclude_group_values,
                     workload_indices=workload_indices)

    if variance_check > 0:
        run_variance_check(
            solution, run_fn,
            n_runs=variance_check,
            backend=f"modal-{_GPU_TYPE.lower()}",
            quiet=quiet,
        )
        return

    run_and_report(
        solution, run_fn,
        force_baseline=force_baseline,
        label=label,
        backend=f"modal-{_GPU_TYPE.lower()}",
        quiet=quiet,
        max_workloads=first,
        group_values=group_values,
        exclude_group_values=exclude_group_values,
        workload_indices=workload_indices,
    )

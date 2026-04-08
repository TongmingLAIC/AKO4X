"""
FlashInfer-Bench NCU Profiler Runner.

Profiles a solution on a specific workload using NVIDIA Nsight Compute (NCU).
Reuses pack_solution() to build Solution objects and the flashinfer-bench NCU agent API.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Solution, TraceSet
from flashinfer_bench.agents import flashinfer_bench_list_ncu_options, flashinfer_bench_run_ncu
from scripts.bench_utils import get_trace_set_path
from scripts.pack_solution import pack_solution


def load_workloads():
    """Load workloads from the trace set. Returns (operator_name, list of Trace objects)."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    operator = config["solution"]["definition"]

    trace_set_path = get_trace_set_path()
    trace_set = TraceSet.from_path(trace_set_path)
    traces = trace_set.workloads.get(operator, [])
    if not traces:
        print(f"Error: No workloads found for operator '{operator}'", file=sys.stderr)
        sys.exit(1)
    return operator, traces


def list_workloads():
    """Print workload table with indices."""
    operator, traces = load_workloads()
    print(f"Workloads for '{operator}' ({len(traces)} total):\n")
    print(f"{'Index':<7} {'UUID':<12} {'Axes'}")
    print(f"{'-----':<7} {'----':<12} {'----'}")
    for i, trace in enumerate(traces):
        wl = trace.workload
        uuid_prefix = wl.uuid[:8]
        axes_str = ", ".join(f"{k}={v}" for k, v in sorted(wl.axes.items()))
        print(f"{i:<7} {uuid_prefix:<12} {axes_str}")


def run_ncu(args):
    """Run NCU profiler on a specific workload."""
    # Isolate Triton JIT cache to this project
    os.environ.setdefault("TRITON_CACHE_DIR", str(PROJECT_ROOT / ".triton_cache"))

    operator, traces = load_workloads()

    if args.workload_index < 0 or args.workload_index >= len(traces):
        print(f"Error: Workload index {args.workload_index} out of range [0, {len(traces) - 1}]",
              file=sys.stderr)
        sys.exit(1)

    trace = traces[args.workload_index]
    workload = trace.workload

    trace_set_path = get_trace_set_path()

    print(f"Packing solution from source files...")
    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text())

    print(f"\nProfiling workload {args.workload_index}: {workload.uuid[:8]}...")
    axes_str = ", ".join(f"{k}={v}" for k, v in sorted(workload.axes.items()))
    print(f"  Axes: {axes_str}")
    print(f"  Set: {args.set}, Page: {args.page}")
    if args.kernel_name:
        print(f"  Kernel filter: {args.kernel_name}")
    print()

    kwargs = {
        "set": args.set,
        "page": args.page,
        "timeout": args.timeout,
        "trace_set_path": trace_set_path,
    }
    if args.kernel_name:
        kwargs["kernel_name"] = args.kernel_name
    if args.sections:
        kwargs["sections"] = args.sections
    if args.max_lines is not None:
        kwargs["max_lines"] = args.max_lines

    result = flashinfer_bench_run_ncu(solution, workload, **kwargs)
    print(result)

    # Auto-save profile output with metadata
    profiles_dir = PROJECT_ROOT / "profiles"
    profiles_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_data = {
        "timestamp": datetime.now().isoformat(),
        "workload_index": args.workload_index,
        "workload_uuid": workload.uuid,
        "axes": dict(workload.axes),
        "ncu_set": args.set,
        "ncu_page": args.page,
        "kernel_filter": args.kernel_name,
        "backend": "local",
        "output": result,
    }
    profile_path = profiles_dir / f"w{args.workload_index}_{timestamp}.json"
    profile_path.write_text(json.dumps(profile_data, indent=2))
    print(f"\nProfile saved to: {profile_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="NCU profiler — profile a solution on a specific workload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python scripts/run_local_profile.py --list                         # list workloads
  python scripts/run_local_profile.py 0                              # NCU profile workload 0
  python scripts/run_local_profile.py 0 --set full --page raw        # full metrics, raw output
  python scripts/run_local_profile.py 0 --page source                # source-correlated counters
  python scripts/run_local_profile.py 0 --kernel-name "my_kernel"    # filter by kernel name
  python scripts/run_local_profile.py --ncu-options                  # list available NCU sets/sections
""",
    )

    parser.add_argument("workload_index", nargs="?", type=int, default=None,
                        help="Index of the workload to profile (see --list)")
    parser.add_argument("--list", action="store_true",
                        help="List workloads with indices")
    parser.add_argument("--ncu-options", action="store_true",
                        help="List available NCU sets and sections")
    parser.add_argument("--set", default="detailed",
                        help="NCU section set to collect (default: detailed)")
    parser.add_argument("--page", default="details", choices=["raw", "details", "source"],
                        help="NCU output page format (default: details)")
    parser.add_argument("--kernel-name", default=None,
                        help="Filter by kernel name (full-match regex; use .*name.* for substring)")
    parser.add_argument("--sections", nargs="+", default=None,
                        help="Additional NCU sections to collect beyond the set")
    parser.add_argument("--max-lines", type=int, default=None,
                        help="Maximum number of output lines")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout in seconds for NCU profiling (default: 60)")

    args = parser.parse_args()

    if args.list:
        list_workloads()
    elif args.ncu_options:
        print(flashinfer_bench_list_ncu_options())
    elif args.workload_index is not None:
        run_ncu(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

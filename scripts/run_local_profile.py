"""
FlashInfer-Bench NCU Profiler Runner.

Profiles a solution on specific workloads using NVIDIA Nsight Compute (NCU)
or lightweight torch.profiler for kernel launch analysis.
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
from scripts.bench_utils import find_group_axis, get_trace_set_path, parse_int_filter
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


def filter_workloads(traces, indices=None, group_values=None, exclude_group_values=None):
    """Filter workloads by index, group, and exclusion. Returns list of (original_index, trace)."""
    group_axis = find_group_axis() if (group_values or exclude_group_values) else ""

    indexed = list(enumerate(traces))

    if indices is not None:
        index_set = set(indices)
        indexed = [(i, t) for i, t in indexed if i in index_set]

    if group_values and group_axis:
        group_set = set(group_values)
        indexed = [(i, t) for i, t in indexed if t.workload.axes.get(group_axis) in group_set]

    if exclude_group_values and group_axis:
        exclude_set = set(exclude_group_values)
        indexed = [(i, t) for i, t in indexed if t.workload.axes.get(group_axis) not in exclude_set]

    return indexed


def list_workloads(traces, indices=None, group_values=None, exclude_group_values=None):
    """Print workload table with indices, optionally filtered."""
    operator = traces[0].workload.uuid.split("-")[0] if traces else "?"
    indexed = filter_workloads(traces, indices, group_values, exclude_group_values)

    total = len(traces)
    shown = len(indexed)
    label = f" (filtered {shown}/{total})" if shown < total else ""
    print(f"Workloads ({total} total{label}):\n")
    print(f"{'Index':<7} {'UUID':<12} {'Axes'}")
    print(f"{'-----':<7} {'----':<12} {'----'}")
    for i, trace in indexed:
        wl = trace.workload
        uuid_prefix = wl.uuid[:8]
        axes_str = ", ".join(f"{k}={v}" for k, v in sorted(wl.axes.items()))
        print(f"{i:<7} {uuid_prefix:<12} {axes_str}")


def run_ncu(args, traces):
    """Run NCU profiler on selected workloads."""
    # Isolate Triton JIT cache to this project
    os.environ.setdefault("TRITON_CACHE_DIR", str(PROJECT_ROOT / ".triton_cache"))

    indexed = filter_workloads(traces, args.indices, args.group_values, args.exclude_group_values)
    if not indexed:
        print("Error: No workloads match the specified filters.", file=sys.stderr)
        sys.exit(1)

    trace_set_path = get_trace_set_path()

    print("Packing solution from source files...")
    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text())

    for idx, trace in indexed:
        workload = trace.workload
        axes_str = ", ".join(f"{k}={v}" for k, v in sorted(workload.axes.items()))
        print(f"\nProfiling workload {idx}: {workload.uuid[:8]}...")
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
            "workload_index": idx,
            "workload_uuid": workload.uuid,
            "axes": dict(workload.axes),
            "ncu_set": args.set,
            "ncu_page": args.page,
            "kernel_filter": args.kernel_name,
            "backend": "local",
            "output": result,
        }
        profile_path = profiles_dir / f"w{idx}_{timestamp}.json"
        profile_path.write_text(json.dumps(profile_data, indent=2))
        print(f"\nProfile saved to: {profile_path}", file=sys.stderr)


def run_launches(args, traces):
    """Run torch.profiler to summarize kernel launches."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    from flashinfer_bench.bench.evaluators.utils import allocate_outputs
    from flashinfer_bench.bench.utils import gen_inputs
    from flashinfer_bench.compile import BuilderRegistry

    os.environ.setdefault("TRITON_CACHE_DIR", str(PROJECT_ROOT / ".triton_cache"))

    indexed = filter_workloads(traces, args.indices, args.group_values, args.exclude_group_values)
    if not indexed:
        print("Error: No workloads match the specified filters.", file=sys.stderr)
        sys.exit(1)

    trace_set_path = get_trace_set_path()
    trace_set = TraceSet.from_path(trace_set_path)

    print("Packing solution from source files...")
    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text())
    definition = trace_set.definitions[solution.definition]

    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, solution)

    for idx, trace in indexed:
        workload = trace.workload
        axes_str = ", ".join(f"{k}={v}" for k, v in sorted(workload.axes.items()))
        print(f"\n{'=' * 60}")
        print(f"Workload {idx}: {workload.uuid[:8]}  ({axes_str})")
        print(f"{'=' * 60}")

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

        print(f"Total CUDA kernel launches: {total_launches}")
        print(f"Total CUDA time: {total_cuda_us / 1000:.3f} ms")
        print()
        print(events.table(sort_by="cuda_time_total", row_limit=args.max_lines if args.max_lines and args.max_lines > 0 else 50))

    runnable.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="NCU profiler / kernel launch analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python scripts/run_local_profile.py --list                         # list workloads
  python scripts/run_local_profile.py --index 0                     # NCU profile workload 0
  python scripts/run_local_profile.py --index 0 --set full          # full metrics
  python scripts/run_local_profile.py --index 0 --launches          # quick launch summary
  python scripts/run_local_profile.py --index 0,3,5 --launches      # batch launch summary
  python scripts/run_local_profile.py --index 0-4 --launches        # range
  python scripts/run_local_profile.py --list --exclude-group 1,14107  # filtered list
  python scripts/run_local_profile.py --ncu-options                  # list NCU sets/sections
""",
    )

    # Workload selection
    parser.add_argument("--index", type=str, default=None, metavar="INDICES",
                        help="Workload indices to profile (e.g. 0, 0,3,5, or 2-8)")
    parser.add_argument("--list", action="store_true",
                        help="List workloads with indices")
    parser.add_argument("--group", type=str, default=None, metavar="VALUES",
                        help="Filter by group axis values (e.g. 8,16 or 32-901)")
    parser.add_argument("--exclude-group", type=str, default=None, metavar="VALUES",
                        help="Exclude by group axis values (e.g. 1,14107)")

    # Mode
    parser.add_argument("--launches", action="store_true",
                        help="Quick kernel launch summary via torch.profiler (instead of NCU)")
    parser.add_argument("--ncu-options", action="store_true",
                        help="List available NCU sets and sections")

    # NCU options
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

    # Parse filter values
    args.indices = parse_int_filter(args.index) if args.index else None
    args.group_values = parse_int_filter(args.group) if args.group else None
    args.exclude_group_values = parse_int_filter(args.exclude_group) if args.exclude_group else None

    if args.ncu_options:
        print(flashinfer_bench_list_ncu_options())
    elif args.list:
        _, traces = load_workloads()
        list_workloads(traces, args.indices, args.group_values, args.exclude_group_values)
    elif args.indices is not None:
        _, traces = load_workloads()
        if args.launches:
            run_launches(args, traces)
        else:
            run_ncu(args, traces)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

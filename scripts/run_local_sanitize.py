"""
FlashInfer-Bench Compute-Sanitizer Runner (Local Backend).

Thin wrapper around flashinfer_bench_run_sanitizer. Requires a host
`compute-sanitizer` binary on PATH (ships with CUDA toolkit). `--tool`
forwards to the `sanitizer_types` argument; `all` passes None (runs all
four). Output is post-processed by summarize_sanitizer_noise to surface
a "NOISE FILTER" banner before the raw sanitizer text.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Solution, TraceSet
from flashinfer_bench.agents import flashinfer_bench_run_sanitizer
from scripts.bench_utils import (
    find_group_axis,
    get_trace_set_path,
    parse_int_filter,
    summarize_sanitizer_noise,
)
from scripts.pack_solution import pack_solution


_VALID_TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck", "all")


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


def run_sanitizer(args, traces):
    """Run compute-sanitizer via flashinfer-bench."""
    if not shutil.which("compute-sanitizer"):
        print("Error: `compute-sanitizer` not found on PATH. Install the CUDA toolkit "
              "(which ships compute-sanitizer) or use the Modal backend.", file=sys.stderr)
        sys.exit(1)

    os.environ.setdefault("TRITON_CACHE_DIR", str(PROJECT_ROOT / ".triton_cache"))

    indexed = filter_workloads(traces, args.indices, args.group_values, args.exclude_group_values)
    if not indexed:
        print("Error: No workloads match the specified filters.", file=sys.stderr)
        sys.exit(1)

    trace_set_path = get_trace_set_path()

    print("Packing solution from source files...")
    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text())

    sanitizer_types = None if args.tool == "all" else [args.tool]

    for idx, trace in indexed:
        workload = trace.workload
        axes_str = ", ".join(f"{k}={v}" for k, v in sorted(workload.axes.items()))
        print(f"\nSanitizing workload {idx}: {workload.uuid[:8]}...")
        print(f"  Axes: {axes_str}")
        print(f"  Tool: {args.tool}")
        print()

        kwargs = {
            "sanitizer_types": sanitizer_types,
            "timeout": args.timeout,
            "trace_set_path": trace_set_path,
        }
        if args.max_lines is not None and args.max_lines > 0:
            kwargs["max_lines"] = args.max_lines

        result = flashinfer_bench_run_sanitizer(solution, workload, **kwargs)
        print(summarize_sanitizer_noise(result))

        san_dir = PROJECT_ROOT / "sanitizer"
        san_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "timestamp": datetime.now().isoformat(),
            "workload_index": idx,
            "workload_uuid": workload.uuid,
            "axes": dict(workload.axes),
            "tool": args.tool,
            "backend": "local",
            "output": result,
        }
        out_path = san_dir / f"w{idx}_{args.tool}_{timestamp}.json"
        out_path.write_text(json.dumps(data, indent=2))
        print(f"\nSanitizer output saved to: {out_path}", file=sys.stderr)


def main():
    # Line-buffer stdout so progress prints stream to caller when piped.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="compute-sanitizer runner (memcheck / racecheck / initcheck / synccheck)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python scripts/run_local_sanitize.py --list                              # list workloads
  python scripts/run_local_sanitize.py --index 5                          # memcheck (default)
  python scripts/run_local_sanitize.py --index 5 --tool initcheck         # uninit reads
  python scripts/run_local_sanitize.py --index 5 --tool racecheck         # SMEM races
  python scripts/run_local_sanitize.py --index 5 --tool all               # run all four
  python scripts/run_local_sanitize.py --index 0,3,5 --tool memcheck      # batch
""",
    )
    parser.add_argument("--index", type=str, default=None, metavar="INDICES",
                        help="Workload indices to sanitize (e.g. 0, 0,3,5, or 2-8)")
    parser.add_argument("--list", action="store_true",
                        help="List workloads with indices")
    parser.add_argument("--group", type=str, default=None, metavar="VALUES",
                        help="Filter by group axis values (e.g. 8,16 or 32-901)")
    parser.add_argument("--exclude-group", type=str, default=None, metavar="VALUES",
                        help="Exclude by group axis values (e.g. 1,14107)")
    parser.add_argument("--tool", default="memcheck", choices=_VALID_TOOLS,
                        help="Sanitizer tool (default: memcheck; 'all' runs every tool)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-tool timeout in seconds (default: 300)")
    parser.add_argument("--max-lines", type=int, default=None,
                        help="Truncate output to N lines")

    args = parser.parse_args()

    args.indices = parse_int_filter(args.index) if args.index else None
    args.group_values = parse_int_filter(args.group) if args.group else None
    args.exclude_group_values = parse_int_filter(args.exclude_group) if args.exclude_group else None

    if args.list:
        _, traces = load_workloads()
        list_workloads(traces, args.indices, args.group_values, args.exclude_group_values)
    elif args.indices is not None:
        _, traces = load_workloads()
        run_sanitizer(args, traces)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

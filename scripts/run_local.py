"""
FlashInfer-Bench Local Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks locally.
Caches reference baseline on first run for stable, efficient subsequent runs.
"""

import argparse
import os
import sys
from functools import partial
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution
from scripts.bench_utils import (
    build_bench_trace_set,
    extract_results,
    find_group_axis,
    get_trace_set_path,
    parse_int_filter,
    run_and_report,
    run_variance_check,
)
from scripts.pack_solution import pack_solution


def run_benchmark(
    solution: Solution,
    config: BenchmarkConfig,
    max_workloads: int = 0,
    group_axis: str = "",
    group_values: list | None = None,
    exclude_group_values: list | None = None,
    workload_indices: list | None = None,
    smoke: bool = False,
) -> dict:
    """Run benchmark locally and return results."""
    trace_set_path = get_trace_set_path()
    bench_trace_set = build_bench_trace_set(
        trace_set_path, solution,
        max_workloads=max_workloads, group_axis=group_axis, group_values=group_values,
        exclude_group_values=exclude_group_values, workload_indices=workload_indices,
        smoke=smoke,
    )
    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)
    return extract_results(result_trace_set, solution.definition)


def main():
    """Pack solution and run benchmark."""
    # Line-buffer stdout so per-run progress prints stream to caller when piped.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Run benchmark with optional trajectory tracking")
    parser.add_argument("label", nargs="?", default=None, help="Optional label for trajectory tracking")
    parser.add_argument("--label", dest="label_flag", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--force-baseline", action="store_true",
                        help="Force re-profiling of reference baseline")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Only print score summary and per-group breakdown")
    parser.add_argument("--first", type=int, default=0, metavar="N",
                        help="Only run first N workloads (quick test mode, does not cache baseline)")
    parser.add_argument("--group", type=str, default=None, metavar="VALUES",
                        help="Only run workloads matching group axis values (comma-separated or range, e.g. --group 8,16 or --group 32-901)")
    parser.add_argument("--exclude-group", type=str, default=None, metavar="VALUES",
                        help="Exclude workloads matching group axis values (e.g. --exclude-group 1,14107)")
    parser.add_argument("--index", type=str, default=None, metavar="INDICES",
                        help="Only run specific workloads by index (e.g. --index 0,3,5 or --index 2-8)")
    parser.add_argument("--variance-check", type=int, default=0, metavar="N",
                        help="Run unchanged solution N times to measure across-run noise (>=2)")
    parser.add_argument("--smoke", action="store_true",
                        help="Run one workload per distinct group-axis bucket (covers every "
                             "group with minimum workload count). Does not cache baseline.")
    args = parser.parse_args()

    label = args.label_flag or args.label

    # Parse group/exclude/index filters
    group_axis = ""
    group_values = None
    exclude_group_values = None
    workload_indices = None

    if args.group or args.exclude_group or args.smoke:
        group_axis = find_group_axis()
        if not group_axis and (args.group or args.exclude_group):
            print("ERROR: --group/--exclude-group requires a variable axis in definition.json, but none found.",
                  file=sys.stderr)
            sys.exit(1)
        # --smoke with no group_axis falls back silently to first workload.
    if args.group:
        group_values = parse_int_filter(args.group)
    if args.exclude_group:
        exclude_group_values = parse_int_filter(args.exclude_group)
    if args.index:
        workload_indices = parse_int_filter(args.index)

    # Isolate Triton JIT cache to this project
    os.environ.setdefault("TRITON_CACHE_DIR", str(PROJECT_ROOT / ".triton_cache"))

    if not args.quiet:
        print("Packing solution from source files...")
    solution_path = pack_solution(quiet=args.quiet)

    if not args.quiet:
        print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    if not args.quiet:
        print(f"Loaded: {solution.name} ({solution.definition})")

    run_fn = partial(run_benchmark, max_workloads=args.first,
                     group_axis=group_axis, group_values=group_values,
                     exclude_group_values=exclude_group_values,
                     workload_indices=workload_indices,
                     smoke=args.smoke)

    if args.variance_check > 0:
        run_variance_check(
            solution, run_fn,
            n_runs=args.variance_check,
            backend="local",
            quiet=args.quiet,
        )
        return

    run_and_report(
        solution, run_fn,
        force_baseline=args.force_baseline,
        label=label,
        backend="local",
        quiet=args.quiet,
        max_workloads=args.first,
        group_values=group_values,
        exclude_group_values=exclude_group_values,
        workload_indices=workload_indices,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()

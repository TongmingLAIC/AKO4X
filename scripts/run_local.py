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
    run_and_report,
)
from scripts.pack_solution import pack_solution


def run_benchmark(
    solution: Solution,
    config: BenchmarkConfig,
    max_workloads: int = 0,
    group_axis: str = "",
    group_values: list | None = None,
) -> dict:
    """Run benchmark locally and return results."""
    trace_set_path = get_trace_set_path()
    bench_trace_set = build_bench_trace_set(
        trace_set_path, solution,
        max_workloads=max_workloads, group_axis=group_axis, group_values=group_values,
    )
    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)
    return extract_results(result_trace_set, solution.definition)


def main():
    """Pack solution and run benchmark."""
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
                        help="Only run workloads matching group axis values (comma-separated, e.g. --group 8,16)")
    args = parser.parse_args()

    label = args.label_flag or args.label

    # Parse --group into axis name + values
    group_axis = ""
    group_values = None
    if args.group:
        group_axis = find_group_axis()
        if not group_axis:
            print("ERROR: --group requires a variable axis in definition.json, but none found.",
                  file=sys.stderr)
            sys.exit(1)
        group_values = [int(v.strip()) for v in args.group.split(",")]

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
                     group_axis=group_axis, group_values=group_values)
    run_and_report(
        solution, run_fn,
        force_baseline=args.force_baseline,
        label=label,
        backend="local",
        quiet=args.quiet,
        max_workloads=args.first,
        group_values=group_values,
    )


if __name__ == "__main__":
    main()

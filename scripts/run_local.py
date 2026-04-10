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
    get_trace_set_path,
    run_and_report,
)
from scripts.pack_solution import pack_solution


def run_benchmark(solution: Solution, config: BenchmarkConfig, max_workloads: int = 0) -> dict:
    """Run benchmark locally and return results."""
    trace_set_path = get_trace_set_path()
    bench_trace_set = build_bench_trace_set(trace_set_path, solution, max_workloads=max_workloads)
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
    args = parser.parse_args()

    label = args.label_flag or args.label

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

    run_fn = partial(run_benchmark, max_workloads=args.first) if args.first else run_benchmark
    run_and_report(
        solution, run_fn,
        force_baseline=args.force_baseline,
        label=label,
        backend="local",
        quiet=args.quiet,
        max_workloads=args.first,
    )


if __name__ == "__main__":
    main()

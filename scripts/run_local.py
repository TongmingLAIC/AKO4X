"""
FlashInfer-Bench Local Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks locally.
Caches reference baseline on first run for stable, efficient subsequent runs.
"""

import argparse
import sys
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


def run_benchmark(solution: Solution, config: BenchmarkConfig) -> dict:
    """Run benchmark locally and return results."""
    trace_set_path = get_trace_set_path()
    bench_trace_set = build_bench_trace_set(trace_set_path, solution)
    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)
    return extract_results(result_trace_set, solution.definition)


def main():
    """Pack solution and run benchmark."""
    parser = argparse.ArgumentParser(description="Run benchmark with optional trajectory tracking")
    parser.add_argument("--label", type=str, default=None, help="Optional label for trajectory tracking")
    parser.add_argument("--force-baseline", action="store_true",
                        help="Force re-profiling of reference baseline")
    args = parser.parse_args()

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    run_and_report(
        solution, run_benchmark,
        force_baseline=args.force_baseline,
        label=args.label,
        backend="local",
    )


if __name__ == "__main__":
    main()

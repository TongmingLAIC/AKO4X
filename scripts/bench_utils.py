"""
Shared utilities for benchmark runners.

Provides baseline caching, scoring, result printing, trajectory tracking,
and the two-phase benchmark orchestration used by both run_local.py and run_modal.py.
"""

import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable

from flashinfer_bench import BenchmarkConfig, BuildSpec, Solution
from flashinfer_bench.agents import pack_solution_from_files

try:
    import tomllib
except ImportError:
    import tomli as tomllib

# --- Project paths ---
PROJECT_ROOT = Path(__file__).parent.parent
BASELINE_PATH = PROJECT_ROOT / "baseline.json"


def get_language() -> str:
    """Read the build language from config.toml."""
    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config["build"]["language"]

# --- Benchmark constants ---
BASELINE_ITERATIONS = 20   # Reference profiling: low to avoid CUPTI cache explosion
SOLUTION_ITERATIONS = 100  # Solution profiling: high for accuracy
NUM_TRIALS = 5
WARMUP_RUNS = 3


# ---------------------------------------------------------------------------
# Baseline caching
# ---------------------------------------------------------------------------

def get_workload_uuids() -> set:
    """Read workload UUIDs from docs/workloads.jsonl."""
    workloads_path = PROJECT_ROOT / "docs" / "workloads.jsonl"
    uuids = set()
    if workloads_path.exists():
        with open(workloads_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    w = json.loads(line)
                    uuids.add(w["workload"]["uuid"])
    return uuids


def load_baseline() -> dict | None:
    """Load cached baseline if it exists and is still valid.

    Returns None if baseline.json is missing or workload UUIDs have changed.
    """
    if not BASELINE_PATH.exists():
        return None

    baseline = json.loads(BASELINE_PATH.read_text())

    # Staleness check: cached workload UUIDs must match current workloads
    current_uuids = get_workload_uuids()
    cached_uuids = set(baseline.get("workloads", {}).keys())
    if current_uuids != cached_uuids:
        print("Baseline cache is stale (workloads changed), will re-profile.", file=sys.stderr)
        return None

    return baseline


def save_baseline(results: dict):
    """Cache reference latency per workload from benchmark results to baseline.json.

    When profiling the reference as a solution, the reference latency is in the
    'latency_ms' field (the solution's own latency).
    """
    workloads = {}
    operator = None
    for def_name, traces in results.items():
        operator = def_name
        for uuid, result in traces.items():
            lat_ms = result.get("latency_ms")
            if lat_ms is not None:
                workloads[uuid] = {"reference_latency_ms": lat_ms}

    if not workloads:
        print("WARNING: Reference baseline produced no valid latencies, skipping cache.",
              file=sys.stderr)
        return

    data = {
        "operator": operator,
        "benchmark_config": {
            "warmup_runs": WARMUP_RUNS,
            "iterations": BASELINE_ITERATIONS,
            "num_trials": NUM_TRIALS,
        },
        "workloads": workloads,
    }
    BASELINE_PATH.write_text(json.dumps(data, indent=2))
    print(f"Baseline cached to {BASELINE_PATH} ({len(workloads)} workloads)")


def inject_baseline(results: dict, baseline: dict) -> dict:
    """Replace reference_latency_ms=0.0 with cached values and recompute speedup_factor."""
    if "workloads" not in baseline:
        print("Warning: baseline.json missing 'workloads' key, skipping baseline injection.", file=sys.stderr)
        return results
    cached = baseline["workloads"]
    for def_name, traces in results.items():
        for uuid, result in traces.items():
            if uuid in cached:
                ref_ms = cached[uuid].get("reference_latency_ms")
                if ref_ms is None:
                    continue
                result["reference_latency_ms"] = ref_ms
                sol_ms = result.get("latency_ms")
                if sol_ms and sol_ms > 0:
                    result["speedup_factor"] = ref_ms / sol_ms
    return results


def make_config(iterations: int) -> BenchmarkConfig:
    """Create BenchmarkConfig with profile_baseline=False (always skip reference timing)."""
    return BenchmarkConfig(
        warmup_runs=WARMUP_RUNS,
        iterations=iterations,
        num_trials=NUM_TRIALS,
        profile_baseline=False,
    )


def pack_reference_as_solution() -> Solution:
    """Pack the reference implementation from definition.json as a Solution object.

    Extracts the pure-Python reference code and packs it as if it were a Triton
    solution, so it can be benchmarked to measure reference latency.
    """
    # Read reference code from definition.json
    def_path = PROJECT_ROOT / "docs" / "definition.json"
    with open(def_path) as f:
        definition = json.load(f)

    ref_code = definition.get("reference", "")
    if not ref_code:
        raise ValueError("No 'reference' field in definition.json")

    # Reference is always Python/Triton code — pack as triton regardless of solution language
    spec = BuildSpec(
        language="triton",
        target_hardware=["cuda"],
        entry_point="kernel.py::run",
        destination_passing_style=False,  # reference is always value-returning
    )

    # Write reference to a temp dir and pack it
    with tempfile.TemporaryDirectory() as tmp_dir:
        ref_path = Path(tmp_dir) / "kernel.py"
        ref_path.write_text(ref_code)

        solution = pack_solution_from_files(
            path=tmp_dir,
            spec=spec,
            name=f"{definition['name']}-reference-baseline",
            definition=definition["name"],
            author="baseline",
        )

    return solution


def extract_results(result_trace_set, definition_name: str) -> dict:
    """Extract benchmark results from a TraceSet into a plain dict.

    Converts Trace objects with Evaluation data into a serializable dict
    keyed by {definition_name: {workload_uuid: {...}}}.
    """
    traces = result_trace_set.traces.get(definition_name, [])
    results = {definition_name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition_name][trace.workload.uuid] = entry

    return results


def run_and_report(
    solution: Solution,
    run_fn: Callable[[Solution, BenchmarkConfig], dict],
    *,
    force_baseline: bool = False,
    label: str = None,
    backend: str = "local",
):
    """Two-phase benchmark orchestration shared by local and modal runners.

    Args:
        solution: The packed Solution to benchmark.
        run_fn: Callable that takes (solution, config) and returns a results dict.
                For local: direct function call. For modal: run_benchmark.remote().
        force_baseline: If True, re-profile reference even if cached.
        label: Optional label for trajectory tracking.
        backend: Backend name for trajectory metadata.
    """
    baseline = None if force_baseline else load_baseline()
    baseline_was_cached = baseline is not None

    if baseline is None:
        # Phase 1: Benchmark the reference implementation to measure its latency
        print("\nProfiling reference baseline...")
        ref_solution = pack_reference_as_solution()
        baseline_results = run_fn(ref_solution, make_config(BASELINE_ITERATIONS))
        if not baseline_results:
            print("No results returned from baseline profiling!")
            return
        save_baseline(baseline_results)
        baseline = load_baseline()
        if baseline is None:
            print("WARNING: Reference baseline failed for all workloads — cannot compute speedup.",
                  file=sys.stderr)

    # Phase 2: Benchmark the actual solution with high iterations
    print("\nUsing cached reference baseline...")
    print("Running benchmark...")
    results = run_fn(solution, make_config(SOLUTION_ITERATIONS))
    if not results:
        print("No results returned!")
        return
    if baseline is not None:
        results = inject_baseline(results, baseline)

    score = compute_score(results)
    print_results(results, score)
    save_trajectory(results, solution, score, label,
                    baseline_cached=baseline_was_cached, backend=backend)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def find_group_axis() -> str:
    """Find the first variable axis from the definition, for grouping results."""
    def_path = PROJECT_ROOT / "docs" / "definition.json"
    if def_path.exists():
        with open(def_path) as f:
            definition = json.load(f)
        for name, info in definition.get("axes", {}).items():
            if info.get("type") == "var":
                return name
    return ""


def compute_score(results: dict) -> dict:
    """Compute final score and per-group breakdown from benchmark results.

    If ANY workload is not PASSED, the kernel is invalid and final_score=None.
    """
    group_axis = find_group_axis()
    workloads_path = PROJECT_ROOT / "docs" / "workloads.jsonl"
    uuid_to_group = {}
    if workloads_path.exists() and group_axis:
        with open(workloads_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    w = json.loads(line)
                    uuid_to_group[w["workload"]["uuid"]] = w["workload"]["axes"].get(group_axis, "?")

    all_speedups = []
    by_group = {}
    passed = failed = error = 0
    all_valid = True

    for def_name, traces in results.items():
        for uuid, result in traces.items():
            status = result.get("status", "")
            if status == "PASSED" and result.get("speedup_factor") is not None:
                sf = result["speedup_factor"]
                all_speedups.append(sf)
                passed += 1
                grp = uuid_to_group.get(uuid, "?")
                by_group.setdefault(grp, []).append(sf)
            else:
                all_valid = False
                if status == "FAILED":
                    failed += 1
                else:
                    error += 1

    if all_valid and all_speedups:
        final_score = sum(all_speedups) / len(all_speedups)
        group_scores = {g: sum(sfs) / len(sfs) for g, sfs in sorted(by_group.items())}
    else:
        final_score = None
        group_scores = {}

    return {
        "final_score": final_score,
        "group_scores": group_scores,
        "group_axis": group_axis,
        "passed": passed,
        "failed": failed,
        "error": error,
        "total": passed + failed + error,
        "min_speedup": min(all_speedups) if all_speedups else None,
        "max_speedup": max(all_speedups) if all_speedups else None,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(results: dict, score: dict = None):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()

    if score:
        print()
        print("\u2550" * 50)
        if score["final_score"] is not None:
            print(f"  FINAL SCORE (mean speedup): {score['final_score']:.2f}x")
            print(f"  Passed: {score['passed']}/{score['total']} | Min: {score['min_speedup']:.2f}x | Max: {score['max_speedup']:.2f}x")
            group_axis = score.get("group_axis", "group")
            group_str = "  ".join(f"{g}\u2192{avg:.2f}x" for g, avg in score["group_scores"].items())
            print(f"  By {group_axis}:  {group_str}")
        else:
            parts = []
            if score["failed"]:
                parts.append(f"{score['failed']} workloads FAILED")
            if score["error"]:
                parts.append(f"{score['error']} ERROR")
            print(f"  FINAL SCORE: INVALID ({', '.join(parts)})")
            print(f"  All {score['total']} workloads must PASS for a valid score.")
            print(f"  Passed: {score['passed']}/{score['total']}")
        print("\u2550" * 50)


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

def save_trajectory(
    results: dict,
    solution: Solution,
    score: dict = None,
    label: str = None,
    *,
    baseline_cached: bool = False,
    backend: str = "local",
):
    """Save kernel and results to trajectory folder."""
    trajectory_dir = PROJECT_ROOT / "trajectory"
    trajectory_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if label:
        folder_name = f"{timestamp}_{label}"
    else:
        folder_name = timestamp

    run_dir = trajectory_dir / folder_name
    run_dir.mkdir(parents=True)

    # Save all solution files and config.toml
    language = get_language()
    solution_dir = PROJECT_ROOT / "solution" / language
    if solution_dir.is_dir():
        for src_file in solution_dir.iterdir():
            if src_file.is_file():
                shutil.copy2(src_file, run_dir / src_file.name)
    config_path = PROJECT_ROOT / "config.toml"
    if config_path.exists():
        shutil.copy2(config_path, run_dir / "config.toml")

    # Save results with metadata
    trajectory_data = {
        "timestamp": datetime.now().isoformat(),
        "label": label,
        "backend": backend,
        "solution_name": solution.name,
        "definition": solution.definition,
        "baseline_cached": baseline_cached,
        "baseline_iterations": BASELINE_ITERATIONS,
        "solution_iterations": SOLUTION_ITERATIONS,
        "score": score,
        "results": results,
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(trajectory_data, f, indent=2)

    print(f"\nTrajectory saved to: {run_dir}")

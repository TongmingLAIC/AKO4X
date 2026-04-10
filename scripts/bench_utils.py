"""
Shared utilities for benchmark runners.

Provides baseline caching, scoring, result printing, trajectory tracking,
and the two-phase benchmark orchestration used by both run_local.py and run_modal.py.
"""

import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable

from flashinfer_bench import BenchmarkConfig, BuildSpec, Solution, TraceSet
from flashinfer_bench.agents import pack_solution_from_files

try:
    import tomllib
except ImportError:
    import tomli as tomllib

# --- Project paths ---
PROJECT_ROOT = Path(__file__).parent.parent
BASELINE_PATH = PROJECT_ROOT / "baseline.json"
EXPERT_BASELINE_PATH = PROJECT_ROOT / "expert_baseline.json"
DEFINITION_PATH = PROJECT_ROOT / "docs" / "definition.json"
WORKLOADS_PATH = PROJECT_ROOT / "docs" / "workloads.jsonl"


def get_language() -> str:
    """Read the build language from config.toml."""
    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config["build"]["language"]


def _load_benchmark_config() -> dict:
    """Read [benchmark] section from config.toml with defaults."""
    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    bench = config.get("benchmark", {})
    return {
        "baseline_iterations": bench.get("baseline_iterations", 100),
        "solution_iterations": bench.get("solution_iterations", 100),
        "num_trials": bench.get("num_trials", 5),
        "warmup_runs": bench.get("warmup_runs", 3),
        "baseline_cache": bench.get("baseline_cache", ""),
        "atol": bench.get("atol", 0.01),
        "rtol": bench.get("rtol", 0.01),
        "required_matched_ratio": bench.get("required_matched_ratio"),
        "use_isolated_runner": bench.get("use_isolated_runner", False),
        "timeout_seconds": bench.get("timeout_seconds", 300),
    }


# --- Benchmark settings (from config.toml [benchmark] section, with defaults) ---
_bench_cfg = _load_benchmark_config()
BASELINE_ITERATIONS = _bench_cfg["baseline_iterations"]
SOLUTION_ITERATIONS = _bench_cfg["solution_iterations"]
NUM_TRIALS = _bench_cfg["num_trials"]
WARMUP_RUNS = _bench_cfg["warmup_runs"]
BASELINE_CACHE = _bench_cfg["baseline_cache"]
ATOL = _bench_cfg["atol"]
RTOL = _bench_cfg["rtol"]
REQUIRED_MATCHED_RATIO = _bench_cfg["required_matched_ratio"]
USE_ISOLATED_RUNNER = _bench_cfg["use_isolated_runner"]
TIMEOUT_SECONDS = _bench_cfg["timeout_seconds"]


# ---------------------------------------------------------------------------
# Trace set loading
# ---------------------------------------------------------------------------

def get_trace_set_path() -> str:
    """Get trace set path from config.toml or FIB_DATASET_PATH env var."""
    config_path = PROJECT_ROOT / "config.toml"
    if config_path.exists():
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        path = config.get("build", {}).get("dataset_path")
        if path:
            return path
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "Dataset path not found. Either set 'dataset_path' in config.toml "
            "or the FIB_DATASET_PATH environment variable."
        )
    return path


def build_bench_trace_set(trace_set_path: str, solution: Solution, max_workloads: int = 0) -> TraceSet:
    """Load trace set and build a single-operator benchmark TraceSet.

    Args:
        max_workloads: If > 0, only include the first N workloads (for quick testing).
    """
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    if max_workloads > 0:
        workloads = workloads[:max_workloads]

    return TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )


def _load_workloads() -> list[dict]:
    """Load and parse all workloads from docs/workloads.jsonl."""
    if not WORKLOADS_PATH.exists():
        return []
    with open(WORKLOADS_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Baseline caching
# ---------------------------------------------------------------------------

def get_workload_uuids() -> set:
    """Read workload UUIDs from docs/workloads.jsonl."""
    return {w["workload"]["uuid"] for w in _load_workloads()}


def load_baseline() -> dict | None:
    """Load cached baseline if it exists and is still valid.

    Returns None if baseline.json is missing or workload UUIDs have changed.
    """
    if not BASELINE_PATH.exists():
        return None

    try:
        baseline = json.loads(BASELINE_PATH.read_text())
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Failed to parse baseline.json ({e}), will re-profile.", file=sys.stderr)
        return None

    # Staleness check: cached workload UUIDs must match current workloads
    current_uuids = get_workload_uuids()
    cached_uuids = set(baseline.get("workloads", {}).keys())
    if current_uuids != cached_uuids:
        print("Baseline cache is stale (workloads changed), will re-profile.", file=sys.stderr)
        return None

    # Source check: invalidate if baseline source changed (e.g. expert added/removed)
    cached_source = baseline.get("source", "reference")
    expected_source = _baseline_source()
    if cached_source != expected_source:
        print(f"Baseline source changed ({cached_source} → {expected_source}), will re-profile.",
              file=sys.stderr)
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
        "source": _baseline_source(),
        "benchmark_config": {
            "warmup_runs": WARMUP_RUNS,
            "iterations": BASELINE_ITERATIONS,
            "num_trials": NUM_TRIALS,
        },
        "workloads": workloads,
    }
    BASELINE_PATH.write_text(json.dumps(data, indent=2))
    print(f"Baseline cached to {BASELINE_PATH} ({len(workloads)} workloads)")

    # Also save to global cache if configured
    global_path = _get_global_baseline_path()
    if global_path:
        global_path.parent.mkdir(parents=True, exist_ok=True)
        global_path.write_text(json.dumps(data, indent=2))
        print(f"Baseline also cached globally to {global_path}")


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
                if sol_ms is not None and sol_ms > 0:
                    result["speedup_factor"] = ref_ms / sol_ms
    return results


def make_config(iterations: int) -> BenchmarkConfig:
    """Create BenchmarkConfig with profile_baseline=False (always skip reference timing)."""
    return BenchmarkConfig(
        warmup_runs=WARMUP_RUNS,
        iterations=iterations,
        num_trials=NUM_TRIALS,
        profile_baseline=False,
        atol=ATOL,
        rtol=RTOL,
        required_matched_ratio=REQUIRED_MATCHED_RATIO,
        use_isolated_runner=USE_ISOLATED_RUNNER,
        timeout_seconds=TIMEOUT_SECONDS,
    )


def make_baseline_config() -> BenchmarkConfig:
    """Create a lightweight BenchmarkConfig for baseline profiling.

    Uses minimal warmup and trials since the baseline only needs to be
    approximately correct — it is cached and reused across runs.
    """
    return BenchmarkConfig(
        warmup_runs=1,
        iterations=BASELINE_ITERATIONS,
        num_trials=1,
        profile_baseline=False,
        atol=ATOL,
        rtol=RTOL,
        required_matched_ratio=REQUIRED_MATCHED_RATIO,
        use_isolated_runner=USE_ISOLATED_RUNNER,
        timeout_seconds=TIMEOUT_SECONDS,
    )


def _baseline_source() -> str:
    """Return expected baseline source based on whether expert solution exists."""
    return "expert" if EXPERT_BASELINE_PATH.exists() else "reference"


def load_expert_solution() -> Solution | None:
    """Load expert baseline solution from expert_baseline.json if it exists."""
    if not EXPERT_BASELINE_PATH.exists():
        return None
    return Solution.model_validate_json(EXPERT_BASELINE_PATH.read_text())


def make_expert_baseline_config() -> BenchmarkConfig:
    """High-iteration config for expert baseline (runs fast, needs accuracy)."""
    return BenchmarkConfig(
        warmup_runs=WARMUP_RUNS,
        iterations=SOLUTION_ITERATIONS,
        num_trials=NUM_TRIALS,
        profile_baseline=False,
        atol=ATOL,
        rtol=RTOL,
        required_matched_ratio=REQUIRED_MATCHED_RATIO,
        use_isolated_runner=USE_ISOLATED_RUNNER,
        timeout_seconds=TIMEOUT_SECONDS,
    )


def _get_global_baseline_path() -> Path | None:
    """Return path to global baseline cache, or None if not configured."""
    if not BASELINE_CACHE:
        return None
    with open(DEFINITION_PATH) as f:
        operator = json.load(f)["name"]
    return Path(BASELINE_CACHE) / operator / "baseline.json"


def _is_solution_reference() -> bool:
    """Check if solution/kernel.py is identical to the reference implementation."""
    with open(DEFINITION_PATH) as f:
        ref_code = json.load(f).get("reference", "")
    solution_path = PROJECT_ROOT / "solution" / "kernel.py"
    if not solution_path.exists() or not ref_code:
        return False
    return solution_path.read_text().replace('\r\n', '\n').strip() == ref_code.replace('\r\n', '\n').strip()


def pack_reference_as_solution() -> Solution:
    """Pack the reference implementation from definition.json as a Solution object.

    Extracts the pure-Python reference code and packs it as if it were a Triton
    solution, so it can be benchmarked to measure reference latency.
    """
    # Read reference code from definition.json
    with open(DEFINITION_PATH) as f:
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


def _truncate_log(log: str, max_chars: int = 3000) -> str:
    """Truncate log to the last max_chars characters, preserving line boundaries."""
    if not log or len(log) <= max_chars:
        return log
    truncated = log[-max_chars:]
    nl = truncated.find("\n")
    if nl != -1 and nl < 200:
        truncated = truncated[nl + 1:]
    return f"[...truncated...]\n{truncated}"


def extract_results(result_trace_set, definition_name: str) -> dict:
    """Extract benchmark results from a TraceSet into a plain dict.

    Converts Trace objects with Evaluation data into a serializable dict
    keyed by {definition_name: {workload_uuid: {...}}}.
    Includes error logs for non-PASSED workloads.
    """
    traces = result_trace_set.traces.get(definition_name, [])
    results = {definition_name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
                "axes": dict(trace.workload.axes),
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                max_abs = trace.evaluation.correctness.max_absolute_error
                max_rel = trace.evaluation.correctness.max_relative_error
                entry["max_abs_error"] = "NaN" if (max_abs is not None and math.isnan(max_abs)) else max_abs
                entry["max_rel_error"] = "NaN" if (max_rel is not None and math.isnan(max_rel)) else max_rel
            # Include error log for non-PASSED workloads
            if trace.evaluation.status.value != "PASSED" and getattr(trace.evaluation, "log", ""):
                entry["error_log"] = _truncate_log(trace.evaluation.log)
            results[definition_name][trace.workload.uuid] = entry

    return results


def run_and_report(
    solution: Solution,
    run_fn: Callable[[Solution, BenchmarkConfig], dict],
    *,
    force_baseline: bool = False,
    label: str = None,
    backend: str = "local",
    quiet: bool = False,
    max_workloads: int = 0,
):
    """Two-phase benchmark orchestration shared by local and modal runners.

    Args:
        solution: The packed Solution to benchmark.
        run_fn: Callable that takes (solution, config) and returns a results dict.
                For local: direct function call. For modal: run_benchmark.remote().
        force_baseline: If True, re-profile reference even if cached.
        label: Optional label for trajectory tracking.
        backend: Backend name for trajectory metadata.
        max_workloads: If > 0, only benchmark the first N workloads (quick test mode).
                       Partial results are NOT saved to baseline cache.
    """
    quick_mode = max_workloads > 0
    if quick_mode and not quiet:
        print(f"\n*** Quick test mode: running first {max_workloads} workload(s) only ***")

    baseline = None if force_baseline else load_baseline()
    baseline_was_cached = baseline is not None

    if baseline is None:
        # Phase 1: Benchmark baseline to measure its latency
        expert = load_expert_solution()

        # Shortcut: solution is unmodified reference AND no expert baseline.
        # In this case the reference IS the baseline — benchmark it once.
        # When expert exists, we must always profile the expert for correct scoring.
        if expert is None and _is_solution_reference():
            if not quiet:
                print("\nSolution is unmodified reference — skipping baseline profiling.")
                print("Running benchmark to establish baseline from solution latency...")
            results = run_fn(solution, make_config(SOLUTION_ITERATIONS))
            if not results:
                print("No results returned!")
                return
            if not quick_mode:
                save_baseline(results)
            baseline = load_baseline()
            if baseline is not None:
                results = inject_baseline(results, baseline)
            score = compute_score(results)
            print_results(results, score, quiet=quiet)
            save_trajectory(results, solution, score, label,
                            baseline_cached=False, backend=backend)
            return

        if expert is not None:
            if not quiet:
                print(f"\nProfiling expert baseline ({expert.name})...")
            baseline_results = run_fn(expert, make_expert_baseline_config())
        else:
            if not quiet:
                print("\nProfiling reference baseline...")
            ref_solution = pack_reference_as_solution()
            baseline_results = run_fn(ref_solution, make_baseline_config())
        if not baseline_results:
            print("No results returned from baseline profiling!")
            return
        if not quick_mode:
            save_baseline(baseline_results)
            baseline = load_baseline()
        else:
            # In quick mode, use baseline_results directly without caching
            baseline = {"workloads": {}}
            for def_name, traces in baseline_results.items():
                for uuid, result in traces.items():
                    lat_ms = result.get("latency_ms")
                    if lat_ms is not None:
                        baseline["workloads"][uuid] = {"reference_latency_ms": lat_ms}
        if baseline is None:
            print("WARNING: Reference baseline failed for all workloads — cannot compute speedup.",
                  file=sys.stderr)

    # If baseline is cached and solution is still the reference, skip Phase 2 entirely
    if baseline_was_cached and _is_solution_reference():
        if not quiet:
            print("\nSolution is unmodified reference and baseline is cached — nothing to benchmark.")
            print("Modify solution/kernel.py first, then re-run.")
        return

    # Phase 2: Benchmark the actual solution with high iterations
    if not quiet:
        if baseline_was_cached:
            print("\nUsing cached reference baseline...")
        print("Running benchmark...")
    results = run_fn(solution, make_config(SOLUTION_ITERATIONS))
    if not results:
        print("No results returned!")
        return
    if baseline is not None:
        results = inject_baseline(results, baseline)

    score = compute_score(results)
    print_results(results, score, quiet=quiet)
    save_trajectory(results, solution, score, label,
                    baseline_cached=baseline_was_cached, backend=backend)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def find_group_axis() -> str:
    """Find the first variable axis from the definition, for grouping results.

    NOTE: A parameterized version exists in generate_context.py (for spawn-time use).
    Keep both in sync if the logic changes.
    """
    if DEFINITION_PATH.exists():
        with open(DEFINITION_PATH) as f:
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
    uuid_to_group = {}
    if group_axis:
        for w in _load_workloads():
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
                if math.isnan(sf):
                    all_valid = False
                    error += 1
                    continue
                all_speedups.append(sf)
                passed += 1
                grp = uuid_to_group.get(uuid, "?")
                by_group.setdefault(grp, {"speedups": [], "latencies": []})
                by_group[grp]["speedups"].append(sf)
                lat = result.get("latency_ms")
                if lat is not None:
                    by_group[grp]["latencies"].append(lat)
            else:
                all_valid = False
                if status == "FAILED":
                    failed += 1
                else:
                    error += 1

    if all_valid and all_speedups:
        final_score = sum(all_speedups) / len(all_speedups)
        if group_axis:
            group_scores = {
                g: {
                    "speedup": sum(d["speedups"]) / len(d["speedups"]),
                    "latency_ms": sum(d["latencies"]) / len(d["latencies"]) if d["latencies"] else None,
                }
                for g, d in sorted(by_group.items())
            }
        else:
            group_scores = {}
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

def _format_error_log(log: str, max_lines: int = 20) -> str:
    """Extract the last max_lines lines from an error log for display."""
    lines = log.rstrip("\n").split("\n")
    if len(lines) <= max_lines:
        return "\n".join(f"    | {l}" for l in lines)
    shown = lines[-max_lines:]
    return f"    | [...{len(lines) - max_lines} lines omitted...]\n" + "\n".join(f"    | {l}" for l in shown)


def print_results(results: dict, score: dict = None, quiet: bool = False):
    """Print benchmark results in a formatted way."""
    error_logs_shown = 0
    max_error_logs = 3

    if not quiet:
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
                    abs_str = "NaN" if abs_err == "NaN" else f"{abs_err:.2e}"
                    rel_str = "NaN" if rel_err == "NaN" else f"{rel_err:.2e}"
                    print(f" | abs_err={abs_str}, rel_err={rel_str}", end="")

                print()

                # Show error log for non-PASSED workloads (up to max_error_logs)
                if status != "PASSED" and result.get("error_log") and error_logs_shown < max_error_logs:
                    print(_format_error_log(result["error_log"]))
                    error_logs_shown += 1
                elif status != "PASSED" and result.get("error_log") and error_logs_shown == max_error_logs:
                    print("    | (additional error logs omitted — see results.json in trajectory/)")
                    error_logs_shown += 1  # only print this notice once

    if score:
        print()
        print("\u2550" * 50)
        if score["final_score"] is not None:
            print(f"  FINAL SCORE (mean speedup): {score['final_score']:.2f}x")
            print(f"  Passed: {score['passed']}/{score['total']} | Min: {score['min_speedup']:.2f}x | Max: {score['max_speedup']:.2f}x")
            if score.get("group_scores"):
                group_axis = score.get("group_axis", "group")
                parts = []
                for g, d in score["group_scores"].items():
                    s = f"{g}\u2192{d['speedup']:.2f}x"
                    if d.get("latency_ms") is not None:
                        s += f"({d['latency_ms']:.3f}ms)"
                    parts.append(s)
                print(f"  By {group_axis}:  {'  '.join(parts)}")
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
    solution_dir = PROJECT_ROOT / "solution"
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

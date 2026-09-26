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

# Benchmark types/loaders come through the adapter — the sole flashinfer_bench
# importer (scripts/benchmark_adapter.py). Dual import: `scripts.<mod>` when
# this file is imported as a package on the host, bare `<mod>` inside the Modal
# container where /root/project/scripts is on sys.path (the runner adds the
# adapter there alongside bench_utils).
try:
    import scripts.benchmark_adapter as adapter
    from scripts.benchmark_adapter import DATASET_PATH_ENV, LEGACY_DATASET_PATH_ENV
except ModuleNotFoundError as exc:
    # Only fall back to the bare module name when the `scripts` package itself
    # isn't importable (the Modal container puts scripts/ directly on sys.path,
    # so there's no `scripts` package there). A ModuleNotFoundError for anything
    # else — notably flashinfer_bench not being installed — is a real error and
    # must surface, not be masked behind "No module named 'benchmark_adapter'".
    if exc.name not in ("scripts", "scripts.benchmark_adapter"):
        raise
    import benchmark_adapter as adapter
    from benchmark_adapter import DATASET_PATH_ENV, LEGACY_DATASET_PATH_ENV

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
        "atol": bench.get("atol", 0.01),
        "rtol": bench.get("rtol", 0.01),
        "required_matched_ratio": bench.get("required_matched_ratio"),
        # Default True (matches `templates/benchmark.md` recommendation): the
        # isolated runner spawns a fresh subprocess per workload so module-level
        # caches (e.g. _GRAPH_CACHE, autotune state) can't alias across workloads.
        # Was False prior to 2026-04-23 — that mismatch silently corrupted 4/23
        # workloads in the v8 DSA session (cute_reduce_v6 INCORRECT_NUMERICAL).
        # Variants that genuinely need persistence must opt in explicitly with
        # `use_isolated_runner = false`.
        "use_isolated_runner": bench.get("use_isolated_runner", True),
        "timeout_seconds": bench.get("timeout_seconds", 300),
    }


# --- Benchmark settings (from config.toml [benchmark] section, with defaults) ---
_bench_cfg = _load_benchmark_config()
BASELINE_ITERATIONS = _bench_cfg["baseline_iterations"]
SOLUTION_ITERATIONS = _bench_cfg["solution_iterations"]
NUM_TRIALS = _bench_cfg["num_trials"]
WARMUP_RUNS = _bench_cfg["warmup_runs"]
ATOL = _bench_cfg["atol"]
RTOL = _bench_cfg["rtol"]
REQUIRED_MATCHED_RATIO = _bench_cfg["required_matched_ratio"]
USE_ISOLATED_RUNNER = _bench_cfg["use_isolated_runner"]
TIMEOUT_SECONDS = _bench_cfg["timeout_seconds"]


# ---------------------------------------------------------------------------
# Trace set loading
# ---------------------------------------------------------------------------

def get_trace_set_path() -> str:
    """Get trace set path from config.toml or the dataset-path env var."""
    config_path = PROJECT_ROOT / "config.toml"
    if config_path.exists():
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        path = config.get("build", {}).get("dataset_path")
        if path:
            return path
    path = os.environ.get(DATASET_PATH_ENV) or os.environ.get(LEGACY_DATASET_PATH_ENV)
    if not path:
        raise EnvironmentError(
            f"Dataset path not found. Either set 'dataset_path' in config.toml "
            f"or the {DATASET_PATH_ENV} environment variable."
        )
    return path


def parse_int_filter(raw: str) -> list[int]:
    """Parse comma-separated integers with optional range syntax.

    Examples: "8,16" -> [8,16], "32-901" -> [32..901], "8,16,32-64" -> [8,16,32..64]

    Raises ValueError with a caller-friendly message for common paste mistakes
    (trailing/leading commas, empty segments, non-integer tokens, reversed ranges).
    """
    if not raw.strip():
        raise ValueError("empty filter string")
    result = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            raise ValueError(
                f"empty segment in filter {raw!r} (check for stray or leading/trailing commas)"
            )
        # Detect range: digits-digits (not a leading negative sign)
        dash = part.find("-", 1)  # skip pos 0 to allow negative ints
        if dash > 0:
            try:
                lo, hi = int(part[:dash]), int(part[dash + 1:])
            except ValueError:
                raise ValueError(f"invalid range {part!r} in filter {raw!r}")
            if lo > hi:
                raise ValueError(f"invalid range {part!r} (start > end)")
            result.extend(range(lo, hi + 1))
        else:
            try:
                result.append(int(part))
            except ValueError:
                raise ValueError(f"invalid integer {part!r} in filter {raw!r}")
    return sorted(set(result))


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

    Invalidates on workload UUID change, source change (reference↔expert),
    or a genuine environment change (gpu / backend). cuda_version is
    informational — minor CUDA bumps shouldn't force re-profile. A baseline
    whose own `environment.gpu/backend` is missing is stale (pre-migration;
    can't verify comparability).

    gpu/backend compare case-insensitively (spawn.py lowercases `--gpu`;
    older/seeded baselines stored "B200" — an exact compare would discard a
    valid frozen baseline on a pure casing artifact). When the *current*
    environment is undetectable (gpu="unknown": a colocated variant config
    without [build] gpu), the frozen baseline is kept with a warning, not
    silently discarded + locally re-profiled. This fixes a buggy
    over-rejection; frozen-for-comparability semantics are unchanged (what
    counts as a *different* environment is the same — see scripts/CLAUDE.md).
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

    # Environment check: invalidate on a genuine hardware/backend change,
    # but tolerate casing and an undetectable current value (see docstring).
    cached_env = baseline.get("environment", {})
    current_env = _detect_environment()
    for key in ("gpu", "backend"):
        cached_raw = cached_env.get(key)
        cached_val = (cached_raw or "").strip().lower()
        current_val = (current_env.get(key) or "").strip().lower()
        if not cached_val:
            print(f"Baseline environment.{key} missing (pre-migration baseline), will re-profile.",
                  file=sys.stderr)
            return None
        if not current_val or current_val == "unknown":
            print(f"Warning: current environment.{key} undetectable "
                  f"(got {current_env.get(key)!r}); cannot verify comparability "
                  f"vs baseline {key}={cached_raw!r} — keeping the cached frozen "
                  f"baseline (a colocated config without [build] {key}?).",
                  file=sys.stderr)
            continue
        if cached_val != current_val:
            print(f"Baseline environment.{key} mismatch ({cached_raw} → "
                  f"{current_env.get(key)}), will re-profile.", file=sys.stderr)
            return None

    return baseline


def _atomic_write_json(path: Path, data) -> None:
    """Write JSON via tmp + rename so a crash mid-write leaves the prior file intact.

    Load-bearing for `baseline.json` and the campaign-archive seed: a
    non-atomic write that fails halfway can leave truncated JSON, which the
    next run's `_archive_is_shell` parse-error path treats as a Round-0 shell
    and overwrites with fresh measurements — silently swapping the frozen
    comparability denominator.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def save_baseline(results: dict, params: dict):
    """Cache reference latency per workload to baseline.json + (first-time) archive seed.

    When profiling the reference as a solution, the reference latency is in the
    'latency_ms' field (the solution's own latency). `params` must be the
    BenchmarkConfig kwargs actually used to produce `results` — its warmup/iterations/
    num_trials are recorded in metadata so a reader can tell how the cached
    latencies were measured.

    Auto-promote: if `[benchmark] archive_seed_path` is set in config.toml and
    the target is absent OR a Round-0 environment-only shell (no workloads),
    write the real baseline there. The archive seed
    (`reference/<family>/baseline.json`) is the campaign golden — once it has
    real workloads it's never overwritten by a run; only the shell→real
    transition (or first-time creation) writes.
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
        "environment": _detect_environment(),
        "benchmark_config": {
            "warmup_runs": params["warmup_runs"],
            "iterations": params["iterations"],
            "num_trials": params["num_trials"],
        },
        "workloads": workloads,
    }
    _atomic_write_json(BASELINE_PATH, data)
    print(f"Baseline cached to {BASELINE_PATH} ({len(workloads)} workloads)")

    archive_path = _get_archive_seed_path()
    if archive_path and (not archive_path.exists() or _archive_is_shell(archive_path)):
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        # Carry forward campaign-only environment fields that the Round-0 shell
        # seeded but _detect_environment() doesn't re-derive (notably the
        # closed-loop `mode`). Without this, the shell→real overwrite silently
        # drops `mode`, and master.read_campaign_mode falls back to its default
        # on every later round — a Mode-3 campaign would downgrade to Mode 2
        # mid-flight. Detected fields always win; only keys absent from the
        # freshly detected environment are inherited from the shell we replace.
        archive_data = data
        if archive_path.exists():
            try:
                prior_env = json.loads(archive_path.read_text()).get("environment", {})
            except (json.JSONDecodeError, ValueError, OSError):
                prior_env = {}
            carried = {k: v for k, v in prior_env.items()
                       if k not in data["environment"]}
            if carried:
                archive_data = {**data,
                                "environment": {**data["environment"], **carried}}
        _atomic_write_json(archive_path, archive_data)
        env = archive_data["environment"]
        print(
            f"\n*** First-time baseline promoted to {archive_path}\n"
            f"*** Environment: gpu={env['gpu']}, backend={env['backend']}, "
            f"cuda={env['cuda_version']}\n"
            f"*** Source: {data['source']}  |  Workloads: {len(workloads)}\n"
            f"*** Review the numbers, then commit to lock this golden:\n"
            f"***   git add {archive_path}\n",
            file=sys.stderr,
        )


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


def _baseline_source() -> str:
    """Return expected baseline source based on whether expert solution exists."""
    return "expert" if EXPERT_BASELINE_PATH.exists() else "reference"


def _detect_environment() -> dict:
    """Capture runtime environment metadata for baseline comparability.

    gpu lives in [build] (already consumed by modal runners). backend lives
    in [benchmark] (spawn.py injects it there; [build] would conflict with
    colocated configs that can't be TOML-appended to). cuda_version is
    best-effort via torch — only gpu+backend participate in staleness.
    """
    config_path = PROJECT_ROOT / "config.toml"
    gpu = "unknown"
    backend = "unknown"
    if config_path.exists():
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        gpu = config.get("build", {}).get("gpu", "unknown")
        backend = config.get("benchmark", {}).get("backend", "unknown")
    cuda_version = "unknown"
    try:
        import torch
        cuda_version = torch.version.cuda or "unknown"
    except (ImportError, AttributeError):
        pass
    return {
        "gpu": gpu,
        "backend": backend,
        "cuda_version": cuda_version,
        "measured_at": datetime.now().isoformat(timespec="seconds"),
    }


def _get_archive_seed_path() -> Path | None:
    """Return parent-repo archive baseline path from config.toml, or None.

    `[benchmark] archive_seed_path` is injected by spawn.py when the child
    is rooted in a known reference/<family>/ archive; manual spawns without
    that root simply skip auto-promote.
    """
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        return None
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    path_str = config.get("benchmark", {}).get("archive_seed_path", "")
    return Path(path_str) if path_str else None


def _archive_is_shell(path: Path) -> bool:
    """True if the archive baseline is a Round-0 environment-only shell.

    A shell has no `workloads` — Round 0 seeds `{"source": "bootstrap",
    "environment": {...}, "workloads": {}}` so spawn_child can read
    gpu/backend before the first real reference profile exists. The first
    real profile must overwrite it, so it counts as "promotable". A parse
    failure is treated as shell too — a corrupt seed shouldn't permanently
    block the auto-promote — but we emit a loud warning since the same path
    could clobber a partially-written real archive (e.g. crash during the
    earlier save_baseline write).
    """
    try:
        return not json.loads(path.read_text()).get("workloads")
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        print(
            f"WARNING: archive baseline at {path} could not be parsed ({exc}); "
            f"treating as Round-0 shell. If this file previously held real "
            f"measured workloads, the next save_baseline auto-promote will "
            f"overwrite the corrupted bytes — back up {path} before any further "
            f"baseline write if you suspect lost data.",
            file=sys.stderr,
        )
        return True


def load_expert_blob() -> str | None:
    """Return the expert baseline solution-blob (expert_baseline.json text) or None."""
    if not EXPERT_BASELINE_PATH.exists():
        return None
    return EXPERT_BASELINE_PATH.read_text()


# --- Plain params-dict builders (adapter.run does BenchmarkConfig(**params)) ----
# Same frozen tolerance values as the make_*_config trio above, but as a plain
# dict so no BenchmarkConfig type crosses into the adapter's run().

def _common_tolerance_params() -> dict:
    return {
        "profile_baseline": False,
        "atol": ATOL,
        "rtol": RTOL,
        "required_matched_ratio": REQUIRED_MATCHED_RATIO,
        "use_isolated_runner": USE_ISOLATED_RUNNER,
        "timeout_seconds": TIMEOUT_SECONDS,
    }


def _config_params(iterations: int) -> dict:
    """Params for a solution run (mirrors make_config)."""
    return {"warmup_runs": WARMUP_RUNS, "iterations": iterations,
            "num_trials": NUM_TRIALS, **_common_tolerance_params()}


def _baseline_params() -> dict:
    """Params for the lightweight reference baseline (mirrors make_baseline_config)."""
    return {"warmup_runs": 1, "iterations": BASELINE_ITERATIONS,
            "num_trials": 1, **_common_tolerance_params()}


def _expert_baseline_params() -> dict:
    """Params for the high-iteration expert baseline (mirrors make_expert_baseline_config)."""
    return {"warmup_runs": WARMUP_RUNS, "iterations": SOLUTION_ITERATIONS,
            "num_trials": NUM_TRIALS, **_common_tolerance_params()}


def _baseline_display_info(baseline: dict | None) -> dict | None:
    """Return display-friendly baseline info: {source, name} or None.

    `name` is the expert solution name when source is "expert", else None.
    Used by print_results to render "Baseline: expert (X)" vs "python-reference".
    """
    if not baseline:
        return None
    source = baseline.get("source", "reference")
    name = None
    if source == "expert":
        expert_blob = load_expert_blob()
        if expert_blob is not None:
            name = adapter.solution_meta(expert_blob)["name"]
    return {"source": source, "name": name}


def _compute_subset_coverage(
    *,
    max_workloads: int = 0,
    group_values: list | None = None,
    exclude_group_values: list | None = None,
    workload_indices: list | None = None,
    smoke: bool = False,
    extremes: bool = False,
) -> dict | None:
    """Compute subset coverage info for display before a filtered bench run.

    Mirrors the filter logic in `build_bench_trace_set` (indices → group →
    exclude → extremes → smoke → first-N) against the full workloads list, to
    show the agent exactly which workloads are included and what slice of the
    group-axis spectrum they represent.

    Returns None when no filter applies (full bench) or workloads file missing.
    Returns a dict with keys: axis, full_min, full_max, full_count,
    selected_values, selected_count, coverage_pct, characterization.
    """
    workloads = _load_workloads()
    if not workloads:
        return None

    axis = find_group_axis()

    def _val(w, a):
        return w["workload"]["axes"].get(a) if a else None

    full_values = [_val(w, axis) for w in workloads] if axis else []

    # Replay the build_bench_trace_set filter order.
    filtered = list(workloads)
    if workload_indices is not None:
        idx_set = set(workload_indices)
        filtered = [w for i, w in enumerate(filtered) if i in idx_set]
    if group_values and axis:
        gs = set(group_values)
        filtered = [w for w in filtered if _val(w, axis) in gs]
    if exclude_group_values and axis:
        xs = set(exclude_group_values)
        filtered = [w for w in filtered if _val(w, axis) not in xs]
    if extremes and axis and full_values:
        uniq_sorted = sorted({v for v in full_values if v is not None})
        if len(uniq_sorted) >= 2:
            pick = {uniq_sorted[0], uniq_sorted[-1]}
            filtered = [w for w in filtered if _val(w, axis) in pick]
        elif uniq_sorted:
            filtered = [w for w in filtered if _val(w, axis) == uniq_sorted[0]]
    if smoke and axis:
        seen = set()
        smoked = []
        for w in filtered:
            v = _val(w, axis)
            if v not in seen:
                seen.add(v)
                smoked.append(w)
        filtered = smoked
    if max_workloads > 0:
        filtered = filtered[:max_workloads]

    selected_values = [_val(w, axis) for w in filtered] if axis else []
    full_count = len(workloads)
    selected_count = len(filtered)
    coverage_pct = (100.0 * selected_count / full_count) if full_count else 0.0

    characterization = ""
    if axis and full_values and selected_values:
        try:
            full_numeric = [v for v in full_values if isinstance(v, (int, float))]
            sel_numeric = [v for v in selected_values if isinstance(v, (int, float))]
            if full_numeric and sel_numeric:
                f_min, f_max = min(full_numeric), max(full_numeric)
                s_min, s_max = min(sel_numeric), max(sel_numeric)
                if s_min == f_min and s_max == s_min:
                    characterization = "launch-overhead-dominated; compute-throughput optimizations won't show here"
                elif s_min == f_min and s_max == f_max:
                    characterization = "input-spectrum extremes; middle-range workloads not sampled"
                else:
                    characterization = "narrow axis range; optimizations targeting excluded values won't show here"
        except (TypeError, ValueError):
            pass

    return {
        "axis": axis or "workload",
        "full_count": full_count,
        "full_min": (min(v for v in full_values if v is not None) if axis and full_values else None),
        "full_max": (max(v for v in full_values if v is not None) if axis and full_values else None),
        "selected_values": selected_values,
        "selected_count": selected_count,
        "coverage_pct": coverage_pct,
        "characterization": characterization,
    }


def select_workload_uuids(
    definition: str = "",
    *,
    max_workloads: int = 0,
    group_values: list | None = None,
    exclude_group_values: list | None = None,
    workload_indices: list | None = None,
    smoke: bool = False,
    extremes: bool = False,
) -> list[str]:
    """Resolve workload filters to a list of uuids in dataset order.

    Benchmark-agnostic: operates on the plain docs/workloads.jsonl dicts, applying
    the same filter order as the old build_bench_trace_set (index → group →
    exclude → extremes → smoke → first-N), with the same empty-match raises. The
    adapter then runs exactly these uuids. `definition` is accepted for symmetry
    with the adapter surface; the local workloads file is single-operator.
    """
    workloads = _load_workloads()
    if not workloads:
        return []
    axis = find_group_axis()

    def _val(w):
        return w["workload"]["axes"].get(axis) if axis else None

    filtered = list(workloads)

    if workload_indices is not None:
        index_set = set(workload_indices)
        filtered = [w for i, w in enumerate(filtered) if i in index_set]
        if not filtered:
            raise ValueError(f"No workloads match indices {sorted(index_set)}.")

    if group_values and axis:
        group_set = set(group_values)
        filtered = [w for w in filtered if _val(w) in group_set]
        if not filtered:
            raise ValueError(
                f"No workloads match {axis} in "
                f"{{{', '.join(str(v) for v in group_values)}}}. "
                f"Check available values in docs/workloads.jsonl."
            )

    if exclude_group_values and axis:
        exclude_set = set(exclude_group_values)
        filtered = [w for w in filtered if _val(w) not in exclude_set]
        if not filtered:
            raise ValueError(
                f"All workloads excluded by {axis} not in "
                f"{{{', '.join(str(v) for v in exclude_group_values)}}}."
            )

    if extremes:
        if not axis:
            raise ValueError(
                "--extremes requires a variable axis in definition.json, but none found. "
                "Use --first 1 or --index for single-workload probes."
            )
        values = [_val(w) for w in filtered]
        numeric_values = sorted({v for v in values if isinstance(v, (int, float))})
        if not numeric_values:
            raise ValueError(
                f"--extremes needs numeric values on axis '{axis}', but none found "
                "in the current workload set."
            )
        if len(numeric_values) < 2:
            print(
                f"Warning: --extremes on axis '{axis}' — only one unique value "
                f"({numeric_values[0]}) available; falling back to single-workload probe.",
                file=sys.stderr,
            )
            target = {numeric_values[0]}
        else:
            target = {numeric_values[0], numeric_values[-1]}
        filtered = [w for w in filtered if _val(w) in target]

    if smoke:
        if axis:
            seen = set()
            picked = []
            for w in filtered:
                v = _val(w)
                if v not in seen:
                    seen.add(v)
                    picked.append(w)
            filtered = picked
        else:
            filtered = filtered[:1]

    if max_workloads > 0:
        filtered = filtered[:max_workloads]

    return [w["workload"]["uuid"] for w in filtered]


def _format_coverage_line(cov: dict) -> str:
    """Render coverage dict as a human-readable line (1-2 lines)."""
    axis = cov["axis"]
    vals = cov["selected_values"]
    # Compact distinct-values presentation; sort if numeric
    try:
        vals_sorted = sorted(set(v for v in vals if v is not None))
    except TypeError:
        vals_sorted = list(dict.fromkeys(vals))
    vals_str = "{" + ", ".join(str(v) for v in vals_sorted) + "}"
    full_range = ""
    if cov.get("full_min") is not None and cov.get("full_max") is not None:
        full_range = f" of full range [{cov['full_min']}..{cov['full_max']}]"
    head = (
        f"Subset: {axis} ∈ {vals_str}{full_range} across "
        f"{cov['full_count']} workloads ({cov['coverage_pct']:.0f}% coverage)"
    )
    if cov["characterization"]:
        return head + f"\nNote: {cov['characterization']}"
    return head


def _unwrap_bench_result(ret):
    """Normalize run_fn return: plain dict (backward-compat) vs wrapped dict.

    run_benchmark may return either a plain results dict (default) or a wrapped
    dict {"results": ..., "autotune_log": ...} when capture_autotune=True was
    passed. Returns (results, autotune_log) — log is "" when not wrapped.
    """
    if isinstance(ret, dict) and "results" in ret and "autotune_log" in ret:
        return ret["results"], ret.get("autotune_log", "") or ""
    return ret, ""


def _parse_autotune_log(log: str) -> list[str]:
    """Extract Triton autotune 'best config' strings from captured stderr.

    Triton prints lines like:
        Triton autotuning for function <name> finished after ..., best config selected: <config>
    We grab the content after "best config selected: " up to end-of-line.
    Returns [] on empty/malformed input. Multiple entries possible if the run
    autotunes multiple distinct kernels.
    """
    import re
    if not log:
        return []
    try:
        return re.findall(
            r"best config selected:\s*(.+?)(?:\r?\n|$)", log, re.IGNORECASE
        )
    except Exception:
        return []


def _autotune_config_digest(configs: list[str]) -> str:
    """Render a list of autotune configs as a compact digest for a table cell.

    Strategy: if there's exactly one config, truncate to ~40 chars. If multiple,
    take an 8-char stable hash so equal runs compare visually.
    """
    import hashlib
    if not configs:
        return "?"
    if len(configs) == 1:
        c = configs[0].strip()
        return (c[:40] + "…") if len(c) > 40 else c
    joined = "\n".join(c.strip() for c in configs)
    h = hashlib.sha256(joined.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"<{len(configs)}cfgs:{h}>"


def _is_solution_reference() -> bool:
    """Check if solution/kernel.py is identical to the reference implementation."""
    with open(DEFINITION_PATH) as f:
        ref_code = json.load(f).get("reference", "")
    solution_path = PROJECT_ROOT / "solution" / "kernel.py"
    if not solution_path.exists() or not ref_code:
        return False
    return solution_path.read_text().replace('\r\n', '\n').strip() == ref_code.replace('\r\n', '\n').strip()


def _pack_reference() -> str:
    """Pack definition.json's reference code as a solution-blob (agnostic file staging).

    The reference is always Python/Triton value-returning code; the adapter owns
    the actual packing (BuildSpec + solution serialization).
    """
    with open(DEFINITION_PATH) as f:
        definition = json.load(f)
    ref_code = definition.get("reference", "")
    if not ref_code:
        raise ValueError("No 'reference' field in definition.json")
    with tempfile.TemporaryDirectory() as tmp_dir:
        (Path(tmp_dir) / "kernel.py").write_text(ref_code)
        return adapter.pack(
            tmp_dir,
            {"language": "triton", "entry_point": "kernel.py::run",
             "destination_passing_style": False},
            name=f"{definition['name']}-reference-baseline",
            definition=definition["name"],
            author="baseline",
        )


def run_and_report(
    solution_blob: str,
    run_fn: Callable[[str, list, dict], dict],
    *,
    force_baseline: bool = False,
    label: str = None,
    backend: str = "local",
    quiet: bool = False,
    max_workloads: int = 0,
    group_values: list | None = None,
    exclude_group_values: list | None = None,
    workload_indices: list | None = None,
    smoke: bool = False,
    extremes: bool = False,
):
    """Two-phase benchmark orchestration shared by local and modal runners.

    Args:
        solution_blob: The packed solution as a JSON blob (str).
        run_fn: Callable (blob, uuids, params) -> results dict. For local: a thin
                call into the adapter; for modal: run_benchmark.remote().
        force_baseline: If True, re-profile reference even if cached.
        label: Optional label for trajectory tracking.
        backend: Backend name for trajectory metadata.
        max_workloads: If > 0, only benchmark the first N workloads (quick test mode).
                       Partial results are NOT saved to baseline cache.
        group_values: If set, only benchmark workloads matching these group axis values.
                      Partial results are NOT saved to baseline cache.
        exclude_group_values: If set, exclude workloads matching these group axis values.
        workload_indices: If set, only benchmark workloads at these indices.
        smoke: If True, reduce to one workload per distinct group_axis value
               (covers every group with minimum workload count). Partial
               results are NOT saved to baseline cache.
        extremes: If True, pick only the min + max group-axis values (input-spectrum
               correctness probe). Partial results are NOT saved to baseline cache.
    """
    meta = adapter.solution_meta(solution_blob)
    uuids = select_workload_uuids(
        meta["definition"],
        max_workloads=max_workloads, group_values=group_values,
        exclude_group_values=exclude_group_values, workload_indices=workload_indices,
        smoke=smoke, extremes=extremes,
    )

    quick_mode = (max_workloads > 0 or bool(group_values) or bool(exclude_group_values)
                  or bool(workload_indices) or smoke or extremes)
    coverage = None
    if quick_mode and not quiet:
        parts = []
        if workload_indices:
            parts.append(f"indices {','.join(str(i) for i in workload_indices)}")
        if max_workloads > 0:
            parts.append(f"first {max_workloads} workload(s)")
        if group_values:
            axis = find_group_axis()
            parts.append(f"{axis}={','.join(str(v) for v in group_values)}")
        if exclude_group_values:
            axis = find_group_axis()
            parts.append(f"excluding {axis}={','.join(str(v) for v in exclude_group_values)}")
        if smoke:
            axis = find_group_axis() or "workload"
            parts.append(f"smoke (one per {axis} bucket)")
        if extremes:
            axis = find_group_axis() or "workload"
            parts.append(f"extremes (min+max {axis})")
        print(f"\n*** Quick test mode: {' + '.join(parts)} ***")
        coverage = _compute_subset_coverage(
            max_workloads=max_workloads,
            group_values=group_values,
            exclude_group_values=exclude_group_values,
            workload_indices=workload_indices,
            smoke=smoke,
            extremes=extremes,
        )
        if coverage:
            print(_format_coverage_line(coverage))

    baseline = None if force_baseline else load_baseline()
    baseline_was_cached = baseline is not None

    if baseline is None:
        # Phase 1: Benchmark baseline to measure its latency
        expert_blob = load_expert_blob()

        # Shortcut: solution is unmodified reference AND no expert baseline.
        # In this case the reference IS the baseline — benchmark it once.
        # When expert exists, we must always profile the expert for correct scoring.
        if expert_blob is None and _is_solution_reference():
            if not quiet:
                print("\nSolution is unmodified reference — skipping baseline profiling.")
                print("Running benchmark to establish baseline from solution latency...")
            params = _config_params(SOLUTION_ITERATIONS)
            results = run_fn(solution_blob, uuids, params)
            if not results:
                print("No results returned!")
                return
            if not quick_mode:
                save_baseline(results, params)
            baseline = load_baseline()
            if baseline is not None:
                results = inject_baseline(results, baseline)
            score = compute_score(results)
            print_results(
                results, score, quiet=quiet,
                is_full_bench=not quick_mode,
                label=label,
                baseline_info=_baseline_display_info(baseline),
            )
            save_trajectory(results, meta, score, label,
                            baseline_cached=False, backend=backend)
            return

        if expert_blob is not None:
            if not quiet:
                print(f"\nProfiling expert baseline ({adapter.solution_meta(expert_blob)['name']})...")
            baseline_params = _expert_baseline_params()
            baseline_results = run_fn(expert_blob, uuids, baseline_params)
        else:
            if not quiet:
                print("\nProfiling reference baseline...")
            ref_blob = _pack_reference()
            baseline_params = _baseline_params()
            baseline_results = run_fn(ref_blob, uuids, baseline_params)
        if not baseline_results:
            print("No results returned from baseline profiling!")
            return
        if not quick_mode:
            save_baseline(baseline_results, baseline_params)
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
            info = _baseline_display_info(baseline)
            if info and info["source"] == "expert":
                name = info["name"] or "unknown"
                print(f"\nUsing cached baseline: expert ({name})")
            else:
                print("\nUsing cached baseline: python-reference")
        print("Running benchmark...")
    results = run_fn(solution_blob, uuids, _config_params(SOLUTION_ITERATIONS))
    if not results:
        print("No results returned!")
        return
    if baseline is not None:
        results = inject_baseline(results, baseline)

    score = compute_score(results)
    print_results(
        results, score, quiet=quiet,
        is_full_bench=not quick_mode,
        label=label,
        baseline_info=_baseline_display_info(baseline),
    )
    save_trajectory(results, meta, score, label,
                    baseline_cached=baseline_was_cached, backend=backend)


# ---------------------------------------------------------------------------
# Variance check
# ---------------------------------------------------------------------------

def run_variance_check(
    solution_blob: str,
    run_fn: Callable[[str, list, dict], dict],
    n_runs: int,
    *,
    backend: str = "local",
    quiet: bool = False,
    label: str = None,
    max_workloads: int = 0,
    group_values: list | None = None,
    exclude_group_values: list | None = None,
    workload_indices: list | None = None,
    smoke: bool = False,
    extremes: bool = False,
):
    """Run the same unchanged solution N times to measure across-run noise.

    Reports across-run mean/std for the overall score and identifies the most
    variable workloads. Saves results to trajectory/noise-floor.json so future
    diff comparisons can use empirical noise as a reference.

    The reference baseline is loaded from cache or profiled once before the loop.
    Each of the N runs uses the same solution and same baseline; differences
    between runs are pure noise (Modal cold-start, GPU thermal, neighbour load).

    Filter combinators: `run_fn` carries any `--smoke / --first / --group /
    --index` filters from the caller, so `--variance-check 3 --smoke` runs one
    workload per group bucket × 3 runs instead of the full 128 × 3. This is
    the mid-cost validation path between a single smoke test (noisy) and a
    full variance check (~3× 5-min). The noise-floor JSON's `per_workload`
    dict will just reflect the filtered subset — document in the label if it
    matters.
    """
    import statistics

    if n_runs < 2:
        print("ERROR: --variance-check requires N >= 2.", file=sys.stderr)
        sys.exit(1)

    meta = adapter.solution_meta(solution_blob)
    uuids = select_workload_uuids(
        meta["definition"],
        max_workloads=max_workloads, group_values=group_values,
        exclude_group_values=exclude_group_values, workload_indices=workload_indices,
        smoke=smoke, extremes=extremes,
    )

    # Ensure baseline exists (profile once if not cached)
    baseline = load_baseline()
    if baseline is None:
        expert_blob = load_expert_blob()
        if expert_blob is not None:
            if not quiet:
                print(f"\nProfiling expert baseline "
                      f"({adapter.solution_meta(expert_blob)['name']}) — one-time cost...")
            baseline_params = _expert_baseline_params()
            baseline_results, _ = _unwrap_bench_result(run_fn(expert_blob, uuids, baseline_params))
        else:
            if not quiet:
                print("\nProfiling reference baseline — one-time cost...")
            ref_blob = _pack_reference()
            baseline_params = _baseline_params()
            baseline_results, _ = _unwrap_bench_result(run_fn(ref_blob, uuids, baseline_params))
        if baseline_results:
            save_baseline(baseline_results, baseline_params)
            baseline = load_baseline()
        if baseline is None:
            print("ERROR: Failed to establish baseline; cannot compute speedups.",
                  file=sys.stderr)
            sys.exit(1)

    # Run the solution N times
    runs = []
    for i in range(n_runs):
        if not quiet:
            print(f"\n=== Variance-check run {i+1}/{n_runs} ===")
        results, autotune_log = _unwrap_bench_result(
            run_fn(solution_blob, uuids, _config_params(SOLUTION_ITERATIONS))
        )
        if not results:
            print(f"  Run {i+1} returned no results; aborting variance check.",
                  file=sys.stderr)
            sys.exit(1)
        results = inject_baseline(results, baseline)
        score = compute_score(results)
        autotune_configs = _parse_autotune_log(autotune_log)
        autotune_digest = _autotune_config_digest(autotune_configs)
        runs.append({
            "score": score, "results": results,
            "autotune_configs": autotune_configs,
            "autotune_digest": autotune_digest,
        })
        if score["final_score"] is not None:
            cfg_suffix = f"   autotune={autotune_digest}" if autotune_configs else ""
            print(f"  Run {i+1} score: {_fmt_speedup(score['final_score'])}{cfg_suffix}")
        elif score["passed"] == 0 and score["failed"] == 0 and score["error"] == 0:
            wl_counts = {d: len(t) for d, t in results.items()}
            print(
                f"  Run {i+1}: HARNESS ANOMALY (empty results; "
                f"per-definition workload count = {wl_counts}). "
                f"Likely a Modal session-reuse failure — try a fresh "
                f"`bash scripts/bench.sh` invocation.",
                file=sys.stderr,
            )
        else:
            print(f"  Run {i+1}: INVALID ({score['failed']} failed, {score['error']} error)")

    # Aggregate
    valid_scores = [r["score"]["final_score"] for r in runs if r["score"]["final_score"] is not None]
    if len(valid_scores) < 2:
        print("\nERROR: Need at least 2 valid runs to compute variance.", file=sys.stderr)
        sys.exit(1)

    score_mean = statistics.mean(valid_scores)
    score_std = statistics.stdev(valid_scores)
    score_cv = score_std / score_mean if score_mean > 0 else 0.0

    # Per-workload aggregation: collect all (uuid -> [speedups])
    per_workload = {}
    for run in runs:
        for def_name, traces in run["results"].items():
            for uuid, result in traces.items():
                if result.get("speedup_factor") is not None:
                    per_workload.setdefault(uuid, []).append(result["speedup_factor"])

    # Compute per-workload mean/std/CV; sort by CV descending
    workload_stats = []
    for uuid, speedups in per_workload.items():
        if len(speedups) < 2:
            continue
        m = statistics.mean(speedups)
        s = statistics.stdev(speedups)
        cv = s / m if m > 0 else 0.0
        workload_stats.append({
            "uuid": uuid,
            "mean": m,
            "std": s,
            "cv": cv,
            "n": len(speedups),
        })
    workload_stats.sort(key=lambda x: x["cv"], reverse=True)

    # Per-group aggregation: for each group (e.g., batch_size), collect the
    # per-run group-mean speedups and report mean ± std across runs. This
    # helps distinguish "entire score shifted" from "one group got noisy".
    group_axis = find_group_axis()
    per_group_runs = {}  # group_value -> [run1_mean, run2_mean, ...]
    for run in runs:
        gs = run["score"].get("group_scores") or {}
        for g, d in gs.items():
            per_group_runs.setdefault(g, []).append(d["speedup"])
    group_stats = []
    for g, speeds in sorted(per_group_runs.items(),
                            key=lambda x: (0, int(x[0])) if str(x[0]).isdigit() else (1, str(x[0]))):
        if len(speeds) < 2:
            continue
        m = statistics.mean(speeds)
        s = statistics.stdev(speeds)
        cv = s / m if m > 0 else 0.0
        group_stats.append({
            "group": g, "mean": m, "std": s, "cv": cv, "n": len(speeds),
        })

    # Print summary
    print()
    print("\u2550" * 60)
    print(f"  Variance check ({n_runs} runs, baseline cached)")
    print(f"  Baseline: {baseline.get('source', 'unknown')}")
    print("\u2550" * 60)
    for i, r in enumerate(runs, 1):
        sc = r["score"]["final_score"]
        digest = r.get("autotune_digest") or ""
        cfg_suffix = f"   autotune={digest}" if digest and digest != "?" else ""
        if sc is not None:
            print(f"  Run {i}: {_fmt_speedup(sc)}{cfg_suffix}")
        else:
            print(f"  Run {i}: INVALID{cfg_suffix}")
    # Summarize autotune drift: do all runs agree on the same selected configs?
    digests = [r.get("autotune_digest") for r in runs if r.get("autotune_configs")]
    if digests:
        unique_digests = set(digests)
        if len(unique_digests) == 1:
            print(f"  Autotune: stable across all runs ({next(iter(unique_digests))})")
        else:
            print(f"  Autotune DRIFT across runs: {len(unique_digests)} distinct "
                  f"config sets. Pin configs with @triton.autotune(key=...) to stabilize.")
    print()
    print(f"  Across-run mean:  {_fmt_speedup(score_mean)}")
    print(f"  Across-run std:   {_fmt_speedup(score_std)}   (CV: {100*score_cv:.1f}%)")
    print(f"  Range:            {_fmt_speedup(min(valid_scores))} \u2014 {_fmt_speedup(max(valid_scores))}")
    print()
    if workload_stats:
        print(f"  Top 5 most variable workloads (out of {len(workload_stats)}):")
        for w in workload_stats[:5]:
            print(f"    {w['uuid'][:8]}...: {_fmt_speedup(w['mean'])} \u00b1 {_fmt_speedup(w['std'])}  "
                  f"(CV {100*w['cv']:.1f}%)")
    if group_stats:
        print()
        axis_label = group_axis or "group"
        print(f"  Per-{axis_label} noise (across {n_runs} runs):")
        for g in group_stats:
            print(f"    {axis_label}={g['group']}: {_fmt_speedup(g['mean'])} \u00b1 {_fmt_speedup(g['std'])}  "
                  f"(CV {100*g['cv']:.1f}%)")
    print("\u2550" * 60)

    # Persist
    noise_floor = {
        "timestamp": datetime.now().isoformat(),
        "backend": backend,
        "n_runs": n_runs,
        "solution_name": meta["name"],
        "definition": meta["definition"],
        "scores": valid_scores,
        "score_mean": score_mean,
        "score_std": score_std,
        "score_cv": score_cv,
        "score_min": min(valid_scores),
        "score_max": max(valid_scores),
        "per_workload": {w["uuid"]: {k: v for k, v in w.items() if k != "uuid"}
                         for w in workload_stats},
        "per_group": {str(g["group"]): {k: v for k, v in g.items() if k != "group"}
                      for g in group_stats},
        "group_axis": group_axis or None,
    }
    out_path = PROJECT_ROOT / "trajectory" / "noise-floor.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(noise_floor, f, indent=2)
    print(f"\n  Saved noise floor to: {out_path.relative_to(PROJECT_ROOT)}")

    # When variance-check is tied to a --label, also persist to the label's
    # trajectory subdir so noise-floor history is preserved across iters (the
    # root noise-floor.json is overwritten each call). This lets agents diff
    # noise floors across iterations and spot when variance itself regressed.
    if label:
        safe_label = _sanitize_label(label).replace(" ", "_")
        label_dir = PROJECT_ROOT / "trajectory" / safe_label
        if label_dir.exists():
            label_out = label_dir / "noise-floor.json"
            with open(label_out, "w") as f:
                json.dump(noise_floor, f, indent=2)
            print(f"  Saved label-scoped noise floor to: {label_out.relative_to(PROJECT_ROOT)}")
        else:
            # Label dir doesn't exist yet (variance-check called before a
            # labeled bench created it). Create a minimal dir for the history.
            label_dir.mkdir(parents=True, exist_ok=True)
            label_out = label_dir / "noise-floor.json"
            with open(label_out, "w") as f:
                json.dump(noise_floor, f, indent=2)
            print(f"  Saved label-scoped noise floor to: {label_out.relative_to(PROJECT_ROOT)} (new)")


# ---------------------------------------------------------------------------
# A/B same-session compare
# ---------------------------------------------------------------------------

# Source file extensions recognized inside a trajectory snapshot. Must stay in
# sync with the copy list in `spawn.py::populate_child` (existing-mode sibling
# copy) and whatever pack_solution_from_files considers a kernel source.
_SNAPSHOT_SOURCE_EXTS = {".py", ".cu", ".cpp", ".hip", ".h", ".hpp", ".cuh"}


def _snapshot_has_sources(snapshot: Path) -> bool:
    return any(f.is_file() and f.suffix.lower() in _SNAPSHOT_SOURCE_EXTS
               for f in snapshot.iterdir())


def _find_trajectory_snapshot(label: str) -> Path:
    """Find a trajectory dir containing kernel sources for the given label.

    save_trajectory stores files flat (kernel.py / config.toml / results.json
    directly under the run dir), not in a `solution/` subdir. Filter candidates
    to those with at least one kernel source file so noise-floor dirs (created
    by --variance-check, which have only noise-floor.json) are skipped rather
    than picked-then-rejected.
    """
    sanitized = _sanitize_label(label)
    safe_label_underscored = sanitized.replace(" ", "_")
    trajectory_root = PROJECT_ROOT / "trajectory"
    candidates: list[Path] = []
    if trajectory_root.exists():
        # Match against the sanitized stored form (post-B1 fix sanitizes at save
        # time too), the underscored variant, and the raw label for old dirs.
        candidates = [
            p for p in trajectory_root.iterdir()
            if p.is_dir() and (
                sanitized in p.name
                or safe_label_underscored in p.name
                or label in p.name
            )
        ]
    # Narrow to ones that actually have kernel sources. Keeps --variance-check
    # dirs (noise-floor.json only) from being picked as A/B candidates.
    with_sources = [p for p in candidates if _snapshot_has_sources(p)]
    if with_sources:
        # Latest by mtime among valid candidates.
        return max(with_sources, key=lambda p: p.stat().st_mtime)
    # Fall back to docs/prior/variants/ — the round-N spawn payload places the
    # parent anchor's kernel there (kernel.py + config.toml), and trajectory/
    # is empty on a fresh spawn before the first labeled bench. Round-2+
    # workflows want `--ab-compare <parent-anchor>` to work on spawn without
    # a manual `mkdir + cp` shim.
    prior_root = PROJECT_ROOT / "docs" / "prior" / "variants"
    if prior_root.exists():
        prior_candidates = [
            p for p in prior_root.iterdir()
            if p.is_dir()
            and (sanitized in p.name or safe_label_underscored in p.name or label in p.name)
            and _snapshot_has_sources(p)
        ]
        if prior_candidates:
            return max(prior_candidates, key=lambda p: p.stat().st_mtime)
    if not trajectory_root.exists() and not (prior_root.exists() and any(prior_root.iterdir())):
        print(f"ERROR: neither trajectory/ nor docs/prior/variants/ exists; "
              f"nothing to compare against.", file=sys.stderr)
    elif not candidates:
        print(f"ERROR: no trajectory dir or docs/prior/variants/ entry matching "
              f"'{label}'. Run `ls trajectory/` and `ls docs/prior/variants/` to "
              f"see available snapshots.", file=sys.stderr)
    else:
        print(f"ERROR: trajectory dir(s) matching '{label}' have no kernel "
              f"sources (likely --variance-check snapshots) and no "
              f"docs/prior/variants/<label>/ fallback. Use a label from a "
              f"standard labeled bench run.", file=sys.stderr)
    sys.exit(1)


def _pack_snapshot(snapshot_dir: Path, base_meta: dict) -> str:
    """Pack a trajectory snapshot's kernel sources as a solution-blob (agnostic staging).

    Reuses base_meta {name, definition, author}; the build spec comes from the
    snapshot's own config.toml if present, else the project config.toml. Only source
    files are staged (the snapshot stores config.toml/results.json flat alongside them).
    """
    config_path = snapshot_dir / "config.toml"
    src = config_path if config_path.exists() else (PROJECT_ROOT / "config.toml")
    with open(src, "rb") as f:
        build = tomllib.load(f)["build"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        staged = Path(tmp_dir)
        for f in snapshot_dir.iterdir():
            if f.is_file() and f.suffix.lower() in _SNAPSHOT_SOURCE_EXTS:
                shutil.copy2(f, staged / f.name)
        return adapter.pack(
            str(staged),
            {"language": build["language"], "entry_point": build["entry_point"],
             "destination_passing_style": build.get("destination_passing_style", False)},
            name=f"{base_meta['name']}__snapshot_{snapshot_dir.name}",
            definition=base_meta["definition"],
            author=base_meta["author"],
        )


def run_ab_compare(
    solution_blob: str,
    run_fn: Callable[[str, list, dict], dict],
    label: str,
    *,
    backend: str = "local",
    quiet: bool = False,
    current_label: str = None,
    max_workloads: int = 0,
    group_values: list | None = None,
    exclude_group_values: list | None = None,
    workload_indices: list | None = None,
    smoke: bool = False,
    extremes: bool = False,
):
    """A/B compare current solution vs a labeled trajectory snapshot.

    Both runs execute back-to-back in the same process / Modal container so
    cross-session drift (thermal, tenant load, cold-start jitter — ±1x on
    Modal B200) nearly cancels. Output is both scores + per-group delta +
    top-5 per-workload movers.

    This is the recommended tool for iterating at sub-1x deltas where
    `--variance-check N` runs far enough apart that drift swamps signal.

    If `current_label` is set, the B-side (current) run is saved as a
    labeled trajectory snapshot at the end so the next iter can use it
    as the `--ab-compare` reference (chained AB without re-grounding).
    """
    import statistics

    snapshot_dir = _find_trajectory_snapshot(label)
    print(f"\n*** A/B compare: current vs {snapshot_dir.name} ***")

    meta = adapter.solution_meta(solution_blob)
    uuids = select_workload_uuids(
        meta["definition"],
        max_workloads=max_workloads, group_values=group_values,
        exclude_group_values=exclude_group_values, workload_indices=workload_indices,
        smoke=smoke, extremes=extremes,
    )

    # Ensure baseline exists (reuse the logic from run_variance_check).
    baseline = load_baseline()
    if baseline is None:
        expert_blob = load_expert_blob()
        if expert_blob is not None:
            if not quiet:
                print(f"\nProfiling expert baseline "
                      f"({adapter.solution_meta(expert_blob)['name']}) — one-time cost...")
            baseline_params = _expert_baseline_params()
            baseline_results = run_fn(expert_blob, uuids, baseline_params)
        else:
            if not quiet:
                print("\nProfiling reference baseline — one-time cost...")
            ref_blob = _pack_reference()
            baseline_params = _baseline_params()
            baseline_results = run_fn(ref_blob, uuids, baseline_params)
        if baseline_results:
            save_baseline(baseline_results, baseline_params)
            baseline = load_baseline()
        if baseline is None:
            print("ERROR: Failed to establish baseline; cannot compute speedups.",
                  file=sys.stderr)
            sys.exit(1)

    snapshot_blob = _pack_snapshot(snapshot_dir, meta)

    # Run side B (current) then side A (snapshot) in the same process so drift cancels.
    print("\n  [B=current] running benchmark...")
    results_b = run_fn(solution_blob, uuids, _config_params(SOLUTION_ITERATIONS))
    if not results_b:
        print("ERROR: current solution returned no results.", file=sys.stderr)
        sys.exit(1)
    results_b = inject_baseline(results_b, baseline)
    score_b = compute_score(results_b)

    print(f"\n  [A={label}] running benchmark...")
    results_a = run_fn(snapshot_blob, uuids, _config_params(SOLUTION_ITERATIONS))
    if not results_a:
        print("ERROR: snapshot solution returned no results.", file=sys.stderr)
        sys.exit(1)
    results_a = inject_baseline(results_a, baseline)
    score_a = compute_score(results_a)

    # Print side-by-side delta summary.
    print()
    print("\u2550" * 68)
    print(f"  A/B compare (same-session, drift cancels)")
    print("\u2550" * 68)
    sa = score_a.get("final_score")
    sb = score_b.get("final_score")
    if sa is None or sb is None:
        print(f"  A={label}: final_score={sa}")
        print(f"  B=current: final_score={sb}")
        print("  One side INVALID — can't compute delta.")
        return
    print(f"  A ({label:<20}): {_fmt_speedup(sa):>9}")
    print(f"  B (current           ): {_fmt_speedup(sb):>9}")
    delta = sb - sa
    sign = "+" if delta >= 0 else ""
    print(f"  \u0394 (B - A)            : {sign}{_fmt_speedup(delta)} "
          f"({sign}{100*delta/sa:>5.2f}%)")

    # Per-group delta
    ga = score_a.get("group_scores") or {}
    gb = score_b.get("group_scores") or {}
    group_axis = score_a.get("group_axis") or score_b.get("group_axis") or "group"
    if ga and gb:
        print()
        print(f"  Per-{group_axis} delta:")
        for g in sorted(set(ga) | set(gb),
                        key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))):
            a = ga.get(g, {}).get("speedup")
            b = gb.get(g, {}).get("speedup")
            if a is not None and b is not None:
                d = b - a
                s = "+" if d >= 0 else ""
                print(f"    {group_axis}={g}: A={_fmt_speedup(a)}  B={_fmt_speedup(b)}  "
                      f"\u0394={s}{_fmt_speedup(d)}")

    # Top-5 per-workload movers (by abs delta)
    workload_deltas = []
    for def_name, traces in results_b.items():
        a_traces = results_a.get(def_name, {})
        for uuid, rb in traces.items():
            ra = a_traces.get(uuid, {})
            sa_w = ra.get("speedup_factor")
            sb_w = rb.get("speedup_factor")
            if sa_w is None or sb_w is None:
                continue
            workload_deltas.append({
                "uuid": uuid, "a": sa_w, "b": sb_w, "delta": sb_w - sa_w,
            })
    workload_deltas.sort(key=lambda x: abs(x["delta"]), reverse=True)
    if workload_deltas:
        print()
        print(f"  Top 5 per-workload movers (by |\u0394|):")
        for w in workload_deltas[:5]:
            s = "+" if w["delta"] >= 0 else ""
            print(f"    {w['uuid'][:8]}...: A={_fmt_speedup(w['a'])}  "
                  f"B={_fmt_speedup(w['b'])}  \u0394={s}{_fmt_speedup(w['delta'])}")
    print("\u2550" * 68)

    # Save the B-side (current) run as a labeled trajectory snapshot so the
    # caller can chain `--ab-compare <prior> --label <new>` from iter N+1.
    if current_label:
        save_trajectory(results_b, meta, score_b, current_label,
                        baseline_cached=True, backend=backend)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def find_group_axis() -> str:
    """Find the first variable axis from the definition, for grouping results."""
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
                # Reject inf as well as NaN: a silent-skip cascade (sol_ms~0
                # surviving the upstream guard) produces inf speedup, which
                # would otherwise propagate to final_score=inf and obscure
                # any honestly-passed workloads in the same run.
                if math.isnan(sf) or math.isinf(sf):
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


def _fmt_speedup(s: float) -> str:
    """Format a speedup factor so very small or very large values stay legible.

    `{:.2f}x` truncates 0.0021 to "0.00x" (looks like an error) and 9876.54
    to "9876.54x" (too many digits). Use 3 significant figures with a
    scientific fallback outside [1e-2, 1e4].
    """
    if s is None:
        return "?"
    if s == 0 or (1e-2 <= abs(s) < 1e4):
        if abs(s) < 10:
            return f"{s:.3g}x"
        return f"{s:.2f}x"
    return f"{s:.2e}x"


def print_results(
    results: dict,
    score: dict = None,
    quiet: bool = False,
    *,
    is_full_bench: bool = True,
    label: str | None = None,
    baseline_info: dict | None = None,
):
    """Print benchmark results in a formatted way.

    New keyword args (all optional, defaults preserve legacy behavior):
        is_full_bench: When False, score box uses "Subset score" header and
            appends the "subset is not a verdict" warning. When True + label
            is set, appends a session-drift calibration warning.
        label: The --label string; gates the drift warning (labeled bench =
            iteration decision moment).
        baseline_info: {source, name} dict from _baseline_display_info(). If
            provided, a "Baseline: ..." line is prefixed above FINAL SCORE.
    """
    error_logs_shown = 0
    max_error_logs = 3
    suggestions_shown = 0
    max_suggestions = 3

    # Map UUID -> position in unfiltered workloads.jsonl, for use in --index suggestions.
    # The same index the user would pass on a follow-up `--index N` command.
    try:
        uuid_to_full_idx = {w["workload"]["uuid"]: i for i, w in enumerate(_load_workloads())}
    except Exception:
        uuid_to_full_idx = {}

    if not quiet:
        for def_name, traces in results.items():
            print(f"\n{def_name}:")
            for workload_uuid, result in traces.items():
                status = result.get("status")
                print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

                if result.get("latency_ms") is not None:
                    print(f" | {result['latency_ms']:.3f} ms", end="")

                if result.get("speedup_factor") is not None:
                    print(f" | {_fmt_speedup(result['speedup_factor'])} speedup", end="")

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

                # Per-status next-step suggestion (cap at max_suggestions to avoid drowning box).
                # Use UNFILTERED workload index so --index N actually re-targets the same workload.
                if status and status != "PASSED" and suggestions_shown < max_suggestions:
                    full_idx = uuid_to_full_idx.get(workload_uuid)
                    if full_idx is not None:
                        suggestion = _suggest_next_step(status, full_idx)
                        if suggestion:
                            print(suggestion)
                            suggestions_shown += 1

    if score:
        print()
        print("═" * 50)
        if baseline_info:
            src = baseline_info.get("source", "reference")
            if src == "expert":
                name = baseline_info.get("name") or "unknown"
                print(f"  Baseline: expert ({name}) ← beat this for >1.0x")
            else:
                print(f"  Baseline: python-reference")
        if score["final_score"] is not None:
            header = "FINAL SCORE (mean speedup)" if is_full_bench else "Subset score"
            print(f"  {header}: {_fmt_speedup(score['final_score'])}")
            print(f"  Passed: {score['passed']}/{score['total']} | Min: {_fmt_speedup(score['min_speedup'])} | Max: {_fmt_speedup(score['max_speedup'])}")
            if score.get("group_scores"):
                group_axis = score.get("group_axis", "group")
                parts = []
                for g, d in score["group_scores"].items():
                    s = f"{g}→{_fmt_speedup(d['speedup'])}"
                    if d.get("latency_ms") is not None:
                        s += f"({d['latency_ms']:.3f}ms)"
                    parts.append(s)
                print(f"  By {group_axis}:  {'  '.join(parts)}")

            # Calibration warnings: drift vs subset-not-verdict, mutually exclusive.
            if not is_full_bench:
                print()
                print("  ⚠ Subset scores do NOT predict full-bench scores.")
                print("    • Use subset runs for CORRECTNESS checks (compile, PASSED count, shape errors).")
                print("    • For performance verdicts, always run full bench.")
                print("    Subset↔full-bench divergence can exceed 2x depending on which workloads")
                print("    are in/out of the subset. Low subset score does not mean the approach")
                print("    fails; high subset score does not mean it works.")
            elif label:
                print()
                print("  ⚠ Session drift: full-bench scores drift ±5-15% across Modal runs")
                print("    for identical code (~80µs fixed per-call overhead, amplified on small seq).")
                print("    For moves < ~5%, use `--ab-compare <label>` to cancel drift.")
                print("    Run `--variance-check 3` to measure this session's noise floor.")
        elif score["passed"] == 0 and score["failed"] == 0 and score["error"] == 0:
            print(f"  FINAL SCORE: HARNESS ANOMALY (empty results; Passed: 0/0)")
            print(f"  Likely a Modal session failure (e.g. cudaErrorDevicesUnavailable on the")
            print(f"  reference baseline) — retry with a fresh `bash scripts/bench.sh` invocation;")
            print(f"  this is not a kernel signal. Same shape as the --variance-check HARNESS")
            print(f"  ANOMALY diagnostic at scripts/bench_utils.py::run_variance_check.")
        else:
            parts = []
            if score["failed"]:
                parts.append(f"{score['failed']} workloads FAILED")
            if score["error"]:
                parts.append(f"{score['error']} ERROR")
            print(f"  FINAL SCORE: INVALID ({', '.join(parts)})")
            print(f"  All {score['total']} workloads must PASS for a valid score.")
            print(f"  Passed: {score['passed']}/{score['total']}")
        print("═" * 50)


def _suggest_next_step(status: str, idx: int) -> str | None:
    """One-line next-step hint printed under a non-PASSED workload result.

    idx is the workload's position in the filtered result list (what the user
    would pass as `--index N` to target it again).
    """
    if status == "INCORRECT_NUMERICAL":
        return f"    → bash scripts/sanitize.sh --index {idx} --tool memcheck   # OOB / race before rollback"
    if status == "RUNTIME_ERROR":
        return f"    → bash scripts/sanitize.sh --index {idx}   # compute-sanitizer can localize"
    if status == "TIMEOUT":
        return f"    → bash scripts/bench.sh --first 1 --index {idx}   # confirm deterministic; then consider raising timeout_seconds"
    if status == "COMPILE_ERROR":
        return f"    → see the language skill for your DSL for compile-error patterns; TRITON_PRINT_AUTOTUNING=1 reveals config-specific failures"
    return None


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

def _sanitize_label(label: str) -> str:
    """Strip chars from `--label` values that would split path components or
    break glob matching. Slashes in particular used to silently create nested
    trajectory dirs (e.g. `20260420_iter-1 … K/scale …` became two dirs), then
    `--ab-compare` matched the outer one and failed for lack of kernel sources
    (observed in v6 indexer session 2026-04-20). Use this anywhere a label is
    embedded in a filesystem path or compared by substring."""
    return (label.replace("/", "_").replace("\\", "_")
                 .replace("\n", " ").replace("\r", " ")
                 .strip())


def save_trajectory(
    results: dict,
    meta: dict,
    score: dict = None,
    label: str = None,
    *,
    baseline_cached: bool = False,
    backend: str = "local",
):
    """Save kernel and results to trajectory folder. `meta` = {name, definition, ...}."""
    trajectory_dir = PROJECT_ROOT / "trajectory"
    trajectory_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if label:
        # Sanitize via the shared `_sanitize_label()` helper (handles `/`, `\`,
        # and newlines — `/` in particular used to silently create nested dirs,
        # see v6 indexer session). Also replace spaces with `_` so the dir is
        # shell-safe and matches the `--ab-compare` glob sanitization applied
        # in `_find_trajectory_snapshot`.
        folder_name = f"{timestamp}_{_sanitize_label(label).replace(' ', '_')}"
    else:
        folder_name = timestamp

    run_dir = trajectory_dir / folder_name
    # %Y%m%d_%H%M%S resolution is 1 second, so two unlabeled saves (or two
    # same-labeled saves) within the same second would collide. Append a
    # numeric suffix on collision so the second save lands cleanly instead of
    # crashing with FileExistsError mid-run after results are already computed.
    suffix = 1
    while run_dir.exists():
        run_dir = trajectory_dir / f"{folder_name}__{suffix}"
        suffix += 1
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
        "solution_name": meta["name"],
        "definition": meta["definition"],
        "baseline_cached": baseline_cached,
        "baseline_iterations": BASELINE_ITERATIONS,
        "solution_iterations": SOLUTION_ITERATIONS,
        "score": score,
        "results": results,
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(trajectory_data, f, indent=2)

    print(f"\nTrajectory saved to: {run_dir}")


# ---------------------------------------------------------------------------
# Sanitizer output post-processing
# ---------------------------------------------------------------------------

def tail_truncate_output(output: str, max_lines: int) -> str:
    """Tail-truncate NCU output so metrics at the end are preserved.

    `flashinfer_bench_run_ncu(..., max_lines=N)` head-truncates, which drops
    the metrics tables (they come after the "==PROF== Profiling ..." progress
    lines). Use this instead: keep the LAST `max_lines` lines and prepend a
    one-line note so callers know what was dropped.
    """
    if max_lines is None or max_lines <= 0:
        return output
    lines = output.split("\n")
    if len(lines) <= max_lines:
        return output
    dropped = len(lines) - max_lines
    note = f"[...{dropped} earlier lines truncated; call without --max-lines to see full output...]"
    return note + "\n" + "\n".join(lines[-max_lines:])


def summarize_sanitizer_noise(output: str) -> str:
    """Prepend a banner explaining benign cuGetProcAddress_v2 library-init noise.

    compute-sanitizer logs every CUDA driver symbol probe done by PyTorch /
    triton / cupti at import time. These are not kernel errors and not
    caused by the solution.
    """
    import re
    noise_count = len(re.findall(
        r"Program hit CUDA_ERROR_INVALID_VALUE.*?cuGetProcAddress_v2",
        output,
    ))
    m = re.search(r"ERROR SUMMARY: (\d+) errors?", output)
    total = int(m.group(1)) if m else None

    if noise_count == 0:
        return output
    if total is not None:
        real = max(total - noise_count, 0)
        real_str = f"{real} apparently-real kernel hit{'s' if real != 1 else ''}"
    else:
        real_str = "unknown remaining hits (no ERROR SUMMARY parsed)"
    banner = (
        "\n>>> NOISE FILTER <<<\n"
        f"    {noise_count} of the reported hits are benign `cuGetProcAddress_v2` probes\n"
        "    from Python library init (PyTorch / triton / cupti probing driver\n"
        "    entry points). They are NOT caused by your kernel.\n"
        f"    Likely real signal: {real_str}.\n"
        "    See the sanitizer skill → 'Benign library-init noise'.\n\n"
    )
    return banner + output

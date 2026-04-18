# Benchmarking Reference

## How Scoring Works

**Score = arithmetic mean of speedup factors across all workloads.** Speedup = reference_latency / your_latency.

The score is a **tracking metric**, not an optimization target. Because the reference is a pure Python/PyTorch implementation, absolute speedup values can range from 300x to 9000x+ depending on workload size — this is expected and does not indicate a problem. What matters is that your kernel's **latency** decreases across iterations.

The score is still useful for tracking progress between runs because the reference latency is cached (see below), so speedup changes reflect real solution improvements.

## Input Behavior

Workload inputs are regenerated on each `run()` call — tensors passed to your kernel are fresh random data every invocation. This means:

- Do **not** cache intermediate results across calls (e.g., pre-computed indices, transposed weights, routing tables).
- Module-level state that depends on input **values** will be stale on the next call.
- You **may** cache structures that depend only on **shapes** (e.g., grid dimensions, block size selection, pre-allocated output buffers).

## Baseline Cache

The first benchmark run profiles the reference implementation and saves results to `baseline.json`. Subsequent runs skip reference profiling and reuse the cache.

- The cache is **automatically invalidated** when the workload set changes (UUIDs in `docs/workloads.jsonl` differ from cached ones).
- To force re-profiling: `bash scripts/bench.sh --force-baseline`
- Do not increase `baseline_iterations` to get a "better" baseline — the reference timing is approximate by design.
- If `baseline.json` already exists when the environment is created (pre-cached by `spawn.py`), baseline profiling is skipped entirely.
- If the solution is still the unmodified reference and a baseline is cached, the benchmark is skipped with a message to modify the kernel first.
- **Cross-session score drift (~0.5–2x)**: The baseline is cached per-env as a fixed latency per workload, but the *absolute* numbers a reference run produces shift 0.5–2% across sessions due to Modal tenant load, driver/image refreshes, and GPU thermal state. As a result, the same unchanged kernel can score 32.25x in one session and 32.67x in another — entirely from baseline drift. When comparing two kernel variants across sessions (especially across different child envs spawned at different times), **re-measure both variants against a single fresh `baseline.json`** (either drop the file and re-profile, or run both variants in the same env back-to-back). The noise floor from `--variance-check` captures within-session noise (typically 0.1–0.5% CV); drift is an across-session effect on top of that.

## Iteration Tuning

The first benchmark run profiles the reference implementation, which can be very slow (pure Python on large workloads). Always use a timeout, especially on the first run:

```bash
timeout 600 bash scripts/bench.sh    # 10-minute timeout
```

Iteration counts are configured in the `[benchmark]` section of `config.toml`:

```toml
[benchmark]
baseline_iterations = 5    # reference profiling iterations (keep low)
solution_iterations = 100  # solution profiling iterations
num_trials = 5             # trials per workload
warmup_runs = 3            # warmup before measurement
timeout_seconds = 300      # per-workload evaluation timeout
use_isolated_runner = true # run each workload in an isolated process
# Correctness tolerances (set per operator type, don't change unless you know why):
# atol = 0.01
# rtol = 0.01
# required_matched_ratio = 0.9
```

- **`baseline_iterations`** is intentionally low (default 5). The cached baseline only needs to be approximately correct. Baseline profiling automatically uses a lightweight configuration (warmup=1, trials=1) to minimize overhead.
- **`solution_iterations`** can stay at 100. Optimized kernels launch few CUDA ops, so profiling is fast.
- **Trim workloads** if benchmarking is still too slow: remove redundant entries from `docs/workloads.jsonl`. Workloads within the same group (same parameter values) are interchangeable — keeping fewer per group speeds up benchmarking without losing coverage.

## Workload Filtering

Use filters to run a subset of workloads for fast iteration. All filter modes skip baseline caching — partial results are not saved to `baseline.json`. Filters compose in order: `--index` → `--group` → `--exclude-group` → `--smoke` → `--first`.

> ⚠️ **Smoke-test noise warning.** Running on ≤3 workloads typically shows **2-3x variance** vs the full-bench mean due to cold-cache / Triton JIT / Modal jitter. Use small-workload filters for **compile/correctness checks only** — do **not** use them to decide whether a code change is a signal or regression. For that, run the full bench or use `--variance-check N`. The bench prints an explicit warning at the end of runs with ≤3 workloads; heed it.

### `--first N` — Quick test

```bash
bash scripts/bench.sh --first 1 --label "debug"   # single workload, fast compile check
bash scripts/bench.sh --first 3                     # first 3 workloads
```

Useful for quickly validating that your kernel compiles and produces correct output.

### `--smoke` — One workload per group bucket

```bash
bash scripts/bench.sh --smoke                       # e.g. 5 workloads for group_axis={1,2,6,7,8}
bash scripts/bench.sh --smoke --exclude-group 1     # skip one bucket
```

Picks the first workload for each distinct value of the operator's `group_axis` (see `docs/definition.json` → `variable_axis`). For an operator with group axis `num_tokens` and values `{1, 2, 6, 7, 8}`, `--smoke` runs 5 workloads total — one per bucket — giving correctness coverage across all shapes at roughly `5/23` of full-bench cost. If the operator has no variable axis, `--smoke` falls back to the first workload only (same as `--first 1`). Composes with other filters; applied after `--group`/`--exclude-group`.

### `--group VALUES` — Filter by group axis

```bash
bash scripts/bench.sh --group 8                    # only workloads where group axis = 8
bash scripts/bench.sh --group 8,16                 # group axis = 8 or 16
bash scripts/bench.sh --group 32-901               # group axis in range [32, 901]
bash scripts/bench.sh --group 8,16,128-512         # mix of exact values and ranges
bash scripts/bench.sh --group 8 --first 2          # first 2 workloads from group 8
```

Filters workloads by the variable axis (the axis that varies across workload groups, e.g. `num_experts`, `seq_len`). Use this to target specific workload sizes without running the full suite.

### `--exclude-group VALUES` — Exclude by group axis

```bash
bash scripts/bench.sh --exclude-group 1,14107      # skip outlier workloads
bash scripts/bench.sh --exclude-group 1-16          # skip small workloads
```

Excludes workloads matching the specified group axis values. Useful for ignoring edge-case workloads (e.g., very small or very large inputs) while benchmarking the typical range. Supports the same comma-separated and range syntax as `--group`.

### `--index INDICES` — Select by workload index

```bash
bash scripts/bench.sh --index 0,3,5               # run workloads 0, 3, and 5
bash scripts/bench.sh --index 2-8                   # run workloads 2 through 8
```

Selects specific workloads by their index in `docs/workloads.jsonl` (same indices shown by `bash scripts/profile.sh --list`). Useful when you know exactly which workloads to test.

Trajectory is still saved for all filter modes, so you can inspect `results.json` for error details.

## Variance Check (`--variance-check N`)

```bash
bash scripts/bench.sh --variance-check 3   # run unchanged solution 3 times
```

Runs the **same unchanged solution N times** back-to-back to measure across-run noise (Modal cold-start, GPU thermal effects, neighbour load). Reports:

- **Overall**: mean / std / CV / range across the N runs.
- **Top 5 most variable workloads** (by CV), so you can identify outliers.
- **Per-group noise** (e.g., `batch_size=1: 46.28 ± 0.12x (CV 0.3%)`, …): lets you see whether noise is spread evenly or concentrated in one group — useful when deciding whether a group-specific speedup change is real.

Useful when you suspect run-to-run noise is large enough to confuse iteration decisions. Runs N times → costs ~N × bench. Result is saved to `trajectory/noise-floor.json` (fields: `scores`, `score_mean/std/cv`, `per_workload`, `per_group`, `group_axis`) so you can refer back to it when interpreting future score deltas.

## Error Diagnostics

When a workload fails (RUNTIME_ERROR, COMPILE_ERROR, INCORRECT_NUMERICAL, etc.), the benchmark now prints the error traceback inline (up to 20 lines for the first 3 failed workloads). The full error log is saved in `trajectory/*/results.json` under the `error_log` field for each failed workload.

This eliminates the need to guess why a kernel failed — check the traceback first.

## Trajectory Tracking

Each labeled run saves a snapshot to `trajectory/YYYYMMDD_HHMMSS_label/`:

- All files from `solution/`
- `config.toml`
- `results.json` with full benchmark data (per-workload latencies, speedups, scores, metadata)

Use `bash scripts/bench.sh --label "description"` to create labeled snapshots.

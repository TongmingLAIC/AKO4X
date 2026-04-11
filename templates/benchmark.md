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

Use filters to run a subset of workloads for fast iteration. All filter modes skip baseline caching — partial results are not saved to `baseline.json`. Filters compose in order: `--index` → `--group` → `--exclude-group` → `--first`.

### `--first N` — Quick test

```bash
bash scripts/bench.sh --first 1 --label "debug"   # single workload, fast compile check
bash scripts/bench.sh --first 3                     # first 3 workloads
```

Useful for quickly validating that your kernel compiles and produces correct output.

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

## Error Diagnostics

When a workload fails (RUNTIME_ERROR, COMPILE_ERROR, INCORRECT_NUMERICAL, etc.), the benchmark now prints the error traceback inline (up to 20 lines for the first 3 failed workloads). The full error log is saved in `trajectory/*/results.json` under the `error_log` field for each failed workload.

This eliminates the need to guess why a kernel failed — check the traceback first.

## Trajectory Tracking

Each labeled run saves a snapshot to `trajectory/YYYYMMDD_HHMMSS_label/`:

- All files from `solution/`
- `config.toml`
- `results.json` with full benchmark data (per-workload latencies, speedups, scores, metadata)

Use `bash scripts/bench.sh --label "description"` to create labeled snapshots.

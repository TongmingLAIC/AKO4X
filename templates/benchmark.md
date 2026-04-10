# Benchmarking Reference

## How Scoring Works

**Score = arithmetic mean of speedup factors across all workloads.** Speedup = reference_latency / your_latency.

The score is a **tracking metric**, not an optimization target. Because the reference is a pure Python/PyTorch implementation, absolute speedup values can range from 300x to 9000x+ depending on workload size — this is expected and does not indicate a problem. What matters is that your kernel's **latency** decreases across iterations.

The score is still useful for tracking progress between runs because the reference latency is cached (see below), so speedup changes reflect real solution improvements.

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

## Quick Test Mode

Use `--first N` to run only the first N workloads for fast iteration:

```bash
bash scripts/bench.sh --first 1 --label "debug"   # single workload, fast compile check
bash scripts/bench.sh --first 3                     # first 3 workloads
```

- Useful for quickly validating that your kernel compiles and produces correct output before running the full suite.
- **Does not cache baseline** — avoids overwriting a valid full baseline with partial results.
- Trajectory is still saved, so you can inspect `results.json` for error details.

## Error Diagnostics

When a workload fails (RUNTIME_ERROR, COMPILE_ERROR, INCORRECT_NUMERICAL, etc.), the benchmark now prints the error traceback inline (up to 20 lines for the first 3 failed workloads). The full error log is saved in `trajectory/*/results.json` under the `error_log` field for each failed workload.

This eliminates the need to guess why a kernel failed — check the traceback first.

## Trajectory Tracking

Each labeled run saves a snapshot to `trajectory/YYYYMMDD_HHMMSS_label/`:

- All files from `solution/`
- `config.toml`
- `results.json` with full benchmark data (per-workload latencies, speedups, scores, metadata)

Use `bash scripts/bench.sh --label "description"` to create labeled snapshots.

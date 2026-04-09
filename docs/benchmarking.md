# Benchmarking

Run benchmarks from the child environment:

```bash
bash scripts/bench.sh "label"
```

Output shows per-workload results and a final score:

```
operator_name:
  Workload 0c23b10...: PASSED | 0.492 ms | 4.17x speedup | abs_err=1.2e-03, rel_err=2.1e-04
  ...

══════════════════════════════════════════════════
  FINAL SCORE (mean speedup): 4.17x
  Passed: 23/23 | Min: 2.10x | Max: 8.34x
  By num_tokens:  1→4.17x  2→3.85x  ...
══════════════════════════════════════════════════
```

The `FINAL SCORE` (mean speedup across all workloads) is the primary metric. All workloads must PASS for a valid score.

## Baseline Caching

The first benchmark run profiles the reference implementation and caches the result to `baseline.json`. Subsequent runs skip reference profiling.

```bash
bash scripts/bench.sh --force-baseline    # Force re-profile reference
```

`baseline.json` auto-invalidates when workloads change.

**Global cache**: When a baseline is computed, it is also saved to the global cache at `baselines/<operator>/baseline.json` (relative to the AKO project root). When `spawn.py` creates a new child environment, it copies the cached baseline automatically — so most child environments never need to profile the reference at all.

**Scratch mode optimization**: If the solution has not been modified from the reference implementation:
- Without cached baseline: the solution is benchmarked once and its latency is saved as the baseline (skipping the separate reference profiling phase).
- With cached baseline: the benchmark is skipped entirely with a message to modify the kernel first.

## Trajectory Tracking

Each labeled run saves a snapshot to `trajectory/<timestamp_label>/`, containing all files from `solution/`, `config.toml`, and `results.json` with per-workload scores and metadata.

## Iteration Tuning

Benchmark iteration counts are configured in the `[benchmark]` section of `config.toml` in the child environment:

```toml
[benchmark]
baseline_iterations = 5     # reference profiling iterations (keep low)
solution_iterations = 100   # solution profiling iterations
num_trials = 5
warmup_runs = 3
baseline_cache = "/path/to/baselines"  # global baseline cache directory
timeout_seconds = 300       # per-workload evaluation timeout
use_isolated_runner = true  # run each workload in an isolated process
# Correctness tolerances (set per operator type at spawn time):
# atol = 0.01
# rtol = 0.01
# required_matched_ratio = 0.9
```

CUPTI-based profiling can be slow. To speed up iteration during development, you can reduce `solution_iterations`, or trim `docs/workloads.jsonl` to benchmark fewer workloads. Both the user and the agent can edit these values.

## Evaluation Parameters

Correctness tolerances and runner settings are configured per operator type. These are set automatically at spawn time from `templates/evaluation.toml` in the parent repo and written into `config.toml`.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `timeout_seconds` | Per-workload evaluation timeout (seconds) | `300` |
| `use_isolated_runner` | Run each workload in an isolated process | `true` |
| `atol` | Absolute tolerance for correctness | `0.01` |
| `rtol` | Relative tolerance for correctness | `0.01` |
| `required_matched_ratio` | Required ratio of elements within tolerance | evaluator default |

Some operators (e.g. MoE FP8) need relaxed tolerances due to numerical precision characteristics. These are pre-configured — you normally don't need to change them.

> **Note on baseline profiling**: Baseline profiling uses a lightweight configuration (warmup=1, trials=1) regardless of the `warmup_runs` and `num_trials` settings, since the reference latency only needs to be approximately correct. The cached baseline is consistent across runs, so speedup changes still reflect real solution improvements.

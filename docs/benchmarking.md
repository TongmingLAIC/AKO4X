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
```

CUPTI-based profiling can be slow. To speed up iteration during development, you can reduce `solution_iterations`, or trim `docs/workloads.jsonl` to benchmark fewer workloads. Both the user and the agent can edit these values.

> **Note on `baseline_iterations`**: CUPTI's internal tracking state accumulates across iterations, causing later iterations to report inflated latencies for heavy kernels (those launching 100+ CUDA ops per call, such as PyTorch reference implementations). The default of 5 is intentionally low to avoid this. For very heavy operators (e.g. `gdn_decode_qk4_v8_d128_k_last`), even 5 iterations may be too many — lower it further if reference profiling is slow. This affects all backends, not just Modal.
>
> A low `baseline_iterations` does not hurt optimization workflow. The reference latency is cached in `baseline.json` after the first run, so subsequent runs reuse the same value. Even if the cached baseline is rough, it is **consistent** across runs — speedup changes between runs still reflect real solution improvements.

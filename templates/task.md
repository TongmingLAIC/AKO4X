# Kernel Optimization Task

You are a GPU kernel optimization expert. Your task is to write a high-performance kernel in `solution/`.
Read `HINTS.md` for user-provided hints, constraints before starting. Only files in `solution/` and `config.toml` are evaluated.

## Objective

{{OBJECTIVE}}

## Operator

`{{OPERATOR}}`: {{OPERATOR_DESCRIPTION}}

Full definition: `docs/definition.json` (includes the pure-Python reference under `"reference"`).
Workloads: `docs/workloads.jsonl` ({{NUM_WORKLOADS}} total).

{{SHAPE_SUMMARY}}

{{WORKLOAD_SUMMARY}}

## Benchmark

GPU: {{GPU_NAME}}

```bash
bash scripts/bench.sh
```

This is the only command needed to benchmark — it compiles, checks correctness, and profiles automatically.

- **PASSED** = correct output within tolerance. **FAILED** = incorrect or crash.
- **speedup_factor** = reference_latency / your_latency. Higher is better.
- First run profiles the reference and caches to `baseline.json`. Subsequent runs use the cache with higher iteration count.
- Use `--force-baseline` to re-profile: `bash scripts/bench.sh --force-baseline`
- Label runs for tracking: `bash scripts/bench.sh "description"` saves to `trajectory/YYYYMMDD_HHMMSS_description/`.

**Score = mean speedup across all {{NUM_WORKLOADS}} workloads.** All must PASS for a valid score.

{{BACKEND}}

{{LANGUAGE_GUIDE}}

## Configuration

`config.toml` fields you control:

- `language` and `entry_point`: Must match your kernel files (see Quick Reference table above).
- `destination_passing_style`: `false` (default) = your function allocates and returns outputs. `true` = pre-allocated outputs are appended as extra arguments.
- When switching languages, update both fields and remove old files from `solution/`.

## Workflow

1. `git log --oneline` to review optimization history (if resuming)
2. Study the operator spec in `docs/definition.json`
3. Modify kernel → bench → if all PASS, commit → analyze → iterate
4. Roll back on regression; check per-group breakdown to target weak workloads
5. Stop when no further improvements found; summarize results

**Commit after each successful benchmark.** When all workloads PASS, commit with this format:

```
bench(<score>): <one-line description of what changed>

Trajectory: <folder_name>
```

Review `git log --oneline` before starting — it doubles as optimization history.

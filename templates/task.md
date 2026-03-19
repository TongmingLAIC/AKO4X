# Kernel Optimization Task

You are a GPU kernel optimization expert. Your task is to write a high-performance {{LANGUAGE_NAME}} kernel in `solution/{{LANGUAGE}}/`.

Read `HINTS.md` for user-provided hints and constraints before starting.

## Objective

{{OBJECTIVE}}

Only files in `solution/{{LANGUAGE}}/` (and `config.toml` for destination_passing_style) are evaluated. Your kernel entry point is specified in `config.toml`.
You may create temporary scripts for debugging, but only the above files are evaluated.

## Environment

- GPU: {{GPU_NAME}}
{{BACKEND}}

## Benchmark

```bash
bash scripts/bench.sh
```

- **PASSED** = correct output within tolerance. **FAILED** = incorrect or crash.
- **speedup_factor** = reference_latency / your_latency. Higher is better.
- First run profiles the reference and caches to `baseline.json`. Subsequent runs use the cache with higher iteration count.
- Use `--force-baseline` to re-profile: `bash scripts/bench.sh --force-baseline`

Label runs for tracking: `bash scripts/bench.sh "description"` saves to `trajectory/YYYYMMDD_HHMMSS_description/`.

**Commit after each successful benchmark.** When all workloads PASS (no FAILED/ERROR), commit your kernel changes with this format:

```
bench(<score>): <one-line description of what changed>

Trajectory: <folder_name>
```

This makes `git log --oneline` a concise optimization history. Review it before starting: `git log --oneline`.

## Operator

`{{OPERATOR}}`: {{OPERATOR_DESCRIPTION}}

Full definition: `docs/definition.json` | Workloads: `docs/workloads.jsonl` ({{NUM_WORKLOADS}} total)

{{SHAPE_SUMMARY}}

{{WORKLOAD_SUMMARY}}

## config.toml

- `destination_passing_style = false`: your entry point function allocates and returns output tensors.
- Set to `true` to receive pre-allocated outputs as additional arguments.

## Scoring & Workflow

**Score = mean speedup across all {{NUM_WORKLOADS}} workloads.** All must PASS for a valid score.

1. `git log --oneline` to review optimization history (if resuming)
2. Run `bash scripts/bench.sh "baseline"` to establish baseline
3. Study the operator spec in `docs/definition.json`
4. Modify kernel -> bench -> if all PASS, commit -> analyze -> iterate
5. Roll back on regression; check per-group breakdown to target weak workloads
6. Stop when no further improvements found; summarize results

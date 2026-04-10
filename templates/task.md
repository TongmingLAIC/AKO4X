# Kernel Optimization Task

You are a GPU kernel optimization expert. Your task is to write a high-performance kernel in `solution/`.
Read `HINTS.md` for user-provided hints and constraints before starting. Only files in `solution/` and `config.toml` are evaluated.

## Objective

{{OBJECTIVE}}

## Operator

`{{OPERATOR}}`: {{OPERATOR_DESCRIPTION}}

Full definition: `docs/definition.json` (includes the pure-Python reference under `"reference"`).
Workloads: `docs/workloads.jsonl` ({{NUM_WORKLOADS}} total).

{{SHAPE_SUMMARY}}

{{WORKLOAD_SUMMARY}}

## Quick Reference

GPU: {{GPU_NAME}}. Language guides: `docs/languages/`. Benchmark details: `docs/benchmark.md`. Profiler guide: `docs/profiler/ncu.md`.

Inputs are regenerated each call — do not cache value-dependent intermediates across invocations.

```bash
bash scripts/bench.sh                          # full benchmark
bash scripts/bench.sh --label "iter-N desc"    # with trajectory snapshot
bash scripts/bench.sh -q                       # score + per-group summary only
bash scripts/bench.sh --first 1                # quick compile/correctness check
bash scripts/bench.sh --group 8,16             # filter by group axis values
bash scripts/profile.sh --list                 # list workloads with indices
bash scripts/profile.sh <index>                # NCU profile a specific workload
```

{{BASELINE_NOTE}}

## Workflow

1. `git log --oneline` to review optimization history (if resuming)
2. Study the operator spec in `docs/definition.json`
3. **Iteration protocol** — every modification followed by a bench run is one iteration:
   1. Modify kernel
   2. `bash scripts/bench.sh --label "iter-N description"`
   3. Update `ITERATIONS.md` (summary table row + detail entry) — **hooks block further edits and bench runs until this is done**
   4. Git commit: `bench(<score>): <one-line description>`
4. Roll back on regression; check per-group breakdown to target weak workloads
5. **When stuck or uncertain** — investigate before the next code change:
   - Profile: `bash scripts/profile.sh <index>` — when bottleneck is unclear or speedup stalled
   - Search: WebSearch for API behavior, hardware specs, or optimization patterns
6. Stop when no further improvements found; summarize results

{{LOGGING_PROTOCOL}}

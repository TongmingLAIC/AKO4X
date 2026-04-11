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
bash scripts/bench.sh --group 32-901           # filter by group axis range
bash scripts/bench.sh --exclude-group 1,14107  # exclude specific group values
bash scripts/bench.sh --index 0,3,5            # run specific workloads by index
bash scripts/profile.sh --list                 # list workloads with indices
bash scripts/profile.sh --index 5              # NCU profile a specific workload
bash scripts/profile.sh --index 5 --launches   # quick kernel launch summary
bash scripts/profile.sh --index 0,3,5 --launches  # batch launch summary
```

{{BASELINE_NOTE}}

## Workflow

1. `git log --oneline` to review optimization history (if resuming)
2. Study the operator spec in `docs/definition.json`
3. **Iteration protocol** — every code change + benchmark is one iteration:
   1. State your bottleneck hypothesis and what you expect the benchmark to show
   2. Modify kernel
   3. `bash scripts/bench.sh --label "iter-N description"`
   4. Update `ITERATIONS.md` — **hooks block further edits and bench runs until this is done**
      - Architectural change → summary row + detail entry
      - Parameter tuning (block size, num_warps, etc.) → Parameter Sweeps row only
   5. Git commit: `bench(<score>): <one-line description>`
4. **When stuck or uncertain** — investigate before the next code change:
   - Profile: `bash scripts/profile.sh --index <N> --launches` for launch overview, `--index <N>` for deep NCU
   - Search: WebSearch for API behavior, hardware specs, or optimization patterns
5. Stop when no further improvements found; summarize results

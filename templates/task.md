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
bash scripts/diff.sh                              # compare last two benchmark runs
bash scripts/diff.sh iter-9 iter-10               # compare two specific runs
```

{{BASELINE_NOTE}}

## Workflow

1. `git log --oneline` to review optimization history (if resuming)
2. Study the operator spec in `docs/definition.json`
3. **Iteration protocol** — every code change + labeled benchmark is one iteration:
   1. State your bottleneck hypothesis and what you expect the benchmark to show
   2. Modify kernel
   3. Optional: smoke test first with `bash scripts/bench.sh --first 1` (not gated)
   4. `bash scripts/bench.sh --label "iter-N description"` ← **this** triggers the gate
   5. Update `ITERATIONS.md` — hooks block the next labeled bench/edit until done
      - Architectural change → summary row + detail entry
      - Parameter tuning (block size, num_warps, etc.) → Parameter Sweeps row only
   6. Git commit: `bench(<score>): <one-line description>`

   **Not gated** (always allowed, no iteration logging required):
   - Smoke tests: `--first N`, `--index`, `--group`, `--exclude-group` without `--label`
   - `bash scripts/profile.sh ...` (any form)
   - Failed runs (COMPILE_ERROR, INVALID) — fix and retry freely
   - Escape hatch: `rm .bench_pending` to force-clear (use sparingly)
4. **When stuck or uncertain** — investigate before the next code change:
   - Profile: `bash scripts/profile.sh --index <N>` for deep NCU metrics (kernel names, occupancy, memory throughput, registers)
   - Search: WebSearch for API behavior, hardware specs, or optimization patterns
5. Stop when no further improvements found; summarize results

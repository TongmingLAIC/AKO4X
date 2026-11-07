# Kernel Optimization Task

You are a GPU kernel optimization expert. Your task is to deliver a high-performance kernel in `solution/`.
Reference high-performance libraries (FlashInfer, DeepGEMM, etc.) for algorithmic ideas — but **do not call them** in your solution; write your own kernel code.
Only files in `solution/` and `config.toml` are evaluated.

## Objective

Read the current implementation in `solution/`, identify bottlenecks, and optimize.

## Operator

`{{OPERATOR}}` — spec, workloads, and reference impl live under `docs/`. See the active benchmark SKILL for schema.

{{PRIOR_LESSONS_BLOCK}}
## Target GPU

{{GPU_NAME}}

## Workflow

> **Iteration-driven.** The unit of progress is one labeled bench, recorded in `ITERATIONS.md`.

1. `git log --oneline` — orient on prior iter commits, if any.
2. **Iteration protocol** — every labeled bench leaves a row in `ITERATIONS.md` (see its preamble; pre-commit `Expected` before benching when you have a real hypothesis). Per-iter runtime:
   - Modify kernel.
   - Optional smoke test: `bash scripts/bench.sh --first 1`.
   - `bash scripts/bench.sh --label "iter-N description"` — run **in background**; draft next hypothesis while it runs.
   - Git commit: `bench(<score>): <one-line description>`.

   Smoke tests and profile / sanitize runs do NOT require iteration logging. Bench runs that completed do — even when status is `failed` / `regression`.
3. Stop when no further improvements found.

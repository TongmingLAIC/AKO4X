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

## Language Reference

| Language | Files in `solution/` | `config.toml` language | `config.toml` entry_point |
|----------|---------------------|----------------------|--------------------------|
| Python / PyTorch | `kernel.py` | `python` | `kernel.py::run` |
| Triton | `kernel.py` | `triton` | `kernel.py::run` |
| TileLang | `kernel.py` | `tilelang` | `kernel.py::run` |
| CuTe DSL | `kernel.py` | `python` | `kernel.py::run` |
| CUDA | `kernel.cu` + `binding.py` | `cuda` | `binding.py::kernel` |
| C++ | `kernel.cpp` + `binding.py` | `cpp` | `binding.py::kernel` |

When you choose a language, read the corresponding guide in `docs/languages/` for patterns, code examples, and constraints.

## Configuration

`config.toml` fields you control:

- `language` and `entry_point`: Must match your kernel files (see Language Reference above).
- `destination_passing_style`: `false` (default) = your function allocates and returns outputs. `true` = pre-allocated outputs are appended as extra arguments.
- When switching languages, update both fields and remove old files from `solution/`.

## Benchmark

GPU: {{GPU_NAME}}

```bash
bash scripts/bench.sh
```

Compiles, checks correctness, and measures latency automatically.

- **PASSED** = correct output within tolerance. **FAILED** / **RUNTIME_ERROR** / **COMPILE_ERROR** = incorrect or crash — the error traceback is printed inline and saved to `trajectory/*/results.json`.
- Focus on reducing **latency** across workloads. {{BASELINE_NOTE}}
- `bash scripts/bench.sh --label "description"` — saves snapshot to `trajectory/`.
- `bash scripts/bench.sh -q` — only prints score, per-group speedup, and latency.
- `bash scripts/bench.sh --first 1` — quick test: only run 1 workload for fast compile/correctness checks (does not cache baseline).
- `bash scripts/bench.sh --group 8` — only run workloads matching group axis value 8 (comma-separated for multiple: `--group 8,16`).
- `timeout 600 bash scripts/bench.sh` — use a timeout, especially on first run.
- Inputs are regenerated each call — do not cache value-dependent intermediates across invocations (see `docs/benchmark.md` § Input Behavior).

Read `docs/benchmark.md` for scoring, baseline caching, iteration tuning, and troubleshooting.

{{PROFILER_GUIDE}}

## Workflow

1. `git log --oneline` to review optimization history (if resuming)
2. Study the operator spec in `docs/definition.json`
3. **Iteration protocol** — every modification followed by a bench run is one iteration:
   1. Modify kernel
   2. `bash scripts/bench.sh --label "iter-N description"`
   3. Update `ITERATIONS.md` (summary table row + detail entry)
   4. Git commit: `bench(<score>): <one-line description>`
4. Roll back on regression; check per-group breakdown to target weak workloads
5. When speedup stalls or a bottleneck is unclear, profile the slowest workloads
6. Stop when no further improvements found; summarize results

{{LOGGING_PROTOCOL}}

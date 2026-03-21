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

- **PASSED** = correct output within tolerance. **FAILED** = incorrect or crash.
- All {{NUM_WORKLOADS}} workloads must PASS.
- Focus on reducing **latency** across workloads. Speedup over the reference is shown for tracking but is not your optimization target — the reference is a slow Python implementation whose timing is approximate.
- Label runs: `bash scripts/bench.sh --label "description"` saves to `trajectory/`.
- Always use a timeout: `timeout 600 bash scripts/bench.sh`. If it times out (especially the first run profiling the reference), lower `baseline_iterations` in `config.toml` or trim redundant workloads — see `docs/benchmark.md`.

Read `docs/benchmark.md` for baseline caching, scoring details, and measurement caveats.

{{BACKEND}}

{{PROFILER_GUIDE}}

## Workflow

1. `git log --oneline` to review optimization history (if resuming)
2. Study the operator spec in `docs/definition.json`
3. Modify kernel → bench → if all PASS, commit → analyze → iterate
4. Roll back on regression; check per-group breakdown to target weak workloads
5. When speedup stalls or a bottleneck is unclear, profile the slowest workloads
6. Stop when no further improvements found; summarize results

**Commit after each successful benchmark.** When all workloads PASS, commit with this format:

```
bench(<score>): <one-line description of what changed>

Trajectory: <folder_name>
```

# Child Environment

Each call to `spawn.py` creates an isolated child environment with everything the agent needs to optimize a kernel.

## Structure

```
ako4fib-run-<label or timestamp>/
├── CLAUDE.md                   # Auto-generated task spec (operator details + instructions)
├── HINTS.md                    # User-editable hints and constraints
├── config.toml                 # Operator configuration (agent updates language/entry_point)
├── .gitignore
├── docs/
│   ├── definition.json         # Operator definition
│   ├── workloads.jsonl         # Benchmark workloads
│   ├── languages/              # Per-language patterns and constraints
│   └── profiler/               # Profiler docs (local backend only)
├── solution/                   # Flat directory — agent puts kernel files here
│   └── kernel.py               # Starting point (reference implementation in scratch mode)
├── scripts/
│   ├── bench_utils.py          # Shared benchmark utilities
│   ├── bench.sh                # Benchmark script
│   ├── run_local.py            # (or run_modal.py) Benchmark runner
│   ├── profile.sh              # NCU profiler script (local only)
│   ├── run_profile.py          # NCU profiler runner (local only)
│   └── pack_solution.py        # Pack kernel for evaluation
├── baseline.json               # Auto-created, cached reference performance
├── trajectory/                 # Auto-created, each run saves kernel snapshot + results.json
└── .claude/
    └── settings.local.json     # Agent permissions
```

## config.toml

The agent updates these fields to match its kernel files:

- **`language`**: `python`, `triton`, `tilelang`, `cuda`, or `cpp`
- **`entry_point`**: e.g. `kernel.py::run` or `binding.py::kernel`
- **`destination_passing_style`**: `false` by default; set `true` when using pre-allocated output buffers

The `[benchmark]` section controls iteration counts:

- **`baseline_iterations`**: Reference profiling iterations (default: `100`)
- **`solution_iterations`**: Solution profiling iterations (default: `100`, higher for accuracy)
- **`num_trials`**: Number of trials per workload (default: `5`)
- **`warmup_runs`**: Warmup runs before timing (default: `3`)

## Benchmarking

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

### Baseline Caching

The first benchmark run profiles the reference implementation (20 iterations) and caches the result to `baseline.json`. Subsequent runs skip reference profiling and use higher iteration counts (100) for more accurate solution timing.

```bash
bash scripts/bench.sh "baseline"          # First run: profiles reference, caches baseline
bash scripts/bench.sh "opt_v1"            # Uses cached baseline, 100 iterations for solution
bash scripts/bench.sh --force-baseline    # Force re-profile reference
```

`baseline.json` auto-invalidates when workloads change.

### Trajectory

Each benchmark run automatically saves a snapshot to `trajectory/<timestamp_label>/`, containing:

- All files from `solution/` and `config.toml` at the time of the run
- `results.json` with per-workload scores, timestamps, and metadata

This allows both users and the agent to review or restore any previous attempt.

## Profiling (Local Backend)

NCU profiling is available via `scripts/profile.sh` in the child environment. See `docs/profiler/ncu.md` within the child environment for detailed usage.

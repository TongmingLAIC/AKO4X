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
│   ├── run_profile.py          # NCU profiler (local only)
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

- **`baseline_iterations`**: Reference profiling iterations (default: `5`)
- **`solution_iterations`**: Solution profiling iterations (default: `100`)
- **`num_trials`**: Number of trials per workload (default: `5`)
- **`warmup_runs`**: Warmup runs before timing (default: `3`)

For benchmarking commands, output interpretation, baseline caching, trajectory tracking, and iteration tuning, see [Usage Guide](usage.md#benchmarking).

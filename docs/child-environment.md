# Child Environment

Each call to `spawn.py` creates an isolated child environment with everything the agent needs to optimize a kernel.

## Workflow

1. `python spawn.py --operator <name>` discovers the operator definition and workloads from the dataset
2. Creates a child environment `ako4fib-run-NNN[-label]/` as a sibling directory
3. You start your code agent (e.g. `claude`) in the child environment
4. The agent reads `CLAUDE.md` (task spec) and `HINTS.md` (user guidance), then:
   - Studies the operator definition and workload distribution
   - Writes or modifies the kernel in `solution/`
   - Benchmarks with `scripts/bench.sh` and analyzes per-workload results
   - Iterates toward higher speedup

## Structure

```
ako4fib-run-NNN/
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
├── trajectory/                 # Auto-created, stores each bench run
└── .claude/
    └── settings.local.json     # Agent permissions
```

## config.toml

The agent updates these fields to match its kernel files:

- **`language`**: `python`, `triton`, `tilelang`, `cuda`, or `cpp`
- **`entry_point`**: e.g. `kernel.py::run` or `binding.py::kernel`
- **`destination_passing_style`**: `false` by default; set `true` when using pre-allocated output buffers

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

## Profiling (Local Backend)

NCU profiling is available via `scripts/profile.sh` in the child environment. See `docs/profiler/ncu.md` within the child environment for detailed usage.

## Profiler

Two modes: `--launches` for quick launch overview, NCU for deep per-kernel analysis.

```bash
bash scripts/profile.sh --list                    # list workloads with indices
bash scripts/profile.sh --index 5 --launches      # quick: kernel launch count + timing
bash scripts/profile.sh --index 0,3,5 --launches  # batch launch summary
bash scripts/profile.sh --index 5                 # deep: NCU detailed metrics
bash scripts/profile.sh --index 5 --page raw      # NCU machine-readable output
```

Start with `--launches` to see the big picture (how many kernels, which dominate), then NCU for hardware counters on the bottleneck kernel. Supports same filters as bench.sh: `--group`, `--exclude-group`, `--index` with range syntax.
Read `docs/profiler/ncu.md` for when to profile, metrics interpretation, and profiling workflows.

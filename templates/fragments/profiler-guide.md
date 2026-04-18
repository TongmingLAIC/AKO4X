## Profiler

NCU (Nsight Compute) deep per-kernel analysis.

```bash
bash scripts/profile.sh --list                    # list workloads with indices
bash scripts/profile.sh --index 5                 # NCU detailed metrics for workload 5
bash scripts/profile.sh --index 5 --page raw      # NCU machine-readable output
bash scripts/profile.sh --index 0,3,5             # batch profile several workloads
```

Supports same filters as bench.sh: `--group`, `--exclude-group`, `--index` with range syntax.
NCU output includes kernel names, occupancy, memory throughput, registers, warp stalls, SMEM usage.
Read `docs/profiler/ncu.md` for when to profile, metrics interpretation, and profiling workflows.

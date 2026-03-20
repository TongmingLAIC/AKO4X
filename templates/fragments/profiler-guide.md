## Profiler

Profile individual workloads with NVIDIA Nsight Compute (NCU) to identify bottlenecks:

```bash
bash scripts/profile.sh --list              # list workloads with indices
bash scripts/profile.sh <index>             # NCU profile (detailed metrics)
bash scripts/profile.sh <index> --page raw  # machine-readable output
```

Don't profile every iteration — reach for NCU when speedup stalls, a workload group is unexpectedly slow, or you need to confirm whether a kernel is memory-bound vs compute-bound.
Read `docs/profiler/ncu.md` for when to profile, metrics interpretation, and profiling workflows.

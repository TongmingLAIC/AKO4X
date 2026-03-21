## Profiler

Profile individual workloads with NVIDIA Nsight Compute (NCU) to identify bottlenecks:

```bash
python scripts/run_profile.py --list              # list workloads with indices
python scripts/run_profile.py <index>             # NCU profile (detailed metrics)
python scripts/run_profile.py <index> --page raw  # machine-readable output
```

Don't profile every iteration — reach for NCU when speedup stalls, a workload group is unexpectedly slow, or you need to confirm whether a kernel is memory-bound vs compute-bound.
Read `docs/profiler/ncu.md` for when to profile, metrics interpretation, and profiling workflows.

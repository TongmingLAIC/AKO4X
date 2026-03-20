## Profiler

Profile individual workloads with NVIDIA Nsight Compute (NCU) to identify bottlenecks:

```bash
bash scripts/profile.sh --list              # list workloads with indices
bash scripts/profile.sh <index>             # NCU profile (detailed metrics)
bash scripts/profile.sh <index> --page raw  # machine-readable output
```

Integrate profiling into your optimization loop: bench → profile bottleneck workloads → optimize → repeat.
Read `docs/profiler/ncu.md` for the full command reference, output interpretation, and profiling strategies.

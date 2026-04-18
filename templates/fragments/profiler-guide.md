## Profiler

`bash scripts/profile.sh` wraps NVIDIA Nsight Compute (`ncu`).

```bash
bash scripts/profile.sh --list
bash scripts/profile.sh --index 5                            # default ncu --set detailed
bash scripts/profile.sh --index 5 --set full --page raw      # more metrics, parseable format
bash scripts/profile.sh --index 5 --sections LaunchStats     # cherry-pick sections (use --ncu-options to list)
```

Same `--index` / `--group` / `--exclude-group` / range syntax as `bench.sh`. `--set`, `--page`, `--sections`, `--kernel-name` forward to `ncu` as-is. Output saves to `profiles/w<idx>_<ts>.json`. See `docs/profiler/ncu.md` for wrapper-specific behavior (CUDA-graph capture interaction, `--ncu-options` discovery).

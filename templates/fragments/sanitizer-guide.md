## Sanitizer

`bash scripts/sanitize.sh` wraps NVIDIA `compute-sanitizer`. Use for `INCORRECT_NUMERICAL` / `RUNTIME_ERROR` cases where the cause isn't visible from kernel source.

```bash
bash scripts/sanitize.sh --list
bash scripts/sanitize.sh --index 5                    # memcheck (default)
bash scripts/sanitize.sh --index 5 --tool initcheck   # racecheck / synccheck / all also valid
```

Same `--index` / `--group` / `--exclude-group` / range syntax as `bench.sh`. Output saves to `sanitizer/w<idx>_<tool>_<ts>.json`. See `docs/sanitizer/sanitizer.md` for wrapper-specific behavior (NOISE FILTER banner, `--tool all` sequential-timeout semantics, CUDA-graph interaction).

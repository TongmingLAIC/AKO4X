## Sanitizer

`compute-sanitizer` catches four classes of GPU correctness bugs. Use when `INCORRECT_NUMERICAL` or `RUNTIME_ERROR` cause is unclear.

```bash
bash scripts/sanitize.sh --list                         # list workloads with indices
bash scripts/sanitize.sh --index 5                      # memcheck (default: OOB, misaligned)
bash scripts/sanitize.sh --index 5 --tool initcheck     # uninitialized memory reads
bash scripts/sanitize.sh --index 5 --tool racecheck     # SMEM race conditions
bash scripts/sanitize.sh --index 5 --tool synccheck     # bad __syncthreads under divergence
bash scripts/sanitize.sh --index 5 --tool all           # run all four (slower)
```

Pick by failure class: garbage values → `initcheck`; near-miss values → `racecheck`; illegal-address → `memcheck`; hangs → `synccheck`; no hypothesis → `all`.
Supports same filters as bench.sh: `--group`, `--exclude-group`, `--index` with range syntax.
Read `docs/sanitizer/sanitizer.md` for when to pick which tool, reading output, and common gotchas.

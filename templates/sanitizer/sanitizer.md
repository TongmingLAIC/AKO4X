# Compute-Sanitizer

Wrapper around NVIDIA `compute-sanitizer`. Same workload-filter syntax as `bench.sh` / `profile.sh`.

## Commands

```bash
bash scripts/sanitize.sh --list
bash scripts/sanitize.sh --index 5                    # default: memcheck
bash scripts/sanitize.sh --index 5 --tool initcheck
bash scripts/sanitize.sh --index 5 --tool racecheck
bash scripts/sanitize.sh --index 5 --tool synccheck
bash scripts/sanitize.sh --index 5 --tool all         # run all four sequentially
bash scripts/sanitize.sh --index 0,3,5 --tool memcheck
bash scripts/sanitize.sh --index 0-18 --group 32-901
bash scripts/sanitize.sh --index 5 --timeout 900
bash scripts/sanitize.sh --index 5 --max-lines 500
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--index` | (required) | Workload indices (e.g. `5`, `0,3,5`, `2-8`) |
| `--list` | — | Print workload table with indices |
| `--tool` | `memcheck` | `memcheck`, `racecheck`, `initcheck`, `synccheck`, or `all` |
| `--group` | (all) | Filter by group axis values |
| `--exclude-group` | (none) | Exclude by group axis values |
| `--timeout` | `300` | Per-tool timeout in seconds |
| `--max-lines` | (unlimited) | Truncate output to N lines |

Output auto-saves to `sanitizer/w<index>_<tool>_<timestamp>.json`.

## Wrapper-specific behavior

- **`--tool all` is sequential, not parallel.** Each of the four tools runs under its own `--timeout`. If one hangs, the remaining tools won't run. Raise `--timeout` if you expect long runs.
- **Output prepended with a `>>> NOISE FILTER <<<` banner** if `cuGetProcAddress_v2` probes are detected (see Gotchas).
- **Bench error codes that most often benefit from sanitizer**: `INCORRECT_NUMERICAL`, `RUNTIME_ERROR` in `bench.sh` output. If `bench.sh` already shows a Python-side `AttributeError` or `ImportError`, sanitizer won't help — fix the Python path first.

## Gotchas

### Benign `cuGetProcAddress_v2` noise inflates ERROR SUMMARY

Almost any run will include 10–30 hits of:

```
========= Program hit CUDA_ERROR_INVALID_VALUE (error 1) due to
=========     "invalid argument" on CUDA API call to cuGetProcAddress_v2.
```

This is not a kernel bug. PyTorch / triton / cupti probe the CUDA driver for optional entry points at import time via `cuGetProcAddress_v2`; when a symbol isn't available the driver returns `INVALID_VALUE` (which the library silently handles), but compute-sanitizer logs every probe. The wrapper's `>>> NOISE FILTER <<<` banner counts these and subtracts them from `ERROR SUMMARY` to give you `apparently-real kernel hits = total − noise`. If the banner says `0 apparently-real kernel hits`, the run is effectively clean.

The distinguishing feature of noise blocks: they never name a kernel, thread, or address — only Python host frames. Real kernel errors always mention `kernel <name>`, `thread (x,y,z)`, `block (bx,by,bz)`, or a specific Address.

### CUDA-graph-captured kernels

Solutions that capture into `torch.cuda.CUDAGraph` launch via `cuGraphLaunch`, which surfaces to sanitizer but with less source info in stack traces. If the sanitizer report's stack traces aren't informative, temporarily disable graph capture (skip the `graph.replay()` path) to get the eager-mode report, fix, then re-enable.

### Output hang when piping to `head -N`

`bash scripts/sanitize.sh ... | head -N` can hang — `head` exits early, SIGPIPE is swallowed somewhere in the Modal/subprocess chain, upstream blocks writing to a full buffer. Use `| tail -N` (reads to EOF) or redirect to a file (`> out.txt`).

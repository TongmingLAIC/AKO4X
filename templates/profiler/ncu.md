# Profiler

Wrapper around NVIDIA Nsight Compute (`ncu`). Same workload-filter syntax as `bench.sh` / `sanitize.sh`.

## Commands

```bash
bash scripts/profile.sh --list
bash scripts/profile.sh --index 5                                 # default: --set detailed --page details
bash scripts/profile.sh --index 5 --set full                      # NCU preset (basic|detailed|full|…)
bash scripts/profile.sh --index 5 --page raw                      # ncu output format (raw|details|source)
bash scripts/profile.sh --index 5 --kernel-name "my_kernel"       # full-match regex; use .*name.* for substring
bash scripts/profile.sh --index 5 --sections LaunchStats,Occupancy # add specific sections on top of --set
bash scripts/profile.sh --index 5 --timeout 120
bash scripts/profile.sh --index 5 --max-lines 200
bash scripts/profile.sh --ncu-options                             # list sets + sections available on this ncu build
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--index` | (required) | Workload indices (e.g. `5`, `0,3,5`, `2-8`) |
| `--list` | — | Print workload table with indices |
| `--group` | (all) | Filter by group axis values |
| `--exclude-group` | (none) | Exclude by group axis values |
| `--ncu-options` | — | `ncu --list-sets` + `ncu --list-sections` |
| `--set` | `detailed` | ncu `--set` preset (forwarded as-is) |
| `--page` | `details` | ncu `--page` format (forwarded as-is) |
| `--kernel-name` | (all) | ncu `--kernel-name` regex (forwarded as-is) |
| `--sections` | (none) | Comma-list of section names added via ncu `--section` (forwarded as-is) |
| `--max-lines` | (unlimited) | Truncate output to N lines |
| `--timeout` | `60` | Timeout in seconds |

Output auto-saves to `profiles/w<index>_<timestamp>.json`.

## Discover → Select sections

`--set` picks an ncu preset bundle; `--sections` adds individual sections on top. To find names valid on the image that runs the profile:

```bash
bash scripts/profile.sh --ncu-options         # enumerate
bash scripts/profile.sh --index 5 --sections <name1>,<name2>,...
```

Section names (e.g. `LaunchStats`, `Occupancy`, `SchedulerStats`, `WarpStateStats`, `MemoryWorkloadAnalysis`, `SourceCounters`) vary by ncu version — always confirm via `--ncu-options` before relying on one.

## Wrapper-specific behavior

### CUDA graph capture interaction

If the solution uses `torch.cuda.CUDAGraph`, kernels launch via `cuGraphLaunch`, not `cuLaunchKernel`. Consequences for this wrapper:

- **`--kernel-name` filter may return "No kernels were profiled"** — graph-captured launches don't surface the same symbol info as direct launches, so ncu's kernel regex can miss them.
- **Full profile (no `--kernel-name`) still works** — ncu profiles every kernel the graph replays, each printed as a separate block headed with `void <kernel_name>(...)`.
- **Workflow for bottleneck attribution**: profile the eager path first (temporarily bypass `graph.replay()`) so `--kernel-name` filters work; re-enable graph capture to measure steady-state wall time via `bench.sh`.

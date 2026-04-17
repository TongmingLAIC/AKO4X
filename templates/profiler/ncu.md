# Profiler Reference

Two profiling modes: **`--launches`** for quick kernel launch overview, **NCU** for deep per-kernel analysis.

## When to Use Each Mode

| Mode | Use When | Speed |
|------|----------|-------|
| `--launches` | Launch overhead diagnosis, kernel count, timing distribution | Fast (single pass) |
| NCU (default) | Memory vs compute bottleneck, occupancy, warp stalls | Slow (kernel replay) |

**Start with `--launches`** to see the big picture (how many kernels, which ones dominate). Switch to NCU only when you need hardware-counter-level detail on a specific kernel.

**Skip profiling when:**

- The kernel isn't correct yet — fix FAILED workloads first.
- You haven't read the bench output yet — per-workload and per-group breakdowns often tell you what to try next without profiling.

## Kernel Launch Summary (`--launches`)

Uses `torch.profiler` to trace CUDA activity in a single pass. Shows kernel names, call counts, and CUDA timing — much faster than NCU.

```bash
# Launch summary for a single workload
bash scripts/profile.sh --index 5 --launches

# Batch: compare launch patterns across workloads
bash scripts/profile.sh --index 0,3,5 --launches
bash scripts/profile.sh --index 0-4 --launches

# Filter workloads by group axis
bash scripts/profile.sh --index 0-18 --launches --exclude-group 1,14107
```

**Reading the output:**
- **Total CUDA kernel launches** — high count (50+) suggests launch overhead; consider fusing kernels.
- **Per-kernel CUDA time** — sorted by total time. If one kernel dominates (>80%), focus optimization there. If many small kernels share time, launch overhead is the bottleneck.

## NCU Deep Profile (default)

NVIDIA Nsight Compute replays each kernel to collect hardware performance counters.

```bash
# NCU profile (detailed metrics, formatted output)
bash scripts/profile.sh --index 5

# Full metrics collection (slower, more data)
bash scripts/profile.sh --index 5 --set full

# Raw machine-readable output (for parsing)
bash scripts/profile.sh --index 5 --page raw

# Source-correlated counters (maps metrics to source lines)
bash scripts/profile.sh --index 5 --page source

# Filter by kernel name (full-match regex; use .*name.* for substring)
bash scripts/profile.sh --index 5 --kernel-name "my_kernel"

# Combine options
bash scripts/profile.sh --index 5 --set full --page raw --kernel-name ".*gemm.*"

# List available NCU sets and sections
bash scripts/profile.sh --ncu-options

# Increase timeout for complex kernels (default: 60s)
bash scripts/profile.sh --index 5 --timeout 120

# Limit output length
bash scripts/profile.sh --index 5 --max-lines 200
```

## Workload Selection

All profiling modes share the same workload filters (same syntax as `bench.sh`):

```bash
bash scripts/profile.sh --list                          # list all workloads
bash scripts/profile.sh --list --exclude-group 1,14107   # filtered list
bash scripts/profile.sh --index 5                       # single workload
bash scripts/profile.sh --index 0,3,5                   # multiple workloads
bash scripts/profile.sh --index 2-8                      # range
bash scripts/profile.sh --index 0-18 --group 32-901      # combined filters
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--index` | (required) | Workload indices (e.g. `5`, `0,3,5`, `2-8`) |
| `--list` | — | Print workload table with indices |
| `--launches` | — | Quick kernel launch summary (torch.profiler) |
| `--group` | (all) | Filter by group axis values (e.g. `8,16` or `32-901`) |
| `--exclude-group` | (none) | Exclude by group axis values |
| `--ncu-options` | — | List available NCU section sets and sections |
| `--set` | `detailed` | NCU section set: `basic`, `detailed`, `full`, etc. |
| `--page` | `details` | Output format: `details`, `raw`, or `source` |
| `--kernel-name` | (all) | Full-match regex to filter profiled kernel names (use `.*name.*` for substring) |
| `--sections` | (none) | Additional NCU sections beyond the set |
| `--max-lines` | (unlimited) | Truncate output to N lines |
| `--timeout` | `60` | Timeout in seconds |

## Understanding NCU Output

### Section Sets

- **basic**: Minimal metrics — launch config, duration, memory throughput. Fast.
- **detailed** (default): Key performance metrics — compute/memory utilization, occupancy, warp stalls. Good balance of speed and insight.
- **full**: All available metrics. Slow but exhaustive. Use when detailed doesn't explain the bottleneck.

### Output Pages

- **details** (default): Human-readable formatted output organized by section. Start here.
- **source**: Maps performance counters to source code lines. Use to find which lines cause stalls or low throughput.
- **raw**: Machine-readable output. Use when you need to parse specific metric values programmatically.

## Tips

- **Use `--kernel-name`**: When your solution launches multiple kernels, filter to the one of interest. The name usually matches your function/kernel name.
- **NCU adds overhead**: Profiled execution is much slower than normal. This is expected — NCU replays kernels to collect metrics.
- **One workload at a time**: Pick the most informative workload (usually the slowest from `bench.sh`).
- **`--page raw` for scripting**: When you need to extract a specific metric value, use raw output and parse the result.
- **Profiles are auto-saved**: Each run saves output and metadata to `profiles/w<index>_<timestamp>.json`. Compare two profiles by reading both files.

## CUDA Graphs and NCU

If your solution captures kernels into a `torch.cuda.CUDAGraph` (a common optimization to amortize kernel-launch dispatch), NCU's kernel filtering behaves differently. Symptoms and workaround:

- **`--kernel-name` may return "No kernels were profiled"** even when the kernel is clearly running — NCU identifies kernels by their launch site, and graph-captured kernels are launched via `cuGraphLaunch`, which doesn't surface the same symbol info as direct `cuLaunchKernel` calls.
- **Full-profile (no `--kernel-name`) still works** — NCU will just profile every kernel the graph replays, including the ones you care about. Parse the output by position or by the `void <kernel_name>(` line in the header.
- **Recommended workflow for graph-captured solutions**:
  1. Profile the *eager* path first — temporarily skip the graph cache (e.g., return before `graph.replay()` or comment out the capture) so each kernel launches individually via `cuLaunchKernel` and `--kernel-name` filters work normally.
  2. Once you've identified the bottleneck, re-enable the graph to measure steady-state wall time via `bench.sh`.
  3. Use `profile.sh --launches` (torch profiler, not NCU) if you only need a launch summary — it traces all activity including graph replay.

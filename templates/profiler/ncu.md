# NCU Profiler Reference

NVIDIA Nsight Compute (NCU) profiles individual GPU kernels to reveal performance bottlenecks.
Use it to guide optimization after identifying slow workloads with `bench.sh`.

## When to Profile

Don't profile routinely — profile when benchmark results alone don't explain performance. Get correctness first: all workloads must PASS before profiling is useful.

**Profile when:**

1. **Speedup has plateaued.** You made 2–3 optimization attempts with little or no improvement. Profiling reveals whether you are targeting the right bottleneck (memory vs compute vs launch overhead).
2. **A workload group is disproportionately slow.** The per-group breakdown from `bench.sh` shows one group lagging behind others. Profile a representative workload from that group to understand what makes it different.
3. **You need to choose between optimization strategies.** You suspect a kernel is memory-bound but aren't sure. Profiling confirms the bottleneck category so you invest effort in the right direction.
4. **After a major architectural change.** New tiling scheme, different algorithm, or language switch — profile to understand the new performance characteristics before micro-optimizing.
5. **Unexpected regression.** A change that should have improved performance made it worse. Profile before and after (see Workflow 4 below) to find what regressed.

**Skip profiling when:**

- The kernel isn't correct yet — fix FAILED workloads first.
- You haven't read the bench output yet — per-workload and per-group breakdowns often tell you what to try next without profiling.

## Command Reference

```bash
# List workloads with indices
bash scripts/profile.sh --list

# Profile a workload (detailed metrics, formatted output)
bash scripts/profile.sh <index>

# Full metrics collection (slower, more data)
bash scripts/profile.sh <index> --set full

# Raw machine-readable output (for parsing)
bash scripts/profile.sh <index> --page raw

# Source-correlated counters (maps metrics to source lines)
bash scripts/profile.sh <index> --page source

# Filter by kernel name (full-match regex; use .*name.* for substring)
bash scripts/profile.sh <index> --kernel-name "my_kernel"

# Combine options
bash scripts/profile.sh <index> --set full --page raw --kernel-name ".*gemm.*"

# List available NCU sets and sections
bash scripts/profile.sh --ncu-options

# Increase timeout for complex kernels (default: 60s)
bash scripts/profile.sh <index> --timeout 120

# Limit output length
bash scripts/profile.sh <index> --max-lines 200
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `<index>` | (required) | Workload index from `--list` |
| `--list` | — | Print workload table with indices |
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

### Key Metrics to Look For

**Memory vs Compute bound:**
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — SM (compute) utilization
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — DRAM (memory) utilization
- If memory throughput >> compute throughput → memory-bound (optimize data movement)
- If compute throughput >> memory throughput → compute-bound (optimize arithmetic)

**Occupancy:**
- `sm__warps_active.avg.pct_of_peak_sustained_elapsed` — achieved occupancy
- Low occupancy often means register pressure or shared memory limits active warps
- Check `launch__registers_per_thread` and `launch__shared_mem_per_block_dynamic`

**Warp stalls (where time is lost):**
- `smsp__warps_issue_stalled_*` — breakdown of stall reasons
- `stalled_long_scoreboard` → waiting for global memory loads
- `stalled_math_pipe_throttle` → compute pipe full (compute-bound)
- `stalled_barrier` → syncthreads overhead
- `stalled_not_selected` → insufficient warps to hide latency

**Launch configuration:**
- `launch__grid_size`, `launch__block_size` — verify your launch config
- `launch__waves_per_multiprocessor` — waves < 1 means underutilization

## Profiling Workflows

### 1. Identify Memory-Bound Kernels

```bash
bash scripts/profile.sh <index> --set detailed
```
Look at "Memory Workload Analysis" section. If DRAM throughput is near peak but compute is low, focus on:
- Reducing global memory accesses (tiling, shared memory caching)
- Improving memory access patterns (coalescing)
- Using vectorized loads (float4, int4)

### 2. Check Occupancy

```bash
bash scripts/profile.sh <index> --set detailed
```
Look at "Occupancy" section. If achieved occupancy is significantly below theoretical:
- Reduce registers per thread (`__launch_bounds__`, compiler flags)
- Reduce shared memory per block
- Adjust block size

### 3. Source-Level Analysis

```bash
bash scripts/profile.sh <index> --page source
```
Maps stall cycles to specific source lines. Useful for finding the exact code causing bottlenecks.

### 4. Compare Before/After

Profile the same workload before and after an optimization. Focus on:
- Did the target metric improve?
- Did any metric regress unexpectedly?

## Tips

- **Choose representative workloads**: Profile workloads with the worst speedup from `bench.sh`, not the fastest ones.
- **Start with `--set detailed`**: The default covers most use cases. Only use `full` when detailed doesn't explain the issue.
- **Use `--kernel-name`**: When your solution launches multiple kernels, filter to the one of interest. The name usually matches your function/kernel name.
- **NCU adds overhead**: Profiled execution is much slower than normal. This is expected — NCU replays kernels to collect metrics.
- **One workload at a time**: NCU profiles a single workload in detail, unlike `bench.sh` which runs all workloads. Pick the most informative one.
- **`--page raw` for scripting**: When you need to extract a specific metric value, use raw output and parse the result.

# Iteration Log

## Summary

<!-- Append one row per iteration. Status: improved / no-change / regression / failed / reverted
     Every 5 iterations: append a progress snapshot (baseline, current best, confirmed bottlenecks, remaining hypotheses). -->

| Iter | Title | Score | Passed | Latency(mean) | Status | Bottleneck |
|------|-------|-------|--------|---------------|--------|------------|

## Rejected Paths

<!-- Record approaches that are genuinely dead ends, so they are not retried.
     Be careful: a partial regression (e.g., some workload groups slower) may mean the approach
     needs tuning, not that it should be abandoned. Check per-group breakdown before rejecting.
- **What:** [approach description]
- **Result:** [what happened]
- **Why it failed:** [root cause analysis]
- **Lesson:** [what this tells you about the kernel]
-->

## Parameter Sweeps

<!-- For rapid parameter tuning (block sizes, num_warps, num_stages, etc.), use this
     compact format instead of full iteration entries. One row per experiment.
     Promote the best result to the Summary table when the sweep concludes. -->

| # | Parameter | Values tried | Best | Score | Latency(mean) | Notes |
|---|-----------|-------------|------|-------|---------------|-------|

## Iterations

<!-- Template — copy for each new iteration:

### Iter N — Short title

- **Bottleneck hypothesis:** memory_bound | compute_bound | launch_bound | sync_bound | register_bound
- **Target lever:** tiling | vectorization | shared_memory | fusion | pipelining | occupancy | algorithmic | other
- **Expected outcome:** Specific prediction with numbers (e.g., "occupancy +20%, latency -15%")

- **Changes:** What was modified and why THIS approach over alternatives
- **Alternatives rejected:**
  - [approach]: [why not]

- **Bench:**
  - Passed: X/Y
  - Score: ___x
  - Latency: ___ ms (mean), ___ ~ ___ ms (min ~ max)
  - By group: <group_key>: <val1>→___x  <val2>→___x  ...
  - Failed: (none) / workload indices and reason

- **Profiler data** (if profiled this iteration):
  - Occupancy: ___
  - DRAM throughput: ___
  - SM efficiency: ___
  - Register spill: yes/no
  - Key stalls: ___

- **Expected vs Actual:** Do NOT skip this field.
  - Predicted: [what you expected]
  - Got: [what actually happened]
  - Gap: [why the prediction was wrong, if it was]

- **Belief update:** What changed in your understanding of this kernel
-->

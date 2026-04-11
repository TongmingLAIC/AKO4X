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

<!-- When tuning a single parameter (block size, num_warps, num_stages, etc.) with
     the same algorithmic approach: use this table. One row per experiment — NO summary
     row or detail entry needed. When the sweep concludes, promote the best config to
     the Summary table as one combined iteration (e.g. "sweep: BLOCK_M best=64"). -->

| # | Parameter | Values tried | Best | Score | Latency(mean) | Notes |
|---|-----------|-------------|------|-------|---------------|-------|

## Iterations

<!-- Template — copy for each new iteration:

### Iter N — Short title

- **Why:** Motivation for this change (1 sentence)
- **Expected:** Specific prediction (1 sentence)
- **Result:** Score: ___x | Passed: X/Y | Latency: ___ ms (mean)
- **Δ:** Expected vs actual — match or mismatch, and why (1-2 sentences). Do NOT skip this field.

-->

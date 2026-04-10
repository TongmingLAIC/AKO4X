# Iteration Log

## Summary

<!-- Append one row per iteration. Status: improved / no-change / regression / failed / reverted -->

| Iter | Title | Score | Passed | Latency(mean) | Status | Bottleneck |
|------|-------|-------|--------|---------------|--------|------------|

## Rejected Paths

<!-- Record approaches that were tried and abandoned, so they are not retried.
- **What:** [approach description]
- **Result:** [what happened]
- **Why it failed:** [root cause analysis]
- **Lesson:** [what this tells you about the kernel]
-->

## Parameter Sweeps

<!-- For rapid parameter tuning (block sizes, num_warps, num_stages, etc.), use this
     compact format instead of full iteration entries. One row per experiment.
     Promote the best result to the Summary table as a normal iteration when the sweep concludes. -->

| # | Parameter | Values tried | Best | Score | Latency(mean) | Notes |
|---|-----------|-------------|------|-------|---------------|-------|

## Iterations

<!-- Template — copy for each new iteration:

### Iter N — Short title

- **Why:** Motivation for this change (1 sentence)
- **Expected:** Specific prediction (1 sentence)
- **Result:** Score: ___x | Passed: X/Y | Latency: ___ ms (mean)
- **Δ:** Expected vs actual — match or mismatch, and why (1-2 sentences)

-->

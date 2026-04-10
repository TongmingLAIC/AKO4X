## Execution Protocol

Every modification-benchmark cycle is one **iteration**. Follow this protocol strictly.

### Before modifying code

1. State your **bottleneck hypothesis** and **target lever** explicitly
2. State what you **expect** the benchmark to show — be specific with numbers
3. Note what alternatives you considered and why you rejected them

### After benchmark completes

Hooks will block further code edits and bench runs until you update ITERATIONS.md.

4. Compare actual results against your expectation — the **Expected vs Actual** section is mandatory
5. If your prediction was wrong, update your mental model and record the **belief update**
6. If results regressed, **revert immediately** and add an entry to the "Rejected Paths" section in ITERATIONS.md
7. Git commit: `bench(<score>): <one-line description>`

### Investigation tools

When bottleneck is unclear or speedup has stalled for several iterations, investigate **before** making the next code change:

- **Profile**: `bash scripts/profile.sh <index>` — use when you need to know if the kernel is memory-bound, compute-bound, or launch-bound, or to check occupancy and register pressure.
- **WebSearch**: use when encountering unfamiliar APIs, hardware characteristics, or optimization patterns (e.g., "triton grouped GEMM best practices", "B200 FP8 tensor core throughput").

When you use a tool for investigation, annotate it inline with a before/after pair:

**Before the tool call:**
> **Investigate**: [why this tool now] — expecting [what signal]

**After getting results:**
> **Found**: [key data point] — means [how this changes your plan]

Keep each annotation to ONE sentence. The findings should flow into the relevant
fields of your next ITERATIONS.md entry. Do NOT create a separate iteration entry for
investigation-only actions. An iteration = a code change + benchmark.

### Every 5 iterations, write a progress snapshot

Append to the end of ITERATIONS.md:

```
### === Snapshot after Iter N ===
- Baseline: [initial score]
- Current best: [score] (Iter X)
- Cumulative speedup: [ratio]
- Confirmed bottlenecks: [what you've verified]
- Rejected approaches: [count, refer to Rejected Paths]
- Remaining hypotheses: [what hasn't been tried]
- Biggest remaining opportunity: [where to focus next]
```

### Iteration Scale

Not all code changes require the same level of documentation:

- **Full iteration** (architecture/algorithm changes): Fill all fields in the iteration template — hypothesis, expected outcome, belief update, etc.
- **Micro experiment** (single parameter change, block size tuning): Use the **Parameter Sweeps** table in ITERATIONS.md — one row per experiment, no hypothesis/belief-update required.
- Rule of thumb: if you're changing ≤2 numeric constants and the approach is the same, it's a micro experiment. Promote the best result to the Summary table when the sweep concludes.

### Rules

- Keep each field in ITERATIONS.md to 1-2 sentences — this is an audit trail, not an essay
- Do NOT skip "Expected vs Actual" — it is the single most valuable record
- Do NOT repeat raw benchmark output verbatim — summarize the key numbers
- When reverting, ALWAYS add to "Rejected Paths" before moving on

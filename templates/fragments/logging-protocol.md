## Execution Protocol

Every modification-benchmark cycle is one **iteration**. Follow this protocol strictly.

### Before modifying code

1. State your **bottleneck hypothesis** and **target lever** explicitly
2. State what you **expect** the benchmark to show — be specific with numbers
3. Note what alternatives you considered and why you rejected them

### After benchmark completes

4. Compare actual results against your expectation — the **Expected vs Actual** section is mandatory
5. If your prediction was wrong, update your mental model and record the **belief update**
6. If results regressed, **revert immediately** and add an entry to the "Rejected Paths" section in ITERATIONS.md

### Tool use annotation

When you use a tool for investigation (profiling, web search, reading docs/papers),
annotate it inline with a before/after pair:

**Before the tool call:**
> **Investigate**: [why this tool now] — expecting [what signal]

**After getting results:**
> **Found**: [key data point] — means [how this changes your plan]

Keep each annotation to ONE sentence. The findings should flow into the relevant
fields of your next ITERATIONS.md entry (Profiler data, Bottleneck hypothesis,
Alternatives rejected, etc.). Do NOT create a separate iteration entry for
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

### Rules

- Keep each field in ITERATIONS.md to 1-2 sentences — this is an audit trail, not an essay
- Do NOT skip "Expected vs Actual" — it is the single most valuable record
- Do NOT repeat raw benchmark output verbatim — summarize the key numbers
- When reverting, ALWAYS add to "Rejected Paths" before moving on

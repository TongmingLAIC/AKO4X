Phase-1 optimization is finished. Now do a **harness retrospective** — only based on the actual evidence in this session (transcript, ITERATIONS.md, git diff, bench output). Do not add anything from general knowledge.

Re-read your phase-1 trail. Was anything in the harness — a SKILL doc, a script, a task description, a reference TRAPS entry — actually misleading or absent or buggy or redundant in a way that cost iterations?

Common places gaps show up (illustrative, not exhaustive):
- SKILL doc misleading or absent
- `scripts/<file>.py` bug / missing flag / unparseable output
- your `CLAUDE.md`'s `## Workflow` unclear
- `docs/prior/README.md` or `docs/prior/TRAPS.md` (if your operator has a prior archive) contradicting what you actually saw, or a recurring trap with no warning

**Deletion is a valid proposal class** — alongside add / rewrite, not a degenerate case of either. If a SKILL section or paragraph (a) you tried and its advice was empirically refuted by your trail, (b) is version-pinned to a toolchain that has already moved out from under it during your session, or (c) is generic kernel-optimization knowledge that demonstrably ate your attention without adding signal — propose deleting it. Same evidence bar as add: cite the trajectory artifact, the iter where the section misled you, or the prompt-budget cost showing up in your `## Notes`. "I didn't happen to use this section" is **not** sufficient evidence — different operators exercise different sections, and absence of use this session does not imply absence of value next session. Deletion needs you to have *tripped over* the content, not just *not exercised* it.

**If your phase-1 made no improvement** despite repeated tries: pay extra attention to "I tried X repeatedly and the harness signal was unhelpful". A harness gap manifests as wasted iterations.

**If you have no concrete evidence** for any of the above: output exactly the section below with `none`. Do **not** invent. `none` is the right answer when phase-1 went smoothly.

---

**Output**: save your retrospective to `PROPOSALS.md` at the child env root. The file IS your output channel — the master reads it directly. **Always write the file**, even when there are no proposals to make. Body is one of:

**(a) literal `none`** — phase-1 went smoothly:

```
none
```

**(b) one or more proposals** — bulleted fields per proposal, separated by `---` on its own line or `## proposal-1` / `## proposal-2` / ... sub-headings:

```
- **scope**: ...
- **evidence pointer**: ...
- **patch**: ...
- **predicted utility**: ...
- **rationale**: ...

---

- **scope**: ...
- ...
```

For acceptance, the master needs to see two things in your text — anywhere, in any field, prose form is fine:

1. **A scope** — file to edit. Must be MUTABLE per the **Closed-loop scope rules** appended below (FROZEN edits get rejected).
2. **Evidence from this phase-1** — ITERATIONS.md line / commit SHA / bench output / sanitizer log / TRAPS entry. Generic or hypothetical reasoning won't pass the gate.

The fields below are a suggested shape (helps you cover each angle), not a parser schema:

- **scope**: file path to edit.
- **evidence pointer**: the phase-1 artifact motivating this change.
- **patch**: diff or new-file content. For a deletion proposal, quote the section header + first line of the block to be removed and mark it `(delete)`.
- **predicted utility**: one sentence — "if this lands, future sessions in <situation> will <do X better>".
- **smoke evidence** (only for `scripts/` proposals): actual stdout / output you ran.
- **rationale**: prose tying it together.

Keep proposals minimal. Patch existing files rather than adding new ones. Don't bundle unrelated fixes into one proposal.

## Session-best handoff (separate output channel)

If phase-1 produced a kernel worth keeping as this session's forensic output — **whether or not it beat the baseline** — package it for the master at `proposed-variants/<your-chosen-name>/` (a fresh dir at the child env root) before ending phase-2. The master promotes from this location into its persistent archive.

A sub-baseline kernel is still worth packaging when it closes (or ceiling-hits) a structural lever — its 5-section header carries the lever-status signal the next session uses to decide whether to fork from it.

**What to package** — four files in `proposed-variants/<name>/`:
- `kernel.py` — the iter you're naming as session-best, with a 5-section header at the top. The five sections: **Identity**, **Delta from prior anchor**, **Lessons on this variant**, **Dead-ends tried**, **Open directions**. Mirror the section names and the per-section discipline of the parent variant's header at `docs/prior/variants/<anchor>/kernel.py` (if your operator has a prior archive). For a sub-baseline kernel, the Identity section MUST state the score AND what the closing evidence was (refuted lever / hit-ceiling-with-NCU / unfinished probe / etc.) — future readers need this to judge whether the kernel is a viable starting point or a dead-end marker.
- `config.toml` — the build config matching this iter (typically the project root `config.toml`; copy whatever was active when this iter was benched).
- `results.json` — the bench output for the same iter. Pull from your `trajectory/<timestamp>_<iter-label>/` snapshot directly — it already has this file, the labels should match your ITERATIONS.md Summary table.
- `variance.json` — optional; include only if you ran `--variance-check` on the session-best iter.

**What NOT to package**:
- Smoke probes (single-workload runs). Those stay in `trajectory/` as forensic detail.
- An iter that obviously regresses below another full-bench iter you ran — unless the iter you're packaging is closing a structural lever and the regression itself is the evidence, in which case the *header* (not the kernel) is the forensic contribution and you should still pick the highest-score iter on that lever's lane.
- A copy of the parent kernel unchanged. If you ended phase-1 with `solution/kernel.py` equal to your spawn-time parent (you explored modifications and ended back at the parent), leave `proposed-variants/` empty.

**When to leave `proposed-variants/` empty**:
- Phase-1 produced no completed-bench kernel (full crash cascade, time-budget exhausted before first labeled bench, etc.).
- Every iter you ran obviously regressed below the parent AND you have no structural-lever-closing header to contribute.

Leaving it empty is the right answer when it's the right answer — the master notices and skips archive. Don't pad with a low-value iter to "fill the channel".

**Choice of name**: descriptive + iter-anchored, e.g. `iter3-persistent-tcgen05` or `iter-7-triton-bn16-singlelaunch`. The master uses this name verbatim as the variant directory name, so pick something that reads well in a `ls` listing.

This is a separate output channel from `PROPOSALS.md` — both can be non-empty in the same retrospective (sub had a session-best kernel AND a harness gap to flag), or either can be empty independently. Package the kernel handoff BEFORE writing `PROPOSALS.md`, so if you run into a session-end limit while writing the retrospective, the kernel output channel is already complete.

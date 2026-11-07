# Closed-loop campaigns (Mode 2 & Mode 3)

The rest of the docs describe the **manual workflow**: you `spawn.py` one child
environment and drive a single agent through it by hand (sometimes called
*Mode 1*). The **closed loop** is an opt-in alternative for unattended,
multi-round optimization: a persistent **master agent** spawns a fresh child each
round, drives a **sub agent** through it, archives the best kernel, and repeats —
no human in the loop between rounds.

| | Manual (Mode 1) | Closed loop (Mode 2 / 3) |
|---|---|---|
| Driver | you, interactively | master agent, autonomously |
| Environments | one child you live in | a fresh child per round |
| Best-kernel memory | you track it | archived to `reference/<family>/` |
| Good for | a quick one-off optimization, hands-on exploration | a sustained multi-round search you walk away from |

If you just want a faster kernel once, the manual workflow in the README
[Quick Start](../README.md#quick-start) is enough. Reach for the closed loop
when you want many rounds of optimization on one operator with persistent
memory across rounds.

## Concepts

- **Campaign** — N rounds of optimization on a single fixed `<family>`, `<gpu>`,
  `<backend>`, and `<mode>`. All four are locked at the start (see *Mode is locked*
  below). `family == operator name, kebab-cased` (operator
  `mla_paged_decode_h16_ckv512_kpe64_ps1` → family
  `mla-paged-decode-h16-ckv512-kpe64-ps1`).
- **Round** — one spawn → optimize → archive cycle. The master picks a parent
  variant, spawns a child seeded from it, runs the sub, and archives the result.
- **Phase-1 / phase-2** — within a round, *phase-1* is the sub doing kernel
  optimization. *Phase-2* (Mode 3 only) is a follow-up retrospective in the same
  sub session, where the sub proposes harness improvements.
- **Family archive** — `reference/<family>/` accumulates the campaign's memory:
  working kernel variants, a `README.md` anchor + history, a `baseline.json`
  scoring denominator, a `TRAPS.md` of silent-bug patterns, and `_failed/` round
  transcripts. `spawn.py` seeds each new child from this archive.
- **Ledger** — `master/harness-ledger.md`, an append-only timeline (one
  round-summary line per Mode-2 round; one entry per harness proposal in Mode 3).

## Family archive lifecycle

The family archive is the campaign's persistent memory across rounds — it's
where the baseline lives, where successful variants accumulate, and what
`spawn.py` seeds new children from.

**Archive seed (golden).** Each operator has a canonical baseline at
`reference/<family>/baseline.json`, committed to git. `spawn.py` copies it
into the child env at spawn time; the child's own `baseline.json` is derived
from this golden. The archive carries an `environment` block (gpu / backend
/ cuda_version / mode / measured_at) that future runs use for staleness
checks: a baseline is invalidated when workloads change, when the source
switches between `expert` and `reference`, or when the recorded environment
doesn't match the current run. The `mode` field is what lets the master
refuse a mode mismatch (see [Mode is locked](#mode-is-locked)).

**First-time auto-promote.** If the archive seed doesn't exist yet (a
brand-new operator — see *Bootstrapping a new family* below), the child's
first reference profile is written to **both** the child's local
`baseline.json` and the parent repo's `reference/<family>/baseline.json`.
A `*** First-time baseline promoted to ...` message goes to stderr with
a `git add` hint. Auto-promote only writes when the archive is missing
or a not-yet-measured placeholder; an archive that already holds real
per-workload latencies is never overwritten by a run, so an in-progress
campaign's denominator stays fixed.

### Bootstrapping a new family

When starting a campaign on a fresh operator that doesn't have an archive
yet:

1. **Create the family directory**: `mkdir reference/<family>`
   (lowercase-kebab name — operator
   `mla_paged_decode_h16_ckv512_kpe64_ps1` → family
   `mla-paged-decode-h16-ckv512-kpe64-ps1`).
2. **Write a placeholder README**: a single line is fine —
   `# <family> — bootstrapped YYYY-MM-DD`.
3. **Spawn a child**: `python spawn.py --operator <name> --backend modal
   --gpu b200 ...`. `spawn.py` matches the operator name against
   `reference/<family>/` via underscore-prefix; the family directory
   must exist for the auto-promote path to wire up.
4. **Run one bench**: `bash scripts/bench.sh --label "bootstrap"`. On
   first reference profile, `save_baseline()` writes
   `reference/<family>/baseline.json` automatically and prints the
   `*** First-time baseline promoted ...` notice.
5. **Review and commit**: inspect the numbers (and the `environment`
   block), then `git add reference/<family>/baseline.json && git commit`.

After this the family is fully bootstrapped: future spawns seed
`child/baseline.json` from the golden, and the staleness check enforces
environment match. To **rebuild** an existing golden (e.g. you upgraded
CUDA and want fresh numbers), `git rm` the file and run a child once —
auto-promote will write a new one. The "never overwrite real latencies"
rule means rebuilding requires the explicit delete.

## The two modes

Both share the same round skeleton; they differ only in whether phase-2 runs.

| | **Mode 2** (default) | **Mode 3** (opt-in) |
|---|---|---|
| Phase-2 retrospective | skipped | runs after every clean phase-1 |
| Harness (`templates/` / `scripts/` / `master/`) | held **static** | **co-evolves** |
| `PROPOSALS.md` channel | none | sub writes it; master evidence-gates and applies accepted edits |
| What the master does each round | spawn, drive phase-1, archive the variant, write a round-summary ledger line | the above **plus** read proposals, gate them, apply harness edits |
| Ledger | one round-summary line per round | one entry per proposal (`accepted` / `rejected: reason`) |

Use **Mode 2** to search for a faster kernel while keeping the harness frozen.
Use **Mode 3** when you also want the harness (skills, runtime, prompts) to learn
from each round — the sub reports what was missing, the master decides what to
keep. (The master is reactive-only: it never self-proposes harness edits;
improvements come exclusively through the sub's phase-2 retrospective.)

## Agent selection

Use `--agent codex` with `campaign_start.py` to run the master and children
with Codex. It writes `AGENTS.md` and records `campaign-agent.json`; subsequent
rounds inherit the selection. Claude remains the default. See
[Codex execution](codex.md) for session handling and permission settings.

## Running a campaign

One-time setup per campaign, from a clean `main`:

```bash
python scripts/campaign_start.py --operator <operator_name> \
    --gpu b200 --backend modal --mode 2
```

`--gpu` (default `b200`), `--backend` (`local` / `modal`, default `modal`), and
`--mode` (`2` / `3`, default `2`) are optional. The script:

1. creates a campaign branch `campaign/<operator>/<YYYYMMDD-HHMM>`,
2. replaces root `CLAUDE.md` (or `AGENTS.md` for Codex) with `master/MASTER.md` (so the master agent's
   auto-loaded context **is** the orchestrator protocol),
3. replaces root `README.md` with a campaign stub,
4. commits the swap as a self-documenting `campaign-mode:` commit (carries
   operator / family / gpu / backend / mode — wrap-up reads it later).

Then start the master and tell it what to run:

```bash
# from the repo root (root CLAUDE.md is now the protocol)
claude
```

Send an initial prompt declaring the **same mode** the setup recorded:

```
Run a Mode-2 campaign on family=<family>, gpu=b200, backend=modal.
```

The master initializes the family archive (Round 0) and then loops rounds on its
own. It prints next steps after setup, so you don't have to memorize this.

## Mode is locked

`<family>`, `<gpu>`, `<backend>`, and `<mode>` are fixed for the whole campaign and
recorded in `reference/<family>/baseline.json`'s `environment` block at Round 0.
Mixing them mid-campaign breaks baseline / variance / score comparability (and for
mode specifically, mixes variants born with phase-2 active and without). The master
**raises and halts** if your prompt declares a mode that differs from the locked
one — to change mode, start a new family. Existing pre-mode baselines have no
locked mode yet; the next `init_campaign` call writes whichever mode you declare
in the prompt (additive migration; defaults to Mode 2 if you don't declare).
From that point on the mode is locked.

## After the campaign — wrap-up

When a campaign concludes, its branch holds the accumulated round commits plus the
non-mergeable `campaign-mode:` base commit. Folding the useful commits back into
`main` (and dropping the base commit) is the **wrap-up protocol**, documented in
the top-of-file comment block of `scripts/campaign_start.py`. It touches shared git
history, so it asks for approval before executing — run it (or have the master run
it) consciously, not as part of optimization.

## Where things live

- `master/MASTER.md` — the master agent's full system prompt and 10-step round
  loop. **This is the authoritative protocol** — read it for round-level detail;
  this page is just the operator-facing overview.
- `master/master.py` — the thin IO layer the master calls (spawn / run sub /
  retrospective / archive / ledger). No decision logic.
- `scripts/campaign_start.py` — campaign setup + the wrap-up protocol comment.

`master/` and `scripts/campaign_start.py` are **parent-side only** — `spawn.py`
never copies them into a child environment, so the sub agent can't see the
orchestration layer it runs inside.

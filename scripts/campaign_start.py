#!/usr/bin/env python3
"""scripts/campaign_start.py — initialize a closed-loop AKO4X campaign branch.

Performs the mechanical setup that isolates master CC from dev-facing root
docs and stages a clean commit-curation surface for wrap-up:

1. Pre-flight: must be on `main`, working tree clean
2. Create branch `campaign/<operator>/<YYYYMMDD-HHMM>`
3. Replace root `CLAUDE.md` with `master/MASTER.md` content
   (so master CC's auto-loaded root context IS the orchestrator protocol)
4. Replace root `README.md` with a campaign stub
   (avoids "no README" footguns from IDE / GitHub repo page)
5. Commit the swap with a self-documenting metadata-rich message —
   the wrap-up phase reads this commit message to recover campaign
   identity without asking the user

The wrap-up procedure (run when the campaign concludes, not the concern
of master CC during optimization) is documented in the comment block
immediately below this docstring — readable to any human / dev CC /
master CC opening this file.

Convention: `family == operator name kebab-cased` (e.g. operator
`mla_paged_decode_h16_ckv512_kpe64_ps1` → family
`mla-paged-decode-h16-ckv512-kpe64-ps1`). The deferred cross-operator
sharing extension is left for a future version.

Usage:
    python scripts/campaign_start.py --operator <name> [--gpu b200] [--backend modal] [--mode 2]

Example:
    python scripts/campaign_start.py \\
        --operator mla_paged_decode_h16_ckv512_kpe64_ps1 \\
        --gpu b200 --backend modal --mode 2
"""

# =============================================================================
# CAMPAIGN WRAP-UP PROTOCOL
# =============================================================================
# Run when the campaign concludes — applies regardless of who does it (master
# CC, a fresh dev CC, or the user directly). Wrap-up is not optimization work
# and not master CC's mandatory job; surface this protocol to whoever asks.
#
# The campaign-mode commit at the base of the branch (created by this script)
# MUST NOT enter main — wrap-up drops it.
#
# 1. Read the campaign-mode commit message for context:
#        git log main..HEAD --format=%B | head -30
#    The bottommost (oldest) commit is `campaign-mode: ...` and carries
#    operator / family / backend / gpu / mode / base-sha — recovers campaign
#    identity (including the Mode 2/3 it locked) without asking the user.
#
# 2. Enumerate campaign commits:
#        git log main..HEAD --oneline
#
# 3. Classify each commit. Typical buckets:
#    - campaign-mode (branch base) ................... DROP (must not enter main)
#    - round-N family=... archive ...  ............... usually KEEP (output)
#    - harness improvements (master/ scripts/ tests/)  usually KEEP
#    - harness-ledger.md appends ..................... usually KEEP
#    - _failed/<round-id>/ additions ................. usually KEEP (forensic)
#    - doc / README iterations during campaign ....... case-by-case (squash noise)
#    Mode caveat: in a Mode-2 campaign (see the campaign-mode commit's `Mode:`
#    field) the harness is held static — a round commit touching templates/ /
#    scripts/ / master/ is a BUG per MASTER.md step 10, not a "harness
#    improvement to keep". Scrutinize any such Mode-2 commit before keeping it.
#
# 4. Present the classification + planned action to the user as a table.
#    Do NOT execute without approval — wrap-up is the only time you touch
#    shared git history; treat it as risky-action.
#
# 5. Execute the approved plan. Common shapes:
#    - Drop just campaign-mode, keep everything else:
#        git rebase --onto <campaign-mode-sha>^ <campaign-mode-sha> HEAD
#    - Selective edits across multiple commits:
#        git rebase -i <campaign-mode-sha>^
#    - Preserve full history (least surgical, leaves both swap + undo on record):
#        git revert <campaign-mode-sha>
#
# 6. Verify with `git diff main` against the cleaned branch and show the
#    net diff to the user. Catch: stray CLAUDE.md / README.md residue
#    from campaign-mode, any experiments/ content (should be gitignored),
#    reference/ files in the wrong operator dir.
#
# 7. Build per-campaign archive bundle into `experiments/<operator>/`
#    (gitignored, local-only). Mirror the layout of any prior campaign
#    bundle in `experiments/`. Standard contents:
#      - README.md ........................ campaign identity (operator,
#        family, GPU, backend, CUDA / Triton / image versions, workload
#        count, expert baseline name, campaign start/end), result curve
#        per round, phase-1 wall-clock durations, accepted harness edits
#        table, artifact layout, reproducibility note.
#      - child-envs/<round-label>/ ........ full child env dirs from
#        `~/Project/AKO/ako4x-run-<round-label>-*/`. Compress (`tar
#        -cJf <round-label>.tar.xz`) only if a round bloats past ~100 MB
#        (e.g. trajectory snapshots of large auxiliary files); the MLA
#        r2 precedent compressed 3.6 GB → 354 MB.
#      - master-claude-sessions/AKO4X-cwd/<uuid>.jsonl — this
#        campaign's master CC session memory from
#        `~/.claude/projects/-home-...-AKO4X/`. Filter by date if
#        multiple sessions exist.
#      - sub-claude-sessions/r{N}/ ........ per-round sub CC local
#        session memory from
#        `~/.claude/projects/-home-...-ako4x-run-<round-label>/`.
#      - reference-snapshot/<family>/ ..... snapshot of
#        `reference/<family>/` as of campaign end (variant archive +
#        README + baseline + _failed/ if any).
#      - ledger-extract.md ................ entries from
#        `master/harness-ledger.md` that this campaign produced. The
#        global ledger remains canonical (single source of truth for the
#        co-evolved harness state); this is a per-campaign convenience
#        view so the gitignored archive bundle is self-contained for
#        retrospective reading. Filter by date range
#        (campaign-mode-commit date → today) plus inspection — entries
#        don't carry a canonical `family=` tag, so use scope path /
#        edit-id keyword to disambiguate when needed.
#
#    Do this BEFORE step 8's `git branch -D`. The child envs and the
#    branch's curated git history are the most retrievable just before
#    deletion.
#
#    After the archive bundle is built AND `diff -rq` confirms each
#    archived child-envs/<round-label>/ is byte-identical to its origin
#    (`~/Project/AKO/ako4x-run-<round-label>-*/`), **delete the
#    originals**: `rm -rf ~/Project/AKO/ako4x-run-<round-label>-*`.
#    Semantics is archive-then-delete, not archive-then-leave — child
#    envs accumulate disk + clutter parent dir across campaigns
#    otherwise. The diff verification is the safety gate; without it,
#    skip the delete and ask the user. (Earlier precedent: rmsnorm-h128
#    wrap-up missed this step; gemm-n2048-k4096 wrap-up retroactively
#    cleaned both campaigns.)
#
# 8. Land:
#        git checkout main && git merge --ff-only <branch>
#        git push origin main
#        git branch -D <branch>
#        git push origin --delete <branch>
#
# Constraints:
# - Never force-push to main.
# - Never drop commits the user hasn't approved.
# - experiments/<operator>/ is gitignored (local-only) — don't try to fold.
# - If a _failed/<round-id>/ commit exists, surface it explicitly during
#   classification — forensic records that the user should consciously
#   keep or drop, not silently dropped.
# =============================================================================

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agent_runtime import AGENTS, SANDBOXES, render_instructions


def _run(cmd, **kw):
    """Run a git/shell command; raise with stderr on failure."""
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, check=False,
        capture_output=True, text=True, **kw,
    )
    if result.returncode != 0:
        sys.exit(
            f"Error running {' '.join(cmd)!r}:\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def preflight():
    """Verify we're on a clean main. Returns base sha (short) for commit msg."""
    branch = _run(["git", "branch", "--show-current"])
    if branch != "main":
        sys.exit(
            f"Error: campaign_start.py must run from main (currently on "
            f"{branch!r}). Switch to main and ensure the campaign starts "
            f"from a known base."
        )
    dirty = _run(["git", "status", "--porcelain"])
    if dirty:
        sys.exit(
            f"Error: working tree is not clean. Commit or stash first:\n{dirty}"
        )
    return _run(["git", "rev-parse", "--short", "HEAD"])


def dataset_preflight(operator):
    """Refuse to create a campaign branch for an operator the dataset can't spawn.

    The git-side setup below (branch + CLAUDE.md/README.md swap + campaign-mode
    commit) makes no dataset call; dataset gaps only surface inside
    spawn.py at master CC's first spawn_child, by which point we've left a
    half-finished campaign branch behind. Validating here turns that 5-step
    rollback into a 1-second hard fail before any state changes.

    Re-uses spawn.py's `resolve_dataset` + `discover_operator` (single source
    of truth for dataset shape). Skipped silently when neither
    AKO_DATASET_PATH nor FIB_DATASET_PATH is exported — preserves the
    existing "warn at end, master CC discovers at spawn time" path for
    users who export per-session in a different shell.
    """
    if not (os.environ.get("AKO_DATASET_PATH") or os.environ.get("FIB_DATASET_PATH")):
        return
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from spawn import resolve_dataset, discover_operator
        dataset_path = resolve_dataset(None)
        discover_operator(dataset_path, operator)
    except SystemExit as e:
        detail = e.code if isinstance(e.code, str) else "(no detail)"
        sys.exit(
            f"Error: operator {operator!r} is not spawnable from the active dataset:\n"
            f"  {detail}\n"
            f"\n"
            f"campaign_start refused to create the branch — no git state was\n"
            f"changed. Fix this first:\n"
            f"  - try a different --operator (this dataset may not ship it), or\n"
            f"  - confirm both files exist:\n"
            f"      <dataset>/definitions/<category>/<operator>.json\n"
            f"      <dataset>/workloads/<category>/<operator>.jsonl"
        )


def validate_operator(operator):
    """Operator name must be safe for filesystem + branch + commit message."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", operator):
        sys.exit(
            f"Error: operator {operator!r} must match [A-Za-z0-9_]+ "
            f"(kebab/slash/space are reserved by the branch and family "
            f"naming conventions)."
        )


def derive_family(operator):
    """family == operator kebab-cased (current convention; see CLAUDE.md)."""
    return operator.replace("_", "-")


def make_branch_name(operator):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    return f"campaign/{operator}/{ts}"


def render_readme_stub(operator, family, backend, gpu, mode, branch, agent="claude"):
    mode_desc = "no harness modification" if mode == 2 else "harness co-evolution"
    task_file = "AGENTS.md" if agent == "codex" else "CLAUDE.md"
    return f"""# AKO4X — Campaign Branch

This branch is a **closed-loop campaign workspace**, not the main repo.

| Field | Value |
|---|---|
| Operator | `{operator}` |
| Family | `{family}` |
| Backend / GPU | `{backend}` / `{gpu}` |
| Mode | `{mode}` ({mode_desc}) |
| Branch | `{branch}` |
| Agent | `{agent}` |

Root `{task_file}` on this branch is `master/MASTER.md` (the closed-loop
orchestrator protocol) — that's the agent context the master needs, not
the dev guide that lives on `main`.

For **developer documentation**, switch to `main`:

```bash
git checkout main
```

For **campaign wrap-up** (folding accumulated commits back into main
when the campaign concludes), see `scripts/campaign_start.py` top-of-
file `CAMPAIGN WRAP-UP PROTOCOL` comment block — same pointer as the
self-documenting `campaign-mode: ...` commit at the base of this
branch. That commit is intentionally non-mergeable; wrap-up drops or
reverts it before landing on main.
"""


def render_commit_message(operator, family, backend, gpu, mode, base_sha, branch, agent="claude"):
    task_file = "AGENTS.md" if agent == "codex" else "CLAUDE.md"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M %z").strip() or \
         datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"campaign-mode: isolate master CC docs ({operator})\n"
        f"\n"
        f"Operator: {operator}\n"
        f"Family:   {family}\n"
        f"Backend:  {backend}\n"
        f"GPU:      {gpu}\n"
        f"Mode:     {mode}\n"
        f"Agent:    {agent}\n"
        f"Started:  {ts} (from main @ {base_sha})\n"
        f"Branch:   {branch}\n"
        f"Script:   scripts/campaign_start.py\n"
        f"\n"
        f"Files changed by this commit:\n"
        f"  {task_file}   <- replaced with master/MASTER.md content\n"
        f"  README.md   <- replaced with campaign stub\n"
        f"  campaign-agent.json <- agent selection for subsequent rounds\n"
        f"\n"
        f"Wrap-up: see scripts/campaign_start.py top-of-file\n"
        f"`CAMPAIGN WRAP-UP PROTOCOL` block. This commit must NOT be\n"
        f"merged into main; drop or revert it during wrap-up.\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Initialize a closed-loop AKO4X campaign branch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--operator", required=True,
                        help="Full operator name from flashinfer-trace "
                             "definitions/ (e.g., "
                             "mla_paged_decode_h16_ckv512_kpe64_ps1).")
    parser.add_argument("--gpu", default="b200",
                        help="GPU slug for master CC's spawn calls "
                             "(default: b200).")
    parser.add_argument("--backend", default="modal", choices=["local", "modal"],
                        help="Backend for master CC's spawn calls "
                             "(default: modal).")
    parser.add_argument("--mode", type=int, default=2, choices=[2, 3],
                        help="Orchestration mode recorded in the campaign-mode "
                             "commit: 2 = no harness modification (default); "
                             "3 = harness co-evolution. Declare the SAME mode in "
                             "the master CC prompt so the locked baseline matches.")
    parser.add_argument("--agent", default="claude", choices=AGENTS)
    parser.add_argument("--codex-sandbox", default="workspace-write", choices=SANDBOXES)
    args = parser.parse_args()

    validate_operator(args.operator)
    base_sha = preflight()
    dataset_preflight(args.operator)
    family = derive_family(args.operator)
    branch = make_branch_name(args.operator)
    task_file = "AGENTS.md" if args.agent == "codex" else "CLAUDE.md"
    master_md = REPO_ROOT / "master" / "MASTER.md"
    if not master_md.is_file():
        sys.exit(f"Error: {master_md} not found. Cannot create the campaign.")

    # 1. Create + check out branch
    _run(["git", "checkout", "-b", branch])

    # 2. Verify source-of-truth file exists, then swap
    task_md = REPO_ROOT / task_file
    readme = REPO_ROOT / "README.md"
    task_md.write_text(render_instructions(master_md.read_text(), args.agent))
    (REPO_ROOT / "campaign-agent.json").write_text(json.dumps({
        "name": args.agent, "sandbox": args.codex_sandbox,
    }, indent=2) + "\n")
    readme.write_text(render_readme_stub(
        args.operator, family, args.backend, args.gpu, args.mode, branch, args.agent,
    ))

    # 3. Commit the swap with self-documenting metadata
    commit_msg = render_commit_message(
        args.operator, family, args.backend, args.gpu, args.mode, base_sha, branch, args.agent,
    )
    _run(["git", "add", task_file, "README.md", "campaign-agent.json"])
    _run(["git", "commit", "-m", commit_msg])
    commit_sha = _run(["git", "rev-parse", "--short", "HEAD"])

    # 4. Print summary + next steps
    fib_dataset = os.environ.get("AKO_DATASET_PATH") or os.environ.get("FIB_DATASET_PATH", "")

    print()
    mode_desc = "no harness modification" if args.mode == 2 else "harness co-evolution"
    print("===== Campaign initialized =====")
    print(f"  Branch:               {branch}")
    print(f"  campaign-mode commit: {commit_sha}")
    print(f"  Operator:             {args.operator}")
    print(f"  Family:               {family}")
    print(f"  Backend / GPU:        {args.backend} / {args.gpu}")
    print(f"  Mode:                 {args.mode} ({mode_desc})")
    print(f"  Started from:         main @ {base_sha}")
    print()
    if not fib_dataset:
        print("WARNING: AKO_DATASET_PATH is not set in this shell. Master CC")
        print("         will need it to spawn children. Set before starting")
        print(f"         your {args.agent} session, e.g.:")
        print('           export AKO_DATASET_PATH="/path/to/flashinfer-trace"')
        print()
    print("Next steps:")
    print(f"  1. cd {REPO_ROOT}")
    print(f"  2. Start the master: `{args.agent}` (root {task_file} is now the protocol)")
    print(f"  3. Send the initial prompt to master CC. This campaign is set up")
    print(f"     for Mode {args.mode} ({mode_desc}); declare the SAME mode in the")
    print(f"     prompt so the locked baseline matches the campaign-mode commit:")
    print(f'         > Run a Mode-{args.mode} campaign on family={family},')
    print(f'         > gpu={args.gpu}, backend={args.backend}.')
    print(f"  4. When the campaign concludes, run wrap-up — protocol is in")
    print(f"     scripts/campaign_start.py (top of file). It applies regardless")
    print(f"     of who does it (master CC, dev CC, or you directly).")
    print("=================================")


if __name__ == "__main__":
    main()

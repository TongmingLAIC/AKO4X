"""Thin IO layer for the AKO4X closed-loop master agent.

This file is intentionally NOT where decisions live. The master CC makes
accept / reject / archive / quarantine decisions in its prompt; master.py
exposes mechanical helpers (spawn, run sub, parse transcripts, write
artifacts to disk).

Functions, see MASTER.md for round-flow context:

  init_campaign(...)             — Round-0 archive setup (idempotent)
  read_campaign_mode(...)        — read locked mode (2 or 3) from baseline.json
  spawn_child(...)               — derive a child env via spawn.py
  run_sub_phase1(...)            — drive sub through phase-1 (claude --print)
  send_retrospective_prompt(...) — drive sub through phase-2 (claude --resume; Mode 3 only)
  archive_variant(...)           — land a successful variant under reference/<family>/
  archive_failed(...)            — land a crash/timeout transcript under reference/<family>/_failed/
  append_ledger(...)             — append one line to harness-ledger.md

Master CC reads sub's proposals directly from `<child>/PROPOSALS.md` via its
own Read tool — there is no master.py parser for that file. The phase-2
transcript at `<child>/.ako/phase2-transcript.jsonl` is preserved as a soft
fallback for master CC to consult if PROPOSALS.md is absent or malformed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from agent_runtime import codex_outcome, launch_command, read_agent, render_instructions, validate_agent

PKG_DIR = Path(__file__).resolve().parent  # AKO4X/master/
ROOT = PKG_DIR.parent                      # AKO4X/


# ── Data classes ─────────────────────────────────────────────────────────


@dataclasses.dataclass
class SubResult:
    session_id: str
    exit_status: int
    transcript_path: Path
    diff: str
    final_kernel_path: Path | None
    iterations_md_path: Path | None
    stderr_tail: str  # last ~4KB of stderr; preserved across timeout/crash for archive_failed
    kernel_changed: bool = False  # final kernel differs from the spawn-time seed (spawn.py always pre-seeds it, so existence proves nothing)
    timed_out: bool = False


@dataclasses.dataclass
class Phase2Result:
    """Phase-2 session outcome. The actual proposals (if any) live in
    `<child>/PROPOSALS.md`, written by sub via the Write tool — master CC
    reads that file directly via its Read tool, no Python parser involved.
    `proposals_md_path` is just a pre-computed pointer (None when the file
    doesn't exist after phase-2).

    Protocol note: retrospective.md instructs sub to ALWAYS write
    PROPOSALS.md (even a no-change report). So `proposals_md_path is None`
    with a CLEAN phase-2 exit (`exit_status == 0`, not `timed_out`) is a
    phase-2 protocol failure, NOT an empty "no proposals" outcome — the
    caller (master CC, MASTER.md step 6) routes it to the failure archive,
    not 'sub had nothing to propose'."""
    transcript_path: Path
    proposals_md_path: Path | None  # None when sub didn't write PROPOSALS.md (or call failed before it could)
    exit_status: int = 0            # claude --resume exit code (-1 on TimeoutExpired)
    timed_out: bool = False         # True iff subprocess.TimeoutExpired fired
    stderr_tail: str = ""           # last ~4KB of stderr; empty on success


# ── Helpers ──────────────────────────────────────────────────────────────


_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_][\w\-.]*$")


def _validate_safe_name(value: str, *, kind: str) -> None:
    """Reject names with path separators, parent-dir traversal, or empty content."""
    if not value or not _SAFE_NAME_RE.match(value):
        raise ValueError(f"unsafe {kind} {value!r}: must match {_SAFE_NAME_RE.pattern}")


def _read_campaign_env(family: str) -> tuple[str, str]:
    """Read (gpu, backend) from reference/<family>/baseline.json's environment.

    The campaign baseline is the single source of truth for hardware/backend.
    Round 0 seeds an environment-only shell (workloads={}, source="bootstrap")
    before round 1; sub's first reference profile overwrites it with the real
    baseline, keeping the same environment. spawn_child reads gpu/backend here
    instead of taking kwargs, so a forgotten arg can't silently fall back to a
    default and run a round on the wrong hardware.

    Fail-loud: a missing baseline.json, unparseable JSON, or absent
    environment.gpu/backend raises, so a mis-set Round 0 surfaces on round 1
    rather than corrupting N rounds.
    """
    baseline = ROOT / "reference" / family / "baseline.json"
    if not baseline.is_file():
        raise RuntimeError(
            f"campaign baseline not found: {baseline}\n"
            f"Round 0 must seed reference/{family}/baseline.json with an "
            f'environment block: {{"source": "bootstrap", "environment": '
            f'{{"gpu": "<gpu>", "backend": "<backend>", ...}}, "workloads": {{}}}}'
        )
    try:
        data = json.loads(baseline.read_text())
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"campaign baseline {baseline} is not valid JSON: {e}")
    env = data.get("environment", {})
    gpu, backend = env.get("gpu"), env.get("backend")
    if not gpu or not backend:
        raise RuntimeError(
            f"campaign baseline {baseline} missing environment.gpu/backend:\n"
            f"  environment={env!r}"
        )
    return gpu, backend


def _shell_baseline(gpu: str, backend: str, mode: int) -> dict:
    """Round-0 environment-only baseline shell. No real latencies — sub's
    first reference profile overwrites it via save_baseline auto-promote
    (which treats a workloads-less archive as promotable)."""
    return {
        "source": "bootstrap",
        "environment": {
            "gpu": gpu,
            "backend": backend,
            "mode": mode,
            "cuda_version": "unknown",
            "measured_at": None,
        },
        "workloads": {},
    }


def read_campaign_mode(family: str) -> int:
    """Read campaign mode (2 or 3) from reference/<family>/baseline.json.

    Mode is locked at init_campaign alongside gpu/backend:
      - **2** (default): no harness modification. Phase-2 retrospective is
        skipped; master only writes reference/<family>/ (operator memory).
      - **3**: harness co-evolution. Phase-2 retrospective runs; sub's
        PROPOSALS.md is evidence-gated and accepted edits land in templates/.

    Backward-compat: if an existing baseline.json lacks the `mode` field
    (campaign predates the mode split), return 2 (the new default) and emit
    a stderr notice. The field gets backfilled the next time init_campaign
    is called for this family — see init_campaign's "additive migration"
    branch.

    Fail-loud on a missing/unparseable baseline or a present-but-invalid
    mode value, so a corrupted archive surfaces on round 1 rather than
    half-running the round on the wrong branch.
    """
    baseline = ROOT / "reference" / family / "baseline.json"
    if not baseline.is_file():
        raise RuntimeError(
            f"campaign baseline not found: {baseline}\n"
            f"call init_campaign(family={family!r}, mode=...) first"
        )
    try:
        data = json.loads(baseline.read_text())
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"campaign baseline {baseline} is not valid JSON: {e}")
    env = data.get("environment", {})
    mode = env.get("mode")
    if mode is None:
        import sys
        print(
            f"[master] reference/{family}/baseline.json has no 'mode' field — "
            f"defaulting to Mode 2 (no harness modification). To run Mode 3 "
            f"(harness co-evolution), declare 'mode=3' in your campaign prompt; "
            f"init_campaign will backfill the field into the existing baseline.",
            file=sys.stderr,
        )
        return 2
    if type(mode) is not int or mode not in (2, 3):
        raise RuntimeError(
            f"campaign baseline {baseline} has invalid mode {mode!r}; must be int 2 or 3"
        )
    return mode


def _sha256_file(path: Path) -> str | None:
    """Content hash of a file, or None if absent. Lets run_sub_phase1 tell a
    sub-modified kernel from the untouched spawn-time seed — spawn.py ALWAYS
    writes solution/kernel.py (extracted reference or copied parent), so file
    existence alone can't distinguish 'sub produced a kernel' from 'sub
    crashed leaving the seed'."""
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_diff_head(cwd: Path, *, timeout: int = 120) -> str | None:
    """`git diff HEAD` text, or None if git failed or timed out. Diagnostic
    only — a slow/broken git must never hang or crash a round."""
    try:
        p = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True, text=True, cwd=cwd, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    return p.stdout if p.returncode == 0 else None


def _run_bounded(
    cmd: list[str], *, cwd: Path, timeout: int
) -> tuple[str, str, int, bool]:
    """Run `cmd` with a hard timeout, killing the whole process group on
    expiry. Returns (stdout, stderr, returncode, timed_out).

    `subprocess.run(timeout=)` only SIGKILLs the direct child: a timed-out
    `claude` leaves its bench/modal grandchildren running, which keep
    writing into a child the master is about to archive as failed.
    `start_new_session=True` puts the child in its own process group so a
    timeout reaps the whole tree. On real timeout, returncode is -1,
    timed_out is True, and whatever output was buffered before the kill
    is preserved (failure-archive callers depend on having SOMETHING to
    fingerprint).

    Orphan-grandchild false-positive: `proc.communicate(timeout=)` blocks
    until BOTH the direct child exits AND the stdout/stderr pipes drain.
    If `claude` exits cleanly but a Modal/bench grandchild inherited the
    pipes and held them open past the deadline, communicate raises
    `TimeoutExpired` even though the session itself was done. We
    distinguish by polling the direct child at the deadline: when its
    returncode is already set, this is the orphan case — SIGKILL the PG
    to reap the orphans but return the child's actual exit code with
    timed_out=False so MASTER.md step-4 branches on the real status.
    (Empirically: r2 of mla-paged-decode 2026-05-20 hit this — `claude`
    stop_reason=end_turn at ~3.23h, but lingering Modal trajectory-snap
    children kept the PG alive, forcing SIGKILL at the 18000s budget;
    pre-fix master.py returned timed_out=True and the step-4 branch
    would have routed an otherwise-clean round to archive_failed.)
    """
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        return "", str(exc), 127, False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return stdout, stderr, proc.returncode, False
    except subprocess.TimeoutExpired:
        # `proc.poll()` returns None iff the direct child is still
        # running; otherwise its actual exit code. Capture BEFORE the
        # PG SIGKILL so the orphan-case detection is based on whether
        # the child exited on its own.
        child_returncode = proc.poll()
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        stdout, stderr = proc.communicate()
        if child_returncode is None:
            # Real timeout: direct child hadn't exited at deadline.
            return stdout or "", stderr or "", -1, True
        # Orphan-grandchild case: direct child exited cleanly; the
        # pipe-drain wait was stuck on grandchildren we just reaped.
        # Surface the child's actual status, not the false timeout.
        return stdout or "", stderr or "", child_returncode, False


# ── 1. init_campaign ─────────────────────────────────────────────────────


def init_campaign(
    family: str,
    *,
    gpu: str = "b200",
    backend: str = "modal",
    mode: int = 2,
) -> None:
    """Idempotent Round-0 campaign initializer.

    Ensures reference/<family>/ exists with a README placeholder and a
    baseline.json whose environment block carries the campaign's
    gpu/backend/mode — the single source of truth spawn_child / mode
    gating reads each round (master CC never re-passes them).

    `mode` (2 or 3): orchestration mode for the campaign.
      - **2** (default): no harness modification. Phase-2 retrospective is
        skipped; master writes only reference/<family>/.
      - **3**: harness co-evolution. Phase-2 retrospective runs; sub's
        PROPOSALS.md is evidence-gated and accepted edits land in templates/.

    State machine on reference/<family>/baseline.json (gpu/backend match
    required for every non-raise row; mode-mismatch is a separate raise):
      - absent                       → mkdir + variants/ + README + seed shell
      - shell, env match, mode absent → backfill `mode` field (additive migration)
      - shell, env match, mode match  → no-op (already initialized)
      - shell, env match, mode differ → RAISE (mode is locked)
      - shell, env differ            → re-seed shell (no real denominator yet)
      - real,  env match, mode absent → backfill `mode` field
      - real,  env match, mode match  → no-op (continuing campaign)
      - real,  env match, mode differ → RAISE (mode is locked)
      - real,  env differ            → RAISE (hardware change is a NEW family)

    Two non-mechanical outcomes: env-differ-on-real (hardware change is a
    new family) and mode-differ (a mid-campaign mode flip would mix
    variants born under phase-2-active with those born without, breaking
    the audit story). On either raise, master CC relays the message and
    halts — does not work around it. Everything else is deterministic IO.
    An existing README is never overwritten (it may already hold step-8
    history from a prior session).
    """
    _validate_safe_name(family, kind="family")
    # `type(mode) is int` rejects bool (a subclass of int) and float — note
    # 2.0 in (2, 3) is True, so a bare membership test would silently accept a
    # float and persist "mode": 2.0 into baseline.json.
    if type(mode) is not int or mode not in (2, 3):
        raise ValueError(f"mode must be int 2 or 3; got {mode!r}")
    # GPU slug is canonically lowercase: spawn.py lowercases --gpu, and
    # bench_utils._detect_environment() reads the child config's lowercase
    # [build] gpu. Seeding the baseline with the same casing keeps
    # load_baseline()'s environment check from discarding a valid frozen
    # baseline on a pure casing artifact ("B200" != "b200").
    gpu = gpu.lower()
    fam_dir = ROOT / "reference" / family
    baseline = fam_dir / "baseline.json"

    # archive_variant() raises if reference/<family>/variants/ is absent; a
    # brand-new family has none until its first successful round, so seed it
    # here. exist_ok=True keeps this idempotent — and on the real-archive
    # conflict path below the dir already exists (a measured archive went
    # through archive_variant ≥ once), so the raise is still untouched.
    (fam_dir / "variants").mkdir(parents=True, exist_ok=True)

    if baseline.is_file():
        try:
            data = json.loads(baseline.read_text())
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"existing {baseline} is not valid JSON: {e}")
        env = data.get("environment", {})
        cur_gpu, cur_backend = env.get("gpu"), env.get("backend")
        cur_mode = env.get("mode")
        # A present-but-malformed stored mode (e.g. hand-edited "3" or 2.0) must
        # surface as corruption, not as a "mode conflict" — the conflict message
        # would misleadingly claim the caller asked for a different mode.
        if cur_mode is not None and (type(cur_mode) is not int or cur_mode not in (2, 3)):
            raise RuntimeError(
                f"existing {baseline} has a corrupt mode field {cur_mode!r}; "
                f"must be int 2 or 3 (or absent for a pre-mode baseline). "
                f"Fix the file by hand before continuing."
            )
        is_real = bool(data.get("workloads"))
        env_match = (cur_gpu or "").lower() == gpu and cur_backend == backend
        if env_match and cur_mode is not None and cur_mode != mode:
            raise RuntimeError(
                f"campaign mode conflict for family {family!r}:\n"
                f"  existing baseline declares mode={cur_mode!r}\n"
                f"  this campaign asked for mode={mode!r}\n"
                f"Mode is locked at init alongside gpu/backend. A mid-campaign "
                f"mode change breaks the audit story (some variants would be "
                f"born under phase-2-active, others without). Either continue "
                f"with mode={cur_mode!r} or start a new family."
            )
        if env_match:
            if cur_mode is None:
                # Additive migration: backfill the mode field without
                # disturbing measurement state (workloads, source, etc.).
                env["mode"] = mode
                data["environment"] = env
                baseline.write_text(json.dumps(data, indent=2))
                kind = "real, continuing campaign" if is_real else "shell"
                print(
                    f"init_campaign: backfilled mode={mode} into "
                    f"reference/{family}/baseline.json [{kind}; "
                    f"gpu={gpu}, backend={backend}]"
                )
                return
            kind = "real, continuing campaign" if is_real else "shell"
            print(f"init_campaign: {family} already initialized "
                  f"[{kind}; gpu={gpu}, backend={backend}, mode={mode}] — no-op")
            return
        if is_real:
            raise RuntimeError(
                f"campaign conflict for family {family!r}:\n"
                f"  existing archive {baseline} was measured on "
                f"gpu={cur_gpu!r}, backend={cur_backend!r}\n"
                f"  this campaign asked for gpu={gpu!r}, backend={backend!r}\n"
                f"A hardware/backend change is a NEW family (e.g. "
                f"reference/{family}-{gpu}/), not a re-seed — "
                f"mixing corrupts the campaign denominator. Halt and consult "
                f"the user (see MASTER.md 'Round 0' — different hardware/"
                f"backend is a new family, not a re-seed)."
            )
        baseline.write_text(json.dumps(_shell_baseline(gpu, backend, mode), indent=2))
        print(f"init_campaign: re-seeded shell for {family} "
              f"[gpu {cur_gpu}→{gpu}, backend {cur_backend}→{backend}, mode={mode}]")
        return

    fam_dir.mkdir(parents=True, exist_ok=True)
    readme = fam_dir / "README.md"
    if not readme.is_file():
        date = datetime.now().strftime("%Y-%m-%d")
        readme.write_text(f"# {family} — campaign started {date}; anchor TBD\n")
    baseline.write_text(json.dumps(_shell_baseline(gpu, backend, mode), indent=2))
    print(f"init_campaign: seeded {family} "
          f"[new family; gpu={gpu}, backend={backend}, mode={mode}] — "
          f"variants/ + README + shell baseline.json created")


# ── 2. spawn_child ───────────────────────────────────────────────────────


def spawn_child(
    operator: str,
    parent_kernel_path: Path | None,
    name_label: str,  # required: must be unique per round (e.g., "round-3-uuid8" or "<round_id>")
    *,
    family: str,
    dataset_path: Path | None = None,
    agent: str | None = None,
) -> Path:
    """Thin wrapper over spawn.py. Returns the created child_dir Path.

    `name_label` is REQUIRED to be unique per round; spawn.py refuses to
    overwrite an existing child dir. Caller (master CC) must include the
    round-id (or campaign + round-N) so sequential rounds don't collide.

    gpu/backend are NOT parameters — they're read from
    reference/<family>/baseline.json's environment block (seeded by Round 0)
    via _read_campaign_env, so a forgotten kwarg can't silently fall back to
    a default and run a round on the wrong hardware. `family` IS forwarded to
    spawn.py as `--family` so spawn-time prior-lessons + the child's
    archive_seed_path bind to the campaign family, not spawn.py's
    operator-prefix discovery (which can't tell `<op>` from `<op>-<gpu>`).

    parent_kernel_path=None is the brand-new-family round-1 bootstrap: with
    no --kernel, spawn.py extracts the operator reference kernel from
    definition.json (no variant exists yet to parent from). Every later
    round passes the chosen variant's kernel.py path.

    Per-round hints belong in the phase-1 prompt, not as a separate file.

    Raises RuntimeError if the campaign baseline is missing/unparseable, or if
    spawn.py fails or its output is unparseable.
    """
    _validate_safe_name(name_label, kind="name_label")
    _validate_safe_name(family, kind="family")
    gpu, backend = _read_campaign_env(family)
    agent_config = read_agent(ROOT / "campaign-agent.json")
    if agent is not None:
        agent_config["name"] = agent
    # Validate before spawning, including explicit overrides.
    agent_config = validate_agent(agent_config)

    # Guard the "continuing campaign + parent=None" footgun. parent=None is
    # documented (above) as the brand-new round-1 bootstrap only — spawn.py
    # then extracts the operator reference kernel from definition.json. In a
    # continuing campaign that fallback is almost never what master meant:
    # the round silently spawns from the PyTorch reference instead of the
    # selected anchor, and the wrong-parent child dir (sibling of repo,
    # outside auto-mode's trusted tree) then has to be hand-removed before
    # re-spawning under the same round label. Raising here turns the LLM
    # reasoning slip into an immediate retry instead of a 5-step rollback.
    if parent_kernel_path is None:
        baseline = ROOT / "reference" / family / "baseline.json"
        variants_dir = ROOT / "reference" / family / "variants"
        if baseline.is_file():
            try:
                bdata = json.loads(baseline.read_text())
            except (json.JSONDecodeError, ValueError):
                bdata = {}
            has_variants = variants_dir.is_dir() and any(variants_dir.iterdir())
            if bdata.get("workloads") and has_variants:
                raise RuntimeError(
                    f"spawn_child: parent_kernel_path=None is the brand-new "
                    f"round-1 bootstrap only — family {family!r} has a real "
                    f"baseline and ≥1 archived variant (continuing campaign). "
                    f"Pick a parent from reference/{family}/variants/ "
                    f"(default: the anchor named in reference/{family}/README.md)."
                )

    cmd = [
        sys.executable, str(ROOT / "spawn.py"),
        "--operator", operator,
        "--family", family,
        "--gpu", gpu,
        "--backend", backend,
        "--name", name_label,
        "--agent", agent_config["name"],
        "--codex-sandbox", agent_config["sandbox"],
    ]
    if parent_kernel_path is not None:
        cmd += ["--kernel", str(parent_kernel_path)]
    if dataset_path is not None:
        cmd += ["--dataset", str(dataset_path)]

    stdout, stderr, returncode, timed_out = _run_bounded(cmd, cwd=ROOT, timeout=600)
    if timed_out:
        raise RuntimeError(
            f"spawn.py timed out after 600s — env setup hung; campaign "
            f"halted rather than frozen. stderr tail:\n{stderr[-2000:]}"
        )
    if returncode != 0:
        raise RuntimeError(f"spawn.py failed (exit {returncode}):\n{stderr}")

    # spawn.py prints `  Path:     <child_dir>` after creating the child.
    # Capture the rest of the line (not `(\S+)`) so a child dir containing
    # spaces isn't silently truncated into a wrong child_dir.
    child_dir = None
    for line in stdout.splitlines():
        m = re.match(r"\s*Path:\s+(.+?)\s*$", line)
        if m:
            child_dir = Path(m.group(1))
            break
    if child_dir is None:
        raise RuntimeError("spawn.py succeeded but no `Path:` line in output:\n" + stdout)

    # Record the spawn-time kernel hash. spawn.py ALWAYS writes
    # solution/kernel.py (extracted reference or copied parent), so
    # run_sub_phase1 needs this to tell a sub-modified kernel from the
    # untouched seed — the phase-1 outcome branch is exit-code + this-hash
    # driven, not "does kernel.py exist" (it always does).
    ako = child_dir / ".ako"
    ako.mkdir(exist_ok=True)
    seed_hash = _sha256_file(child_dir / "solution" / "kernel.py")
    if seed_hash is not None:
        (ako / "seed-kernel.sha256").write_text(seed_hash + "\n")

    return child_dir


# ── 3. run_sub_phase1 ────────────────────────────────────────────────────


def run_sub_phase1(child_dir: Path, prompt: str, *, timeout: int = 18000) -> SubResult:
    """Drive sub through phase-1 (kernel optimization).

    Uses the child's recorded agent. Claude accepts a caller-generated UUID;
    Codex supplies its thread ID in JSONL. Phase 2 resumes that exact session.

    Captures stdout into `<child>/.ako/phase1-transcript.jsonl`, stderr into
    `<child>/.ako/phase1-stderr.log`, and preserves the last ~4KB of stderr
    in SubResult.stderr_tail. On TimeoutExpired we still write whatever
    output the subprocess emitted before being killed — failure-archive
    callers depend on having SOMETHING to fingerprint.

    Caller decides outcome branch (improved / no-improvement / crash /
    timeout) from exit_status, timed_out, and presence of final_kernel_path.
    """
    ako = child_dir / ".ako"
    ako.mkdir(exist_ok=True)
    config = read_agent(ako / "agent.json")
    session_id = str(uuid.uuid4()) if config["name"] == "claude" else ""
    (ako / "session-id.txt").unlink(missing_ok=True)

    transcript_path = ako / "phase1-transcript.jsonl"
    stderr_log = ako / "phase1-stderr.log"

    cmd = launch_command(config, prompt, session_id)
    stdout, stderr, exit_status, timed_out = _run_bounded(
        cmd, cwd=child_dir, timeout=timeout
    )

    # Sub can wipe untracked working-tree state during phase-1 (e.g. `git
    # clean -fd` after a revert), which deletes the .ako dir we created
    # above. Re-create defensively so we don't lose the captured stdout /
    # stderr to a FileNotFoundError when sub-side cleanup races us.
    ako.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(stdout)
    if config["name"] == "codex":
        session_id, error = codex_outcome(stdout)
        if error:
            stderr += f"\n{error}\n"
            if exit_status == 0:
                exit_status = 1
    (ako / "agent.json").write_text(json.dumps(config, indent=2) + "\n")
    if session_id:
        (ako / "session-id.txt").write_text(session_id + "\n")
    stderr_log.write_text(stderr)
    stderr_tail = stderr[-4096:] if len(stderr) > 4096 else stderr

    diff_text = _git_diff_head(child_dir) or ""

    final_kernel = child_dir / "solution" / "kernel.py"
    final_exists = final_kernel.is_file()
    seed_hash_file = ako / "seed-kernel.sha256"
    seed_hash = seed_hash_file.read_text().strip() if seed_hash_file.is_file() else None
    # kernel_changed: sub left a kernel that differs from the spawn-time seed.
    # Missing seed hash (env predates spawn_child seeding, or .ako lost) → we
    # can't prove "unchanged", so default to changed; that never silently
    # discards a real result, and crash routing is exit-code driven anyway.
    kernel_changed = final_exists and (
        seed_hash is None or _sha256_file(final_kernel) != seed_hash
    )
    iterations_md = child_dir / "ITERATIONS.md"
    return SubResult(
        session_id=session_id,
        exit_status=exit_status,
        transcript_path=transcript_path,
        diff=diff_text,
        final_kernel_path=final_kernel if final_exists else None,
        iterations_md_path=iterations_md if iterations_md.is_file() else None,
        stderr_tail=stderr_tail,
        kernel_changed=kernel_changed,
        timed_out=timed_out,
    )


# ── 4. send_retrospective_prompt ────────────────────────────────────────


def send_retrospective_prompt(
    child_dir: Path,
    session_id: str,
    *,
    retrospective_template_path: Path = ROOT / "templates" / "retrospective.md",
    scope_path: Path | None = ROOT / "templates" / "closed-loop-scope.md",
    timeout: int = 1800,
) -> Phase2Result:
    """Phase-2: continue the same sub session and inject the retrospective prompt.

    Resumes the recorded agent's exact session so it keeps phase-1 context
    (transcript, ITERATIONS.md, git history,
    bench output) when reflecting on harness gaps. The retrospective template
    (`templates/retrospective.md`, a ROOT-anchored default) is read and
    passed as the prompt argument.

    `scope_path` defaults to `templates/closed-loop-scope.md` (ROOT-anchored):
    its contents are appended to the retrospective body separated by
    `\n\n---\n\n`, so sub sees closed-loop scope rules in phase-2 context only
    (kept out of phase-1 noise). Pass `scope_path=None` to skip injection.

    Captures stdout into `<child>/.ako/phase2-transcript.jsonl`, stderr into
    `<child>/.ako/phase2-stderr.log`.

    Returns Phase2Result with `proposals_md_path` pointing at
    `<child>/PROPOSALS.md` if sub wrote it, else None. Master CC reads that
    file directly (no Python parser); the transcript is kept as a soft
    fallback for when PROPOSALS.md is absent or malformed.
    """
    template_text = retrospective_template_path.read_text()
    if scope_path is not None:
        template_text = template_text.rstrip() + "\n\n---\n\n" + scope_path.read_text()
    ako = child_dir / ".ako"
    ako.mkdir(exist_ok=True)
    config = read_agent(ako / "agent.json")
    recorded = ako / "session-id.txt"
    if not session_id or (recorded.is_file() and recorded.read_text().strip() != session_id):
        raise ValueError("retrospective requires the recorded phase-1 session ID")
    template_text = render_instructions(template_text, config["name"])
    transcript_path = ako / "phase2-transcript.jsonl"
    stderr_log = ako / "phase2-stderr.log"

    cmd = launch_command(config, template_text, session_id, resume=True)
    stdout, stderr, exit_status, timed_out = _run_bounded(
        cmd, cwd=child_dir, timeout=timeout
    )

    # Defensive: sub may have wiped .ako between phase-1 and phase-2 (see
    # run_sub_phase1 for the failure mode that motivated this).
    ako.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(stdout)
    if config["name"] == "codex":
        _, error = codex_outcome(stdout, session_id)
        if error:
            stderr += f"\n{error}\n"
            if exit_status == 0:
                exit_status = 1
    stderr_log.write_text(stderr)
    stderr_tail = stderr[-4096:] if len(stderr) > 4096 else stderr

    proposals_md = child_dir / "PROPOSALS.md"
    # None here with exit_status == 0 and not timed_out is anomalous: sub is
    # told to always write PROPOSALS.md. Master CC treats that as a phase-2
    # protocol failure (MASTER.md step 6), not a clean 'no proposals'.
    proposals_md_path = proposals_md if proposals_md.is_file() else None
    return Phase2Result(
        transcript_path=transcript_path, proposals_md_path=proposals_md_path,
        exit_status=exit_status, timed_out=timed_out, stderr_tail=stderr_tail,
    )


# ── 5. archive_variant ──────────────────────────────────────────────────


def archive_variant(
    family: str,
    name: str,
    kernel_path: Path,
    parent: str,
    spawn_meta: dict,
    header_text: str | None,
    *,
    config_path: Path,
    result_json_path: Path | None = None,
    variance_json_path: Path | None = None,
) -> Path:
    """Land a new variant under reference/<family>/variants/<name>/.

    `header_text` is either:
      - a master-composed 5-section header string (Layer 2 fallback case —
        kernel.py has no header yet, archive_variant prepends `header_text`
        with `# ` line convention assumed by caller), OR
      - `None` for the Layer 1 (sub-packaged) case where the sub already
        wrote a 5-section header into `proposed-variants/<name>/kernel.py`
        and master is promoting the file as-is. `kernel_path` is written
        verbatim; no prepending.

    See MASTER.md step 8 "Artifact source — two-tier lookup" for the
    Layer 1 / Layer 2 distinction. Caller is responsible for header content
    in both cases.

    `config_path` is REQUIRED and must exist — `config.toml` is a mandatory
    archive member (templates/agent/lessons-convention.md): a future spawn
    only reuses a variant's build config if it's colocated, so a variant
    archived without it silently re-spawns on generated defaults
    (language=python, kernel.py::run) and is non-reproducible. Fail-loud
    rather than skip, unlike the optional result/variance JSON.

    `parent` must be "null" or a sibling variant name; otherwise ValueError.
    Refuses to overwrite an existing variant directory.
    """
    _validate_safe_name(family, kind="family")
    _validate_safe_name(name, kind="variant name")
    family_dir = ROOT / "reference" / family
    variants_dir = family_dir / "variants"
    if not variants_dir.is_dir():
        raise ValueError(f"family variants dir does not exist: {variants_dir}")

    sibling_names = {p.name for p in variants_dir.iterdir() if p.is_dir()}
    if parent != "null" and parent not in sibling_names:
        raise ValueError(
            f"parent {parent!r} is not a sibling variant in {family_dir}; "
            f"have: {sorted(sibling_names)}"
        )

    target = variants_dir / name
    if target.exists():
        raise ValueError(f"variant directory already exists: {target}")

    if not config_path.is_file():
        raise ValueError(
            f"config.toml is a required archive member but was not found: "
            f"{config_path} (a variant without it re-spawns on generated "
            f"defaults and is non-reproducible)"
        )

    # Stage into a sibling temp dir then rename atomically. If any write
    # fails, the partial state is cleaned up so a retry doesn't trip the
    # overwrite guard above.
    import shutil
    staging = variants_dir / f".{name}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        kernel_text = kernel_path.read_text()
        if header_text:
            header_block = header_text.rstrip("\n") + "\n\n"
            (staging / "kernel.py").write_text(header_block + kernel_text)
        else:
            # Layer 1 (sub-packaged): kernel.py already carries its 5-section header.
            (staging / "kernel.py").write_text(kernel_text)
        (staging / "config.toml").write_text(config_path.read_text())
        (staging / "parent.txt").write_text(parent + "\n")
        (staging / "spawn.json").write_text(json.dumps(spawn_meta, indent=2) + "\n")
        if result_json_path and result_json_path.is_file():
            (staging / "result.json").write_text(result_json_path.read_text())
        if variance_json_path and variance_json_path.is_file():
            (staging / "variance.json").write_text(variance_json_path.read_text())
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


# ── 6. archive_failed ───────────────────────────────────────────────────


def archive_failed(
    child_dir: Path,
    round_id: str,
    family: str,
    *,
    exit_kind: str,
    last_action: str = "",
    last_stderr_tail: str = "",
) -> Path:
    """Land a failed-round bundle under reference/<family>/_failed/<round-id>/.

    Always writes summary.md with mechanical fields filled. The two
    extraction-heavy fields (`cited_skills`, `top_frame`) are placeholdered;
    master CC fills them by reading the transcript directly during ledger
    commit (a real `extract_failure_fingerprint` helper can be added if
    reliability becomes an issue).

    Idempotent: if the failed directory already exists, refuses (caller should
    pass a fresh round_id; collisions are a logic bug).
    """
    _validate_safe_name(family, kind="family")
    _validate_safe_name(round_id, kind="round_id")
    family_dir = ROOT / "reference" / family
    if not family_dir.is_dir():
        raise ValueError(f"family directory does not exist: {family_dir}")

    failed_parent = family_dir / "_failed"
    failed_parent.mkdir(parents=True, exist_ok=True)
    target = failed_parent / round_id
    if target.exists():
        raise ValueError(f"_failed/{round_id} already exists: {target}")

    import shutil
    staging = failed_parent / f".{round_id}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        src_transcript = child_dir / ".ako" / "phase1-transcript.jsonl"
        if src_transcript.is_file():
            (staging / "phase1-transcript.jsonl").write_text(src_transcript.read_text())

        diff_text = _git_diff_head(child_dir)
        if diff_text is not None:
            (staging / "git-diff.patch").write_text(diff_text)

        iterations = child_dir / "ITERATIONS.md"
        if iterations.is_file():
            (staging / "ITERATIONS.md").write_text(iterations.read_text())

        summary = (
            f"# Failure summary — {round_id}\n\n"
            f"- **exit_kind**: {exit_kind}\n"
            f"- **last_action**: {last_action}\n"
            f"- **last_stderr_tail**: {last_stderr_tail}\n"
            f"- **cited_skills**: <TO_FILL_BY_MASTER_CC>\n"
            f"- **top_frame**: <TO_FILL_BY_MASTER_CC>\n"
        )
        (staging / "summary.md").write_text(summary)
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


# ── 7. append_ledger ────────────────────────────────────────────────────


def append_ledger(entry: str, *, ledger_path: Path | None = None) -> None:
    """Append a one-line ledger entry to harness-ledger.md.

    Written as a markdown list item: a leading `- ` is added automatically
    (pass the bare entry without it) so the file renders as a list rather
    than collapsing into one paragraph. A bullet the caller supplies anyway
    is tolerated — not doubled.

    Logical format (canonical spec in harness-ledger.md "## Format"):
      `YYYY-MM-DD round-N edit-id scope: short rationale (accepted|rejected: reason)`

    Multi-line entries are rejected — keep it grep-able.
    """
    p = ledger_path or (PKG_DIR / "harness-ledger.md")
    if not p.is_file():
        raise FileNotFoundError(f"ledger missing: {p}")
    e = entry.strip()
    if "\n" in e:
        raise ValueError(f"ledger entry must be one line; got multi-line: {entry!r}")
    if e.startswith("- "):              # tolerate a caller-supplied bullet
        e = e[2:].lstrip()
    with p.open("a") as f:
        f.write(f"- {e}\n")

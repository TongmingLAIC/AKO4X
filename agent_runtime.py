"""Agent-specific launch and instruction conventions (no model or GPU imports)."""

from __future__ import annotations

import json
from pathlib import Path

AGENTS = ("claude", "codex")
SANDBOXES = ("read-only", "workspace-write", "danger-full-access")


def validate_agent(config: dict) -> dict:
    name = config.get("name", "claude")
    if name not in AGENTS:
        raise ValueError(f"unsupported agent: {name!r}")
    sandbox = config.get("sandbox", "workspace-write")
    if sandbox not in SANDBOXES:
        raise ValueError(f"unsupported Codex sandbox: {sandbox!r}")
    return {**config, "name": name, "sandbox": sandbox}


def read_agent(path: Path) -> dict:
    """Missing metadata means a legacy Claude workspace; corrupt metadata fails."""
    return validate_agent(json.loads(path.read_text()) if path.is_file() else {})


def render_instructions(text: str, agent: str) -> str:
    if agent == "codex":
        return (text.replace("CLAUDE.md", "AGENTS.md")
                .replace(".claude/skills/", ".agents/skills/")
                .replace(".claude/commands/review.md", "docs/iteration-review.md")
                .replace("slash commands under `.claude/commands/`",
                         "review guidance in `docs/iteration-review.md`"))
    return text


def launch_command(config: dict, prompt: str, session_id: str = "", *, resume=False) -> list[str]:
    config = validate_agent(config)
    if config["name"] == "claude":
        session = ["--resume" if resume else "--session-id", session_id]
        return ["claude", "--print", "--verbose", "--output-format", "stream-json",
                *session, prompt]
    # Resume's subparser does not accept --sandbox; pass it to exec itself.
    cmd = ["codex", "exec", "--sandbox", config["sandbox"],
           "-c", 'approval_policy="never"']
    if resume:
        cmd += ["resume", "--json", session_id]
    else:
        cmd += ["--json"]
    return [*cmd, prompt]


def codex_outcome(stdout: str, expected_session: str = "") -> tuple[str, str]:
    """Read documented JSONL lifecycle events; never invent a resumable ID."""
    session_id = expected_session
    completed = False
    error = ""
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "thread.started":
            found = event.get("thread_id")
            if not isinstance(found, str) or not found.strip():
                error = "Codex returned an invalid thread ID"
            elif session_id and session_id != found:
                error = "Codex returned a different thread ID"
            else:
                session_id = found
        elif kind in ("error", "turn.failed"):
            error = f"Codex emitted {kind}: {event.get('error', event.get('message', ''))}"
        elif kind == "turn.completed":
            completed = True
    if not session_id:
        error = error or "Codex did not emit thread.started; no resumable session"
    if not completed:
        error = error or "Codex did not emit turn.completed"
    return session_id, error

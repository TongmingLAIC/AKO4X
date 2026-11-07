# Codex execution

Select Codex independently of the benchmark backend. CUDA kernels still need a
CUDA worker (local or Modal); selecting Codex does not make them run on Metal.
Install and sign in to Codex CLI before starting an optimization session. Use
Python 3.10+ (3.11+ needs no extra packages for the orchestration tests).

## One workspace

```bash
python spawn.py --operator <operator> --dataset /path/to/flashinfer-trace \
  --backend modal --gpu b200 --agent codex --name trial
cd ../ako4x-run-trial
codex
```

The generated `AGENTS.md` contains the task. Skills live at
`.agents/skills/<name>/`, and `scripts/AGENTS.md` carries the shared-runtime
contract. All are rendered from the existing canonical templates. Claude
workspaces keep `CLAUDE.md`, `.claude/skills`, and their existing hooks.

Codex's benchmark wrapper prints the shared review prompt after every
`[advisory].frequency` labeled benchmarks. This is an advisory, not a pass/fail
gate, and it preserves the benchmark exit code. It uses the spawning Python
interpreter by default; set `AKO_PYTHON` to override it after moving a workspace.

## Multiple rounds

From a clean `main` containing this change:

```bash
python scripts/campaign_start.py --operator <operator> \
  --gpu b200 --backend modal --mode 2 --agent codex
codex
```

Ask the master to run the generated family, GPU, backend, and mode. The campaign
commit records `campaign-agent.json`; subsequent `master.spawn_child()` calls
inherit it. `master.run_sub_phase1()` uses the child's `.ako/agent.json`.
`master.send_retrospective_prompt()` resumes the same session for Mode 3.
Omitting the agent retains Claude, including for legacy workspaces.

Codex launches with `codex exec --json` and returns its own thread ID. The
runner saves that ID and the raw event stream, requires `turn.completed`, and
treats `turn.failed`, `error`, or a missing session as failure even if the
process exits zero. Retrospective uses `codex exec resume` with that exact ID,
never `--last`. A timeout keeps partial events and kills the process group.
The campaign's scoring and archive decision protocol are unchanged.

Headless Codex uses `workspace-write` and `approval_policy="never"`. No personal
configuration or credentials are rewritten. `--codex-sandbox` on spawn or
campaign setup explicitly selects another supported policy if required by the
execution environment; blocked commands remain failures, with no automatic
escalation. Interactive `codex` uses the user's normal settings. No model is
selected by the harness. Agent exit success is not evidence of kernel
correctness or speed: those still come from the benchmark artifacts.

## Validation

```bash
python -m unittest discover -s tests -v
python -m compileall -q agent_runtime.py spawn.py master scripts
```

These tests run on CPU and require neither Codex/Claude credentials nor CUDA
packages. They create temporary Git workspaces, simulate agent lifecycle events,
check session recovery and failure handling, and execute the benchmark wrapper
with a failing test command. They do not certify a model-driven GPU campaign.

Protocol references: [Codex non-interactive mode](https://developers.openai.com/codex/noninteractive)
and [skill discovery](https://developers.openai.com/codex/skills).

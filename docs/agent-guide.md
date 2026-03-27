# Agent Guide

## Permissions

Each child environment includes `.claude/settings.local.json` which controls what tools the agent can use. By default, broad access is granted (file I/O, bash, web search, etc.). To restrict permissions, edit this file before launching the agent.

For fully automated (unattended) runs, Claude Code supports skipping all permission prompts:

```bash
claude --dangerously-skip-permissions
```

> **Warning**: This disables all safety confirmations. Only use in trusted, isolated environments.

## Working with the Agent

**Interrupt and guide.** You can interrupt the agent at any time to provide hints, suggest a different optimization strategy, or point it toward a specific bottleneck. The agent will incorporate your feedback and continue.

**Resuming after the agent stops.** The agent may stop after several iterations when it believes it has exhausted its ideas. You have two options:

- **Continue in the same environment**: send another prompt (e.g. "Keep optimizing, try a different approach"). The git history and trajectory are preserved.
- **Spawn a new `existing` environment**: use `--mode existing --kernel /path/to/solution/` to start a fresh environment from the current kernel. This gives the agent a clean slate without prior optimization history.

**Longer sessions.** By default the agent decides when to stop. To request more iterations upfront, include it in your initial prompt:

```
Read CLAUDE.md and optimize the kernel. Iterate at least 20 rounds.
```

## Monitoring Progress

While the agent works, you can monitor its progress without interrupting:

- **`ITERATIONS.md`** — The agent maintains a summary table of each iteration with scores, strategies tried, and results
- **`git log --oneline`** — Each benchmark run is committed with the format `bench(<score>): <description>`
- **`trajectory/`** — Each labeled run saves the kernel snapshot and full results to `trajectory/<timestamp_label>/`

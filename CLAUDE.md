# AKO4FIB — Developer Guide

Template repository for spawning GPU kernel optimization environments.
**Not** an optimization environment — use `spawn.py` to create one.
User-facing docs: [README.md](README.md) and [docs/](docs/).

## Architecture

Three layers:

1. **`spawn.py`** — CLI. Creates child environments from templates + dataset + scripts.
2. **`templates/`** — `task.md` (master template with `{{PLACEHOLDER}}`), `fragments/` (composable blocks), `languages/`, `profiler/`, `agent/` (config + hooks), `iterations.md` / `iterations-trace.md` (light / heavy iteration templates).
3. **`scripts/`** — Copied into children. `bench_utils.py` (shared core), `run_local.py` / `run_modal.py` (runners), `pack_solution.py`, `generate_context.py` (spawn-time only).

## Key Constraints

- **Scripts must be self-contained**: `scripts/` is copied into children. No imports from parent.
- **Children are disposable**: This repo is source of truth. Fix here, re-spawn.
- **Operator data is external**: `definition.json`, `workloads.jsonl` come from the dataset at spawn time.
- **bench_utils.py is shared**: Used by both runners and copied to Modal image. No heavy deps.
- **config.toml merges evaluation overrides**: Base defaults in `populate_child()`, per-op_type in `templates/evaluation.toml`.
- **Hooks enforce iteration protocol**: `templates/agent/hooks/` → child `.claude/hooks/`. Marker-file gate: bench.sh → must edit ITERATIONS.md → unblocked. See `populate_child()`.
- **`--trace` controls template weight**: Light (4 fields) vs heavy (full audit trail). Hooks are the same in both modes.

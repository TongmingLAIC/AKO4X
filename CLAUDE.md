# AKO4FIB — Developer Guide

This is the **template repository** for spawning isolated GPU kernel optimization environments.
It is NOT an optimization environment itself — use `spawn.py` to create one.

For user-facing documentation (installation, usage, language support, benchmarking, troubleshooting),
see [README.md](README.md) and the [docs/](docs/) directory.

## Architecture Overview

AKO4FIB has three layers:

1. **`spawn.py`** — CLI entry point. Discovers operators from a dataset, resolves configuration
   (GPU, backend, mode), and orchestrates child environment creation by combining templates,
   dataset files, and scripts.

2. **Template system** (`templates/`) — `task.md` is the master template with `{{PLACEHOLDER}}`
   markers. Fragment files (`fragments/`) provide composable blocks (objective, backend instructions,
   language guide, profiler guide). Per-language docs (`languages/`) and profiler docs (`profiler/`)
   are copied verbatim into children. The agent config (`agent/claude.json`) controls output
   filenames and permissions.

3. **Runtime scripts** (`scripts/`) — Copied into each child environment. `bench_utils.py` is the
   shared core (baseline caching, scoring, trajectory tracking). `run_local.py` / `run_modal.py`
   are backend-specific runners. `pack_solution.py` bridges `config.toml` to flashinfer-bench's
   Solution model. `generate_context.py` is used at spawn time only (not copied to children).

## Directory Structure

```
AKO4FIB/
├── spawn.py                           # Creates isolated child environments
├── pyproject.toml                     # Package metadata and dependencies
├── docs/                              # User-facing documentation
│   ├── installation.md                # Prerequisites, installation, Modal setup
│   ├── usage.md                       # spawn.py options, modes, language selection
│   ├── child-environment.md           # Structure, config.toml, benchmarking, profiling
│   └── troubleshooting.md             # CUPTI, TVM FFI, version issues
├── templates/
│   ├── task.md                        # Main task template (assembled into child's CLAUDE.md)
│   ├── hints.md                       # Default HINTS.md template (copied to children)
│   ├── gitignore                      # .gitignore template for child environments
│   ├── fragments/
│   │   ├── objective-scratch.md       # Objective for scratch mode
│   │   ├── objective-existing.md      # Objective for existing mode
│   │   ├── language-guide.md          # Inline language reference (table + pointer)
│   │   ├── profiler-guide.md          # Inline profiler summary (local backend only)
│   │   ├── backend-local.md           # Local execution instructions
│   │   └── backend-modal.md           # Modal execution instructions
│   ├── languages/                     # Per-language detail docs (copied to child docs/languages/)
│   ├── profiler/                      # Per-tool profiler docs (copied to child docs/profiler/)
│   └── agent/
│       └── claude.json                # Agent config (output filenames + permissions)
└── scripts/
    ├── generate_context.py            # Extracts operator metadata for template rendering
    ├── bench_utils.py                 # Shared benchmark utilities (baseline caching, scoring)
    ├── bench.sh                       # Local benchmark script
    ├── bench_modal.sh                 # Modal benchmark script
    ├── profile.sh                     # Local NCU profiler script
    ├── run_profile.py                 # NCU profiler runner (local only)
    ├── run_local.py                   # Local benchmark runner
    ├── run_modal.py                   # Modal benchmark runner
    └── pack_solution.py               # Pack kernel for evaluation
```

## spawn.py Pipeline

The `main()` function in `spawn.py` drives the pipeline. Key entry points:

- **`populate_child()`** — the main orchestration function. Creates the child directory structure, copies files, generates `config.toml` (inline template, not a separate file), and wires up backend-specific scripts.
- **`render_template()`** — assembles the task file from `templates/task.md` + fragments. Block placeholders (alone on a line, e.g. `{{OBJECTIVE}}`) replace the entire line; inline placeholders (e.g. `{{OPERATOR}}`) are substituted within lines.
- **`generate_context()`** (in `scripts/generate_context.py`) — extracts template variables (shapes, workload distribution) from the dataset.

## Template System

`templates/task.md` is the skeleton with `{{PLACEHOLDER}}` markers. See `render_template()` in `spawn.py` for the full list of block vs inline placeholders. Fragments in `templates/fragments/` provide the substitution content; `templates/languages/` and `templates/profiler/` are copied verbatim into children.

### How to Modify Templates

| What to change | Where to edit |
|----------------|---------------|
| Task template (main skeleton) | `templates/task.md` |
| Objective text | `templates/fragments/objective-scratch.md` or `objective-existing.md` |
| Language quick-reference | `templates/fragments/language-guide.md` |
| Language detail docs | `templates/languages/*.md` |
| Profiler inline summary | `templates/fragments/profiler-guide.md` (local only) |
| Profiler detail docs | `templates/profiler/ncu.md` |
| Backend instructions | `templates/fragments/backend-local.md` or `backend-modal.md` |
| Agent config | `templates/agent/claude.json` |
| Hints template | `templates/hints.md` |
| config.toml defaults | `populate_child()` in `spawn.py` (inline string) |
| Benchmark scripts | `scripts/bench.sh`, `scripts/bench_modal.sh`, etc. |
| User-facing docs | `docs/*.md` |

## Agent Config System

Each agent type has a JSON config in `templates/agent/` (currently only `claude.json`). Fields: `task_filename`, `hints_filename`, `permissions`. The `permissions` field drives `.claude/settings.local.json` generation in child environments. Use `--agent <name>` at spawn time to select a config.

## Extending the System

### Adding a New Backend

1. Create `templates/fragments/backend-{name}.md` with execution instructions
2. Create `scripts/run_{name}.py` implementing the benchmark runner
3. Create `scripts/bench_{name}.sh` wrapper script
4. Update `spawn.py`: add to `--backend` choices in `parse_args()`, add copy logic in `populate_child()`

### Adding a New Language

1. Create `templates/languages/{name}.md` with language-specific patterns and constraints
2. Update `templates/fragments/language-guide.md` — add a row to the quick reference table
3. Update `infer_language()` in `spawn.py` if the file extension is new
4. Update the language table in `docs/usage.md` and the features list in `README.md`
5. No changes needed in bench scripts — flashinfer-bench handles compilation via its builder system

### Adding a New Agent Type

1. Create `templates/agent/{name}.json` with `task_filename`, `hints_filename`, and `permissions`
2. Use `--agent {name}` at spawn time

## Key Development Constraints

- **Scripts must be self-contained**: Everything in `scripts/` is copied into child environments. No imports from the parent repo — children are standalone.
- **Operator data is external**: `definition.json`, `workloads.jsonl`, and reference kernels come from the dataset at spawn time. This repo never stores operator-specific data.
- **config.toml is generated inline**: The template is a string literal in `populate_child()`, not a separate template file.
- **bench_utils.py is shared**: Used by both `run_local.py` and `run_modal.py`, and also copied into the Modal image. Keep it self-contained with no heavy dependencies beyond what's already required.
- **Child environments are disposable**: This repo is the source of truth. Any fix to scripts or templates must be made here, then new children spawned to pick up changes.

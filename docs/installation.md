# Installation

> **Note**: This page covers the base AKO install + setup for the **default
> benchmark (flashinfer-bench)**. The subsections under [Default benchmark
> setup](#default-benchmark-setup-flashinfer-bench) are FIB-specific. If
> you've swapped benchmarks via [Porting](porting.md), those subsections
> will differ — consult your benchmark's docs.

## Prerequisites

- **NVIDIA GPU**: CUDA-capable (A100, H100, B200, etc.). The default
  benchmark additionally requires **CUDA Driver >= 13.0** — see
  [CUPTI Driver Mismatch](troubleshooting.md#cupti-driver-mismatch) for
  why. If your driver is older, use the Modal backend instead.
- **Git LFS**: required for on-demand tensor downloads
  (`sudo apt install git-lfs`).
- **Python**: 3.10+ (3.12 recommended; `pyproject.toml` sets `requires-python = ">=3.10"` with no upper cap, but 3.14 currently hits pip resolution issues with the default dep set).

## Base install

```bash
pip install .
```

This installs AKO plus the dependencies pinned in `pyproject.toml` —
which today includes the default benchmark (`flashinfer-bench`) and its
transitive deps (`cupti-python >= 13.0`, etc.). If you plan to swap the
benchmark, edit `pyproject.toml`'s dependency first and follow
[Porting](porting.md).

## Default benchmark setup (flashinfer-bench)

The remaining steps wire up the default benchmark. Skip or replace them
if you've ported AKO to a different benchmark.

### Dataset

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace
export AKO_DATASET_PATH=/path/to/flashinfer-trace   # FIB_DATASET_PATH also works
```

Keep this environment activated when running `spawn.py` and when
launching `claude` in child environments — the benchmark scripts use
`python` from `PATH`, so the activated environment must contain all
dependencies.

### Optional language / backend extras

```bash
pip install ".[modal]"        # Modal cloud backend
pip install ".[tilelang]"     # TileLang kernels
pip install ".[cutlass-dsl]"  # CuTe DSL kernels

# DeepGEMM — only needed to run the DSA-indexer expert baseline locally
# (flashinfer_deepgemm_wrapper_2ba145). Requires --no-build-isolation
# because DeepGEMM's build links against the env's torch.
pip install --no-build-isolation ".[deep-gemm]"
```

### macOS (driving Modal, no local GPU)

On macOS you use Modal for all kernel execution. The resolver cannot
install `flashinfer-python` / `cupti-python` / `triton` (no macOS
wheels), and without a `flashinfer` module `flashinfer_bench` can't even
import. Two extra steps on top of the base install — **`uv` is
required** (`--overrides` is uv-specific; plain `pip` has no equivalent
for replacing a transitive requirement's platform marker):

```bash
# 1. Skip the Linux-only deps during resolution
uv pip install --overrides compat/overrides-macos.txt ".[modal]"

# 2. Install the import-time shim so `import flashinfer_bench` loads
#    (the shim raises RuntimeError if any flashinfer function is
#    actually invoked locally — real execution happens on Modal)
uv pip install ./compat/flashinfer-shim
```

Do **not** install `./compat/flashinfer-shim` on Linux — it would mask
the real `flashinfer` package.

### Modal backend

```bash
modal setup                          # authenticate

# Pull actual tensors (Modal has no git — LFS pointers won't work)
cd /path/to/flashinfer-trace
git lfs pull

# Create and upload trace volume
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/flashinfer-trace/
```

### Verify CUPTI is wired up

```bash
python -c "
from importlib.metadata import version
try:
    v = version('cupti-python')
    major = int(v.split('.')[0])
    assert major >= 13, f'cupti-python {v} < 13.0.0'
    from cupti import cupti
    cupti.activity_enable(cupti.ActivityKind.RUNTIME)
    cupti.activity_disable(cupti.ActivityKind.RUNTIME)
    print(f'CUPTI: available (cupti-python {v})')
except Exception as e:
    print(f'CUPTI: unavailable ({e}) — will fall back to CUDA events')
"
```

If this reports "unavailable" because of a driver mismatch, see
[Troubleshooting → CUPTI Driver Mismatch](troubleshooting.md#cupti-driver-mismatch).

## First spawn

```bash
python spawn.py --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 --name my_run
cd ../ako4x-run-my_run
claude
# Send an initial prompt, e.g. "Read CLAUDE.md and optimize the kernel using Triton."
```

`python spawn.py --help` lists every flag; `python spawn.py` with no
arguments lists available operators. The module docstring at the top of
`spawn.py` shows common invocation patterns (custom kernel, Modal-backed
run, custom dataset, custom task template). For closed-loop campaigns
(Mode 2 / Mode 3), see [Closed-loop campaigns](closed-loop.md) instead.

For Codex, add `--agent codex` to spawn and launch `codex` in the child.
See [Codex execution](codex.md) for discovery, permissions, and headless sessions.

## Working with the agent

Once `claude` is running inside the child env, the agent reads its own
`CLAUDE.md` for task framing and loads SKILLs on demand. A few
operator-side things worth knowing.

### Permissions

Each Claude child env includes `.claude/settings.local.json` controlling what
tools the agent may use. By default broad access is granted (file I/O,
bash, web search, etc.); edit it before launch to restrict. For fully
unattended runs, Claude Code supports:

```bash
claude --dangerously-skip-permissions
```

> **Warning**: This disables all safety confirmations. Only use in
> trusted, isolated environments.

### Interrupt and guide

You can interrupt the agent at any time to suggest a different strategy
or point it at a specific bottleneck. The agent incorporates your
feedback and continues.

### Resuming after the agent stops

The agent may stop when it believes it has exhausted its ideas. Two
options:

- **Continue in the same env**: send another prompt (e.g. "Try a
  different approach"). Git history and `trajectory/` are preserved.
- **Spawn a fresh env from the current kernel**: `python spawn.py
  --operator <name> --kernel /path/to/old/solution/` starts a new env
  seeded with that kernel, no prior optimization history.

To request more iterations upfront, include it in your initial prompt:
"Iterate at least 20 rounds."

### Monitoring progress

Without interrupting:

- **`ITERATIONS.md`** — one Summary row per labeled bench plus a
  free-form `## Notes` section the agent uses for hypothesis records
  and end-of-session synthesis. The protocol is documented in the
  file's own preamble.
- **`git log --oneline`** — every benchmark run is committed as
  `bench(<score>): <description>`.
- **`trajectory/<timestamp_label>/`** — each labeled run snapshots
  the kernel + `results.json` for later A/B inspection.

# kernel-opt-agent

A tool for automated GPU kernel optimization using Code Agents (e.g. Claude Code). Built on the [flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench) SDK, it supports any flashinfer-bench compatible dataset, including user-defined operators.

## Quick Start

```bash
# List available operators
python spawn.py

# Create an optimization environment (local, auto-detect GPU, from scratch)
python spawn.py --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64

# Enter the child environment and start your code agent
# (keep the same shell so your Python environment stays activated)
cd ../kernel-opt-agent-run-001
claude
```

## Setup Options

```bash
python spawn.py \
  --operator <name> \
  [--dataset /path/to/dataset] \    # default: $FIB_DATASET_PATH
  [--mode scratch|existing] \       # default: scratch
  [--backend local|modal] \         # default: local
  [--gpu <name>] \                  # local: auto-detected; modal: required
  [--agent claude] \                # default: claude
  [--kernel /path/to/kernel.py] \   # required for existing mode
  [--name <label>] \                # optional label for the run
  [--task /path/to/task.md] \       # custom task template
  [--hints /path/to/hints.md]       # custom hints for the agent
```

### Examples

```bash
# Local, auto-detect GPU, from scratch, with label
python spawn.py --operator gdn_decode_qk4_v8_d128_k_last --name "gdn_exp_1"

# Local with explicit GPU
python spawn.py --operator gdn_decode_qk4_v8_d128_k_last --gpu h100

# Local, from existing kernel
python spawn.py --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 \
    --mode existing --kernel /path/to/kernel.py

# From existing kernel directory (copies all files in directory to solution/)
python spawn.py --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 \
    --mode existing --kernel /path/to/kernel_dir/

# Modal B200, from scratch (--gpu required for modal)
python spawn.py --operator moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
    --backend modal --gpu b200 --name "moe_b200"

# Custom dataset
python spawn.py --operator my_custom_op --dataset /path/to/my/traceset

# Custom task template and hints
python spawn.py --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 \
    --task ./my_task.md --hints ./my_hints.md

# Legacy flag aliases still work (--prompt → --task, --info → --hints)
python spawn.py --operator my_op --prompt ./my_task.md --info ./my_hints.md
```

### Language Selection

The child environment is **language-agnostic**. The agent chooses the kernel language at runtime and updates `config.toml` accordingly.

To guide the agent toward a specific language, use `HINTS.md`:

```bash
# Create environment, then edit HINTS.md before launching the agent
python spawn.py --operator my_op
cd ../kernel-opt-agent-run-*/
echo "Use Triton for this kernel." >> HINTS.md
claude

# Or provide a hints file at setup time
echo "Write the kernel in CUDA." > my_hints.md
python spawn.py --operator my_op --hints my_hints.md
```

The agent can write kernels in Python/PyTorch, Triton, TileLang, CuTe DSL, CUDA, or C++. See the Language Reference section in the generated `CLAUDE.md` for details on each language.

### Existing Mode

When using `--mode existing --kernel /path/to/kernel`:

- **Single file**: `--kernel /path/to/kernel.py` copies the file plus siblings (`.py`, `.cu`, `.cpp`, `.h` files) to `solution/`
- **Directory**: `--kernel /path/to/dir/` copies all files in the directory to `solution/`
- **Colocated config.toml**: If a `config.toml` exists alongside the kernel, it is used directly (useful for pre-configured language/entry_point/DPS settings)
- **Otherwise**: Language is inferred from file extension (`.cu`→cuda, `.cpp`→cpp, `.py`→python)

### Language/Package Notes

Some languages require extra packages:

- **TileLang**: `pip install tilelang` — without it, solutions report COMPILE_ERROR
- **CuTe DSL**: `pip install nvidia-cutlass-dsl` — CuTe DSL kernels are Python-defined (`@CuTeDSL.kernel` decorator) and compiled to PTX at runtime
- **Triton**: Pre-installed with PyTorch, no extra installation needed

**TVM FFI builder limitation (CUDA/C++):** The TVMFFIBuilder compiles `.cu`/`.cpp` files into a shared library but does not support linking external CUDA libraries (cuBLAS, cuDNN, etc.) — the `extra_ldflags` parameter is not passed to the underlying `tvm_ffi.cpp.build()`. To use these libraries, load them at runtime via `dlopen`/`dlsym` from within the `.cu` file.

## How It Works

1. `python spawn.py --operator <name>` discovers the operator definition and workloads from the dataset
2. Creates a child environment `kernel-opt-agent-run-NNN[-label]/` with everything the agent needs
3. You start your code agent (e.g. `claude`) in the child environment
4. The agent reads `CLAUDE.md` and `HINTS.md`, then iteratively optimizes the kernel

### Child Environment Structure

```
kernel-opt-agent-run-NNN/
├── CLAUDE.md                   # Auto-generated task spec (operator details + instructions)
├── HINTS.md                    # User-editable hints and constraints
├── config.toml                 # Operator configuration (agent updates language/entry_point)
├── .gitignore
├── docs/
│   ├── definition.json         # Operator definition
│   ├── workloads.jsonl         # Benchmark workloads
│   └── languages/              # Per-language patterns and constraints
│       ├── python.md
│       ├── triton.md
│       ├── tilelang.md
│       ├── cute-dsl.md
│       ├── cuda.md
│       └── cpp.md
├── solution/                   # Flat directory — agent puts kernel files here
│   └── kernel.py               # Starting point (reference implementation in scratch mode)
├── scripts/
│   ├── bench_utils.py          # Shared benchmark utilities
│   ├── bench.sh                # Benchmark script
│   ├── run_local.py/run_modal.py  # Benchmark runner
│   └── pack_solution.py        # Pack kernel for evaluation
├── baseline.json               # Auto-created, cached reference performance
├── trajectory/                 # Auto-created, stores each bench run
└── .claude/
    └── settings.local.json     # Permission fallback
```

## Customization

### Custom Task Template

The default task template (`templates/task.md`) defines the agent's instructions. Override it with `--task`:

```bash
python spawn.py --operator my_op --task ./my_task.md
```

### Custom Hints

`HINTS.md` is placed in each child environment for ad-hoc hints (including language preferences). Override the template with `--hints`:

```bash
python spawn.py --operator my_op --hints ./my_analysis.md
```

Or edit `HINTS.md` directly in the child environment before launching.

### Custom Dataset

Any flashinfer-bench compatible trace set works. The expected structure:

```
dataset/
├── definitions/
│   └── <type>/
│       └── <operator>.json
└── workloads/
    └── <type>/
        └── <operator>.jsonl
```

**Important:** Your `<operator>.json` must contain a `reference` field with **plain PyTorch code**:

```json
{
  "name": "my_operator",
  "reference": "import torch\n\n@torch.no_grad()\ndef run(input, weight):\n    return torch.matmul(input, weight)",
  ...
}
```

The `reference` serves as the mathematical specification and correctness baseline. FlashInfer-Bench will:
1. Execute it as Python to generate reference outputs for correctness checks
2. Profile it to measure baseline performance (when `profile_baseline=True`)

⚠️ **Critical Constraint**: The `reference` must be executable Python/PyTorch code. FlashInfer-Bench hardcodes reference execution as `language=PYTHON, entry_point=main.py::run`. Using CUDA/C++/Triton code in the `reference` field will cause compilation errors and benchmark failures.

## Benchmark Output

The benchmark script prints per-workload results and a final score:

```
operator_name:
  Workload 0c23b10...: PASSED | 0.492 ms | 4.17x speedup | abs_err=1.2e-03, rel_err=2.1e-04
  ...

══════════════════════════════════════════════════
  FINAL SCORE (mean speedup): 4.17x
  Passed: 23/23 | Min: 2.10x | Max: 8.34x
  By num_tokens:  1→4.17x  2→3.85x  ...
══════════════════════════════════════════════════
```

The `score` field (= `FINAL SCORE` = mean speedup across all workloads) is the primary metric. All workloads must PASS for a valid score.

### Baseline Caching

The first benchmark run profiles the reference implementation and caches the result to `baseline.json`. Subsequent runs skip reference profiling for faster, more stable comparisons and use higher iteration counts (100 vs 20) for more accurate solution timing.

```bash
bash scripts/bench.sh "baseline"          # First run: profiles reference, caches baseline
bash scripts/bench.sh "opt_v1"            # Uses cached baseline, 100 iterations for solution
bash scripts/bench.sh --force-baseline    # Force re-profile reference
```

## Installation

### Prerequisites

- **NVIDIA GPU**: CUDA-capable GPU (A100, H100, B200, etc.)
- **CUDA Driver**: 13.0+ **required** for CUPTI profiling (must pair with `cupti-python >= 13.0.0`). If `cupti-python` is not installed or version < 13.0, benchmarking falls back to CUDA events automatically. But if `cupti-python >= 13.0` is installed while the **driver** is < 13.0, CUPTI fails at runtime — see note below
- **Python**: 3.10 – 3.13 (3.12 recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/flashinfer-ai/kernel-opt-agent.git
cd kernel-opt-agent

# Create and activate a virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies (includes flashinfer-bench from source and cupti-python>=13)
pip install .

# For Modal cloud backend:
# pip install ".[modal]"

# For TileLang or CuTe DSL:
# pip install ".[tilelang]"
# pip install ".[cutlass-dsl]"

# Set dataset path (only needed for spawn.py; child envs store it in config.toml)
export FIB_DATASET_PATH=/path/to/flashinfer-trace
```

> **Note**: Keep this environment activated when running `spawn.py` and when launching `claude` in child environments. The benchmark scripts use `python` from `PATH`, so the activated environment must contain all dependencies.

### Modal Backend Setup (Optional)

For cloud benchmarking on Modal:

```bash
# Authenticate with Modal
modal setup

# Create and upload trace volume
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/flashinfer-trace/
```

### Verify Installation

```bash
# Check GPU
nvidia-smi

# Check flashinfer-bench
python -c "import flashinfer_bench; print(flashinfer_bench.__version__)"

# Check CUPTI profiling support (requires cupti-python >= 13.0.0 and CUDA 13.0+ driver)
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

# List available operators
python spawn.py --dataset /path/to/flashinfer-trace
```

**⚠️ Note**: Local benchmarking uses CUPTI profiling by default via `cupti-python`. There are two fallback scenarios:
- **`cupti-python` not installed or version < 13.0**: Benchmarking gracefully falls back to CUDA events at import time. This works fine but produces noisier measurements, especially with low iteration counts.
- **`cupti-python >= 13.0` installed, but CUDA driver < 13.0**: CUPTI calls fail at runtime with `NotSupportedError`, causing all workloads to report RUNTIME_ERROR. This is **not** caught by the current fallback logic.

An upstream fix in flashinfer to catch this runtime failure is pending but not yet merged. **Current workarounds**: ensure CUDA driver >= 13.0 when `cupti-python >= 13.0` is installed, or use the Modal backend (`--backend modal`).

**⚠️ PyPI versions**: `pyproject.toml` pins `flashinfer-bench` to the git main branch (not PyPI's v0.1.2, which lags behind) and `cupti-python>=13.0.0`. Both are installed automatically via `pip install .`.

## Requirements Summary

See **Installation** section above for detailed setup instructions.

**Quick reference:**

**Local backend (default):**
- NVIDIA GPU with CUDA 13.0+ driver **required** for CUPTI profiling (graceful fallback if `cupti-python` is missing/old, but driver < 13.0 with `cupti-python >= 13.0` causes runtime failure — see note above)
- Dataset path (stored in child env's `config.toml` at spawn time; `FIB_DATASET_PATH` env var as fallback)

**Modal backend:**
- `--gpu` flag required (e.g., `--gpu b200`, `--gpu h100`)
- Modal authenticated and volume uploaded
- No local GPU or CUDA driver required

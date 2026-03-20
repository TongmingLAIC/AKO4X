# AKO4FIB — Agentic Kernel Optimization for FlashInfer-Bench

A tool for automated GPU kernel optimization using Code Agents (e.g. Claude Code). Built on the [flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench) SDK, it supports any flashinfer-bench compatible dataset, including user-defined operators.

## Prerequisites

- **NVIDIA GPU**: CUDA-capable GPU (A100, H100, B200, etc.)
- **CUDA Driver**: 13.0+ recommended for CUPTI profiling (see [Troubleshooting](#troubleshooting) for fallback behavior)
- **Python**: 3.10–3.13 (3.12 recommended)

## Installation

```bash
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

### Verify Installation

```bash
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

### Modal Backend Setup (Optional)

For cloud benchmarking on Modal:

```bash
# Authenticate with Modal
modal setup

# Create and upload trace volume
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/flashinfer-trace/
```

## Quick Start

```bash
# List available operators
python spawn.py

# Create an optimization environment (local, auto-detect GPU, from scratch)
python spawn.py --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64

# Enter the child environment and start your code agent
# (keep the same shell so your Python environment stays activated)
cd ../ako4fib-run-001
claude
```

## Usage

### spawn.py Options

```bash
python spawn.py \
  --operator <name> \
  [--dataset /path/to/dataset] \    # default: $FIB_DATASET_PATH
  [--mode scratch|existing] \       # default: scratch
  [--backend local|modal] \         # default: local
  [--gpu <name>] \                  # local: auto-detected; modal: required
  [--agent claude] \                # default: claude
  [--kernel /path/to/kernel.py] \   # required for existing mode
  [--name <label>]                  # optional label for the run
```

### Examples

```bash
# Local, auto-detect GPU, from scratch, with label
python spawn.py --operator gdn_decode_qk4_v8_d128_k_last --name "gdn_exp_1"

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
```

### Modes

#### Scratch Mode (default)

The PyTorch reference implementation is extracted from the dataset to `solution/kernel.py`. The agent optimizes it in-place (Python/Triton) or rewrites it in any supported language (CUDA, C++, TileLang, CuTe DSL).

#### Existing Mode

When using `--mode existing --kernel /path/to/kernel`:

- **Single file**: `--kernel /path/to/kernel.py` copies the file plus siblings (`.py`, `.cu`, `.cpp`, `.h` files) to `solution/`
- **Directory**: `--kernel /path/to/dir/` copies all files in the directory to `solution/`
- **Colocated config.toml**: If a `config.toml` exists alongside the kernel, it is used directly (useful for pre-configured language/entry_point/DPS settings)
- **Otherwise**: Language is inferred from file extension (`.cu`→cuda, `.cpp`→cpp, `.py`→python)

### Language Selection

The child environment is **language-agnostic**. The agent chooses the kernel language at runtime and updates `config.toml` accordingly.

#### Supported Languages

| Language | Files in `solution/` | config.toml `language` | config.toml `entry_point` |
|----------|---------------------|----------------------|--------------------------|
| Python / PyTorch | `kernel.py` | `python` | `kernel.py::run` |
| Triton | `kernel.py` | `triton` | `kernel.py::run` |
| TileLang | `kernel.py` | `tilelang` | `kernel.py::run` |
| CuTe DSL | `kernel.py` | `python` | `kernel.py::run` |
| CUDA | `kernel.cu` + `binding.py` | `cuda` | `binding.py::kernel` |
| C++ | `kernel.cpp` + `binding.py` | `cpp` | `binding.py::kernel` |

For compiled languages (CUDA, C++), `binding.py` uses TVM FFI (`from tvm.ffi import register_func`) to expose compiled code to Python. Alternatively, use `kernel.cu::symbol_name` with `TVM_FFI_DLL_EXPORT_TYPED_FUNC` to export the function directly.

#### Package Requirements

Some languages require extra packages:

- **TileLang**: `pip install tilelang` — without it, solutions report COMPILE_ERROR
- **CuTe DSL**: `pip install nvidia-cutlass-dsl` — CuTe DSL kernels are Python-defined (`@CuTeDSL.kernel` decorator) and compiled to PTX at runtime
- **Triton**: Pre-installed with PyTorch, no extra installation needed

#### Guiding Language Choice

To guide the agent toward a specific language, edit `HINTS.md` in the child environment before launching the agent:

```bash
python spawn.py --operator my_op
cd ../ako4fib-run-*/
echo "Use Triton for this kernel." >> HINTS.md
claude
```

Per-language documentation is available in `docs/languages/` within each child environment.

## Child Environment

### How It Works

1. `python spawn.py --operator <name>` discovers the operator definition and workloads from the dataset
2. Creates a child environment `ako4fib-run-NNN[-label]/` with everything the agent needs
3. You start your code agent (e.g. `claude`) in the child environment
4. The agent reads `CLAUDE.md` and `HINTS.md`, then iteratively optimizes the kernel

### Structure

```
ako4fib-run-NNN/
├── CLAUDE.md                   # Auto-generated task spec (operator details + instructions)
├── HINTS.md                    # User-editable hints and constraints
├── config.toml                 # Operator configuration (agent updates language/entry_point)
├── .gitignore
├── docs/
│   ├── definition.json         # Operator definition
│   ├── workloads.jsonl         # Benchmark workloads
│   ├── languages/              # Per-language patterns and constraints
│   └── profiler/               # Profiler docs (local backend only)
├── solution/                   # Flat directory — agent puts kernel files here
│   └── kernel.py               # Starting point (reference implementation in scratch mode)
├── scripts/
│   ├── bench_utils.py          # Shared benchmark utilities
│   ├── bench.sh                # Benchmark script
│   ├── run_local.py            # (or run_modal.py) Benchmark runner
│   ├── profile.sh              # NCU profiler script (local only)
│   ├── run_profile.py          # NCU profiler runner (local only)
│   └── pack_solution.py        # Pack kernel for evaluation
├── baseline.json               # Auto-created, cached reference performance
├── trajectory/                 # Auto-created, stores each bench run
└── .claude/
    └── settings.local.json     # Agent permissions
```

### config.toml

The agent updates these fields to match its kernel files:

- **`language`**: `python`, `triton`, `tilelang`, `cuda`, or `cpp`
- **`entry_point`**: e.g. `kernel.py::run` or `binding.py::kernel`
- **`destination_passing_style`**: `false` by default; set `true` when using pre-allocated output buffers

### Benchmarking

Run benchmarks from the child environment:

```bash
bash scripts/bench.sh "label"
```

Output shows per-workload results and a final score:

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

The `FINAL SCORE` (mean speedup across all workloads) is the primary metric. All workloads must PASS for a valid score.

#### Baseline Caching

The first benchmark run profiles the reference implementation (20 iterations) and caches the result to `baseline.json`. Subsequent runs skip reference profiling and use higher iteration counts (100) for more accurate solution timing.

```bash
bash scripts/bench.sh "baseline"          # First run: profiles reference, caches baseline
bash scripts/bench.sh "opt_v1"            # Uses cached baseline, 100 iterations for solution
bash scripts/bench.sh --force-baseline    # Force re-profile reference
```

`baseline.json` auto-invalidates when workloads change.

### Profiling (Local Backend)

NCU profiling is available via `scripts/profile.sh` in the child environment. See `docs/profiler/ncu.md` for detailed usage.

## Custom Datasets

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
2. Profile it to measure baseline performance

**Critical**: The `reference` must be executable Python/PyTorch code. FlashInfer-Bench hardcodes reference execution as `language=PYTHON, entry_point=main.py::run`. Using CUDA/C++/Triton code in the `reference` field will cause compilation errors and benchmark failures.

## Troubleshooting

### CUPTI Driver Mismatch

Local benchmarking uses CUPTI profiling by default via `cupti-python`. There are two fallback scenarios:

- **`cupti-python` not installed or version < 13.0**: Benchmarking gracefully falls back to CUDA events at import time. This works fine but produces noisier measurements, especially with low iteration counts.
- **`cupti-python >= 13.0` installed, but CUDA driver < 13.0**: CUPTI calls fail at runtime with `NotSupportedError`, causing all workloads to report RUNTIME_ERROR. This is **not** caught by the current fallback logic.

**Workarounds**: Ensure CUDA driver >= 13.0 when `cupti-python >= 13.0` is installed, or use the Modal backend (`--backend modal`). An upstream fix in flashinfer-bench is pending.

### TVM FFI Builder: No External Library Linking

The TVMFFIBuilder (used for CUDA/C++ solutions) compiles `.cu`/`.cpp` files via `tvm_ffi.cpp.build()` but does not pass `extra_ldflags` to the linker. Libraries like cuBLAS, cuDNN, etc. cannot be linked at compile time.

**Workaround**: Use `dlopen`/`dlsym` to load CUDA libraries at runtime from within the `.cu` file. For example, `cublasGemmEx` can be called via `dlopen("libcublas.so.12", RTLD_NOW)` + `dlsym`. Prefer `cublasGemmEx` (accepts `float*` alpha/beta) over `cublasHgemm` (accepts `__half*` alpha/beta) — the `__half` type may have ABI issues across shared library boundaries via dlopen.

### flashinfer-bench Version

`pyproject.toml` pins `flashinfer-bench` to the git main branch (not PyPI's v0.1.2, which lags behind). Both `flashinfer-bench` and `cupti-python>=13.0.0` are installed automatically via `pip install .`.

## Contributing

For architecture details and development guidance, see [CLAUDE.md](CLAUDE.md).

## License

<!-- TODO: Add license -->

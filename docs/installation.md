# Installation

## Prerequisites

- **NVIDIA GPU**: CUDA-capable GPU (A100, H100, B200, etc.)
- **CUDA Driver**: 13.0+ recommended for CUPTI profiling (see [Troubleshooting](troubleshooting.md) for fallback behavior)
- **Python**: 3.10–3.13 (3.12 recommended)

## Installation

```bash
pip install .
```

Optional extras for specific kernel languages or backends:

```bash
# Modal cloud backend
pip install ".[modal]"

# TileLang kernels
pip install ".[tilelang]"

# CuTe DSL kernels
pip install ".[cutlass-dsl]"
```

Set the dataset path (only needed for `spawn.py`; child environments store it in `config.toml`):

```bash
export FIB_DATASET_PATH=/path/to/flashinfer-trace
```

> **Note**: Keep this environment activated when running `spawn.py` and when launching `claude` in child environments. The benchmark scripts use `python` from `PATH`, so the activated environment must contain all dependencies.

### Verify Installation

Check CUPTI profiling support (requires `cupti-python >= 13.0.0` and CUDA 13.0+ driver):

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

## Modal Backend Setup

For cloud benchmarking on Modal:

```bash
# Authenticate with Modal
modal setup

# Create and upload trace volume
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/flashinfer-trace/
```

See [Usage Guide](usage.md) for how to spawn with `--backend modal`.

## Common Issues

If you encounter CUPTI driver mismatches, TVM FFI linking errors, or flashinfer-bench version problems, see [Troubleshooting](troubleshooting.md).

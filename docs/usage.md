# Usage Guide

> **Note**: AKO4FIB currently ships with [Claude Code](https://docs.anthropic.com/en/docs/claude-code) as the default agent. Community contributions for other agents are welcome — see the [roadmap](../README.md#contributing).

## spawn.py Options

```bash
python spawn.py \
  --operator <name> \
  [--dataset /path/to/dataset] \    # default: $FIB_DATASET_PATH
  [--mode scratch|existing] \       # default: scratch
  [--backend local|modal] \         # default: local
  [--gpu <name>] \                  # local: auto-detected; modal: required
  [--agent claude] \                # currently only claude is shipped
  [--kernel /path/to/kernel.py] \   # required for existing mode
  [--name <label>]                  # optional label for the run
```

## Examples

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

# Custom dataset (see https://bench.flashinfer.ai/docs/tutorials/bring-your-own-kernel)
python spawn.py --operator my_custom_op --dataset /path/to/my/traceset
```

## Modes

### Scratch Mode (default)

The PyTorch reference implementation is extracted from the dataset to `solution/kernel.py`. The agent optimizes it in-place (Python/Triton) or rewrites it in any supported language (CUDA, C++, TileLang, CuTe DSL).

### Existing Mode

When using `--mode existing --kernel /path/to/kernel`:

- **Single file**: `--kernel /path/to/kernel.py` copies the file plus siblings (`.py`, `.cu`, `.cpp`, `.h` files) to `solution/`
- **Directory**: `--kernel /path/to/dir/` copies all files in the directory to `solution/`
- **Colocated config.toml**: If a `config.toml` exists alongside the kernel, it is used directly (useful for pre-configured language/entry_point/DPS settings)
- **Otherwise**: Language is inferred from file extension (`.cu`→cuda, `.cpp`→cpp, `.py`→python)

## Language Selection

The child environment supports multiple kernel languages. By default, the agent selects a language based on the operator characteristics, but you can guide this choice via `HINTS.md` or your initial prompt.

### Supported Languages

| Language | Files in `solution/` | config.toml `language` | config.toml `entry_point` |
|----------|---------------------|----------------------|--------------------------|
| Python / PyTorch | `kernel.py` | `python` | `kernel.py::run` |
| Triton | `kernel.py` | `triton` | `kernel.py::run` |
| TileLang | `kernel.py` | `tilelang` | `kernel.py::run` |
| CuTe DSL | `kernel.py` | `python` | `kernel.py::run` |
| CUDA | `kernel.cu` + `binding.py` | `cuda` | `binding.py::kernel` |
| C++ | `kernel.cpp` + `binding.py` | `cpp` | `binding.py::kernel` |

### Package Requirements

Some languages require extra packages:

- **TileLang**: `pip install tilelang` — without it, solutions report COMPILE_ERROR
- **CuTe DSL**: `pip install nvidia-cutlass-dsl` — CuTe DSL kernels are Python-defined (`@CuTeDSL.kernel` decorator) and compiled to PTX at runtime

Per-language documentation is available in `docs/languages/` within each child environment.

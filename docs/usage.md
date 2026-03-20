# Usage Guide

## spawn.py Options

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

# Custom dataset
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

The child environment is **language-agnostic**. The agent chooses the kernel language at runtime and updates `config.toml` accordingly.

### Supported Languages

| Language | Files in `solution/` | config.toml `language` | config.toml `entry_point` |
|----------|---------------------|----------------------|--------------------------|
| Python / PyTorch | `kernel.py` | `python` | `kernel.py::run` |
| Triton | `kernel.py` | `triton` | `kernel.py::run` |
| TileLang | `kernel.py` | `tilelang` | `kernel.py::run` |
| CuTe DSL | `kernel.py` | `python` | `kernel.py::run` |
| CUDA | `kernel.cu` + `binding.py` | `cuda` | `binding.py::kernel` |
| C++ | `kernel.cpp` + `binding.py` | `cpp` | `binding.py::kernel` |

For compiled languages (CUDA, C++), `binding.py` uses TVM FFI (`from tvm.ffi import register_func`) to expose compiled code to Python. Alternatively, use `kernel.cu::symbol_name` with `TVM_FFI_DLL_EXPORT_TYPED_FUNC` to export the function directly.

### Package Requirements

Some languages require extra packages:

- **TileLang**: `pip install tilelang` — without it, solutions report COMPILE_ERROR
- **CuTe DSL**: `pip install nvidia-cutlass-dsl` — CuTe DSL kernels are Python-defined (`@CuTeDSL.kernel` decorator) and compiled to PTX at runtime
- **Triton**: Pre-installed with PyTorch, no extra installation needed

### Guiding Language Choice

To steer the agent toward a specific language, edit `HINTS.md` in the child environment before launching the agent:

```bash
python spawn.py --operator my_op
cd ../ako4fib-run-*/
# Open HINTS.md and add your guidance, e.g.:
#   "Use Triton for this kernel."
#   "Start with the PyTorch reference and optimize with CUDA."
claude
```

You can also include language guidance directly in your initial prompt to the agent.

Per-language documentation is available in `docs/languages/` within each child environment.

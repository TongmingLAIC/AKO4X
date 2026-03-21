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

## Benchmark Tuning

Benchmark iteration counts are configured in the `[benchmark]` section of `config.toml` in the child environment:

```toml
[benchmark]
baseline_iterations = 100   # reference profiling iterations
solution_iterations = 100   # solution profiling iterations
num_trials = 5
warmup_runs = 3
```

CUPTI-based profiling can be slow. To speed up iteration during development, you can reduce `baseline_iterations` and `solution_iterations`, or trim `docs/workloads.jsonl` to benchmark fewer workloads. Both the user and the agent can edit these values.

> **Modal backend note**: On Modal, CUPTI activity tracing with high iteration counts on slow kernels (e.g. the PyTorch reference) can cause CUPTI buffer bloat, leading to inflated and inaccurate timings. If you observe this, lower `baseline_iterations` (e.g. to `20`) in `config.toml`. This issue does not affect local benchmarking.

## Agent Permissions

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

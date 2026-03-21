# AKO4FIB — Agentic Kernel Optimization for FlashInfer-Bench

A tool for automated GPU kernel optimization using Code Agents (e.g. Claude Code). Built on the [flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench) SDK, it supports any flashinfer-bench compatible dataset, including the official [FlashInfer-Trace](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace) and user-defined operators.

## Quick Install

```bash
pip install .

# Download a dataset (LFS pointers only — tensors are fetched on demand)
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace
# Or the MLSys 2026 Contest subset: https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest

export FIB_DATASET_PATH=/path/to/flashinfer-trace
```

See [Installation](docs/installation.md) for prerequisites, optional extras (Modal, TileLang, CuTe DSL), and verification steps.

## Quick Start

> **Note**: Run all commands below — including `claude` — in the same Python environment where you installed AKO4FIB, so the agent can access all dependencies.

```bash
# List available operators
python spawn.py

# Create an optimization environment
python spawn.py --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 --name my_run

# Enter the child environment and start optimizing
cd ../ako4fib-run-my_run
claude
# Send your initial prompt, e.g. "Optimize this kernel using Triton"
```

## How It Works

1. **Spawn** — `spawn.py` discovers an operator from the dataset and creates an isolated child environment with task spec (`CLAUDE.md`), benchmark scripts, and reference kernel

2. **Configure** (optional) — Edit `HINTS.md` in the child environment to provide guidance: language preference, optimization strategies, or constraints

3. **Run the agent** — Launch `claude` in the child environment and send your initial prompt. The agent reads its task spec which contains the operator definition, workload distribution, and optimization instructions. You can also provide additional direction in your initial prompt

4. **Iterate** — The agent writes or modifies the kernel, benchmarks it, analyzes per-workload results, and iterates toward higher speedup. Each benchmark run is automatically saved to `trajectory/` with the kernel snapshot and scores, so no progress is lost

The primary metric is **mean speedup** across all workloads against the PyTorch reference.

## Features

- **6 kernel languages**: Python/PyTorch, Triton, CUDA, C++, TileLang, CuTe DSL (Code Agent will automatically select the best language if not specified)
- **2 backends**: Local GPU execution and [Modal](https://modal.com/) cloud
- **NCU profiling**: Integrated NVIDIA Nsight Compute profiling (local backend)
- **Baseline caching**: Automatic reference profiling with cached results
- **Trajectory tracking**: Every benchmark run saves kernel snapshot and results for review
- **Git integration**: Each child environment is a git repo; the agent commits progress automatically
- **Custom datasets**: Any flashinfer-bench compatible trace set (see [Bring Your Own Kernel](https://bench.flashinfer.ai/docs/tutorials/bring-your-own-kernel)). You can also use the Code Agent to help create custom datasets.
- **Two modes**: Write from scratch or optimize an existing kernel

## Documentation

| Guide | Description |
|-------|-------------|
| [Installation](docs/installation.md) | Prerequisites, installation, Modal setup |
| [Usage Guide](docs/usage.md) | spawn.py options, modes, language selection |
| [Child Environment](docs/child-environment.md) | Structure, config.toml, benchmarking, profiling |
| [Troubleshooting](docs/troubleshooting.md) | CUPTI, TVM FFI, version issues |

## Contributing

We welcome contributions! For architecture details and development guidance, see [CLAUDE.md](CLAUDE.md).

**Roadmap** — areas we'd like to develop with the community:

- Support for more code agents beyond Claude Code (see [Adding a New Agent Type](CLAUDE.md#adding-a-new-agent-type))
- Additional kernel languages and backends
- Unify `scratch`/`existing` modes — let the agent decide based on context

Check [open issues](../../issues) for specific tasks, or open a new one to discuss your ideas.

## License

License information will be added before the first public release.

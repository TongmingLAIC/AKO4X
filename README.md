# AKO4FIB — Agentic Kernel Optimization for FlashInfer-Bench

Give it an operator, get back a faster GPU kernel — automatically. AKO4FIB uses Code Agents (e.g. Claude Code) to write, benchmark, and iteratively optimize GPU kernels. Built on [flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench).

## Quick Start

```bash
# Install
pip install .

# Download a dataset (LFS pointers only — tensors are fetched on demand)
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace
export FIB_DATASET_PATH=/path/to/flashinfer-trace

# Spawn an optimization environment
python spawn.py --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 --name my_run

# Enter the child environment and start optimizing
cd ../ako4fib-run-my_run
claude
# Send your initial prompt, e.g. "Optimize this kernel using Triton"
```

> **CUDA Driver >= 13.0 required.** Check with `nvidia-smi`. See [Installation](docs/installation.md) for prerequisites, optional extras, and verification steps.

## How It Works

1. **Spawn** — `spawn.py` creates an isolated environment with the operator spec, benchmark scripts, and a reference kernel
2. **Configure** (optional) — Edit `HINTS.md` to guide the agent: language preference, optimization strategies, constraints
3. **Run** — Launch `claude` and send your initial prompt. The agent reads its task spec and begins optimizing
4. **Iterate** — The agent modifies the kernel, benchmarks, analyzes results, and repeats until latency stops improving

## Features

- **6 kernel languages**: Python/PyTorch, Triton, CUDA, C++, TileLang, CuTe DSL — the agent selects automatically or you choose via `HINTS.md`
- **2 execution backends**: Local GPU or [Modal](https://modal.com/) cloud
- **2 modes**: Write from scratch or optimize an existing kernel (`--mode existing`)
- **NCU profiling**: Integrated NVIDIA Nsight Compute profiling for bottleneck analysis (local backend)
- **Progress monitoring**: Track optimization via `ITERATIONS.md`, git history (`git log --oneline`), and per-run snapshots in `trajectory/`
- **Custom datasets**: Any [flashinfer-bench compatible trace set](https://bench.flashinfer.ai/docs/tutorials/bring-your-own-kernel), including user-defined operators

## Documentation

| Guide | What it covers |
|-------|---------------|
| [Installation](docs/installation.md) | Prerequisites, optional extras (Modal, TileLang, CuTe DSL), verification |
| [Usage Guide](docs/usage.md) | spawn.py options, modes, language selection |
| [Benchmarking](docs/benchmarking.md) | Benchmark commands, output interpretation, baseline caching, iteration tuning |
| [Agent Guide](docs/agent-guide.md) | Agent permissions, collaboration tips, resuming and extending sessions |
| [Child Environment](docs/child-environment.md) | Directory structure and config.toml reference |
| [Troubleshooting](docs/troubleshooting.md) | CUPTI driver mismatches, flashinfer-bench versions, TVM FFI linking |

## Contributing

We welcome contributions! For architecture details and development guidance, see [CLAUDE.md](CLAUDE.md).

**Roadmap:**

- Support for more code agents beyond Claude Code (see [Adding a New Agent Type](CLAUDE.md#adding-a-new-agent-type))
- Additional kernel languages and backends
- Unify `scratch`/`existing` modes — let the agent decide based on context

Check [open issues](../../issues) for specific tasks, or open a new one to discuss your ideas.

## License

This project is licensed under the [MIT License](LICENSE).

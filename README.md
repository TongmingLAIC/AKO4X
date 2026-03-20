# AKO4FIB — Agentic Kernel Optimization for FlashInfer-Bench

A tool for automated GPU kernel optimization using Code Agents (e.g. Claude Code). Built on the [flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench) SDK, it supports any flashinfer-bench compatible dataset, including the official [FlashInfer-Trace](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace) and user-defined operators.

## Quick Install

```bash
pip install .
export FIB_DATASET_PATH=/path/to/flashinfer-trace
```

For Modal backend, TileLang, or CuTe DSL extras, see [Getting Started](docs/getting-started.md).

## Quick Start

```bash
# List available operators
python spawn.py

# Create an optimization environment
python spawn.py --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64

# Enter the child environment and start your code agent
cd ../ako4fib-run-001
claude
```

## How It Works

1. **Spawn** — `spawn.py` discovers an operator from the dataset and creates an isolated child environment with task spec, benchmark scripts, and reference kernel
2. **Optimize** — The agent reads the task spec and hints, studies the operator definition, writes or modifies a kernel, benchmarks it, and iterates toward higher speedup
3. **Measure** — Benchmarks report per-workload speedup against the PyTorch reference; the primary metric is the mean speedup across all workloads

## Features

- **6 kernel languages**: Python/PyTorch, Triton, CUDA, C++, TileLang, CuTe DSL
- **2 backends**: Local GPU execution and [Modal](https://modal.com/) cloud
- **NCU profiling**: Integrated NVIDIA Nsight Compute profiling (local backend)
- **Baseline caching**: Automatic reference profiling with cached results
- **Custom datasets**: Any flashinfer-bench compatible trace set
- **Two modes**: Write from scratch or optimize an existing kernel

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Prerequisites, installation, Modal setup |
| [Usage Guide](docs/usage.md) | spawn.py options, modes, language selection |
| [Child Environment](docs/child-environment.md) | Structure, config.toml, benchmarking, profiling |
| [Custom Datasets](docs/custom-datasets.md) | Dataset schema and reference requirements |
| [Troubleshooting](docs/troubleshooting.md) | CUPTI, TVM FFI, version issues |

## Contributing

For architecture details and development guidance, see [CLAUDE.md](CLAUDE.md).

## License

License information will be added before the first public release.

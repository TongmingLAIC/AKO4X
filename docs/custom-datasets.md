# Custom Datasets

Any flashinfer-bench compatible trace set works with AKO4FIB. The expected structure:

```
dataset/
├── definitions/
│   └── <type>/
│       └── <operator>.json
└── workloads/
    └── <type>/
        └── <operator>.jsonl
```

## Reference Field

Your `<operator>.json` must contain a `reference` field with **plain PyTorch code**:

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

# Python / PyTorch

Single file `solution/kernel.py` with a `run()` function:

```python
import torch

@torch.no_grad()
def run(input, weight):
    # Your implementation using PyTorch ops
    return output
```

Plain PyTorch — no special compilation step, no extra packages.

## Config

- `config.toml` language: `python`
- `config.toml` entry_point: `kernel.py::run`

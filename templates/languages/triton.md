# Triton

Single file `solution/kernel.py`. Define kernels with `@triton.jit` and call them from `run()`:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(x_ptr, out_ptr, n: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)

@torch.no_grad()
def run(input, weight):
    output = torch.empty_like(input)
    n = input.numel()
    _kernel[(n + 1023) // 1024,](input, output, n)
    return output
```

## Config

- `config.toml` language: `triton`
- `config.toml` entry_point: `kernel.py::run`

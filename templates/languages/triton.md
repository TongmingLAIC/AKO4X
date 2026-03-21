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

## Autotune Tips

- **See selected configs**: `TRITON_PRINT_AUTOTUNING=1 bash scripts/bench.sh` prints which config autotune picked for each kernel invocation.
- **Variance from re-selection**: If benchmark scores fluctuate (>5%), check whether autotune is picking different configs across runs. If so, hardcode the best config and remove `@triton.autotune`.
- **In-place / DPS kernels**: Autotune trials reuse the same input tensors. If the kernel writes in-place, earlier trials corrupt inputs for later ones — use `restore_value` on the affected pointer arguments.
- **Stale compilation cache**: Benchmark and profiler use a project-local Triton cache (`.triton_cache/`). If you suspect stale compiled kernels, `rm -rf .triton_cache/` to force recompilation.

## Config

- `config.toml` language: `triton`
- `config.toml` entry_point: `kernel.py::run`

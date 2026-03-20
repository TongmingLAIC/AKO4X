# CuTe DSL (NVIDIA CUTLASS Python DSL)

CuTe DSL kernels are Python-defined and compiled to PTX at runtime. Requires `nvidia-cutlass-dsl` package.

Single file `solution/kernel.py` with the same `run()` interface:

```python
import torch
from cutlass.cute.runtime import from_dlpack

@torch.no_grad()
def run(input, weight):
    # Convert torch tensors to CuTe tensors
    cute_input = from_dlpack(input)
    cute_weight = from_dlpack(weight)

    # Allocate output
    output = torch.empty_like(input)
    cute_output = from_dlpack(output)

    # Call CuTe DSL kernel (defined with @CuTeDSL.kernel or @CuTeDSL.jit)
    my_cute_kernel(cute_input, cute_weight, cute_output)

    return output
```

Key points:
- Use `@CuTeDSL.kernel` for fully compiled kernels or `@CuTeDSL.jit` for JIT compilation
- Convert between torch and CuTe tensors with `from_dlpack()`
- Torch tensors passed to CuTe DSL must be contiguous

## Config

- `config.toml` language: `python` (CuTe DSL compiles at runtime, uses Python language setting)
- `config.toml` entry_point: `kernel.py::run`

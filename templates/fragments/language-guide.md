## Language Reference

### Quick Reference

| Language | Files in `solution/` | `config.toml` language | `config.toml` entry_point |
|----------|---------------------|----------------------|--------------------------|
| Python / PyTorch | `kernel.py` | `python` | `kernel.py::run` |
| Triton | `kernel.py` | `triton` | `kernel.py::run` |
| TileLang | `kernel.py` | `tilelang` | `kernel.py::run` |
| CuTe DSL | `kernel.py` | `python` | `kernel.py::run` |
| CUDA | `kernel.cu` + `binding.py` | `cuda` | `binding.py::kernel` |
| C++ | `kernel.cpp` + `binding.py` | `cpp` | `binding.py::kernel` |

### Python / Triton / TileLang

Single file `solution/kernel.py` with a `run()` function:

```python
import torch

@torch.no_grad()
def run(input, weight):
    # Your implementation (Triton: @triton.jit kernels; TileLang: TileLang primitives)
    return output
```

TileLang requires `tilelang` package; CuTe DSL requires `nvidia-cutlass-dsl` package.

### CuTe DSL

Same `run()` pattern. Kernels use `@CuTeDSL.kernel` / `@CuTeDSL.jit` decorators, compiled to PTX at runtime. Convert torch tensors via `from cutlass.cute.runtime import from_dlpack`.

### CUDA

Two files: `solution/kernel.cu` and `solution/binding.py` (TVM FFI bindings).

```python
# binding.py
import torch
from tvm.ffi import register_func

@register_func("flashinfer.kernel")
def kernel(input, weight):
    output = torch.empty_like(input)
    # Launch CUDA kernel
    return output
```

Standard CUDA kernel in `kernel.cu`. Alternative: use entry_point `kernel.cu::symbol_name` with `TVM_FFI_DLL_EXPORT_TYPED_FUNC(symbol_name, function)` to export directly (bypassing binding.py).

### C++

Same as CUDA but with `solution/kernel.cpp`. Load CUDA runtime APIs via `dlopen("libcudart.so", RTLD_NOW)` / `dlsym`.

### TVM FFI Builder Limitations

- **No external library linking**: The builder does not pass `extra_ldflags`. Libraries like cuBLAS, cuDNN, CUTLASS cannot be linked at compile time.
- **Workaround**: Use `dlopen`/`dlsym` to load CUDA libraries at runtime. Prefer `cublasGemmEx` (`float*` alpha/beta) over `cublasHgemm` (`__half*` alpha/beta) to avoid ABI issues across dlopen boundaries.

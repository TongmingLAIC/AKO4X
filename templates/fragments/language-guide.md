## Language Guide

You may write your kernel in any supported language. Update `config.toml` to match before running `bash scripts/bench.sh`.

### Quick Reference

| Language | Files in `solution/` | `config.toml` language | `config.toml` entry_point | Extra packages |
|----------|---------------------|----------------------|--------------------------|----------------|
| Python / PyTorch | `kernel.py` | `python` | `kernel.py::run` | (none) |
| Triton | `kernel.py` | `triton` | `kernel.py::run` | (pre-installed) |
| TileLang | `kernel.py` | `tilelang` | `kernel.py::run` | `pip install tilelang` |
| CuTe DSL | `kernel.py` | `python` | `kernel.py::run` | `pip install nvidia-cutlass-dsl` |
| CUDA | `kernel.cu` + `binding.py` | `cuda` | `binding.py::kernel` | (none) |
| C++ | `kernel.cpp` + `binding.py` | `cpp` | `binding.py::kernel` | (none) |

### Python / Triton / TileLang Pattern

Single file `solution/kernel.py` with a `run()` function:

```python
import torch

@torch.no_grad()
def run(input, weight):
    # Your implementation
    return output
```

- **Triton**: Use `@triton.jit` kernels called from `run()`.
- **TileLang**: Use `TileLang` primitives; requires `pip install tilelang` in the `fi-bench` env.

### CuTe DSL Pattern

CuTe DSL kernels are Python-defined (`@CuTeDSL.kernel` / `@CuTeDSL.jit` decorators) and compiled to PTX at runtime. Use `language = "python"` in config.toml.

```python
import torch
from cutlass.cute.runtime import from_dlpack

@torch.no_grad()
def run(input, weight):
    # Convert tensors: cute_tensor = from_dlpack(torch_tensor)
    # Call CuTe DSL kernel
    return output
```

Requires `pip install nvidia-cutlass-dsl` in the `fi-bench` env.

### CUDA Pattern

Two files: `solution/kernel.cu` (kernel code) and `solution/binding.py` (Python bindings via TVM FFI).

**binding.py:**
```python
import torch
from tvm.ffi import register_func

@register_func("flashinfer.kernel")
def kernel(input, weight):
    # Allocate output
    output = torch.empty_like(input)
    # Launch CUDA kernel (via ctypes or torch custom ops)
    return output
```

**kernel.cu:**
```cuda
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void my_kernel(const half* input, half* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}
```

The TVM FFI builder compiles `.cu` files automatically. Alternatively, use entry_point `kernel.cu::symbol_name` with `TVM_FFI_DLL_EXPORT_TYPED_FUNC(symbol_name, function)` to export the function directly (bypassing binding.py).

### C++ Pattern

Same as CUDA but with `solution/kernel.cpp`. Use `language = "cpp"` in config.toml.

To call CUDA runtime APIs from C++, load them via `dlopen`/`dlsym`:
```cpp
void* handle = dlopen("libcudart.so", RTLD_NOW);
```

### TVM FFI Builder Limitations

- **No external library linking**: The builder does not pass `extra_ldflags` to the linker. Libraries like cuBLAS, cuDNN, CUTLASS cannot be linked at compile time.
- **Workaround**: Use `dlopen`/`dlsym` to load CUDA libraries at runtime from within `.cu`/`.cpp` files. For example, load cuBLAS via `dlopen("libcublas.so.12", RTLD_NOW)` + `dlsym`. Prefer `cublasGemmEx` (accepts `float*` alpha/beta) over `cublasHgemm` (accepts `__half*` alpha/beta) to avoid ABI issues across shared library boundaries.

### Destination Passing Style (DPS)

In `config.toml`, `destination_passing_style = false` (default) means your `run()`/`kernel()` function allocates and returns output tensors. Set to `true` to receive pre-allocated output tensors as additional arguments (appended after input args).

### Important Notes

- **Update config.toml before benchmarking**: Set `language` and `entry_point` to match your kernel files.
- **When switching languages**: Remove old files from `solution/` to avoid confusion.
- **Reference implementation**: The pure-Python reference is in `docs/definition.json` (under `"reference"`). Study it to understand the operator semantics.

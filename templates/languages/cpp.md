# C++

Two files: `solution/kernel.cpp` (kernel code) and `solution/binding.py` (Python bindings via TVM FFI).

## binding.py

```python
import torch
from tvm.ffi import register_func

@register_func("kernel")
def kernel(input, weight):
    output = torch.empty_like(input)
    # Call C++ compiled code
    return output
```

## kernel.cpp

Standard C++ with access to CUDA headers. To call CUDA runtime APIs, load them via `dlopen`/`dlsym`:

```cpp
#include <dlfcn.h>

void* cuda_rt = dlopen("libcudart.so", RTLD_NOW);
auto cudaMalloc_fn = (decltype(&cudaMalloc))dlsym(cuda_rt, "cudaMalloc");
```

The TVM FFI builder compiles `.cpp` files automatically.

## TVM FFI Builder Limitations

- **No external library linking**: The builder does not pass `extra_ldflags` to the linker. Libraries like cuBLAS, cuDNN, CUTLASS cannot be linked at compile time.
- **Workaround**: Use `dlopen`/`dlsym` to load CUDA libraries at runtime from within the `.cpp` file:

```cpp
#include <dlfcn.h>

void* cublas_handle = dlopen("libcublas.so.12", RTLD_NOW);
auto cublasGemmEx_fn = (decltype(&cublasGemmEx))dlsym(cublas_handle, "cublasGemmEx");
```

- Prefer `cublasGemmEx` (accepts `float*` alpha/beta) over `cublasHgemm` (accepts `__half*` alpha/beta) — the `__half` type may have ABI issues across shared library boundaries via dlopen.

## Config

- `config.toml` language: `cpp`
- `config.toml` entry_point: `binding.py::kernel`

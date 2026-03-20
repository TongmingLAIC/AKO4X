# CUDA

Two files: `solution/kernel.cu` (kernel code) and `solution/binding.py` (Python bindings via TVM FFI).

## binding.py

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

## kernel.cu

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

The TVM FFI builder compiles `.cu` files automatically.

## Alternative: Direct Symbol Export

Instead of `binding.py`, you can export the function directly from `.cu` using `TVM_FFI_DLL_EXPORT_TYPED_FUNC`:

```cuda
#include <tvm/runtime/c_runtime_api.h>

void my_kernel_impl(DLTensor* input, DLTensor* output) {
    // Launch kernel using input->data, output->data
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_kernel, my_kernel_impl);
```

Then set entry_point to `kernel.cu::my_kernel` in config.toml. The `binding.py` stub is packed but not used.

## TVM FFI Builder Limitations

- **No external library linking**: The builder does not pass `extra_ldflags` to the linker. Libraries like cuBLAS, cuDNN, CUTLASS cannot be linked at compile time.
- **Workaround**: Use `dlopen`/`dlsym` to load CUDA libraries at runtime from within the `.cu` file:

```cuda
#include <dlfcn.h>

void* cublas_handle = dlopen("libcublas.so.12", RTLD_NOW);
auto cublasGemmEx_fn = (decltype(&cublasGemmEx))dlsym(cublas_handle, "cublasGemmEx");
```

- Prefer `cublasGemmEx` (accepts `float*` alpha/beta) over `cublasHgemm` (accepts `__half*` alpha/beta) — the `__half` type may have ABI issues across shared library boundaries via dlopen.

## Config

- `config.toml` language: `cuda`
- `config.toml` entry_point: `binding.py::kernel` (default) or `kernel.cu::symbol_name` (direct export)

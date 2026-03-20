# CUDA

Two entry point patterns — direct symbol export (recommended) or Python binding.
The TVM FFI builder compiles `.cu` files automatically in both cases.

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

## Recommended: Direct Symbol Export

Export the function directly from `.cu` using `TVM_FFI_DLL_EXPORT_TYPED_FUNC`.
The wrapper function receives `tvm::ffi::TensorView` args (use `.data_ptr()`, `.size(dim)` to access data).
Requires `destination_passing_style = true` — outputs are passed as trailing arguments.

```cuda
#include <tvm/ffi/container/tensor.h>

__global__ void my_cuda_kernel(const half* in, half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}

void my_kernel(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
    int n = input.size(0);
    my_cuda_kernel<<<(n+255)/256, 256>>>(
        static_cast<const half*>(input.data_ptr()),
        static_cast<half*>(output.data_ptr()), n);
    cudaDeviceSynchronize();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_kernel, my_kernel);
```

Then set `entry_point = "kernel.cu::my_kernel"` and `destination_passing_style = true` in config.toml.

## Alternative: Python Binding

Use `binding.py` when you need full Python control (e.g., calling cuBLAS via dlopen, or torch custom ops).
You handle kernel launching yourself; the compiled `.cu` is loaded but not directly wired to the binding.

```python
import torch
from tvm.ffi import register_func

@register_func("kernel")
def kernel(A, B):
    output = torch.empty(A.shape[0], B.shape[0], device=A.device, dtype=A.dtype)
    # Launch via ctypes, dlopen, torch custom ops, etc.
    return output
```

Set `entry_point = "binding.py::kernel"` and `destination_passing_style = false` in config.toml.

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
- Direct export: `entry_point = "kernel.cu::symbol_name"`, `destination_passing_style = true`
- Python binding: `entry_point = "binding.py::kernel"`, `destination_passing_style = false`

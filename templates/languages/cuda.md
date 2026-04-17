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

## Tips

- **bfloat16 intrinsics**: The TVM FFI builder does not add `-D__CUDA_NO_BFLOAT16_CONVERSIONS__`, so `__nv_bfloat162float` and related intrinsics work out of the box. If you hit bfloat16 conversion errors, make sure you're using `language = "cuda"`, not `language = "python"` with `torch.utils.cpp_extension` (which adds that define by default).
- **Mixed Triton + CUDA**: Use `language = "python"` with `kernel.py` that imports both Triton and loads CUDA via `cpp_extension`. Place `cpp_extension.load_inline()` at module level (outside `run()`) so the `.so` is compiled once and available across benchmark subprocess invocations.
- **`load_inline(name=...)` caches by name, NOT by source**: `torch.utils.cpp_extension.load_inline` writes the compiled `.so` to `~/.cache/torch_extensions/<name>/` and on subsequent imports re-uses whatever is there if the **name** matches. If you edit the CUDA source but keep the same `name=`, the cache may still serve the *old* binary. Two ways this bites:
  - **When preserving multiple variants side-by-side** (e.g. under `reference/<op>/variants/`), give each variant a unique `name=` so they don't alias each other (e.g., `name="radix_topk_graph"` vs `name="radix_topk_no_graph"`).
  - **When iterating in a single Python process** (unusual inside the isolated-runner bench, but common in ad-hoc debugging), bump the `name=` after each meaningful source change (or `rm -rf ~/.cache/torch_extensions/<name>/`) to guarantee a fresh compile.
  Benchmarks run in fresh subprocesses (see `use_isolated_runner`), so this doesn't affect `bash scripts/bench.sh`, but it does affect interactive REPL / notebook use.
- **Which path to use**: Pure CUDA → `language = "cuda"` (TVM FFI, recommended). Need PyTorch ops or mixed Triton + CUDA → `language = "python"`.

## Config

- `config.toml` language: `cuda`
- Direct export: `entry_point = "kernel.cu::symbol_name"`, `destination_passing_style = true`
- Python binding: `entry_point = "binding.py::kernel"`, `destination_passing_style = false`

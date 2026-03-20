# Troubleshooting

## CUPTI Driver Mismatch

Local benchmarking uses CUPTI profiling by default via `cupti-python`. There are two fallback scenarios:

- **`cupti-python` not installed or version < 13.0**: Benchmarking gracefully falls back to CUDA events at import time. This works fine but produces noisier measurements, especially with low iteration counts.
- **`cupti-python >= 13.0` installed, but CUDA driver < 13.0**: CUPTI calls fail at runtime with `NotSupportedError`, causing all workloads to report RUNTIME_ERROR. This is **not** caught by the current fallback logic.

**Workarounds**: Ensure CUDA driver >= 13.0 when `cupti-python >= 13.0` is installed, or use the Modal backend (`--backend modal`). An upstream fix in flashinfer-bench is pending.

## flashinfer-bench Version

`pyproject.toml` pins `flashinfer-bench` to the git main branch (not PyPI's v0.1.2, which lags behind). Both `flashinfer-bench` and `cupti-python>=13.0.0` are installed automatically via `pip install .`.

## TVM FFI Builder: No External Library Linking

The TVMFFIBuilder (used for CUDA/C++ solutions) compiles `.cu`/`.cpp` files via `tvm_ffi.cpp.build()` but does not pass `extra_ldflags` to the linker. Libraries like cuBLAS, cuDNN, etc. cannot be linked at compile time.

**Workaround**: Use `dlopen`/`dlsym` to load CUDA libraries at runtime from within the `.cu` file. For example, `cublasGemmEx` can be called via `dlopen("libcublas.so.12", RTLD_NOW)` + `dlsym`. Prefer `cublasGemmEx` (accepts `float*` alpha/beta) over `cublasHgemm` (accepts `__half*` alpha/beta) — the `__half` type may have ABI issues across shared library boundaries via dlopen.

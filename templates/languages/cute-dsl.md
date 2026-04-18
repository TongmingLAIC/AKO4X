# CuTe DSL (NVIDIA CUTLASS Python DSL)

Python-defined kernels that compile to PTX at runtime. Requires `nvidia-cutlass-dsl` package. Install in the Modal image with `--no-deps` (see the build-image commit history) to avoid downgrading `apache-tvm-ffi` and breaking TileLang.

Single file `solution/kernel.py` with the standard `run()` interface:

```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def _my_kernel(X: cute.Tensor, Y: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    # ... per-thread logic ...

@cute.jit
def _my_jit(X: cute.Tensor, Y: cute.Tensor, n: cutlass.Constexpr):
    _my_kernel(X, Y).launch(grid=((n + 127) // 128, 1, 1), block=(128, 1, 1))

@torch.no_grad()
def run(input, weight):
    output = torch.empty_like(input)
    _my_jit(from_dlpack(input), from_dlpack(output), input.shape[0])
    return output
```

## AKO4FIB / FlashInfer-specific notes

- **Tensor conversion**: `from_dlpack(torch_tensor, assumed_align=N)` — default alignment is element-size (2 bytes for bf16). Required `assumed_align=16` for `cp.async` 128-bit copy atoms, otherwise ncu fails with `src ptr alignment (16 bits) does not meet requirement (128 bits)`.
- **Probing that the runtime works**: `flashinfer-bench` loads the `cutlass` module at kernel-build time; if `import cutlass.cute` fails the whole bench reports `COMPILE_ERROR`. Fail-loud inside the kernel (e.g., `assert _CUTE_OK` after an `import … except` guard) to surface the real error message instead of a generic import failure.
- **Decorator APIs actually used on the Modal image** (versions on the `nvidia-cutlass-dsl>=4.3.4` pip wheel):
  - `@cute.kernel` — the device-side function
  - `@cute.jit` — the host-side JIT launcher that calls `.launch(grid=..., block=...)`
  - `cute.arch.thread_idx()`, `cute.arch.block_idx()`, `cute.arch.sync_threads()`, `cute.arch.cp_async_commit_group()`, `cute.arch.cp_async_wait_group(N)` — low-level device intrinsics
  - `cute.math.exp`, `cute.math.log`, `cute.math.exp2` — **note**: `cute.arch` does **not** have `log`

## Config

- `config.toml` language: `python` (CuTe DSL compiles at runtime, uses the Python builder path)
- `config.toml` entry_point: `kernel.py::run`

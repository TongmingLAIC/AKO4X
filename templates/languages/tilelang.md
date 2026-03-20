# TileLang

Single file `solution/kernel.py`. Requires `tilelang` package — without it, all workloads report COMPILE_ERROR.

TileLang uses a factory pattern: an outer function decorated with `@tilelang.jit` returns a `@T.prim_func` kernel. The JIT compiler infers the target (CUDA/HIP) from input tensors at first call.

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def my_kernel_factory(M: int, N: int, block_size: int = 256, dtype: str = "float16"):
    @T.prim_func
    def my_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M * N, block_size), threads=block_size) as bx:
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < M * N:
                    C[idx] = A[idx] + B[idx]
    return my_kernel

@torch.no_grad()
def run(input, weight):
    M, N = input.shape
    output = torch.empty_like(input)
    # First call compiles; subsequent calls reuse the compiled kernel
    my_kernel_factory(M, N, dtype=str(input.dtype).split(".")[-1])(input, weight, output)
    return output
```

Key points:
- `@T.prim_func` defines the kernel as a TIR function with typed tensor signatures (`T.Tensor`)
- `@tilelang.jit` wraps a factory function that captures compile-time parameters (shapes, tile sizes, dtypes)
- Use `T.Kernel(grid, threads=N)` to specify launch configuration
- Use `T.Parallel`, `T.Serial` for loop annotations and `T.alloc_shared()`, `T.alloc_fragment()` for memory
- High-level ops: `T.gemm()`, `T.copy()`, `T.reduce()` for common tile patterns

## Config

- `config.toml` language: `tilelang`
- `config.toml` entry_point: `kernel.py::run`

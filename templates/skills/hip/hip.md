# HIP — Detailed Reference

Companion to `SKILL.md`. Reference for writing HIP C++ kernels on AMD Instinct
GPUs (gfx942 = MI300X/MI325X, gfx950 = MI350X/MI355X). Content here is drawn
from validated kernel work, not speculation — claims are tied to specific
patterns that have been measured.

## Entry point: hipcc-built extension via torch

The reliable pattern is a `.hip` device-code file with a sibling pybind11
`binding.cpp`, both loaded at runtime via `torch.utils.cpp_extension.load`:

```python
from torch.utils.cpp_extension import load
mod = load(
    name="my_kernel",
    sources=["kernel.hip", "binding.cpp"],
    extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
)
def kernel(x, w, eps):
    y = torch.empty_like(x)
    mod.launch(x, w, y, eps)
    return y
```

`binding.cpp` exposes the launcher to torch:

```cpp
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

extern "C" void my_launch(const void* x, const void* w, void* y,
                          int n_rows, int hidden, float eps, hipStream_t s);

void launch(torch::Tensor x, torch::Tensor w, torch::Tensor y, double eps) {
    my_launch(x.data_ptr(), w.data_ptr(), y.data_ptr(),
              int(x.size(0)), int(x.size(1)), float(eps),
              c10::hip::getCurrentHIPStream().stream());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch, "HIP kernel launcher");
}
```

Always pass an explicit stream from `c10::hip::getCurrentHIPStream().stream()`
— do not rely on the null stream when the kernel may be launched under
`hipStreamBeginCapture` / `hipGraphLaunch`.

## Wave size — fixed at 64 on CDNA

All CDNA GPUs (gfx9xx) run Wave64. Block sizes must be multiples of 64.

Practical defaults:
- Element-wise / single-pass reductions: 256 threads (4 waves) per block.
- MFMA-heavy GEMMs at moderate tile size: **8 waves/block (512 threads)** —
  measured 1.35× speedup over 4-wave on an FP8 GEMM at BM=BN=128, attributed
  to latency hiding across more in-flight waves per CU.
- Decode-style attention with small per-query work: 64 threads (1 wave).

## The bf16 / fp8 vector-load trap

Clang on AMD rejects `ext_vector_type(8)` over `__hip_bfloat16` and `__hip_fp8_*`
types. Additionally, ROCm 7.0's `__hip_fp8_e4m3_*::operator float()` is marked
`__FP8_HOST__` only — calling it from `__device__` code is a compile error.

**Idiom**: vector loads/stores via `uint4` + `union`, scalar decode inline.

```cpp
typedef union { uint4 u; __hip_bfloat16 bf[8]; } bf16x8_t;

// 128-bit vectorized load:
bf16x8_t v;
v.u = *reinterpret_cast<const uint4*>(x_ptr + offset);
#pragma unroll
for (int k = 0; k < 8; ++k) {
    float f = static_cast<float>(v.bf[k]);
    // ...
}
```

For FP8 e4m3fn on `__device__` code, decode the raw byte manually (fn format:
sign + 4-bit exp bias-7 + 3-bit mantissa, NaN = `s.1111.111`):

```cpp
__device__ __forceinline__ float fp8_e4m3fn_to_float(unsigned char raw) {
    unsigned int sign = raw & 0x80u;
    int exp  = (raw >> 3) & 0xF;
    unsigned int mant = raw & 0x7u;
    if (exp == 0xF && mant == 0x7) return __builtin_nanf("");
    float val = (exp == 0)
        ? mant * (1.0f / 512.0f)
        : (1.0f + mant * 0.125f) * exp2f(float(exp - 7));
    return sign ? -val : val;
}
```

## LDS bank conflicts — measured, not theoretical

LDS is 32 banks × 4 bytes wide on both CDNA archs (64 KiB total per CU).
Wave64 means 64 lanes hit the LDS per access, twice the contention surface
of NVIDIA's 32-lane warps.

**Worked example** (validated speedup): an FP8 GEMM with `__shared__
unsigned char A_lds[128][64]` (row stride = 64 B = 16 banks). Each MFMA
operand load reads 8 contiguous bytes from 16 consecutive rows at the same
column → all 16 rows map to the same 2 bank-pair groups (an 8-way conflict
that serializes each operand load).

**Fix**: add 8 bytes of padding per row (stride 72 B = 18 banks). Since
`gcd(18, 32) = 2`, 16 consecutive rows produce 16 distinct bank-pair offsets
→ conflict-free. Measured **1.76× speedup** on FP8 GEMM at (M=4096, N=7168,
K=2048) by adding `[BK+8]` to the LDS array. Occupancy dropped from 4 → 3
blocks/CU (18 KB × 3 = 54 KB ≤ 64 KB) but the conflict elimination dominated.

**General rule**: when `BK_bytes` shares a factor > 2 with 32, pad to break
the GCD. The exact pad depends on the access pattern's stride.

**Diagnose**: `rocprofv3 --pmc LDS_BANK_CONFLICT` (the rocprof equivalent of
NCU's `l1tex_data_bank_conflicts`).

## MFMA selection — dispatch overhead is not the bottleneck

Validated on FP8 GEMM at BM=BN=128 (MI355X): swapping
`mfma_f32_16x16x32_fp8_fp8` (32 dispatches per wave per outer iter) for
`mfma_f32_32x32x16_fp8_fp8` (16 dispatches per wave per outer iter, same
total FLOPS) yielded **zero speedup**. The bigger MFMA produces 4× more
output per dispatch but takes proportionally more cycles. At this tile
shape, neither MFMA dispatch overhead nor instruction count is the
bottleneck.

Both MFMAs are valid choices; pick based on register-pressure / scheduling
tradeoffs (32x32x16 needs `floatx16` accumulators = 16 fp32/lane; 16x16x32
needs `floatx4` = 4 fp32/lane).

## Documented dead-ends — do not re-attempt without a new mechanism

These hypotheses were tested in our campaign and rejected:

- **Software double-buffer (2-stage LDS pipeline)** — with regular global
  loads, the HIP compiler already issues loads ahead of MFMA work via
  instruction scheduling. Explicit 2-stage software pipelining does NOT add
  overlap that wasn't already there; doubles LDS footprint and hurts
  occupancy. To buy real overlap, you need
  `__builtin_amdgcn_global_load_lds_*` (direct DRAM → LDS, bypasses VGPR
  staging) — but that intrinsic writes to `M0 + lane_id × 4` contiguous LDS
  addresses, which is incompatible with the padded LDS layout used for the
  bank-conflict fix above. Resolving that requires a different LDS layout.

- **Naive persistent kernels on CDNA** — folding `N_rows >> CU_COUNT` worth
  of work into a fixed grid of `~CU_COUNT` long-lived blocks. The CDNA
  wavefront scheduler already hides launch overhead for short blocks, and
  reducing total block count starves the chip-level work-stealing /
  load-balancing the scheduler does. Persistent patterns pay off only when
  block launch is genuinely the dominant cost (small per-block work and
  thousands of launches per second).

- **Wider blocks (e.g. BM = 256) chasing DRAM reduction** — reducing
  redundant per-block global loads is real arithmetic, but the LDS-footprint
  increase often drops blocks/CU from 2 → 1 and total grid size below the
  CU count, which removes the scheduler's room to hide. Measured **slower**
  on FP8 GEMM despite ~25% less DRAM traffic.

## Useful __builtin_amdgcn intrinsics

- `__builtin_amdgcn_mfma_*` — matrix instructions. Signatures take packed
  operands (`long` = 8×FP8, etc.); see compiler headers for the exact
  per-shape signature.
- `__builtin_amdgcn_ds_bpermute(addr, val)` — cross-lane permute (the
  `__shfl_sync` equivalent on AMD).
- `__builtin_amdgcn_readlane(val, lane)` — broadcast from a specific lane.
- `__builtin_amdgcn_global_load_lds_*` — direct DRAM → LDS path on CDNA;
  bypasses VGPR pressure. **Caveat**: writes to contiguous LDS addresses
  starting at `M0`, which constrains the LDS layout (cannot be combined
  naively with padded-row swizzles — see Dead-ends above).

## hipcc flags worth knowing

- `--offload-arch=gfx950` (or `gfx942`) — required; default is host arch.
- `-O3` — usual perf flag (Torch's cpp_extension already passes this).
- `--save-temps` — dumps `.s` (gfx ISA). Use to verify MFMA selection.
- `-fno-gpu-rdc` — non-relocatable device code (default; faster compile;
  used by `torch.utils.cpp_extension.load`).

## Torch+ROCm runtime caveats

- **`torch._scaled_mm` / hipBLASLt** on torch 2.12+rocm7.1 has a known bug
  where small FP32 shapes like (64, 128, 2048) hit
  `HIPBLAS_STATUS_INVALID_VALUE`. Workaround: set
  `TORCH_BLAS_PREFER_HIPBLASLT=0` before importing torch. BF16 path on the
  same shapes works fine. Affects FP32 reference matmuls in benchmark
  drivers, not the FP8 path itself.

---
name: hip
description: HIP C++ (.hip) kernel reference for AMD Instinct GPUs (CDNA3 gfx942 = MI300X/MI325X, CDNA4 gfx950 = MI350X/MI355X). Covers wave64 fixed size, the LDS bank-conflict math (32 banks × 4B; stride/conflict relationship; padding fixes), MFMA tile selection for BF16/FP8, the bf16/fp8 vector-load trap (Clang's ext_vector_type rejects bf16 on AMD; use uint4 + union), occupancy/tile-size tradeoffs validated on MI355X, and dead-end patterns (persistent kernels on CDNA, naive 2-stage double-buffering when the compiler already overlaps). Use when writing or debugging hand-written HIP C++ kernels on Instinct hardware.
---

# HIP

Reference for HIP C++ kernel writers targeting AMD Instinct GPUs. Detailed
guide: `hip.md`.

## When to consult

- Writing a HIP `.hip` kernel — `hipcc` + a sibling `binding.cpp` (pybind11) or
  `binding.py` (Python C extension) loaded via `torch.utils.cpp_extension.load`.
- Resolving correctness or perf issues traceable to LDS bank conflicts,
  wavefront-scheduler behavior, or MFMA operand layouts.
- Choosing MFMA instruction shape — `mfma_f32_16x16x32_fp8_fp8` vs
  `mfma_f32_32x32x16_fp8_fp8` for FP8 GEMM, `mfma_f32_*_bf16` for BF16.
- Avoiding the bf16/fp8 vector-load trap: Clang's `ext_vector_type(8)` rejects
  `__hip_bfloat16` and the host-only `operator float()` on FP8 types — both
  require the `uint4` + `union` idiom on AMD.

## Architectures targeted

- **gfx942** — MI300X, MI325X (CDNA3). Wave64. 304 CUs. 64 KiB LDS/CU.
- **gfx950** — MI350X, MI355X (CDNA4). Wave64. 256 CUs. 64 KiB LDS/CU.
  Adds OCP `float8_e4m3fn` in hardware (gfx942 is `fnuz`-only).

Use `hipcc --offload-arch=gfx942,gfx950` for a fat binary; or single-arch for
slightly smaller binaries + access to arch-specific intrinsics.

## COUPLED references

- `aiter` — AMD's tuned kernel library; check its baseline before writing from
  scratch (and for the FP8 GEMM bar to beat).

---
name: aiter
description: AITER — AMD's tuned kernel library for Instinct GPUs (github.com/ROCm/aiter). The closest analogue on AMD to FlashInfer-python on NVIDIA. Use when (a) studying a tuned reference impl before writing your own kernel, (b) calling AITER ops as part of a hybrid solution, or (c) setting the comparison bar for a new operator. Covers the actual 25.9 top-level API surface (gemm_a8w8 family, rms_norm, layernorm2d_fwd), the gfx950 OCP vs gfx942 fnuz FP8 dtype split, the bpreshuffle weight-shuffle requirement, and the JIT-at-first-call behavior.
---

# AITER

Reference for using and beating AITER. Detailed guide: `aiter.md`.

## When to consult

- Studying a tuned AMD reference impl before writing your own kernel.
- Calling AITER ops directly as part of a hybrid solution (pre/post-process
  in custom HIP, heavy compute via AITER).
- Setting the baseline for a new operator — AITER ships per-shape autotuned
  CSVs for common LLM kernel shapes.
- Resolving the gfx942 vs gfx950 FP8 dtype split (`float8_e4m3fnuz` vs
  `float8_e4m3fn`) — AITER rejects the wrong one with a confusing error.

## Quick API map (AITER 25.9, top-level)

Most ops are exported directly at `aiter.<name>`, not nested under
`aiter.ops.<group>`. Probe with `dir(aiter)` to confirm names for your
installed version — the surface shifts across releases.

- FP8 GEMM (per-token A scale + per-channel B scale):
  `aiter.gemm_a8w8(XQ, WQ, x_scale, w_scale, bias=None, dtype=torch.bfloat16, splitK=None)`
- FP8 GEMM with pre-shuffled weights (fast path on MI300/MI350):
  `aiter.gemm_a8w8_bpreshuffle(XQ, WQ_shuffled, x_scale, w_scale, bias=None, dtype=torch.float16)`
  — requires `WQ_shuffled = aiter.ops.shuffle.shuffle_weight(WQ)` first;
  silently returns wrong values without the shuffle.
- RMSNorm BF16: `aiter.rms_norm(input, weight, epsilon)` (signature varies
  across versions; check `inspect.signature` on your install).
- LayerNorm BF16: `aiter.layernorm2d_fwd(input, weight, bias, epsilon)`.

## COUPLED references

- `hip` — for the underlying HIP runtime semantics and writing the
  hand-tuned kernel that beats (or composes with) AITER.

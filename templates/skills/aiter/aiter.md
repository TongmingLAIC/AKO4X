# AITER — Detailed Reference

Companion to `SKILL.md`. AITER is AMD's tuned kernel library for Instinct
GPUs — the closest analogue on AMD to FlashInfer-python on NVIDIA.

Repo: [github.com/ROCm/aiter](https://github.com/ROCm/aiter). Install:
`pip install aiter` (included in AKO4X's `[rocm]` extra).

## Validated environment

All API claims in this doc are pinned to **AITER 25.9** as shipped in the
`rocm/pytorch-private:mxfp8-gfx950-v26.6` container (torch 2.12+rocm7.1,
Python 3.12, MI355X gfx950). Different containers / versions may shift
surfaces; verify with `inspect.signature` on your installed version.

## API surface (25.9)

AITER 25.9 exports most operators at the **top level** (`aiter.<name>`),
not nested under `aiter.ops.<group>`. The `aiter.ops.*` submodules exist
but largely re-export the same functions. Probe what's available:

```python
import aiter, inspect
fns = [x for x in dir(aiter) if "gemm" in x.lower() or "norm" in x.lower()]
print(fns)
print(inspect.signature(aiter.gemm_a8w8))
```

### FP8 GEMM family (most common AITER call)

`gemm_a8w8` — generic CK FP8 path:
```python
y = aiter.gemm_a8w8(
    XQ,                 # [M, K] float8_e4m3fn
    WQ,                 # [N, K] float8_e4m3fn (row-major weight layout)
    x_scale,            # [M, 1] float32 (per-token)
    w_scale,            # [1, N] float32 (per-channel)
    bias=None,
    dtype=torch.bfloat16,
    splitK=None,
)
```

`gemm_a8w8_bpreshuffle` — faster MI350 FP8 path; requires the weight
tensor pre-shuffled into the tile-permuted layout the kernel expects:

```python
from aiter.ops.shuffle import shuffle_weight
WQ_shuffled = shuffle_weight(WQ)  # one-shot at weight-load time
y = aiter.gemm_a8w8_bpreshuffle(XQ, WQ_shuffled, x_scale, w_scale, None, torch.bfloat16)
```

**Trap**: calling `bpreshuffle` with a row-major weight (no `shuffle_weight`
first) silently returns wrong values — correctness check will fail, but no
exception. Always shuffle first.

Other GEMM variants in 25.9: `gemm_a8w8_ck`, `gemm_a8w8_asm`,
`gemm_a8w8_blockscale`, `mi350_a8w8_blockscale_ASM`,
`gemm_a4w4`, `gemm_a4w4_blockscale`, `gemm_a8w8_tune` (autotune entry).

### Normalization

```python
# RMSNorm
y = aiter.rms_norm(x, weight, epsilon)         # signature varies by version

# LayerNorm
y = aiter.layernorm2d_fwd(x, weight, bias, epsilon)
```

### Attention / MoE

`aiter.ops.mla.*`, `aiter.ops.attention.*`, `aiter.ops.moe.*` —
op-by-op surface, fastest-moving area in AITER. Check release notes /
`dir()` for your version.

## The gfx942 vs gfx950 FP8 dtype split

AITER on **gfx950** (MI350X / MI355X) requires `torch.float8_e4m3fn` (OCP).
Passing `torch.float8_e4m3fnuz` (the AMD-historical "finite, no -0, no
NaN" format used on gfx942) produces:

```
RuntimeError: Weights and activations should both be int8/fp8!
```

— a confusing message that does **not** mention the dtype mismatch. On
gfx942 (MI300X / MI325X) the situation may reverse depending on the AITER
version. Detect with:

```python
import torch
gfx = torch.cuda.get_device_properties(0).gcnArchName  # e.g. "gfx950"
fp8_dtype = torch.float8_e4m3fn if gfx == "gfx950" else torch.float8_e4m3fnuz
```

## JIT compilation at first call

AITER compiles op groups lazily via `hipcc` at first call. Wheel does not
ship pre-compiled binaries for arbitrary shapes. First-call cost:

| op group              | one-time JIT cost |
|----------------------|------------------:|
| `module_aiter_enum`  | ~14 s             |
| `module_gemm_common` | ~17 s             |
| `module_gemm_a8w8`   | ~150–170 s        |
| `module_gemm_a8w8_bpreshuffle` | ~135 s   |

The compiled artifacts are cached under
`/opt/venv/lib/python3.*/site-packages/aiter/jit/build/` (in-container) or
`~/.cache/aiter/` (host-side install). Warm subsequent runs hit the cache.

Benchmark harnesses should warm AITER paths once before timing.

## Tuned-config CSVs

AITER ships per-shape best-known configs in `.csv` files next to the
ops, e.g. `aiter/configs/a8w8_tuned_gemm.csv`. For an off-tuned shape
the runtime prints:

```
[aiter] shape is M:..., N:..., K:..., q_dtype_w:..., not found tuned config
in .../a8w8_tuned_gemm.csv, will use default config!
```

— a real signal that the operator is on AITER's untuned path and
likely has 10–30% headroom over what AITER delivers out-of-the-box.

## Reading AITER kernels

The most useful study material is `aiter/op_tests/<op>.py` — each
contains a naive PyTorch reference, the AITER op call, a correctness
check, and a timing loop. Read the matching `.csv` to see what tile /
block sizes AITER picked for your shape.

## Beating AITER — known regimes

Validated wins from AKO4X-driven kernel work on MI355X:

- **bpreshuffle on FP8 GEMM** beats hipBLASLt's `torch._scaled_mm` by
  about 1–5 % on DSR1-FFN-like shapes (M=4096, N=7168, K=2048); generic
  `gemm_a8w8` is roughly 12 % slower than the bpreshuffle path on the same
  shape. So if your kernel needs to match a "real" inference baseline,
  bpreshuffle is the bar — generic `gemm_a8w8` is not.
- **Hand-tuned MFMA with proper LDS swizzle** can close most of the
  remaining gap to bpreshuffle at moderate tile sizes (see the `hip`
  SKILL's LDS bank-conflict worked example).

Known dead-ends (do not retry without a new mechanism):
- Beating AITER on shapes that route to its tuned table for the exact
  shape you're hitting — the table-hit path is usually within a few % of
  the practical ceiling on common LLM shapes.
- Beating `aiter.ops.gemm` square-shape paths — those mostly thin-wrap
  hipBLASLt.

## Caveats

- Some AITER ops require `HIP_FORCE_DEV_KERNARG=1` to avoid host-side
  kernel-arg marshaling overhead. Symptom: per-call latency is ~5 µs
  higher than the kernel itself measures via `hipEvent` timing.
- AITER versions iterate fast; pin the version in your reproducer and
  re-baseline on bumps. The error messages and surface shift across
  releases — `inspect.signature` is the source of truth.

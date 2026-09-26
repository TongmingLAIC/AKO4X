# ROCm port — status

This page tracks AKO4X support for AMD Instinct GPUs (CDNA3 / CDNA4 via ROCm).
The port lands incrementally; this document is the authoritative scope marker
for what is and isn't in `main` yet.

## In `main` today (this PR)

- **`spawn.py` auto-detection works on AMD hosts.** `resolve_gpu()` falls back
  to `rocm-smi` when `nvidia-smi` is missing, matching AMD Instinct datacenter
  cards (MI300X / MI325X / MI350X / MI355X). NVIDIA detection behavior is
  unchanged.
- **`.hip` kernels classify correctly.** `infer_language()` recognizes the
  `.hip` extension (AMD's `.cu` equivalent) and tags the operator with
  language `hip`. The classification is purely descriptive — the actual ROCm
  build path lives downstream (see "Deferred" below).
- **Opt-in `[rocm]` dependency extra.** `pip install -e ".[rocm]"` pulls
  AITER. The ROCm flavor of torch is not pinned here; use AMD's wheel index
  (`pip install torch --index-url https://download.pytorch.org/whl/rocmX.Y`)
  or a `rocm/pytorch-*` container.

## Deferred (incoming follow-up PRs)

- **ROCm SKILL bundle.** SKILLs for `hip`, `aiter`, `flydsl`, `ck`, and
  `profiler-rocprof` — analogous to the existing `cuda` / `tilelang` /
  `cute-dsl` / `profiler-ncu` set. Lets sub Claude write AMD-native kernels
  without re-deriving the toolchain. Tracked as a separate PR; coexists with
  the existing NVIDIA SKILLs (no removals).
- **Example operator + closed-loop case study.** End-to-end demo of the
  master/sub round loop running on AMD hardware, with one operator archive
  (variants + dead-end headers + RESULTS.md) showing how the loop converges.
  Pending its own PR.
- **`benchmark_adapter.py` ROCm path.** `scripts/benchmark_adapter.py` is
  currently the sole `flashinfer_bench` importer (NVIDIA-shaped). A ROCm
  integration requires either porting the dataset side or wrapping a
  different bench (AITER's own harness, torchbench-rocm, or a custom
  plain-data adapter). Deferred pending maintainer guidance on preferred
  direction.

## Trying it on AMD today

After this PR lands:

```bash
pip install -e ".[rocm]"
# On an AMD Instinct box (gfx942 / gfx950):
python spawn.py --operator <op> --backend local
# resolve_gpu() detects MI300X/MI325X/MI350X/MI355X via rocm-smi.
```

The standalone-operator-archive pattern (without `flashinfer-bench` dataset
integration) is documented in the upcoming case-study PR.

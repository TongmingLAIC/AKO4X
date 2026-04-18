# DSA Sparse Attention — Variant Comparison

Five preserved kernels. `cute_reduce_v5` is the current production kernel
(added from `ako4fib-run-dsa-sparse-b200-v5` session on 2026-04-18 after two
optimization sessions totaling 20 iterations on the v5 env). Backend: Modal
B200, all 23 workloads.

Raw data: `variance/<name>.json` (per-run scores, per-workload mean/std).

## Headline numbers

| Variant            | Mean ± Std     | CV%  | T=1    | T=2   | T=6   | T=7   | T=8   | LOC | Deps |
|--------------------|----------------|------|--------|-------|-------|-------|-------|-----|------|
| **`cute_reduce_v5`** | **71.05 ± 0.31** | 0.4% | **100.79** | 89.99 | 59.82 | 57.92 | 57.76 | 523 | triton + tilelang + cutlass-dsl(–no-deps) |
| `cute_reduce`      | 70.46 ± 0.32  | 0.5% | 100.4 | 88.5 | 60.0 | 58.3 | 58.1 | 505 | triton + tilelang + cutlass-dsl(–no-deps) |
| `hybrid_dual_ns`   | 47.43 ± 0.11  | 0.2% | 56.3 | 52.4 | 45.0 | 43.8 | 43.7 | 377 | triton + tilelang + cutlass-dsl(–no-deps) |
| `pure_tilelang`    | 45.37 ± 0.03  | 0.1% | 49.7 | 47.4 | 44.9 | 43.7 | 43.6 | 263 | triton + tilelang                          |
| `pure_triton`      | 45.25 ± 0.04  | 0.1% | 53.0 | 48.2 | 45.1 | 42.3 | 42.5 | 178 | triton only                                |

Per-T values are the arithmetic mean of mean-speedups within the group (each
row-T has 1–8 workloads; see `docs/workloads.jsonl`).

## Takeaways

1. **`cute_reduce_v5` wins by +0.59x headline over `cute_reduce`** via a pure
   source-level refactor of the CuTe DSL reduce: split the fused single-loop
   `for s: (m, w, l_g+=, acc+=)` into two unrolled passes (one for l_g, one
   for acc). This lets nvcc interleave 32 independent PO loads with the FMA
   backbone; compiler CSEs the duplicated `exp(m - m_g)` so the "redundant"
   source-level compute is free. Net +1.51x over the v5 re-measured baseline
   (69.54); headline gain vs v4's `cute_reduce` is smaller because v5 and v4
   differ in noise-floor baseline. T=1 confirmed consistently above 100x
   across all 5 variance-check runs (100.79 ± 0.62).
2. **`cute_reduce` (v4) wins by a massive +23x margin** over `hybrid_dual_ns`.
   The only structural change is the reduce kernel — replacing the Triton
   reduce with a CuTe DSL reduce (simple probe-style, no layout machinery,
   ~70 LOC) gave 2x reduce-kernel throughput. Wins stack on top of the
   iter-22 dual-NS fwd dispatch. **T=1 broke 100x** for the first time.
3. **Hybrid (`hybrid_dual_ns`) wins by −2.0x margin** over the best single-path
   variant (`pure_tilelang`). Margin is ~18× the v3 across-run std.
4. **The hybrid win is concentrated in T=1/2**: +6.6x (T=1) and +5.0x (T=2) vs
   `pure_tilelang`; +3.3x (T=1) and +4.2x (T=2) vs `pure_triton`. This is
   the dual-NS effect (Triton NS=32 BK=64 gives more blocks at low T so SM
   utilization climbs from ~0.1 waves to ~0.2 waves).
5. **The CuTe reduce win is universal**: every T value gained +14 to +42x
   once the Triton reduce was replaced. The Triton reduce, being launch-bound
   at low T and gather-patterned at high T, was a much bigger bottleneck
   than expected — the v3 LESSONS had estimated it at ~2 µs but it was
   closer to 2-3x that after accounting for the effective kernel launch.
6. **At T≥6 the three pre-CuTe variants collapse within 0.5x** of each
   other. The HBM random-gather pattern was believed to be the floor, but
   `cute_reduce` shows the reduce kernel was actually a major contributor,
   not just the fwd.
7. **Noise floor by env**: v3 CV 0.03–0.11x (tight); v4 CV 0.32–0.9x (~5x
   looser); v5 CV 0.04x (baseline re-measure) to 0.31x (current best —
   roughly matches v4). Treat ≤0.5x as noise on v4/v5 envs; ≥1.5x is real
   signal. **Always run `bash scripts/bench.sh --variance-check 5` before
   accepting any sub-1x change — single-run scores hide sign-changing
   regressions at this noise floor** (v5 sessions 1-2 confirmed three
   separate micro-opts passed single-run but regressed in variance-check).

## When to use which variant

- **`cute_reduce_v5`** — **current production kernel.** Highest score.
  Recommended starting point for any new sub-env. Same deps as `cute_reduce`
  but incorporates the v5 iter-1 split-loop fix (+0.45–1.5x free) and leaves
  cp.async `assumed_align=16` infrastructure in place (harmless, ready for
  future fused-kernel work).
- **`cute_reduce`** — v4-era CuTe DSL reduce variant. Retained as comparison
  baseline for the iter-1 split-loop win. Still a valid starting point if
  you want the minimal-diff CuTe DSL reduce without any v5-specific
  additions.
- **`hybrid_dual_ns`** — v3-era production kernel with Triton reduce,
  retained as the comparison baseline for the CuTe DSL breakthrough. Use
  only if CuTe DSL reduce fails to compile.
- **`pure_tilelang`** — drop-in when you need to simplify compile (no Triton
  fwd path to JIT) and can accept the T<3 penalty. Same build deps as hybrid
  minus cutlass-dsl.
- **`pure_triton`** — minimal dependency footprint (just `torch`+`triton`).
  Useful as a fallback if TileLang fails to compile on a new backend, or
  as a starting point for a pure-Triton rewrite. Closest in spirit to the
  reference MLA kernels in flashinfer-python.

## Reproducing these numbers

```bash
cd ako4fib-run-dsa-sparse-b200-v4   # or any spawned sub-env
# (back up current solution/kernel.py first!)
cp ../AKO4FIB/reference/dsa-sparse-attention/variants/<variant>.py solution/kernel.py
bash scripts/bench.sh --variance-check 5
# result: trajectory/noise-floor.json  (~158 lines of JSON)
```

## Raw data

- `variance/hybrid_dual_ns.json` — 5 runs, 4 valid (1 × INCORRECT_NUMERICAL
  on workload `38389961...`, a known flaky input pattern).
- `variance/pure_triton.json` — 5 runs, 5 valid.
- `variance/pure_tilelang.json` — 5 runs, 5 valid.
- `cute_reduce` — not yet captured via `--variance-check`. Numbers above
  are from 5 standalone `bench.sh` runs on v4: 70.77 / 70.70 / 70.01 / 70.13 / 70.70.
- `cute_reduce_v5` — `--variance-check 5` measured on v5 Modal B200
  (2026-04-18): runs 70.63 / 71.28 / 71.05 / 71.51 / 70.77, mean 71.05 ±
  0.31 CV 0.44%. Earlier 5-run variance-check on the same kernel form:
  mean 71.05 ± 0.31. Top-5 most-variable workloads: CV 0.3–1.0%.
  Per-T CV ≤ 0.7%.

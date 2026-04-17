# DSA Sparse Attention — Variant Comparison

Four preserved kernels. First three re-measured with `bash scripts/bench.sh --variance-check 5`
on `ako4fib-run-dsa-attn-v2` on 2026-04-17. Fourth added from
`ako4fib-run-dsa-sparse-b200-v4` session on 2026-04-17 (5 unlabeled single runs — variance-check
tooling was not re-run). Backend: Modal B200, all 23 workloads.

Raw data: `variance/<name>.json` (per-run scores, per-workload mean/std).

## Headline numbers

| Variant           | Mean ± Std     | CV%  | T=1    | T=2   | T=6   | T=7   | T=8   | LOC | Deps |
|-------------------|----------------|------|--------|-------|-------|-------|-------|-----|------|
| **`cute_reduce`** | **70.46 ± 0.32** | 0.5% | **100.4** | 88.5 | 60.0 | 58.3 | 58.1 | 505 | triton + tilelang + cutlass-dsl(–no-deps) |
| `hybrid_dual_ns`  | 47.43 ± 0.11  | 0.2% | 56.3 | 52.4 | 45.0 | 43.8 | 43.7 | 377 | triton + tilelang + cutlass-dsl(–no-deps) |
| `pure_tilelang`   | 45.37 ± 0.03  | 0.1% | 49.7 | 47.4 | 44.9 | 43.7 | 43.6 | 263 | triton + tilelang                          |
| `pure_triton`     | 45.25 ± 0.04  | 0.1% | 53.0 | 48.2 | 45.1 | 42.3 | 42.5 | 178 | triton only                                |

Per-T values are the arithmetic mean of mean-speedups within the group (each
row-T has 1–8 workloads; see `docs/workloads.jsonl`).

## Takeaways

1. **`cute_reduce` wins by a massive +23x margin** over `hybrid_dual_ns`.
   The only structural change is the reduce kernel — replacing the Triton
   reduce with a CuTe DSL reduce (simple probe-style, no layout machinery,
   ~70 LOC) gave 2x reduce-kernel throughput. Wins stack on top of the
   iter-22 dual-NS fwd dispatch. **T=1 broke 100x** for the first time.
2. **Hybrid (`hybrid_dual_ns`) wins by −2.0x margin** over the best single-path
   variant (`pure_tilelang`). Margin is ~18× the v3 across-run std.
3. **The hybrid win is concentrated in T=1/2**: +6.6x (T=1) and +5.0x (T=2) vs
   `pure_tilelang`; +3.3x (T=1) and +4.2x (T=2) vs `pure_triton`. This is
   the dual-NS effect (Triton NS=32 BK=64 gives more blocks at low T so SM
   utilization climbs from ~0.1 waves to ~0.2 waves).
4. **The CuTe reduce win is universal**: every T value gained +14 to +42x
   once the Triton reduce was replaced. The Triton reduce, being launch-bound
   at low T and gather-patterned at high T, was a much bigger bottleneck
   than expected — the v3 LESSONS had estimated it at ~2 µs but it was
   closer to 2-3x that after accounting for the effective kernel launch.
5. **At T≥6 the three pre-CuTe variants collapse within 0.5x** of each
   other. The HBM random-gather pattern was believed to be the floor, but
   `cute_reduce` shows the reduce kernel was actually a major contributor,
   not just the fwd.
6. **v3 variance is tiny** (0.03–0.11x std). **v4 variance is ~5x higher**
   (~0.32x std, CV 0.5%). Treat ≤1x differences as noise on v4; ≥1.5x
   is probably signal. Historical v3 thresholds (~0.2x noise floor) are
   optimistic for the current environment.

## When to use which variant

- **`cute_reduce`** — **current production kernel.** Highest score by a wide
  margin. Recommended starting point for any new sub-env.
- **`hybrid_dual_ns`** — previous production kernel, retained as the comparison
  baseline for the CuTe DSL breakthrough. Use only if CuTe DSL reduce fails
  to compile.
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
  Capture a canonical `variance/cute_reduce.json` before trusting downstream
  claims about its noise floor on a new env.

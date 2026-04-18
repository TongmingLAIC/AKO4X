# DSA TopK Indexer — Variant Comparison

Three preserved kernels. `warp_coop_v3` is the current production
recommendation (ako4fib-run-indexer_v3, sessions 2026-04-17/18, 32+
labeled iters; not yet captured via `--variance-check 3`). `graph_cached`
remains the v2-stable baseline. `no_graph` is the no-graph fallback.
Backend: Modal B200, all 128 workloads.

Raw data: `variance/<name>.json` (per-run scores, per-workload mean/std).

## Headline numbers

| Variant          | Mean ± Std         | CV%  | B=1   | B=8   | B=12  | B=15  | B=30  | LOC | Deps |
|------------------|--------------------|------|-------|-------|-------|-------|-------|-----|------|
| `warp_coop_v3`   | **41.12-41.99** *  | n/a  | ~55.4 | ~51.2 | 23.45 | 32.22 | 31.66 | 437 | torch + triton + load_inline CUDA |
| `graph_cached`   | 41.35 ± 0.16       | 0.4% | 56.8  | 50.9  | 22.88 | 31.89 | 31.10 | 331 | torch + triton + load_inline CUDA |
| `no_graph`       | 32.25 ± 0.03       | 0.1% | 49.8  | 40.8  | 8.7   | 21.1  | 23.0  | 270 | torch + triton + load_inline CUDA |

\* `warp_coop_v3` numbers are best-observed across multiple v3 sessions
(score range across full benches 41.12-41.99). Per-B values are the
best-observed within those sessions. **No `--variance-check 3` capture
yet** — Modal v3-env across-session noise is ~±2x (39-42x range for
the same code), so a tight variance check is needed to publish a stable
"Mean ± Std". Within-session, the slow-path tail improvements over
graph_cached (B=12/14/30) reproduce consistently.

Per-B values are the arithmetic mean of mean-speedups within the group;
B=12 has 1 workload, B=15 has 21 workloads (15 of which are slow-path,
the largest weighted contributor — see LESSONS.md).

## Takeaways

1. **CUDA graph capture is still the dominant lever.** `graph_cached`
   beats `no_graph` by **+9.10x (≈+28%)**. The scoring (Triton FP8
   `tl.dot` [64,128]×[128,64]) and radix (CUDA CUB two-pass 8+8-bit)
   kernels in those two are *byte-identical* — the only delta is the
   Python wrapper that captures score+radix into a `torch.cuda.CUDAGraph`
   and replays it for same-pointer calls.
2. **`warp_coop_v3` adds a smaller but reproducible win on top of
   graph_cached** (~+0.5x on slow-path tail). Three independent
   radix-internal optimizations stack:
   - Warp-cooperative gather (ballot+popc): one atomic per warp instead
     of one per surviving element. 32× fewer atomicAdds in the gather.
   - Warp-cooperative phase-1 hi-byte histogram (`__match_any_sync`):
     one atomic per unique bucket per warp. **Works here despite v1
     LESSONS' "uniform = no win" note** because score data is post-ReLU
     positive-only with a narrow exp range, giving high warp-internal
     bucket-collision rate.
   - Warp-only 256-bin threshold scan replaces CUB `BlockScan<int, 1024>`,
     removing the BT=1024 constraint and letting block size drop to
     BT=128 (more blocks/SM, less per-barrier sync cost).
3. **Slow-path latency breakdown still dominates the score**. At B=30,
   `warp_coop_v3` measures ~4.2 µs (vs ~4.4 µs for graph_cached). Of
   that: ~1 µs CUDA graph replay (hardware floor), ~2 µs Triton FP8
   score (compute-bound, near the FP8 tensor-core floor), ~1.0-1.2 µs
   warp-coop radix (down from 1.5 µs in graph_cached).
4. **Neither variant breaks the radix SM-underutilization floor.** Even
   warp_coop_v3 still launches 30 blocks on 148 SMs (≈20% util). The
   single biggest untapped lever is **CuTe DSL for the radix** — the
   sibling op (`reference/dsa-sparse-attention/`) saw +23x replacing
   its launch-bound Triton reduce with a ~70-LOC CuTe DSL kernel; our
   radix is the same archetype. See LESSONS.md *What remains untried*.
5. **Across-run variance**: graph_cached on v2 is tight (CV 0.4%);
   v3 sessions on the same code show ±2x drift (39-42x range), pointing
   at Modal B200 thermal/tenant-load drift specific to v3 env spawns.
   Per-B slow-path numbers are stable within session and reproduce the
   v3 win pattern across drift.

## When to use which variant

- **`warp_coop_v3`** — recommended starting point for new sub-envs.
  Best-observed score, slow-path tail consistently better than
  graph_cached. Same dependencies as graph_cached. **Run
  `--variance-check 3` against your env baseline before relying on
  the headline number** (the v3 env-drift caveat above).
- **`graph_cached`** — v2-stable baseline. Use if you want a kernel
  with documented CV-0.4% variance check, or if you want to verify
  warp_coop_v3's win is real on your env (run both, A/B compare).
- **`no_graph`** — drop-in when CUDA graphs are unavailable or fail
  to capture (e.g., kernel does something graph-hostile like host-side
  synchronization, or a PyTorch build without graph support). Also
  serves as the reference for what the kernel looks like *without* the
  graph capture layer — useful for studying the raw scoring/radix
  design.

## Reproducing these numbers

```bash
cd ako4fib-run-indexer_v3   # or any spawned sub-env
# (back up current solution/kernel.py first!)
cp ../AKO4FIB/reference/dsa-topk-indexer/variants/<variant>.py solution/kernel.py
bash scripts/bench.sh --variance-check 3
# result: trajectory/noise-floor.json  (~160 lines of JSON, 128 workloads)
```

## Raw data

- `variance/graph_cached.json` — 3 runs (41.45, 41.17, 41.43x). Identical
  copy of `ako4fib-run-indexer_v2/trajectory/noise-floor.json` after iter-9.
- `variance/no_graph.json` — 3 runs, fresh bench 2026-04-17 after
  swapping `ako4fib-run-indexer_optimize`'s HEAD kernel into v2's bench
  infra (same baseline cache so speedups are apples-to-apples).
- `warp_coop_v3` — not yet captured via `--variance-check`. Numbers above
  are best-observed across the v3 session full-bench results documented
  in `ako4fib-run-indexer_v3/ITERATIONS.md` iter-1/8/16/26/30/32. Run
  `--variance-check 3` in a v3-equivalent env to capture stable numbers.

Both JSON files contain: `scores[]`, `score_mean`, `score_std`, `score_cv`,
`score_min`, `score_max`, and `per_workload` (nested dict of workload UUID →
mean, std, cv, min, max).

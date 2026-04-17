# DSA TopK Indexer — Variant Comparison

Two preserved kernels, each re-measured with `bash scripts/bench.sh --variance-check 3`
against the cached baseline in `ako4fib-run-indexer_v2` on 2026-04-17.
Backend: Modal B200, all 128 workloads.

Raw data: `variance/<name>.json` (per-run scores, per-workload mean/std).

## Headline numbers

| Variant          | Mean ± Std      | CV%  | B=1  | B=8  | B=12 | B=15 | B=30 | LOC | Deps |
|------------------|-----------------|------|------|------|------|------|------|-----|------|
| `graph_cached`   | **41.35 ± 0.16**| 0.4% | 56.8 | 50.9 | 22.8 | 31.9 | 31.5 | 331 | torch + triton + load_inline CUDA |
| `no_graph`       | 32.25 ± 0.03    | 0.1% | 49.8 | 40.8 | 8.7  | 21.1 | 23.0 | 270 | torch + triton + load_inline CUDA |

Per-B values are the arithmetic mean of mean-speedups within the group; B=12
has 1 workload, B=30 has 18 workloads (see `docs/workloads.jsonl` in the child env).

## Takeaways

1. **CUDA graph capture accounts for the entire gap**: `graph_cached` beats
   `no_graph` by **+9.10x (≈+28%)**. The scoring (Triton FP8 `tl.dot`
   [64,128]×[128,64]) and radix (CUDA CUB two-pass 8+8-bit) kernels are
   *byte-identical* between the two variants — the only delta is the Python
   wrapper that captures score+radix into a `torch.cuda.CUDAGraph` and replays
   it for same-pointer calls.
2. **The graph win is concentrated on slow-path latencies**. For slow-path
   groups (B≥12 with M>32) the replay amortizes ~2-3µs of Triton+CUDA
   dispatch overhead per call, roughly halving wall-clock latency:
   B=30 drops 10µs → 4.4µs, B=12 drops 13µs → 5.2µs. Fast-path groups
   (B≤11 with M≤32) also benefit — e.g., B=3 goes 2.4µs → 1.7µs.
3. **Variance is very tight** (std ≤ 0.2x → CV ≤ 0.5%). A ≥0.5x
   difference between configurations is real signal; anything ≤0.2x is noise.
   This was only achievable *after* graph capture; earlier runs of
   `no_graph`-class kernels showed CV ~1–3% (see `LESSONS.md` for why
   variance dropped so much).
4. **Neither variant touches the radix SM-underutilization floor**.
   With 30 batches → 30 blocks on B200's 148 SMs, the radix uses ≤20%
   of SMs. Multi-block / cooperative radix schemes were tried 3× across
   the two envs and always regressed (documented in `LESSONS.md`) because
   the per-kernel launch overhead on Modal B200 (~1-3µs) exceeds the
   parallelism savings at this workload size.

## When to use which variant

- **`graph_cached`** — default production kernel. Best raw score. Requires
  `torch.cuda.CUDAGraph` + `torch.cuda.graph_pool_handle()` (PyTorch ≥ 2.0).
- **`no_graph`** — drop-in when CUDA graphs are unavailable or fail to
  capture (e.g., kernel does something graph-hostile like host-side
  synchronization, or a PyTorch build without graph support). Also serves as
  the reference for what the kernel looks like *without* the graph
  capture layer — useful for studying the raw scoring/radix design.

## Reproducing these numbers

```bash
cd ako4fib-run-indexer_v2
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

Both JSON files contain: `scores[]`, `score_mean`, `score_std`, `score_cv`,
`score_min`, `score_max`, and `per_workload` (nested dict of workload UUID →
mean, std, cv, min, max).

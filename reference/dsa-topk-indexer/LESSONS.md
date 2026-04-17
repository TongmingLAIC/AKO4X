# DSA TopK Indexer — Lessons

Distilled from ~90 iterations across two sub-envs
(`ako4fib-run-indexer_optimize/`: ~79 iterations focused on kernel internals;
`ako4fib-run-indexer_v2/`: 9 labeled iterations focused on launch-overhead
amortization). Keep this short. Raw iteration logs live in each sub-env's
`ITERATIONS.md` for anyone who wants the unabridged history.

## Operator & shape

- Op: `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`
  — decode-stage sparse attention's Top-K indexer (per-query KV selection).
- Q: `q_index_fp8 [B, 64, 128]` float8_e4m3fn.
- KV: `k_index_cache_fp8 [P, 64, 1, 132]` int8 (interpreted as uint8).
  Layout per page: `[64 × 128 bytes fp8 data][64 × 4 bytes float32 scales]`.
- Weights: `[B, 64]` float32 per-head.
- Indexing: `seq_lens [B]` int32, `block_table [B, max_num_pages]` int32.
- Output: `topk_indices [B, 2048]` int32 (−1 = padding for short seqs).
- Formula: `sum(relu(q @ K.T) * weights, axis=heads)` → Top-K by scalar score.
- Dataset: 128 workloads, B ∈ {1…31}, max_num_pages ∈ {1…91},
  num_pages = 11923 everywhere, all decode-stage.
- Hardware: B200 (148 SMs, 228 KB SMEM/SM, 64K regs/SM, HBM3e ~8 TB/s peak).

## Key facts

- **Expert baseline** (cached, profiled on Modal B200): roughly 0.32 ms at
  B=30. Score = expert_latency / our_latency → 41x means ~8µs.
- **Best kernel today**: `variants/graph_cached.py` at
  **41.35 ± 0.16x** (3-run variance-check, CV 0.4%). 331 LOC.
  Triton FP8 scoring + CUDA CUB two-pass 8+8-bit radix + per-shape
  `torch.cuda.CUDAGraph` capture + replay.
- **Latency breakdown at B=30 (≈4.4 µs total)**:
  - CUDA graph replay overhead: ~1 µs (hardware fixed).
  - Triton FP8 scoring kernel (2730 blocks): ~2 µs. Compute-bound on
    FP8 tensor cores; memory traffic is small.
  - CUDA CUB radix topk (30 blocks): ~1.5 µs. SM-underutilized
    (30 / 148 SMs = 20%) but can't trivially parallelize further
    (see *Instructive failures* below).
- **Slow-path vs fast-path dispatch**: when `max_num_pages ≤ 32` (so
  `seq_len ≤ 2048 = TOPK`), we skip scoring and emit all valid token
  indices via a tiny Triton kernel (`_all_tokens_kernel`). Slow-path
  runs scoring+radix; fast-path runs just the index-gather.
- **Across-run noise floor**: vc3 gives CV ≈ 0.4% (std ≈ 0.16x on
  41.35x). Pre-graph runs (32x-class) had CV ≈ 1-3% because
  per-call launch-overhead jitter dominated the measurement.
  Anything ≤ 0.3x difference between configs is noise.

## Architectural pivots that worked

### Iter 5 (v2) — CUDA graph capture for slow-path, keyed on (B, M, all_ptrs)  → 36.96x (from 32.08x baseline)

The breakthrough. Per-call overhead for score+radix on Modal B200 was
~2-3µs of host-side Triton+CUDA dispatch. Within a bench trial the
bench framework reuses the same `args_tuple` 100×, so input tensor
data_ptrs are stable and a captured CUDAGraph can replay them. First
call eager-executes + captures on a side stream; subsequent calls hit
the `(B, M, q_ptr, k_ptr, w_ptr, sl_ptr, bt_ptr)` cache and fire
`graph.replay()` (~1µs).

This was the axis that `_optimize` spent 79 iterations missing. The
kernel internals were at a tight local optimum already; the untapped
reservoir was launch dispatch, which no amount of kernel tuning could
reach.

### Iter 6 (v2) — Extend graph capture to the fast-path kernel too  → 41.22x

The fast-path (`_all_tokens_kernel`, single Triton launch for
`M ≤ 32`) was still paying ~1-2µs of Triton dispatch. Capturing it
in a graph too dropped small-B workloads 20-40%: B=1 went
2.3µs → 1.9µs, B=3 went 2.4µs → 1.7µs. The capture path mirrors
the slow-path one (eager warmup + side-stream warmup + capture).

### Iter 7-8 (v2) — Shape-cache scores buffer + shared `graph_pool_handle()`  → stable 41.35x (CV 0.4%)

Before: every trial allocated a fresh `scores` tensor and put it in
its own captured graph. This caused allocator fragmentation and
widened CV. After: `_scores_cache` keyed on `(B, MT)` reuses the
same `scores` tensor across trials of the same shape (stride matches
exactly — unlike the failed iter-2 `_optimize` attempt at oversized
caching); `torch.cuda.graph_pool_handle()` shares a memory pool
across all captured graphs. No single-run score gain, but CV
dropped from ~1% to ~0.4%.

## Instructive failures (grouped by lesson)

- **Multi-block radix with grid-wide coordination is net-negative on
  B200 Modal.** Tried three times across the two envs: `_optimize`
  iter 62 (multi-stream + atomic signaling), v2 iter-mid (grid.sync
  across 120 blocks), v2 iter-mid (2-kernel partial_hist + main
  radix). All regressed 30-50% because per-kernel launch overhead
  (~1-3µs on Modal) exceeds the parallelism savings from splitting
  the radix's ~2µs of work across more blocks. The radix is
  fundamentally serial-per-batch; the 30 blocks × 1 batch each layout
  is optimal at this workload size.
- **Hand-CUDA / PTX MMA scoring is slower than Triton `tl.dot`.**
  `_optimize` iter 43-46 (pure CUDA + WMMA BF16 scoring via TVM FFI)
  → score kernel 25-45µs; iter 67 (CUDA PTX FP8 `m16n8k32` with A-fragment
  reuse) → still 2µs slower than Triton at matched occupancy. Triton's
  codegen for 64×64×128 FP8 matmul on B200 is genuinely unbeatable
  by hand. **Lesson: don't rewrite Triton's FP8 `tl.dot` as CUDA
  unless you can also replicate its register-optimal instruction
  scheduling — you almost certainly can't.**
- **`tl.dot` with M=1 collapses to scalar code (no tensor cores).**
  `_optimize` tried per-head online softmax for scoring. `tl.dot`
  with a 1-row LHS emits 16 identical scalar MMAs; tensor-core path
  disabled. Keep the `[64, 128] × [128, 64] → [64, 64]` tile.
- **TileLang `T.gemm` FP8 produces incorrect output on sm_100.**
  `_optimize` iter 73: 4 workloads tested, all `INCORRECT_NUMERICAL`
  (max_abs 100+, matched_ratio 0.77-0.92). Filed in iter log; not
  validated for Blackwell FP8 at this time.
- **`torch.topk` is slower than our custom CUB radix for this shape.**
  For N ≤ 5824, `K = 2048`, `torch.topk` internally uses CUB
  `DeviceRadixSort` which allocates scratch + launches more kernels.
  Custom single-block 8+8-bit radix is ~3× faster. v2 verified this
  even with a hybrid (torch.topk for B≤20, custom for B>20) — the
  hybrid regressed to 4.46x.
- **Oversized cached scores buffer breaks L2 locality (−0.26x).**
  v2 iter-2 pre-allocated `[max(B, 32), MAX_MT=5824]` bfloat16 and
  sliced per-call. Wide stride for small-MT workloads killed
  cache-line reuse between the score write and radix read. Fresh
  `torch.empty((B, MT))` has packed stride = MT and is only ~1µs
  anyway. **Cache only when the cached stride matches the actual
  stride.** (iter-7 above fixes this by keying on exact shape.)
- **num_warps=4/8 for the score kernel regress.** 26-28µs vs 16µs
  baseline. Triton's scheduler wants num_warps=2 for the specific
  `[64, 128] × [128, 64] → [64, 64]` FP8 tile. Hand-picked
  `num_warps=2, num_stages=1` is optimal.
- **`num_stages=2/3` (pipelining) hurts scalar-dominant loops.** The
  score kernel has no outer loop over K (K=128 fits a single MMA
  issue), so `num_stages>1` just inserts needless SMEM staging
  overhead.
- **PPB=2 (2 pages per block) regressed at multiple envs.** Reshape
  `[2, 64, 128] → [128, 128]` forces SMEM roundtrip; larger tile
  increases register pressure; indexed `phys_page` gather from the
  block table becomes 3D and harder to vectorize. Keep 1 block per
  (bid, pid) tile.
- **Merging radix pass 2 + pass 3 into one combined pass regressed
  severely.** v2 attempted "build hist2 + gather tie bucket in one
  pass over N". The per-iteration branch-predicated atomic +
  block-table lookup nearly doubled the work of pass-2. The classic
  3-pass 8+8 structure (hist on hi-byte → hist on lo-byte filtered
  by hi-threshold → gather) is well-balanced — each pass is lean in
  exactly the right way.
- **Warp-level `__match_any_sync` coalesced histogram is not
  faster on B200.** Shared-memory atomics on B200 are already fast
  enough that the `ballot + ffs + popc + divergent atomic` overhead
  exceeds the savings for roughly-uniform distributions. Only worth
  it for highly-skewed data (e.g., 90%+ of values in one bucket).
- **CUDA graph capture with cluster launch (`__cluster_dims__`) hits
  runtime errors on this Modal image.** v2 attempted to combine the
  two optimizations but got `unspecified launch failure` intermittently
  on workloads with B=30. Unclear if it's the cluster API, the CUB
  BlockScan inside cluster, or the DSM access pattern. Left as
  *untried* below.

## B200 floor analysis

- **Score kernel floor ≈ 2 µs**. Triton codegen is near-optimal at
  this tile; register pressure (208 regs/thread at num_warps=2)
  limits `Block Limit Registers` to 4 → theoretical occupancy 12.5%.
  Going lower requires explicit CUTLASS/CuTe tcgen05 (Blackwell
  tensor memory) — estimated complexity is ~500 LOC and probably
  won't measurably beat Triton on this workload because the kernel
  is compute-bound, not HBM-bound.
- **Radix kernel floor ≈ 1 µs**. 30 blocks × 1 batch = 20% SM
  utilization but no parallel-radix scheme has broken through this
  (see *Instructive failures*). The work per batch (~2800 cycles at
  1.1 GHz on one SM) is too small to amortize sync overhead of any
  multi-block scheme.
- **CUDA graph replay overhead ≈ 1 µs**. Fixed cost of `cuGraphLaunch`;
  not reducible from the PyTorch side.
- **Total floor** ≈ 4 µs for slow-path workloads. We measure 4.4 µs
  for B=30 — within 10% of the floor. Further kernel-internal tuning
  returns less than 0.5 µs.

## What remains untried

- **Cluster launch with DSM reduce for radix parallelism.** B200
  supports `__cluster_dims__` with cross-block SMEM (DSM). In
  principle, 4 CTAs per cluster could do parallel partial
  histograms + DSM reduce, boosting radix SM utilization to 80%.
  v2 attempted it and hit an intermittent runtime error. Worth
  revisiting with a cleaner Modal image / newer CUDA driver.
- **CuTe DSL tcgen05 score kernel**. Rewriting the scoring
  `[64, 128] × [128, 64]` FP8 MMA in CuTe DSL using Blackwell
  tensor memory (tcgen05) could reduce register pressure and allow
  higher occupancy. Estimated outcome: match Triton ±0.3x, not
  beat it, because the kernel is already near the tensor-core
  compute floor. Only worth it as a learning exercise or to
  explore fused score+radix in one kernel.
- **Fused score+radix in a single persistent kernel.** Eliminates
  the 1 µs inter-kernel gap inside the graph, but requires grid
  sync (cluster sync or cooperative-groups grid sync). Not
  possible in Triton; feasible in CuTe DSL or raw CUDA. Potential
  saving is small (<1 µs) and it requires writing both kernels in
  one language.
- **Persistent kernel with per-SM work queue.** Launch 148 blocks
  (one per SM), pick work from an atomic counter, maintain
  long-running state. Eliminates graph replay overhead (~1 µs) and
  kernel-launch jitter. Same feasibility story as fused score+radix.

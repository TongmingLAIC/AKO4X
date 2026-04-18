# DSA TopK Indexer — Lessons

Distilled from ~120 iterations across three sub-envs
(`ako4fib-run-indexer_optimize/`: ~79 iters, kernel internals;
`ako4fib-run-indexer_v2/`: 9 labeled iters, launch-overhead amortization;
`ako4fib-run-indexer_v3/`: ~38 labeled iters, radix internals + tooling
explorations). Keep this short. Raw iteration logs live in each sub-env's
`ITERATIONS.md`.

**tl;dr:** three big levers, in order.
1. **CUDA graph capture** of the score+radix slow-path keyed on
   `(B, M, all_data_ptrs)`. v2 iter-5/6/7-8: 32x → 41.35x. The single
   biggest win in this op's history. Without graph capture you live
   in the 32x range; with it you live in the 41x range. **Always start
   from `variants/warp_coop_v3.py`** (or `graph_cached.py`); never
   ship a no-graph version.
2. **Warp-cooperative radix** (v3 iter-8/16/26/30: 41.35x → 41.99x peak).
   Replace per-element `atomicAdd` in the gather and phase-1 hi-byte
   histogram with `__ballot_sync` / `__match_any_sync` + leader-only
   atomic. Replace CUB `BlockScan<int, 1024>` with a single-warp
   shuffle-based 256-bin scan, which removes the BT=1024 constraint
   and lets the slow-path block size drop to 128 (more blocks/SM,
   less per-barrier sync cost). Stable +0.5x on slow-path tail
   (B≥12 with M>32) across session drift.
3. **(Untried at top-K, predicted high impact)** CuTe DSL for the
   **radix kernel** (not the score kernel). The sibling op (DSA sparse
   attention) replaced its launch-bound Triton reduce with a ~70-LOC
   CuTe DSL kernel and got **+23x overall** (47x → 70x). Our radix is
   the same archetype: launch-bound, SM-underutilized (30 blocks on
   148 SMs ≈ 20% util at the slow-path), small per-block work
   (~1-1.5 µs). See *What remains untried — top priority* below.

After (1) and (2), v3's best is `warp_coop_v3.py` at peak 41.99x
(vs `graph_cached.py`'s 41.35x). The score kernel is at the FP8
tensor-core compute floor — see *Score kernel floor analysis* below.

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
- **Best kernel today**: `variants/warp_coop_v3.py` at peak **41.99x**
  single-run, **41.12-41.99x across multiple v3 sessions** (see also
  `graph_cached.py` at 41.35 ± 0.16x for the v2-stable comparison
  baseline). Triton FP8 scoring (with eviction hints) + CUDA radix with
  warp-coop gather/hist + warp-only scan + BT=128 + per-shape CUDAGraph.
- **Latency breakdown at B=30 (≈4.2 µs slow-path)**:
  - CUDA graph replay overhead: ~1 µs (hardware fixed).
  - Triton FP8 scoring kernel (2730 blocks): ~2 µs. Compute-bound on
    FP8 tensor cores; memory traffic small (Q/K mostly L2-resident
    after eviction hints).
  - CUDA warp-coop radix topk (30 blocks): ~1.0-1.2 µs (down from
    ~1.5 µs in v2's CUB variant). SM-underutilized (30 / 148 SMs
    = 20%) but no parallel-radix scheme has beaten this layout —
    see *Instructive failures*.
- **Slow-path vs fast-path dispatch**: when `max_num_pages ≤ 32` (so
  `seq_len ≤ 2048 = TOPK`), we skip scoring and emit all valid token
  indices via a tiny Triton kernel (`_all_tokens_kernel`). Slow-path
  runs scoring+radix; fast-path runs just the index-gather.
- **Across-run noise floor**: vc3 on v2 gives CV ≈ 0.4% (std ≈ 0.16x
  on 41.35x). v3 sessions vary 39-42x due to Modal B200 thermal/
  tenant-load drift; treat ≤ 0.5x as noise on v3 envs. Within-session
  changes are stable; the slow-path per-batch deltas (B=12/14/30)
  reproduce across drift.
- **Workload weighting reality check**: B=30 is the loudest single
  workload but NOT the largest contributor. The slow-path tail is
  dominated by **B=15** (15 workloads, ~3.4x weighted gap to fast
  baseline), then B=30 (11 wl, 2.7x), B=29 (7 wl, 1.8x), B=8 (8 wl,
  1.6x), B=4 slow-path (6 wl, 1.1x), B=14 (4 wl, 0.9x). Tune for
  B=15/30 first, not just B=30.

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
reservoir was launch dispatch.

### Iter 6 (v2) — Extend graph capture to the fast-path kernel too  → 41.22x

The fast-path (`_all_tokens_kernel`, single Triton launch for `M ≤ 32`)
was still paying ~1-2µs of Triton dispatch. Capturing it in a graph
too dropped small-B workloads 20-40%: B=1 went 2.3µs → 1.9µs, B=3
went 2.4µs → 1.7µs.

### Iter 7-8 (v2) — Shape-cache scores buffer + shared `graph_pool_handle()`  → stable 41.35x (CV 0.4%)

`_scores_cache` keyed on `(B, MT)` reuses the same `scores` tensor
across trials of the same shape (stride matches exactly);
`torch.cuda.graph_pool_handle()` shares a memory pool across all
captured graphs. No single-run gain, but CV dropped from ~1% to ~0.4%.

### Iter 1 (v3) — L2 eviction hints on the score kernel  → +1.6x

`tl.load(..., eviction_policy="evict_last")` on Q (reused across the
M blocks for each batch); `evict_first` on K and the per-token scale
(one-shot per block). Q L2-cache retention is the dominant effect;
small-batch workloads gained the most (B=1: 51.91→54.23, B=6:
58.66→61.33). Slow-path slightly improved too.

### Iter 8 (v3) — Warp-cooperative radix gather (ballot+popc)  → +0.6x

Replace per-element `atomicAdd(&s_out_cnt, 1)` in the gather with
`__ballot_sync` + `__popc` + `__shfl_sync` for the warp's base index
→ **one atomic per warp** instead of up to 32. Required converting
the gather loop to a warp-uniform iteration count (use `ceil(N/BT)`
and predicate, not `for i = tid; i < N; i += BT`) so `__ballot_sync(
FULL_MASK, ...)` always sees all 32 lanes (matters when N mod BT
lands mid-warp).

### Iter 16 (v3) — Warp-cooperative phase-1 histogram (match_any)  → +0.32x

`__match_any_sync(FULL_MASK, bucket)` groups same-bucket lanes →
one atomic per unique bucket per warp. v1 LESSONS predicted this
wouldn't help ("uniform distributions favor naive atomics"), but
**score data is post-ReLU positive-only with a narrow exp range
(0x3D-0x43 hi-byte after `bf16_ord`)**, giving very high
warp-internal collision rate. The lesson: re-test warp-coop on
data with non-uniform distribution.

Phase-2 lo-byte hist tried the same pattern (iter-15) and regressed
slightly — the hi-byte filter in phase 2 leaves ~95% of lanes with
sentinel bucket=-1, so match_any overhead exceeds savings.

### Iter 26 (v3) — Warp-only 256-bin threshold scan replaces CUB BlockScan

CUB `BlockScan<int, BT>` requires `BT == 1024` to scan our 256
buckets in parallel — wasting threads 256-1023 on every scan call
plus the SMEM temp_storage roundtrip. Replaced with a single-warp
scan: warp 0 does 8 iterations of 32-lane shuffle-based prefix-sum
over `hist[0..255]`, other warps wait at `__syncthreads`. Same
correctness, slightly fewer barriers, and **removes the BT=1024
constraint** that had been load-bearing in v1/v2. This is the
prerequisite for iter-30.

### Iter 30 (v3) — Radix BT=1024 → BT=128  → +0.5x slow-path

With CUB BlockScan gone, BT is no longer tied to scan size. BT=128
gives more blocks per SM (less per-barrier sync cost, smaller team
per `__syncthreads`). Same launch geometry (B blocks, one per batch)
but each block uses 128 threads instead of 1024. Sweep verified
BT=64 and BT=256 are both worse for slow-path; BT=128 is the sweet
spot. Slow-path B=12/14/30 each gained 0.3-0.6x.

## Score kernel floor analysis (unchanged from v2)

- **Score kernel floor ≈ 2 µs**. Triton FP8 codegen is near-optimal
  at this tile; register pressure (208 regs/thread at num_warps=2)
  limits `Block Limit Registers` to 4 → theoretical occupancy 12.5%
  (achieved ~7.4% per NCU). NCU "Est. Speedup 87.5%" from raising
  occupancy is misleading: the kernel is FP8 tensor-core compute-
  bound, not register-stall-bound, so reducing register pressure
  via TMEM (CuTe DSL) wouldn't translate to wall-clock savings.
- **Radix kernel floor ≈ 1 µs**. 30 blocks on 148 SMs = 20% SM util.
  The work per batch (~2800 cycles at 1.1 GHz on one SM) is too
  small to amortize sync overhead of any multi-block scheme tried
  to date. CuTe DSL is the most-promising untried path here — see
  the next section.
- **CUDA graph replay overhead ≈ 1 µs**. Fixed cost of `cuGraphLaunch`;
  not reducible from the PyTorch side.
- **Total slow-path floor** ≈ 4 µs. v3's `warp_coop_v3` measures
  4.2 µs at B=30 — within 5% of the floor. Score kernel internal
  tuning returns less than 0.5 µs.

## What remains untried — top priority

### CuTe DSL for the **radix** kernel (not the score kernel)

**This is the highest-EV untried lever.** The sibling op
(`reference/dsa-sparse-attention/`) replaced its launch-bound Triton
reduce kernel with a ~70-LOC CuTe DSL kernel (`cute_reduce_v5.py`)
and got **+23x overall** (47.43x → 70.46x → 71.05x with the v5 split-
loop refinement). The pattern that worked is dead simple — direct
tensor indexing, scalar fp32 loops, `cutlass.Float32(value)` for
casts, `cute.math.exp/log`, no layout/copy-atom/MMA machinery.

**Why this is the right target for top-K, not the score kernel**:
- Score kernel is FP8 tensor-core compute-bound. CuTe DSL would emit
  the same `tcgen05.mma.kind::f8f8` as Triton; v1 LESSONS iter-67
  (hand-PTX FP8) confirmed this loses ~2 µs. The "match Triton ±0.3x"
  prediction stands for the score kernel.
- Radix kernel is exactly the sibling-reduce archetype: tiny per-block
  work (~1-1.5 µs), launch-bound on Triton/CUB but launch overhead
  is significantly lower for CuTe DSL kernels (skip Triton/MLIR, go
  straight to `nvcc -arch=sm_100a`). 30 blocks × 148 SMs = grid-size-
  limited regime where per-launch overhead dominates the per-block
  compute.

**Concrete plan** (if a future session takes this on):
1. Port the v3 warp-coop radix CUDA kernel to CuTe DSL.
2. Use scalar-loop pattern from sibling's `_cute_reduce_kernel`
   (variants/cute_reduce.py L380-414 in upstream). No TMA, no TMEM
   allocator, no warp specialization.
3. Keep the warp-coop pattern (`__match_any_sync`, `__ballot_sync`)
   if CuTe DSL exposes them as `cute.arch.*` intrinsics; else fall
   back to per-lane atomics (CuTe DSL's tighter codegen may make
   the warp-coop overhead unnecessary).
4. Expected outcome based on sibling: **+10-20x overall**. The
   sibling's reduce was 2-3 µs; ours is 1-1.5 µs. Even halving it
   (sibling halved theirs) would shave ~0.6 µs off slow-path,
   pushing B=30 from ~31x to ~36-38x → **+3-5x overall score**.

### Score kernel: status quo

The Triton `tl.dot` FP8 path is the production choice. The CuTe DSL
score-kernel direction was investigated in v3 (a runnable scalar-FMA
proof-of-concept landed in `solution/kernel.py` behind
`USE_CUTE_SCORE` env flag, dormant) but the FP8-MMA tcgen05 version
would need 300-500 LOC of warp-spec + TMA + TMEM scaffolding to even
match Triton, with no expected upside. Skip unless someone wants the
~500-LOC fused score+radix angle (see below).

### Other backlog (lower priority)

- **Cluster launch with DSM reduce for radix parallelism.** B200
  `__cluster_dims__(2,1,1)` with DSM-reduced histograms WAS attempted
  in v3 iter-3 (and v2 before that). Kernel produces correct output
  in eager mode but **CUDA graph replay fails with "unspecified
  launch failure"** intermittently — same failure as v2. Triton's
  `num_ctas=2` (which lowers to the same cluster-launch path) hits
  the identical incompatibility (v3 iter-17). Conclusion: cluster
  launch is incompatible with PyTorch's `torch.cuda.graph` capture
  on this Modal image. Not retried unless the Modal image upgrades
  CUDA driver / PyTorch.
- **Fused score+radix in a single persistent kernel.** Eliminates
  the ~200ns inter-kernel gap inside the graph. CuTe DSL or raw CUDA;
  needs grid sync. Saving is small (<0.5 µs) and writing both kernels
  in one language is high effort. Defer until after the radix CuTe
  DSL win is captured.
- **Persistent kernel with per-SM work queue** (Triton). v3 iter-33/
  34/35 tried this — round-robin AND contiguous-slice variants both
  regressed heavily on slow-path (B=12: 23 → 17), even with
  `tl.range(num_stages=2)`. **Triton's `tl.range` doesn't pipeline
  across inner iterations**, so a serial tile loop loses the implicit
  wave-parallelism that the `(B, M)` grid gets from the GPU scheduler.
  Don't retry in Triton; only viable in CuTe DSL with explicit
  cp.async pipelining.

## Instructive failures (grouped by lesson)

- **Multi-block radix with grid-wide coordination is net-negative on
  B200 Modal.** Tried four times: `_optimize` iter 62 (multi-stream +
  atomic signaling), v2 iter-mid (grid.sync across 120 blocks), v2
  iter-mid (2-kernel partial_hist + main radix), v3 iter-3 (cluster-
  launch radix with DSM). All regressed when CUDA graph capture
  involved, because per-kernel launch overhead and/or cluster-launch+
  graph incompatibility (~1-3µs on Modal) exceeds the parallelism
  savings from splitting the radix's ~1-2µs of work across more blocks.
  The radix is fundamentally serial-per-batch within the current
  toolchain; the 30-blocks × 1-batch-each layout is optimal.
- **Hand-CUDA / PTX MMA scoring is slower than Triton `tl.dot`.**
  `_optimize` iter 43-46 (pure CUDA + WMMA BF16 scoring via TVM FFI)
  → score kernel 25-45µs; iter 67 (CUDA PTX FP8 `m16n8k32` with
  A-fragment reuse) → still 2µs slower than Triton at matched
  occupancy. **Lesson: don't rewrite Triton's FP8 `tl.dot` as
  hand-CUDA unless you can also replicate its register-optimal
  instruction scheduling.**
- **`tl.dot` with M=1 collapses to scalar code (no tensor cores).**
  `_optimize` tried per-head online softmax for scoring. `tl.dot`
  with a 1-row LHS emits 16 identical scalar MMAs; tensor-core path
  disabled. Keep the `[64, 128] × [128, 64] → [64, 64]` tile.
- **TileLang `T.gemm` FP8 is broken on sm_100 — confirmed three
  times.** `_optimize` iter 73 (4 workloads → INCORRECT_NUMERICAL,
  matched_ratio 0.77-0.92), v3 iter-36 (full warp-spec port,
  matched_ratio stable at 0.967 across 6 sub-variants). v3 also
  tested a **BF16 upcast in the SMEM load** (T.cast fp8→bf16 inside
  the kernel) — same 0.967 ratio, ruling out FP8 MMA precision and
  pointing at TileLang's sm_100 codegen for this specific tile shape.
  Don't retry on this Modal image.
- **`torch.topk` is slower than our custom CUB radix for this shape.**
  For N ≤ 5824, `K = 2048`, `torch.topk` internally uses CUB
  `DeviceRadixSort` which allocates scratch + launches more kernels.
  Custom single-block 8+8-bit radix is ~3× faster; even the v3
  warp-coop variant (with the BlockScan removed) is faster than
  `torch.topk`.
- **Oversized cached scores buffer breaks L2 locality (−0.26x).**
  v2 iter-2 pre-allocated `[max(B, 32), MAX_MT=5824]` bfloat16 and
  sliced per-call. Wide stride for small-MT workloads killed
  cache-line reuse between the score write and radix read. Cache
  only when the cached stride matches the actual stride.
- **num_warps=4/8 for the score kernel regress.** 26-28µs vs 16µs
  baseline. v3 also tested `num_warps=1` (iter-17): heavy slow-path
  regression (B=30: 32→29x). Triton's scheduler wants
  `num_warps=2, num_stages=1` for the `[64, 128] × [128, 64] → [64, 64]`
  FP8 tile. This is rock-solid across three sub-envs.
- **`num_stages=2/3` (pipelining) hurts scalar-dominant loops.** The
  score kernel has no outer K loop (K=128 fits a single MMA issue),
  so `num_stages>1` just inserts needless SMEM staging overhead.
- **PPB=2 (2 pages per block) regressed at multiple envs.** Three
  attempts across the three envs failed: original reshape
  `[2, 64, 128] → [128, 128]` forces SMEM roundtrip; v3's "inner
  loop, same MMA tile" variant (iter-2) regressed slow-path 20%
  (B=12: 22.84→16.11) because Triton's per-iteration scheduling for
  the single-MMA inner loop is suboptimal. Keep 1 block per (bid,
  pid) tile.
- **Merging radix pass 2 + pass 3 into one combined pass regressed
  severely.** v2 attempted "build hist2 + gather tie bucket in one
  pass over N". The per-iteration branch-predicated atomic +
  block-table lookup nearly doubled the work of pass-2.
- **Warp-coop hist for *uniform* distributions is not faster.**
  v1 LESSONS noted this; **v3 iter-16 confirmed the inverse: for
  *clustered* distributions (post-ReLU positive-only scores → narrow
  bf16 hi-byte range) it IS faster.** Always test on actual data
  distribution, not theoretical uniform.
- **Phase-2 warp-coop hist** (v3 iter-15): the filter `hi == thr_hi`
  leaves ~95% of lanes with sentinel; match_any overhead on -1
  sentinels exceeds savings. Per-thread atomic in phase-2 is optimal.
- **Persistent score kernel in Triton fails three ways** (v3
  iter-33/34/35): round-robin tile assignment scatters bids and
  loses Q L2 reuse; contiguous slicing keeps Q reuse but `tl.range`
  emits serial inner-loop code with no cross-iteration pipelining;
  `tl.range(num_stages=2)` inserts SMEM staging for Q/K loads that
  reuse within a single tile, actively harmful. **Triton can't
  express the persistent + pipelined pattern that would actually
  win.** A future CuTe DSL persistent kernel could.
- **CUDA graph capture with cluster launch (`__cluster_dims__`) hits
  runtime errors on this Modal image.** Reproduced in v2 and again
  in v3 iter-3 (raw CUDA cluster launch) and v3 iter-17 (Triton
  `num_ctas=2`). Symptom: kernel produces correct output on first
  invocation, then `unspecified launch failure` on subsequent
  trials. Conclusion: not a kernel bug, a Modal/PyTorch/CUDA-driver
  combination incompatibility. Skip cluster launch + graph capture
  until the Modal image upgrades.
- **PyTorch dlpack does not export `float8_e4m3fn`.** Affects any
  attempt to pass FP8 tensors to CuTe DSL via `from_dlpack`.
  Workarounds: (a) `.view(torch.uint8)` and bit-reinterpret in the
  kernel (CuTe DSL doesn't expose easy uint8→fp8 reinterpret as of
  4.3.4 — manual dequant arithmetic needed), or (b) `q.to(torch.
  bfloat16)` on host (cheap for Q since it's small; expensive for
  the K page cache unless cached by data_ptr — see v3 iter-38).
  **Implication for radix CuTe DSL port: NOT affected** since the
  radix consumes bf16 scores, not fp8.

## CuTe DSL gotchas hit in v3 (port these carefully)

When porting the radix kernel to CuTe DSL, expect these API quirks
(all hit in v3 iter-37/38 while exploring CuTe DSL on the score
kernel — same issues will apply to a radix port):

- **No `cute.cast` and no `cute.select`.** Cast via the type
  constructor: `cutlass.Float32(int_val)`. Conditionals: use Python
  `if`-statements (not a `where`/`select` function). For `torch.where`-
  style 3-arg select, use an `if-else` block or compute both branches
  and combine with arithmetic.
- **Early `return` is forbidden in `@cute.kernel`.** Wrap the body in
  an `if` guard instead. The CuTe DSL preprocessor explicitly rejects
  early exit with `DSLAstPreprocessorError: Early exit (return) is not
  allowed`.
- **No `cute.exp` / `cute.log`.** Use `cute.math.exp` / `cute.math.log`
  / `cute.math.exp2`.
- **`cute.arch` does not have `log`** (also confirmed by sibling LESSONS).
- **`cute.range_constexpr(N)` for compile-time loops**, not Python
  `range(N)` — the latter generates per-iteration MLIR ops instead of
  unrolling.
- **PyTorch dlpack rejects `float8_e4m3fn`.** See above.
- **Lazy compile via `cute.compile(host_fn, *cute_tensors)`**, then
  call the compiled object with the same `cute_tensors`. Cache by
  shape — recompile per `(B, M)`.

The minimal-LOC pattern (sibling's `_cute_reduce_kernel`,
`variants/cute_reduce.py` L380-414) sidesteps all of these by sticking
to direct tensor indexing + scalar fp32 + `cutlass.Float32(...)` casts
+ `cute.math.exp/log` + `cute.range_constexpr`. ~50 LOC. Use as
template.

## Cross-references

- `reference/dsa-sparse-attention/` — sibling op, same B200 backend.
  **Read its LESSONS.md before writing any CuTe DSL kernel here.**
  Particularly the "CuTe DSL reduce breakthrough" (v4 iter 25–26)
  and "v5 iter 1 split-loop refinement" sections. The minimal-LOC
  pattern at `variants/cute_reduce.py` L380-414 is the template
  for our radix CuTe DSL port.

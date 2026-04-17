# DSA Sparse Attention — Lessons

Distilled from 48 iterations across two sub-envs
(`ako4fib-run-dsa-attn/`, `ako4fib-run-dsa-attn-v2/`). Keep this short.
Raw iteration logs live in each sub-env's `ITERATIONS.md` for anyone who
wants the unabridged history.

## Operator & shape

- Op: `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`
  — decode-stage attention with Top-K sparse KV cache selection.
- Q: `q_nope [T,16,512] bf16` + `q_pe [T,16,64] bf16`.
- KV: `ckv [P,64,512] bf16` + `kpe [P,64,64] bf16` reshaped to
  `[P*64, 512]` / `[P*64, 64]` for gather.
- Indices: `sparse_indices [T,2048] int32` per token (Top-K = 2048).
- Output: `output [T,16,512] bf16` + `lse [T,16] fp32`.
- Dataset: 23 workloads, T ∈ {1 × 1, 2 × 8, 6 × 3, 7 × 3, 8 × 8},
  `num_pages = 8462`, all decode-stage (T ≤ 8).
- Hardware: B200 (148 SMs, 228 KB SMEM/SM, ~8 TB/s HBM peak,
  ~3 TB/s effective for random scatter/gather).

## Key facts

- **Expert baseline** (cached, profiled on a different GPU): ~0.53 ms.
  Score = expert_latency / our_latency so ~10 µs → ~47x.
- **Best kernel today**: `hybrid_dual_ns` at **47.43 ± 0.11x** (5-run
  variance-check). 377 LOC. Triton NS=32 BK=64 for T<3, TileLang NS=16
  BI=128 for T≥3, shared Triton reduce kernel, CUDA graph cache.
- **Latency breakdown at T=1 (≈10 µs total)**:
  - CUDA graph replay overhead: ~3 µs (hardware fixed).
  - Fwd split-K kernel (16–32 blocks): ~5 µs. HBM random gather
    (~2.3 MB at ~3 TB/s effective) dominates.
  - Reduce kernel (16 head blocks): ~2 µs. Launch-overhead bound at T=1.
- **Across-run noise floor**: CV ≈ 0.1–0.2% (std ~ 0.03–0.11x).
  Anything ≤0.2x difference between configs is noise, not signal.

## Architectural pivots that worked

### Iter 9 — TileLang split-K fwd + Triton reduce (→45.29x, matched Triton baseline)

Wrote the forward as a TileLang prim_func with on-chip `T.Pipelined` gather
+ GEMM + online softmax + split-K store. Kept the tiny Triton reduce kernel
as-is. Gained nothing over pure Triton on average but unlocked the T≥6
regime: TileLang's warp-specialized codegen hides HBM latency better when
grid is big enough. Prereq for iter 11.

### Iter 11 — Hybrid dispatch (T<3 Triton, T≥3 TileLang) → 45.90x

At T=1, grid is tiny (NS × T = 16 blocks on 148 SMs = 11% wave). Triton's
compact codegen wins because launch overhead is a bigger fraction of total
time; TileLang's warp-spec scaffolding actually hurts. At T≥6 the inverse
holds. Dispatch on T at the Python level (no kernel change), use Triton
for T<3 and TileLang for T≥3. Both paths share the same reduce kernel.

### Iter 22 — Dual-NS (Triton NS=32 BK=64 at low T, TileLang NS=16 BI=128 at high T) → 47.0x

The pivotal insight: NS ("number of splits") is the real knob. More splits
mean more blocks (better SM fill) but a heavier reduce kernel. Optimal NS
depends on T: T=1/2 wants NS=32 (double the blocks → 0.2 waves on B200),
T≥6 already has enough blocks at NS=16. Parameterize the Triton reduce
kernel on NUM_SPLITS and dispatch with the correct NS per path. Net +3%.
This is the current production kernel. BK=32/NS=64 regressed at T<3
(smaller BK kills MMA efficiency); BK=128/NS=16 regresses at low T.

## Instructive failures (grouped by lesson)

- **Register pressure kills fused single-kernel variants.** Iter 1 (fused
  split+reduce via atomics) and iter 6 (fused online softmax across all
  heads) both regressed to 23–38x. Triton has no `__launch_bounds__`;
  accumulators alive at the end of the block force register allocation
  for the whole body. Split-K + separate reduce is the robust shape.
- **`tl.dot` with M=1 collapses to scalar code (14× slower).** Iter 3
  tried per-head online softmax with M=1 tensor-core MMA. No tensor-core
  path; also generates 16 identical MMA instances. Lesson: if you
  compute attention per-head, do it via a warp-wide manual MMA, not
  `tl.dot`.
- **Hand-CUDA / WMMA is 5–6× slower than Triton codegen.** Iter 7
  (full CUDA + WMMA + per-warp scratch) → 8.10x; iter 8 (float4
  vectorized + parallel softmax) → 10.93x. Triton's `cp.async`,
  bank-conflict-free SMEM layout, and register-optimal MMA scheduling
  are unreachable without CUTLASS 3.0 / CuTe. If you want to beat
  Triton by hand, you must use CUTLASS/CuTe, not raw CUDA.
- **`data_ptr` is reused across workload processes → pre-sort cache
  aliasing.** Iter 12 cached sorted indices keyed by
  `sparse_indices.data_ptr()`. `INCORRECT_NUMERICAL` on 17/23 workloads:
  each workload runs in its own process, PyTorch's tensor allocator
  recycles the same address, but the contents differ. Lesson: cache by
  a value-dependent key, or don't cache across `run()` invocations.
- **Thread-count sweep: 256 is the sweet spot.** 128 (iter 18) and 384
  (iter 16, invalid: tile mod) and 512 (iter 17) all regress 3–9%.
- **Vectorized/bulk reduce loses sequential HBM pattern.** Iter 20
  loaded the full `[16, 512]` PO tile in one `tl.load` and multiplied by
  a broadcast weight vector. 12–15% regression. The per-split sequential
  loop was actually emitting coalesced loads; the "optimized" bulk load
  was a gather pattern in disguise.
- **SMEM-indexed parallel gather breaks a compiler optimization (~50%
  regression).** Iter 21 tried to stage sparse indices into SMEM. The
  original per-thread gather pattern was being specialized by the
  compiler (probably into a TMA-like path); staging through SMEM
  defeats it. If it's not broken, don't stage.
- **`tl.range(num_stages=N)` inserts SMEM staging for scalar loads
  (−30%).** Iter 33 wrapped the reduce kernel's split loop with
  `tl.range(num_stages=2)`. Triton inserts a `cp.async` SMEM pipeline,
  which is terrible when your "loads" are scalar broadcasts that hit
  L1. Keep `tl.static_range` for scalar-dominant inner loops.
- **2-heads-per-block reduce (iter 13, iter 36) halves parallelism
  more than it saves work.** 20–30% regressions. The reduce kernel is
  already small; halving the grid below 128 blocks on 148 SMs tanks
  utilization.
- **`pip install nvidia-cutlass-dsl` without `--no-deps` silently
  downgrades `apache-tvm-ffi` and breaks TileLang.** Iter 40 caught
  this: `NestedLoopChecker` raises a cryptic
  `'_NestedLoopCheckVisitor' object has no attribute '_inst'` on T≥6
  only (T<3 still worked because those went through the Triton path).
  Fix: `pip install --no-deps nvidia-cutlass-dsl>=4.3.4` (iter 43).

Collapsed one-liners (kept for completeness): most TileLang parameter
sweeps (iter 23–30, 39) landed within ±2% of iter 22 — not worth
enumerating. Cache-modifier `evict_first` hints (iter 32) were noise-flat
because Triton's default is already near-optimal. CuTe DSL probe
(iter 38/44/45/46/47) confirmed the Modal toolchain works end-to-end.

## HBM-floor analysis & CuTe DSL study (iter 48)

At ~10 µs total, ~5 µs goes to fwd split-K, dominated by random
gather of ~72 KB per block from `ckv_cache` + `kpe_cache`. Total HBM
traffic per invocation is ~2.3 MB at T=1 scaled up to ~18.4 MB at T=8.
At ~3 TB/s effective random bandwidth, the theoretical minimum is
~0.8 µs — we're already within 6× of that floor, and the remainder
is launch overhead + per-block setup that no codegen can eliminate.

FA4 (`Dao-AILab/flash-attention`, `flash_attn/cute/`) has the exact
top-k sparse-gather pattern in `topk_gather_kv.py`
(`CpasyncGatherKVManager`) and Blackwell MLA forward in
`flash_fwd_mla_sm100.py`. Core primitives:
```
cute.make_ptr(dtype, idx_i64, gmem, align=16)
cute.copy(gmem_tiled_copy, src_ptr_tensor, sX_tile, pred=should_load)
# with cpasync.CopyG2SOp for async gather, tcgen05.MmaF16BF16Op for tensor cores.
```
A competitive CuTe DSL port would need ~500–2000 LOC and depends on ~30
internal `flash_attn.cute.*` + `quack.*` modules (named_barrier,
softmax, mask, tile_scheduler, pipeline_custom, blackwell_helpers,
mma_sm100_desc, block_sparsity, paged_kv, pack_gqa, seqlen_info, …).
Without local GPU, every debug iteration costs ~1–2 min via Modal.
**Expected best-case outcome: match Triton+TileLang, not beat it**,
because the bottleneck is HBM random gather bandwidth, which CuTe DSL
emits the same `LDG.E` instructions for as Triton.

## What remains untried

- **Persistent kernel with cooperative grid-wide reduce.** Single
  launch, 148 blocks on B200, each block picks work from an atomic
  counter, barrier via `cooperative_groups::grid()`, last block does
  the reduce. Eliminates the 3-µs CUDA-graph replay overhead (2 kernel
  launches → 1). Blocked on Triton not having grid sync; doable in
  CuTe DSL or raw CUDA but only worth it if one can hit 47x baseline
  in the compute-only portion first.
- **FP8 data path.** If we cast `ckv_cache` / `kpe_cache` to FP8 at
  prefill time (outside our scope), the gather shrinks 2x and we can
  still MMA in BF16 via CUTLASS's mixed-precision tensor cores. This
  is FA4's direction for Hopper and Blackwell. Changes the KV cache
  format so it's a system-level decision, not a kernel tweak.
- **Focused CuTe DSL port across multiple sessions.** See iter 48
  analysis. Feasible with a dedicated 2–3 day window; probably won't
  win measurably against the current 47x but would unlock warp-spec
  experiments (producer/consumer gather vs compute) that Triton and
  TileLang can't express.

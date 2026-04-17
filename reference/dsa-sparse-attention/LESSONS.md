# DSA Sparse Attention — Lessons

Distilled from 84 iterations across three sub-envs
(`ako4fib-run-dsa-attn/`, `ako4fib-run-dsa-attn-v2/`, `ako4fib-run-dsa-sparse-b200-v4/`).
Keep this short. Raw iteration logs live in each sub-env's `ITERATIONS.md`.

**tl;dr:** the single biggest lever is replacing the Triton reduce with a
**CuTe DSL reduce** (section "CuTe DSL reduce breakthrough" below) — that
one change turned a 47.43x kernel into a **70.46x kernel** (+23x). Skip
everything else and try that first on any new sub-env.

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
  Score = expert_latency / our_latency so ~10 µs → ~47x; ~5 µs → ~100x.
- **Best kernel today**: `cute_reduce` at **70.46 ± 0.32x** (5-run on v4).
  505 LOC. Same hybrid dual-NS fwd as before (Triton T<3 / TileLang T≥3),
  but **reduce replaced by a CuTe DSL kernel**. T=1 breaks 100x for the
  first time. CUDA graph cache + tensor-identity fast path.
- **Previous best**: `hybrid_dual_ns` at 47.43 ± 0.11x (377 LOC). Retained
  as fallback and as comparison baseline for the CuTe breakthrough.
- **Latency breakdown at T=1 on the new kernel (≈5.4 µs total)**:
  - CUDA graph replay overhead: ~3 µs (hardware fixed, dominant now).
  - Fwd split-K kernel (32 blocks Triton): ~1.5 µs. Still HBM-gather bound.
  - Reduce kernel (CuTe DSL, 64 blocks): ~0.9 µs. Was ~2 µs with Triton.
- **Across-run noise floor**: CV ≈ 0.1–0.2% on v2/v3, ~0.5–0.9% on v4.
  Treat ≤0.2x as noise on v2/v3 envs; ≤1x as noise on v4-style envs.
  v4 variance is ~5x higher likely because Modal B200 GPU/thermal state
  is less consistent in the new env bootstrap.

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
This was the prior production kernel. BK=32/NS=64 regressed at T<3
(smaller BK kills MMA efficiency); BK=128/NS=16 regresses at low T.

## CuTe DSL reduce breakthrough (v4 iter 25–26, 2026-04-17) → 70.46x, +23x

Replacing the Triton reduce kernel with a CuTe DSL reduce — same algorithm,
same inputs, different codegen — gave the single biggest win in the whole
trajectory. **Every T value gained +14 to +42x.** T=1: 57 → 99. T=8: 44 → 58.

**Why it works.** The reduce is launch-bound at low T (tiny per-block work,
dominated by kernel setup) and latency-bound at high T (sequential po loads
from the split-K buffer). CuTe DSL compiles to `nvcc -arch=sm_100a` directly
and skips the Triton/MLIR compilation stages. For a small kernel like this
the per-launch overhead and per-instruction efficiency matter more than
tensor-core utilization, and CuTe wins on both.

**The simple pattern that works** (~70 LOC, no layout/copy-atom machinery):
```python
@cute.kernel
def _cute_reduce_kernel(PO, PM, PL, OUT, LSE, NS: constexpr, D_CHUNK: constexpr):
    tid, _, _ = cute.arch.thread_idx()
    tok, hd, dc = cute.arch.block_idx()
    d = tid + dc * D_CHUNK

    m_g = cutlass.Float32(-1e30)
    for s in cutlass.range_constexpr(NS):
        m_val = cutlass.Float32(PM[tok, s, hd])
        if m_val > m_g:
            m_g = m_val

    l_g = cutlass.Float32(0.0)
    acc = cutlass.Float32(0.0)
    for s in cutlass.range_constexpr(NS):
        m_val = cutlass.Float32(PM[tok, s, hd])
        w_s   = cute.math.exp(m_val - m_g)
        l_g   = l_g + w_s * cutlass.Float32(PL[tok, s, hd])
        acc   = acc + w_s * cutlass.Float32(PO[tok, s, hd, d])
    OUT[tok, hd, d] = cutlass.BFloat16(acc / l_g)

    if tid == 0 and dc == 0:
        LSE[tok, hd] = (m_g + cute.math.log(l_g)) * cutlass.Float32(1.4426950408889634)

@cute.jit
def _cute_reduce_jit(PO, PM, PL, OUT, LSE, T_sz, NS, H, D, D_CHUNK):
    _cute_reduce_kernel(PO, PM, PL, OUT, LSE, NS, D_CHUNK).launch(
        grid=(T_sz, H, D // D_CHUNK), block=(D_CHUNK, 1, 1),
    )
# Call:  _cute_reduce_jit(from_dlpack(po_4d), from_dlpack(pm), ..., Tv, NS, H, D, 128)
```

**Tuned config** (v4 iter 27–29 sweep): `D_CHUNK=128` beat 64 and 256 by ~0.7x.
That gives grid `(T, 16, 4)` with 128 threads/block, 1 D element/thread.

**What did NOT help on top** (each tested in v4 iter 30–35):
- Online single-pass softmax (2 exps per iter vs 1): −0.8x.
- 2-D-elements-per-thread with halved block size: −1.4x (parallelism loss).
- SMEM-precomputed w, thread 0 compute + broadcast: neutral (compiler CSEs
  the "redundant" per-thread compute; SMEM sync offsets any saving).
- Routing T=6 through Triton fwd on top of CuTe reduce: still −10x at T=6
  (TileLang NS=16/BI=128 fwd is strictly better than Triton NS=32 at mid-T,
  even when reduce is fast).

**Gotchas in CuTe DSL we hit** (v4 iter 14, 20, 23):
- `allocator.allocate_array(dtype, num_elems)` returns a `_Pointer` — does
  NOT support `[]=`. To get indexable storage, wrap with
  `cute.make_tensor(ptr, cute.make_layout(N))`.
- `cute.arch` has NO `log` — use `cute.math.log` (also `cute.math.exp/exp2`).
- Triton `num_ctas=2` fails when grid dim0 < 2 (at T=1 our grid is `(1, 32)`).
- TileLang `safe_idx` fragment cached inside nested `T.Parallel(BI, D)` gather
  collapses T≥6 to ~29x — the compiler can't fuse fragment scalar access into
  the parallel schedule. Keep inline `T.if_then_else(idx>=0, idx, 0)` at each
  use site.
- A first-attempt TileLang reduce with a SMEM po tile inside `T.Pipelined`
  made nvcc segfault. The CuTe DSL reduce works; don't retry the TileLang
  variant without a minimal repro.

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

## HBM-floor analysis & CuTe DSL study (iter 48, revised by v4 iter 25–26)

**The old analysis was wrong about the reduce.** We used to estimate reduce
at ~2 µs out of a ~10 µs T=1 call and the fwd split-K at ~5 µs; we
concluded "HBM gather in the fwd is the floor, reduce is already tiny."
The v4 CuTe reduce shattered that: a reduce rewrite alone cut T=1 latency
from ~10 µs to ~5.4 µs. The actual budget was closer to fwd ≈ 1.5 µs,
reduce ≈ 3 µs, CUDA graph ≈ 3 µs once you instrument the CuTe version.

The fwd-side HBM-floor reasoning is still roughly right:
total HBM traffic per invocation is ~2.3 MB at T=1 scaled up to ~18.4 MB
at T=8. At ~3 TB/s effective random bandwidth, the fwd theoretical
minimum is ~0.8 µs. After the CuTe reduce, T=1 fwd is within ~2× of
that floor; T=8 fwd is within 1.2× (it's HBM-bound and hard to shave).

**Implication:** the v3 LESSONS framed CuTe DSL fwd as "high effort, low
expected win." That was predicated on the wrong cost model. The CuTe
reduce win suggests a focused CuTe fwd could plausibly push T=1 from 100x
toward 120–140x, because its launch overhead + per-instruction efficiency
matters more at small grids than we thought.

FA4 (`Dao-AILab/flash-attention`, `flash_attn/cute/`) has the exact
top-k sparse-gather pattern in `topk_gather_kv.py`
(`CpasyncGatherKVManager`) and Blackwell MLA forward in
`flash_fwd_mla_sm100.py`. Core primitives:
```
cute.make_ptr(dtype, idx_i64, gmem, align=16)
cute.copy(gmem_tiled_copy, src_ptr_tensor, sX_tile, pred=should_load)
# with cpasync.CopyG2SOp for async gather, tcgen05.MmaF16BF16Op for tensor cores.
```
A competitive CuTe DSL **fwd** port still needs ~500–2000 LOC and depends
on ~30 internal modules. But the v4 reduce breakthrough showed that a
much simpler probe-style pattern (~70 LOC, no layout/tiled-copy/MMA
machinery) can beat Triton for kernels that aren't tensor-core-bound.

## What remains untried (post v4 iter 36)

- **CuTe DSL fwd.** The reduce win re-opens this. Now that we know the
  simple probe-style pattern beats Triton for launch-bound kernels, a
  CuTe fwd — even at first without tensor cores for the MMA, or with
  `cute.gemm` atoms for the MMA and probe-style for everything else —
  is worth trying. Expected: T=1 100x → 120-140x, T=8 58x → 65-75x.
  Fwd is harder because of the gather + softmax + MMA pipeline, but
  the winning pattern (direct `A[i, j, k]` indexing, `cute.math.exp`,
  `cutlass.range_constexpr`) is the same.
- **cp.async pipelining of the po load in the CuTe reduce.** The current
  reduce has each thread issue 32 sequential po loads. `cute.arch.cp_async_*`
  primitives + a double-buffered SMEM staging could overlap load with
  compute. Tried briefly (v4) but `allocate_array` returns a non-indexable
  pointer — use `cute.make_tensor(ptr, layout)` to get a writable handle.
  Expected: +0.5 to +1 µs on the reduce, i.e. T=1 100x → ~115x.
- **Persistent kernel with cooperative grid-wide reduce.** Still worth
  ~3 µs of CUDA-graph replay overhead if you eliminate the 2-kernel
  design. Blocked on Triton not having grid sync; CuTe DSL has it via
  cluster launches (max size 8) or cooperative kernel launches.
- **FP8 data path.** Unchanged from v3 — cast `ckv_cache` / `kpe_cache`
  to FP8 at prefill time (outside our scope), gather shrinks 2x, MMA in
  BF16 via CUTLASS mixed-precision tensor cores. This is FA4's direction.
  Changes the KV cache format → system-level decision, not a kernel tweak.

## Operational notes for the next session

- **Always start from `variants/cute_reduce.py`**, not `hybrid_dual_ns.py`.
  The CuTe DSL reduce is a +23x free lunch and there is no downside
  (same deps — `triton + tilelang + cutlass-dsl`).
- v4 Modal B200 environments have higher across-run variance than v2/v3
  (CV ~0.5-0.9% vs 0.1-0.2%). Don't waste iterations on ≤0.5x moves;
  ≥1.5x is the detection threshold for real signal.
- On a fresh env, run `bash scripts/bench.sh --variance-check 5` first
  with the byte-identical `variants/cute_reduce.py` to establish the
  env's noise floor before iterating.

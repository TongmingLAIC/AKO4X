# DSA Sparse Attention — Lessons

Distilled from 104 iterations across four sub-envs
(`ako4fib-run-dsa-attn/`, `ako4fib-run-dsa-attn-v2/`, `ako4fib-run-dsa-sparse-b200-v4/`,
`ako4fib-run-dsa-sparse-b200-v5/`). Keep this short. Raw iteration logs
live in each sub-env's `ITERATIONS.md`.

**tl;dr:** two big levers, in order:
1. Replace the Triton reduce with a **CuTe DSL reduce** (v4 iter 25–26, the
   "CuTe DSL reduce breakthrough" section below). This turned a 47.43x kernel
   into a 70.46x kernel (+23x). **Skip everything else on any new sub-env
   and start from `variants/cute_reduce_v5.py`.**
2. **Split the CuTe reduce's fused inner loop into two unrolled passes**
   (one for l_g, one for acc) so nvcc can interleave 32 PO loads with the
   FMA backbone (v5 iter 1, +1.51x on v5 baseline). Applied in
   `variants/cute_reduce_v5.py`. The compiler CSEs the duplicated
   `exp(m - m_g)` across both passes, so the "redundant" source-level
   compute is free.

After these two, the reduce kernel is a **tight local optimum** — v5
sessions 1–2 (20 iterations) confirmed every attempt at further tweaks
(rmem caching, SMEM staging, cp.async prefetch, N-way acc splits, param
sweeps) either ties or regresses. Further gains require a structural
change — full CuTe DSL fwd + cooperative-launch fusion — which was not
completed on v5 and is flagged as next-session work.

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
- **Best kernel today**: `cute_reduce_v5` at **71.05 ± 0.31x** (5-run
  variance-check on v5). 523 LOC. Same hybrid dual-NS fwd as before
  (Triton T<3 / TileLang T≥3), **reduce is a CuTe DSL kernel with the
  fused single-loop split into two unrolled passes** (one l_g, one acc).
  T=1 = 100.79 ± 0.62x (consistently >100 across all 5 variance-check runs).
  CUDA graph cache + tensor-identity fast path.
- **Prior best**: `cute_reduce` at 70.46 ± 0.32x (505 LOC, v4 iter 25–29).
  Same hybrid fwd + CuTe DSL reduce, but with the fused single-loop form
  before the iter-1 split. Retained as comparison baseline.
- **Previous best**: `hybrid_dual_ns` at 47.43 ± 0.11x (377 LOC). Retained
  as fallback and as comparison baseline for the CuTe breakthrough.
- **Latency breakdown at T=1 on the current kernel (≈5.4 µs total)**:
  - CUDA graph replay overhead: ~3 µs (hardware fixed, dominant now).
  - Fwd split-K kernel (32 blocks Triton): ~1.5 µs. Still HBM-gather bound.
  - Reduce kernel (CuTe DSL, 64 blocks): ~0.9 µs. Was ~2 µs with Triton.
- **Across-run noise floor**: CV ≈ 0.1–0.2% on v2/v3, ~0.5–0.9% on v4,
  0.04–0.44% on v5 (tight at baseline, loosens as absolute score climbs).
  Treat ≤0.2x as noise on v2/v3 envs; ≤0.5x as noise on v4/v5 envs.
  v4/v5 variance is ~5x looser than v2/v3 likely because Modal B200 GPU/
  thermal state is less consistent in the newer env bootstraps.
- **Single-run vs variance-check honesty**: on v5, single-run noise std
  is ~0.3x — *the same order as every surviving micro-opt*. Three separate
  kernel variants (iter 6/10/13) passed single-run bench and looked like
  wins until a 5-run variance-check revealed them as −1.4 to −2.0x real
  regressions. **Never accept a sub-1x change without `--variance-check 5`.**

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

## CuTe DSL reduce — iter-1 split-loop refinement (v5 iter 1, 2026-04-18) → 71.05x

A pure source-level refactor on top of the v4 CuTe DSL reduce: split the
fused single-loop into two unrolled passes, one building `l_g` from PM/PL,
one building `acc` from PO.

```python
# BEFORE (v4 cute_reduce.py, single-loop fused):
l_g = cutlass.Float32(0.0)
acc = cutlass.Float32(0.0)
for s in cutlass.range_constexpr(NS):
    m_val = cutlass.Float32(PM[tok, s, hd])
    w_s   = cute.math.exp(m_val - m_g)
    l_g   = l_g + w_s * cutlass.Float32(PL[tok, s, hd])
    acc   = acc + w_s * cutlass.Float32(PO[tok, s, hd, d])

# AFTER (v5 cute_reduce_v5.py, two unrolled passes):
l_g = cutlass.Float32(0.0)
for s in cutlass.range_constexpr(NS):
    m_val = cutlass.Float32(PM[tok, s, hd])
    l_val = cutlass.Float32(PL[tok, s, hd])
    l_g = l_g + cute.math.exp(m_val - m_g) * l_val

acc = cutlass.Float32(0.0)
for s in cutlass.range_constexpr(NS):
    m_val = cutlass.Float32(PM[tok, s, hd])
    w_s = cute.math.exp(m_val - m_g)
    o_val = cutlass.Float32(PO[tok, s, hd, d])
    acc = acc + w_s * o_val
```

**Why it works.** The fused loop chains every `acc += w*o` FMA to the
previous FMA on the `acc` accumulator. Each PO load must wait for the
prior FMA to retire before the next load can issue. Separating into two
passes gives nvcc a clean pure-load-accumulate pattern for the 32 PO
reads that it can pipeline against the FMA backbone independently.

**The "duplicated" `exp(m_val - m_g)` is free.** The compiler CSEs it
across both passes — no wasted arithmetic.

**Measured gain (v5):**
- **Net +1.51x over v5 re-measured baseline (69.54 ± 0.04x).**
- Headline +0.45x single-run, +1.51x in 5-run variance-check, ~12× the
  v5 noise floor.
- Biggest at low T: T=1 +0.40x, T=2 +1.11x. T≥6 is HBM-bound, so reduce
  micro-opts don't move it.

## v5 exhausted levers (2026-04-18, 20 iterations across 2 sessions)

After applying the iter-1 split-loop and settling at 71.05 ± 0.31x, the
CuTe DSL reduce became a tight local optimum. **Every further attempt
either tied within noise or regressed.** Don't re-run these on any new
sub-env:

- **rmem tensor caching of weights** (iter 2, 6, 10, 11, 12, 13):
  `cute.make_rmem_tensor((NS,), cutlass.Float32)` + `.fill(0.0)` + hoist
  `exp(PM[s] - m_g)` into the rmem tensor. Passed at single-run bench
  (70.4–70.6x) for iter 6/10/13, but a 5-run variance-check revealed
  iter-13 at 69.02 ± 0.11x vs the pure-exp form at 71.05 ± 0.31x. That's
  **−2.03x real, hidden by single-run noise.** Root cause: init + spill
  overhead exceeds the saved exp recompute (which nvcc CSEs anyway).
  Adding a second rmem tensor (iter 11/12/13/14 for PL / PO / combined)
  always regressed further due to register spill to local memory.
  **Conclusion: only ONE NS-sized rmem tensor fits in the reduce without
  spilling; even that is net-negative.** Prefer no rmem caching.
- **`cute.make_rmem_tensor` does NOT zero-initialize** on
  `nvidia-cutlass-dsl>=4.3.4` (v5 Modal image). Without explicit
  `.fill(0.0)`, one T=6 TileLang-path workload (`385742b2`) hits
  `INCORRECT_NUMERICAL` because the compiler reorders weight-reads in
  pass 2 across some weight-writes in pass 1. Fix is simple
  (`weights.fill(0.0)` after `make_rmem_tensor`) — but prefer no rmem
  tensor in the reduce at all (see above).
- **Sync SMEM staging of PO** (v5 iter 18): cooperatively load PO tile
  into SMEM via regular `sPO[s, tid] = PO[...]` + `sync_threads()`, then
  acc loop reads SMEM. Measured: **−1.43x** (T=1 −2.49, T=2 −3.86).
  Adds 32 SMEM writes + 1 barrier + 32 SMEM reads per thread without
  eliminating GMEM reads. Breaks nvcc's interleaved GMEM/FMA schedule.
  **Lesson: adding sync without async overlap is pure overhead.**
- **cp.async PO prefetch via full tiled_copy** (v5 iter 19b): real
  async overlap attempt. Built `cute.make_copy_atom(cpasync.CopyG2SOp(
  cache_mode=cpasync.LoadCacheMode.GLOBAL), bf16, num_bits_per_copy=128)`,
  thread layout `(D_CHUNK//16, 16)` + val layout `(1, 8)`,
  `cute.make_tiled_copy_tv(...)`, `cute.local_tile(PO[tok, None, hd, None],
  (NS, D_CHUNK), (0, dc))`, `thr_copy.partition_S/D`, `cute.copy(...)` +
  `cp_async_commit_group()` / `cp_async_wait_group(0)` / `sync_threads()`.
  Required `from_dlpack(tensor, assumed_align=16)` for 128-bit src ptr
  alignment. Compiled + correct on B200 Modal image. **Measured: −1.35x**
  (T=1 −2.23, T=2 −3.73, T≥6 flat). At only 64 blocks on 148 SMs, the
  reduce is **grid-size-limited** (6.27% achieved occupancy per NCU), not
  latency-limited. The baseline's FMA/GMEM interleave already hides
  latency via intra-block ILP; cp.async machinery adds serialized overhead
  (commit → wait → sync) that exceeds the ~0.3–0.5 µs of PO-load latency
  that could be overlapped with the m_g/l_g compute.
  - **Silver lining**: the tiled_copy scaffolding compiles + runs correctly
    on B200 Modal image with bf16 PO. **This is the hard part of writing
    a future CuTe DSL fused fwd+reduce kernel** — reuse the pattern.
- **N-way acc split for ILP** (v5 iter 17, 20): 2-way split was flat
  (+0.07x, 0.25× std — nvcc already reassociates the FMA chain at SASS
  level). 4-way split regressed **−1.76x uniformly** due to register spill
  (4 acc regs + per-iter m/o intermediates overflow the budget).
  **The 1-accumulator baseline is the register-budget optimum.**
- **D_CHUNK sweep re-test** (v5): D_CHUNK=64 (97.95x T=1) and D_CHUNK=256
  (97.50x T=1) both worse than D_CHUNK=128's 99.9–100.5x, confirming
  v4's sweep result on v5 env.
- **Triton num_warps sweep re-test** (v5 iter 7/7b): num_warps=4 gave
  T=1=96.4, T=2=83.3 (regressed). num_warps=16 crashed T=1 smoke to 83x.
  8 remains optimal.
- **TileLang params re-test** (v5 iter 8/9): BI=64/NI=2 regressed T=6/8
  by 1.7–3.9x. `TL_DISABLE_WARP_SPECIALIZED=True` hit RUNTIME_ERROR on
  all T≥6 — the current TileLang kernel relies on warp-spec.
- **evict_first cache hints on Triton** (v5 iter 4): −0.47x uniform;
  nvcc's default cache schedule is already near-optimal, hints perturb
  it negatively. Don't retry on this kernel.

## NCU profile of the v5 reduce kernel at T=1 (for future-session tuning)

- 48 regs/thread, 128 threads/block, 0 KB SMEM.
- Block Limit Registers = 10 per SM (not binding — only 64 blocks total
  across 148 SMs).
- Achieved Occupancy: **6.27%** (theoretical 62.5% — limited by small grid).
- SM Active Cycles / Elapsed Cycles: 4941 / 17536 = **28% active**.
- Memory Throughput: 35.54 GB/s (out of ~8 TB/s HBM peak = 0.4%).

**Diagnosis: grid-parallelism-limited.** Most SMs idle. The 72% inactive
time is kernel-launch and wind-down overhead, not memory/compute stalls.
The ONLY meaningful lever for this specific kernel is to **eliminate
the kernel launch entirely** by fusing with the fwd.

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

## What remains untried (post v5 session 2)

- **CuTe DSL fwd using `warp.MmaF16BF16Op` tensor cores.** The real
  remaining structural lever. Expected: T=1 100x → 120-140x, T=8 58x →
  65-75x. Build deps confirmed on v5 Modal image. Start from
  `ampere/flash_attention_v2.py` (1344 LOC) and simplify aggressively.
  Core primitives:
  ```
  cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=LoadCacheMode.GLOBAL),
                      bf16, num_bits_per_copy=128)
  cute.make_tiled_copy_tv(atom, thr_layout, val_layout)
  warp.MmaF16BF16Op(bf16, Float32, (16, 8, 16))
  cute.make_tiled_mma(op, (num_threads // 32, 1, 1), permutation_mnk=...)
  cute.gemm(tiled_mma, acc, fragA, fragB, acc)
  ```
  Scope estimate: 300-500 LOC before first compile; 2-4 iterations
  to correctness; several more to match Triton's 1.5 µs fwd time.
  **cp.async tiled_copy infrastructure from v5 iter 19b is the foundation
  — it validated compile+correctness on B200 Modal image with bf16 and
  assumed_align=16.**
- **Cooperative-launch fused fwd+reduce via `cute.arch.cooperative=True`
  + `GlobalBarrier`.** Grid-wide sync is viable at low T: for T=1 the
  cooperative grid is 32 blocks (NS=32), well under the B200 148-block
  cooperative launch max. Primitives verified (see
  `ampere/cooperative_launch.py`). Viability depends on the CuTe DSL
  fwd above. Expected: eliminate the ~0.9 µs reduce kernel plus some of
  the ~3 µs CUDA-graph-replay overhead → T=1 toward 130-150x.
- **`tcgen05.MmaF16BF16Op` (Blackwell async MMA)** via the pattern in
  `blackwell/tutorial_gemm/fp16_gemm_0.py` (`TmemAllocator`,
  `PipelineTmaUmma`). Potentially faster than Ampere-style warp MMA on
  B200 but more complex. Do warp-MMA first.
- **FP8 data path.** Unchanged — cast `ckv_cache` / `kpe_cache` to FP8
  at prefill time (outside our scope), gather shrinks 2x, MMA in BF16
  via CUTLASS mixed-precision tensor cores. System-level decision, not
  a kernel tweak.

## Operational notes for the next session

- **Always start from `variants/cute_reduce_v5.py`** (not `cute_reduce.py`
  or `hybrid_dual_ns.py`). The CuTe DSL reduce + iter-1 split-loop is a
  free +23x + +0.5x over everything that came before; same deps
  (`triton + tilelang + cutlass-dsl --no-deps`).
- v4/v5 Modal B200 environments have higher across-run variance than v2/v3
  (CV 0.4–0.9% vs 0.1–0.2%). **Do not accept sub-1x changes on single-run
  bench** — single-run noise on v5 is ~0.3x, same order as every surviving
  micro-opt, and three separate kernel variants passed single-run but
  regressed in variance-check during v5 sessions 1–2.
- **Always run `bash scripts/bench.sh --variance-check 5`** before
  accepting or rejecting any change that looks ≤1x. Tracks noise floor
  to `trajectory/noise-floor.json`.
- **Don't re-attempt** these on a fresh env — all tried on v5, all
  regressed: rmem tensor caching of weights (any form), sync SMEM
  staging of PO, cp.async PO prefetch via tiled_copy, N-way acc split
  ≥ 4, D_CHUNK ≠ 128, Triton num_warps ≠ 8, TileLang BI=64 or
  warp-spec-disabled or num_stages=2, evict_first cache hints, 2-heads-
  per-block reduce. See "v5 exhausted levers" section above for details.
- The remaining real lever is a **CuTe DSL fwd + cooperative fusion** —
  500+ LOC scope, multiple iterations to get compile+correct, reuse the
  cp.async tiled_copy scaffolding from v5 iter 19b. See "What remains
  untried" section above.

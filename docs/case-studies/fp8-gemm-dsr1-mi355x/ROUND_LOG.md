# fp8_gemm_dsr1 ROCm Campaign — Round Log (Rounds 1–5)

**Hardware:** AMD Instinct MI355X (gfx950, CDNA4), 64 CUs, HBM3e ~9 TB/s  
**Software:** `rocm/pytorch-private:mxfp8-gfx950-v26.6` (torch 2.12.0+rocm7.1, AITER 25.9)  
**Shape:** M=4096, N=7168, K=2048, 120.3 GFLOP per call  
**Anchor throughout:** `aiter_a8w8_bpreshuffle` at 72.4 µs / 1661 TFLOP/s  
**Methodology:** 5 warmup + 100 timed iterations, correctness atol ≤ max(0.5, 0.05·out_amax)

---

## Round 1 — v6_lds_swizzle — WIN

**Direction:** Add 8-byte row padding to LDS arrays to eliminate bank conflicts.

**Hypothesis:** v2's 64-byte row stride (=16 LDS banks) creates 8-way conflicts for MFMA
operand loads: 16 consecutive rows in a lane group alias into only 2 bank-pair groups.
Padding to 72 bytes (=18 banks, gcd(18,32)=2) gives 16 distinct bank offsets → conflict-free.

**Result:** 136.1 µs / 884 TFLOP/s — **WIN, 1.79× over v2 (243.7 µs)**

**Lesson:** LDS bank conflicts were the dominant bottleneck. The 8-way serialization cost
dominated everything else. Occupancy dropped from 4 blocks/CU to 3 (18KB LDS → 54KB/CU ≤ 64KB),
but eliminating conflicts more than compensated.

---

## Round 2 — v7_8wave — WIN (NEW ANCHOR)

**Direction:** Double wave count per block from 4 (WAVES_N=2) to 8 (WAVES_N=4). LDS unchanged.

**Hypothesis:** With 12 waves/CU (3 blocks × 4 waves), the 64-cycle MFMA latency is only
partially hidden. At 4-cycle MFMA issue rate, 12×4=48 cycles < 64 → stalls on re-issue.
Doubling to 24 waves/CU (3 blocks × 8 waves) gives 96-cycle gaps between re-issues on the
same wave — fully hiding the latency.

**Result:** 101.6 µs / 1184 TFLOP/s — **WIN, 1.34× over v6**

**Lesson:** Wave count was the secondary bottleneck after bank conflicts. LDS cost is zero
(18KB unchanged → still 3 blocks/CU). Gain is purely MFMA pipeline utilization.

**DRAM analysis:** v7 reads 917 MB at ~9 TB/s → theoretical minimum ≈ 102 µs. This matches
our measurement exactly — v7 is at the DRAM bandwidth ceiling.

---

## Round 3 — v8_wider_n — DEAD-END

**Direction:** Widen the N dimension: BN=128→256. Reduces DRAM reads by ~25% (917→685 MB).

**Hypothesis:** v7 is DRAM-BW limited. 25% fewer DRAM bytes → ~25% speedup → ~76 µs.

**Result:** 111.9 µs / 1075 TFLOP/s — **DEAD-END, 10% slower than v7**

**Lesson:** BN=256 forces LDS to 27KB → 2 blocks/CU (was 3). Waves/CU drops to 16 (2×8)
— exactly at the MFMA latency threshold (16×4=64 cycles = MFMA latency). Any overhead
causes stalls. Occupancy loss outweighs the DRAM reduction.

---

## Round 4 — v9_256n_16w — DEAD-END

**Direction:** Combine BN=256 with 16 waves/block (32 waves/CU). Addresses v8's threshold
problem: 32 waves/CU provides 128-cycle gap >> 64-cycle MFMA latency.

**Hypothesis:** v8's failure was MFMA latency exposure. More waves will recover the loss
while keeping the 25% DRAM reduction.

**Result:** 113.5 µs / 1060 TFLOP/s — **DEAD-END, 12% slower than v7**

**Lesson:** Wave count was NOT the limiting factor in v8. The root cause is structural:
BN=256 gives 2 blocks/CU × 64 CUs = 128 concurrent blocks vs v7's 3×64=192 blocks.
Fewer concurrent blocks → less wavefront ILP across the chip → pipeline efficiency drops.
The 25% DRAM reduction is outweighed by the ~33% fewer concurrent blocks.

**Conclusion:** v7's 128×128 tile at 3 blocks/CU is the sweet spot. BN=256 with any wave
count is worse. Larger tiles systematically lose concurrent-block ILP.

---

## Round 5 — v10_global_lds — FAILED (correctness)

**Direction:** Replace VGPR-staged uint4 tile loads with `__builtin_amdgcn_global_load_lds`.
Hypothesis: direct DRAM→LDS bypasses the VGPR round-trip, potentially saving VGPRs and
instruction count.

**Result:** **FAILED correctness check** (atol up to 1176, gate ≤8.75). Not timed.

**Root cause (ISA-level):** `global_load_lds_dword` uses M0 (a scalar register) as the LDS
base address and stores each lane's data to `M0 + lane_id × element_size`. This produces a
contiguous LDS region of size 64×4=256 bytes starting at M0. The compiler correctly identifies
lane 0's LDS address, places it in M0 via `v_readfirstlane_b32 + s_mov_b32 m0`, and issues
the instruction — but ALL lanes then write to the contiguous range M0..M0+252, regardless
of our per-lane-computed LDS addresses.

**Why this mismatches our layout:** Our padded tile has ROW_STRIDE=72 bytes (=BK+LDS_PAD=64+8).
With 4 threads per row, row k starts at byte 72k — NOT at 64k. The hardware's assumption of
consecutive 4-byte strides between lanes would need zero padding (stride=64/4=16 words between
row-starts, not 18). The 8-byte pad gap means lanes 4, 8, 12 … are always 2 LDS words (8 bytes)
away from their intended locations.

**Attempted fixes:** Three approaches for the LDS pointer address space (plain `void*`,
`(lds_uint*)` C-style cast, 1D flat array) all produced the same root error — each
changed which threads' data ended up at wrong locations but could not fix the M0-stride
mismatch. The LLVM IR confirmed `global.load.lds` was being emitted correctly with AS3
pointers; the fault is in the instruction semantics vs our layout.

**What would work:** (a) Zero padding (ROW_STRIDE=64) makes lanes stride exactly 4 bytes
apart, but restores 8-way bank conflicts on MFMA operand reads. (b) A transposed tile where
all 64 lanes of a wave contribute to the same row, padded to multiples of 64 bytes. Either
requires redesigning the entire tile layout and MFMA read phase.

---

## Summary

| round | variant | result | µs | TFLOP/s | key insight |
|------:|---------|--------|---:|--------:|-------------|
| 1 | v6_lds_swizzle | WIN | 136.1 | 884 | 8-byte row pad eliminates 8-way LDS bank conflicts — dominant bottleneck |
| 2 | v7_8wave | WIN (BEST) | 101.6 | 1184 | 8 waves/block → 24 waves/CU fully hides 64-cycle MFMA latency |
| 3 | v8_wider_n | DEAD-END | 111.9 | 1075 | BN=256 → 2 blocks/CU → 16 waves/CU → at MFMA latency threshold |
| 4 | v9_256n_16w | DEAD-END | 113.5 | 1060 | BN=256 → 128 vs 192 concurrent blocks → chip-wide ILP loss |
| 5 | v10_global_lds | FAILED | — | — | global_load_lds requires contiguous LDS stride; padded layout incompatible |

**Best achieved:** v7_8wave at 101.6 µs / 1184 TFLOP/s  
**AITER gap:** 72.4 µs / 1661 TFLOP/s (1.40× behind)  
**DRAM analysis:** v7 is at the HBM3e BW ceiling (917 MB / 9 TB/s ≈ 102 µs). Closing the
1.40× gap requires reading less DRAM per call — not further kernel microoptimization.

## Paths Forward (Beyond 5 Rounds)

1. **Stream-K / persistent kernel:** Cross-block L2 reuse to amortize the 917 MB DRAM footprint
   across multiple blocks in flight. AITER bpreshuffle likely achieves similar reuse through its
   preshuffled weight layout.
2. **Preshuffled B weights:** Offline layout transformation (done once at model load) reduces
   effective DRAM reads per inference token — the key to AITER bpreshuffle's 1.40× advantage.
3. **global_load_lds with a non-padded, transposed tile:** Requires redesigning both tile-load
   and MFMA-read phases; could eliminate VGPR staging AND bank conflicts simultaneously if the
   new layout provides conflict-free MFMA reads without row padding.

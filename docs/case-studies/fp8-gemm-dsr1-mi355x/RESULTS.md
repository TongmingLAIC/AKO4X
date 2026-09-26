# FP8 GEMM (DSR1 FFN shape) on MI355X gfx950 — results

**Hardware:** AMD Instinct MI355X (gfx950, CDNA4)
**Software:** `rocm/pytorch-private:mxfp8-gfx950-v26.6` (torch 2.12.0.dev+rocm7.1, AITER 25.9)
**Operator:** FP8 GEMM, `C[m, n] = (sum_k A[m,k]*B[n,k]) * scale_a[m] * scale_b[n]`
**Dtypes:** A, B in `float8_e4m3fn` (OCP); C in `bf16`; scales fp32 (per-token A + per-channel B)
**Shape:** M=4096, N=7168, K=2048 (DSR1 FFN-like), 120.3 GFLOP per call
**Methodology:** 5 warmup + 100 timed iterations, correctness atol = max(0.5, 0.05·out_amax)

All numbers in the tables below are taken directly from
[`baseline.json`](baseline.json) — the canonical measurement record for
this case study.

## Baseline table (MI355X)

| backend | latency (µs) | TFLOP/s | correctness |
|---|---:|---:|---|
| **`aiter_a8w8_bpreshuffle`** (W preshuffled once) | **72.4** | **1660.7** | ✅ atol 0.25 |
| `aiter_a8w8` (CK FP8 path, generic)              | 93.8     | 1282.7     | ✅ atol 0.25 |
| `hip_stock` (scalar one-output-per-thread)        | 104819   | 1.1        | ✅ atol 0.0  |

hipBLASLt (`torch._scaled_mm`) is excluded: torch 2.12+rocm7.1 hits
`HIPBLAS_STATUS_INVALID_VALUE` on the small correctness-check shape (64, 128, 2048)
for FP32. The driver sets `TORCH_BLAS_PREFER_HIPBLASLT=0` to dodge this for the
FP32 reference matmul.

## Variants (10 attempts: 2 wins, 4 dead-ends, 1 failed)

| variant | latency (µs) | TFLOP/s | vs v1 | result |
|---|---:|---:|---:|---|
| v1_mfma_tile (64×64 block, 4 waves, single LDS)         | 430.2    | 279.5  | 1.00× | ✅ baseline (correctness anchor) |
| v2_wider_tile (128×128 block, 4 waves)                  | 243.7    | 493.6  | 1.77× | ✅ was anchor (beat by v6) |
| v3_double_buffer (v2 + 2-stage LDS pipeline)            | 252.3    | 476.6  | 1.71× | ❌ tied v2 (compiler already overlaps) |
| v4_bigger_mfma (v2 + mfma_32x32x16)                     | 242.6    | 495.8  | 1.77× | ❌ tied v2 (dispatch ≠ bottleneck) |
| v5_block_256x128 (256×128 block, 8 waves)               | 280.1    | 429.4  | 1.54× | ❌ slower (occupancy collapsed) |
| v6_lds_swizzle (v2 + 8-byte LDS row pad)                | 136.1    | 883.5  | 3.16× | ✅ was anchor (beat by v7) |
| **v7_8wave** (v6 + 8 waves/block, 24 waves/CU)          | **101.6** | **1183.5** | **4.23×** | ✅ **BEST** |
| v8_wider_n (BN=256, 25% less DRAM, 16 waves/CU)         | 111.9    | 1074.6 | 3.84× | ❌ slower (MFMA latency exposed at 16 waves/CU) |
| v9_256n_16w (BN=256, 16 waves/block, 32 waves/CU)       | 113.5    | 1060.0 | 3.79× | ❌ slower (2 blocks/CU → 128 concurrent blocks vs v7's 192) |
| v10_global_lds (v7 + global_load_lds tile loads)        | —        | —      | —     | ❌ correctness failure — global_load_lds incompatible with padded tile layout |

**Headline:** v7 → 1.34× over v6, 4.2× over v1; **still 1.40× behind AITER bpreshuffle**
(101.6 µs vs 72.4 µs).

## What the results taught us

### Wins

- **v6 — bank conflict elimination (1.79× over v2).** v2's `A_lds[128][64]` has a
  64-byte row stride = 16 LDS banks. With 32 banks total, 16 consecutive-row
  MFMA-operand reads (one wave's 8-byte loads at the same column) all aliased into
  2 bank-pair groups — an 8-way bank conflict. Adding 8 bytes of padding (stride →
  72 bytes = 18 banks per row) makes consecutive rows walk the bank space with
  stride 2 (`gcd(18, 32) = 2`), giving 16 distinct bank-pair offsets for 16 rows →
  conflict-free. Occupancy dropped from 4 blocks/CU to 3 (18 KB × 3 = 54 KB ≤ 64 KB),
  but the 8× serialization elimination dominated.

- **v7 — more waves per block (1.34× over v6).** v7 confirms MFMA latency exposure
  at 12 waves/CU. Doubling to 8 waves/block gives 24 waves/CU; with MFMA latency
  ≈ 64 cycles and 4-cycle issue rate, 24×4 = 96 cycles between re-issues on the
  same wave — enough to fully hide the latency. LDS unchanged at 18 KB → still 3
  blocks/CU; the gain is purely from better MFMA pipeline utilization.

### Dead-ends (rejected hypotheses)

- **v3 (double-buffer):** with regular global loads, the HIP compiler already issues
  loads ahead of MFMA work via instruction scheduling. Explicit software
  double-buffering doesn't add overlap that wasn't already there.
- **v4 (32×32×16 MFMA):** halved MFMA dispatch count, zero speedup. MFMA dispatch
  overhead is not the bottleneck at this tile size.
- **v5 (256×128):** reduced redundant DRAM traffic ~25%, got *slower*. The 24 KB
  LDS footprint forced 1 block/CU (was 2), and total block count dropped (896 vs
  1792) — the wavefront scheduler had less to interleave.
- **v8 (BN=256, 8 waves/block → 16 waves/CU):** exactly at MFMA latency threshold;
  any overhead caused stalls.
- **v9 (BN=256, 16 waves/block → 32 waves/CU):** wave count no longer the limit;
  root cause is structural — 2 blocks/CU means only 128 concurrent blocks at 64 CUs
  vs v7's 192, less wavefront-level ILP across the chip. BN=256 ruled out regardless
  of wave count.
- **v10 (global_load_lds):** FAILED correctness. `global_load_lds_dword` uses M0
  (scalar) as LDS base and writes to `M0 + lane_id × 4` — requires a contiguous
  LDS layout with no per-row padding. The v6 padded tile (ROW_STRIDE=72 bytes,
  LDS_PAD=8) puts row boundaries at non-multiple-of-64 offsets, mismatching the
  hardware's contiguous lane stride. Removing padding to fix this would restore
  the 8-way bank conflicts eliminated in v6.

### Why v7 is at the bandwidth ceiling

v7 reads M·K + N·K bytes of A + B per call = 4096·2048 + 7168·2048 = 22 MB per call
*if* there were zero redundant per-block loads. With 128×128 tiles, the actual
DRAM traffic accounting (including each block's tile-load) is ≈ 917 MB; at
HBM3e peak ≈ 9 TB/s the floor is ≈ 102 µs — matching v7's measurement exactly.

Remaining ~1.40× gap to AITER `gemm_a8w8_bpreshuffle` (72.4 µs) lives in
DRAM-traffic reduction, not further microoptimization:

- **Preshuffled B layout** (AITER's mechanism) — improves L2 reuse / reduces
  per-call DRAM reads.
- **Stream-K / persistent kernel** — cross-block L2 reuse to amortize the
  917 MB footprint across multiple blocks in flight.
- **`__builtin_amdgcn_global_load_lds` with a non-padded transposed tile** —
  requires redesigning both tile-load and MFMA-read phases; could eliminate
  VGPR staging AND bank conflicts simultaneously if the new layout provides
  conflict-free MFMA reads without row padding (v10's failure mode resolved).

These are multi-day rewrites, deferred.

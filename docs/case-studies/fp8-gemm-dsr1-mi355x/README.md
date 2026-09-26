# Case study: FP8 GEMM closed-loop on AMD MI355X

End-to-end demonstration that the AKO4X closed-loop pattern (per-round
measurement → structured artifact → next round informed by accumulated
dead-ends) generalizes to AMD Instinct hardware.

This is a **standalone case-study artifact** — a self-contained record of
one campaign with all kernel sources, measurements, and analysis. It is
deliberately not a `reference/<family>/` archive: the variants are
`kernel.hip + binding.cpp` (not `kernel.py`), there is no `parent.txt`,
and `baseline.json` uses a bespoke `{shape, rows}` schema rather than the
workload-hash-keyed `flashinfer-bench` shape that the closed-loop machinery
seeds from. A contract-conforming `reference/<family>/` for this operator
can land later, alongside the ROCm benchmark adapter that consumes it.

## TL;DR

- **Operator:** FP8 GEMM, M=4096 × N=7168 × K=2048 (DSR1 FFN-like),
  e4m3fn → bf16, per-token A scale + per-channel B scale. 120 GFLOP.
- **Hardware:** AMD Instinct MI355X (gfx950, CDNA4), 64 CUs, HBM3e ~9 TB/s.
- **Software:** `rocm/pytorch-private:mxfp8-gfx950-v26.6` (torch 2.12+rocm7.1,
  AITER 25.9).
- **5 rounds, 2 wins, 2 dead-ends, 1 failure, ~1h 53m of agent wall-clock.**
- **Result:** **`v7_8wave` at 101.6 µs / 1184 TFLOP/s** — 4.2× faster than
  the first working MFMA kernel (v1 at 430 µs), 2.4× faster than the entry
  point of the closed-loop campaign (v2 at 244 µs), and **1.40× behind**
  AITER's `gemm_a8w8_bpreshuffle` bar (72.4 µs). v7 matches the HBM3e DRAM
  bandwidth ceiling (917 MB / 9 TB/s ≈ 102 µs).
- **Loop pattern was validated.** All 5 rounds produced structured artifacts
  (variants with Identity / Delta / Lessons / Dead-ends / Open-directions
  headers, ROUND_LOG entries with ISA-level postmortems).

All headline numbers above match `baseline.json` (the canonical
measurement record); see [`RESULTS.md`](RESULTS.md) for the full table.

## The dataset-less loop pattern

The canonical AKO4X closed-loop driven by `master/MASTER.md` +
`scripts/campaign_start.py` + `spawn.py` is tightly coupled to the
`flashinfer_bench` dataset (definitions/<op_type>/<op>.json + workloads
JSONL). That dataset's scoring path is NVIDIA-shaped today — a ROCm
equivalent is the deferred follow-up tracked by the adapter-stub PR
(`scripts/benchmark_adapter_rocm.py` + `docs/rocm-adapter-contract.md`).

To unblock the closed-loop demo without that work, this campaign used a
**dataset-less variant** of the loop:

- One Claude session played both master and engineer (instead of master
  spawning sub via `claude --print --session-id`). Single context per
  round, simpler accounting.
- The benchmark loop ran via a self-contained `benchmark_fp8_gemm.py`
  in this directory — not via `scripts/benchmark_adapter.py`. No
  `flashinfer_bench` dependency, no dataset definitions needed.
- Each round wrote its variant directly to `variants/<slug>/`. The
  "archive on win, document dead-end on tie/loss" discipline from the
  canonical pattern was preserved — see the variant headers and
  `ROUND_LOG.md`.

This is meaningfully less than the full canonical loop (no master/sub
split, no harness ledger, no scope gate), but it preserves the
load-bearing part: **per-round measurement → structured artifact → next
round informed by accumulated dead-ends**. Enough to demonstrate the
iteration pattern works on AMD; the full master/sub split can land later
once the adapter is implemented and ROCm has its own contract-conforming
`reference/<family>/`.

## The 5 rounds in one paragraph each

**Round 1 (`v6_lds_swizzle` — WIN, 1.79× over the entry anchor).** Entered
the campaign with v2 (a hand-tuned 128×128 tile, 4 waves, single-buffered
LDS) at 244 µs. The agent picked LDS bank-conflict elimination from the
"untried levers" list, computed the bank math (row stride 64 B = 16 banks
in a 32-bank LDS → 8-way conflict per MFMA operand load; padding to 72 B
gives `gcd(18, 32) = 2` → 16 distinct bank-pair offsets → conflict-free),
and verified the resulting kernel at **136 µs**. Occupancy dropped from
4 → 3 blocks/CU (18 KB × 3 = 54 KB ≤ 64 KB) but the conflict elimination
dominated.

**Round 2 (`v7_8wave` — WIN, 1.34× over v6).** Doubled the wave count per
block from 4 → 8 (24 waves/CU). Hypothesis: at 4-cycle MFMA issue and
64-cycle MFMA latency, 12 waves/CU (3 blocks × 4 waves) gives 48-cycle
gaps between re-issues — exposing stalls. 24 waves gives 96-cycle gaps.
Measured **101.6 µs / 1184 TFLOP/s**. Reads 917 MB at ~9 TB/s →
theoretical DRAM floor ≈ 102 µs. v7 is at the HBM3e bandwidth ceiling.

**Round 3 (`v8_wider_n` — DEAD-END).** Tested whether reducing DRAM reads
~25% via BN=128 → 256 would speed things up. Measured **112 µs**
(~10% slower). BN=256 forces LDS to 27 KB → 2 blocks/CU; total waves/CU
drops to 16, which is exactly at the 64-cycle MFMA latency threshold
(16×4 = 64). Any overhead causes pipeline stalls.

**Round 4 (`v9_256n_16w` — DEAD-END).** Asked: is v8's loss really MFMA
latency exposure, or is it something else? Tried BN=256 + 16 waves/block
(32 waves/CU) — should clear the latency threshold. Measured **113.5 µs**
(~12% slower than v7). Root cause is structural, not latency: BN=256
gives 128 concurrent blocks across the chip (vs v7's 192), and fewer
concurrent blocks means less chip-wide ILP. Larger N-direction tiles
systematically lose this regardless of wave count. v7's 128×128 /
3 blocks/CU is the sweet spot.

**Round 5 (`v10_global_lds` — FAILED correctness).** Attempted to use
`__builtin_amdgcn_global_load_lds_dword` for the tile-load phase (bypass
VGPR staging). Hit a fundamental incompatibility: the intrinsic writes to
`M0 + lane_id × 4` contiguous LDS addresses, but v6's bank-conflict fix
uses a 72-byte row stride (=18 LDS words, not 16). Three address-space
cast attempts confirmed the LLVM IR was emitting `global.load.lds`
correctly — the fault is in the instruction's contiguous-stride semantics
vs the padded layout. Resolving this requires redesigning both the
tile-load and MFMA-read phases to use a non-padded transposed layout —
out of scope for one round.

Full per-round detail with hypotheses, falsification criteria, and the
ISA analysis for round 5: [`ROUND_LOG.md`](ROUND_LOG.md).

## What the loop validated

1. **The Identity / Delta / Lessons / Dead-ends header convention works
   identically on AMD.** All 10 variants in this case study carry the
   same header shape as the NVIDIA-side `reference/<family>/` archives.
   Future readers (or future agents) get the same cumulative-knowledge
   effect.

2. **Dead-end documentation actually steered subsequent rounds.** Round
   3's "untried levers" list explicitly excluded the v3 / v4 / v5
   dead-end hypotheses (SW double-buffer, MFMA-shape tuning, naive
   bigger block). Round 4's hypothesis ("is v8's loss really MFMA
   latency exposure?") was a direct refinement of round 3's dead-end
   note. This is the load-bearing claim of the archive pattern, and it
   held.

3. **ISA-level postmortems live in the archive.** Round 5's failure
   isn't "we couldn't get it working"; it's a precise diagnosis of
   `global_load_lds_dword`'s `M0 + lane_id × 4` semantics
   incompatibility with the padded LDS layout — useful immediately to
   anyone attempting the same lever later.

## What's in here

```
fp8-gemm-dsr1-mi355x/
├── README.md           (this file — case-study writeup)
├── RESULTS.md          (headline measurement table + dead-end analysis)
├── ROUND_LOG.md        (round-by-round narrative + ISA-level postmortems)
├── baseline.json       (canonical measurements: AITER × 2, hip_stock, v1..v10)
├── benchmark_fp8_gemm.py    (driver: auto-discovers variants/<name>/{kernel.hip, binding.cpp})
├── baselines/
│   ├── torch_naive.py       (fp32 reference, einsum-routed to dodge hipBLASLt bug)
│   ├── hip_stock.hip        (scalar one-output-per-thread, manual e4m3fn decode)
│   └── hip_stock_binding.cpp
└── variants/
    ├── v1_mfma_tile/         (430 µs — first working MFMA kernel; correctness anchor)
    ├── v2_wider_tile/        (244 µs — round-0 entry; 128×128 block; 1.77× over v1)
    ├── v3_double_buffer/     (DEAD-END — SW pipeline doesn't add overlap)
    ├── v4_bigger_mfma/       (DEAD-END — 32×32×16 ties 16×16×32; dispatch ≠ bottleneck)
    ├── v5_block_256x128/     (DEAD-END — bigger block; occupancy collapsed)
    ├── v6_lds_swizzle/       (136 µs — round-1 WIN; 8B LDS pad, 1.79× over v2)
    ├── v7_8wave/             (101.6 µs — round-2 WIN; 8 waves/block, 1.34× over v6 — BEST)
    ├── v8_wider_n/           (DEAD-END — BN=256 → 16 waves/CU at MFMA latency threshold)
    ├── v9_256n_16w/          (DEAD-END — BN=256 + 32 waves/CU → 128 concurrent blocks vs v7's 192)
    └── v10_global_lds/       (FAILED — global_load_lds incompatible with padded LDS layout)
```

Each variant header carries an Identity / Delta / Lessons / Dead-ends /
Open-directions block. Variants are auto-discovered by
`benchmark_fp8_gemm.py` — drop a `variants/<slug>/{kernel.hip,
binding.cpp}` in place and it gets built + benchmarked next run.

## What's NOT in this case study

- **No master/sub split.** Single Claude played both master and engineer.
- **No `scripts/benchmark_adapter.py` integration.** Standalone driver only.
- **No multi-shape sweep.** Single shape; multi-shape sweeps would
  benefit from the `flashinfer-bench` workload-hash schema.

All three are paths the ROCm adapter PR is meant to unlock.

## Reproduction

On an AMD MI350X / MI355X host with docker + KFD access:

```bash
docker run --rm \
    --device /dev/kfd --device /dev/dri \
    --group-add video --group-add render \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    -v $PWD:/work -w /work/docs/case-studies/fp8-gemm-dsr1-mi355x \
    rocm/pytorch-private:mxfp8-gfx950-v26.6 \
    python benchmark_fp8_gemm.py --iters 100 --warmup 5 --out baseline.json
```

`torch._scaled_mm` (hipBLASLt) is excluded from the canonical run: torch
2.12+rocm7.1 hits `HIPBLAS_STATUS_INVALID_VALUE` on the small
correctness-check shape (64, 128, 2048) for FP32. The driver sets
`TORCH_BLAS_PREFER_HIPBLASLT=0` to dodge this for the FP32 reference
matmul.

First run JITs the AITER ops (~5 minutes total — `gemm_a8w8` ≈ 150 s,
`bpreshuffle` ≈ 135 s, others < 20 s each). Subsequent runs hit the cache
under `/opt/venv/lib/python*/site-packages/aiter/jit/build/`.

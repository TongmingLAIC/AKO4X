# References — DSA Sparse Attention (Blackwell B200)

Curated links for the next optimization session. Organized so you can
jump to exactly the bit you need.

## FlashAttention 4 (primary source)

The exact pattern we need — Blackwell top-k sparse gather + MLA.
If you port one thing to CuTe DSL, port from here.

- **Paper**: FlashAttention-4: Algorithm and Kernel Pipelining Co-Design
  for Asymmetric Hardware Scaling, Zadouri et al., arXiv 2603.05451
  (2026-03-05). HTML render lets you jump between sections.
  - Abstract & PDF: https://arxiv.org/abs/2603.05451
  - HTML (browsable): https://arxiv.org/html/2603.05451v1
  - Colfax deep-dive: https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/
- **Repo**: Dao-AILab/flash-attention `flash_attn/cute/`
  - Directory index: https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute
  - **`flash_fwd_mla_sm100.py`** — Blackwell MLA forward
    (tcgen05 + warp-spec + tx-barriers). Closest reference to our kernel:
    https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_mla_sm100.py
  - **`topk_gather_kv.py`** — `CpasyncGatherKVManager`. This IS the
    top-k sparse gather pattern our kernel does by hand:
    https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/topk_gather_kv.py
  - `flash_fwd_sm100.py` — standard FMHA fwd for SM100, simpler starting
    point than MLA:
    https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py
  - `softmax.py`, `mask.py`, `named_barrier.py`, `paged_kv.py` — support
    modules the fwd depends on.

## NVIDIA CUTLASS CuTe DSL

- **Announcement blog** (start here if you've never used CuTe DSL):
  https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/
- **Python DSL docs root**:
  https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html
- **Examples directory**:
  https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL
- **Tutorials** (ordered from simple → FMHA):
  - `ampere/elementwise_apply.py` — the simplest `@cute.kernel`
    template (thread-value layouts, `cute.composition`, predicated copy).
    https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/elementwise_apply.py
  - `blackwell/tutorial_gemm/fp16_gemm_0.py` — minimal tcgen05 MMA GEMM
    on SM100. ~200 LOC, covers TMA atoms + tensor memory allocator +
    pipeline barriers.
    https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py
  - `blackwell/fmha.py` — the FMHA reference FA4 is adapted from.
    https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py
  - `ampere/flash_attention_v2.py` — FlashAttention-2 in CuTe DSL on
    Ampere. Simpler than Blackwell (no tcgen05) and therefore a
    better first adaptation target.
    https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py
- **Blackwell architecture tutorial (Colfax)** — Tensor Memory, tcgen05,
  warp-spec patterns. Much more concrete than the NVIDIA docs.
  - Writing GEMM kernels for Blackwell:
    https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/

## Prior work in this repo

- `LESSONS.md` — distilled trajectory from 48 iterations across two
  sub-envs. Start here before proposing anything radical.
- `VARIANTS.md` — the three preserved kernels with 5-run variance data.
- `variants/hybrid_dual_ns.py` (47.43 ± 0.11x) — current production kernel.
- `variants/pure_tilelang.py` (45.37 ± 0.03x) — TileLang-only fallback.
- `variants/pure_triton.py` (45.25 ± 0.04x) — minimal dependency fallback.

## Benchmark / infra references

- `docs/benchmark.md` (in the spawned child env) — how `bench.sh` works,
  baseline caching, variance-check, trajectory diff.
- `docs/profiler/ncu.md` — NCU profiling invocation. Use
  `bash scripts/profile.sh --index N --launches` for a launch summary
  first; full NCU (`--index N` alone) is for targeted deep-dives.
- Nsight Compute CLI docs:
  https://docs.nvidia.com/nsight-compute/NsightComputeCli/
- Modal CUDA 12.8 devel image:
  `nvidia/cuda:12.8.0-devel-ubuntu22.04` on Docker Hub / nvcr.io.
- `scripts/bench.sh --variance-check N` — run same solution N times,
  report across-run mean / std / CV + most-variable workloads.
  Output: `trajectory/noise-floor.json`. Use this to establish the
  noise floor before deciding if a candidate optimization is signal.

## Hardware references

- B200 (SM100): 148 SMs, 228 KB SMEM/SM, 4th-gen tensor cores
  (tcgen05), asynchronous MMA, tensor memory. HBM3e with ~8 TB/s peak,
  **~3 TB/s effective for random scatter/gather** (our operating
  bottleneck).
- Key constraint: tensor-core throughput doubled vs H100 (SM90), but
  SMEM bandwidth and exponential units scaled less. FA4's design
  philosophy (asymmetric hardware scaling) is specifically responding
  to this — see §2 of the FA4 paper.

## Related operators worth cross-checking

- `flashinfer-python`'s `BatchMLAPagedAttentionWrapper` and
  `BatchDecodeMlaWithPagedKVCacheWrapper` — MLA reference decodes with
  a similar access pattern (paged KV cache). Not directly applicable
  (dense vs sparse KV) but same Q × K^T softmax × V skeleton.
- DeepGEMM (`deepseek-ai/DeepGEMM`) — tiny, focused, highly-optimized
  FP8 GEMM kernels. Not attention but instructive for the
  "small handwritten kernel vs giant framework" trade.

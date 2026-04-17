# References — DSA TopK Indexer (Blackwell B200)

Curated links for the next optimization session. Organized so you can jump
to exactly the bit you need.

## Operator & reference implementations

The exact operator we optimize. The reference Python implementation and
related sparse-attention wrappers are the canonical correctness baseline.

- **Flashinfer-bench operator spec** — the definition used by the harness:
  https://github.com/flashinfer-ai/flashinfer-bench
  - Per-operator `definition.json` ships the `"reference"` field as a pure
    PyTorch implementation; our bench framework diff'd against it for
    every iteration. The indexer's reference is in
    `ako4fib-run-indexer_v2/docs/definition.json` ("reference" key).
- **FlashInfer** — top-K indexer reference kernel for DSA-style decode:
  https://github.com/flashinfer-ai/flashinfer
  - `csrc/dsa_sparse_attn/` hosts the CUDA implementation we ultimately
    compete with; our kernel is not allowed to call into it but studying
    its shapes and quantization layout (`page_size=64`, `128 bytes fp8 +
    4 bytes scale per token`) is essential.
- **DeepGEMM** — FP8 matmul reference kit, same quant layout:
  https://github.com/deepseek-ai/DeepGEMM
  - `deep_gemm/fp8_gemm.cu` is the canonical fast FP8 GEMM on B200. Its
    `[64, 128] × [128, 64]` tile shape + `page_size=64, head_dim=128`
    layout is identical to our scoring shape, so the register-schedule
    patterns translate directly.
- **SGLang DSA sparse attention** — same formula, different harness:
  https://github.com/sgl-project/sglang
  - Their `weights = weights_proj(x) * n_heads^-0.5 * q_scale *
    softmax_scale` line is the source of our `weights [B, 64]` fp32
    contract. Useful if you suspect an interpretation bug.

## NVIDIA CUTLASS CuTe DSL (for Blackwell rewrite)

The remaining untried axis from `LESSONS.md` (CuTe tcgen05 score kernel)
would come from these references.

- **Announcement blog** (start here if you've never used CuTe DSL):
  https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/
- **Python DSL docs root**:
  https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html
- **Examples directory**:
  https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL
- **Tutorials** (ordered from simple → FP8 GEMM):
  - `ampere/elementwise_apply.py` — simplest `@cute.kernel` template
    (thread-value layouts, `cute.composition`, predicated copy):
    https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/elementwise_apply.py
  - `blackwell/tutorial_gemm/fp16_gemm_0.py` — minimal tcgen05 MMA GEMM
    on SM100. ~200 LOC, covers TMA atoms + tensor memory allocator +
    pipeline barriers:
    https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py
  - `blackwell/blockwise_gemm/blockwise_gemm.py` — FP8 blockwise GEMM,
    closest reference for our scoring shape:
    https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/blockwise_gemm/blockwise_gemm.py
  - Example 81 (C++ but useful prose): FP8 blockwise-scaled GEMM for SM100:
    https://github.com/NVIDIA/cutlass/blob/main/examples/81_blackwell_gemm_blockwise/81_blackwell_gemm_blockwise.cu
- **Blackwell architecture tutorial (Colfax)** — Tensor Memory, tcgen05,
  warp-spec patterns. Much more concrete than the NVIDIA docs:
  - Writing GEMM kernels for Blackwell using tensor memory:
    https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/

## CUDA Graph API (for the optimization that actually worked)

This is what iter-5/6 of v2 relied on. If you're porting to a new PyTorch
or graph framework, these are the primitives.

- **CUDA Graph programming guide** — full API reference:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
- **`torch.cuda.CUDAGraph` PyTorch docs**:
  https://docs.pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html
- **`torch.cuda.graph` context manager** (what we use):
  https://docs.pytorch.org/docs/stable/generated/torch.cuda.graph.html
- **`torch.cuda.graph_pool_handle()`** — shared memory pool across
  captured graphs (iter-8 in v2):
  https://docs.pytorch.org/docs/stable/generated/torch.cuda.graph_pool_handle.html
- **PyGraph paper** (pointer-indirection trick for dynamic inputs in
  captured graphs — this is what makes our `data_ptr()`-keyed cache
  "just work" on Modal B200):
  https://arxiv.org/html/2503.19779v1

## CUB library (for the radix topk)

The radix kernel uses `cub::BlockScan` for the parallel 256-bin threshold
finding in each radix pass.

- **CUB documentation root**:
  https://nvlabs.github.io/cub/
- **`BlockScan<T, BLOCK_DIM_X>::InclusiveSum`** — what we use:
  https://nvlabs.github.io/cub/classcub_1_1_block_scan.html
- Note: we tried `cub::BlockScan<int, BLOCK_DIM_X, 4>` (items-per-thread)
  for a 4096-bucket 12-bit radix pass and it compiled but didn't beat the
  current two-pass 8+8 structure — see `LESSONS.md`.

## Prior work in this repo

- `LESSONS.md` — distilled trajectory. Start here before proposing
  anything radical.
- `VARIANTS.md` — 2 preserved kernels with vc3 variance data.
- `variants/graph_cached.py` (41.35 ± 0.16x) — current production kernel.
- `variants/no_graph.py` (~32.7x) — fallback / illustrates graph's
  contribution.
- **`reference/dsa-sparse-attention/`** — sibling operator's distillation
  (47.43x hybrid kernel). Same Modal B200 backend, similar harness, same
  `--variance-check` methodology. Cross-reference for patterns that
  transfer across DSA operators.

## Benchmark / infra references

- `docs/benchmark.md` (in the spawned child env) — how `bench.sh`
  works, baseline caching, variance-check, trajectory diff.
- `docs/profiler/ncu.md` — NCU profiling invocation. Use
  `bash scripts/profile.sh --index N --launches` for a launch summary
  first; full NCU (`--index N` alone) is for targeted deep-dives.
- **NVIDIA Nsight Compute CLI docs**:
  https://docs.nvidia.com/nsight-compute/NsightComputeCli/
- **`bench_gpu_time_with_cupti`** (flashinfer-python) — the timing
  function we measure against. CUPTI-based, so measures GPU kernel
  activity only (not host-side Python):
  https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/testing.py
- **`scripts/bench.sh --variance-check N`** — run same solution N times,
  report across-run mean / std / CV + most-variable workloads.
  Output: `trajectory/noise-floor.json`. Use this to establish the
  noise floor before deciding if a candidate optimization is signal.
- Modal CUDA 12.8 devel image used by this harness:
  `nvidia/cuda:12.8.0-devel-ubuntu22.04` on Docker Hub / nvcr.io.

## Hardware references

- **B200 (SM100) overview**: 148 SMs, 228 KB SMEM/SM, 5th-gen tensor
  cores (tcgen05), asynchronous MMA, 256 KB tensor memory per SM.
  HBM3e with ~8 TB/s peak. Modal B200 has the usual shared-tenant
  jitter on top; our per-run noise floor reflects that.
- **Microbenchmarking NVIDIA Blackwell** (academic survey of SM100):
  https://arxiv.org/html/2512.02189v1
  - §4 (MMA single-instruction latency, tcgen05 vs Hopper wgmma)
    and §6 (TMEM read/write bandwidth) are the numbers we consulted
    when estimating the CuTe DSL rewrite's upper bound.
- **Key B200 constraint for our kernel**: FP8 tensor cores are very
  fast (10+ PFLOPS) but our scoring is compute-bound by register
  allocation, not throughput. 208 regs/thread at num_warps=2
  → `Block Limit Registers = 4` → 12.5% occupancy. This is the
  tightest knob; see `LESSONS.md` for why we can't reduce it in
  Triton.

## Related operators worth cross-checking

- **FlashInfer `BatchMLAPagedAttentionWrapper` and `BatchDecodeMlaWith
  PagedKVCacheWrapper`** — MLA paged-decode wrappers that share our
  paged-KV format. Useful for cross-checking how `block_table` is
  supposed to index into `k_index_cache_fp8`:
  https://github.com/flashinfer-ai/flashinfer
- **DeepGEMM FP8 GEMM** — small, focused, highly-optimized FP8 matmul.
  Not top-K but instructive for the "small handwritten kernel vs
  giant framework" trade, and for register-allocation intuitions on
  64×64×128 FP8 on B200:
  https://github.com/deepseek-ai/DeepGEMM
- **FlashAttention 4** (`Dao-AILab/flash-attention`, `flash_attn/cute/`)
  — not directly applicable (dense attention), but `topk_gather_kv.py`
  demonstrates a fast top-K sparse-gather pattern:
  https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute

## Competition / meta context

- **FlashInfer-Bench competition root**:
  https://bench.flashinfer.ai/
- **"Bring your own kernel" tutorial** (how operator/solution format
  is specified):
  https://bench.flashinfer.ai/docs/tutorials/bring-your-own-kernel
- **MLSys 2026 autokernel track** (why this benchmark exists):
  https://mlsys26.flashinfer.ai/
- **`flashinfer-trace` dataset** (where workloads come from):
  https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace
- **Sibling harness `AKO4ALL`** (general-purpose counterpart to this
  parent repo): https://github.com/TongmingLAIC/AKO4ALL
- **`autoresearch`** (design inspiration for agent loops):
  https://github.com/karpathy/autoresearch
- **`autokernel`** (prior art on autonomous kernel optimization):
  https://github.com/RightNow-AI/autokernel

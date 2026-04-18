# Variant: warp_coop_v3
# Source: ako4fib-run-indexer_v3/solution/kernel.py (sessions 2026-04-17/18, 32+ iters)
# Architecture: Same skeleton as graph_cached.py — Triton FP8 scoring + CUDA CUB
#               radix topk + per-shape CUDA graph capture. The score kernel is
#               unchanged except for L2 eviction hints. The radix kernel was
#               rewritten with three orthogonal micro-opts (all retained):
#                 1. Warp-cooperative gather: ballot+popc → 1 atomic per warp
#                    (instead of 1 per surviving element). 32× fewer atomics.
#                 2. Warp-cooperative phase-1 hi-byte histogram: __match_any_sync
#                    groups same-bucket lanes → 1 atomic per unique bucket per
#                    warp. Score distribution is post-ReLU positive-only with a
#                    narrow exponent range, so match rate is high (unlike the
#                    "uniform distribution" assumption in v1 LESSONS that said
#                    match_any won't help).
#                 3. Warp-only 256-bin threshold scan: replaces CUB
#                    BlockScan<int, 1024> (which wastes threads 256-1023) with
#                    a single-warp scan using 8×32 shuffle rounds. Removes the
#                    BT-must-be-1024 constraint, enabling smaller block size.
#               After (3), BT=128 became the slow-path optimum (vs the
#               BlockScan-mandated BT=1024).
# Measured: peak 41.99x single-run; 41.12-41.99x across multiple runs;
#           v2 graph_cached re-measured at 39.36-41.35x in same v3 sessions.
#           Slow-path improvements are stable across session drift:
#             B=12: 22.88 → 23.45 (+0.57x)
#             B=14: 31.97 → 32.48 (+0.51x)
#             B=30: 31.10 → 31.66 (+0.56x)
#           Headline gain over graph_cached is small (~+0.5x) but consistent
#           across runs at the slow-path tail (B≥12 with M>32). Fast-path
#           workloads (M≤32) are unchanged from graph_cached.
# Build deps: torch, triton, torch.utils.cpp_extension.load_inline (CUDA C++).
# Notable:
#   - num_warps=2, num_stages=1 on the score kernel remains optimal — same
#     as graph_cached (v1 LESSONS confirmed) — see ../LESSONS.md.
#   - The warp-coop hist DOES help here even though v1 LESSONS said it
#     wouldn't, because score data clusters in a narrow bf16 hi-byte range
#     after ReLU (positive-only, exp ~127-133), giving high warp-internal
#     bucket-collision rate.

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline


# ── CUDA kernels: warp-coop radix topk + fast-path all-tokens ──
_cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

// bf16 bit-to-ordinal: smaller ordinal = larger float (top-K scan from 0).
__device__ __forceinline__ unsigned short bf16_ord(unsigned short b) {
    return (b & 0x8000u) ? b : (unsigned short)((~b) & 0x7FFFu);
}

// Warp-only inclusive scan over the 256-bin hi/lo histogram.
// Thread 0..31 of warp 0 do 8 iterations of 32-lane shuffle prefix-sum;
// other warps wait at the next __syncthreads(). Replaces CUB
// BlockScan<int, BT> (which forced BT==1024 to align with the scan size).
__device__ __forceinline__ void find_threshold_warp(
    int* hist, int base_count, int TOPK, int* out_thr, int* out_count_lt
) {
    if (threadIdx.x < 32) {
        int lane = threadIdx.x;
        int prefix = 0;
        int need = TOPK - base_count;
        #pragma unroll
        for (int iter = 0; iter < 8; iter++) {
            int idx = iter * 32 + lane;
            int val = hist[idx];
            int incl = val;
            #pragma unroll
            for (int offset = 1; offset < 32; offset *= 2) {
                int tmp = __shfl_up_sync(0xFFFFFFFFu, incl, offset);
                if (lane >= offset) incl += tmp;
            }
            int my_incl = incl + prefix;
            int my_excl = my_incl - val;
            if (my_incl >= need && my_excl < need) {
                *out_thr = idx;
                *out_count_lt = base_count + my_excl;
            }
            prefix += __shfl_sync(0xFFFFFFFFu, incl, 31);
        }
    }
}

template<int BT>
__global__ __launch_bounds__(BT)
void radix_select_topk_kernel(
    const __nv_bfloat16* __restrict__ scores,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    int* __restrict__ output,
    int MT, int stride_s, int stride_bt, int stride_out, int TOPK, int PS
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int sl = seq_lens[bid];

    if (sl <= TOPK) {
        for (int k = tid; k < TOPK; k += BT) {
            int tok = -1;
            if (k < sl) {
                int pg = k / PS;
                tok = __ldg(&block_table[bid * stride_bt + pg]) * PS + (k & (PS - 1));
            }
            output[bid * stride_out + k] = tok;
        }
        return;
    }

    const unsigned short* __restrict__ raw =
        reinterpret_cast<const unsigned short*>(scores + bid * stride_s);
    const unsigned long long* __restrict__ raw4 =
        reinterpret_cast<const unsigned long long*>(raw);
    const int N = sl;
    const int N4 = N >> 2;
    const int N4_tail = N4 << 2;

    __shared__ int hist[256];
    __shared__ int s_thr_hi, s_count_lt, s_out_cnt, s_cnt_tie;

    // Vectorized init: 64 int4 stores instead of 256 int stores.
    if (tid < 64) {
        reinterpret_cast<int4*>(hist)[tid] = make_int4(0, 0, 0, 0);
    }
    if (tid == 0) { s_thr_hi = 0; s_count_lt = 0; }
    __syncthreads();

    // Phase 1 hi-byte histogram with warp-coop match_any: each warp issues
    // ONE atomic per unique bucket instead of 1 per element. Score data is
    // post-ReLU positive only with narrow exp range → high match rate.
    {
        const unsigned FULL_MASK = 0xFFFFFFFFu;
        const int lane = tid & 31;
        const int n_iters4 = (N4 + BT - 1) / BT;
        for (int iter = 0; iter < n_iters4; iter++) {
            int i = iter * BT + tid;
            bool in_range = (i < N4);
            unsigned long long v = in_range ? raw4[i] : 0ULL;
            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                unsigned short b = (unsigned short)(v >> (16 * sub));
                int bucket = in_range ? (int)(bf16_ord(b) >> 8) : -1;
                unsigned peers = __match_any_sync(FULL_MASK, bucket);
                int leader = __ffs(peers) - 1;
                int count = __popc(peers);
                if (bucket >= 0 && lane == leader) {
                    atomicAdd(&hist[bucket], count);
                }
            }
        }
        for (int i = N4_tail + tid; i < N; i += BT)
            atomicAdd(&hist[(int)(bf16_ord(raw[i]) >> 8)], 1);
    }
    __syncthreads();

    find_threshold_warp(hist, 0, TOPK, &s_thr_hi, &s_count_lt);
    __syncthreads();

    int thr_hi = s_thr_hi;
    int count_lt_stage1 = s_count_lt;

    if (tid < 64) {
        reinterpret_cast<int4*>(hist)[tid] = make_int4(0, 0, 0, 0);
    }
    __syncthreads();

    // Phase 2 (lo-byte filtered): no warp-coop here. The filter
    // `hi == thr_hi` masks ~95% of lanes, so match_any with sentinel -1
    // adds overhead exceeding the savings. Per-thread atomic is optimal.
    for (int i = tid; i < N4; i += BT) {
        unsigned long long v = raw4[i];
        unsigned short b0 = (unsigned short)(v), b1 = (unsigned short)(v >> 16);
        unsigned short b2 = (unsigned short)(v >> 32), b3 = (unsigned short)(v >> 48);
        unsigned short o0 = bf16_ord(b0), o1 = bf16_ord(b1), o2 = bf16_ord(b2), o3 = bf16_ord(b3);
        if ((int)(o0 >> 8) == thr_hi) atomicAdd(&hist[(int)(o0 & 0xFFu)], 1);
        if ((int)(o1 >> 8) == thr_hi) atomicAdd(&hist[(int)(o1 & 0xFFu)], 1);
        if ((int)(o2 >> 8) == thr_hi) atomicAdd(&hist[(int)(o2 & 0xFFu)], 1);
        if ((int)(o3 >> 8) == thr_hi) atomicAdd(&hist[(int)(o3 & 0xFFu)], 1);
    }
    for (int i = N4_tail + tid; i < N; i += BT) {
        unsigned short o = bf16_ord(raw[i]);
        if ((int)(o >> 8) == thr_hi) atomicAdd(&hist[(int)(o & 0xFFu)], 1);
    }
    __syncthreads();

    // Merged init for second find_threshold + gather counters.
    if (tid == 0) {
        s_thr_hi = 0;
        s_count_lt = count_lt_stage1;
        s_out_cnt = 0;
        s_cnt_tie = 0;
    }
    __syncthreads();
    find_threshold_warp(hist, count_lt_stage1, TOPK, &s_thr_hi, &s_count_lt);
    __syncthreads();

    // Pack threshold locally — no extra __syncthreads() required.
    int threshold16 = (thr_hi << 8) | s_thr_hi;
    int count_lt = s_count_lt;

    // Warp-cooperative gather: 32× fewer atomics via ballot+popc.
    // Warp-uniform iteration count so __ballot_sync(FULL_MASK, ...) always
    // sees all 32 lanes (matters when N mod BT lands mid-warp).
    const unsigned FULL_MASK = 0xFFFFFFFFu;
    const int lane = tid & 31;
    const unsigned lane_bit_below = (1u << lane) - 1u;
    const int n_iters = (N + BT - 1) / BT;
    for (int iter = 0; iter < n_iters; iter++) {
        int i = iter * BT + tid;
        bool in_range = (i < N);
        int key = in_range ? (int)bf16_ord(raw[i]) : (threshold16 + 1);
        bool surv_lt  = in_range && (key <  threshold16);
        bool surv_tie = in_range && (key == threshold16);

        unsigned mask_lt  = __ballot_sync(FULL_MASK, surv_lt);
        unsigned mask_tie = __ballot_sync(FULL_MASK, surv_tie);
        int n_lt  = __popc(mask_lt);
        int n_tie = __popc(mask_tie);
        int warp_rank_lt  = __popc(mask_lt  & lane_bit_below);
        int warp_rank_tie = __popc(mask_tie & lane_bit_below);

        int base_lt  = 0;
        int base_tie = 0;
        if (lane == 0) {
            if (n_lt)  base_lt  = atomicAdd(&s_out_cnt, n_lt);
            if (n_tie) base_tie = atomicAdd(&s_cnt_tie, n_tie);
        }
        base_lt  = __shfl_sync(FULL_MASK, base_lt,  0);
        base_tie = __shfl_sync(FULL_MASK, base_tie, 0);

        if (surv_lt) {
            int pg = i / PS;
            int tok = __ldg(&block_table[bid * stride_bt + pg]) * PS + (i & (PS - 1));
            output[bid * stride_out + base_lt + warp_rank_lt] = tok;
        }
        if (surv_tie) {
            int pos = count_lt + base_tie + warp_rank_tie;
            if (pos < TOPK) {
                int pg = i / PS;
                int tok = __ldg(&block_table[bid * stride_bt + pg]) * PS + (i & (PS - 1));
                output[bid * stride_out + pos] = tok;
            }
        }
    }
}

torch::Tensor radix_select_topk(
    torch::Tensor scores, torch::Tensor block_table, torch::Tensor seq_lens,
    torch::Tensor output, int TOPK, int PS
) {
    int B = scores.size(0);
    int MT = scores.size(1);
    constexpr int BT = 128;  // smaller block fits more per-SM (was 1024 before warp scan removed CUB constraint)
    radix_select_topk_kernel<BT><<<B, BT>>>(
        reinterpret_cast<const __nv_bfloat16*>(scores.data_ptr()),
        block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
        output.data_ptr<int>(),
        MT, scores.stride(0), block_table.stride(0), output.stride(0), TOPK, PS);
    return output;
}
"""

_cuda_module = load_inline(
    name="radix_topk_warp_coop_v3",
    cpp_sources=[
        "torch::Tensor radix_select_topk(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int);",
    ],
    cuda_sources=_cuda_src,
    functions=["radix_select_topk"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


# ── Triton kernels ──

@triton.jit
def _all_tokens_kernel(
    bt_ptr, sl_ptr, out_ptr,
    stride_bt_b, stride_out_b,
    TOPK: tl.constexpr, PS: tl.constexpr,
):
    bid = tl.program_id(0)
    seq_len = tl.load(sl_ptr + bid)
    k = tl.arange(0, TOPK)
    valid = k < seq_len
    pg = k // PS
    off = (k % PS).to(tl.int32)
    phys = tl.load(bt_ptr + bid * stride_bt_b + pg, mask=valid, other=0)
    tok = phys.to(tl.int32) * PS + off
    tok = tl.where(valid, tok, -1)
    tl.store(out_ptr + bid * stride_out_b + k, tok)


@triton.jit
def _dsa_score_kernel(
    q_ptr, k_fp8_ptr, k_f32_ptr, w_ptr, sl_ptr, bt_ptr, out_ptr,
    stride_q_b, stride_bt_b, stride_s_b,
    M,
    KS: tl.constexpr, KSF: tl.constexpr, SOF: tl.constexpr,
    NH: tl.constexpr, PS: tl.constexpr, HD: tl.constexpr,
):
    bid = tl.program_id(0)
    pid = tl.program_id(1)
    ti = tl.arange(0, PS)
    out_base = bid * stride_s_b + pid * PS
    seq_len = tl.load(sl_ptr + bid)
    n_pages = tl.cdiv(seq_len, PS)
    if pid >= n_pages:
        return
    phys_page = tl.load(bt_ptr + bid * stride_bt_b + pid)
    hi = tl.arange(0, NH)
    di = tl.arange(0, HD)
    # eviction hints: Q is reused across M blocks per batch (evict_last keeps
    # it in L2); K and scale are one-shot per block (evict_first).
    q = tl.load(q_ptr + bid * stride_q_b + hi[:, None] * HD + di[None, :],
                eviction_policy="evict_last")
    k = tl.load(k_fp8_ptr + phys_page * KS + ti[:, None] * HD + di[None, :],
                eviction_policy="evict_first")
    S = tl.dot(q, tl.trans(k))
    S = tl.maximum(S, 0.0)
    w = tl.load(w_ptr + bid * NH + hi, eviction_policy="evict_last")
    S *= w[:, None]
    scores = tl.sum(S, axis=0)
    scale = tl.load(k_f32_ptr + phys_page * KSF + SOF + ti,
                    eviction_policy="evict_first")
    scores *= scale
    # Radix reads only positions [0, seq_len) so positions >= seq_len are dead;
    # skip the tl.where(valid, ..., -inf) mask entirely.
    tl.store(out_ptr + out_base + ti, scores.to(tl.bfloat16))


# ── Buffer caching + per-shape graph cache ──
_output_buf = None
_scores_cache = {}
_graph_cache = {}
_graph_pool = None


def _run_kernels(q_index_fp8, k_fp8, k_f32, weights, seq_lens, block_table, scores, output, B, M, MT):
    _dsa_score_kernel[(B, M)](
        q_index_fp8, k_fp8, k_f32, weights, seq_lens, block_table, scores,
        q_index_fp8.stride(0), block_table.stride(0), MT,
        M,
        KS=8448, KSF=2112, SOF=2048, NH=64, PS=64, HD=128,
        num_warps=2, num_stages=1,
    )
    _cuda_module.radix_select_topk(scores, block_table, seq_lens, output, 2048, 64)


@torch.no_grad()
def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    global _output_buf, _graph_cache, _graph_pool

    B, M = block_table.shape
    device = block_table.device

    if _output_buf is None or _output_buf.shape[0] < B:
        _output_buf = torch.empty(max(B, 32), 2048, dtype=torch.int32, device=device)
    output = _output_buf[:B]

    cache_key = (
        B, M,
        q_index_fp8.data_ptr(), k_index_cache_fp8.data_ptr(),
        weights.data_ptr(), seq_lens.data_ptr(), block_table.data_ptr(),
    )
    cached = _graph_cache.get(cache_key)
    if cached is not None:
        graph, cached_output = cached[0], cached[1]
        graph.replay()
        return (cached_output,)

    if M <= 32:
        # Fast path (single kernel, all valid tokens emitted).
        def run_fast():
            _all_tokens_kernel[(B,)](
                block_table, seq_lens, output,
                block_table.stride(0), output.stride(0),
                TOPK=2048, PS=64,
            )
        run_fast()
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run_fast()
            torch.cuda.current_stream().wait_stream(stream)
            if _graph_pool is None:
                _graph_pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, pool=_graph_pool):
                run_fast()
            _graph_cache[cache_key] = (graph, output, None)
        except Exception:
            pass
        return (output,)

    NP = k_index_cache_fp8.shape[0]
    MT = M * 64
    k_raw = k_index_cache_fp8.view(torch.uint8).view(NP, 8448)
    k_fp8 = k_raw.view(torch.float8_e4m3fn)
    k_f32 = k_raw.view(torch.float32)

    shape_key = (B, MT)
    scores = _scores_cache.get(shape_key)
    if scores is None:
        scores = torch.empty((B, MT), dtype=torch.bfloat16, device=device)
        _scores_cache[shape_key] = scores

    _run_kernels(q_index_fp8, k_fp8, k_f32, weights, seq_lens, block_table, scores, output, B, M, MT)

    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _run_kernels(q_index_fp8, k_fp8, k_f32, weights, seq_lens, block_table, scores, output, B, M, MT)
        torch.cuda.current_stream().wait_stream(stream)

        if _graph_pool is None:
            _graph_pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, pool=_graph_pool):
            _run_kernels(q_index_fp8, k_fp8, k_f32, weights, seq_lens, block_table, scores, output, B, M, MT)
        _graph_cache[cache_key] = (graph, output, scores)
    except Exception:
        pass

    return (output,)

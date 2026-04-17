# Variant: graph_cached
# Source: ako4fib-run-indexer_v2/solution/kernel.py (final commit 96d860b)
# Architecture: Triton FP8 scoring + CUDA CUB radix topk + per-shape CUDA
#               graph capture (score + radix captured as a single replay unit).
# Measured: 41.35 +/- 0.16x (CV 0.4%, 3-run variance-check on 2026-04-17).
#   Per-group: B=1->56.2  B=4->49.9  B=8->50.2  B=11->49.1  B=16->27.0
#              B=30->30.2  (full table in ../VARIANTS.md)
# Build deps: torch, triton, torch.utils.cpp_extension.load_inline (CUDA C++).
# Notable: The CUDA-graph layer (lines ~230-331 of the packed file) is the
#          entire performance gap vs no_graph.py; the Triton scoring and CUDA
#          radix kernels are byte-identical across the two variants.

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline


# ── CUDA kernels: radix topk + fast-path all-tokens ──
_cuda_src = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cub/block/block_scan.cuh>

// bf16 bit-to-ordinal: smaller ordinal = larger float (for top-K scan from 0)
__device__ __forceinline__ unsigned short bf16_ord(unsigned short b) {
    return (b & 0x8000u) ? b : (unsigned short)((~b) & 0x7FFFu);
}

__device__ __forceinline__ void hist4(unsigned long long v, int* hist) {
    unsigned short b0 = (unsigned short)(v);
    unsigned short b1 = (unsigned short)(v >> 16);
    unsigned short b2 = (unsigned short)(v >> 32);
    unsigned short b3 = (unsigned short)(v >> 48);
    atomicAdd(&hist[(int)(bf16_ord(b0) >> 8)], 1);
    atomicAdd(&hist[(int)(bf16_ord(b1) >> 8)], 1);
    atomicAdd(&hist[(int)(bf16_ord(b2) >> 8)], 1);
    atomicAdd(&hist[(int)(bf16_ord(b3) >> 8)], 1);
}

template <int BT>
__device__ __forceinline__ void find_threshold_parallel(
    int* hist, int base_count, int TOPK, int* out_thr, int* out_count_lt
) {
    typedef cub::BlockScan<int, BT> BlockScanT;
    __shared__ typename BlockScanT::TempStorage scan_storage;
    int my_val = (threadIdx.x < 256) ? hist[threadIdx.x] : 0;
    int incl;
    BlockScanT(scan_storage).InclusiveSum(my_val, incl);
    int excl = incl - my_val;
    int need = TOPK - base_count;
    if (threadIdx.x < 256 && incl >= need && excl < need) {
        *out_thr = threadIdx.x;
        *out_count_lt = base_count + excl;
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

    if (tid < 256) hist[tid] = 0;
    if (tid == 0) { s_thr_hi = 0; s_count_lt = 0; }
    __syncthreads();

    for (int i = tid; i < N4; i += BT) hist4(raw4[i], hist);
    for (int i = N4_tail + tid; i < N; i += BT)
        atomicAdd(&hist[(int)(bf16_ord(raw[i]) >> 8)], 1);
    __syncthreads();

    find_threshold_parallel<BT>(hist, 0, TOPK, &s_thr_hi, &s_count_lt);
    __syncthreads();

    int thr_hi = s_thr_hi;
    int count_lt_stage1 = s_count_lt;

    if (tid < 256) hist[tid] = 0;
    __syncthreads();

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

    if (tid == 0) { s_thr_hi = 0; s_count_lt = count_lt_stage1; }
    __syncthreads();
    find_threshold_parallel<BT>(hist, count_lt_stage1, TOPK, &s_thr_hi, &s_count_lt);
    __syncthreads();

    if (tid == 0) {
        s_thr_hi = (thr_hi << 8) | s_thr_hi;
        s_out_cnt = 0;
        s_cnt_tie = 0;
    }
    __syncthreads();

    int threshold16 = s_thr_hi;
    int count_lt = s_count_lt;

    for (int i = tid; i < N; i += BT) {
        int key = (int)bf16_ord(raw[i]);
        if (key <= threshold16) {
            int pg = i / PS;
            int tok = __ldg(&block_table[bid * stride_bt + pg]) * PS + (i & (PS - 1));
            if (key < threshold16) {
                int rank = atomicAdd(&s_out_cnt, 1);
                output[bid * stride_out + rank] = tok;
            } else {
                int rank = atomicAdd(&s_cnt_tie, 1);
                int pos = count_lt + rank;
                if (pos < TOPK)
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
    radix_select_topk_kernel<1024><<<B, 1024>>>(
        reinterpret_cast<const __nv_bfloat16*>(scores.data_ptr()),
        block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
        output.data_ptr<int>(),
        MT, scores.stride(0), block_table.stride(0), output.stride(0), TOPK, PS);
    return output;
}

"""

_cuda_module = load_inline(
    name="radix_topk_base_v2",
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
    q = tl.load(q_ptr + bid * stride_q_b + hi[:, None] * HD + di[None, :])
    k = tl.load(k_fp8_ptr + phys_page * KS + ti[:, None] * HD + di[None, :])
    S = tl.dot(q, tl.trans(k))
    S = tl.maximum(S, 0.0)
    w = tl.load(w_ptr + bid * NH + hi)
    S *= w[:, None]
    scores = tl.sum(S, axis=0)
    scale = tl.load(k_f32_ptr + phys_page * KSF + SOF + ti)
    scores *= scale
    valid = (pid * PS + ti) < seq_len
    scores = tl.where(valid, scores, float('-inf'))
    tl.store(out_ptr + out_base + ti, scores.to(tl.bfloat16))


# ── Buffer caching ──
_output_buf = None
_scores_cache = {}  # (B, MT) -> reusable scores tensor (shared across trials of same shape)
_graph_cache = {}  # key = (B, M, q_ptr, k_ptr, w_ptr, sl_ptr, bt_ptr) -> (graph, output_slice, scores_ref)
_graph_pool = None  # shared CUDA graph memory pool across all graphs


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

    # Graph cache keyed on shape + all tensor data pointers.
    # data_ptr() is slightly slower than id() but catches cross-trial pointer reuse
    # (PyTorch caching allocator often reuses same address for same-shape tensors).
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
        # Fast path (single kernel)
        def run_fast():
            _all_tokens_kernel[(B,)](
                block_table, seq_lens, output,
                block_table.stride(0), output.stride(0),
                TOPK=2048, PS=64,
            )
        run_fast()  # eager warmup (also serves as result if graph capture fails)
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run_fast()  # side-stream warmup
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

    # Reuse scores tensor per-shape (exact stride) to keep graph memory pool stable
    shape_key = (B, MT)
    scores = _scores_cache.get(shape_key)
    if scores is None:
        scores = torch.empty((B, MT), dtype=torch.bfloat16, device=device)
        _scores_cache[shape_key] = scores

    # Run once eagerly (warmup for graph capture)
    _run_kernels(q_index_fp8, k_fp8, k_f32, weights, seq_lens, block_table, scores, output, B, M, MT)

    # Capture into graph on dedicated stream
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

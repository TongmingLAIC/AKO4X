// === Identity ===
// v3_double_buffer — v2's tile shape + double-buffered LDS so the next K
// iteration's global loads overlap with the current iteration's MFMA work.
// Halves the effective sync cost.
//
// === DEAD-END: did not beat v2 ===
// v3 measured 273µs vs v2's 265µs — no measurable speedup. Two reasons:
// (1) On regular global loads, the compiler/scheduler already issues loads
//     ahead of MFMA work; explicit double-buffer doesn't add overlap that
//     wasn't already there.
// (2) We're compute-bound at this tile shape (9% of peak MFMA throughput),
//     not load-latency-bound. Halving sync cost matters less than getting
//     more compute per dispatch.
// LDS footprint also doubled to 32KB (was 16KB), which can hurt occupancy
// on CUs with multiple blocks.
//
// To actually overlap loads with compute, would need:
//   * __builtin_amdgcn_global_load_lds — DRAM→LDS direct, no VGPR staging.
//   * Manual s_waitcnt control.
// Pursued instead in v4+ via bigger MFMA (32x32x16) and other levers.
//
// === Delta from v2_wider_tile ===
// 1) A_lds and B_lds are [2][...] — two LDS buffers, alternated each K iter.
// 2) Prologue: load buffer 0, sync.
// 3) Main loop iters [1..N): kick off load into buf^1, run MFMA on buf, sync, swap.
// 4) Epilogue iter: run MFMA on final buf.
// 5) Each iteration still has 1 __syncthreads (was 2 in v2).
//
// === Lessons ===
// LDS doubles: 16KB → 32KB. Still well within MI350X's 64KB/CU LDS limit
// (and 8 blocks/CU max for occupancy purposes).
//
// === Open directions (v4+) ===
// * __builtin_amdgcn_global_load_lds — DRAM→LDS direct (no VGPR staging).
// * Hoist scale_a/scale_b loads + packed bf16 store in epilogue.
// * mfma_f32_32x32x16_fp8_fp8 — bigger MFMA, half the dispatch count.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define BM 128
#define BN 128
#define BK 64
#define WAVES_M 2
#define WAVES_N 2
#define WAVE_SIZE 64
#define NUM_WAVES (WAVES_M * WAVES_N)
#define NUM_THREADS (NUM_WAVES * WAVE_SIZE)

#define WAVE_TILE_M (BM / WAVES_M)
#define WAVE_TILE_N (BN / WAVES_N)
#define MFMA_M 16
#define MFMA_N 16
#define MFMA_K 32
#define WAVE_MFMA_M (WAVE_TILE_M / MFMA_M)   // 4
#define WAVE_MFMA_N (WAVE_TILE_N / MFMA_N)   // 4
#define BLOCK_MFMA_K (BK / MFMA_K)           // 2

#define A_BYTES (BM * BK)
#define BYTES_PER_THREAD (A_BYTES / NUM_THREADS)
#define U4_PER_ROW (BK / 16)
#define U4_PER_THREAD (BYTES_PER_THREAD / 16)

typedef long long i64;
typedef float float4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ i64 load_fp8x8(const unsigned char* src) {
    return *reinterpret_cast<const i64*>(src);
}

template <int ROWS>
__device__ __forceinline__ void cooperative_load(
    unsigned char (&dst)[ROWS][BK], const unsigned char* base, int outer_off, int k0, int K, int tid)
{
    #pragma unroll
    for (int i = 0; i < U4_PER_THREAD; ++i) {
        int u4_idx = tid * U4_PER_THREAD + i;
        int row = u4_idx / U4_PER_ROW;
        int col_u4 = u4_idx % U4_PER_ROW;
        int col_b = col_u4 * 16;
        const unsigned char* src = base + (outer_off + row) * K + (k0 + col_b);
        *reinterpret_cast<uint4*>(&dst[row][col_b]) = *reinterpret_cast<const uint4*>(src);
    }
}

__device__ __forceinline__ void compute_block(
    unsigned char (&A_lds)[BM][BK], unsigned char (&B_lds)[BN][BK],
    float4_t (&acc)[WAVE_MFMA_M][WAVE_MFMA_N],
    int wave_m_off, int wave_n_off, int lane_id)
{
    int lane_outer = lane_id & 15;
    int lane_k_grp = lane_id >> 4;
    #pragma unroll
    for (int kk = 0; kk < BLOCK_MFMA_K; ++kk) {
        int k_off = kk * MFMA_K + lane_k_grp * 8;
        #pragma unroll
        for (int mi = 0; mi < WAVE_MFMA_M; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < WAVE_MFMA_N; ++ni) {
                int a_row = wave_m_off + mi * MFMA_M + lane_outer;
                int b_row = wave_n_off + ni * MFMA_N + lane_outer;
                i64 a_reg = load_fp8x8(&A_lds[a_row][k_off]);
                i64 b_reg = load_fp8x8(&B_lds[b_row][k_off]);
                acc[mi][ni] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                    a_reg, b_reg, acc[mi][ni], 0, 0, 0);
            }
        }
    }
}

__global__ void __launch_bounds__(NUM_THREADS)
fp8_gemm_v3_kernel(
    const unsigned char* __restrict__ a,
    const unsigned char* __restrict__ b,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    __hip_bfloat16* __restrict__ c,
    int M, int N, int K)
{
    const int block_m = blockIdx.x * BM;
    const int block_n = blockIdx.y * BN;
    const int tid = threadIdx.x;
    const int wave_id = tid / WAVE_SIZE;
    const int lane_id = tid & (WAVE_SIZE - 1);
    const int wave_m_off = (wave_id / WAVES_N) * WAVE_TILE_M;
    const int wave_n_off = (wave_id % WAVES_N) * WAVE_TILE_N;

    __shared__ unsigned char A_lds[2][BM][BK];
    __shared__ unsigned char B_lds[2][BN][BK];

    float4_t acc[WAVE_MFMA_M][WAVE_MFMA_N];
    #pragma unroll
    for (int i = 0; i < WAVE_MFMA_M; ++i)
        #pragma unroll
        for (int j = 0; j < WAVE_MFMA_N; ++j)
            acc[i][j] = (float4_t){0.f, 0.f, 0.f, 0.f};

    // Prologue: load buffer 0.
    cooperative_load(A_lds[0], a, block_m, 0, K, tid);
    cooperative_load(B_lds[0], b, block_n, 0, K, tid);
    __syncthreads();

    int buf = 0;
    int k0 = BK;
    for (; k0 < K; k0 += BK) {
        int next = buf ^ 1;
        // Kick off next-iter loads (independent of current MFMA work).
        cooperative_load(A_lds[next], a, block_m, k0, K, tid);
        cooperative_load(B_lds[next], b, block_n, k0, K, tid);
        // Compute on current buffer while loads are in flight.
        compute_block(A_lds[buf], B_lds[buf], acc, wave_m_off, wave_n_off, lane_id);
        __syncthreads();
        buf = next;
    }
    // Final compute on last loaded buffer.
    compute_block(A_lds[buf], B_lds[buf], acc, wave_m_off, wave_n_off, lane_id);

    // Epilogue
    int lane_n = lane_id & 15;
    int lane_m_grp = lane_id >> 4;
    #pragma unroll
    for (int mi = 0; mi < WAVE_MFMA_M; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WAVE_MFMA_N; ++ni) {
            int n_out = block_n + wave_n_off + ni * MFMA_N + lane_n;
            float sb = (n_out < N) ? scale_b[n_out] : 0.f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int m_out = block_m + wave_m_off + mi * MFMA_M + lane_m_grp * 4 + i;
                if (m_out < M && n_out < N) {
                    float sa = scale_a[m_out];
                    float v = acc[mi][ni][i] * sa * sb;
                    c[m_out * N + n_out] = static_cast<__hip_bfloat16>(v);
                }
            }
        }
    }
}

extern "C" void fp8_gemm_v3_launch(
    const void* a, const void* b,
    const void* scale_a, const void* scale_b,
    void* c, int M, int N, int K, hipStream_t stream)
{
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(NUM_THREADS);
    fp8_gemm_v3_kernel<<<grid, block, 0, stream>>>(
        static_cast<const unsigned char*>(a),
        static_cast<const unsigned char*>(b),
        static_cast<const float*>(scale_a),
        static_cast<const float*>(scale_b),
        static_cast<__hip_bfloat16*>(c),
        M, N, K);
}

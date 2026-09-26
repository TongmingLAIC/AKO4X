// === Identity ===
// v4_bigger_mfma — same tile shape as v2 (128×128 block), but uses
// mfma_f32_32x32x16_fp8_fp8 instead of 16x16x32. Each MFMA now produces
// 32×32 = 1024 fp32 outputs (was 16×16 = 256), so per wave per outer iter
// we issue 16 MFMA dispatches instead of 32. Same total FLOPS, half the
// dispatch overhead.
//
// === DEAD-END: tied v2, did not beat ===
// v4 measured 264.9µs vs v2's 264.7µs — exact tie. Confirmed: MFMA dispatch
// overhead is NOT the bottleneck. v2/v3/v4 all hit the same ~265µs ceiling
// regardless of MFMA shape or LDS pipelining, which means we're bandwidth-bound
// from per-block LDS-load amplification (A loaded N/BN = 56× redundantly,
// B loaded M/BM = 32× redundantly). The fix is wider tiles (v5+), not
// MFMA-shape tuning.
//
// === Delta from v2_wider_tile ===
// 1) MFMA: f32_32x32x16_fp8_fp8 (was 16x16x32). Bigger output tile per MFMA.
// 2) MFMA_K = 16 (was 32). BLOCK_MFMA_K = BK/16 = 4 (was 2). 4 K-steps per outer iter.
// 3) Per wave: WAVE_MFMA_M = WAVE_MFMA_N = 2 (was 4). 2×2×4 = 16 MFMAs/outer (was 32).
// 4) Accumulator: floatx16 per MFMA (16 fp32 per lane), was floatx4 (4 fp32 per lane).
//    Per wave: 2×2 = 4 accumulators × 16 fp32 = 64 fp32 in registers (same as v2).
//
// === Lane mapping for mfma_f32_32x32x16_fp8_fp8 (wave64) ===
// A operand (32×16, M×K), per lane = 8 fp8 along K:
//   lane_m   = lane_id & 31       (0..31, row)
//   lane_k_grp = lane_id >> 5     (0..1, K-group of 8 fp8)
//   pack8 = A[lane_m, k_outer + lane_k_grp*8 .. +8]
// B operand (16×32, K×N) row-major as b[N,K], per lane = 8 fp8 along K:
//   lane_n     = lane_id & 31
//   lane_k_grp = lane_id >> 5
//   pack8 = b[lane_n, k_outer + lane_k_grp*8 .. +8]
// D output (32×32 fp32), per lane = 16 fp32 in 4 groups of 4 rows each:
//   lane_n     = lane_id & 31
//   lane_m_base= (lane_id >> 5) * 4   (0 or 4 within each group)
//   For g in 0..3, i in 0..3:
//     d_idx = g*4 + i      (flat index into floatx16)
//     D[g*8 + lane_m_base + i, lane_n] = d[d_idx]
//
// === Open directions (v5+) ===
// * __builtin_amdgcn_global_load_lds direct path (the real overlap win).
// * BK=128 + 2-stage software pipeline (more compute per outer iter).

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

#define WAVE_TILE_M (BM / WAVES_M)   // 64
#define WAVE_TILE_N (BN / WAVES_N)   // 64
#define MFMA_M 32
#define MFMA_N 32
#define MFMA_K 16
#define WAVE_MFMA_M (WAVE_TILE_M / MFMA_M)   // 2
#define WAVE_MFMA_N (WAVE_TILE_N / MFMA_N)   // 2
#define BLOCK_MFMA_K (BK / MFMA_K)           // 4

#define A_BYTES (BM * BK)
#define BYTES_PER_THREAD (A_BYTES / NUM_THREADS)
#define U4_PER_ROW (BK / 16)
#define U4_PER_THREAD (BYTES_PER_THREAD / 16)

typedef long long i64;
typedef float float16_t __attribute__((ext_vector_type(16)));

__device__ __forceinline__ i64 load_fp8x8(const unsigned char* src) {
    return *reinterpret_cast<const i64*>(src);
}

__global__ void __launch_bounds__(NUM_THREADS)
fp8_gemm_v4_kernel(
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

    __shared__ unsigned char A_lds[BM][BK];
    __shared__ unsigned char B_lds[BN][BK];

    float16_t acc[WAVE_MFMA_M][WAVE_MFMA_N];
    #pragma unroll
    for (int i = 0; i < WAVE_MFMA_M; ++i)
        #pragma unroll
        for (int j = 0; j < WAVE_MFMA_N; ++j)
            #pragma unroll
            for (int k = 0; k < 16; ++k)
                acc[i][j][k] = 0.f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Cooperative load A
        #pragma unroll
        for (int i = 0; i < U4_PER_THREAD; ++i) {
            int u4_idx = tid * U4_PER_THREAD + i;
            int row = u4_idx / U4_PER_ROW;
            int col_b = (u4_idx % U4_PER_ROW) * 16;
            *reinterpret_cast<uint4*>(&A_lds[row][col_b]) =
                *reinterpret_cast<const uint4*>(a + (block_m + row) * K + (k0 + col_b));
        }
        // Cooperative load B
        #pragma unroll
        for (int i = 0; i < U4_PER_THREAD; ++i) {
            int u4_idx = tid * U4_PER_THREAD + i;
            int row = u4_idx / U4_PER_ROW;
            int col_b = (u4_idx % U4_PER_ROW) * 16;
            *reinterpret_cast<uint4*>(&B_lds[row][col_b]) =
                *reinterpret_cast<const uint4*>(b + (block_n + row) * K + (k0 + col_b));
        }
        __syncthreads();

        int lane_outer = lane_id & 31;
        int lane_k_grp = lane_id >> 5;

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
                    acc[mi][ni] = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                        a_reg, b_reg, acc[mi][ni], 0, 0, 0);
                }
            }
        }
        __syncthreads();
    }

    // Epilogue — 32x32 D layout: 4 groups of 4 rows, lane_m_base = (lane>>5)*4.
    int lane_n = lane_id & 31;
    int lane_m_base = (lane_id >> 5) * 4;
    #pragma unroll
    for (int mi = 0; mi < WAVE_MFMA_M; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WAVE_MFMA_N; ++ni) {
            int n_out = block_n + wave_n_off + ni * MFMA_N + lane_n;
            float sb = (n_out < N) ? scale_b[n_out] : 0.f;
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int m_out = block_m + wave_m_off + mi * MFMA_M + g * 8 + lane_m_base + i;
                    if (m_out < M && n_out < N) {
                        float sa = scale_a[m_out];
                        float v = acc[mi][ni][g * 4 + i] * sa * sb;
                        c[m_out * N + n_out] = static_cast<__hip_bfloat16>(v);
                    }
                }
            }
        }
    }
}

extern "C" void fp8_gemm_v4_launch(
    const void* a, const void* b,
    const void* scale_a, const void* scale_b,
    void* c, int M, int N, int K, hipStream_t stream)
{
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(NUM_THREADS);
    fp8_gemm_v4_kernel<<<grid, block, 0, stream>>>(
        static_cast<const unsigned char*>(a),
        static_cast<const unsigned char*>(b),
        static_cast<const float*>(scale_a),
        static_cast<const float*>(scale_b),
        static_cast<__hip_bfloat16*>(c),
        M, N, K);
}

// === Identity ===
// v7_8wave — v6's padded 128×128 tile, but with 8 waves per block (WAVES_M=2,
// WAVES_N=4) instead of 4. Each wave covers a 64×32 sub-tile. NUM_THREADS=512.
// Goal: 24 waves/CU (3 blocks × 8 waves) vs v6's 12 (3 × 4), giving better
// MFMA latency hiding (need ~16+ waves to cover the 64-cycle latency at
// 4-cycle issue rate; 12 is insufficient).
//
// === Delta from v6_lds_swizzle ===
// 1) WAVES_N: 2 → 4. WAVE_TILE_N: 64 → 32. WAVE_MFMA_N: 4 → 2.
// 2) NUM_WAVES: 4 → 8. NUM_THREADS: 256 → 512.
// 3) BYTES_PER_THREAD: 32 → 16 (= 1 uint4). Store loop iteration count: 2 → 1.
// 4) LDS unchanged: 128×72 × 2 = 18 432 bytes → still 3 blocks/CU.
// 5) Waves/CU: 3 × 8 = 24 (was 12). Should fully hide 64-cycle MFMA latency.
//
// === Lessons ===
// LDS bank conflicts (8-way in v2) were the dominant bottleneck — 8-byte row
// pad (v6) eliminated them and gave 1.76× speedup.
//
// === Dead-ends (inherited) ===
// * v3: SW double-buffer with regular loads — compiler already overlaps.
// * v4: mfma_32x32x16 — dispatch not bottleneck.
// * v5: BM=256 — occupancy collapse.
//
// === Open directions ===
// * __builtin_amdgcn_global_load_lds — true async DRAM→LDS, no VGPR staging.
// * Multi-stage SW pipeline (3- or 4-stage).
// * Persistent kernel / stream-K for better L2 reuse.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define BM 128
#define BN 128
#define BK 64
#define LDS_PAD 8
#define WAVES_M 2
#define WAVES_N 4          // <-- 4 (was 2)
#define WAVE_SIZE 64
#define NUM_WAVES (WAVES_M * WAVES_N)   // 8
#define NUM_THREADS (NUM_WAVES * WAVE_SIZE)  // 512

#define WAVE_TILE_M (BM / WAVES_M)   // 64
#define WAVE_TILE_N (BN / WAVES_N)   // 32  <-- was 64
#define MFMA_M 16
#define MFMA_N 16
#define MFMA_K 32
#define WAVE_MFMA_M (WAVE_TILE_M / MFMA_M)   // 4
#define WAVE_MFMA_N (WAVE_TILE_N / MFMA_N)   // 2  <-- was 4
#define BLOCK_MFMA_K (BK / MFMA_K)           // 2

#define A_BYTES (BM * BK)             // 8192
#define BYTES_PER_THREAD (A_BYTES / NUM_THREADS)  // 16 = 1 uint4

typedef long long i64;
typedef float float4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ i64 load_fp8x8(const unsigned char* src) {
    return *reinterpret_cast<const i64*>(src);
}

__global__ void __launch_bounds__(NUM_THREADS)
fp8_gemm_v7_kernel(
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
    const int wave_m_idx = wave_id / WAVES_N;
    const int wave_n_idx = wave_id % WAVES_N;
    const int wave_m_off = wave_m_idx * WAVE_TILE_M;
    const int wave_n_off = wave_n_idx * WAVE_TILE_N;

    __shared__ unsigned char A_lds[BM][BK + LDS_PAD];
    __shared__ unsigned char B_lds[BN][BK + LDS_PAD];

    float4_t acc[WAVE_MFMA_M][WAVE_MFMA_N];
    #pragma unroll
    for (int i = 0; i < WAVE_MFMA_M; ++i)
        #pragma unroll
        for (int j = 0; j < WAVE_MFMA_N; ++j)
            acc[i][j] = (float4_t){0.f, 0.f, 0.f, 0.f};

    // 1 uint4 per thread per tile (512 threads × 16 bytes = 8192 bytes = A tile).
    constexpr int U4_PER_ROW = BK / 16;       // 4
    constexpr int U4_PER_THREAD = BYTES_PER_THREAD / 16;  // 1

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile
        #pragma unroll
        for (int i = 0; i < U4_PER_THREAD; ++i) {
            int u4_idx = tid * U4_PER_THREAD + i;
            int row = u4_idx / U4_PER_ROW;
            int col_u4 = u4_idx % U4_PER_ROW;
            int col_b = col_u4 * 16;
            const unsigned char* src = a + (block_m + row) * K + (k0 + col_b);
            *reinterpret_cast<uint4*>(&A_lds[row][col_b]) = *reinterpret_cast<const uint4*>(src);
        }
        // Load B tile
        #pragma unroll
        for (int i = 0; i < U4_PER_THREAD; ++i) {
            int u4_idx = tid * U4_PER_THREAD + i;
            int row = u4_idx / U4_PER_ROW;
            int col_u4 = u4_idx % U4_PER_ROW;
            int col_b = col_u4 * 16;
            const unsigned char* src = b + (block_n + row) * K + (k0 + col_b);
            *reinterpret_cast<uint4*>(&B_lds[row][col_b]) = *reinterpret_cast<const uint4*>(src);
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BLOCK_MFMA_K; ++kk) {
            int k_lds = kk * MFMA_K;
            int lane_outer = lane_id & 15;
            int lane_k_grp = lane_id >> 4;
            int k_off = k_lds + lane_k_grp * 8;

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
        __syncthreads();
    }

    // Epilogue: each wave writes its 64×32 sub-tile.
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

extern "C" void fp8_gemm_v7_launch(
    const void* a, const void* b,
    const void* scale_a, const void* scale_b,
    void* c, int M, int N, int K, hipStream_t stream)
{
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(NUM_THREADS);
    hipLaunchKernelGGL(( fp8_gemm_v7_kernel), dim3(grid), dim3(block), 0, stream,
        static_cast<const unsigned char*>(a),
        static_cast<const unsigned char*>(b),
        static_cast<const float*>(scale_a),
        static_cast<const float*>(scale_b),
        static_cast<__hip_bfloat16*>(c),
        M, N, K);
}

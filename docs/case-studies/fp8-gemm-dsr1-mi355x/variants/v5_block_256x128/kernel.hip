// === Identity ===
// v5_block_256x128 — bigger output block (256×128) with 8 waves in a 4×2
// grid. Attacks the bandwidth-bound ceiling that v2/v3/v4 all hit at 265µs.
//
// === DEAD-END: slower than v2 (305µs vs 265µs) ===
// Bandwidth-bound thesis FALSIFIED. Reducing B's redundant DRAM load by 2×
// did NOT translate into a speedup — v5 measured slower. Likely causes:
// (1) 24KB LDS + 512 threads → only 1 block/CU (was 2 at v2's 16KB).
// (2) Only 16×56 = 896 blocks vs v2's 32×56 = 1792 — less work for the
//     wavefront scheduler to hide latency across.
// Real bottleneck at the 265µs ceiling is probably LDS bank conflicts
// (no swizzle) and lack of async global→LDS loads. Both are deeper
// rewrites; deferred to v6+ (would need __builtin_amdgcn_global_load_lds
// + per-bank LDS layout).
//
// === Delta from v2_wider_tile ===
// 1) BM=256 (was 128). B is now loaded only M/BM = 16× redundantly (was 32×).
//    Total LDS-load amplification drops from ~940MB to ~704MB (25% less).
// 2) 8 waves (4×2) instead of 4 (2×2). Each wave still owns a 64×64 sub-tile
//    so per-wave MFMA work is identical to v2 (4×4×2 = 32 MFMAs/outer).
// 3) LDS: A 256×64=16KB + B 128×64=8KB = 24KB (was 16KB). Still ≤ 2 blocks/CU.
// 4) NUM_THREADS=512 (was 256). Bytes/thread for A: 16384/512=32=2×uint4 (same as v2).
//
// === Why this should help if the bandwidth thesis is right ===
// v2/v3/v4 ceiling = 265µs at 454 TFLOP/s. If pure-bandwidth model is correct,
// 25% less DRAM traffic → ~199µs / ~604 TFLOP/s expected. Anything close
// confirms the diagnosis; significant miss falsifies it.
//
// === Open directions ===
// * v6 = 256×128 + double buffer (now that we know bandwidth matters).
// * BK=128 — fewer outer iters, larger LDS footprint (would drop to 1 block/CU).

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define BM 256
#define BN 128
#define BK 64
#define WAVES_M 4
#define WAVES_N 2
#define WAVE_SIZE 64
#define NUM_WAVES (WAVES_M * WAVES_N)
#define NUM_THREADS (NUM_WAVES * WAVE_SIZE)

#define WAVE_TILE_M (BM / WAVES_M)   // 64
#define WAVE_TILE_N (BN / WAVES_N)   // 64
#define MFMA_M 16
#define MFMA_N 16
#define MFMA_K 32
#define WAVE_MFMA_M (WAVE_TILE_M / MFMA_M)   // 4
#define WAVE_MFMA_N (WAVE_TILE_N / MFMA_N)   // 4
#define BLOCK_MFMA_K (BK / MFMA_K)           // 2

#define A_BYTES (BM * BK)              // 16384
#define B_BYTES (BN * BK)              // 8192
#define A_BYTES_PER_THREAD (A_BYTES / NUM_THREADS)  // 32
#define B_BYTES_PER_THREAD (B_BYTES / NUM_THREADS)  // 16
#define U4_PER_ROW (BK / 16)           // 4
#define A_U4_PER_THREAD (A_BYTES_PER_THREAD / 16)   // 2
#define B_U4_PER_THREAD (B_BYTES_PER_THREAD / 16)   // 1

typedef long long i64;
typedef float float4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ i64 load_fp8x8(const unsigned char* src) {
    return *reinterpret_cast<const i64*>(src);
}

__global__ void __launch_bounds__(NUM_THREADS)
fp8_gemm_v5_kernel(
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

    float4_t acc[WAVE_MFMA_M][WAVE_MFMA_N];
    #pragma unroll
    for (int i = 0; i < WAVE_MFMA_M; ++i)
        #pragma unroll
        for (int j = 0; j < WAVE_MFMA_N; ++j)
            acc[i][j] = (float4_t){0.f, 0.f, 0.f, 0.f};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A: 2 uint4 per thread
        #pragma unroll
        for (int i = 0; i < A_U4_PER_THREAD; ++i) {
            int u4_idx = tid * A_U4_PER_THREAD + i;
            int row = u4_idx / U4_PER_ROW;
            int col_b = (u4_idx % U4_PER_ROW) * 16;
            *reinterpret_cast<uint4*>(&A_lds[row][col_b]) =
                *reinterpret_cast<const uint4*>(a + (block_m + row) * K + (k0 + col_b));
        }
        // Load B: 1 uint4 per thread
        {
            int u4_idx = tid;  // 512 threads * 1 uint4 = 512 uint4 = 8KB ✓
            int row = u4_idx / U4_PER_ROW;
            int col_b = (u4_idx % U4_PER_ROW) * 16;
            *reinterpret_cast<uint4*>(&B_lds[row][col_b]) =
                *reinterpret_cast<const uint4*>(b + (block_n + row) * K + (k0 + col_b));
        }
        __syncthreads();

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
        __syncthreads();
    }

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

extern "C" void fp8_gemm_v5_launch(
    const void* a, const void* b,
    const void* scale_a, const void* scale_b,
    void* c, int M, int N, int K, hipStream_t stream)
{
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(NUM_THREADS);
    fp8_gemm_v5_kernel<<<grid, block, 0, stream>>>(
        static_cast<const unsigned char*>(a),
        static_cast<const unsigned char*>(b),
        static_cast<const float*>(scale_a),
        static_cast<const float*>(scale_b),
        static_cast<__hip_bfloat16*>(c),
        M, N, K);
}

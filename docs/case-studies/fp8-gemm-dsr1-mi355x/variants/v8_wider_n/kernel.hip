// === Identity ===
// v8_wider_n — BM=128, BN=256 (was 128). Each block processes a 128×256
// output tile. 1792 blocks → 896 blocks; DRAM reads: 917 MB → 685 MB (−25%).
// Same 8-wave layout (WAVES_M=2, WAVES_N=4), same 8-byte LDS padding.
// LDS grows to 27 KB (was 18 KB) → 2 blocks/CU (was 3).
// Waves/CU: 2 × 8 = 16 (was 24). Still enough to hide 64-cycle MFMA latency
// (16 × 4-cycle issue = 64 cycles — exactly at the boundary).
//
// === Delta from v7_8wave ===
// 1) BN: 128 → 256. Grid N-dim: 56 → 28 tiles.
// 2) B_lds[BN][BK+LDS_PAD]: 256×72 = 18 432 B (was 128×72 = 9 216 B).
//    Total LDS: 9 216 + 18 432 = 27 648 B ≈ 27 KB (was 18 KB).
// 3) WAVE_TILE_N = BN/WAVES_N = 256/4 = 64 (restored from v7's 32 to v6's 64).
//    WAVE_MFMA_N = 64/16 = 4 (was 2 in v7). Each wave computes 4×4 MFMA grid.
// 4) B_BYTES = 16 384 B. B_U4_PER_THREAD = 2 (was 1).
//    B store loop: 2 uint4 per thread (same as A_U4_PER_THREAD in v6).
//
// === Lessons ===
// v6: 8-byte LDS padding eliminates 8-way bank conflicts (dominant bottleneck).
// v7: 8 waves/block closes 64-cycle MFMA latency gap (secondary bottleneck).
// v8 hypothesis: v7's 102 µs ≈ 917 MB / 9 TB/s — DRAM bandwidth limited.
// 25% DRAM reduction expected to give ~76 µs if hypothesis is correct.
//
// === Dead-ends (inherited) ===
// * v3: SW double-buffer with regular VGPR loads — no benefit.
// * v4: bigger MFMA shape — dispatch not bottleneck.
// * v5: BM=256 — occupancy collapse (1 block/CU).
//
// === DEAD-END: 109.8 µs — slower than v7's 102.0 µs ===
// Occupancy drop (24 → 16 waves/CU) hurt more than 25% DRAM reduction helped.
// With 16 waves × 4-cycle MFMA throughput = 64 cycles between reissues on the
// same wave — exactly equal to MFMA latency, so any additional overhead (LDS
// reads, address math) causes pipeline stalls. The 685 MB → 76 µs DRAM-BW
// ceiling was not reached because the MFMA pipeline starved first.
//
// Fix: restore 3 blocks/CU (tile size ≤ 18 KB), OR pair BN=256 with
// 16 waves/block (32 waves/CU) to provide headroom above the 64-cycle threshold.
//
// === Open directions ===
// * BN=256 + 16 waves/block (32 waves/CU) — test if latency headroom recovers.
// * __builtin_amdgcn_global_load_lds — async DRAM→LDS, no VGPR staging.
// * Persistent kernel for better L2 reuse across blocks.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define BM 128
#define BN 256          // <-- was 128
#define BK 64
#define LDS_PAD 8
#define WAVES_M 2
#define WAVES_N 4
#define WAVE_SIZE 64
#define NUM_WAVES (WAVES_M * WAVES_N)   // 8
#define NUM_THREADS (NUM_WAVES * WAVE_SIZE)  // 512

#define WAVE_TILE_M (BM / WAVES_M)   // 64
#define WAVE_TILE_N (BN / WAVES_N)   // 64  (restored; was 32 in v7)
#define MFMA_M 16
#define MFMA_N 16
#define MFMA_K 32
#define WAVE_MFMA_M (WAVE_TILE_M / MFMA_M)   // 4
#define WAVE_MFMA_N (WAVE_TILE_N / MFMA_N)   // 4  (restored; was 2 in v7)
#define BLOCK_MFMA_K (BK / MFMA_K)           // 2

#define A_BYTES (BM * BK)                           // 8192
#define A_BYTES_PER_THREAD (A_BYTES / NUM_THREADS)  // 16 = 1 uint4
#define B_BYTES (BN * BK)                           // 16384
#define B_BYTES_PER_THREAD (B_BYTES / NUM_THREADS)  // 32 = 2 uint4

typedef long long i64;
typedef float float4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ i64 load_fp8x8(const unsigned char* src) {
    return *reinterpret_cast<const i64*>(src);
}

__global__ void __launch_bounds__(NUM_THREADS)
fp8_gemm_v8_kernel(
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

    constexpr int U4_PER_ROW = BK / 16;                  // 4
    constexpr int A_U4_PER_THREAD = A_BYTES_PER_THREAD / 16;  // 1
    constexpr int B_U4_PER_THREAD = B_BYTES_PER_THREAD / 16;  // 2

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile (1 uint4 per thread)
        #pragma unroll
        for (int i = 0; i < A_U4_PER_THREAD; ++i) {
            int u4_idx = tid * A_U4_PER_THREAD + i;
            int row = u4_idx / U4_PER_ROW;
            int col_u4 = u4_idx % U4_PER_ROW;
            int col_b = col_u4 * 16;
            const unsigned char* src = a + (block_m + row) * K + (k0 + col_b);
            *reinterpret_cast<uint4*>(&A_lds[row][col_b]) = *reinterpret_cast<const uint4*>(src);
        }
        // Load B tile (2 uint4 per thread)
        #pragma unroll
        for (int i = 0; i < B_U4_PER_THREAD; ++i) {
            int u4_idx = tid * B_U4_PER_THREAD + i;
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

    // Epilogue: write 128×256 output tile.
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

extern "C" void fp8_gemm_v8_launch(
    const void* a, const void* b,
    const void* scale_a, const void* scale_b,
    void* c, int M, int N, int K, hipStream_t stream)
{
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(NUM_THREADS);
    hipLaunchKernelGGL(( fp8_gemm_v8_kernel), dim3(grid), dim3(block), 0, stream,
        static_cast<const unsigned char*>(a),
        static_cast<const unsigned char*>(b),
        static_cast<const float*>(scale_a),
        static_cast<const float*>(scale_b),
        static_cast<__hip_bfloat16*>(c),
        M, N, K);
}

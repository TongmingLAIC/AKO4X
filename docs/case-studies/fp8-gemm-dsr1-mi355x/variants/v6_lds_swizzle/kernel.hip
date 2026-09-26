// === Identity ===
// v6_lds_swizzle — same 128×128 tile as v2, but with an 8-byte pad per LDS
// row. Row stride changes from BK=64 bytes (16 banks) to BK+8=72 bytes (18
// banks). GCD(18, 32) = 2, so 16 consecutive rows produce 16 distinct base-
// bank offsets: {0,18,4,22,8,26,12,30,16,2,20,6,24,10,28,14} mod 32.
// Each wave's 16 MFMA-operand lanes (reading 8 bytes from 16 consecutive rows
// at the same column) therefore hit 16 unique bank-pairs → zero conflicts.
// v2 had an 8-way bank conflict for every MFMA operand load (even rows all
// mapped to bank 0, odd rows all to bank 16).
//
// === Delta from v2_wider_tile ===
// 1) #define LDS_PAD 8 added.
// 2) A_lds[BM][BK + LDS_PAD], B_lds[BN][BK + LDS_PAD] (physical stride 72B).
//    All A/B stores and loads go through the same 2D-array indexing — C
//    accounts for the padded stride automatically. Data region is columns 0..63;
//    padding bytes 64..71 are never written or read.
// 3) LDS: 128×72 × 2 = 18 432 bytes (≈ 18 KB, was 16 KB).
//    Occupancy: floor(64 KB / 18 KB) = 3 blocks/CU (was 4).
//
// === Lessons ===
// (from prior variants)
// uint4 vectorized loads always win when stride/alignment allow.
// MFMA accumulator stays in fp32 in regs across the full K loop.
//
// === Dead-ends (inherited) ===
// * v3: software double-buffer with regular global loads — compiler already
//   overlaps loads with MFMAs; explicit buffering adds nothing.
// * v4: mfma_f32_32x32x16 — halved MFMA dispatch, no speedup (dispatch not
//   bottleneck).
// * v5: BM=256 (wider tile to cut B's DRAM redundancy) — occupancy collapsed
//   to 1 block/CU; slower than v2.
//
// === Open directions ===
// * __builtin_amdgcn_global_load_lds — true async DRAM→LDS (no VGPR staging).
// * Multi-stage SW pipeline (3- or 4-stage) with manual s_waitcnt.
// * BK=128 — fewer outer K iters, larger LDS but conflict-free with same pad.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define BM 128
#define BN 128
#define BK 64
#define LDS_PAD 8          // <-- new: pad each LDS row by 8 bytes
#define WAVES_M 2
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

#define A_BYTES (BM * BK)             // 8192 (logical; physical row is BK+LDS_PAD)
#define BYTES_PER_THREAD (A_BYTES / NUM_THREADS)  // 32 = 2 uint4

typedef long long i64;
typedef float float4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ i64 load_fp8x8(const unsigned char* src) {
    return *reinterpret_cast<const i64*>(src);
}

__global__ void __launch_bounds__(NUM_THREADS)
fp8_gemm_v6_kernel(
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

    // Padded LDS: row stride = BK + LDS_PAD = 72 bytes (18 × 4B banks).
    // With 32 banks, 16 consecutive rows produce 16 distinct bank-pair offsets
    // → conflict-free 8-byte MFMA operand loads.
    __shared__ unsigned char A_lds[BM][BK + LDS_PAD];
    __shared__ unsigned char B_lds[BN][BK + LDS_PAD];

    float4_t acc[WAVE_MFMA_M][WAVE_MFMA_N];
    #pragma unroll
    for (int i = 0; i < WAVE_MFMA_M; ++i)
        #pragma unroll
        for (int j = 0; j < WAVE_MFMA_N; ++j)
            acc[i][j] = (float4_t){0.f, 0.f, 0.f, 0.f};

    // Store loop: covers the data region [0..BK) of each row.
    // Padding bytes [BK..BK+LDS_PAD) are never written (don't affect correctness).
    constexpr int U4_PER_ROW = BK / 16;          // 4 (logical data columns per row)
    constexpr int U4_PER_THREAD = BYTES_PER_THREAD / 16;  // 2

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile from global → LDS
        #pragma unroll
        for (int i = 0; i < U4_PER_THREAD; ++i) {
            int u4_idx = tid * U4_PER_THREAD + i;
            int row = u4_idx / U4_PER_ROW;
            int col_u4 = u4_idx % U4_PER_ROW;
            int col_b = col_u4 * 16;
            const unsigned char* src = a + (block_m + row) * K + (k0 + col_b);
            *reinterpret_cast<uint4*>(&A_lds[row][col_b]) = *reinterpret_cast<const uint4*>(src);
        }
        // Load B tile from global → LDS
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

        // MFMA loop: each 8-byte load from A_lds/B_lds is now conflict-free.
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

    // Epilogue: scale + convert acc to bf16 and store to C.
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

extern "C" void fp8_gemm_v6_launch(
    const void* a, const void* b,
    const void* scale_a, const void* scale_b,
    void* c, int M, int N, int K, hipStream_t stream)
{
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(NUM_THREADS);
    hipLaunchKernelGGL(( fp8_gemm_v6_kernel), dim3(grid), dim3(block), 0, stream,
        static_cast<const unsigned char*>(a),
        static_cast<const unsigned char*>(b),
        static_cast<const float*>(scale_a),
        static_cast<const float*>(scale_b),
        static_cast<__hip_bfloat16*>(c),
        M, N, K);
}

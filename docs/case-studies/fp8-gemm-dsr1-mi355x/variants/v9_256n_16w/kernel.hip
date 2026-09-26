// === Identity ===
// v9_256n_16w — BN=256, 16 waves/block (WAVES_M=2, WAVES_N=8), 1024 threads.
// Fixes v8's failure: BN=256 reduces DRAM 25% (917→685 MB, 1792→896 blocks),
// and 16 waves/block gives 2 blocks/CU × 16 = 32 waves/CU — 2× the 16-wave
// threshold needed to hide 64-cycle MFMA latency (32 × 4 = 128 cycles > 64).
// v8 had only 16 waves/CU (8 waves × 2 blocks), exactly at the threshold —
// any overhead stalled the pipeline and cost more than the DRAM savings.
//
// === Delta from v8_wider_n ===
// 1) WAVES_N: 4 → 8. WAVE_TILE_N: 64 → 32. WAVE_MFMA_N: 4 → 2.
// 2) NUM_WAVES: 8 → 16. NUM_THREADS: 512 → 1024.
// 3) Waves/CU: 2 × 16 = 32 (v8 had 2 × 8 = 16).
// 4) A store: 8 bytes/thread (i64) because 8192 B / 1024 threads = 8 B.
//    B store: 16 bytes/thread (uint4) because 16384 B / 1024 threads = 16 B.
//    (v8 had 1 uint4 per thread for A; v9 uses 1 i64 per thread for A.)
//
// === Lessons ===
// v6: LDS bank-conflict pad (8B/row) — dominant bottleneck eliminated.
// v7: 8 waves/block — MFMA latency hiding bottleneck eliminated.
// v8: BN=256 with 8 waves/block → 16 waves/CU → pipeline stall > DRAM win.
//
// === Dead-ends (inherited) ===
// * v3: SW double-buffer (VGPR loads) — compiler already overlaps.
// * v4: bigger MFMA shape — dispatch not bottleneck.
// * v5: BM=256 (8 waves, no pad) — occupancy collapse + no pad.
// * v8: BN=256 + 8 waves/block — 16 waves/CU insufficient, latency exposed.
//
// === DEAD-END: 113.5 µs — worse than v7's 101.6 µs ===
// Even with 32 waves/CU (vs v8's 16), BN=256 is ~12% slower than v7. This
// rules out MFMA latency exposure as the cause of v8's failure. The BN=256
// penalty is structural: 2 blocks/CU puts 128 concurrent blocks in flight
// (vs v7's 192), reducing wavefront ILP and pipeline efficiency. The 25% DRAM
// reduction is outweighed by the ~33% fewer concurrent blocks.
// Conclusion: v7's 128×128 tile at 3 blocks/CU is the sweet spot; BN=256
// with any wave count is consistently worse.
//
// === Open directions ===
// * global_load_lds — bypass VGPR staging for direct DRAM→LDS.
// * Persistent / stream-K kernel for L2 reuse across M-blocks.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define BM 128
#define BN 256
#define BK 64
#define LDS_PAD 8
#define WAVES_M 2
#define WAVES_N 8          // <-- 8 (was 4 in v8)
#define WAVE_SIZE 64
#define NUM_WAVES (WAVES_M * WAVES_N)   // 16
#define NUM_THREADS (NUM_WAVES * WAVE_SIZE)  // 1024

#define WAVE_TILE_M (BM / WAVES_M)   // 64
#define WAVE_TILE_N (BN / WAVES_N)   // 32  <-- was 64 in v8
#define MFMA_M 16
#define MFMA_N 16
#define MFMA_K 32
#define WAVE_MFMA_M (WAVE_TILE_M / MFMA_M)   // 4
#define WAVE_MFMA_N (WAVE_TILE_N / MFMA_N)   // 2  <-- was 4 in v8
#define BLOCK_MFMA_K (BK / MFMA_K)           // 2

#define A_BYTES (BM * BK)                            // 8192
#define A_I64_PER_ROW (BK / 8)                       // 8  (8-byte stores for A)
#define A_I64_PER_THREAD (A_BYTES / NUM_THREADS / 8) // 1

#define B_BYTES (BN * BK)                            // 16384
#define B_U4_PER_ROW (BK / 16)                       // 4  (16-byte stores for B)
#define B_U4_PER_THREAD (B_BYTES / NUM_THREADS / 16) // 1

typedef long long i64;
typedef float float4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ i64 load_fp8x8(const unsigned char* src) {
    return *reinterpret_cast<const i64*>(src);
}

__global__ void __launch_bounds__(NUM_THREADS)
fp8_gemm_v9_kernel(
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

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile: 8 bytes per thread (i64), 1024 threads × 8 B = 8192 B ✓
        {
            int a_row = tid / A_I64_PER_ROW;        // 0..127
            int a_col = (tid % A_I64_PER_ROW) * 8;  // 0,8,16,...,56
            const unsigned char* src = a + (block_m + a_row) * K + (k0 + a_col);
            *reinterpret_cast<i64*>(&A_lds[a_row][a_col]) = *reinterpret_cast<const i64*>(src);
        }
        // Load B tile: 16 bytes per thread (uint4), 1024 threads × 16 B = 16384 B ✓
        {
            int b_row = tid / B_U4_PER_ROW;         // 0..255
            int b_col = (tid % B_U4_PER_ROW) * 16;  // 0,16,32,48
            const unsigned char* src = b + (block_n + b_row) * K + (k0 + b_col);
            *reinterpret_cast<uint4*>(&B_lds[b_row][b_col]) = *reinterpret_cast<const uint4*>(src);
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

    // Epilogue: write 128×256 output tile (WAVE_MFMA_N=2, wave covers 64×32).
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

extern "C" void fp8_gemm_v9_launch(
    const void* a, const void* b,
    const void* scale_a, const void* scale_b,
    void* c, int M, int N, int K, hipStream_t stream)
{
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(NUM_THREADS);
    hipLaunchKernelGGL(( fp8_gemm_v9_kernel), dim3(grid), dim3(block), 0, stream,
        static_cast<const unsigned char*>(a),
        static_cast<const unsigned char*>(b),
        static_cast<const float*>(scale_a),
        static_cast<const float*>(scale_b),
        static_cast<__hip_bfloat16*>(c),
        M, N, K);
}

// === Identity ===
// v1_mfma_tile — first handwritten MFMA-based FP8 GEMM on gfx950.
// Uses mfma_f32_16x16x32_fp8_fp8 (CDNA4 native). Goal: CORRECTNESS first;
// performance iteration in v2+.
//
// === Delta from hip_stock ===
// 1) MFMA tensor cores instead of scalar fp32 accumulate (~1000× more FLOP/cycle).
// 2) Block tiles 64×64 outputs, 4 wave64 waves in a 2×2 wave grid.
// 3) Single-buffered LDS staging of A_tile [64, BK] and B_tile [64, BK], BK=64.
// 4) Per-token A scale + per-channel B scale applied in epilogue (after fp32 acc).
//
// === Layout convention (mfma_f32_16x16x32_fp8_fp8) ===
// Per wave64: A is 16×32 fp8, B is 32×16 fp8, D is 16×16 fp32.
// Lane mapping (this kernel's convention; iterate if MFMA output mismatches):
//   - For A and B operand loads:
//       lane_m_or_n = lane_id & 15        (0..15, the "outer" dim)
//       lane_k_grp  = lane_id >> 4        (0..3, K-group of 8 fp8 elements)
//     Each lane packs 8 fp8 = 1 uint64 from src[lane_m_or_n, k_start + lane_k_grp*8 .. +8].
//   - For D output:
//       lane_n      = lane_id & 15
//       lane_m_grp  = lane_id >> 4        (0..3, M-group of 4 fp32 outputs)
//     Each lane writes D[lane_m_grp*4 + i, lane_n] for i in 0..3.
//
// === Dead-ends to avoid (carried from RMSNorm/LayerNorm phase) ===
// * Persistent kernel — hurts at large grid on CDNA.
// * Bank conflicts in LDS — pad A/B tiles along K if MFMA load shows conflicts.
//
// === Open directions (v2+) ===
// * Double-buffered LDS (overlap global load with MFMA compute).
// * __builtin_amdgcn_global_load_lds — DRAM→LDS direct, skips VGPR staging.
// * Wider tiles (128×128 outputs, 4×4 wave grid).
// * mfma_f32_32x32x16_fp8_fp8 — bigger tile, fewer MFMA dispatches.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define BM 64
#define BN 64
#define BK 64
#define WAVES_M 2
#define WAVES_N 2
#define WAVE_SIZE 64
#define NUM_WAVES (WAVES_M * WAVES_N)
#define NUM_THREADS (NUM_WAVES * WAVE_SIZE)

#define WAVE_TILE_M 32   // each wave covers BM/WAVES_M = 32 rows of output
#define WAVE_TILE_N 32   // each wave covers BN/WAVES_N = 32 cols of output
#define MFMA_M 16
#define MFMA_N 16
#define MFMA_K 32
#define WAVE_MFMA_M (WAVE_TILE_M / MFMA_M)  // 2
#define WAVE_MFMA_N (WAVE_TILE_N / MFMA_N)  // 2
#define BLOCK_MFMA_K (BK / MFMA_K)          // 2

typedef long long i64;
typedef float float4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ i64 load_fp8x8(const unsigned char* src) {
    // 8 fp8 bytes → 64-bit packed value, little-endian.
    return *reinterpret_cast<const i64*>(src);
}

__global__ void __launch_bounds__(NUM_THREADS)
fp8_gemm_mfma_kernel(
    const unsigned char* __restrict__ a,  // [M, K] e4m3fn
    const unsigned char* __restrict__ b,  // [N, K] e4m3fn
    const float* __restrict__ scale_a,    // [M]
    const float* __restrict__ scale_b,    // [N]
    __hip_bfloat16* __restrict__ c,       // [M, N] bf16
    int M, int N, int K)
{
    const int block_m = blockIdx.x * BM;
    const int block_n = blockIdx.y * BN;
    const int tid = threadIdx.x;
    const int wave_id = tid / WAVE_SIZE;
    const int lane_id = tid & (WAVE_SIZE - 1);
    const int wave_m_idx = wave_id / WAVES_N;       // 0..WAVES_M-1
    const int wave_n_idx = wave_id % WAVES_N;       // 0..WAVES_N-1
    const int wave_m_off = wave_m_idx * WAVE_TILE_M;  // row offset within block
    const int wave_n_off = wave_n_idx * WAVE_TILE_N;  // col offset within block

    __shared__ unsigned char A_lds[BM][BK];
    __shared__ unsigned char B_lds[BN][BK];

    // Per-wave accumulators (one float4 per MFMA tile).
    float4_t acc[WAVE_MFMA_M][WAVE_MFMA_N];
    #pragma unroll
    for (int i = 0; i < WAVE_MFMA_M; ++i)
        #pragma unroll
        for (int j = 0; j < WAVE_MFMA_N; ++j)
            acc[i][j] = (float4_t){0.f, 0.f, 0.f, 0.f};

    // ---- Main K loop ----
    for (int k0 = 0; k0 < K; k0 += BK) {
        // Cooperative load A[block_m:block_m+BM, k0:k0+BK] -> A_lds.
        // BM*BK = 64*64 = 4096 bytes, 256 threads -> 16 bytes/thread = 1 uint4.
        // Thread layout: 64 threads cover BK (1 byte each won't work — use 4 threads/row × 64 rows).
        // Simpler: each thread loads 16 contiguous bytes from one (row, col_start) pair.
        {
            int tot = BM * BK;                      // 4096 bytes
            int per_thread = tot / NUM_THREADS;     // 16 bytes
            int byte_off = tid * per_thread;
            int row = byte_off / BK;                // 0..63
            int col = byte_off % BK;                // 0, 16, 32, 48
            int gm = block_m + row;
            const unsigned char* src = a + gm * K + (k0 + col);
            // 16-byte aligned load.
            *reinterpret_cast<uint4*>(&A_lds[row][col]) = *reinterpret_cast<const uint4*>(src);
        }
        {
            int tot = BN * BK;
            int per_thread = tot / NUM_THREADS;
            int byte_off = tid * per_thread;
            int row = byte_off / BK;
            int col = byte_off % BK;
            int gn = block_n + row;
            const unsigned char* src = b + gn * K + (k0 + col);
            *reinterpret_cast<uint4*>(&B_lds[row][col]) = *reinterpret_cast<const uint4*>(src);
        }
        __syncthreads();

        // ---- MFMA loop over BK in chunks of MFMA_K=32 ----
        #pragma unroll
        for (int kk = 0; kk < BLOCK_MFMA_K; ++kk) {
            int k_lds = kk * MFMA_K;
            // Each lane picks (lane_outer, lane_k_grp) → 8 fp8 = 1 i64.
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

    // ---- Epilogue: apply per-token A scale + per-channel B scale, cast to bf16 ----
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

extern "C" void fp8_gemm_v1_launch(
    const void* a, const void* b,
    const void* scale_a, const void* scale_b,
    void* c, int M, int N, int K, hipStream_t stream)
{
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(NUM_THREADS);
    fp8_gemm_mfma_kernel<<<grid, block, 0, stream>>>(
        static_cast<const unsigned char*>(a),
        static_cast<const unsigned char*>(b),
        static_cast<const float*>(scale_a),
        static_cast<const float*>(scale_b),
        static_cast<__hip_bfloat16*>(c),
        M, N, K);
}

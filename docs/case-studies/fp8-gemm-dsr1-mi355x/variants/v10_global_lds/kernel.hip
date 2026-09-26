// === Identity ===
// v10_global_lds — v7's tile config (BM=128, BN=128, BK=64, 8 waves/block),
// but tile loads use __builtin_amdgcn_global_load_lds instead of
// VGPR-staged uint4 loads. Hypothesis: direct DRAM→LDS bypasses VGPR round-trip.
//
// === Delta from v7_8wave ===
// 1) A/B tile loads: VGPR-staged uint4 → 4 × global_load_lds(size=4) each.
//    No other change. LDS layout, wave count, MFMA loop identical to v7.
//
// === Lessons ===
// v6: 8-byte LDS row pad eliminates 8-way bank conflicts (dominant bottleneck).
// v7: 8 waves/block (24/CU) fully hides 64-cycle MFMA latency (1.35× over v6).
// v8/v9: BN=256 tiles reduce DRAM 25% but cut blocks/CU (2 vs 3) → ILP loss.
//
// === Dead-ends (inherited) ===
// * v3: SW double-buffer with regular loads — compiler already overlaps.
// * v4: bigger MFMA shape — dispatch not bottleneck.
// * v5: BM=256 — 1 block/CU, occupancy collapsed.
// * v8: BN=256, 8 waves/block — 16 waves/CU, latency exposed.
// * v9: BN=256, 16 waves/block — structural ILP penalty (128 vs 192 blocks).
//
// === FAILED: correctness — global_load_lds incompatible with padded tile layout ===
// global_load_lds_dword generates: LDS_dst = M0 + lane_id * element_size.
// M0 is a SCALAR register (uniform per wavefront). The compiler extracts lane 0's
// computed LDS address into M0 (via v_readfirstlane), then ALL 64 lanes write to
// M0 + lane_id*4 — a contiguous 256-byte LDS region starting at lane 0's address.
//
// Our padded tile has ROW_STRIDE=72 bytes=18 dwords, 4 threads/row. Row boundaries
// are at 72, 144, 216 … bytes — NOT at multiples of 64. The lane_id*4 stride
// assumes consecutive 4-byte positions, which skips the LDS_PAD=8 bytes inserted
// after each 64-byte row of data. Result: all lanes 4+ write to wrong LDS slots.
//
// To use global_load_lds correctly we would need either:
// (a) No LDS padding — restores 8-way bank conflicts (v2 baseline).
// (b) A completely different tile organization (transposed, or padded to multiples
//     of 64 bytes per row so lane_id*4 steps align with row boundaries).
// Both would require redesigning the MFMA read phase as well.
//
// Root cause of the atol failures (736 → 205 → 1112 across three attempts):
// address-space annotation changes shifted which lanes overwrote which LDS slots,
// but the fundamental M0+lane_id*4 vs padded-row mismatch was always the bug.
//
// === Open directions ===
// * Stream-K / persistent kernel — reuse L2-cached A/B across block boundaries.
//   v7 is at ~9 TB/s (DRAM BW ceiling); only path to break AITER gap without
//   preshuffling is to read less DRAM via cross-block tile reuse.
// * Preshuffled B (like AITER bpreshuffle) — offline weight layout change cuts
//   DRAM reads per inference by reusing shuffled data across many requests.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define BM 128
#define BN 128
#define BK 64
#define LDS_PAD 8
#define WAVES_M 2
#define WAVES_N 4
#define WAVE_SIZE 64
#define NUM_WAVES (WAVES_M * WAVES_N)   // 8
#define NUM_THREADS (NUM_WAVES * WAVE_SIZE)  // 512

#define WAVE_TILE_M (BM / WAVES_M)   // 64
#define WAVE_TILE_N (BN / WAVES_N)   // 32
#define MFMA_M 16
#define MFMA_N 16
#define MFMA_K 32
#define WAVE_MFMA_M (WAVE_TILE_M / MFMA_M)   // 4
#define WAVE_MFMA_N (WAVE_TILE_N / MFMA_N)   // 2
#define BLOCK_MFMA_K (BK / MFMA_K)           // 2

#define A_BYTES (BM * BK)             // 8192
#define BYTES_PER_THREAD (A_BYTES / NUM_THREADS)  // 16 = 1 uint4

typedef long long i64;
typedef float float4_t __attribute__((ext_vector_type(4)));
// LDS pointer type — __builtin_amdgcn_global_load_lds requires the dst arg to be
// in addrspace(3). Plain unsigned char* is flat (addrspace(0)) and silently
// produces wrong results. C-style cast (NOT reinterpret_cast — that is rejected by
// the compiler) from __shared__ to lds_uint* preserves the AS3 annotation.
typedef unsigned int __attribute__((address_space(3))) lds_uint;

__device__ __forceinline__ i64 load_fp8x8(const unsigned char* src) {
    return *reinterpret_cast<const i64*>(src);
}

__global__ void __launch_bounds__(NUM_THREADS)
fp8_gemm_v10_kernel(
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

    // Flat 1D shared arrays — 2D indexing of __shared__ may lose addrspace(3)
    // during index-expression evaluation, corrupting global_load_lds dst addresses.
    // Manual index arithmetic on these 1D arrays preserves AS3 through C-style cast.
    __shared__ unsigned char A_storage[BM * (BK + LDS_PAD)];
    __shared__ unsigned char B_storage[BN * (BK + LDS_PAD)];
    constexpr int ROW_STRIDE = BK + LDS_PAD;  // 72

    float4_t acc[WAVE_MFMA_M][WAVE_MFMA_N];
    #pragma unroll
    for (int i = 0; i < WAVE_MFMA_M; ++i)
        #pragma unroll
        for (int j = 0; j < WAVE_MFMA_N; ++j)
            acc[i][j] = (float4_t){0.f, 0.f, 0.f, 0.f};

    // 1 uint4 (16 bytes) per thread per tile; loaded as 4 × 4-byte global_load_lds.
    constexpr int U4_PER_ROW = BK / 16;       // 4
    constexpr int U4_PER_THREAD = BYTES_PER_THREAD / 16;  // 1

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile: 512 threads × 16 bytes = 8192 bytes ✓
        // C-style cast from 1D __shared__ array to lds_uint* preserves addrspace(3).
        // Manual index arithmetic keeps all pointer ops in AS3.
        {
            int u4_idx = tid * U4_PER_THREAD;
            int a_row = u4_idx / U4_PER_ROW;
            int a_col = (u4_idx % U4_PER_ROW) * 16;
            const unsigned int* src = reinterpret_cast<const unsigned int*>(
                a + (block_m + a_row) * K + (k0 + a_col));
            lds_uint* dst = (lds_uint*)A_storage + (a_row * ROW_STRIDE + a_col) / 4;
            __builtin_amdgcn_global_load_lds(src,     dst,     4, 0, 0);
            __builtin_amdgcn_global_load_lds(src + 1, dst + 1, 4, 0, 0);
            __builtin_amdgcn_global_load_lds(src + 2, dst + 2, 4, 0, 0);
            __builtin_amdgcn_global_load_lds(src + 3, dst + 3, 4, 0, 0);
        }
        // Load B tile: same pattern.
        {
            int u4_idx = tid * U4_PER_THREAD;
            int b_row = u4_idx / U4_PER_ROW;
            int b_col = (u4_idx % U4_PER_ROW) * 16;
            const unsigned int* src = reinterpret_cast<const unsigned int*>(
                b + (block_n + b_row) * K + (k0 + b_col));
            lds_uint* dst = (lds_uint*)B_storage + (b_row * ROW_STRIDE + b_col) / 4;
            __builtin_amdgcn_global_load_lds(src,     dst,     4, 0, 0);
            __builtin_amdgcn_global_load_lds(src + 1, dst + 1, 4, 0, 0);
            __builtin_amdgcn_global_load_lds(src + 2, dst + 2, 4, 0, 0);
            __builtin_amdgcn_global_load_lds(src + 3, dst + 3, 4, 0, 0);
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
                    i64 a_reg = load_fp8x8(A_storage + a_row * ROW_STRIDE + k_off);
                    i64 b_reg = load_fp8x8(B_storage + b_row * ROW_STRIDE + k_off);
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

extern "C" void fp8_gemm_v10_launch(
    const void* a, const void* b,
    const void* scale_a, const void* scale_b,
    void* c, int M, int N, int K, hipStream_t stream)
{
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(NUM_THREADS);
    hipLaunchKernelGGL(( fp8_gemm_v10_kernel), dim3(grid), dim3(block), 0, stream,
        static_cast<const unsigned char*>(a),
        static_cast<const unsigned char*>(b),
        static_cast<const float*>(scale_a),
        static_cast<const float*>(scale_b),
        static_cast<__hip_bfloat16*>(c),
        M, N, K);
}

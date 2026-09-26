// === Identity ===
// hip_stock — naive FP8 GEMM, one output per thread, fp32 accum, scalar loads.
// Correctness anchor for variants. Expected to be very slow; the only goal is
// "matches torch_naive on a 32x32 probe so we trust the variant comparison."
//
// Shape:
//   a       [M, K] fp8_e4m3fnuz (row-major, K-contig)
//   b       [N, K] fp8_e4m3fnuz (row-major, K-contig — weight layout)
//   scale_a [M] fp32
//   scale_b [N] fp32
//   c       [M, N] bf16
//
// Grid: (ceil(M/BM), ceil(N/BN)), block: (BM*BN) — one thread per output.
// No LDS, no MFMA, no vectorization.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define BM 16
#define BN 16

// e4m3fn (OCP) manual decode — ROCm 7.0's hip_fp8.h `operator float()` is host-only,
// so we decode the raw byte directly. fn format: 1 sign + 4 exp (bias 7) + 3 mant,
// max ±448; NaN = s.1111.111 (0x7F / 0xFF); ±0 distinct; no infinity.
__device__ __forceinline__ float fp8_e4m3fn_to_float(unsigned char raw) {
    unsigned int sign = raw & 0x80u;
    int exp = (raw >> 3) & 0xF;
    unsigned int mant = raw & 0x7u;
    if (exp == 0xF && mant == 0x7) return __builtin_nanf("");
    float val;
    if (exp == 0) {
        val = static_cast<float>(mant) * (1.0f / 512.0f);   // 2^-6 * mant/8 = mant * 2^-9
    } else {
        val = (1.0f + static_cast<float>(mant) * 0.125f) * exp2f(static_cast<float>(exp - 7));
    }
    return sign ? -val : val;
}

__global__ void fp8_gemm_stock_kernel(
    const unsigned char* __restrict__ a,
    const unsigned char* __restrict__ b,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    __hip_bfloat16* __restrict__ c,
    int M, int N, int K)
{
    int m = blockIdx.x * BM + threadIdx.x;
    int n = blockIdx.y * BN + threadIdx.y;
    if (m >= M || n >= N) return;

    float acc = 0.f;
    const unsigned char* a_row = a + m * K;
    const unsigned char* b_row = b + n * K;
    for (int k = 0; k < K; ++k) {
        float av = fp8_e4m3fn_to_float(a_row[k]);
        float bv = fp8_e4m3fn_to_float(b_row[k]);
        acc += av * bv;
    }
    acc *= scale_a[m] * scale_b[n];
    c[m * N + n] = static_cast<__hip_bfloat16>(acc);
}

extern "C" void fp8_gemm_launch(
    const void* a, const void* b,
    const void* scale_a, const void* scale_b,
    void* c, int M, int N, int K, hipStream_t stream)
{
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(BM, BN);
    fp8_gemm_stock_kernel<<<grid, block, 0, stream>>>(
        static_cast<const unsigned char*>(a),
        static_cast<const unsigned char*>(b),
        static_cast<const float*>(scale_a),
        static_cast<const float*>(scale_b),
        static_cast<__hip_bfloat16*>(c),
        M, N, K);
}

"""Reference FP8 GEMM with per-token (a) + per-channel (b) scales.

Shape contract (DSR1 FFN gate_proj-ish):
    a       : [M, K] float8_e4m3fnuz   (row-major, K-contiguous)
    b       : [N, K] float8_e4m3fnuz   (row-major, K-contiguous — "weight" layout)
    scale_a : [M]    float32           (per-token, one scale per row of a)
    scale_b : [N]    float32           (per-channel, one scale per row of b)
    out     : [M, N] bfloat16

Math:
    out[m, n] = sum_k( float(a[m,k]) * float(b[n,k]) ) * scale_a[m] * scale_b[n]

We reference in fp32 to keep the correctness anchor tight; the FP8 product is
small enough that fp32 accumulation matches what a properly written MFMA
kernel would produce, modulo BF16 quantization on the final cast.
"""
import torch


def fp8_gemm_naive(a, b, scale_a, scale_b):
    a_f = a.to(torch.float32)
    b_f = b.to(torch.float32)
    # einsum (NOT @ b.t()) — hipBLASLt on torch 2.12+rocm7.1 hits
    # HIPBLAS_STATUS_INVALID_VALUE on small fp32 shapes like (64, 128, 2048).
    # einsum routes through a different kernel path that handles all shapes.
    acc = torch.einsum("mk,nk->mn", a_f, b_f)
    acc = acc * scale_a.view(-1, 1) * scale_b.view(1, -1)
    return acc.to(torch.bfloat16)

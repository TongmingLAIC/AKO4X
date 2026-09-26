#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

extern "C" void fp8_gemm_launch(
    const void* a, const void* b,
    const void* scale_a, const void* scale_b,
    void* c, int M, int N, int K, hipStream_t stream);

void launch(torch::Tensor a, torch::Tensor b,
            torch::Tensor scale_a, torch::Tensor scale_b,
            torch::Tensor c)
{
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() &&
                scale_a.is_contiguous() && scale_b.is_contiguous() &&
                c.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "a/b/c must be 2-D");
    TORCH_CHECK(a.size(1) == b.size(1), "a.K must equal b.K (b is [N,K])");
    TORCH_CHECK(c.size(0) == a.size(0) && c.size(1) == b.size(0), "c shape mismatch");
    TORCH_CHECK(scale_a.numel() == a.size(0), "scale_a must be [M]");
    TORCH_CHECK(scale_b.numel() == b.size(0), "scale_b must be [N]");
    fp8_gemm_launch(
        a.data_ptr(), b.data_ptr(),
        scale_a.data_ptr(), scale_b.data_ptr(),
        c.data_ptr(),
        static_cast<int>(a.size(0)),
        static_cast<int>(b.size(0)),
        static_cast<int>(a.size(1)),
        c10::hip::getCurrentHIPStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch, "hip_stock FP8 GEMM (per-token + per-channel scales)");
}

#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

extern "C" void fp8_gemm_v1_launch(
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
    fp8_gemm_v1_launch(
        a.data_ptr(), b.data_ptr(),
        scale_a.data_ptr(), scale_b.data_ptr(),
        c.data_ptr(),
        static_cast<int>(a.size(0)),
        static_cast<int>(b.size(0)),
        static_cast<int>(a.size(1)),
        c10::hip::getCurrentHIPStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch, "v1_mfma_tile FP8 GEMM");
}

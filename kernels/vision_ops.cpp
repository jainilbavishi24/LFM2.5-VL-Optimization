// PyTorch bindings for the vision-encoder kernels in vision_ops.cu (compiled by the host compiler).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

void launch_add_layernorm(void* h, const void* y, const void* w, const void* b, void* x, int rows, int D, float eps,
                          cudaStream_t st);

#define CHECK_BF16(t) TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(), #t " must be contiguous CUDA bf16")

// h += y (optional, in place); x = LayerNorm(h) * w + b
void add_layernorm(torch::Tensor h, c10::optional<torch::Tensor> y, torch::Tensor w, torch::Tensor b, torch::Tensor x,
                   double eps) {
    CHECK_BF16(h); CHECK_BF16(w); CHECK_BF16(b); CHECK_BF16(x);
    if (y.has_value()) { CHECK_BF16((*y)); TORCH_CHECK(y->numel() == h.numel(), "y shape"); }
    const int D = h.size(-1), rows = h.numel() / D;
    TORCH_CHECK(D % 128 == 0 && D <= 2048 && w.numel() == D && b.numel() == D && x.numel() == h.numel(),
                "D must be a multiple of 128, <= 2048");
    launch_add_layernorm(h.data_ptr(), y.has_value() ? y->data_ptr() : nullptr, w.data_ptr(), b.data_ptr(),
                         x.data_ptr(), rows, D, (float)eps, at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_layernorm", &add_layernorm, "h += y (optional); x = LayerNorm(h) * w + b (bf16, fp32 statistics)");
}

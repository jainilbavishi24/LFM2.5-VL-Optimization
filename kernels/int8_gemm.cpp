// PyTorch bindings for the INT8 prefill / vision GEMMs in int8_gemm.cu (compiled by the host compiler).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

void launch_gemm_i8(const void *A, const float *sa, const void *B, const float *sb, const void *bias, void *Y, int M,
                    int N, int K, bool gelu, bool grouped, cudaStream_t st);
void launch_gemm_i8_q(const void *A, const float *saT, const void *B, const float *sb, const void *bias, void *Yq,
                      float *YsT, int M, int N, int K, bool gelu, cudaStream_t st);
void launch_gemm_w8(const void *A, const void *B, const float *sb, const void *bias, void *Y, int M, int N, int K,
                    bool gelu, cudaStream_t st);
void launch_quant_rows(const void *x, void *q, float *s, int rows, int K, cudaStream_t st);
void launch_quant_groups(const void *x, void *q, float *sT, int rows, int K, cudaStream_t st);
void launch_add_layernorm_q(void *h, const void *y, const void *w, const void *b, void *q, float *s, int rows, int D,
                            float eps, bool grouped, cudaStream_t st);
void launch_add_rmsnorm_q(void *h, const void *y, const void *w, void *q, float *s, int rows, int H, float eps,
                          bool grouped, cudaStream_t st);
void launch_silu_mul_q(const void *g, void *q, float *s, int rows, int I, bool grouped, cudaStream_t st);

#define CHECK_T(t, dt) TORCH_CHECK(t.is_cuda() && t.scalar_type() == dt && t.is_contiguous(), #t " must be contiguous CUDA " #dt)

static void check_common(const torch::Tensor &w, const torch::Tensor &sw, const c10::optional<torch::Tensor> &bias,
                         const torch::Tensor &y, int M, int K) {
    CHECK_T(w, at::kChar);
    CHECK_T(sw, at::kFloat);
    CHECK_T(y, at::kBFloat16);
    const int N = w.size(0);
    TORCH_CHECK(w.dim() == 2 && w.size(1) == K, "w must be [N, K]");
    TORCH_CHECK(K % 16 == 0 && N % 8 == 0, "need K % 16 == 0 and N % 8 == 0");
    TORCH_CHECK(sw.numel() == N && y.numel() == (int64_t)M * N, "sw / y shape");
    if (bias.has_value()) {
        CHECK_T((*bias), at::kBFloat16);
        TORCH_CHECK(bias->numel() == N, "bias shape");
    }
}

// W8A8: y[M, N] = (x_q[M, K] . w[N, K]^T) * sx[m] * sw[n] (+ bias) (GELU), bf16 out. sx [M]: one scale per row;
// sx [K/64, M]: one scale per row and group of 64 input channels (from quant_rows with a 2-d scale tensor)
void gemm_i8(torch::Tensor xq, torch::Tensor sx, torch::Tensor w, torch::Tensor sw, c10::optional<torch::Tensor> bias,
             torch::Tensor y, bool gelu) {
    CHECK_T(xq, at::kChar);
    CHECK_T(sx, at::kFloat);
    const int K = xq.size(-1), M = xq.numel() / K;
    const bool grouped = sx.dim() == 2;
    TORCH_CHECK(grouped ? (K % 64 == 0 && sx.size(0) == K / 64 && sx.size(1) == M) : sx.numel() == M, "sx shape");
    check_common(w, sw, bias, y, M, K);
    launch_gemm_i8(xq.data_ptr(), sx.data_ptr<float>(), w.data_ptr(), sw.data_ptr<float>(),
                   bias.has_value() ? bias->data_ptr() : nullptr, y.data_ptr(), M, w.size(0), K, gelu, grouped,
                   at::cuda::getCurrentCUDAStream());
}

// grouped W8A8 with an INT8 output for the next W8A8 GEMM: (yq [M, N], ysT [N/64, M]) = quant(gemm_i8(...) as bf16)
void gemm_i8_q(torch::Tensor xq, torch::Tensor sxT, torch::Tensor w, torch::Tensor sw, c10::optional<torch::Tensor> bias,
               torch::Tensor yq, torch::Tensor ysT, bool gelu) {
    CHECK_T(xq, at::kChar);
    CHECK_T(sxT, at::kFloat);
    CHECK_T(w, at::kChar);
    CHECK_T(sw, at::kFloat);
    CHECK_T(yq, at::kChar);
    CHECK_T(ysT, at::kFloat);
    const int K = xq.size(-1), M = xq.numel() / K, N = w.size(0);
    TORCH_CHECK(sxT.dim() == 2 && K % 64 == 0 && sxT.size(0) == K / 64 && sxT.size(1) == M, "sxT shape");
    TORCH_CHECK(w.dim() == 2 && w.size(1) == K && N % 128 == 0 && sw.numel() == N, "w [N, K] with N % 128 == 0");
    TORCH_CHECK(yq.numel() == (int64_t)M * N && ysT.dim() == 2 && ysT.size(0) == N / 64 && ysT.size(1) == M, "yq / ysT shape");
    if (bias.has_value()) { CHECK_T((*bias), at::kBFloat16); TORCH_CHECK(bias->numel() == N, "bias shape"); }
    launch_gemm_i8_q(xq.data_ptr(), sxT.data_ptr<float>(), w.data_ptr(), sw.data_ptr<float>(),
                     bias.has_value() ? bias->data_ptr() : nullptr, yq.data_ptr(), ysT.data_ptr<float>(), M, N, K, gelu,
                     at::cuda::getCurrentCUDAStream());
}

// W8A16: y[M, N] = (x[M, K] . w[N, K]^T) * sw[n] (+ bias) (GELU), bf16 in / out
void gemm_w8(torch::Tensor x, torch::Tensor w, torch::Tensor sw, c10::optional<torch::Tensor> bias, torch::Tensor y,
             bool gelu) {
    CHECK_T(x, at::kBFloat16);
    const int K = x.size(-1), M = x.numel() / K;
    check_common(w, sw, bias, y, M, K);
    launch_gemm_w8(x.data_ptr(), w.data_ptr(), sw.data_ptr<float>(), bias.has_value() ? bias->data_ptr() : nullptr,
                   y.data_ptr(), M, w.size(0), K, gelu, at::cuda::getCurrentCUDAStream());
}

// per-row (token) symmetric quantization: s = max(|x|) / 127, q = round(x / s); with s of shape [K/64, rows]: one
// scale per row and group of 64 channels (transposed, as gemm_i8 reads it)
void quant_rows(torch::Tensor x, torch::Tensor q, torch::Tensor s) {
    CHECK_T(x, at::kBFloat16);
    CHECK_T(q, at::kChar);
    CHECK_T(s, at::kFloat);
    const int K = x.size(-1), rows = x.numel() / K;
    if (s.dim() == 2) {
        TORCH_CHECK(K % 128 == 0 && s.size(0) == K / 64 && s.size(1) == rows && q.numel() == x.numel(), "grouped shapes");
        launch_quant_groups(x.data_ptr(), q.data_ptr(), s.data_ptr<float>(), rows, K, at::cuda::getCurrentCUDAStream());
        return;
    }
    TORCH_CHECK(K % 128 == 0 && K <= 2048, "K must be a multiple of 128, <= 2048");
    TORCH_CHECK(q.numel() == x.numel() && s.numel() == rows, "q / s shape");
    launch_quant_rows(x.data_ptr(), q.data_ptr(), s.data_ptr<float>(), rows, K, at::cuda::getCurrentCUDAStream());
}

static bool check_qs(const torch::Tensor &q, const torch::Tensor &s, int rows, int K) {
    CHECK_T(q, at::kChar);
    CHECK_T(s, at::kFloat);
    const bool grouped = s.dim() == 2;
    TORCH_CHECK(q.numel() == (int64_t)rows * K, "q shape");
    TORCH_CHECK(grouped ? (K % 64 == 0 && s.size(0) == K / 64 && s.size(1) == rows) : s.numel() == rows, "s shape");
    return grouped;
}

// vision: h += y (optional, in place); (q, s) = quant(LayerNorm(h) * w + b)
void add_layernorm_q(torch::Tensor h, c10::optional<torch::Tensor> y, torch::Tensor w, torch::Tensor b, torch::Tensor q,
                     torch::Tensor s, double eps) {
    CHECK_T(h, at::kBFloat16);
    CHECK_T(w, at::kBFloat16);
    CHECK_T(b, at::kBFloat16);
    if (y.has_value()) { CHECK_T((*y), at::kBFloat16); TORCH_CHECK(y->numel() == h.numel(), "y shape"); }
    const int D = h.size(-1), rows = h.numel() / D;
    TORCH_CHECK(D % 128 == 0 && D <= 2048 && w.numel() == D && b.numel() == D, "D must be a multiple of 128, <= 2048");
    const bool grouped = check_qs(q, s, rows, D);
    launch_add_layernorm_q(h.data_ptr(), y.has_value() ? y->data_ptr() : nullptr, w.data_ptr(), b.data_ptr(), q.data_ptr(),
                           s.data_ptr<float>(), rows, D, (float)eps, grouped, at::cuda::getCurrentCUDAStream());
}

// LM prefill: h += y (optional, in place); (q, s) = quant(w * rmsnorm(h))
void add_rmsnorm_q(torch::Tensor h, c10::optional<torch::Tensor> y, torch::Tensor w, torch::Tensor q, torch::Tensor s,
                   double eps) {
    CHECK_T(h, at::kBFloat16);
    CHECK_T(w, at::kBFloat16);
    if (y.has_value()) { CHECK_T((*y), at::kBFloat16); TORCH_CHECK(y->numel() == h.numel(), "y shape"); }
    const int H = h.size(-1), rows = h.numel() / H;
    TORCH_CHECK(w.numel() == H && H % 64 == 0, "w shape");
    const bool grouped = check_qs(q, s, rows, H);
    launch_add_rmsnorm_q(h.data_ptr(), y.has_value() ? y->data_ptr() : nullptr, w.data_ptr(), q.data_ptr(),
                         s.data_ptr<float>(), rows, H, (float)eps, grouped, at::cuda::getCurrentCUDAStream());
}

// LM prefill: g [rows, 2I] = w1 | w3 outputs -> (q, s) = quant(silu(g1) * g3)
void silu_mul_q(torch::Tensor g, torch::Tensor q, torch::Tensor s) {
    CHECK_T(g, at::kBFloat16);
    const int I = g.size(-1) / 2, rows = g.numel() / (2 * I);
    TORCH_CHECK(I % 64 == 0 && I <= 20000, "I shape");
    const bool grouped = check_qs(q, s, rows, I);
    launch_silu_mul_q(g.data_ptr(), q.data_ptr(), s.data_ptr<float>(), rows, I, grouped, at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_layernorm_q", &add_layernorm_q, "vision: h += y; INT8 (per row / group of 64) LayerNorm output");
    m.def("add_rmsnorm_q", &add_rmsnorm_q, "LM prefill: h += y; INT8 (per row / group of 64) RMSNorm output");
    m.def("silu_mul_q", &silu_mul_q, "LM prefill: INT8 (per row / group of 64) SwiGLU output");
    m.def("gemm_i8", &gemm_i8, "W8A8 GEMM, int8 tensor cores, scales (+bias, GELU) in the epilogue, bf16 out");
    m.def("gemm_i8_q", &gemm_i8_q, "grouped W8A8 GEMM writing the next W8A8 GEMM's INT8 input (+ group scales)");
    m.def("gemm_w8", &gemm_w8, "W8A16 GEMM, int8 weights converted in registers, bf16 tensor cores, bf16 out");
    m.def("quant_rows", &quant_rows, "bf16 -> int8 with one absmax scale per row");
}

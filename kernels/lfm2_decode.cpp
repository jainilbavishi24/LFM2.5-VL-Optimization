// PyTorch bindings for the fused decode kernels in lfm2_decode.cu (compiled by the host compiler).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

void launch_add_rmsnorm(void* h, const void* y, const void* w, void* x, int rows, int H, float eps, cudaStream_t st);
void launch_shortconv_step(const void* bcx, void* state, const void* w, const void* bias, void* y, int B, int H, cudaStream_t st);
void launch_qk_norm_rope_kv(const void* qkv, const void* qw, const void* kw, const float* inv_freq, float scaling,
                            const int64_t* pos, void* q_out, void* kc, void* vc, int B, int NQ, int NKV, int max_len,
                            float eps, cudaStream_t st);
void launch_decode_attention(const void* q, const void* kc, const void* vc, const int64_t* pos, void* out,
                             int B, int NQ, int NKV, int max_len, float scale, cudaStream_t st);
void launch_silu_mul(const void* g, void* y, int B, int I, cudaStream_t st);
void launch_gemv(const void* W, bool int8, const float* scale, const void* x, void* y, int N, int K, cudaStream_t st);
void launch_decode_attention_split(const void* q, const void* kc, const void* vc, const int64_t* pos, void* out, float* ws,
                                   int B, int NQ, int NKV, int max_len, int S, float scale, cudaStream_t st);

#define CHECK_BF16(t) TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(), #t " must be contiguous CUDA bf16")
static inline cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }

void add_rmsnorm(torch::Tensor h, c10::optional<torch::Tensor> y, torch::Tensor w, torch::Tensor x, double eps) {
    CHECK_BF16(h); CHECK_BF16(w); CHECK_BF16(x);
    if (y.has_value()) CHECK_BF16((*y));
    const int H = h.size(-1), rows = h.numel() / H;
    launch_add_rmsnorm(h.data_ptr(), y.has_value() ? y->data_ptr() : nullptr, w.data_ptr(), x.data_ptr(), rows, H, (float)eps, stream());
}

void shortconv_step(torch::Tensor bcx, torch::Tensor state, torch::Tensor w, c10::optional<torch::Tensor> bias, torch::Tensor y) {
    CHECK_BF16(bcx); CHECK_BF16(state); CHECK_BF16(w); CHECK_BF16(y);
    TORCH_CHECK(state.size(-1) == 3, "conv state length must be 3");
    launch_shortconv_step(bcx.data_ptr(), state.data_ptr(), w.data_ptr(), bias.has_value() ? bias->data_ptr() : nullptr,
                          y.data_ptr(), bcx.size(0), bcx.size(-1) / 3, stream());
}

void qk_norm_rope_kv(torch::Tensor qkv, torch::Tensor qw, torch::Tensor kw, torch::Tensor inv_freq, double scaling,
                     torch::Tensor pos, torch::Tensor q_out, torch::Tensor kc, torch::Tensor vc, double eps) {
    CHECK_BF16(qkv); CHECK_BF16(qw); CHECK_BF16(kw); CHECK_BF16(q_out); CHECK_BF16(kc); CHECK_BF16(vc);
    TORCH_CHECK(inv_freq.scalar_type() == at::kFloat && pos.scalar_type() == at::kLong, "inv_freq fp32, pos int64");
    TORCH_CHECK(kc.size(3) == 64, "head_dim 64 only");
    launch_qk_norm_rope_kv(qkv.data_ptr(), qw.data_ptr(), kw.data_ptr(), inv_freq.data_ptr<float>(), (float)scaling,
                           pos.data_ptr<int64_t>(), q_out.data_ptr(), kc.data_ptr(), vc.data_ptr(), qkv.size(0),
                           q_out.size(1), kc.size(1), kc.size(2), (float)eps, stream());
}

void decode_attention(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor pos, torch::Tensor out, double scale) {
    CHECK_BF16(q); CHECK_BF16(kc); CHECK_BF16(vc); CHECK_BF16(out);
    TORCH_CHECK(kc.size(3) == 64, "head_dim 64 only");
    launch_decode_attention(q.data_ptr(), kc.data_ptr(), vc.data_ptr(), pos.data_ptr<int64_t>(), out.data_ptr(),
                            q.size(0), q.size(1), kc.size(1), kc.size(2), (float)scale, stream());
}

void decode_attention_split(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor pos, torch::Tensor out,
                            torch::Tensor ws, double scale) {
    CHECK_BF16(q); CHECK_BF16(kc); CHECK_BF16(vc); CHECK_BF16(out);
    TORCH_CHECK(kc.size(3) == 64 && q.size(1) == 4 * kc.size(1), "head_dim 64 and 4 query heads per KV head only");
    TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.is_contiguous() && ws.dim() == 4 && ws.size(3) == 66, "ws: fp32 [B, NQ, S, 66]");
    launch_decode_attention_split(q.data_ptr(), kc.data_ptr(), vc.data_ptr(), pos.data_ptr<int64_t>(), out.data_ptr(),
                                  ws.data_ptr<float>(), q.size(0), q.size(1), kc.size(1), kc.size(2), ws.size(2), (float)scale, stream());
}

// y [1, N] = x [1, K] @ W^T, W [N, K] BF16 or INT8 (then scale [N] fp32 is required)
void gemv(torch::Tensor W, c10::optional<torch::Tensor> scale, torch::Tensor x, torch::Tensor y) {
    CHECK_BF16(x); CHECK_BF16(y);
    TORCH_CHECK(W.is_cuda() && W.is_contiguous() && W.dim() == 2, "W must be a contiguous CUDA matrix");
    const bool int8 = W.scalar_type() == at::kChar;
    TORCH_CHECK(int8 || W.scalar_type() == at::kBFloat16, "W must be bf16 or int8");
    TORCH_CHECK(!int8 || (scale.has_value() && scale->scalar_type() == at::kFloat), "int8 W needs fp32 scale");
    const int N = W.size(0), K = W.size(1);
    TORCH_CHECK(x.numel() == K && y.numel() == N, "batch 1 only");
    TORCH_CHECK(K % (32 * (int8 ? 16 : 8)) == 0, "K must be a multiple of 256 (bf16) / 512 (int8)");
    launch_gemv(W.data_ptr(), int8, scale.has_value() ? scale->data_ptr<float>() : nullptr, x.data_ptr(), y.data_ptr(), N, K, stream());
}

void silu_mul(torch::Tensor g, torch::Tensor y) {
    CHECK_BF16(g); CHECK_BF16(y);
    launch_silu_mul(g.data_ptr(), y.data_ptr(), g.size(0), g.size(-1) / 2, stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_rmsnorm", &add_rmsnorm, "h += y (optional); x = w * rmsnorm(h)");
    m.def("shortconv_step", &shortconv_step, "LFM2 gated short-conv decode step");
    m.def("qk_norm_rope_kv", &qk_norm_rope_kv, "q/k head RMSNorm + RoPE, k/v into the static cache at pos");
    m.def("decode_attention", &decode_attention, "GQA decode attention over cache[0..pos]");
    m.def("decode_attention_split", &decode_attention_split, "GQA decode attention, split over positions (flash-decoding)");
    m.def("gemv", &gemv, "batch-1 GEMV, BF16 or INT8 weights");
    m.def("silu_mul", &silu_mul, "silu(a) * b on [a | b]");
}

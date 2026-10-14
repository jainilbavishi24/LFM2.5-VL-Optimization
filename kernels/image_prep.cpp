// PyTorch bindings for the image preprocessing kernels in image_prep.cu (compiled by the host compiler).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

void launch_resize_h(const void* in, void* tmp, const int* bounds, const int16_t* w, int K, int prec,
                     int H_in, int W_in, int W_out, cudaStream_t st);
void launch_resize_v_patchify(const void* tmp, float* out, const int* bounds, const int16_t* w, int K, int prec,
                              int W_out, int H_out, int tile_h, int tile_w, int tiles_x, int t0, int max_patches,
                              const float* mean, const float* std, cudaStream_t st);

static inline cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }

static void check_weights(const torch::Tensor& bounds, const torch::Tensor& w, int64_t n) {
    TORCH_CHECK(bounds.is_cuda() && bounds.scalar_type() == at::kInt && bounds.is_contiguous() && bounds.numel() == 2 * n,
                "bounds: contiguous CUDA int32 [n, 2]");
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kShort && w.is_contiguous() && w.dim() == 2 && w.size(0) == n,
                "weights: contiguous CUDA int16 [n, K]");
}

// img: uint8 [H, W, 3] -> tmp: uint8 [H, W_out, 3]
void resize_h(torch::Tensor img, torch::Tensor tmp, torch::Tensor bounds, torch::Tensor w, int64_t prec) {
    TORCH_CHECK(img.is_cuda() && img.scalar_type() == at::kByte && img.is_contiguous() && img.dim() == 3 && img.size(2) == 3,
                "img: contiguous CUDA uint8 [H, W, 3]");
    TORCH_CHECK(tmp.is_cuda() && tmp.scalar_type() == at::kByte && tmp.is_contiguous() && tmp.size(0) == img.size(0) && tmp.size(2) == 3,
                "tmp: contiguous CUDA uint8 [H, W_out, 3]");
    check_weights(bounds, w, tmp.size(1));
    launch_resize_h(img.data_ptr(), tmp.data_ptr(), bounds.data_ptr<int>(), (const int16_t*)w.data_ptr(), w.size(1), prec,
                    img.size(0), img.size(1), tmp.size(1), stream());
}

// tmp: uint8 [H_in, W_out, 3] -> out: fp32 [n_tiles, max_patches, 768], resized to H_out rows, tiles of tile_h x tile_w
// starting at tile t0 (row-major over tiles_x columns)
void resize_v_patchify(torch::Tensor tmp, torch::Tensor out, torch::Tensor bounds, torch::Tensor w, int64_t prec,
                       int64_t H_out, int64_t tile_h, int64_t tile_w, int64_t tiles_x, int64_t t0,
                       std::vector<double> mean, std::vector<double> std) {
    TORCH_CHECK(tmp.is_cuda() && tmp.scalar_type() == at::kByte && tmp.is_contiguous() && tmp.size(2) == 3, "tmp: CUDA uint8 [H, W, 3]");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat && out.is_contiguous() && out.dim() == 3 && out.size(2) == 768,
                "out: contiguous CUDA fp32 [n, max_patches, 768]");
    TORCH_CHECK(tile_h % 16 == 0 && tile_w % 16 == 0 && tmp.size(1) % tile_w == 0 && H_out % tile_h == 0, "16-aligned tiles");
    const int64_t tiles = (H_out / tile_h) * tiles_x;
    TORCH_CHECK(t0 + tiles <= out.size(0) && (tile_h / 16) * (tile_w / 16) <= out.size(1), "out too small");
    TORCH_CHECK(mean.size() == 3 && std.size() == 3, "mean/std: 3 channels");
    check_weights(bounds, w, H_out);
    const float m[3] = {(float)mean[0], (float)mean[1], (float)mean[2]}, s[3] = {(float)std[0], (float)std[1], (float)std[2]};
    launch_resize_v_patchify(tmp.data_ptr(), out.data_ptr<float>(), bounds.data_ptr<int>(), (const int16_t*)w.data_ptr(),
                             w.size(1), prec, tmp.size(1), H_out, tile_h, tile_w, tiles_x, t0, out.size(1), m, s, stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("resize_h", &resize_h, "antialiased bilinear resize, horizontal pass (uint8, int16 fixed-point weights)");
    m.def("resize_v_patchify", &resize_v_patchify,
          "vertical resize pass + normalize + tile split + 16x16 patchify into padded pixel_values");
}

// GPU image preprocessing for LFM2-VL (SigLIP2 NaFlex tiles): antialiased bilinear resize + rescale/normalize +
// tile split + 16x16 patchify, written straight into the padded `pixel_values` layout the vision tower reads.
//
// Bit-exact with the HF CPU path (torchvision -> torch's uint8 AVX resize, which follows Pillow): separable resize,
// horizontal pass first into a uint8 buffer, then vertical; int16 fixed-point weights with a per-pass precision,
// accumulator starts at 1 << (prec - 1), result >> prec and saturated to [0, 255]. The weights are computed on the host
// (same double-precision formula as ATen) so only the integer arithmetic runs here.
//
// Kernels only + raw-pointer launchers (torch bindings in image_prep.cpp; nvcc 13 cannot parse some ATen headers).
#include <cuda_runtime.h>
#include <cstdint>

// in: uint8 HWC [H_in, W_in, 3] -> tmp: uint8 [H_in, W_out, 3]
__global__ void resize_h_kernel(const uint8_t* __restrict__ in, uint8_t* __restrict__ tmp, const int* __restrict__ bounds,
                                const int16_t* __restrict__ w, int K, int prec, int H_in, int W_in, int W_out) {
    const int xo = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    if (xo >= W_out) return;
    const int xmin = bounds[2 * xo], xsize = bounds[2 * xo + 1];
    const int16_t* wx = w + (size_t)xo * K;
    const uint8_t* row = in + ((size_t)y * W_in + xmin) * 3;
    int s0 = 1 << (prec - 1), s1 = s0, s2 = s0;
    for (int k = 0; k < xsize; ++k) {
        const int wk = wx[k];
        s0 += wk * row[3 * k];
        s1 += wk * row[3 * k + 1];
        s2 += wk * row[3 * k + 2];
    }
    uint8_t* o = tmp + ((size_t)y * W_out + xo) * 3;
    o[0] = (uint8_t)min(max(s0 >> prec, 0), 255);
    o[1] = (uint8_t)min(max(s1 >> prec, 0), 255);
    o[2] = (uint8_t)min(max(s2 >> prec, 0), 255);
}

// tmp: uint8 [H_in, W_out, 3] -> out pixel (yo, xo) of the resized image, normalized, at its tile/patch slot:
//   tile  = t0 + (yo / tile_h) * tiles_x + xo / tile_w
//   patch = ((yo % tile_h) / 16) * (tile_w / 16) + (xo % tile_w) / 16
//   out[tile][patch][((yo % 16) * 16 + xo % 16) * 3 + c] = (u8 - mean) / std     (mean = std = 127.5 for LFM2-VL)
__global__ void resize_v_patchify_kernel(const uint8_t* __restrict__ tmp, float* __restrict__ out, const int* __restrict__ bounds,
                                         const int16_t* __restrict__ w, int K, int prec, int W_out, int H_out,
                                         int tile_h, int tile_w, int tiles_x, int t0, int max_patches,
                                         float m0, float m1, float m2, float s0f, float s1f, float s2f) {
    const int xo = blockIdx.x * blockDim.x + threadIdx.x, yo = blockIdx.y;
    if (xo >= W_out) return;
    const int ymin = bounds[2 * yo], ysize = bounds[2 * yo + 1];
    const int16_t* wy = w + (size_t)yo * K;
    const uint8_t* col = tmp + ((size_t)ymin * W_out + xo) * 3;
    const size_t stride = (size_t)W_out * 3;
    int a0 = 1 << (prec - 1), a1 = a0, a2 = a0;
    for (int k = 0; k < ysize; ++k) {
        const int wk = wy[k];
        const uint8_t* p = col + k * stride;
        a0 += wk * p[0];
        a1 += wk * p[1];
        a2 += wk * p[2];
    }
    const float v0 = (float)min(max(a0 >> prec, 0), 255), v1 = (float)min(max(a1 >> prec, 0), 255),
                v2 = (float)min(max(a2 >> prec, 0), 255);
    const int ty = yo / tile_h, tx = xo / tile_w, ry = yo - ty * tile_h, rx = xo - tx * tile_w;
    const size_t tile = t0 + ty * tiles_x + tx;
    const size_t patch = (ry >> 4) * (tile_w >> 4) + (rx >> 4);
    float* o = out + (tile * max_patches + patch) * 768 + (((ry & 15) << 4) + (rx & 15)) * 3;
    o[0] = (v0 - m0) / s0f;  // same op order as torchvision normalize (sub, then div) -> bit-identical floats
    o[1] = (v1 - m1) / s1f;
    o[2] = (v2 - m2) / s2f;
}

void launch_resize_h(const void* in, void* tmp, const int* bounds, const int16_t* w, int K, int prec,
                     int H_in, int W_in, int W_out, cudaStream_t st) {
    dim3 grid((W_out + 127) / 128, H_in);
    resize_h_kernel<<<grid, 128, 0, st>>>((const uint8_t*)in, (uint8_t*)tmp, bounds, w, K, prec, H_in, W_in, W_out);
}

void launch_resize_v_patchify(const void* tmp, float* out, const int* bounds, const int16_t* w, int K, int prec,
                              int W_out, int H_out, int tile_h, int tile_w, int tiles_x, int t0, int max_patches,
                              const float* mean, const float* std, cudaStream_t st) {
    dim3 grid((W_out + 127) / 128, H_out);
    resize_v_patchify_kernel<<<grid, 128, 0, st>>>((const uint8_t*)tmp, out, bounds, w, K, prec, W_out, H_out, tile_h,
                                                   tile_w, tiles_x, t0, max_patches, mean[0], mean[1], mean[2],
                                                   std[0], std[1], std[2]);
}

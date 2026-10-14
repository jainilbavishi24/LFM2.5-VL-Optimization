// Fused kernels for the SigLIP2 vision encoder (prefill-sized, 9216 x 1152 activations at 1080p).
//   add_layernorm: h += y (bf16-rounded like PyTorch's add), x = LayerNorm(h) * w + b  -- one pass instead of two
//   (residual add, then LayerNorm), one warp per row, statistics in fp32 (two-pass mean / variance in registers).
// Kernels only + raw-pointer launchers (torch bindings in vision_ops.cpp).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int MAXC = 16;  // up to D = 16 * 128 = 2048

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ void load4(const __nv_bfloat16* p, float* f) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&u.x), b = *reinterpret_cast<const __nv_bfloat162*>(&u.y);
    f[0] = __low2float(a); f[1] = __high2float(a); f[2] = __low2float(b); f[3] = __high2float(b);
}

__device__ __forceinline__ void store4(__nv_bfloat16* p, const float* f) {
    uint2 u;
    *reinterpret_cast<__nv_bfloat162*>(&u.x) = __floats2bfloat162_rn(f[0], f[1]);
    *reinterpret_cast<__nv_bfloat162*>(&u.y) = __floats2bfloat162_rn(f[2], f[3]);
    *reinterpret_cast<uint2*>(p) = u;
}

__global__ void add_layernorm_kernel(__nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ y,
                                     const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ b,
                                     __nv_bfloat16* __restrict__ x, int rows, int D, float eps) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= rows) return;  // whole warp exits together
    const int C = D >> 7;     // chunks of 4 elements per lane
    __nv_bfloat16* hr = h + (size_t)row * D;
    const __nv_bfloat16* yr = y ? y + (size_t)row * D : nullptr;
    float v[MAXC][4];
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < MAXC; ++c) {
        if (c < C) {
            const int i = (c * 32 + lane) * 4;
            load4(hr + i, v[c]);
            if (yr) {
                float t[4];
                load4(yr + i, t);
#pragma unroll
                for (int j = 0; j < 4; ++j) v[c][j] = __bfloat162float(__float2bfloat16_rn(v[c][j] + t[j]));
                store4(hr + i, v[c]);
            }
            s += v[c][0] + v[c][1] + v[c][2] + v[c][3];
        }
    }
    const float mean = warp_sum(s) / D;
    float q = 0.f;
#pragma unroll
    for (int c = 0; c < MAXC; ++c)
        if (c < C)
#pragma unroll
            for (int j = 0; j < 4; ++j) { const float d = v[c][j] - mean; q += d * d; }
    const float rstd = rsqrtf(warp_sum(q) / D + eps);
    __nv_bfloat16* xr = x + (size_t)row * D;
#pragma unroll
    for (int c = 0; c < MAXC; ++c) {
        if (c < C) {
            const int i = (c * 32 + lane) * 4;
            float wf[4], bf[4], o[4];
            load4(w + i, wf);
            load4(b + i, bf);
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = (v[c][j] - mean) * rstd * wf[j] + bf[j];
            store4(xr + i, o);
        }
    }
}

}  // namespace

void launch_add_layernorm(void* h, const void* y, const void* w, const void* b, void* x, int rows, int D, float eps,
                          cudaStream_t st) {
    const int threads = 256, rows_per_block = threads / 32;
    add_layernorm_kernel<<<(rows + rows_per_block - 1) / rows_per_block, threads, 0, st>>>(
        (__nv_bfloat16*)h, (const __nv_bfloat16*)y, (const __nv_bfloat16*)w, (const __nv_bfloat16*)b,
        (__nv_bfloat16*)x, rows, D, eps);
}

// INT8 tensor-core GEMMs for the prefill / vision path: Y[M, N] = X[M, K] W[N, K]^T, BF16 out, with the INT8 scales
// (+ bias, + GELU) applied in registers before the single BF16 store. cuBLAS' INT8 GEMM writes int32, and reading that
// back for the scales costs more than the INT8 math saves on wide outputs (profiling/int8_gemm_probe.py).
//
//   gemm_i8  (W8A8):  X int8 with one scale per row (token), W int8 with one scale per row (output channel);
//                     mma.m16n8k32 s8 -> exact int32 accumulation; y = (acc * sx[m]) * sw[n] + b[n]
//   gemm_w8  (W8A16): X bf16, W int8; W converted to bf16 in registers (exact: |q| <= 127) right before the
//                     mma.m16n8k16 bf16 -> fp32 accumulation; y = acc * sw[n] + b[n]
//   quant_rows:       X bf16 -> int8 + per-row scale (absmax / 127, round half to even), one warp per row
//
// Tiles: BM (64 | 128) x 128 x 64 bytes of K per stage, 8 warps (2 x 4, warp tile BM/2 x 32), cp.async multistage
// pipeline, XOR-swizzled shared memory (conflict-free ldmatrix / LDS), zero-filled M / N / K tails.
// Needs sm_80+ (mma.sync int8 / bf16, cp.async). Kernels + raw-pointer launchers (torch bindings in int8_gemm.cpp).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

typedef __nv_bfloat16 bf16;

namespace {

constexpr int BN = 128, NT = 256;

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void *src, bool pred) {
    const int n = pred ? 16 : 0;  // src-size 0: the 16 bytes are zero-filled
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t i8x2_to_bf16x2(int lo, int hi) {
    __nv_bfloat162 r = __floats2bfloat162_rn((float)lo, (float)hi);  // exact for |q| <= 127
    return *reinterpret_cast<uint32_t *>(&r);
}

__device__ __forceinline__ float gelu_tanh(float v) {  // = F.gelu(approximate="tanh")
    return 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}

// Epilogue for one m16n8 accumulator tile: fragment c0, c1 = row g, cols 2t, 2t+1; c2, c3 = row g + 8.
// v = (acc * sx[row]) * sw[col] (+ bias[col]); no FMA contraction, so W8A8 matches torch._int_mm + the same fp32 ops.
template <bool GELU>
__device__ __forceinline__ void store_tile(bf16 *Y, int M, int N, int row0, int col, const float (&v)[4],
                                           const float *sx, const float *sw, const bf16 *bias) {
    if (col >= N)
        return;  // N % 8 == 0: col + 1 < N too
    const float w0 = sw[col], w1 = sw[col + 1];
    const float b0 = bias ? __bfloat162float(bias[col]) : 0.f, b1 = bias ? __bfloat162float(bias[col + 1]) : 0.f;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int row = row0 + h * 8;
        if (row >= M)
            continue;
        const float s = sx ? sx[row] : 1.f;
        float y0 = __fmul_rn(v[2 * h], s), y1 = __fmul_rn(v[2 * h + 1], s);
        y0 = __fadd_rn(__fmul_rn(y0, w0), b0);
        y1 = __fadd_rn(__fmul_rn(y1, w1), b1);
        if (GELU) {
            y0 = gelu_tanh(y0);
            y1 = gelu_tanh(y1);
        }
        *reinterpret_cast<__nv_bfloat162 *>(Y + (size_t)row * N + col) = __floats2bfloat162_rn(y0, y1);
    }
}

// ------------------------------------------------------------------------------------------------ W8A8
// smem rows of 64 bytes = 4 chunks of 16; chunk c of row r is stored at slot c ^ ((r >> 1) & 3): the 8 rows one
// ldmatrix phase reads (same logical chunk) land on 8 distinct 16-byte bank groups.
__device__ __forceinline__ int swz_i8(int r, int c) { return (r * 4 + (c ^ ((r >> 1) & 3))) << 4; }

template <int BM, int STAGES, bool GELU>
__global__ void __launch_bounds__(NT) gemm_i8_kernel(const int8_t *__restrict__ A, const float *__restrict__ sa,
                                                      const int8_t *__restrict__ B, const float *__restrict__ sb,
                                                      const bf16 *__restrict__ bias, bf16 *__restrict__ Y, int M,
                                                      int N, int K) {
    constexpr int BK = 64, MI = BM / 32;  // MI m16 tiles per warp (warp tile BM/2 x 32)
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t *sA = smem, *sB = smem + STAGES * BM * BK;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp >> 2, wn = warp & 3;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, KT = (K + BK - 1) / BK;

    auto load = [&](int stage, int kt) {
        const int k0 = kt * BK;
#pragma unroll
        for (int it = 0; it < BM * 4 / NT; ++it) {
            const int i = tid + it * NT, r = i >> 2, c = i & 3, gr = m0 + r, gk = k0 + c * 16;
            const bool p = gr < M && gk < K;
            cp_async16(smem_u32(sA + stage * BM * BK + swz_i8(r, c)), A + (p ? (size_t)gr * K + gk : 0), p);
        }
#pragma unroll
        for (int it = 0; it < BN * 4 / NT; ++it) {
            const int i = tid + it * NT, r = i >> 2, c = i & 3, gr = n0 + r, gk = k0 + c * 16;
            const bool p = gr < N && gk < K;
            cp_async16(smem_u32(sB + stage * BN * BK + swz_i8(r, c)), B + (p ? (size_t)gr * K + gk : 0), p);
        }
    };

    int acc[MI][4][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                acc[i][j][e] = 0;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT)
            load(s, s);
        cp_async_commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();  // stage kt landed for everyone; stage kt - 1 is no longer read
        if (kt + STAGES - 1 < KT)
            load((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        cp_async_commit();
        const uint8_t *a_s = sA + (kt % STAGES) * BM * BK, *b_s = sB + (kt % STAGES) * BN * BK;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {  // two k32 steps per 64-byte stage
            uint32_t af[MI][4], bfr[4][2];
#pragma unroll
            for (int i = 0; i < MI; ++i) {  // a0..a3 = rows 0-7 / 8-15 x k 0-15 / 16-31
                const int r = wm * (BM / 2) + i * 16 + (lane & 15), c = ks * 2 + (lane >> 4);
                ldmatrix_x4(af[i], smem_u32(a_s + swz_i8(r, c)));
            }
#pragma unroll
            for (int j = 0; j < 4; j += 2) {  // b0, b1 of two n8 tiles
                const int r = wn * 32 + j * 8 + (lane & 7) + ((lane >> 4) << 3), c = ks * 2 + ((lane >> 3) & 1);
                uint32_t t[4];
                ldmatrix_x4(t, smem_u32(b_s + swz_i8(r, c)));
                bfr[j][0] = t[0];
                bfr[j][1] = t[1];
                bfr[j + 1][0] = t[2];
                bfr[j + 1][1] = t[3];
            }
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    mma_s8(acc[i][j], af[i], bfr[j][0], bfr[j][1]);
        }
    }
    cp_async_wait<0>();

    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float v[4] = {(float)acc[i][j][0], (float)acc[i][j][1], (float)acc[i][j][2], (float)acc[i][j][3]};
            store_tile<GELU>(Y, M, N, m0 + wm * (BM / 2) + i * 16 + g, n0 + wn * 32 + j * 8 + t4 * 2, v, sa, sb, bias);
        }
}

// ------------------------------------------------------------------------------------------------ W8A16
// Per stage: X tile BM x 32 bf16 (64-byte rows, 4 chunks), W tile BN x 32 int8 (32-byte rows, 2 chunks).
// Fragments are read with plain LDS: for each k16 step a thread (g = lane / 4, t = lane % 4) takes the 4 consecutive
// k at 4t..4t+3 of its row, for X (one 64-bit load) and for W (one 32-bit load = 4 int8). The mma's logical k slots
// {2t, 2t+1, 2t+8, 2t+9} are mapped to those physical k for both operands alike, so the dot product is unchanged.
__device__ __forceinline__ int swz_xa(int r, int c) { return (r * 4 + (c ^ (r & 2))) << 4; }       // 64-bit LDS
__device__ __forceinline__ int swz_wb(int r, int c) { return (r * 2 + (c ^ ((r >> 2) & 1))) << 4; }  // 32-bit LDS

template <int BM, int STAGES, bool GELU>
__global__ void __launch_bounds__(NT) gemm_w8_kernel(const bf16 *__restrict__ A, const int8_t *__restrict__ B,
                                                      const float *__restrict__ sb, const bf16 *__restrict__ bias,
                                                      bf16 *__restrict__ Y, int M, int N, int K) {
    constexpr int BK = 32, MI = BM / 32;
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t *sA = smem, *sB = smem + STAGES * BM * BK * 2;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp >> 2, wn = warp & 3;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, KT = (K + BK - 1) / BK;

    auto load = [&](int stage, int kt) {
        const int k0 = kt * BK;
#pragma unroll
        for (int it = 0; it < BM * 4 / NT; ++it) {
            const int i = tid + it * NT, r = i >> 2, c = i & 3, gr = m0 + r, gk = k0 + c * 8;
            const bool p = gr < M && gk < K;
            cp_async16(smem_u32(sA + stage * BM * BK * 2 + swz_xa(r, c)), A + (p ? (size_t)gr * K + gk : 0), p);
        }
#pragma unroll
        for (int it = 0; it < BN * 2 / NT; ++it) {
            const int i = tid + it * NT, r = i >> 1, c = i & 1, gr = n0 + r, gk = k0 + c * 16;
            const bool p = gr < N && gk < K;
            cp_async16(smem_u32(sB + stage * BN * BK + swz_wb(r, c)), B + (p ? (size_t)gr * K + gk : 0), p);
        }
    };

    float acc[MI][4][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                acc[i][j][e] = 0.f;

    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT)
            load(s, s);
        cp_async_commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < KT)
            load((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        cp_async_commit();
        const uint8_t *a_s = sA + (kt % STAGES) * BM * BK * 2, *b_s = sB + (kt % STAGES) * BN * BK;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {  // two k16 steps per stage
            uint32_t af[MI][4], bfr[4][2];
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {  // rows g, g + 8; bytes ks*32 + 8*t4 = chunk ks*2 + t4/2, offset (t4&1)*8
                    const int r = wm * (BM / 2) + i * 16 + g + h * 8;
                    const uint2 v = *reinterpret_cast<const uint2 *>(a_s + swz_xa(r, ks * 2 + (t4 >> 1)) + (t4 & 1) * 8);
                    af[i][h] = v.x;      // k 4t, 4t+1  -> logical slots 2t, 2t+1
                    af[i][2 + h] = v.y;  // k 4t+2, 4t+3 -> logical slots 2t+8, 2t+9
                }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = wn * 32 + j * 8 + g;
                const int q = *reinterpret_cast<const int *>(b_s + swz_wb(r, ks) + t4 * 4);
                bfr[j][0] = i8x2_to_bf16x2((q << 24) >> 24, (q << 16) >> 24);
                bfr[j][1] = i8x2_to_bf16x2((q << 8) >> 24, q >> 24);
            }
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    mma_bf16(acc[i][j], af[i], bfr[j][0], bfr[j][1]);
        }
    }
    cp_async_wait<0>();

#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            store_tile<GELU>(Y, M, N, m0 + wm * (BM / 2) + i * 16 + g, n0 + wn * 32 + j * 8 + t4 * 2, acc[i][j],
                             nullptr, sb, bias);
}

// ------------------------------------------------------------------------------------------------ quantize rows
constexpr int QMAXC = 16;  // K up to 16 * 128 = 2048

__global__ void quant_rows_kernel(const bf16 *__restrict__ x, int8_t *__restrict__ q, float *__restrict__ s, int rows,
                                  int K) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= rows)
        return;
    const int C = K >> 7;
    const bf16 *xr = x + (size_t)row * K;
    float v[QMAXC][4];
    float amax = 0.f;
#pragma unroll
    for (int c = 0; c < QMAXC; ++c)
        if (c < C) {
            const uint2 u = *reinterpret_cast<const uint2 *>(xr + (c * 32 + lane) * 4);
            const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162 *>(&u.x);
            const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162 *>(&u.y);
            v[c][0] = __low2float(a);
            v[c][1] = __high2float(a);
            v[c][2] = __low2float(b);
            v[c][3] = __high2float(b);
#pragma unroll
            for (int e = 0; e < 4; ++e)
                amax = fmaxf(amax, fabsf(v[c][e]));
        }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float sc = fmaxf(amax, 1e-8f) / 127.f;
    if (lane == 0)
        s[row] = sc;
#pragma unroll
    for (int c = 0; c < QMAXC; ++c)
        if (c < C) {
            char4 o;
            o.x = (signed char)max(-127, min(127, __float2int_rn(v[c][0] / sc)));
            o.y = (signed char)max(-127, min(127, __float2int_rn(v[c][1] / sc)));
            o.z = (signed char)max(-127, min(127, __float2int_rn(v[c][2] / sc)));
            o.w = (signed char)max(-127, min(127, __float2int_rn(v[c][3] / sc)));
            *reinterpret_cast<char4 *>(q + (size_t)row * K + (c * 32 + lane) * 4) = o;
        }
}

}  // namespace

// 48 KB of shared memory per block at BM = 128: two blocks per SM on Ada (100 KB) and Ampere (Jetson Orin)
constexpr int I8_STAGES = 3, W8_STAGES = 4;

// BM = 128 for prefill-sized M; 64 when 128-row tiles would leave most SMs idle (short prompts, one image tile)
static int pick_bm(int M, int N) {
    static int sms = 0;
    if (!sms) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    }
    const long tiles128 = (long)((M + 127) / 128) * ((N + BN - 1) / BN);
    return tiles128 >= 2L * sms ? 128 : 64;
}

#define LAUNCH_GEMM(KERN, BM_, ST_, GELU_, SMEM_, ...)                                                        \
    {                                                                                                          \
        static bool cfg = false;                                                                               \
        auto k = KERN<BM_, ST_, GELU_>;                                                                        \
        if (!cfg) {                                                                                            \
            cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(SMEM_));                \
            cfg = true;                                                                                        \
        }                                                                                                      \
        k<<<dim3((N + BN - 1) / BN, (M + BM_ - 1) / BM_), NT, SMEM_, st>>>(__VA_ARGS__);                       \
    }

void launch_gemm_i8(const void *A, const float *sa, const void *B, const float *sb, const void *bias, void *Y, int M,
                    int N, int K, bool gelu, cudaStream_t st) {
    const int8_t *a = (const int8_t *)A, *b = (const int8_t *)B;
    const bf16 *bi = (const bf16 *)bias;
    bf16 *y = (bf16 *)Y;
    const int bm = pick_bm(M, N);
    if (bm == 128) {
        constexpr size_t sm = (size_t)I8_STAGES * (128 + BN) * 64;
        if (gelu) LAUNCH_GEMM(gemm_i8_kernel, 128, I8_STAGES, true, sm, a, sa, b, sb, bi, y, M, N, K)
        else LAUNCH_GEMM(gemm_i8_kernel, 128, I8_STAGES, false, sm, a, sa, b, sb, bi, y, M, N, K)
    } else {
        constexpr size_t sm = (size_t)I8_STAGES * (64 + BN) * 64;
        if (gelu) LAUNCH_GEMM(gemm_i8_kernel, 64, I8_STAGES, true, sm, a, sa, b, sb, bi, y, M, N, K)
        else LAUNCH_GEMM(gemm_i8_kernel, 64, I8_STAGES, false, sm, a, sa, b, sb, bi, y, M, N, K)
    }
}

void launch_gemm_w8(const void *A, const void *B, const float *sb, const void *bias, void *Y, int M, int N, int K,
                    bool gelu, cudaStream_t st) {
    const bf16 *a = (const bf16 *)A, *bi = (const bf16 *)bias;
    const int8_t *b = (const int8_t *)B;
    bf16 *y = (bf16 *)Y;
    const int bm = pick_bm(M, N);
    if (bm == 128) {
        constexpr size_t sm = (size_t)W8_STAGES * (128 * 64 + BN * 32);
        if (gelu) LAUNCH_GEMM(gemm_w8_kernel, 128, W8_STAGES, true, sm, a, b, sb, bi, y, M, N, K)
        else LAUNCH_GEMM(gemm_w8_kernel, 128, W8_STAGES, false, sm, a, b, sb, bi, y, M, N, K)
    } else {
        constexpr size_t sm = (size_t)W8_STAGES * (64 * 64 + BN * 32);
        if (gelu) LAUNCH_GEMM(gemm_w8_kernel, 64, W8_STAGES, true, sm, a, b, sb, bi, y, M, N, K)
        else LAUNCH_GEMM(gemm_w8_kernel, 64, W8_STAGES, false, sm, a, b, sb, bi, y, M, N, K)
    }
}

void launch_quant_rows(const void *x, void *q, float *s, int rows, int K, cudaStream_t st) {
    const int threads = 256, rpb = threads / 32;
    quant_rows_kernel<<<(rows + rpb - 1) / rpb, threads, 0, st>>>((const bf16 *)x, (int8_t *)q, s, rows, K);
}

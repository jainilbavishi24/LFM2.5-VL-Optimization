// INT8 tensor-core GEMMs for the prefill / vision path: Y[M, N] = X[M, K] W[N, K]^T, BF16 out, with the INT8 scales
// (+ bias, + GELU) applied in registers before the single BF16 store. cuBLAS' INT8 GEMM writes int32, and reading that
// back for the scales costs more than the INT8 math saves on wide outputs (profiling/int8_gemm_probe.py).
//
//   gemm_i8  (W8A8):  X int8 with one scale per row (token), W int8 with one scale per row (output channel);
//                     mma.m16n8k32 s8 -> exact int32 accumulation; y = (acc * sx[m]) * sw[n] + b[n]
//                     Grouped: one X scale per row and group of 64 input channels (sxT[K/64][M]): after each
//                     64-wide K stage the int32 partial sums are folded into fp32, acc = fma(float(part), sxT[g][m], acc)
//                     (outlier channels then only coarsen their own group: W8A8 at the W8A16 quality level)
//   gemm_w8  (W8A16): X bf16, W int8; W converted to bf16 in registers (exact: |q| <= 127) right before the
//                     mma.m16n8k16 bf16 -> fp32 accumulation; y = acc * sw[n] + b[n]
//   quant_rows:       X bf16 -> int8 + per-row scale (absmax / 127, round half to even), one warp per row
//   quant_groups:     the same with one scale per row and group of 64 (written transposed, [K/64][M])
//
// Tiles: BM (64 | 128) x 128 x 64 bytes of K per stage, 8 warps (2 x 4, warp tile BM/2 x 32), cp.async multistage
// pipeline, XOR-swizzled shared memory (conflict-free ldmatrix / LDS), zero-filled M / N / K tails.
// Needs sm_80+ (mma.sync int8 / bf16, cp.async). Kernels + raw-pointer launchers (torch bindings in int8_gemm.cpp).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

typedef __nv_bfloat16 bf16;

namespace {

constexpr int BN = 128, NT = 256;

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void *src, bool pred) {
    const int n = pred ? 16 : 0;  // src-size 0: the 16 bytes are zero-filled
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void *src, bool pred) {
    const int n = pred ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(n));
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

// 4 int8 -> 2 x bf16x2 without the (quarter-rate) int -> float conversions: q + 128 is placed in the mantissa of
// 2^23 (fp32 bits 0x4B0000uu), 2^23 + 128 is subtracted (exact), and the upper halves of the small-integer fp32 values
// are their exact bf16 encodings. 1 LOP + 6 PRMT + 4 FADD per 4 weights (the trick vLLM's Marlin kernels use).
__device__ __forceinline__ void i8x4_to_bf16x4(int q, uint32_t &lo, uint32_t &hi) {
    const uint32_t u = (uint32_t)q ^ 0x80808080u, base = 0x4B000000u;
    const float a = __uint_as_float(__byte_perm(u, base, 0x7650)) - 8388736.f;
    const float b = __uint_as_float(__byte_perm(u, base, 0x7651)) - 8388736.f;
    const float c = __uint_as_float(__byte_perm(u, base, 0x7652)) - 8388736.f;
    const float d = __uint_as_float(__byte_perm(u, base, 0x7653)) - 8388736.f;
    lo = __byte_perm(__float_as_uint(a), __float_as_uint(b), 0x7632);
    hi = __byte_perm(__float_as_uint(c), __float_as_uint(d), 0x7632);
}

// exact int -> float for |v| < 2^22 (a group of 64: |v| <= 127 * 127 * 64): integer add into the mantissa of 1.5 * 2^23
__device__ __forceinline__ float i2f_small(int v) { return __int_as_float(v + 0x4B400000) - 12582912.f; }

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

template <int BM, int STAGES, bool GELU, bool GROUPED>
__global__ void __launch_bounds__(NT) gemm_i8_kernel(const int8_t *__restrict__ A, const float *__restrict__ sa,
                                                      const int8_t *__restrict__ B, const float *__restrict__ sb,
                                                      const bf16 *__restrict__ bias, bf16 *__restrict__ Y, int M,
                                                      int N, int K) {
    constexpr int BK = 64, MI = BM / 32;  // MI m16 tiles per warp (warp tile BM/2 x 32)
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t *sA = smem, *sB = smem + STAGES * BM * BK;
    float *sS = reinterpret_cast<float *>(sB + STAGES * BN * BK);  // GROUPED: [STAGES][BM] X scales of the stage's group
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp >> 2, wn = warp & 3;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, KT = (K + BK - 1) / BK;

    auto load = [&](int stage, int kt) {
        const int k0 = kt * BK;
        if (GROUPED && tid < BM)
            cp_async4(smem_u32(sS + stage * BM + tid), sa + (size_t)kt * M + m0 + tid, m0 + tid < M);
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
    float facc[GROUPED ? MI : 1][4][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[i][j][e] = 0;
                if (GROUPED)
                    facc[GROUPED ? i : 0][j][e] = 0.f;
            }

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT)
            load(s, s);
        cp_async_commit();
    }
    const int g = lane >> 2, t4 = lane & 3;
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();  // stage kt landed for everyone; stage kt - 1 is no longer read
        if (kt + STAGES - 1 < KT)
            load((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        cp_async_commit();
        const uint8_t *a_s = sA + (kt % STAGES) * BM * BK, *b_s = sB + (kt % STAGES) * BN * BK;
        uint32_t af[2][MI][4], bfr[2][4][2];  // both k32 steps of the stage
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
            for (int i = 0; i < MI; ++i) {  // a0..a3 = rows 0-7 / 8-15 x k 0-15 / 16-31
                const int r = wm * (BM / 2) + i * 16 + (lane & 15), c = ks * 2 + (lane >> 4);
                ldmatrix_x4(af[ks][i], smem_u32(a_s + swz_i8(r, c)));
            }
#pragma unroll
            for (int j = 0; j < 4; j += 2) {  // b0, b1 of two n8 tiles
                const int r = wn * 32 + j * 8 + (lane & 7) + ((lane >> 4) << 3), c = ks * 2 + ((lane >> 3) & 1);
                uint32_t t[4];
                ldmatrix_x4(t, smem_u32(b_s + swz_i8(r, c)));
                bfr[ks][j][0] = t[0];
                bfr[ks][j][1] = t[1];
                bfr[ks][j + 1][0] = t[2];
                bfr[ks][j + 1][1] = t[3];
            }
        }
        // GROUPED: fold this group's int32 sums into fp32 with the row's scale. Tile i - 1 is folded right after the
        // last mma of tile i is issued, so the ALU work overlaps the tensor cores instead of following all of them.
        const float *s_s = sS + (kt % STAGES) * BM;
        auto fold = [&](int i) {
            const float s0 = s_s[wm * (BM / 2) + i * 16 + g], s1 = s_s[wm * (BM / 2) + i * 16 + g + 8];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    facc[GROUPED ? i : 0][j][e] =
                        __fmaf_rn(i2f_small(acc[i][j][e]), e < 2 ? s0 : s1, facc[GROUPED ? i : 0][j][e]);
                    acc[i][j][e] = 0;
                }
        };
#pragma unroll
        for (int i = 0; i < MI; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                mma_s8(acc[i][j], af[0][i], bfr[0][j][0], bfr[0][j][1]);
#pragma unroll
        for (int i = 0; i < MI; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                mma_s8(acc[i][j], af[1][i], bfr[1][j][0], bfr[1][j][1]);
            if (GROUPED && i > 0)
                fold(i - 1);
        }
        if (GROUPED)
            fold(MI - 1);
    }
    cp_async_wait<0>();

#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float v[4];
#pragma unroll
            for (int e = 0; e < 4; ++e)
                v[e] = GROUPED ? facc[GROUPED ? i : 0][j][e] : (float)acc[i][j][e];
            store_tile<GELU>(Y, M, N, m0 + wm * (BM / 2) + i * 16 + g, n0 + wn * 32 + j * 8 + t4 * 2, v,
                             GROUPED ? nullptr : sa, sb, bias);
        }
}

// ------------------------------------------------------------------------------------------------ W8A16
// Per stage: X tile BM x 64 bf16 (128-byte rows, 8 chunks), W tile BN x 64 int8 (64-byte rows, 4 chunks).
// Fragments are read with plain LDS: for each k16 step a thread (g = lane / 4, t = lane % 4) takes the 4 consecutive
// k at 4t..4t+3 of its row, for X (one 64-bit load) and for W (one 32-bit load = 4 int8). The mma's logical k slots
// {2t, 2t+1, 2t+8, 2t+9} are mapped to those physical k for both operands alike, so the dot product is unchanged.
__device__ __forceinline__ int swz_xa(int r, int c) { return (r * 8 + (c ^ ((r & 3) << 1))) << 4; }  // 64-bit LDS

template <int BM, int STAGES, bool GELU>
__global__ void __launch_bounds__(NT) gemm_w8_kernel(const bf16 *__restrict__ A, const int8_t *__restrict__ B,
                                                      const float *__restrict__ sb, const bf16 *__restrict__ bias,
                                                      bf16 *__restrict__ Y, int M, int N, int K) {
    constexpr int BK = 64, MI = BM / 32;
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t *sA = smem, *sB = smem + STAGES * BM * BK * 2;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp >> 2, wn = warp & 3;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, KT = (K + BK - 1) / BK;

    auto load = [&](int stage, int kt) {
        const int k0 = kt * BK;
#pragma unroll
        for (int it = 0; it < BM * 8 / NT; ++it) {
            const int i = tid + it * NT, r = i >> 3, c = i & 7, gr = m0 + r, gk = k0 + c * 8;
            const bool p = gr < M && gk < K;
            cp_async16(smem_u32(sA + stage * BM * BK * 2 + swz_xa(r, c)), A + (p ? (size_t)gr * K + gk : 0), p);
        }
#pragma unroll
        for (int it = 0; it < BN * 4 / NT; ++it) {
            const int i = tid + it * NT, r = i >> 2, c = i & 3, gr = n0 + r, gk = k0 + c * 16;
            const bool p = gr < N && gk < K;
            cp_async16(smem_u32(sB + stage * BN * BK + swz_i8(r, c)), B + (p ? (size_t)gr * K + gk : 0), p);
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
        for (int ks = 0; ks < 4; ++ks) {  // four k16 steps per stage
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
                const int q = *reinterpret_cast<const int *>(b_s + swz_i8(r, ks) + t4 * 4);
                i8x4_to_bf16x4(q, bfr[j][0], bfr[j][1]);
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

// one warp per row, a 128-wide chunk at a time: lanes 0-15 hold group 2c, lanes 16-31 group 2c + 1 (4 values each)
__global__ void quant_groups_kernel(const bf16 *__restrict__ x, int8_t *__restrict__ q, float *__restrict__ sT, int rows,
                                    int K) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= rows)
        return;
    const bf16 *xr = x + (size_t)row * K;
    const int C = K >> 7;
    for (int c0 = 0; c0 < C; c0 += 4) {  // 4 chunks per round: their loads are in flight together
        float v[4][4];
#pragma unroll
        for (int u = 0; u < 4; ++u)
            if (c0 + u < C) {
                const uint2 w = *reinterpret_cast<const uint2 *>(xr + ((c0 + u) * 32 + lane) * 4);
                const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162 *>(&w.x);
                const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162 *>(&w.y);
                v[u][0] = __low2float(a);
                v[u][1] = __high2float(a);
                v[u][2] = __low2float(b);
                v[u][3] = __high2float(b);
            }
#pragma unroll
        for (int u = 0; u < 4; ++u)
            if (c0 + u < C) {
                const int c = c0 + u;
                float amax = fmaxf(fmaxf(fabsf(v[u][0]), fabsf(v[u][1])), fmaxf(fabsf(v[u][2]), fabsf(v[u][3])));
#pragma unroll
                for (int o = 8; o > 0; o >>= 1)
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                const float sc = fmaxf(amax, 1e-8f) / 127.f;
                if ((lane & 15) == 0)
                    sT[(size_t)(2 * c + (lane >> 4)) * rows + row] = sc;
                char4 o;
                o.x = (signed char)max(-127, min(127, __float2int_rn(v[u][0] / sc)));
                o.y = (signed char)max(-127, min(127, __float2int_rn(v[u][1] / sc)));
                o.z = (signed char)max(-127, min(127, __float2int_rn(v[u][2] / sc)));
                o.w = (signed char)max(-127, min(127, __float2int_rn(v[u][3] / sc)));
                *reinterpret_cast<char4 *>(q + (size_t)row * K + (c * 32 + lane) * 4) = o;
            }
    }
}

}  // namespace

// 48 KB of shared memory per block at BM = 128: two blocks per SM on Ada (100 KB) and Ampere (Jetson Orin)
constexpr int I8_STAGES = 3, W8_STAGES = 3;

// BM = 128 for prefill-sized M; 64 when 128-row tiles would leave most SMs idle (short prompts, one image tile)
static int pick_bm(int M, int N) {
    static const int force = getenv("LFM2_INT8_BM") ? atoi(getenv("LFM2_INT8_BM")) : 0;  // 64 / 128: for A/B tests
    if (force)
        return force;
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

template <int BM, bool GELU, bool GROUPED>
static void run_i8(const int8_t *a, const float *sa, const int8_t *b, const float *sb, const bf16 *bi, bf16 *y, int M,
                   int N, int K, cudaStream_t st) {
    constexpr size_t sm = (size_t)I8_STAGES * ((BM + BN) * 64 + (GROUPED ? BM * 4 : 0));
    auto k = gemm_i8_kernel<BM, I8_STAGES, GELU, GROUPED>;
    static bool cfg = false;
    if (!cfg) {
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
        cfg = true;
    }
    k<<<dim3((N + BN - 1) / BN, (M + BM - 1) / BM), NT, sm, st>>>(a, sa, b, sb, bi, y, M, N, K);
}

template <int BM>
static void run_i8_bm(const int8_t *a, const float *sa, const int8_t *b, const float *sb, const bf16 *bi, bf16 *y,
                      int M, int N, int K, bool gelu, bool grouped, cudaStream_t st) {
    if (grouped) {
        if (gelu) run_i8<BM, true, true>(a, sa, b, sb, bi, y, M, N, K, st);
        else run_i8<BM, false, true>(a, sa, b, sb, bi, y, M, N, K, st);
    } else {
        if (gelu) run_i8<BM, true, false>(a, sa, b, sb, bi, y, M, N, K, st);
        else run_i8<BM, false, false>(a, sa, b, sb, bi, y, M, N, K, st);
    }
}

// grouped: sa = sxT [K/64][M] (one scale per row and group of 64), else sa = sx [M]
void launch_gemm_i8(const void *A, const float *sa, const void *B, const float *sb, const void *bias, void *Y, int M,
                    int N, int K, bool gelu, bool grouped, cudaStream_t st) {
    const int8_t *a = (const int8_t *)A, *b = (const int8_t *)B;
    const bf16 *bi = (const bf16 *)bias;
    bf16 *y = (bf16 *)Y;
    if (pick_bm(M, N) == 128) run_i8_bm<128>(a, sa, b, sb, bi, y, M, N, K, gelu, grouped, st);
    else run_i8_bm<64>(a, sa, b, sb, bi, y, M, N, K, gelu, grouped, st);
}

void launch_gemm_w8(const void *A, const void *B, const float *sb, const void *bias, void *Y, int M, int N, int K,
                    bool gelu, cudaStream_t st) {
    const bf16 *a = (const bf16 *)A, *bi = (const bf16 *)bias;
    const int8_t *b = (const int8_t *)B;
    bf16 *y = (bf16 *)Y;
    const int bm = pick_bm(M, N);
    if (bm == 128) {
        constexpr size_t sm = (size_t)W8_STAGES * (128 * 128 + BN * 64);
        if (gelu) LAUNCH_GEMM(gemm_w8_kernel, 128, W8_STAGES, true, sm, a, b, sb, bi, y, M, N, K)
        else LAUNCH_GEMM(gemm_w8_kernel, 128, W8_STAGES, false, sm, a, b, sb, bi, y, M, N, K)
    } else {
        constexpr size_t sm = (size_t)W8_STAGES * (64 * 128 + BN * 64);
        if (gelu) LAUNCH_GEMM(gemm_w8_kernel, 64, W8_STAGES, true, sm, a, b, sb, bi, y, M, N, K)
        else LAUNCH_GEMM(gemm_w8_kernel, 64, W8_STAGES, false, sm, a, b, sb, bi, y, M, N, K)
    }
}

void launch_quant_groups(const void *x, void *q, float *sT, int rows, int K, cudaStream_t st) {
    const int threads = 256, rpb = threads / 32;
    quant_groups_kernel<<<(rows + rpb - 1) / rpb, threads, 0, st>>>((const bf16 *)x, (int8_t *)q, sT, rows, K);
}

void launch_quant_rows(const void *x, void *q, float *s, int rows, int K, cudaStream_t st) {
    const int threads = 256, rpb = threads / 32;
    quant_rows_kernel<<<(rows + rpb - 1) / rpb, threads, 0, st>>>((const bf16 *)x, (int8_t *)q, s, rows, K);
}

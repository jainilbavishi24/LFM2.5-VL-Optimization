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
//   add_layernorm_q / add_rmsnorm_q / silu_mul_q: the vision / LM norm and SwiGLU kernels with the next GEMM's INT8
//                     input as output (per row or per group of 64): the bf16 value is computed exactly as in the
//                     unfused kernel (vision_ops.cu, lfm2_decode.cu) and quantized in registers, so the activation
//                     never makes a bf16 round trip through memory and needs no extra launch
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

__device__ __forceinline__ signed char q8(float v, float sc) {
    return (signed char)max(-127, min(127, __float2int_rn(v / sc)));
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

// QOUT (with GROUPED): instead of BF16, write the output already quantized for the next W8A8 GEMM: INT8 Yq [M, N] and
// one scale per row and group of 64 columns, YsT [N/64][M] (a 128-wide tile = 2 groups, each spread over 2 warps whose
// row maxima meet in shared memory). The quantized value is the bf16-rounded output, as quant_rows would see it.
template <int BM, int STAGES, bool GELU, bool GROUPED, bool QOUT>
__global__ void __launch_bounds__(NT) gemm_i8_kernel(const int8_t *__restrict__ A, const float *__restrict__ sa,
                                                      const int8_t *__restrict__ B, const float *__restrict__ sb,
                                                      const bf16 *__restrict__ bias, bf16 *__restrict__ Y, int M,
                                                      int N, int K, int8_t *__restrict__ Yq, float *__restrict__ YsT) {
    static_assert(!QOUT || GROUPED, "QOUT reuses the fp32 accumulators of the grouped path");
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

    if (QOUT) {
        __syncthreads();  // every warp is done with the pipeline buffers: reuse them for the row maxima
        float *red = reinterpret_cast<float *>(smem);  // [BM][4 warps along N]
#pragma unroll
        for (int i = 0; i < MI; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {  // final values, bf16-rounded exactly as store_tile writes them (N % 128 == 0)
                const int col = n0 + wn * 32 + j * 8 + t4 * 2;
                const float w0 = sb[col], w1 = sb[col + 1];
                const float b0 = bias ? __bfloat162float(bias[col]) : 0.f, b1 = bias ? __bfloat162float(bias[col + 1]) : 0.f;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float y = __fadd_rn(__fmul_rn(facc[GROUPED ? i : 0][j][e], (e & 1) ? w1 : w0), (e & 1) ? b1 : b0);
                    if (GELU)
                        y = gelu_tanh(y);
                    facc[GROUPED ? i : 0][j][e] = __bfloat162float(__float2bfloat16_rn(y));
                }
            }
#pragma unroll
        for (int i = 0; i < MI; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                float m = 0.f;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    m = fmaxf(m, fmaxf(fabsf(facc[GROUPED ? i : 0][j][2 * h]), fabsf(facc[GROUPED ? i : 0][j][2 * h + 1])));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                if (t4 == 0)
                    red[(wm * (BM / 2) + i * 16 + g + 8 * h) * 4 + wn] = m;
            }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < MI; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = wm * (BM / 2) + i * 16 + g + 8 * h, row = m0 + r;
                if (row >= M)
                    continue;
                const float sc = fmaxf(fmaxf(red[r * 4 + wn], red[r * 4 + (wn ^ 1)]), 1e-8f) / 127.f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    char2 c;
                    c.x = q8(facc[GROUPED ? i : 0][j][2 * h], sc);
                    c.y = q8(facc[GROUPED ? i : 0][j][2 * h + 1], sc);
                    *reinterpret_cast<char2 *>(Yq + (size_t)row * N + n0 + wn * 32 + j * 8 + t4 * 2) = c;
                }
                if (t4 == 0 && (wn & 1) == 0)
                    YsT[(size_t)((n0 + wn * 32) >> 6) * M + row] = sc;
            }
        return;
    }
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

// ------------------------------------------------------------------------------------------------ fused producers
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// one warp per row holding v[c][0..3] = elements (c * 32 + lane) * 4 + 0..3: write its INT8 row and scale(s)
template <int MAXC, bool GROUPED>
__device__ __forceinline__ void warp_quant_store(const float (&v)[MAXC][4], int C, int8_t *__restrict__ q,
                                                 float *__restrict__ s, int row, int rows, int K, int lane) {
    float sc_row = 0.f;
    if (!GROUPED) {
        float amax = 0.f;
#pragma unroll
        for (int c = 0; c < MAXC; ++c)
            if (c < C)
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v[c][0]), fabsf(v[c][1])), fmaxf(fabsf(v[c][2]), fabsf(v[c][3]))));
        sc_row = fmaxf(wmax(amax), 1e-8f) / 127.f;
        if (lane == 0)
            s[row] = sc_row;
    }
#pragma unroll
    for (int c = 0; c < MAXC; ++c)
        if (c < C) {
            float sc = sc_row;
            if (GROUPED) {  // lanes 0-15: group 2c, lanes 16-31: group 2c + 1
                float amax = fmaxf(fmaxf(fabsf(v[c][0]), fabsf(v[c][1])), fmaxf(fabsf(v[c][2]), fabsf(v[c][3])));
#pragma unroll
                for (int o = 8; o > 0; o >>= 1)
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                sc = fmaxf(amax, 1e-8f) / 127.f;
                if ((lane & 15) == 0)
                    s[(size_t)(2 * c + (lane >> 4)) * rows + row] = sc;
            }
            char4 o;
            o.x = q8(v[c][0], sc);
            o.y = q8(v[c][1], sc);
            o.z = q8(v[c][2], sc);
            o.w = q8(v[c][3], sc);
            *reinterpret_cast<char4 *>(q + (size_t)row * K + (c * 32 + lane) * 4) = o;
        }
}

// vision: h += y (bf16-rounded); x = LayerNorm(h) * w + b, rounded to bf16 exactly as vision_ops.cu add_layernorm,
// then quantized. One warp per row, D <= 2048, D % 128 == 0.
template <bool GROUPED>
__global__ void add_layernorm_q_kernel(bf16 *__restrict__ h, const bf16 *__restrict__ y, const bf16 *__restrict__ w,
                                       const bf16 *__restrict__ b, int8_t *__restrict__ q, float *__restrict__ s,
                                       int rows, int D, float eps) {
    constexpr int MAXC = 16;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= rows)
        return;
    const int C = D >> 7;
    bf16 *hr = h + (size_t)row * D;
    const bf16 *yr = y ? y + (size_t)row * D : nullptr;
    float v[MAXC][4];
    float sum = 0.f;
#pragma unroll
    for (int c = 0; c < MAXC; ++c) {
        if (c < C) {
            const int i = (c * 32 + lane) * 4;
            const uint2 u = *reinterpret_cast<const uint2 *>(hr + i);
            const __nv_bfloat162 a0 = *reinterpret_cast<const __nv_bfloat162 *>(&u.x), a1 = *reinterpret_cast<const __nv_bfloat162 *>(&u.y);
            v[c][0] = __low2float(a0); v[c][1] = __high2float(a0); v[c][2] = __low2float(a1); v[c][3] = __high2float(a1);
            if (yr) {
                const uint2 t = *reinterpret_cast<const uint2 *>(yr + i);
                const __nv_bfloat162 t0 = *reinterpret_cast<const __nv_bfloat162 *>(&t.x), t1 = *reinterpret_cast<const __nv_bfloat162 *>(&t.y);
                const float tf[4] = {__low2float(t0), __high2float(t0), __low2float(t1), __high2float(t1)};
#pragma unroll
                for (int j = 0; j < 4; ++j) v[c][j] = __bfloat162float(__float2bfloat16_rn(v[c][j] + tf[j]));
                uint2 o;
                *reinterpret_cast<__nv_bfloat162 *>(&o.x) = __floats2bfloat162_rn(v[c][0], v[c][1]);
                *reinterpret_cast<__nv_bfloat162 *>(&o.y) = __floats2bfloat162_rn(v[c][2], v[c][3]);
                *reinterpret_cast<uint2 *>(hr + i) = o;
            }
            sum += v[c][0] + v[c][1] + v[c][2] + v[c][3];
        }
    }
    const float mean = wsum(sum) / D;
    float qs = 0.f;
#pragma unroll
    for (int c = 0; c < MAXC; ++c)
        if (c < C)
#pragma unroll
            for (int j = 0; j < 4; ++j) { const float d = v[c][j] - mean; qs += d * d; }
    const float rstd = rsqrtf(wsum(qs) / D + eps);
#pragma unroll
    for (int c = 0; c < MAXC; ++c) {
        if (c < C) {
            const int i = (c * 32 + lane) * 4;
            const uint2 uw = *reinterpret_cast<const uint2 *>(w + i), ub = *reinterpret_cast<const uint2 *>(b + i);
            const __nv_bfloat162 w0 = *reinterpret_cast<const __nv_bfloat162 *>(&uw.x), w1 = *reinterpret_cast<const __nv_bfloat162 *>(&uw.y);
            const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162 *>(&ub.x), b1 = *reinterpret_cast<const __nv_bfloat162 *>(&ub.y);
            const float wf[4] = {__low2float(w0), __high2float(w0), __low2float(w1), __high2float(w1)};
            const float bf[4] = {__low2float(b0), __high2float(b0), __low2float(b1), __high2float(b1)};
#pragma unroll
            for (int j = 0; j < 4; ++j) v[c][j] = __bfloat162float(__float2bfloat16_rn((v[c][j] - mean) * rstd * wf[j] + bf[j]));
        }
    }
    warp_quant_store<MAXC, GROUPED>(v, C, q, s, row, rows, D, lane);
}

// LM prefill: h += y (bf16-rounded); x = bf16(w * bf16(h * rsqrt(mean(h^2) + eps))) exactly as lfm2_decode.cu
// add_rmsnorm (one 256-thread block per row, same reduction order), then quantized.
constexpr int RN_THREADS = 256;
template <bool ADD, bool GROUPED>
__global__ void add_rmsnorm_q_kernel(bf16 *__restrict__ h, const bf16 *__restrict__ y, const bf16 *__restrict__ w,
                                     int8_t *__restrict__ q, float *__restrict__ s, int rows, int H, float eps) {
    extern __shared__ float rsm[];  // H floats + reduction scratch
    float *hv = rsm, *red = rsm + H;
    const size_t off = (size_t)blockIdx.x * H;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float ss = 0.f;
    for (int i = threadIdx.x; i < H; i += RN_THREADS) {
        float v = __bfloat162float(h[off + i]);
        if (ADD) {
            v = __bfloat162float(__float2bfloat16(v + __bfloat162float(y[off + i])));
            h[off + i] = __float2bfloat16(v);
        }
        hv[i] = v;
        ss += v * v;
    }
    ss = wsum(ss);  // block_sum<8> of lfm2_decode.cu
    if (lane == 0)
        red[warp] = ss;
    __syncthreads();
    ss = wsum(lane < RN_THREADS / 32 ? red[lane] : 0.f);
    __syncthreads();
    const float rs = rsqrtf(ss / H + eps);
    float amax = 0.f;
    for (int i = threadIdx.x; i < H; i += RN_THREADS) {
        const float n = __bfloat162float(__float2bfloat16(hv[i] * rs));
        const float x = __bfloat162float(__float2bfloat16(__bfloat162float(w[i]) * n));
        hv[i] = x;
        amax = fmaxf(amax, fabsf(x));
    }
    if (GROUPED) {
        __syncthreads();
        for (int gi = warp; gi < H / 64; gi += RN_THREADS / 32) {
            const float a = hv[gi * 64 + lane], b = hv[gi * 64 + 32 + lane];
            const float sc = fmaxf(wmax(fmaxf(fabsf(a), fabsf(b))), 1e-8f) / 127.f;
            q[off + gi * 64 + lane] = q8(a, sc);
            q[off + gi * 64 + 32 + lane] = q8(b, sc);
            if (lane == 0)
                s[(size_t)gi * rows + blockIdx.x] = sc;
        }
    } else {
        amax = wmax(amax);
        if (lane == 0)
            red[warp] = amax;
        __syncthreads();
        const float sc = fmaxf(wmax(lane < RN_THREADS / 32 ? red[lane] : 0.f), 1e-8f) / 127.f;
        if (threadIdx.x == 0)
            s[blockIdx.x] = sc;
        for (int i = threadIdx.x; i < H; i += RN_THREADS)
            q[off + i] = q8(hv[i], sc);
    }
}

// LM prefill SwiGLU: g = [rows, 2I] (w1 | w3 outputs); a = bf16(bf16(silu(g1)) * g3) exactly as lfm2_decode.cu
// silu_mul, then quantized (the w2 input). One 256-thread block per row; a goes through shared memory.
template <bool GROUPED>
__global__ void silu_mul_q_kernel(const bf16 *__restrict__ g, int8_t *__restrict__ q, float *__restrict__ s, int rows,
                                  int I) {
    extern __shared__ float rsm[];  // I floats + scratch
    float *av = rsm, *red = rsm + I;
    const bf16 *gr = g + (size_t)blockIdx.x * 2 * I;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float amax = 0.f;
    for (int i = threadIdx.x; i < I; i += RN_THREADS) {
        const float a = __bfloat162float(gr[i]), u = __bfloat162float(gr[I + i]);
        const float v = __bfloat162float(__float2bfloat16(__bfloat162float(__float2bfloat16(a / (1.f + __expf(-a)))) * u));
        av[i] = v;
        amax = fmaxf(amax, fabsf(v));
    }
    const size_t off = (size_t)blockIdx.x * I;
    if (GROUPED) {
        __syncthreads();
        for (int gi = warp; gi < I / 64; gi += RN_THREADS / 32) {
            const float a = av[gi * 64 + lane], b = av[gi * 64 + 32 + lane];
            const float sc = fmaxf(wmax(fmaxf(fabsf(a), fabsf(b))), 1e-8f) / 127.f;
            q[off + gi * 64 + lane] = q8(a, sc);
            q[off + gi * 64 + 32 + lane] = q8(b, sc);
            if (lane == 0)
                s[(size_t)gi * rows + blockIdx.x] = sc;
        }
    } else {
        amax = wmax(amax);
        if (lane == 0)
            red[warp] = amax;
        __syncthreads();
        const float sc = fmaxf(wmax(lane < RN_THREADS / 32 ? red[lane] : 0.f), 1e-8f) / 127.f;
        if (threadIdx.x == 0)
            s[blockIdx.x] = sc;
        for (int i = threadIdx.x; i < I; i += RN_THREADS)
            q[off + i] = q8(av[i], sc);
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

template <int BM, bool GELU, bool GROUPED, bool QOUT = false>
static void run_i8(const int8_t *a, const float *sa, const int8_t *b, const float *sb, const bf16 *bi, bf16 *y, int M,
                   int N, int K, cudaStream_t st, int8_t *yq = nullptr, float *ysT = nullptr) {
    constexpr size_t sm = (size_t)I8_STAGES * ((BM + BN) * 64 + (GROUPED ? BM * 4 : 0));
    auto k = gemm_i8_kernel<BM, I8_STAGES, GELU, GROUPED, QOUT>;
    static bool cfg = false;
    if (!cfg) {
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
        cfg = true;
    }
    k<<<dim3((N + BN - 1) / BN, (M + BM - 1) / BM), NT, sm, st>>>(a, sa, b, sb, bi, y, M, N, K, yq, ysT);
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

// grouped W8A8 whose output is the next W8A8 GEMM's input: yq int8 [M, N] + ysT [N/64][M] (N % 128 == 0)
void launch_gemm_i8_q(const void *A, const float *saT, const void *B, const float *sb, const void *bias, void *Yq,
                      float *YsT, int M, int N, int K, bool gelu, cudaStream_t st) {
    const int8_t *a = (const int8_t *)A, *b = (const int8_t *)B;
    const bf16 *bi = (const bf16 *)bias;
    int8_t *yq = (int8_t *)Yq;
    if (pick_bm(M, N) == 128) {
        if (gelu) run_i8<128, true, true, true>(a, saT, b, sb, bi, nullptr, M, N, K, st, yq, YsT);
        else run_i8<128, false, true, true>(a, saT, b, sb, bi, nullptr, M, N, K, st, yq, YsT);
    } else {
        if (gelu) run_i8<64, true, true, true>(a, saT, b, sb, bi, nullptr, M, N, K, st, yq, YsT);
        else run_i8<64, false, true, true>(a, saT, b, sb, bi, nullptr, M, N, K, st, yq, YsT);
    }
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

void launch_add_layernorm_q(void *h, const void *y, const void *w, const void *b, void *q, float *s, int rows, int D,
                            float eps, bool grouped, cudaStream_t st) {
    const int threads = 256, rpb = threads / 32, blocks = (rows + rpb - 1) / rpb;
    if (grouped)
        add_layernorm_q_kernel<true><<<blocks, threads, 0, st>>>((bf16 *)h, (const bf16 *)y, (const bf16 *)w,
                                                                 (const bf16 *)b, (int8_t *)q, s, rows, D, eps);
    else
        add_layernorm_q_kernel<false><<<blocks, threads, 0, st>>>((bf16 *)h, (const bf16 *)y, (const bf16 *)w,
                                                                  (const bf16 *)b, (int8_t *)q, s, rows, D, eps);
}

void launch_add_rmsnorm_q(void *h, const void *y, const void *w, void *q, float *s, int rows, int H, float eps,
                          bool grouped, cudaStream_t st) {
    const size_t sm = (H + 32) * sizeof(float);
    bf16 *hh = (bf16 *)h;
    const bf16 *yy = (const bf16 *)y, *ww = (const bf16 *)w;
    int8_t *qq = (int8_t *)q;
    if (y && grouped) add_rmsnorm_q_kernel<true, true><<<rows, RN_THREADS, sm, st>>>(hh, yy, ww, qq, s, rows, H, eps);
    else if (y) add_rmsnorm_q_kernel<true, false><<<rows, RN_THREADS, sm, st>>>(hh, yy, ww, qq, s, rows, H, eps);
    else if (grouped) add_rmsnorm_q_kernel<false, true><<<rows, RN_THREADS, sm, st>>>(hh, yy, ww, qq, s, rows, H, eps);
    else add_rmsnorm_q_kernel<false, false><<<rows, RN_THREADS, sm, st>>>(hh, yy, ww, qq, s, rows, H, eps);
}

void launch_silu_mul_q(const void *g, void *q, float *s, int rows, int I, bool grouped, cudaStream_t st) {
    const size_t sm = (I + 32) * sizeof(float);
    static bool cfg = false;
    if (!cfg) {  // I = 8192: 32 KB of shared memory, more for wider models
        cudaFuncSetAttribute(silu_mul_q_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        cudaFuncSetAttribute(silu_mul_q_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        cfg = true;
    }
    if (grouped) silu_mul_q_kernel<true><<<rows, RN_THREADS, sm, st>>>((const bf16 *)g, (int8_t *)q, s, rows, I);
    else silu_mul_q_kernel<false><<<rows, RN_THREADS, sm, st>>>((const bf16 *)g, (int8_t *)q, s, rows, I);
}

void launch_quant_rows(const void *x, void *q, float *s, int rows, int K, cudaStream_t st) {
    const int threads = 256, rpb = threads / 32;
    quant_rows_kernel<<<(rows + rpb - 1) / rpb, threads, 0, st>>>((const bf16 *)x, (int8_t *)q, s, rows, K);
}

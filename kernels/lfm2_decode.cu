// O2-O4: fused CUDA kernels for one LFM2 decode step (batch B, one token per sequence), BF16 in/out,
// FP32 accumulation. Each kernel replaces a chain of small PyTorch ops in transformers' modeling_lfm2.py and
// keeps its rounding points (BF16 after every op HF does in BF16), so results match the eager model closely.
// Host wrappers + bindings: lfm2_decode.cpp (kept out of this file: nvcc 13 cannot parse some ATen headers).
// All kernels launch on PyTorch's current stream and read the decode position from device memory, so the
// whole step stays capturable in one CUDA graph (src/lfm2opt/runtime.py, fused=True).
//
//   add_rmsnorm      h += y (optional); x = w * rmsnorm(h)                 [Lfm2RMSNorm + residual add]
//   shortconv_step   y = C * conv3(shift(state, B*x))                       [Lfm2ShortConv decode path]
//   qk_norm_rope_kv  q,k: per-head RMSNorm + RoPE; k,v -> cache[pos]       [Lfm2Attention before SDPA]
//   decode_attention softmax(q K^T * scale) V over cache[0..pos], GQA      [SDPA, q_len = 1]
//   decode_attention_split  same, flash-decoding (split over positions + combine) for long contexts
//   silu_mul         silu(a) * b on the merged w1|w3 projection            [Lfm2MLP]
//   gemv             batch-1 y = W x, W in BF16 or INT8 (+ per-row scale)    [nn.Linear, O6]
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;
#define B2F(x) __bfloat162float(x)
#define F2B(x) __float2bfloat16(x)
__device__ __forceinline__ float rbf(float x) { return B2F(F2B(x)); }  // round to BF16 and back

// Programmatic Dependent Launch (sm_90+): kernels are launched with programmatic stream serialization, so the next
// kernel can start while this one drains. Contract: before pdl_wait() a kernel may only read immutable data
// (weights) and must not write anything; pdl_wait() returns once the previous grid has completed and its writes are
// visible. pdl_trigger() lets the next kernel launch early. Both are no-ops for launches without the attribute.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;"); }
static bool g_pdl = true;
void set_pdl(bool on) { g_pdl = on; }
template <typename... KArgs, typename... Args>
static void launch_k(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = st;
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = g_pdl ? 1 : 0;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
template <int NWARPS>
__device__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) red[w] = v;
    __syncthreads();
    float t = (l < NWARPS) ? red[l] : 0.f;
    t = warp_sum(t);
    __syncthreads();
    return t;
}


// ------------------------------------------------------------------------------------------------ add_rmsnorm
// One block per row. HF: h = bf16(h + y); var = mean(h_f32^2); n = bf16(h * rsqrt(var + eps)); x = bf16(w * n)
constexpr int NORM_THREADS = 256;
template <bool ADD>
__global__ void add_rmsnorm_kernel(bf16* __restrict__ h, const bf16* __restrict__ y, const bf16* __restrict__ w,
                                   bf16* __restrict__ x, int H, float eps) {
    pdl_trigger();
    pdl_wait();
    extern __shared__ float smem[];  // H floats + reduction scratch
    float* hv = smem;
    float* red = smem + H;
    const size_t off = (size_t)blockIdx.x * H;
    float ss = 0.f;
    for (int i = threadIdx.x; i < H; i += NORM_THREADS) {
        float v = B2F(h[off + i]);
        if (ADD) {
            v = rbf(v + B2F(y[off + i]));
            h[off + i] = F2B(v);
        }
        hv[i] = v;
        ss += v * v;
    }
    ss = block_sum<NORM_THREADS / 32>(ss, red);
    const float rs = rsqrtf(ss / H + eps);
    for (int i = threadIdx.x; i < H; i += NORM_THREADS) x[off + i] = F2B(B2F(w[i]) * rbf(hv[i] * rs));
}

void launch_add_rmsnorm(void* h, const void* y, const void* w, void* x, int rows, int H, float eps, cudaStream_t st) {
    const size_t sm = (H + 32) * sizeof(float);
    if (y) launch_k(add_rmsnorm_kernel<true>, rows, NORM_THREADS, sm, st, (bf16*)h, (const bf16*)y, (const bf16*)w, (bf16*)x, H, eps);
    else launch_k(add_rmsnorm_kernel<false>, rows, NORM_THREADS, sm, st, (bf16*)h, nullptr, (const bf16*)w, (bf16*)x, H, eps);
}

// ------------------------------------------------------------------------------------------------ shortconv_step
// bcx = in_proj(x) laid out [B, 3H] = (B | C | x). HF decode: bx = bf16(B*x); window = (s1, s2, bx) -> state;
// conv = bf16(sum_k w[k] * window[k] (+bias)); y = bf16(C * conv)
__global__ void shortconv_step_kernel(const bf16* __restrict__ bcx, bf16* __restrict__ state, const bf16* __restrict__ w,
                                      const bf16* __restrict__ bias, bf16* __restrict__ y, int H) {
    pdl_trigger();
    pdl_wait();
    constexpr int L = 3;
    const int b = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    const bf16* row = bcx + (size_t)b * 3 * H;
    const float bx = rbf(B2F(row[i]) * B2F(row[2 * H + i]));
    bf16* st = state + ((size_t)b * H + i) * L;
    const float s1 = B2F(st[1]), s2 = B2F(st[2]);
    st[0] = st[1]; st[1] = st[2]; st[2] = F2B(bx);
    float acc = B2F(w[i * L + 0]) * s1 + B2F(w[i * L + 1]) * s2 + B2F(w[i * L + 2]) * bx;
    if (bias) acc += B2F(bias[i]);
    y[(size_t)b * H + i] = F2B(B2F(row[H + i]) * rbf(acc));
}

void launch_shortconv_step(const void* bcx, void* state, const void* w, const void* bias, void* y, int B, int H, cudaStream_t st) {
    dim3 grid((H + 255) / 256, B);
    launch_k(shortconv_step_kernel, grid, 256, 0, st, (const bf16*)bcx, (bf16*)state, (const bf16*)w, (const bf16*)bias, (bf16*)y, H);
}

// ------------------------------------------------------------------------------------------------ qk_norm_rope_kv
// qkv = merged q|k|v projection [B, (NQ + 2*NKV) * D]. One block of D threads per (b, head).
// q/k heads: n = bf16(x * rsqrt(mean(x^2) + eps)); x = bf16(w * n); RoPE as HF in BF16:
//   cos/sin = bf16(cos/sin(pos * inv_freq[d % (D/2)]) * scaling); out = bf16(bf16(x*cos) + bf16(rot(x)*sin))
// q -> q_out[b, head]; k, v -> cache[b, kv_head, pos]
template <int D>
__global__ void qk_norm_rope_kv_kernel(const bf16* __restrict__ qkv, const bf16* __restrict__ qw, const bf16* __restrict__ kw,
                                       const float* __restrict__ inv_freq, float scaling, const int64_t* __restrict__ pos_p,
                                       bf16* __restrict__ q_out, bf16* __restrict__ kc, bf16* __restrict__ vc,
                                       int NQ, int NKV, int max_len, float eps) {
    pdl_trigger();
    pdl_wait();
    __shared__ float xs[D];
    __shared__ float red[D / 32];
    const int b = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const int64_t pos = *pos_p;
    const int NH = NQ + 2 * NKV;
    float x = B2F(qkv[((size_t)b * NH + head) * D + d]);
    if (head >= NQ + NKV) {  // value head: straight into the cache
        const int vh = head - NQ - NKV;
        vc[(((size_t)b * NKV + vh) * max_len + pos) * D + d] = F2B(x);
        return;
    }
    const bool is_q = head < NQ;
    const float ss = block_sum<D / 32>(x * x, red);
    x = rbf(B2F((is_q ? qw : kw)[d]) * rbf(x * rsqrtf(ss / D + eps)));
    xs[d] = x;
    __syncthreads();
    const int half = D / 2;
    const float rot = d < half ? -xs[d + half] : xs[d - half];
    const float f = (float)pos * inv_freq[d % half];
    const float c = rbf(cosf(f) * scaling), s = rbf(sinf(f) * scaling);
    const bf16 out = F2B(rbf(x * c) + rbf(rot * s));
    if (is_q) q_out[((size_t)b * NQ + head) * D + d] = out;
    else kc[(((size_t)b * NKV + (head - NQ)) * max_len + pos) * D + d] = out;
}

void launch_qk_norm_rope_kv(const void* qkv, const void* qw, const void* kw, const float* inv_freq, float scaling,
                            const int64_t* pos, void* q_out, void* kc, void* vc, int B, int NQ, int NKV, int max_len,
                            float eps, cudaStream_t st) {
    dim3 grid(B, NQ + 2 * NKV);
    launch_k(qk_norm_rope_kv_kernel<64>, grid, 64, 0, st, (const bf16*)qkv, (const bf16*)qw, (const bf16*)kw, inv_freq, scaling,
        pos, (bf16*)q_out, (bf16*)kc, (bf16*)vc, NQ, NKV, max_len, eps);
}

// ------------------------------------------------------------------------------------------------ decode_attention
// One block per (b, query head); 8 warps split the positions 0..pos (t = warp, warp+8, ...), each keeps an
// online softmax (running max m, sum l, accumulator over its 2 dims per lane); warps are merged at the end.
constexpr int ATT_WARPS = 8;
__global__ void decode_attention_kernel(const bf16* __restrict__ q, const bf16* __restrict__ kc, const bf16* __restrict__ vc,
                                        const int64_t* __restrict__ pos_p, bf16* __restrict__ out,
                                        int NQ, int NKV, int max_len, float scale) {
    pdl_trigger();
    pdl_wait();
    constexpr int D = 64;
    __shared__ float sm_m[ATT_WARPS], sm_l[ATT_WARPS], sm_acc[ATT_WARPS][D];
    const int b = blockIdx.x, h = blockIdx.y, kvh = h / (NQ / NKV);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int len = (int)(*pos_p) + 1;
    const __nv_bfloat162* qp = reinterpret_cast<const __nv_bfloat162*>(q + ((size_t)b * NQ + h) * D);
    const float2 qf = __bfloat1622float2(qp[lane]);
    const float q0 = qf.x * scale, q1 = qf.y * scale;
    const size_t base = ((size_t)b * NKV + kvh) * max_len * D;
    const __nv_bfloat162* K = reinterpret_cast<const __nv_bfloat162*>(kc + base);
    const __nv_bfloat162* V = reinterpret_cast<const __nv_bfloat162*>(vc + base);
    float m = -INFINITY, l = 0.f, a0 = 0.f, a1 = 0.f;
    for (int t = warp; t < len; t += ATT_WARPS) {
        const float2 k = __bfloat1622float2(K[(size_t)t * (D / 2) + lane]);
        const float s = warp_sum(q0 * k.x + q1 * k.y);
        const float mn = fmaxf(m, s);
        const float corr = __expf(m - mn), p = __expf(s - mn);
        const float2 v = __bfloat1622float2(V[(size_t)t * (D / 2) + lane]);
        l = l * corr + p;
        a0 = a0 * corr + p * v.x;
        a1 = a1 * corr + p * v.y;
        m = mn;
    }
    if (lane == 0) { sm_m[warp] = m; sm_l[warp] = l; }
    sm_acc[warp][2 * lane] = a0;
    sm_acc[warp][2 * lane + 1] = a1;
    __syncthreads();
    if (warp == 0) {
        float M = -INFINITY;
        for (int w = 0; w < ATT_WARPS; ++w) M = fmaxf(M, sm_m[w]);
        float L = 0.f, o0 = 0.f, o1 = 0.f;
        for (int w = 0; w < ATT_WARPS; ++w) {
            if (sm_l[w] == 0.f) continue;  // warp saw no positions
            const float c = __expf(sm_m[w] - M);
            L += sm_l[w] * c;
            o0 += sm_acc[w][2 * lane] * c;
            o1 += sm_acc[w][2 * lane + 1] * c;
        }
        __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(out + ((size_t)b * NQ + h) * D);
        op[lane] = __floats2bfloat162_rn(o0 / L, o1 / L);
    }
}

void launch_decode_attention(const void* q, const void* kc, const void* vc, const int64_t* pos, void* out,
                             int B, int NQ, int NKV, int max_len, float scale, cudaStream_t st) {
    dim3 grid(B, NQ);
    launch_k(decode_attention_kernel, grid, ATT_WARPS * 32, 0, st, (const bf16*)q, (const bf16*)kc, (const bf16*)vc, pos, (bf16*)out,
                                                             NQ, NKV, max_len, scale);
}

// ------------------------------------------------------------------------------------------------ decode_attention_split
// Flash-decoding for long contexts. Grid (B*NKV, S): block (b, kv head, split) covers positions
// [s*chunk, (s+1)*chunk) ∩ [0, pos]; its 4 warps are the 4 query heads that share this KV head (K/V rows are
// read once from DRAM, then hit L1 for the other warps). Inside a warp, 4 lane groups of 8 take 4 positions per
// iteration; each lane holds 8 of the 64 dims (one 16-byte load of K and V). Partial (max, sum, acc[64]) per
// (b, q head, split) go to ws; attn_combine merges the S partials.
constexpr int SPLIT_WARPS = 4;  // = query heads per KV head (32 / 8)
__global__ void decode_attention_split_kernel(const bf16* __restrict__ q, const bf16* __restrict__ kc, const bf16* __restrict__ vc,
                                              const int64_t* __restrict__ pos_p, float* __restrict__ ws,
                                              int NQ, int NKV, int max_len, int S, float scale) {
    pdl_trigger();
    pdl_wait();
    constexpr int D = 64;
    const int bk = blockIdx.x, s = blockIdx.y;
    const int b = bk / NKV, kvh = bk % NKV;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = kvh * (NQ / NKV) + warp;
    const int grp = lane >> 3, sub = lane & 7;            // position slot in this iteration, 8-dim slice
    const int len = (int)(*pos_p) + 1;
    const int chunk = (max_len + S - 1) / S;
    const int t0 = s * chunk, t1 = min(len, t0 + chunk);
    float* out = ws + (((size_t)b * NQ + h) * S + s) * (D + 2);
    if (t0 >= t1) {                                        // empty split
        if (lane == 0) { out[0] = -INFINITY; out[1] = 0.f; }
        return;
    }
    float qv[8];
    {
        const uint4 raw = reinterpret_cast<const uint4*>(q + ((size_t)b * NQ + h) * D)[sub];
        const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) { const float2 f = __bfloat1622float2(p2[j]); qv[2 * j] = f.x * scale; qv[2 * j + 1] = f.y * scale; }
    }
    const size_t base = ((size_t)b * NKV + kvh) * max_len * D;
    const uint4* K = reinterpret_cast<const uint4*>(kc + base);
    const uint4* V = reinterpret_cast<const uint4*>(vc + base);
    float m = -INFINITY, l = 0.f, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int it = t0; it < t1; it += 4) {                  // same trip count for all lanes (shuffles need the full warp)
        const int t = it + grp;
        const bool valid = t < t1;
        float dot = 0.f;
        uint4 vraw = make_uint4(0, 0, 0, 0);
        if (valid) {
            const uint4 kraw = K[(size_t)t * (D / 8) + sub];
            vraw = V[(size_t)t * (D / 8) + sub];
            const __nv_bfloat162* k2 = reinterpret_cast<const __nv_bfloat162*>(&kraw);
#pragma unroll
            for (int j = 0; j < 4; ++j) { const float2 f = __bfloat1622float2(k2[j]); dot += qv[2 * j] * f.x + qv[2 * j + 1] * f.y; }
        }
        dot += __shfl_xor_sync(0xffffffff, dot, 4);
        dot += __shfl_xor_sync(0xffffffff, dot, 2);
        dot += __shfl_xor_sync(0xffffffff, dot, 1);
        if (valid) {
            const float mn = fmaxf(m, dot), corr = __expf(m - mn), p = __expf(dot - mn);
            l = l * corr + p;
            const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&vraw);
#pragma unroll
            for (int j = 0; j < 4; ++j) { const float2 f = __bfloat1622float2(v2[j]); acc[2 * j] = acc[2 * j] * corr + p * f.x; acc[2 * j + 1] = acc[2 * j + 1] * corr + p * f.y; }
            m = mn;
        }
    }
    // merge the 4 lane groups (same sub = same dims): xor 8 and 16
#pragma unroll
    for (int o = 8; o <= 16; o <<= 1) {
        const float m2 = __shfl_xor_sync(0xffffffff, m, o), l2 = __shfl_xor_sync(0xffffffff, l, o);
        const float mn = fmaxf(m, m2);
        const float c1 = (m == -INFINITY) ? 0.f : __expf(m - mn), c2 = (m2 == -INFINITY) ? 0.f : __expf(m2 - mn);
        l = l * c1 + l2 * c2;
#pragma unroll
        for (int j = 0; j < 8; ++j) { const float a2 = __shfl_xor_sync(0xffffffff, acc[j], o); acc[j] = acc[j] * c1 + a2 * c2; }
        m = mn;
    }
    if (grp == 0) {
#pragma unroll
        for (int j = 0; j < 8; ++j) out[2 + sub * 8 + j] = acc[j];
        if (sub == 0) { out[0] = m; out[1] = l; }
    }
}

// One block of 64 threads per (b, q head): out = sum_s acc_s e^(m_s - M) / sum_s l_s e^(m_s - M)
__global__ void attn_combine_kernel(const float* __restrict__ ws, bf16* __restrict__ out, int S) {
    pdl_trigger();
    pdl_wait();
    constexpr int D = 64;
    const int bh = blockIdx.x, d = threadIdx.x;
    const float* w = ws + (size_t)bh * S * (D + 2);
    float M = -INFINITY;
    for (int s = 0; s < S; ++s) M = fmaxf(M, w[s * (D + 2)]);
    float L = 0.f, o = 0.f;
    for (int s = 0; s < S; ++s) {
        const float ls = w[s * (D + 2) + 1];
        if (ls == 0.f) continue;
        const float c = __expf(w[s * (D + 2)] - M);
        L += ls * c;
        o += w[s * (D + 2) + 2 + d] * c;
    }
    out[(size_t)bh * D + d] = F2B(o / L);
}

void launch_decode_attention_split(const void* q, const void* kc, const void* vc, const int64_t* pos, void* out, float* ws,
                                   int B, int NQ, int NKV, int max_len, int S, float scale, cudaStream_t st) {
    dim3 grid(B * NKV, S);
    launch_k(decode_attention_split_kernel, grid, SPLIT_WARPS * 32, 0, st, (const bf16*)q, (const bf16*)kc, (const bf16*)vc, pos, ws,
                                                                     NQ, NKV, max_len, S, scale);
    launch_k(attn_combine_kernel, B * NQ, 64, 0, st, ws, (bf16*)out, S);
}

// ------------------------------------------------------------------------------------------------ gemv (O6)
// y[n] = scale[n] * sum_k W[n, k] * x[k] for batch-1 decode. W is BF16 or INT8 (weight-only, symmetric
// per-output-row scale), row-major [N, K]. One warp per output row (8 rows per block); x lives in shared
// memory as FP32; each lane streams 16-byte chunks of its row (8 BF16 or 16 INT8 weights) with no-L1-allocate
// loads, UNROLL chunks in flight per lane to keep enough bytes outstanding to saturate DRAM.
constexpr int GEMV_WARPS = 8;
__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
template <typename WT> struct WTraits;
template <> struct WTraits<bf16> {
    static constexpr int PER = 8;  // weights per 16-byte chunk
    __device__ static float dot(const uint4& c, const float* xs, int stride) {
        const __nv_bfloat162* w = reinterpret_cast<const __nv_bfloat162*>(&c);
        float a = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) { const float2 f = __bfloat1622float2(w[j]); a += f.x * xs[(2 * j) * stride] + f.y * xs[(2 * j + 1) * stride]; }
        return a;
    }
};
template <> struct WTraits<int8_t> {
    static constexpr int PER = 16;
    __device__ static float dot(const uint4& c, const float* xs, int stride) {
        const int8_t* w = reinterpret_cast<const int8_t*>(&c);
        float a = 0.f;
#pragma unroll
        for (int j = 0; j < 16; ++j) a += (float)w[j] * xs[j * stride];
        return a;
    }
};
template <typename WT, int ROWS, int UNROLL>
__global__ void gemv_kernel(const WT* __restrict__ W, const float* __restrict__ scale, const bf16* __restrict__ x,
                            bf16* __restrict__ y, int N, int K) {
    extern __shared__ float xs[];
    pdl_trigger();
    constexpr int PER = WTraits<WT>::PER;
    const int chunks = K / PER;                             // K is a multiple of 32 * PER (checked on the host)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * GEMV_WARPS + warp) * ROWS;  // this warp's ROWS consecutive output rows
    const bool active = n0 < N;
    const uint4* rows[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) rows[r] = reinterpret_cast<const uint4*>(W + (size_t)min(n0 + r, N - 1) * K);
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
    int c = lane;
    uint4 buf[ROWS][UNROLL];
    bool more = active && c + 32 * (UNROLL - 1) < chunks;
    if (more) {  // weights do not depend on the previous kernel: start streaming them before the dependency wait
#pragma unroll
        for (int u = 0; u < UNROLL; ++u)
#pragma unroll
            for (int r = 0; r < ROWS; ++r) buf[r][u] = ld_stream(rows[r] + c + 32 * u);
    }
    pdl_wait();
    // natural layout (a transposed, bank-conflict-free layout measured ~5 % slower here: profiling/bench_gemv.py)
    for (int k = threadIdx.x; k < K; k += blockDim.x) xs[k] = B2F(x[k]);
    __syncthreads();
    if (!active) return;
    while (more) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const float* xp = xs + (c + 32 * u) * PER;      // x slice loaded once, reused by all ROWS rows
#pragma unroll
            for (int r = 0; r < ROWS; ++r) acc[r] += WTraits<WT>::dot(buf[r][u], xp, 1);
        }
        c += 32 * UNROLL;
        more = c + 32 * (UNROLL - 1) < chunks;
        if (more) {
#pragma unroll
            for (int u = 0; u < UNROLL; ++u)
#pragma unroll
                for (int r = 0; r < ROWS; ++r) buf[r][u] = ld_stream(rows[r] + c + 32 * u);
        }
    }
    for (; c < chunks; c += 32) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[r] += WTraits<WT>::dot(ld_stream(rows[r] + c), xs + c * PER, 1);
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float a = warp_sum(acc[r]);
        if (lane == 0 && n0 + r < N) y[n0 + r] = F2B(scale ? a * scale[n0 + r] : a);
    }
}

// Variant without the shared-memory stage: each lane loads its own x slice (PER BF16 values = 16 or 32 bytes) with
// cached loads right when it needs it; x is tiny and shared by all warps, so it is served by L1/L2. Removes the
// block-wide x -> shared prologue + __syncthreads that dominates the small matrices.
template <int PER>
__device__ __forceinline__ void load_x(const bf16* __restrict__ x, int chunk, float* xv) {
    const uint4* p = reinterpret_cast<const uint4*>(x + (size_t)chunk * PER);
#pragma unroll
    for (int q = 0; q < PER / 8; ++q) {
        const uint4 raw = __ldg(p + q);
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) { const float2 f = __bfloat1622float2(h[j]); xv[8 * q + 2 * j] = f.x; xv[8 * q + 2 * j + 1] = f.y; }
    }
}
template <typename WT, int ROWS, int UNROLL>
__global__ void gemv_direct_kernel(const WT* __restrict__ W, const float* __restrict__ scale, const bf16* __restrict__ x,
                                   bf16* __restrict__ y, int N, int K) {
    pdl_trigger();
    constexpr int PER = WTraits<WT>::PER;
    const int chunks = K / PER;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * GEMV_WARPS + warp) * ROWS;
    if (n0 >= N) return;                                    // no block-level sync in this kernel: early exit is fine
    const uint4* rows[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) rows[r] = reinterpret_cast<const uint4*>(W + (size_t)min(n0 + r, N - 1) * K);
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
    int c = lane;
    uint4 buf[ROWS][UNROLL];
    bool more = c + 32 * (UNROLL - 1) < chunks;
    if (more) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u)
#pragma unroll
            for (int r = 0; r < ROWS; ++r) buf[r][u] = ld_stream(rows[r] + c + 32 * u);
    }
    pdl_wait();
    while (more) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            float xv[PER];
            load_x<PER>(x, c + 32 * u, xv);
#pragma unroll
            for (int r = 0; r < ROWS; ++r) acc[r] += WTraits<WT>::dot(buf[r][u], xv, 1);
        }
        c += 32 * UNROLL;
        more = c + 32 * (UNROLL - 1) < chunks;
        if (more) {
#pragma unroll
            for (int u = 0; u < UNROLL; ++u)
#pragma unroll
                for (int r = 0; r < ROWS; ++r) buf[r][u] = ld_stream(rows[r] + c + 32 * u);
        }
    }
    for (; c < chunks; c += 32) {
        float xv[PER];
        load_x<PER>(x, c, xv);
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[r] += WTraits<WT>::dot(ld_stream(rows[r] + c), xv, 1);
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float a = warp_sum(acc[r]);
        if (lane == 0 && n0 + r < N) y[n0 + r] = F2B(scale ? a * scale[n0 + r] : a);
    }
}
static int g_gemv_impl = 1;  // 1: direct x loads (default: 1.1-1.23 TB/s on every decode shape), 0: x staged in shared memory
void set_gemv_impl(int impl) { g_gemv_impl = impl; }


// rows per warp (0 = auto): more rows reuse each x slice and keep more loads in flight
template <typename WT>
static void gemv_dispatch(const WT* W, const float* scale, const bf16* x, bf16* y, int N, int K, int rows, cudaStream_t st) {
    // measured (profiling/bench_gemv.py): direct loads -> 1 row/warp with 4 chunks in flight is best on every shape;
    // staged-x kernel -> 2 rows/warp, 4 for lm_head
    if (rows <= 0) rows = g_gemv_impl == 1 ? 1 : (N >= 32768 ? 4 : 2);
    const int per_block = GEMV_WARPS * rows;
    const dim3 grid((N + per_block - 1) / per_block);
    if (g_gemv_impl == 1) {
        if (rows == 4) launch_k(gemv_direct_kernel<WT, 4, 2>, grid, GEMV_WARPS * 32, 0, st, W, scale, x, y, N, K);
        else if (rows == 2) launch_k(gemv_direct_kernel<WT, 2, 2>, grid, GEMV_WARPS * 32, 0, st, W, scale, x, y, N, K);
        else launch_k(gemv_direct_kernel<WT, 1, 4>, grid, GEMV_WARPS * 32, 0, st, W, scale, x, y, N, K);
        return;
    }
    const size_t sm = (size_t)K * sizeof(float);
    if (rows == 4) launch_k(gemv_kernel<WT, 4, 2>, grid, GEMV_WARPS * 32, sm, st, W, scale, x, y, N, K);
    else if (rows == 2) launch_k(gemv_kernel<WT, 2, 2>, grid, GEMV_WARPS * 32, sm, st, W, scale, x, y, N, K);
    else launch_k(gemv_kernel<WT, 1, 4>, grid, GEMV_WARPS * 32, sm, st, W, scale, x, y, N, K);
}

void launch_gemv(const void* W, bool int8, const float* scale, const void* x, void* y, int N, int K, int rows, cudaStream_t st) {
    if (int8) gemv_dispatch<int8_t>((const int8_t*)W, scale, (const bf16*)x, (bf16*)y, N, K, rows, st);
    else gemv_dispatch<bf16>((const bf16*)W, scale, (const bf16*)x, (bf16*)y, N, K, rows, st);
}

// ------------------------------------------------------------------------------------------------ gemv_w4 (O6b)
// INT4 weight-only GEMV, symmetric scale per group of `group` (multiple of 32, default 128) along K. Wp [N, K/2] bytes: element 2j in the low
// nibble, 2j+1 in the high nibble, stored as q + 8 (q in [-8, 7]); S [N, K/128] fp32 scales. Each lane streams
// 16-byte chunks (32 weights); a group of 128 = 4 chunks, so one scale per chunk.
template <int ROWS>
__global__ void gemv_w4_kernel(const uint8_t* __restrict__ Wp, const float* __restrict__ S, const bf16* __restrict__ x,
                               bf16* __restrict__ y, int N, int K, int group) {
    pdl_trigger();
    const int chunks = K / 32, groups = K / group, chunks_per_group = group / 32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * GEMV_WARPS + warp) * ROWS;
    if (n0 >= N) return;                                    // no block-level sync: early exit is fine
    const uint4* rows[ROWS];
    const float* srow[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int n = min(n0 + r, N - 1);
        rows[r] = reinterpret_cast<const uint4*>(Wp + (size_t)n * (K / 2));
        srow[r] = S + (size_t)n * groups;
    }
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
    int c = lane;
    uint4 buf[ROWS];
    float sc[ROWS];
    bool more = c < chunks;
    if (more) {  // prefetch the first weights + scales before the dependency wait (they are constants)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) { buf[r] = ld_stream(rows[r] + c); sc[r] = __ldg(srow[r] + c / chunks_per_group); }
    }
    pdl_wait();
    while (more) {
        float xv[32];
        load_x<32>(x, c, xv);                                // this chunk's 32 inputs, direct from L1/L2
        uint4 nbuf[ROWS];
        float nsc[ROWS];
        const int cn = c + 32;
        const bool next = cn < chunks;
        if (next) {                                          // keep the next chunk in flight while computing this one
#pragma unroll
            for (int r = 0; r < ROWS; ++r) { nbuf[r] = ld_stream(rows[r] + cn); nsc[r] = __ldg(srow[r] + cn / chunks_per_group); }
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const uint32_t* w = reinterpret_cast<const uint32_t*>(&buf[r]);
            float a = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t v = w[j];
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const uint32_t byte = (v >> (8 * b)) & 0xFF;
                    a += (float)((int)(byte & 0xF) - 8) * xv[8 * j + 2 * b] + (float)((int)(byte >> 4) - 8) * xv[8 * j + 2 * b + 1];
                }
            }
            acc[r] += a * sc[r];
        }
        if (next) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) { buf[r] = nbuf[r]; sc[r] = nsc[r]; }
        }
        c = cn;
        more = next;
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float a = warp_sum(acc[r]);
        if (lane == 0 && n0 + r < N) y[n0 + r] = F2B(a);
    }
}

void launch_gemv_w4(const void* Wp, const float* S, const void* x, void* y, int N, int K, int group, int rows, cudaStream_t st) {
    if (rows <= 0) rows = N >= 32768 ? 2 : 1;
    const size_t sm = 0;  // x is read directly (no shared-memory stage)
    const dim3 grid((N + GEMV_WARPS * rows - 1) / (GEMV_WARPS * rows));
    if (rows == 4) launch_k(gemv_w4_kernel<4>, grid, GEMV_WARPS * 32, sm, st, (const uint8_t*)Wp, S, (const bf16*)x, (bf16*)y, N, K, group);
    else if (rows == 2) launch_k(gemv_w4_kernel<2>, grid, GEMV_WARPS * 32, sm, st, (const uint8_t*)Wp, S, (const bf16*)x, (bf16*)y, N, K, group);
    else launch_k(gemv_w4_kernel<1>, grid, GEMV_WARPS * 32, sm, st, (const uint8_t*)Wp, S, (const bf16*)x, (bf16*)y, N, K, group);
}

// ------------------------------------------------------------------------------------------------ silu_mul
// g = merged w1|w3 projection [B, 2I]: y = bf16(bf16(silu(a)) * b)
__global__ void silu_mul_kernel(const bf16* __restrict__ g, bf16* __restrict__ y, int I) {
    pdl_trigger();
    pdl_wait();
    const int b = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= I) return;
    const float a = B2F(g[(size_t)b * 2 * I + i]), u = B2F(g[(size_t)b * 2 * I + I + i]);
    y[(size_t)b * I + i] = F2B(rbf(a / (1.f + __expf(-a))) * u);
}

void launch_silu_mul(const void* g, void* y, int B, int I, cudaStream_t st) {
    dim3 grid((I + 255) / 256, B);
    launch_k(silu_mul_kernel, grid, 256, 0, st, (const bf16*)g, (bf16*)y, I);
}

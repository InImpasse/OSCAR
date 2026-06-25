#pragma once

#include "common.cuh"
#include "q2_0.cuh"
#include "convert.cuh"
#include "vecdotq.cuh"

#include <cstdint>

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
#ifndef GGML_CUDA_OSCAR2_KQ_SCALE
#define GGML_CUDA_OSCAR2_KQ_SCALE 1.0f
#endif

// log(2) = 0.6931, by adding this to the KQ maximum used for the softmax the numerical range representable
//     by the VKQ accumulators is effectively being shifted up by a factor of 2.
// This reduces issues with numerical overflow but also causes larger values to be flushed to zero.
// However, as the output from FlashAttention will usually be used as an input for a matrix multiplication this should be negligible.
// Still, the value range should be shifted as much as necessary but as little as possible.
// The macro on the following line shifts it by a factor of 2**3=8, as was needed to fix https://github.com/ggml-org/llama.cpp/issues/18606 .
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

typedef void (* fattn_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33);

typedef float (*vec_dot_KQ_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) half2 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            ggml_cuda_mad(sum,                tmp[k_KQ_1] , ((const half2  *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            ggml_cuda_mad(sum, __half22float2(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_bf16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const nv_bfloat162 * K_bf16 = (const nv_bfloat162 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) nv_bfloat162 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_bf16 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            // FIXME replace macros in vector FA kernel with templating and use FP32 for BF16
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]));
#else
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

static __device__ __forceinline__ float turbo2_centroid_fattn(const uint8_t q) {
    switch (q & 0x03) {
        case 0:  return -0.133462f;
        case 1:  return -0.039994f;
        case 2:  return  0.039994f;
        default: return  0.133462f;
    }
}

static __device__ __forceinline__ float turbo3_centroid_fattn(const uint8_t q) {
    switch (q & 0x07) {
        case 0:  return -0.190685f;
        case 1:  return -0.117832f;
        case 2:  return -0.065717f;
        case 3:  return -0.021460f;
        case 4:  return  0.021460f;
        case 5:  return  0.065717f;
        case 6:  return  0.117832f;
        default: return  0.190685f;
    }
}

static __device__ __forceinline__ uint8_t turbo3_index_fattn(const block_turbo3_0 & b, const int j) {
    const uint8_t low = (b.qs[j / 4] >> (2*(j & 3))) & 0x03;
    const uint8_t hi  = (b.signs[j / 8] >> (j & 7)) & 0x01;
    return low | (hi << 2);
}

static __device__ __forceinline__ float oscar2_v_centroid_fattn_fast(const int q) {
    const int qi = q & 0x07;
    const int mi = (qi & 0x04) ? (7 - qi) : qi;
    const float mag =
        mi == 0 ? 1.3500f :
        mi == 1 ? 0.8600f :
        mi == 2 ? 0.5200f : 0.1850f;
    return (qi & 0x04) ? mag : -mag;
}

static __device__ __forceinline__ half2 oscar2_v_centroid_pair_h2(const int idx0, const int idx1) {
    return make_half2(oscar2_v_centroid_fattn_fast(idx0), oscar2_v_centroid_fattn_fast(idx1));
}

template<bool is_k>
static __device__ __forceinline__ half2 oscar2_dequantize_pair_h2(
        const block_oscar2_kv & b, const uint8_t qs, const uint8_t rs, const int shift0) {
    const int idx0 = ((qs >> shift0) & 0x03) | (((rs >> (shift0/2 + 0)) & 0x01) << 2);
    const int idx1 = ((qs >> (shift0 + 2)) & 0x03) | (((rs >> (shift0/2 + 1)) & 0x01) << 2);
    const float d = __half2float(b.d);
    const float m = __half2float(b.m);
    if constexpr (is_k) {
        return make_half2(m + d * oscar2_centroid_3bit_cuda(idx0), m + d * oscar2_centroid_3bit_cuda(idx1));
    } else {
        return make_half2(m + d * oscar2_v_centroid_fattn_fast(idx0), m + d * oscar2_v_centroid_fattn_fast(idx1));
    }
}

static __device__ __forceinline__ void oscar2_dequantize_4_v_h2(
        const block_oscar2_kv & b, const uint8_t qs, const uint8_t rs, half2 & v01, half2 & v23) {
    const int idx0 = ((qs >> 0) & 0x03) | (((rs >> 0) & 0x01) << 2);
    const int idx1 = ((qs >> 2) & 0x03) | (((rs >> 1) & 0x01) << 2);
    const int idx2 = ((qs >> 4) & 0x03) | (((rs >> 2) & 0x01) << 2);
    const int idx3 = ((qs >> 6) & 0x03) | (((rs >> 3) & 0x01) << 2);
    const half2 m = __half2half2(b.m);
    const half2 d = __half2half2(b.d);
    v01 = __hfma2(d, oscar2_v_centroid_pair_h2(idx0, idx1), m);
    v23 = __hfma2(d, oscar2_v_centroid_pair_h2(idx2, idx3), m);
}

static __device__ __forceinline__ void oscar2_dequantize_4_v_f2(
        const block_oscar2_kv & b, const uint8_t qs, const uint8_t rs, float2 & v01, float2 & v23) {
    const int idx0 = ((qs >> 0) & 0x03) | (((rs >> 0) & 0x01) << 2);
    const int idx1 = ((qs >> 2) & 0x03) | (((rs >> 1) & 0x01) << 2);
    const int idx2 = ((qs >> 4) & 0x03) | (((rs >> 2) & 0x01) << 2);
    const int idx3 = ((qs >> 6) & 0x03) | (((rs >> 3) & 0x01) << 2);
    const float m = __half2float(b.m);
    const float d = __half2float(b.d);
    v01 = make_float2(m + d * oscar2_v_centroid_fattn_fast(idx0), m + d * oscar2_v_centroid_fattn_fast(idx1));
    v23 = make_float2(m + d * oscar2_v_centroid_fattn_fast(idx2), m + d * oscar2_v_centroid_fattn_fast(idx3));
}


template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo2_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    static_assert(D % QK_TURBO2 == 0, "bad D for turbo2 KQ");
    const block_turbo2_0 * K_t2 = (const block_turbo2_0 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads) {
        const int k2 = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int i0 = 2*k2 + 0;
        const int i1 = 2*k2 + 1;
        const block_turbo2_0 & b0 = K_t2[i0 / QK_TURBO2];
        const block_turbo2_0 & b1 = K_t2[i1 / QK_TURBO2];
        const uint8_t qbyte0 = b0.qs[(i0 % QK_TURBO2) / 4];
        const uint8_t qbyte1 = b1.qs[(i1 % QK_TURBO2) / 4];
        const uint8_t q0 = (qbyte0 >> (2*((i0 % QK_TURBO2) & 3))) & 0x03;
        const uint8_t q1 = (qbyte1 >> (2*((i1 % QK_TURBO2) & 3))) & 0x03;
        const float2 k = make_float2(__half2float(b0.norm)*turbo2_centroid_fattn(q0),
                                     __half2float(b1.norm)*turbo2_centroid_fattn(q1));

#ifdef V_DOT2_F32_F16_AVAILABLE
        ggml_cuda_mad(sum, k, __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads]));
#else
        ggml_cuda_mad(sum, k, ((const float2 *) Q_v)[k_KQ_0/nthreads]);
#endif // V_DOT2_F32_F16_AVAILABLE
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo3_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    static_assert(D % QK_TURBO3 == 0, "bad D for turbo3 KQ");
    const block_turbo3_0 * K_t3 = (const block_turbo3_0 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads) {
        const int k2 = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int i0 = 2*k2 + 0;
        const int i1 = 2*k2 + 1;
        const block_turbo3_0 & b0 = K_t3[i0 / QK_TURBO3];
        const block_turbo3_0 & b1 = K_t3[i1 / QK_TURBO3];
        const uint8_t q0 = turbo3_index_fattn(b0, i0 % QK_TURBO3);
        const uint8_t q1 = turbo3_index_fattn(b1, i1 % QK_TURBO3);
        const float2 k = make_float2(__half2float(b0.norm)*turbo3_centroid_fattn(q0),
                                     __half2float(b1.norm)*turbo3_centroid_fattn(q1));

#ifdef V_DOT2_F32_F16_AVAILABLE
        ggml_cuda_mad(sum, k, __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads]));
#else
        ggml_cuda_mad(sum, k, ((const float2 *) Q_v)[k_KQ_0/nthreads]);
#endif // V_DOT2_F32_F16_AVAILABLE
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q4_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x - (8/QI8_1)*Q_ds.y);
    }

    return sum;
}

// packed q2 byte -> sign/high int32 for dp4a KQ dot (256 entries each)
static const __device__ uint32_t Q2_0_FATTN_SIGN_LUT[256] = {
    0xffffffff, 0xffffffff, 0xffffff01, 0xffffff01, 0xffffffff, 0xffffffff, 0xffffff01, 0xffffff01,
    0xffff01ff, 0xffff01ff, 0xffff0101, 0xffff0101, 0xffff01ff, 0xffff01ff, 0xffff0101, 0xffff0101,
    0xffffffff, 0xffffffff, 0xffffff01, 0xffffff01, 0xffffffff, 0xffffffff, 0xffffff01, 0xffffff01,
    0xffff01ff, 0xffff01ff, 0xffff0101, 0xffff0101, 0xffff01ff, 0xffff01ff, 0xffff0101, 0xffff0101,
    0xff01ffff, 0xff01ffff, 0xff01ff01, 0xff01ff01, 0xff01ffff, 0xff01ffff, 0xff01ff01, 0xff01ff01,
    0xff0101ff, 0xff0101ff, 0xff010101, 0xff010101, 0xff0101ff, 0xff0101ff, 0xff010101, 0xff010101,
    0xff01ffff, 0xff01ffff, 0xff01ff01, 0xff01ff01, 0xff01ffff, 0xff01ffff, 0xff01ff01, 0xff01ff01,
    0xff0101ff, 0xff0101ff, 0xff010101, 0xff010101, 0xff0101ff, 0xff0101ff, 0xff010101, 0xff010101,
    0xffffffff, 0xffffffff, 0xffffff01, 0xffffff01, 0xffffffff, 0xffffffff, 0xffffff01, 0xffffff01,
    0xffff01ff, 0xffff01ff, 0xffff0101, 0xffff0101, 0xffff01ff, 0xffff01ff, 0xffff0101, 0xffff0101,
    0xffffffff, 0xffffffff, 0xffffff01, 0xffffff01, 0xffffffff, 0xffffffff, 0xffffff01, 0xffffff01,
    0xffff01ff, 0xffff01ff, 0xffff0101, 0xffff0101, 0xffff01ff, 0xffff01ff, 0xffff0101, 0xffff0101,
    0xff01ffff, 0xff01ffff, 0xff01ff01, 0xff01ff01, 0xff01ffff, 0xff01ffff, 0xff01ff01, 0xff01ff01,
    0xff0101ff, 0xff0101ff, 0xff010101, 0xff010101, 0xff0101ff, 0xff0101ff, 0xff010101, 0xff010101,
    0xff01ffff, 0xff01ffff, 0xff01ff01, 0xff01ff01, 0xff01ffff, 0xff01ffff, 0xff01ff01, 0xff01ff01,
    0xff0101ff, 0xff0101ff, 0xff010101, 0xff010101, 0xff0101ff, 0xff0101ff, 0xff010101, 0xff010101,
    0x01ffffff, 0x01ffffff, 0x01ffff01, 0x01ffff01, 0x01ffffff, 0x01ffffff, 0x01ffff01, 0x01ffff01,
    0x01ff01ff, 0x01ff01ff, 0x01ff0101, 0x01ff0101, 0x01ff01ff, 0x01ff01ff, 0x01ff0101, 0x01ff0101,
    0x01ffffff, 0x01ffffff, 0x01ffff01, 0x01ffff01, 0x01ffffff, 0x01ffffff, 0x01ffff01, 0x01ffff01,
    0x01ff01ff, 0x01ff01ff, 0x01ff0101, 0x01ff0101, 0x01ff01ff, 0x01ff01ff, 0x01ff0101, 0x01ff0101,
    0x0101ffff, 0x0101ffff, 0x0101ff01, 0x0101ff01, 0x0101ffff, 0x0101ffff, 0x0101ff01, 0x0101ff01,
    0x010101ff, 0x010101ff, 0x01010101, 0x01010101, 0x010101ff, 0x010101ff, 0x01010101, 0x01010101,
    0x0101ffff, 0x0101ffff, 0x0101ff01, 0x0101ff01, 0x0101ffff, 0x0101ffff, 0x0101ff01, 0x0101ff01,
    0x010101ff, 0x010101ff, 0x01010101, 0x01010101, 0x010101ff, 0x010101ff, 0x01010101, 0x01010101,
    0x01ffffff, 0x01ffffff, 0x01ffff01, 0x01ffff01, 0x01ffffff, 0x01ffffff, 0x01ffff01, 0x01ffff01,
    0x01ff01ff, 0x01ff01ff, 0x01ff0101, 0x01ff0101, 0x01ff01ff, 0x01ff01ff, 0x01ff0101, 0x01ff0101,
    0x01ffffff, 0x01ffffff, 0x01ffff01, 0x01ffff01, 0x01ffffff, 0x01ffffff, 0x01ffff01, 0x01ffff01,
    0x01ff01ff, 0x01ff01ff, 0x01ff0101, 0x01ff0101, 0x01ff01ff, 0x01ff01ff, 0x01ff0101, 0x01ff0101,
    0x0101ffff, 0x0101ffff, 0x0101ff01, 0x0101ff01, 0x0101ffff, 0x0101ffff, 0x0101ff01, 0x0101ff01,
    0x010101ff, 0x010101ff, 0x01010101, 0x01010101, 0x010101ff, 0x010101ff, 0x01010101, 0x01010101,
    0x0101ffff, 0x0101ffff, 0x0101ff01, 0x0101ff01, 0x0101ffff, 0x0101ffff, 0x0101ff01, 0x0101ff01,
    0x010101ff, 0x010101ff, 0x01010101, 0x01010101, 0x010101ff, 0x010101ff, 0x01010101, 0x01010101,
};

static const __device__ uint32_t Q2_0_FATTN_HIGH_LUT[256] = {
    0xffffffff, 0xffffff00, 0xffffff00, 0xffffff01, 0xffff00ff, 0xffff0000, 0xffff0000, 0xffff0001,
    0xffff00ff, 0xffff0000, 0xffff0000, 0xffff0001, 0xffff01ff, 0xffff0100, 0xffff0100, 0xffff0101,
    0xff00ffff, 0xff00ff00, 0xff00ff00, 0xff00ff01, 0xff0000ff, 0xff000000, 0xff000000, 0xff000001,
    0xff0000ff, 0xff000000, 0xff000000, 0xff000001, 0xff0001ff, 0xff000100, 0xff000100, 0xff000101,
    0xff00ffff, 0xff00ff00, 0xff00ff00, 0xff00ff01, 0xff0000ff, 0xff000000, 0xff000000, 0xff000001,
    0xff0000ff, 0xff000000, 0xff000000, 0xff000001, 0xff0001ff, 0xff000100, 0xff000100, 0xff000101,
    0xff01ffff, 0xff01ff00, 0xff01ff00, 0xff01ff01, 0xff0100ff, 0xff010000, 0xff010000, 0xff010001,
    0xff0100ff, 0xff010000, 0xff010000, 0xff010001, 0xff0101ff, 0xff010100, 0xff010100, 0xff010101,
    0x00ffffff, 0x00ffff00, 0x00ffff00, 0x00ffff01, 0x00ff00ff, 0x00ff0000, 0x00ff0000, 0x00ff0001,
    0x00ff00ff, 0x00ff0000, 0x00ff0000, 0x00ff0001, 0x00ff01ff, 0x00ff0100, 0x00ff0100, 0x00ff0101,
    0x0000ffff, 0x0000ff00, 0x0000ff00, 0x0000ff01, 0x000000ff, 0x00000000, 0x00000000, 0x00000001,
    0x000000ff, 0x00000000, 0x00000000, 0x00000001, 0x000001ff, 0x00000100, 0x00000100, 0x00000101,
    0x0000ffff, 0x0000ff00, 0x0000ff00, 0x0000ff01, 0x000000ff, 0x00000000, 0x00000000, 0x00000001,
    0x000000ff, 0x00000000, 0x00000000, 0x00000001, 0x000001ff, 0x00000100, 0x00000100, 0x00000101,
    0x0001ffff, 0x0001ff00, 0x0001ff00, 0x0001ff01, 0x000100ff, 0x00010000, 0x00010000, 0x00010001,
    0x000100ff, 0x00010000, 0x00010000, 0x00010001, 0x000101ff, 0x00010100, 0x00010100, 0x00010101,
    0x00ffffff, 0x00ffff00, 0x00ffff00, 0x00ffff01, 0x00ff00ff, 0x00ff0000, 0x00ff0000, 0x00ff0001,
    0x00ff00ff, 0x00ff0000, 0x00ff0000, 0x00ff0001, 0x00ff01ff, 0x00ff0100, 0x00ff0100, 0x00ff0101,
    0x0000ffff, 0x0000ff00, 0x0000ff00, 0x0000ff01, 0x000000ff, 0x00000000, 0x00000000, 0x00000001,
    0x000000ff, 0x00000000, 0x00000000, 0x00000001, 0x000001ff, 0x00000100, 0x00000100, 0x00000101,
    0x0000ffff, 0x0000ff00, 0x0000ff00, 0x0000ff01, 0x000000ff, 0x00000000, 0x00000000, 0x00000001,
    0x000000ff, 0x00000000, 0x00000000, 0x00000001, 0x000001ff, 0x00000100, 0x00000100, 0x00000101,
    0x0001ffff, 0x0001ff00, 0x0001ff00, 0x0001ff01, 0x000100ff, 0x00010000, 0x00010000, 0x00010001,
    0x000100ff, 0x00010000, 0x00010000, 0x00010001, 0x000101ff, 0x00010100, 0x00010100, 0x00010101,
    0x01ffffff, 0x01ffff00, 0x01ffff00, 0x01ffff01, 0x01ff00ff, 0x01ff0000, 0x01ff0000, 0x01ff0001,
    0x01ff00ff, 0x01ff0000, 0x01ff0000, 0x01ff0001, 0x01ff01ff, 0x01ff0100, 0x01ff0100, 0x01ff0101,
    0x0100ffff, 0x0100ff00, 0x0100ff00, 0x0100ff01, 0x010000ff, 0x01000000, 0x01000000, 0x01000001,
    0x010000ff, 0x01000000, 0x01000000, 0x01000001, 0x010001ff, 0x01000100, 0x01000100, 0x01000101,
    0x0100ffff, 0x0100ff00, 0x0100ff00, 0x0100ff01, 0x010000ff, 0x01000000, 0x01000000, 0x01000001,
    0x010000ff, 0x01000000, 0x01000000, 0x01000001, 0x010001ff, 0x01000100, 0x01000100, 0x01000101,
    0x0101ffff, 0x0101ff00, 0x0101ff00, 0x0101ff01, 0x010100ff, 0x01010000, 0x01010000, 0x01010001,
    0x010100ff, 0x01010000, 0x01010000, 0x01010001, 0x010101ff, 0x01010100, 0x01010100, 0x01010101,
};

static constexpr int Q2_0_FATTN_ONES_I32 = 0x01010101;

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q2_0_chunk(
    const block_q2_0 * K_q2_0, const int k_KQ, const int u, const float Q_d) {

    const int ib   = k_KQ / QI8_1;
    const int byte = k_KQ % QI8_1;

    const uint8_t packed = K_q2_0[ib].qs[byte];
    const int sign_i = (int) Q2_0_FATTN_SIGN_LUT[packed];
    const int high_i = (int) Q2_0_FATTN_HIGH_LUT[packed];

    const int sum_sign = ggml_cuda_dp4a(sign_i, u, 0);
    const int sum_high = ggml_cuda_dp4a(high_i, u, 0);
    const int usum     = ggml_cuda_dp4a(Q2_0_FATTN_ONES_I32, u, 0);

    const float d = __half2float(K_q2_0[ib].d);
    const float m = __half2float(K_q2_0[ib].m);

    return Q_d * (d*(Q2_0_LM_C2*sum_sign + (Q2_0_LM_C3 - Q2_0_LM_C2)*sum_high) + m*usum);
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q2_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q2_0 * K_q2_0 = (const block_q2_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += vec_dot_fattn_vec_KQ_q2_0_chunk<D, nthreads>(K_q2_0, k_KQ, Q_q8[k_KQ_0/nthreads], Q_ds.x);
    }

    return sum;
}

template<bool is_k, int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_oscar2_chunk(
    const block_oscar2_kv * K_oscar2, const int k_KQ, const int u, const float2 Q_ds) {

    const int ib   = k_KQ / (QK_OSCAR2_KV / 4);
    const int byte = k_KQ % (QK_OSCAR2_KV / 4);

    const uint8_t packed = K_oscar2[ib].qs[byte];
    const uint8_t high  = K_oscar2[ib].rs[(4*byte) / 8] >> ((4*byte) & 7);
    const float d = __half2float(K_oscar2[ib].d);
    const float m = __half2float(K_oscar2[ib].m);
    int v = 0;
    uint8_t * vq = (uint8_t *) &v;
    vq[0] = ((packed >> 0) & 0x03) | (((high >> 0) & 0x01) << 2);
    vq[1] = ((packed >> 2) & 0x03) | (((high >> 1) & 0x01) << 2);
    vq[2] = ((packed >> 4) & 0x03) | (((high >> 2) & 0x01) << 2);
    vq[3] = ((packed >> 6) & 0x03) | (((high >> 3) & 0x01) << 2);

    const int8_t * uq = (const int8_t *) &u;
    float sumi = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sumi += oscar2_centroid_3bit_cuda(vq[i]) * (float) uq[i];
    }
    const int usum = ggml_cuda_dp4a(0x01010101, u, 0);
    return d * Q_ds.x * sumi + m*Q_ds.x*usum;
}

template<bool is_k, int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_oscar2(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_oscar2_kv * K_oscar2 = (const block_oscar2_kv *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += vec_dot_fattn_vec_KQ_oscar2_chunk<is_k, D, nthreads>(K_oscar2, k_KQ, Q_q8[k_KQ_0/nthreads], Q_ds);
    }

    return GGML_CUDA_OSCAR2_KQ_SCALE * sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_1 * K_q4_1 = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q4_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q4_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_0 * K_q5_0 = (const block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_0;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q5_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x - (16/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_1 * K_q5_1 = (const block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_1;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q5_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q5_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q8_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        int v;
        ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        const float2 * Q_ds = (const float2 *) Q_ds_v;
        const float Q_d = Q_ds[k_KQ_0/nthreads].x;

        sum += vec_dot_q8_0_q8_1_impl<float, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    return sum;
}

template <typename Tds, int ni>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float * __restrict__ x, const float scale, int * __restrict__ yq32, void * __restrict__ yds) {

    float vals[sizeof(int)] = {0.0f};
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
        vals[l] = (ni == WARP_SIZE || threadIdx.x < ni) ? scale * x[4*threadIdx.x + l] : 0.0f;
    }

    float amax = fabsf(vals[0]);
    float sum  = vals[0];
#pragma unroll
    for (int l = 1; l < int(sizeof(int)); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        sum +=             __shfl_xor_sync(0xFFFFFFFF, sum,  mask, 32);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0 && (ni == WARP_SIZE || threadIdx.x < ni)) {
        if (std::is_same<Tds, half2>::value) {
            ((half2  *) yds)[threadIdx.x/QI8_1] =  make_half2(d, sum);
        } else {
            ((float2 *) yds)[threadIdx.x/QI8_1] = make_float2(d, sum);
        }
    }
}

typedef void (*dequantize_V_t)(const void *, void *, const int64_t);

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_f16(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    if constexpr (std::is_same_v<T, half>) {
        ggml_cuda_memcpy_1<ne*sizeof(half)>(dst, (const half *) vx + i0);
    } else if constexpr (std::is_same_v<T, float>) {
        static_assert(ne % 2 == 0, "bad ne");
        __align__(16) half2 tmp[ne/2];
        ggml_cuda_memcpy_1<ne*sizeof(half)>(tmp, (const half *) vx + i0);
        float2 * dst_f2 = (float2 *) dst;
#pragma unroll
        for (int l = 0; l < ne/2; ++l) {
            dst_f2[l] = __half22float2(tmp[l]);
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_bf16(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    static_assert(std::is_same_v<T, float>, "BF16 V dequantization only supports float output");
    static_assert(ne % 2 == 0, "bad ne");
    __align__(16) nv_bfloat162 tmp[ne/2];
    ggml_cuda_memcpy_1<ne*sizeof(nv_bfloat16)>(tmp, (const nv_bfloat16 *) vx + i0);
    float2 * dst_f2 = (float2 *) dst;
#pragma unroll
    for (int l = 0; l < ne/2; ++l) {
        dst_f2[l] = ggml_cuda_cast<float2>(tmp[l]);
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_0(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i0          /  QK4_0;
    const int     iqs   =  i0          % (QK4_0/2);
    const int     shift = (i0 % QK4_0) / (QK4_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = __vsubss4(q, 0x08080808);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q2_0(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q2_0 * x = (const block_q2_0 *) vx;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        static_assert(ne % 2 == 0, "bad ne");
#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            float vals[2];
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                const int64_t i = i0 + l0 + l;
                vals[l] = q2_0_dequantize_scalar_cuda(x, i);
            }
            ((half2 *) dst)[l0/2] = make_half2(vals[0], vals[1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            const int64_t i = i0 + l;
            ((float *) dst)[l] = q2_0_dequantize_scalar_cuda(x, i);
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <bool is_k, typename T, int ne>
static __device__ __forceinline__ void dequantize_V_oscar2(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_oscar2_kv * x = (const block_oscar2_kv *) vx;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        static_assert(ne % 2 == 0, "bad ne");
        if constexpr (ne == 4 || ne == 8) {
            const int64_t ib = i0 / QK_OSCAR2_KV;
            const int j0 = i0 % QK_OSCAR2_KV;
            const block_oscar2_kv & b = x[ib];
#pragma unroll
            for (int l0 = 0; l0 < ne; l0 += 4) {
                const int j = j0 + l0;
                const uint8_t qs = b.qs[j / 4];
                const uint8_t rs = b.rs[j / 8] >> (j & 7);
                ((half2 *) dst)[l0/2 + 0] = oscar2_dequantize_pair_h2<is_k>(b, qs, rs, 0);
                ((half2 *) dst)[l0/2 + 1] = oscar2_dequantize_pair_h2<is_k>(b, qs, rs, 4);
            }
        } else {
#pragma unroll
            for (int l0 = 0; l0 < ne; l0 += 2) {
                float vals[2];
#pragma unroll
                for (int l = 0; l < 2; ++l) {
                    const int64_t i = i0 + l0 + l;
                    vals[l] = oscar2_dequantize_scalar_cuda<is_k>(x, i);
                }
                ((half2 *) dst)[l0/2] = make_half2(vals[0], vals[1]);
            }
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (ne == 4 || ne == 8) {
            const int64_t ib = i0 / QK_OSCAR2_KV;
            const int j0 = i0 % QK_OSCAR2_KV;
            const block_oscar2_kv & b = x[ib];
            const float d = __half2float(b.d);
            const float m = __half2float(b.m);
#pragma unroll
            for (int l0 = 0; l0 < ne; l0 += 4) {
                const int j = j0 + l0;
                const uint8_t qs = b.qs[j / 4];
                const uint8_t rs = b.rs[j / 8] >> (j & 7);
                const int idx0 = ((qs >> 0) & 0x03) | (((rs >> 0) & 0x01) << 2);
                const int idx1 = ((qs >> 2) & 0x03) | (((rs >> 1) & 0x01) << 2);
                const int idx2 = ((qs >> 4) & 0x03) | (((rs >> 2) & 0x01) << 2);
                const int idx3 = ((qs >> 6) & 0x03) | (((rs >> 3) & 0x01) << 2);
                if constexpr (is_k) {
                    ((float *) dst)[l0 + 0] = m + d * oscar2_centroid_3bit_cuda(idx0);
                    ((float *) dst)[l0 + 1] = m + d * oscar2_centroid_3bit_cuda(idx1);
                    ((float *) dst)[l0 + 2] = m + d * oscar2_centroid_3bit_cuda(idx2);
                    ((float *) dst)[l0 + 3] = m + d * oscar2_centroid_3bit_cuda(idx3);
                } else {
                    ((float *) dst)[l0 + 0] = m + d * oscar2_v_centroid_fattn_fast(idx0);
                    ((float *) dst)[l0 + 1] = m + d * oscar2_v_centroid_fattn_fast(idx1);
                    ((float *) dst)[l0 + 2] = m + d * oscar2_v_centroid_fattn_fast(idx2);
                    ((float *) dst)[l0 + 3] = m + d * oscar2_v_centroid_fattn_fast(idx3);
                }
            }
        } else {
#pragma unroll
            for (int l = 0; l < ne; ++l) {
                const int64_t i = i0 + l;
                ((float *) dst)[l] = oscar2_dequantize_scalar_cuda<is_k>(x, i);
            }
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo2_0(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_turbo2_0 * x = (const block_turbo2_0 *) vx;

    const int64_t ib   = i0 / QK_TURBO2;
    const int     j0   = i0 % QK_TURBO2;
    const block_turbo2_0 & b = x[ib];
    const float   norm = __half2float(b.norm);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        static_assert(ne % 2 == 0, "bad ne");
        if constexpr (ne == 4) {
            const uint8_t qs_byte = b.qs[j0 / 4];
            const uint8_t idx0 = (qs_byte >> 0) & 3;
            const uint8_t idx1 = (qs_byte >> 2) & 3;
            const uint8_t idx2 = (qs_byte >> 4) & 3;
            const uint8_t idx3 = (qs_byte >> 6) & 3;
            ((half2 *) dst)[0] = make_half2(turbo2_centroid_fattn(idx0) * norm, turbo2_centroid_fattn(idx1) * norm);
            ((half2 *) dst)[1] = make_half2(turbo2_centroid_fattn(idx2) * norm, turbo2_centroid_fattn(idx3) * norm);
        } else {
#pragma unroll
            for (int l0 = 0; l0 < ne; l0 += 2) {
                float vals[2];
#pragma unroll
                for (int l = 0; l < 2; ++l) {
                    const int64_t i = i0 + l0 + l;
                    const block_turbo2_0 & bl = x[i / QK_TURBO2];
                    const uint8_t qbyte = bl.qs[(i % QK_TURBO2) / 4];
                    const uint8_t q = (qbyte >> (2*((i % QK_TURBO2) & 3))) & 0x03;
                    vals[l] = __half2float(bl.norm) * turbo2_centroid_fattn(q);
                }
                ((half2 *) dst)[l0/2] = make_half2(vals[0], vals[1]);
            }
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (ne == 4) {
            const uint8_t qs_byte = b.qs[j0 / 4];
            const uint8_t idx0 = (qs_byte >> 0) & 3;
            const uint8_t idx1 = (qs_byte >> 2) & 3;
            const uint8_t idx2 = (qs_byte >> 4) & 3;
            const uint8_t idx3 = (qs_byte >> 6) & 3;
            ((float2 *) dst)[0] = make_float2(turbo2_centroid_fattn(idx0) * norm, turbo2_centroid_fattn(idx1) * norm);
            ((float2 *) dst)[1] = make_float2(turbo2_centroid_fattn(idx2) * norm, turbo2_centroid_fattn(idx3) * norm);
        } else {
#pragma unroll
            for (int l = 0; l < ne; ++l) {
                const int64_t i = i0 + l;
                const block_turbo2_0 & bl = x[i / QK_TURBO2];
                const uint8_t qbyte = bl.qs[(i % QK_TURBO2) / 4];
                const uint8_t q = (qbyte >> (2*((i % QK_TURBO2) & 3))) & 0x03;
                ((float *) dst)[l] = __half2float(bl.norm) * turbo2_centroid_fattn(q);
            }
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo3_0(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx;

    const int64_t ib   = i0 / QK_TURBO3;
    const int     j0   = i0 % QK_TURBO3;
    const block_turbo3_0 & b = x[ib];
    const float   norm = __half2float(b.norm);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        static_assert(ne % 2 == 0, "bad ne");
        if constexpr (ne == 4) {
            if ((j0 % 4) == 0) {
                const uint8_t qbyte = b.qs[j0 / 4];
                const uint8_t sbyte = b.signs[j0 / 8];
                const int sshift = j0 % 8;
                const uint8_t idx0 = ((qbyte >> 0) & 3) | (((sbyte >> (sshift + 0)) & 1) << 2);
                const uint8_t idx1 = ((qbyte >> 2) & 3) | (((sbyte >> (sshift + 1)) & 1) << 2);
                const uint8_t idx2 = ((qbyte >> 4) & 3) | (((sbyte >> (sshift + 2)) & 1) << 2);
                const uint8_t idx3 = ((qbyte >> 6) & 3) | (((sbyte >> (sshift + 3)) & 1) << 2);
                ((half2 *) dst)[0] = make_half2(turbo3_centroid_fattn(idx0) * norm, turbo3_centroid_fattn(idx1) * norm);
                ((half2 *) dst)[1] = make_half2(turbo3_centroid_fattn(idx2) * norm, turbo3_centroid_fattn(idx3) * norm);
            } else {
#pragma unroll
                for (int l0 = 0; l0 < ne; l0 += 2) {
                    float vals[2];
#pragma unroll
                    for (int l = 0; l < 2; ++l) {
                        const int64_t i = i0 + l0 + l;
                        const block_turbo3_0 & bl = x[i / QK_TURBO3];
                        const uint8_t q = turbo3_index_fattn(bl, i % QK_TURBO3);
                        vals[l] = __half2float(bl.norm) * turbo3_centroid_fattn(q);
                    }
                    ((half2 *) dst)[l0/2] = make_half2(vals[0], vals[1]);
                }
            }
        } else {
#pragma unroll
            for (int l0 = 0; l0 < ne; l0 += 2) {
                float vals[2];
#pragma unroll
                for (int l = 0; l < 2; ++l) {
                    const int64_t i = i0 + l0 + l;
                    const block_turbo3_0 & bl = x[i / QK_TURBO3];
                    const uint8_t q = turbo3_index_fattn(bl, i % QK_TURBO3);
                    vals[l] = __half2float(bl.norm) * turbo3_centroid_fattn(q);
                }
                ((half2 *) dst)[l0/2] = make_half2(vals[0], vals[1]);
            }
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (ne == 4) {
            if ((j0 % 4) == 0) {
                const uint8_t qbyte = b.qs[j0 / 4];
                const uint8_t sbyte = b.signs[j0 / 8];
                const int sshift = j0 % 8;
                const uint8_t idx0 = ((qbyte >> 0) & 3) | (((sbyte >> (sshift + 0)) & 1) << 2);
                const uint8_t idx1 = ((qbyte >> 2) & 3) | (((sbyte >> (sshift + 1)) & 1) << 2);
                const uint8_t idx2 = ((qbyte >> 4) & 3) | (((sbyte >> (sshift + 2)) & 1) << 2);
                const uint8_t idx3 = ((qbyte >> 6) & 3) | (((sbyte >> (sshift + 3)) & 1) << 2);
                ((float2 *) dst)[0] = make_float2(turbo3_centroid_fattn(idx0) * norm, turbo3_centroid_fattn(idx1) * norm);
                ((float2 *) dst)[1] = make_float2(turbo3_centroid_fattn(idx2) * norm, turbo3_centroid_fattn(idx3) * norm);
            } else {
#pragma unroll
                for (int l = 0; l < ne; ++l) {
                    const int64_t i = i0 + l;
                    const block_turbo3_0 & bl = x[i / QK_TURBO3];
                    const uint8_t q = turbo3_index_fattn(bl, i % QK_TURBO3);
                    ((float *) dst)[l] = __half2float(bl.norm) * turbo3_centroid_fattn(q);
                }
            }
        } else {
#pragma unroll
            for (int l = 0; l < ne; ++l) {
                const int64_t i = i0 + l;
                const block_turbo3_0 & bl = x[i / QK_TURBO3];
                const uint8_t q = turbo3_index_fattn(bl, i % QK_TURBO3);
                ((float *) dst)[l] = __half2float(bl.norm) * turbo3_centroid_fattn(q);
            }
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_1(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int64_t ib    =  i0          /  QK4_1;
    const int     iqs   =  i0          % (QK4_1/2);
    const int     shift = (i0 % QK4_1) / (QK4_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_0(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int64_t ib    =  i0          /  QK5_0;
    const int     idq   =  i0          %  QK5_0;
    const int     iqs   =  i0          % (QK5_0/2);
    const int     shift = (i0 % QK5_0) / (QK5_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = __vsubss4(q, 0x10101010);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_1(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int64_t ib    =  i0          /  QK5_1;
    const int     idq   =  i0          %  QK5_1;
    const int     iqs   =  i0          % (QK5_1/2);
    const int     shift = (i0 % QK5_1) / (QK5_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q8_0(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i0 / QK8_0;
    const int     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_cuda_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same<T, half>::value) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same<T, float>::value) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * qs[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <ggml_type type_K, int D, int nthreads>
constexpr __device__ vec_dot_KQ_t get_vec_dot_KQ() {
    if constexpr (type_K == GGML_TYPE_F16) {
        return vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TURBO2_0) {
        return vec_dot_fattn_vec_KQ_turbo2_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TURBO3_0) {
        return vec_dot_fattn_vec_KQ_turbo3_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q2_0) {
        return vec_dot_fattn_vec_KQ_q2_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_OSCAR2_KV) {
        return vec_dot_fattn_vec_KQ_oscar2<true, D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_0) {
        return vec_dot_fattn_vec_KQ_q4_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_1) {
        return vec_dot_fattn_vec_KQ_q4_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_0) {
        return vec_dot_fattn_vec_KQ_q5_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_1) {
        return vec_dot_fattn_vec_KQ_q5_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q8_0) {
        return vec_dot_fattn_vec_KQ_q8_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_BF16) {
        return vec_dot_fattn_vec_KQ_bf16<D, nthreads>;
    } else {
        static_assert(type_K == -1, "bad type");
        return nullptr;
    }
}

template <ggml_type type_V, typename T, int ne>
constexpr __device__ dequantize_V_t get_dequantize_V() {
    if constexpr (type_V == GGML_TYPE_F16) {
        return dequantize_V_f16<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TURBO2_0) {
        return dequantize_V_turbo2_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TURBO3_0) {
        return dequantize_V_turbo3_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q2_0) {
        return dequantize_V_q2_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_OSCAR2_KV) {
        return dequantize_V_oscar2<false, T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_0) {
        return dequantize_V_q4_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_1) {
        return dequantize_V_q4_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_0) {
        return dequantize_V_q5_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_1) {
        return dequantize_V_q5_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q8_0) {
        return dequantize_V_q8_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_BF16) {
        return dequantize_V_bf16<float, ne>;
    } else {
        static_assert(type_V == -1, "bad type");
        return nullptr;
    }
}

template <int ncols1>
__launch_bounds__(FATTN_KQ_STRIDE/2, 1)
static __global__ void flash_attn_mask_to_KV_max(
        const half2 * __restrict__ mask, int * __restrict__ KV_max, const int ne30, const int s31, const int s33) {
    const int ne31     = gridDim.x;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;

    mask += sequence*s33 + jt*ncols1*s31;

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    ggml_cuda_pdl_sync();
    __syncthreads();

    int KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float2 tmp = __half22float2(mask[j*s31 + KV_max_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    if (threadIdx.x != 0) {
        return;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

template<int D>
static __device__ __forceinline__ float flash_attn_hp_fused_finish_row(
        const char * __restrict__ Q,
        const char * __restrict__ K_hp,
        const char * __restrict__ V_hp,
        const char * __restrict__ mask_hp,
        const float scale,
        const float logit_softcap,
        const int row,
        const int col,
        const int head,
        const int sequence,
        const int ne01, const int ne02,
        const int ne11_hp, const int ne12_hp,
        const int nb01, const int nb02, const int nb03,
        const int nb11_hp, const int nb12_hp, const int64_t nb13_hp,
        const int nb21_hp, const int nb22_hp, const int64_t nb23_hp,
        const int nb31_hp, const int64_t nb33_hp,
        float dst_val, float max_val, float rowsum) {
    if (!K_hp || !V_hp || !mask_hp) {
        return dst_val / rowsum;
    }

    if constexpr (D != 128) {
        return dst_val / rowsum;
    }
    const int tid = threadIdx.x;

    const int gqa_ratio_hp = ne02 / ne12_hp;
    const int hkv_hp = head / gqa_ratio_hp;

    const float q = scale * ((const float *) (Q + nb03*sequence + nb02*head + nb01*col))[tid];
    const char * K_hp_row = K_hp + nb13_hp*sequence + nb12_hp*hkv_hp;
    const char * V_hp_row = V_hp + nb23_hp*sequence + nb22_hp*hkv_hp;
    const char * mask_hp_seq = mask_hp + nb33_hp*sequence + nb31_hp*col;

    __shared__ float score_shared[D];
    for (int key = 0; key < ne11_hp; ++key) {
        const float mask_val = *(const float *) (mask_hp_seq + key*sizeof(float));
        if (mask_val < -1.0e30f) {
            continue;
        }

        const half * K_h = (const half *) (K_hp_row + key*nb11_hp);
        score_shared[tid] = q * __half2float(K_h[tid]);
        __syncthreads();

        for (int offset = D/2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                score_shared[tid] += score_shared[tid + offset];
            }
            __syncthreads();
        }

        float score = score_shared[0];
        if (logit_softcap != 0.0f) {
            score = logit_softcap*tanhf(score);
        }
        score += mask_val;

        const float max_new = fmaxf(max_val, score + FATTN_KQ_MAX_OFFSET);
        const float scale_old = expf(max_val - max_new);
        const float scale_hp = expf(score - max_new);
        const half * V_h = (const half *) (V_hp_row + key*nb21_hp);
        dst_val = dst_val*scale_old + scale_hp*__half2float(V_h[tid]);
        rowsum = rowsum*scale_old + scale_hp;
        max_val = max_new;
        __syncthreads();
    }

    GGML_UNUSED(row);
    return dst_val / rowsum;
}

template<int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_uniform(
        float * __restrict__ dst,
        float2 * __restrict__ dst_final_meta,
        const char * __restrict__ Q,
        const char * __restrict__ K_hp,
        const char * __restrict__ V_hp,
        const char * __restrict__ mask_hp,
        const float2 * __restrict__ dst_fixup,
        const float scale,
        const float logit_softcap,
        const int nb01, const int nb02, const int nb03,
        const int ne11_hp, const int ne12_hp,
        const int nb11_hp, const int nb12_hp, const int64_t nb13_hp,
        const int nb21_hp, const int nb22_hp, const int64_t nb23_hp,
        const int nb31_hp, const int64_t nb33_hp,
        const int ne01, const int ne02,
        const int ne12, const int nblocks_stream_k,
        const int gqa_ratio,
        const int blocks_per_tile,
        const uint3 fd_iter_j_z_ne12,
        const uint3 fd_iter_j_z,
        const uint3 fd_iter_j) {
    constexpr int ncols = ncols1*ncols2;
    ggml_cuda_pdl_lc();

    const int tile_idx = blockIdx.x; // One block per output tile.
    const int j        = blockIdx.y;
    const int c        = blockIdx.z;
    const int jc       = j*ncols2 + c;
    const int tid      = threadIdx.x;

    // nblocks_stream_k is a multiple of ntiles_dst (== gridDim.x), so each tile gets the same number of blocks.
    const int b_first = tile_idx * blocks_per_tile;
    const int b_last  = b_first + blocks_per_tile - 1;

    const float * dst_fixup_data = ((const float *) dst_fixup) + nblocks_stream_k*(2*2*ncols);

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(tile_idx, fd_iter_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y,    fd_iter_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y,    fd_iter_j);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm2.y;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    const int row = sequence*ne02*ne01 + jt*ne02*ncols1 + zt_Q + j*ne02 + c;
    dst += int64_t(row)*D + tid;

    ggml_cuda_pdl_sync();
    // Load the partial result that needs a fixup
    float dst_val = *dst;
    float max_val;
    float rowsum;
    {
        const float2 tmp = dst_fixup[b_last*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Combine with all previous blocks in this tile.
    for (int bidx = b_last - 1; bidx >= b_first; --bidx) {
        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(nblocks_stream_k + bidx)*ncols + jc];

        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;
    }

    // Write back final result:
    const int col = jt*ncols1 + j;
    const int head = zt_Q + c;
    *dst = flash_attn_hp_fused_finish_row<D>(
        Q, K_hp, V_hp, mask_hp, scale, logit_softcap, row, col, head, sequence,
        ne01, ne02, ne11_hp, ne12_hp, nb01, nb02, nb03,
        nb11_hp, nb12_hp, nb13_hp, nb21_hp, nb22_hp, nb23_hp, nb31_hp, nb33_hp,
        dst_val, max_val, rowsum);
    if (dst_final_meta && tid == 0) {
        dst_final_meta[row] = make_float2(max_val, rowsum);
    }
}

// General fixup kernel for the case where the number of blocks per tile is not uniform across tiles
// (blocks_num.x not a multiple of ntiles_dst)
template <int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_general(
        float * __restrict__ dst,
        float2 * __restrict__ dst_final_meta,
        const char * __restrict__ Q,
        const char * __restrict__ K_hp,
        const char * __restrict__ V_hp,
        const char * __restrict__ mask_hp,
        const float2 * __restrict__ dst_fixup,
        const float scale,
        const float logit_softcap,
        const int nb01, const int nb02, const int nb03,
        const int ne11_hp, const int ne12_hp,
        const int nb11_hp, const int nb12_hp, const int64_t nb13_hp,
        const int nb21_hp, const int nb22_hp, const int64_t nb23_hp,
        const int nb31_hp, const int64_t nb33_hp,
        const int ne01, const int ne02,
        const int gqa_ratio,
        const int total_work,
        const uint3 fd_iter_k_j_z_ne12,
        const uint3 fd_iter_k_j_z,
        const uint3 fd_iter_k_j,
        const uint3 fd_iter_k) {
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int kbc0      = int64_t(bidx0 + 0)*total_work / gridDim.x;
    const int kbc0_stop = int64_t(bidx0 + 1)*total_work / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, fd_iter_k) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, fd_iter_k) == fastdiv(kbc0_stop, fd_iter_k) && fastmodulo(kbc0_stop, fd_iter_k) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(kbc0, fd_iter_k_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y, fd_iter_k_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y, fd_iter_k_j);
    const uint2 dm3 = fast_div_modulo(dm2.y, fd_iter_k);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm3.x;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    const int row = sequence*ne02*ne01 + jt*ne02*ncols1 + zt_Q + j*ne02 + c;
    dst += int64_t(row)*D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    ggml_cuda_pdl_sync();
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    const int tile_kbc0 = fastdiv(kbc0, fd_iter_k);
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = int64_t(bidx)*total_work / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, fd_iter_k) == 0 || fastdiv(kbc, fd_iter_k) < tile_kbc0) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    const int col = jt*ncols1 + j;
    const int head = zt_Q + c;
    *dst = flash_attn_hp_fused_finish_row<D>(
        Q, K_hp, V_hp, mask_hp, scale, logit_softcap, row, col, head, sequence,
        ne01, ne02, ne11_hp, ne12_hp, nb01, nb02, nb03,
        nb11_hp, nb12_hp, nb13_hp, nb21_hp, nb22_hp, nb23_hp, nb31_hp, nb33_hp,
        dst_val, max_val, rowsum);
    if (dst_final_meta && tid == 0) {
        dst_final_meta[row] = make_float2(max_val, rowsum);
    }
}

template<int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_combine_results(
        const float  * __restrict__ VKQ_parts,
        const float2 * __restrict__ VKQ_meta,
        float * __restrict__ dst,
        float2 * __restrict__ dst_final_meta,
        const int parallel_blocks) {
    ggml_cuda_pdl_lc();
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;

    const int j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    ggml_cuda_pdl_sync();
    for (int i = tid; i < 2*parallel_blocks; i += D) {
        ((float *) meta)[i] = ((const float *)VKQ_meta) [i];
    }

    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
    for (int l = 0; l < parallel_blocks; ++l) {
        const float KQ_max_scale = expf(meta[l].x - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
    if (dst_final_meta && tid == 0) {
        dst_final_meta[j_dst_unrolled] = make_float2(kqmax, VKQ_denominator);
    }
}

template <int DV, int ncols1, int ncols2>
void launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const int warp_size = WARP_SIZE
) {
    constexpr int ncols = ncols1 * ncols2;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * final_meta = dst->src[8];
    const bool fused_hp_fixup = getenv("LLAMA_KV_HP_STAGED_FUSED_FIXUP") != nullptr;
    const ggml_tensor * K_hp = fused_hp_fixup ? dst->src[5] : nullptr;
    const ggml_tensor * V_hp = fused_hp_fixup ? dst->src[6] : nullptr;
    const ggml_tensor * mask_hp = fused_hp_fixup ? dst->src[7] : nullptr;

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    ggml_cuda_pool_alloc<half>   K_f16(pool);
    ggml_cuda_pool_alloc<half>   V_f16(pool);
    ggml_cuda_pool_alloc<int>    KV_max(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    const char * V_data = (const char *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        K_f16.alloc(ggml_nelements(K));
        if (ggml_is_contiguously_allocated(K)) {
            to_fp16_cuda_t to_fp16 = K->type == GGML_TYPE_OSCAR2_KV ?
                dequantize_row_oscar2_kv_f16_cuda<true> : ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16.ptr, ggml_nelements(K), main_stream);

            nb11 = nb11*bs*sizeof(half)/ts;
            nb12 = nb12*bs*sizeof(half)/ts;
            nb13 = nb13*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            const int64_t s01 = nb11 / ts;
            const int64_t s02 = nb12 / ts;
            const int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16.ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (char *) K_f16.ptr;
    }

    if (need_f16_V && V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } else {
            const size_t bs = ggml_blck_size(V->type);
            const size_t ts = ggml_type_size(V->type);

            V_f16.alloc(ggml_nelements(V));
            if (ggml_is_contiguously_allocated(V)) {
                to_fp16_cuda_t to_fp16 = V->type == GGML_TYPE_OSCAR2_KV ?
                    dequantize_row_oscar2_kv_f16_cuda<false> : ggml_get_to_fp16_cuda(V->type);
                to_fp16(V_data, V_f16.ptr, ggml_nelements(V), main_stream);
                V_data = (char *) V_f16.ptr;

                nb21 = nb21*bs*sizeof(half)/ts;
                nb22 = nb22*bs*sizeof(half)/ts;
                nb23 = nb23*bs*sizeof(half)/ts;
            } else {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
                const int64_t s01 = nb21 / ts;
                const int64_t s02 = nb22 / ts;
                const int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16.ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * sizeof(half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (char *) V_f16.ptr;
        }
    }

    const int ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    if (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        const int s31 = mask->nb[1] / sizeof(half2);
        const int s33 = mask->nb[3] / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);

        const int ne_KV_max = blocks_num_KV_max.x*blocks_num_KV_max.y;
        const int iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        flash_attn_mask_to_KV_max<ncols1><<<blocks_num_KV_max, block_dim_KV_max, 0, main_stream>>>
            ((const half2 *) mask->data, KV_max.ptr, iter_k, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));
    GGML_ASSERT(max_blocks_per_sm > 0);
    int parallel_blocks = max_blocks_per_sm;

    const int ntiles_KV = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by KV cache length.

    dim3 blocks_num;
    if (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const int max_blocks = max_blocks_per_sm*nsm;
        const int tiles_nwaves = (ntiles_dst + max_blocks - 1) / max_blocks;
        const int tiles_efficiency_percent = 100 * ntiles_dst / (max_blocks*tiles_nwaves);

        const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || amd_wmma_available(cc) || tiles_efficiency_percent < 75;

        blocks_num.x = ntiles_dst;
        blocks_num.y = 1;
        blocks_num.z = 1;

        if(use_stream_k) {
            const int nblocks_stream_k_raw = std::min(max_blocks, ntiles_KV*ntiles_dst);
            // Round down to a multiple of ntiles_dst so that each output tile gets the same number of blocks (avoids fixup).
            // Only do this if the occupancy loss from rounding is acceptable.
            const int nblocks_stream_k_rounded = (nblocks_stream_k_raw / ntiles_dst) * ntiles_dst;
            const int max_efficiency_loss_percent = 5;
            const int efficiency_loss_percent = nblocks_stream_k_rounded > 0
                ? 100 * (nblocks_stream_k_raw - nblocks_stream_k_rounded) / nblocks_stream_k_raw
                : 100;
            const int nblocks_stream_k = efficiency_loss_percent <= max_efficiency_loss_percent
                ? nblocks_stream_k_rounded
                : nblocks_stream_k_raw;

            blocks_num.x = nblocks_stream_k;
        }

        if (ntiles_dst % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_num.x) * ncols * (2 + DV/2)));
        }
    } else {
        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KV);

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_dst * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);

    float2 * dst_meta_direct = !stream_k && parallel_blocks == 1 && final_meta ? (float2 *) final_meta->data : dst_tmp_meta.ptr;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num, block_dim, nbytes_shared, main_stream);
    ggml_cuda_kernel_launch(fattn_kernel, launch_params,
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? ((const char *) mask->data) : nullptr,
        sinks ? ((const char *) sinks->data) : nullptr,
        KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_meta_direct,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13,
        nb21, nb22, nb23,
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0
    );
    CUDA_CHECK(cudaGetLastError());

    if (stream_k) {
        if ((int)blocks_num.x % ntiles_dst == 0 && (int)blocks_num.x > ntiles_dst) {
            // Optimized fixup: nblocks_stream_k is a multiple of ntiles_dst, launch one block per tile.
            const int nblocks_sk  = (int)blocks_num.x;
            const int bpt         = nblocks_sk / ntiles_dst;

            const uint3 fd0 = init_fastdiv_values(ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd1 = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
            const uint3 fd2 = init_fastdiv_values(ntiles_x);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {(unsigned)ntiles_dst, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, final_meta ? (float2 *) final_meta->data : nullptr,
                 (const char *) Q->data,
                 K_hp ? (const char *) K_hp->data : nullptr,
                 V_hp ? (const char *) V_hp->data : nullptr,
                 mask_hp ? (const char *) mask_hp->data : nullptr,
                 dst_tmp_meta.ptr, scale, logit_softcap,
                 Q->nb[1], Q->nb[2], Q->nb[3],
                 K_hp ? K_hp->ne[1] : 0, K_hp ? K_hp->ne[2] : 1,
                 K_hp ? K_hp->nb[1] : 0, K_hp ? K_hp->nb[2] : 0, K_hp ? K_hp->nb[3] : 0,
                 V_hp ? V_hp->nb[1] : 0, V_hp ? V_hp->nb[2] : 0, V_hp ? V_hp->nb[3] : 0,
                 mask_hp ? mask_hp->nb[1] : 0, mask_hp ? mask_hp->nb[3] : 0,
                 Q->ne[1], Q->ne[2], K->ne[2], nblocks_sk,
                 gqa_ratio, bpt, fd0, fd1, fd2);
        } else if (ntiles_dst % blocks_num.x != 0) {
            // General fixup for the cases where nblocks_stream_k < ntiles_dst.
            const int total_work = ntiles_KV * ntiles_dst;

            const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd_k_j_z      = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
            const uint3 fd_k_j        = init_fastdiv_values(ntiles_KV * ntiles_x);
            const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_general<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, final_meta ? (float2 *) final_meta->data : nullptr,
                 (const char *) Q->data,
                 K_hp ? (const char *) K_hp->data : nullptr,
                 V_hp ? (const char *) V_hp->data : nullptr,
                 mask_hp ? (const char *) mask_hp->data : nullptr,
                 dst_tmp_meta.ptr, scale, logit_softcap,
                 Q->nb[1], Q->nb[2], Q->nb[3],
                 K_hp ? K_hp->ne[1] : 0, K_hp ? K_hp->ne[2] : 1,
                 K_hp ? K_hp->nb[1] : 0, K_hp ? K_hp->nb[2] : 0, K_hp ? K_hp->nb[3] : 0,
                 V_hp ? V_hp->nb[1] : 0, V_hp ? V_hp->nb[2] : 0, V_hp ? V_hp->nb[3] : 0,
                 mask_hp ? mask_hp->nb[1] : 0, mask_hp ? mask_hp->nb[3] : 0,
                 Q->ne[1], Q->ne[2], gqa_ratio, total_work,
                 fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
        }
    } else if (parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream);
        ggml_cuda_kernel_launch(flash_attn_combine_results<DV>, launch_params,
            dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data,
            final_meta ? (float2 *) final_meta->data : nullptr, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}

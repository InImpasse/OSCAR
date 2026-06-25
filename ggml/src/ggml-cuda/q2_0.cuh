#pragma once

#include "common.cuh"

static __device__ __forceinline__ float q2_0_centroid_cuda(const int q) {
    switch (q & 0x03) {
        case 0:  return Q2_0_LM_C0;
        case 1:  return Q2_0_LM_C1;
        case 2:  return Q2_0_LM_C2;
        default: return Q2_0_LM_C3;
    }
}

static __device__ __forceinline__ float oscar2_k_centroid_cuda(const int q) {
    switch (q & 0x07) {
        case 0:  return OSCAR2_K_C0;
        case 1:  return OSCAR2_K_C1;
        case 2:  return OSCAR2_K_C2;
        case 3:  return OSCAR2_K_C3;
        case 4:  return OSCAR2_K_C4;
        case 5:  return OSCAR2_K_C5;
        case 6:  return OSCAR2_K_C6;
        default: return OSCAR2_K_C7;
    }
}

static __device__ __forceinline__ float oscar2_v_centroid_cuda(const int q) {
    switch (q & 0x03) {
        case 0:  return OSCAR2_V_C0;
        case 1:  return OSCAR2_V_C1;
        case 2:  return OSCAR2_V_C2;
        default: return OSCAR2_V_C3;
    }
}

static __device__ __forceinline__ uint8_t q2_0_quantize_lm_cuda(const float v, const float inv_sigma) {
    const float vs = v * inv_sigma;
    if (vs < Q2_0_LM_T0) return 0;
    if (vs < Q2_0_LM_T1) return 1;
    if (vs < Q2_0_LM_T2) return 2;
    return 3;
}

static __device__ __forceinline__ uint8_t q2_0_quantize_symmetric_cuda(
        const float v, const float inv_sigma, const float t0, const float t1, const float t2) {
    const float vs = v * inv_sigma;
    if (vs < t0) return 0;
    if (vs < t1) return 1;
    if (vs < t2) return 2;
    return 3;
}

static __device__ __forceinline__ uint8_t oscar2_k_quantize_cuda(const float v, const float inv_sigma) {
    GGML_UNUSED(v);
    GGML_UNUSED(inv_sigma);
    return 0;
}

static __device__ __forceinline__ uint8_t oscar2_v_quantize_cuda(const float v, const float inv_sigma) {
    return q2_0_quantize_symmetric_cuda(v, inv_sigma, OSCAR2_V_T0, OSCAR2_V_T1, OSCAR2_V_T2);
}

static __host__ __device__ __forceinline__ float oscar2_centroid_3bit_cuda(const int q) {
    switch (q & 0x07) {
        case 0: return OSCAR2_K_C0;
        case 1: return OSCAR2_K_C1;
        case 2: return OSCAR2_K_C2;
        case 3: return OSCAR2_K_C3;
        case 4: return OSCAR2_K_C4;
        case 5: return OSCAR2_K_C5;
        case 6: return OSCAR2_K_C6;
        default: return OSCAR2_K_C7;
    }
}

static __host__ __device__ __forceinline__ float oscar2_v_centroid_3bit_cuda(const int q) {
    switch (q & 0x07) {
        case 0: return OSCAR2_V3_C0;
        case 1: return OSCAR2_V3_C1;
        case 2: return OSCAR2_V3_C2;
        case 3: return OSCAR2_V3_C3;
        case 4: return OSCAR2_V3_C4;
        case 5: return OSCAR2_V3_C5;
        case 6: return OSCAR2_V3_C6;
        default: return OSCAR2_V3_C7;
    }
}

static __device__ __forceinline__ float q2_0_dequantize_scalar_cuda(const block_q2_0 * x, const int i) {
    const int ib   = i / QK2_0;
    const int iq   = i % QK2_0;
    const int byte = iq / 4;
    const int sh   = 2 * (iq % 4);
    const int q    = (x[ib].qs[byte] >> sh) & 0x03;
    return __half2float(x[ib].d) * q2_0_centroid_cuda(q) + __half2float(x[ib].m);
}

template<bool is_k>
static __device__ __forceinline__ float oscar2_dequantize_scalar_cuda(const block_oscar2_kv * x, const int i) {
    const int ib   = i / QK_OSCAR2_KV;
    const int iq   = i % QK_OSCAR2_KV;
    const int byte = iq / 4;
    const int sh   = 2 * (iq % 4);
    int q          = (x[ib].qs[byte] >> sh) & 0x03;
    q |= (((x[ib].rs[iq / 8] >> (iq % 8)) & 0x01) << 2);
    const float m = __half2float(x[ib].m);
    const float c = is_k ? oscar2_centroid_3bit_cuda(q) : oscar2_v_centroid_3bit_cuda(q);
    return m + __half2float(x[ib].d) * c;
}

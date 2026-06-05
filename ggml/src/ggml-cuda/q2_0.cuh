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

static __device__ __forceinline__ uint8_t q2_0_quantize_lm_cuda(const float v, const float inv_sigma) {
    const float vs = v * inv_sigma;
    if (vs < Q2_0_LM_T0) return 0;
    if (vs < Q2_0_LM_T1) return 1;
    if (vs < Q2_0_LM_T2) return 2;
    return 3;
}

static __device__ __forceinline__ float q2_0_dequantize_scalar_cuda(const block_q2_0 * x, const int i) {
    const int ib   = i / QK2_0;
    const int iq   = i % QK2_0;
    const int byte = iq / 4;
    const int sh   = 2 * (iq % 4);
    const int q    = (x[ib].qs[byte] >> sh) & 0x03;
    return __half2float(x[ib].d) * q2_0_centroid_cuda(q) + __half2float(x[ib].m);
}

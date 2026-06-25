#pragma once

#include "ggml-common.h"
#include "convert.cuh"
#include "q2_0.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q2_0_block(const float * __restrict__ x, block_q2_0 * __restrict__ y) {
    float mean = 0.0f;
    for (int j = 0; j < QK2_0; ++j) {
        mean += x[j];
    }
    mean /= QK2_0;

    float sum_sq = 0.0f;

    for (int j = 0; j < QK2_0; ++j) {
        const float v = x[j] - mean;
        sum_sq += v * v;
    }

    const float sigma    = sqrtf(sum_sq / QK2_0);
    const float inv_sigma = sigma > 1e-8f ? 1.0f / sigma : 0.0f;

    y->d = sigma;
    y->m = mean;

    for (int j = 0; j < QK2_0 / 4; ++j) {
        uint8_t packed = 0;
        for (int b = 0; b < 4; ++b) {
            packed |= q2_0_quantize_lm_cuda(x[j * 4 + b] - mean, inv_sigma) << (2 * b);
        }
        y->qs[j] = packed;
    }
}

template<uint8_t (*quantize)(float, float), bool use_high_bit>
static __device__ void quantize_f32_oscar2_block(const float * __restrict__ x, block_oscar2_kv * __restrict__ y) {
    float mean = 0.0f;
    for (int j = 0; j < QK_OSCAR2_KV; ++j) {
        mean += x[j];
    }
    mean /= QK_OSCAR2_KV;

    float sum_sq = 0.0f;
    for (int j = 0; j < QK_OSCAR2_KV; ++j) {
        const float v = x[j] - mean;
        sum_sq += v * v;
    }

    const float sigma = sqrtf(sum_sq / QK_OSCAR2_KV);
    const float inv_sigma = sigma > 1e-8f ? 1.0f / sigma : 0.0f;
    y->d = sigma;
    y->m = mean;

    for (int j = 0; j < QK_OSCAR2_KV / 4; ++j) {
        uint8_t packed = 0;
        for (int b = 0; b < 4; ++b) {
            const uint8_t q = quantize(x[j * 4 + b] - mean, inv_sigma);
            packed |= (q & 0x03) << (2 * b);
        }
        y->qs[j] = packed;
    }

    for (int j = 0; j < QK_OSCAR2_KV / 8; ++j) {
        uint8_t packed = 0;
        if constexpr (use_high_bit) {
            for (int b = 0; b < 8; ++b) {
                const int idx = j * 8 + b;
                const uint8_t q = quantize(x[idx] - mean, inv_sigma);
                packed |= ((q >> 2) & 0x01) << b;
            }
        }
        y->rs[j] = packed;
    }
}

static __device__ void quantize_f32_oscar2_k_block(const float * __restrict__ x, block_oscar2_kv * __restrict__ y) {
    float sum_sq = 0.0f;
    for (int j = 0; j < QK_OSCAR2_KV; ++j) {
        sum_sq += x[j] * x[j];
    }

    const float d  = sqrtf(sum_sq / QK_OSCAR2_KV);
    const float id = d > 1e-8f ? 1.0f / d : 0.0f;
    y->d = d;
    y->m = 0.0f;

    for (int j = 0; j < QK_OSCAR2_KV / 4; ++j) {
        y->qs[j] = 0;
    }
    for (int j = 0; j < QK_OSCAR2_KV / 8; ++j) {
        y->rs[j] = 0;
    }

    for (int idx = 0; idx < QK_OSCAR2_KV; ++idx) {
        const float v = x[idx] * id;
        int q = 0;
        float best = fabsf(v - oscar2_centroid_3bit_cuda(0));
        for (int qi = 1; qi < 8; ++qi) {
            const float err = fabsf(v - oscar2_centroid_3bit_cuda(qi));
            if (err < best) {
                best = err;
                q = qi;
            }
        }
        y->qs[idx / 4] |= (q & 0x03) << (2 * (idx % 4));
        y->rs[idx / 8] |= ((q >> 2) & 0x01) << (idx % 8);
    }
}

static __device__ void quantize_f32_oscar2_k_residual_block(const float * __restrict__ x, block_oscar2_kv * __restrict__ y) {
    float mean = 0.0f;
    for (int j = 0; j < QK_OSCAR2_KV; ++j) {
        mean += x[j];
    }
    mean /= QK_OSCAR2_KV;

    float sum_sq = 0.0f;
    for (int j = 0; j < QK_OSCAR2_KV; ++j) {
        const float v = x[j] - mean;
        sum_sq += v * v;
    }

    const float d  = sqrtf(sum_sq / QK_OSCAR2_KV);
    const float id = d > 1e-8f ? 1.0f / d : 0.0f;
    y->d = d;
    y->m = mean;

    for (int j = 0; j < QK_OSCAR2_KV / 4; ++j) {
        y->qs[j] = 0;
    }
    for (int j = 0; j < QK_OSCAR2_KV / 8; ++j) {
        y->rs[j] = 0;
    }

    for (int idx = 0; idx < QK_OSCAR2_KV; ++idx) {
        const float v = (x[idx] - mean) * id;
        int q = 0;
        float best = fabsf(v - oscar2_centroid_3bit_cuda(0));
        for (int qi = 1; qi < 8; ++qi) {
            const float err = fabsf(v - oscar2_centroid_3bit_cuda(qi));
            if (err < best) {
                best = err;
                q = qi;
            }
        }
        y->qs[idx / 4] |= (q & 0x03) << (2 * (idx % 4));
        y->rs[idx / 8] |= ((q >> 2) & 0x01) << (idx % 8);
    }
}

static __device__ void quantize_f32_oscar2_v_block(const float * __restrict__ x, block_oscar2_kv * __restrict__ y) {
    float mean = 0.0f;
    for (int j = 0; j < QK_OSCAR2_KV; ++j) {
        mean += x[j];
    }
    mean /= QK_OSCAR2_KV;

    float sum_sq = 0.0f;
    for (int j = 0; j < QK_OSCAR2_KV; ++j) {
        const float v = x[j] - mean;
        sum_sq += v * v;
    }

    const float d  = sqrtf(sum_sq / QK_OSCAR2_KV);
    const float id = d > 1e-8f ? 1.0f / d : 0.0f;
    y->d = d;
    y->m = mean;

    for (int j = 0; j < QK_OSCAR2_KV / 4; ++j) {
        y->qs[j] = 0;
    }
    for (int j = 0; j < QK_OSCAR2_KV / 8; ++j) {
        y->rs[j] = 0;
    }

    for (int idx = 0; idx < QK_OSCAR2_KV; ++idx) {
        const float v = (x[idx] - mean) * id;
        int q = 0;
        float best = fabsf(v - oscar2_v_centroid_3bit_cuda(0));
        for (int qi = 1; qi < 8; ++qi) {
            const float err = fabsf(v - oscar2_v_centroid_3bit_cuda(qi));
            if (err < best) {
                best = err;
                q = qi;
            }
        }
        y->qs[idx / 4] |= (q & 0x03) << (2 * (idx % 4));
        y->rs[idx / 8] |= ((q >> 2) & 0x01) << (idx % 8);
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q2_0(const char * cxi, char * cdsti) {
    quantize_f32_q2_0_block((const float *)cxi, (block_q2_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}

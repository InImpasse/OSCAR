#pragma once

#include "q2_0.cuh"

static inline bool q2_0_cuda_owht_enabled() {
    const char * kv_owht = getenv("LLAMA_KV_Q2_0_OWHT");
    if (kv_owht && atoi(kv_owht)) {
        return true;
    }
    const char * cuda_owht = getenv("LLAMA_CUDA_Q2_0_OWHT");
    return cuda_owht && atoi(cuda_owht);
}

static inline bool q2_0_cuda_apply_hadamard() {
    const char * no_hadamard = getenv("LLAMA_KV_NO_HADAMARD");
    return !(no_hadamard && atoi(no_hadamard));
}

static inline float q2_0_cuda_clip_ratio() {
    const char * clip_ratio = getenv("LLAMA_KV_CLIP_RATIO");
    return clip_ratio ? (float) atof(clip_ratio) : 0.0f;
}

static inline float q2_0_cuda_clip_ratio_for_cache(const char * name) {
    const bool is_k_cache = name && name[0] == 'c' && name[1] == 'a' && name[2] == 'c' && name[3] == 'h' && name[4] == 'e' && name[5] == '_' && name[6] == 'k';
    const bool is_v_cache = name && name[0] == 'c' && name[1] == 'a' && name[2] == 'c' && name[3] == 'h' && name[4] == 'e' && name[5] == '_' && name[6] == 'v';
    const char * clip_ratio = nullptr;

    if (is_k_cache) {
        clip_ratio = getenv("LLAMA_KV_CLIP_RATIO_K");
    } else if (is_v_cache) {
        clip_ratio = getenv("LLAMA_KV_CLIP_RATIO_V");
    }

    if (!clip_ratio) {
        clip_ratio = getenv("LLAMA_KV_CLIP_RATIO");
    }
    return clip_ratio ? (float) atof(clip_ratio) : 0.0f;
}

static __host__ __device__ __forceinline__ int q2_0_group_size_cuda(const int n_dims) {
    return n_dims >= Q2_0_OWHT_GROUP_SIZE ? Q2_0_OWHT_GROUP_SIZE : QK2_0;
}

static __device__ __forceinline__ void q2_0_hadamard_cuda(float * x, const int n) {
    for (int h = 1; h < n; h <<= 1) {
        for (int i = 0; i < n; i += h << 1) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j];
                const float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    const float scale = rsqrtf((float) n);
    for (int i = 0; i < n; ++i) {
        x[i] *= scale;
    }
}

static __device__ __forceinline__ void q2_0_decode_group_owht_cuda(
        const block_q2_0 * x, const int group_block, const int actual_nb, float * tmp, const bool apply_hadamard) {
    const float mean = __half2float(x[group_block].m);
    const int actual_n = actual_nb * QK2_0;

    for (int ib = 0; ib < actual_nb; ++ib) {
        const float sigma = __half2float(x[group_block + ib].d);
        float * blk = tmp + ib * QK2_0;

        for (int j = 0; j < QK2_0 / 4; ++j) {
            const uint8_t packed = x[group_block + ib].qs[j];
            for (int b = 0; b < 4; ++b) {
                blk[j * 4 + b] = sigma * q2_0_centroid_cuda((packed >> (2 * b)) & 0x03);
            }
        }
    }

    if (apply_hadamard) {
        q2_0_hadamard_cuda(tmp, actual_n);
    }

    for (int i = 0; i < actual_n; ++i) {
        tmp[i] += mean;
    }
}

template<int D>
static __device__ __forceinline__ void q2_0_dequantize_row_owht_cuda(const block_q2_0 * x, float * out, const bool apply_hadamard) {
    const int group_size = q2_0_group_size_cuda(D);

    for (int group_elem = 0; group_elem < D; group_elem += group_size) {
        const int remaining   = D - group_elem;
        const int actual_n    = remaining >= group_size ? group_size : remaining;
        const int actual_nb   = actual_n / QK2_0;
        const int group_block = group_elem / QK2_0;

        q2_0_decode_group_owht_cuda(x, group_block, actual_nb, out + group_elem, apply_hadamard);
    }
}

static __device__ __forceinline__ float q2_0_dequantize_scalar_owht_cuda(
        const block_q2_0 * x, const int i, const int n_dims, const bool apply_hadamard) {
    const int group_size  = q2_0_group_size_cuda(n_dims);
    const int group_elem  = (i / group_size) * group_size;
    const int remaining   = n_dims - group_elem;
    const int actual_n    = remaining >= group_size ? group_size : remaining;
    const int actual_nb   = actual_n / QK2_0;
    const int group_block = group_elem / QK2_0;
    float tmp[Q2_0_OWHT_GROUP_SIZE];

    q2_0_decode_group_owht_cuda(x, group_block, actual_nb, tmp, apply_hadamard);
    return tmp[i - group_elem];
}

static __device__ __forceinline__ void q2_0_clip_group_cuda(float * tmp, const int actual_n, const float clip_ratio) {
    if (!(clip_ratio > 0.0f && clip_ratio < 1.0f)) {
        return;
    }

    float absv[Q2_0_OWHT_GROUP_SIZE];
    for (int i = 0; i < actual_n; ++i) {
        absv[i] = fabsf(tmp[i]);
    }

    for (int i = 1; i < actual_n; ++i) {
        const float v = absv[i];
        int j = i - 1;
        while (j >= 0 && absv[j] > v) {
            absv[j + 1] = absv[j];
            --j;
        }
        absv[j + 1] = v;
    }

    int idx = (int) (clip_ratio * actual_n);
    if (idx >= actual_n) {
        idx = actual_n - 1;
    }
    const float thr = absv[idx];
    for (int i = 0; i < actual_n; ++i) {
        tmp[i] = fminf(fmaxf(tmp[i], -thr), thr);
    }
}

static __device__ __forceinline__ void q2_0_quantize_group_owht_cuda(
        const float * __restrict__ src, block_q2_0 * __restrict__ dst, const int actual_n,
        const bool apply_hadamard, const float clip_ratio) {
    float tmp[Q2_0_OWHT_GROUP_SIZE];

    float mean = 0.0f;
    for (int j = 0; j < actual_n; ++j) {
        mean += src[j];
    }
    mean /= actual_n;

    for (int j = 0; j < actual_n; ++j) {
        tmp[j] = src[j] - mean;
    }

    if (apply_hadamard) {
        q2_0_hadamard_cuda(tmp, actual_n);
    }

    q2_0_clip_group_cuda(tmp, actual_n, clip_ratio);

    const int actual_nb = actual_n / QK2_0;
    const half mean_h = __float2half(mean);
    for (int ib = 0; ib < actual_nb; ++ib) {
        dst[ib].m = mean_h;
    }

    for (int ib = 0; ib < actual_nb; ++ib) {
        const float * blk = tmp + ib * QK2_0;

        float sum_sq = 0.0f;
        for (int j = 0; j < QK2_0; ++j) {
            sum_sq += blk[j] * blk[j];
        }

        const float sigma = sqrtf(sum_sq / QK2_0);
        const float inv_sigma = sigma > 1e-8f ? 1.0f / sigma : 0.0f;
        dst[ib].d = __float2half(sigma);

        for (int j = 0; j < QK2_0 / 4; ++j) {
            uint8_t packed = 0;
            for (int b = 0; b < 4; ++b) {
                packed |= q2_0_quantize_lm_cuda(blk[j * 4 + b], inv_sigma) << (2 * b);
            }
            dst[ib].qs[j] = packed;
        }
    }
}

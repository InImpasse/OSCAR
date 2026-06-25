#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-oscar2-v2.cuh"

#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <type_traits>

static int ggml_cuda_fattn_vec_get_nthreads_host(const int cc) {
    return 128;
    GGML_UNUSED(cc);
}

static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device() {
    return 128;
}

static bool turbo_vec_stream_k_env_enabled() {
    const char * env = getenv("LLAMA_TURBO_VEC_STREAM_K");
    return env && env[0] != '\0' && env[0] != '0';
}

struct fattn_vec_oscar2_two_tier_params {
    int stride;
    int tail;
    int original_len;
    int effective_len;
    int mode;
    int weighted;
    int mid_tokens;
    int far_stride;
    int far_mode;
    int far_sampled;
    int mid_sampled;
    int v_avg;
    int k_avg;
    float weight_exp;
};

static __constant__ fattn_vec_oscar2_two_tier_params fattn_vec_oscar2_two_tier;

static __device__ __forceinline__ int oscar2_two_tier_physical_key(
        const int key_logical,
        const int two_tier_stride,
        const int two_tier_tail,
        const int two_tier_original_len,
        const int two_tier_prefix_sampled,
        const int two_tier_mode,
        const int three_tier_mid_tokens,
        const int three_tier_far_stride,
        const int three_tier_far_mode,
        const int three_tier_far_sampled,
        const int three_tier_mid_sampled) {
    if (two_tier_stride <= 1) {
        return key_logical;
    }
    if (three_tier_far_stride > 1 && three_tier_mid_tokens > 0) {
        const int mid_start = two_tier_original_len - two_tier_tail - three_tier_mid_tokens;
        if (key_logical < three_tier_far_sampled) {
            return min(key_logical*three_tier_far_stride + three_tier_far_mode, mid_start - 1);
        }
        if (key_logical < three_tier_far_sampled + three_tier_mid_sampled) {
            const int mid_logical = key_logical - three_tier_far_sampled;
            if (two_tier_stride == 4 && two_tier_mode == 3) {
                return min(mid_start + (mid_logical << 2) + 3, two_tier_original_len - two_tier_tail - 1);
            }
            return min(mid_start + mid_logical*two_tier_stride + two_tier_mode,
                       two_tier_original_len - two_tier_tail - 1);
        }
        return two_tier_original_len - two_tier_tail + (key_logical - three_tier_far_sampled - three_tier_mid_sampled);
    }
    if (key_logical >= two_tier_prefix_sampled) {
        return two_tier_original_len - two_tier_tail + (key_logical - two_tier_prefix_sampled);
    }
    if (two_tier_stride == 4 && two_tier_mode == 3) {
        return min((key_logical << 2) + 3, two_tier_original_len - two_tier_tail - 1);
    }
    return min(key_logical*two_tier_stride + two_tier_mode, two_tier_original_len - two_tier_tail - 1);
}

static __device__ __forceinline__ int oscar2_two_tier_group_size(
        const int key_logical,
        const int two_tier_stride,
        const int two_tier_prefix_sampled,
        const int two_tier_prefix_full_groups,
        const int two_tier_prefix_last_group_size,
        const int three_tier_far_stride,
        const int three_tier_far_sampled,
        const int three_tier_mid_sampled,
        const int three_tier_far_full_groups,
        const int three_tier_far_last_group_size,
        const int three_tier_mid_full_groups,
        const int three_tier_mid_last_group_size) {
    if (three_tier_far_stride > 1) {
        if (key_logical < three_tier_far_sampled) {
            return key_logical < three_tier_far_full_groups ? three_tier_far_stride :
                (three_tier_far_last_group_size > 0 ? three_tier_far_last_group_size : three_tier_far_stride);
        }
        if (key_logical < three_tier_far_sampled + three_tier_mid_sampled) {
            const int mid_logical = key_logical - three_tier_far_sampled;
            return mid_logical < three_tier_mid_full_groups ? two_tier_stride :
                (three_tier_mid_last_group_size > 0 ? three_tier_mid_last_group_size : two_tier_stride);
        }
        return 1;
    }
    if (key_logical < two_tier_prefix_sampled) {
        return key_logical < two_tier_prefix_full_groups ? two_tier_stride :
            (two_tier_prefix_last_group_size > 0 ? two_tier_prefix_last_group_size : two_tier_stride);
    }
    return 1;
}

static __device__ __forceinline__ int oscar2_two_tier_group_start(
        const int key_logical,
        const int two_tier_stride,
        const int two_tier_tail,
        const int two_tier_original_len,
        const int two_tier_prefix_sampled,
        const int three_tier_mid_tokens,
        const int three_tier_far_stride,
        const int three_tier_far_sampled,
        const int three_tier_mid_sampled) {
    if (two_tier_stride <= 1) {
        return key_logical;
    }
    const int prefix_len = two_tier_original_len - two_tier_tail;
    if (three_tier_far_stride > 1 && three_tier_mid_tokens > 0) {
        const int mid_start = prefix_len - three_tier_mid_tokens;
        if (key_logical < three_tier_far_sampled) {
            return key_logical*three_tier_far_stride;
        }
        if (key_logical < three_tier_far_sampled + three_tier_mid_sampled) {
            return mid_start + (key_logical - three_tier_far_sampled)*two_tier_stride;
        }
        return prefix_len + (key_logical - three_tier_far_sampled - three_tier_mid_sampled);
    }
    if (key_logical < two_tier_prefix_sampled) {
        return key_logical*two_tier_stride;
    }
    return prefix_len + (key_logical - two_tier_prefix_sampled);
}

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__

template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, bool raw_output = false, bool raw_normalized_output = false, bool raw_oscar2_two_tier = false, bool raw_oscar2_diag_no_v = false> // D == head size
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 1)
static __global__ void flash_attn_ext_vec(
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
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_lc();
#ifdef FLASH_ATTN_AVAILABLE

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

#ifdef GGML_USE_HIP
#ifdef RDNA
    constexpr int nthreads_KQ_q = 2;
#else
    constexpr int nthreads_KQ_q = 4;
#endif // RDNA
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#else
    constexpr int nthreads_KQ_q = (D/4 < 32 ? D/4 : 32);
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#endif // GGML_USE_HIP

    constexpr int nthreads    = ggml_cuda_fattn_vec_get_nthreads_device();
    constexpr bool K_is_turbo = type_K == GGML_TYPE_TURBO2_0 || type_K == GGML_TYPE_TURBO3_0;
    constexpr bool K_uses_centroid_lut = K_is_turbo || type_K == GGML_TYPE_OSCAR2_KV;
    constexpr bool V_is_turbo = type_V == GGML_TYPE_TURBO2_0 || type_V == GGML_TYPE_TURBO3_0;
    constexpr bool V_is_oscar2 = type_V == GGML_TYPE_OSCAR2_KV;
    constexpr bool type_K_uses_Q_reg = type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_BF16 || K_uses_centroid_lut;
    constexpr int nthreads_KQ = type_K == GGML_TYPE_OSCAR2_KV ? 1 :
                                (K_uses_centroid_lut ? 1 : (type_K_uses_Q_reg ? 128 / cpy_nb : nthreads_KQ_q));
    constexpr int nthreads_V  = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 128 / cpy_nb :
                                V_is_oscar2 ? (nthreads_V_q / 2 < 1 ? 1 : nthreads_V_q / 2) :
                                V_is_turbo  ? (nthreads_V_q / 4 < 1 ? 1 : nthreads_V_q / 4) : nthreads_V_q;

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");

    constexpr int V_rows_per_thread =
        (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 2*cpy_ne :
        (V_is_oscar2 && D == 128) ? 8 : 4;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;
    constexpr bool oscar2_direct_prob =
        type_K == GGML_TYPE_OSCAR2_KV && type_V == GGML_TYPE_OSCAR2_KV && nthreads_KQ == 1;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ>();
    constexpr bool Q_q8_1 = !type_K_uses_Q_reg;
#ifdef V_DOT2_F32_F16_AVAILABLE
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, half,  V_rows_per_thread>();
#else
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, float, V_rows_per_thread>();
#endif // V_DOT2_F32_F16_AVAILABLE

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = nthreads / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    constexpr int ne_KQ      = ncols*D;
    constexpr int ne_combine = nwarps*V_cols_per_iter*D;
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2            VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ half   KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#else
    float2           VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ float  KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#endif // V_DOT2_F32_F16_AVAILABLE

    constexpr int n_centroids_lut =
        (D <= 256 && type_K == GGML_TYPE_TURBO2_0) ? 4 :
        (D <= 256 && (type_K == GGML_TYPE_TURBO3_0 || type_K == GGML_TYPE_OSCAR2_KV)) ? 8 : 0;
    constexpr int lut_stride = n_centroids_lut > 0 ? n_centroids_lut + 1 : 1;
    __shared__ half turbo_lut[ncols][n_centroids_lut > 0 ? D : 1][lut_stride];

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2  Q_reg[ncols][(D/2)/nthreads_KQ]; // Will be initialized completely.
#else
    __align__(16) float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // V_DOT2_F32_F16_AVAILABLE
    int    Q_i32[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    float2  Q_ds[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];

    ggml_cuda_pdl_sync();
    if constexpr (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 1 && ic0 + j >= int(ne01.z)) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    if (i0 + WARP_SIZE <= int(D/sizeof(int)) || i < int(D/sizeof(int))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
            } else {
                const float * Q_f = (const float *) (Q + j*nb01);
                constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<float2, nthreads_quantize>
                        (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#ifdef V_DOT2_F32_F16_AVAILABLE
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;

                __align__(16) float2 tmp[cpy_ne] = {{0.0f, 0.0f}};
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0/nthreads_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#else
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x *= scale;
                Q_reg[j][k].y *= scale;
            }
        }
#endif // V_DOT2_F32_F16_AVAILABLE
    }

    if constexpr (n_centroids_lut > 0) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float * Q_f = (const float *) (Q + j*nb01);
            for (int d = tid; d < D; d += nthreads) {
                const float q_val = (ncols == 1 || ic0 + j < int(ne01.z)) ? Q_f[d] * scale : 0.0f;
                if constexpr (type_K == GGML_TYPE_TURBO2_0) {
                    turbo_lut[j][d][0] = __float2half(q_val * -0.133462f);
                    turbo_lut[j][d][1] = __float2half(q_val * -0.039994f);
                    turbo_lut[j][d][2] = __float2half(q_val *  0.039994f);
                    turbo_lut[j][d][3] = __float2half(q_val *  0.133462f);
                } else if constexpr (type_K == GGML_TYPE_TURBO3_0) {
                    turbo_lut[j][d][0] = __float2half(q_val * -0.190685f);
                    turbo_lut[j][d][1] = __float2half(q_val * -0.117832f);
                    turbo_lut[j][d][2] = __float2half(q_val * -0.065717f);
                    turbo_lut[j][d][3] = __float2half(q_val * -0.021460f);
                    turbo_lut[j][d][4] = __float2half(q_val *  0.021460f);
                    turbo_lut[j][d][5] = __float2half(q_val *  0.065717f);
                    turbo_lut[j][d][6] = __float2half(q_val *  0.117832f);
                    turbo_lut[j][d][7] = __float2half(q_val *  0.190685f);
                } else if constexpr (type_K == GGML_TYPE_OSCAR2_KV) {
                    turbo_lut[j][d][0] = __float2half(q_val * OSCAR2_K_C0);
                    turbo_lut[j][d][1] = __float2half(q_val * OSCAR2_K_C1);
                    turbo_lut[j][d][2] = __float2half(q_val * OSCAR2_K_C2);
                    turbo_lut[j][d][3] = __float2half(q_val * OSCAR2_K_C3);
                    turbo_lut[j][d][4] = __float2half(q_val * OSCAR2_K_C4);
                    turbo_lut[j][d][5] = __float2half(q_val * OSCAR2_K_C5);
                    turbo_lut[j][d][6] = __float2half(q_val * OSCAR2_K_C6);
                    turbo_lut[j][d][7] = __float2half(q_val * OSCAR2_K_C7);
                }
            }
        }
        __syncthreads();
    }

    int two_tier_stride = 1;
    int two_tier_tail = 0;
    int two_tier_original_len = ne11;
    int two_tier_prefix_sampled = 0;
    int two_tier_effective_len = ne11;
    int two_tier_mode = 0;
    int two_tier_weighted = 0;
    int two_tier_prefix_full_groups = 0;
    int two_tier_prefix_last_group_size = 0;
    int three_tier_mid_tokens = 0;
    int three_tier_far_stride = 1;
    int three_tier_far_mode = 0;
    int three_tier_far_sampled = 0;
    int three_tier_mid_sampled = 0;
    int three_tier_far_full_groups = 0;
    int three_tier_far_last_group_size = 0;
    int three_tier_mid_full_groups = 0;
    int three_tier_mid_last_group_size = 0;
    int two_tier_v_avg = 0;
    int two_tier_k_avg = 0;
    float two_tier_weight_exp = 1.0f;
    if constexpr (raw_oscar2_two_tier) {
        const fattn_vec_oscar2_two_tier_params p = fattn_vec_oscar2_two_tier;
        if (p.stride > 1 && p.effective_len > 0 && p.original_len >= p.effective_len) {
            two_tier_stride = p.stride;
            two_tier_tail = p.tail;
            two_tier_original_len = p.original_len;
            two_tier_effective_len = p.effective_len;
            two_tier_prefix_sampled = two_tier_effective_len - two_tier_tail;
            two_tier_mode = p.mode;
            two_tier_weighted = p.weighted;
            const int prefix_len = two_tier_original_len - two_tier_tail;
            two_tier_prefix_full_groups = prefix_len / two_tier_stride;
            two_tier_prefix_last_group_size = prefix_len - two_tier_prefix_full_groups*two_tier_stride;
            three_tier_mid_tokens = p.mid_tokens;
            three_tier_far_stride = p.far_stride > 1 ? p.far_stride : 1;
            three_tier_far_mode = p.far_mode;
            three_tier_far_sampled = p.far_sampled;
            three_tier_mid_sampled = p.mid_sampled;
            two_tier_v_avg = p.v_avg;
            two_tier_k_avg = p.k_avg;
            two_tier_weight_exp = p.weight_exp > 0.0f ? p.weight_exp : 1.0f;
            if (three_tier_far_stride > 1 && three_tier_mid_tokens > 0) {
                const int far_len = prefix_len - three_tier_mid_tokens;
                three_tier_far_full_groups = far_len / three_tier_far_stride;
                three_tier_far_last_group_size = far_len - three_tier_far_full_groups*three_tier_far_stride;
                three_tier_mid_full_groups = three_tier_mid_tokens / two_tier_stride;
                three_tier_mid_last_group_size = three_tier_mid_tokens - three_tier_mid_full_groups*two_tier_stride;
            }
        }
    }
    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : two_tier_effective_len;
    const char * K_base = K;
    const char * V_base = V;
    const half * maskh_base = maskh;
    GGML_UNUSED(K_base);
    GGML_UNUSED(V_base);
    GGML_UNUSED(maskh_base);
    K     += blockIdx.y*nthreads * nb11;
    V     += blockIdx.y*nthreads * nb21;
    maskh += blockIdx.y*nthreads;
    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             // Increment pointers after each loop:
             K += gridDim.y*nthreads*nb11, V += gridDim.y*nthreads*nb21, maskh += gridDim.y*nthreads) {
        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[ncols]; // KQ in registers.

        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1))) + i_KQ_0;
            const int key_logical = k_VKQ_0 + i_KQ;
            int key_physical = key_logical;
            if constexpr (raw_oscar2_two_tier) {
                key_physical = oscar2_two_tier_physical_key(
                    key_logical, two_tier_stride, two_tier_tail, two_tier_original_len,
                    two_tier_prefix_sampled, two_tier_mode,
                    three_tier_mid_tokens, three_tier_far_stride, three_tier_far_mode,
                    three_tier_far_sampled, three_tier_mid_sampled);
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum;
                if constexpr (n_centroids_lut > 0 && type_K == GGML_TYPE_TURBO2_0) {
                    const block_turbo2_0 * K_turbo = (const block_turbo2_0 *) (K + i_KQ*nb11);
                    sum = 0.0f;
#pragma unroll
                    for (int d0 = 0; d0 < D; d0 += 8) {
                        const int ib = d0 / QK_TURBO2;
                        const int jj = d0 % QK_TURBO2;
                        const float norm = __half2float(K_turbo[ib].norm);
                        const uint8_t qs0 = K_turbo[ib].qs[jj / 4];
                        const uint8_t qs1 = K_turbo[ib].qs[jj / 4 + 1];
                        sum += (__half2float(turbo_lut[j][d0  ][(qs0 >> 0) & 3]) +
                                __half2float(turbo_lut[j][d0+1][(qs0 >> 2) & 3]) +
                                __half2float(turbo_lut[j][d0+2][(qs0 >> 4) & 3]) +
                                __half2float(turbo_lut[j][d0+3][(qs0 >> 6) & 3]) +
                                __half2float(turbo_lut[j][d0+4][(qs1 >> 0) & 3]) +
                                __half2float(turbo_lut[j][d0+5][(qs1 >> 2) & 3]) +
                                __half2float(turbo_lut[j][d0+6][(qs1 >> 4) & 3]) +
                                __half2float(turbo_lut[j][d0+7][(qs1 >> 6) & 3])) * norm;
                    }
                } else if constexpr (n_centroids_lut > 0 && type_K == GGML_TYPE_TURBO3_0) {
                    const block_turbo3_0 * K_turbo = (const block_turbo3_0 *) (K + i_KQ*nb11);
                    sum = 0.0f;
#pragma unroll
                    for (int d0 = 0; d0 < D; d0 += 4) {
                        const int ib = d0 / QK_TURBO3;
                        const int jj = d0 % QK_TURBO3;
                        const float norm = __half2float(K_turbo[ib].norm);
                        const uint8_t qs = K_turbo[ib].qs[jj / 4];
                        const uint8_t signs = K_turbo[ib].signs[jj / 8];
                        const int sshift = jj & 7;
                        const uint8_t idx0 = ((qs >> 0) & 3) | (((signs >> (sshift + 0)) & 1) << 2);
                        const uint8_t idx1 = ((qs >> 2) & 3) | (((signs >> (sshift + 1)) & 1) << 2);
                        const uint8_t idx2 = ((qs >> 4) & 3) | (((signs >> (sshift + 2)) & 1) << 2);
                        const uint8_t idx3 = ((qs >> 6) & 3) | (((signs >> (sshift + 3)) & 1) << 2);
                        sum += (__half2float(turbo_lut[j][d0  ][idx0]) +
                                __half2float(turbo_lut[j][d0+1][idx1]) +
                                __half2float(turbo_lut[j][d0+2][idx2]) +
                                __half2float(turbo_lut[j][d0+3][idx3])) * norm;
                    }
                } else if constexpr (n_centroids_lut > 0 && type_K == GGML_TYPE_OSCAR2_KV) {
                    sum = 0.0f;
                    const int d_lane = threadIdx.x % nthreads_KQ;
                    int k_avg_group_start = key_physical;
                    int k_avg_group_size = 1;
                    if constexpr (raw_oscar2_two_tier) {
                        if (two_tier_k_avg && two_tier_stride > 1) {
                            k_avg_group_start = oscar2_two_tier_group_start(
                                key_logical, two_tier_stride, two_tier_tail, two_tier_original_len,
                                two_tier_prefix_sampled, three_tier_mid_tokens, three_tier_far_stride,
                                three_tier_far_sampled, three_tier_mid_sampled);
                            k_avg_group_size = oscar2_two_tier_group_size(
                                key_logical, two_tier_stride, two_tier_prefix_sampled,
                                two_tier_prefix_full_groups, two_tier_prefix_last_group_size,
                                three_tier_far_stride, three_tier_far_sampled, three_tier_mid_sampled,
                                three_tier_far_full_groups, three_tier_far_last_group_size,
                                three_tier_mid_full_groups, three_tier_mid_last_group_size);
                        }
                    }
	                    for (int g = 0; g < k_avg_group_size; ++g) {
	                        const block_oscar2_kv * K_oscar2 = (const block_oscar2_kv *)
	                            ((raw_oscar2_two_tier && two_tier_stride > 1) ?
	                                (K_base + (k_avg_group_start + g)*nb11) : (K + i_KQ*nb11));
                            const float norm = __half2float(K_oscar2->d);
                            const float mean = __half2float(K_oscar2->m);
	#pragma unroll
	                        for (int d0 = 4*d_lane; d0 < D; d0 += 4*nthreads_KQ) {
	                            const uint8_t qs = K_oscar2->qs[d0 / 4];
	                            const uint8_t res = K_oscar2->rs[d0 / 8] >> (d0 & 7);
	                            const uint8_t idx0 = ((qs >> 0) & 3) | (((res >> 0) & 1) << 2);
	                            const uint8_t idx1 = ((qs >> 2) & 3) | (((res >> 1) & 1) << 2);
	                            const uint8_t idx2 = ((qs >> 4) & 3) | (((res >> 2) & 1) << 2);
	                            const uint8_t idx3 = ((qs >> 6) & 3) | (((res >> 3) & 1) << 2);
                                float mean_term = 0.0f;
                                if (mean != 0.0f && (ncols == 1 || ic0 + j < int(ne01.z))) {
                                    const float * Q_f = (const float *) (Q + j*nb01);
                                    mean_term = mean * scale * (Q_f[d0 + 0] + Q_f[d0 + 1] + Q_f[d0 + 2] + Q_f[d0 + 3]);
                                }
                                float group_sum = (__half2float(turbo_lut[j][d0  ][idx0]) +
                                                   __half2float(turbo_lut[j][d0+1][idx1]) +
                                                   __half2float(turbo_lut[j][d0+2][idx2]) +
                                                   __half2float(turbo_lut[j][d0+3][idx3])) * norm +
                                                   mean_term;
                                sum += group_sum;
	                        }
	                    }
                    if (k_avg_group_size > 1) {
                        sum /= float(k_avg_group_size);
                    }
	                    if constexpr (nthreads_KQ > 1) {
	                        sum = warp_reduce_sum<nthreads_KQ>(sum);
	                    }
                } else {
                    sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                    sum = warp_reduce_sum<nthreads_KQ>(sum);
                }

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                if (mask && (ncols == 1 || ic0 + j < int(ne01.z))) {
                    if constexpr (raw_oscar2_two_tier) {
                        const half * mask_row = two_tier_stride > 1 ? maskh_base : maskh;
                        const int i_mask = two_tier_stride > 1 ? key_physical : i_KQ;
                        sum += slope*__half2float(mask_row[j*two_tier_original_len + i_mask]);
                    } else {
                        sum += slope*__half2float(maskh[j*ne11 + i_KQ]);
                    }
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);

                if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            if constexpr (raw_oscar2_two_tier) {
                if (two_tier_weighted && two_tier_stride > 1) {
                    const int key_logical = k_VKQ_0 + tid;
                    const float group_size = float(oscar2_two_tier_group_size(
                        key_logical, two_tier_stride, two_tier_prefix_sampled,
                        two_tier_prefix_full_groups, two_tier_prefix_last_group_size,
                        three_tier_far_stride, three_tier_far_sampled, three_tier_mid_sampled,
                        three_tier_far_full_groups, three_tier_far_last_group_size,
                        three_tier_mid_full_groups, three_tier_mid_last_group_size));
                    KQ_reg[j] *= two_tier_weight_exp == 1.0f ? group_size : expf(logf(group_size)*two_tier_weight_exp);
                }
            }
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

            if constexpr (!raw_oscar2_diag_no_v) {
#ifdef V_DOT2_F32_F16_AVAILABLE
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                    VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
                }
#else
#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                    VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                    VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
                }
#endif // V_DOT2_F32_F16_AVAILABLE
            }
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif // GGML_USE_HIP

        if constexpr (!raw_oscar2_diag_no_v) {
#pragma unroll
            for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
                const int k = threadIdx.y*WARP_SIZE + k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);
                const int key_logical = k_VKQ_0 + k;
                int key_physical = key_logical;
                if constexpr (raw_oscar2_two_tier) {
                    key_physical = oscar2_two_tier_physical_key(
                        key_logical, two_tier_stride, two_tier_tail, two_tier_original_len,
                        two_tier_prefix_sampled, two_tier_mode,
                        three_tier_mid_tokens, three_tier_far_stride, three_tier_far_mode,
                        three_tier_far_sampled, three_tier_mid_sampled);
                }

#ifdef V_DOT2_F32_F16_AVAILABLE
            half2 KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                if constexpr (oscar2_direct_prob) {
                    KQ_k[j] = __half2half2(__shfl_sync(0xFFFFFFFF, KQ_reg[j], k & (WARP_SIZE - 1), WARP_SIZE));
                } else {
                    KQ_k[j] = __half2half2(KQ[j*nthreads + k]);
                }
            }
#pragma unroll
            {
#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                    half2 tmp[V_rows_per_thread/2];
                    if constexpr (type_V == GGML_TYPE_BF16) {
                        float2 tmp_f[V_rows_per_thread/2];
                        dequantize_V(V + k*nb21, tmp_f,
                            2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                        for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
                            tmp[i_VKQ_1] = __float22half2_rn(tmp_f[i_VKQ_1]);
                        }
                    } else if constexpr (type_V == GGML_TYPE_OSCAR2_KV && V_rows_per_thread == 4 && D == 128) {
                        const int i0 = 2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread;
                        if constexpr (raw_oscar2_two_tier) {
                            if (two_tier_v_avg && two_tier_stride > 1) {
                                const int group_start = oscar2_two_tier_group_start(
                                    key_logical, two_tier_stride, two_tier_tail, two_tier_original_len,
                                    two_tier_prefix_sampled, three_tier_mid_tokens, three_tier_far_stride,
                                    three_tier_far_sampled, three_tier_mid_sampled);
                                const int group_size = oscar2_two_tier_group_size(
                                    key_logical, two_tier_stride, two_tier_prefix_sampled,
                                    two_tier_prefix_full_groups, two_tier_prefix_last_group_size,
                                    three_tier_far_stride, three_tier_far_sampled, three_tier_mid_sampled,
                                    three_tier_far_full_groups, three_tier_far_last_group_size,
                                    three_tier_mid_full_groups, three_tier_mid_last_group_size);
                                float2 avg0 = make_float2(0.0f, 0.0f);
                                float2 avg1 = make_float2(0.0f, 0.0f);
                                for (int g = 0; g < group_size; ++g) {
                                    const block_oscar2_kv * V_oscar2 = (const block_oscar2_kv *) (V_base + (group_start + g)*nb21);
                                    const block_oscar2_kv & b = V_oscar2[i0 / QK_OSCAR2_KV];
                                    const int j0 = i0 % QK_OSCAR2_KV;
                                    const uint8_t qs = b.qs[j0 / 4];
                                    const uint8_t rs = b.rs[j0 / 8] >> (j0 & 7);
                                    float2 v0;
                                    float2 v1;
                                    oscar2_dequantize_4_v_f2(b, qs, rs, v0, v1);
                                    avg0.x += v0.x;
                                    avg0.y += v0.y;
                                    avg1.x += v1.x;
                                    avg1.y += v1.y;
                                }
                                const float inv_group_size = 1.0f / float(group_size);
                                tmp[0] = make_half2(avg0.x*inv_group_size, avg0.y*inv_group_size);
                                tmp[1] = make_half2(avg1.x*inv_group_size, avg1.y*inv_group_size);
                            } else {
                                const block_oscar2_kv * V_oscar2 = (const block_oscar2_kv *)
                                    (two_tier_stride > 1 ? (V_base + key_physical*nb21) : (V + k*nb21));
                                const block_oscar2_kv & b = V_oscar2[i0 / QK_OSCAR2_KV];
                                const int j0 = i0 % QK_OSCAR2_KV;
                                const uint8_t qs = b.qs[j0 / 4];
                                const uint8_t rs = b.rs[j0 / 8] >> (j0 & 7);
                                oscar2_dequantize_4_v_h2(b, qs, rs, tmp[0], tmp[1]);
                            }
                        } else {
                            const block_oscar2_kv * V_oscar2 = (const block_oscar2_kv *) (V + k*nb21);
                            const block_oscar2_kv & b = V_oscar2[i0 / QK_OSCAR2_KV];
                            const int j0 = i0 % QK_OSCAR2_KV;
                            const uint8_t qs = b.qs[j0 / 4];
                            const uint8_t rs = b.rs[j0 / 8] >> (j0 & 7);
                            oscar2_dequantize_4_v_h2(b, qs, rs, tmp[0], tmp[1]);
                        }
                    } else {
                        dequantize_V(V + k*nb21, tmp,
                            2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
                    }
#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                        for (int j = 0; j < ncols; ++j) {
                            VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                        }
                    }
                }
            }
#else
            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                if constexpr (oscar2_direct_prob) {
                    KQ_k[j] = __shfl_sync(0xFFFFFFFF, KQ_reg[j], k & (WARP_SIZE - 1), WARP_SIZE);
                } else {
                    KQ_k[j] = KQ[j*nthreads + k];
                }
            }
            if constexpr (type_V == GGML_TYPE_TURBO2_0) {
                const block_turbo2_0 * vb = (const block_turbo2_0 *) (V + k*nb21);
                int prev_ib = -1;
                float sc[4];
#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                    const int i0 = 2*i_VKQ_0 + (threadIdx.x % nthreads_V)*V_rows_per_thread;
                    const int ib = i0 / QK_TURBO2;
                    const int j0 = i0 % QK_TURBO2;

                    if (ib != prev_ib) {
                        prev_ib = ib;
                        const float norm = __half2float(vb[ib].norm);
                        sc[0] = -0.133462f * norm;
                        sc[1] = -0.039994f * norm;
                        sc[2] =  0.039994f * norm;
                        sc[3] =  0.133462f * norm;
                    }

                    const uint8_t qs_byte = vb[ib].qs[j0 / 4];
                    const uint8_t idx0 = (qs_byte >> 0) & 3;
                    const uint8_t idx1 = (qs_byte >> 2) & 3;
                    const uint8_t idx2 = (qs_byte >> 4) & 3;
                    const uint8_t idx3 = (qs_byte >> 6) & 3;
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + 0].x += sc[idx0]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 0].y += sc[idx1]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 1].x += sc[idx2]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 1].y += sc[idx3]*KQ_k[j];
                    }
                }
            } else {
#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                    float2 tmp[V_rows_per_thread/2];
                    dequantize_V(V + k*nb21, tmp,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                        for (int j = 0; j < ncols; ++j) {
                            VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x*KQ_k[j];
                            VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y*KQ_k[j];
                        }
                    }
                }
            }
#endif // V_DOT2_F32_F16_AVAILABLE
            }
        }

#ifdef V_DOT2_F32_F16_AVAILABLE
#endif // V_DOT2_F32_F16_AVAILABLE
    }

    if (sinks && blockIdx.y == 0) {
        const float sink = ((const float *) sinks)[head];

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = expf(KQ_max[j] - kqmax_new_j);
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + (threadIdx.x == 0 ? expf(sink - KQ_max[j]) : 0.0f);

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 1 && ic0 + j_VKQ >= int(ne01.z)) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

#ifdef V_DOT2_F32_F16_AVAILABLE
        half2 * VKQ_tmp = (half2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread*sizeof(half)>(VKQ_tmp + i_VKQ, &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
        }
#else
        float2 * VKQ_tmp = (float2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // V_DOT2_F32_F16_AVAILABLE

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                if ((!raw_output || raw_normalized_output) && gridDim.y == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence*int(ne01.z) + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
            }
        }

        if (j_VKQ < ncols-1) {
            __syncthreads();
        }

    }

    if (dst_meta && tid < ncols && (ncols == 1 || ic0 + tid < int(ne01.z))) {
        dst_meta[((sequence*int(ne01.z) + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template<int D, int nthreads_KQ>
static __device__ __forceinline__ float mixed_oscar2_kq_lut_chunk(
        const block_oscar2_kv * __restrict__ k_row,
        const int chunk,
        const float * __restrict__ q_centroid_lut) {
    static_assert(D == 128, "mixed oscar2 KQ LUT helper is specialized for D=128");
    const float kd = __half2float(k_row->d);
    float sum = 0.0f;
#pragma unroll
    for (int d0 = 4*chunk; d0 < D; d0 += 4*nthreads_KQ) {
        const uint8_t packed = k_row->qs[d0 / 4];
        const uint8_t res = k_row->rs[d0 / 8] >> (d0 & 7);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int d = d0 + b;
            uint8_t q = (packed >> (2*b)) & 0x03;
            q |= (((res >> b) & 0x01) << 2);
            sum += q_centroid_lut[d*8 + q];
        }
    }
    return kd * sum;
}

template<int D, int ncols>
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 1)
static __global__ void flash_attn_ext_mixed_oscar2_f16_vec(
        const char * __restrict__ Q,
        const char * __restrict__ K_lp,
        const char * __restrict__ V_lp,
        const char * __restrict__ mask_lp,
        const char * __restrict__ K_hp,
        const char * __restrict__ V_hp,
        const char * __restrict__ mask_hp,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const int32_t ne01, const int32_t ne02, const int32_t ne03,
        const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne11_lp, const int32_t ne12_lp,
        const int32_t nb11_lp, const int32_t nb12_lp, const int64_t nb13_lp,
        const int32_t nb21_lp, const int32_t nb22_lp, const int64_t nb23_lp,
        const int32_t ne11_hp, const int32_t ne12_hp,
        const int32_t nb11_hp, const int32_t nb12_hp, const int64_t nb13_hp,
        const int32_t nb21_hp, const int32_t nb22_hp, const int64_t nb23_hp,
        const int32_t nb31_lp, const int64_t nb33_lp,
        const int32_t nb31_hp, const int64_t nb33_hp) {
#ifdef FLASH_ATTN_AVAILABLE
    static_assert(D == 128, "mixed oscar2/f16 vec is currently specialized for D=128");

    constexpr int nthreads = ggml_cuda_fattn_vec_get_nthreads_device();
    constexpr int nwarps = nthreads / WARP_SIZE;
    constexpr int nthreads_KQ = D/4 < 32 ? D/4 : 32;
    constexpr int nthreads_V_lp = (D/4 < 32 ? D/4 : 32) / 4;
    constexpr int nthreads_V_hp = nthreads_V_lp;
    constexpr int V_rows_per_thread_lp = 4;
    constexpr int V_rows_per_thread_hp = V_rows_per_thread_lp;
    constexpr int V_cols_per_iter_lp = WARP_SIZE / nthreads_V_lp;
    constexpr int V_cols_per_iter_hp = WARP_SIZE / nthreads_V_hp;
    constexpr int V_cols_per_iter_max = V_cols_per_iter_lp > V_cols_per_iter_hp ? V_cols_per_iter_lp : V_cols_per_iter_hp;
    constexpr int ne_KQ = ncols*nthreads;
    constexpr int ne_combine = nwarps*V_cols_per_iter_max*D;

    const int ic0 = blockIdx.x * ncols;
    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio_lp = ne02 / ne12_lp;
    const int gqa_ratio_hp = ne02 / ne12_hp;
    const int hkv_lp = head / gqa_ratio_lp;
    const int hkv_hp = head / gqa_ratio_hp;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;

    Q    += nb03*sequence + nb02*head + nb01*ic0;
    K_lp += nb13_lp*sequence + nb12_lp*hkv_lp;
    V_lp += nb23_lp*sequence + nb22_lp*hkv_lp;
    K_hp += nb13_hp*sequence + nb12_hp*hkv_hp;
    V_hp += nb23_hp*sequence + nb22_hp*hkv_hp;
    const char * mask_lp_seq = mask_lp + nb33_lp*sequence + nb31_lp*ic0;
    const char * mask_hp_seq = mask_hp + nb33_hp*sequence + nb31_hp*ic0;

    float2 VKQ[ncols][D/(2*nthreads_V_lp)] = {{{0.0f, 0.0f}}};
    __shared__ float KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
    __shared__ float K_lut[ncols][D][8];

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    int    Q_i32[ncols][D/(sizeof(int)*nthreads_KQ)];
    float2 Q_ds [ncols][D/(sizeof(int)*nthreads_KQ)];
    float  Q_hp [ncols][D/nthreads_KQ];

#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
        if (j >= ncols) {
            break;
        }

        int    * tmp_q_i32 = (int    *) &KQ[j*D];
        float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

        if (ic0 + j >= ne01) {
#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;
                if (i < int(D/sizeof(int))) {
                    tmp_q_i32[i] = 0;
                }
            }
            if (threadIdx.x < D/QK8_1) {
                tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
            }
        } else {
            const float * Q_f = (const float *) (Q + j*nb01);
            constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                quantize_q8_1_to_shared<float2, nthreads_quantize>(
                    Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        int    * tmp_q_i32 = (int    *) &KQ[j*D];
        float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));
        const float * Q_f = (const float *) (Q + j*nb01);
#pragma unroll
        for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
            const int i = i0 + (threadIdx.x % nthreads_KQ);
            Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
            Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
        }
#pragma unroll
        for (int d0 = 0; d0 < D; d0 += nthreads_KQ) {
            const int d = d0 + (threadIdx.x % nthreads_KQ);
            Q_hp[j][d0/nthreads_KQ] = (ic0 + j < ne01) ? scale * Q_f[d] : 0.0f;
        }
    }

    if (threadIdx.y == 0) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const int chunk = threadIdx.x;
            const int u = Q_i32[j][0];
            const int8_t * uq = (const int8_t *) &u;
            const float qd = Q_ds[j][0].x;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int d = 4*chunk + b;
                const float qv = qd * (float) uq[b];
                K_lut[j][d][0] = qv * OSCAR2_K_C0;
                K_lut[j][d][1] = qv * OSCAR2_K_C1;
                K_lut[j][d][2] = qv * OSCAR2_K_C2;
                K_lut[j][d][3] = qv * OSCAR2_K_C3;
                K_lut[j][d][4] = qv * OSCAR2_K_C4;
                K_lut[j][d][5] = qv * OSCAR2_K_C5;
                K_lut[j][d][6] = qv * OSCAR2_K_C6;
                K_lut[j][d][7] = qv * OSCAR2_K_C7;
            }
        }
    }

    __syncthreads();

    auto process_tile = [&](auto hp_tag, const int k0, const char * K_base, const char * V_base, const char * mask_base,
                            const int nbK1, const int nbV1, const int nbM1, const int ne11) {
        constexpr bool hp = decltype(hp_tag)::value;
        float KQ_reg[ncols];
        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (threadIdx.x & ~(nthreads_KQ - 1)) + i_KQ_0;
            const int key = k0 + i_KQ;
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = -INFINITY;
                if (key < ne11 && ic0 + j < ne01) {
                    if constexpr (hp) {
                        const half * K_h = (const half *) (K_base + i_KQ*nbK1);
                        sum = 0.0f;
#pragma unroll
                        for (int d0 = 0; d0 < D; d0 += nthreads_KQ) {
                            const int d = d0 + (threadIdx.x % nthreads_KQ);
                            sum += Q_hp[j][d0/nthreads_KQ] * __half2float(K_h[d]);
                        }
                        sum = warp_reduce_sum<nthreads_KQ>(sum);
                    } else {
                        const int chunk = threadIdx.x % nthreads_KQ;
                        sum = mixed_oscar2_kq_lut_chunk<D, nthreads_KQ>(
                            (const block_oscar2_kv *) (K_base + i_KQ*nbK1), chunk, &K_lut[j][0][0]);
                        sum = warp_reduce_sum<nthreads_KQ>(sum);
                    }

                    const float m = *(const float *) (mask_base + j*nbM1 + key*sizeof(float));
                    sum += m;
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);
                if ((threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V_lp) {
                VKQ[j][i_VKQ_0/nthreads_V_lp].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V_lp].y *= KQ_max_scale;
            }
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif

        if constexpr (hp) {
#pragma unroll
            for (int k00 = 0; k00 < WARP_SIZE; k00 += V_cols_per_iter_hp) {
                const int k = threadIdx.y*WARP_SIZE + k00 + threadIdx.x / nthreads_V_hp;
                if (k0 + k >= ne11) {
                    continue;
                }
                const half2 * V_h2 = (const half2 *) (V_base + k*nbV1);
                float KQ_k[ncols];
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    KQ_k[j] = KQ[j*nthreads + k];
                }
#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V_hp*V_rows_per_thread_hp/2) {
                    const int i_VKQ = i_VKQ_0 + (threadIdx.x % nthreads_V_hp)*(V_rows_per_thread_hp/2);
                    float2 tmp[V_rows_per_thread_hp/2];
#pragma unroll
                    for (int ii = 0; ii < V_rows_per_thread_hp/2; ++ii) {
                        tmp[ii] = __half22float2(V_h2[i_VKQ + ii]);
                    }
#pragma unroll
                    for (int ii = 0; ii < V_rows_per_thread_hp/2; ++ii) {
#pragma unroll
                        for (int j = 0; j < ncols; ++j) {
                            VKQ[j][i_VKQ_0/nthreads_V_lp + ii].x += tmp[ii].x*KQ_k[j];
                            VKQ[j][i_VKQ_0/nthreads_V_lp + ii].y += tmp[ii].y*KQ_k[j];
                        }
                    }
                }
            }
        } else {
#pragma unroll
            for (int k00 = 0; k00 < WARP_SIZE; k00 += V_cols_per_iter_lp) {
                const int k = threadIdx.y*WARP_SIZE + k00 + threadIdx.x / nthreads_V_lp;
                if (k0 + k >= ne11) {
                    continue;
                }
                float KQ_k[ncols];
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    KQ_k[j] = KQ[j*nthreads + k];
                }
#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V_lp*V_rows_per_thread_lp/2) {
                    float2 tmp[V_rows_per_thread_lp/2];
                    dequantize_V_oscar2<false, float, V_rows_per_thread_lp>(
                        V_base + k*nbV1, tmp, 2*i_VKQ_0 + (threadIdx.x % nthreads_V_lp)*V_rows_per_thread_lp);
#pragma unroll
                    for (int ii = 0; ii < V_rows_per_thread_lp/2; ++ii) {
#pragma unroll
                        for (int j = 0; j < ncols; ++j) {
                            VKQ[j][i_VKQ_0/nthreads_V_lp + ii].x += tmp[ii].x*KQ_k[j];
                            VKQ[j][i_VKQ_0/nthreads_V_lp + ii].y += tmp[ii].y*KQ_k[j];
                        }
                    }
                }
            }
        }
    };

    for (int k0 = blockIdx.y*nthreads; k0 < ne11_lp; k0 += gridDim.y*nthreads) {
        process_tile(std::false_type{}, k0, K_lp + k0*nb11_lp, V_lp + k0*nb21_lp, mask_lp_seq, nb11_lp, nb21_lp, nb31_lp, ne11_lp);
    }
    for (int k0 = blockIdx.y*nthreads; k0 < ne11_hp; k0 += gridDim.y*nthreads) {
        process_tile(std::true_type{}, k0, K_hp + k0*nb11_hp, V_hp + k0*nb21_hp, mask_hp_seq, nb11_hp, nb21_hp, nb31_hp, ne11_hp);
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ic0 + j_VKQ >= ne01) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

        float2 * VKQ_tmp = (float2 *) KQ + threadIdx.y*(V_cols_per_iter_lp*D/2)
            + (threadIdx.x / nthreads_V_lp)*(D/2);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V_lp) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V_lp].x *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V_lp].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V_lp*V_rows_per_thread_lp/2) {
            const int i_VKQ = i_VKQ_0 + (threadIdx.x % nthreads_V_lp)*(V_rows_per_thread_lp/2);
            ggml_cuda_memcpy_1<V_rows_per_thread_lp/2*sizeof(float)>(VKQ_tmp + i_VKQ,
                &VKQ[j_VKQ][i_VKQ_0/nthreads_V_lp]);
            ggml_cuda_memcpy_1<V_rows_per_thread_lp/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread_lp/4,
                &VKQ[j_VKQ][i_VKQ_0/nthreads_V_lp + V_rows_per_thread_lp/4]);
        }

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }
        __syncthreads();

        if (tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
            float dst_val = 0.0f;
#pragma unroll
            for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                for (int v = 0; v < V_cols_per_iter_lp; ++v) {
                    dst_val += KQ[w*V_cols_per_iter_lp*D + v*D + tid];
                }
            }
            if (gridDim.y == 1) {
                dst_val /= KQ_sum[j_VKQ];
            }
            dst[(((sequence*ne01 + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D + tid] = dst_val;
        }

        if (j_VKQ < ncols - 1) {
            __syncthreads();
        }
    }

    if (dst_meta && tid < ncols && ic0 + tid < ne01) {
        dst_meta[((sequence*ne01 + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#else
    GGML_UNUSED_VARS(Q, K_lp, V_lp, mask_lp, K_hp, V_hp, mask_hp, dst, dst_meta, scale,
        ne01, ne02, ne03, nb01, nb02, nb03,
        ne11_lp, ne12_lp, nb11_lp, nb12_lp, nb13_lp, nb21_lp, nb22_lp, nb23_lp,
        ne11_hp, ne12_hp, nb11_hp, nb12_hp, nb13_hp, nb21_hp, nb22_hp, nb23_hp,
        nb31_lp, nb33_lp, nb31_hp, nb33_hp);
    NO_DEVICE_CODE;
#endif
}

template<int D, int ncols>
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 1)
static __global__ void flash_attn_ext_mixed_oscar2_lp_raw_vec(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float      * __restrict__ dst_num,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float logit_softcap,
        const int normalize_output,
        const int mask_f16,
        const int32_t ne01, const int32_t ne02, const int32_t ne03,
        const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne11, const int32_t ne12,
        const int32_t nb11, const int32_t nb12, const int64_t nb13,
        const int32_t nb21, const int32_t nb22, const int64_t nb23,
        const int32_t nb31, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE
    static_assert(D == 128, "mixed oscar2 raw LP vec is currently specialized for D=128");

    constexpr int nthreads = ggml_cuda_fattn_vec_get_nthreads_device();
    constexpr int nwarps = nthreads / WARP_SIZE;
    constexpr int nthreads_KQ = 2;
    constexpr int nthreads_V = 16;
    constexpr int V_rows_per_thread = 4;
    constexpr int V_cols_per_iter = WARP_SIZE / nthreads_V;
    constexpr int ne_KQ = ncols*D;
    constexpr int ne_combine = nwarps*V_cols_per_iter*D;

    const int ic0 = blockIdx.x * ncols;
    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12;
    const int hkv = head / gqa_ratio;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;

    Q += nb03*sequence + nb02*head + nb01*ic0;
    K += nb13*sequence + nb12*hkv;
    V += nb23*sequence + nb22*hkv;
    const char * mask_seq = mask + nb33*sequence + nb31*ic0;

    float2 VKQ[ncols][D/(2*nthreads_V)] = {{{0.0f, 0.0f}}};
    __shared__ float KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
    __shared__ float K_lut[ncols][D][8];

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        const float * Q_f = (const float *) (Q + j*nb01);
        for (int d = tid; d < D; d += nthreads) {
            const float qv = (ic0 + j < ne01) ? Q_f[d] * scale : 0.0f;
            K_lut[j][d][0] = qv * OSCAR2_K_C0;
            K_lut[j][d][1] = qv * OSCAR2_K_C1;
            K_lut[j][d][2] = qv * OSCAR2_K_C2;
            K_lut[j][d][3] = qv * OSCAR2_K_C3;
            K_lut[j][d][4] = qv * OSCAR2_K_C4;
            K_lut[j][d][5] = qv * OSCAR2_K_C5;
            K_lut[j][d][6] = qv * OSCAR2_K_C6;
            K_lut[j][d][7] = qv * OSCAR2_K_C7;
        }
    }

    __syncthreads();

    for (int k0 = blockIdx.y*nthreads; k0 < ne11; k0 += gridDim.y*nthreads) {
        float KQ_reg[ncols];
        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (threadIdx.x & ~(nthreads_KQ - 1)) + i_KQ_0;
            const int key = k0 + i_KQ;
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = -INFINITY;
                if (key < ne11 && ic0 + j < ne01) {
                    const int chunk = threadIdx.x % nthreads_KQ;
                    sum = mixed_oscar2_kq_lut_chunk<D, nthreads_KQ>(
                        (const block_oscar2_kv *) (K + key*nb11), chunk, &K_lut[j][0][0]);
                    sum = warp_reduce_sum<nthreads_KQ>(sum);
                    if (logit_softcap != 0.0f) {
                        sum = logit_softcap*tanhf(sum);
                    }
                    if (mask_f16) {
                        sum += __half2float(*(const half *) (mask_seq + j*nb31 + key*sizeof(half)));
                    } else {
                        sum += *(const float *) (mask_seq + j*nb31 + key*sizeof(float));
                    }
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);
                if ((threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif

#pragma unroll
        for (int k00 = 0; k00 < WARP_SIZE; k00 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k00 + threadIdx.x / nthreads_V;
            if (k0 + k >= ne11) {
                continue;
            }

            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }

#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                float2 tmp[V_rows_per_thread/2];
                dequantize_V_oscar2<false, float, V_rows_per_thread>(
                    V + (k0 + k)*nb21, tmp, 2*i_VKQ_0 + (threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                for (int ii = 0; ii < V_rows_per_thread/2; ++ii) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + ii].x += tmp[ii].x*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + ii].y += tmp[ii].y*KQ_k[j];
                    }
                }
            }
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ic0 + j_VKQ >= ne01) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

        float2 * VKQ_tmp = (float2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (threadIdx.x / nthreads_V)*(D/2);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (threadIdx.x % nthreads_V)*(V_rows_per_thread/2);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,
                &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4,
                &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }
        __syncthreads();

        if (tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
            float dst_val = 0.0f;
#pragma unroll
            for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                for (int v = 0; v < V_cols_per_iter; ++v) {
                    dst_val += KQ[w*V_cols_per_iter*D + v*D + tid];
                }
            }
            if (normalize_output && gridDim.y == 1) {
                dst_val /= KQ_sum[j_VKQ];
            }
            dst_num[(((sequence*ne01 + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D + tid] = dst_val;
        }

        if (j_VKQ < ncols - 1) {
            __syncthreads();
        }
    }

    if (tid < ncols && ic0 + tid < ne01) {
        dst_meta[((sequence*ne01 + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, dst_num, dst_meta, scale, logit_softcap, normalize_output, mask_f16,
        ne01, ne02, ne03, nb01, nb02, nb03,
        ne11, ne12, nb11, nb12, nb13, nb21, nb22, nb23, nb31, nb33);
    NO_DEVICE_CODE;
#endif
}

template<int D>
__launch_bounds__(D, 1)
static __global__ void flash_attn_ext_mixed_oscar2_hp_combine(
        const char * __restrict__ Q,
        const char * __restrict__ K_hp,
        const char * __restrict__ V_hp,
        const char * __restrict__ mask_hp,
        const float * __restrict__ lp_num,
        const float2 * __restrict__ lp_meta,
        float * __restrict__ dst,
        const float scale,
        const float logit_softcap,
        const int parallel_blocks,
        const int lp_normalized,
        const int32_t ne01, const int32_t ne02, const int32_t ne03,
        const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne11_hp, const int32_t ne12_hp,
        const int32_t nb11_hp, const int32_t nb12_hp, const int64_t nb13_hp,
        const int32_t nb21_hp, const int32_t nb22_hp, const int64_t nb23_hp,
        const int32_t nb31_hp, const int64_t nb33_hp) {
#ifdef FLASH_ATTN_AVAILABLE
    static_assert(D == 128, "mixed oscar2 HP combine is currently specialized for D=128");

    const int tid = threadIdx.x;
    const int col = blockIdx.x;
    const int head = blockIdx.y;
    const int sequence = blockIdx.z;
    const int gqa_ratio_hp = ne02 / ne12_hp;
    const int hkv_hp = head / gqa_ratio_hp;

    const int row = (sequence*ne01 + col)*ne02 + head;
    const float * lp_num_row = lp_num + int64_t(row)*parallel_blocks*D;
    const float2 * lp_meta_row = lp_meta + int64_t(row)*parallel_blocks;

    float kqmax = -INFINITY;
    for (int part = 0; part < parallel_blocks; ++part) {
        const float2 meta = lp_meta_row[part];
        if (isfinite(meta.x) && isfinite(meta.y) && meta.y > 0.0f) {
            kqmax = fmaxf(kqmax, meta.x);
        }
    }

    float numerator = 0.0f;
    float denom = 0.0f;
    for (int part = 0; part < parallel_blocks; ++part) {
        const float w = expf(lp_meta_row[part].x - kqmax);
        numerator += w * lp_num_row[part*D + tid];
        denom += w * lp_meta_row[part].y;
    }

    Q    += nb03*sequence + nb02*head + nb01*col;
    K_hp += nb13_hp*sequence + nb12_hp*hkv_hp;
    V_hp += nb23_hp*sequence + nb22_hp*hkv_hp;
    const char * mask_hp_seq = mask_hp + nb33_hp*sequence + nb31_hp*col;

    const float q = scale * ((const float *) Q)[tid];
    __shared__ float warp_partials[WARP_SIZE];

    for (int key = 0; key < ne11_hp; ++key) {
        const float mask_val = *(const float *) (mask_hp_seq + key*sizeof(float));
        if (mask_val < -1.0e30f) {
            continue;
        }

        const half * K_h = (const half *) (K_hp + key*nb11_hp);
        float partial = q * __half2float(K_h[tid]);
        partial = warp_reduce_sum(partial);

        if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
            warp_partials[threadIdx.x / WARP_SIZE] = partial;
        }
        __syncthreads();

        float score = threadIdx.x < D/WARP_SIZE ? warp_partials[threadIdx.x] : 0.0f;
        if (threadIdx.x < WARP_SIZE) {
            score = warp_reduce_sum(score);
            if (threadIdx.x == 0) {
                warp_partials[0] = score;
            }
        }
        __syncthreads();

        score = warp_partials[0] + mask_val;
        const float kqmax_new = fmaxf(kqmax, score + FATTN_KQ_MAX_OFFSET);
        const float old_scale = expf(kqmax - kqmax_new);
        const float p = expf(score - kqmax_new);
        const half * V_h = (const half *) (V_hp + key*nb21_hp);
        numerator = numerator*old_scale + p*__half2float(V_h[tid]);
        denom = denom*old_scale + p;
        kqmax = kqmax_new;
        __syncthreads();
    }

    dst[int64_t(row)*D + tid] = numerator / denom;
#else
    GGML_UNUSED_VARS(Q, K_hp, V_hp, mask_hp, lp_num, lp_meta, dst, scale, parallel_blocks,
        ne01, ne02, ne03, nb01, nb02, nb03,
        ne11_hp, ne12_hp, nb11_hp, nb12_hp, nb13_hp, nb21_hp, nb22_hp, nb23_hp,
        nb31_hp, nb33_hp);
    NO_DEVICE_CODE;
#endif
}

template<int D, int ncols, bool mask_skip, bool use_logit_softcap>
__launch_bounds__(WARP_SIZE*ncols, 1)
static __global__ void flash_attn_ext_mixed_oscar2_hp_combine_warp(
        const char * __restrict__ Q,
        const char * __restrict__ K_hp,
        const char * __restrict__ V_hp,
        const char * __restrict__ mask_hp,
        const float * __restrict__ lp_num,
        const float2 * __restrict__ lp_meta,
        float * __restrict__ dst,
        const float scale,
        const float logit_softcap,
        const int parallel_blocks,
        const int lp_normalized,
        const int32_t ne01, const int32_t ne02, const int32_t ne03,
        const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne11_hp, const int32_t ne12_hp,
        const int32_t hp_sink, const int32_t ne11_lp_eff,
        const int32_t compact_hp_mask,
        const int32_t nb11_hp, const int32_t nb12_hp, const int64_t nb13_hp,
        const int32_t nb21_hp, const int32_t nb22_hp, const int64_t nb23_hp,
        const int32_t nb31_hp, const int64_t nb33_hp) {
#ifdef FLASH_ATTN_AVAILABLE
    static_assert(D == 128, "mixed oscar2 warp HP combine is specialized for D=128");

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int col = blockIdx.x*ncols + warp;
    if (warp >= ncols || col >= ne01) {
        return;
    }

    const int head = blockIdx.y;
    const int sequence = blockIdx.z;
    const int gqa_ratio_hp = ne02 / ne12_hp;
    const int hkv_hp = head / gqa_ratio_hp;
    const int row = (sequence*ne01 + col)*ne02 + head;

    const float * lp_num_row = lp_num + int64_t(row)*parallel_blocks*D;
    const float2 * lp_meta_row = lp_meta + int64_t(row)*parallel_blocks;

    float kqmax = lp_meta_row[0].x;
    for (int part = 1; part < parallel_blocks; ++part) {
        kqmax = fmaxf(kqmax, lp_meta_row[part].x);
    }

    float numerator[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        numerator[r] = 0.0f;
    }
    float denom = 0.0f;
    if (lp_normalized && parallel_blocks == 1 &&
            isfinite(lp_meta_row[0].x) && isfinite(lp_meta_row[0].y) && lp_meta_row[0].y > 0.0f) {
        kqmax = lp_meta_row[0].x;
        denom = lp_meta_row[0].y;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            numerator[r] = denom * lp_num_row[lane + r*WARP_SIZE];
        }
    } else {
        for (int part = 0; part < parallel_blocks; ++part) {
            if (!isfinite(lp_meta_row[part].x) || !isfinite(lp_meta_row[part].y) || lp_meta_row[part].y <= 0.0f) {
                continue;
            }
            const float w = expf(lp_meta_row[part].x - kqmax);
            const float lp_factor = lp_normalized ? lp_meta_row[part].y : 1.0f;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                numerator[r] += w * lp_factor * lp_num_row[part*D + lane + r*WARP_SIZE];
            }
            denom += w * lp_meta_row[part].y;
        }
    }

    Q    += nb03*sequence + nb02*head + nb01*col;
    K_hp += nb13_hp*sequence + nb12_hp*hkv_hp;
    V_hp += nb23_hp*sequence + nb22_hp*hkv_hp;
    const char * mask_hp_seq = mask_hp + nb33_hp*sequence + nb31_hp*col;

    float q[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        q[r] = scale * ((const float *) Q)[lane + r*WARP_SIZE];
    }

    int key_begin = 0;
    int key_end = ne11_hp;
    bool hp_mask_all_valid = false;
    if constexpr (mask_skip) {
        if (compact_hp_mask) {
            const int sink_valid = min(hp_sink, min(ne11_hp, col + 1));
            int recent_valid = 0;
            if (col >= ne11_lp_eff) {
                recent_valid = min(ne11_hp - hp_sink, col - ne11_lp_eff + 1);
            }
            key_end = min(ne11_hp, sink_valid + max(0, recent_valid));
            hp_mask_all_valid = true;
        } else {
            while (key_begin < key_end) {
                const float mask_val = *(const float *) (mask_hp_seq + key_begin*sizeof(float));
                if (mask_val >= -1.0e30f) {
                    break;
                }
                ++key_begin;
            }
            while (key_end > key_begin) {
                const float mask_val = *(const float *) (mask_hp_seq + (key_end - 1)*sizeof(float));
                if (mask_val >= -1.0e30f) {
                    break;
                }
                --key_end;
            }
            hp_mask_all_valid = key_begin == 0 && key_end == ne11_hp &&
                ne11_hp > 0 &&
                *(const float *) (mask_hp_seq + 0*sizeof(float)) == 0.0f &&
                *(const float *) (mask_hp_seq + (ne11_hp - 1)*sizeof(float)) == 0.0f;
        }
    }

    for (int key = key_begin; key < key_end; ++key) {
        if constexpr (mask_skip) {
            const float mask_val = hp_mask_all_valid ? 0.0f : *(const float *) (mask_hp_seq + key*sizeof(float));
            if (!hp_mask_all_valid && mask_val < -1.0e30f) {
                    continue;
                }

            const half * K_h = (const half *) (K_hp + key*nb11_hp);
            float score = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                score += q[r] * __half2float(K_h[lane + r*WARP_SIZE]);
            }
            score = warp_reduce_sum(score);
            if constexpr (use_logit_softcap) {
                score = logit_softcap*tanhf(score);
            }
            score += mask_val;

            const float kqmax_new = fmaxf(kqmax, score + FATTN_KQ_MAX_OFFSET);
            const float old_scale = __expf(kqmax - kqmax_new);
            const float p = __expf(score - kqmax_new);
            const half * V_h = (const half *) (V_hp + key*nb21_hp);
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                numerator[r] = numerator[r]*old_scale + p*__half2float(V_h[lane + r*WARP_SIZE]);
            }
            denom = denom*old_scale + p;
            kqmax = kqmax_new;
        } else {
            const half * K_h = (const half *) (K_hp + key*nb11_hp);
            float score = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                score += q[r] * __half2float(K_h[lane + r*WARP_SIZE]);
            }
            score = warp_reduce_sum(score);
            if constexpr (use_logit_softcap) {
                score = logit_softcap*tanhf(score);
            }
            score += *(const float *) (mask_hp_seq + key*sizeof(float));

            const float kqmax_new = fmaxf(kqmax, score + FATTN_KQ_MAX_OFFSET);
            const float old_scale = __expf(kqmax - kqmax_new);
            const float p = __expf(score - kqmax_new);
            const half * V_h = (const half *) (V_hp + key*nb21_hp);
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                numerator[r] = numerator[r]*old_scale + p*__half2float(V_h[lane + r*WARP_SIZE]);
            }
            denom = denom*old_scale + p;
            kqmax = kqmax_new;
        }
    }

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        dst[int64_t(row)*D + lane + r*WARP_SIZE] = numerator[r] / denom;
    }
#else
    GGML_UNUSED_VARS(Q, K_hp, V_hp, mask_hp, lp_num, lp_meta, dst, scale, logit_softcap, parallel_blocks, lp_normalized,
        ne01, ne02, ne03, nb01, nb02, nb03,
        ne11_hp, ne12_hp, hp_sink, ne11_lp_eff, compact_hp_mask,
        nb11_hp, nb12_hp, nb13_hp, nb21_hp, nb22_hp, nb23_hp,
        nb31_hp, nb33_hp);
    NO_DEVICE_CODE;
#endif
}

template<int D, bool mask_skip, bool use_logit_softcap>
__launch_bounds__(WARP_SIZE, 1)
static __global__ void flash_attn_ext_mixed_oscar2_hp_combine_qtile4_warp(
        const char * __restrict__ Q,
        const char * __restrict__ K_hp,
        const char * __restrict__ V_hp,
        const char * __restrict__ mask_hp,
        const float * __restrict__ lp_num,
        const float2 * __restrict__ lp_meta,
        float * __restrict__ dst,
        const float scale,
        const float logit_softcap,
        const int32_t ne01, const int32_t ne02, const int32_t ne03,
        const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne11_hp, const int32_t ne12_hp,
        const int32_t hp_sink, const int32_t ne11_lp_eff,
        const int32_t compact_hp_mask,
        const int32_t nb11_hp, const int32_t nb12_hp, const int64_t nb13_hp,
        const int32_t nb21_hp, const int32_t nb22_hp, const int64_t nb23_hp,
        const int32_t nb31_hp, const int64_t nb33_hp) {
#ifdef FLASH_ATTN_AVAILABLE
    static_assert(D == 128, "mixed oscar2 qtile4 HP combine is specialized for D=128");

    const int lane = threadIdx.x;
    const int col0 = blockIdx.x*4;
    const int head = blockIdx.y;
    const int sequence = blockIdx.z;
    const int gqa_ratio_hp = ne02 / ne12_hp;
    const int hkv_hp = head / gqa_ratio_hp;

    Q    += nb03*sequence + nb02*head;
    K_hp += nb13_hp*sequence + nb12_hp*hkv_hp;
    V_hp += nb23_hp*sequence + nb22_hp*hkv_hp;

    float numerator[4][4];
    float denom[4];
    float kqmax[4];
    float q[4][4];
    int key_end_q[4];

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = col0 + j;
        const int row = (sequence*ne01 + col)*ne02 + head;
        if (col < ne01) {
            const float2 meta = lp_meta[row];
            kqmax[j] = meta.x;
            denom[j] = meta.y;
            if constexpr (mask_skip) {
                if (compact_hp_mask) {
                    const int sink_valid = min(hp_sink, min(ne11_hp, col + 1));
                    int recent_valid = 0;
                    if (col >= ne11_lp_eff) {
                        recent_valid = min(ne11_hp - hp_sink, col - ne11_lp_eff + 1);
                    }
                    key_end_q[j] = min(ne11_hp, sink_valid + max(0, recent_valid));
                } else {
                    key_end_q[j] = ne11_hp;
                }
            } else {
                key_end_q[j] = ne11_hp;
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int d = lane + r*WARP_SIZE;
                numerator[j][r] = denom[j] * lp_num[int64_t(row)*D + d];
                q[j][r] = scale * ((const float *) (Q + nb01*col))[d];
            }
        } else {
            kqmax[j] = -INFINITY;
            denom[j] = 0.0f;
            key_end_q[j] = 0;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                numerator[j][r] = 0.0f;
                q[j][r] = 0.0f;
            }
        }
    }

    for (int key = 0; key < ne11_hp; ++key) {
        const half * K_h = (const half *) (K_hp + key*nb11_hp);
        const half * V_h = (const half *) (V_hp + key*nb21_hp);

        float v[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            v[r] = __half2float(V_h[lane + r*WARP_SIZE]);
        }

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = col0 + j;
            if (col >= ne01) {
                continue;
            }

            float mask_val = 0.0f;
            if constexpr (mask_skip) {
                if (compact_hp_mask) {
                    if (key >= key_end_q[j]) {
                        continue;
                    }
                } else {
                    const char * mask_hp_seq = mask_hp + nb33_hp*sequence + nb31_hp*col;
                    mask_val = *(const float *) (mask_hp_seq + key*sizeof(float));
                    if (mask_val < -1.0e30f) {
                        continue;
                    }
                }
            } else {
                const char * mask_hp_seq = mask_hp + nb33_hp*sequence + nb31_hp*col;
                mask_val = *(const float *) (mask_hp_seq + key*sizeof(float));
            }

            float score = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                score += q[j][r] * __half2float(K_h[lane + r*WARP_SIZE]);
            }
            score = warp_reduce_sum(score);
            if constexpr (use_logit_softcap) {
                score = logit_softcap*tanhf(score);
            }
            score += mask_val;

            const float kqmax_new = fmaxf(kqmax[j], score + FATTN_KQ_MAX_OFFSET);
            const float old_scale = __expf(kqmax[j] - kqmax_new);
            const float p = __expf(score - kqmax_new);
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                numerator[j][r] = numerator[j][r]*old_scale + p*v[r];
            }
            denom[j] = denom[j]*old_scale + p;
            kqmax[j] = kqmax_new;
        }
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = col0 + j;
        if (col >= ne01) {
            continue;
        }
        const int row = (sequence*ne01 + col)*ne02 + head;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int d = lane + r*WARP_SIZE;
            dst[int64_t(row)*D + d] = numerator[j][r] / denom[j];
        }
    }
#else
    GGML_UNUSED_VARS(Q, K_hp, V_hp, mask_hp, lp_num, lp_meta, dst, scale, logit_softcap,
        ne01, ne02, ne03, nb01, nb02, nb03,
        ne11_hp, ne12_hp, hp_sink, ne11_lp_eff, compact_hp_mask,
        nb11_hp, nb12_hp, nb13_hp, nb21_hp, nb22_hp, nb23_hp,
        nb31_hp, nb33_hp);
    NO_DEVICE_CODE;
#endif
}

template<int D, int ncols>
__launch_bounds__(WARP_SIZE*ncols, 1)
static __global__ void flash_attn_ext_mixed_oscar2_copy_lp_warp(
        const float * __restrict__ lp_num,
        float * __restrict__ dst,
        const int32_t ne01, const int32_t ne02, const int32_t ne03) {
#ifdef FLASH_ATTN_AVAILABLE
    static_assert(D == 128, "mixed oscar2 copy LP is currently specialized for D=128");
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int col = blockIdx.x*ncols + warp;
    if (warp >= ncols || col >= ne01) {
        return;
    }
    const int head = blockIdx.y;
    const int sequence = blockIdx.z;
    const int row = (sequence*ne01 + col)*ne02 + head;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int d = lane + r*WARP_SIZE;
        dst[int64_t(row)*D + d] = lp_num[int64_t(row)*D + d];
    }
#else
    GGML_UNUSED_VARS(lp_num, dst, ne01, ne02, ne03);
    NO_DEVICE_CODE;
#endif
}

template<int D, int ncols, bool use_logit_softcap, bool mask_skip>
__launch_bounds__(WARP_SIZE*ncols, 1)
static __global__ void flash_attn_ext_mixed_oscar2_graph_combine_warp(
        const char * __restrict__ Q,
        const float * __restrict__ lp_out,
        const float2 * __restrict__ lp_meta,
        const char * __restrict__ K_hp,
        const char * __restrict__ V_hp,
        const char * __restrict__ mask_hp,
        float * __restrict__ dst,
        const float scale,
        const float logit_softcap,
        const int32_t ne01, const int32_t ne02, const int32_t ne03,
        const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne11_hp, const int32_t ne12_hp,
        const int32_t nb11_hp, const int32_t nb12_hp, const int64_t nb13_hp,
        const int32_t nb21_hp, const int32_t nb22_hp, const int64_t nb23_hp,
        const int32_t nb31_hp, const int64_t nb33_hp) {
#ifdef FLASH_ATTN_AVAILABLE
    static_assert(D == 128, "mixed graph combine warp is specialized for D=128");

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int col = blockIdx.x*ncols + warp;
    if (warp >= ncols || col >= ne01) {
        return;
    }

    const int head = blockIdx.y;
    const int sequence = blockIdx.z;
    const int gqa_ratio_hp = ne02 / ne12_hp;
    const int hkv_hp = head / gqa_ratio_hp;
    const int row = (sequence*ne01 + col)*ne02 + head;

    Q    += nb03*sequence + nb02*head + nb01*col;
    K_hp += nb13_hp*sequence + nb12_hp*hkv_hp;
    V_hp += nb23_hp*sequence + nb22_hp*hkv_hp;
    const char * mask_hp_seq = mask_hp + nb33_hp*sequence + nb31_hp*col;

    float numerator[4];
    const float2 lp_m_raw = lp_meta ? lp_meta[row] : make_float2(0.0f, 1.0f);
    const bool lp_valid = isfinite(lp_m_raw.x) && isfinite(lp_m_raw.y) && lp_m_raw.y > 0.0f;
    const float2 lp_m = lp_valid ? lp_m_raw : make_float2(-INFINITY, 0.0f);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const float lp_v = lp_valid ? lp_out[int64_t(row)*D + lane + r*WARP_SIZE] : 0.0f;
        numerator[r] = (lp_valid && isfinite(lp_v)) ? lp_m.y * lp_v : 0.0f;
    }
    float kqmax = lp_m.x;
    float denom = lp_m.y;

    float q[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        q[r] = scale * ((const float *) Q)[lane + r*WARP_SIZE];
    }

    int key_begin = 0;
    int key_end = ne11_hp;
    if constexpr (mask_skip) {
        while (key_begin < key_end && *(const float *) (mask_hp_seq + key_begin*sizeof(float)) < -1.0e30f) {
            ++key_begin;
        }
        while (key_end > key_begin && *(const float *) (mask_hp_seq + (key_end - 1)*sizeof(float)) < -1.0e30f) {
            --key_end;
        }
    }

    for (int key = key_begin; key < key_end; ++key) {
        const float mask_val = *(const float *) (mask_hp_seq + key*sizeof(float));
        if constexpr (mask_skip) {
            if (mask_val < -1.0e30f) {
                continue;
            }
        }
        const half * K_h = (const half *) (K_hp + key*nb11_hp);
        float score = 0.0f;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            score += q[r] * __half2float(K_h[lane + r*WARP_SIZE]);
        }
        score = warp_reduce_sum(score);
        if constexpr (use_logit_softcap) {
            score = logit_softcap*tanhf(score);
        }
        score += mask_val;

        const float kqmax_new = fmaxf(kqmax, score + FATTN_KQ_MAX_OFFSET);
        const float old_scale = expf(kqmax - kqmax_new);
        const float p = expf(score - kqmax_new);
        const half * V_h = (const half *) (V_hp + key*nb21_hp);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            numerator[r] = numerator[r]*old_scale + p*__half2float(V_h[lane + r*WARP_SIZE]);
        }
        denom = denom*old_scale + p;
        kqmax = kqmax_new;
    }

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        dst[int64_t(row)*D + lane + r*WARP_SIZE] = numerator[r] / denom;
    }
#else
    GGML_UNUSED_VARS(Q, lp_out, K_hp, V_hp, mask_hp, dst, scale, logit_softcap,
        ne01, ne02, ne03, nb01, nb02, nb03,
        ne11_hp, ne12_hp, nb11_hp, nb12_hp, nb13_hp, nb21_hp, nb22_hp, nb23_hp,
        nb31_hp, nb33_hp);
    NO_DEVICE_CODE;
#endif
}

template<int D, int ncols>
static void ggml_cuda_flash_attn_ext_mixed_vec_split_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    const ggml_tensor * Q       = dst->src[0];
    const ggml_tensor * K_lp    = dst->src[1];
    const ggml_tensor * V_lp    = dst->src[2];
    const ggml_tensor * mask_lp = dst->src[3];
    const ggml_tensor * K_hp    = dst->src[5];
    const ggml_tensor * V_hp    = dst->src[6];
    const ggml_tensor * mask_hp = dst->src[7];

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(ggml_cuda_info().devices[ctx.device].cc);
    const int nwarps = nthreads / WARP_SIZE;
    const int ntiles_x = (Q->ne[1] + ncols - 1) / ncols;
    const int ntiles_dst = ntiles_x * Q->ne[2] * Q->ne[3];
    const int ntiles_KV = (K_lp->ne[1] + nthreads - 1) / nthreads;

    int max_blocks_per_sm = 1;
    const dim3 block_dim(WARP_SIZE, nwarps, 1);
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, flash_attn_ext_mixed_oscar2_lp_raw_vec<D, ncols>, block_dim.x*block_dim.y, 0));
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    int parallel_blocks = std::max(1, std::min(max_blocks_per_sm, ntiles_KV));
    const int blocks_per_wave = nsm * max_blocks_per_sm;
    int nwaves_best = 0;
    int efficiency_percent_best = 0;
    for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
        const int nblocks_total = ntiles_dst * parallel_blocks_test;
        const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
        const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);
        if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
            break;
        }
        if (efficiency_percent > efficiency_percent_best) {
            nwaves_best = nwaves;
            efficiency_percent_best = efficiency_percent;
            parallel_blocks = parallel_blocks_test;
        }
    }
    if (const char * env = getenv("LLAMA_KV_MIXED_VEC_MAIN_PARTS")) {
        parallel_blocks = std::max(1, std::min(atoi(env), ntiles_KV));
    }

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float>  lp_num(pool);
    ggml_cuda_pool_alloc<float2> lp_meta(pool);
    lp_num.alloc(parallel_blocks*ggml_nelements(dst));
    lp_meta.alloc(parallel_blocks*ggml_nrows(dst));

    float scale = 1.0f;
    memcpy(&scale, dst->op_params, sizeof(float));

    const dim3 blocks_lp(ntiles_x, parallel_blocks, Q->ne[2]*Q->ne[3]);
    const auto params_lp = ggml_cuda_kernel_launch_params(blocks_lp, block_dim, 0, ctx.stream());
    ggml_cuda_kernel_launch(flash_attn_ext_mixed_oscar2_lp_raw_vec<D, ncols>, params_lp,
        (const char *) Q->data,
        (const char *) K_lp->data, (const char *) V_lp->data, (const char *) mask_lp->data,
        lp_num.ptr, lp_meta.ptr, scale,
        0.0f, 0, 0,
        Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K_lp->ne[1], K_lp->ne[2], K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
        V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
        mask_lp->nb[1], mask_lp->nb[3]);
    CUDA_CHECK(cudaGetLastError());

    const dim3 blocks_hp(ntiles_x, Q->ne[2], Q->ne[3]);
    const dim3 threads_hp(WARP_SIZE*ncols, 1, 1);
    const auto params_hp = ggml_cuda_kernel_launch_params(blocks_hp, threads_hp, 0, ctx.stream());
    ggml_cuda_kernel_launch((flash_attn_ext_mixed_oscar2_hp_combine_warp<D, ncols, false>), params_hp,
        (const char *) Q->data,
        (const char *) K_hp->data, (const char *) V_hp->data, (const char *) mask_hp->data,
        lp_num.ptr, lp_meta.ptr, (float *) dst->data, scale, 0.0f, parallel_blocks, 0,
        Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K_hp->ne[1], K_hp->ne[2], K_hp->nb[1], K_hp->nb[2], K_hp->nb[3],
        V_hp->nb[1], V_hp->nb[2], V_hp->nb[3],
        mask_hp->nb[1], mask_hp->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}

template<int D, int ncols>
static bool ggml_cuda_flash_attn_ext_mixed_vec_raw_launch_lp_tile2(
        const ggml_cuda_kernel_launch_params & params_lp,
        const ggml_tensor * Q,
        const ggml_tensor * K_lp,
        const ggml_tensor * V_lp,
        const ggml_tensor * mask_lp,
        float * dst,
        float2 * lp_meta,
        const float scale,
        const float logit_softcap,
        const uint3 ne01,
        const int ne11_lp_eff,
        const bool normalize_lp) {
    if (!normalize_lp) {
        return false;
    }

    if (logit_softcap == 0.0f) {
        ggml_cuda_kernel_launch((flash_attn_ext_vec<D, ncols, GGML_TYPE_OSCAR2_KV, GGML_TYPE_OSCAR2_KV, false, true, true>), params_lp,
            (const char *) Q->data,
            (const char *) K_lp->data, (const char *) V_lp->data,
            (const char *) mask_lp->data,
            nullptr, nullptr,
            dst, lp_meta,
            scale, 0.0f, 0.0f, 0.0f, uint32_t(Q->ne[2]), 0.0f,
            Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K_lp->ne[0], ne11_lp_eff, K_lp->ne[2], K_lp->ne[3], K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
            V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
            mask_lp->ne[1], mask_lp->ne[2], mask_lp->ne[3], mask_lp->nb[1], mask_lp->nb[2], mask_lp->nb[3]);
    } else {
        ggml_cuda_kernel_launch((flash_attn_ext_vec<D, ncols, GGML_TYPE_OSCAR2_KV, GGML_TYPE_OSCAR2_KV, true, true, true>), params_lp,
            (const char *) Q->data,
            (const char *) K_lp->data, (const char *) V_lp->data,
            (const char *) mask_lp->data,
            nullptr, nullptr,
            dst, lp_meta,
            scale, 0.0f, 0.0f, 0.0f, uint32_t(Q->ne[2]), logit_softcap,
            Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K_lp->ne[0], ne11_lp_eff, K_lp->ne[2], K_lp->ne[3], K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
            V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
            mask_lp->ne[1], mask_lp->ne[2], mask_lp->ne[3], mask_lp->nb[1], mask_lp->nb[2], mask_lp->nb[3]);
    }
    return true;
}

template<int D, int ncols>
static void ggml_cuda_flash_attn_ext_mixed_vec_raw_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    const ggml_tensor * Q       = dst->src[0];
    const ggml_tensor * K_lp    = dst->src[1];
    const ggml_tensor * V_lp    = dst->src[2];
    const ggml_tensor * mask_lp = dst->src[3];
    const ggml_tensor * K_hp    = dst->src[5];
    const ggml_tensor * V_hp    = dst->src[6];
    const ggml_tensor * mask_hp = dst->src[7];
    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(K_lp->type == GGML_TYPE_OSCAR2_KV);
    GGML_ASSERT(V_lp->type == GGML_TYPE_OSCAR2_KV);
    GGML_ASSERT(mask_lp->type == GGML_TYPE_F16);
    GGML_ASSERT(K_hp->type == GGML_TYPE_F16);
    GGML_ASSERT(V_hp->type == GGML_TYPE_F16);
    GGML_ASSERT(mask_hp->type == GGML_TYPE_F32);

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(ggml_cuda_info().devices[ctx.device].cc);
    const int nwarps = nthreads / WARP_SIZE;
    const int ntiles_x = (Q->ne[1] + ncols - 1) / ncols;
    const int ntiles_dst = ntiles_x * Q->ne[2] * Q->ne[3];
    int hp_sink = 0;
    if (const char * env = getenv("LLAMA_KV_HP_SINK")) {
        hp_sink = std::max(0, atoi(env));
    }
    const int hp_recent_tail = std::max<int>(0, K_hp->ne[1] - std::min<int64_t>(hp_sink, K_hp->ne[1]));
    const int ne11_lp_eff = std::max<int>(0, K_lp->ne[1] - hp_recent_tail);
    const int ntiles_KV = std::max(1, (ne11_lp_eff + nthreads - 1) / nthreads);

    int max_blocks_per_sm = 1;
    const dim3 block_dim(WARP_SIZE, nwarps, 1);

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, flash_attn_ext_vec<D, ncols, GGML_TYPE_OSCAR2_KV, GGML_TYPE_OSCAR2_KV, false, true>,
        block_dim.x*block_dim.y, 0));
    GGML_ASSERT(max_blocks_per_sm > 0);

    int parallel_blocks = std::max(1, std::min(max_blocks_per_sm, ntiles_KV));
    const int blocks_per_wave = ggml_cuda_info().devices[ctx.device].nsm * max_blocks_per_sm;
    int nwaves_best = 0;
    int efficiency_percent_best = 0;
    for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
        const int nblocks_total = ntiles_dst * parallel_blocks_test;
        const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
        const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);
        if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
            break;
        }
        if (efficiency_percent > efficiency_percent_best) {
            nwaves_best = nwaves;
            efficiency_percent_best = efficiency_percent;
            parallel_blocks = parallel_blocks_test;
        }
    }
    const bool normalize_lp_enabled = []() {
        const char * env = getenv("LLAMA_KV_MIXED_VEC_RAW_NORM_LP");
        return env == nullptr || (env[0] != '\0' && env[0] != '0');
    }();
    if (const char * env = getenv("LLAMA_KV_MIXED_VEC_RAW_PARTS")) {
        if (strcmp(env, "auto") != 0) {
            parallel_blocks = std::max(1, std::min(atoi(env), ntiles_KV));
        }
    } else if (normalize_lp_enabled) {
        parallel_blocks = 1;
    }
    const bool normalize_lp = parallel_blocks == 1 && normalize_lp_enabled;
    if (const char * env = getenv("LLAMA_KV_MIXED_VEC_RAW_DEBUG")) {
        if (env[0] != '\0' && env[0] != '0') {
            fprintf(stderr, "mixed_raw: q=%lld k_lp=%lld k_hp=%lld hp_sink=%d tail=%d ne11_lp_eff=%d ntiles_kv=%d parts=%d ncols=%d norm_lp=%d\n",
                    (long long) Q->ne[1], (long long) K_lp->ne[1], (long long) K_hp->ne[1],
                    hp_sink, hp_recent_tail, ne11_lp_eff, ntiles_KV, parallel_blocks, ncols,
                    normalize_lp ? 1 : 0);
        }
    }

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float>  lp_num(pool);
    ggml_cuda_pool_alloc<float2> lp_meta(pool);
    if (!normalize_lp) {
        lp_num.alloc(parallel_blocks*ggml_nelements(dst));
    }
    lp_meta.alloc(parallel_blocks*ggml_nrows(dst));

    float scale = 1.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);
    const dim3 blocks_lp(ntiles_x, parallel_blocks, Q->ne[2]*Q->ne[3]);
    const auto params_lp = ggml_cuda_kernel_launch_params(blocks_lp, block_dim, 0, ctx.stream());
    const bool use_dedicated_lp = getenv("LLAMA_KV_MIXED_VEC_RAW_DEDICATED_LP") != nullptr;
    const bool use_lp_tile2 = getenv("LLAMA_KV_MIXED_VEC_RAW_LP_TILE2") != nullptr;
    const bool lp_diag_no_v = getenv("LLAMA_KV_MIXED_VEC_LP_DIAG_NO_V") != nullptr;
    const bool raw_debug = getenv("LLAMA_KV_MIXED_VEC_RAW_DEBUG") != nullptr;
    fattn_vec_oscar2_two_tier_params two_tier = { 1, 0, ne11_lp_eff, ne11_lp_eff, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1.0f };
    if (const char * env = getenv("LLAMA_KV_MIXED_VEC_LP_TWO_TIER_STRIDE")) {
        const int stride = std::max(1, atoi(env));
        if (stride > 1 && normalize_lp) {
            int tail = 0;
            if (const char * env_tail = getenv("LLAMA_KV_MIXED_VEC_LP_TWO_TIER_TAIL")) {
                tail = std::max(0, atoi(env_tail));
            }
            tail = std::min(tail, ne11_lp_eff);
            const int prefix_len = ne11_lp_eff - tail;
            const int prefix_sampled = (prefix_len + stride - 1) / stride;
            int mode = 0;
            if (const char * env_mode = getenv("LLAMA_KV_MIXED_VEC_LP_TWO_TIER_MODE")) {
                if (strcmp(env_mode, "end") == 0) {
                    mode = stride - 1;
                } else if (strcmp(env_mode, "mid") == 0) {
                    mode = stride / 2;
                }
            }
            int weighted = 0;
            if (const char * env_weighted = getenv("LLAMA_KV_MIXED_VEC_LP_TWO_TIER_WEIGHTED")) {
                weighted = env_weighted[0] != '\0' && env_weighted[0] != '0';
            }
            int mid_tokens = 0;
            int far_stride = 1;
            int far_mode = 0;
            int far_sampled = 0;
            int mid_sampled = 0;
            int v_avg = 0;
            if (const char * env_v_avg = getenv("LLAMA_KV_MIXED_VEC_LP_TWO_TIER_V_AVG")) {
                v_avg = env_v_avg[0] != '\0' && env_v_avg[0] != '0';
            }
            int k_avg = 0;
            if (const char * env_k_avg = getenv("LLAMA_KV_MIXED_VEC_LP_TWO_TIER_K_AVG")) {
                k_avg = env_k_avg[0] != '\0' && env_k_avg[0] != '0';
            }
            float weight_exp = 1.0f;
            if (const char * env_weight_exp = getenv("LLAMA_KV_MIXED_VEC_LP_TWO_TIER_WEIGHT_EXP")) {
                weight_exp = std::max(0.0f, (float) atof(env_weight_exp));
            }
            if (const char * env_far_stride = getenv("LLAMA_KV_MIXED_VEC_LP_THREE_TIER_FAR_STRIDE")) {
                far_stride = std::max(1, atoi(env_far_stride));
            }
            if (far_stride > 1) {
                if (const char * env_mid = getenv("LLAMA_KV_MIXED_VEC_LP_THREE_TIER_MID_TOKENS")) {
                    mid_tokens = std::max(0, atoi(env_mid));
                }
                mid_tokens = std::min(mid_tokens, prefix_len);
                const int far_len = prefix_len - mid_tokens;
                far_sampled = (far_len + far_stride - 1) / far_stride;
                mid_sampled = (mid_tokens + stride - 1) / stride;
                far_mode = far_stride - 1;
                if (const char * env_far_mode = getenv("LLAMA_KV_MIXED_VEC_LP_THREE_TIER_FAR_MODE")) {
                    if (strcmp(env_far_mode, "mid") == 0) {
                        far_mode = far_stride / 2;
                    } else if (strcmp(env_far_mode, "start") == 0) {
                        far_mode = 0;
                    }
                }
            }
            const int effective_len = far_stride > 1 && mid_tokens > 0 ? far_sampled + mid_sampled + tail : prefix_sampled + tail;
            two_tier = { stride, tail, ne11_lp_eff, effective_len, mode, weighted,
                mid_tokens, far_stride, far_mode, far_sampled, mid_sampled, v_avg, k_avg, weight_exp };
        }
    }
    const int ne11_lp_launch = two_tier.effective_len;
    if (two_tier.stride > 1) {
        CUDA_CHECK(cudaMemcpyToSymbolAsync(
            fattn_vec_oscar2_two_tier, &two_tier, sizeof(two_tier), 0, cudaMemcpyHostToDevice, ctx.stream()));
        if (raw_debug) {
            fprintf(stderr, "mixed_raw: lp_two_tier=1 stride=%d tail=%d original=%d effective=%d mode=%d weighted=%d weight_exp=%.3f mid_tokens=%d far_stride=%d far_sampled=%d mid_sampled=%d v_avg=%d k_avg=%d\n",
                two_tier.stride, two_tier.tail, two_tier.original_len, two_tier.effective_len,
                two_tier.mode, two_tier.weighted, two_tier.weight_exp, two_tier.mid_tokens, two_tier.far_stride,
                two_tier.far_sampled, two_tier.mid_sampled, two_tier.v_avg, two_tier.k_avg);
        }
    }
    if (ggml_cuda_flash_attn_ext_mixed_oscar2_v2_lp_try(
            ctx, Q, K_lp, V_lp, mask_lp,
            normalize_lp ? (float *) dst->data : lp_num.ptr, lp_meta.ptr,
            scale, logit_softcap, ne11_lp_launch, parallel_blocks, normalize_lp, lp_diag_no_v, ncols)) {
        if (raw_debug) {
            fprintf(stderr, "mixed_raw: oscar2_v2_lp=1 ncols=%d norm_lp=%d\n", ncols, normalize_lp ? 1 : 0);
        }
    } else if (use_dedicated_lp) {
        ggml_cuda_kernel_launch(flash_attn_ext_mixed_oscar2_lp_raw_vec<D, ncols>, params_lp,
            (const char *) Q->data,
            (const char *) K_lp->data, (const char *) V_lp->data,
            (const char *) mask_lp->data,
            normalize_lp ? (float *) dst->data : lp_num.ptr, lp_meta.ptr, scale, logit_softcap,
            normalize_lp ? 1 : 0, 1,
            Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            ne11_lp_launch, K_lp->ne[2], K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
            V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
            mask_lp->nb[1], mask_lp->nb[3]);
    } else if (use_lp_tile2 && ggml_cuda_flash_attn_ext_mixed_vec_raw_launch_lp_tile2<D, ncols>(
            params_lp, Q, K_lp, V_lp, mask_lp,
            normalize_lp ? (float *) dst->data : lp_num.ptr, lp_meta.ptr,
            scale, logit_softcap, ne01, ne11_lp_launch, normalize_lp)) {
        if (raw_debug) {
            fprintf(stderr, "mixed_raw: lp_tile2=1 path=generic_stub ncols=%d norm_lp=%d\n", ncols, normalize_lp ? 1 : 0);
        }
        // env-gated LP_TILE2 path handled by helper
    } else if (logit_softcap == 0.0f && normalize_lp) {
        if (two_tier.stride > 1) {
            ggml_cuda_kernel_launch((flash_attn_ext_vec<D, ncols, GGML_TYPE_OSCAR2_KV, GGML_TYPE_OSCAR2_KV, false, true, true, true>), params_lp,
                (const char *) Q->data,
                (const char *) K_lp->data, (const char *) V_lp->data,
                (const char *) mask_lp->data,
                nullptr, nullptr,
                (float *) dst->data, lp_meta.ptr,
                scale, 0.0f, 0.0f, 0.0f, uint32_t(Q->ne[2]), 0.0f,
                Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
                K_lp->ne[0], ne11_lp_launch, K_lp->ne[2], K_lp->ne[3], K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
                V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
                mask_lp->ne[1], mask_lp->ne[2], mask_lp->ne[3], mask_lp->nb[1], mask_lp->nb[2], mask_lp->nb[3]);
        } else if (lp_diag_no_v) {
            ggml_cuda_kernel_launch((flash_attn_ext_vec<D, ncols, GGML_TYPE_OSCAR2_KV, GGML_TYPE_OSCAR2_KV, false, true, true, false, true>), params_lp,
            (const char *) Q->data,
            (const char *) K_lp->data, (const char *) V_lp->data,
            (const char *) mask_lp->data,
            nullptr, nullptr,
            (float *) dst->data, lp_meta.ptr,
            scale, 0.0f, 0.0f, 0.0f, uint32_t(Q->ne[2]), 0.0f,
            Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K_lp->ne[0], ne11_lp_launch, K_lp->ne[2], K_lp->ne[3], K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
            V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
            mask_lp->ne[1], mask_lp->ne[2], mask_lp->ne[3], mask_lp->nb[1], mask_lp->nb[2], mask_lp->nb[3]);
        } else {
            ggml_cuda_kernel_launch((flash_attn_ext_vec<D, ncols, GGML_TYPE_OSCAR2_KV, GGML_TYPE_OSCAR2_KV, false, true, true>), params_lp,
            (const char *) Q->data,
            (const char *) K_lp->data, (const char *) V_lp->data,
            (const char *) mask_lp->data,
            nullptr, nullptr,
            (float *) dst->data, lp_meta.ptr,
            scale, 0.0f, 0.0f, 0.0f, uint32_t(Q->ne[2]), 0.0f,
            Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K_lp->ne[0], ne11_lp_launch, K_lp->ne[2], K_lp->ne[3], K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
            V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
            mask_lp->ne[1], mask_lp->ne[2], mask_lp->ne[3], mask_lp->nb[1], mask_lp->nb[2], mask_lp->nb[3]);
        }
    } else if (logit_softcap == 0.0f) {
        if (two_tier.stride > 1) {
            ggml_cuda_kernel_launch((flash_attn_ext_vec<D, ncols, GGML_TYPE_OSCAR2_KV, GGML_TYPE_OSCAR2_KV, false, true, false, true>), params_lp,
                (const char *) Q->data,
                (const char *) K_lp->data, (const char *) V_lp->data,
                (const char *) mask_lp->data,
                nullptr, nullptr,
                lp_num.ptr, lp_meta.ptr,
                scale, 0.0f, 0.0f, 0.0f, uint32_t(Q->ne[2]), 0.0f,
                Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
                K_lp->ne[0], ne11_lp_launch, K_lp->ne[2], K_lp->ne[3], K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
                V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
                mask_lp->ne[1], mask_lp->ne[2], mask_lp->ne[3], mask_lp->nb[1], mask_lp->nb[2], mask_lp->nb[3]);
        } else {
            ggml_cuda_kernel_launch((flash_attn_ext_vec<D, ncols, GGML_TYPE_OSCAR2_KV, GGML_TYPE_OSCAR2_KV, false, true, false>), params_lp,
            (const char *) Q->data,
            (const char *) K_lp->data, (const char *) V_lp->data,
            (const char *) mask_lp->data,
            nullptr, nullptr,
            lp_num.ptr, lp_meta.ptr,
            scale, 0.0f, 0.0f, 0.0f, uint32_t(Q->ne[2]), 0.0f,
            Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K_lp->ne[0], ne11_lp_launch, K_lp->ne[2], K_lp->ne[3], K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
            V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
            mask_lp->ne[1], mask_lp->ne[2], mask_lp->ne[3], mask_lp->nb[1], mask_lp->nb[2], mask_lp->nb[3]);
        }
    } else if (normalize_lp) {
        if (two_tier.stride > 1) {
            ggml_cuda_kernel_launch((flash_attn_ext_vec<D, ncols, GGML_TYPE_OSCAR2_KV, GGML_TYPE_OSCAR2_KV, true, true, true, true>), params_lp,
                (const char *) Q->data,
                (const char *) K_lp->data, (const char *) V_lp->data,
                (const char *) mask_lp->data,
                nullptr, nullptr,
                (float *) dst->data, lp_meta.ptr,
                scale, 0.0f, 0.0f, 0.0f, uint32_t(Q->ne[2]), logit_softcap,
                Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
                K_lp->ne[0], ne11_lp_launch, K_lp->ne[2], K_lp->ne[3], K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
                V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
                mask_lp->ne[1], mask_lp->ne[2], mask_lp->ne[3], mask_lp->nb[1], mask_lp->nb[2], mask_lp->nb[3]);
        } else if (lp_diag_no_v) {
            ggml_cuda_kernel_launch((flash_attn_ext_vec<D, ncols, GGML_TYPE_OSCAR2_KV, GGML_TYPE_OSCAR2_KV, true, true, true, false, true>), params_lp,
            (const char *) Q->data,
            (const char *) K_lp->data, (const char *) V_lp->data,
            (const char *) mask_lp->data,
            nullptr, nullptr,
            (float *) dst->data, lp_meta.ptr,
            scale, 0.0f, 0.0f, 0.0f, uint32_t(Q->ne[2]), logit_softcap,
            Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K_lp->ne[0], ne11_lp_launch, K_lp->ne[2], K_lp->ne[3], K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
            V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
            mask_lp->ne[1], mask_lp->ne[2], mask_lp->ne[3], mask_lp->nb[1], mask_lp->nb[2], mask_lp->nb[3]);
        } else {
            ggml_cuda_kernel_launch((flash_attn_ext_vec<D, ncols, GGML_TYPE_OSCAR2_KV, GGML_TYPE_OSCAR2_KV, true, true, true>), params_lp,
            (const char *) Q->data,
            (const char *) K_lp->data, (const char *) V_lp->data,
            (const char *) mask_lp->data,
            nullptr, nullptr,
            (float *) dst->data, lp_meta.ptr,
            scale, 0.0f, 0.0f, 0.0f, uint32_t(Q->ne[2]), logit_softcap,
            Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K_lp->ne[0], ne11_lp_launch, K_lp->ne[2], K_lp->ne[3], K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
            V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
            mask_lp->ne[1], mask_lp->ne[2], mask_lp->ne[3], mask_lp->nb[1], mask_lp->nb[2], mask_lp->nb[3]);
        }
    } else {
        if (two_tier.stride > 1) {
            ggml_cuda_kernel_launch((flash_attn_ext_vec<D, ncols, GGML_TYPE_OSCAR2_KV, GGML_TYPE_OSCAR2_KV, true, true, false, true>), params_lp,
                (const char *) Q->data,
                (const char *) K_lp->data, (const char *) V_lp->data,
                (const char *) mask_lp->data,
                nullptr, nullptr,
                lp_num.ptr, lp_meta.ptr,
                scale, 0.0f, 0.0f, 0.0f, uint32_t(Q->ne[2]), logit_softcap,
                Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
                K_lp->ne[0], ne11_lp_launch, K_lp->ne[2], K_lp->ne[3], K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
                V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
                mask_lp->ne[1], mask_lp->ne[2], mask_lp->ne[3], mask_lp->nb[1], mask_lp->nb[2], mask_lp->nb[3]);
        } else {
            ggml_cuda_kernel_launch((flash_attn_ext_vec<D, ncols, GGML_TYPE_OSCAR2_KV, GGML_TYPE_OSCAR2_KV, true, true, false>), params_lp,
            (const char *) Q->data,
            (const char *) K_lp->data, (const char *) V_lp->data,
            (const char *) mask_lp->data,
            nullptr, nullptr,
            lp_num.ptr, lp_meta.ptr,
            scale, 0.0f, 0.0f, 0.0f, uint32_t(Q->ne[2]), logit_softcap,
            Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K_lp->ne[0], ne11_lp_launch, K_lp->ne[2], K_lp->ne[3], K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
            V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
            mask_lp->ne[1], mask_lp->ne[2], mask_lp->ne[3], mask_lp->nb[1], mask_lp->nb[2], mask_lp->nb[3]);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    const char * lp_only_diag_env = getenv("LLAMA_KV_MIXED_VEC_RAW_LP_ONLY_DIAG");
    if (normalize_lp && lp_only_diag_env && lp_only_diag_env[0] != '\0' && lp_only_diag_env[0] != '0') {
        return;
    }
    const dim3 blocks_hp(ntiles_x, Q->ne[2], Q->ne[3]);
    const dim3 threads_hp(WARP_SIZE*ncols, 1, 1);
    const auto params_hp = ggml_cuda_kernel_launch_params(blocks_hp, threads_hp, 0, ctx.stream());
    const char * copy_lp_diag_env = getenv("LLAMA_KV_MIXED_VEC_COPY_LP_DIAG");
    if (normalize_lp && copy_lp_diag_env && copy_lp_diag_env[0] != '\0' && copy_lp_diag_env[0] != '0') {
        ggml_cuda_kernel_launch((flash_attn_ext_mixed_oscar2_copy_lp_warp<D, ncols>), params_hp,
            (const float *) dst->data, (float *) dst->data,
            Q->ne[1], Q->ne[2], Q->ne[3]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    bool hp_mask_skip = true;
    if (const char * env = getenv("LLAMA_KV_MIXED_VEC_HP_MASK_SKIP")) {
        hp_mask_skip = env[0] != '\0' && env[0] != '0';
    }
    bool compact_hp_mask = true;
    if (const char * env = getenv("LLAMA_KV_MIXED_VEC_HP_COMPACT_MASK")) {
        compact_hp_mask = env[0] != '\0' && env[0] != '0';
    }
    bool hp_qtile4_enabled = true;
    if (const char * env = getenv("LLAMA_KV_MIXED_VEC_HP_QTILE")) {
        hp_qtile4_enabled = env[0] != '\0' && env[0] != '0';
    }
    const bool hp_qtile4 = normalize_lp && parallel_blocks == 1 && ncols == 4 && hp_qtile4_enabled;
    if (hp_qtile4) {
        const dim3 threads_qtile(WARP_SIZE, 1, 1);
        const auto params_qtile = ggml_cuda_kernel_launch_params(blocks_hp, threads_qtile, 0, ctx.stream());
        if (hp_mask_skip && logit_softcap != 0.0f) {
            ggml_cuda_kernel_launch((flash_attn_ext_mixed_oscar2_hp_combine_qtile4_warp<D, true, true>), params_qtile,
                (const char *) Q->data,
                (const char *) K_hp->data, (const char *) V_hp->data, (const char *) mask_hp->data,
                (const float *) dst->data, lp_meta.ptr,
                (float *) dst->data, scale, logit_softcap,
                Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
                K_hp->ne[1], K_hp->ne[2], hp_sink, ne11_lp_eff, compact_hp_mask ? 1 : 0,
                K_hp->nb[1], K_hp->nb[2], K_hp->nb[3],
                V_hp->nb[1], V_hp->nb[2], V_hp->nb[3],
                mask_hp->nb[1], mask_hp->nb[3]);
        } else if (hp_mask_skip) {
            ggml_cuda_kernel_launch((flash_attn_ext_mixed_oscar2_hp_combine_qtile4_warp<D, true, false>), params_qtile,
                (const char *) Q->data,
                (const char *) K_hp->data, (const char *) V_hp->data, (const char *) mask_hp->data,
                (const float *) dst->data, lp_meta.ptr,
                (float *) dst->data, scale, logit_softcap,
                Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
                K_hp->ne[1], K_hp->ne[2], hp_sink, ne11_lp_eff, compact_hp_mask ? 1 : 0,
                K_hp->nb[1], K_hp->nb[2], K_hp->nb[3],
                V_hp->nb[1], V_hp->nb[2], V_hp->nb[3],
                mask_hp->nb[1], mask_hp->nb[3]);
        } else if (logit_softcap != 0.0f) {
            ggml_cuda_kernel_launch((flash_attn_ext_mixed_oscar2_hp_combine_qtile4_warp<D, false, true>), params_qtile,
                (const char *) Q->data,
                (const char *) K_hp->data, (const char *) V_hp->data, (const char *) mask_hp->data,
                (const float *) dst->data, lp_meta.ptr,
                (float *) dst->data, scale, logit_softcap,
                Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
                K_hp->ne[1], K_hp->ne[2], hp_sink, ne11_lp_eff, compact_hp_mask ? 1 : 0,
                K_hp->nb[1], K_hp->nb[2], K_hp->nb[3],
                V_hp->nb[1], V_hp->nb[2], V_hp->nb[3],
                mask_hp->nb[1], mask_hp->nb[3]);
        } else {
            ggml_cuda_kernel_launch((flash_attn_ext_mixed_oscar2_hp_combine_qtile4_warp<D, false, false>), params_qtile,
                (const char *) Q->data,
                (const char *) K_hp->data, (const char *) V_hp->data, (const char *) mask_hp->data,
                (const float *) dst->data, lp_meta.ptr,
                (float *) dst->data, scale, logit_softcap,
                Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
                K_hp->ne[1], K_hp->ne[2], hp_sink, ne11_lp_eff, compact_hp_mask ? 1 : 0,
                K_hp->nb[1], K_hp->nb[2], K_hp->nb[3],
                V_hp->nb[1], V_hp->nb[2], V_hp->nb[3],
                mask_hp->nb[1], mask_hp->nb[3]);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (hp_mask_skip && logit_softcap != 0.0f) {
        ggml_cuda_kernel_launch((flash_attn_ext_mixed_oscar2_hp_combine_warp<D, ncols, true, true>), params_hp,
            (const char *) Q->data,
            (const char *) K_hp->data, (const char *) V_hp->data, (const char *) mask_hp->data,
            normalize_lp ? (const float *) dst->data : lp_num.ptr, lp_meta.ptr,
            (float *) dst->data, scale, logit_softcap, parallel_blocks, normalize_lp ? 1 : 0,
            Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K_hp->ne[1], K_hp->ne[2], hp_sink, ne11_lp_eff, compact_hp_mask ? 1 : 0,
            K_hp->nb[1], K_hp->nb[2], K_hp->nb[3],
            V_hp->nb[1], V_hp->nb[2], V_hp->nb[3],
            mask_hp->nb[1], mask_hp->nb[3]);
    } else if (hp_mask_skip) {
        ggml_cuda_kernel_launch((flash_attn_ext_mixed_oscar2_hp_combine_warp<D, ncols, true, false>), params_hp,
            (const char *) Q->data,
            (const char *) K_hp->data, (const char *) V_hp->data, (const char *) mask_hp->data,
            normalize_lp ? (const float *) dst->data : lp_num.ptr, lp_meta.ptr,
            (float *) dst->data, scale, logit_softcap, parallel_blocks, normalize_lp ? 1 : 0,
            Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K_hp->ne[1], K_hp->ne[2], hp_sink, ne11_lp_eff, compact_hp_mask ? 1 : 0,
            K_hp->nb[1], K_hp->nb[2], K_hp->nb[3],
            V_hp->nb[1], V_hp->nb[2], V_hp->nb[3],
            mask_hp->nb[1], mask_hp->nb[3]);
    } else if (logit_softcap != 0.0f) {
        ggml_cuda_kernel_launch((flash_attn_ext_mixed_oscar2_hp_combine_warp<D, ncols, false, true>), params_hp,
            (const char *) Q->data,
            (const char *) K_hp->data, (const char *) V_hp->data, (const char *) mask_hp->data,
            normalize_lp ? (const float *) dst->data : lp_num.ptr, lp_meta.ptr,
            (float *) dst->data, scale, logit_softcap, parallel_blocks, normalize_lp ? 1 : 0,
            Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K_hp->ne[1], K_hp->ne[2], hp_sink, ne11_lp_eff, compact_hp_mask ? 1 : 0,
            K_hp->nb[1], K_hp->nb[2], K_hp->nb[3],
            V_hp->nb[1], V_hp->nb[2], V_hp->nb[3],
            mask_hp->nb[1], mask_hp->nb[3]);
    } else {
        ggml_cuda_kernel_launch((flash_attn_ext_mixed_oscar2_hp_combine_warp<D, ncols, false, false>), params_hp,
            (const char *) Q->data,
            (const char *) K_hp->data, (const char *) V_hp->data, (const char *) mask_hp->data,
            normalize_lp ? (const float *) dst->data : lp_num.ptr, lp_meta.ptr,
            (float *) dst->data, scale, logit_softcap, parallel_blocks, normalize_lp ? 1 : 0,
            Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K_hp->ne[1], K_hp->ne[2], hp_sink, ne11_lp_eff, compact_hp_mask ? 1 : 0,
            K_hp->nb[1], K_hp->nb[2], K_hp->nb[3],
            V_hp->nb[1], V_hp->nb[2], V_hp->nb[3],
            mask_hp->nb[1], mask_hp->nb[3]);
    }
    CUDA_CHECK(cudaGetLastError());
}

template<int D>
__launch_bounds__(D, 1)
static __global__ void flash_attn_ext_mixed_oscar2_graph_combine(
        const char * __restrict__ Q,
        const char * __restrict__ K_lp,
        const float * __restrict__ lp_out,
        const char * __restrict__ mask_lp,
        const char * __restrict__ K_hp,
        const char * __restrict__ V_hp,
        const char * __restrict__ mask_hp,
        float * __restrict__ dst,
        const float scale,
        const int32_t ne01, const int32_t ne02, const int32_t ne03,
        const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne11_lp, const int32_t ne12_lp,
        const int32_t nb11_lp, const int32_t nb12_lp, const int64_t nb13_lp,
        const int32_t ne11_hp, const int32_t ne12_hp,
        const int32_t nb11_hp, const int32_t nb12_hp, const int64_t nb13_hp,
        const int32_t nb21_hp, const int32_t nb22_hp, const int64_t nb23_hp,
        const int32_t nb31_lp, const int64_t nb33_lp,
        const int32_t nb31_hp, const int64_t nb33_hp) {
#ifdef FLASH_ATTN_AVAILABLE
    static_assert(D == 128, "mixed graph combine is currently specialized for D=128");
    GGML_UNUSED(K_lp);
    GGML_UNUSED(mask_lp);
    GGML_UNUSED(ne11_lp);
    GGML_UNUSED(ne12_lp);
    GGML_UNUSED(nb11_lp);
    GGML_UNUSED(nb12_lp);
    GGML_UNUSED(nb13_lp);
    GGML_UNUSED(nb31_lp);
    GGML_UNUSED(nb33_lp);

    const int tid = threadIdx.x;
    const int col = blockIdx.x;
    const int head = blockIdx.y;
    const int sequence = blockIdx.z;
    const int gqa_ratio_lp = ne02 / ne12_lp;
    const int gqa_ratio_hp = ne02 / ne12_hp;
    GGML_UNUSED(gqa_ratio_lp);
    const int hkv_hp = head / gqa_ratio_hp;
    const int row = (sequence*ne01 + col)*ne02 + head;

    Q    += nb03*sequence + nb02*head + nb01*col;
    K_hp += nb13_hp*sequence + nb12_hp*hkv_hp;
    V_hp += nb23_hp*sequence + nb22_hp*hkv_hp;
    const char * mask_hp_seq = mask_hp + nb33_hp*sequence + nb31_hp*col;

    __shared__ float scratch[WARP_SIZE];

    float kqmax = 0.0f;
    float denom = 1.0f;
    float numerator = lp_out[int64_t(row)*D + tid];

    const float q = scale * ((const float *) Q)[tid];
    for (int key = 0; key < ne11_hp; ++key) {
        const half * K_h = (const half *) (K_hp + key*nb11_hp);
        float partial = q * __half2float(K_h[tid]);
        partial = warp_reduce_sum(partial);

        if ((tid & (WARP_SIZE - 1)) == 0) {
            scratch[tid / WARP_SIZE] = partial;
        }
        __syncthreads();

        float score = tid < D/WARP_SIZE ? scratch[tid] : 0.0f;
        if (tid < WARP_SIZE) {
            score = warp_reduce_sum(score);
            if (tid == 0) {
                scratch[0] = score;
            }
        }
        __syncthreads();

        score = scratch[0] + *(const float *) (mask_hp_seq + key*sizeof(float));
        const float kqmax_new = fmaxf(kqmax, score + FATTN_KQ_MAX_OFFSET);
        const float old_scale = expf(kqmax - kqmax_new);
        const float p = expf(score - kqmax_new);
        const half * V_h = (const half *) (V_hp + key*nb21_hp);
        numerator = numerator*old_scale + p*__half2float(V_h[tid]);
        denom = denom*old_scale + p;
        kqmax = kqmax_new;
        __syncthreads();
    }

    dst[int64_t(row)*D + tid] = numerator / denom;
#else
    GGML_UNUSED_VARS(Q, K_lp, lp_out, mask_lp, K_hp, V_hp, mask_hp, dst, scale,
        ne01, ne02, ne03, nb01, nb02, nb03,
        ne11_lp, ne12_lp, nb11_lp, nb12_lp, nb13_lp,
        ne11_hp, ne12_hp, nb11_hp, nb12_hp, nb13_hp, nb21_hp, nb22_hp, nb23_hp,
        nb31_lp, nb33_lp, nb31_hp, nb33_hp);
    NO_DEVICE_CODE;
#endif
}

template<int D, int ncols>
static void ggml_cuda_flash_attn_ext_mixed_graph_combine_case_ncols(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    const ggml_tensor * Q       = dst->src[0];
    const ggml_tensor * K_lp    = dst->src[1];
    const ggml_tensor * lp_out  = dst->src[2];
    const ggml_tensor * mask_lp = dst->src[3];
    const ggml_tensor * K_hp    = dst->src[5];
    const ggml_tensor * V_hp    = dst->src[6];
    const ggml_tensor * mask_hp = dst->src[7];
    const ggml_tensor * lp_meta_sidecar = dst->src[8];

    float scale = 1.0f;
    memcpy(&scale, dst->op_params, sizeof(float));
    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const dim3 blocks((Q->ne[1] + ncols - 1) / ncols, Q->ne[2], Q->ne[3]);
    const dim3 threads(WARP_SIZE*ncols, 1, 1);
    const auto params = ggml_cuda_kernel_launch_params(blocks, threads, 0, ctx.stream());
    GGML_UNUSED(K_lp);
    GGML_UNUSED(mask_lp);
    const bool mask_skip = []() {
        const char * env = getenv("LLAMA_KV_HP_STAGED_MASK_SKIP");
        return env && env[0] != '\0' && env[0] != '0';
    }();
    if (mask_skip && logit_softcap != 0.0f) {
        ggml_cuda_kernel_launch((flash_attn_ext_mixed_oscar2_graph_combine_warp<D, ncols, true, true>), params,
            (const char *) Q->data,
            (const float *) lp_out->data,
            lp_meta_sidecar ? (const float2 *) lp_meta_sidecar->data : nullptr,
            (const char *) K_hp->data, (const char *) V_hp->data, (const char *) mask_hp->data,
            (float *) dst->data, scale, logit_softcap,
            Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K_hp->ne[1], K_hp->ne[2], K_hp->nb[1], K_hp->nb[2], K_hp->nb[3],
            V_hp->nb[1], V_hp->nb[2], V_hp->nb[3],
            mask_hp->nb[1], mask_hp->nb[3]);
    } else if (mask_skip) {
        ggml_cuda_kernel_launch((flash_attn_ext_mixed_oscar2_graph_combine_warp<D, ncols, false, true>), params,
            (const char *) Q->data,
            (const float *) lp_out->data,
            lp_meta_sidecar ? (const float2 *) lp_meta_sidecar->data : nullptr,
            (const char *) K_hp->data, (const char *) V_hp->data, (const char *) mask_hp->data,
            (float *) dst->data, scale, logit_softcap,
            Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K_hp->ne[1], K_hp->ne[2], K_hp->nb[1], K_hp->nb[2], K_hp->nb[3],
            V_hp->nb[1], V_hp->nb[2], V_hp->nb[3],
            mask_hp->nb[1], mask_hp->nb[3]);
    } else if (logit_softcap != 0.0f) {
        ggml_cuda_kernel_launch((flash_attn_ext_mixed_oscar2_graph_combine_warp<D, ncols, true, false>), params,
            (const char *) Q->data,
            (const float *) lp_out->data,
            lp_meta_sidecar ? (const float2 *) lp_meta_sidecar->data : nullptr,
            (const char *) K_hp->data, (const char *) V_hp->data, (const char *) mask_hp->data,
            (float *) dst->data, scale, logit_softcap,
            Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K_hp->ne[1], K_hp->ne[2], K_hp->nb[1], K_hp->nb[2], K_hp->nb[3],
            V_hp->nb[1], V_hp->nb[2], V_hp->nb[3],
            mask_hp->nb[1], mask_hp->nb[3]);
    } else {
        ggml_cuda_kernel_launch((flash_attn_ext_mixed_oscar2_graph_combine_warp<D, ncols, false, false>), params,
            (const char *) Q->data,
            (const float *) lp_out->data,
            lp_meta_sidecar ? (const float2 *) lp_meta_sidecar->data : nullptr,
            (const char *) K_hp->data, (const char *) V_hp->data, (const char *) mask_hp->data,
            (float *) dst->data, scale, logit_softcap,
            Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K_hp->ne[1], K_hp->ne[2], K_hp->nb[1], K_hp->nb[2], K_hp->nb[3],
            V_hp->nb[1], V_hp->nb[2], V_hp->nb[3],
            mask_hp->nb[1], mask_hp->nb[3]);
    }
    CUDA_CHECK(cudaGetLastError());
}

template<int D>
static void ggml_cuda_flash_attn_ext_mixed_graph_combine_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (getenv("LLAMA_KV_HP_STAGED_COMBINE_NCOLS8")) {
        ggml_cuda_flash_attn_ext_mixed_graph_combine_case_ncols<D, 8>(ctx, dst);
    } else {
        ggml_cuda_flash_attn_ext_mixed_graph_combine_case_ncols<D, 4>(ctx, dst);
    }
}

template<int D, int ncols>
static void ggml_cuda_flash_attn_ext_mixed_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    const ggml_tensor * Q       = dst->src[0];
    const ggml_tensor * K_lp    = dst->src[1];
    const ggml_tensor * V_lp    = dst->src[2];
    const ggml_tensor * mask_lp = dst->src[3];
    const ggml_tensor * K_hp    = dst->src[5];
    const ggml_tensor * V_hp    = dst->src[6];
    const ggml_tensor * mask_hp = dst->src[7];

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(ggml_cuda_info().devices[ctx.device].cc);
    const int nwarps = nthreads / WARP_SIZE;
    const int ntiles_x = (Q->ne[1] + ncols - 1) / ncols;
    const int ntiles_dst = ntiles_x * Q->ne[2] * Q->ne[3];
    const int ntiles_KV = std::max<int>((K_lp->ne[1] + nthreads - 1) / nthreads,
                                        (K_hp->ne[1] + nthreads - 1) / nthreads);

    int max_blocks_per_sm = 1;
    const dim3 block_dim(WARP_SIZE, nwarps, 1);
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, flash_attn_ext_mixed_oscar2_f16_vec<D, ncols>, block_dim.x*block_dim.y, 0));
    int parallel_blocks = std::max(1, std::min(max_blocks_per_sm, ntiles_KV));
    if (const char * env = getenv("LLAMA_KV_MIXED_VEC_MAIN_PARTS")) {
        parallel_blocks = std::max(1, std::min(atoi(env), ntiles_KV));
    }

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    float * out = (float *) dst->data;
    if (parallel_blocks > 1) {
        dst_tmp.alloc(parallel_blocks*ggml_nelements(dst));
        dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(dst));
        out = dst_tmp.ptr;
    }

    float scale = 1.0f;
    memcpy(&scale, dst->op_params, sizeof(float));

    const dim3 blocks(ntiles_x, parallel_blocks, Q->ne[2]*Q->ne[3]);
    const auto params = ggml_cuda_kernel_launch_params(blocks, block_dim, 0, ctx.stream());
    ggml_cuda_kernel_launch(flash_attn_ext_mixed_oscar2_f16_vec<D, ncols>, params,
        (const char *) Q->data,
        (const char *) K_lp->data, (const char *) V_lp->data, (const char *) mask_lp->data,
        (const char *) K_hp->data, (const char *) V_hp->data, (const char *) mask_hp->data,
        out, dst_tmp_meta.ptr, scale,
        Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K_lp->ne[1], K_lp->ne[2], K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
        V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
        K_hp->ne[1], K_hp->ne[2], K_hp->nb[1], K_hp->nb[2], K_hp->nb[3],
        V_hp->nb[1], V_hp->nb[2], V_hp->nb[3],
        mask_lp->nb[1], mask_lp->nb[3], mask_hp->nb[1], mask_hp->nb[3]);
    CUDA_CHECK(cudaGetLastError());

    if (parallel_blocks > 1) {
        const dim3 blocks_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const dim3 threads_combine(D, 1, 1);
        const size_t shmem = parallel_blocks*sizeof(float2);
        const auto combine_params = ggml_cuda_kernel_launch_params(blocks_combine, threads_combine, shmem, ctx.stream());
        ggml_cuda_kernel_launch(flash_attn_combine_results<D>, combine_params,
            out, dst_tmp_meta.ptr, (float *) dst->data, nullptr, parallel_blocks);
        CUDA_CHECK(cudaGetLastError());
    }

    GGML_UNUSED(ntiles_dst);
}

template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(cc);
    const int nwarps   = nthreads / WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap>;
    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;
    const bool stream_k =
        (type_K == GGML_TYPE_TURBO2_0 || type_V == GGML_TYPE_TURBO2_0 ||
         type_K == GGML_TYPE_TURBO3_0 || type_V == GGML_TYPE_TURBO3_0) &&
        turbo_vec_stream_k_env_enabled();
    launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, stream_k);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if constexpr (type_K == GGML_TYPE_Q2_0 || type_V == GGML_TYPE_Q2_0) {
        if (Q->ne[1] > 2) {
            constexpr int cols_per_block = 4;
            if (logit_softcap == 0.0f) {
                constexpr bool use_logit_softcap = false;
                ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
            } else {
                constexpr bool use_logit_softcap = true;
                ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
            }
            return;
        }
    }

    if constexpr (type_K == GGML_TYPE_TURBO2_0 || type_V == GGML_TYPE_TURBO2_0 ||
                  type_K == GGML_TYPE_TURBO3_0 || type_V == GGML_TYPE_TURBO3_0) {
        constexpr bool mixed_non_turbo_K_turbo_V =
            type_K != GGML_TYPE_TURBO2_0 && type_K != GGML_TYPE_TURBO3_0 &&
            (type_V == GGML_TYPE_TURBO2_0 || type_V == GGML_TYPE_TURBO3_0);
        constexpr int cols_per_block = mixed_non_turbo_K_turbo_V ? 4 : 2;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 2;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template void ggml_cuda_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_TURBO2_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_TURBO3_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q2_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_BF16); \

EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q2_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_TURBO2_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_TURBO3_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q2_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q2_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_BF16)

extern DECL_FATTN_VEC_CASE(512, GGML_TYPE_Q2_0, GGML_TYPE_Q2_0);

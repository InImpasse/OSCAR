#include "fattn-oscar2-v2.cuh"

#include "fattn-common.cuh"

#include <cstdlib>
#include <cstdio>

template<bool use_logit_softcap>
__launch_bounds__(WARP_SIZE*4, 1)
static __global__ void flash_attn_oscar2_v2_lp_warp4(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float * __restrict__ dst,
        float2 * __restrict__ dst_meta,
        const float scale,
        const float logit_softcap,
        const int32_t ne01, const int32_t ne02, const int32_t ne03,
        const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int32_t ne11, const int32_t ne12,
        const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int64_t nb21, const int64_t nb22, const int64_t nb23,
        const int64_t nb31, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE
    constexpr int D = 128;
    constexpr int ncols = 4;

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int col = blockIdx.x*ncols + warp;
    if (warp >= ncols || col >= ne01) {
        return;
    }

    const int head = blockIdx.y;
    const int sequence = blockIdx.z;
    const int gqa_ratio = ne02 / ne12;
    const int hkv = head / gqa_ratio;
    const int row = (sequence*ne01 + col)*ne02 + head;

    Q += nb03*sequence + nb02*head + nb01*col;
    K += nb13*sequence + nb12*hkv;
    V += nb23*sequence + nb22*hkv;
    const half * mask_row = (const half *) (mask + nb33*sequence + nb31*col);

    float q[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        q[r] = scale * ((const float *) Q)[lane + r*WARP_SIZE];
    }

    float numerator[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float kqmax = -FLT_MAX/2.0f;
    float denom = 0.0f;

    for (int key = 0; key < ne11; ++key) {
        const block_oscar2_kv & kb = *(const block_oscar2_kv *) (K + key*nb11);

        float score = 0.0f;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int d = lane + r*WARP_SIZE;
            const uint8_t qs = kb.qs[d / 4];
            const uint8_t rs = kb.rs[d / 8];
            const int sh = 2*(d & 3);
            const int idx = ((qs >> sh) & 0x03) | (((rs >> (d & 7)) & 0x01) << 2);
            score += q[r] * __half2float(kb.d) * oscar2_centroid_3bit_cuda(idx);
        }
        score = warp_reduce_sum(score);
        if constexpr (use_logit_softcap) {
            score = logit_softcap*tanhf(score);
        }
        score += __half2float(mask_row[key]);

        const float kqmax_new = fmaxf(kqmax, score + FATTN_KQ_MAX_OFFSET);
        const float old_scale = __expf(kqmax - kqmax_new);
        const float p = __expf(score - kqmax_new);

        const block_oscar2_kv & vb = *(const block_oscar2_kv *) (V + key*nb21);
        const float vm = __half2float(vb.m);
        const float vd = __half2float(vb.d);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int d = lane + r*WARP_SIZE;
            const uint8_t qs = vb.qs[d / 4];
            const uint8_t rs = vb.rs[d / 8];
            const int sh = 2*(d & 3);
            const int idx = ((qs >> sh) & 0x03) | (((rs >> (d & 7)) & 0x01) << 2);
            const float v = vm + vd * oscar2_v_centroid_3bit_cuda(idx);
            numerator[r] = numerator[r]*old_scale + p*v;
        }
        denom = denom*old_scale + p;
        kqmax = kqmax_new;
    }

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        dst[int64_t(row)*D + lane + r*WARP_SIZE] = numerator[r] / denom;
    }
    if (lane == 0) {
        dst_meta[row] = make_float2(kqmax, denom);
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, dst, dst_meta, scale, logit_softcap,
        ne01, ne02, ne03, nb01, nb02, nb03,
        ne11, ne12, nb11, nb12, nb13, nb21, nb22, nb23, nb31, nb33);
    NO_DEVICE_CODE;
#endif
}

template<bool use_logit_softcap>
__launch_bounds__(WARP_SIZE*4, 1)
static __global__ void flash_attn_oscar2_v2_lp_tile4_shared_v(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float * __restrict__ dst,
        float2 * __restrict__ dst_meta,
        const float scale,
        const float logit_softcap,
        const int32_t ne01, const int32_t ne02, const int32_t ne03,
        const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int32_t ne11, const int32_t ne12,
        const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int64_t nb21, const int64_t nb22, const int64_t nb23,
        const int64_t nb31, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE
    constexpr int D = 128;
    constexpr int ncols = 4;

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int col0 = blockIdx.x*ncols;
    const int col = col0 + warp;
    const int head = blockIdx.y;
    const int sequence = blockIdx.z;
    const int gqa_ratio = ne02 / ne12;
    const int hkv = head / gqa_ratio;

    Q += nb03*sequence + nb02*head;
    K += nb13*sequence + nb12*hkv;
    V += nb23*sequence + nb22*hkv;

    float q[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        q[r] = (col < ne01) ? scale * ((const float *) (Q + nb01*col))[lane + r*WARP_SIZE] : 0.0f;
    }

    float numerator[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float kqmax = -FLT_MAX/2.0f;
    float denom = 0.0f;

    __shared__ float sh_v[D];
    __shared__ float sh_p[ncols];
    __shared__ float sh_old_scale[ncols];

    for (int key = 0; key < ne11; ++key) {
        const block_oscar2_kv & kb = *(const block_oscar2_kv *) (K + key*nb11);

        float score = 0.0f;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int d = lane + r*WARP_SIZE;
            const uint8_t qs = kb.qs[d / 4];
            const uint8_t rs = kb.rs[d / 8];
            const int sh = 2*(d & 3);
            const int idx = ((qs >> sh) & 0x03) | (((rs >> (d & 7)) & 0x01) << 2);
            score += q[r] * __half2float(kb.d) * oscar2_centroid_3bit_cuda(idx);
        }
        score = warp_reduce_sum(score);
        if constexpr (use_logit_softcap) {
            score = logit_softcap*tanhf(score);
        }
        if (col < ne01) {
            const half * mask_row = (const half *) (mask + nb33*sequence + nb31*col);
            score += __half2float(mask_row[key]);
        } else {
            score = -INFINITY;
        }

        const float kqmax_new = fmaxf(kqmax, score + FATTN_KQ_MAX_OFFSET);
        const float old_scale = __expf(kqmax - kqmax_new);
        const float p = __expf(score - kqmax_new);
        if (lane == 0) {
            sh_p[warp] = p;
            sh_old_scale[warp] = old_scale;
        }

        if (warp == 0) {
            const block_oscar2_kv & vb = *(const block_oscar2_kv *) (V + key*nb21);
            const float vm = __half2float(vb.m);
            const float vd = __half2float(vb.d);
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int d = lane + r*WARP_SIZE;
                const uint8_t qs = vb.qs[d / 4];
                const uint8_t rs = vb.rs[d / 8];
                const int sh = 2*(d & 3);
                const int idx = ((qs >> sh) & 0x03) | (((rs >> (d & 7)) & 0x01) << 2);
                sh_v[d] = vm + vd * oscar2_v_centroid_3bit_cuda(idx);
            }
        }

        __syncthreads();

        const float p_warp = sh_p[warp];
        const float old_scale_warp = sh_old_scale[warp];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int d = lane + r*WARP_SIZE;
            numerator[r] = numerator[r]*old_scale_warp + p_warp*sh_v[d];
        }
        denom = denom*old_scale + p;
        kqmax = kqmax_new;

        __syncthreads();
    }

    if (col < ne01) {
        const int row = (sequence*ne01 + col)*ne02 + head;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            dst[int64_t(row)*D + lane + r*WARP_SIZE] = numerator[r] / denom;
        }
        if (lane == 0) {
            dst_meta[row] = make_float2(kqmax, denom);
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, dst, dst_meta, scale, logit_softcap,
        ne01, ne02, ne03, nb01, nb02, nb03,
        ne11, ne12, nb11, nb12, nb13, nb21, nb22, nb23, nb31, nb33);
    NO_DEVICE_CODE;
#endif
}

bool ggml_cuda_flash_attn_ext_mixed_oscar2_v2_lp_try(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * Q,
        const ggml_tensor * K_lp,
        const ggml_tensor * V_lp,
        const ggml_tensor * mask_lp,
        float * lp_num,
        float2 * lp_meta,
        float scale,
        float logit_softcap,
        int ne11_lp_launch,
        int parallel_blocks,
        bool normalize_lp,
        bool lp_diag_no_v,
        int ncols) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(lp_num);
    GGML_UNUSED(lp_meta);
    GGML_UNUSED(scale);
    GGML_UNUSED(logit_softcap);
    GGML_UNUSED(ne11_lp_launch);
    GGML_UNUSED(parallel_blocks);
    GGML_UNUSED(normalize_lp);
    GGML_UNUSED(lp_diag_no_v);

    const char * env = getenv("LLAMA_KV_MIXED_VEC_OSCAR2_V2");
    if (env == nullptr || env[0] == '\0' || env[0] == '0') {
        return false;
    }

    if (Q == nullptr || K_lp == nullptr || V_lp == nullptr || mask_lp == nullptr) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 ||
            K_lp->type != GGML_TYPE_OSCAR2_KV ||
            V_lp->type != GGML_TYPE_OSCAR2_KV ||
            mask_lp->type != GGML_TYPE_F16 ||
            Q->ne[0] != 128 ||
            ncols != 4) {
        return false;
    }

    const char * force_env = getenv("LLAMA_KV_MIXED_VEC_OSCAR2_V2_FORCE");
    const bool force = force_env != nullptr && force_env[0] != '\0' && force_env[0] != '0';
    if (force && normalize_lp && parallel_blocks == 1 && !lp_diag_no_v) {
        ggml_cuda_set_device(ctx.device);
        const dim3 blocks((Q->ne[1] + 3) / 4, Q->ne[2], Q->ne[3]);
        const dim3 threads(WARP_SIZE*4, 1, 1);
        const auto params = ggml_cuda_kernel_launch_params(blocks, threads, 0, ctx.stream());
        const char * tile_env = getenv("LLAMA_KV_MIXED_VEC_OSCAR2_V2_TILE");
        const bool tile = tile_env != nullptr && tile_env[0] != '\0' && tile_env[0] != '0';
        if (logit_softcap != 0.0f) {
            if (tile) {
                ggml_cuda_kernel_launch((flash_attn_oscar2_v2_lp_tile4_shared_v<true>), params,
                    (const char *) Q->data,
                    (const char *) K_lp->data,
                    (const char *) V_lp->data,
                    (const char *) mask_lp->data,
                    lp_num, lp_meta,
                    scale, logit_softcap,
                    Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
                    ne11_lp_launch, K_lp->ne[2], K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
                    V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
                    mask_lp->nb[1], mask_lp->nb[3]);
            } else {
                ggml_cuda_kernel_launch((flash_attn_oscar2_v2_lp_warp4<true>), params,
                    (const char *) Q->data,
                    (const char *) K_lp->data,
                    (const char *) V_lp->data,
                    (const char *) mask_lp->data,
                    lp_num, lp_meta,
                    scale, logit_softcap,
                    Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
                    ne11_lp_launch, K_lp->ne[2], K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
                    V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
                    mask_lp->nb[1], mask_lp->nb[3]);
            }
        } else {
            if (tile) {
                ggml_cuda_kernel_launch((flash_attn_oscar2_v2_lp_tile4_shared_v<false>), params,
                    (const char *) Q->data,
                    (const char *) K_lp->data,
                    (const char *) V_lp->data,
                    (const char *) mask_lp->data,
                    lp_num, lp_meta,
                    scale, logit_softcap,
                    Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
                    ne11_lp_launch, K_lp->ne[2], K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
                    V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
                    mask_lp->nb[1], mask_lp->nb[3]);
            } else {
                ggml_cuda_kernel_launch((flash_attn_oscar2_v2_lp_warp4<false>), params,
                    (const char *) Q->data,
                    (const char *) K_lp->data,
                    (const char *) V_lp->data,
                    (const char *) mask_lp->data,
                    lp_num, lp_meta,
                    scale, logit_softcap,
                    Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
                    ne11_lp_launch, K_lp->ne[2], K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
                    V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
                    mask_lp->nb[1], mask_lp->nb[3]);
            }
        }
        CUDA_CHECK(cudaGetLastError());
        if (getenv("LLAMA_KV_MIXED_VEC_RAW_DEBUG")) {
            fprintf(stderr,
                "mixed_oscar2_v2_lp: force=1 kernel=%s q=%lld k_eff=%d ncols=%d norm=%d softcap=%.6g\n",
                tile ? "tile4_shared_v" : "warp4",
                (long long) Q->ne[1],
                ne11_lp_launch,
                ncols,
                normalize_lp ? 1 : 0,
                logit_softcap);
        }
        return true;
    }

    if (getenv("LLAMA_KV_MIXED_VEC_RAW_DEBUG")) {
        fprintf(stderr,
            "mixed_oscar2_v2_lp: stub q=%lld k_eff=%d parts=%d ncols=%d norm=%d no_v=%d softcap=%.6g fallback=1\n",
            (long long) Q->ne[1],
            ne11_lp_launch,
            parallel_blocks,
            ncols,
            normalize_lp ? 1 : 0,
            lp_diag_no_v ? 1 : 0,
            logit_softcap);
    }

    return false;
}

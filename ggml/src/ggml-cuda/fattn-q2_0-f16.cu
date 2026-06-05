#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn.cuh"
#include "q2_0-owht.cuh"

template<int D>
static __device__ __forceinline__ float dot_q2_k(const float * q, const block_q2_0 * k_row) {
    float sum = 0.0f;
#pragma unroll
    for (int d = 0; d < D; ++d) {
        sum += q[d] * q2_0_dequantize_scalar_cuda(k_row, d);
    }
    return sum;
}

template<int D>
static __device__ __forceinline__ float dot_f32_k(const float * q, const float * k) {
    float sum = 0.0f;
#pragma unroll
    for (int d = 0; d < D; ++d) {
        sum += q[d] * k[d];
    }
    return sum;
}

template<int D>
static __device__ __forceinline__ float dot_f16_k(const float * q, const half * k_row) {
    float sum = 0.0f;
#pragma unroll
    for (int d = 0; d < D; ++d) {
        sum += q[d] * __half2float(k_row[d]);
    }
    return sum;
}

// Fused LP (Q2_0) + HP (F16) joint-softmax attention (one K·Q dot per key).
template<int D>
static __global__ void flash_attn_q2_0_f16_kernel(
        const float      * __restrict__ Q,
        const block_q2_0 * __restrict__ K_lp,
        const block_q2_0 * __restrict__ V_lp,
        const float      * __restrict__ mask_lp,
        const half       * __restrict__ K_hp,
        const half       * __restrict__ V_hp,
        const float      * __restrict__ mask_hp,
        float            * __restrict__ dst,
        const float scale,
        const int n_q,
        const int n_head,
        const int n_head_kv,
        const int n_kv_lp,
        const int n_kv_hp,
        const int n_stream,
        const int nb_q1,
        const int nb_q2,
        const int nb_q3,
        const int nb_klp1,
        const int nb_klp2,
        const int nb_klp3,
        const int nb_vlp1,
        const int nb_vlp2,
        const int nb_vlp3,
        const int nb_khp1,
        const int nb_khp2,
        const int nb_khp3,
        const int nb_vhp1,
        const int nb_vhp2,
        const int nb_vhp3,
        const int nb_mlp1,
        const int nb_mlp3,
        const int nb_mhp1,
        const int nb_mhp3,
        const bool use_owht,
        const bool apply_hadamard) {
    const int token  = blockIdx.x;
    const int head   = blockIdx.y;
    const int stream = blockIdx.z;
    const int hkv    = head / (n_head / n_head_kv);
    const int tid    = threadIdx.x;

    const char * q_base = (const char *) Q + stream*nb_q3 + head*nb_q2 + token*nb_q1;

    __shared__ float q_sh[D];
    for (int d = tid; d < D; d += blockDim.x) {
        q_sh[d] = ((const float *) q_base)[d];
    }
    __syncthreads();

    __shared__ float m_sh;
    __shared__ float l_sh;
    if (tid == 0) {
        m_sh = -FLT_MAX/2.0f;
        l_sh = 0.0f;
    }
    __syncthreads();

    float o_val = 0.0f;
    const bool owns_dim = tid < D;

    auto online_update = [&](const float s, auto get_v) {
        if (!isfinite(s)) {
            return;
        }
        const float m_new     = fmaxf(m_sh, s);
        const float alpha     = expf(s - m_new);
        const float scale_old = expf(m_sh - m_new);
        if (owns_dim) {
            o_val = o_val * scale_old + alpha * get_v(tid);
        }
        if (tid == 0) {
            l_sh = l_sh * scale_old + alpha;
            m_sh = m_new;
        }
        __syncthreads();
    };

    for (int j = 0; j < n_kv_lp; ++j) {
        const float m = *(const float *) ((const char *) mask_lp + stream*nb_mlp3 + token*nb_mlp1 + j*sizeof(float));
        if (!isfinite(m)) {
            continue;
        }
        const block_q2_0 * k_row = (const block_q2_0 *) ((const char *) K_lp + stream*nb_klp3 + hkv*nb_klp2 + j*nb_klp1);
        const block_q2_0 * v_row = (const block_q2_0 *) ((const char *) V_lp + stream*nb_vlp3 + hkv*nb_vlp2 + j*nb_vlp1);
        if (use_owht) {
            float k_dec[D];
            float v_dec[D];
            q2_0_dequantize_row_owht_cuda<D>(k_row, k_dec, apply_hadamard);
            q2_0_dequantize_row_owht_cuda<D>(v_row, v_dec, apply_hadamard);
            const float s = dot_f32_k<D>(q_sh, k_dec) * scale + m;
            online_update(s, [&](const int d) { return v_dec[d]; });
        } else {
            const float s = dot_q2_k<D>(q_sh, k_row) * scale + m;
            online_update(s, [&](const int d) { return q2_0_dequantize_scalar_cuda(v_row, d); });
        }
    }

    for (int j = 0; j < n_kv_hp; ++j) {
        const float m = *(const float *) ((const char *) mask_hp + stream*nb_mhp3 + token*nb_mhp1 + j*sizeof(float));
        if (!isfinite(m)) {
            continue;
        }
        const half * k_row = (const half *) ((const char *) K_hp + stream*nb_khp3 + hkv*nb_khp2 + j*nb_khp1);
        const half * v_row = (const half *) ((const char *) V_hp + stream*nb_vhp3 + hkv*nb_vhp2 + j*nb_vhp1);
        const float s = dot_f16_k<D>(q_sh, k_row) * scale + m;
        online_update(s, [&](const int d) { return __half2float(v_row[d]); });
    }

    if (owns_dim) {
        dst[((stream*n_q + token)*n_head + head)*D + tid] = o_val / l_sh;
    }
}

template<int D>
static void launch_flash_attn_q2_0_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q       = dst->src[0];
    const ggml_tensor * K_lp    = dst->src[1];
    const ggml_tensor * V_lp    = dst->src[2];
    const ggml_tensor * mask_lp = dst->src[3];
    const ggml_tensor * K_hp    = dst->src[4];
    const ggml_tensor * V_hp    = dst->src[5];
    const ggml_tensor * mask_hp = dst->src[6];

    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    const bool use_owht = q2_0_cuda_owht_enabled();
    const bool apply_hadamard = q2_0_cuda_apply_hadamard();

    const dim3 blocks(Q->ne[1], Q->ne[2], Q->ne[3]);
    const dim3 threads(D <= 128 ? 128 : (D <= 256 ? 256 : 512), 1, 1);
    const auto params = ggml_cuda_kernel_launch_params(blocks, threads, 0, ctx.stream());
    ggml_cuda_kernel_launch(flash_attn_q2_0_f16_kernel<D>, params,
        (const float *) Q->data,
        (const block_q2_0 *) K_lp->data,
        (const block_q2_0 *) V_lp->data,
        (const float *) mask_lp->data,
        (const half *) K_hp->data,
        (const half *) V_hp->data,
        (const float *) mask_hp->data,
        (float *) dst->data,
        scale,
        Q->ne[1], Q->ne[2], K_lp->ne[2], K_lp->ne[1], K_hp->ne[1], Q->ne[3],
        Q->nb[1], Q->nb[2], Q->nb[3],
        K_lp->nb[1], K_lp->nb[2], K_lp->nb[3],
        V_lp->nb[1], V_lp->nb[2], V_lp->nb[3],
        K_hp->nb[1], K_hp->nb[2], K_hp->nb[3],
        V_hp->nb[1], V_hp->nb[2], V_hp->nb[3],
        mask_lp->nb[1], mask_lp->nb[3],
        mask_hp->nb[1], mask_hp->nb[3],
        use_owht, apply_hadamard);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_flash_attn_ext_q2_0_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    switch (dst->src[0]->ne[0]) {
        case 64:  launch_flash_attn_q2_0_f16< 64>(ctx, dst); break;
        case 128: launch_flash_attn_q2_0_f16<128>(ctx, dst); break;
        case 256: launch_flash_attn_q2_0_f16<256>(ctx, dst); break;
        case 512: launch_flash_attn_q2_0_f16<512>(ctx, dst); break;
        default: GGML_ABORT("unsupported q2_0+f16 attention head size");
    }
}

bool ggml_cuda_flash_attn_ext_q2_0_f16_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    const ggml_tensor * Q       = dst->src[0];
    const ggml_tensor * K_lp    = dst->src[1];
    const ggml_tensor * V_lp    = dst->src[2];
    const ggml_tensor * mask_lp = dst->src[3];
    const ggml_tensor * K_hp    = dst->src[4];
    const ggml_tensor * V_hp    = dst->src[5];
    const ggml_tensor * mask_hp = dst->src[6];

    if (!Q || !K_lp || !V_lp || !mask_lp || !K_hp || !V_hp || !mask_hp) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K_lp->type != GGML_TYPE_Q2_0 || V_lp->type != GGML_TYPE_Q2_0 ||
            K_hp->type != GGML_TYPE_F16 || V_hp->type != GGML_TYPE_F16 ||
            mask_lp->type != GGML_TYPE_F32 || mask_hp->type != GGML_TYPE_F32) {
        return false;
    }
    return Q->ne[0] == 64 || Q->ne[0] == 128 || Q->ne[0] == 256 || Q->ne[0] == 512;
}

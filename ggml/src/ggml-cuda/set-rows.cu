#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include "q2_0-owht.cuh"

#include <cstring>

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

template <bool is_v>
static __device__ __forceinline__ uint8_t oscar2_quantize_norm_cuda(float v) {
    int q = 0;
    float best;
    if constexpr (is_v) {
        best = fabsf(v - oscar2_v_centroid_3bit_cuda(0));
        for (int qi = 1; qi < 8; ++qi) {
            const float err = fabsf(v - oscar2_v_centroid_3bit_cuda(qi));
            if (err < best) {
                best = err;
                q = qi;
            }
        }
    } else {
        best = fabsf(v - oscar2_centroid_3bit_cuda(0));
        for (int qi = 1; qi < 8; ++qi) {
            const float err = fabsf(v - oscar2_centroid_3bit_cuda(qi));
            if (err < best) {
                best = err;
                q = qi;
            }
        }
    }
    return (uint8_t) q;
}

static __device__ __forceinline__ uint8_t turbo2_quantize_cuda(const float v) {
    if (v < -0.086728f) return 0;
    if (v <  0.0f)      return 1;
    if (v <  0.086728f) return 2;
    return 3;
}

static __device__ __forceinline__ float turbo2_centroid_cuda(const uint8_t q) {
    switch (q & 0x03) {
        case 0:  return -0.133462f;
        case 1:  return -0.039994f;
        case 2:  return  0.039994f;
        default: return  0.133462f;
    }
}

static __device__ __forceinline__ uint8_t turbo3_quantize_cuda(const float v) {
    if (v < -0.154259f) return 0;
    if (v < -0.091775f) return 1;
    if (v < -0.043589f) return 2;
    if (v <  0.0f)      return 3;
    if (v <  0.043589f) return 4;
    if (v <  0.091775f) return 5;
    if (v <  0.154259f) return 6;
    return 7;
}

static __device__ __forceinline__ float turbo3_centroid_cuda(const uint8_t q) {
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

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename idx_t, bool is_v>
static __global__ void k_set_rows_oscar2_parallel(const float * __restrict__ src0,
                                                  const idx_t * __restrict__ src1,
                                                  block_oscar2_kv * __restrict__ dst,
                                                  const int64_t ne_total,
                                                  const int64_t s01,
                                                  const int64_t s02,
                                                  const int64_t s03,
                                                  const int64_t s10,
                                                  const int64_t s11,
                                                  const int64_t s12,
                                                  const int64_t s1,
                                                  const int64_t s2,
                                                  const int64_t s3,
                                                  const uint3   ne00,
                                                  const uint3   ne01,
                                                  const uint3   ne02,
                                                  const uint3   ne11_fd,
                                                  const uint3   ne12_fd) {
    constexpr int qk = QK_OSCAR2_KV;
    const int64_t iblock = int64_t(blockIdx.x);
    const int tid = threadIdx.x;

    if (iblock >= ne_total || tid >= qk) {
        return;
    }

    const int64_t i_base = iblock * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_oscar2_kv * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_oscar2_kv);
    block_oscar2_kv * dst_block = dst_row_ptr + i00 / qk;

    const float x = src0_row[i00 + tid];
    __shared__ float shared_reduce[4];

    float mean = 0.0f;
    if constexpr (is_v) {
        mean = block_reduce<block_reduce_method::SUM, qk>(x, shared_reduce) / qk;
    }
    __syncthreads();

    const float centered = x - mean;
    const float sum_sq = block_reduce<block_reduce_method::SUM, qk>(centered * centered, shared_reduce);
    const float d = sqrtf(sum_sq / qk);
    const float id = d > 1e-8f ? 1.0f / d : 0.0f;
    __syncthreads();

    if (tid == 0) {
        dst_block->d = d;
        dst_block->m = is_v ? mean : 0.0f;
    }
    if (tid < qk / 4) {
        uint8_t packed = 0;
        const int base = 4*tid;
        packed |= (oscar2_quantize_norm_cuda<is_v>((src0_row[i00 + base + 0] - mean) * id) & 0x03) << 0;
        packed |= (oscar2_quantize_norm_cuda<is_v>((src0_row[i00 + base + 1] - mean) * id) & 0x03) << 2;
        packed |= (oscar2_quantize_norm_cuda<is_v>((src0_row[i00 + base + 2] - mean) * id) & 0x03) << 4;
        packed |= (oscar2_quantize_norm_cuda<is_v>((src0_row[i00 + base + 3] - mean) * id) & 0x03) << 6;
        dst_block->qs[tid] = packed;
    }
    if (tid < qk / 8) {
        uint8_t packed = 0;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const uint8_t q = oscar2_quantize_norm_cuda<is_v>((src0_row[i00 + 8*tid + b] - mean) * id);
            packed |= ((q >> 2) & 0x01) << b;
        }
        dst_block->rs[tid] = packed;
    }
}

template<typename idx_t, bool is_v>
static bool set_rows_cuda_oscar2_parallel(
        const float * src0_d, const idx_t * src1_d, block_oscar2_kv * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne11, const int64_t ne12,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    const char * env = getenv("LLAMA_KV_OSCAR2_SET_ROWS_PAR");
    if (env == nullptr || env[0] == '\0' || env[0] == '0') {
        return false;
    }
    if constexpr (is_v) {
        if (strcmp(env, "k") == 0) {
            return false;
        }
    } else {
        if (strcmp(env, "v") == 0) {
            return false;
        }
    }
    if (ne00 % QK_OSCAR2_KV != 0 || ne00 == 0 || ne01 == 0 || ne02 == 0 || ne11 == 0 || ne12 == 0) {
        return false;
    }

    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / QK_OSCAR2_KV;
    if (ne_total <= 0) {
        return true;
    }

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
    const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
    const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
    const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
    const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

    const dim3 grid_size(ne_total);
    const dim3 block_size(QK_OSCAR2_KV);
    const auto launch_params = ggml_cuda_kernel_launch_params(grid_size, block_size, 0, stream);
    ggml_cuda_kernel_launch((k_set_rows_oscar2_parallel<idx_t, is_v>), launch_params,
            src0_d, src1_d, dst_d, ne_total, s01, s02, s03, s10, s11, s12, s1, s2, s3,
            ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    return true;
}

template <typename idx_t>
static __global__ void k_set_rows_q2_0_owht(const float * __restrict__ src0,
                                            const idx_t * __restrict__ src1,
                                            block_q2_0 * __restrict__ dst,
                                            const int64_t ne_total,
                                            const int64_t ne10,
                                            const int64_t ne11,
                                            const int64_t ne12,
                                            const int64_t ne13,
                                            const int64_t s01,
                                            const int64_t s02,
                                            const int64_t s03,
                                            const int64_t s10,
                                            const int64_t s11,
                                            const int64_t s12,
                                            const int64_t s1,
                                            const int64_t s2,
                                            const int64_t s3,
                                            const uint3   ne_groups,
                                            const uint3   ne01,
                                            const uint3   ne02,
                                            const uint3   ne11_fd,
                                            const uint3   ne12_fd,
                                            const int64_t ne00,
                                            const int     group_size,
                                            const bool    apply_hadamard,
                                            const float   clip_ratio) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2 div_mod;

    div_mod                 = fast_div_modulo(tmp, ne_groups);
    const int64_t group_idx = div_mod.y;
    tmp                     = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const int64_t i00 = group_idx * group_size;
    const int64_t actual_n = min((int64_t) group_size, ne00 - i00);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_q2_0 * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_q2_0);

    q2_0_quantize_group_owht_cuda(src0_row + i00, dst_row_ptr + i00 / QK2_0, (int) actual_n, apply_hadamard, clip_ratio);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename idx_t>
static void set_rows_cuda_q2_0_owht(
        const float * src0_d, const idx_t * src1_d, block_q2_0 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream, const char * dst_name) {

    GGML_ASSERT(ne00 % QK2_0 == 0);

    const int group_size = q2_0_group_size_cuda((int) ne00);
    const bool apply_hadamard = q2_0_cuda_apply_hadamard();
    const float clip_ratio = q2_0_cuda_clip_ratio_for_cache(dst_name);
    const int64_t groups_per_row = (ne00 + group_size - 1) / group_size;
    const int64_t ne_total = groups_per_row * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne_groups_fd = init_fastdiv_values((uint32_t) groups_per_row);
        const uint3 ne01_fd      = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd      = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd      = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd      = init_fastdiv_values((uint32_t) ne12);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_size, block_size, 0, stream);
        ggml_cuda_kernel_launch(k_set_rows_q2_0_owht<idx_t>, launch_params,
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3,
            ne_groups_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd, ne00, group_size, apply_hadamard, clip_ratio);
    }
}

template <typename idx_t>
static __global__ void k_set_rows_turbo2_0(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turbo2_0 * __restrict__ dst,
        const int64_t ne_total,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const uint3   ne_blocks,
        const uint3   ne01,
        const uint3   ne02,
        const uint3   ne11_fd,
        const uint3   ne12_fd) {
    const int lane = threadIdx.x;
    const int64_t i = blockIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2 div_mod;

    div_mod                  = fast_div_modulo(tmp, ne_blocks);
    const int64_t block_idx  = div_mod.y;
    tmp                      = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo2_0 * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_turbo2_0);
    block_turbo2_0 * dst_block = dst_row_ptr + block_idx;

    const float x = src0_row[block_idx * QK_TURBO2 + lane];
    float norm_sq = x * x;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        norm_sq += __shfl_xor_sync(0xffffffff, norm_sq, offset);
    }

    __shared__ float warp_sums[4];
    if ((lane & (WARP_SIZE - 1)) == 0) {
        warp_sums[lane / WARP_SIZE] = norm_sq;
    }
    __syncthreads();

    float total = lane < 4 ? warp_sums[lane] : 0.0f;
    if (lane < WARP_SIZE) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            total += __shfl_xor_sync(0xffffffff, total, offset);
        }
    }
    __shared__ float inv_norm_sh;
    if (lane == 0) {
        const float norm = sqrtf(total);
        inv_norm_sh = norm > 1e-10f ? 1.0f / norm : 0.0f;
    }
    __syncthreads();

    const uint8_t q = turbo2_quantize_cuda(x * inv_norm_sh);

    const uint8_t my_bits = q & 0x03;
    uint8_t qs_byte = 0;
    const int warp_lane = lane & (WARP_SIZE - 1);
    const int lane4 = warp_lane & ~3;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint8_t contrib = __shfl_sync(0xffffffff, my_bits, lane4 + k);
        qs_byte |= contrib << (2 * k);
    }
    if ((lane & 3) == 0) {
        dst_block->qs[lane / 4] = qs_byte;
    }

    float recon_sq = turbo2_centroid_cuda(q) * turbo2_centroid_cuda(q);
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        recon_sq += __shfl_xor_sync(0xffffffff, recon_sq, offset);
    }
    if ((lane & (WARP_SIZE - 1)) == 0) {
        warp_sums[lane / WARP_SIZE] = recon_sq;
    }
    __syncthreads();

    total = lane < 4 ? warp_sums[lane] : 0.0f;
    if (lane < WARP_SIZE) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            total += __shfl_xor_sync(0xffffffff, total, offset);
        }
    }
    if (lane == 0) {
        const float src_norm = inv_norm_sh > 0.0f ? 1.0f / inv_norm_sh : 0.0f;
        const float recon_norm = sqrtf(total);
        const float corrected = recon_norm > 1e-10f ? src_norm / recon_norm : src_norm;
        dst_block->norm = __float2half(corrected);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename idx_t>
static void set_rows_cuda_turbo2_0(
        const float * src0_d, const idx_t * src1_d, block_turbo2_0 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_TURBO2 == 0);

    const int64_t blocks_per_row = ne00 / QK_TURBO2;
    const int64_t ne_total = blocks_per_row * ne01 * ne02 * ne03;
    const dim3 block_size(QK_TURBO2);
    const dim3 grid_size(ne_total);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne_blocks_fd = init_fastdiv_values((uint32_t) blocks_per_row);
        const uint3 ne01_fd      = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd      = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd      = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd      = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_turbo2_0<idx_t><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13,
            s01, s02, s03, s10, s11, s12, s1, s2, s3,
            ne_blocks_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <typename idx_t>
static __global__ void k_set_rows_turbo3_0(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turbo3_0 * __restrict__ dst,
        const int64_t ne_total,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const uint3   ne_blocks,
        const uint3   ne01,
        const uint3   ne02,
        const uint3   ne11_fd,
        const uint3   ne12_fd) {
    const int lane = threadIdx.x;
    const int64_t i = blockIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2 div_mod;

    div_mod                  = fast_div_modulo(tmp, ne_blocks);
    const int64_t block_idx  = div_mod.y;
    tmp                      = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo3_0 * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_turbo3_0);
    block_turbo3_0 * dst_block = dst_row_ptr + block_idx;

    const float x = src0_row[block_idx * QK_TURBO3 + lane];
    float norm_sq = x * x;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        norm_sq += __shfl_xor_sync(0xffffffff, norm_sq, offset);
    }

    __shared__ float inv_norm_sh;
    if (lane == 0) {
        const float norm = sqrtf(norm_sq);
        inv_norm_sh = norm > 1e-10f ? 1.0f / norm : 0.0f;
    }
    __syncthreads();

    const uint8_t q = turbo3_quantize_cuda(x * inv_norm_sh);

    const uint8_t low_bits = q & 0x03;
    uint8_t qs_byte = 0;
    const int lane4 = lane & ~3;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint8_t contrib = __shfl_sync(0xffffffff, low_bits, lane4 + k);
        qs_byte |= contrib << (2 * k);
    }
    if ((lane & 3) == 0) {
        dst_block->qs[lane / 4] = qs_byte;
    }

    const uint8_t high_bit = (q >> 2) & 0x01;
    uint8_t signs_byte = 0;
    const int lane8 = lane & ~7;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const uint8_t contrib = __shfl_sync(0xffffffff, high_bit, lane8 + k);
        signs_byte |= contrib << k;
    }
    if ((lane & 7) == 0) {
        dst_block->signs[lane / 8] = signs_byte;
    }

    float recon_sq = turbo3_centroid_cuda(q) * turbo3_centroid_cuda(q);
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        recon_sq += __shfl_xor_sync(0xffffffff, recon_sq, offset);
    }
    if (lane == 0) {
        const float src_norm = inv_norm_sh > 0.0f ? 1.0f / inv_norm_sh : 0.0f;
        const float recon_norm = sqrtf(recon_sq);
        const float corrected = recon_norm > 1e-10f ? src_norm / recon_norm : src_norm;
        dst_block->norm = __float2half(corrected);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename idx_t>
static void set_rows_cuda_turbo3_0(
        const float * src0_d, const idx_t * src1_d, block_turbo3_0 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_TURBO3 == 0);

    const int64_t blocks_per_row = ne00 / QK_TURBO3;
    const int64_t ne_total = blocks_per_row * ne01 * ne02 * ne03;
    const dim3 block_size(QK_TURBO3);
    const dim3 grid_size(ne_total);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne_blocks_fd = init_fastdiv_values((uint32_t) blocks_per_row);
        const uint3 ne01_fd      = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd      = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd      = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd      = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_turbo3_0<idx_t><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13,
            s01, s02, s03, s10, s11, s12, s1, s2, s3,
            ne_blocks_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * __restrict__ src0,
                                  const idx_t * __restrict__ src1,
                                  dst_t * __restrict__ dst,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    ggml_cuda_pdl_lc();

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_size, block_size, 0, stream);
        ggml_cuda_kernel_launch(k_set_rows<src_t, idx_t, dst_t>, launch_params,
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
            s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
            ne11_fd, ne12_fd);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q2_0) {
        if (q2_0_cuda_owht_enabled()) {
            set_rows_cuda_q2_0_owht<idx_t>(
                src0_d, src1_d, (block_q2_0*)dst->data,
                ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream, dst->name
            );
        } else {
            set_rows_cuda_quant<idx_t, block_q2_0, QK2_0, quantize_f32_q2_0_block>(
                src0_d, src1_d, (block_q2_0*)dst->data,
                ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream
            );
        }
    } else if (dst->type == GGML_TYPE_OSCAR2_KV) {
        const bool is_v = strstr(dst->name, "cache_v") != nullptr;
        if (is_v) {
            if (!set_rows_cuda_oscar2_parallel<idx_t, true>(
                    src0_d, src1_d, (block_oscar2_kv*)dst->data,
                    ne00, ne01, ne02, ne03,
                    ne11, ne12,
                    nb01, nb02, nb03,
                    nb10, nb11, nb12,
                    nb1, nb2, nb3,
                    stream)) {
                set_rows_cuda_quant<idx_t, block_oscar2_kv, QK_OSCAR2_KV, quantize_f32_oscar2_v_block>(
                    src0_d, src1_d, (block_oscar2_kv*)dst->data,
                    ne00, ne01, ne02, ne03,
                    ne10, ne11, ne12, ne13,
                    nb01, nb02, nb03,
                    nb10, nb11, nb12,
                    nb1, nb2, nb3,
                    stream
                );
            }
        } else {
            const char * env_k_residual = getenv("LLAMA_KV_OSCAR2_K_RESIDUAL");
            const bool use_k_residual = env_k_residual && env_k_residual[0] != '\0' && env_k_residual[0] != '0';
            if (!use_k_residual && !set_rows_cuda_oscar2_parallel<idx_t, false>(
                    src0_d, src1_d, (block_oscar2_kv*)dst->data,
                    ne00, ne01, ne02, ne03,
                    ne11, ne12,
                    nb01, nb02, nb03,
                    nb10, nb11, nb12,
                    nb1, nb2, nb3,
                    stream)) {
                if (use_k_residual) {
                    set_rows_cuda_quant<idx_t, block_oscar2_kv, QK_OSCAR2_KV, quantize_f32_oscar2_k_residual_block>(
                        src0_d, src1_d, (block_oscar2_kv*)dst->data,
                        ne00, ne01, ne02, ne03,
                        ne10, ne11, ne12, ne13,
                        nb01, nb02, nb03,
                        nb10, nb11, nb12,
                        nb1, nb2, nb3,
                        stream
                    );
                } else {
                    set_rows_cuda_quant<idx_t, block_oscar2_kv, QK_OSCAR2_KV, quantize_f32_oscar2_k_block>(
                    src0_d, src1_d, (block_oscar2_kv*)dst->data,
                    ne00, ne01, ne02, ne03,
                    ne10, ne11, ne12, ne13,
                    nb01, nb02, nb03,
                    nb10, nb11, nb12,
                    nb1, nb2, nb3,
                    stream
                    );
                }
            }
        }
    } else if (dst->type == GGML_TYPE_TURBO2_0) {
        set_rows_cuda_turbo2_0<idx_t>(
            src0_d, src1_d, (block_turbo2_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TURBO3_0) {
        set_rows_cuda_turbo3_0<idx_t>(
            src0_d, src1_d, (block_turbo3_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
    }
}

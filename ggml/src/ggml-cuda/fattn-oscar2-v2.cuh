#pragma once

#include "common.cuh"

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
        int ncols);

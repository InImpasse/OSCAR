// OSCAR INT2 K-side precision diagnostic: q4 K + oscar2 V, D=128 only.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_OSCAR2_KV);

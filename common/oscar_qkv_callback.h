#pragma once

#include "oscar_oskv.h"

#include <memory>
#include <string>

struct common_params;
struct ggml_tensor;

// Dedicated post-RoPE Q/K/V callback for OSCAR rotation calibration.
struct oscar_qkv_cb_user_data {
    struct impl;
    std::unique_ptr<impl> pimpl;

    oscar_qkv_cb_user_data();
    ~oscar_qkv_cb_user_data();

    oscar_qkv_cb_user_data(const oscar_qkv_cb_user_data &) = delete;
    oscar_qkv_cb_user_data & operator=(const oscar_qkv_cb_user_data &) = delete;

    explicit oscar_qkv_cb_user_data(common_params & params);

    void set_dump_root(const std::string & dump_root);
    bool begin_prompt(int prompt_id);
    bool finish_prompt(uint32_t n_tokens, std::string & out_path, std::string & error);
};

bool oscar_qkv_cb_eval(struct ggml_tensor * t, bool ask, void * user_data);

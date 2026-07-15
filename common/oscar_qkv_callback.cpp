#include "oscar_qkv_callback.h"

#include "common.h"
#include "log.h"
#include "oscar_oskv.h"

#include <cstdio>
#include <cstdio>
#include <filesystem>
#include <vector>

struct oscar_qkv_cb_user_data::impl {
    oscar_oskv::writer writer;
    std::vector<uint8_t> staging;
    std::string dump_root;
    int prompt_id = 0;
    std::string current_path;
};

oscar_qkv_cb_user_data::oscar_qkv_cb_user_data() : pimpl(std::make_unique<impl>()) {}

oscar_qkv_cb_user_data::~oscar_qkv_cb_user_data() = default;

oscar_qkv_cb_user_data::oscar_qkv_cb_user_data(common_params & params)
    : pimpl(std::make_unique<impl>()) {
    params.cb_eval           = oscar_qkv_cb_eval;
    params.cb_eval_user_data = this;
}

void oscar_qkv_cb_user_data::set_dump_root(const std::string & dump_root) {
    pimpl->dump_root = dump_root;
    if (!dump_root.empty()) {
        std::filesystem::create_directories(dump_root);
    }
}

bool oscar_qkv_cb_user_data::begin_prompt(int prompt_id) {
    pimpl->prompt_id = prompt_id;
    pimpl->writer.reset();
    pimpl->current_path.clear();

    if (pimpl->dump_root.empty()) {
        LOG_ERR("%s: dump root is empty\n", __func__);
        return false;
    }

    char name[32];
    std::snprintf(name, sizeof(name), "prompt_%05d.oskv", prompt_id);
    pimpl->current_path = (std::filesystem::path(pimpl->dump_root) / name).string();
    return true;
}

bool oscar_qkv_cb_user_data::finish_prompt(uint32_t n_tokens, std::string & out_path, std::string & error) {
    if (pimpl->current_path.empty()) {
        error = "prompt dump path is not initialized";
        return false;
    }

    if (!pimpl->writer.write_file(pimpl->current_path, n_tokens, error)) {
        return false;
    }

    out_path = pimpl->current_path;
    std::fprintf(
        stdout,
        "oskv_dump_complete: prompt=%d tokens=%u path=%s\n",
        pimpl->prompt_id,
        n_tokens,
        out_path.c_str());
    std::fflush(stdout);
    pimpl->writer.reset();
    return true;
}

bool oscar_qkv_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = static_cast<oscar_qkv_cb_user_data *>(user_data);
    auto * pimpl = cb_data->pimpl.get();

    oscar_oskv::qkv_kind kind;
    int layer = -1;
    const bool matches = oscar_oskv::parse_qkv_name(t->name, kind, layer);

    if (ask) {
        return matches;
    }

    if (!matches || t->type != GGML_TYPE_F32) {
        return true;
    }

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * data = nullptr;
    if (is_host) {
        data = static_cast<const uint8_t *>(t->data);
    } else {
        const size_t n_bytes = ggml_nbytes(t);
        pimpl->staging.resize(n_bytes);
        ggml_backend_tensor_get(t, pimpl->staging.data(), 0, n_bytes);
        data = pimpl->staging.data();
    }

    pimpl->writer.capture_tensor(t, data);
    return true;
}

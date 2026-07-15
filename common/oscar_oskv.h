#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_tensor;

namespace oscar_oskv {

constexpr char     k_magic[4]     = { 'O', 'S', 'K', 'V' };
constexpr char     k_footer[4]    = { 'E', 'N', 'D', 'O' };
constexpr uint32_t k_format_version = 1;

enum class qkv_kind : uint8_t {
    q = 0,
    k = 1,
    v = 2,
};

bool parse_qkv_name(const char * name, qkv_kind & kind, int & layer);

struct tensor_capture {
    std::vector<float> data;
    int64_t            ne[4] = { 0, 0, 0, 0 };
};

struct layer_pass {
    std::unordered_map<std::string, tensor_capture> tensors;
};

// Accumulates post-RoPE Q/K/V tensors for one prompt and writes a compact OSKV file.
class writer {
public:
    void reset();

    void capture_tensor(const ggml_tensor * t, const uint8_t * data);

    bool write_file(const std::string & path, uint32_t n_tokens, std::string & error) const;

private:
    std::unordered_map<std::string, int>                 dump_counts_;
    std::map<int, std::map<int, layer_pass>>             layers_; // layer -> pass_idx -> tensors

    static bool select_best_pass(const std::map<int, layer_pass> & passes, layer_pass & out, int & n_tokens);
};

} // namespace oscar_oskv

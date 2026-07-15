#include "oscar_oskv.h"

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace oscar_oskv {

namespace {

constexpr size_t k_header_size = 64;
constexpr size_t k_layer_entry_size = 64;

struct file_header {
    char     magic[4];
    uint32_t version;
    uint32_t n_tokens;
    uint32_t n_layers;
    uint32_t flags;
    uint8_t  reserved[44];
};

struct layer_entry {
    uint32_t layer_id;
    uint16_t n_heads_q;
    uint16_t n_heads_k;
    uint16_t n_heads_v;
    uint16_t head_dim;
    uint32_t reserved;
    uint64_t q_offset;
    uint64_t k_offset;
    uint64_t v_offset;
    uint64_t q_size;
    uint64_t k_size;
    uint64_t v_size;
};

static_assert(sizeof(file_header) == k_header_size, "OSKV header must be 64 bytes");
static_assert(sizeof(layer_entry) == k_layer_entry_size, "OSKV layer entry must be 64 bytes");

std::string kind_name(qkv_kind kind) {
    switch (kind) {
        case qkv_kind::q: return "Qcur";
        case qkv_kind::k: return "Kcur";
        case qkv_kind::v: return "Vcur";
    }
    return {};
}

} // namespace

void writer::reset() {
    dump_counts_.clear();
    layers_.clear();
}

bool parse_qkv_name(const char * name, qkv_kind & kind, int & layer) {
    if (name == nullptr || name[0] == '\0') {
        return false;
    }

    const std::string_view view(name);
    const size_t dash = view.rfind('-');
    if (dash == std::string_view::npos || dash + 1 >= view.size()) {
        return false;
    }

    const std::string_view prefix = view.substr(0, dash);
    const std::string_view suffix = view.substr(dash + 1);
    if (prefix.find('_') != std::string_view::npos) {
        return false;
    }
    for (const char ch : suffix) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            return false;
        }
    }

    if (prefix == "Qcur") {
        kind = qkv_kind::q;
    } else if (prefix == "Kcur") {
        kind = qkv_kind::k;
    } else if (prefix == "Vcur") {
        kind = qkv_kind::v;
    } else {
        return false;
    }

    layer = std::stoi(std::string(suffix));
    return layer >= 0;
}

void writer::capture_tensor(const ggml_tensor * t, const uint8_t * data) {
    if (t == nullptr || data == nullptr) {
        return;
    }
    if (t->type != GGML_TYPE_F32) {
        return;
    }

    qkv_kind kind;
    int layer = -1;
    if (!oscar_oskv::parse_qkv_name(t->name, kind, layer)) {
        return;
    }

    const std::string tensor_name = kind_name(kind) + "-" + std::to_string(layer);
    const int pass_idx = dump_counts_[tensor_name]++;

    tensor_capture capture;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        capture.ne[i] = t->ne[i];
    }

    const size_t n_elems = static_cast<size_t>(t->ne[0] * t->ne[1] * t->ne[2] * std::max<int64_t>(t->ne[3], 1));
    capture.data.resize(n_elems);
    std::memcpy(capture.data.data(), data, n_elems * sizeof(float));

    layers_[layer][pass_idx].tensors[kind_name(kind)] = std::move(capture);
}

bool writer::select_best_pass(const std::map<int, layer_pass> & passes, layer_pass & out, int & n_tokens) {
    bool found = false;
    int best_tokens = -1;

    for (const auto & [pass_idx, pass] : passes) {
        (void) pass_idx;
        const auto q_it = pass.tensors.find("Qcur");
        const auto k_it = pass.tensors.find("Kcur");
        const auto v_it = pass.tensors.find("Vcur");
        if (q_it == pass.tensors.end() || k_it == pass.tensors.end() || v_it == pass.tensors.end()) {
            continue;
        }

        const int tokens = static_cast<int>(q_it->second.ne[2]);
        if (tokens <= 1) {
            continue;
        }
        if (int(k_it->second.ne[2]) != tokens || int(v_it->second.ne[2]) != tokens) {
            continue;
        }
        if (!found || tokens > best_tokens) {
            found = true;
            best_tokens = tokens;
            out = pass;
            n_tokens = tokens;
        }
    }

    return found;
}

static bool select_pass_with_tokens(const std::map<int, layer_pass> & passes, int target_tokens, layer_pass & out) {
    int best_tokens = -1;
    bool found = false;
    for (const auto & [pass_idx, pass] : passes) {
        (void) pass_idx;
        const auto q_it = pass.tensors.find("Qcur");
        const auto k_it = pass.tensors.find("Kcur");
        const auto v_it = pass.tensors.find("Vcur");
        if (q_it == pass.tensors.end() || k_it == pass.tensors.end() || v_it == pass.tensors.end()) {
            continue;
        }
        const int tokens = static_cast<int>(q_it->second.ne[2]);
        if (tokens <= 1 || tokens != target_tokens) {
            continue;
        }
        if (int(k_it->second.ne[2]) != tokens || int(v_it->second.ne[2]) != tokens) {
            continue;
        }
        if (!found || tokens > best_tokens) {
            found = true;
            best_tokens = tokens;
            out = pass;
        }
    }
    return found;
}

static bool find_common_token_count(const std::map<int, std::map<int, layer_pass>> & layers, int & n_tokens) {
    std::vector<int> candidates;
    for (const auto & [layer_id, passes] : layers) {
        (void) layer_id;
        for (const auto & [pass_idx, pass] : passes) {
            (void) pass_idx;
            const auto q_it = pass.tensors.find("Qcur");
            const auto k_it = pass.tensors.find("Kcur");
            const auto v_it = pass.tensors.find("Vcur");
            if (q_it == pass.tensors.end() || k_it == pass.tensors.end() || v_it == pass.tensors.end()) {
                continue;
            }
            const int tokens = static_cast<int>(q_it->second.ne[2]);
            if (tokens <= 1) {
                continue;
            }
            if (int(k_it->second.ne[2]) == tokens && int(v_it->second.ne[2]) == tokens) {
                candidates.push_back(tokens);
            }
        }
    }
    if (candidates.empty()) {
        return false;
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    for (auto it = candidates.rbegin(); it != candidates.rend(); ++it) {
        const int candidate = *it;
        bool all_layers = true;
        for (const auto & [layer_id, passes] : layers) {
            (void) layer_id;
            layer_pass pass;
            if (!select_pass_with_tokens(passes, candidate, pass)) {
                all_layers = false;
                break;
            }
        }
        if (all_layers) {
            n_tokens = candidate;
            return true;
        }
    }
    return false;
}

bool writer::write_file(const std::string & path, uint32_t n_tokens, std::string & error) const {
    if (layers_.empty()) {
        error = "no Q/K/V tensors captured";
        return false;
    }

    struct selected_layer {
        int layer_id;
        layer_pass pass;
    };

    std::vector<selected_layer> selected;
    selected.reserve(layers_.size());

    int common_tokens = 0;
    if (n_tokens == 0) {
        if (!find_common_token_count(layers_, common_tokens)) {
            error = "no common prefill token count across layers";
            return false;
        }
        n_tokens = static_cast<uint32_t>(common_tokens);
    } else {
        common_tokens = static_cast<int>(n_tokens);
    }

    for (const auto & [layer_id, passes] : layers_) {
        layer_pass pass;
        if (!select_pass_with_tokens(passes, common_tokens, pass)) {
            continue;
        }
        selected.push_back({ layer_id, std::move(pass) });
    }

    if (selected.empty()) {
        error = "no complete prefill Q/K/V passes found";
        return false;
    }

    std::sort(selected.begin(), selected.end(), [](const selected_layer & a, const selected_layer & b) {
        return a.layer_id < b.layer_id;
    });

    std::vector<uint8_t> payload;
    std::vector<layer_entry> entries;
    entries.reserve(selected.size());

    const uint64_t payload_base = k_header_size + selected.size() * k_layer_entry_size;

    for (const auto & item : selected) {
        const auto & q = item.pass.tensors.at("Qcur");
        const auto & k = item.pass.tensors.at("Kcur");
        const auto & v = item.pass.tensors.at("Vcur");

        layer_entry entry {};
        entry.layer_id = static_cast<uint32_t>(item.layer_id);
        entry.n_heads_q = static_cast<uint16_t>(q.ne[1]);
        entry.n_heads_k = static_cast<uint16_t>(k.ne[1]);
        entry.n_heads_v = static_cast<uint16_t>(v.ne[1]);
        entry.head_dim = static_cast<uint16_t>(q.ne[0]);
        entry.q_offset = payload_base + payload.size();
        entry.q_size = q.data.size() * sizeof(float);
        payload.insert(payload.end(),
                       reinterpret_cast<const uint8_t *>(q.data.data()),
                       reinterpret_cast<const uint8_t *>(q.data.data()) + entry.q_size);
        entry.k_offset = payload_base + payload.size();
        entry.k_size = k.data.size() * sizeof(float);
        payload.insert(payload.end(),
                       reinterpret_cast<const uint8_t *>(k.data.data()),
                       reinterpret_cast<const uint8_t *>(k.data.data()) + entry.k_size);
        entry.v_offset = payload_base + payload.size();
        entry.v_size = v.data.size() * sizeof(float);
        payload.insert(payload.end(),
                       reinterpret_cast<const uint8_t *>(v.data.data()),
                       reinterpret_cast<const uint8_t *>(v.data.data()) + entry.v_size);
        entries.push_back(entry);
    }

    file_header header {};
    std::memcpy(header.magic, k_magic, sizeof(k_magic));
    header.version = k_format_version;
    header.n_tokens = n_tokens;
    header.n_layers = static_cast<uint32_t>(entries.size());

    namespace fs = std::filesystem;
    fs::create_directories(fs::path(path).parent_path());

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        error = "failed to open OSKV output file: " + path;
        return false;
    }

    out.write(reinterpret_cast<const char *>(&header), sizeof(header));
    out.write(reinterpret_cast<const char *>(entries.data()), entries.size() * sizeof(layer_entry));
    out.write(reinterpret_cast<const char *>(payload.data()), payload.size());
    out.write(k_footer, sizeof(k_footer));
    if (!out) {
        error = "failed while writing OSKV output file: " + path;
        return false;
    }

    return true;
}

} // namespace oscar_oskv

#include "oscar_qkv_callback.h"
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct calibrate_cli {
    std::string dataset;
    std::string dump_root;
    std::string dump_format = "oskv";
    int max_prompts = -1;
    int token_budget = -1;
};

static void print_usage(int /*argc*/, char ** argv) {
    LOG(
        "usage:\n"
        "  %s -m model.gguf -p \"prompt text\" [options]\n"
        "  %s -m model.gguf --dataset prompts.jsonl --dump-root out/raw [options]\n"
        "\n"
        "Dump post-RoPE Qcur/Kcur/Vcur tensors for OSCAR KV rotation calibration.\n"
        "Default output uses compact per-prompt OSKV files under --dump-root.\n"
        "Legacy debug raw dumps remain available via --dump-format legacy.\n"
        "\n"
        "Multi-prompt options:\n"
        "  --dataset PATH         JSONL or plain-text prompt file\n"
        "  --dump-root PATH       Root directory for per-prompt dumps\n"
        "  --dump-format FORMAT   oskv (default) or legacy\n"
        "  --max-prompts N        Limit number of prompts from the dataset\n"
        "  --token-budget N       Stop after dumping at least N prefill tokens\n"
        "\n"
        "Example:\n"
        "  %s -m model.gguf --dataset prompts.jsonl --dump-root out/oskv -n 1 -c 4096 -ngl 99 -fa on --no-warmup\n",
        argv[0], argv[0], argv[0]);
}

static bool parse_calibrate_cli(int & argc, char ** argv, calibrate_cli & cli) {
    std::vector<char *> kept;
    kept.reserve(argc);
    kept.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto take_value = [&](std::string & out) -> bool {
            if (i + 1 >= argc) {
                LOG_ERR("missing value for %s\n", arg.c_str());
                return false;
            }
            out = argv[++i];
            return true;
        };

        if (arg == "--dataset") {
            if (!take_value(cli.dataset)) {
                return false;
            }
            continue;
        }
        if (arg == "--dump-root") {
            if (!take_value(cli.dump_root)) {
                return false;
            }
            continue;
        }
        if (arg == "--dump-format") {
            if (!take_value(cli.dump_format)) {
                return false;
            }
            continue;
        }
        if (arg == "--max-prompts") {
            std::string value;
            if (!take_value(value)) {
                return false;
            }
            cli.max_prompts = std::stoi(value);
            continue;
        }
        if (arg == "--token-budget") {
            std::string value;
            if (!take_value(value)) {
                return false;
            }
            cli.token_budget = std::stoi(value);
            continue;
        }

        kept.push_back(argv[i]);
    }

    for (int i = 0; i < (int) kept.size(); ++i) {
        argv[i] = kept[i];
    }
    argc = (int) kept.size();
    return true;
}

static std::string trim_copy(const std::string & value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(start, end - start);
}

static std::string extract_json_string(const std::string & line, const std::string & key) {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = line.find(needle);
    if (key_pos == std::string::npos) {
        return {};
    }
    const size_t colon = line.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return {};
    }
    const size_t quote_start = line.find('"', colon + 1);
    if (quote_start == std::string::npos) {
        return {};
    }
    std::string out;
    bool escape = false;
    for (size_t i = quote_start + 1; i < line.size(); ++i) {
        const char ch = line[i];
        if (escape) {
            out.push_back(ch);
            escape = false;
            continue;
        }
        if (ch == '\\') {
            escape = true;
            continue;
        }
        if (ch == '"') {
            break;
        }
        out.push_back(ch);
    }
    return out;
}

static std::string prompt_from_line(const std::string & line) {
    const std::string trimmed = trim_copy(line);
    if (trimmed.empty()) {
        return {};
    }
    if (trimmed.front() == '{') {
        for (const char * key : {"prompt", "question", "text", "input"}) {
            const std::string value = extract_json_string(trimmed, key);
            if (!value.empty()) {
                return value;
            }
        }
        return {};
    }
    return trimmed;
}

static bool read_prompts(const std::string & dataset_path, int max_prompts, std::vector<std::string> & prompts) {
    std::ifstream in(dataset_path);
    if (!in) {
        LOG_ERR("%s: failed to open dataset %s\n", __func__, dataset_path.c_str());
        return false;
    }

    std::string line;
    while (std::getline(in, line)) {
        const std::string prompt = prompt_from_line(line);
        if (prompt.empty()) {
            continue;
        }
        prompts.push_back(prompt);
        if (max_prompts > 0 && (int) prompts.size() >= max_prompts) {
            break;
        }
    }

    if (prompts.empty()) {
        LOG_ERR("%s: dataset %s did not contain any prompts\n", __func__, dataset_path.c_str());
        return false;
    }
    return true;
}

static bool reset_context_state(llama_context * ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        LOG_ERR("%s: failed to get llama memory\n", __func__);
        return false;
    }
    llama_memory_clear(mem, true);
    llama_perf_context_reset(ctx);
    llama_synchronize(ctx);
    return true;
}

static bool run_prompt_oskv(
    llama_context * ctx,
    const std::string & prompt,
    oscar_qkv_cb_user_data & cb_data,
    int prompt_id,
    size_t & prompt_tokens
) {
    prompt_tokens = 0;
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    if (!reset_context_state(ctx)) {
        return false;
    }
    if (!cb_data.begin_prompt(prompt_id)) {
        return false;
    }

    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, add_bos);
    if (tokens.empty()) {
        LOG_ERR("%s: no input tokens for prompt\n", __func__);
        return false;
    }

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s: failed to eval prompt\n", __func__);
        return false;
    }

    std::string out_path;
    std::string error;
    if (!cb_data.finish_prompt(static_cast<uint32_t>(tokens.size()), out_path, error)) {
        LOG_ERR("%s: failed to finalize OSKV dump for prompt %d: %s\n", __func__, prompt_id, error.c_str());
        return false;
    }

    prompt_tokens = tokens.size();
    return true;
}

static bool run_dataset_oskv(
    llama_model * model,
    common_params & params,
    oscar_qkv_cb_user_data & cb_data,
    const calibrate_cli & cli,
    const std::vector<std::string> & prompts
) {
    cb_data.set_dump_root(cli.dump_root);
    params.cb_eval           = oscar_qkv_cb_eval;
    params.cb_eval_user_data = &cb_data;

    const llama_context_params cparams = common_context_params_to_llama(params);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        LOG_ERR("%s: failed to create persistent context\n", __func__);
        return false;
    }

    bool ok = true;
    size_t dumped_tokens = 0;
    size_t captured_prompts = 0;
    for (size_t i = 0; i < prompts.size(); ++i) {
        if (cli.token_budget > 0 && dumped_tokens >= (size_t) cli.token_budget) {
            std::fprintf(
                stdout,
                "token budget reached before prompt %zu: dumped_tokens=%zu budget=%d\n",
                i + 1,
                dumped_tokens,
                cli.token_budget
            );
            std::fflush(stdout);
            break;
        }
        params.prompt = prompts[i];
        std::fprintf(stdout, "processing prompt %zu/%zu\n", i + 1, prompts.size());
        std::fflush(stdout);
        size_t prompt_tokens = 0;
        if (!run_prompt_oskv(ctx, prompts[i], cb_data, (int) i + 1, prompt_tokens)) {
            ok = false;
            break;
        }
        dumped_tokens += prompt_tokens;
        captured_prompts += 1;
        if (cli.token_budget > 0 && dumped_tokens >= (size_t) cli.token_budget) {
            std::fprintf(
                stdout,
                "token budget reached after prompt %zu: dumped_tokens=%zu budget=%d\n",
                i + 1,
                dumped_tokens,
                cli.token_budget
            );
            std::fflush(stdout);
            break;
        }
    }
    std::fprintf(
        stdout,
        "dataset dump summary: captured_prompts=%zu dumped_tokens=%zu\n",
        captured_prompts,
        dumped_tokens
    );
    std::fflush(stdout);

    llama_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    common_params params;
    calibrate_cli cli;

    common_init();

    if (!parse_calibrate_cli(argc, argv, cli)) {
        return 1;
    }

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DEBUG, print_usage)) {
        return 1;
    }

    if (cli.dump_format != "oskv") {
        LOG_ERR("%s: only --dump-format oskv is supported in this build\n", __func__);
        return 1;
    }

    std::vector<std::string> prompts;
    const bool multi_prompt = !cli.dataset.empty();
    if (multi_prompt) {
        if (cli.dump_root.empty()) {
            LOG_ERR("%s: --dump-root is required with --dataset\n", __func__);
            return 1;
        }
        if (!read_prompts(cli.dataset, cli.max_prompts, prompts)) {
            return 1;
        }
    } else if (params.prompt.empty()) {
        LOG_ERR("%s: provide -p/--prompt or --dataset\n", __func__);
        return 1;
    } else if (cli.dump_root.empty()) {
        LOG_ERR("%s: --dump-root is required for OSKV output\n", __func__);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    oscar_qkv_cb_user_data cb_data(params);

    bool ok = false;
    auto llama_init = common_init_from_params(params, true);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        LOG_ERR("%s: failed to init model\n", __func__);
        return 1;
    }

    if (multi_prompt) {
        ok = run_dataset_oskv(model, params, cb_data, cli, prompts);
    } else {
        ok = run_dataset_oskv(model, params, cb_data, cli, { params.prompt });
    }

    if (!ok) {
        return 1;
    }

    llama_backend_free();
    return 0;
}

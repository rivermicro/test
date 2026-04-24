#include "cli.hpp"

#include "ggml-backend.h"
#include "llama.h"
#include "paths.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

Options make_default_options() {
    Options options;
    options.model_path = "model/llama3.2-1b.gguf";
    options.system_prompt = "You are Yedera, a helpful local assistant running on garage server.";
    options.debug = true;
    return options;
}

int parse_int(const char * value, const char * flag_name) {
    try {
        return std::stoi(value);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string("invalid value for ") + flag_name + ": " + value);
    }
}

float parse_float(const char * value, const char * flag_name) {
    try {
        return std::stof(value);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string("invalid value for ") + flag_name + ": " + value);
    }
}

uint32_t parse_seed(const std::string & value, const char * flag_name) {
    if (value == "random") {
        return LLAMA_DEFAULT_SEED;
    }

    try {
        unsigned long long parsed = std::stoull(value);
        if (parsed > UINT32_MAX) {
            throw std::out_of_range("seed out of range");
        }
        return static_cast<uint32_t>(parsed);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string("invalid value for ") + flag_name + ": " + value);
    }
}

bool parse_bool_value(const std::string & value, const char * key_name) {
    if (value == "true" || value == "1" || value == "yes" || value == "on") {
        return true;
    }

    if (value == "false" || value == "0" || value == "no" || value == "off") {
        return false;
    }

    throw std::runtime_error(std::string("invalid boolean for ") + key_name + ": " + value);
}

std::string strip_wrapping_quotes(const std::string & value) {
    if (value.size() >= 2) {
        const char first = value.front();
        const char last = value.back();
        if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
            return value.substr(1, value.size() - 2);
        }
    }

    return value;
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string resolve_config_relative_path(const std::filesystem::path & config_path, const std::string & value) {
    if (value.empty()) {
        return value;
    }

    const std::filesystem::path parsed_path(value);
    if (parsed_path.is_absolute()) {
        return parsed_path.lexically_normal().string();
    }

    const std::filesystem::path config_directory = config_path.parent_path();
    if (config_directory.empty()) {
        return parsed_path.lexically_normal().string();
    }

    return (config_directory / parsed_path).lexically_normal().string();
}

std::filesystem::path default_config_path() {
    std::error_code error;
    const std::filesystem::path current_directory = std::filesystem::current_path(error);
    if (error) {
        throw std::runtime_error("failed to resolve the current working directory for yedera.conf");
    }

    const std::filesystem::path current_config_path = current_directory / "yedera.conf";
    if (std::filesystem::exists(current_config_path)) {
        return current_config_path;
    }

    if (const char * home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
        const std::filesystem::path home_config_path = std::filesystem::path(home) / ".yedera" / "yedera.conf";
        if (std::filesystem::exists(home_config_path)) {
            return home_config_path;
        }
    }

    return current_config_path;
}

template <typename T>
void apply_override(T & target, const std::optional<T> & source) {
    if (source.has_value()) {
        target = *source;
    }
}

void validate_options(const Options & options) {
    if (options.n_ctx <= 0) {
        throw std::runtime_error("--ctx-size must be positive");
    }

    if (options.n_predict <= 0) {
        throw std::runtime_error("--n-predict must be positive");
    }

    if (options.temperature < 0.0f) {
        throw std::runtime_error("--temperature must be non-negative");
    }

    if (options.top_p <= 0.0f || options.top_p > 1.0f) {
        throw std::runtime_error("--top-p must be in the range (0, 1]");
    }

    if (options.min_p < 0.0f || options.min_p > 1.0f) {
        throw std::runtime_error("--min-p must be in the range [0, 1]");
    }
}

bool has_gpu_backend_device() {
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (device != nullptr && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            return true;
        }
    }

    return false;
}

OptionOverrides load_config_file(const std::filesystem::path & config_path, bool required) {
    std::ifstream input(config_path);
    if (!input.is_open()) {
        if (required) {
            if (!std::filesystem::exists(config_path)) {
                throw std::runtime_error("config file not found: " + config_path.string());
            }
            throw std::runtime_error("failed to open config file: " + config_path.string());
        }
        return {};
    }

    OptionOverrides overrides;
    std::string line;
    int line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;

        const std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }

        const size_t separator = trimmed.find('=');
        if (separator == std::string::npos) {
            throw std::runtime_error("invalid config line " + std::to_string(line_number) + " in " + config_path.string());
        }

        const std::string key = trim(trimmed.substr(0, separator));
        const std::string value = strip_wrapping_quotes(trim(trimmed.substr(separator + 1)));

        try {
            if (key == "model_path") {
                overrides.model_path = value;
            } else if (key == "model_embeddings") {
                overrides.model_embeddings = value;
            } else if (key == "rag_documents_path") {
                overrides.rag_documents_path = value;
            } else if (key == "prompt") {
                overrides.system_prompt = value;
            } else if (key == "user_prompt") {
                overrides.user_prompt = value;
            } else if (key == "system_prompt") {
                overrides.system_prompt = value;
            } else if (key == "ctx_size") {
                overrides.n_ctx = parse_int(value.c_str(), key.c_str());
            } else if (key == "n_predict") {
                overrides.n_predict = parse_int(value.c_str(), key.c_str());
            } else if (key == "n_gpu_layers") {
                overrides.n_gpu_layers = parse_int(value.c_str(), key.c_str());
            } else if (key == "temperature") {
                overrides.temperature = parse_float(value.c_str(), key.c_str());
            } else if (key == "top_p") {
                overrides.top_p = parse_float(value.c_str(), key.c_str());
            } else if (key == "min_p") {
                overrides.min_p = parse_float(value.c_str(), key.c_str());
            } else if (key == "seed") {
                overrides.seed = parse_seed(value, key.c_str());
            } else if (key == "interactive") {
                overrides.interactive = parse_bool_value(value, key.c_str());
            } else if (key == "verbose") {
                overrides.verbose = parse_bool_value(value, key.c_str());
            } else if (key == "debug") {
                overrides.debug = parse_bool_value(value, key.c_str());
            } else {
                throw std::runtime_error("unknown config key '" + key + "'");
            }
        } catch (const std::exception & error) {
            throw std::runtime_error(
                "invalid config line " + std::to_string(line_number) + " in " + config_path.string() + ": " + error.what());
        }
    }

    return overrides;
}

} // namespace

void print_usage(const char * program_name) {
    std::cout
        << "Usage: " << program_name << " [command] [options] [prompt]\n\n"
        << "Commands:\n"
        << "  list                      List model files under the model/ directory next to the binary\n\n"
        << "Options:\n"
        << "  --config PATH             Config file path (default: ./yedera.conf, then ~/.yedera/yedera.conf)\n"
        << "  -m, --model-path PATH      Path to a local GGUF model file\n"
        << "  -p, --prompt TEXT          One-shot user prompt\n"
        << "  -i, --interactive          Start an interactive chat session\n"
        << "  -s, --system-prompt TEXT   Assistant system prompt\n"
        << "  -c, --ctx-size N           Context size (default: 2048)\n"
        << "  -n, --n-predict N          Max generated tokens per response (default: 256)\n"
        << "  --temperature N            Sampling temperature (default: 0.8)\n"
        << "  --top-p N                  Nucleus sampling top-p (default: 0.95)\n"
        << "  --min-p N                  Minimum-p filter (default: 0.05)\n"
        << "  --seed N|random            Sampling seed (default: random)\n"
        << "  -ngl, --n-gpu-layers N     GPU offload layers (default: auto; all layers on NVIDIA when available)\n"
        << "  -v, --verbose              Keep llama.cpp logging enabled\n"
        << "  -h, --help                 Show this help\n\n"
        << "Interactive chat commands:\n"
        << "  /learn PATH               Learn a file or directory for the current session\n"
        << "  //TEXT                    Send a literal prompt that starts with /\n"
        << "  ESC on an empty line      Toggle between `> ` chat mode and `: ` file-learn mode\n"
        << "                            In file-learn mode, relative paths and wildcards resolve under rag_documents_path\n";
}

OptionOverrides parse_args(int argc, char ** argv) {
    OptionOverrides options;
    std::vector<std::string> positional;

    for (int index = 1; index < argc; ++index) {
        const std::string current = argv[index];

        auto require_value = [&](const char * flag_name) -> const char * {
            if (index + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + flag_name);
            }
            return argv[++index];
        };

        if (current == "--config") {
            options.config_path = require_value(current.c_str());
        } else if (current == "-m" || current == "--model-path") {
            options.model_path = require_value(current.c_str());
        } else if (current == "-p" || current == "--prompt") {
            options.user_prompt = require_value(current.c_str());
        } else if (current == "-i" || current == "--interactive") {
            options.interactive = true;
        } else if (current == "-s" || current == "--system-prompt") {
            options.system_prompt = require_value(current.c_str());
        } else if (current == "-c" || current == "--ctx-size") {
            options.n_ctx = parse_int(require_value(current.c_str()), current.c_str());
        } else if (current == "-n" || current == "--n-predict") {
            options.n_predict = parse_int(require_value(current.c_str()), current.c_str());
        } else if (current == "--temperature") {
            options.temperature = parse_float(require_value(current.c_str()), current.c_str());
        } else if (current == "--top-p") {
            options.top_p = parse_float(require_value(current.c_str()), current.c_str());
        } else if (current == "--min-p") {
            options.min_p = parse_float(require_value(current.c_str()), current.c_str());
        } else if (current == "--seed") {
            options.seed = parse_seed(require_value(current.c_str()), current.c_str());
        } else if (current == "-ngl" || current == "--n-gpu-layers") {
            options.n_gpu_layers = parse_int(require_value(current.c_str()), current.c_str());
        } else if (current == "-v" || current == "--verbose") {
            options.verbose = true;
        } else if (current == "-h" || current == "--help") {
            options.help = true;
        } else {
            positional.push_back(current);
        }
    }

    if (!positional.empty()) {
        std::string prompt = positional.front();
        for (size_t index = 1; index < positional.size(); ++index) {
            prompt += " ";
            prompt += positional[index];
        }
        options.user_prompt = prompt;
    }

    return options;
}

Options resolve_options(const OptionOverrides & cli_overrides) {
    Options options = make_default_options();
    const bool explicit_config = cli_overrides.config_path.has_value();
    const std::filesystem::path config_path = explicit_config ? std::filesystem::path(*cli_overrides.config_path) : default_config_path();
    options.config_path = config_path.string();

    const OptionOverrides config_overrides = load_config_file(config_path, true);

    apply_override(options.model_path, config_overrides.model_path);
    apply_override(options.model_embeddings, config_overrides.model_embeddings);
    apply_override(options.rag_documents_path, config_overrides.rag_documents_path);
    apply_override(options.prompt, config_overrides.user_prompt);
    apply_override(options.system_prompt, config_overrides.system_prompt);
    apply_override(options.n_ctx, config_overrides.n_ctx);
    apply_override(options.n_predict, config_overrides.n_predict);
    apply_override(options.n_gpu_layers, config_overrides.n_gpu_layers);
    apply_override(options.temperature, config_overrides.temperature);
    apply_override(options.top_p, config_overrides.top_p);
    apply_override(options.min_p, config_overrides.min_p);
    apply_override(options.seed, config_overrides.seed);
    apply_override(options.interactive, config_overrides.interactive);
    apply_override(options.verbose, config_overrides.verbose);
    apply_override(options.debug, config_overrides.debug);

    if (config_overrides.model_path.has_value()) {
        options.model_path = resolve_config_relative_path(config_path, options.model_path);
    }
    if (config_overrides.model_embeddings.has_value()) {
        options.model_embeddings = resolve_config_relative_path(config_path, options.model_embeddings);
    }
    if (config_overrides.rag_documents_path.has_value()) {
        options.rag_documents_path = resolve_config_relative_path(config_path, options.rag_documents_path);
    }

    apply_override(options.model_path, cli_overrides.model_path);
    apply_override(options.model_embeddings, cli_overrides.model_embeddings);
    apply_override(options.rag_documents_path, cli_overrides.rag_documents_path);
    apply_override(options.prompt, cli_overrides.user_prompt);
    apply_override(options.system_prompt, cli_overrides.system_prompt);
    apply_override(options.n_ctx, cli_overrides.n_ctx);
    apply_override(options.n_predict, cli_overrides.n_predict);
    apply_override(options.n_gpu_layers, cli_overrides.n_gpu_layers);
    apply_override(options.temperature, cli_overrides.temperature);
    apply_override(options.top_p, cli_overrides.top_p);
    apply_override(options.min_p, cli_overrides.min_p);
    apply_override(options.seed, cli_overrides.seed);
    apply_override(options.interactive, cli_overrides.interactive);
    apply_override(options.verbose, cli_overrides.verbose);
    apply_override(options.debug, cli_overrides.debug);

    const bool gpu_layers_explicit = config_overrides.n_gpu_layers.has_value() || cli_overrides.n_gpu_layers.has_value();
    if (!gpu_layers_explicit) {
        options.n_gpu_layers = llama_supports_gpu_offload() && has_gpu_backend_device() ? -1 : 0;
    }

    if (options.prompt.empty()) {
        options.interactive = true;
    }

    if (options.system_prompt.empty()) {
        throw std::runtime_error("missing assistant prompt in config; set prompt = \"...\" in " + options.config_path + " or pass --system-prompt");
    }

    if (options.model_path.empty()) {
        throw std::runtime_error("missing model path; set model_path in " + options.config_path + " or pass --model-path");
    }

    options.help = cli_overrides.help;
    validate_options(options);

    return options;
}

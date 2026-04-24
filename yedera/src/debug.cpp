#include "debug.hpp"

#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#ifndef YEDERA_CUDA_ENABLED
#define YEDERA_CUDA_ENABLED 0
#endif

#ifndef YEDERA_CUDA_VERSION
#define YEDERA_CUDA_VERSION "unknown"
#endif

namespace {

std::string describe_model_path(const std::string & configured_path) {
    const std::filesystem::path configured_model_path(configured_path);
    std::error_code model_path_error;
    const std::filesystem::path absolute_model_path = std::filesystem::absolute(configured_model_path, model_path_error).lexically_normal();
    if (!model_path_error && !configured_model_path.is_absolute()) {
        return configured_model_path.string() + " (resolved to " + absolute_model_path.string() + ")";
    }

    return configured_model_path.string();
}

void finish_model_load_progress_line(DebugState * debug_state) {
    if (debug_state != nullptr && debug_state->load_progress_active) {
        std::fputc('\n', stderr);
        std::fflush(stderr);
        debug_state->load_progress_active = false;
    }
}

void print_debug_message(const std::string & message) {
    std::cerr << "[debug] " << message << '\n';
}

bool should_emit_debug_log(std::string_view message) {
    return message.find("llama_model_load_from_file_impl: using device") != std::string_view::npos ||
           message.find("load_tensors: offloading") != std::string_view::npos ||
           message.find("load_tensors: offloaded") != std::string_view::npos ||
           (message.find("load_tensors:") != std::string_view::npos && message.find("model buffer size") != std::string_view::npos) ||
           (message.find("llama_kv_cache:") != std::string_view::npos && message.find("KV buffer size") != std::string_view::npos) ||
           (message.find("llama_context:") != std::string_view::npos && message.find("compute buffer size =") != std::string_view::npos);
}

void llama_log_router(enum ggml_log_level level, const char * text, void * user_data) {
    auto * debug_state = static_cast<DebugState *>(user_data);
    if (text == nullptr) {
        return;
    }

    if (debug_state != nullptr && debug_state->verbose) {
        finish_model_load_progress_line(debug_state);
        std::fprintf(stderr, "%s", text);
        return;
    }

    if (level >= GGML_LOG_LEVEL_ERROR) {
        finish_model_load_progress_line(debug_state);
        std::fprintf(stderr, "%s", text);
        return;
    }

    if (debug_state != nullptr && debug_state->enabled && should_emit_debug_log(text)) {
        finish_model_load_progress_line(debug_state);
        std::fprintf(stderr, "%s", text);
    }
}

} // namespace

void install_startup_log_router() {
    llama_log_set(llama_log_router, nullptr);
}

void install_runtime_log_router(DebugState * debug_state) {
    llama_log_set(llama_log_router, debug_state);
}

bool handle_model_load_progress(float progress, void * user_data) {
    auto * debug_state = static_cast<DebugState *>(user_data);
    if (debug_state == nullptr || !debug_state->enabled) {
        return true;
    }

    const int percent = std::clamp(static_cast<int>(progress * 100.0f + 0.5f), 0, 100);
    const int bucket = percent / 5;
    if (bucket != debug_state->load_progress_bucket) {
        debug_state->load_progress_bucket = bucket;
        debug_state->load_progress_active = true;
        std::fprintf(stderr, "\r[debug] model load: %d%%", percent);
        std::fflush(stderr);
    }

    if (percent >= 100) {
        finish_model_load_progress_line(debug_state);
    }

    return true;
}

void print_backend_debug_summary(const Options & options) {
    if (!options.debug) {
        return;
    }

    print_debug_message("using model " + describe_model_path(options.model_path));

    if (!options.model_embeddings.empty()) {
        print_debug_message("using embeddings model " + describe_model_path(options.model_embeddings));
        print_debug_message("RAG mode enabled");
    } else {
        print_debug_message("RAG mode disabled");
    }

    if (YEDERA_CUDA_ENABLED) {
        print_debug_message(std::string("CUDA backend built in: yes (toolkit ") + YEDERA_CUDA_VERSION + ")");
    } else {
        print_debug_message("CUDA backend built in: no");
    }

    size_t accelerator_count = 0;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (device == nullptr || ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }

        ++accelerator_count;

        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(device, &props);
        const char * backend_name = "unknown";
        if (ggml_backend_reg_t backend_reg = ggml_backend_dev_backend_reg(device); backend_reg != nullptr) {
            backend_name = ggml_backend_reg_name(backend_reg);
        }

        const char * description = props.description != nullptr ? props.description : ggml_backend_dev_description(device);
        const size_t memory_mib = props.memory_total / (1024 * 1024);
        print_debug_message(
            "detected device " + std::to_string(accelerator_count) + ": backend " + backend_name + ", " +
            (description != nullptr ? std::string(description) : std::string("unknown device")) + ", " +
            std::to_string(memory_mib) + " MiB total");
    }

    if (accelerator_count == 0) {
        print_debug_message("no non-CPU backend devices detected");
    }

    if (options.n_gpu_layers < 0) {
        print_debug_message("offload request: all eligible model layers to GPU");
    } else if (options.n_gpu_layers == 0) {
        print_debug_message("offload request: CPU-only");
    } else {
        print_debug_message("offload request: up to " + std::to_string(options.n_gpu_layers) + " model layers to GPU");
    }

    print_debug_message(
        "CPU/host work: chat template formatting, tokenization, sampling, and any tensors not offloaded by llama.cpp remain on CPU/host");
}
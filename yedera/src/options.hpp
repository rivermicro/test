#pragma once

#include "llama.h"

#include <cstdint>
#include <optional>
#include <string>

struct Options {
    std::string config_path;
    std::string model_path;
    std::string model_embeddings;
    std::string rag_documents_path;
    std::string prompt;
    std::string system_prompt;
    int n_ctx = 2048;
    int n_predict = 256;
    int n_gpu_layers = 0;
    float temperature = 0.8f;
    float top_p = 0.95f;
    float min_p = 0.05f;
    uint32_t seed = LLAMA_DEFAULT_SEED;
    bool interactive = false;
    bool verbose = false;
    bool debug = false;
    bool help = false;
};

struct OptionOverrides {
    std::optional<std::string> config_path;
    std::optional<std::string> model_path;
    std::optional<std::string> model_embeddings;
    std::optional<std::string> rag_documents_path;
    std::optional<std::string> user_prompt;
    std::optional<std::string> system_prompt;
    std::optional<int> n_ctx;
    std::optional<int> n_predict;
    std::optional<int> n_gpu_layers;
    std::optional<float> temperature;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<uint32_t> seed;
    std::optional<bool> interactive;
    std::optional<bool> verbose;
    std::optional<bool> debug;
    bool help = false;
};
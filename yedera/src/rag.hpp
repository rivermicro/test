#pragma once

#include "options.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct RagState;

using RagStatePtr = std::unique_ptr<RagState, void (*)(RagState *)>;
using RagProgressCallback = std::function<void(const std::string & source_path, size_t completed, size_t total)>;

void destroy_rag_state(RagState * rag_state);
RagStatePtr create_rag_state(const Options & options);
void learn_rag_source(
    RagStatePtr & rag_state,
    const Options & options,
    const std::string & source_path,
    const RagProgressCallback & progress_callback = {});
void learn_rag_sources(
    RagStatePtr & rag_state,
    const Options & options,
    const std::vector<std::filesystem::path> & source_paths,
    const RagProgressCallback & progress_callback = {});
void replace_rag_sources(
    RagStatePtr & rag_state,
    const Options & options,
    const std::vector<std::filesystem::path> & source_paths,
    const RagProgressCallback & progress_callback = {});
void forget_rag_sources(
    RagStatePtr & rag_state,
    const std::vector<std::filesystem::path> & source_paths);
void clear_rag_sources(RagStatePtr & rag_state);
size_t rag_chunk_count(const RagState * rag_state);
size_t rag_chunk_count_for_source(const RagState * rag_state, const std::filesystem::path & source_path);
bool should_use_rag_for_input(const std::string & user_input);
std::string format_rag_prompt(const std::string & retrieved_context, const std::string & user_input);
std::string augment_prompt_with_rag(RagState * rag_state, const std::string & user_input);
std::string augment_prompt_with_rag(
    RagState * rag_state,
    const std::string & user_input,
    std::vector<std::string> & retrieved_sources);

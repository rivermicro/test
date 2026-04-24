#pragma once

#include "options.hpp"

#include <memory>
#include <string>

struct RagState;

using RagStatePtr = std::unique_ptr<RagState, void (*)(RagState *)>;

void destroy_rag_state(RagState * rag_state);
RagStatePtr create_rag_state(const Options & options);
void learn_rag_source(RagStatePtr & rag_state, const Options & options, const std::string & source_path);
std::string augment_prompt_with_rag(RagState * rag_state, const std::string & user_input);
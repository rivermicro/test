#include "rag.hpp"

#include "llama.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kChunkSize = 800;
constexpr size_t kChunkOverlap = 160;
constexpr size_t kTopK = 3;

void debug_log(const Options & options, const std::string & message) {
    if (options.debug) {
        std::cerr << "[debug] RAG: " << message << '\n';
    }
}

void debug_file_log(const Options & options, const std::string & file_path) {
    if (options.debug) {
        std::cerr << "[rag] " << file_path << '\n';
    }
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }

    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

bool contains_text(const std::string & value, const std::string & needle) {
    return value.find(needle) != std::string::npos;
}

bool contains_digit(const std::string & value) {
    return std::any_of(value.begin(), value.end(), [](unsigned char character) {
        return std::isdigit(character);
    });
}

bool looks_like_lookup_request(const std::string & input) {
    return contains_digit(input) || contains_text(input, "item") || contains_text(input, "price") ||
           contains_text(input, "buy it now") || contains_text(input, "sku") || contains_text(input, "record") ||
           contains_text(input, "lookup") || contains_text(input, "find ");
}

bool is_supported_document_file(const std::filesystem::path & path) {
    if (!std::filesystem::is_regular_file(path)) {
        return false;
    }

    const std::string extension = to_lower(path.extension().string());
    return extension == ".txt" || extension == ".md" || extension == ".markdown" || extension == ".rst" ||
           extension == ".log" || extension == ".csv" || extension == ".json" || extension == ".yaml" ||
           extension == ".yml";
}

std::vector<std::filesystem::path> collect_document_files(const std::filesystem::path & documents_path) {
    std::vector<std::filesystem::path> files;

    if (std::filesystem::is_regular_file(documents_path)) {
        if (is_supported_document_file(documents_path)) {
            files.push_back(documents_path);
        }
        return files;
    }

    for (const auto & entry : std::filesystem::recursive_directory_iterator(documents_path)) {
        if (is_supported_document_file(entry.path())) {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

std::string read_text_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open RAG document: " + path.string());
    }

    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::vector<std::string> split_into_chunks(const std::string & text) {
    std::vector<std::string> chunks;
    size_t start = 0;

    while (start < text.size()) {
        size_t end = std::min(text.size(), start + kChunkSize);
        if (end < text.size()) {
            const size_t min_break = start + (kChunkSize / 2);
            const size_t split = text.find_last_of("\n.?! ", end);
            if (split != std::string::npos && split > min_break) {
                end = split + 1;
            }
        }

        if (end <= start) {
            end = std::min(text.size(), start + kChunkSize);
        }

        const std::string chunk = trim(text.substr(start, end - start));
        if (!chunk.empty()) {
            chunks.push_back(chunk);
        }

        if (end >= text.size()) {
            break;
        }

        start = end > kChunkOverlap ? end - kChunkOverlap : end;
    }

    return chunks;
}

std::filesystem::path normalize_source_path(const std::filesystem::path & file_path) {
    std::error_code error;
    const std::filesystem::path absolute_path = std::filesystem::absolute(file_path, error);
    if (!error) {
        return absolute_path.lexically_normal();
    }

    return file_path.lexically_normal();
}

std::vector<float> normalize_embedding(const float * embedding, int embedding_size) {
    std::vector<float> normalized(static_cast<size_t>(embedding_size));

    double magnitude_sq = 0.0;
    for (int index = 0; index < embedding_size; ++index) {
        magnitude_sq += static_cast<double>(embedding[index]) * static_cast<double>(embedding[index]);
    }

    const double scale = magnitude_sq > 0.0 ? 1.0 / std::sqrt(magnitude_sq) : 1.0;
    for (int index = 0; index < embedding_size; ++index) {
        normalized[static_cast<size_t>(index)] = static_cast<float>(embedding[index] * scale);
    }

    return normalized;
}

float cosine_similarity(const std::vector<float> & left, const std::vector<float> & right) {
    const size_t size = std::min(left.size(), right.size());
    float similarity = 0.0f;
    for (size_t index = 0; index < size; ++index) {
        similarity += left[index] * right[index];
    }
    return similarity;
}

class EmbeddingEngine {
public:
    EmbeddingEngine(const Options & options, const std::filesystem::path & model_path)
        : options_(options), model_path_(model_path) {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = options_.n_gpu_layers;
        model_ = llama_model_load_from_file(model_path_.c_str(), model_params);
        if (model_ == nullptr) {
            throw std::runtime_error("failed to load embeddings model: " + model_path_.string());
        }

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = std::max(512, std::min(options_.n_ctx, 2048));
        context_params.n_batch = context_params.n_ctx;
        context_params.n_ubatch = context_params.n_batch;
        context_params.embeddings = true;

        context_ = llama_init_from_model(model_, context_params);
        if (context_ == nullptr) {
            llama_model_free(model_);
            model_ = nullptr;
            throw std::runtime_error("failed to create embeddings context");
        }

        embedding_size_ = llama_model_n_embd_out(model_);
        if (embedding_size_ <= 0) {
            throw std::runtime_error("embeddings model does not expose an output embedding size: " + model_path_.string());
        }
    }

    ~EmbeddingEngine() {
        if (context_ != nullptr) {
            llama_free(context_);
        }
        if (model_ != nullptr) {
            llama_model_free(model_);
        }
    }

    std::vector<float> embed(const std::string & text) {
        const std::vector<llama_token> tokens = tokenize(text);
        if (tokens.empty()) {
            throw std::runtime_error("cannot embed empty text");
        }
        if (static_cast<int>(tokens.size()) > llama_n_ctx(context_)) {
            throw std::runtime_error("text exceeds embeddings context window for model: " + model_path_.string());
        }

        llama_memory_clear(llama_get_memory(context_), true);

        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data()), static_cast<int32_t>(tokens.size()));
        if (llama_decode(context_, batch) < 0) {
            throw std::runtime_error("failed to compute embeddings for text");
        }

        const enum llama_pooling_type pooling_type = llama_pooling_type(context_);
        const float * embedding = pooling_type == LLAMA_POOLING_TYPE_NONE
            ? llama_get_embeddings_ith(context_, static_cast<int32_t>(tokens.size()) - 1)
            : llama_get_embeddings_seq(context_, 0);
        if (embedding == nullptr) {
            embedding = llama_get_embeddings_ith(context_, static_cast<int32_t>(tokens.size()) - 1);
        }
        if (embedding == nullptr) {
            throw std::runtime_error("failed to read embeddings output for model: " + model_path_.string());
        }

        return normalize_embedding(embedding, embedding_size_);
    }

private:
    std::vector<llama_token> tokenize(const std::string & text) const {
        const llama_vocab * vocab = llama_model_get_vocab(model_);
        const int token_count = -llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0, true, true);
        if (token_count <= 0) {
            throw std::runtime_error("failed to tokenize text for embeddings");
        }

        std::vector<llama_token> tokens(static_cast<size_t>(token_count));
        if (llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(), token_count, true, true) < 0) {
            throw std::runtime_error("failed to tokenize text for embeddings");
        }

        const llama_token eos_token = llama_vocab_eos(vocab);
        if (eos_token >= 0 && (tokens.empty() || tokens.back() != eos_token)) {
            tokens.push_back(eos_token);
        }

        return tokens;
    }

    Options options_;
    std::filesystem::path model_path_;
    llama_model * model_ = nullptr;
    llama_context * context_ = nullptr;
    int embedding_size_ = 0;
};

struct IndexedChunk {
    std::string source;
    std::string text;
    std::vector<float> embedding;
};

std::string display_source_path(const std::filesystem::path & file_path) {
    return normalize_source_path(file_path).string();
}

} // namespace

struct RagState {
    RagState(const Options & options_in, const std::filesystem::path & embeddings_model_path)
        : options(options_in), engine(options_in, embeddings_model_path) {
    }

    Options options;
    EmbeddingEngine engine;
    std::vector<IndexedChunk> chunks;
};

void destroy_rag_state(RagState * rag_state) {
    delete rag_state;
}

std::filesystem::path require_embeddings_model_path(const Options & options) {
    if (options.model_embeddings.empty()) {
        throw std::runtime_error("RAG sources are set but model_embeddings is missing");
    }

    const std::filesystem::path embeddings_model_path(options.model_embeddings);
    if (!std::filesystem::exists(embeddings_model_path)) {
        throw std::runtime_error("model_embeddings must point to a local GGUF file: " + embeddings_model_path.string());
    }

    return embeddings_model_path;
}

RagStatePtr create_empty_rag_state(const Options & options) {
    return RagStatePtr(new RagState(options, require_embeddings_model_path(options)), destroy_rag_state);
}

void index_rag_sources(
    RagState & rag_state,
    const Options & options,
    const std::vector<std::filesystem::path> & source_paths,
    const RagProgressCallback & progress_callback) {
    if (source_paths.empty()) {
        throw std::runtime_error("RAG source does not exist");
    }

    std::vector<std::filesystem::path> document_files;
    std::set<std::string> document_file_keys;
    for (const auto & source_path : source_paths) {
        if (!std::filesystem::exists(source_path)) {
            throw std::runtime_error("RAG source does not exist: " + source_path.string());
        }

        debug_log(options, "indexing documents from " + display_source_path(source_path));
        for (const auto & document_file : collect_document_files(source_path)) {
            const std::filesystem::path normalized_file = normalize_source_path(document_file);
            if (document_file_keys.insert(normalized_file.string()).second) {
                document_files.push_back(normalized_file);
            }
        }
    }

    std::sort(document_files.begin(), document_files.end());
    if (document_files.empty()) {
        throw std::runtime_error("no readable RAG documents found in: " + source_paths.front().string());
    }

    debug_log(options, "learning content from files");

    size_t processed_documents = 0;
    size_t indexed_documents = 0;
    size_t indexed_chunks = 0;
    for (const auto & document_file : document_files) {
        const std::string file_contents = read_text_file(document_file);
        const std::vector<std::string> document_chunks = split_into_chunks(file_contents);
        const std::string source = display_source_path(document_file);
        debug_file_log(options, source);
        if (document_chunks.empty()) {
            ++processed_documents;
            if (progress_callback) {
                progress_callback(source, processed_documents, document_files.size());
            }
            continue;
        }

        ++indexed_documents;
        indexed_chunks += document_chunks.size();
        for (const std::string & chunk_text : document_chunks) {
            rag_state.chunks.push_back({source, chunk_text, rag_state.engine.embed(chunk_text)});
        }

        ++processed_documents;
        if (progress_callback) {
            progress_callback(source, processed_documents, document_files.size());
        }
    }

    if (indexed_chunks == 0) {
        throw std::runtime_error("no non-empty RAG document chunks found in: " + source_paths.front().string());
    }

    debug_log(
        options,
        "indexed " + std::to_string(indexed_chunks) +
            " chunks from " + std::to_string(indexed_documents) + " documents");
}

void index_rag_source(
    RagState & rag_state,
    const Options & options,
    const std::filesystem::path & source_path,
    const RagProgressCallback & progress_callback = {}) {
    index_rag_sources(rag_state, options, {source_path}, progress_callback);
}

RagStatePtr create_rag_state(const Options & options) {
    if (options.rag_documents_path.empty()) {
        return RagStatePtr(nullptr, destroy_rag_state);
    }

    RagStatePtr rag_state = create_empty_rag_state(options);
    index_rag_source(*rag_state, options, std::filesystem::path(options.rag_documents_path));
    return rag_state;
}

void learn_rag_source(
    RagStatePtr & rag_state,
    const Options & options,
    const std::string & source_path,
    const RagProgressCallback & progress_callback) {
    if (trim(source_path).empty()) {
        throw std::runtime_error("/learn requires a file or directory path");
    }

    learn_rag_sources(rag_state, options, {std::filesystem::path(source_path)}, progress_callback);
}

void learn_rag_sources(
    RagStatePtr & rag_state,
    const Options & options,
    const std::vector<std::filesystem::path> & source_paths,
    const RagProgressCallback & progress_callback) {
    if (source_paths.empty()) {
        throw std::runtime_error("/learn requires a file or directory path");
    }

    if (!rag_state) {
        rag_state = create_empty_rag_state(options);
    }

    index_rag_sources(*rag_state, options, source_paths, progress_callback);
}

bool should_use_rag_for_input(const std::string & user_input) {
    const std::string input = to_lower(trim(user_input));
    if (input.empty()) {
        return false;
    }

    if (looks_like_lookup_request(input)) {
        return true;
    }

    if (contains_text(input, "my name") || contains_text(input, "who am i") || contains_text(input, "who i am")) {
        return false;
    }

    if (contains_text(input, "what did i") || contains_text(input, "what have i") ||
        contains_text(input, "do you remember") || contains_text(input, "remember that")) {
        return false;
    }

    if (contains_text(input, "i am ") || contains_text(input, "i'm ") || contains_text(input, "call me ")) {
        return false;
    }

    return true;
}

std::string format_rag_prompt(const std::string & retrieved_context, const std::string & user_input) {
    return "Answer using the retrieved context when it contains the requested facts. "
           "The current conversation and the user's own statements are authoritative for session facts, identity, names, and preferences. "
           "For direct lookup questions, reply with the exact facts from the retrieved context. "
           "Do not refuse, speculate, or add policy commentary when the answer is present in the retrieved context. "
           "If the retrieved context does not contain the answer, say that the answer was not found in the retrieved context.\n\n"
           "Retrieved context:\n" +
           retrieved_context + "User question:\n" + user_input;
}

std::string augment_prompt_with_rag(RagState * rag_state, const std::string & user_input) {
    if (rag_state == nullptr || !should_use_rag_for_input(user_input)) {
        return user_input;
    }

    const std::vector<float> query_embedding = rag_state->engine.embed(user_input);

    struct ScoredChunk {
        size_t index = 0;
        float score = 0.0f;
    };

    std::vector<ScoredChunk> scored_chunks;
    scored_chunks.reserve(rag_state->chunks.size());
    for (size_t index = 0; index < rag_state->chunks.size(); ++index) {
        scored_chunks.push_back({index, cosine_similarity(query_embedding, rag_state->chunks[index].embedding)});
    }

    std::sort(scored_chunks.begin(), scored_chunks.end(), [](const ScoredChunk & left, const ScoredChunk & right) {
        return left.score > right.score;
    });

    const size_t retrieved_count = std::min(kTopK, scored_chunks.size());
    if (retrieved_count == 0) {
        return user_input;
    }

    std::string retrieved_context;
    for (size_t rank = 0; rank < retrieved_count; ++rank) {
        const IndexedChunk & chunk = rag_state->chunks[scored_chunks[rank].index];
        retrieved_context += chunk.text;
        retrieved_context += "\n\n";
    }

    if (rag_state->options.debug) {
        std::cerr << "[retrieved " << retrieved_count << " chunks]\n";
    }

    return format_rag_prompt(retrieved_context, user_input);
}

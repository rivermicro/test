#include "rag.hpp"

#include "llama.h"

#include <fnmatch.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
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

std::string shell_quote(std::string_view value) {
    std::string quoted = "'";
    for (const char character : value) {
        if (character == '\'') {
            quoted += "'\\''";
        } else {
            quoted += character;
        }
    }
    quoted += "'";
    return quoted;
}

std::optional<std::string> try_capture_command_output(const std::string & command) {
    std::array<char, 4096> buffer{};
    std::string output;

    FILE * pipe = popen((command + " 2>/dev/null").c_str(), "r");
    if (pipe == nullptr) {
        return std::nullopt;
    }

    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }

    const int status = pclose(pipe);
    if (status != 0) {
        return std::nullopt;
    }

    return output;
}

std::string capture_command_output(const std::string & command, const std::string & error_message) {
    const std::optional<std::string> output = try_capture_command_output(command);
    if (!output.has_value()) {
        throw std::runtime_error(error_message);
    }

    return *output;
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
           extension == ".yml" || extension == ".pdf" || extension == ".doc" || extension == ".docx";
}

bool contains_wildcard(const std::string & value) {
    return value.find('*') != std::string::npos || value.find('?') != std::string::npos;
}

std::filesystem::path wildcard_search_root(const std::filesystem::path & pattern_path) {
    std::filesystem::path root;
    for (const auto & component : pattern_path) {
        const std::string component_text = component.string();
        if (contains_wildcard(component_text)) {
            break;
        }
        root /= component;
    }

    if (root.empty()) {
        return std::filesystem::current_path();
    }

    return root;
}

bool has_matched_directory_ancestor(
    const std::filesystem::path & candidate_path,
    const std::set<std::filesystem::path> & matched_directories) {
    std::filesystem::path current = candidate_path.parent_path();
    while (!current.empty()) {
        if (matched_directories.find(current) != matched_directories.end()) {
            return true;
        }
        current = current.parent_path();
    }

    return false;
}

std::vector<std::filesystem::path> expand_recursive_wildcard_paths(const std::filesystem::path & pattern_path) {
    std::vector<std::filesystem::path> matches;
    const std::filesystem::path search_root = wildcard_search_root(pattern_path);
    std::error_code error;
    if (!std::filesystem::exists(search_root, error) || error) {
        return matches;
    }

    const std::string pattern_text = pattern_path.lexically_normal().string();
    std::set<std::filesystem::path> matched_directories;
    std::set<std::string> matched_paths;
    for (const auto & entry : std::filesystem::recursive_directory_iterator(search_root)) {
        const std::filesystem::path candidate = entry.path().lexically_normal();
        const std::string candidate_text = candidate.string();
        if (fnmatch(pattern_text.c_str(), candidate_text.c_str(), 0) != 0) {
            continue;
        }

        if (has_matched_directory_ancestor(candidate, matched_directories)) {
            continue;
        }

        if (matched_paths.insert(candidate_text).second) {
            matches.push_back(candidate);
        }

        if (entry.is_directory()) {
            matched_directories.insert(candidate);
        }
    }

    std::sort(matches.begin(), matches.end());
    return matches;
}

std::filesystem::path resolve_startup_index_path(const Options & options, const std::string & entry) {
    const std::filesystem::path configured_path(entry);
    if (configured_path.is_absolute() || options.rag_documents_path.empty()) {
        return configured_path.lexically_normal();
    }

    return (std::filesystem::path(options.rag_documents_path) / configured_path).lexically_normal();
}

std::vector<std::filesystem::path> resolve_startup_source_paths(const Options & options) {
    std::vector<std::filesystem::path> source_paths;
    std::set<std::string> source_keys;

    for (const std::string & entry : options.index_at_startup) {
        const std::filesystem::path resolved_path = resolve_startup_index_path(options, entry);
        const std::vector<std::filesystem::path> expanded_paths = contains_wildcard(resolved_path.string())
            ? expand_recursive_wildcard_paths(resolved_path)
            : std::vector<std::filesystem::path>{resolved_path};
        const std::vector<std::filesystem::path> candidate_paths = expanded_paths.empty()
            ? std::vector<std::filesystem::path>{resolved_path}
            : expanded_paths;

        for (const auto & candidate_path : candidate_paths) {
            const std::filesystem::path normalized_path = candidate_path.lexically_normal();
            if (source_keys.insert(normalized_path.string()).second) {
                source_paths.push_back(normalized_path);
            }
        }
    }

    return source_paths;
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

std::string read_file_prefix(const std::filesystem::path & path, size_t max_bytes) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open RAG document: " + path.string());
    }

    std::string prefix(max_bytes, '\0');
    input.read(prefix.data(), static_cast<std::streamsize>(max_bytes));
    prefix.resize(static_cast<size_t>(input.gcount()));
    return prefix;
}

bool starts_with_bytes(const std::string & value, std::string_view prefix) {
    return value.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), value.begin());
}

bool looks_like_rtf_document(const std::filesystem::path & path) {
    std::string prefix = read_file_prefix(path, 16);
    if (starts_with_bytes(prefix, "\xEF\xBB\xBF")) {
        prefix.erase(0, 3);
    }

    return starts_with_bytes(prefix, "{\\rtf");
}

std::string extract_pdf_text(const std::filesystem::path & path) {
    return capture_command_output(
        "pdftotext -enc UTF-8 -nopgbrk " + shell_quote(path.string()) + " -",
        "failed to extract text from PDF document: " + path.string());
}

std::string extract_docx_text(const std::filesystem::path & path) {
    return capture_command_output(
        "pandoc --from=docx --to=plain --wrap=none " + shell_quote(path.string()),
        "failed to extract text from DOCX document: " + path.string());
}

std::string extract_rtf_doc_text(const std::filesystem::path & path) {
    return capture_command_output(
        "pandoc --from=rtf --to=plain --wrap=none " + shell_quote(path.string()),
        "failed to extract text from DOC document: " + path.string());
}

std::string extract_legacy_doc_text(const std::filesystem::path & path) {
    if (const std::optional<std::string> antiword_output =
            try_capture_command_output("antiword " + shell_quote(path.string()))) {
        return *antiword_output;
    }

    if (const std::optional<std::string> catdoc_output =
            try_capture_command_output("catdoc " + shell_quote(path.string()))) {
        return *catdoc_output;
    }

    throw std::runtime_error(
        "failed to extract text from DOC document: " + path.string() +
        ". Install antiword or catdoc for legacy .doc files, or provide an RTF-backed .doc file.");
}

std::string read_document_file(const std::filesystem::path & path) {
    const std::string extension = to_lower(path.extension().string());
    if (extension == ".pdf") {
        return extract_pdf_text(path);
    }

    if (extension == ".docx") {
        return extract_docx_text(path);
    }

    if (extension == ".doc") {
        return looks_like_rtf_document(path) ? extract_rtf_doc_text(path) : extract_legacy_doc_text(path);
    }

    return read_text_file(path);
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

std::vector<std::filesystem::path> resolve_document_files(const std::vector<std::filesystem::path> & source_paths) {
    if (source_paths.empty()) {
        throw std::runtime_error("RAG source does not exist");
    }

    std::vector<std::filesystem::path> document_files;
    std::set<std::string> document_file_keys;
    for (const auto & source_path : source_paths) {
        for (const auto & document_file : collect_document_files(source_path)) {
            const std::filesystem::path normalized_file = normalize_source_path(document_file);
            if (document_file_keys.insert(normalized_file.string()).second) {
                document_files.push_back(normalized_file);
            }
        }
    }

    std::sort(document_files.begin(), document_files.end());
    return document_files;
}

std::set<std::string> resolve_source_keys(const std::vector<std::filesystem::path> & source_paths) {
    std::set<std::string> source_keys;

    for (const auto & source_path : source_paths) {
        std::error_code error;
        if (std::filesystem::exists(source_path, error) && !error) {
            if (std::filesystem::is_directory(source_path, error) && !error) {
                for (const auto & document_file : collect_document_files(source_path)) {
                    source_keys.insert(display_source_path(document_file));
                }
                continue;
            }

            source_keys.insert(display_source_path(source_path));
            continue;
        }

        source_keys.insert(display_source_path(source_path));
    }

    return source_keys;
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

namespace {

void erase_rag_sources(RagState & rag_state, const std::set<std::string> & source_keys) {
    if (source_keys.empty()) {
        return;
    }

    rag_state.chunks.erase(
        std::remove_if(
            rag_state.chunks.begin(),
            rag_state.chunks.end(),
            [&](const IndexedChunk & chunk) {
                return source_keys.find(chunk.source) != source_keys.end();
            }),
        rag_state.chunks.end());
}

} // namespace

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

    for (const auto & source_path : source_paths) {
        if (!std::filesystem::exists(source_path)) {
            throw std::runtime_error("RAG source does not exist: " + source_path.string());
        }

        debug_log(options, "indexing documents from " + display_source_path(source_path));
    }

    const std::vector<std::filesystem::path> document_files = resolve_document_files(source_paths);
    if (document_files.empty()) {
        throw std::runtime_error("no readable RAG documents found in: " + source_paths.front().string());
    }

    debug_log(options, "learning content from files");

    size_t processed_documents = 0;
    size_t indexed_documents = 0;
    size_t indexed_chunks = 0;
    for (const auto & document_file : document_files) {
        const std::string file_contents = read_document_file(document_file);
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

    const std::vector<std::filesystem::path> source_paths = resolve_startup_source_paths(options);
    if (source_paths.empty()) {
        debug_log(options, "startup indexing disabled");
        return RagStatePtr(nullptr, destroy_rag_state);
    }

    RagStatePtr rag_state = create_empty_rag_state(options);
    index_rag_sources(*rag_state, options, source_paths, {});
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

    erase_rag_sources(*rag_state, resolve_source_keys(source_paths));

    index_rag_sources(*rag_state, options, source_paths, progress_callback);
}

void replace_rag_sources(
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

    rag_state->chunks.clear();
    index_rag_sources(*rag_state, options, source_paths, progress_callback);
}

void forget_rag_sources(
    RagStatePtr & rag_state,
    const std::vector<std::filesystem::path> & source_paths) {
    if (!rag_state || source_paths.empty()) {
        return;
    }

    erase_rag_sources(*rag_state, resolve_source_keys(source_paths));
}

void clear_rag_sources(RagStatePtr & rag_state) {
    if (!rag_state) {
        return;
    }

    rag_state->chunks.clear();
}

size_t rag_chunk_count(const RagState * rag_state) {
    return rag_state == nullptr ? 0 : rag_state->chunks.size();
}

size_t rag_chunk_count_for_source(const RagState * rag_state, const std::filesystem::path & source_path) {
    if (rag_state == nullptr) {
        return 0;
    }

    const std::string normalized_source = display_source_path(source_path);
    return static_cast<size_t>(std::count_if(
        rag_state->chunks.begin(),
        rag_state->chunks.end(),
        [&](const IndexedChunk & chunk) {
            return chunk.source == normalized_source;
        }));
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

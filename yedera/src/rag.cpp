#include "rag.hpp"

#include "llama.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

class TemporaryDirectory {
public:
    explicit TemporaryDirectory(std::string_view name_template) {
        const std::filesystem::path temp_root = std::filesystem::temp_directory_path();
        std::string path_template = (temp_root / std::string(name_template)).string();
        std::vector<char> writable_path(path_template.begin(), path_template.end());
        writable_path.push_back('\0');

        char * created_path = mkdtemp(writable_path.data());
        if (created_path == nullptr) {
            throw std::runtime_error("failed to create temporary directory");
        }

        path_ = created_path;
    }

    TemporaryDirectory(const TemporaryDirectory &) = delete;
    TemporaryDirectory & operator=(const TemporaryDirectory &) = delete;

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    const std::filesystem::path & path() const {
        return path_;
    }

private:
    std::filesystem::path path_;
};

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

int ocr_page_number(const std::filesystem::path & image_path) {
    const std::string stem = image_path.stem().string();
    constexpr std::string_view prefix = "page-";
    if (stem.rfind(prefix, 0) != 0) {
        return 0;
    }

    try {
        return std::stoi(stem.substr(prefix.size()));
    } catch (const std::exception &) {
        return 0;
    }
}

std::vector<std::filesystem::path> collect_ocr_page_images(const std::filesystem::path & directory) {
    std::vector<std::filesystem::path> page_images;
    for (const auto & entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        if (to_lower(entry.path().extension().string()) == ".png") {
            page_images.push_back(entry.path());
        }
    }

    std::sort(page_images.begin(), page_images.end(), [](const auto & left, const auto & right) {
        const int left_page = ocr_page_number(left);
        const int right_page = ocr_page_number(right);
        if (left_page != right_page) {
            return left_page < right_page;
        }
        return left < right;
    });
    return page_images;
}

std::optional<std::string> try_extract_pdf_text_with_ocr(const std::filesystem::path & path) {
    TemporaryDirectory temp_dir("yedera-pdf-ocr-XXXXXX");
    const std::filesystem::path page_prefix = temp_dir.path() / "page";
    const std::string render_command =
        "pdftoppm -r 200 -png " + shell_quote(path.string()) + " " + shell_quote(page_prefix.string());
    if (!try_capture_command_output(render_command).has_value()) {
        return std::nullopt;
    }

    const std::vector<std::filesystem::path> page_images = collect_ocr_page_images(temp_dir.path());
    if (page_images.empty()) {
        return std::nullopt;
    }

    std::string text;
    for (const auto & page_image : page_images) {
        const std::optional<std::string> page_text =
            try_capture_command_output("tesseract " + shell_quote(page_image.string()) + " stdout");
        if (!page_text.has_value()) {
            return std::nullopt;
        }

        if (!text.empty() && !page_text->empty() && text.back() != '\n') {
            text += '\n';
        }
        text += *page_text;
    }

    return text;
}

std::string extract_pdf_text(const std::filesystem::path & path) {
    const std::optional<std::string> embedded_text = try_capture_command_output(
        "pdftotext -enc UTF-8 -nopgbrk " + shell_quote(path.string()) + " -");
    if (embedded_text.has_value() && !trim(*embedded_text).empty()) {
        return *embedded_text;
    }

    const std::optional<std::string> ocr_text = try_extract_pdf_text_with_ocr(path);
    if (ocr_text.has_value()) {
        return *ocr_text;
    }

    if (embedded_text.has_value()) {
        throw std::runtime_error(
            "failed to extract text from PDF document: " + path.string() +
            ". pdftotext found no embedded text, and OCR fallback failed. Install pdftoppm and tesseract for scanned PDFs.");
    }

    throw std::runtime_error(
        "failed to extract text from PDF document: " + path.string() +
        ". Install poppler-utils for pdftotext/pdftoppm and tesseract for OCR.");
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

bool quiet_model_load_progress(float, void *) {
    return true;
}

class EmbeddingEngine {
public:
    EmbeddingEngine(const Options & options, const std::filesystem::path & model_path)
        : options_(options), model_path_(model_path) {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = options_.n_gpu_layers;
        model_params.progress_callback = quiet_model_load_progress;
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
        const std::string source = display_source_path(document_file);
        if (progress_callback) {
            progress_callback(source, 0, 0);
        }

        const std::string file_contents = read_document_file(document_file);
        const std::vector<std::string> document_chunks = split_into_chunks(file_contents);
        debug_file_log(options, source);
        if (document_chunks.empty()) {
            ++processed_documents;
            if (progress_callback) {
                progress_callback(source, 1, 1);
            }
            continue;
        }

        ++indexed_documents;
        indexed_chunks += document_chunks.size();
        size_t indexed_file_chunks = 0;
        for (const std::string & chunk_text : document_chunks) {
            rag_state.chunks.push_back({source, chunk_text, rag_state.engine.embed(chunk_text)});
            ++indexed_file_chunks;
            if (progress_callback) {
                progress_callback(source, indexed_file_chunks, document_chunks.size());
            }
        }

        ++processed_documents;
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
    (void) options;
    return RagStatePtr(nullptr, destroy_rag_state);
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

std::string augment_prompt_with_rag(
    RagState * rag_state,
    const std::string & user_input,
    std::vector<std::string> & retrieved_sources) {
    retrieved_sources.clear();
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
    std::set<std::string> seen_sources;
    for (size_t rank = 0; rank < retrieved_count; ++rank) {
        const IndexedChunk & chunk = rag_state->chunks[scored_chunks[rank].index];
        retrieved_context += chunk.text;
        retrieved_context += "\n\n";
        if (seen_sources.insert(chunk.source).second) {
            retrieved_sources.push_back(chunk.source);
        }
    }

    return format_rag_prompt(retrieved_context, user_input);
}

std::string augment_prompt_with_rag(RagState * rag_state, const std::string & user_input) {
    std::vector<std::string> retrieved_sources;
    return augment_prompt_with_rag(rag_state, user_input, retrieved_sources);
}

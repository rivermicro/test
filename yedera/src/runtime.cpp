#include "runtime.hpp"

#include "debug.hpp"
#include "llama.h"
#include "rag.hpp"
#include "token_list.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fnmatch.h>
#include <iostream>
#include <memory>
#include <poll.h>
#include <set>
#include <stdexcept>
#include <string>
#include <termios.h>
#include <unistd.h>
#include <vector>

namespace {

enum class PromptMode {
    Chat,
    LearnFile,
};

const char * prompt_text(PromptMode mode) {
    return mode == PromptMode::Chat ? "> " : ": ";
}

struct InteractiveReadResult {
    enum class Kind {
        Input,
        Empty,
        Eof,
    } kind = Kind::Eof;

    std::string text;
};

class TerminalModeGuard {
public:
    explicit TerminalModeGuard(int fd)
        : fd_(fd) {
        if (tcgetattr(fd_, &original_) != 0) {
            throw std::runtime_error("failed to read terminal mode");
        }

        termios raw = original_;
        raw.c_lflag &= static_cast<unsigned long>(~(ICANON | ECHO | ISIG));
        raw.c_cc[VMIN] = 1;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(fd_, TCSANOW, &raw) != 0) {
            throw std::runtime_error("failed to enable interactive terminal mode");
        }

        active_ = true;
    }

    TerminalModeGuard(const TerminalModeGuard &) = delete;
    TerminalModeGuard & operator=(const TerminalModeGuard &) = delete;

    ~TerminalModeGuard() {
        if (active_) {
            tcsetattr(fd_, TCSANOW, &original_);
        }
    }

private:
    int fd_ = -1;
    bool active_ = false;
    termios original_{};
};

class RuntimeLogRouterGuard {
public:
    RuntimeLogRouterGuard(DebugState * restore_state, bool enabled, bool verbose)
        : restore_state_(restore_state), temporary_state_(*restore_state) {
        temporary_state_.enabled = enabled;
        temporary_state_.verbose = verbose;
        temporary_state_.load_progress_bucket = -1;
        temporary_state_.load_progress_active = false;
        install_runtime_log_router(&temporary_state_);
    }

    RuntimeLogRouterGuard(const RuntimeLogRouterGuard &) = delete;
    RuntimeLogRouterGuard & operator=(const RuntimeLogRouterGuard &) = delete;

    ~RuntimeLogRouterGuard() {
        install_runtime_log_router(restore_state_);
    }

private:
    DebugState * restore_state_ = nullptr;
    DebugState temporary_state_{};
};

void redraw_prompt(PromptMode mode, const std::string & buffer) {
    std::cout << '\r' << "\x1b[2K" << prompt_text(mode) << buffer << std::flush;
}

bool has_pending_input(int fd, int timeout_ms) {
    pollfd descriptor{};
    descriptor.fd = fd;
    descriptor.events = POLLIN;

    const int result = poll(&descriptor, 1, timeout_ms);
    return result > 0 && (descriptor.revents & POLLIN) != 0;
}

std::string read_escape_sequence(int fd) {
    std::string sequence;
    while (sequence.size() < 8 && has_pending_input(fd, sequence.empty() ? 0 : 50)) {
        char character = 0;
        const ssize_t bytes_read = read(fd, &character, 1);
        if (bytes_read <= 0) {
            break;
        }

        sequence.push_back(character);
        if (std::isalpha(static_cast<unsigned char>(character)) || character == '~') {
            break;
        }
    }

    return sequence;
}

void discard_escape_sequence(int fd) {
    while (has_pending_input(fd, 0)) {
        char discard = 0;
        const ssize_t bytes_read = read(fd, &discard, 1);
        if (bytes_read <= 0) {
            break;
        }
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

bool starts_with(const std::string & value, const std::string & prefix) {
    return value.rfind(prefix, 0) == 0;
}

std::string display_source_path(const std::string & source_path) {
    std::error_code error;
    const std::filesystem::path absolute_path = std::filesystem::absolute(source_path, error);
    if (!error) {
        return absolute_path.lexically_normal().string();
    }

    return std::filesystem::path(source_path).lexically_normal().string();
}

bool contains_text(const std::string & value, const std::string & needle) {
    return value.find(needle) != std::string::npos;
}

bool quiet_model_load_progress(float, void *) {
    return true;
}

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

bool is_session_memory_statement(const std::string & text) {
    const std::string input = to_lower(trim(text));
    return contains_text(input, "my name is ") || contains_text(input, "call me ");
}

bool contains_wildcard(const std::string & path_pattern) {
    return path_pattern.find_first_of("*?[") != std::string::npos;
}

std::filesystem::path wildcard_search_root(const std::filesystem::path & pattern_path) {
    std::filesystem::path search_root;
    for (const auto & component : pattern_path) {
        if (contains_wildcard(component.string())) {
            break;
        }

        search_root /= component;
    }

    if (!search_root.empty()) {
        return search_root.lexically_normal();
    }

    std::error_code error;
    const std::filesystem::path current_directory = std::filesystem::current_path(error);
    return error ? std::filesystem::path(".") : current_directory.lexically_normal();
}

bool has_matched_directory_ancestor(const std::filesystem::path & path, const std::set<std::string> & matched_directory_keys) {
    std::filesystem::path parent = path.parent_path();
    while (!parent.empty()) {
        if (matched_directory_keys.find(parent.lexically_normal().string()) != matched_directory_keys.end()) {
            return true;
        }

        const std::filesystem::path next_parent = parent.parent_path();
        if (next_parent == parent) {
            break;
        }
        parent = next_parent;
    }

    return false;
}

std::vector<std::filesystem::path> expand_recursive_wildcard_paths(const std::filesystem::path & pattern_path) {
    std::vector<std::filesystem::path> matches;

    std::error_code error;
    const std::filesystem::path search_root = wildcard_search_root(pattern_path);
    if (!std::filesystem::exists(search_root, error) || error) {
        return matches;
    }

    const std::string pattern = pattern_path.lexically_normal().string();
    std::filesystem::recursive_directory_iterator end;
    for (std::filesystem::recursive_directory_iterator iterator(
             search_root,
             std::filesystem::directory_options::skip_permission_denied,
             error);
         !error && iterator != end;
         iterator.increment(error)) {
        const std::filesystem::path candidate = iterator->path().lexically_normal();
        if (fnmatch(pattern.c_str(), candidate.string().c_str(), 0) == 0) {
            matches.push_back(candidate);
        }
    }

    if (error) {
        return {};
    }

    std::sort(matches.begin(), matches.end());
    matches.erase(std::unique(matches.begin(), matches.end()), matches.end());

    std::vector<std::filesystem::path> filtered_matches;
    std::set<std::string> matched_directory_keys;
    for (const auto & match : matches) {
        if (has_matched_directory_ancestor(match, matched_directory_keys)) {
            continue;
        }

        filtered_matches.push_back(match);

        std::error_code directory_error;
        if (std::filesystem::is_directory(match, directory_error) && !directory_error) {
            matched_directory_keys.insert(match.string());
        }
    }

    return filtered_matches;
}

std::filesystem::path resolve_learn_entry_path(
    const Options & options,
    const std::string & source_path,
    bool relative_to_rag_documents_path) {
    const std::filesystem::path entered_path(source_path);
    if (entered_path.is_absolute() || !relative_to_rag_documents_path || options.rag_documents_path.empty()) {
        return entered_path.lexically_normal();
    }

    return (std::filesystem::path(options.rag_documents_path) / entered_path).lexically_normal();
}

std::vector<std::filesystem::path> expand_single_learn_entry_paths(
    const Options & options,
    const std::string & source_path,
    bool relative_to_rag_documents_path) {
    const std::string trimmed_path = trim(source_path);
    if (trimmed_path.empty()) {
        return {};
    }

    const std::filesystem::path resolved_path =
        resolve_learn_entry_path(options, trimmed_path, relative_to_rag_documents_path);
    const std::string pattern = resolved_path.string();
    if (!contains_wildcard(pattern)) {
        return {resolved_path};
    }

    std::vector<std::filesystem::path> paths = expand_recursive_wildcard_paths(resolved_path);
    if (paths.empty()) {
        return {resolved_path};
    }

    return paths;
}

std::vector<std::filesystem::path> expand_learn_entry_paths(
    const Options & options,
    const std::string & source_path,
    bool relative_to_rag_documents_path) {
    const std::vector<std::string> entries = parse_token_list(trim(source_path), "learn path");
    if (entries.empty()) {
        return {};
    }

    std::vector<std::filesystem::path> paths;
    std::set<std::string> seen_paths;
    for (const std::string & entry : entries) {
        for (const auto & expanded_path : expand_single_learn_entry_paths(options, entry, relative_to_rag_documents_path)) {
            const std::filesystem::path normalized_path = expanded_path.lexically_normal();
            if (seen_paths.insert(normalized_path.string()).second) {
                paths.push_back(normalized_path);
            }
        }
    }

    return paths;
}

class EscLearnProgressRenderer {
public:
    void on_progress(const std::string & source_path, size_t completed, size_t total) {
        const std::string display_path = display_source_path(source_path);
        if (display_path != current_source_path_) {
            current_source_path_ = display_path;
            active_ = false;
            std::cout << "+ " << current_source_path_ << std::flush;
        }

        if (total == 0) {
            active_ = true;
            return;
        }

        const size_t percent = std::min<size_t>(100, (completed * 100) / total);
        if (percent >= 100) {
            std::cout << '\r' << "\x1b[2K" << "+ " << current_source_path_ << '\n' << std::flush;
            active_ = false;
            return;
        }

        std::cout << '\r' << "\x1b[2K" << "+ " << current_source_path_ << ' ' << percent << '%' << std::flush;
        active_ = true;
    }

    void finish() {
        if (active_) {
            std::cout << '\r' << "\x1b[2K" << "+ " << current_source_path_ << '\n' << std::flush;
            active_ = false;
        }
    }

private:
    std::string current_source_path_;
    bool active_ = false;
};

InteractiveReadResult read_interactive_input(PromptMode & mode, const std::vector<std::string> & chat_history) {
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
        std::cout << prompt_text(mode) << std::flush;
        std::string line;
        if (!std::getline(std::cin, line)) {
            return {InteractiveReadResult::Kind::Eof, ""};
        }

        if (line.empty()) {
            return {InteractiveReadResult::Kind::Empty, ""};
        }

        return {InteractiveReadResult::Kind::Input, std::move(line)};
    }

    TerminalModeGuard terminal_mode(STDIN_FILENO);
    std::string buffer;
    std::string draft_buffer;
    size_t history_index = chat_history.size();
    std::cout << prompt_text(mode) << std::flush;

    while (true) {
        char character = 0;
        const ssize_t bytes_read = read(STDIN_FILENO, &character, 1);
        if (bytes_read == 0) {
            std::cout << '\n' << std::flush;
            return buffer.empty()
                ? InteractiveReadResult{InteractiveReadResult::Kind::Eof, ""}
                : InteractiveReadResult{InteractiveReadResult::Kind::Input, std::move(buffer)};
        }

        if (bytes_read < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error("failed to read interactive input");
        }

        const unsigned char byte = static_cast<unsigned char>(character);
        if (character == '\r' || character == '\n') {
            std::cout << '\n' << std::flush;
            if (buffer.empty()) {
                return {InteractiveReadResult::Kind::Empty, ""};
            }
            return {InteractiveReadResult::Kind::Input, std::move(buffer)};
        }

        if (byte == 3) {
            std::cout << '\n' << std::flush;
            return {InteractiveReadResult::Kind::Eof, ""};
        }

        if (byte == 4) {
            continue;
        }

        if (byte == 27) {
            if (has_pending_input(STDIN_FILENO, 50)) {
                const std::string sequence = read_escape_sequence(STDIN_FILENO);
                if (mode == PromptMode::Chat && sequence == "[A" && !chat_history.empty()) {
                    if (history_index == chat_history.size()) {
                        draft_buffer = buffer;
                    }
                    if (history_index > 0) {
                        --history_index;
                        buffer = chat_history[history_index];
                        redraw_prompt(mode, buffer);
                    }
                    continue;
                }

                if (mode == PromptMode::Chat && sequence == "[B") {
                    if (history_index < chat_history.size()) {
                        ++history_index;
                        buffer = history_index == chat_history.size() ? draft_buffer : chat_history[history_index];
                        redraw_prompt(mode, buffer);
                    }
                    continue;
                }

                discard_escape_sequence(STDIN_FILENO);
                continue;
            }

            if (buffer.empty()) {
                mode = mode == PromptMode::Chat ? PromptMode::LearnFile : PromptMode::Chat;
                redraw_prompt(mode, buffer);
            }
            continue;
        }

        if (byte == 127 || byte == 8) {
            if (!buffer.empty()) {
                buffer.pop_back();
                redraw_prompt(mode, buffer);
            }
            continue;
        }

        if (std::isprint(byte)) {
            buffer.push_back(static_cast<char>(byte));
            std::cout << static_cast<char>(byte) << std::flush;
        }
    }
}

struct OwnedMessage {
    std::string role;
    std::string content;
};

std::vector<llama_chat_message> to_llama_messages(const std::vector<OwnedMessage> & messages) {
    std::vector<llama_chat_message> raw_messages;
    raw_messages.reserve(messages.size());

    for (const auto & message : messages) {
        raw_messages.push_back({message.role.c_str(), message.content.c_str()});
    }

    return raw_messages;
}

bool template_uses_channel_messages(llama_model * model) {
    const char * template_text = llama_model_chat_template(model, nullptr);
    if (template_text == nullptr) {
        return false;
    }

    const std::string_view view(template_text);
    return view.find("<|channel|>") != std::string_view::npos &&
        view.find("<|message|>") != std::string_view::npos;
}

size_t find_channel_message_end(const std::string & response, size_t start_offset) {
    size_t end_offset = response.find("<|return|>", start_offset);

    const size_t call_offset = response.find("<|call|>", start_offset);
    if (call_offset != std::string::npos && (end_offset == std::string::npos || call_offset < end_offset)) {
        end_offset = call_offset;
    }

    const size_t message_end_offset = response.find("<|end|>", start_offset);
    if (message_end_offset != std::string::npos && (end_offset == std::string::npos || message_end_offset < end_offset)) {
        end_offset = message_end_offset;
    }

    const size_t next_message_offset = response.find("<|start|>", start_offset);
    if (next_message_offset != std::string::npos && (end_offset == std::string::npos || next_message_offset < end_offset)) {
        end_offset = next_message_offset;
    }

    return end_offset == std::string::npos ? response.size() : end_offset;
}

bool channel_message_invokes_function(const std::string & response, size_t channel_offset, size_t message_offset) {
    const size_t function_offset = response.find("to=functions.", channel_offset);
    return function_offset != std::string::npos && function_offset < message_offset;
}

std::string extract_channel_message(const std::string & response, const std::string & channel_name) {
    const std::string marker = std::string("<|channel|>") + channel_name;
    const size_t channel_offset = response.rfind(marker);
    if (channel_offset == std::string::npos) {
        return "";
    }

    const size_t message_offset = response.find("<|message|>", channel_offset);
    if (message_offset == std::string::npos) {
        return "";
    }

    if (channel_name == "commentary" && channel_message_invokes_function(response, channel_offset, message_offset)) {
        return "";
    }

    const size_t content_offset = message_offset + std::string("<|message|>").size();
    const size_t end_offset = find_channel_message_end(response, content_offset);
    return response.substr(content_offset, end_offset - content_offset);
}

std::string extract_channel_name_at_offset(const std::string & response, size_t channel_offset) {
    const size_t name_offset = channel_offset + std::string("<|channel|>").size();
    const size_t message_offset = response.find("<|message|>", name_offset);
    if (message_offset == std::string::npos) {
        return "";
    }

    std::string channel_name = response.substr(name_offset, message_offset - name_offset);
    const size_t suffix_offset = channel_name.find_first_of(" \t\r\n");
    if (suffix_offset != std::string::npos) {
        channel_name.resize(suffix_offset);
    }

    return channel_name;
}

std::string extract_last_channel_message(const std::string & response) {
    const std::string marker = "<|channel|>";
    const size_t channel_offset = response.rfind(marker);
    if (channel_offset == std::string::npos) {
        return "";
    }

    const size_t message_offset = response.find("<|message|>", channel_offset + marker.size());
    if (message_offset == std::string::npos) {
        return "";
    }

    const size_t content_offset = message_offset + std::string("<|message|>").size();
    const size_t end_offset = find_channel_message_end(response, content_offset);
    return response.substr(content_offset, end_offset - content_offset);
}

std::string extract_streamable_channel_message(const std::string & response) {
    const std::string final_message = extract_channel_message(response, "final");
    if (!final_message.empty() || response.find("<|channel|>final") != std::string::npos) {
        return final_message;
    }

    const std::string marker = "<|channel|>";
    const size_t channel_offset = response.rfind(marker);
    if (channel_offset == std::string::npos) {
        return "";
    }

    const std::string channel_name = extract_channel_name_at_offset(response, channel_offset);
    if (channel_name.empty() || channel_name == "analysis") {
        return "";
    }

    const size_t message_offset = response.find("<|message|>", channel_offset + marker.size());
    if (message_offset == std::string::npos) {
        return "";
    }

    if (channel_name == "commentary" && channel_message_invokes_function(response, channel_offset, message_offset)) {
        return "";
    }

    const size_t content_offset = message_offset + std::string("<|message|>").size();
    const size_t end_offset = find_channel_message_end(response, content_offset);
    return response.substr(content_offset, end_offset - content_offset);
}

std::string format_channel_response_for_display(const std::string & response) {
    if (response.find("<|channel|>") == std::string::npos) {
        return response;
    }

    const std::string final_message = extract_channel_message(response, "final");
    if (!final_message.empty() || response.find("<|channel|>final") != std::string::npos) {
        return final_message;
    }

    const std::string commentary_message = extract_channel_message(response, "commentary");
    if (!commentary_message.empty() || response.find("<|channel|>commentary") != std::string::npos) {
        return commentary_message;
    }

    return extract_last_channel_message(response);
}

bool response_has_channel_terminator(const std::string & response) {
    return response.find("<|return|>") != std::string::npos ||
        response.find("<|call|>") != std::string::npos;
}

std::string format_messages(llama_model * model, std::vector<OwnedMessage> & messages, std::vector<char> & buffer, bool add_assistant) {
    const char * template_text = llama_model_chat_template(model, nullptr);
    if (template_text == nullptr) {
        throw std::runtime_error("model does not expose a chat template");
    }

    auto raw_messages = to_llama_messages(messages);
    int rendered = llama_chat_apply_template(template_text, raw_messages.data(), raw_messages.size(), add_assistant, buffer.data(), buffer.size());
    if (rendered > static_cast<int>(buffer.size())) {
        buffer.resize(rendered);
        raw_messages = to_llama_messages(messages);
        rendered = llama_chat_apply_template(template_text, raw_messages.data(), raw_messages.size(), add_assistant, buffer.data(), buffer.size());
    }

    if (rendered < 0) {
        throw std::runtime_error("failed to apply chat template");
    }

    return std::string(buffer.data(), rendered);
}

std::vector<OwnedMessage> build_prompt_history(const std::vector<OwnedMessage> & messages, const std::string & user_input) {
    if (should_use_rag_for_input(user_input)) {
        return messages;
    }

    std::vector<OwnedMessage> prompt_history;
    for (const auto & message : messages) {
        if (message.role == "system") {
            prompt_history.push_back(message);
            continue;
        }

        if (message.role == "user" && is_session_memory_statement(message.content)) {
            prompt_history.push_back(message);
        }
    }

    return prompt_history;
}

} // namespace

void run_inference(const Options & options) {
    if (!std::filesystem::exists(options.model_path)) {
        throw std::runtime_error("model path does not exist: " + options.model_path);
    }

    DebugState debug_state;
    debug_state.enabled = options.debug;
    debug_state.verbose = options.verbose;
    install_runtime_log_router(&debug_state);

    print_backend_debug_summary(options);

    RagStatePtr rag_state = create_rag_state(options);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = options.n_gpu_layers;
    if (options.debug) {
        model_params.progress_callback = handle_model_load_progress;
        model_params.progress_callback_user_data = &debug_state;
    } else {
        model_params.progress_callback = quiet_model_load_progress;
    }

    llama_model * model = llama_model_load_from_file(options.model_path.c_str(), model_params);
    if (model == nullptr) {
        throw std::runtime_error("failed to load model: " + options.model_path);
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = options.n_ctx;
    context_params.n_batch = std::min(options.n_ctx, 512);

    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        llama_model_free(model);
        throw std::runtime_error("failed to create llama context");
    }

    auto sampler_params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    if (options.top_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(options.top_p, 1));
    }
    if (options.min_p > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_min_p(options.min_p, 1));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(options.temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(options.seed));

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<char> formatted_buffer(static_cast<size_t>(std::max(4096, options.n_ctx * 4)));
    const bool model_uses_structured_channels = template_uses_channel_messages(model);

    auto generate = [&](const std::string & prompt) {
        llama_memory_clear(llama_get_memory(context), true);
        llama_sampler_reset(sampler);
        llama_memory_t memory = llama_get_memory(context);

        const int token_count = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
        if (token_count <= 0) {
            throw std::runtime_error("failed to tokenize prompt");
        }

        std::vector<llama_token> prompt_tokens(static_cast<size_t>(token_count));
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
            throw std::runtime_error("failed to tokenize prompt");
        }

        const int ctx_size = llama_n_ctx(context);
        if (static_cast<int>(prompt_tokens.size()) >= ctx_size) {
            throw std::runtime_error("context size exceeded; increase --ctx-size or shorten the conversation");
        }

        const int prompt_tokens_to_keep = static_cast<int>(prompt_tokens.size());

        auto ensure_decode_capacity = [&](int incoming_tokens) {
            while (true) {
                const int ctx_used = llama_memory_seq_pos_max(memory, 0) + 1;
                if (ctx_used + incoming_tokens <= ctx_size) {
                    return;
                }

                if (!llama_memory_can_shift(memory)) {
                    break;
                }

                const int discardable_tokens = ctx_used - prompt_tokens_to_keep;
                if (discardable_tokens <= 0) {
                    break;
                }

                const int tokens_needed = ctx_used + incoming_tokens - ctx_size;
                const int tokens_to_discard = std::min(
                    discardable_tokens,
                    std::max(tokens_needed, std::max(1, discardable_tokens / 2)));

                if (!llama_memory_seq_rm(memory, 0, prompt_tokens_to_keep, prompt_tokens_to_keep + tokens_to_discard)) {
                    break;
                }

                llama_memory_seq_add(memory, 0, prompt_tokens_to_keep + tokens_to_discard, ctx_used, -tokens_to_discard);
            }

            throw std::runtime_error("context size exceeded; increase --ctx-size or shorten the conversation");
        };

        const size_t max_batch_tokens = std::max<size_t>(1, llama_n_batch(context));
        auto decode_tokens = [&](llama_token * tokens, size_t token_size) {
            size_t offset = 0;
            while (offset < token_size) {
                const size_t chunk_size = std::min(max_batch_tokens, token_size - offset);
                ensure_decode_capacity(static_cast<int>(chunk_size));
                llama_batch batch = llama_batch_get_one(tokens + offset, static_cast<int32_t>(chunk_size));

                const int decode_result = llama_decode(context, batch);
                if (decode_result != 0) {
                    throw std::runtime_error("llama_decode failed with code " + std::to_string(decode_result));
                }

                offset += chunk_size;
            }
        };

        decode_tokens(prompt_tokens.data(), prompt_tokens.size());

        std::string response;
        std::string streamed_response;

        int generated = 0;
        int initial_eog_retries = 0;
        const bool unlimited_generation = options.n_predict < 0;
        while (unlimited_generation || generated < options.n_predict) {
            llama_token token = llama_sampler_sample(sampler, context, -1);
            if (llama_vocab_is_eog(vocab, token)) {
                if (generated == 0 && initial_eog_retries < 8) {
                    ++initial_eog_retries;
                    continue;
                }
                break;
            }

            initial_eog_retries = 0;

            char piece_buffer[256];
            const int piece_length = llama_token_to_piece(vocab, token, piece_buffer, sizeof(piece_buffer), 0, true);
            if (piece_length < 0) {
                throw std::runtime_error("failed to decode token piece");
            }

            const std::string piece(piece_buffer, static_cast<size_t>(piece_length));
            response += piece;
            if (!model_uses_structured_channels) {
                std::cout << piece << std::flush;
            } else {
                const std::string current_streamable_response = extract_streamable_channel_message(response);
                if (current_streamable_response.size() >= streamed_response.size() &&
                    current_streamable_response.compare(0, streamed_response.size(), streamed_response) == 0) {
                    const std::string delta = current_streamable_response.substr(streamed_response.size());
                    if (!delta.empty()) {
                        std::cout << delta << std::flush;
                        streamed_response = current_streamable_response;
                    }
                }
            }
            ++generated;

            if (model_uses_structured_channels && response_has_channel_terminator(response)) {
                break;
            }

            if (unlimited_generation || generated < options.n_predict) {
                decode_tokens(&token, 1);
            }
        }

        if (model_uses_structured_channels) {
            const std::string display_response = format_channel_response_for_display(response);
            if (display_response.size() >= streamed_response.size() &&
                display_response.compare(0, streamed_response.size(), streamed_response) == 0) {
                const std::string tail = display_response.substr(streamed_response.size());
                if (!tail.empty()) {
                    std::cout << tail << std::flush;
                }
            } else if (streamed_response.empty() && !display_response.empty()) {
                std::cout << display_response << std::flush;
            }
            return display_response;
        }

        return response;
    };

    std::vector<OwnedMessage> messages;
    if (!options.system_prompt.empty()) {
        messages.push_back({"system", options.system_prompt});
    }

    auto run_turn = [&](const std::string & user_input) {
        const bool use_rag = should_use_rag_for_input(user_input);
        std::vector<OwnedMessage> prompt_messages = build_prompt_history(messages, user_input);
        std::vector<std::string> retrieved_sources;
        prompt_messages.push_back({"user", augment_prompt_with_rag(rag_state.get(), user_input, retrieved_sources)});
        const std::string rendered = format_messages(model, prompt_messages, formatted_buffer, true);
        const std::string response = generate(rendered);
        std::cout << '\n';
        for (const std::string & source : retrieved_sources) {
            std::cout << "→ " << source << '\n';
        }
        messages.push_back({"user", user_input});
        if (use_rag) {
            messages.push_back({"assistant", response});
        }
    };

    if (!options.prompt.empty()) {
        run_turn(options.prompt);
    }

    if (options.interactive) {
        PromptMode prompt_mode = PromptMode::Chat;
        std::vector<std::string> chat_history;
        while (true) {
            const InteractiveReadResult input = read_interactive_input(prompt_mode, chat_history);
            if (input.kind == InteractiveReadResult::Kind::Eof) {
                break;
            }

            if (input.kind == InteractiveReadResult::Kind::Empty) {
                continue;
            }

            const std::string & user_input = input.text;

            if (prompt_mode == PromptMode::LearnFile) {
                try {
                    Options quiet_learn_options = options;
                    quiet_learn_options.debug = false;

                    const std::string trimmed_input = trim(user_input);
                    if (trimmed_input == "-") {
                        clear_rag_sources(rag_state);
                        std::cout << "[rag] forgot all\n";
                        prompt_mode = PromptMode::Chat;
                        continue;
                    }

                    if (starts_with(trimmed_input, "-")) {
                        const std::string forget_path = trim(trimmed_input.substr(1));
                        const std::vector<std::filesystem::path> source_paths =
                            expand_learn_entry_paths(options, forget_path, true);
                        forget_rag_sources(rag_state, source_paths);
                        for (const auto & source_path : source_paths) {
                            std::cout << "[rag] forgot " << display_source_path(source_path.string()) << '\n';
                        }
                        prompt_mode = PromptMode::Chat;
                        continue;
                    }

                    const std::vector<std::filesystem::path> source_paths =
                        expand_learn_entry_paths(options, user_input, true);
                    EscLearnProgressRenderer progress_renderer;
                    RuntimeLogRouterGuard quiet_log_router(&debug_state, false, false);
                    if (trimmed_input == "*") {
                        replace_rag_sources(
                            rag_state,
                            quiet_learn_options,
                            source_paths,
                            [&](const std::string & source_path, size_t completed, size_t total) {
                                progress_renderer.on_progress(source_path, completed, total);
                            });
                    } else {
                        learn_rag_sources(
                            rag_state,
                            quiet_learn_options,
                            source_paths,
                            [&](const std::string & source_path, size_t completed, size_t total) {
                                progress_renderer.on_progress(source_path, completed, total);
                            });
                    }
                    progress_renderer.finish();
                    prompt_mode = PromptMode::Chat;
                } catch (const std::exception & error) {
                    std::cerr << "error: " << error.what() << '\n';
                }
                continue;
            }

            chat_history.push_back(user_input);

            const std::string trimmed_input = trim(user_input);
            if (starts_with(trimmed_input, "//")) {
                run_turn(trimmed_input.substr(1));
                continue;
            }

            if (starts_with(trimmed_input, "/learn")) {
                const std::string source_path = trim(trimmed_input.substr(std::string("/learn").size()));
                try {
                    const std::vector<std::filesystem::path> source_paths =
                        expand_learn_entry_paths(options, source_path, false);
                    learn_rag_sources(rag_state, options, source_paths, {});
                    for (const auto & learned_path : source_paths) {
                        std::cout << "[rag] learned " << display_source_path(learned_path.string()) << '\n';
                    }
                } catch (const std::exception & error) {
                    std::cerr << "error: " << error.what() << '\n';
                }
                continue;
            }

            run_turn(user_input);
        }
    }

    llama_sampler_free(sampler);
    llama_free(context);
    llama_model_free(model);
}

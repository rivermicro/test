#include "runtime.hpp"

#include "debug.hpp"
#include "llama.h"
#include "rag.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <memory>
#include <poll.h>
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

InteractiveReadResult read_interactive_input(PromptMode & mode) {
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

        if (byte == 3 || byte == 4) {
            std::cout << '\n' << std::flush;
            return {InteractiveReadResult::Kind::Eof, ""};
        }

        if (byte == 27) {
            if (has_pending_input(STDIN_FILENO, 10)) {
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

    auto generate = [&](const std::string & prompt) {
        const bool is_first = llama_memory_seq_pos_max(llama_get_memory(context), 0) == -1;
        const int token_count = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, is_first, true);
        if (token_count <= 0) {
            throw std::runtime_error("failed to tokenize prompt");
        }

        std::vector<llama_token> prompt_tokens(static_cast<size_t>(token_count));
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
            throw std::runtime_error("failed to tokenize prompt");
        }

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        std::string response;

        int generated = 0;
        while (generated < options.n_predict) {
            const int ctx_size = llama_n_ctx(context);
            const int ctx_used = llama_memory_seq_pos_max(llama_get_memory(context), 0) + 1;
            if (ctx_used + batch.n_tokens > ctx_size) {
                throw std::runtime_error("context size exceeded; increase --ctx-size or shorten the conversation");
            }

            const int decode_result = llama_decode(context, batch);
            if (decode_result != 0) {
                throw std::runtime_error("llama_decode failed with code " + std::to_string(decode_result));
            }

            llama_token token = llama_sampler_sample(sampler, context, -1);
            if (llama_vocab_is_eog(vocab, token)) {
                break;
            }

            char piece_buffer[256];
            const int piece_length = llama_token_to_piece(vocab, token, piece_buffer, sizeof(piece_buffer), 0, true);
            if (piece_length < 0) {
                throw std::runtime_error("failed to decode token piece");
            }

            const std::string piece(piece_buffer, static_cast<size_t>(piece_length));
            std::cout << piece << std::flush;
            response += piece;
            batch = llama_batch_get_one(&token, 1);
            ++generated;
        }

        return response;
    };

    std::vector<OwnedMessage> messages;
    if (!options.system_prompt.empty()) {
        messages.push_back({"system", options.system_prompt});
    }

    int previous_length = 0;

    auto run_turn = [&](const std::string & user_input) {
        messages.push_back({"user", augment_prompt_with_rag(rag_state.get(), user_input)});
        const std::string rendered = format_messages(model, messages, formatted_buffer, true);
        const std::string incremental_prompt = rendered.substr(static_cast<size_t>(previous_length));
        const std::string response = generate(incremental_prompt);
        std::cout << '\n';
        messages.push_back({"assistant", response});
        previous_length = static_cast<int>(format_messages(model, messages, formatted_buffer, false).size());
    };

    if (!options.prompt.empty()) {
        run_turn(options.prompt);
    }

    if (options.interactive) {
        PromptMode prompt_mode = PromptMode::Chat;
        while (true) {
            const InteractiveReadResult input = read_interactive_input(prompt_mode);
            if (input.kind == InteractiveReadResult::Kind::Eof) {
                break;
            }

            if (input.kind == InteractiveReadResult::Kind::Empty) {
                if (prompt_mode == PromptMode::Chat) {
                    break;
                }
                continue;
            }

            const std::string & user_input = input.text;

            if (prompt_mode == PromptMode::LearnFile) {
                try {
                    learn_rag_source(rag_state, options, user_input);
                    std::cout << "[rag] learned " << display_source_path(user_input) << '\n';
                } catch (const std::exception & error) {
                    std::cerr << "error: " << error.what() << '\n';
                }
                continue;
            }

            const std::string trimmed_input = trim(user_input);
            if (starts_with(trimmed_input, "//")) {
                run_turn(trimmed_input.substr(1));
                continue;
            }

            if (starts_with(trimmed_input, "/learn")) {
                const std::string source_path = trim(trimmed_input.substr(std::string("/learn").size()));
                try {
                    learn_rag_source(rag_state, options, source_path);
                    std::cout << "[rag] learned " << display_source_path(source_path) << '\n';
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
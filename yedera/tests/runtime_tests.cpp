#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#if defined(__linux__)
#include <pty.h>
#include <sys/wait.h>
#endif

#include <unistd.h>

#ifndef YEDERA_BINARY_PATH
#error "YEDERA_BINARY_PATH must be defined"
#endif

namespace {

struct TempDirectory {
    TempDirectory() {
        char path_template[] = "/tmp/yedera-runtime-tests-XXXXXX";
        char * created = mkdtemp(path_template);
        if (created == nullptr) {
            throw std::runtime_error("failed to create temp directory");
        }
        path = created;
    }

    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }

    std::filesystem::path path;
};

struct TestFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void expect_true(bool condition, const std::string & message) {
    if (!condition) {
        throw TestFailure(message);
    }
}

void expect_contains(std::string_view haystack, std::string_view needle, const std::string & message) {
    if (haystack.find(needle) == std::string_view::npos) {
        throw TestFailure(message + ": missing '" + std::string(needle) + "'");
    }
}

void expect_not_contains(std::string_view haystack, std::string_view needle, const std::string & message) {
    if (haystack.find(needle) != std::string_view::npos) {
        throw TestFailure(message + ": unexpected '" + std::string(needle) + "'");
    }
}

void write_file(const std::filesystem::path & path, const std::string & content) {
    std::ofstream output(path);
    if (!output.is_open()) {
        throw std::runtime_error("failed to write test file: " + path.string());
    }
    output << content;
}

std::string shell_quote(std::string_view value) {
    std::string quoted = "'";
    for (const char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

std::string run_command(const std::string & command) {
    std::array<char, 4096> buffer{};
    std::string output;

    FILE * pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        throw std::runtime_error("failed to run command: " + command);
    }

    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }

    const int status = pclose(pipe);
    if (status != 0) {
        throw TestFailure(
            "command failed with status " + std::to_string(status) + ": " + command + "\nOutput:\n" + output);
    }

    return output;
}

#if defined(__linux__)
struct PtyStep {
    std::string input;
    std::string wait_for;
    int timeout_ms = 60000;
};

std::string read_pty_until(
    int master_fd,
    std::string & output,
    std::string_view needle,
    int timeout_ms,
    size_t search_start = 0) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    std::array<char, 4096> buffer{};

    while (needle.empty() || output.find(needle, search_start) == std::string::npos) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            throw TestFailure("timed out waiting for PTY output: " + std::string(needle) + "\nOutput:\n" + output);
        }

        const int remaining_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
        pollfd descriptor{};
        descriptor.fd = master_fd;
        descriptor.events = POLLIN;
        const int poll_result = poll(&descriptor, 1, remaining_ms);
        if (poll_result < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error("failed to poll pseudo-terminal output");
        }

        if (poll_result == 0) {
            continue;
        }

        const ssize_t count = read(master_fd, buffer.data(), buffer.size());
        if (count == 0) {
            break;
        }
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EIO) {
                break;
            }
            throw std::runtime_error("failed to read pseudo-terminal output");
        }

        output.append(buffer.data(), static_cast<size_t>(count));
    }

    return output;
}

std::string run_command_in_pty(const std::vector<std::string> & argv_values, const std::vector<PtyStep> & steps) {
    int master_fd = -1;
    const pid_t child_pid = forkpty(&master_fd, nullptr, nullptr, nullptr);
    if (child_pid < 0) {
        throw std::runtime_error("failed to create pseudo-terminal");
    }

    if (child_pid == 0) {
        std::vector<char *> argv;
        argv.reserve(argv_values.size() + 1);
        for (const std::string & value : argv_values) {
            argv.push_back(const_cast<char *>(value.c_str()));
        }
        argv.push_back(nullptr);
        execv(argv_values.front().c_str(), argv.data());
        _exit(127);
    }

    std::string output;
    for (const PtyStep & step : steps) {
        const size_t wait_start = output.size();
        size_t written = 0;
        while (written < step.input.size()) {
            const ssize_t count = write(master_fd, step.input.data() + written, 1);
            if (count < 0) {
                if (errno == EINTR) {
                    continue;
                }
                close(master_fd);
                throw std::runtime_error("failed to write pseudo-terminal input");
            }
            written += static_cast<size_t>(count);

            const unsigned char byte = static_cast<unsigned char>(step.input[written - 1]);
            if (byte == 27) {
                usleep(25000);
            } else if (byte == '\n') {
                usleep(5000);
            }
        }

        if (!step.wait_for.empty()) {
            read_pty_until(master_fd, output, step.wait_for, step.timeout_ms, wait_start);
        }
    }

    read_pty_until(master_fd, output, "", 1000);

    close(master_fd);

    int status = 0;
    if (waitpid(child_pid, &status, 0) < 0) {
        throw std::runtime_error("failed to wait for pseudo-terminal child");
    }

    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        throw TestFailure("pseudo-terminal command failed\nOutput:\n" + output);
    }

    return output;
}
#endif

struct CommandResult {
    int status = 0;
    std::string output;
};

CommandResult run_command_capture(const std::string & command) {
    std::array<char, 4096> buffer{};
    std::string output;

    FILE * pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        throw std::runtime_error("failed to run command: " + command);
    }

    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }

    const int status = pclose(pipe);
    return {status, std::move(output)};
}

std::string normalize_terminal_output(std::string_view raw_output) {
    std::string normalized;
    std::string current_line;
    size_t cursor = 0;

    auto flush_line = [&]() {
        normalized += current_line;
        normalized += '\n';
        current_line.clear();
        cursor = 0;
    };

    for (const char ch : raw_output) {
        if (ch == '\r') {
            cursor = 0;
            continue;
        }

        if (ch == '\n') {
            flush_line();
            continue;
        }

        if (cursor >= current_line.size()) {
            current_line.push_back(ch);
        } else {
            current_line[cursor] = ch;
        }
        ++cursor;
    }

    if (!current_line.empty()) {
        normalized += current_line;
    }

    return normalized;
}

std::filesystem::path binary_path() {
    return std::filesystem::path(YEDERA_BINARY_PATH);
}

void test_list_command_uses_executable_relative_model_dir() {
    const TempDirectory temp_dir;
    const std::filesystem::path temp_binary = temp_dir.path / "yedera";

    std::filesystem::copy_file(binary_path(), temp_binary, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::permissions(
        temp_binary,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec |
            std::filesystem::perms::group_read | std::filesystem::perms::group_exec |
            std::filesystem::perms::others_read | std::filesystem::perms::others_exec,
        std::filesystem::perm_options::replace);

    std::filesystem::create_directories(temp_dir.path / "model" / "nested");
    write_file(temp_dir.path / "model" / "alpha.gguf", "");
    write_file(temp_dir.path / "model" / "nested" / "beta.gguf", "");

    const std::string output = run_command(shell_quote(temp_binary.string()) + " list 2>&1");
    const std::string expected =
        (temp_dir.path / "model" / "alpha.gguf").lexically_normal().string() + "\n" +
        (temp_dir.path / "model" / "nested" / "beta.gguf").lexically_normal().string() + "\n";
    expect_true(output == expected, "list output should show absolute model paths");
}

void test_config_is_discovered_from_current_directory() {
    const std::filesystem::path source_model_path = binary_path().parent_path() / "model" / "llama3.2-1b.gguf";
    expect_true(std::filesystem::exists(source_model_path), "runtime smoke test model is missing");

    const TempDirectory temp_dir;
    const std::filesystem::path temp_binary = temp_dir.path / "yedera";
    const std::filesystem::path temp_config = temp_dir.path / "yedera.conf";

    std::filesystem::copy_file(binary_path(), temp_binary, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::permissions(
        temp_binary,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec |
            std::filesystem::perms::group_read | std::filesystem::perms::group_exec |
            std::filesystem::perms::others_read | std::filesystem::perms::others_exec,
        std::filesystem::perm_options::replace);

    write_file(
        temp_config,
        "prompt = \"You are a test assistant.\"\n"
        "model_path = \"" + source_model_path.string() + "\"\n"
        "n_gpu_layers = 0\n"
        "debug = true\n"
        "verbose = false\n");

    const std::string raw_output = run_command(
        "cd " + shell_quote(temp_dir.path.string()) + " && " + shell_quote(temp_binary.string()) +
        " --prompt 'Reply with one word.' --n-predict 1 --temperature 0 --top-p 1 --min-p 0 --seed 1 2>&1");
    const std::string output = normalize_terminal_output(raw_output);

    expect_contains(output, "[debug] using model ", "current-dir config should be discovered automatically");
    expect_contains(output, source_model_path.string(), "current-dir config should provide the configured model path");
    expect_contains(output, "[debug] RAG mode disabled", "current-dir config without embeddings model should report RAG disabled");
    expect_contains(output, "[debug] offload request: CPU-only", "current-dir config should provide the configured offload mode");
}

void test_config_falls_back_to_home_directory() {
    const std::filesystem::path source_model_path = binary_path().parent_path() / "model" / "llama3.2-1b.gguf";
    expect_true(std::filesystem::exists(source_model_path), "runtime smoke test model is missing");

    const TempDirectory temp_dir;
    const std::filesystem::path temp_binary = temp_dir.path / "yedera";
    const std::filesystem::path home_directory = temp_dir.path / "home";
    const std::filesystem::path home_config = home_directory / ".yedera" / "yedera.conf";

    std::filesystem::copy_file(binary_path(), temp_binary, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::permissions(
        temp_binary,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec |
            std::filesystem::perms::group_read | std::filesystem::perms::group_exec |
            std::filesystem::perms::others_read | std::filesystem::perms::others_exec,
        std::filesystem::perm_options::replace);

    std::filesystem::create_directories(home_config.parent_path());
    write_file(
        home_config,
        "prompt = \"You are a test assistant.\"\n"
        "model_path = \"" + source_model_path.string() + "\"\n"
        "n_gpu_layers = 0\n"
        "debug = true\n"
        "verbose = false\n");

    const std::string raw_output = run_command(
        "cd " + shell_quote(temp_dir.path.string()) + " && HOME=" + shell_quote(home_directory.string()) + " " + shell_quote(temp_binary.string()) +
        " --prompt 'Reply with one word.' --n-predict 1 --temperature 0 --top-p 1 --min-p 0 --seed 1 2>&1");
    const std::string output = normalize_terminal_output(raw_output);

    expect_contains(output, "[debug] using model ", "home config should be discovered automatically");
    expect_contains(output, source_model_path.string(), "home config should provide the configured model path");
}

void test_missing_config_reports_error() {
    const std::filesystem::path source_model_path = binary_path().parent_path() / "model" / "llama3.2-1b.gguf";
    expect_true(std::filesystem::exists(source_model_path), "runtime smoke test model is missing");

    const TempDirectory temp_dir;
    const std::filesystem::path temp_binary = temp_dir.path / "yedera";
    const std::filesystem::path missing_config = temp_dir.path / "yedera.conf";

    std::filesystem::copy_file(binary_path(), temp_binary, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::permissions(
        temp_binary,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec |
            std::filesystem::perms::group_read | std::filesystem::perms::group_exec |
            std::filesystem::perms::others_read | std::filesystem::perms::others_exec,
        std::filesystem::perm_options::replace);

    expect_true(!std::filesystem::exists(missing_config), "test precondition failed: config should not exist yet");

    const CommandResult result = run_command_capture(
        "cd " + shell_quote(temp_dir.path.string()) + " && HOME=" + shell_quote((temp_dir.path / "home").string()) +
        " " + shell_quote(temp_binary.string()) +
        " --model-path " + shell_quote(source_model_path.string()) +
        " --system-prompt 'You are an autogenerated config test assistant.'"
        " --n-gpu-layers 0 --n-predict 1 --temperature 0 --top-p 1 --min-p 0 --seed 1"
        " --prompt 'Reply with one word.' 2>&1");
    const std::string output = normalize_terminal_output(result.output);

    expect_true(result.status != 0, "missing config should cause the binary to return an error");
    expect_true(!std::filesystem::exists(missing_config), "missing config should not be auto-generated");
    expect_contains(output, "error: config file not found: " + missing_config.string(), "missing config should report a not-found error");
}

void test_debug_output_stays_filtered() {
    const std::filesystem::path model_path = binary_path().parent_path() / "model" / "llama3.2-1b.gguf";
    expect_true(std::filesystem::exists(model_path), "runtime smoke test model is missing");

    const TempDirectory temp_dir;
    const std::filesystem::path config_path = temp_dir.path / "yedera.conf";
    write_file(
        config_path,
        "prompt = \"You are a test assistant.\"\n"
        "model_path = \"" + model_path.string() + "\"\n"
        "n_gpu_layers = 0\n"
        "debug = true\n"
        "verbose = false\n");

    const std::string raw_output = run_command(
        shell_quote(binary_path().string()) +
        " --config " + shell_quote(config_path.string()) +
        " --prompt 'Reply with one word.' --n-predict 1 --temperature 0 --top-p 1 --min-p 0 --seed 1 2>&1");
    const std::string output = normalize_terminal_output(raw_output);

    expect_contains(output, "[debug] using model ", "debug output should include the model path");
    expect_contains(output, "[debug] RAG mode disabled", "debug output should report RAG disabled when no embeddings model is configured");
    expect_contains(output, "[debug] offload request: CPU-only", "debug output should include the resolved offload mode");
    expect_contains(output, "[debug] model load: 100%", "debug output should report terminal load progress");
    expect_not_contains(output, "[debug] model load: 0%", "single-line progress should collapse intermediate percentages");
    expect_not_contains(output, "ggml_cuda_init:", "filtered debug output should not include raw backend initialization noise");
}

void test_rag_retrieval_reports_indexed_documents() {
    const std::filesystem::path model_path = binary_path().parent_path() / "model" / "llama3.2-1b.gguf";
    expect_true(std::filesystem::exists(model_path), "runtime smoke test model is missing");

    const TempDirectory temp_dir;
    const std::filesystem::path rag_dir = temp_dir.path / "rag";
    const std::filesystem::path config_path = temp_dir.path / "yedera.conf";
    std::filesystem::create_directories(rag_dir);
    write_file(rag_dir / "note.md", "garage retrieval keyword\n\nThe answer lives in the local note file.\n");
    write_file(rag_dir / "odd.txt", "the obsidian teacup is catalogued under shelf delta\n");
    write_file(
        config_path,
        "prompt = \"You are a test assistant.\"\n"
        "model_path = \"" + model_path.string() + "\"\n"
        "model_embeddings = \"" + model_path.string() + "\"\n"
        "rag_documents_path = \"" + rag_dir.string() + "\"\n"
        "n_gpu_layers = 0\n"
        "debug = true\n"
        "verbose = false\n");

    const std::string raw_output = run_command(
        shell_quote(binary_path().string()) +
        " --config " + shell_quote(config_path.string()) +
        " --prompt 'garage retrieval keyword' --n-predict 1 --temperature 0 --top-p 1 --min-p 0 --seed 1 2>&1");
    const std::string output = normalize_terminal_output(raw_output);

    expect_contains(output, "[debug] using model ", "RAG debug output should include the main model path");
    expect_contains(output, "[debug] using embeddings model ", "RAG debug output should include the embeddings model path");
    expect_contains(output, "[debug] RAG mode enabled", "RAG debug output should report RAG enabled when an embeddings model is configured");
    expect_contains(output, "[debug] RAG: indexing documents from " + rag_dir.lexically_normal().string(), "RAG debug output should show the configured document directory");
    expect_contains(output, "[debug] RAG: learning content from files", "RAG debug output should announce file learning");
    expect_contains(output, "[rag] " + (rag_dir / "note.md").lexically_normal().string(), "RAG debug output should list learned markdown files");
    expect_contains(output, "[rag] " + (rag_dir / "odd.txt").lexically_normal().string(), "RAG debug output should list learned text files");
    expect_contains(output, "[debug] RAG: indexed ", "RAG should index the configured documents");
    expect_contains(output, "[retrieved ", "RAG should report the retrieved document chunks");
}

void test_interactive_learn_command_does_not_conflict_with_prompt() {
    const std::filesystem::path model_path = binary_path().parent_path() / "model" / "llama3.2-1b.gguf";
    expect_true(std::filesystem::exists(model_path), "runtime smoke test model is missing");

    const TempDirectory temp_dir;
    const std::filesystem::path learn_file = temp_dir.path / "learn-note.md";
    const std::filesystem::path config_path = temp_dir.path / "yedera.conf";
    write_file(learn_file, "prompt mode special keyword\n\nThis file should be learned during the chat session.\n");
    write_file(
        config_path,
        "prompt = \"You are a test assistant.\"\n"
        "model_path = \"" + model_path.string() + "\"\n"
        "model_embeddings = \"" + model_path.string() + "\"\n"
        "n_gpu_layers = 0\n"
        "debug = true\n"
        "verbose = false\n");

    const std::string learn_command = "/learn " + learn_file.string();
    const std::string escaped_prompt = "//learn prompt mode special keyword";
    const std::string raw_output = run_command(
        "printf '%s\n%s\n\n' " + shell_quote(learn_command) + " " + shell_quote(escaped_prompt) + " | " +
        shell_quote(binary_path().string()) +
        " --config " + shell_quote(config_path.string()) +
        " --interactive --n-predict 1 --temperature 0 --top-p 1 --min-p 0 --seed 1 2>&1");
    const std::string output = normalize_terminal_output(raw_output);

    expect_contains(output, "[debug] using embeddings model ", "interactive learn should still enable RAG mode");
    expect_contains(output, "[debug] RAG: indexing documents from " + learn_file.lexically_normal().string(), "interactive /learn should index the requested file");
    expect_contains(output, "[rag] " + learn_file.lexically_normal().string(), "interactive /learn should list the learned file");
    expect_contains(output, "[rag] learned " + learn_file.lexically_normal().string(), "interactive /learn should acknowledge the learned source");
    expect_contains(output, "[retrieved ", "escaped //learn prompt should still reach normal RAG retrieval");
}

void test_escape_file_entry_resolves_paths_and_returns_to_chat_prompt() {
#if !defined(__linux__)
    return;
#else
    const std::filesystem::path model_path = binary_path().parent_path() / "model" / "llama3.2-1b.gguf";
    expect_true(std::filesystem::exists(model_path), "runtime smoke test model is missing");

    const TempDirectory temp_dir;
    const std::filesystem::path rag_dir = temp_dir.path / "rag";
    std::filesystem::create_directories(rag_dir);
    const std::filesystem::path relative_file = rag_dir / "escape-note.md";
    const std::filesystem::path absolute_file = temp_dir.path / "absolute-note.md";
    const std::filesystem::path wildcard_a = rag_dir / "wild-a.md";
    const std::filesystem::path wildcard_b = rag_dir / "wild-b.md";
    const std::filesystem::path config_path = temp_dir.path / "yedera.conf";
    write_file(relative_file, "escape mode keyword\n\nThis file is learned from the ESC prompt mode.\n");
    write_file(absolute_file, "absolute escape keyword\n\nThis absolute file is learned from the ESC prompt mode.\n");
    write_file(wildcard_a, "wildcard escape alpha\n\nThis wildcard file is learned from the ESC prompt mode.\n");
    write_file(wildcard_b, "wildcard escape beta\n\nThis wildcard file is learned from the ESC prompt mode.\n");
    write_file(
        config_path,
        "prompt = \"You are a test assistant.\"\n"
        "model_path = \"" + model_path.string() + "\"\n"
        "model_embeddings = \"" + model_path.string() + "\"\n"
        "rag_documents_path = \"" + rag_dir.string() + "\"\n"
        "n_gpu_layers = 0\n"
        "debug = true\n"
        "verbose = false\n");

    const std::string raw_output = run_command_in_pty(
        {
            binary_path().string(),
            "--config",
            config_path.string(),
            "--interactive",
            "--n-predict",
            "1",
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--min-p",
            "0",
            "--seed",
            "1",
        },
        {
            {"\x1b", "\r\x1b[2K: "},
            {"\x1b", "\r\x1b[2K> "},
            {"\x1b", "\r\x1b[2K: "},
            {"escape-note.md\n", "[rag] learned " + relative_file.lexically_normal().string()},
            {"\x1b", "\r\x1b[2K: "},
            {absolute_file.string() + "\n", "[rag] learned " + absolute_file.lexically_normal().string()},
            {"\x1b", "\r\x1b[2K: "},
            {"wild-*.md\n", "[rag] learned " + wildcard_b.lexically_normal().string()},
            {"escape mode keyword\n", "[retrieved "},
            {"\n", ""},
        });
    const std::string output = normalize_terminal_output(raw_output);

    expect_contains(raw_output, "\r\x1b[2K: ", "ESC on an empty entry should switch to the learn-file prompt");
    expect_contains(raw_output, "\r\x1b[2K> ", "ESC on an empty learn prompt should switch back to the chat prompt");
    expect_contains(output, relative_file.lexically_normal().string() + " <100% rag tuning>", "relative ESC file entry should show RAG tuning progress for the resolved file");
    expect_contains(output, "[rag] learned " + relative_file.lexically_normal().string(), "relative ESC file entry should resolve under rag_documents_path");
    expect_contains(output, "[rag] learned " + absolute_file.lexically_normal().string(), "absolute ESC file entry should be used as entered");
    expect_contains(output, wildcard_a.lexically_normal().string() + " <50% rag tuning>", "wildcard ESC file entry should report progress across matched files");
    expect_contains(output, wildcard_b.lexically_normal().string() + " <100% rag tuning>", "wildcard ESC file entry should complete progress at 100%");
    expect_contains(output, "[retrieved ", "after toggling back to chat, the normal chat prompt should still run RAG retrieval");
#endif
}

void test_chat_prompt_arrow_keys_navigate_history() {
#if !defined(__linux__)
    return;
#else
    const std::filesystem::path model_path = binary_path().parent_path() / "model" / "llama3.2-1b.gguf";
    expect_true(std::filesystem::exists(model_path), "runtime smoke test model is missing");

    const TempDirectory temp_dir;
    const std::filesystem::path config_path = temp_dir.path / "yedera.conf";
    write_file(
        config_path,
        "prompt = \"You are a test assistant.\"\n"
        "model_path = \"" + model_path.string() + "\"\n"
        "n_gpu_layers = 0\n"
        "debug = false\n"
        "verbose = false\n");

    const std::string raw_output = run_command_in_pty(
        {
            binary_path().string(),
            "--config",
            config_path.string(),
            "--interactive",
            "--n-predict",
            "1",
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--min-p",
            "0",
            "--seed",
            "1",
        },
        {
            {"history alpha\n", "\n> "},
            {"\x1b[A", "\r\x1b[2K> history alpha"},
            {"\x1b[B", "\r\x1b[2K> "},
            {"\n", ""},
        });

    expect_contains(raw_output, "\r\x1b[2K> history alpha", "up arrow should recall the previous chat entry");
    expect_contains(raw_output, "\r\x1b[2K> ", "down arrow should move forward in chat history");
#endif
}

} // namespace

int main() {
    try {
        test_list_command_uses_executable_relative_model_dir();
        test_config_is_discovered_from_current_directory();
        test_config_falls_back_to_home_directory();
        test_missing_config_reports_error();
        test_debug_output_stays_filtered();
        test_rag_retrieval_reports_indexed_documents();
        test_interactive_learn_command_does_not_conflict_with_prompt();
        test_escape_file_entry_resolves_paths_and_returns_to_chat_prompt();
        test_chat_prompt_arrow_keys_navigate_history();
        std::cout << "runtime tests passed\n";
        return 0;
    } catch (const TestFailure & error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    } catch (const std::exception & error) {
        std::cerr << "unexpected error: " << error.what() << '\n';
        return 1;
    }
}

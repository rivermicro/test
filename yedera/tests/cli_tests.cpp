#include "cli.hpp"
#include "pull.hpp"
#include "rag.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct TempDirectory {
    TempDirectory() {
        char path_template[] = "/tmp/yedera-config-tests-XXXXXX";
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

template <typename T>
void expect_equal(const T & actual, const T & expected, const std::string & message) {
    if (!(actual == expected)) {
        throw TestFailure(message);
    }
}

void write_file(const std::filesystem::path & path, const std::string & content) {
    std::ofstream output(path);
    if (!output.is_open()) {
        throw std::runtime_error("failed to write test file: " + path.string());
    }
    output << content;
}

void test_parse_args_collects_flags_and_prompt() {
    std::vector<std::string> values = {
        "yedera",
        "--config", "/tmp/test.conf",
        "--model-path", "model.gguf",
        "--temperature", "0.2",
        "--seed", "7",
        "hello",
        "from",
        "tests"
    };

    std::vector<char *> argv;
    argv.reserve(values.size());
    for (std::string & value : values) {
        argv.push_back(value.data());
    }

    const OptionOverrides overrides = parse_args(static_cast<int>(argv.size()), argv.data());
    expect_true(overrides.config_path.has_value(), "expected --config to be parsed");
    expect_equal(*overrides.config_path, std::string("/tmp/test.conf"), "unexpected config path");
    expect_true(overrides.model_path.has_value(), "expected --model-path to be parsed");
    expect_equal(*overrides.model_path, std::string("model.gguf"), "unexpected model path");
    expect_true(overrides.temperature.has_value(), "expected --temperature to be parsed");
    expect_equal(*overrides.temperature, 0.2f, "unexpected temperature");
    expect_true(overrides.seed.has_value(), "expected --seed to be parsed");
    expect_equal(*overrides.seed, static_cast<uint32_t>(7), "unexpected seed");
    expect_true(overrides.user_prompt.has_value(), "expected positional prompt to be combined");
    expect_equal(*overrides.user_prompt, std::string("hello from tests"), "unexpected combined prompt");
}

void test_resolve_options_reads_config_and_cli_overrides() {
    const TempDirectory temp_dir;
    const std::filesystem::path config_path = temp_dir.path / "yedera.conf";
    write_file(
        config_path,
        "prompt = \"system prompt\"\n"
        "model_path = \"model-from-config.gguf\"\n"
        "model_embeddings = \"model-embeddings.gguf\"\n"
        "rag_documents_path = \"docs\"\n"
        "user_prompt = \"config prompt\"\n"
        "n_predict = 42\n"
        "n_gpu_layers = 0\n"
        "debug = false\n");

    OptionOverrides overrides;
    overrides.config_path = config_path.string();
    overrides.user_prompt = std::string("cli prompt");
    overrides.debug = true;
    overrides.temperature = 0.1f;

    const Options resolved = resolve_options(overrides);
    expect_equal(resolved.config_path, config_path.string(), "unexpected resolved config path");
    expect_equal(resolved.model_path, (temp_dir.path / "model-from-config.gguf").lexically_normal().string(), "unexpected resolved model path");
    expect_equal(resolved.model_embeddings, (temp_dir.path / "model-embeddings.gguf").lexically_normal().string(), "unexpected resolved embeddings model");
    expect_equal(resolved.rag_documents_path, (temp_dir.path / "docs").lexically_normal().string(), "unexpected resolved rag documents path");
    expect_equal(resolved.system_prompt, std::string("system prompt"), "unexpected resolved system prompt");
    expect_equal(resolved.prompt, std::string("cli prompt"), "CLI prompt should override config prompt");
    expect_equal(resolved.n_predict, 42, "unexpected resolved n_predict");
    expect_equal(resolved.n_gpu_layers, 0, "explicit config n_gpu_layers should be preserved");
    expect_equal(resolved.temperature, 0.1f, "CLI temperature should override config");
    expect_true(resolved.debug, "CLI debug override should be applied");
}

void test_resolve_options_defaults_to_interactive_without_prompt() {
    const TempDirectory temp_dir;
    const std::filesystem::path config_path = temp_dir.path / "yedera.conf";
    write_file(
        config_path,
        "prompt = \"system prompt\"\n"
        "model_path = \"model.gguf\"\n"
        "n_gpu_layers = 0\n"
        "interactive = false\n");

    OptionOverrides overrides;
    overrides.config_path = config_path.string();

    const Options resolved = resolve_options(overrides);
    expect_true(resolved.interactive, "missing prompt should force interactive mode");
    expect_equal(resolved.prompt, std::string(), "prompt should remain empty when not configured");
}

void test_resolve_options_rejects_invalid_debug_value() {
    const TempDirectory temp_dir;
    const std::filesystem::path config_path = temp_dir.path / "yedera.conf";
    write_file(
        config_path,
        "prompt = \"system prompt\"\n"
        "model_path = \"model.gguf\"\n"
        "n_gpu_layers = 0\n"
        "debug = maybe\n");

    OptionOverrides overrides;
    overrides.config_path = config_path.string();

    bool threw = false;
    try {
        (void) resolve_options(overrides);
    } catch (const std::runtime_error & error) {
        threw = std::string(error.what()).find("invalid boolean for debug") != std::string::npos;
    }

    expect_true(threw, "invalid debug value should produce a config parse error");
}

void test_resolve_options_resolves_relative_config_paths() {
    const TempDirectory temp_dir;
    const std::filesystem::path config_dir = temp_dir.path / "config";
    std::filesystem::create_directories(config_dir);
    const std::filesystem::path config_path = config_dir / "yedera.conf";
    write_file(
        config_path,
        "prompt = \"system prompt\"\n"
        "model_path = \"model/chat.gguf\"\n"
        "model_embeddings = \"model/embed.gguf\"\n"
        "rag_documents_path = \"rag\"\n"
        "n_gpu_layers = 0\n");

    OptionOverrides overrides;
    overrides.config_path = config_path.string();

    const Options resolved = resolve_options(overrides);
    expect_equal(resolved.model_path, (config_dir / "model" / "chat.gguf").lexically_normal().string(), "model_path should resolve relative to the config file");
    expect_equal(resolved.model_embeddings, (config_dir / "model" / "embed.gguf").lexically_normal().string(), "model_embeddings should resolve relative to the config file");
    expect_equal(resolved.rag_documents_path, (config_dir / "rag").lexically_normal().string(), "rag_documents_path should resolve relative to the config file");
}

void test_plan_model_download_maps_known_alias() {
    const TempDirectory temp_dir;
    const std::filesystem::path configured_path = temp_dir.path / "model" / "nomic-embed-text-v1.5.gguf";

    const std::optional<ModelDownloadPlan> plan = plan_model_download(configured_path.string());
    expect_true(plan.has_value(), "known model alias should produce a download plan");
    expect_equal(plan->destination_path, configured_path, "download plan should preserve the configured destination path");
    expect_equal(
        plan->download_url,
        std::string("https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf?download=true"),
        "download plan should resolve the configured alias to the expected URL");
}

void test_plan_model_download_skips_existing_file() {
    const TempDirectory temp_dir;
    const std::filesystem::path configured_path = temp_dir.path / "model" / "llama3.2-1b.gguf";
    std::filesystem::create_directories(configured_path.parent_path());
    write_file(configured_path, "already here");

    const std::optional<ModelDownloadPlan> plan = plan_model_download(configured_path.string());
    expect_true(!plan.has_value(), "existing configured file should not trigger a download plan");
}

void test_format_rag_prompt_prefers_direct_fact_lookup() {
    const std::string prompt = format_rag_prompt(
        "pinu phone number: 49232\nlisa phone number: 11123\n\n",
        "whats lisa and pinu phone number?");

    expect_true(
        prompt.find("The current conversation and the user's own statements are authoritative") != std::string::npos,
        "RAG prompt should not override session facts from the chat");
    expect_true(
        prompt.find("For direct lookup questions, reply with the exact facts from the retrieved context.") != std::string::npos,
        "RAG prompt should instruct the model to answer direct fact lookups from retrieved context");
    expect_true(
        prompt.find("Do not refuse, speculate, or add policy commentary") != std::string::npos,
        "RAG prompt should suppress generic refusals when context already contains the answer");
    expect_true(
        prompt.find("pinu phone number: 49232") != std::string::npos,
        "RAG prompt should include the retrieved facts");
}

void test_should_use_rag_skips_session_memory_prompts() {
    expect_true(!should_use_rag_for_input("what is my name?"), "name questions should use chat memory instead of RAG");
    expect_true(!should_use_rag_for_input("my name is cris"), "user-provided identity facts should not trigger RAG");
    expect_true(should_use_rag_for_input("item id for 365905899470"), "document lookup prompts should still use RAG");
    expect_true(should_use_rag_for_input("i am looking for the price of item 365905899470"), "lookup prompts with first-person wording should still use RAG");
}

} // namespace

int main() {
    try {
        test_parse_args_collects_flags_and_prompt();
        test_resolve_options_reads_config_and_cli_overrides();
        test_resolve_options_defaults_to_interactive_without_prompt();
        test_resolve_options_rejects_invalid_debug_value();
        test_resolve_options_resolves_relative_config_paths();
        test_plan_model_download_maps_known_alias();
        test_plan_model_download_skips_existing_file();
        test_format_rag_prompt_prefers_direct_fact_lookup();
        test_should_use_rag_skips_session_memory_prompts();
        std::cout << "cli tests passed\n";
        return 0;
    } catch (const TestFailure & error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    } catch (const std::exception & error) {
        std::cerr << "unexpected error: " << error.what() << '\n';
        return 1;
    }
}

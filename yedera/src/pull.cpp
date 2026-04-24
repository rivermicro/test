#include "pull.hpp"

#include "http.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>

namespace {

struct PullTarget {
    std::string download_url;
    std::string local_filename;
};

const std::map<std::string, PullTarget> & known_pull_targets() {
    static const std::map<std::string, PullTarget> targets = {
        {
            "nomic-embed-text-v1.5.gguf",
            {
                "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf?download=true",
                "nomic-embed-text-v1.5.gguf",
            },
        },
        {
            "nomic-embed-text-v1.5.f16.gguf",
            {
                "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf?download=true",
                "nomic-embed-text-v1.5.f16.gguf",
            },
        },
        {
            "llama3.2-1b.gguf",
            {
                "https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF/resolve/main/llama-3.2-1b-instruct-q4_k_m.gguf?download=true",
                "llama3.2-1b.gguf",
            },
        },
        {
            "llama3.2-3b.gguf",
            {
                "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf?download=true",
                "llama3.2-3b.gguf",
            },
        },
    };

    return targets;
}

bool is_url(const std::string & value) {
    return value.rfind("https://", 0) == 0 || value.rfind("http://", 0) == 0;
}

std::string filename_from_url(const std::string & url) {
    const common_http_url parts = common_http_parse_url(url);
    std::string path = parts.path;
    const size_t query_offset = path.find_first_of("?#");
    if (query_offset != std::string::npos) {
        path = path.substr(0, query_offset);
    }

    const std::string filename = std::filesystem::path(path).filename().string();
    if (filename.empty()) {
        throw std::runtime_error("could not determine a filename from URL: " + url);
    }

    return filename;
}

PullTarget resolve_pull_target(const std::string & value) {
    const auto known = known_pull_targets().find(value);
    if (known != known_pull_targets().end()) {
        return known->second;
    }

    if (is_url(value)) {
        return {value, filename_from_url(value)};
    }

    std::string supported;
    for (const auto & entry : known_pull_targets()) {
        if (!supported.empty()) {
            supported += ", ";
        }
        supported += entry.first;
    }

    throw std::runtime_error(
        "unknown model alias '" + value + "'; pass a direct https URL or one of: " + supported);
}

std::string basename_for_model_path(const std::string & configured_model_path) {
    return std::filesystem::path(configured_model_path).filename().string();
}

size_t parse_content_length(const httplib::Response & response) {
    if (!response.has_header("Content-Length")) {
        return 0;
    }

    try {
        return static_cast<size_t>(std::stoull(response.get_header_value("Content-Length")));
    } catch (const std::exception &) {
        return 0;
    }
}

void ensure_parent_directory(const std::filesystem::path & path) {
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) {
        throw std::runtime_error("failed to create directory: " + path.parent_path().string());
    }
}

void remove_if_exists(const std::filesystem::path & path) {
    std::error_code error;
    std::filesystem::remove(path, error);
}

void print_progress(size_t downloaded, size_t total_size, size_t & last_percent) {
    if (total_size == 0) {
        return;
    }

    const size_t percent = (downloaded * 100) / total_size;
    if (percent == last_percent && downloaded != total_size) {
        return;
    }

    last_percent = percent;
    std::cerr << "\r[pull] " << percent << "% (" << (downloaded / (1024 * 1024)) << " / "
              << (total_size / (1024 * 1024)) << " MiB)" << std::flush;
    if (downloaded == total_size) {
        std::cerr << '\n';
    }
}

void download_to_path(const std::string & url, const std::filesystem::path & destination_path) {
    auto [cli, parts] = common_http_client(url);

    cli.set_default_headers({{"User-Agent", "yedera"}});
    cli.set_follow_location(true);
    cli.set_connection_timeout(30);
    cli.set_read_timeout(300, 0);
    cli.set_write_timeout(300, 0);

#if !defined(CPPHTTPLIB_OPENSSL_SUPPORT)
    if (parts.scheme == "https") {
        throw std::runtime_error("HTTPS download support is not enabled in this build");
    }
#endif

    const auto head = cli.Head(parts.path);
    if (!head) {
        throw std::runtime_error("failed to contact remote host for: " + url + " (" + httplib::to_string(head.error()) + ")");
    }
    if (head->status < 200 || head->status >= 300) {
        throw std::runtime_error("remote server returned HTTP " + std::to_string(head->status) + " for: " + url);
    }

    const size_t total_size = parse_content_length(*head);
    const std::filesystem::path temporary_path = destination_path.string() + ".downloadInProgress";
    ensure_parent_directory(destination_path);
    remove_if_exists(temporary_path);

    std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("failed to open download destination: " + temporary_path.string());
    }

    size_t downloaded = 0;
    size_t last_percent = static_cast<size_t>(-1);

    const auto result = cli.Get(
        parts.path,
        [&](const httplib::Response & response) {
            return response.status >= 200 && response.status < 300;
        },
        [&](const char * data, size_t data_length) {
            output.write(data, static_cast<std::streamsize>(data_length));
            if (!output) {
                return false;
            }

            downloaded += data_length;
            print_progress(downloaded, total_size, last_percent);
            return true;
        });

    output.close();

    if (!result) {
        remove_if_exists(temporary_path);
        throw std::runtime_error("download failed for: " + url + " (" + httplib::to_string(result.error()) + ")");
    }
    if (result->status < 200 || result->status >= 300) {
        remove_if_exists(temporary_path);
        throw std::runtime_error("remote server returned HTTP " + std::to_string(result->status) + " for: " + url);
    }

    if (total_size > 0 && downloaded != total_size) {
        remove_if_exists(temporary_path);
        throw std::runtime_error(
            "download size mismatch for: " + url + " (expected " + std::to_string(total_size) +
            " bytes, got " + std::to_string(downloaded) + ")");
    }

    std::error_code rename_error;
    std::filesystem::rename(temporary_path, destination_path, rename_error);
    if (rename_error) {
        remove_if_exists(temporary_path);
        throw std::runtime_error("failed to finalize downloaded file: " + destination_path.string());
    }
}

} // namespace

std::optional<ModelDownloadPlan> plan_model_download(const std::string & configured_model_path) {
    if (configured_model_path.empty()) {
        return std::nullopt;
    }

    const std::filesystem::path destination_path(configured_model_path);
    if (std::filesystem::exists(destination_path)) {
        return std::nullopt;
    }

    const PullTarget target = resolve_pull_target(basename_for_model_path(configured_model_path));
    return ModelDownloadPlan{target.download_url, destination_path};
}

void ensure_configured_models_available(const Options & options) {
    auto ensure_model = [](const std::string & configured_model_path, const char * field_name) {
        if (configured_model_path.empty()) {
            return;
        }

        const std::filesystem::path destination_path(configured_model_path);
        if (std::filesystem::exists(destination_path)) {
            return;
        }

        const std::optional<ModelDownloadPlan> plan = plan_model_download(configured_model_path);
        if (!plan.has_value()) {
            return;
        }

        std::cerr << "[download] " << field_name << ": " << destination_path.string() << '\n';
        std::cerr << "[download] source: " << plan->download_url << '\n';

        download_to_path(plan->download_url, plan->destination_path);
    };

    ensure_model(options.model_path, "model_path");
    ensure_model(options.model_embeddings, "model_embeddings");
}
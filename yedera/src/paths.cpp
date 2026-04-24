#include "paths.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

std::filesystem::path get_executable_path() {
    std::error_code error;
    const std::filesystem::path executable_path = std::filesystem::read_symlink("/proc/self/exe", error);
    if (error) {
        throw std::runtime_error("failed to resolve executable path");
    }

    return executable_path;
}

std::filesystem::path get_model_directory() {
    return get_executable_path().parent_path() / "model";
}

int handle_list_command() {
    const std::filesystem::path model_dir = get_model_directory();
    if (!std::filesystem::exists(model_dir)) {
        std::cout << "No models found in " << model_dir.string() << '\n';
        return 0;
    }

    std::vector<std::string> entries;
    for (const auto & entry : std::filesystem::recursive_directory_iterator(model_dir)) {
        if (entry.is_regular_file()) {
            entries.push_back(entry.path().lexically_normal().string());
        }
    }

    std::sort(entries.begin(), entries.end());

    if (entries.empty()) {
        std::cout << "No models found in " << model_dir.string() << '\n';
        return 0;
    }

    for (const auto & path : entries) {
        std::cout << path << '\n';
    }

    return 0;
}
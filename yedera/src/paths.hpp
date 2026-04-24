#pragma once

#include <filesystem>

std::filesystem::path get_executable_path();
std::filesystem::path get_model_directory();
int handle_list_command();
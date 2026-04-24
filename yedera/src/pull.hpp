#pragma once

#include "options.hpp"

#include <filesystem>
#include <optional>
#include <string>

struct ModelDownloadPlan {
	std::string download_url;
	std::filesystem::path destination_path;
};

std::optional<ModelDownloadPlan> plan_model_download(const std::string & configured_model_path);
void ensure_configured_models_available(const Options & options);
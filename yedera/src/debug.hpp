#pragma once

#include "options.hpp"

struct DebugState {
    bool enabled = false;
    bool verbose = false;
    int load_progress_bucket = -1;
    bool load_progress_active = false;
};

void install_startup_log_router();
void install_runtime_log_router(DebugState * debug_state);
bool handle_model_load_progress(float progress, void * user_data);
void print_backend_debug_summary(const Options & options);
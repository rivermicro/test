#pragma once

#include "options.hpp"

void print_usage(const char * program_name);
OptionOverrides parse_args(int argc, char ** argv);
Options resolve_options(const OptionOverrides & cli_overrides);
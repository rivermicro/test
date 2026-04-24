#include "cli.hpp"
#include "debug.hpp"
#include "paths.hpp"
#include "pull.hpp"
#include "runtime.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
int main(int argc, char ** argv) {
    try {
        install_startup_log_router();

        if (argc >= 2 && std::string(argv[1]) == "list") {
            return handle_list_command();
        }

        const OptionOverrides cli_overrides = parse_args(argc, argv);
        if (cli_overrides.help) {
            print_usage(argv[0]);
            return 0;
        }

        const Options options = resolve_options(cli_overrides);
        ensure_configured_models_available(options);
        run_inference(options);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
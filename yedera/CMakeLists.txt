cmake_minimum_required(VERSION 3.20)

project(yedera LANGUAGES CXX)

include(GNUInstallDirs)
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build all libraries static" FORCE)
set(YEDERA_LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp")

find_package(CUDAToolkit QUIET)
set(YEDERA_ENABLE_CUDA OFF)
if (CUDAToolkit_FOUND)
    set(YEDERA_ENABLE_CUDA ON)
    message(STATUS "CUDA toolkit found; enabling NVIDIA backend")
else()
    message(STATUS "CUDA toolkit not found; building CPU-only backend")
endif()

if (NOT EXISTS "${YEDERA_LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Vendored llama.cpp checkout not found at ${YEDERA_LLAMA_CPP_DIR}")
endif()

set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${YEDERA_ENABLE_CUDA} CACHE BOOL "" FORCE)
set(GGML_CUDA_NO_VMM ${YEDERA_ENABLE_CUDA} CACHE BOOL "" FORCE)

add_subdirectory("${YEDERA_LLAMA_CPP_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp")

if (LLAMA_HTTPLIB AND NOT TARGET cpp-httplib)
    add_subdirectory("${YEDERA_LLAMA_CPP_DIR}/vendor/cpp-httplib" "${CMAKE_BINARY_DIR}/cpp-httplib")
endif()

find_package(Threads REQUIRED)

add_executable(yedera
    src/main.cpp
    src/cli.cpp
    src/debug.cpp
    src/paths.cpp
    src/pull.cpp
    src/rag.cpp
    src/runtime.cpp)
add_custom_command(TARGET yedera POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E rm -f
        $<TARGET_FILE_DIR:yedera>/yedera.cfg
    COMMAND ${CMAKE_COMMAND} -E rm -f
        $<TARGET_FILE_DIR:yedera>/yedera.conf
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/yedera.conf
        $<TARGET_FILE_DIR:yedera>/yedera.conf)
target_include_directories(yedera PRIVATE
    ${YEDERA_LLAMA_CPP_DIR}/common
    ${YEDERA_LLAMA_CPP_DIR}/vendor)
target_link_libraries(yedera PRIVATE llama Threads::Threads cpp-httplib)
target_compile_features(yedera PRIVATE cxx_std_17)
target_compile_definitions(yedera PRIVATE
    YEDERA_CUDA_ENABLED=$<BOOL:${YEDERA_ENABLE_CUDA}>
    YEDERA_CUDA_VERSION="${CUDAToolkit_VERSION}")
if (NOT YEDERA_ENABLE_CUDA)
    target_link_options(yedera PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-static>)
endif()
target_link_options(yedera PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-static-libgcc>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-static-libstdc++>)

install(TARGETS yedera RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES yedera.conf DESTINATION ${CMAKE_INSTALL_BINDIR})

if (BUILD_TESTING)
    add_test(
        NAME yedera-cli-positional-prompt-test
        COMMAND ${CMAKE_COMMAND}
            -DCASE=positional_prompt
            -DYEDERA_BINARY=$<TARGET_FILE:yedera>
            -DYEDERA_BINARY_DIR=$<TARGET_FILE_DIR:yedera>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_integration_test.cmake)

    add_test(
        NAME yedera-cli-no-prompt-chat-test
        COMMAND ${CMAKE_COMMAND}
            -DCASE=no_prompt_chat
            -DYEDERA_BINARY=$<TARGET_FILE:yedera>
            -DYEDERA_BINARY_DIR=$<TARGET_FILE_DIR:yedera>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_integration_test.cmake)

    add_test(
        NAME yedera-cli-current-dir-config-test
        COMMAND ${CMAKE_COMMAND}
            -DCASE=current_dir_config
            -DYEDERA_BINARY=$<TARGET_FILE:yedera>
            -DYEDERA_BINARY_DIR=$<TARGET_FILE_DIR:yedera>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_integration_test.cmake)

    add_test(
        NAME yedera-cli-home-config-test
        COMMAND ${CMAKE_COMMAND}
            -DCASE=home_config
            -DYEDERA_BINARY=$<TARGET_FILE:yedera>
            -DYEDERA_BINARY_DIR=$<TARGET_FILE_DIR:yedera>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_integration_test.cmake)

    add_test(
        NAME yedera-cli-missing-config-test
        COMMAND ${CMAKE_COMMAND}
            -DCASE=missing_config
            -DYEDERA_BINARY=$<TARGET_FILE:yedera>
            -DYEDERA_BINARY_DIR=$<TARGET_FILE_DIR:yedera>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_integration_test.cmake)

    add_test(
        NAME yedera-cli-no-startup-rag-test
        COMMAND ${CMAKE_COMMAND}
            -DCASE=no_startup_rag
            -DYEDERA_BINARY=$<TARGET_FILE:yedera>
            -DYEDERA_BINARY_DIR=$<TARGET_FILE_DIR:yedera>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_integration_test.cmake)

    add_test(
        NAME yedera-cli-long-output-context-shift-test
        COMMAND ${CMAKE_COMMAND}
            -DCASE=long_output_context_shift
            -DYEDERA_BINARY=$<TARGET_FILE:yedera>
            -DYEDERA_BINARY_DIR=$<TARGET_FILE_DIR:yedera>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_integration_test.cmake)

    add_test(
        NAME yedera-cli-unlimited-n-predict-test
        COMMAND ${CMAKE_COMMAND}
            -DCASE=unlimited_n_predict
            -DYEDERA_BINARY=$<TARGET_FILE:yedera>
            -DYEDERA_BINARY_DIR=$<TARGET_FILE_DIR:yedera>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_integration_test.cmake)

    add_test(
        NAME yedera-cli-learn-multi-entry-test
        COMMAND ${CMAKE_COMMAND}
            -DCASE=learn_multi_entry_syntax
            -DYEDERA_BINARY=$<TARGET_FILE:yedera>
            -DYEDERA_BINARY_DIR=$<TARGET_FILE_DIR:yedera>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_integration_test.cmake)

    set_tests_properties(
        yedera-cli-positional-prompt-test
        yedera-cli-no-prompt-chat-test
        yedera-cli-current-dir-config-test
        yedera-cli-home-config-test
        yedera-cli-missing-config-test
        yedera-cli-no-startup-rag-test
        yedera-cli-long-output-context-shift-test
        yedera-cli-unlimited-n-predict-test
        yedera-cli-learn-multi-entry-test
        PROPERTIES TIMEOUT 180)
endif()

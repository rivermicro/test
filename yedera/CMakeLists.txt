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
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
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
    add_executable(yedera-config-tests
        tests/cli_tests.cpp
        src/cli.cpp
        src/pull.cpp
        src/paths.cpp)
    target_link_libraries(yedera-config-tests PRIVATE llama Threads::Threads cpp-httplib)
    target_compile_features(yedera-config-tests PRIVATE cxx_std_17)
    target_include_directories(yedera-config-tests PRIVATE
        src
        ${YEDERA_LLAMA_CPP_DIR}/common
        ${YEDERA_LLAMA_CPP_DIR}/vendor)
    add_test(NAME yedera-config-tests COMMAND yedera-config-tests)

    add_executable(yedera-runtime-tests
        tests/runtime_tests.cpp)
    target_compile_features(yedera-runtime-tests PRIVATE cxx_std_17)
    target_compile_definitions(yedera-runtime-tests PRIVATE
        YEDERA_BINARY_PATH="$<TARGET_FILE:yedera>")
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(yedera-runtime-tests PRIVATE util)
    endif()
    add_dependencies(yedera-runtime-tests yedera)
    add_test(NAME yedera-runtime-tests COMMAND yedera-runtime-tests)
endif()
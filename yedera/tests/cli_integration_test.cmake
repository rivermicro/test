cmake_minimum_required(VERSION 3.20)

include(CMakeParseArguments)

if (NOT DEFINED CASE)
    message(FATAL_ERROR "CASE is required")
endif()

if (NOT DEFINED YEDERA_BINARY)
    message(FATAL_ERROR "YEDERA_BINARY is required")
endif()

if (NOT DEFINED YEDERA_BINARY_DIR)
    message(FATAL_ERROR "YEDERA_BINARY_DIR is required")
endif()

set(MODEL_PATH "${YEDERA_BINARY_DIR}/model/llama3.2-1b.gguf")
if (NOT EXISTS "${MODEL_PATH}")
    message(FATAL_ERROR "runtime smoke test model is missing: ${MODEL_PATH}")
endif()

string(RANDOM LENGTH 10 ALPHABET 0123456789abcdef TEST_SUFFIX)
set(TEST_DIR "${YEDERA_BINARY_DIR}/test-tmp/${CASE}-${TEST_SUFFIX}")
file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}")

function(expect_contains haystack needle message)
    string(FIND "${haystack}" "${needle}" index)
    if (index EQUAL -1)
        message(FATAL_ERROR "${message}: missing '${needle}'\nOutput:\n${haystack}")
    endif()
endfunction()

function(expect_not_contains haystack needle message)
    string(FIND "${haystack}" "${needle}" index)
    if (NOT index EQUAL -1)
        message(FATAL_ERROR "${message}: unexpected '${needle}'\nOutput:\n${haystack}")
    endif()
endfunction()

function(write_test_pdf output_path text)
    set(script_path "${TEST_DIR}/write_test_pdf.py")
    file(WRITE "${script_path}" [=[
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = sys.argv[2]
stream = f"BT\n/F1 12 Tf\n72 720 Td\n({text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')}) Tj\nET\n"
objects = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
    f"<< /Length {len(stream)} >>\nstream\n{stream}endstream",
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
]
pdf = "%PDF-1.4\n"
offsets = [0]
for index, obj in enumerate(objects, start=1):
    offsets.append(len(pdf))
    pdf += f"{index} 0 obj\n{obj}\nendobj\n"
xref_offset = len(pdf)
pdf += f"xref\n0 {len(objects) + 1}\n"
pdf += "0000000000 65535 f \n"
for offset in offsets[1:]:
    pdf += f"{offset:010d} 00000 n \n"
pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
pdf += f"startxref\n{xref_offset}\n%%EOF\n"
path.write_bytes(pdf.encode("utf-8"))
]=])

    execute_process(
        COMMAND python3 "${script_path}" "${output_path}" "${text}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)

    if (NOT result EQUAL 0)
        message(FATAL_ERROR "failed to create test pdf ${output_path}\nOutput:\n${stdout}${stderr}")
    endif()
endfunction()

function(run_yedera)
    set(options EXPECT_FAILURE)
    set(oneValueArgs OUTPUT STATUS WORKING_DIRECTORY HOME INPUT_FILE)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(RUN "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (DEFINED RUN_HOME)
        if (DEFINED RUN_INPUT_FILE)
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E env HOME=${RUN_HOME} ${YEDERA_BINARY} ${RUN_ARGS}
                WORKING_DIRECTORY "${RUN_WORKING_DIRECTORY}"
                INPUT_FILE "${RUN_INPUT_FILE}"
                RESULT_VARIABLE result
                OUTPUT_VARIABLE stdout
                ERROR_VARIABLE stderr)
        else()
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E env HOME=${RUN_HOME} ${YEDERA_BINARY} ${RUN_ARGS}
                WORKING_DIRECTORY "${RUN_WORKING_DIRECTORY}"
                RESULT_VARIABLE result
                OUTPUT_VARIABLE stdout
                ERROR_VARIABLE stderr)
        endif()
    else()
        if (DEFINED RUN_INPUT_FILE)
            execute_process(
                COMMAND ${YEDERA_BINARY} ${RUN_ARGS}
                WORKING_DIRECTORY "${RUN_WORKING_DIRECTORY}"
                INPUT_FILE "${RUN_INPUT_FILE}"
                RESULT_VARIABLE result
                OUTPUT_VARIABLE stdout
                ERROR_VARIABLE stderr)
        else()
            execute_process(
                COMMAND ${YEDERA_BINARY} ${RUN_ARGS}
                WORKING_DIRECTORY "${RUN_WORKING_DIRECTORY}"
                RESULT_VARIABLE result
                OUTPUT_VARIABLE stdout
                ERROR_VARIABLE stderr)
        endif()
    endif()

    if (RUN_EXPECT_FAILURE)
        if (result EQUAL 0)
            message(FATAL_ERROR "command unexpectedly succeeded\nOutput:\n${stdout}${stderr}")
        endif()
    else()
        if (NOT result EQUAL 0)
            message(FATAL_ERROR "command failed with status ${result}\nOutput:\n${stdout}${stderr}")
        endif()
    endif()

    set(${RUN_OUTPUT} "${stdout}${stderr}" PARENT_SCOPE)
    set(${RUN_STATUS} "${result}" PARENT_SCOPE)
endfunction()

if (CASE STREQUAL "positional_prompt")
    file(WRITE "${TEST_DIR}/yedera.conf"
        "prompt = \"You are a test assistant.\"\n"
        "model_path = \"${MODEL_PATH}\"\n"
        "n_gpu_layers = 0\n"
        "debug = true\n"
        "verbose = false\n")

    run_yedera(
        OUTPUT output
        STATUS status
        WORKING_DIRECTORY "${TEST_DIR}"
        ARGS "Reply with one word." --n-predict 1 --temperature 0 --top-p 1 --min-p 0 --seed 1)

    expect_contains("${output}" "[debug] using model " "positional prompt should run inference through the main binary")
    expect_not_contains("${output}" "> " "positional prompt should not enter interactive chat mode")
elseif (CASE STREQUAL "no_prompt_chat")
    file(WRITE "${TEST_DIR}/yedera.conf"
        "prompt = \"You are a test assistant.\"\n"
        "model_path = \"${MODEL_PATH}\"\n"
        "n_gpu_layers = 0\n"
        "debug = true\n"
        "verbose = false\n")
    file(WRITE "${TEST_DIR}/stdin.txt" "\nReply with one word.\n")

    run_yedera(
        OUTPUT output
        STATUS status
        WORKING_DIRECTORY "${TEST_DIR}"
        INPUT_FILE "${TEST_DIR}/stdin.txt"
        ARGS --n-predict 1 --temperature 0 --top-p 1 --min-p 0 --seed 1)

    expect_contains("${output}" "> > " "empty Enter should keep chat mode active and redraw the chat prompt")
    expect_contains("${output}" "[debug] using model " "chat mode should still run through the main binary")
    expect_not_contains("${output}" ".........." "chat mode should not emit llama.cpp dot progress during startup")
elseif (CASE STREQUAL "current_dir_config")
    file(WRITE "${TEST_DIR}/yedera.conf"
        "prompt = \"You are a test assistant.\"\n"
        "model_path = \"${MODEL_PATH}\"\n"
        "n_gpu_layers = 0\n"
        "debug = true\n"
        "verbose = false\n")

    run_yedera(
        OUTPUT output
        STATUS status
        WORKING_DIRECTORY "${TEST_DIR}"
        ARGS "Reply with one word." --n-predict 1 --temperature 0 --top-p 1 --min-p 0 --seed 1)

    expect_contains("${output}" "[debug] using model " "current-dir config should be discovered automatically")
    expect_contains("${output}" "${MODEL_PATH}" "current-dir config should provide the configured model path")
    expect_contains("${output}" "[debug] RAG mode disabled" "current-dir config without embeddings should disable RAG")
elseif (CASE STREQUAL "home_config")
    set(HOME_DIR "${TEST_DIR}/home")
    file(MAKE_DIRECTORY "${HOME_DIR}/.yedera")
    file(WRITE "${HOME_DIR}/.yedera/yedera.conf"
        "prompt = \"You are a test assistant.\"\n"
        "model_path = \"${MODEL_PATH}\"\n"
        "n_gpu_layers = 0\n"
        "debug = true\n"
        "verbose = false\n")

    run_yedera(
        OUTPUT output
        STATUS status
        WORKING_DIRECTORY "${TEST_DIR}"
        HOME "${HOME_DIR}"
        ARGS "Reply with one word." --n-predict 1 --temperature 0 --top-p 1 --min-p 0 --seed 1)

    expect_contains("${output}" "[debug] using model " "home config should be discovered automatically")
    expect_contains("${output}" "${MODEL_PATH}" "home config should provide the configured model path")
elseif (CASE STREQUAL "missing_config")
    set(HOME_DIR "${TEST_DIR}/home")
    run_yedera(
        EXPECT_FAILURE
        OUTPUT output
        STATUS status
        WORKING_DIRECTORY "${TEST_DIR}"
        HOME "${HOME_DIR}"
        ARGS --model-path "${MODEL_PATH}" --system-prompt "You are an autogenerated config test assistant." --n-gpu-layers 0 --n-predict 1 --temperature 0 --top-p 1 --min-p 0 --seed 1 "Reply with one word.")

    expect_contains("${output}" "error: config file not found: ${TEST_DIR}/yedera.conf" "missing config should report a not-found error")
elseif (CASE STREQUAL "no_startup_rag")
    file(MAKE_DIRECTORY "${TEST_DIR}/rag")
    file(WRITE "${TEST_DIR}/rag/note.md" "garage retrieval keyword\n\nThe answer lives in the local note file.\n")
    file(WRITE "${TEST_DIR}/yedera.conf"
        "prompt = \"You are a test assistant.\"\n"
        "model_path = \"${MODEL_PATH}\"\n"
        "model_embeddings = \"${MODEL_PATH}\"\n"
        "rag_documents_path = \"${TEST_DIR}/rag\"\n"
        "n_gpu_layers = 0\n"
        "debug = true\n"
        "verbose = false\n")

    run_yedera(
        OUTPUT output
        STATUS status
        WORKING_DIRECTORY "${TEST_DIR}"
        ARGS --config "${TEST_DIR}/yedera.conf" "garage retrieval keyword" --n-predict 1 --temperature 0 --top-p 1 --min-p 0 --seed 1)

    expect_contains("${output}" "[debug] RAG mode enabled" "interactive RAG should remain available when embeddings are configured")
    expect_not_contains("${output}" "[rag] ${TEST_DIR}/rag/note.md" "startup should not learn files automatically")
    expect_not_contains("${output}" "→ " "startup should not list retrieved source files before any interactive learning")
elseif (CASE STREQUAL "long_output_context_shift")
    file(WRITE "${TEST_DIR}/yedera.conf"
        "prompt = \"You are a test assistant.\"\n"
        "model_path = \"${MODEL_PATH}\"\n"
        "n_gpu_layers = 0\n"
        "debug = false\n"
        "verbose = false\n")

    run_yedera(
        OUTPUT output
        STATUS status
        WORKING_DIRECTORY "${TEST_DIR}"
        ARGS --config "${TEST_DIR}/yedera.conf" --ctx-size 256 --n-predict 384 --temperature 0 --top-p 1 --min-p 0 --seed 1 "write a long sf story with spaceships")

    expect_not_contains("${output}" "context size exceeded" "long generation should shift context instead of failing")
elseif (CASE STREQUAL "unlimited_n_predict")
    file(WRITE "${TEST_DIR}/yedera.conf"
        "prompt = \"You are a test assistant.\"\n"
        "model_path = \"${MODEL_PATH}\"\n"
        "n_predict = -1\n"
        "n_gpu_layers = 0\n"
        "debug = false\n"
        "verbose = false\n")

    run_yedera(
        OUTPUT output
        STATUS status
        WORKING_DIRECTORY "${TEST_DIR}"
        ARGS --config "${TEST_DIR}/yedera.conf" --temperature 0 --top-p 1 --min-p 0 --seed 1 "Reply with one word.")

    expect_not_contains("${output}" "--n-predict must be positive" "n_predict = -1 should be accepted for unlimited generation")
elseif (CASE STREQUAL "learn_multi_entry_syntax")
    file(MAKE_DIRECTORY "${TEST_DIR}/rag")
    file(WRITE "${TEST_DIR}/rag/alpha.txt" "alpha learn token\n")
    file(WRITE "${TEST_DIR}/rag/file1.txt" "file1 learn token\n")
    file(WRITE "${TEST_DIR}/rag/file3.txt" "file3 learn token\n")
    write_test_pdf("${TEST_DIR}/rag/manual.pdf" "manual learn token")
    write_test_pdf("${TEST_DIR}/rag/file2.pdf" "file2 learn token")
    file(WRITE "${TEST_DIR}/yedera.conf"
        "prompt = \"You are a test assistant.\"\n"
        "model_path = \"${MODEL_PATH}\"\n"
        "model_embeddings = \"${MODEL_PATH}\"\n"
        "rag_documents_path = \"${TEST_DIR}/rag\"\n"
        "n_gpu_layers = 0\n"
        "debug = true\n"
        "verbose = false\n")
    file(WRITE "${TEST_DIR}/stdin.txt"
        "/learn *.pdf\n"
        "/learn \"*.pdf\"\n"
        "/learn \"*.txt\", *.pdf\n"
        "/learn \"*.txt\" \"*.pdf\"\n"
        "/learn file1.txt, file2.pdf, file3.txt, *.txt\n"
        "/learn \"file1.txt\", \"file2.pdf\", \"file3.txt\", *.txt\n"
        "/learn \"file1.txt\" \"file2.pdf\" \"file3.txt\" *.txt\n"
        "alpha learn token\n")

    run_yedera(
        OUTPUT output
        STATUS status
        WORKING_DIRECTORY "${TEST_DIR}/rag"
        INPUT_FILE "${TEST_DIR}/stdin.txt"
        ARGS --config "${TEST_DIR}/yedera.conf" --n-predict 8 --temperature 0 --top-p 1 --min-p 0 --seed 1)

    expect_not_contains("${output}" "error:" "multi-entry /learn syntax should not produce parsing or file errors")
    expect_not_contains("${output}" ".........." "learn mode should not emit llama.cpp dot progress while loading the embeddings model")
    expect_contains("${output}" "[rag] learned ${TEST_DIR}/rag/manual.pdf" "bare *.pdf should be accepted in learn mode")
    expect_contains("${output}" "[rag] learned ${TEST_DIR}/rag/file2.pdf" "quoted *.pdf and explicit pdf entries should be accepted in learn mode")
    expect_contains("${output}" "[rag] learned ${TEST_DIR}/rag/alpha.txt" "mixed quoted wildcard lists should accept txt patterns in learn mode")
    expect_contains("${output}" "[rag] learned ${TEST_DIR}/rag/file1.txt" "comma-separated explicit file names should be accepted in learn mode")
    expect_contains("${output}" "[rag] learned ${TEST_DIR}/rag/file3.txt" "space-separated quoted file names should be accepted in learn mode")
    expect_contains("${output}" "→ ${TEST_DIR}/rag/alpha.txt" "learn-mode multi-entry syntax should still list retrieved source files")
else()
    message(FATAL_ERROR "unknown CASE: ${CASE}")
endif()

file(REMOVE_RECURSE "${TEST_DIR}")

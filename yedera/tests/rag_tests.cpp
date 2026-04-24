#include "rag.hpp"

#include <chrono>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef YEDERA_BINARY_PATH
#error "YEDERA_BINARY_PATH must be defined"
#endif

namespace {

struct TempDirectory {
    TempDirectory() {
        char path_template[] = "/tmp/yedera-rag-tests-XXXXXX";
        char * created = mkdtemp(path_template);
        if (created == nullptr) {
            throw std::runtime_error("failed to create temp directory");
        }
        path = created;
    }

    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }

    std::filesystem::path path;
};

struct TestFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void expect_true(bool condition, const std::string & message) {
    if (!condition) {
        throw TestFailure(message);
    }
}

template <typename T>
void expect_equal(const T & actual, const T & expected, const std::string & message) {
    if (!(actual == expected)) {
        throw TestFailure(message);
    }
}

void expect_contains(const std::string & haystack, const std::string & needle, const std::string & message) {
    if (haystack.find(needle) == std::string::npos) {
        throw TestFailure(message + ": missing '" + needle + "'");
    }
}

void expect_not_contains(const std::string & haystack, const std::string & needle, const std::string & message) {
    if (haystack.find(needle) != std::string::npos) {
        throw TestFailure(message + ": unexpected '" + needle + "'");
    }
}

void write_file(const std::filesystem::path & path, const std::string & content) {
    std::ofstream output(path);
    if (!output.is_open()) {
        throw std::runtime_error("failed to write test file: " + path.string());
    }
    output << content;
}

void write_binary_file(const std::filesystem::path & path, const std::string & content) {
    std::ofstream output(path, std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("failed to write binary test file: " + path.string());
    }
    output.write(content.data(), static_cast<std::streamsize>(content.size()));
}

std::string shell_quote(std::string_view value) {
    std::string quoted = "'";
    for (const char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

std::string run_command(const std::string & command) {
    std::array<char, 4096> buffer{};
    std::string output;

    FILE * pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        throw std::runtime_error("failed to run command: " + command);
    }

    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }

    const int status = pclose(pipe);
    if (status != 0) {
        throw TestFailure(
            "command failed with status " + std::to_string(status) + ": " + command + "\nOutput:\n" + output);
    }

    return output;
}

std::string escape_pdf_text(std::string_view text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (const char ch : text) {
        if (ch == '\\' || ch == '(' || ch == ')') {
            escaped.push_back('\\');
        }
        escaped.push_back(ch);
    }
    return escaped;
}

void write_pdf_file(const std::filesystem::path & path, const std::vector<std::string> & lines) {
    std::string content_stream = "BT\n/F1 12 Tf\n72 720 Td\n";
    for (size_t index = 0; index < lines.size(); ++index) {
        if (index > 0) {
            content_stream += "0 -16 Td\n";
        }
        content_stream += "(" + escape_pdf_text(lines[index]) + ") Tj\n";
    }
    content_stream += "ET\n";

    const std::vector<std::string> objects = {
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        "<< /Length " + std::to_string(content_stream.size()) + " >>\nstream\n" + content_stream + "endstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    };

    std::string pdf = "%PDF-1.4\n";
    std::vector<size_t> offsets;
    offsets.reserve(objects.size() + 1);
    offsets.push_back(0);
    for (size_t index = 0; index < objects.size(); ++index) {
        offsets.push_back(pdf.size());
        pdf += std::to_string(index + 1) + " 0 obj\n" + objects[index] + "\nendobj\n";
    }

    const size_t xref_offset = pdf.size();
    pdf += "xref\n0 " + std::to_string(objects.size() + 1) + "\n";
    pdf += "0000000000 65535 f \n";
    for (size_t index = 1; index < offsets.size(); ++index) {
        char offset_buffer[32];
        std::snprintf(offset_buffer, sizeof(offset_buffer), "%010zu 00000 n \n", offsets[index]);
        pdf += offset_buffer;
    }
    pdf += "trailer\n<< /Size " + std::to_string(objects.size() + 1) + " /Root 1 0 R >>\n";
    pdf += "startxref\n" + std::to_string(xref_offset) + "\n%%EOF\n";

    write_binary_file(path, pdf);
}

void write_docx_file(const std::filesystem::path & path, const std::string & title, const std::string & body) {
    const std::filesystem::path source_path = path.parent_path() / (path.stem().string() + ".source.md");
    write_file(source_path, "# " + title + "\n\n" + body + "\n");
    run_command("pandoc -s " + shell_quote(source_path.string()) + " -o " + shell_quote(path.string()));
    std::error_code error;
    std::filesystem::remove(source_path, error);
}

void write_doc_file(const std::filesystem::path & path, const std::string & title, const std::string & body) {
    const std::filesystem::path source_path = path.parent_path() / (path.stem().string() + ".source.md");
    write_file(source_path, "# " + title + "\n\n" + body + "\n");
    run_command("pandoc -s -t rtf " + shell_quote(source_path.string()) + " -o " + shell_quote(path.string()));
    std::error_code error;
    std::filesystem::remove(source_path, error);
}

std::filesystem::path binary_path() {
    return std::filesystem::path(YEDERA_BINARY_PATH);
}

std::filesystem::path model_path() {
    return binary_path().parent_path() / "model" / "llama3.2-1b.gguf";
}

Options make_rag_options() {
    const std::filesystem::path embeddings_model = model_path();
    expect_true(std::filesystem::exists(embeddings_model), "RAG test embeddings model is missing");

    Options options;
    options.model_embeddings = embeddings_model.string();
    options.n_gpu_layers = 0;
    options.debug = false;
    return options;
}

std::string format_duration(double seconds) {
    const size_t rounded_seconds = static_cast<size_t>(seconds + 0.5);
    const size_t hours = rounded_seconds / 3600;
    const size_t minutes = (rounded_seconds % 3600) / 60;
    const size_t remaining_seconds = rounded_seconds % 60;

    if (hours > 0) {
        return std::to_string(hours) + "h " + std::to_string(minutes) + "m " + std::to_string(remaining_seconds) + "s";
    }

    if (minutes > 0) {
        return std::to_string(minutes) + "m " + std::to_string(remaining_seconds) + "s";
    }

    return std::to_string(remaining_seconds) + "s";
}

template <typename TestFunction>
void run_test_case(
    size_t index,
    size_t total,
    const char * name,
    const TestFunction & test_function,
    const std::chrono::steady_clock::time_point & suite_start) {
    std::cout << "[" << index << "/" << total << "] " << name << " ..." << std::endl;

    const auto test_start = std::chrono::steady_clock::now();
    try {
        test_function();
    } catch (const TestFailure & error) {
        throw TestFailure(std::string(name) + ": " + error.what());
    } catch (const std::exception & error) {
        throw TestFailure(std::string(name) + ": " + error.what());
    }

    const auto test_end = std::chrono::steady_clock::now();
    const double test_seconds = std::chrono::duration<double>(test_end - test_start).count();
    const double suite_seconds = std::chrono::duration<double>(test_end - suite_start).count();
    const double average_seconds = suite_seconds / static_cast<double>(index);
    const double eta_seconds = average_seconds * static_cast<double>(total - index);

    std::cout
        << "[" << index << "/" << total << "] " << name
        << " ok (" << format_duration(test_seconds) << ", eta " << format_duration(eta_seconds) << ")"
        << std::endl;
}

void test_learn_all_files_from_rag_folder() {
    const TempDirectory temp_dir;
    const std::filesystem::path rag_dir = temp_dir.path / "rag";
    std::filesystem::create_directories(rag_dir);
    const std::filesystem::path file_a = rag_dir / "porcelain.md";
    const std::filesystem::path file_b = rag_dir / "forklift.md";
    write_file(file_a, "porcelain thunder sentinel is stored in the saffron crate.\n");
    write_file(file_b, "forklift dusk window begins after moonrise.\n");

    RagStatePtr rag_state(nullptr, destroy_rag_state);
    const Options options = make_rag_options();
    replace_rag_sources(rag_state, options, {rag_dir});

    expect_equal(rag_chunk_count(rag_state.get()), static_cast<size_t>(2), "learning the rag folder should index both files");
    expect_equal(rag_chunk_count_for_source(rag_state.get(), file_a), static_cast<size_t>(1), "folder learning should keep one chunk for porcelain.md");
    expect_equal(rag_chunk_count_for_source(rag_state.get(), file_b), static_cast<size_t>(1), "folder learning should keep one chunk for forklift.md");
    expect_contains(
        augment_prompt_with_rag(rag_state.get(), "where is the porcelain thunder sentinel stored?"),
        "saffron crate",
        "folder learning should retrieve the porcelain file content");
    expect_contains(
        augment_prompt_with_rag(rag_state.get(), "when does forklift dusk window begin?"),
        "after moonrise",
        "folder learning should retrieve the forklift file content");
}

void test_learn_single_file_adds_only_target_file() {
    const TempDirectory temp_dir;
    const std::filesystem::path file_a = temp_dir.path / "porcelain.md";
    const std::filesystem::path file_b = temp_dir.path / "forklift.md";
    write_file(file_a, "porcelain thunder sentinel rests behind the basil tunnel.\n");
    write_file(file_b, "forklift dusk window begins after moonrise.\n");

    RagStatePtr rag_state(nullptr, destroy_rag_state);
    const Options options = make_rag_options();
    learn_rag_sources(rag_state, options, {file_a}, {});

    expect_equal(rag_chunk_count(rag_state.get()), static_cast<size_t>(1), "single-file learning should add only one file");
    expect_equal(rag_chunk_count_for_source(rag_state.get(), file_a), static_cast<size_t>(1), "single-file learning should index the requested file");
    expect_equal(rag_chunk_count_for_source(rag_state.get(), file_b), static_cast<size_t>(0), "single-file learning should leave unrelated files absent");
    expect_contains(
        augment_prompt_with_rag(rag_state.get(), "where does porcelain thunder sentinel rest?"),
        "basil tunnel",
        "single-file learning should retrieve the requested file content");
}

void test_learning_different_files_is_incremental() {
    const TempDirectory temp_dir;
    const std::filesystem::path file_a = temp_dir.path / "alpha.md";
    const std::filesystem::path file_b = temp_dir.path / "beta.md";
    write_file(file_a, "alpha brass marker stands by the west gate.\n");
    write_file(file_b, "beta copper marker stands by the east gate.\n");

    RagStatePtr rag_state(nullptr, destroy_rag_state);
    const Options options = make_rag_options();
    learn_rag_sources(rag_state, options, {file_a}, {});
    learn_rag_sources(rag_state, options, {file_b}, {});

    expect_equal(rag_chunk_count(rag_state.get()), static_cast<size_t>(2), "learning different files should remain incremental");
    expect_equal(rag_chunk_count_for_source(rag_state.get(), file_a), static_cast<size_t>(1), "incremental learning should preserve the first file");
    expect_equal(rag_chunk_count_for_source(rag_state.get(), file_b), static_cast<size_t>(1), "incremental learning should add the second file");
}

void test_relearn_same_file_replaces_changed_content() {
    const TempDirectory temp_dir;
    const std::filesystem::path file_a = temp_dir.path / "porcelain.md";
    write_file(file_a, "porcelain thunder sentinel uses the old latch code alpha-12.\n");

    RagStatePtr rag_state(nullptr, destroy_rag_state);
    const Options options = make_rag_options();
    learn_rag_sources(rag_state, options, {file_a}, {});
    expect_contains(
        augment_prompt_with_rag(rag_state.get(), "what latch code does porcelain thunder sentinel use?"),
        "alpha-12",
        "the initial file content should be retrievable");

    write_file(file_a, "porcelain thunder sentinel uses the new latch code beta-34.\n");
    learn_rag_sources(rag_state, options, {file_a}, {});

    const std::string prompt = augment_prompt_with_rag(rag_state.get(), "what latch code does porcelain thunder sentinel use?");
    expect_equal(rag_chunk_count(rag_state.get()), static_cast<size_t>(1), "relearning the same file should not duplicate chunks");
    expect_equal(rag_chunk_count_for_source(rag_state.get(), file_a), static_cast<size_t>(1), "relearning the same file should overwrite the previous file chunk");
    expect_contains(prompt, "beta-34", "relearning the same file should retrieve the updated content");
    expect_not_contains(prompt, "alpha-12", "relearning the same file should forget the previous content for that file");
}

void test_forget_single_file_removes_only_target_file() {
    const TempDirectory temp_dir;
    const std::filesystem::path file_a = temp_dir.path / "alpha.md";
    const std::filesystem::path file_b = temp_dir.path / "beta.md";
    write_file(file_a, "alpha brass marker stands by the west gate.\n");
    write_file(file_b, "beta copper marker stands by the east gate.\n");

    RagStatePtr rag_state(nullptr, destroy_rag_state);
    const Options options = make_rag_options();
    learn_rag_sources(rag_state, options, {file_a, file_b}, {});
    forget_rag_sources(rag_state, {file_a});

    expect_equal(rag_chunk_count(rag_state.get()), static_cast<size_t>(1), "forgetting one file should keep the remaining learned files");
    expect_equal(rag_chunk_count_for_source(rag_state.get(), file_a), static_cast<size_t>(0), "forgetting one file should remove that file's chunks");
    expect_equal(rag_chunk_count_for_source(rag_state.get(), file_b), static_cast<size_t>(1), "forgetting one file should preserve unrelated files");
}

void test_forget_all_clears_rag_state() {
    const TempDirectory temp_dir;
    const std::filesystem::path file_a = temp_dir.path / "alpha.md";
    const std::filesystem::path file_b = temp_dir.path / "beta.md";
    write_file(file_a, "alpha brass marker stands by the west gate.\n");
    write_file(file_b, "beta copper marker stands by the east gate.\n");

    RagStatePtr rag_state(nullptr, destroy_rag_state);
    const Options options = make_rag_options();
    learn_rag_sources(rag_state, options, {file_a, file_b}, {});
    clear_rag_sources(rag_state);

    const std::string question = "where does alpha brass marker stand?";
    expect_equal(rag_chunk_count(rag_state.get()), static_cast<size_t>(0), "forget all should clear every learned chunk");
    expect_equal(augment_prompt_with_rag(rag_state.get(), question), question, "forget all should leave no RAG context to inject into the prompt");
}

void test_forget_unknown_file_is_a_noop() {
    const TempDirectory temp_dir;
    const std::filesystem::path file_a = temp_dir.path / "alpha.md";
    const std::filesystem::path missing_file = temp_dir.path / "missing.md";
    write_file(file_a, "alpha brass marker stands by the west gate.\n");

    RagStatePtr rag_state(nullptr, destroy_rag_state);
    const Options options = make_rag_options();
    learn_rag_sources(rag_state, options, {file_a}, {});
    forget_rag_sources(rag_state, {missing_file});

    expect_equal(rag_chunk_count(rag_state.get()), static_cast<size_t>(1), "forgetting an unknown file should not disturb learned chunks");
    expect_equal(rag_chunk_count_for_source(rag_state.get(), file_a), static_cast<size_t>(1), "forgetting an unknown file should keep the known file intact");
}

void test_learn_pdf_doc_and_docx_files_extract_text() {
    const TempDirectory temp_dir;
    const std::filesystem::path rag_dir = temp_dir.path / "rag";
    std::filesystem::create_directories(rag_dir);

    const std::filesystem::path pdf_file = rag_dir / "manual.pdf";
    const std::filesystem::path doc_file = rag_dir / "notes.doc";
    const std::filesystem::path docx_file = rag_dir / "summary.docx";
    write_pdf_file(pdf_file, {"PDF sentinel location: basil tunnel crate 77.", "PDF sentinel code: amber-22."});
    write_doc_file(doc_file, "DOC Sentinel", "DOC sentinel latch code: cobalt-34.");
    write_docx_file(docx_file, "DOCX Sentinel", "DOCX sentinel corridor: copper tunnel.");

    RagStatePtr rag_state(nullptr, destroy_rag_state);
    const Options options = make_rag_options();
    replace_rag_sources(rag_state, options, {rag_dir});

    expect_true(rag_chunk_count_for_source(rag_state.get(), pdf_file) > 0, "PDF ingestion should index at least one chunk");
    expect_true(rag_chunk_count_for_source(rag_state.get(), doc_file) > 0, "DOC ingestion should index at least one chunk");
    expect_true(rag_chunk_count_for_source(rag_state.get(), docx_file) > 0, "DOCX ingestion should index at least one chunk");

    expect_contains(
        augment_prompt_with_rag(rag_state.get(), "what is the PDF sentinel location?"),
        "basil tunnel crate 77",
        "PDF ingestion should make extracted text retrievable");
    expect_contains(
        augment_prompt_with_rag(rag_state.get(), "what is the DOC sentinel latch code?"),
        "cobalt-34",
        "DOC ingestion should make extracted text retrievable");
    expect_contains(
        augment_prompt_with_rag(rag_state.get(), "what is the DOCX sentinel corridor?"),
        "copper tunnel",
        "DOCX ingestion should make extracted text retrievable");
}

} // namespace

int main() {
    try {
        const std::vector<std::pair<const char *, void (*)()>> tests = {
            {"learn all files from rag folder", test_learn_all_files_from_rag_folder},
            {"learn single file adds only target file", test_learn_single_file_adds_only_target_file},
            {"learning different files is incremental", test_learning_different_files_is_incremental},
            {"relearn same file replaces changed content", test_relearn_same_file_replaces_changed_content},
            {"forget single file removes only target file", test_forget_single_file_removes_only_target_file},
            {"forget all clears rag state", test_forget_all_clears_rag_state},
            {"forget unknown file is a noop", test_forget_unknown_file_is_a_noop},
            {"learn pdf doc and docx files extract text", test_learn_pdf_doc_and_docx_files_extract_text},
        };

        std::cout << "running " << tests.size() << " rag tests" << std::endl;
        const auto suite_start = std::chrono::steady_clock::now();
        for (size_t index = 0; index < tests.size(); ++index) {
            run_test_case(index + 1, tests.size(), tests[index].first, tests[index].second, suite_start);
        }

        const double total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - suite_start).count();
        std::cout << "rag tests passed in " << format_duration(total_seconds) << '\n';
        return 0;
    } catch (const TestFailure & error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    } catch (const std::exception & error) {
        std::cerr << "unexpected error: " << error.what() << '\n';
        return 1;
    }
}
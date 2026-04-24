# yedera

`yedera` is a small C++ command-line chat client with a vendored `llama.cpp` source tree under `vendor/llama.cpp`.

It follows the same low-level API flow as `llama.cpp/examples/simple-chat/simple-chat.cpp` and loads a local GGUF model file directly. It does not call Ollama or any remote or local inference server.

The project vendors `llama.cpp` and builds the embedded `llama.cpp` stack as static libraries. If the CUDA toolkit is available at configure time, `yedera` enables the NVIDIA backend automatically and offloads layers to CUDA by default. CPU-only builds keep the final `yedera` executable fully static; CUDA-enabled builds link against the NVIDIA runtime libraries available on the host.

Command line values override config values.

## Build

```bash
cmake -S yedera -B yedera/build
cmake --build yedera/build -j
```

On machines with NVIDIA CUDA installed, that build enables CUDA automatically. To force CPU-only execution at runtime, set `n_gpu_layers = 0` in `yedera.conf` or pass `--n-gpu-layers 0`.

## Install

```bash
cmake --install yedera/build --prefix ~/.local
```

That installs the binary to `~/.local/bin/yedera` and the default config to `~/.local/bin/yedera.conf`.

## Run

List downloaded model files from the `model/` directory next to the binary:

```bash
./yedera/build/yedera list
```

One-shot prompt using a direct GGUF model path:

```bash
./yedera/build/yedera --model-path /path/to/model.gguf "Say hello in five words."
```

Interactive chat using the model configured in `yedera.conf`:

```bash
./yedera/build/yedera
```

Inside interactive chat, use `/learn PATHS` to index one or more extra files or directories for the current session. Quoted entries, wildcards, and comma-separated lists are accepted, for example `/learn "*.txt", *.pdf` or `/learn "file1.txt" "file2.pdf"`. Re-learning the same file replaces that file's existing RAG chunks instead of appending duplicates. If you want to send a literal prompt that starts with `/`, prefix it with a second slash, for example `//learn this is part of the prompt`.

When the current entry is empty, pressing `Esc` toggles the prompt between normal chat mode `> ` and file-learn mode `: `. In `: ` mode, entering `PATHS` learns matching files without leaving the chat loop, and re-entering the same file replaces that file's previous RAG chunks. Quoted entries, wildcards, and comma-separated lists are accepted there too, for example `: "*.txt", *.pdf` or `: "file1.txt" "file2.pdf"`. Enter `*` to rebuild the session RAG state from every file under `rag_documents_path`, `-PATH` to forget one learned file, or `-` to forget all session RAG content. Relative entries resolve under `rag_documents_path`; absolute entries are used as entered. Each learned file prints its absolute path with RAG tuning progress, then the prompt returns to normal chat mode `> `. Press `Esc` again on an empty `: ` prompt to return to normal chat mode without learning.

Using config defaults instead of passing the model each time:

```bash
# edit ~/.local/bin/yedera.conf and set model_path to your local GGUF file
~/.local/bin/yedera "Summarize this project in one sentence."
```

Direct GGUF path override:

```bash
./yedera/build/yedera --model-path /path/to/model.gguf "Explain GGUF in one sentence."
```

Set `debug = true` in `yedera.conf` to print startup diagnostics to `stderr`, including CUDA detection and toolkit version, model loading progress, and GPU/CPU offload details.

For local RAG, point `model_embeddings` at a local GGUF embeddings model and `rag_documents_path` at a file or directory of text documents. Documents are learned only during the session through `/learn` or `Esc` file-learn mode; yedera does not index RAG documents at startup. Relative paths in `yedera.conf` resolve from the config file directory, so `model_embeddings = "model/...gguf"` and `rag_documents_path = "rag"` work next to the binary. PDF ingestion uses embedded text first through `pdftotext`; when no embedded text is found, it falls back to OCR with `pdftoppm` and `tesseract`.

## Options

Command:

- `list`: list regular files found under the `model/` directory as absolute paths

- `-m`, `--model-path`: path to a local GGUF model file
- `--config`: config file path override. Default lookup is `./yedera.conf`, then `~/.yedera/yedera.conf`
- `[prompt]`: run one inference and exit
- if no prompt is provided: start a chat loop
- `-p`, `--prompt`: explicit one-shot prompt flag, equivalent to passing `[prompt]`
- `-i`, `--interactive`: force chat mode even when a prompt is configured elsewhere
- interactive chat command `/learn PATHS`: learn one or more files or directories for the current session without leaving the chat loop; quoted entries, wildcards, and comma-separated lists are accepted
- interactive chat escape `//TEXT`: send a literal prompt that starts with `/`
- interactive chat key `Esc` on an empty prompt: toggle between `> ` chat mode and `: ` file-learn mode; relative entries in file-learn mode resolve under `rag_documents_path`, quoted entries, wildcards, and comma-separated lists are accepted, `*` rebuilds from `rag_documents_path`, `-PATH` forgets one learned file, and `-` forgets all learned RAG content
- `-s`, `--system-prompt`: override the assistant system prompt from config
- `-c`, `--ctx-size`: context size, default `2048`
- `-n`, `--n-predict`: max generated tokens per response, default `256`
- `--temperature`: sampling temperature, default `0.8`
- `--top-p`: nucleus sampling value, default `0.95`
- `--min-p`: minimum-p filter value, default `0.05`
- `--seed`: integer seed or `random`
- `-ngl`, `--n-gpu-layers`: GPU offload layers. If omitted, `yedera` offloads all layers when a non-CPU backend device is available and otherwise falls back to CPU-only. Set `0` to force CPU-only execution.
- `-v`, `--verbose`: keep `llama.cpp` logging enabled

## Config File

Example config:

```ini
prompt = "You are Yedera, a helpful local assistant running on llama.cpp."
model_path = "/absolute/path/to/model.gguf"
temperature = 0.7
top_p = 0.9
seed = random
```

If `n_gpu_layers` is omitted, `yedera` automatically uses NVIDIA GPU offload when the build includes CUDA and a GPU is visible. Set `n_gpu_layers = 0` to force CPU-only mode or a positive integer to cap offloaded layers.

Supported keys are `prompt`, `model_path`, `model_embeddings`, `rag_documents_path`, `user_prompt`, `ctx_size`, `n_predict`, `n_gpu_layers`, `temperature`, `top_p`, `min_p`, `seed`, `interactive`, `verbose`, and `debug`. `system_prompt` is still accepted as a backward-compatible alias for `prompt`.

When `model_path` or `model_embeddings` points at a built-in alias under `model/` and the file is missing, `yedera` downloads the GGUF automatically at startup into the configured path. When `model_embeddings` and `rag_documents_path` are configured, `/learn` and `Esc` file-learn mode can index documents for the current session and prepend the most relevant chunks to each user turn.

The assistant prompt now lives directly in [yedera.conf](yedera.conf) as `prompt = "..."`.

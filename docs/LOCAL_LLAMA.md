# Running MGE_GOAT on your GPU (local llama.cpp)

MGE_GOAT can route work to a local [llama.cpp](https://github.com/ggml-org/llama.cpp)
`llama-server` instead of remote APIs. These steps are verified for an
**RTX 3050 (8 GB, Ampere / sm_86)** on Linux; adjust the arch for other cards.

> GPU-aware routing: `mge gpu` shows detected VRAM. A `[models.local]` route with
> `min_free_vram_mb` is only used when the GPU has at least that much free —
> otherwise MGE_GOAT routes to remote APIs. If the server is simply down, the
> normal fallback chain handles it (the local attempt fails to connect and we
> fall through to the next route).

## 1. Build llama.cpp with CUDA

```bash
# Prereqs: NVIDIA driver + CUDA toolkit (nvcc on PATH)
nvidia-smi --query-gpu=compute_cap --format=csv   # should print 8.6 for RTX 3050
nvcc --version                                     # must succeed

git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp

# NOTE: LLAMA_CUBLAS is REMOVED — using it hard-errors. Use GGML_CUDA.
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release -j"$(nproc)"   # binaries in build/bin/
```

Optional flags: `-DGGML_CUDA_F16=ON` (~15–20% faster on Ampere),
`-DLLAMA_CURL=ON` (enables `-hf` auto-download).
If you change flags/arch later, `rm -rf build` first (stale CUDA cache causes
link failures). If a run logs `offloaded 0/N layers`, CUDA didn't link — rebuild.

## 2. Get a model that fits 8 GB **and** does tool calling

Recommended for the agentic loop: **Qwen3-4B-Instruct-2507, Q4_K_M (~2.5 GB)**.

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF \
  Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf --local-dir ./models
```

Alternatives: `Qwen2.5-Coder-7B-Instruct` Q4_K_M (~4.7 GB) is the strongest pure
coder but its tool calling is unreliable — use it only for code-gen, not the
agentic loop. `IBM Granite-4.0-H-Micro` (3B, ~1.9 GB) is the smallest dependable
tool-calling fallback.

## 3. Start the server

```bash
./build/bin/llama-server \
  -m models/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 99 -fa --jinja -c 16384 \
  --host 127.0.0.1 --port 8080 --alias local
```

- `-ngl 99` full GPU offload · `-fa` flash attention (OK on Ampere)
- **`--jinja` is REQUIRED** — without it, a request carrying `tools[]` returns HTTP 500
- `-c 16384` caps KV-cache VRAM · `--alias local` is the id reported by `/v1/models`
  (must match `[models.local].model` below)

Readiness check: `curl http://127.0.0.1:8080/health` → `{"status":"ok"}` when loaded.

## 4. Point MGE_GOAT at it

In `~/.config/mge/config.toml`, the `local` provider already exists. Add a route:

```toml
[models.local]
provider = "local"
model = "local"            # must match llama-server --alias
min_free_vram_mb = 4000    # ~2.5GB model + headroom; skip local if less is free
```

Use it directly with `mge chat --route local` / `mge tui --route local`, or add
`"local"` to the front of another route's `fallback`/make it the `default_route`
so MGE_GOAT prefers the GPU when it's available and falls back to remote when not.

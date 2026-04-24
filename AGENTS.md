# LLM Model Manager — Agent Notes

## Architecture
- **Single-file app**: `app.py` (FastAPI + vanilla HTML frontend). No test/lint/typecheck config.
- **Two-container deployment**: `llm-model-manager` (API/UI) + `llama-swap` (llama-server proxy).
- **State**: `/models/served/state.json` + `config.yaml`. **Cache**: `/models/.cache` (HF cache).
- **Frontend CSS**: compiled from `input.css` → `static/output.css` via Tailwind standalone CLI (no Node.js). Rebuild: `./tailwindcss-linux-x64 -i ./input.css -o ./static/output.css --minify`.

## Running
```bash
export MODELS=/path/to/models HF_TOKEN=hf_...
# Default backend is CUDA
./up.sh
# Or pick a backend explicitly
LLAMA_BACKEND=vulkan ./up.sh
LLAMA_BACKEND=sycl ./up.sh
LLAMA_BACKEND=openvino ./up.sh
```

- `up.sh` uses `LLAMA_BACKEND` env var to pick the compose override: `cuda`, `vulkan`, `sycl`, or `openvino`.
- Default compose file: `docker-compose.yml` + chosen override. llama-swap Dockerfile is set via `LLAMA_DOCKERFILE` env var (default: `Dockerfile.llama.cuda`).
- **Do not** run `docker compose up` directly — always use `up.sh` so the backend flag is applied consistently.
- **`up.sh` bugs**: SYCL override path is `compose.sycl.yml` (missing `docker-` prefix). `openvino` backend falls through to the error case — no case arm for it despite `docker-compose.openvino.yml` existing.

## Key paths & env
| Variable | Purpose |
|---|---|
| `MODELS` | Host volume mounted as `/models` inside containers |
| `HF_TOKEN` | HuggingFace token for gated models |
| `LOG_LEVEL` | Python log level (default: INFO) |
| `LLAMA_BACKEND` | GPU backend: `cuda`, `vulkan`, `sycl`, `openvino` |
| `LLAMA_DOCKERFILE` | llama-swap Dockerfile path (default: `Dockerfile.llama.cuda`) |
| `LLAMA_SWAP_CONTAINER` | llama-swap container name (default: `llama-swap`) |

## Gotchas
- **Model downloads use PTY** (`pty.openpty()`) to capture `tqdm` progress. Don't change the subprocess env.
- **Revision pinning**: state stores resolved commit SHAs, not branches. Upstream `main` advances don't break pinned models. After a download completes, `refs/main` is updated to point to the downloaded SHA so `scan_cache_dir` can find it.
- **Auto-prune**: on startup and after config deletion, unreferenced cache revisions are deleted automatically.
- **RPC mode**: toggle via `/api/rpc_mode` — only models with matching `rpc` param are synced.
- **llama-swap restarts** happen asynchronously after every config change (in a background thread).
- **Quant regex** (`QUANT_REGEX`) drives file matching for GGUF variants and mmproj files. Changing it requires cache re-scan.
- **No tests or linting** exist. Any changes to `app.py` should be verified by reading the code carefully — it's the only source file.

## llama-swap backends
- `Dockerfile.llama.cuda` — NVIDIA CUDA (default)
- `Dockerfile.llama.vulkan` — Vulkan
- `Dockerfile.llama.sycl` — Intel SYCL
- `Dockerfile.llama.openvino` — OpenVINO

Each backend has a matching `docker-compose.<backend>.yml` override for GPU device reservations.

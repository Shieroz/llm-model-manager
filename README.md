# LLM Model Manager

Manage HuggingFace GGUF models with a single-panel UI. Downloads, pins revisions, and auto-generates llama-swap configs.

## Quick Start

```bash
export MODELS=/path/to/models HF_TOKEN=hf_...
./up.sh
```

Open `http://localhost:8000` → deploy a model → it appears at `http://localhost:8080` (llama-swap API).

## Backends

```bash
LLAMA_BACKEND=vulkan ./up.sh   # AMD/Intel Vulkan
LLAMA_BACKEND=sycl ./up.sh     # Intel SYCL
LLAMA_BACKEND=openvino ./up.sh # Intel OpenVINO
```

## How It Works

Two containers share a volume at `/models`:

| Container | Port | Role |
|---|---|---|
| `llm-model-manager` | 8000 | UI + API — downloads GGUF from HF, writes `config.yaml` |
| `llama-swap` | 8080 | llama-server proxy — reads `config.yaml`, serves models |

State is persisted in `/models/served/state.json`. Model cache lives at `/models/.cache` (HF cache).

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MODELS` | *(required)* | Host path mounted as `/models` |
| `HF_TOKEN` | *(required)* | HuggingFace token for gated models |
| `LOG_LEVEL` | `INFO` | Python log level |
| `LLAMA_BACKEND` | `cuda` | GPU backend: `cuda`, `vulkan`, `sycl`, `openvino` |
| `LLAMA_DOCKERFILE` | `Dockerfile.llama.cuda` | llama-swap Dockerfile to use |

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Frontend UI |
| `GET` | `/api/models` | List cached models & revisions |
| `GET` | `/api/quants?repo=...` | Available quant variants for a HF repo |
| `GET` | `/api/commits?repo=...` | Commit history with pin status |
| `POST` | `/api/setup` | Deploy a model (downloads if needed) |
| `DELETE` | `/api/configs/{name}` | Remove a config (auto-prunes cache) |
| `POST` | `/api/revisions/delete` | Delete a cached revision |
| `POST` | `/api/rpc_mode` | Toggle RPC mode for combining GPU power |
| `WS` | `/ws` | Real-time state updates |

## CSS (no Node.js)

```bash
./tailwindcss-linux-x64 -i ./input.css -o ./static/output.css --minify
```

## Development

Single-file app (`app.py`). No tests, no linting. Verify changes by reading the code.

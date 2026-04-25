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

Modular Python backend (`src/backend/`) with vanilla HTML/CSS/JS frontend (`src/frontend/`).

### Tests

127 tests across 9 test files. Run locally without Docker:

```bash
./up.sh test
```

Covers: backend (config, models, state, cache, hf_hub, sync, websocket, api) + frontend (HTML structure, form elements, API routes, styling, layout, WebSocket).

### Frontend Structure

The frontend SPA has been modularized from a single ~700-line inline script into 9 separate JS modules:

| Module | Purpose |
|---|---|
| `state.js` | Shared application state |
| `utils.js` | DOM helpers, debounce, JSON validation |
| `api.js` | API communication functions |
| `filter.js` | Client-side search/filter |
| `localmodels.js` | Disk storage data fetching |
| `render.js` | UI rendering (cards, progress bars) |
| `ws.js` | WebSocket connection management |
| `form.js` | Form handling and validation |
| `app.js` | Main entry point, wires modules together |

### Source Structure

| File | Purpose |
|---|---|
| `src/backend/app.py` | FastAPI setup, lifespan, route definitions |
| `src/backend/api.py` | All route handlers |
| `src/backend/config.py` | Constants, paths, regex patterns |
| `src/backend/models.py` | Pydantic request/response models |
| `src/backend/state.py` | State file read/write |
| `src/backend/cache.py` | HF cache scanning and pruning |
| `src/backend/hf_hub.py` | HuggingFace API calls |
| `src/backend/sync.py` | llama-swap config sync |
| `src/backend/download.py` | PTY-based model downloads |
| `src/backend/websocket.py` | WebSocket connection manager |
| `src/frontend/index.html` | SPA HTML (forms, layout, styling) |
| `src/frontend/js/*.js` | 9 modular JavaScript files |

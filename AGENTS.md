# LLM Model Manager — Agent Notes

## Architecture
- **Modular app**: `src/backend/` (FastAPI) + `src/frontend/` (vanilla HTML/CSS/JS). No lint/typecheck config.
- **Test suite**: 139 tests across 9 test files — all passing. Run with `./dev.sh test` (no Docker).
- **Two-container deployment**: `llm-model-manager` (API/UI) + `llama-swap` (llama-server proxy).
- **State**: `/models/served/state.json` + `config.yaml`. **Cache**: `/models/.cache` (HF cache).
- **Frontend CSS**: compiled from `input.css` → `static/output.css` via Tailwind standalone CLI (no Node.js). Rebuild: `./tailwindcss-linux-x64 -i ./input.css -o ./static/output.css --minify`.
- **Source layout**:
  - `src/backend/app.py` — FastAPI setup, lifespan, route definitions
  - `src/backend/api.py` — all route handlers (quants, commits, models, rpc, revisions, setup, configs)
  - `src/backend/config.py` — constants, paths, QUANT_REGEX, MTP_HEAD_HINTS/is_mtp_head_file, LLAMA_SHORT_FLAGS
  - `src/backend/models.py` — Pydantic models (ModelSetup incl. mtp_head, RevisionDeleteReq, RpcModeReq)
  - `src/backend/state.py` — load_state, save_state (atomic write), iter_configs, format_bytes
  - `src/backend/cache.py` — scan_cache, in_use_revisions, prune_unreferenced_revisions/files
  - `src/backend/hf_hub.py` — resolve_sha, get_commits, pre_flight_size (incl. head)
  - `src/backend/sync.py` — sync_system (mmproj + MTP head symlinks, --model-draft), restart_llama_swap
  - `src/backend/download.py` — process_model (disk-based progress, stall watchdog, resume)
  - `src/backend/websocket.py` — ConnectionManager, manager singleton
  - `src/backend/tests/` — 106 pytest tests
  - `src/frontend/index.html` — SPA HTML (forms, layout, styling)
  - `src/frontend/js/*.js` — 9 modular JS files (state, utils, api, filter, localmodels, render, ws, form, app)
  - `src/frontend/tests/` — 33 pytest E2E tests (HTML structure, form, API, styling, layout)

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

- `up.sh` uses `LLAMA_BACKEND` env var to pick the compose override: `cuda`, `vulkan`, `sycl`.
- Default compose file: `docker-compose.yml` + chosen override. llama-swap Dockerfile is set via `LLAMA_DOCKERFILE` env var (default: `Dockerfile.llama.cuda`).
- **Do not** run `docker compose up` directly — always use `up.sh` so the backend flag is applied consistently.
- **`up.sh` bugs**: SYCL override path is `compose.sycl.yml` (missing `docker-` prefix). `openvino` backend falls through to the error case — no case arm for it despite `docker-compose.openvino.yml` existing.
- **Tests**: `./dev.sh test` runs pytest locally (no Docker). 139 tests: 106 backend (config, models, state, cache, hf_hub, sync, websocket, api) + 33 frontend E2E (HTML structure, form elements, API routes, styling, layout, WebSocket).
- **Dev mode**: `./dev.sh dev` starts the container with selective source mounts via `docker-compose.dev.yml` (preserves compiled CSS, adds `--reload`). Backend path fixed: `COPY src/backend/ ./backend` (was overwriting `/app/`).

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
- **Download progress is computed from bytes-on-disk**, not stdout. huggingface_hub (≥1.x) no longer emits parseable `tqdm` byte counts, so `process_model` polls the repo's `blobs/` dir each second and reports **bytes fetched this session ÷ bytes still needed** (`expected_total = dl_total`, recorded at setup via `pre_flight_size`). A PTY (`pty.openpty()`) is still opened, but only drained in a thread to keep the child unblocked + capture errors.
- **Download reliability**: `HF_HUB_DOWNLOAD_TIMEOUT=30` makes a dead-but-open LFS socket raise (then huggingface_hub resumes the file from its `.incomplete`); a 300 s **stall watchdog** in `process_model` aborts with status `error` if zero bytes flow, so a config never sticks at `downloading`. `hf_xet` (in requirements) accelerates Xet-backed repos transparently; LFS repos use the normal path. The old `HF_HUB_ENABLE_HF_TRANSFER` is deprecated/ignored.
- **`save_state` is atomic** (temp file + `os.replace`). Required because the 1 Hz progress writer and the 1 Hz WebSocket broadcaster's reader race — a non-atomic rewrite caused torn reads → empty config list → UI flicker.
- **File-glob precision**: download patterns anchor the tag to a separator (`*mmproj*[-._]F16.gguf`) so `F16` doesn't also pull `BF16`. `pre_flight_size`/`sync` already use `QUANT_REGEX` (which requires a `[-._]` prefix), so keep download globs consistent or sizes won't match.
- **Revision pinning**: state stores resolved commit SHAs, not branches. Upstream `main` advances don't break pinned models. After a download completes, `refs/main` is updated to point to the downloaded SHA so `scan_cache_dir` can find it.
- **Auto-prune**: on startup and after config deletion, unreferenced cache revisions are deleted automatically. `prune_unreferenced_files` also removes orphan GGUF variants (unused quant/mmproj) inside still-referenced revisions; MTP heads are protected by `is_mtp_head_file` (name hint + <4 GB size cap).
- **RPC mode**: toggle via `/api/rpc_mode` — only models with matching `rpc` param are synced.
- **llama-swap restarts** happen asynchronously after every config change (in a background thread).
- **Quant regex** (`QUANT_REGEX`) drives file matching for GGUF variants and mmproj files. Changing it requires cache re-scan.
- **Tests mock module-level imports**: fixtures patch `backend.state`, `backend.cache`, `backend.sync`, `backend.config` attributes. API tests patch `backend.api.HfApi` (where it's imported in-function) and `backend.api.sync_system` (to avoid deep `os.path` mocking).
- **Sync tests** create mock cache objects matching `scan_cache_dir` return structure: `repos` → `repo_id`, `revisions` → `commit_hash`, `files` → `file_name`/`file_path`/`size_on_disk`.
- **Frontend tests** use httpx + BeautifulSoup (no browser required). A test server fixture starts an isolated backend with patched paths.

## RPC Mode

Toggle via `/api/rpc_mode` — only models with `"rpc": true` in their params are synced when RPC mode is active. Filtered view in the UI shows RPC or Local configs based on the toggle state.

## MTP (Multi-Token Prediction)

Speculative decoding via llama.cpp's `--spec-type draft-mtp`. Three distribution patterns, handled simply:

1. **Grafted** (head baked into the quant, e.g. `unsloth/Qwen3.6-27B-MTP-GGUF`): the deploy form detects an `mtp` token in the repo name and auto-injects `spec-type: draft-mtp` + `spec-draft-n-max` into the params JSON. Delete them there to disable.
2. **Separate head, same repo** (e.g. gemma-4's `mtp-gemma-4-31B-it.gguf`): `/api/quants` returns a `heads` list (detected by `is_mtp_head_file`, excluded from quants). The form shows a draft-head dropdown (mmproj-style). On deploy, `mtp_head` (rfilename) is stored in state; `download.py` fetches it, `sync.py` symlinks it to `{name}-mtp-head.gguf` and passes `--model-draft` (+ `--spec-type draft-mtp` if not already in params, skipping any `model-draft`/`md` in params).
3. **Separate head, different repo**: not managed. Users deploy the head repo as its own config and copy its served symlink path (shown on each Disk Storage config card) into the main model's `model-draft` param by hand.

Notes: `mtp_head` only forces a re-download when it's a *new non-empty* head (removing one just drops `--model-draft` on the next sync). The grafted heuristic only gates auto-injection of flags; the head dropdown is driven by real file detection.

## Frontend Module Structure
The inline JS (~700 lines) has been modularized into 9 files loaded via `<script>` tags:
- `state.js` — Shared application state (configs, models, expanded sets, RPC mode)
- `utils.js` — DOM helpers, debounce, JSON validation, status display
- `api.js` — API communication (rpcMode, fetchQuants, fetchCommits, setupConfig, deleteConfig, deleteRevision)
- `filter.js` — Client-side search/filter for configs and storage
- `localmodels.js` — Local models / disk storage data fetching
- `render.js` — UI rendering (config cards incl. served symlink paths, storage cards, progress bars, quant badges)
- `ws.js` — WebSocket connection with auto-reconnect
- `form.js` — Form handling, JSON validation, mmproj + MTP draft-head wiring (grafted auto-inject / head dropdown)
- `app.js` — Main entry point, wires all modules, exposes functions for inline HTML handlers

## llama-swap backends
- `Dockerfile.llama.cuda` — NVIDIA CUDA (default)
- `Dockerfile.llama.vulkan` — Vulkan
- `Dockerfile.llama.sycl` — Intel SYCL
- `Dockerfile.llama.openvino` — OpenVINO

Each backend has a matching `docker-compose.<backend>.yml` override for GPU device reservations.

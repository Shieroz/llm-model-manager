# Multi-Instance llama-swap — Feature Plan

> **Branch:** `feature/multi-instance-llama-swap`
> **Goal:** Allow users to configure, manage, and serve models across multiple concurrent llama-swap containers, each with its own backend, GPU, and port. Architecture the system so the frontend becomes a one-stop suite for all model and server management.

---

## Table of Contents

1. [Direct Path Migration (Phase 0)](#1-direct-path-migration-phase-0)
2. [Config-Driven Instance Management](#2-config-driven-instance-management)
3. [Container Lifecycle & Image Management](#3-container-lifecycle--image-management)
4. [Per-Instance Model Assignment](#4-per-instance-model-assignment)
5. [Backend Architecture](#5-backend-architecture)
6. [Frontend Architecture](#6-frontend-architecture)
7. [Implementation Phases](#7-implementation-phases)
8. [Future Roadmap](#8-future-roadmap)

---

## 1. Direct Path Migration (Phase 0)

> **Priority:** First — replaces the symlink layer before any multi-instance work.
> **Scope:** `sync.py`, `config.py`, `test_sync.py`

### 1.1 Current Symlink Chain

```
/models/served/<name>-Q4K_M.gguf  →  /models/.cache/repos/.../snapshots/<sha>/<file>.gguf  →  /models/.cache/.../blobs/<hash>
```

The current approach creates symlinks in `/models/served/` pointing to HF Hub's snapshot directories. This adds complexity:
- Symlink creation/removal on every sync
- `FileExistsError` handling
- Sharded model symlink logic (naming, first-shard tracking)
- mmproj symlink logic
- Potential for broken symlinks if cache structure changes

### 1.2 huggingface_hub Library Support

> `huggingface_hub` already provides exactly what we need — no path construction required.

- `scan_cache_dir()` returns `CachedFileInfo` objects with `file_path` pointing directly to the snapshot directory (e.g., `/models/.cache/repos/hf/unsloth/Llama-3.1-8B/snapshots/abc123/model-Q4K_M.gguf`)
- `try_to_load_from_cache(repo_id, filename, revision)` can resolve a single file path without a full cache scan (future optimization)
- `scan_cache_dir()` is still needed for sharded models and the Disk Storage UI listing (needs all files in a revision)

### 1.3 New: Direct Cache Paths

HF Hub's snapshot directories use deterministic paths:
```
/models/.cache/repos/hf/<namespace>/<repo>/snapshots/<commit_sha>/<file>.gguf
```

Since `commit_sha` is fixed per revision, the path is predictable and stable. llama.cpp accepts file paths directly — no symlink layer needed.

**New approach:** Use `cf.file_path` (from `huggingface_hub.scan_cache_dir`) directly as the `--model` argument.

### 1.4 Changes in `sync.py`

| Current (symlinks) | New (direct paths) |
|---|---|
| `os.symlink(cf.file_path, symlink_path)` | Use `cf.file_path` directly |
| `glob.glob(served/*.gguf)` → unlink cleanup | No cleanup needed |
| `FileExistsError` handling on symlink | No conflict handling |
| Sharded: create `<name>-Q4K_M-00001-of-00010.gguf` symlink | Use first shard's `cf.file_path` directly |
| mmproj: create `<name>-mmproj-FP16.gguf` symlink | Use mmproj `cf.file_path` directly |
| `SERVED_DIR` used for symlink targets | `SERVED_DIR` removed entirely |

**Sharded models:** llama-server accepts the first shard path and auto-discovers remaining shards in the same directory. No special handling needed — just pass the first shard's `cf.file_path`.

**mmproj files:** Pass the mmproj file's `cf.file_path` directly to `--mmproj`.

### 1.5 Why This Is Better

- **Deterministic:** `<commit_sha>` guarantees exact files — no ambiguity
- **No race conditions:** llama-server starts with valid paths immediately
- **No broken symlinks:** cache pruning doesn't affect served configs
- **Simpler code:** removes ~30 lines of symlink management logic
- **Fewer edge cases:** no `FileExistsError`, no cleanup loops, no symlink chains
- **Library-supported:** `cf.file_path` is already the direct snapshot path — no construction needed

### 1.6 What Stays the Same

- The llama-swap `config.yaml` is still generated (per-instance in multi-instance mode)
- Models are still grouped by `instance_setup`
- The external API (state.json, config.yaml format, llama-swap interaction) remains unchanged
- Existing state files don't need migration

### 1.7 Phase 0 Checklist

- [ ] Use `cf.file_path` directly (already provided by `huggingface_hub.CachedFileInfo`) — no path construction needed
- [ ] Remove symlink creation in `sync_system()` — use `cf.file_path` directly as `--model` path
- [ ] Remove symlink cleanup (`glob.glob` + `os.unlink`)
- [ ] Remove `FileExistsError` handling
- [ ] Simplify sharded model handling — first shard's `cf.file_path` is the entry point
- [ ] Simplify mmproj handling — pass mmproj `cf.file_path` directly to `--mmproj`
- [ ] Remove `SERVED_DIR` from `config.py` (no longer needed)
- [ ] Update `sync_system()` — verify signature stays the same (`sync_system(state, restart)`)
- [ ] Update tests: verify `config.yaml` contains direct cache paths instead of symlink paths
- [ ] Run full test suite — all 127 tests passing

---

## 2. Config-Driven Instance Management

### 2.1 Config File

**Path:** `/models/config.yml` (priority) or `/models/config.yaml`
**Template:** `config.example.yml` (git-tracked, clearly marked as example)
**User files:** `.gitignore`d

```yaml
# /models/config.yml
hf_token: ""  # fallback if not in .env; .env always takes priority

instances:
  - name: cuda-0
    dockerfile: /path/to/Dockerfile.llama.cuda
    devices:
      - type: nvidia
        count: all
    port: 8080

  - name: vulkan-0
    dockerfile: /home/user/custom/Dockerfile.llama.vulkan
    devices:
      - /dev/dri/renderD128
      - /dev/dri/card0
    port: 8081
```

**Field rules:**
| Field | Required | Type | Description |
|---|---|---|---|
| `name` | No | `string` | Container name / instance key. Auto-generated from `dockerfile` path if omitted. |
| `dockerfile` | **Yes** | `string` | Absolute path to Dockerfile on host. No defaults — user must opt in. |
| `devices` | No | `list` | GPU device specification (see 2.2). If omitted, no GPU access. |
| `port` | No | `int` | HTTP port for llama-server. Auto-incremented from 8080 if omitted. |

**GPU device spec formats:**

```yaml
# NVIDIA (uses Docker device_requests with nvidia driver)
- type: nvidia
  count: all        # or [0, 1, 2] for specific GPUs

# Direct device passthrough (AMD Vulkan, Intel SYCL, OpenVINO)
- /dev/dri/renderD128
- /dev/dri/card0
```

**Config discovery:** `/models/config.yml` → `/models/config.yaml` → **empty** (no default instance)

> **Rationale:** `dockerfile` is required. Users must explicitly configure instances. This avoids wasting time building images for unsupported hardware. If no config exists, the manager runs with zero instances — the UI/API still functions, but model cards show a warning that no instance is available.

### 2.2 Config Reload

`POST /api/instances/reload` — hot-reload without restarting the manager:

1. Read `config.yml` from disk
2. Diff against current running instances
3. **New instances:** build image → create container → start
4. **Removed instances:** stop container → remove
5. **Changed `dockerfile` path:** stop → remove old → rebuild → create → start
6. **Changed `devices` or `port`:** stop → recreate with new settings
7. **Changed `name`:** stop old → create new with new name
8. Re-sync models to all affected instances
9. Broadcast updated instance list via WebSocket

---

## 3. Container Lifecycle & Image Management

### 3.1 Docker SDK Integration

The manager uses the `docker` Python SDK (already a dependency) to manage containers. No more docker-compose overrides for llama-swap.

**Current behavior:** `up.sh` uses docker-compose with backend-specific override files.
**New behavior:** Manager controls container lifecycle directly. `up.sh` only starts the manager container.

### 3.2 Container Naming & Image Tags

| Container | Name | Image Tag |
|---|---|---|
| Model Manager | `llm-model-manager` | `local/llm-model-manager:latest` |
| llama-swap instance 0 | `<name>` from config (default: `llama-swap-cuda-0`) | `local/llama-swap-<name>:latest` |

### 3.3 Volume Layout

```
/models/
  config.yml              ← instance config (gitignored)
  config.example.yml      ← template (git-tracked)
  state.json              ← model state
  instances/
    cuda-0/
      config.yaml         ← per-instance llama-swap config
    vulkan-0/
      config.yaml
  .cache/                 ← HF cache (direct paths point here)
```

> **Note:** `served/` directory is removed entirely in Phase 0. No symlinks needed — `cf.file_path` from `scan_cache_dir()` points directly to HF snapshot directories. Per-instance llama-swap configs live under `/models/instances/<name>/config.yaml`.

### 3.4 Image Management (Future)

> **TODO:** Implement image management UI and API.

**Why:** llama.cpp Docker images are huge (several GB each). Users need to:
- See image sizes across instances
- Delete unused images to free disk space
- Choose image tags (e.g., specific llama.cpp versions)
- Understand which images are in use by running containers

**Future API endpoints:**
- `GET /api/images` — list local images with size, tags, last used
- `DELETE /api/images/<id>` — prune unused images
- `POST /api/instances/<name>/rebuild` — force rebuild of one instance

**Future UI:** "Images" section in sidebar showing local Docker images, sizes, and which instances use them.

---

## 4. Per-Instance Model Assignment

### 4.1 State File Changes

Each model config in `state.json` gets an `instance_setup` field:

```json
{
  "_meta": {"rpc_mode": false},
  "llama3-8b": {
    "repo": "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
    "quant": "Q4_K_M",
    "instance_setup": "cuda-0",
    "params": {"ctx_size": 8192},
    "status": "ready",
    "revision": "abc123def456"
  },
  "llama3-70b": {
    "repo": "unsloth/Llama-3.1-70B-Instruct-bnb-4bit",
    "quant": "Q4_K_M",
    "instance_setup": ["cuda-0", "cuda-1"],
    "params": {"ctx_size": 8192},
    "status": "ready",
    "revision": "def789ghi012"
  }
}
```

**Assignment types:**
| Value | Behavior |
|---|---|
| `"cuda-0"` | Model served only by cuda-0 |
| `["cuda-0", "cuda-1"]` | Model replicated across both instances |
| `[]` or `null` | Auto-assign to first available instance |
| `["nonexistent"]` | Model excluded from sync, warning shown in UI |

### 4.2 Sync Behavior

**`sync_instance(instance_name, models, restart)`:**
1. Generate llama-swap `config.yaml` containing only models assigned to this instance
2. Write to `/models/instances/<name>/config.yaml`
3. Optionally restart the container (only this one, not others)

**`sync_all_instances(state, restart)`:**
1. Group models by `instance_setup`
2. For each instance, call `sync_instance()`
3. Restart only instances with changed models

**Selective restart:** When a model is deleted/updated, only the instances listed in its `instance_setup` are restarted. Other instances continue serving their models uninterrupted.

### 4.3 Backward Compatibility

| Scenario | Behavior |
|---|---|
| No config file | 0 instances, UI warning, models show "No instance available" |
| Existing state without `instance_setup` | Auto-assigned to first available instance on next sync |
| `instance_setup` references missing instance | Model excluded, UI shows "Instance unavailable" |
| Old `LLAMA_BACKEND`/`LLAMA_DOCKERFILE` env vars | Ignored (config file is the single source of truth) |

---

## 5. Backend Architecture

### 5.1 New Module: `docker_manager.py`

```
src/backend/docker_manager.py
```

**Responsibilities:**
- Parse `config.yml` / `config.yaml` (YAML)
- Resolve `HF_TOKEN` (`.env` wins over config file)
- Container lifecycle: build, create, start, stop, restart, remove
- Image management (future)
- Config reconciliation (diff, rebuild, recreate)
- GPU device mapping (nvidia driver vs direct passthrough)

**API:**

```python
class DockerManager:
    def load_config(self) -> list[dict]:
        """Parse config.yml, resolve HF_TOKEN, validate, return instance definitions."""

    def ensure_instances(self, state: dict) -> None:
        """At startup: build/start all instances, sync models."""

    def reconcile_instances(self, new_instances: list[dict]) -> list[str]:
        """Diff current vs new, build/recreate/remove as needed.
        Returns list of instance names that need model re-sync."""

    def get_running_instances(self) -> dict[str, dict]:
        """Return {name: {status, image, port, devices, ...}} for all llama-swap containers."""

    def build_instance(self, instance: dict) -> None:
        """docker build from instance['dockerfile'] path."""

    def create_container(self, instance: dict) -> None:
        """docker create with devices, ports, volumes, config mount."""

    def restart_instance(self, name: str) -> None:
        """docker restart specific container."""

    def write_instance_config(self, name: str, models: dict) -> None:
        """Write per-instance config.yaml to shared volume."""

    def get_models_per_instance(self, state: dict) -> dict[str, list[dict]]:
        """Group models by instance_setup. Handles string, list, null, empty."""
```

### 5.2 Modified Modules

**`config.py`** — Split across phases:
- **Phase 0:** Remove `SERVED_DIR`, `CONFIG_PATH`
- **Phase 1+:** Add `INSTANCE_CONFIG_PATH`

**`sync.py`** — Refactor (after Phase 0 direct path migration):
- `sync_system()` → `sync_instance(instance_name, models, restart)` — per-instance sync
- `restart_llama_swap()` → `restart_instance(name)` — specific container
- New `sync_all_instances(state, restart)` — orchestrates all instances
- New `get_models_per_instance(state)` — groups models by instance_setup

**`models.py`** — Add to `ModelSetup`:
```python
instance_setup: list[str] = []  # empty = auto-assign to first available instance
```

**`api.py`** — Changes:
- `setup_model()` — accepts `instance_setup`, auto-assigns if empty
- `delete_config()` — finds affected instances, selectively restarts
- New endpoints:
  - `GET /api/instances` — list instances with status
  - `POST /api/instances/reload` — reload config.yml, reconcile containers

**`app.py`** — Changes:
- `lifespan()` — load instances config, ensure containers running, `sync_all_instances(restart=False)`
- Wire new API routes

### 5.3 API Endpoints Summary

| Route | Method | Description |
|---|---|---|
| `/api/instances` | GET | List instances: `[{name, status, port, devices, image}]` |
| `/api/instances/reload` | POST | Reload config.yml, reconcile containers |
| `/api/setup` | POST | Create/update model (now with `instance_setup` field) |
| `/api/configs/{name}` | DELETE | Delete model config (selectively restarts affected instances) |

*(All existing endpoints remain unchanged.)*

---

## 6. Frontend Architecture

### 6.1 Architecture Goal

The frontend should be a **one-stop management suite** for:
- Deploying new model configs (create)
- Editing existing configs (edit)
- Duplicating configs (duplicate)
- Deleting model configs (delete)
- Managing llama-swap instances (view, reload, rebuild)
- Monitoring disk storage and cache
- Real-time status via WebSocket

### 6.2 State Module: `state.js`

```javascript
let instances = [];  // [{name, status, port, devices, image}]
```

### 6.3 API Module: `api.js`

```javascript
async function fetchInstances() { ... }  // GET /api/instances
```

### 6.4 Render Module: `render.js`

New function:
```javascript
function renderInstances() { ... }  // sidebar section with instance cards
```

Instance card shows:
- Name (clickable, links to container logs in future)
- Status indicator (● running / ○ stopped / ⚠ error)
- Dockerfile path (truncated)
- GPU devices
- Port
- Model count (how many configs assigned)
- Actions: Rebuild, Stop, Start, Logs (future)

### 6.5 Form Module: `form.js`

**New field** (after `symlink_name`, before `parameters`):
```html
<div>
    <label class="block text-sm text-gray-400">Instance(s)</label>
    <select id="instance_setup" multiple class="mt-1 w-full bg-gray-700 border-gray-600 rounded p-2 text-white" size="4">
        <!-- populated dynamically from /api/instances -->
    </select>
    <p class="text-xs text-gray-500 mt-1">Hold Ctrl/Cmd to select multiple instances</p>
</div>
```

**Form payload update:**
```javascript
const payload = {
    hf_repo: ...,
    quant: ...,
    mmproj: ...,
    symlink_name: ...,
    original_name: ...,
    parameters: ...,
    revision: ...,
    instance_setup: Array.from(document.querySelectorAll('#instance_setup option:checked')).map(o => o.value)
};
```

### 6.6 App Module: `app.js`

- Load instances on init (`Api.fetchInstances()`)
- Populate instance dropdown in form
- Auto-assign to first instance when editing (if `instance_setup` is empty)
- Show instance badge on config cards:
  - Single: `cuda-0` (blue badge)
  - Multiple: `cuda-0 +1` (clickable, hover shows all)
  - None: `No instance` (yellow badge)

### 6.7 WebSocket Update

Broadcast payload extended:
```json
{
  "type": "update",
  "data": [...configs...],
  "rpc_mode": false,
  "instances": [
    {"name": "cuda-0", "status": "running", "port": 8080, "devices": ["nvidia"]},
    {"name": "vulkan-0", "status": "running", "port": 8081, "devices": ["direct"]}
  ]
}
```

### 6.8 HTML Structure Update

**Sidebar** (left column, between form and storage):
```
┌─ Deploy New Config ───────────────┐
│ [HF Repo] [Branch] [Commit]       │
│ [Quant] [mmproj] [Symlink Name]   │
│ [Instance(s) ▼]                   │  ← NEW
│ [Parameters JSON]                 │
│ [Provision] [Clear]               │
└────────────────────────────────────┘

┌─ Instances ───────────────────────┐
│ ● cuda-0          NVIDIA  8080   3│  ← NEW
│ ● vulkan-0        AMD     8081   1│
│                                   │
│ 2 instances running               │
└────────────────────────────────────┘

┌─ Disk Storage ────────────────────┐
│ [search]                          │
│ unsloth/Qwen3...                  │
│   abc123def456  14.2GB  Q4_K_M   │
└────────────────────────────────────┘
```

**Config card** (right column) — new badge:
```html
<span class="text-xs bg-blue-900 text-blue-300 px-2 py-0.5 rounded border border-blue-700 font-mono" title="cuda-0, cuda-1">cuda-0 +1</span>
```

### 6.9 Future Frontend Enhancements (TODO)

- **Image management UI** — list local images, sizes, delete unused
- **Container logs viewer** — WebSocket stream of container logs
- **Instance CRUD** — add/edit/remove instances from UI (not just config file)
- **Model deployment wizard** — step-by-step guided model setup
- **Performance monitoring** — GPU utilization, memory, tokens/sec per instance
- **Backup/restore** — export/import config.yml and state.json

---

## 7. Implementation Phases

### Phase 0: Direct Path Migration

**Files:** `src/backend/sync.py`, `src/backend/config.py`, `src/backend/tests/test_sync.py`

- [ ] Use `cf.file_path` directly (already provided by `huggingface_hub.CachedFileInfo`) — no path construction needed
- [ ] Remove symlink creation in `sync_system()` — use `cf.file_path` directly as `--model` path
- [ ] Remove symlink cleanup (`glob.glob` + `os.unlink`)
- [ ] Remove `FileExistsError` handling
- [ ] Simplify sharded model handling — first shard's `cf.file_path` is the entry point
- [ ] Simplify mmproj handling — pass mmproj `cf.file_path` directly to `--mmproj`
- [ ] Remove `SERVED_DIR` from `config.py` (no longer needed)
- [ ] Update `sync_system()` — verify signature stays the same (`sync_system(state, restart)`)
- [ ] Update tests: verify `config.yaml` contains direct cache paths instead of symlink paths
- [ ] Run full test suite — all 127 tests passing

### Phase 1: Core Backend (Config + Docker Manager)

**Files:** `config.example.yml`, `.gitignore`, `src/backend/docker_manager.py`, `src/backend/config.py`

- [ ] Create `config.example.yml` with all options documented
- [ ] Add `config.yml`, `config.yaml`, `user/` to `.gitignore`
- [ ] Implement `DockerManager.load_config()` — parse YAML, validate, resolve HF_TOKEN
- [ ] Implement `DockerManager.ensure_instances()` — build/start containers
- [ ] Implement `DockerManager.reconcile_instances()` — diff and update
- [ ] Implement `DockerManager.get_running_instances()` — scan Docker
- [ ] Implement `DockerManager.build_instance()` — docker build
- [ ] Implement `DockerManager.create_container()` — docker create with devices/ports
- [ ] Implement `DockerManager.write_instance_config()` — per-instance config.yaml
- [ ] Implement `DockerManager.get_models_per_instance()` — group by instance_setup
- [ ] Unit tests: `test_docker_manager.py` (config parsing, validation, grouping)

### Phase 2: Per-Instance Sync

**Files:** `src/backend/sync.py`

- [ ] Refactor `sync_system()` → `sync_instance(instance_name, models, restart)`
- [ ] Refactor `restart_llama_swap()` → `restart_instance(name)`
- [ ] Implement `sync_all_instances(state, restart)`
- [ ] Update `get_models_per_instance()` to handle string/list/null/empty
- [ ] Update tests: `test_sync.py` (per-instance sync, selective restart)
- [ ] Backward compatibility: migration for existing state without `instance_setup`

### Phase 3: API + State Changes

**Files:** `src/backend/models.py`, `src/backend/api.py`, `src/backend/app.py`

- [ ] Add `instance_setup: list[str] = []` to `ModelSetup`
- [ ] Update `setup_model()` — accept `instance_setup`, auto-assign if empty
- [ ] Update `delete_config()` — find affected instances, selective restart
- [ ] Add `GET /api/instances` endpoint
- [ ] Add `POST /api/instances/reload` endpoint
- [ ] Update `lifespan()` — load instances, ensure running, sync all
- [ ] Wire new routes in `app.py`

### Phase 4: Frontend — Instance Data Flow

**Files:** `src/frontend/js/state.js`, `src/frontend/js/api.js`, `src/frontend/js/ws.js`

- [ ] Add `instances` array to `AppState`
- [ ] Add `Api.fetchInstances()`
- [ ] Extend WebSocket broadcast to include `instances` data
- [ ] Update `LocalModels.fetch()` to also fetch instances on init

### Phase 5: Frontend — Instance UI

**Files:** `src/frontend/js/render.js`, `src/frontend/index.html`

- [ ] Add instances section between form and storage in HTML
- [ ] Implement `Render.renderInstances()` — instance cards with status
- [ ] Add instance badge to config cards (single/multiple/none)
- [ ] CSS for instance badges and cards

### Phase 6: Frontend — Form Integration

**Files:** `src/frontend/js/form.js`, `src/frontend/js/app.js`, `src/frontend/index.html`

- [ ] Add instance multi-select dropdown to form HTML
- [ ] Populate dropdown from `AppState.instances`
- [ ] Update form payload to include `instance_setup` array
- [ ] Auto-assign first instance when editing (if empty)
- [ ] Handle form reset with instance field

### Phase 7: Tests

**Files:** `src/backend/tests/test_docker_manager.py`, `src/backend/tests/test_sync.py`, `src/frontend/tests/`

- [ ] Backend: config parsing, validation, auto-assignment, selective sync
- [ ] Backend: reconcile (add/remove/rebuild/recreate)
- [ ] Backend: backward compatibility (missing instance_setup)
- [ ] Frontend: HTML structure (new sections, fields)
- [ ] Frontend: form submission with instance_setup
- [ ] Frontend: render instances section

### Phase 8: Polish & Edge Cases

- [ ] Config reload error handling (invalid YAML, missing dockerfile)
- [ ] GPU device validation (invalid device paths)
- [ ] Port conflict detection
- [ ] Container name collision handling
- [ ] Logging improvements (instance lifecycle events)
- [ ] Update `up.sh` — remove llama-swap compose logic, only start manager
- [ ] Update `Dockerfile` — copy config.example.yml
- [ ] Update README with config.yml documentation

---

## 8. Future Roadmap

### 8.1 Image Management (TODO — High Priority)

llama.cpp Docker images are large (several GB). Users need:
- View local images, sizes, tags
- Delete unused images
- Force rebuild per instance
- Image history and build logs

**API:**
- `GET /api/images` — list images
- `DELETE /api/images/<id>` — remove image
- `POST /api/instances/<name>/rebuild` — rebuild one instance

**UI:** "Images" section in sidebar.

### 8.2 Frontend Instance CRUD (TODO)

Currently instances are managed via config file only. Future: allow managing instances from the UI.

**API:**
- `POST /api/instances` — add new instance
- `PUT /api/instances/<name>` — update instance
- `DELETE /api/instances/<name>` — remove instance

**Behavior:** Updates config.yml atomically, triggers reload.

### 8.3 Container Logs (TODO)

Stream container logs via WebSocket for debugging.

**API:**
- `GET /ws/logs?instance=<name>` — WebSocket endpoint for logs

### 8.4 Performance Monitoring (TODO)

GPU utilization, memory, tokens/sec per instance.

**Approach:** Docker stats API + llama-swap metrics endpoint.

### 8.5 MLX Support (TODO)

Future-proof the instance architecture for Apple MLX backend:
- `devices` spec should support MLX-specific requirements
- Dropdown compatibility check (instance backend vs model compatibility)
- Example: `"backend": "mlx"` in instance config

### 8.6 Multi-Model Deployment Wizard (TODO)

Guided workflow for deploying new models:
1. Enter HF repo → fetch quants
2. Select quant → preview params
3. Select instance(s) → confirm
4. Deploy → track progress

---

## 9. Migration Notes

### Breaking Changes

| Change | Impact | Migration |
|---|---|---|
| `up.sh` no longer manages llama-swap | Users must configure `config.yml` | Provide migration script or docs |
| `LLAMA_BACKEND`/`LLAMA_DOCKERFILE` env vars deprecated | Old setups break | Log warning, suggest config.yml |
| docker-compose overrides for llama-swap removed | `docker-compose.*.yml` files become irrelevant | Keep for manager container only |

### Non-Breaking

| Change | Impact |
|---|---|
| New `instance_setup` field in state | Existing state works (auto-assigned) |
| New `GET /api/instances` endpoint | Backward compatible (new endpoint) |
| New `instance_setup` field in form | Backward compatible (optional field) |

---

## 10. File Change Summary

### New Files

| File | Purpose |
|---|---|
| `config.example.yml` | Template showing all config options |
| `config.yml` / `config.yaml` | User config (gitignored) |
| `src/backend/docker_manager.py` | Container lifecycle management |
| `src/backend/tests/test_docker_manager.py` | Unit tests for docker_manager |

### Modified Files

| File | Changes |
|---|---|
| `.gitignore` | Add `config.yml`, `config.yaml`, `user/` |
| `src/backend/config.py` | **Phase 0:** Remove `SERVED_DIR`, `CONFIG_PATH` → **Phase 1+:** Add `INSTANCE_CONFIG_PATH` |
| `src/backend/sync.py` | **Phase 0:** Remove all symlink logic, use direct paths → **Phase 2:** Per-instance sync, selective restart |
| `src/backend/models.py` | Add `instance_setup` field |
| `src/backend/api.py` | New endpoints, selective sync |
| `src/backend/app.py` | Instance init in lifespan |
| `src/frontend/js/state.js` | Add `instances` array |
| `src/frontend/js/api.js` | Add `fetchInstances()` |
| `src/frontend/js/render.js` | Add `renderInstances()`, instance badges |
| `src/frontend/js/form.js` | Instance multi-select, payload |
| `src/frontend/js/app.js` | Load instances, auto-assign |
| `src/frontend/index.html` | Instance section, multi-select |
| `src/backend/tests/test_sync.py` | Phase 0: direct path assertions → Phase 2: per-instance sync |
| `src/frontend/tests/` | Update for new HTML structure |
| `up.sh` | Remove llama-swap compose logic |
| `Dockerfile` | Copy `config.example.yml` |

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from huggingface_hub import HfApi, scan_cache_dir
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import subprocess
import threading
import os
import sys
import glob
import yaml
import json
import shutil
import asyncio
import pty
import re
import time
import logging
import docker as docker_sdk

log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
logger.info(f"Model Manager booted. Log level set to: {log_level_str}")

CACHE_DIR = "/models/.cache"
SERVED_DIR = "/models/served"
CONFIG_PATH = os.path.join(SERVED_DIR, "config.yaml")
STATE_FILE = os.path.join(SERVED_DIR, "state.json")
LLAMA_SWAP_CONTAINER = os.environ.get("LLAMA_SWAP_CONTAINER", "llama-swap")
QUANT_REGEX = r'[-._]((?:UD-)?[A-Za-z]*Q[0-9][A-Za-z0-9_]*|BF16|F16|F32|MXFP4_MOE)(?:-\d{5}-of-\d{5})?\.gguf$'
SHA_RE = re.compile(r'/snapshots/([a-f0-9]{40})/')

# llama-server short flags (single dash) — everything else uses double dash
LLAMA_SHORT_FLAGS = {
    "a", "b", "bs", "c", "cb", "cd", "cl", "cmoe", "cmoed", "cpent", "cram",
    "ctk", "ctkd", "ctv", "ctvd", "ctxcp", "dev", "devd", "dio", "dr", "dt",
    "e", "fa", "fit", "fitc", "fitt", "h", "hf", "hfd", "hff", "hffv", "hft",
    "hfv", "j", "jf", "kvo", "kvu", "l", "lcd", "lcs", "lv", "m", "md", "mg",
    "mm", "mmu", "mu", "mv", "n", "ncmoe", "ngl", "ngld", "np", "ot", "otd",
    "r", "rea", "s", "sm", "sp", "sps", "t", "tb", "tbd", "td", "to", "ts",
    "ub", "v",
}

class ModelSetup(BaseModel):
    hf_repo: str
    quant: str
    mmproj: str = ""
    symlink_name: str
    original_name: str = ""
    parameters: str
    revision: str = "latest"

class RevisionDeleteReq(BaseModel):
    repo: str
    revision: str

class RpcModeReq(BaseModel):
    enabled: bool

# --- UTILS & STATE ---
def format_bytes(bytes_num):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_num < 1000.0: return f"{bytes_num:.1f} {unit}"
        bytes_num /= 1000.0
    return f"{bytes_num:.1f} PB"

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f: return json.load(f)
        except: pass
    return {}

def save_state(state):
    os.makedirs(SERVED_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f: json.dump(state, f, indent=2)

def iter_configs(state):
    for name, data in state.items():
        if name == "_meta" or not isinstance(data, dict):
            continue
        yield name, data

def restart_llama_swap():
    try:
        client = docker_sdk.from_env()
        container = client.containers.get(LLAMA_SWAP_CONTAINER)
        container.restart()
        logger.info(f"Restarted {LLAMA_SWAP_CONTAINER} to apply new config.")
    except Exception as e:
        logger.error(f"Failed to restart {LLAMA_SWAP_CONTAINER}: {e}")

# --- HF HUB WRAPPERS ---
def resolve_sha(repo, ref="main"):
    """Resolve a repo ref (branch/tag/SHA) to a concrete commit SHA. Raises on failure."""
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = api.model_info(repo, revision=ref)
    return info.sha

def get_commits(repo, limit=50):
    logger.info(f"get_commits: fetching for {repo}, limit={limit}")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    logger.info(f"get_commits: HfApi created, calling list_repo_commits")
    commits = list(api.list_repo_commits(repo, repo_type="model"))[:limit]
    logger.info(f"get_commits: got {len(commits)} commits")
    result = []
    for c in commits:
        msg = ""
        if hasattr(c, "short_commit_message") and c.short_commit_message:
            msg = c.short_commit_message
        elif hasattr(c, "message") and c.message:
            msg = c.message.split("\n", 1)[0][:120]
        # committer_timestamp is a float (epoch seconds) or datetime
        ts = getattr(c, "committer_timestamp", None)
        if ts and isinstance(ts, str):
            try:
                ts = float(ts)
            except ValueError:
                ts = 0
        elif ts and hasattr(ts, "timestamp"):
            ts = ts.timestamp()
        result.append({
            "sha": c.commit_id,
            "date": int(ts) if ts else 0,
            "message": msg,
        })
    # Sort by date descending (newest first)
    result.sort(key=lambda x: x["date"], reverse=True)
    return result

def pre_flight_size(repo, sha, quant, mmproj):
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = api.model_info(repo, revision=sha, files_metadata=True)
    total = 0
    for f in info.siblings:
        if not f.size or not f.rfilename.endswith(".gguf"):
            continue
        match = re.search(QUANT_REGEX, f.rfilename, re.IGNORECASE)
        if not match:
            continue
        is_mmproj_file = "mmproj" in f.rfilename.lower()
        if is_mmproj_file and mmproj and match.group(1).upper() == mmproj.upper():
            total += f.size
        elif not is_mmproj_file and match.group(1).upper() == quant.upper():
            total += f.size
    return total

# --- CACHE INSPECTION & PRUNING ---
def scan_cache():
    """Scan the HF cache. Returns HFCacheInfo or None if cache dir is absent/empty."""
    if not os.path.isdir(CACHE_DIR):
        return None
    try:
        return scan_cache_dir(CACHE_DIR)
    except Exception as e:
        logger.warning(f"scan_cache_dir failed: {e}")
        return None

def in_use_revisions(state):
    """Set of (repo_id, commit_sha) tuples referenced by state."""
    return {
        (d["repo"], d["revision"])
        for _, d in iter_configs(state)
        if d.get("repo") and d.get("revision")
    }

def prune_unreferenced_revisions(state):
    """Delete any cached revision (and orphan repo dir) not referenced by state. Silent, automatic."""
    in_use = in_use_revisions(state)
    cache = scan_cache()
    if cache is not None:
        shas_to_delete = []
        for repo in cache.repos:
            for rev in repo.revisions:
                if (repo.repo_id, rev.commit_hash) not in in_use:
                    shas_to_delete.append(rev.commit_hash)
        if shas_to_delete:
            logger.info(f"Pruning {len(shas_to_delete)} unreferenced revision(s) from cache.")
            try:
                cache.delete_revisions(*shas_to_delete).execute()
            except Exception as e:
                logger.error(f"delete_revisions failed: {e}")

    # Second pass: corrupted repo dirs (empty/broken snapshots) that scan_cache_dir excludes.
    expected_dirs = {
        f"models--{d['repo'].replace('/', '--')}"
        for _, d in iter_configs(state) if d.get("repo")
    }
    if os.path.isdir(CACHE_DIR):
        for entry in os.listdir(CACHE_DIR):
            if not entry.startswith("models--"):
                continue
            if entry in expected_dirs:
                continue
            full = os.path.join(CACHE_DIR, entry)
            if os.path.isdir(full):
                logger.info(f"Removing orphan repo dir: {entry}")
                shutil.rmtree(full, ignore_errors=True)

# --- SYNC (state → filesystem + llama-swap config) ---
def sync_system(state, restart=True):
    """Rebuild served-dir symlinks and llama-swap config.yaml from state."""
    logger.info("Syncing served dir and llama-swap config from state...")
    os.makedirs(SERVED_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(SERVED_DIR, "*.gguf")):
        try: os.unlink(f)
        except: pass

    cache = scan_cache()
    rev_lookup = {}
    if cache is not None:
        for repo in cache.repos:
            for rev in repo.revisions:
                rev_lookup[(repo.repo_id, rev.commit_hash)] = rev

    swap_models = {}
    state_changed = False
    rpc_mode = state.get("_meta", {}).get("rpc_mode", False)

    for name, data in iter_configs(state):
        if data.get("status") in ("downloading", "error"):
            continue

        has_rpc = bool(data.get("params", {}).get("rpc"))
        if rpc_mode != has_rpc:
            continue

        repo = data.get("repo")
        sha = data.get("revision")
        quant = data.get("quant", "")
        mmproj = data.get("mmproj", "")
        params = data.get("params", {})

        rev = rev_lookup.get((repo, sha)) if repo and sha else None
        if rev is None:
            logger.warning(f"Snapshot missing for {name} (repo={repo}, sha={(sha or '')[:12]}). Marking missing.")
            if data.get("status") != "missing":
                data["status"] = "missing"
                state_changed = True
            continue

        quant_files = []
        mmproj_file = None
        for cf in rev.files:
            fname = cf.file_name
            match = re.search(QUANT_REGEX, fname, re.IGNORECASE)
            if not match:
                continue
            matched_q = match.group(1).upper()
            is_mmproj_file = "mmproj" in fname.lower()
            if is_mmproj_file:
                if mmproj and matched_q == mmproj.upper():
                    mmproj_file = cf
            elif matched_q == quant.upper():
                quant_files.append(cf)

        if not quant_files or (mmproj and mmproj_file is None):
            logger.warning(f"Required files missing in snapshot for {name}. Marking missing.")
            if data.get("status") != "missing":
                data["status"] = "missing"
                state_changed = True
            continue

        if data.get("status") == "missing":
            data["status"] = "ready"
            state_changed = True

        quant_files.sort(key=lambda cf: cf.file_name)
        first_shard = None
        for cf in quant_files:
            shard_match = re.search(r'(-\d{5}-of-\d{5}\.gguf)$', cf.file_name, re.IGNORECASE)
            if shard_match:
                symlink_filename = f"{name}-{quant}{shard_match.group(1)}"
                if "-00001-of-" in shard_match.group(1):
                    first_shard = os.path.join(SERVED_DIR, symlink_filename)
            else:
                symlink_filename = f"{name}-{quant}.gguf"
                if first_shard is None:
                    first_shard = os.path.join(SERVED_DIR, symlink_filename)
            symlink_path = os.path.join(SERVED_DIR, symlink_filename)
            try:
                os.symlink(str(cf.file_path), symlink_path)
            except FileExistsError:
                pass

        model_path = first_shard

        cmd_args = ["llama-server", "--model", model_path, "--host", "0.0.0.0", "--port", "${PORT}"]

        if mmproj_file is not None:
            mmproj_path = os.path.join(SERVED_DIR, f"{name}-mmproj-{mmproj.upper()}.gguf")
            try:
                os.symlink(str(mmproj_file.file_path), mmproj_path)
            except FileExistsError:
                pass
            cmd_args.extend(["--mmproj", mmproj_path])

        for k, v in params.items():
            if k in ("host", "port", "model", "mmproj"):
                continue
            flag = f"-{k}" if k in LLAMA_SHORT_FLAGS else f"--{k}"
            if v is None or (isinstance(v, bool) and v):
                cmd_args.append(flag)
            elif isinstance(v, bool) and not v:
                continue
            elif isinstance(v, (dict, list)):
                json_str = json.dumps(v, separators=(',', ':'))
                escaped = json_str.replace('\\', '\\\\').replace('"', '\\"')
                cmd_args.extend([flag, f'"{escaped}"'])
            else:
                cmd_args.extend([flag, str(v)])

        model_id = f"{name}-{quant}"
        inner_cmd = " ".join(cmd_args)
        wrapped_cmd = f"sh -c 'exec {inner_cmd} >/proc/1/fd/1 2>&1'"
        swap_models[model_id] = {
            "cmd": wrapped_cmd,
            "proxy": "http://127.0.0.1:${PORT}",
        }

    if state_changed:
        save_state(state)

    with open(CONFIG_PATH, 'w') as f:
        yaml.dump({"models": swap_models}, f, default_flow_style=False, sort_keys=False)
    logger.info("llama-swap config.yaml rewritten.")

    if restart:
        threading.Thread(target=restart_llama_swap, daemon=True).start()

# --- WEBSOCKET MANAGER ---
class ConnectionManager:
    def __init__(self): self.active_connections: list[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket): self.active_connections.remove(websocket)
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try: await connection.send_text(message)
            except: pass

manager = ConnectionManager()

# --- DOWNLOAD WORKER (PTY for tqdm progress) ---
def process_model(req: ModelSetup):
    """Download via huggingface_hub.snapshot_download with a pre-resolved SHA in req.revision."""
    logger.info(f"Starting background download for {req.symlink_name} ({req.revision[:12]})...")
    patterns = [f"*{req.quant}.gguf", f"*{req.quant}-*-of-*.gguf"]
    if req.mmproj:
        patterns.append(f"*mmproj*{req.mmproj}*.gguf")

    script_code = f"""
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id={req.hf_repo!r},
    allow_patterns={patterns!r},
    cache_dir={CACHE_DIR!r},
    revision={req.revision!r},
)
"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    refs_dir = os.path.join(
        CACHE_DIR,
        f"models--{req.hf_repo.replace('/', '--')}",
        "refs",
    )
    script_path = os.path.join(CACHE_DIR, f"dl_{req.symlink_name}.py")
    with open(script_path, "w") as f: f.write(script_code)

    cmd = [sys.executable, script_path]
    master, slave = pty.openpty()
    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    try:
        process = subprocess.Popen(cmd, stdin=slave, stdout=slave, stderr=slave, close_fds=True, env=env)
    except Exception as e:
        logger.error(f"Subprocess failed to launch for {req.symlink_name}: {e}")
        state = load_state()
        if req.symlink_name in state:
            state[req.symlink_name]["status"] = "error"
            state[req.symlink_name]["error_msg"] = f"Subprocess failed: {e}"[:120]
            save_state(state)
        os.close(slave)
        os.close(master)
        return

    os.close(slave)
    buffer = ""
    error_log = []
    last_ui_update = 0

    while True:
        try:
            data = os.read(master, 32768).decode('utf-8', errors='replace')
            if not data: break
            buffer += data
            if len(error_log) < 100: error_log.append(data)

            now = time.time()
            if now - last_ui_update > 0.5:
                lines = buffer.replace('\r', '\n').split('\n')
                buffer = lines.pop()

                for line in reversed(lines):
                    clean_line = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', line)
                    data_m = re.search(r'([0-9.]+[A-Za-z]?B?)\s*/\s*([0-9.]+[A-Za-z]?B?)', clean_line, re.IGNORECASE)

                    if data_m:
                        perc_m = re.search(r'(\d{1,3})%', clean_line)
                        speed_m = re.search(r'([0-9.]+[A-Za-z]?B?/s)', clean_line, re.IGNORECASE)
                        eta_m = re.search(r'<([0-9:]+)', clean_line)

                        state = load_state()
                        if req.symlink_name in state and state[req.symlink_name].get("status") == "downloading":
                            dl_str = data_m.group(1).upper()
                            tot_str = data_m.group(2).upper()
                            if not dl_str.endswith('B'): dl_str += "B"
                            if not tot_str.endswith('B'): tot_str += "B"

                            state[req.symlink_name]["progress_str"] = {
                                "percent": perc_m.group(1) if perc_m else "0",
                                "downloaded": dl_str,
                                "total": tot_str,
                                "speed": speed_m.group(1).upper() if speed_m else "--",
                                "eta": eta_m.group(1) if eta_m else "--"
                            }
                            save_state(state)

                        last_ui_update = now
                        break
        except OSError: break

    process.wait()
    os.close(master)
    try: os.unlink(script_path)
    except: pass

    state = load_state()
    if process.returncode == 0:
        logger.info(f"Download completed for {req.symlink_name}.")
        # Ensure refs/main points to the downloaded SHA so scan_cache_dir can find the snapshot.
        # Without this, if HF's main branch advanced after the original download, the ref becomes
        # stale and the snapshot is invisible to scan_cache_dir.
        try:
            os.makedirs(refs_dir, exist_ok=True)
            refs_main = os.path.join(refs_dir, "main")
            with open(refs_main, "w") as rf:
                rf.write(req.revision)
            logger.debug(f"Updated refs/main → {req.revision[:12]} for {req.hf_repo}")
        except Exception as e:
            logger.warning(f"Could not update refs/main for {req.hf_repo}: {e}")
        if req.symlink_name in state:
            state[req.symlink_name]["status"] = "ready"
            save_state(state)
        sync_system(state)
        # Previous revision for this config may now be orphaned.
        prune_unreferenced_revisions(state)
    else:
        err_text = "".join(error_log[-10:]).replace('\n', ' ')
        exc_match = re.search(r'([A-Za-z]+Error:.*)', err_text)
        final_err = exc_match.group(1) if exc_match else err_text
        logger.error(f"Download failed for {req.symlink_name}. Error: {final_err}")

        if req.symlink_name in state:
            state[req.symlink_name]["status"] = "error"
            state[req.symlink_name]["error_msg"] = f"Failed: {final_err}"[:120]
            save_state(state)

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    state = load_state()
    prune_unreferenced_revisions(state)
    sync_system(state, restart=False)

    resumed = 0
    for name, data in iter_configs(state):
        if data.get("status") == "downloading" and data.get("revision"):
            resumed += 1
            req = ModelSetup(
                hf_repo=data["repo"],
                quant=data["quant"],
                mmproj=data.get("mmproj", ""),
                symlink_name=name,
                parameters=json.dumps(data["params"]),
                revision=data["revision"],
            )
            threading.Thread(target=process_model, args=(req,), daemon=True).start()
        elif data.get("status") == "downloading":
            logger.warning(f"Abandoning resume for {name}: no pinned revision in state.")
            data["status"] = "error"
            data["error_msg"] = "Interrupted download without pinned revision; redeploy."
            save_state(state)

    if resumed > 0: logger.info(f"Resumed {resumed} interrupted download(s).")

    async def broadcast_state():
        while True:
            await asyncio.sleep(1)
            if manager.active_connections:
                state_data = load_state()
                configs = [{"name": k, **v} for k, v in state_data.items() if k != "_meta"]
                rpc_mode = state_data.get("_meta", {}).get("rpc_mode", False)
                await manager.broadcast(json.dumps({"type": "update", "data": configs, "rpc_mode": rpc_mode}))

    broadcast_task = asyncio.create_task(broadcast_state())
    yield
    broadcast_task.cancel()

app = FastAPI(title="Local LLM Model Manager", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- API ---
@app.get("/")
async def serve_ui(): return FileResponse("index.html")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon(): return FileResponse("favicon.ico")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/quants")
async def get_quants(repo: str):
    try:
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        logger.info(f"Scanning repo '{repo}' for quant files...")
        info = api.model_info(repo, files_metadata=True)
        quants_dict = {}
        mmproj_dict = {}

        for f in info.siblings:
            if not f.rfilename.endswith(".gguf"): continue
            match = re.search(QUANT_REGEX, f.rfilename, re.IGNORECASE)
            if not match: continue
            q_name = match.group(1).upper()
            q_size = f.size or 0
            if "mmproj" in f.rfilename.lower():
                mmproj_dict[q_name] = mmproj_dict.get(q_name, 0) + q_size
            else:
                quants_dict[q_name] = quants_dict.get(q_name, 0) + q_size

        quants_list = [{"name": q, "size_str": format_bytes(s), "raw": s} for q, s in quants_dict.items()]
        quants_list.sort(key=lambda x: x["raw"], reverse=True)
        mmproj_list = [{"name": q, "size_str": format_bytes(s), "raw": s} for q, s in mmproj_dict.items()]
        mmproj_list.sort(key=lambda x: x["raw"], reverse=True)

        return {"quants": quants_list, "mmprojs": mmproj_list}
    except Exception as e:
        logger.error(f"Failed to fetch quants for {repo}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/commits")
async def api_get_commits(repo: str):
    logger.info(f"Fetching commits for repo: {repo}")
    try:
        state = load_state()
        pinned_shas = {
            d["revision"]
            for _, d in iter_configs(state)
            if d.get("repo") == repo and d.get("revision")
        }
        commits = get_commits(repo)
        for c in commits:
            c["pinned"] = c["sha"] in pinned_shas
        logger.info(f"Got {len(commits)} commits for {repo}")
        return {"commits": commits}
    except Exception as e:
        logger.error(f"Failed to fetch commits for {repo}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/models")
async def get_local_models():
    cache = scan_cache()
    if cache is None:
        return {"models": []}

    state = load_state()
    used_by = {}
    for name, data in iter_configs(state):
        key = (data.get("repo"), data.get("revision"))
        if None in key: continue
        used_by.setdefault(key, []).append(name)

    models = []
    for repo in sorted(cache.repos, key=lambda r: r.repo_id):
        revisions = []
        for rev in sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True):
            files = []
            for cf in sorted(rev.files, key=lambda f: f.file_name):
                if not cf.file_name.endswith(".gguf"): continue
                match = re.search(QUANT_REGEX, cf.file_name, re.IGNORECASE)
                files.append({
                    "name": cf.file_name,
                    "size_str": format_bytes(cf.size_on_disk),
                    "quant": match.group(1).upper() if match else None,
                    "is_mmproj": "mmproj" in cf.file_name.lower(),
                })
            if not files:
                continue
            revisions.append({
                "sha": rev.commit_hash,
                "short_sha": rev.commit_hash[:12],
                "size_str": format_bytes(rev.size_on_disk),
                "last_modified": rev.last_modified,
                "refs": sorted(rev.refs),
                "files": files,
                "used_by": used_by.get((repo.repo_id, rev.commit_hash), []),
            })
        if not revisions:
            continue
        models.append({
            "repo": repo.repo_id,
            "size_str": format_bytes(repo.size_on_disk),
            "revisions": revisions,
        })
    return {"models": models}

@app.post("/api/rpc_mode")
async def toggle_rpc(req: RpcModeReq):
    state = load_state()
    state.setdefault("_meta", {})["rpc_mode"] = req.enabled
    save_state(state)
    sync_system(state)
    return {"status": f"RPC mode enabled: {req.enabled}"}

@app.post("/api/revisions/delete")
async def delete_revision(req: RevisionDeleteReq):
    state = load_state()
    blockers = [
        name for name, data in iter_configs(state)
        if data.get("repo") == req.repo and data.get("revision") == req.revision
    ]
    if blockers:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Revision is pinned by existing configs. Delete them first.",
                "used_by": blockers,
                "repo": req.repo,
                "revision": req.revision,
            },
        )

    cache = scan_cache()
    if cache is None:
        raise HTTPException(status_code=404, detail="Cache is empty.")

    known = any(
        rev.commit_hash == req.revision
        for repo in cache.repos for rev in repo.revisions
        if repo.repo_id == req.repo
    )
    if not known:
        raise HTTPException(status_code=404, detail="Revision not found in cache.")

    try:
        cache.delete_revisions(req.revision).execute()
    except Exception as e:
        logger.error(f"delete_revisions failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {e}")

    sync_system(state)
    return {"status": "Deleted"}

@app.post("/api/setup")
async def setup_model(req: ModelSetup, background_tasks: BackgroundTasks):
    req.symlink_name = req.symlink_name.replace("/", "-")
    if req.original_name: req.original_name = req.original_name.replace("/", "-")
    logger.info(f"Setup {req.symlink_name} ({req.quant}, rev={req.revision[:12] if req.revision != 'latest' else 'latest'})")

    try:
        params_dict = json.loads(req.parameters) if req.parameters.strip() else {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    try:
        ref = "main" if req.revision == "latest" else req.revision
        resolved_sha = resolve_sha(req.hf_repo, ref)
    except Exception as e:
        logger.error(f"Could not resolve revision '{req.revision}' for {req.hf_repo}: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid repo/revision: {e}")
    req.revision = resolved_sha

    state = load_state()
    if req.original_name and req.original_name != req.symlink_name and req.original_name in state:
        del state[req.original_name]

    existing = state.get(req.symlink_name) or {}
    needs_download = (
        not existing
        or existing.get("repo") != req.hf_repo
        or existing.get("quant") != req.quant
        or existing.get("mmproj", "") != req.mmproj
        or existing.get("revision") != resolved_sha
        or existing.get("status") == "missing"
    )

    warning_msg = ""
    if needs_download:
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            model_size = pre_flight_size(req.hf_repo, resolved_sha, req.quant, req.mmproj)
        except Exception as e:
            logger.error(f"Pre-flight API check failed: {e}")
            raise HTTPException(status_code=400, detail=f"API Error: {e}")

        total, used, free = shutil.disk_usage(CACHE_DIR)
        if model_size > free:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient Storage. Model requires {format_bytes(model_size)}, but only {format_bytes(free)} is available.",
            )
        if (used + model_size) / total > 0.8:
            warning_msg = " (Warning: This will push disk usage above 80%)"

    state[req.symlink_name] = {
        "repo": req.hf_repo,
        "quant": req.quant,
        "mmproj": req.mmproj,
        "params": params_dict,
        "status": "downloading" if needs_download else "ready",
        "revision": resolved_sha,
    }
    save_state(state)

    if needs_download:
        background_tasks.add_task(process_model, req)
        return {"status": f"Provisioning started.{warning_msg}"}
    else:
        sync_system(state)
        return {"status": "Config updated!"}

@app.delete("/api/configs/{symlink_name}")
async def delete_config(symlink_name: str):
    logger.info(f"Deleting config {symlink_name}")
    state = load_state()
    if symlink_name in state:
        del state[symlink_name]
        save_state(state)
        prune_unreferenced_revisions(state)
        sync_system(state)
    return {"status": "Config deleted!"}

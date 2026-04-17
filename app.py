from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from huggingface_hub import HfApi
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import subprocess
import threading
import os
import sys
import glob
import yaml
import json
import shlex
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

class DeleteModelReq(BaseModel):
    repo: str
    quant: str
    is_mmproj: bool

class RpcModeReq(BaseModel):
    enabled: bool

# --- UTILS & STATE MANAGEMENT ---
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

def restart_llama_swap():
    try:
        client = docker_sdk.from_env()
        container = client.containers.get(LLAMA_SWAP_CONTAINER)
        container.restart()
        logger.info(f"Restarted {LLAMA_SWAP_CONTAINER} to apply new config.")
    except Exception as e:
        logger.error(f"Failed to restart {LLAMA_SWAP_CONTAINER}: {e}")

def sync_system(state, restart=True):
    logger.info("Syncing system state and regenerating symlinks...")
    os.makedirs(SERVED_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(SERVED_DIR, "*.gguf")):
        try: os.unlink(f)
        except: pass

    swap_models = {}
    state_changed = False

    rpc_mode = state.get("_meta", {}).get("rpc_mode", False)

    for name, data in state.items():
        if name == "_meta" or data.get("status") in ["downloading", "error"]: continue

        has_rpc = bool(data.get("params", {}).get("rpc"))
        if rpc_mode and not has_rpc: continue
        if not rpc_mode and has_rpc: continue

        repo, quant, params = data["repo"], data["quant"], data["params"]
        mmproj = data.get("mmproj", "")
        logger.debug(f"Processing config: {name} ({quant}) from {repo}")

        search_pattern = f"{CACHE_DIR}/models--{repo.replace('/', '--')}/**/*.gguf"
        all_files = glob.glob(search_pattern, recursive=True)

        files = []
        for f in all_files:
            if "mmproj" in f.lower(): continue
            match = re.search(QUANT_REGEX, f, re.IGNORECASE)
            if match and match.group(1).upper() == quant.upper():
                files.append(f)

        files = sorted(files)

        mmproj_missing = False
        if mmproj:
            found_mmproj = False
            for f in all_files:
                if "mmproj" in f.lower():
                    match = re.search(QUANT_REGEX, f, re.IGNORECASE)
                    if match and match.group(1).upper() == mmproj.upper():
                        found_mmproj = True
                        break
            if not found_mmproj:
                mmproj_missing = True

        if not files or mmproj_missing:
            logger.warning(f"Files missing for {name}. Tagging as missing.")
            if data.get("status") != "missing":
                data["status"] = "missing"
                state_changed = True
            continue
        else:
            if data.get("status") == "missing":
                data["status"] = "ready"
                state_changed = True

        first_shard = None
        for f in files:
            match = re.search(r'(-\d{5}-of-\d{5}\.gguf)$', f, re.IGNORECASE)
            if match:
                suffix = match.group(1)
                symlink_filename = f"{name}-{quant}{suffix}"
                symlink_path = os.path.join(SERVED_DIR, symlink_filename)
                try:
                    os.symlink(f, symlink_path)
                except FileExistsError:
                    pass
                if "-00001-of-" in suffix: first_shard = symlink_path
            else:
                symlink_filename = f"{name}-{quant}.gguf"
                symlink_path = os.path.join(SERVED_DIR, symlink_filename)
                try:
                    os.symlink(f, symlink_path)
                except FileExistsError:
                    pass
                if not first_shard: first_shard = symlink_path

        model_path = first_shard or files[0]

        # Build llama-server CLI args from params
        cmd_args = ["llama-server", "--model", model_path, "--host", "0.0.0.0", "--port", "${PORT}"]

        if mmproj:
            for f in all_files:
                if "mmproj" in f.lower():
                    match = re.search(QUANT_REGEX, f, re.IGNORECASE)
                    if match and match.group(1).upper() == mmproj.upper():
                        mmproj_path = os.path.join(SERVED_DIR, f"{name}-mmproj.gguf")
                        os.symlink(f, mmproj_path)
                        cmd_args.extend(["--mmproj", mmproj_path])
                        logger.debug(f"Created mmproj symlink: {mmproj_path}")
                        break

        for k, v in params.items():
            if k in ("host", "port", "model", "mmproj"):
                continue
            flag = f"-{k}" if k in LLAMA_SHORT_FLAGS else f"--{k}"
            if v is None or (isinstance(v, bool) and v):
                cmd_args.append(flag)
            elif isinstance(v, bool) and not v:
                continue
            elif isinstance(v, (dict, list)):
                # Embed JSON in double quotes with escaped inner quotes so the inner
                # sh (inside the outer sh -c '...') preserves the double quotes instead
                # of stripping them during word processing.
                json_str = json.dumps(v, separators=(',', ':'))
                escaped = json_str.replace('\\', '\\\\').replace('"', '\\"')
                cmd_args.extend([flag, f'"{escaped}"'])
            else:
                cmd_args.extend([flag, str(v)])

        model_id = f"{name}-{quant}"
        inner_cmd = " ".join(cmd_args)
        # Forward llama-server stdout+stderr to container PID 1 stdout so they
        # appear in `docker logs`. exec replaces the shell so signals pass through.
        wrapped_cmd = f"sh -c 'exec {inner_cmd} >/proc/1/fd/1 2>&1'"
        swap_models[model_id] = {
            "cmd": wrapped_cmd,
            "proxy": "http://127.0.0.1:${PORT}",
        }

    if state_changed: save_state(state)

    swap_config = {"models": swap_models}
    with open(CONFIG_PATH, 'w') as f: yaml.dump(swap_config, f, default_flow_style=False, sort_keys=False)
    logger.info("llama-swap config.yaml successfully rewritten.")
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

# --- PTY SUBPROCESS WRAPPER ---
def process_model(req: ModelSetup, params_dict: dict):
    logger.info(f"Starting background download for {req.symlink_name}...")
    patterns = f"['*{req.quant}.gguf', '*{req.quant}-*-of-*.gguf']"
    if req.mmproj:
        patterns = f"['*{req.quant}.gguf', '*{req.quant}-*-of-*.gguf', '*mmproj*{req.mmproj}*.gguf']"
        
    script_code = f"""
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='''{req.hf_repo}''',
    allow_patterns={patterns},
    cache_dir='''{CACHE_DIR}'''
)
"""
    os.makedirs(CACHE_DIR, exist_ok=True)
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
        logger.info(f"Download completed successfully for {req.symlink_name}.")
        if req.symlink_name in state:
            state[req.symlink_name]["status"] = "ready"
            save_state(state)
        sync_system(state)
    else:
        err_text = "".join(error_log[-10:]).replace('\n', ' ')
        exc_match = re.search(r'([A-Za-z]+Error:.*)', err_text)
        final_err = exc_match.group(1) if exc_match else err_text
        logger.error(f"Download failed for {req.symlink_name}. Error: {final_err}")
        
        if req.symlink_name in state:
            state[req.symlink_name]["status"] = "error"
            state[req.symlink_name]["error_msg"] = f"Failed: {final_err}"[:120]
            save_state(state)

# --- LIFESPAN (AUTO-RESUME & WS BROADCASTER) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    state = load_state()
    sync_system(state, restart=False)
    resumed = 0
    for name, data in state.items():
        if data.get("status") == "downloading":
            resumed += 1
            req = ModelSetup(hf_repo=data["repo"], quant=data["quant"], symlink_name=name, parameters=json.dumps(data["params"]))
            threading.Thread(target=process_model, args=(req, data["params"]), daemon=True).start()

    if resumed > 0: logger.info(f"Resumed {resumed} interrupted downloads.")
    
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

# --- API ENDPOINTS ---
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
        logger.info(f"Found {len(info.siblings)} files in the repo. Processing...")
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

        logger.info(f"Found {len(quants_dict)} unique quants in {repo}.")

        quants_list = [{"name": q, "size_str": format_bytes(s), "raw": s} for q, s in quants_dict.items()]
        quants_list.sort(key=lambda x: x["raw"], reverse=True)

        mmproj_list = [{"name": q, "size_str": format_bytes(s), "raw": s} for q, s in mmproj_dict.items()]
        mmproj_list.sort(key=lambda x: x["raw"], reverse=True)
        
        return {"quants": quants_list, "mmprojs": mmproj_list}
    except Exception as e:
        logger.error(f"Failed to fetch quants for {repo}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/models")
async def get_local_models():
    models_data = []
    if not os.path.exists(CACHE_DIR): return {"models": []}
    
    for repo_dir in glob.glob(os.path.join(CACHE_DIR, "models--*")):
        repo_name = os.path.basename(repo_dir).replace("models--", "").replace("--", "/")
        quants_map = {}
        mmprojs_map = {}
        
        for f in glob.glob(os.path.join(repo_dir, "**/*.gguf"), recursive=True):
            filename = os.path.basename(f)
            size = os.path.getsize(f)
            
            match = re.search(QUANT_REGEX, filename, re.IGNORECASE)
            if not match: continue
            quant = match.group(1).upper()
            
            if "mmproj" in filename.lower():
                if quant not in mmprojs_map: mmprojs_map[quant] = {"total_size": 0, "files": []}
                mmprojs_map[quant]["total_size"] += size
                mmprojs_map[quant]["files"].append({"name": filename, "size_str": format_bytes(size)})
            else:
                if quant not in quants_map: quants_map[quant] = {"total_size": 0, "files": []}
                quants_map[quant]["total_size"] += size
                quants_map[quant]["files"].append({"name": filename, "size_str": format_bytes(size)})
        
        if quants_map or mmprojs_map:
            models_data.append({
                "repo": repo_name,
                "quants": [{"quant": k, "size_str": format_bytes(v["total_size"]), "files": sorted(v["files"], key=lambda x: x["name"])} for k, v in quants_map.items()],
                "mmprojs": [{"quant": k, "size_str": format_bytes(v["total_size"]), "files": sorted(v["files"], key=lambda x: x["name"])} for k, v in mmprojs_map.items()]
            })
            
    return {"models": models_data}

@app.post("/api/rpc_mode")
async def toggle_rpc(req: RpcModeReq):
    state = load_state()
    if "_meta" not in state:
        state["_meta"] = {}
    state["_meta"]["rpc_mode"] = req.enabled
    save_state(state)
    sync_system(state)
    return {"status": f"RPC mode enabled: {req.enabled}"}

@app.post("/api/models/delete")
async def delete_local_model(req: DeleteModelReq):
    repo_dir = os.path.join(CACHE_DIR, f"models--{req.repo.replace('/', '--')}")
    search_pattern = os.path.join(repo_dir, "**/*.gguf")

    blobs_to_delete = set()
    symlinks_to_delete = []

    for f in glob.glob(search_pattern, recursive=True):
        filename = os.path.basename(f)
        match = re.search(QUANT_REGEX, filename, re.IGNORECASE)
        if not match: continue
        if match.group(1).upper() != req.quant.upper(): continue
        is_mmproj_file = "mmproj" in filename.lower()
        if is_mmproj_file != req.is_mmproj: continue

        symlinks_to_delete.append(f)
        if os.path.islink(f):
            blob_path = os.path.realpath(f)
            if blob_path.startswith(repo_dir):
                blobs_to_delete.add(blob_path)

    if not symlinks_to_delete:
        raise HTTPException(status_code=404, detail="Files not found on disk")

    for path in blobs_to_delete | set(symlinks_to_delete):
        try: os.remove(path)
        except: pass

    # Remove snapshot dirs that are now empty
    snapshots_dir = os.path.join(repo_dir, "snapshots")
    if os.path.isdir(snapshots_dir):
        for entry in os.listdir(snapshots_dir):
            entry_path = os.path.join(snapshots_dir, entry)
            if os.path.isdir(entry_path) and not os.listdir(entry_path):
                shutil.rmtree(entry_path, ignore_errors=True)

    # Remove the entire repo cache dir if no blob files remain
    remaining_blobs = [f for f in glob.glob(os.path.join(repo_dir, "**/*"), recursive=True) if os.path.isfile(f)]
    if not remaining_blobs:
        shutil.rmtree(repo_dir, ignore_errors=True)

    state = load_state()
    sync_system(state)
    return {"status": "Deleted successfully"}

@app.post("/api/setup")
async def setup_model(req: ModelSetup, background_tasks: BackgroundTasks):
    logger.info(f"Received setup request for {req.symlink_name} ({req.quant})")
    req.symlink_name = req.symlink_name.replace("/", "-")
    if req.original_name: req.original_name = req.original_name.replace("/", "-")

    try: params_dict = json.loads(req.parameters) if req.parameters.strip() else {}
    except Exception as e:
        logger.error(f"JSON Parsing failed during setup: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
    state = load_state()
    if req.original_name and req.original_name != req.symlink_name and req.original_name in state:
        del state[req.original_name]
        
    needs_download = (
        req.symlink_name not in state or
        state[req.symlink_name].get("repo") != req.hf_repo or
        state[req.symlink_name].get("quant") != req.quant or
        state[req.symlink_name].get("mmproj", "") != req.mmproj or
        state[req.symlink_name].get("status") == "missing"
    )

    warning_msg = ""
    if needs_download:
        logger.info(f"Model not found in cache. Initiating pre-flight storage check...")
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            api = HfApi(token=os.environ.get("HF_TOKEN"))
            info = api.model_info(req.hf_repo, files_metadata=True)
            model_size = 0
            for f in info.siblings:
                if f.size and f.rfilename.endswith(".gguf"):
                    is_mmproj = "mmproj" in f.rfilename.lower()
                    match = re.search(QUANT_REGEX, f.rfilename, re.IGNORECASE)
                    if not match: continue

                    if is_mmproj and req.mmproj and match.group(1).upper() == req.mmproj.upper():
                        model_size += f.size
                    elif not is_mmproj and match.group(1).upper() == req.quant.upper():
                        model_size += f.size
        except Exception as e:
            logger.error(f"Pre-flight API check failed: {e}")
            raise HTTPException(status_code=400, detail=f"API Error: {e}")

        total, used, free = shutil.disk_usage(CACHE_DIR)
        logger.debug(f"Storage Status: {format_bytes(free)} free, {format_bytes(model_size)} required.")
        
        if model_size > free:
            logger.error("Insufficient storage space. Aborting setup.")
            raise HTTPException(status_code=400, detail=f"Insufficient Storage. Model requires {format_bytes(model_size)}, but only {format_bytes(free)} is available.")
            
        if (used + model_size) / total > 0.8:
            warning_msg = f" (Warning: This will push disk usage above 80%)"

    state[req.symlink_name] = {
        "repo": req.hf_repo, "quant": req.quant, "mmproj": req.mmproj, "params": params_dict,
        "status": "downloading" if needs_download else "ready",
    }
    save_state(state)

    if needs_download:
        background_tasks.add_task(process_model, req, params_dict)
        return {"status": f"Provisioning started.{warning_msg}"}
    else:
        logger.info(f"Model {req.symlink_name} already downloaded. Updating config directly.")
        sync_system(state)
        return {"status": "Config updated!"}

@app.delete("/api/configs/{symlink_name}")
async def delete_config(symlink_name: str):
    logger.info(f"Deleting config for {symlink_name}...")
    state = load_state()
    if symlink_name in state:
        del state[symlink_name]
        save_state(state)
        sync_system(state)
    return {"status": "Config deleted!"}
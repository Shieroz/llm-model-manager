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
import configparser
import json
import shutil
import asyncio
import pty
import re
import time
import logging

log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
# Safely fallback to INFO if the user types an invalid level in docker-compose
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
INI_PATH = os.path.join(SERVED_DIR, "models.ini")
STATE_FILE = os.path.join(SERVED_DIR, "state.json")
QUANT_REGEX = r'[-._]((?:UD-)?[A-Za-z]*Q[0-9][A-Za-z0-9_]*|BF16|F16|F32|MXFP4_MOE)(?:-\d{5}-of-\d{5})?\.gguf$'

class ModelSetup(BaseModel):
    hf_repo: str             
    quant: str               
    symlink_name: str             
    original_name: str = ""       
    parameters: str 

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

def sync_system(state):
    logger.info("Syncing system state and regenerating symlinks...")
    os.makedirs(SERVED_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(SERVED_DIR, "*.gguf")):
        try: os.unlink(f)
        except: pass
        
    config = configparser.ConfigParser()
    config.optionxform = str
    
    for name, data in state.items():
        if data.get("status") != "ready": continue 
            
        repo, quant, params = data["repo"], data["quant"], data["params"]
        logger.debug(f"Processing config: {name} ({quant}) from {repo}")

        search_pattern = f"{CACHE_DIR}/models--{repo.replace('/', '--')}/**/*.gguf"
        all_files = glob.glob(search_pattern, recursive=True)
        
        files = []
        for f in all_files:
            match = re.search(QUANT_REGEX, f, re.IGNORECASE)
            if match and match.group(1).upper() == quant.upper():
                files.append(f)
                
        files = sorted(files)
        if not files:
            logger.warning(f"No matching files found for {name} ({quant}). Skipping.")
            continue
            
        section_name = f"{name}-{quant}"
        config.add_section(section_name)
        
        # --- SHARD HANDLING ---
        first_shard = None
        for f in files:
            # Check if the file uses the Hugging Face split naming convention
            match = re.search(r'(-\d{5}-of-\d{5}\.gguf)$', f, re.IGNORECASE)
            
            if match:
                suffix = match.group(1)
                symlink_filename = f"{name}-{quant}{suffix}"
                symlink_path = os.path.join(SERVED_DIR, symlink_filename)
                os.symlink(f, symlink_path)
                logger.debug(f"Created shard symlink: {symlink_filename}")
                
                # We must hand exactly the "-00001-of-..." file to llama.cpp
                if "-00001-of-" in suffix:
                    first_shard = symlink_path
            else:
                # Fallback for standard, non-sharded models (e.g., 0.8B models)
                symlink_filename = f"{name}-{quant}.gguf"
                symlink_path = os.path.join(SERVED_DIR, symlink_filename)
                os.symlink(f, symlink_path)
                logger.debug(f"Created standard symlink: {symlink_filename}")
                if not first_shard:
                    first_shard = symlink_path
                    
        # Point the config to the first shard; llama.cpp will auto-load the rest
        config.set(section_name, "model", first_shard or files[0])
        # ----------------------
        
        for k, v in params.items():
            if isinstance(v, (dict, list)): config.set(section_name, k, json.dumps(v))
            elif isinstance(v, bool): config.set(section_name, k, str(v).lower())
            else: config.set(section_name, k, str(v))
                
    with open(INI_PATH, 'w') as f: config.write(f)
    logger.info("models.ini successfully rewritten.")

def restart_llama_container():
    container_name = os.environ.get("LLAMA_CONTAINER_NAME", "llama-cpp")
    logger.info(f"Issuing restart command to container: {container_name}")
    try:
        subprocess.run(["curl", "-s", "--unix-socket", "/var/run/docker.sock", "-X", "POST", f"http://localhost/containers/{container_name}/restart"], check=True)
        logger.info(f"{container_name} restarted successfully.")
    except Exception as e:
        logger.error(f"Failed to restart {container_name}: {e}")

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
    script_code = f"""
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='''{req.hf_repo}''',
    allow_patterns=['''*{req.quant}.gguf''', '''*{req.quant}-*-of-*.gguf'''],
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
        os.close(slave); os.close(master)
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
        restart_llama_container()
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
                configs = [{"name": k, **v} for k, v in load_state().items()]
                await manager.broadcast(json.dumps({"type": "update", "data": configs}))
                
    broadcast_task = asyncio.create_task(broadcast_state())
    yield
    broadcast_task.cancel()

app = FastAPI(title="Local LLM Model Manager", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- API ENDPOINTS ---
@app.get("/")
async def serve_ui(): return FileResponse("index.html")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("favicon.ico")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/quants")
async def get_quants(repo: str):
    """Scans a HF repo for .gguf files, extracts tags, and sums file sizes (handles shards)."""
    try:
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        logger.info(f"Scanning repo '{repo}' for quant files...")
        info = api.model_info(repo, files_metadata=True)
        logger.info(f"Found {len(info.siblings)} files in the repo. Processing...")
        quants_dict = {}
        
        for f in info.siblings:
            if not f.rfilename.endswith(".gguf"): continue
            match = re.search(QUANT_REGEX, f.rfilename, re.IGNORECASE)
            if match:
                q_name = match.group(1).upper()
                q_size = f.size or 0
                quants_dict[q_name] = quants_dict.get(q_name, 0) + q_size
                logger.debug(f"Identified quant '{q_name}' in file '{f.rfilename}' with size {format_bytes(q_size)}")

        # Format sizes and sort by raw byte size (largest first)
        logger.info(f"Found {len(quants_dict)} unique quants in {repo}.")
        quants_list = [{"name": q, "size_str": format_bytes(s), "raw": s} for q, s in quants_dict.items()]
        quants_list.sort(key=lambda x: x["raw"], reverse=True)
        
        return {"quants": quants_list}
    except Exception as e:
        logger.error(f"Failed to fetch quants for {repo}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

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
        
    needs_download = req.symlink_name not in state or state[req.symlink_name]["repo"] != req.hf_repo or state[req.symlink_name]["quant"] != req.quant

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
                    match = re.search(QUANT_REGEX, f.rfilename, re.IGNORECASE)
                    if match and match.group(1).upper() == req.quant.upper():
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
        "repo": req.hf_repo, "quant": req.quant, "params": params_dict,
        "status": "downloading" if needs_download else "ready",
    }
    save_state(state)

    if needs_download:
        background_tasks.add_task(process_model, req, params_dict)
        return {"status": f"Provisioning started.{warning_msg}"}
    else:
        logger.info(f"Model {req.symlink_name} already downloaded. Updating config directly.")
        sync_system(state)
        restart_llama_container()
        return {"status": "Config updated and container restarted!"}

@app.delete("/api/configs/{symlink_name}")
async def delete_config(symlink_name: str):
    logger.info(f"Deleting config for {symlink_name}...")
    state = load_state()
    if symlink_name in state:
        del state[symlink_name]
        save_state(state)
        sync_system(state)
        restart_llama_container()
    return {"status": "Config deleted!"}
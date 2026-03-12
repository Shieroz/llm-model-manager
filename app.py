from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from huggingface_hub import snapshot_download, HfApi
from pydantic import BaseModel
import subprocess
import threading
import fnmatch
import time
import os
import glob
import configparser
import json
import shutil
import asyncio

CACHE_DIR = "/models/.cache"
SERVED_DIR = "/models/served"
INI_PATH = os.path.join(SERVED_DIR, "models.ini")
STATE_FILE = os.path.join(SERVED_DIR, "state.json")

class ModelSetup(BaseModel):
    hf_repo: str             
    quant: str               
    symlink_name: str             
    original_name: str = ""       
    parameters: str 

# --- UTILS & STATE MANAGEMENT ---
def format_bytes(bytes_num):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_num < 1024.0: return f"{bytes_num:.1f} {unit}"
        bytes_num /= 1024.0
    return f"{bytes_num:.1f} PB"

def get_dir_size(path):
    total = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp): total += os.path.getsize(fp)
    except: pass
    return total

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
    os.makedirs(SERVED_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(SERVED_DIR, "*.gguf")):
        try: os.unlink(f)
        except: pass
        
    config = configparser.ConfigParser()
    config.optionxform = str
    
    for name, data in state.items():
        if data.get("status") != "ready": continue 
            
        repo, quant, params = data["repo"], data["quant"], data["params"]
        search_pattern = f"{CACHE_DIR}/models--{repo.replace('/', '--')}/**/*{quant}*.gguf"
        files = sorted(glob.glob(search_pattern, recursive=True))
        
        if not files: continue 
            
        symlink_filename = f"{name}-{quant}.gguf"
        symlink_path = os.path.join(SERVED_DIR, symlink_filename)
        os.symlink(files[0], symlink_path)
        
        section_name = f"{name}-{quant}"
        config.add_section(section_name)
        config.set(section_name, "model", symlink_path)
        
        for k, v in params.items():
            if isinstance(v, (dict, list)): config.set(section_name, k, json.dumps(v))
            elif isinstance(v, bool): config.set(section_name, k, str(v).lower())
            else: config.set(section_name, k, str(v))
                
    with open(INI_PATH, 'w') as f: config.write(f)

def restart_llama_container():
    container_name = os.environ.get("LLAMA_CONTAINER_NAME", "llama-cpp")
    print(f"Restarting {container_name} to apply changes...")
    try: subprocess.run(["curl", "-s", "--unix-socket", "/var/run/docker.sock", "-X", "POST", f"http://localhost/containers/{container_name}/restart"], check=True)
    except Exception as e: print(f"Failed to restart {container_name}: {e}")

# --- WEBSOCKET MANAGER ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try: await connection.send_text(message)
            except: pass

manager = ConnectionManager()

# --- BACKGROUND WORKER WITH TELEMETRY ---
def process_model(req: ModelSetup, params_dict: dict, total_size: int):
    repo_cache_dir = os.path.join(CACHE_DIR, f"models--{req.hf_repo.replace('/', '--')}")
    baseline_size = get_dir_size(repo_cache_dir)
    download_status = {"running": True}

    def monitor():
        last_downloaded = 0
        last_time = time.time()
        while download_status["running"]:
            time.sleep(1)
            current_size = get_dir_size(repo_cache_dir)
            downloaded = max(0, current_size - baseline_size)
            if total_size > 0: downloaded = min(downloaded, total_size)
            
            now = time.time()
            speed = (downloaded - last_downloaded) / (now - last_time) if now - last_time > 0 else 0
            last_downloaded = downloaded
            last_time = now
            eta = (total_size - downloaded) / speed if speed > 0 and total_size > 0 else 0
            
            state = load_state()
            if req.symlink_name in state and state[req.symlink_name].get("status") == "downloading":
                state[req.symlink_name]["progress"] = {
                    "downloaded": downloaded, "total": total_size, "speed": speed, "eta": eta
                }
                save_state(state)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    try:
        snapshot_download(repo_id=req.hf_repo, allow_patterns=f"*{req.quant}*", cache_dir=CACHE_DIR)
        download_status["running"] = False
        monitor_thread.join(timeout=2)
        
        state = load_state()
        if req.symlink_name in state:
            state[req.symlink_name]["status"] = "ready"
            save_state(state)
            
        sync_system(state)
        restart_llama_container()
        print(f"Successfully provisioned: {req.symlink_name}")
        
    except Exception as e:
        download_status["running"] = False
        state = load_state()
        if req.symlink_name in state:
            state[req.symlink_name]["status"] = "error"
            state[req.symlink_name]["error_msg"] = str(e)
            save_state(state)

# --- LIFESPAN (AUTO-RESUME & WS BROADCASTER) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Resume interrupted downloads
    state = load_state()
    for name, data in state.items():
        if data.get("status") == "downloading":
            req = ModelSetup(hf_repo=data["repo"], quant=data["quant"], symlink_name=name, parameters=json.dumps(data["params"]))
            threading.Thread(target=process_model, args=(req, data["params"], data.get("expected_size", 0)), daemon=True).start()
    
    # 2. Start WebSocket Broadcast loop
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

# --- API ENDPOINTS ---
@app.get("/")
async def serve_ui():
    return FileResponse("index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True: await websocket.receive_text() # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/setup")
async def setup_model(req: ModelSetup, background_tasks: BackgroundTasks):
    req.symlink_name = req.symlink_name.replace("/", "-")
    if req.original_name: req.original_name = req.original_name.replace("/", "-")

    try: params_dict = json.loads(req.parameters) if req.parameters.strip() else {}
    except Exception as e: raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
    state = load_state()
    if req.original_name and req.original_name != req.symlink_name and req.original_name in state:
        del state[req.original_name]
        
    needs_download = req.symlink_name not in state or state[req.symlink_name]["repo"] != req.hf_repo or state[req.symlink_name]["quant"] != req.quant

    warning_msg = ""
    model_size = 0
    
    if needs_download:
        # Pre-flight Storage Check
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            api = HfApi(token=os.environ.get("HF_TOKEN"))
            info = api.model_info(req.hf_repo)
            model_size = sum(f.size for f in info.siblings if fnmatch.fnmatch(f.rfilename, f"*{req.quant}*") and f.size)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch model info from Hugging Face: {e}")

        total, used, free = shutil.disk_usage(CACHE_DIR)
        
        if model_size > free:
            raise HTTPException(status_code=400, detail=f"Storage Check Failed. Model requires {format_bytes(model_size)}, but only {format_bytes(free)} is available.")
            
        if (used + model_size) / total > 0.8:
            warning_msg = f" (Warning: This download will push your disk usage above 80%)"

    state[req.symlink_name] = {
        "repo": req.hf_repo, "quant": req.quant, "params": params_dict,
        "status": "downloading" if needs_download else "ready",
        "expected_size": model_size
    }
    save_state(state)

    if needs_download:
        background_tasks.add_task(process_model, req, params_dict, model_size)
        return {"status": f"Provisioning started.{warning_msg}"}
    else:
        sync_system(state)
        restart_llama_container()
        return {"status": "Config updated and container restarted!"}

@app.delete("/api/configs/{symlink_name}")
async def delete_config(symlink_name: str):
    state = load_state()
    if symlink_name in state:
        del state[symlink_name]
        save_state(state)
        sync_system(state)
        restart_llama_container()
    return {"status": "Config deleted!"}
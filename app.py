from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from huggingface_hub import snapshot_download
from pydantic import BaseModel
import subprocess
import threading
import os
import glob
import configparser
import json

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

# --- STATE MANAGEMENT ---
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except: pass
    return {}

def save_state(state):
    os.makedirs(SERVED_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def sync_system(state):
    os.makedirs(SERVED_DIR, exist_ok=True)
    
    # Clean old symlinks
    for f in glob.glob(os.path.join(SERVED_DIR, "*.gguf")):
        try: os.unlink(f)
        except: pass
        
    config = configparser.ConfigParser()
    config.optionxform = str
    
    for name, data in state.items():
        if data.get("status") != "ready":
            continue # Don't symlink models that are still downloading
            
        repo = data["repo"]
        quant = data["quant"]
        params = data["params"]
        
        search_pattern = f"{CACHE_DIR}/models--{repo.replace('/', '--')}/**/*{quant}*.gguf"
        files = sorted(glob.glob(search_pattern, recursive=True))
        
        if not files:
            continue 
            
        symlink_filename = f"{name}-{quant}.gguf"
        symlink_path = os.path.join(SERVED_DIR, symlink_filename)
        os.symlink(files[0], symlink_path)
        
        section_name = f"{name}-{quant}"
        config.add_section(section_name)
        config.set(section_name, "model", symlink_path)
        
        for k, v in params.items():
            if isinstance(v, (dict, list)):
                config.set(section_name, k, json.dumps(v))
            elif isinstance(v, bool):
                config.set(section_name, k, str(v).lower())
            else:
                config.set(section_name, k, str(v))
                
    with open(INI_PATH, 'w') as f:
        config.write(f)

def restart_llama_container():
    container_name = os.environ.get("LLAMA_CONTAINER_NAME", "llama-cpp")
    print(f"Restarting {container_name} to apply changes...")
    try:
        subprocess.run(["curl", "-s", "--unix-socket", "/var/run/docker.sock", "-X", "POST", f"http://localhost/containers/{container_name}/restart"], check=True)
    except Exception as e:
        print(f"Failed to restart {container_name}: {e}")

# --- BACKGROUND WORKER ---
def process_model(req: ModelSetup, params_dict: dict):
    print(f"Downloading {req.hf_repo} ({req.quant})...")
    try:
        snapshot_download(
            repo_id=req.hf_repo,
            allow_patterns=f"*{req.quant}*",
            cache_dir=CACHE_DIR
        )
        # Success: Update state to ready
        state = load_state()
        if req.symlink_name in state:
            state[req.symlink_name]["status"] = "ready"
            save_state(state)
            
        sync_system(state)
        restart_llama_container()
        print(f"Successfully provisioned: {req.symlink_name}")
        
    except Exception as e:
        # Failure: Update state to error
        print(f"Download failed for {req.symlink_name}: {e}")
        state = load_state()
        if req.symlink_name in state:
            state[req.symlink_name]["status"] = "error"
            state[req.symlink_name]["error_msg"] = str(e)
            save_state(state)

# --- STARTUP SWEEPER (AUTO-RESUME) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Look for interrupted downloads and resume them
    state = load_state()
    resumed = 0
    for name, data in state.items():
        if data.get("status") == "downloading":
            print(f"Auto-resuming interrupted download for: {name}")
            req = ModelSetup(hf_repo=data["repo"], quant=data["quant"], symlink_name=name, parameters=json.dumps(data["params"]))
            # Spin up a daemon thread so it doesn't block the FastAPI boot sequence
            threading.Thread(target=process_model, args=(req, data["params"]), daemon=True).start()
            resumed += 1
    if resumed > 0:
        print(f"Resumed {resumed} background downloads.")
    yield
    # Shutdown logic goes here (if needed)

app = FastAPI(title="Local LLM Model Manager", lifespan=lifespan)

# --- API ENDPOINTS ---
@app.post("/api/setup")
async def setup_model(req: ModelSetup, background_tasks: BackgroundTasks):
    req.symlink_name = req.symlink_name.replace("/", "-")
    if req.original_name:
        req.original_name = req.original_name.replace("/", "-")

    try: params_dict = json.loads(req.parameters) if req.parameters.strip() else {}
    except Exception as e: raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
    state = load_state()
    
    # Handle renaming cleanup
    if req.original_name and req.original_name != req.symlink_name and req.original_name in state:
        del state[req.original_name]
        
    # Check if this requires a download (new deployment or changing repo/quant)
    needs_download = False
    if req.symlink_name not in state or state[req.symlink_name]["repo"] != req.hf_repo or state[req.symlink_name]["quant"] != req.quant:
        needs_download = True

    # Save state immediately
    state[req.symlink_name] = {
        "repo": req.hf_repo,
        "quant": req.quant,
        "params": params_dict,
        "status": "downloading" if needs_download else "ready"
    }
    save_state(state)

    if needs_download:
        background_tasks.add_task(process_model, req, params_dict)
        return {"status": "Provisioning... Container will restart when finished."}
    else:
        # Instant param update
        sync_system(state)
        restart_llama_container()
        return {"status": "Config updated and container restarted!"}

@app.get("/api/configs")
async def get_configs():
    return [{"name": k, **v} for k, v in load_state().items()]

@app.delete("/api/configs/{symlink_name}")
async def delete_config(symlink_name: str):
    state = load_state()
    if symlink_name in state:
        del state[symlink_name]
        save_state(state)
        sync_system(state)
        restart_llama_container()
    return {"status": "Config deleted!"}

# --- FRONTEND WEB UI ---
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LLM Config Manager</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script>tailwind.config = { darkMode: 'class' }</script>
    </head>
    <body class="bg-gray-900 text-gray-100 p-8 font-sans">
        <div class="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-12 gap-8">
            <div class="md:col-span-5 bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700 h-fit sticky top-8">
                <h2 id="formTitle" class="text-2xl font-bold mb-4 text-blue-400">Deploy New Config</h2>
                <form id="setupForm" class="space-y-4">
                    <input type="hidden" id="original_name" value="">
                    <div><label class="block text-sm text-gray-400">HF Repo *</label><input type="text" id="hf_repo" required class="mt-1 w-full bg-gray-700 border-gray-600 rounded p-2 text-white focus:ring-blue-500"></div>
                    <div><label class="block text-sm text-gray-400">Quantization *</label><input type="text" id="quant" required class="mt-1 w-full bg-gray-700 border-gray-600 rounded p-2 text-white focus:ring-blue-500"></div>
                    <div><label class="block text-sm text-gray-400">Symlink Name *</label><input type="text" id="symlink_name" required class="mt-1 w-full bg-gray-700 border-gray-600 rounded p-2 text-white focus:ring-blue-500"></div>
                    <div><label class="block text-sm text-gray-400">Parameters (JSON) *</label><textarea id="parameters" required rows="8" class="mt-1 w-full bg-gray-700 border-gray-600 rounded p-2 text-white font-mono text-sm">{"temp": 0.6}</textarea></div>
                    <div class="flex gap-2">
                        <button type="submit" id="submitBtn" class="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition">Provision Model</button>
                        <button type="button" id="clearBtn" onclick="resetForm()" class="bg-gray-600 hover:bg-gray-500 text-white py-2 px-4 rounded transition">Clear</button>
                    </div>
                    <p id="statusMsg" class="text-sm mt-2 hidden"></p>
                </form>
            </div>
            <div class="md:col-span-7">
                <div class="flex justify-between items-end mb-4">
                    <h2 class="text-2xl font-bold text-green-400">Active Configurations</h2>
                    <span class="text-xs text-gray-500 flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span> Auto-Sync Active</span>
                </div>
                <div id="configList" class="space-y-4"></div>
            </div>
        </div>

        <script>
            let currentConfigs = [];
            async function loadConfigs() {
                try {
                    const res = await fetch('/api/configs');
                    currentConfigs = await res.json();
                    renderConfigs();
                } catch (e) { console.error("Polling error"); }
            }

            function renderConfigs() {
                const list = document.getElementById('configList');
                list.innerHTML = '';
                if (currentConfigs.length === 0) { list.innerHTML = '<p class="text-gray-500 italic">No configs active.</p>'; return; }
                
                currentConfigs.forEach(conf => {
                    const isDownloading = conf.status === 'downloading';
                    const isError = conf.status === 'error';
                    const card = document.createElement('div');
                    
                    let statusBadge = '';
                    if (isDownloading) statusBadge = `<span class="text-xs bg-yellow-900 text-yellow-300 px-2 py-0.5 rounded border border-yellow-700 animate-pulse">Downloading...</span>`;
                    else if (isError) statusBadge = `<span class="text-xs bg-red-900 text-red-300 px-2 py-0.5 rounded border border-red-700">Error</span>`;

                    const paramsHtml = Object.entries(conf.params).map(([k, v]) => 
                        `<span class="bg-gray-900 border border-gray-600 text-xs px-2 py-1 rounded mr-1 mb-1 inline-block"><span class="text-blue-300">${k}:</span> <span class="font-mono text-gray-300">${typeof v === 'object' ? JSON.stringify(v) : v}</span></span>`
                    ).join('');

                    card.className = `bg-gray-800 p-4 rounded-lg shadow border ${isDownloading ? 'border-yellow-600 opacity-80' : isError ? 'border-red-600' : 'border-gray-700 hover:border-blue-500'} flex justify-between items-start transition duration-200`;
                    
                    let buttons = '';
                    if (isDownloading) {
                        buttons = `<p class="text-xs text-yellow-500 mt-2 text-right">Building cache...</p>`;
                    } else if (isError) {
                         buttons = `<p class="text-xs text-red-400 mb-2 max-w-xs break-words">${conf.error_msg || "Download failed"}</p>
                                    <button onclick="editConfig('${conf.name}')" class="bg-yellow-600 hover:bg-yellow-500 text-white text-xs py-1 px-3 rounded transition w-full mb-1">Retry/Edit</button>
                                    <button onclick="deleteConfig('${conf.name}')" class="bg-red-600 hover:bg-red-500 text-white text-xs py-1 px-3 rounded transition w-full">Delete</button>`;
                    } else {
                        buttons = `
                            <button onclick="duplicateConfig('${conf.name}')" class="bg-green-600 hover:bg-green-500 text-white text-xs py-1 px-3 rounded transition">Duplicate</button>
                            <button onclick="editConfig('${conf.name}')" class="bg-blue-600 hover:bg-blue-500 text-white text-xs py-1 px-3 rounded transition mt-1">Edit</button>
                            <button onclick="deleteConfig('${conf.name}')" class="bg-red-600 hover:bg-red-500 text-white text-xs py-1 px-3 rounded transition mt-1">Delete</button>`;
                    }

                    card.innerHTML = `
                        <div class="flex-1">
                            <h3 class="text-lg font-bold text-white flex items-center gap-2">
                                ${conf.name}
                                <span class="text-xs bg-blue-900 text-blue-300 px-2 py-0.5 rounded border border-blue-700 font-mono tracking-wider">${conf.quant}</span>
                                ${statusBadge}
                            </h3>
                            <p class="text-xs text-gray-500 mb-3 font-mono mt-1">Repo: ${conf.repo}</p>
                            <div class="flex flex-wrap mt-2">${paramsHtml}</div>
                        </div>
                        <div class="flex flex-col ml-4 min-w-[80px]">${buttons}</div>
                    `;
                    list.appendChild(card);
                });
            }

            function duplicateConfig(name) {
                const conf = currentConfigs.find(c => c.name === name);
                if(!conf) return;
                resetForm();
                document.getElementById('formTitle').textContent = `Duplicate Config: ${name}`;
                document.getElementById('formTitle').className = "text-2xl font-bold mb-4 text-green-400";
                document.getElementById('hf_repo').value = conf.repo;
                document.getElementById('quant').value = conf.quant;
                document.getElementById('symlink_name').value = conf.name + "-COPY";
                document.getElementById('submitBtn').textContent = "Provision Duplicate";
                document.getElementById('submitBtn').className = "flex-1 bg-green-600 hover:bg-green-500 text-white font-bold py-2 px-4 rounded transition";
                document.getElementById('clearBtn').textContent = "Cancel";
                document.getElementById('parameters').value = JSON.stringify(conf.params, null, 2);
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }

            function editConfig(name) {
                const conf = currentConfigs.find(c => c.name === name);
                if(!conf) return;
                document.getElementById('formTitle').textContent = `Edit Config: ${name}`;
                document.getElementById('formTitle').className = "text-2xl font-bold mb-4 text-yellow-400";
                document.getElementById('hf_repo').value = conf.repo;
                document.getElementById('quant').value = conf.quant;
                document.getElementById('symlink_name').value = conf.name;
                document.getElementById('original_name').value = conf.name;
                document.getElementById('submitBtn').textContent = "Update Config";
                document.getElementById('submitBtn').className = "flex-1 bg-yellow-600 hover:bg-yellow-500 text-white font-bold py-2 px-4 rounded transition";
                document.getElementById('clearBtn').textContent = "Cancel Edit";
                document.getElementById('clearBtn').className = "bg-red-900 hover:bg-red-800 text-white py-2 px-4 rounded transition";
                document.getElementById('parameters').value = JSON.stringify(conf.params, null, 2);
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }

            function resetForm() {
                document.getElementById('formTitle').textContent = "Deploy New Config";
                document.getElementById('formTitle').className = "text-2xl font-bold mb-4 text-blue-400";
                document.getElementById('setupForm').reset();
                document.getElementById('original_name').value = ''; 
                document.getElementById('submitBtn').textContent = "Provision Model";
                document.getElementById('submitBtn').className = "flex-1 bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition";
                document.getElementById('clearBtn').textContent = "Clear";
                document.getElementById('clearBtn').className = "bg-gray-600 hover:bg-gray-500 text-white py-2 px-4 rounded transition";
                document.getElementById('parameters').value = '{\\n  "temp": 0.6,\\n  "top-p": 0.95\\n}';
                document.getElementById('statusMsg').className = "hidden";
            }

            async function deleteConfig(name) {
                if(confirm(`Delete ${name}?`)) {
                    await fetch(`/api/configs/${name}`, { method: 'DELETE' });
                    loadConfigs();
                    if(document.getElementById('original_name').value === name) resetForm();
                }
            }

            document.getElementById('setupForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const btn = document.getElementById('submitBtn');
                const status = document.getElementById('statusMsg');
                btn.disabled = true; btn.classList.add('opacity-50');
                status.className = "text-sm mt-2 text-yellow-400 block animate-pulse";
                status.textContent = "Processing request...";

                const payload = {
                    hf_repo: document.getElementById('hf_repo').value,
                    quant: document.getElementById('quant').value,
                    symlink_name: document.getElementById('symlink_name').value,
                    original_name: document.getElementById('original_name').value,
                    parameters: document.getElementById('parameters').value
                };

                const res = await fetch('/api/setup', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
                const data = await res.json();
                if(res.ok) {
                    status.className = "text-sm mt-2 text-green-400 block";
                    status.textContent = data.status;
                    setTimeout(() => { resetForm(); btn.disabled = false; btn.classList.remove('opacity-50'); }, 2000);
                    loadConfigs();
                } else {
                    status.className = "text-sm mt-2 text-red-400 block";
                    status.textContent = "Error: " + (data.detail || "Invalid parameters");
                    btn.disabled = false; btn.classList.remove('opacity-50');
                }
            });

            loadConfigs();
            setInterval(loadConfigs, 3000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
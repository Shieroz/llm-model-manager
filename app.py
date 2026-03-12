from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel
import subprocess
import os
import glob
import configparser
import json

app = FastAPI(title="Local LLM Model Manager")

CACHE_DIR = "/models/.cache"
SERVED_DIR = "/models/served"
INI_PATH = os.path.join(SERVED_DIR, "models.ini")

class ModelSetup(BaseModel):
    hf_repo: str = ""
    quant: str = ""
    symlink_name: str
    original_name: str = ""       # Tracks the old name during a rename operation
    parameters: str

def restart_llama_container():
    """Dynamically restarts the target container using the compose environment variable."""
    container_name = os.environ.get("LLAMA_CONTAINER_NAME", "llama-cpp")
    print(f"Restarting {container_name} to apply changes...")
    try:
        subprocess.run([
            "curl", "-s", "--unix-socket", "/var/run/docker.sock", 
            "-X", "POST", f"http://localhost/containers/{container_name}/restart"
        ], check=True)
        print(f"{container_name} restarted successfully.")
    except Exception as e:
        print(f"Failed to restart {container_name}: {e}")

def write_to_ini(symlink_name: str, params_dict: dict):
    os.makedirs(SERVED_DIR, exist_ok=True)
    config = configparser.ConfigParser()
    config.optionxform = str 
    
    if os.path.exists(INI_PATH):
        config.read(INI_PATH)
    
    section_name = symlink_name 
    
    if config.has_section(section_name):
        config.remove_section(section_name)
    config.add_section(section_name)
    
    symlink_path = os.path.join(SERVED_DIR, f"{symlink_name}.gguf")
    config.set(section_name, "model", symlink_path)
    
    for key, value in params_dict.items():
        if isinstance(value, (dict, list)):
            config.set(section_name, key, json.dumps(value))
        elif isinstance(value, bool):
            config.set(section_name, key, str(value).lower())
        else:
            config.set(section_name, key, str(value))
            
    with open(INI_PATH, 'w') as configfile:
        config.write(configfile)

def process_model(req: ModelSetup, params_dict: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SERVED_DIR, exist_ok=True)

    print(f"Downloading {req.hf_repo} ({req.quant})...")
    snapshot_download(
        repo_id=req.hf_repo,
        allow_patterns=f"*{req.quant}*",
        cache_dir=CACHE_DIR
    )

    search_pattern = f"{CACHE_DIR}/models--{req.hf_repo.replace('/', '--')}/**/*{req.quant}*.gguf"
    files = sorted(glob.glob(search_pattern, recursive=True))
    
    if not files:
        raise FileNotFoundError("Model downloaded, but GGUF file not found in cache.")

    target_file = files[0]
    symlink_path = os.path.join(SERVED_DIR, f"{req.symlink_name}.gguf")

    if os.path.exists(symlink_path) or os.path.islink(symlink_path):
        os.unlink(symlink_path)
    os.symlink(target_file, symlink_path)

    write_to_ini(req.symlink_name, params_dict)
    print(f"Successfully provisioned: {req.symlink_name}")
    restart_llama_container()

@app.post("/api/setup")
async def setup_model(req: ModelSetup, background_tasks: BackgroundTasks):
    req.symlink_name = req.symlink_name.replace("/", "-")
    if req.original_name:
        req.original_name = req.original_name.replace("/", "-")

    try:
        params_dict = json.loads(req.parameters) if req.parameters.strip() else {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
    # If repo is provided, it's a new download/provision
    if req.hf_repo and req.quant:
        background_tasks.add_task(process_model, req, params_dict)
        return {"status": "Provisioning in background. Container will restart when finished."}
    
    # --- EDIT MODE ---
    target_name = req.symlink_name
    old_name = req.original_name
    
    # Handle Renaming
    if old_name and old_name != target_name:
        old_symlink_path = os.path.join(SERVED_DIR, f"{old_name}.gguf")
        new_symlink_path = os.path.join(SERVED_DIR, f"{target_name}.gguf")
        
        if not os.path.islink(old_symlink_path):
            raise HTTPException(status_code=404, detail="Original symlink not found for renaming.")
            
        target_file = os.readlink(old_symlink_path)
        
        if os.path.exists(new_symlink_path) or os.path.islink(new_symlink_path):
            os.unlink(new_symlink_path)
        os.symlink(target_file, new_symlink_path)
        
        os.unlink(old_symlink_path)
        
        config = configparser.ConfigParser()
        config.optionxform = str
        if os.path.exists(INI_PATH):
            config.read(INI_PATH)
            if config.has_section(old_name):
                config.remove_section(old_name)
                with open(INI_PATH, 'w') as configfile:
                    config.write(configfile)
    else:
        # Standard Edit (No Rename)
        symlink_path = os.path.join(SERVED_DIR, f"{target_name}.gguf")
        if not os.path.exists(symlink_path):
            raise HTTPException(status_code=404, detail="Symlink does not exist.")
            
    write_to_ini(target_name, params_dict)
    restart_llama_container()
    return {"status": "Config updated and container restarted!"}

@app.get("/api/configs")
async def get_configs():
    config = configparser.ConfigParser()
    config.optionxform = str
    if os.path.exists(INI_PATH):
        config.read(INI_PATH)
    
    configs = []
    for section in config.sections():
        params = dict(config.items(section))
        file_path = params.pop("model", f"{section}.gguf") 
        configs.append({"name": section, "file": file_path, "params": params})
    return configs

@app.delete("/api/configs/{symlink_name}")
async def delete_config(symlink_name: str):
    file_name = f"{symlink_name}.gguf"
    symlink_path = os.path.join(SERVED_DIR, file_name)
    
    if os.path.exists(symlink_path):
        os.unlink(symlink_path)
        
    config = configparser.ConfigParser()
    config.optionxform = str
    if os.path.exists(INI_PATH):
        config.read(INI_PATH)
        if config.has_section(symlink_name):
            config.remove_section(symlink_name)
            with open(INI_PATH, 'w') as configfile:
                config.write(configfile)
                
    restart_llama_container()
    return {"status": "Config deleted and container restarted!"}

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
                    <div>
                        <label class="block text-sm font-medium text-gray-400">HF Repo <span class="text-xs text-gray-500">(Leave blank to edit existing)</span></label>
                        <input type="text" id="hf_repo" class="mt-1 w-full bg-gray-700 border border-gray-600 rounded p-2 text-white focus:ring-blue-500">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-400">Quantization <span class="text-xs text-gray-500">(Leave blank to edit)</span></label>
                        <input type="text" id="quant" class="mt-1 w-full bg-gray-700 border border-gray-600 rounded p-2 text-white focus:ring-blue-500">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-400">Symlink Name *</label>
                        <input type="text" id="symlink_name" required class="mt-1 w-full bg-gray-700 border border-gray-600 rounded p-2 text-white focus:ring-blue-500">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-400">Parameters (JSON) *</label>
                        <textarea id="parameters" required rows="8" class="mt-1 w-full bg-gray-700 border border-gray-600 rounded p-2 text-white focus:ring-blue-500 font-mono text-sm">{"temp": 0.6}</textarea>
                    </div>
                    <div class="flex gap-2">
                        <button type="submit" id="submitBtn" class="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition">
                            Provision Model
                        </button>
                        <button type="button" id="clearBtn" onclick="resetForm()" class="bg-gray-600 hover:bg-gray-500 text-white py-2 px-4 rounded transition">
                            Clear
                        </button>
                    </div>
                    <p id="statusMsg" class="text-sm mt-2 hidden"></p>
                </form>
            </div>

            <div class="md:col-span-7">
                <div class="flex justify-between items-end mb-4">
                    <h2 class="text-2xl font-bold text-green-400">Active Configurations</h2>
                    <span class="text-xs text-gray-500 flex items-center gap-1">
                        <span class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span> Auto-Sync Active
                    </span>
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
                } catch (e) { console.error("Polling error", e); }
            }

            function renderConfigs() {
                const list = document.getElementById('configList');
                list.innerHTML = '';
                if (currentConfigs.length === 0) {
                    list.innerHTML = '<p class="text-gray-500 italic">No configurations currently active.</p>';
                    return;
                }
                currentConfigs.forEach(conf => {
                    const card = document.createElement('div');
                    card.className = "bg-gray-800 p-4 rounded-lg shadow border border-gray-700 flex justify-between items-start hover:border-blue-500 transition duration-200";
                    const paramsHtml = Object.entries(conf.params)
                        .map(([k, v]) => `<span class="bg-gray-900 border border-gray-600 text-xs px-2 py-1 rounded mr-1 mb-1 inline-block"><span class="text-blue-300">${k}:</span> <span class="font-mono text-gray-300">${v}</span></span>`)
                        .join('');
                    card.innerHTML = `
                        <div class="flex-1">
                            <h3 class="text-lg font-bold text-white">${conf.name}</h3>
                            <p class="text-xs text-gray-500 mb-3 font-mono">${conf.file}</p>
                            <div class="flex flex-wrap mt-2">${paramsHtml}</div>
                        </div>
                        <div class="flex flex-col gap-2 ml-4">
                            <button onclick="editConfig('${conf.name}')" class="bg-blue-600 hover:bg-blue-500 text-white text-xs py-1 px-3 rounded transition">Edit</button>
                            <button onclick="deleteConfig('${conf.name}')" class="bg-red-600 hover:bg-red-500 text-white text-xs py-1 px-3 rounded transition">Delete</button>
                        </div>
                    `;
                    list.appendChild(card);
                });
            }

            function editConfig(name) {
                const conf = currentConfigs.find(c => c.name === name);
                if(!conf) return;
                
                document.getElementById('formTitle').textContent = `Edit Config: ${name}`;
                document.getElementById('formTitle').className = "text-2xl font-bold mb-4 text-yellow-400";
                document.getElementById('hf_repo').value = '';
                document.getElementById('quant').value = '';
                document.getElementById('symlink_name').value = conf.name;
                document.getElementById('original_name').value = conf.name; // Track the original name
                
                document.getElementById('submitBtn').textContent = "Update Parameters";
                document.getElementById('submitBtn').className = "flex-1 bg-yellow-600 hover:bg-yellow-500 text-white font-bold py-2 px-4 rounded transition";
                
                document.getElementById('clearBtn').textContent = "Cancel Edit";
                document.getElementById('clearBtn').className = "bg-red-900 hover:bg-red-800 text-white py-2 px-4 rounded transition";
                
                let paramsObj = {};
                for (let [k, v] of Object.entries(conf.params)) {
                    try { paramsObj[k] = JSON.parse(v); } 
                    catch(e) { 
                        if (!isNaN(v) && v.trim() !== '') paramsObj[k] = Number(v);
                        else if (v.toLowerCase() === 'true') paramsObj[k] = true;
                        else if (v.toLowerCase() === 'false') paramsObj[k] = false;
                        else paramsObj[k] = v; 
                    }
                }
                document.getElementById('parameters').value = JSON.stringify(paramsObj, null, 2);
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }

            function resetForm() {
                document.getElementById('formTitle').textContent = "Deploy New Config";
                document.getElementById('formTitle').className = "text-2xl font-bold mb-4 text-blue-400";
                document.getElementById('setupForm').reset();
                document.getElementById('original_name').value = ''; // Clear tracker
                
                document.getElementById('submitBtn').textContent = "Provision Model";
                document.getElementById('submitBtn').className = "flex-1 bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition";
                
                document.getElementById('clearBtn').textContent = "Clear";
                document.getElementById('clearBtn').className = "bg-gray-600 hover:bg-gray-500 text-white py-2 px-4 rounded transition";
                
                document.getElementById('parameters').value = '{\\n  "temp": 0.6,\\n  "top-p": 0.95\\n}';
                document.getElementById('statusMsg').className = "hidden";
            }

            async function deleteConfig(name) {
                if(confirm(`Delete the ${name} config? (Model files remain safely cached)`)) {
                    await fetch(`/api/configs/${name}`, { method: 'DELETE' });
                    loadConfigs();
                    if(document.getElementById('original_name').value === name) resetForm();
                }
            }

            document.getElementById('setupForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const btn = document.getElementById('submitBtn');
                const status = document.getElementById('statusMsg');
                
                btn.disabled = true;
                btn.classList.add('opacity-50');
                status.className = "text-sm mt-2 text-yellow-400 block animate-pulse";
                status.textContent = "Processing request...";

                const payload = {
                    hf_repo: document.getElementById('hf_repo').value,
                    quant: document.getElementById('quant').value,
                    symlink_name: document.getElementById('symlink_name').value,
                    original_name: document.getElementById('original_name').value,
                    parameters: document.getElementById('parameters').value
                };

                const res = await fetch('/api/setup', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await res.json();

                if(res.ok) {
                    status.className = "text-sm mt-2 text-green-400 block";
                    status.textContent = data.status;
                    setTimeout(() => { 
                        resetForm(); 
                        btn.disabled = false; 
                        btn.classList.remove('opacity-50'); 
                    }, 2000);
                    loadConfigs();
                } else {
                    status.className = "text-sm mt-2 text-red-400 block";
                    status.textContent = "Error: " + (data.detail || "Invalid parameters");
                    btn.disabled = false; 
                    btn.classList.remove('opacity-50');
                }
            });

            loadConfigs();
            setInterval(loadConfigs, 3000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
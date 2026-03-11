from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
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
    hf_repo: str                  
    quant: str                    
    symlink_name: str             
    parameters: str # Received as a JSON string from the UI form         

def process_model(req: ModelSetup):
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SERVED_DIR, exist_ok=True)

    print(f"Checking cache or downloading {req.hf_repo} ({req.quant})...")
    download_cmd = [
        "huggingface-cli", "download", req.hf_repo,
        "--include", f"*{req.quant}*",
        "--cache-dir", CACHE_DIR
    ]
    subprocess.run(download_cmd, check=True)

    search_pattern = f"{CACHE_DIR}/models--{req.hf_repo.replace('/', '--')}/**/*{req.quant}*.gguf"
    files = sorted(glob.glob(search_pattern, recursive=True))
    
    if not files:
        raise FileNotFoundError("Model downloaded, but GGUF file not found in cache.")

    target_file = files[0]
    symlink_path = os.path.join(SERVED_DIR, f"{req.symlink_name}.gguf")

    if os.path.exists(symlink_path) or os.path.islink(symlink_path):
        os.unlink(symlink_path)
    os.symlink(target_file, symlink_path)

    config = configparser.ConfigParser()
    if os.path.exists(INI_PATH):
        config.read(INI_PATH)
    
    section_name = f"{req.symlink_name}.gguf"
    if not config.has_section(section_name):
        config.add_section(section_name)
    
    # Parse the JSON string from the UI into a dictionary
    params_dict = json.loads(req.parameters) if req.parameters else {}
    for key, value in params_dict.items():
        config.set(section_name, key, str(value))
        
    with open(INI_PATH, 'w') as configfile:
        config.write(configfile)
    print(f"Successfully provisioned: {section_name}")

@app.post("/api/setup")
async def setup_model(req: ModelSetup, background_tasks: BackgroundTasks):
    try:
        json.loads(req.parameters) # Validate JSON before starting
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON in parameters")
    background_tasks.add_task(process_model, req)
    return {"status": "Provisioning in background. This may take a while if downloading."}

@app.get("/api/configs")
async def get_configs():
    config = configparser.ConfigParser()
    if os.path.exists(INI_PATH):
        config.read(INI_PATH)
    
    configs = []
    for section in config.sections():
        # Strip the .gguf extension for a cleaner UI display
        clean_name = section.replace(".gguf", "")
        configs.append({
            "name": clean_name,
            "file": section,
            "params": dict(config.items(section))
        })
    return configs

@app.delete("/api/configs/{symlink_name}")
async def delete_config(symlink_name: str):
    file_name = f"{symlink_name}.gguf"
    symlink_path = os.path.join(SERVED_DIR, file_name)
    
    if os.path.exists(symlink_path):
        os.unlink(symlink_path)
        
    config = configparser.ConfigParser()
    if os.path.exists(INI_PATH):
        config.read(INI_PATH)
        if config.has_section(file_name):
            config.remove_section(file_name)
            with open(INI_PATH, 'w') as configfile:
                config.write(configfile)
                
    return {"status": "Config deleted"}

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
        <script>
            tailwind.config = { darkMode: 'class' }
        </script>
    </head>
    <body class="bg-gray-900 text-gray-100 p-8 font-sans">
        <div class="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8">
            
            <div class="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
                <h2 class="text-2xl font-bold mb-4 text-blue-400">Deploy New Config</h2>
                <form id="setupForm" class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-400">HF Repo (e.g., unsloth/Qwen3.5-122B-A10B-GGUF)</label>
                        <input type="text" id="hf_repo" required class="mt-1 w-full bg-gray-700 border border-gray-600 rounded p-2 text-white focus:ring-blue-500 focus:border-blue-500">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-400">Quantization (e.g., UD-Q4_K_XL)</label>
                        <input type="text" id="quant" required class="mt-1 w-full bg-gray-700 border border-gray-600 rounded p-2 text-white focus:ring-blue-500">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-400">Symlink Name (e.g., Qwen-Coding)</label>
                        <input type="text" id="symlink_name" required class="mt-1 w-full bg-gray-700 border border-gray-600 rounded p-2 text-white focus:ring-blue-500">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-400">Parameters (JSON format)</label>
                        <textarea id="parameters" rows="4" class="mt-1 w-full bg-gray-700 border border-gray-600 rounded p-2 text-white focus:ring-blue-500 font-mono text-sm">{"temp": 0.1, "top-p": 0.95, "top-k": 40}</textarea>
                    </div>
                    <button type="submit" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition">
                        Provision Model
                    </button>
                    <p id="statusMsg" class="text-sm mt-2 hidden"></p>
                </form>
            </div>

            <div>
                <h2 class="text-2xl font-bold mb-4 text-green-400">Active Configurations</h2>
                <button onclick="loadConfigs()" class="mb-4 text-sm bg-gray-700 hover:bg-gray-600 text-white py-1 px-3 rounded transition">↻ Refresh List</button>
                <div id="configList" class="space-y-4">
                    </div>
            </div>
        </div>

        <script>
            async function loadConfigs() {
                const res = await fetch('/api/configs');
                const configs = await res.json();
                const list = document.getElementById('configList');
                list.innerHTML = '';
                
                if (configs.length === 0) {
                    list.innerHTML = '<p class="text-gray-500 italic">No configurations currently active.</p>';
                    return;
                }

                configs.forEach(conf => {
                    const card = document.createElement('div');
                    card.className = "bg-gray-800 p-4 rounded-lg shadow border border-gray-700 flex justify-between items-start";
                    
                    const paramsHtml = Object.entries(conf.params)
                        .map(([k, v]) => `<span class="bg-gray-700 text-xs px-2 py-1 rounded mr-1 mb-1 inline-block">${k}: ${v}</span>`)
                        .join('');

                    card.innerHTML = `
                        <div class="flex-1">
                            <h3 class="text-lg font-bold text-white">${conf.name}</h3>
                            <p class="text-xs text-gray-400 mb-2 font-mono">${conf.file}</p>
                            <div class="flex flex-wrap mt-2">${paramsHtml}</div>
                        </div>
                        <button onclick="deleteConfig('${conf.name}')" class="ml-4 bg-red-600 hover:bg-red-700 text-white text-sm py-1 px-3 rounded transition">Delete</button>
                    `;
                    list.appendChild(card);
                });
            }

            async function deleteConfig(name) {
                if(confirm(`Are you sure you want to delete the ${name} config? (This does not delete the 75GB model file from cache)`)) {
                    await fetch(`/api/configs/${name}`, { method: 'DELETE' });
                    loadConfigs();
                }
            }

            document.getElementById('setupForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const btn = e.target.querySelector('button');
                const status = document.getElementById('statusMsg');
                
                btn.disabled = true;
                btn.classList.add('opacity-50');
                status.className = "text-sm mt-2 text-yellow-400 block";
                status.textContent = "Request sent. Check Docker logs for download progress.";

                const payload = {
                    hf_repo: document.getElementById('hf_repo').value,
                    quant: document.getElementById('quant').value,
                    symlink_name: document.getElementById('symlink_name').value,
                    parameters: document.getElementById('parameters').value
                };

                const res = await fetch('/api/setup', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if(res.ok) {
                    status.className = "text-sm mt-2 text-green-400 block";
                    status.textContent = "Processing started successfully!";
                    setTimeout(() => { loadConfigs(); btn.disabled = false; btn.classList.remove('opacity-50'); }, 2000);
                } else {
                    status.className = "text-sm mt-2 text-red-400 block";
                    status.textContent = "Error: Invalid JSON parameters.";
                    btn.disabled = false; btn.classList.remove('opacity-50');
                }
            });

            // Initial load
            loadConfigs();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
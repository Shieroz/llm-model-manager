import logging
import os
import sys
import threading

import asyncio
import json

from contextlib import asynccontextmanager
from fastapi import BackgroundTasks, FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api import (
    delete_config,
    delete_revision,
    get_local_models,
    get_quants,
    setup_model,
    toggle_rpc,
    api_get_commits,
    websocket_endpoint,
)
from backend.models import ModelSetup, RevisionDeleteReq, RpcModeReq
from backend.state import iter_configs, load_state, save_state
from backend.sync import sync_system
from backend.websocket import manager

log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.info(f"Model Manager booted. Log level set to: {log_level_str}")


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    state = load_state()
    from backend.cache import prune_unreferenced_revisions

    prune_unreferenced_revisions(state)
    sync_system(state, restart=False)

    resumed = 0
    for name, data in iter_configs(state):
        if data.get("status") == "downloading" and data.get("revision"):
            resumed += 1
            from backend.download import process_model
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

    if resumed > 0:
        logger.info(f"Resumed {resumed} interrupted download(s).")

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


# --- App ---
app = FastAPI(title="Local LLM Model Manager", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


# --- UI Routes ---
@app.get("/")
async def serve_ui():
    return FileResponse("frontend/index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("frontend/favicon.ico")


# --- WebSocket ---
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket_endpoint(websocket)


# --- API Routes ---
@app.get("/api/quants")
async def api_quants(repo: str):
    return await get_quants(repo)


@app.get("/api/commits")
async def api_commits(repo: str):
    return await api_get_commits(repo)


@app.get("/api/models")
async def api_models():
    return await get_local_models()


@app.post("/api/rpc_mode")
async def api_rpc(req: RpcModeReq):
    return await toggle_rpc(req)


@app.post("/api/revisions/delete")
async def api_revision_delete(req: RevisionDeleteReq):
    return await delete_revision(req)


@app.post("/api/setup")
async def api_setup(req: ModelSetup, background_tasks: BackgroundTasks):
    return await setup_model(req, background_tasks)


@app.delete("/api/configs/{symlink_name}")
async def api_config_delete(symlink_name: str):
    return await delete_config(symlink_name)

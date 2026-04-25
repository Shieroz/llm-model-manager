import json
import logging
import os
import re
import shutil

from fastapi import BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect

from backend.cache import scan_cache, prune_unreferenced_revisions
from backend.config import QUANT_REGEX
from backend.hf_hub import get_branches, get_commits, pre_flight_size, resolve_sha
from backend.models import ModelSetup, RevisionDeleteReq, RpcModeReq
from backend.state import format_bytes, iter_configs, load_state, save_state
from backend.sync import sync_system
from backend.websocket import manager

log = logging.getLogger(__name__)


# --- WebSocket ---
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# --- API: Quants ---
async def get_quants(repo: str):
    from huggingface_hub import HfApi
    try:
        api = HfApi(token=None)
        log.info(f"Scanning repo '{repo}' for quant files...")
        info = api.model_info(repo, files_metadata=True)
        quants_dict = {}
        mmproj_dict = {}

        for f in info.siblings:
            if not f.rfilename.endswith(".gguf"):
                continue
            match = re.search(QUANT_REGEX, f.rfilename, re.IGNORECASE)
            if not match:
                continue
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

        repo_name = repo.rsplit("/", 1)[-1] if "/" in repo else repo
        return {"quants": quants_list, "mmprojs": mmproj_list, "repoName": repo_name}
    except Exception as e:
        log.error(f"Failed to fetch quants for {repo}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# --- API: Commits ---
async def api_get_commits(repo: str, revision: str = "main"):
    log.info(f"Fetching commits for repo: {repo}, revision: {revision}")
    try:
        state = load_state()
        pinned_shas = {
            d["revision"]
            for _, d in iter_configs(state)
            if d.get("repo") == repo and d.get("revision")
        }
        commits = get_commits(repo, revision=revision)
        for c in commits:
            c["pinned"] = c["sha"] in pinned_shas
        log.info(f"Got {len(commits)} commits for {repo}")
        return {"commits": commits}
    except Exception as e:
        log.error(f"Failed to fetch commits for {repo}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


# --- API: Branches ---
async def api_get_branches(repo: str):
    log.info(f"Fetching branches for repo: {repo}")
    try:
        branches = get_branches(repo)
        log.info(f"Got {len(branches)} branches for {repo}")
        return {"branches": branches}
    except Exception as e:
        log.error(f"Failed to fetch branches for {repo}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


def _get_branch_heads(repo_id):
    """Get {branch: sha} mapping for all branches of a repo from HF. Returns empty dict on failure."""
    try:
        branches = get_branches(repo_id)
    except Exception:
        return {}
    heads = {}
    for branch in branches:
        try:
            heads[branch] = resolve_sha(repo_id, branch)
        except Exception:
            continue
    return heads


# --- API: Local Models ---
async def get_local_models():
    cache = scan_cache()
    if cache is None:
        return {"models": []}

    state = load_state()
    used_by = {}
    for name, data in iter_configs(state):
        key = (data.get("repo"), data.get("revision"))
        if None in key:
            continue
        used_by.setdefault(key, []).append(name)

    models = []
    for repo in sorted(cache.repos, key=lambda r: r.repo_id):
        branch_heads = _get_branch_heads(repo.repo_id)
        log.info(f"get_local_models: {repo.repo_id} branch_heads={branch_heads}")
        revisions = []
        for rev in sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True):
            files = []
            for cf in sorted(rev.files, key=lambda f: f.file_name):
                if not cf.file_name.endswith(".gguf"):
                    continue
                match = re.search(QUANT_REGEX, cf.file_name, re.IGNORECASE)
                files.append({
                    "name": cf.file_name,
                    "size_str": format_bytes(cf.size_on_disk),
                    "quant": match.group(1).upper() if match else None,
                    "is_mmproj": "mmproj" in cf.file_name.lower(),
                })
            if not files:
                continue
            refs = list(rev.refs)
            for branch, sha in branch_heads.items():
                if rev.commit_hash == sha and branch not in refs:
                    refs.append(branch)
            log.debug(f"get_local_models: {repo.repo_id} {rev.commit_hash[:12]} refs={refs}")
            revisions.append({
                "sha": rev.commit_hash,
                "short_sha": rev.commit_hash[:12],
                "size_str": format_bytes(rev.size_on_disk),
                "last_modified": rev.last_modified,
                "refs": sorted(refs),
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


# --- API: RPC Mode ---
async def toggle_rpc(req: RpcModeReq):
    state = load_state()
    state.setdefault("_meta", {})["rpc_mode"] = req.enabled
    save_state(state)
    sync_system(state)
    return {"status": f"RPC mode enabled: {req.enabled}"}


# --- API: Delete Revision ---
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
        log.error(f"delete_revisions failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {e}")

    sync_system(state)
    return {"status": "Deleted"}


# --- API: Setup Model ---
async def setup_model(req: ModelSetup, background_tasks: BackgroundTasks):
    from backend.download import process_model

    req.symlink_name = req.symlink_name.replace("/", "-")
    if req.original_name:
        req.original_name = req.original_name.replace("/", "-")
    log.info(f"Setup {req.symlink_name} ({req.quant}, rev={req.revision[:12] if req.revision != 'latest' else 'latest'})")

    try:
        params_dict = json.loads(req.parameters) if req.parameters.strip() else {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    try:
        ref = "main" if req.revision == "latest" else req.revision
        resolved_sha = resolve_sha(req.hf_repo, ref)
    except Exception as e:
        log.error(f"Could not resolve revision '{req.revision}' for {req.hf_repo}: {e}")
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
        os.makedirs("/models/.cache", exist_ok=True)
        try:
            model_size = pre_flight_size(req.hf_repo, resolved_sha, req.quant, req.mmproj)
        except Exception as e:
            log.error(f"Pre-flight API check failed: {e}")
            raise HTTPException(status_code=400, detail=f"API Error: {e}")

        total, used, free = shutil.disk_usage("/models/.cache")
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


# --- API: Delete Config ---
async def delete_config(symlink_name: str):
    log.info(f"Deleting config {symlink_name}")
    state = load_state()
    if symlink_name in state:
        del state[symlink_name]
        save_state(state)
        prune_unreferenced_revisions(state)
        sync_system(state)
    return {"status": "Config deleted!"}

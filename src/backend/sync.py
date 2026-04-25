import glob
import json
import logging
import os
import re
import threading

import yaml

import docker as docker_sdk

from backend.config import CACHE_DIR, CONFIG_PATH, LLAMA_SHORT_FLAGS, LLAMA_SWAP_CONTAINER, QUANT_REGEX, SERVED_DIR
from backend.state import iter_configs, load_state, save_state

log = logging.getLogger(__name__)


def restart_llama_swap() -> None:
    try:
        client = docker_sdk.from_env()
        container = client.containers.get(LLAMA_SWAP_CONTAINER)
        container.restart()
        log.info(f"Restarted {LLAMA_SWAP_CONTAINER} to apply new config.")
    except Exception as e:
        log.error(f"Failed to restart {LLAMA_SWAP_CONTAINER}: {e}")


def sync_system(state: dict, restart: bool = True) -> None:
    """Rebuild served-dir symlinks and llama-swap config.yaml from state."""
    log.info("Syncing served dir and llama-swap config from state...")
    os.makedirs(SERVED_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(SERVED_DIR, "*.gguf")):
        try:
            os.unlink(f)
        except Exception:
            pass

    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir(CACHE_DIR) if os.path.isdir(CACHE_DIR) else None
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
            log.warning(f"Snapshot missing for {name} (repo={repo}, sha={(sha or '')[:12]}). Marking missing.")
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
            log.warning(f"Required files missing in snapshot for {name}. Marking missing.")
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
    log.info("llama-swap config.yaml rewritten.")

    if restart:
        threading.Thread(target=restart_llama_swap, daemon=True).start()

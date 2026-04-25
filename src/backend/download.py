import logging
import os
import pty
import re
import subprocess
import sys
import time

from backend.config import CACHE_DIR
from backend.models import ModelSetup
from backend.state import load_state, save_state
from backend.sync import sync_system
from backend.cache import prune_unreferenced_revisions

log = logging.getLogger(__name__)


def process_model(req: ModelSetup) -> None:
    """Download via huggingface_hub.snapshot_download with a pre-resolved SHA in req.revision."""
    log.info(f"Starting background download for {req.symlink_name} ({req.revision[:12]})...")
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
    with open(script_path, "w") as f:
        f.write(script_code)

    cmd = [sys.executable, script_path]
    master, slave = pty.openpty()
    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    try:
        process = subprocess.Popen(cmd, stdin=slave, stdout=slave, stderr=slave, close_fds=True, env=env)
    except Exception as e:
        log.error(f"Subprocess failed to launch for {req.symlink_name}: {e}")
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
            if not data:
                break
            buffer += data
            if len(error_log) < 100:
                error_log.append(data)

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
                            if not dl_str.endswith('B'):
                                dl_str += "B"
                            if not tot_str.endswith('B'):
                                tot_str += "B"

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
        except OSError:
            break

    process.wait()
    os.close(master)
    try:
        os.unlink(script_path)
    except Exception:
        pass

    state = load_state()
    if process.returncode == 0:
        log.info(f"Download completed for {req.symlink_name}.")
        # Ensure refs/main points to the downloaded SHA so scan_cache_dir can find the snapshot.
        # Without this, if HF's main branch advanced after the original download, the ref becomes
        # stale and the snapshot is invisible to scan_cache_dir.
        try:
            os.makedirs(refs_dir, exist_ok=True)
            refs_main = os.path.join(refs_dir, "main")
            with open(refs_main, "w") as rf:
                rf.write(req.revision)
            log.debug(f"Updated refs/main → {req.revision[:12]} for {req.hf_repo}")
        except Exception as e:
            log.warning(f"Could not update refs/main for {req.hf_repo}: {e}")
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
        log.error(f"Download failed for {req.symlink_name}. Error: {final_err}")

        if req.symlink_name in state:
            state[req.symlink_name]["status"] = "error"
            state[req.symlink_name]["error_msg"] = f"Failed: {final_err}"[:120]
            save_state(state)

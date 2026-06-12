import logging
import os
import pty
import re
import subprocess
import sys
import threading
import time

from backend.config import CACHE_DIR
from backend.models import ModelSetup
from backend.state import format_bytes, load_state, save_state
from backend.sync import sync_system
from backend.cache import prune_unreferenced_revisions

log = logging.getLogger(__name__)


def _fmt_eta(secs: float) -> str:
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def process_model(req: ModelSetup) -> None:
    """Download via huggingface_hub.snapshot_download with a pre-resolved SHA in req.revision."""
    log.info(f"Starting background download for {req.symlink_name} ({req.revision[:12]})...")
    patterns = [f"*{req.quant}.gguf", f"*{req.quant}-*-of-*.gguf"]
    if req.mmproj:
        # Anchor the tag to a separator so e.g. mmproj=F16 doesn't also pull mmproj-BF16
        # (the bare glob *F16*.gguf matches BF16). Mirrors QUANT_REGEX's [-._] prefix.
        patterns.append(f"*mmproj*[-._]{req.mmproj}.gguf")
        patterns.append(f"*mmproj*[-._]{req.mmproj}-*-of-*.gguf")
    if req.mtp_head:
        patterns.append(req.mtp_head)  # exact rfilename of the draft head (same repo)

    # HF_HUB_DOWNLOAD_TIMEOUT bounds each socket read: a dead-but-open connection (the
    # LFS stalls we keep hitting near the end) then raises instead of hanging forever,
    # and huggingface_hub resumes the file from its partial .incomplete via http_backoff.
    # hf_xet (if the repo is Xet-backed) accelerates transparently; LFS repos fall back
    # to the normal HTTP path. The old hf_transfer env var is deprecated and ignored.
    script_code = f"""
import os
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '30'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id={req.hf_repo!r},
    allow_patterns={patterns!r},
    cache_dir={CACHE_DIR!r},
    revision={req.revision!r},
    max_workers=4,
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
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
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

    # Expected total bytes (selected quant + mmproj + head), recorded at setup time.
    st0 = load_state()
    expected_total = int((st0.get(req.symlink_name) or {}).get("dl_total", 0) or 0)
    blob_dir = os.path.join(CACHE_DIR, f"models--{req.hf_repo.replace('/', '--')}", "blobs")

    def blob_bytes() -> int:
        total = 0
        try:
            with os.scandir(blob_dir) as it:
                for e in it:
                    try:
                        total += e.stat().st_size
                    except OSError:
                        pass
        except FileNotFoundError:
            pass
        return total

    # Drain the PTY in a thread so the child never blocks on a full terminal buffer,
    # keeping the tail for error reporting. Progress is derived from bytes-on-disk, not
    # from stdout: huggingface_hub (>=1.x) no longer emits parseable tqdm byte counts.
    error_log = []

    def drain():
        while True:
            try:
                data = os.read(master, 32768).decode('utf-8', errors='replace')
            except OSError:
                break
            if not data:
                break
            error_log.append(data)
            if len(error_log) > 100:
                del error_log[0]

    drain_thread = threading.Thread(target=drain, daemon=True)
    drain_thread.start()

    # Progress tracks THIS session's download, not total bytes on disk: measure bytes
    # fetched since start against bytes that still need fetching. This keeps the bar
    # meaningful whether it's a fresh pull (start≈0) or just a head re-fetch over an
    # already-present quant+mmproj (where total-on-disk is already near the full size).
    STALL_LIMIT = 300
    start_bytes = blob_bytes()
    start_t = time.time()
    to_fetch = max(expected_total - start_bytes, 0)  # 0 if everything's already present
    last_growth_bytes = start_bytes
    last_growth_t = start_t
    stalled = False

    while process.poll() is None:
        time.sleep(1.0)
        cur = blob_bytes()
        now = time.time()

        if cur > last_growth_bytes:
            last_growth_bytes, last_growth_t = cur, now
        elif now - last_growth_t > STALL_LIMIT:
            log.error(f"Download for {req.symlink_name} stalled {STALL_LIMIT}s with no progress; aborting.")
            stalled = True
            process.kill()
            break

        done = max(cur - start_bytes, 0)
        # Average speed since start: smooth and always present once bytes flow, unlike a
        # 1 s instantaneous delta that jitters to 0 between chunks/files.
        elapsed = now - start_t
        speed = done / elapsed if elapsed > 0 else 0
        if to_fetch > 0:
            pct = min(99, int(done / to_fetch * 100))
            remaining = max(0, to_fetch - done)
            eta = remaining / speed if speed > 1 else 0
            total_str = format_bytes(to_fetch)
        else:  # nothing to fetch (re-link only); the child just verifies and exits
            pct, eta, total_str = 99, 0, "--"

        state = load_state()
        if req.symlink_name in state and state[req.symlink_name].get("status") == "downloading":
            state[req.symlink_name]["progress_str"] = {
                "percent": str(pct),
                "downloaded": format_bytes(done),
                "total": total_str,
                "speed": f"{format_bytes(speed)}/s" if speed > 1 else "--",
                "eta": _fmt_eta(eta) if eta else "--",
            }
            save_state(state)

    process.wait()
    drain_thread.join(timeout=2)
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
    elif stalled:
        log.error(f"Download for {req.symlink_name} aborted: stalled with no progress.")
        if req.symlink_name in state:
            state[req.symlink_name]["status"] = "error"
            state[req.symlink_name]["error_msg"] = "Network stall: download made no progress. Retry to resume."
            save_state(state)
    else:
        err_text = "".join(error_log[-10:]).replace('\n', ' ')
        exc_match = re.search(r'([A-Za-z]+Error:.*)', err_text)
        final_err = exc_match.group(1) if exc_match else err_text
        log.error(f"Download failed for {req.symlink_name}. Error: {final_err}")

        if req.symlink_name in state:
            state[req.symlink_name]["status"] = "error"
            state[req.symlink_name]["error_msg"] = f"Failed: {final_err}"[:120]
            save_state(state)

import logging
import os
import shutil

from huggingface_hub import scan_cache_dir

from backend.config import CACHE_DIR
from backend.state import iter_configs

log = logging.getLogger(__name__)


def scan_cache():
    """Scan the HF cache. Returns HFCacheInfo or None if cache dir is absent/empty."""
    if not os.path.isdir(CACHE_DIR):
        return None
    try:
        return scan_cache_dir(CACHE_DIR)
    except Exception as e:
        log.warning(f"scan_cache_dir failed: {e}")
        return None


def in_use_revisions(state: dict) -> set:
    """Set of (repo_id, commit_sha) tuples referenced by state."""
    return {
        (d["repo"], d["revision"])
        for _, d in iter_configs(state)
        if d.get("repo") and d.get("revision")
    }


def prune_unreferenced_revisions(state: dict) -> None:
    """Delete any cached revision (and orphan repo dir) not referenced by state. Silent, automatic."""
    in_use = in_use_revisions(state)
    cache = scan_cache()
    if cache is not None:
        shas_to_delete = []
        for repo in cache.repos:
            for rev in repo.revisions:
                if (repo.repo_id, rev.commit_hash) not in in_use:
                    shas_to_delete.append(rev.commit_hash)
        if shas_to_delete:
            log.info(f"Pruning {len(shas_to_delete)} unreferenced revision(s) from cache.")
            try:
                cache.delete_revisions(*shas_to_delete).execute()
            except Exception as e:
                log.error(f"delete_revisions failed: {e}")

    # Second pass: corrupted repo dirs (empty/broken snapshots) that scan_cache_dir excludes.
    expected_dirs = {
        f"models--{d['repo'].replace('/', '--')}"
        for _, d in iter_configs(state) if d.get("repo")
    }
    if os.path.isdir(CACHE_DIR):
        for entry in os.listdir(CACHE_DIR):
            if not entry.startswith("models--"):
                continue
            if entry in expected_dirs:
                continue
            full = os.path.join(CACHE_DIR, entry)
            if os.path.isdir(full):
                log.info(f"Removing orphan repo dir: {entry}")
                shutil.rmtree(full, ignore_errors=True)

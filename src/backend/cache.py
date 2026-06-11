import logging
import os
import re
import shutil

from huggingface_hub import scan_cache_dir

from backend.config import CACHE_DIR, QUANT_REGEX
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

    prune_unreferenced_files(state)

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


def _file_quant(fname: str):
    """Quant tag a GGUF filename encodes (e.g. 'UD-Q4_K_XL', 'F16'), or None."""
    m = re.search(QUANT_REGEX, fname, re.IGNORECASE)
    return m.group(1).upper() if m else None


def in_use_files(state: dict) -> dict:
    """Map (repo, sha) -> {'quants': set, 'mmprojs': set} of quant tags state references."""
    out = {}
    for _, d in iter_configs(state):
        repo, sha = d.get("repo"), d.get("revision")
        if not repo or not sha:
            continue
        entry = out.setdefault((repo, sha), {"quants": set(), "mmprojs": set()})
        if d.get("quant"):
            entry["quants"].add(d["quant"].upper())
        if d.get("mmproj"):
            entry["mmprojs"].add(d["mmproj"].upper())
    return out


def prune_unreferenced_files(state: dict) -> None:
    """Delete orphan GGUF files (unused quant/mmproj variants) living inside a revision
    that is otherwise still referenced — revision-level pruning can't reach these (e.g.
    a config switched quant, leaving the old .gguf behind). Silent, automatic.

    Only recognized GGUF variants are touched; metadata and unrecognized GGUFs (e.g. a
    separate MTP draft head) are left alone. Blobs are reference-counted so shared
    content is never removed while another snapshot still points at it."""
    cache = scan_cache()
    if cache is None:
        return
    wanted = in_use_files(state)

    blob_refs = {}
    for repo in cache.repos:
        for rev in repo.revisions:
            for f in rev.files:
                blob_refs[str(f.blob_path)] = blob_refs.get(str(f.blob_path), 0) + 1

    for repo in cache.repos:
        for rev in repo.revisions:
            w = wanted.get((repo.repo_id, rev.commit_hash))
            if w is None:
                continue  # whole revision unreferenced -> prune_unreferenced_revisions handles it
            for f in rev.files:
                fname = f.file_name
                low = fname.lower()
                if not low.endswith(".gguf"):
                    continue
                if "mtp" in low:
                    # Conservatively skip MTP/draft-head files (e.g. gemma's mtp-*.gguf):
                    # they aren't tracked in state yet and some are runtime heads we must
                    # not delete. A grafted main GGUF that embeds "MTP" in its filename
                    # simply won't be pruned here (residual disk, never data loss).
                    continue
                quant = _file_quant(fname)
                if quant is None:
                    continue  # unrecognized gguf (e.g. an MTP draft head) -> leave alone
                is_mmproj = "mmproj" in fname.lower()
                if quant in (w["mmprojs"] if is_mmproj else w["quants"]):
                    continue  # referenced
                try:
                    fp, bp = str(f.file_path), str(f.blob_path)
                    if os.path.islink(fp) or os.path.exists(fp):
                        os.remove(fp)
                    blob_refs[bp] = blob_refs.get(bp, 1) - 1
                    if blob_refs[bp] <= 0 and os.path.exists(bp):
                        os.remove(bp)
                    log.info(
                        f"Pruned orphan file {repo.repo_id}@{rev.commit_hash[:12]}/{fname} "
                        f"({f.size_on_disk / 1e9:.2f} GB)"
                    )
                except Exception as e:
                    log.error(f"Failed to prune orphan file {fname}: {e}")

import logging
import re

from huggingface_hub import HfApi

from backend.config import QUANT_REGEX

log = logging.getLogger(__name__)


def resolve_sha(repo: str, ref: str = "main") -> str:
    """Resolve a repo ref (branch/tag/SHA) to a concrete commit SHA. Raises on failure."""
    api = HfApi(token=None)
    info = api.model_info(repo, revision=ref)
    return info.sha


def get_commits(repo: str, limit: int = 50) -> list:
    log.info(f"get_commits: fetching for {repo}, limit={limit}")
    api = HfApi(token=None)
    log.info(f"get_commits: HfApi created, calling list_repo_commits")
    commits = list(api.list_repo_commits(repo, repo_type="model"))[:limit]
    log.info(f"get_commits: got {len(commits)} commits")
    result = []
    for c in commits:
        msg = ""
        if hasattr(c, "short_commit_message") and c.short_commit_message:
            msg = c.short_commit_message
        elif hasattr(c, "message") and c.message:
            msg = c.message.split("\n", 1)[0][:120]
        # committer_timestamp is a float (epoch seconds) or datetime
        ts = getattr(c, "committer_timestamp", None)
        if ts and isinstance(ts, str):
            try:
                ts = float(ts)
            except ValueError:
                ts = 0
        elif ts and hasattr(ts, "timestamp"):
            ts = ts.timestamp()
        result.append({
            "sha": c.commit_id,
            "date": int(ts) if ts else 0,
            "message": msg,
        })
    # Sort by date descending (newest first)
    result.sort(key=lambda x: x["date"], reverse=True)
    return result


def pre_flight_size(repo: str, sha: str, quant: str, mmproj: str) -> int:
    api = HfApi(token=None)
    info = api.model_info(repo, revision=sha, files_metadata=True)
    total = 0
    for f in info.siblings:
        if not f.size or not f.rfilename.endswith(".gguf"):
            continue
        match = re.search(QUANT_REGEX, f.rfilename, re.IGNORECASE)
        if not match:
            continue
        is_mmproj_file = "mmproj" in f.rfilename.lower()
        if is_mmproj_file and mmproj and match.group(1).upper() == mmproj.upper():
            total += f.size
        elif not is_mmproj_file and match.group(1).upper() == quant.upper():
            total += f.size
    return total

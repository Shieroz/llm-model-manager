import os
import re

# --- Paths ---
CACHE_DIR = "/models/.cache"
SERVED_DIR = "/models/served"
CONFIG_PATH = os.path.join(SERVED_DIR, "config.yaml")
STATE_FILE = os.path.join(SERVED_DIR, "state.json")
LLAMA_SWAP_CONTAINER = os.environ.get("LLAMA_SWAP_CONTAINER", "llama-swap")

# --- Patterns ---
QUANT_REGEX = r'[-._]((?:UD-)?[A-Za-z]*Q[0-9][A-Za-z0-9_]*|BF16|F16|F32|MXFP4_MOE)(?:-\d{5}-of-\d{5})?\.gguf$'
SHA_RE = re.compile(r'/snapshots/([a-f0-9]{40})/')

# --- MTP draft heads (separate-head models shipped alongside the main quant) ---
# Filename hints that mark a runtime MTP/speculative draft head (e.g. gemma-4's
# gemma4-assistant, NextN, EAGLE). Heads are small, so a size cap keeps a grafted
# main GGUF that happens to embed "MTP" in its filename out of the head list.
MTP_HEAD_HINTS = ("mtp", "assistant", "nextn", "eagle")
MTP_HEAD_MAX_BYTES = 4 * 1024 ** 3


def is_mtp_head_file(fname: str, size: int = 0) -> bool:
    """True if a GGUF looks like a small separate MTP draft head (not a quant/mmproj)."""
    low = fname.lower()
    if not low.endswith(".gguf") or "mmproj" in low:
        return False
    if not any(h in low for h in MTP_HEAD_HINTS):
        return False
    return size == 0 or size < MTP_HEAD_MAX_BYTES

# llama-server short flags (single dash) — everything else uses double dash
LLAMA_SHORT_FLAGS = {
    "a", "b", "bs", "c", "cb", "cd", "cl", "cmoe", "cmoed", "cpent", "cram",
    "ctk", "ctkd", "ctv", "ctvd", "ctxcp", "dev", "devd", "dio", "dr", "dt",
    "e", "fa", "fit", "fitc", "fitt", "h", "hf", "hfd", "hff", "hffv", "hft",
    "hfv", "j", "jf", "kvo", "kvu", "l", "lcd", "lcs", "lv", "m", "md", "mg",
    "mm", "mmu", "mu", "mv", "n", "ncmoe", "ngl", "ngld", "np", "ot", "otd",
    "r", "rea", "s", "sm", "sp", "sps", "t", "tb", "tbd", "td", "to", "ts",
    "ub", "v",
}

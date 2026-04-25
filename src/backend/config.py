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

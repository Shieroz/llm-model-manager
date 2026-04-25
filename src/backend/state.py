import json
import os

from backend.config import SERVED_DIR, STATE_FILE


def format_bytes(bytes_num: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_num < 1000.0:
            return f"{bytes_num:.1f} {unit}"
        bytes_num /= 1000.0
    return f"{bytes_num:.1f} PB"


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(state: dict) -> None:
    os.makedirs(SERVED_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def iter_configs(state: dict):
    for name, data in state.items():
        if name == "_meta" or not isinstance(data, dict):
            continue
        yield name, data

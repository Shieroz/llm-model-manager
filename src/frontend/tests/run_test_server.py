"""Test server wrapper that patches SERVED_DIR to a temp directory."""
import os
import sys

# Patch SERVED_DIR before importing the app
_test_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tmp", "test-llm-models")
os.makedirs(_test_dir, exist_ok=True)
os.makedirs(os.path.join(_test_dir, "served"), exist_ok=True)

# Patch the config module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import backend.config
backend.config.SERVED_DIR = os.path.join(_test_dir, "served")
backend.config.CACHE_DIR = _test_dir
backend.config.CONFIG_PATH = os.path.join(_test_dir, "served", "config.yaml")
backend.config.STATE_FILE = os.path.join(_test_dir, "served", "state.json")

from backend.app import app  # noqa: E402

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=18976)

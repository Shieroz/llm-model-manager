import os
from unittest.mock import patch

import pytest


@pytest.fixture()
def tmp_state_dir(tmp_path):
    """Temporarily override state/cache paths for tests that need them."""
    state_file = str(tmp_path / "state.json")
    served_dir = str(tmp_path)
    config_path = str(tmp_path / "config.yaml")
    cache_dir = str(tmp_path / ".cache")
    
    with patch("backend.state.SERVED_DIR", served_dir):
        with patch("backend.state.STATE_FILE", state_file):
            with patch("backend.cache.CACHE_DIR", cache_dir):
                with patch("backend.sync.SERVED_DIR", served_dir):
                    with patch("backend.sync.CONFIG_PATH", config_path):
                        with patch("backend.sync.CACHE_DIR", cache_dir):
                            with patch("backend.config.SERVED_DIR", served_dir):
                                with patch("backend.config.STATE_FILE", state_file):
                                    with patch("backend.config.CONFIG_PATH", config_path):
                                        with patch("backend.config.CACHE_DIR", cache_dir):
                                            yield tmp_path


@pytest.fixture()
def empty_state(tmp_state_dir):
    """Provide a fresh empty state file."""
    from backend.state import save_state
    save_state({})
    return {}


@pytest.fixture()
def sample_state(tmp_state_dir):
    """Provide a state with a sample config."""
    from backend.state import save_state
    state = {
        "_meta": {"rpc_mode": False},
        "test-model": {
            "repo": "test/repo",
            "quant": "Q4_K_M",
            "mmproj": "",
            "params": {"ctx_size": 4096},
            "status": "ready",
            "revision": "abc123def456",
        },
    }
    save_state(state)
    return state


@pytest.fixture()
def download_state(tmp_state_dir):
    """Provide a state with a downloading model."""
    from backend.state import save_state
    state = {
        "downloading-model": {
            "repo": "test/download",
            "quant": "Q5_K_M",
            "mmproj": "",
            "params": {},
            "status": "downloading",
            "revision": "deadbeef1234",
        },
    }
    save_state(state)
    return state

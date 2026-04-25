import json
import os
from unittest.mock import patch

import pytest

from backend.state import format_bytes, load_state, save_state, iter_configs


class TestFormatBytes:
    def test_bytes(self):
        assert format_bytes(500) == "500.0 B"

    def test_kilobytes(self):
        assert format_bytes(1500) == "1.5 KB"

    def test_megabytes(self):
        assert format_bytes(1500000) == "1.5 MB"

    def test_gigabytes(self):
        assert format_bytes(1500000000) == "1.5 GB"

    def test_terabytes(self):
        assert format_bytes(1500000000000) == "1.5 TB"

    def test_very_large(self):
        assert format_bytes(1500000000000000) == "1.5 PB"

    def test_boundary(self):
        assert format_bytes(999.9) == "999.9 B"
        assert format_bytes(1000.0) == "1.0 KB"


class TestSaveLoadState:
    def test_save_and_load(self, tmp_state_dir):
        state = {"key": "value", "count": 42}
        save_state(state)
        loaded = load_state()
        assert loaded == state

    def test_save_creates_directory(self, tmp_state_dir):
        # Use a subdirectory as the served dir
        nested = tmp_state_dir / "subdir"
        with patch("backend.state.SERVED_DIR", str(nested)):
            with patch("backend.state.STATE_FILE", str(nested / "state.json")):
                with patch("backend.config.SERVED_DIR", str(nested)):
                    with patch("backend.config.STATE_FILE", str(nested / "state.json")):
                        save_state({"test": True})
                        assert nested.exists()
                        assert (nested / "state.json").exists()

    def test_load_nonexistent_returns_empty(self, tmp_state_dir):
        state_file = os.path.join(str(tmp_state_dir), "state.json")
        if os.path.exists(state_file):
            os.remove(state_file)
        assert load_state() == {}

    def test_load_corrupted_json_returns_empty(self, tmp_state_dir):
        state_file = os.path.join(str(tmp_state_dir), "state.json")
        with open(state_file, "w") as f:
            f.write("not valid json{{{")
        assert load_state() == {}


class TestIterConfigs:
    def test_iterates_over_configs(self, empty_state):
        from backend import config
        empty_state["model1"] = {"repo": "a", "revision": "b"}
        empty_state["model2"] = {"repo": "c", "revision": "d"}
        save_state(empty_state)

        configs = list(iter_configs(empty_state))
        names = [name for name, _ in configs]
        assert "model1" in names
        assert "model2" in names

    def test_skips_meta(self, empty_state):
        empty_state["_meta"] = {"rpc_mode": False}
        configs = list(iter_configs(empty_state))
        names = [name for name, _ in configs]
        assert "_meta" not in names

    def test_skips_non_dict(self, empty_state):
        empty_state["bad"] = "not a dict"
        empty_state["good"] = {"repo": "x"}
        configs = list(iter_configs(empty_state))
        names = [name for name, _ in configs]
        assert "bad" not in names
        assert "good" in names

    def test_returns_empty_for_empty_state(self, empty_state):
        configs = list(iter_configs(empty_state))
        assert configs == []

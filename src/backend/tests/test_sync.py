import json
import os
from unittest.mock import MagicMock, patch

import pytest
import yaml

from backend.sync import sync_system, restart_llama_swap
from backend.state import save_state


class TestRestartLlamaSwap:
    @patch("backend.sync.docker_sdk")
    def test_restarts_container(self, mock_docker):
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_client.containers.get.return_value = mock_container
        mock_docker.from_env.return_value = mock_client

        restart_llama_swap()
        mock_container.restart.assert_called_once()

    @patch("backend.sync.docker_sdk")
    def test_handles_failure(self, mock_docker, caplog):
        mock_docker.from_env.side_effect = Exception("docker not available")
        restart_llama_swap()


class TestSyncSystem:
    @patch("backend.cache.scan_cache_dir")
    def test_writes_config_yaml(self, mock_scan, sample_state, tmp_state_dir):
        mock_cache = MagicMock()
        mock_cache.repos = []
        mock_scan.return_value = mock_cache

        sync_system(sample_state, restart=False)

        config_path = os.path.join(str(tmp_state_dir), "config.yaml")
        assert os.path.exists(config_path)
        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert "models" in data

    @patch("backend.cache.scan_cache_dir")
    def test_skips_downloading_models(self, mock_scan, download_state, tmp_state_dir):
        mock_cache = MagicMock()
        mock_cache.repos = []
        mock_scan.return_value = mock_cache

        sync_system(download_state, restart=False)

        with open(os.path.join(str(tmp_state_dir), "config.yaml")) as f:
            data = yaml.safe_load(f)
        assert "downloading-model-Q5_K_M" not in data.get("models", {})

    @patch("backend.cache.scan_cache_dir")
    def test_handles_missing_snapshot(self, mock_scan, sample_state, tmp_state_dir):
        mock_cache = MagicMock()
        mock_cache.repos = []
        mock_scan.return_value = mock_cache

        sync_system(sample_state, restart=False)
        assert sample_state["test-model"]["status"] == "missing"

    @patch("backend.sync.threading")
    @patch("backend.cache.scan_cache_dir")
    def test_restarts_llama_swap_by_default(self, mock_scan, mock_threading, sample_state, tmp_state_dir):
        mock_cache = MagicMock()
        mock_cache.repos = []
        mock_scan.return_value = mock_cache

        sync_system(sample_state, restart=True)
        mock_threading.Thread.assert_called_once()

    @patch("backend.sync.threading")
    @patch("backend.cache.scan_cache_dir")
    def test_does_not_restart_when_disabled(self, mock_scan, mock_threading, sample_state, tmp_state_dir):
        mock_cache = MagicMock()
        mock_cache.repos = []
        mock_scan.return_value = mock_cache

        sync_system(sample_state, restart=False)
        mock_threading.Thread.assert_not_called()

    @patch("backend.sync.threading")
    @patch("backend.cache.scan_cache_dir")
    def test_does_not_restart_when_config_unchanged(self, mock_scan, mock_threading, sample_state, tmp_state_dir):
        mock_cache = MagicMock()
        mock_cache.repos = []
        mock_scan.return_value = mock_cache

        sync_system(sample_state, restart=True)
        mock_threading.Thread.assert_called_once()

        # Re-syncing identical state must not evict the loaded model.
        mock_threading.Thread.reset_mock()
        sync_system(sample_state, restart=True)
        mock_threading.Thread.assert_not_called()

    @patch("backend.sync.threading")
    @patch("backend.cache.scan_cache_dir")
    def test_restarts_when_config_changes(self, mock_scan, mock_threading, sample_state, tmp_state_dir):
        mock_cache = MagicMock()
        mock_cache.repos = []
        mock_scan.return_value = mock_cache

        sync_system(sample_state, restart=True)

        config_path = os.path.join(str(tmp_state_dir), "config.yaml")
        with open(config_path, "a") as f:
            f.write("# drift\n")

        mock_threading.Thread.reset_mock()
        sync_system(sample_state, restart=True)
        mock_threading.Thread.assert_called_once()

    @patch("backend.sync.os.path.isdir", return_value=True)
    @patch("huggingface_hub.scan_cache_dir")
    def test_emits_model_draft_for_same_repo_head(self, mock_scan, mock_isdir, tmp_state_dir):
        state = {
            "_meta": {"rpc_mode": False},
            "headcfg": {
                "repo": "u/main-GGUF", "quant": "Q4_K_M", "mmproj": "",
                "params": {}, "status": "ready", "revision": "mainsha",
                "mtp_head": "mtp-head.gguf",
            },
        }
        save_state(state)

        def mk(name):
            cf = MagicMock()
            cf.file_name = name
            cf.file_path = f"/models/.cache/blobs/{name}"
            cf.size_on_disk = 100
            return cf

        rev = MagicMock()
        rev.commit_hash = "mainsha"
        rev.files = [mk("model-Q4_K_M.gguf"), mk("mtp-head.gguf")]
        repo = MagicMock()
        repo.repo_id = "u/main-GGUF"
        repo.revisions = [rev]
        cache = MagicMock()
        cache.repos = [repo]
        mock_scan.return_value = cache

        sync_system(state, restart=False)

        with open(os.path.join(str(tmp_state_dir), "config.yaml")) as f:
            data = yaml.safe_load(f)
        cmd = data["models"]["headcfg-Q4_K_M"]["cmd"]
        assert "--model-draft" in cmd
        assert "headcfg-mtp-head.gguf" in cmd
        assert "--spec-type draft-mtp" in cmd
        assert state["headcfg"]["status"] == "ready"

    @patch("backend.sync.os.path.isdir", return_value=True)
    @patch("huggingface_hub.scan_cache_dir")
    def test_marks_missing_when_head_absent(self, mock_scan, mock_isdir, tmp_state_dir):
        state = {
            "_meta": {"rpc_mode": False},
            "headcfg": {
                "repo": "u/main-GGUF", "quant": "Q4_K_M", "mmproj": "",
                "params": {}, "status": "ready", "revision": "mainsha",
                "mtp_head": "mtp-head.gguf",
            },
        }
        save_state(state)
        cf = MagicMock()
        cf.file_name = "model-Q4_K_M.gguf"
        cf.file_path = "/models/.cache/blobs/model-Q4_K_M.gguf"
        cf.size_on_disk = 100
        rev = MagicMock()
        rev.commit_hash = "mainsha"
        rev.files = [cf]  # quant present, head missing
        repo = MagicMock()
        repo.repo_id = "u/main-GGUF"
        repo.revisions = [rev]
        cache = MagicMock()
        cache.repos = [repo]
        mock_scan.return_value = cache

        sync_system(state, restart=False)
        assert state["headcfg"]["status"] == "missing"

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

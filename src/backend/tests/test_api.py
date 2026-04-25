from unittest.mock import MagicMock, patch

import pytest

from backend.api import (
    get_quants,
    api_get_commits,
    get_local_models,
    toggle_rpc,
    delete_revision,
    setup_model,
    delete_config,
)
from backend.models import ModelSetup, RevisionDeleteReq, RpcModeReq
from backend.state import load_state, save_state


class TestGetQuants:
    @patch("huggingface_hub.HfApi")
    @pytest.mark.asyncio
    async def test_returns_quants(self, mock_api_class):
        mock_api = MagicMock()
        file1 = MagicMock()
        file1.rfilename = "model-Q4_K_M.gguf"
        file1.size = 5000000000
        file2 = MagicMock()
        file2.rfilename = "model-Q5_K_M.gguf"
        file2.size = 6000000000
        mock_api.model_info.return_value = MagicMock(siblings=[file1, file2])
        mock_api_class.return_value = mock_api

        result = await get_quants("test/repo")
        assert "quants" in result
        assert len(result["quants"]) == 2
        names = [q["name"] for q in result["quants"]]
        assert "Q4_K_M" in names
        assert "Q5_K_M" in names

    @patch("huggingface_hub.HfApi")
    @pytest.mark.asyncio
    async def test_separates_mmproj(self, mock_api_class):
        mock_api = MagicMock()
        file1 = MagicMock()
        file1.rfilename = "model-Q4_K_M.gguf"
        file1.size = 5000000000
        file2 = MagicMock()
        file2.rfilename = "mmproj-F16.gguf"
        file2.size = 900000000
        mock_api.model_info.return_value = MagicMock(siblings=[file1, file2])
        mock_api_class.return_value = mock_api

        result = await get_quants("test/repo")
        assert len(result["quants"]) == 1
        assert len(result["mmprojs"]) == 1
        assert result["mmprojs"][0]["name"] == "F16"

    @patch("huggingface_hub.HfApi")
    @pytest.mark.asyncio
    async def test_raises_on_api_failure(self, mock_api_class):
        mock_api_class.side_effect = Exception("API error")
        with pytest.raises(Exception) as exc_info:
            await get_quants("test/repo")
        assert "API error" in str(exc_info.value)


class TestGetCommits:
    @patch("backend.api.get_commits")
    @patch("backend.api.load_state")
    @pytest.mark.asyncio
    async def test_returns_commits_with_pin_status(self, mock_load, mock_get):
        mock_load.return_value = {
            "pinned-model": {
                "repo": "test/repo",
                "revision": "pinned-sha",
            }
        }
        mock_get.return_value = [
            {"sha": "other-sha", "date": 200, "message": "other"},
            {"sha": "pinned-sha", "date": 100, "message": "pinned"},
        ]

        result = await api_get_commits("test/repo")
        assert len(result["commits"]) == 2
        # Commits sorted newest first, pinned-sha is the second one
        pinned = [c for c in result["commits"] if c["pinned"]]
        assert len(pinned) == 1
        assert pinned[0]["sha"] == "pinned-sha"

    @patch("backend.api.get_commits")
    @pytest.mark.asyncio
    async def test_returns_commits(self, mock_get):
        mock_get.return_value = [
            {"sha": "abc", "date": 100, "message": "test"},
        ]

        result = await api_get_commits("test/repo")
        assert result["commits"][0]["sha"] == "abc"


class TestGetLocalModels:
    @patch("backend.api.scan_cache")
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_cache(self, mock_scan):
        mock_scan.return_value = None
        result = await get_local_models()
        assert result == {"models": []}

    @patch("backend.api.scan_cache")
    @pytest.mark.asyncio
    async def test_returns_cached_models(self, mock_scan, sample_state):
        mock_cache = MagicMock()
        mock_repo = MagicMock()
        mock_repo.repo_id = "test/repo"
        mock_repo.size_on_disk = 5000000000
        mock_rev = MagicMock()
        mock_rev.commit_hash = "abc123def456"
        mock_rev.size_on_disk = 5000000000
        mock_rev.last_modified = 1000000
        mock_rev.refs = ["main"]
        mock_file = MagicMock()
        mock_file.file_name = "model-Q4_K_M.gguf"
        mock_file.size_on_disk = 5000000000
        mock_rev.files = [mock_file]
        mock_repo.revisions = [mock_rev]
        mock_cache.repos = [mock_repo]
        mock_scan.return_value = mock_cache

        result = await get_local_models()
        assert len(result["models"]) == 1
        assert result["models"][0]["repo"] == "test/repo"
        assert result["models"][0]["revisions"][0]["sha"] == "abc123def456"


class TestToggleRpc:
    @pytest.mark.asyncio
    async def test_enables_rpc_mode(self, empty_state):
        req = RpcModeReq(enabled=True)
        result = await toggle_rpc(req)
        state = load_state()
        assert state["_meta"]["rpc_mode"] is True
        assert "RPC mode enabled: True" in result["status"]

    @pytest.mark.asyncio
    async def test_disables_rpc_mode(self, empty_state):
        save_state({"_meta": {"rpc_mode": True}})
        req = RpcModeReq(enabled=False)
        result = await toggle_rpc(req)
        state = load_state()
        assert state["_meta"]["rpc_mode"] is False


class TestDeleteRevision:
    @patch("backend.api.scan_cache")
    @pytest.mark.asyncio
    async def test_deletes_revision(self, mock_scan, tmp_state_dir):
        mock_cache = MagicMock()
        mock_repo = MagicMock()
        mock_repo.repo_id = "test/repo"
        mock_rev = MagicMock()
        mock_rev.commit_hash = "abc123"
        mock_repo.revisions = [mock_rev]
        mock_cache.repos = [mock_repo]
        mock_scan.return_value = mock_cache

        save_state({})
        req = RevisionDeleteReq(repo="test/repo", revision="abc123")
        result = await delete_revision(req)
        assert result["status"] == "Deleted"
        mock_cache.delete_revisions.assert_called_once_with("abc123")

    @patch("backend.api.scan_cache")
    @pytest.mark.asyncio
    async def test_blocks_when_revision_in_use(self, mock_scan, sample_state):
        mock_scan.return_value = MagicMock()
        mock_scan.return_value.repos = []

        req = RevisionDeleteReq(repo="test/repo", revision="abc123def456")
        with pytest.raises(Exception) as exc_info:
            await delete_revision(req)
        assert "pinned" in str(exc_info.value).lower()

    @patch("backend.api.scan_cache")
    @pytest.mark.asyncio
    async def test_returns_404_when_cache_empty(self, mock_scan):
        mock_scan.return_value = None
        req = RevisionDeleteReq(repo="test/repo", revision="abc123")
        with pytest.raises(Exception) as exc_info:
            await delete_revision(req)
        assert "empty" in str(exc_info.value).lower()


class TestSetupModel:
    @patch("backend.api.resolve_sha")
    @patch("backend.api.pre_flight_size")
    @patch("backend.api.shutil")
    @patch("backend.api.os")
    @patch("backend.api.sync_system")
    @pytest.mark.asyncio
    async def test_sets_up_existing_model(
        self, mock_sync, mock_os, mock_shutil, mock_size, mock_resolve, tmp_state_dir
    ):
        mock_resolve.return_value = "abc123"
        mock_size.return_value = 0
        mock_shutil.disk_usage.return_value = (100, 10, 90)

        mock_bg = MagicMock()
        req = ModelSetup(
            hf_repo="test/repo",
            quant="Q4_K_M",
            symlink_name="test-model",
            parameters="{}",
            revision="latest",
        )

        save_state({"test-model": {"repo": "test/repo", "quant": "Q4_K_M", "revision": "abc123", "status": "ready"}})
        result = await setup_model(req, mock_bg)

        state = load_state()
        assert "test-model" in state
        assert state["test-model"]["status"] == "ready"
        assert state["test-model"]["revision"] == "abc123"

    @patch("backend.api.resolve_sha")
    @patch("backend.api.pre_flight_size")
    @patch("backend.api.shutil")
    @patch("backend.api.os")
    @pytest.mark.asyncio
    async def test_triggers_download_when_needed(
        self, mock_os, mock_shutil, mock_size, mock_resolve, tmp_state_dir
    ):
        mock_resolve.return_value = "abc123"
        mock_size.return_value = 1000
        mock_shutil.disk_usage.return_value = (100000, 1000, 5000)

        mock_bg = MagicMock()
        req = ModelSetup(
            hf_repo="test/repo",
            quant="Q4_K_M",
            symlink_name="new-model",
            parameters="{}",
            revision="latest",
        )

        save_state({})
        result = await setup_model(req, mock_bg)

        assert "Provisioning" in result["status"]
        mock_bg.add_task.assert_called_once()


class TestDeleteConfig:
    @pytest.mark.asyncio
    async def test_deletes_config(self, sample_state):
        result = await delete_config("test-model")
        state = load_state()
        assert "test-model" not in state
        assert result["status"] == "Config deleted!"

    @pytest.mark.asyncio
    async def test_does_nothing_when_missing(self, empty_state):
        result = await delete_config("nonexistent")
        assert result["status"] == "Config deleted!"

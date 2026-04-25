from unittest.mock import MagicMock, patch

import pytest

from backend.hf_hub import resolve_sha, get_commits, pre_flight_size


class TestResolveSha:
    @patch("backend.hf_hub.HfApi")
    def test_returns_sha(self, mock_api_class):
        mock_api = MagicMock()
        mock_api.model_info.return_value = MagicMock(sha="abc123def456")
        mock_api_class.return_value = mock_api

        sha = resolve_sha("test/repo", "main")
        assert sha == "abc123def456"
        mock_api.model_info.assert_called_once_with("test/repo", revision="main")

    @patch("backend.hf_hub.HfApi")
    def test_resolves_with_ref(self, mock_api_class):
        mock_api = MagicMock()
        mock_api.model_info.return_value = MagicMock(sha="sha123")
        mock_api_class.return_value = mock_api

        resolve_sha("test/repo", "v1.0")
        mock_api.model_info.assert_called_once_with("test/repo", revision="v1.0")


class TestGetCommits:
    @patch("backend.hf_hub.HfApi")
    def test_returns_commits_list(self, mock_api_class):
        mock_api = MagicMock()
        commit1 = MagicMock()
        commit1.commit_id = "aaa"
        commit1.short_commit_message = "first commit"
        commit1.committer_timestamp = 1000000
        commit2 = MagicMock()
        commit2.commit_id = "bbb"
        commit2.short_commit_message = "second commit"
        commit2.committer_timestamp = 2000000
        mock_api.list_repo_commits.return_value = [commit1, commit2]
        mock_api_class.return_value = mock_api

        result = get_commits("test/repo")
        assert len(result) == 2
        assert result[0]["sha"] == "bbb"
        assert result[0]["date"] == 2000000
        assert result[0]["message"] == "second commit"
        assert result[1]["sha"] == "aaa"
        assert result[1]["message"] == "first commit"

    @patch("backend.hf_hub.HfApi")
    def test_uses_message_attribute_if_no_short(self, mock_api_class):
        mock_api = MagicMock()
        commit = MagicMock()
        commit.commit_id = "abc"
        commit.short_commit_message = None
        commit.message = "long message\nwith newline"
        commit.committer_timestamp = 1000
        mock_api.list_repo_commits.return_value = [commit]
        mock_api_class.return_value = mock_api

        result = get_commits("test/repo")
        assert result[0]["message"] == "long message"

    @patch("backend.hf_hub.HfApi")
    def test_limits_commits(self, mock_api_class):
        mock_api = MagicMock()
        mock_api.list_repo_commits.return_value = [MagicMock()] * 100
        mock_api_class.return_value = mock_api

        result = get_commits("test/repo", limit=10)
        assert len(result) == 10
        mock_api.list_repo_commits.assert_called_once_with("test/repo", repo_type="model")

    @patch("backend.hf_hub.HfApi")
    def test_sorts_newest_first(self, mock_api_class):
        mock_api = MagicMock()
        old = MagicMock()
        old.commit_id = "old"
        old.short_commit_message = "old"
        old.committer_timestamp = 100
        new = MagicMock()
        new.commit_id = "new"
        new.short_commit_message = "new"
        new.committer_timestamp = 999
        mock_api.list_repo_commits.return_value = [old, new]
        mock_api_class.return_value = mock_api

        result = get_commits("test/repo")
        assert result[0]["sha"] == "new"
        assert result[1]["sha"] == "old"


class TestPreFlightSize:
    @patch("backend.hf_hub.HfApi")
    def test_counts_matching_quant(self, mock_api_class):
        mock_api = MagicMock()
        file1 = MagicMock()
        file1.rfilename = "model-Q4_K_M.gguf"
        file1.size = 5000000000
        file2 = MagicMock()
        file2.rfilename = "model-Q5_K_M.gguf"
        file2.size = 6000000000
        mock_api.model_info.return_value = MagicMock(siblings=[file1, file2])
        mock_api_class.return_value = mock_api

        size = pre_flight_size("test/repo", "abc", "Q4_K_M", "")
        assert size == 5000000000

    @patch("backend.hf_hub.HfApi")
    def test_counts_mmproj_when_specified(self, mock_api_class):
        mock_api = MagicMock()
        file1 = MagicMock()
        file1.rfilename = "model-Q4_K_M.gguf"
        file1.size = 5000000000
        file2 = MagicMock()
        file2.rfilename = "mmproj-F16.gguf"
        file2.size = 900000000
        mock_api.model_info.return_value = MagicMock(siblings=[file1, file2])
        mock_api_class.return_value = mock_api

        size = pre_flight_size("test/repo", "abc", "Q4_K_M", "F16")
        assert size == 5900000000

    @patch("backend.hf_hub.HfApi")
    def test_excludes_non_gguf(self, mock_api_class):
        mock_api = MagicMock()
        file1 = MagicMock()
        file1.rfilename = "model.txt"
        file1.size = 999999999
        mock_api.model_info.return_value = MagicMock(siblings=[file1])
        mock_api_class.return_value = mock_api

        size = pre_flight_size("test/repo", "abc", "Q4_K_M", "")
        assert size == 0

    @patch("backend.hf_hub.HfApi")
    def test_excludes_files_without_size(self, mock_api_class):
        mock_api = MagicMock()
        file1 = MagicMock()
        file1.rfilename = "model-Q4_K_M.gguf"
        file1.size = None
        mock_api.model_info.return_value = MagicMock(siblings=[file1])
        mock_api_class.return_value = mock_api

        size = pre_flight_size("test/repo", "abc", "Q4_K_M", "")
        assert size == 0

    @patch("backend.hf_hub.HfApi")
    def test_ignores_mmproj_when_not_specified(self, mock_api_class):
        mock_api = MagicMock()
        file1 = MagicMock()
        file1.rfilename = "model-Q4_K_M.gguf"
        file1.size = 5000000000
        file2 = MagicMock()
        file2.rfilename = "mmproj-F16.gguf"
        file2.size = 900000000
        mock_api.model_info.return_value = MagicMock(siblings=[file1, file2])
        mock_api_class.return_value = mock_api

        size = pre_flight_size("test/repo", "abc", "Q4_K_M", "")
        assert size == 5000000000

from unittest.mock import MagicMock, patch

import pytest

from backend.cache import scan_cache, in_use_revisions, prune_unreferenced_revisions


class TestScanCache:
    @patch("backend.cache.scan_cache_dir")
    @patch("backend.cache.os")
    def test_returns_cache_info(self, mock_os, mock_scan, tmp_state_dir):
        mock_os.path.isdir.return_value = True
        mock_cache = MagicMock()
        mock_scan.return_value = mock_cache
        result = scan_cache()
        assert result == mock_cache
        mock_scan.assert_called_once()

    @patch("backend.cache.os")
    def test_returns_none_when_cache_dir_absent(self, mock_os, tmp_state_dir):
        mock_os.path.isdir.return_value = False
        result = scan_cache()
        assert result is None

    @patch("backend.cache.scan_cache_dir")
    def test_returns_none_on_exception(self, mock_scan, tmp_state_dir):
        mock_scan.side_effect = Exception("test error")
        result = scan_cache()
        assert result is None


class TestInUseRevisions:
    def test_empty_state(self, empty_state):
        result = in_use_revisions(empty_state)
        assert result == set()

    def test_single_config(self, sample_state):
        result = in_use_revisions(sample_state)
        assert ("test/repo", "abc123def456") in result

    def test_multiple_configs(self, tmp_state_dir):
        from backend.state import save_state
        state = {
            "model1": {"repo": "a/b", "revision": "rev1"},
            "model2": {"repo": "c/d", "revision": "rev2"},
            "model3": {"repo": "e/f", "revision": "rev3"},
        }
        save_state(state)
        result = in_use_revisions(state)
        assert result == {("a/b", "rev1"), ("c/d", "rev2"), ("e/f", "rev3")}

    def test_skips_missing_repo(self, tmp_state_dir):
        from backend.state import save_state
        state = {
            "bad": {"revision": "rev1"},
            "good": {"repo": "a/b", "revision": "rev2"},
        }
        save_state(state)
        result = in_use_revisions(state)
        assert result == {("a/b", "rev2")}

    def test_skips_missing_revision(self, tmp_state_dir):
        from backend.state import save_state
        state = {
            "bad": {"repo": "a/b"},
            "good": {"repo": "c/d", "revision": "rev2"},
        }
        save_state(state)
        result = in_use_revisions(state)
        assert result == {("c/d", "rev2")}


class TestPruneUnreferencedRevisions:
    @patch("backend.cache.scan_cache")
    def test_no_revisions_to_prune(self, mock_scan, sample_state):
        mock_cache = MagicMock()
        mock_cache.repos = []
        mock_scan.return_value = mock_cache
        prune_unreferenced_revisions(sample_state)

    def test_removes_orphan_repo_dirs(self, tmp_state_dir):
        cache_dir = tmp_state_dir / ".cache"
        cache_dir.mkdir(exist_ok=True)

        orphan = cache_dir / "models--orphan--repo"
        orphan.mkdir(exist_ok=True)

        used = cache_dir / "models--test--repo"
        used.mkdir(exist_ok=True)

        prune_unreferenced_revisions({"test-model": {"repo": "test/repo"}})

        assert not orphan.exists()
        assert used.exists()

    def test_handles_empty_cache_dir(self, tmp_state_dir, sample_state):
        cache_dir = tmp_state_dir / ".cache"
        cache_dir.mkdir(exist_ok=True)
        prune_unreferenced_revisions(sample_state)

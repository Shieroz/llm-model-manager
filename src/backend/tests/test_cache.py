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


class TestInUseFiles:
    def test_maps_quant_and_mmproj(self, sample_state):
        from backend.cache import in_use_files
        result = in_use_files(sample_state)
        key = ("test/repo", "abc123def456")
        assert result[key]["quants"] == {"Q4_K_M"}
        assert result[key]["mmprojs"] == set()


class TestPruneUnreferencedFiles:
    def _mock_file(self, tmp_path, name):
        blob = tmp_path / f"blob-{name}"
        blob.write_text("x")
        link = tmp_path / name
        link.symlink_to(blob)
        f = MagicMock()
        f.file_name = name
        f.file_path = link
        f.blob_path = blob
        f.size_on_disk = 1
        return f, blob, link

    def _mock_cache(self, sha, files):
        rev = MagicMock()
        rev.commit_hash = sha
        rev.files = files
        repo = MagicMock()
        repo.repo_id = "test/repo"
        repo.revisions = [rev]
        cache = MagicMock()
        cache.repos = [repo]
        return cache

    @patch("backend.cache.scan_cache")
    def test_removes_orphan_quant_keeps_referenced(self, mock_scan, sample_state, tmp_path):
        from backend.cache import prune_unreferenced_files
        used_f, used_blob, used_link = self._mock_file(tmp_path, "model-Q4_K_M.gguf")
        orphan_f, orphan_blob, orphan_link = self._mock_file(tmp_path, "model-Q8_0.gguf")
        mock_scan.return_value = self._mock_cache("abc123def456", [used_f, orphan_f])
        prune_unreferenced_files(sample_state)
        assert used_link.exists() and used_blob.exists()
        assert not orphan_link.exists() and not orphan_blob.exists()

    @patch("backend.cache.scan_cache")
    def test_protects_mtp_head_file(self, mock_scan, sample_state, tmp_path):
        from backend.cache import prune_unreferenced_files
        head_f, head_blob, head_link = self._mock_file(tmp_path, "mtp-model-Q8_0.gguf")
        mock_scan.return_value = self._mock_cache("abc123def456", [head_f])
        prune_unreferenced_files(sample_state)
        assert head_link.exists()  # "mtp" in name -> never auto-deleted

    @patch("backend.cache.scan_cache")
    def test_skips_unreferenced_revision(self, mock_scan, sample_state, tmp_path):
        from backend.cache import prune_unreferenced_files
        f, blob, link = self._mock_file(tmp_path, "model-Q8_0.gguf")
        mock_scan.return_value = self._mock_cache("UNREFERENCED_SHA", [f])
        prune_unreferenced_files(sample_state)
        assert link.exists()  # revision not in state -> left to revision-level prune

import json

import pytest
from pydantic import ValidationError

from backend.models import ModelSetup, RevisionDeleteReq, RpcModeReq


class TestModelSetup:
    def test_minimal_fields(self):
        req = ModelSetup(
            hf_repo="test/repo",
            quant="Q4_K_M",
            symlink_name="my-model",
            parameters="{}",
            revision="abc123",
        )
        assert req.hf_repo == "test/repo"
        assert req.quant == "Q4_K_M"
        assert req.mmproj == ""
        assert req.symlink_name == "my-model"
        assert req.original_name == ""
        assert req.parameters == "{}"
        assert req.revision == "abc123"

    def test_all_fields(self):
        req = ModelSetup(
            hf_repo="test/repo",
            quant="Q5_K_M",
            mmproj="BF16",
            symlink_name="my-model",
            original_name="old-name",
            parameters='{"ctx_size": 8192}',
            revision="def456",
        )
        assert req.mmproj == "BF16"
        assert req.original_name == "old-name"
        assert json.loads(req.parameters) == {"ctx_size": 8192}

    def test_revision_defaults_to_latest(self):
        req = ModelSetup(
            hf_repo="test/repo",
            quant="Q4_K_M",
            symlink_name="my-model",
            parameters="{}",
        )
        assert req.revision == "latest"

    def test_mmproj_defaults_to_empty(self):
        req = ModelSetup(
            hf_repo="test/repo",
            quant="Q4_K_M",
            symlink_name="my-model",
            parameters="{}",
        )
        assert req.mmproj == ""


class TestRevisionDeleteReq:
    def test_valid(self):
        req = RevisionDeleteReq(repo="test/repo", revision="abc123")
        assert req.repo == "test/repo"
        assert req.revision == "abc123"

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            RevisionDeleteReq(repo="test/repo")

        with pytest.raises(ValidationError):
            RevisionDeleteReq(revision="abc123")


class TestRpcModeReq:
    def test_enabled_true(self):
        req = RpcModeReq(enabled=True)
        assert req.enabled is True

    def test_enabled_false(self):
        req = RpcModeReq(enabled=False)
        assert req.enabled is False

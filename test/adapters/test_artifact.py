"""Tests for artifact directory structure and helpers."""
import json

import pytest

from rdagent.adapters.artifact_utils import (
    SCHEMA_VERSION,
    create_round_dir,
    create_run_dir,
    load_artifact,
    resolve_artifact,
    save_artifact,
)


@pytest.fixture
def tmp_base(tmp_path):
    return tmp_path / "artifacts"


class TestCreateRunDir:
    def test_creates_directory(self, tmp_base):
        run_dir = create_run_dir(tmp_base, "run_001")
        assert run_dir.is_dir()
        assert (run_dir / "trace.json").exists()
        assert (run_dir / "round_manifest.json").exists()

    def test_trace_json_has_required_fields(self, tmp_base):
        run_dir = create_run_dir(tmp_base, "run_002")
        data = load_artifact(run_dir / "trace.json")
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["artifact_type"] == "trace"
        assert data["hist"] == []
        assert data["dag_parent"] == []

    def test_idempotent(self, tmp_base):
        run_dir1 = create_run_dir(tmp_base, "run_003")
        run_dir2 = create_run_dir(tmp_base, "run_003")
        assert run_dir1 == run_dir2


class TestCreateRoundDir:
    def test_creates_round_directory(self, tmp_base):
        run_dir = create_run_dir(tmp_base, "run_010")
        round_dir = create_round_dir(run_dir, 0)
        assert round_dir.is_dir()
        assert (round_dir / "implementations").is_dir()
        assert (round_dir / "manifest.json").exists()

    def test_manifest_has_required_fields(self, tmp_base):
        run_dir = create_run_dir(tmp_base, "run_011")
        round_dir = create_round_dir(run_dir, 0)
        data = load_artifact(round_dir / "manifest.json")
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["loop_idx"] == 0
        assert "created_at" in data


class TestResolveArtifact:
    def test_resolves_existing(self, tmp_base):
        run_dir = create_run_dir(tmp_base, "run_020")
        round_dir = create_round_dir(run_dir, 0)
        save_artifact(round_dir / "hypothesis.json", {"schema_version": 1, "data": "test"})
        path = resolve_artifact(run_dir, 0, "hypothesis.json")
        assert path.exists()

    def test_raises_for_missing(self, tmp_base):
        run_dir = create_run_dir(tmp_base, "run_021")
        create_round_dir(run_dir, 0)
        with pytest.raises(FileNotFoundError):
            resolve_artifact(run_dir, 0, "nonexistent.json")


class TestSaveLoadArtifact:
    def test_roundtrip(self, tmp_base):
        tmp_base.mkdir(parents=True, exist_ok=True)
        path = tmp_base / "test.json"
        original = {"schema_version": 1, "key": "value"}
        save_artifact(path, original)
        loaded = load_artifact(path)
        assert loaded["key"] == "value"
        assert "updated_at" in loaded

    def test_load_missing_schema_version(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"no_schema": True}))
        with pytest.raises(ValueError, match="Missing schema_version"):
            load_artifact(path)

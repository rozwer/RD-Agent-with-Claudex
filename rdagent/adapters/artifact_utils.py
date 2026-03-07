"""Artifact directory helpers for the RDLoop adapter layer."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def create_run_dir(base: Path, run_id: str) -> Path:
    """Create <base>/<run_id>/ with empty trace.json and round_manifest.json."""
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_artifact(
        run_dir / "trace.json",
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "trace",
            "artifact_id": run_id,
            "class_path": "rdagent.core.proposal.Trace",
            "scen_class_path": "",
            "knowledge_base_class_path": None,
            "hist": [],
            "dag_parent": [],
            "idx2loop_id": {},
            "current_selection": [-1],
        },
    )
    save_artifact(
        run_dir / "round_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "round_manifest",
            "artifact_id": run_id,
            "rounds": [],
        },
    )
    return run_dir


def create_round_dir(run_dir: Path, round_idx: int) -> Path:
    """Create round_<N>/ and implementations/ subdirectory. Initialize manifest.json."""
    round_dir = run_dir / f"round_{round_idx}"
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "implementations").mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    save_artifact(
        round_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "session_id": "",
            "rdloop_class_path": "",
            "prop_setting": {},
            "loop_idx": round_idx,
            "step_idx": {},
            "latest_checkpoint": {"loop_id": round_idx, "step_idx": 0, "step_name": ""},
            "trace_ref": "../trace.json",
            "created_at": now,
        },
    )
    return round_dir


def resolve_artifact(run_dir: Path, round_idx: int, name: str) -> Path:
    """Return run_dir/round_<N>/<name>. Raise FileNotFoundError if missing."""
    path = run_dir / f"round_{round_idx}" / name
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return path


def load_artifact(path: Path) -> dict:
    """Load JSON artifact and validate schema_version."""
    with open(path) as f:
        data = json.load(f)
    if "schema_version" not in data:
        raise ValueError(f"Missing schema_version in {path}")
    return data


def save_artifact(path: Path, data: dict) -> None:
    """Save JSON artifact with updated_at timestamp."""
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

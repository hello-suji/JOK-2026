from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def read_text(path: Path) -> str:
    """Read a text file with UTF-8, falling back to Latin-1 on decode errors."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def write_json(data: Any, path: str | Path) -> None:
    """Write JSON artifact with the repository's default formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
    tmp_path.replace(output_path)


def write_json_atomic(data: Any, path: str | Path) -> None:
    """Write JSON atomically with fsync, without changing write_json semantics."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def read_json_object(path: str | Path) -> Optional[Dict[str, Any]]:
    """Read a JSON object, returning None for missing/invalid/non-object files."""
    input_path = Path(path)
    if not input_path.exists():
        return None
    try:
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if (
        data.get("schema_version") == "root-artifact-reference-v1"
        and isinstance(data.get("artifact_ref"), dict)
    ):
        try:
            # Local import avoids the artifact_hash -> file_io import cycle.
            from src.utils.artifact_hash import resolve_content_addressed_json

            resolved = resolve_content_addressed_json(
                data["artifact_ref"],
                input_path.parent,
                expected_instance_id=data.get("instance_id"),
                expected_outer_iteration=data.get("outer_iteration"),
                expected_candidate_sha256=data.get("candidate_sha256"),
                expected_artifact_type=data.get("artifact_type"),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        if isinstance(resolved, dict):
            resolved = dict(resolved)
            resolved.setdefault("source_iteration", data.get("source_iteration"))
            resolved.setdefault("candidate_id", data.get("candidate_id"))
            resolved.setdefault("candidate_hash", data.get("candidate_sha256"))
            return resolved
        return None
    return data

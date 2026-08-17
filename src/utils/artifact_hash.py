from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.utils.file_io import write_json_atomic


def sha256_text(text: str) -> str:
    """Return a stable sha256 for generated patch artifacts."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of the exact bytes stored at ``path``."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_evidence_reference(
    path: str | Path,
    run_root: str | Path,
    *,
    instance_id: str,
    outer_iteration: int,
    candidate_sha256: str | None,
    artifact_type: str,
) -> dict[str, Any]:
    """Build a strict, portable ``evidence_ref-v1`` for an existing artifact.

    Paths are always relative to the instance run root.  The reference owns
    enough identity to reject accidental reuse across instances, iterations,
    and generated candidates.
    """
    root = Path(run_root).resolve()
    artifact_path = Path(path).resolve()
    try:
        relative_path = artifact_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("evidence artifact must be inside the run root") from exc
    if not artifact_path.is_file():
        raise FileNotFoundError(f"evidence artifact is missing: {artifact_path}")
    if outer_iteration < 1:
        raise ValueError("outer_iteration must be a positive integer")
    return {
        "schema_version": "evidence_ref-v1",
        "relative_path": relative_path.as_posix(),
        "sha256": sha256_file(artifact_path),
        "instance_id": str(instance_id),
        "outer_iteration": int(outer_iteration),
        "candidate_sha256": str(candidate_sha256 or "") or None,
        "artifact_type": str(artifact_type),
        "byte_size": artifact_path.stat().st_size,
    }


def write_content_addressed_json(
    value: Mapping[str, Any] | list[Any],
    root: str | Path,
    *,
    schema_version: str,
    filename_prefix: str = "evidence",
    instance_id: str = "",
    outer_iteration: int = 1,
    candidate_sha256: str | None = None,
    artifact_type: str = "json_evidence",
) -> dict[str, Any]:
    """Write one canonical JSON artifact and return a compact reference.

    Existing files with the same digest are left untouched, making this safe
    for resume and root-alias synchronization.
    """
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    digest = sha256_text(encoded)
    directory = Path(root) / "canonical_evidence"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{filename_prefix}-{digest}.json"
    if not path.exists():
        write_json_atomic(json.loads(encoded), path)
    reference = build_evidence_reference(
        path,
        root,
        instance_id=instance_id,
        outer_iteration=outer_iteration,
        candidate_sha256=candidate_sha256,
        artifact_type=artifact_type,
    )
    reference["content_schema_version"] = schema_version
    return reference


def resolve_content_addressed_json(
    reference: Mapping[str, Any],
    run_root: str | Path,
    *,
    expected_instance_id: str | None = None,
    expected_outer_iteration: int | None = None,
    expected_candidate_sha256: str | None = None,
    expected_artifact_type: str | None = None,
) -> Any:
    """Resolve a strict evidence reference, failing closed on ownership drift."""
    if reference.get("schema_version") != "evidence_ref-v1":
        raise ValueError("unsupported evidence reference schema")
    raw_relative_path = str(reference.get("relative_path") or "")
    relative_path = Path(raw_relative_path)
    if not raw_relative_path or relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("evidence reference path must be a safe run-relative path")
    root = Path(run_root).resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("evidence reference escapes the run root") from exc
    ownership_expectations = {
        "instance_id": expected_instance_id,
        "outer_iteration": expected_outer_iteration,
        "candidate_sha256": expected_candidate_sha256,
        "artifact_type": expected_artifact_type,
    }
    for field, expected in ownership_expectations.items():
        if expected is not None and reference.get(field) != expected:
            raise ValueError(f"evidence reference {field} ownership mismatch")
    if not path.is_file():
        raise FileNotFoundError(f"canonical evidence artifact is missing: {path}")
    if int(reference.get("byte_size") or -1) != path.stat().st_size:
        raise ValueError("canonical evidence byte-size mismatch")
    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    expected = str(reference.get("sha256") or "")
    if digest != expected:
        raise ValueError("canonical evidence digest mismatch")
    payload = json.loads(raw_bytes.decode("utf-8"))
    if isinstance(payload, Mapping):
        payload_instance = payload.get("instance_id")
        if payload_instance is None and isinstance(payload.get("payload"), Mapping):
            payload_instance = payload["payload"].get("instance_id")
        if payload_instance is not None and str(payload_instance) != str(reference["instance_id"]):
            raise ValueError("canonical evidence payload instance ownership mismatch")
        payload_iteration = payload.get("outer_iteration", payload.get("iteration"))
        if payload_iteration is not None and int(payload_iteration) != int(reference["outer_iteration"]):
            raise ValueError("canonical evidence payload iteration ownership mismatch")
        payload_candidate = payload.get("candidate_sha256") or payload.get("generated_patch_sha256")
        if payload_candidate is None and isinstance(payload.get("payload"), Mapping):
            payload_candidate = payload["payload"].get("generated_patch_sha256")
        reference_candidate = reference.get("candidate_sha256")
        if payload_candidate and reference_candidate and str(payload_candidate) != str(reference_candidate):
            raise ValueError("canonical evidence payload candidate ownership mismatch")
    return payload

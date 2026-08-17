from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.contracts.feature_flags import V22FeatureFlags, resolve_feature_flags
from src.utils.artifact_hash import sha256_file, sha256_text
from src.utils.file_io import read_json_object


M8_INPUT_FINGERPRINT_SCHEMA = "m8-input-fingerprint-v1"


def _artifact_component(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return {"path": path.name, "sha256": sha256_file(path)}
    except (OSError, UnicodeError):
        return None


def _m6_identity_artifact(output_dir: Path) -> dict[str, Any] | None:
    for name in (
        "alignment_execution.json",
        "m6_execution_result.json",
        "execution_result.json",
    ):
        path = output_dir / name
        payload = read_json_object(path) or {}
        inner = payload.get("payload") if isinstance(payload.get("payload"), Mapping) else payload
        if inner.get("generated_patch_sha256"):
            return _artifact_component(path)
    return None


def _feature_component(
    feature_flags: V22FeatureFlags | Mapping[str, Any] | None,
    *,
    recorded: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if feature_flags is None:
        recorded_component = (
            recorded.get("components", {}).get("feature_flags")
            if isinstance(recorded, Mapping)
            and isinstance(recorded.get("components"), Mapping)
            else None
        )
        if isinstance(recorded_component, Mapping):
            return dict(recorded_component)
        feature_flags = resolve_feature_flags()
    resolved = (
        feature_flags
        if isinstance(feature_flags, V22FeatureFlags)
        else resolve_feature_flags(feature_flags)
    )
    values = resolved.to_dict()
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return {
        "config_id": resolved.config_id,
        "sha256": sha256_text(canonical),
        "values": values,
    }


def build_m8_input_fingerprint(
    output_dir: str | Path,
    *,
    feature_flags: V22FeatureFlags | Mapping[str, Any] | None = None,
    recorded: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Bind an M8 result to all mutable evidence that controls its meaning."""
    root = Path(output_dir)
    components = {
        "generated_patch": _artifact_component(root / "generated_test.patch"),
        "generated_candidate": _artifact_component(root / "generated_test.json"),
        "m6_evidence": _m6_identity_artifact(root),
        "m7_admission": _artifact_component(root / "alignment_result.json"),
        "feature_flags": _feature_component(feature_flags, recorded=recorded),
        "m8_execution": _artifact_component(root / "m8_final_execution_result.json"),
    }
    if any(value is None for value in components.values()):
        return None
    canonical = json.dumps(components, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": M8_INPUT_FINGERPRINT_SCHEMA,
        "sha256": sha256_text(canonical),
        "components": components,
    }


def m8_input_fingerprint_is_fresh(
    output_dir: str | Path,
    final_eval: Mapping[str, Any],
    *,
    feature_flags: V22FeatureFlags | Mapping[str, Any] | None = None,
) -> bool:
    recorded = final_eval.get("m8_input_fingerprint")
    if not isinstance(recorded, Mapping):
        return False
    if recorded.get("schema_version") != M8_INPUT_FINGERPRINT_SCHEMA:
        return False
    current = build_m8_input_fingerprint(
        output_dir,
        feature_flags=feature_flags,
        recorded=recorded,
    )
    return bool(current and current.get("sha256") == recorded.get("sha256"))

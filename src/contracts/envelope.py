from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from src.contracts.feature_flags import V22FeatureFlags, core_only_feature_flags, resolve_feature_flags


SCHEMA_VERSION = "v22.common_contracts.v1"


@dataclass
class TraceabilityEnvelope:
    schema_version: str
    instance_id: str
    run_id: str
    module: str
    config_id: str
    created_at: str
    feature_flags: dict[str, bool]
    payload: dict[str, Any]
    iteration: Optional[int] = None
    seed: Optional[int] = None
    model: Optional[str] = None
    prompt_version: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactEnvelope:
    schema_version: str
    method_version: str
    module: str
    status: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    token_usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_envelope(
    *,
    instance_id: str,
    run_id: str,
    module: str,
    payload: Mapping[str, Any],
    feature_flags: V22FeatureFlags | Mapping[str, bool] | None = None,
    config_id: str | None = None,
    iteration: int | None = None,
    seed: int | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    created_at: str | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> TraceabilityEnvelope:
    flags = feature_flags or core_only_feature_flags()
    resolved_flags = flags if isinstance(flags, V22FeatureFlags) else resolve_feature_flags(flags)
    flag_dict = resolved_flags.to_dict()
    envelope = TraceabilityEnvelope(
        schema_version=schema_version,
        instance_id=instance_id,
        run_id=run_id,
        module=module,
        config_id=config_id or _config_id_from_flags(resolved_flags),
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        feature_flags=flag_dict,
        payload=dict(payload),
        iteration=iteration,
        seed=seed,
        model=model,
        prompt_version=prompt_version,
    )
    validate_envelope(envelope.to_dict())
    return envelope


def _config_id_from_flags(flags: V22FeatureFlags | Mapping[str, bool]) -> str:
    if isinstance(flags, V22FeatureFlags):
        return flags.config_id
    return resolve_feature_flags(flags).config_id


def is_legacy_artifact(data: Mapping[str, Any]) -> bool:
    return "schema_version" not in data


def validate_envelope(data: TraceabilityEnvelope | Mapping[str, Any]) -> None:
    value = data.to_dict() if isinstance(data, TraceabilityEnvelope) else dict(data)
    required = {
        "schema_version",
        "instance_id",
        "run_id",
        "module",
        "config_id",
        "created_at",
        "feature_flags",
        "payload",
    }
    missing = sorted(k for k in required if k not in value)
    if missing:
        raise ValueError(f"envelope missing required fields: {', '.join(missing)}")
    if not isinstance(value["feature_flags"], dict):
        raise TypeError("feature_flags must be an object")
    if not isinstance(value["payload"], dict):
        raise TypeError("payload must be an object")


def make_artifact_envelope(
    *,
    module: str,
    status: str,
    method_version: str = "v22",
    inputs: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
    scores: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    token_usage: Mapping[str, Any] | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> ArtifactEnvelope:
    envelope = ArtifactEnvelope(
        schema_version=schema_version,
        method_version=method_version,
        module=module,
        status=status,
        inputs=dict(inputs or {}),
        outputs=dict(outputs or {}),
        scores=dict(scores or {}),
        provenance=dict(provenance or {}),
        diagnostics=dict(diagnostics or {}),
        token_usage=dict(token_usage or {}),
    )
    validate_artifact_envelope(envelope.to_dict())
    return envelope


def wrap_artifact(
    artifact: Mapping[str, Any],
    *,
    module: str,
    status: str,
    method_version: str = "v22",
    inputs: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
    scores: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    token_usage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach a generic envelope without removing legacy top-level fields."""
    legacy = dict(artifact)
    envelope = make_artifact_envelope(
        module=module,
        status=status,
        method_version=method_version,
        inputs=inputs,
        outputs=outputs,
        scores=scores,
        provenance=provenance,
        diagnostics={
            "legacy_top_level_keys": sorted(legacy),
            **dict(diagnostics or {}),
        },
        token_usage=token_usage,
    ).to_dict()
    return {**legacy, "artifact_envelope": envelope}


def validate_artifact_envelope(data: ArtifactEnvelope | Mapping[str, Any]) -> None:
    value = data.to_dict() if isinstance(data, ArtifactEnvelope) else dict(data)
    required = {
        "schema_version",
        "method_version",
        "module",
        "status",
        "inputs",
        "outputs",
        "scores",
        "provenance",
        "diagnostics",
        "token_usage",
    }
    missing = sorted(k for k in required if k not in value)
    if missing:
        raise ValueError(f"artifact envelope missing required fields: {', '.join(missing)}")
    for key in ("inputs", "outputs", "scores", "provenance", "diagnostics", "token_usage"):
        if not isinstance(value[key], dict):
            raise TypeError(f"{key} must be an object")


def deterministic_artifact_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

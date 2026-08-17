from __future__ import annotations

from typing import Any, Mapping, Optional

from src.contracts.models import FinalSetMembership

M1_TO_M7_MODULES = frozenset({"m1", "m2", "m3", "m4", "m5", "m5a", "m6", "m7"})
TRUSTED_SECTION_KEYS = frozenset({"inputs", "outputs", "scores", "provenance"})
FORBIDDEN_M1_TO_M7_TRUST_KEYS = frozenset(
    {
        "golden_patch",
        "golden_patch_data",
        "golden_patch_lines",
        "patch",
        "patch_lines",
        "patched_source",
        "patched_repo",
        "patched_repo_path",
        "post_patch",
        "post_patch_execution",
        "post_patch_execution_data",
        "post_patch_outcome",
        "post_patch_results",
        "fail_to_pass",
        "f_to_p",
        "patch_hit_rate",
        "phr",
        "m8_evaluation",
        "m8_results",
    }
)


def validate_score_range(value: Optional[float], *, name: str = "score") -> None:
    if value is None:
        return
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def validate_final_set_membership(membership: FinalSetMembership | Mapping[str, Any]) -> None:
    value = (
        membership
        if isinstance(membership, FinalSetMembership)
        else FinalSetMembership(**dict(membership))
    )
    if value.in_t_f2p and not value.in_t_final:
        raise ValueError("in_t_f2p implies in_t_final")


def validate_payload_model(model: Any) -> None:
    if not hasattr(model, "to_dict"):
        raise TypeError("contract payload must provide to_dict()")
    data = model.to_dict()
    if not isinstance(data, dict):
        raise TypeError("contract payload to_dict() must return dict")


def validate_artifact_trust_boundary(artifact: Mapping[str, Any]) -> None:
    envelope = artifact.get("artifact_envelope", artifact)
    if not isinstance(envelope, Mapping):
        raise TypeError("artifact envelope must be an object")
    module = str(envelope.get("module") or artifact.get("module") or "").lower()
    if module in M1_TO_M7_MODULES:
        for section_name in TRUSTED_SECTION_KEYS:
            section = envelope.get(section_name, {})
            if not isinstance(section, Mapping):
                continue
            path = _find_forbidden_key(section, FORBIDDEN_M1_TO_M7_TRUST_KEYS)
            if path:
                raise ValueError(
                    f"{module.upper()} artifact trusted section declares forbidden data: {path}"
                )
    validate_trusted_provenance(envelope.get("provenance", {}))


def validate_trusted_provenance(provenance: Mapping[str, Any]) -> None:
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be an object")
    trusted = provenance.get("trusted", {})
    if trusted in ({}, None):
        return
    if not isinstance(trusted, Mapping):
        raise TypeError("trusted provenance must be an object")
    source = trusted.get("source")
    if source != "repository_validator":
        raise ValueError("trusted provenance must come from repository_validator")
    if trusted.get("validated") is not True:
        raise ValueError("trusted provenance must be explicitly validated")


def validate_m8_does_not_mutate_m7_admission(
    before_candidates: list[Mapping[str, Any]],
    after_candidates: list[Mapping[str, Any]],
) -> None:
    before = _admission_membership_by_id(before_candidates)
    after = _admission_membership_by_id(after_candidates)
    if before != after:
        raise ValueError("M8 results cannot mutate M7 admission membership")


def _admission_membership_by_id(candidates: list[Mapping[str, Any]]) -> dict[str, bool]:
    from src.contracts.final_sets import admitted_to_final_set

    return {
        str(candidate.get("candidate_id") or candidate.get("test_id") or index): admitted_to_final_set(candidate)
        for index, candidate in enumerate(candidates)
    }


def _find_forbidden_key(value: Mapping[str, Any], forbidden: frozenset[str], prefix: str = "") -> str:
    for key, item in value.items():
        key_text = str(key)
        path = f"{prefix}.{key_text}" if prefix else key_text
        if key_text.lower() in forbidden:
            return path
        if isinstance(item, Mapping):
            found = _find_forbidden_key(item, forbidden, path)
            if found:
                return found
    return ""

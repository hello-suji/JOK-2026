from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Mapping


CANONICAL_FEATURE_FLAGS: tuple[str, ...] = (
    "m2_formula_ranking",
    "m3_adaptive_self_consistency",
    "m4_scenario_score",
    "m4_rra",
    "m4_multi_formula_consensus",
    "m6_cumulative_fp",
    "m6_sbfl",
    "m6_execution_stability",
    "m6_pyurify",
    "m7_sbfl_weighted_coverage",
    "m8_dynamic_slice_cc",
    "m1_llm_refinement",
    "m2_llm_semantic_matching",
    "m5_candor",
    "m5a_llm_error_refinement",
    "m6_flitsr",
    "m6_artemis",
    "m7_llm_scenario_refinement",
    "m8_ssd",
)

CORE_MIGRATION_FLAGS: tuple[str, ...] = CANONICAL_FEATURE_FLAGS[:11]
OPTIONAL_EXTENSION_FLAGS: tuple[str, ...] = CANONICAL_FEATURE_FLAGS[11:]

LEGACY_FLAG_ALIASES: dict[str, str] = {
    "enable_m1_llm_clue_refinement": "m1_llm_refinement",
    "enable_m2_llm_semantic_matching": "m2_llm_semantic_matching",
    "enable_m5_candor": "m5_candor",
    "enable_m5a_llm_error_refinement": "m5a_llm_error_refinement",
    "enable_m6_flitsr": "m6_flitsr",
    "enable_m6_artemis": "m6_artemis",
    "enable_m7_llm_feedback_refinement": "m7_llm_scenario_refinement",
}

FEATURE_FLAG_ENV_PREFIX = "V22_FEATURE_FLAG_"


@dataclass(frozen=True)
class V22FeatureFlags:
    m2_formula_ranking: bool = False
    m3_adaptive_self_consistency: bool = False
    m4_scenario_score: bool = False
    m4_rra: bool = False
    m4_multi_formula_consensus: bool = False
    m6_cumulative_fp: bool = False
    m6_sbfl: bool = False
    m6_execution_stability: bool = False
    m6_pyurify: bool = False
    m7_sbfl_weighted_coverage: bool = False
    m8_dynamic_slice_cc: bool = False
    m1_llm_refinement: bool = False
    m2_llm_semantic_matching: bool = False
    m5_candor: bool = False
    m5a_llm_error_refinement: bool = False
    m6_flitsr: bool = False
    m6_artemis: bool = False
    m7_llm_scenario_refinement: bool = False
    m8_ssd: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    def to_legacy_alias_dict(self) -> dict[str, bool]:
        values = self.to_dict()
        return {legacy: values[canonical] for legacy, canonical in LEGACY_FLAG_ALIASES.items()}

    @property
    def config_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"v22-flags-{digest}"

    @property
    def enable_m1_llm_clue_refinement(self) -> bool:
        return self.m1_llm_refinement

    @property
    def enable_m2_llm_semantic_matching(self) -> bool:
        return self.m2_llm_semantic_matching

    @property
    def enable_m5_candor(self) -> bool:
        return self.m5_candor

    @property
    def enable_m5a_llm_error_refinement(self) -> bool:
        return self.m5a_llm_error_refinement

    @property
    def enable_m6_flitsr(self) -> bool:
        return self.m6_flitsr

    @property
    def enable_m6_artemis(self) -> bool:
        return self.m6_artemis

    @property
    def enable_m7_llm_feedback_refinement(self) -> bool:
        return self.m7_llm_scenario_refinement


def full_feature_flags() -> V22FeatureFlags:
    return V22FeatureFlags(**{name: True for name in CANONICAL_FEATURE_FLAGS})


def core_only_feature_flags() -> V22FeatureFlags:
    return V22FeatureFlags(**{name: False for name in CANONICAL_FEATURE_FLAGS})


def resolve_feature_flags(
    values: Mapping[str, Any] | None = None,
    *,
    base: V22FeatureFlags | None = None,
    env: Mapping[str, str] | None = None,
) -> V22FeatureFlags:
    resolved = (base or core_only_feature_flags()).to_dict()
    provided = _normalize_flag_keys(values or {})
    unknown = sorted(set(provided) - set(CANONICAL_FEATURE_FLAGS))
    if unknown:
        raise ValueError(f"unknown v22 feature flags: {', '.join(unknown)}")
    for name, value in provided.items():
        if not isinstance(value, bool):
            raise TypeError(f"feature flag {name} must be bool")
        resolved[name] = value
    for name, value in _feature_flags_from_env(os.environ if env is None else env).items():
        resolved[name] = value
    return V22FeatureFlags(**resolved)


def _normalize_flag_keys(values: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for name, value in values.items():
        canonical = LEGACY_FLAG_ALIASES.get(str(name), str(name))
        if canonical in normalized and normalized[canonical] != value:
            raise ValueError(f"conflicting values for v22 feature flag: {canonical}")
        normalized[canonical] = value
    return normalized


def _feature_flags_from_env(env: Mapping[str, str]) -> dict[str, bool]:
    resolved: dict[str, bool] = {}
    env_names = {f"{FEATURE_FLAG_ENV_PREFIX}{name.upper()}": name for name in CANONICAL_FEATURE_FLAGS}
    env_names.update({
        f"{FEATURE_FLAG_ENV_PREFIX}{legacy.upper()}": canonical
        for legacy, canonical in LEGACY_FLAG_ALIASES.items()
    })
    for env_name, canonical in env_names.items():
        if env_name in env:
            resolved[canonical] = _parse_env_bool(env_name, env[env_name])
    return resolved


def _parse_env_bool(name: str, value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise TypeError(f"environment feature flag {name} must be exactly 'true' or 'false'")


@dataclass(frozen=True)
class ExtensionStatus:
    CANDOR_enabled: bool
    FLITSR_enabled: bool
    ARTEMIS_enabled: bool
    SSD_enabled: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


def extension_status(flags: V22FeatureFlags | Mapping[str, Any] | None = None) -> ExtensionStatus:
    resolved = flags if isinstance(flags, V22FeatureFlags) else resolve_feature_flags(flags)
    return ExtensionStatus(
        CANDOR_enabled=resolved.m5_candor,
        FLITSR_enabled=resolved.m6_flitsr,
        ARTEMIS_enabled=resolved.m6_artemis,
        SSD_enabled=resolved.m8_ssd,
    )

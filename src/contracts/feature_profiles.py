from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.contracts.feature_flags import (
    CANONICAL_FEATURE_FLAGS,
    LEGACY_FLAG_ALIASES,
    V22FeatureFlags,
    core_only_feature_flags,
    resolve_feature_flags,
)


LEGACY_FEATURE_PROFILE = "legacy"
V22_CORE_FEATURE_PROFILE = "v22-core"
V22_FINAL_VERIFIED_FEATURE_PROFILE = "v22-final-verified"
V26_FEATURE_PROFILE = "v26"
V27_FEATURE_PROFILE = "v27"
V27R1_FEATURE_PROFILE = "v27r1"
V29_FEATURE_PROFILE = "v29"
V30_FEATURE_PROFILE = "v30"
V31_FEATURE_PROFILE = "v31"
V36_FEATURE_PROFILE = "v36"
V37_FEATURE_PROFILE = "v37"
FEATURE_PROFILE_NAMES: tuple[str, ...] = (
    LEGACY_FEATURE_PROFILE,
    V22_CORE_FEATURE_PROFILE,
    V22_FINAL_VERIFIED_FEATURE_PROFILE,
    V26_FEATURE_PROFILE,
    V27_FEATURE_PROFILE,
    V27R1_FEATURE_PROFILE,
    V29_FEATURE_PROFILE,
    V30_FEATURE_PROFILE,
    V31_FEATURE_PROFILE,
    V36_FEATURE_PROFILE,
    V37_FEATURE_PROFILE,
)

V22_CORE_ENABLED_FLAGS: tuple[str, ...] = (
    "m2_formula_ranking",
    "m3_adaptive_self_consistency",
    "m6_execution_stability",
    "m6_cumulative_fp",
    "m6_sbfl",
)

V22_FINAL_VERIFIED_FLAGS: dict[str, bool] = {
    "m1_llm_refinement": True,
    "m2_formula_ranking": True,
    "m2_llm_semantic_matching": True,
    "m3_adaptive_self_consistency": False,
    "m4_scenario_score": False,
    "m4_rra": False,
    "m4_multi_formula_consensus": False,
    "m5_candor": False,
    "m5a_llm_error_refinement": False,
    "m6_cumulative_fp": True,
    "m6_sbfl": True,
    "m6_execution_stability": False,
    "m6_pyurify": True,
    "m6_flitsr": False,
    "m6_artemis": False,
    "m7_sbfl_weighted_coverage": True,
    "m7_llm_scenario_refinement": True,
    "m8_dynamic_slice_cc": True,
    "m8_ssd": False,
}

V22_FINAL_FLAGS = V22_FINAL_VERIFIED_FLAGS
V22_FINAL_FEATURE_PROFILE = V22_FINAL_VERIFIED_FEATURE_PROFILE

V26_FLAGS: dict[str, bool] = {
    "m1_llm_refinement": False,
    "m2_formula_ranking": True,
    "m2_llm_semantic_matching": True,
    "m3_adaptive_self_consistency": False,
    "m4_scenario_score": False,
    "m4_rra": False,
    "m4_multi_formula_consensus": False,
    "m5_candor": False,
    "m5a_llm_error_refinement": True,
    "m6_cumulative_fp": True,
    "m6_sbfl": True,
    "m6_execution_stability": True,
    "m6_pyurify": True,
    "m6_flitsr": False,
    "m6_artemis": False,
    "m7_sbfl_weighted_coverage": True,
    "m7_llm_scenario_refinement": True,
    "m8_dynamic_slice_cc": False,
    "m8_ssd": False,
}

# v27 retains the validated v26 feature selection.  Revision-specific behavior
# is selected by the explicit profile identity in the pipeline, not by silently
# changing the meaning of a v26 flag bundle.
V27_FLAGS: dict[str, bool] = dict(V26_FLAGS)

# v27r1 is a correction revision, not a feature-selection experiment.  Its
# behavioral changes are keyed by the explicit profile identity.
V27R1_FLAGS: dict[str, bool] = dict(V27_FLAGS)

# v29 keeps the validated v27r1 generation selection and enables the existing
# independent M8 dynamic-slice CC implementation required by the v29 report.
# Formula and routing changes are selected by the explicit profile identity.
V29_FLAGS: dict[str, bool] = {
    **V27R1_FLAGS,
    # v29 primary experiment explicitly disables the optional stability pass.
    # Historical v26/v27/v27r1 profiles retain their existing behavior.
    "m6_execution_stability": False,
    "m8_dynamic_slice_cc": True,
}

# v30 keeps every v29 formula, threshold, and extension selection.  The
# methodology revisions are selected by the profile identity in orchestration
# and by the explicit v30 contracts; they are deliberately not represented as
# unrelated feature flags.  This makes v29 replay stable while allowing the
# v30 path to carry richer hypotheses, provenance, and bounded evidence.
V30_FLAGS: dict[str, bool] = dict(V29_FLAGS)

# v31 is deliberately isolated from v30.  It retains the validated feature
# selection while routing the generation/localization corrections through an
# explicit profile identity.
V31_FLAGS: dict[str, bool] = dict(V30_FLAGS)

# v36 is an isolated methodology profile.  Historical profiles retain their
# exact feature bundles and orchestration semantics.  Optional algorithms
# whose v36 procedures remain underspecified stay disabled rather than being
# represented by a heuristic approximation.
V36_FLAGS: dict[str, bool] = {
    **V31_FLAGS,
    "m1_llm_refinement": False,
    "m2_formula_ranking": True,
    "m2_llm_semantic_matching": True,
    "m3_adaptive_self_consistency": False,
    "m4_scenario_score": True,
    "m4_rra": True,
    "m4_multi_formula_consensus": False,
    "m5_candor": False,
    "m5a_llm_error_refinement": True,
    "m6_cumulative_fp": True,
    "m6_sbfl": True,
    "m6_execution_stability": False,
    "m6_pyurify": False,
    "m6_flitsr": False,
    "m6_artemis": False,
    "m7_sbfl_weighted_coverage": True,
    "m7_llm_scenario_refinement": True,
    "m8_dynamic_slice_cc": True,
    "m8_ssd": False,
}

# v37 is a correction profile. It inherits the v36 feature selection while
# selecting revised behavior through its explicit profile identity.
V37_FLAGS: dict[str, bool] = dict(V36_FLAGS)


@dataclass(frozen=True)
class FeatureFlagResolution:
    requested_feature_profile: str | None
    explicit_feature_overrides: dict[str, bool]
    effective_feature_flags: V22FeatureFlags
    feature_flag_resolution_provenance: dict[str, Any]

    def metadata(self) -> dict[str, Any]:
        return {
            "requested_feature_profile": self.requested_feature_profile,
            "explicit_feature_overrides": dict(self.explicit_feature_overrides),
            "effective_feature_flags": self.effective_feature_flags.to_dict(),
            "feature_flag_resolution_provenance": dict(
                self.feature_flag_resolution_provenance
            ),
        }


def feature_profile_flags(profile: str | None) -> V22FeatureFlags:
    """Return canonical flags for a named reproducible v22 feature profile."""
    if profile is None or profile == LEGACY_FEATURE_PROFILE:
        return core_only_feature_flags()
    if profile == V22_FINAL_VERIFIED_FEATURE_PROFILE:
        return resolve_feature_flags(
            V22_FINAL_VERIFIED_FLAGS,
            base=core_only_feature_flags(),
            env={},
        )
    if profile == V26_FEATURE_PROFILE:
        return resolve_feature_flags(
            V26_FLAGS,
            base=core_only_feature_flags(),
            env={},
        )
    if profile in {V27_FEATURE_PROFILE, V27R1_FEATURE_PROFILE, V29_FEATURE_PROFILE, V30_FEATURE_PROFILE, V31_FEATURE_PROFILE}:
        return resolve_feature_flags(
            V31_FLAGS
            if profile == V31_FEATURE_PROFILE
            else V30_FLAGS
            if profile == V30_FEATURE_PROFILE
            else V29_FLAGS
            if profile == V29_FEATURE_PROFILE
            else V27R1_FLAGS
            if profile == V27R1_FEATURE_PROFILE
            else V27_FLAGS,
            base=core_only_feature_flags(),
            env={},
        )
    if profile in {V36_FEATURE_PROFILE, V37_FEATURE_PROFILE}:
        return resolve_feature_flags(
            V37_FLAGS if profile == V37_FEATURE_PROFILE else V36_FLAGS,
            base=core_only_feature_flags(),
            env={},
        )
    if profile != V22_CORE_FEATURE_PROFILE:
        raise ValueError(f"unknown v22 feature profile: {profile}")

    enabled = {name: False for name in CANONICAL_FEATURE_FLAGS}
    for name in V22_CORE_ENABLED_FLAGS:
        enabled[name] = True
    return resolve_feature_flags(enabled, base=core_only_feature_flags(), env={})


def resolve_feature_profile(
    profile: str | None = None,
    explicit_overrides: Mapping[str, Any] | None = None,
) -> FeatureFlagResolution:
    """Resolve feature flags with profile precedence and strict canonical validation.

    Precedence is:
    canonical defaults < selected named profile < explicit per-flag overrides.
    """
    normalized_overrides = _normalize_explicit_overrides(explicit_overrides or {})
    if profile in {V36_FEATURE_PROFILE, V37_FEATURE_PROFILE}:
        retired = {
            "m1_llm_refinement",
            "m3_adaptive_self_consistency",
            "m4_multi_formula_consensus",
            "m5_candor",
            "m6_execution_stability",
        }
        enabled_retired = sorted(
            key for key in retired if normalized_overrides.get(key) is True
        )
        if enabled_retired:
            raise ValueError(
                f"{profile} retired feature(s) cannot be enabled: "
                + ", ".join(enabled_retired)
            )
    profile_flags = feature_profile_flags(profile)
    canonical_overrides = resolve_feature_flags(
        explicit_overrides or {},
        base=profile_flags,
        env={},
    )
    requested_profile = profile
    return FeatureFlagResolution(
        requested_feature_profile=requested_profile,
        explicit_feature_overrides=normalized_overrides,
        effective_feature_flags=canonical_overrides,
        feature_flag_resolution_provenance={
            "precedence": [
                "canonical_defaults",
                "selected_named_profile",
                "explicit_per_flag_overrides",
            ],
            "canonical_defaults": core_only_feature_flags().to_dict(),
            "selected_profile": requested_profile,
            "selected_profile_flags": profile_flags.to_dict(),
            "explicit_override_keys": sorted(normalized_overrides),
        },
    )


def parse_feature_flag_override(value: str) -> tuple[str, bool]:
    """Parse one CLI feature-flag override in canonical KEY=true|false form."""
    if "=" not in value:
        raise ValueError(f"malformed feature flag override: {value}")
    key, raw_bool = value.split("=", 1)
    if not key or raw_bool == "":
        raise ValueError(f"malformed feature flag override: {value}")
    if raw_bool == "true":
        return key, True
    if raw_bool == "false":
        return key, False
    raise TypeError(f"feature flag {key} must be exactly 'true' or 'false'")


def parse_feature_flag_overrides(values: Sequence[str] | None) -> dict[str, bool]:
    parsed: dict[str, bool] = {}
    for raw_value in values or ():
        key, bool_value = parse_feature_flag_override(raw_value)
        canonical_key = LEGACY_FLAG_ALIASES.get(key, key)
        resolve_feature_flags({canonical_key: bool_value}, env={})
        if canonical_key in parsed and parsed[canonical_key] != bool_value:
            raise ValueError(f"conflicting values for v22 feature flag: {canonical_key}")
        parsed[canonical_key] = bool_value
    return parsed


def _normalize_explicit_overrides(values: Mapping[str, Any]) -> dict[str, bool]:
    if not values:
        return {}
    normalized: dict[str, bool] = {}
    for key, value in values.items():
        canonical_key = LEGACY_FLAG_ALIASES.get(str(key), str(key))
        if canonical_key not in CANONICAL_FEATURE_FLAGS:
            resolve_feature_flags({canonical_key: value}, env={})
        if not isinstance(value, bool):
            resolve_feature_flags({canonical_key: value}, env={})
        if canonical_key in normalized and normalized[canonical_key] != value:
            raise ValueError(f"conflicting values for v22 feature flag: {canonical_key}")
        normalized[canonical_key] = value
    return normalized

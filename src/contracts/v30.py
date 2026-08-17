"""Explicit v30 producer/consumer contracts.

The v30 path keeps the legacy dictionaries at module boundaries for backward
compatibility, but normalizes the high-risk values into small typed records.
These records are intentionally patch-free: their provenance can only point to
the issue, pre-patch source view, or pre-patch execution.
"""
from __future__ import annotations

import posixpath
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


V30_MAX_LOCALIZATION_HYPOTHESES = 3
V30_MAX_SOURCE_FILES = 16
V30_MAX_TEST_FILES = 32
V30_MAX_TEST_NODES = 256
V30_MAX_SUPPLEMENTAL_EXECUTIONS = 6
V30_MIN_DISTINCT_PASSING_TESTS = 3


def _text(value: Any) -> str:
    return str(value or "").strip()


@dataclass(frozen=True)
class LocalizationHypothesis:
    hypothesis_id: str
    source_file: str
    function_name: str = ""
    class_name: str = ""
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    issue_clue_support: list[str] = field(default_factory=list)
    static_source_support: list[str] = field(default_factory=list)
    candidate_test_files: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    primary: bool = False
    uncertainty: str = ""

    def __post_init__(self) -> None:
        if not _text(self.hypothesis_id) or not _text(self.source_file):
            raise ValueError("v30 localization hypotheses require hypothesis_id and source_file")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("hypothesis confidence must be in [0, 1]")
        if any(_contains_forbidden_runtime_field(item) for item in self.provenance):
            raise ValueError("localization provenance contains forbidden post-patch field")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HypothesisScenario:
    scenario_id: str
    hypothesis_id: str
    setup: list[str]
    trigger: list[str]
    target: dict[str, Any]
    observed_buggy_behavior: str
    expected_behavior: str
    oracle: str
    issue_evidence: list[str] = field(default_factory=list)
    uncertainty: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = {
            "scenario_id": self.scenario_id,
            "hypothesis_id": self.hypothesis_id,
            "observed_buggy_behavior": self.observed_buggy_behavior,
            "expected_behavior": self.expected_behavior,
            "oracle": self.oracle,
        }
        if any(not _text(value) for value in required.values()):
            raise ValueError("v30 scenarios require complete hypothesis-bound fields")
        if not isinstance(self.target, Mapping) or not _text(self.target.get("source_file")):
            raise ValueError("v30 scenarios require a target source_file")
        if not self.setup or not self.trigger:
            raise ValueError("v30 scenarios require setup and trigger steps")
        if any(_contains_forbidden_runtime_field(item) for item in self.provenance):
            raise ValueError("scenario provenance contains forbidden post-patch field")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TestGenerationContextV30:
    framework: str
    runner: str
    imports: list[str]
    fixtures: list[str]
    setup_conventions: list[str]
    target_api: str
    test_examples: list[str]
    hypothesis_id: str
    scenario_id: str
    oracle_constraints: list[str]
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _text(self.framework) or not _text(self.runner):
            raise ValueError("v30 generation context requires framework and runner")
        if not _text(self.hypothesis_id) or not _text(self.scenario_id):
            raise ValueError("v30 generation context requires hypothesis/scenario identity")
        if any(_contains_forbidden_runtime_field(item) for item in self.provenance):
            raise ValueError("generation context provenance contains forbidden post-patch field")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class M5AResultV30:
    status: str
    deterministic_actions: list[dict[str, Any]] = field(default_factory=list)
    llm_call_count: int = 0
    before_hash: str = ""
    after_hash: str = ""
    failure_category: str = ""
    route_owner: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.llm_call_count < 0 or self.llm_call_count > 1:
            raise ValueError("v30 M5-A permits zero or one LLM call")
        if self.status not in {"VALID", "REPAIRED", "FAILED", "NO_EFFECT", "NOT_APPLICABLE"}:
            raise ValueError(f"unknown v30 M5-A status: {self.status}")
        if self.status == "FAILED" and not _text(self.failure_category):
            raise ValueError("failed v30 M5-A result requires failure_category")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SupplementalPassEvidenceV30:
    candidate_id: str
    candidate_hash: str
    test_nodeid: str
    source_file: str
    source_checkout: str
    outer_iteration: int
    covered_sut_lines: list[dict[str, Any]]
    source_mapping_status: str
    raw_artifact_ref: str = ""
    raw_artifact_sha256: str = ""
    summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_checkout != "pre_patch":
            raise ValueError("M1-M7 supplemental evidence must be pre_patch")
        if self.source_mapping_status not in {"VALID", "VALID_PRE_PATCH_SOURCE"}:
            raise ValueError("supplemental evidence has invalid source mapping")
        if self.outer_iteration < 1:
            raise ValueError("outer_iteration must be positive")
        if not _text(self.candidate_id) or not _text(self.test_nodeid):
            raise ValueError("supplemental evidence requires candidate identity")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeedbackRouteV30:
    diagnosis: str
    earliest_causal_owner: str
    requested_modules: list[str]
    expected_artifact: str
    previous_fingerprint: str
    new_fingerprint: str
    material_change: bool
    no_effect: bool = False
    escalation: str = ""

    def __post_init__(self) -> None:
        allowed = {"M1", "M2", "M3", "M4", "M5", "M5-A", "M6", "M7"}
        if self.earliest_causal_owner not in allowed:
            raise ValueError("invalid v30 feedback causal owner")
        if self.no_effect and self.material_change:
            raise ValueError("no_effect route cannot report material_change")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _contains_forbidden_runtime_field(value: Any) -> bool:
    forbidden = {
        "golden_patch", "golden_patch_lines", "patched_repo", "post_patch",
        "post_patch_outcome", "m8_results", "fail_to_pass", "patch_hit_rate",
    }
    if isinstance(value, str):
        return value.lower() in forbidden
    return str(value).lower() in forbidden


def normalize_localization_hypotheses(
    raw: Any,
    *,
    max_hypotheses: int = V30_MAX_LOCALIZATION_HYPOTHESES,
) -> list[dict[str, Any]]:
    """Normalize and deterministically bound M2 hypotheses."""
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        source_file = _text(item.get("source_file") or item.get("file_path"))
        if not source_file:
            continue
        hypothesis = dict(item)
        hypothesis.setdefault("hypothesis_id", f"h{index + 1}")
        normalized_source = posixpath.normpath(source_file.replace("\\", "/"))
        while normalized_source.startswith("./"):
            normalized_source = normalized_source[2:]
        if normalized_source == ".." or normalized_source.startswith("../"):
            continue
        hypothesis["source_file"] = normalized_source
        hypothesis["confidence"] = max(0.0, min(1.0, float(item.get("confidence", item.get("score", 0.0)) or 0.0)))
        hypothesis["primary"] = index == 0
        hypothesis.setdefault("evidence", [])
        hypothesis.setdefault("alternatives", [])
        hypothesis.setdefault("provenance", {"source": "m2_pre_patch_ranked_evidence"})
        normalized.append(hypothesis)
    normalized.sort(key=lambda item: (
        -float(item.get("confidence", 0.0)),
        str(item.get("source_file")),
        str(item.get("qualified_name") or item.get("function_name") or ""),
        tuple(item.get("line_range") or ()),
    ))
    # A bounded hypothesis set is useful only if duplicate source identities
    # do not consume every slot.  Prefer one hypothesis per canonical source,
    # then fill any remaining capacity with lower-ranked functions from those
    # sources.  This preserves ranking while retaining directly issue-named
    # alternative files.
    diverse: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    seen_identities: set[tuple[Any, ...]] = set()
    for item in normalized:
        source_identity = str(item.get("source_file") or "").casefold()
        identity = (
            source_identity,
            str(item.get("qualified_name") or item.get("function_name") or "").casefold(),
            tuple(item.get("line_range") or ()),
        )
        if identity in seen_identities:
            continue
        seen_identities.add(identity)
        if source_identity in seen_sources:
            deferred.append(item)
            continue
        seen_sources.add(source_identity)
        diverse.append(item)
    bounded = (diverse + deferred)[:max_hypotheses]
    for index, item in enumerate(bounded):
        item["hypothesis_id"] = f"h{index + 1}"
        item["primary"] = index == 0
    return bounded


def validate_v30_artifact_reference(value: Mapping[str, Any]) -> None:
    """Validate the compact canonical-evidence reference contract."""
    for key in ("path", "sha256", "schema_version"):
        if not _text(value.get(key)):
            raise ValueError(f"artifact reference requires {key}")
    digest = _text(value.get("sha256"))
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
        raise ValueError("artifact reference sha256 must be a hexadecimal digest")

"""Typed, pre-patch-only contracts for the v31 correction revision.

The v31 contracts make the M5 import/oracle/target handoff explicit without
changing the v30 wire format.  All provenance is restricted to issue text,
pre-patch source, or pre-patch execution evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


V31_SCHEMA_VERSION = "v31-generation-contract-v1"
V31_MAX_IMPORTS = 64
V31_MAX_TARGET_HYPOTHESES = 3


def _text(value: Any) -> str:
    return str(value or "").strip()


def _forbidden(value: Any) -> bool:
    fields = {
        "golden_patch", "golden_patch_lines", "patched_repo", "post_patch",
        "post_patch_outcome", "m8_results", "fail_to_pass", "patch_hit_rate",
    }
    if isinstance(value, Mapping):
        return any(_forbidden(key) or _forbidden(item) for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(_forbidden(item) for item in value)
    return str(value).lower() in fields


@dataclass(frozen=True)
class ImportManifestEntryV31:
    import_line: str
    module: str
    symbol: str = ""
    provenance: str = "pre_patch_source"
    verified_module: bool = True
    verified_symbol: bool = False

    def __post_init__(self) -> None:
        if not _text(self.import_line) or not _text(self.module):
            raise ValueError("v31 import manifest entries require import_line and module")
        if self.provenance not in {"pre_patch_source", "pre_patch_test", "stdlib", "framework"}:
            raise ValueError("v31 import manifest provenance must be pre-patch")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OracleContractV31:
    oracle_type: str
    property: str
    evidence: list[str] = field(default_factory=list)
    allowed_forms: list[str] = field(default_factory=list)
    forbidden_forms: list[str] = field(default_factory=list)
    issue_identifiers: list[str] = field(default_factory=list)
    target_relation: str = ""

    def __post_init__(self) -> None:
        if not _text(self.oracle_type) or not _text(self.property):
            raise ValueError("v31 oracle contract requires oracle_type and property")
        if _forbidden(self.evidence) or _forbidden(self.target_relation):
            raise ValueError("v31 oracle contract contains forbidden runtime provenance")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetHypothesisV31:
    hypothesis_id: str
    source_file: str
    symbol: str = ""
    line_range: list[int] = field(default_factory=list)
    confidence: float = 0.0
    provenance: str = "m2_pre_patch_ranked_evidence"
    evidence: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not _text(self.hypothesis_id) or not _text(self.source_file):
            raise ValueError("v31 target hypotheses require identity and source file")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("v31 target confidence must be in [0, 1]")
        if self.line_range and (len(self.line_range) != 2 or any(int(v) < 1 for v in self.line_range)):
            raise ValueError("v31 line_range must be a positive [start, end] pair")
        if _forbidden(self.provenance) or _forbidden(self.evidence):
            raise ValueError("v31 target provenance contains forbidden runtime field")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TestGenerationContractV31:
    schema_version: str
    framework: str
    runner: str
    shape: str
    target_test_file: str
    target_source_file: str
    target_symbol: str
    allowed_imports: list[ImportManifestEntryV31]
    nearby_patterns: list[str]
    fixture_candidates: list[str]
    target_invocation_candidates: list[str]
    observed_behavior: str
    expected_behavior: str
    oracle: OracleContractV31
    forbidden_patterns: list[str] = field(default_factory=list)
    skeleton_source: str = ""
    target_hypotheses: list[TargetHypothesisV31] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.schema_version != V31_SCHEMA_VERSION:
            raise ValueError("unexpected v31 generation contract schema")
        if not _text(self.framework) or not _text(self.runner) or not _text(self.shape):
            raise ValueError("v31 generation contract requires framework, runner, and shape")
        if not _text(self.target_source_file) or not _text(self.target_test_file):
            raise ValueError("v31 generation contract requires target files")
        if len(self.allowed_imports) > V31_MAX_IMPORTS:
            raise ValueError("v31 import manifest exceeds bound")
        if _forbidden(self.__dict__):
            raise ValueError("v31 generation contract contains forbidden runtime field")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OracleTraceV31:
    assertion: str
    behavior_contract_field: str
    issue_evidence: list[str] = field(default_factory=list)
    target_relation: str = ""

    def __post_init__(self) -> None:
        if not _text(self.assertion) or not _text(self.behavior_contract_field):
            raise ValueError("v31 oracle trace requires assertion and behavior field")
        if _forbidden(self.__dict__):
            raise ValueError("v31 oracle trace contains forbidden runtime field")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class M5TelemetryV31:
    generation_attempt_id: str
    candidate_hash: str
    framework_valid: bool
    imports_valid: bool
    target_invocation_valid: bool
    oracle_valid: bool
    syntax_valid: bool
    repair_actions: list[str] = field(default_factory=list)
    rejection_reason: str = ""
    rejected_semantic_fingerprint: str = ""
    contract_schema_version: str = V31_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_target_hypotheses(raw: Any) -> list[TargetHypothesisV31]:
    """Build a deterministic, bounded target identity list from M2 evidence."""
    if not isinstance(raw, list):
        return []
    values: list[TargetHypothesisV31] = []
    for index, item in enumerate(raw[:V31_MAX_TARGET_HYPOTHESES]):
        if not isinstance(item, Mapping):
            continue
        source_file = _text(item.get("source_file") or item.get("file_path"))
        if not source_file:
            continue
        raw_lines = item.get("line_range") or []
        lines = [int(v) for v in raw_lines[:2]] if isinstance(raw_lines, list) else []
        if len(lines) != 2:
            lines = []
        values.append(TargetHypothesisV31(
            hypothesis_id=_text(item.get("hypothesis_id")) or f"h{index + 1}",
            source_file=source_file.replace("\\", "/"),
            symbol=_text(item.get("function_name") or item.get("symbol")),
            line_range=lines,
            confidence=max(0.0, min(1.0, float(item.get("confidence", item.get("score", 0.0)) or 0.0))),
            evidence=[str(v) for v in item.get("evidence", []) if _text(v)],
        ))
    values.sort(key=lambda value: (-value.confidence, value.source_file, value.symbol))
    return values[:V31_MAX_TARGET_HYPOTHESES]

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from src.contracts.feature_flags import V22FeatureFlags, resolve_feature_flags
from src.executor.alignment_runner import (
    compute_ochiai_sbfl,
    normalize_pre_patch_execution_status,
)


BLOCKING_UNSPECIFIED = "BLOCKING_UNSPECIFIED"
DISABLED = "DISABLED"


@dataclass(frozen=True)
class FlitsrResult:
    status: str
    final_order: list[dict[str, Any]] = field(default_factory=list)
    reduction_rounds: list[dict[str, Any]] = field(default_factory=list)
    selected_basis_elements: list[dict[str, Any]] = field(default_factory=list)
    remaining_tests: dict[str, list[str]] = field(default_factory=dict)
    canonical_spectra: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_flitsr_reduction(
    spectra: Iterable[Mapping[str, Any]],
    *,
    feature_flags: V22FeatureFlags | Mapping[str, Any] | None = None,
) -> FlitsrResult:
    """Prepare deterministic FLITSR inputs without inventing undefined steps.

    The approved repository specification requires FLITSR artifacts but does
    not define the reduction selection formula, removal rule, or termination
    rule. Therefore the enabled path returns ``BLOCKING_UNSPECIFIED`` after
    canonical unique F/P spectra are built. Stability reruns are collapsed by
    canonical test identity before any artifact is produced.
    """
    flags = (
        feature_flags
        if isinstance(feature_flags, V22FeatureFlags)
        else resolve_feature_flags(feature_flags)
    )
    canonical = canonical_unique_spectra(spectra)
    remaining = _remaining_tests(canonical)
    metadata = {
        "feature_flag": "m6_flitsr",
        "input_policy": "canonical_unique_pre_patch_spectra",
        "stability_reruns_are_observations": False,
        "deterministic_ordering": [
            "canonical_test_id_asc",
            "source_file_asc",
            "line_no_asc",
        ],
    }
    if not flags.m6_flitsr:
        return FlitsrResult(
            status=DISABLED,
            remaining_tests=remaining,
            canonical_spectra=canonical,
            diagnostics=["m6_flitsr disabled by feature flag"],
            metadata={**metadata, "enabled": False},
        )
    return FlitsrResult(
        status=BLOCKING_UNSPECIFIED,
        reduction_rounds=[],
        selected_basis_elements=[],
        remaining_tests=remaining,
        final_order=[],
        canonical_spectra=canonical,
        diagnostics=[
            "BLOCKING: FLITSR reduction formula is not defined by the approved specification",
            "BLOCKING: FLITSR test-removal rule is not defined by the approved specification",
            "BLOCKING: FLITSR termination rule is not defined by the approved specification",
        ],
        metadata={
            **metadata,
            "enabled": True,
            "base_sbfl": compute_ochiai_sbfl(canonical).to_dict(),
        },
    )


def canonical_unique_spectra(
    spectra: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(spectra):
        record = dict(item)
        if record.get("flaky") or record.get("is_stable") is False:
            continue
        canonical_id = str(
            record.get("canonical_test_id")
            or record.get("test_id")
            or record.get("test_nodeid")
            or f"pre_patch_execution_{index}"
        )
        if canonical_id in records:
            continue
        covered_lines = record.get("covered_lines")
        if covered_lines is None:
            covered_lines = record.get("covered_sut_lines")
        normalized_lines = _normalized_covered_lines(
            covered_lines if isinstance(covered_lines, list) else []
        )
        normalized = dict(record)
        normalized["canonical_test_id"] = canonical_id
        normalized["test_id"] = canonical_id
        normalized["covered_lines"] = normalized_lines
        records[canonical_id] = normalized
    return [records[key] for key in sorted(records)]


def _normalized_covered_lines(lines: Iterable[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for line in lines:
        if not isinstance(line, Mapping):
            continue
        source_file = str(line.get("source_file") or "")
        line_no = line.get("line_no")
        if not source_file or not isinstance(line_no, int):
            continue
        key = (source_file, line_no)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {"source_file": source_file, "line_no": line_no, "element_type": "line"}
        )
    normalized.sort(key=lambda item: (item["source_file"], item["line_no"]))
    return normalized


def _remaining_tests(records: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    failing: list[str] = []
    passing: list[str] = []
    error: list[str] = []
    for record in records:
        test_id = str(record.get("canonical_test_id") or record.get("test_id") or "")
        status = normalize_pre_patch_execution_status(record).value
        if status == "FAIL":
            failing.append(test_id)
        elif status == "PASS":
            passing.append(test_id)
        elif status == "ERROR":
            error.append(test_id)
    return {
        "failing_tests": failing,
        "passing_tests": passing,
        "error_tests": error,
    }

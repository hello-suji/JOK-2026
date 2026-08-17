from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.contracts.models import (
    AlignmentResult,
    ExecutionResult,
    GeneratedTest,
    IssueClue,
)
from src.contracts.status import (
    CandidateStatus,
    ExecutionStatus,
    ValidationStatus,
    legacy_failure_type_to_statuses,
)
from src.utils.file_io import read_json_object


def _read_mapping(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    data = read_json_object(value)
    if data is None:
        raise ValueError(f"not a readable JSON object: {value}")
    return data


def _canonical_text(value: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value), {
            "legacy_value": list(value),
            "legacy_value_type": "list",
        }
    if value is None:
        return "", {"legacy_value": None, "missing_from_legacy_artifact": True}
    return str(value), {"legacy_value": value, "legacy_value_type": type(value).__name__}


def read_legacy_clue(value: str | Path | Mapping[str, Any]) -> IssueClue:
    data = _read_mapping(value)
    observed, observed_meta = _canonical_text(data.get("observed_behavior"))
    expected, expected_meta = _canonical_text(data.get("expected_behavior"))
    steps = data.get("steps_to_reproduce", data.get("repro_conditions", []))
    if steps is None:
        steps = []
    if not isinstance(steps, list):
        steps = [str(steps)]
    metadata = {
        "adapter": "legacy_clue",
        "observed_behavior": observed_meta,
        "expected_behavior": expected_meta,
        "legacy_keys": sorted(data.keys()),
    }
    return IssueClue(
        instance_id=str(data.get("instance_id", "")),
        observed_behavior=observed,
        expected_behavior=expected,
        steps_to_reproduce=[str(item) for item in steps],
        identifiers=data.get("identifiers", {}) if isinstance(data.get("identifiers"), dict) else {},
        defect_location_hints=[
            dict(item)
            for item in (data.get("defect_location_hints") or [])
            if isinstance(item, Mapping)
        ],
        raw_issue_text=str(data.get("raw_issue_text", "")),
        metadata=metadata,
    )


def read_legacy_context(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    data = _read_mapping(value)
    return {"payload": data, "metadata": {"adapter": "legacy_context"}}


def read_legacy_scenario(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    data = _read_mapping(value)
    return {"payload": data, "metadata": {"adapter": "legacy_scenario"}}


def read_legacy_generated_test(value: str | Path | Mapping[str, Any]) -> GeneratedTest:
    data = _read_mapping(value)
    test_id = str(data.get("test_id") or data.get("scenario_id") or data.get("instance_id") or "")
    metadata = {
        "adapter": "legacy_generated_test",
        "legacy_patch_field": "test_patch" if "test_patch" in data else None,
        "legacy_hash_field": "patch_sha256" if "patch_sha256" in data else None,
        "legacy_values": {
            "test_patch": data.get("test_patch"),
            "patch_sha256": data.get("patch_sha256"),
        },
    }
    return GeneratedTest(
        instance_id=str(data.get("instance_id", "")),
        test_id=test_id,
        generated_patch_path=str(data.get("generated_patch_path", "")),
        generated_patch_sha256=str(
            data.get("generated_patch_sha256") or data.get("patch_sha256") or ""
        ),
        candidate_status=CandidateStatus.GENERATED,
        diagnostic_only=False,
        metadata=metadata,
    )


def read_legacy_alignment_execution(value: str | Path | Mapping[str, Any]) -> ExecutionResult:
    data = _read_mapping(value)
    has_error = bool(data.get("has_error"))
    has_failure = bool(data.get("has_failure"))
    if has_error:
        status = ExecutionStatus.ERROR
    elif has_failure:
        status = ExecutionStatus.FAIL
    elif data.get("test_results"):
        status = ExecutionStatus.PASS
    else:
        status = ExecutionStatus.NOT_RUN
    return ExecutionResult(
        instance_id=str(data.get("instance_id", "")),
        run_id=str(data.get("run_id", "")),
        execution_status=status,
        test_results=data.get("test_results", {}) if isinstance(data.get("test_results"), dict) else {},
        raw_output=str(data.get("raw_output", "")),
        metadata={"adapter": "legacy_alignment_execution"},
    )


def read_legacy_alignment_result(value: str | Path | Mapping[str, Any]) -> AlignmentResult:
    data = _read_mapping(value)
    converted = legacy_failure_type_to_statuses(data.get("failure_type"))
    return AlignmentResult(
        instance_id=str(data.get("instance_id", "")),
        execution_status=converted["execution_status"] or ExecutionStatus.NOT_RUN.value,
        validation_status=converted["validation_status"] or ValidationStatus.NOT_RUN.value,
        m7_alignment_status=converted["m7_alignment_status"],
        m7_decision_status=data.get("m7_decision_status") or converted["m7_decision_status"],
        diagnostic_only=(data.get("m7_decision_status") or converted["m7_decision_status"]) != "ALIGNED",
        legacy_failure_type=converted["legacy_failure_type"],
        metadata={
            "adapter": "legacy_alignment_result",
            "legacy_failure_type": data.get("failure_type"),
            "compatibility_m7_alignment_status": converted["m7_alignment_status"],
            "legacy_payload": data,
        },
    )


def read_legacy_final_evaluation(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    data = _read_mapping(value)
    return {"payload": data, "metadata": {"adapter": "legacy_final_evaluation"}}


def read_legacy_batch_summary(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    data = _read_mapping(value)
    return {"payload": data, "metadata": {"adapter": "legacy_batch_summary"}}

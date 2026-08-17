from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class ExecutionStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    NOT_RUN = "NOT_RUN"


class ValidationStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    NOT_RUN = "NOT_RUN"


class M7AlignmentStatus(str, Enum):
    ALIGNED = "ALIGNED"
    NOT_FAILED = "NOT_FAILED"
    NO_COVERAGE = "NO_COVERAGE"
    WEAK_ALIGNMENT = "WEAK_ALIGNMENT"


class M7DecisionStatus(str, Enum):
    NOT_VALID = "NOT_VALID"
    ERROR = "ERROR"
    NOT_FAILED = "NOT_FAILED"
    NO_COVERAGE = "NO_COVERAGE"
    WEAK_ALIGNMENT = "WEAK_ALIGNMENT"
    ALIGNED = "ALIGNED"


class CandidateStatus(str, Enum):
    GENERATED = "GENERATED"
    POSTPROCESSED = "POSTPROCESSED"
    INVALID = "INVALID"


_M7_VALUES = {item.value for item in M7AlignmentStatus}
_M7_DECISION_VALUES = {item.value for item in M7DecisionStatus}
_CANDIDATE_VALUES = {item.value for item in CandidateStatus}


def _status_text(value: Any) -> str:
    if value is None:
        return ""
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return str(enum_value).upper()
    return str(value).upper()


def coerce_execution_status(value: Any) -> ExecutionStatus:
    if isinstance(value, ExecutionStatus):
        return value
    text = _status_text(value)
    if text in {"PASSED", "PASS"}:
        return ExecutionStatus.PASS
    if text in {"FAILED", "FAIL"}:
        return ExecutionStatus.FAIL
    if text == "ERROR":
        return ExecutionStatus.ERROR
    if text in {"NOT_RUN", "SKIP", "SKIPPED", ""}:
        return ExecutionStatus.NOT_RUN
    raise ValueError(f"unknown execution_status: {value!r}")


def coerce_validation_status(value: Any) -> ValidationStatus:
    if isinstance(value, ValidationStatus):
        return value
    text = _status_text(value)
    if text == "VALID":
        return ValidationStatus.VALID
    if text in {"INVALID", "NOT_VALID", "NOT_COLLECTED"}:
        return ValidationStatus.INVALID
    if text in {"NOT_RUN", ""}:
        return ValidationStatus.NOT_RUN
    raise ValueError(f"unknown validation_status: {value!r}")


def coerce_m7_alignment_status(value: Any) -> Optional[M7AlignmentStatus]:
    if value is None or value == "":
        return None
    if isinstance(value, M7AlignmentStatus):
        return value
    text = _status_text(value)
    if text in _M7_VALUES:
        return M7AlignmentStatus(text)
    if text in {"ERROR", "NOT_VALID", "NOT_COLLECTED"}:
        return None
    raise ValueError(f"unknown m7_alignment_status: {value!r}")


def coerce_m7_decision_status(value: Any) -> Optional[M7DecisionStatus]:
    if value is None or value == "":
        return None
    if isinstance(value, M7DecisionStatus):
        return value
    text = _status_text(value)
    if text == "NO_FAIL":
        text = M7DecisionStatus.NOT_FAILED.value
    if text == "NOT_COLLECTED":
        text = M7DecisionStatus.NOT_VALID.value
    if text in _M7_DECISION_VALUES:
        return M7DecisionStatus(text)
    raise ValueError(f"unknown m7_decision_status: {value!r}")


def m7_decision_to_alignment_status(value: Any) -> Optional[M7AlignmentStatus]:
    decision = coerce_m7_decision_status(value)
    if decision is None:
        return None
    if decision.value in _M7_VALUES:
        return M7AlignmentStatus(decision.value)
    return None


def coerce_candidate_status(value: Any) -> CandidateStatus:
    if value is None or value == "":
        return CandidateStatus.GENERATED
    if isinstance(value, CandidateStatus):
        return value
    text = _status_text(value)
    if text in _CANDIDATE_VALUES:
        return CandidateStatus(text)
    raise ValueError(f"unknown candidate_status: {value!r}")


def legacy_failure_type_to_statuses(value: Any) -> dict[str, str | None]:
    """Convert legacy failure_type without fabricating M7 outcomes."""
    text = _status_text(value)
    result: dict[str, str | None] = {
        "execution_status": ExecutionStatus.NOT_RUN.value,
        "validation_status": ValidationStatus.NOT_RUN.value,
        "m7_decision_status": None,
        "m7_alignment_status": None,
        "legacy_failure_type": text or None,
    }
    if text == "ERROR":
        result["execution_status"] = ExecutionStatus.ERROR.value
        result["validation_status"] = ValidationStatus.VALID.value
        result["m7_decision_status"] = M7DecisionStatus.ERROR.value
        return result
    if text in {"NOT_VALID", "NOT_COLLECTED"}:
        result["validation_status"] = ValidationStatus.INVALID.value
        result["m7_decision_status"] = M7DecisionStatus.NOT_VALID.value
        return result
    if text in {"NO_FAIL", "NOT_FAILED"}:
        result["validation_status"] = ValidationStatus.VALID.value
        result["m7_decision_status"] = M7DecisionStatus.NOT_FAILED.value
        result["m7_alignment_status"] = M7AlignmentStatus.NOT_FAILED.value
        return result
    if text in _M7_VALUES:
        result["validation_status"] = ValidationStatus.VALID.value
        result["m7_decision_status"] = text
        result["m7_alignment_status"] = text
        return result
    if not text:
        return result
    return result


def is_canonical_m7_alignment(value: Any) -> bool:
    try:
        return coerce_m7_alignment_status(value) is not None
    except ValueError:
        return False


def is_canonical_m7_decision(value: Any) -> bool:
    try:
        return coerce_m7_decision_status(value) is not None
    except ValueError:
        return False

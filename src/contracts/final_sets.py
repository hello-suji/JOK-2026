from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from src.contracts.models import FinalSetMembership
from src.contracts.status import (
    CandidateStatus,
    ExecutionStatus,
    M7AlignmentStatus,
    coerce_candidate_status,
    coerce_execution_status,
    coerce_m7_alignment_status,
    m7_decision_to_alignment_status,
)


def is_valid_candidate(candidate: Mapping[str, Any]) -> bool:
    try:
        status = coerce_candidate_status(candidate.get("candidate_status"))
    except ValueError:
        return False
    return status != CandidateStatus.INVALID


def is_diagnostic_only(candidate: Mapping[str, Any]) -> bool:
    value = candidate.get("diagnostic_only", False)
    if isinstance(value, bool):
        return value
    return True


def admitted_to_final_set(candidate: Mapping[str, Any]) -> bool:
    """Return the v29 authoritative membership predicate.

    ``T_final`` membership is defined exactly by canonical M7 ALIGNED status.
    Candidate/diagnostic fields are derived metadata and must not silently
    override the M7 verdict; contradictory records are rejected by boundary
    validators rather than changing this predicate.
    """
    try:
        if "m7_decision_status" in candidate:
            status = m7_decision_to_alignment_status(candidate.get("m7_decision_status"))
        else:
            status = coerce_m7_alignment_status(candidate.get("m7_alignment_status"))
    except ValueError:
        return False
    return status == M7AlignmentStatus.ALIGNED


def build_t_final(candidates: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [candidate for candidate in candidates if admitted_to_final_set(candidate)]


def is_f_to_p(candidate: Mapping[str, Any]) -> bool:
    if not admitted_to_final_set(candidate):
        return False
    try:
        before = coerce_execution_status(candidate.get("pre_patch_outcome"))
        after = coerce_execution_status(candidate.get("post_patch_outcome"))
    except ValueError:
        return False
    return before == ExecutionStatus.FAIL and after == ExecutionStatus.PASS


def build_t_f2p(candidates: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [candidate for candidate in candidates if is_f_to_p(candidate)]


def validate_final_set_membership(membership: FinalSetMembership | Mapping[str, Any]) -> None:
    value = (
        membership
        if isinstance(membership, FinalSetMembership)
        else FinalSetMembership(**dict(membership))
    )
    if value.in_t_f2p and not value.in_t_final:
        raise ValueError("in_t_f2p implies in_t_final")


def rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator

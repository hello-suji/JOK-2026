from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from src.contracts.status import (
    CandidateStatus,
    ExecutionStatus,
    M7AlignmentStatus,
    M7DecisionStatus,
    ValidationStatus,
    coerce_m7_decision_status,
    m7_decision_to_alignment_status,
)


@dataclass
class FinalSetMembership:
    in_t_final: bool = False
    in_t_f2p: bool = False

    def __post_init__(self) -> None:
        if self.in_t_f2p and not self.in_t_final:
            raise ValueError("in_t_f2p implies in_t_final")

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass
class IssueClue:
    instance_id: str
    observed_behavior: str
    expected_behavior: str
    steps_to_reproduce: list[str]
    identifiers: dict[str, list[str]] = field(default_factory=dict)
    defect_location_hints: list[dict[str, Any]] = field(default_factory=list)
    raw_issue_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FaultCandidate:
    instance_id: str
    file_path: str
    function_name: str = ""
    line_no: Optional[int] = None
    score: Optional[float] = None
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Scenario:
    instance_id: str
    scenario_id: str
    description: str = ""
    steps: list[str] = field(default_factory=list)
    oracle: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratedTest:
    instance_id: str
    test_id: str
    generated_patch_path: str = ""
    generated_patch_sha256: str = ""
    candidate_status: CandidateStatus = CandidateStatus.GENERATED
    diagnostic_only: bool = False
    final_set_membership: FinalSetMembership = field(default_factory=FinalSetMembership)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_status, CandidateStatus):
            self.candidate_status = CandidateStatus(str(self.candidate_status))
        if isinstance(self.final_set_membership, dict):
            self.final_set_membership = FinalSetMembership(**self.final_set_membership)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidate_status"] = self.candidate_status.value
        return data


@dataclass
class ExecutionResult:
    instance_id: str
    run_id: str
    execution_status: ExecutionStatus
    test_results: dict[str, str] = field(default_factory=dict)
    raw_output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.execution_status, ExecutionStatus):
            self.execution_status = ExecutionStatus(str(self.execution_status))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["execution_status"] = self.execution_status.value
        return data


@dataclass
class CoverageResult:
    instance_id: str
    coverage_data: dict[str, Any] = field(default_factory=dict)
    checked_coverage: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SBFLResult:
    instance_id: str
    suspiciousness: list[dict[str, Any]] = field(default_factory=list)
    formula: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlignmentResult:
    instance_id: str
    execution_status: ExecutionStatus = ExecutionStatus.NOT_RUN
    validation_status: ValidationStatus = ValidationStatus.NOT_RUN
    m7_alignment_status: Optional[M7AlignmentStatus] = None
    m7_decision_status: Optional[M7DecisionStatus] = None
    diagnostic_only: bool = False
    legacy_failure_type: Optional[str] = None
    alignment_verdict: str = "NOT_ALIGNED"
    admission_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.execution_status, ExecutionStatus):
            self.execution_status = ExecutionStatus(str(self.execution_status))
        if not isinstance(self.validation_status, ValidationStatus):
            self.validation_status = ValidationStatus(str(self.validation_status))
        if self.m7_alignment_status is not None and not isinstance(
            self.m7_alignment_status, M7AlignmentStatus
        ):
            self.m7_alignment_status = M7AlignmentStatus(str(self.m7_alignment_status))
        if self.m7_decision_status is not None and not isinstance(
            self.m7_decision_status, M7DecisionStatus
        ):
            coerced = coerce_m7_decision_status(self.m7_decision_status)
            if coerced is None:
                raise ValueError("m7_decision_status is required when provided")
            self.m7_decision_status = coerced
        if self.m7_decision_status is not None and self.m7_alignment_status is None:
            self.m7_alignment_status = m7_decision_to_alignment_status(self.m7_decision_status)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["execution_status"] = self.execution_status.value
        data["validation_status"] = self.validation_status.value
        data["m7_alignment_status"] = (
            self.m7_alignment_status.value if self.m7_alignment_status else None
        )
        data["m7_decision_status"] = (
            self.m7_decision_status.value if self.m7_decision_status else None
        )
        return data


M7_DECISION_RECORD_SCHEMA_VERSION = "m7-decision-record-v1"
M7_FEEDBACK_DECISION_SCHEMA_VERSION = "m7-feedback-decision-v1"


@dataclass
class M7FeedbackDecision:
    m7_decision_status: M7DecisionStatus
    outer_iteration: int
    source_stage: str
    failure_category: str
    feedback_summary: str
    selected_feedback_branch: str
    next_start_stage: str
    failure_reason: str = ""
    assumption_gap: str = ""
    next_scenario_change: str = ""
    admissible_alternatives: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    conservative_gate_assessment: Optional[str] = None
    recommended_change: str = ""
    change_owner_module: str = ""
    previous_feedback_effect: str = "NOT_APPLICABLE"
    confidence: Optional[float] = None
    admission_path: Optional[str] = None
    route_destination: str = ""
    modules_requested_for_next: list[str] = field(default_factory=list)
    cause_code: str = ""
    cause_source: str = "RULE"
    cause_hypothesis: str = ""
    cause_confidence: Optional[float] = None
    cause_evidence: dict[str, Any] = field(default_factory=dict)
    alternative_causes_considered: list[str] = field(default_factory=list)
    allowed_restart_stages: list[str] = field(default_factory=list)
    selected_restart_stage: str = ""
    concrete_repair_instruction: str = ""
    policy_validation: dict[str, Any] = field(default_factory=dict)
    fields_to_preserve: list[str] = field(default_factory=list)
    fields_to_change: list[str] = field(default_factory=list)
    prohibited_prior_fingerprints: list[str] = field(default_factory=list)
    target_reuse_allowed: bool = False
    target_reuse_justification: str = ""
    evidence_used: dict[str, Any] = field(default_factory=dict)
    remaining_outer_iterations: int = 0
    deterministic_fallback_used: bool = False
    feedback_provenance: str = "DETERMINISTIC_FALLBACK"
    llm_attempted: bool = True
    llm_succeeded: bool = False
    parse_succeeded: bool = False
    fallback_reason: Optional[str] = None
    model_request_elapsed_sec: Optional[float] = None
    raw_response: str = ""
    llm_route: Optional[str] = None
    route_consistency_status: str = "NOT_CHECKED"
    final_route: Optional[str] = None
    schema_version: str = M7_FEEDBACK_DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.m7_decision_status, M7DecisionStatus):
            decision = coerce_m7_decision_status(self.m7_decision_status)
            if decision is None:
                raise ValueError("m7_decision_status is required")
            self.m7_decision_status = decision
        if self.next_start_stage not in {"M2", "M3", "M4", "M5", "M6", "M8"}:
            raise ValueError(f"next_start_stage must be an explicit module boundary: {self.next_start_stage!r}")
        provenance = str(self.feedback_provenance or "").upper()
        if provenance not in {
            "RULE_CONCLUSIVE",
            "LLM",
            "LLM_DIAGNOSIS",
            "DETERMINISTIC_FALLBACK",
            "NOT_APPLICABLE",
        }:
            raise ValueError(f"unknown feedback_provenance: {self.feedback_provenance!r}")
        self.feedback_provenance = provenance
        if self.cause_source not in {
            "RULE",
            "LLM",
            "LLM_DIAGNOSIS",
            "DETERMINISTIC_FALLBACK",
        }:
            raise ValueError(f"unknown cause_source: {self.cause_source!r}")
        if not self.selected_restart_stage:
            self.selected_restart_stage = self.next_start_stage
        if self.selected_restart_stage not in {"M2", "M3", "M4", "M5", "M6", "M8"}:
            raise ValueError(f"selected_restart_stage must be an explicit module boundary: {self.selected_restart_stage!r}")
        if not self.allowed_restart_stages:
            self.allowed_restart_stages = [self.selected_restart_stage]
        invalid_allowed = sorted(set(self.allowed_restart_stages) - {"M2", "M3", "M4", "M5", "M6", "M8"})
        if invalid_allowed:
            raise ValueError(f"allowed_restart_stages contains invalid module boundary: {invalid_allowed[0]!r}")
        if self.selected_restart_stage not in self.allowed_restart_stages:
            raise ValueError("selected_restart_stage must be in allowed_restart_stages")
        if self.route_destination and self.route_destination not in {"→M2", "→M3", "→M5", "→M8"}:
            raise ValueError(f"unknown M7 route_destination: {self.route_destination!r}")
        if self.admission_path not in {None, "DIRECT", "CONSERVATIVE_OVERRIDE"}:
            raise ValueError(f"unknown admission_path: {self.admission_path!r}")
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        if self.feedback_provenance in {"LLM", "LLM_DIAGNOSIS"}:
            self.deterministic_fallback_used = False
            self.llm_attempted = True
            self.llm_succeeded = True
            self.parse_succeeded = True
            self.fallback_reason = None
            self.cause_source = self.feedback_provenance
        elif self.feedback_provenance == "RULE_CONCLUSIVE":
            self.deterministic_fallback_used = False
            self.llm_attempted = False
            self.llm_succeeded = False
            self.parse_succeeded = False
            self.fallback_reason = None
            self.cause_source = "RULE"
        elif self.feedback_provenance == "DETERMINISTIC_FALLBACK":
            self.deterministic_fallback_used = True
            self.llm_attempted = bool(self.llm_attempted)
            self.llm_succeeded = bool(self.llm_succeeded)
            self.parse_succeeded = bool(self.parse_succeeded)
            if not self.fallback_reason:
                raise ValueError("fallback_reason is required for deterministic fallback")
            self.cause_source = "DETERMINISTIC_FALLBACK"
        else:
            self.deterministic_fallback_used = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["m7_decision_status"] = self.m7_decision_status.value
        return data


@dataclass
class M7DecisionRecord:
    instance_id: str
    outer_iteration: int
    m7_decision_status: M7DecisionStatus
    source_stage: str
    validation_status: ValidationStatus
    execution_status: ExecutionStatus
    failure_category: str
    evaluation_evidence: dict[str, Any]
    feedback_decision: M7FeedbackDecision | dict[str, Any]
    previous_iteration_reference: dict[str, Any] = field(default_factory=dict)
    remaining_outer_iterations: int = 0
    loop_terminated: bool = False
    termination_reason: str = ""
    created_at: str = ""
    schema_version: str = M7_DECISION_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.m7_decision_status, M7DecisionStatus):
            decision = coerce_m7_decision_status(self.m7_decision_status)
            if decision is None:
                raise ValueError("m7_decision_status is required")
            self.m7_decision_status = decision
        if not isinstance(self.validation_status, ValidationStatus):
            self.validation_status = ValidationStatus(str(self.validation_status))
        if not isinstance(self.execution_status, ExecutionStatus):
            self.execution_status = ExecutionStatus(str(self.execution_status))
        if isinstance(self.feedback_decision, dict):
            self.feedback_decision = M7FeedbackDecision(**self.feedback_decision)

    @property
    def m7_alignment_status(self) -> Optional[M7AlignmentStatus]:
        return m7_decision_to_alignment_status(self.m7_decision_status)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["m7_decision_status"] = self.m7_decision_status.value
        data["m7_alignment_status"] = (
            self.m7_alignment_status.value if self.m7_alignment_status else None
        )
        data["validation_status"] = self.validation_status.value
        data["execution_status"] = self.execution_status.value
        feedback = self.feedback_decision
        data["feedback_decision"] = (
            feedback.to_dict() if isinstance(feedback, M7FeedbackDecision) else dict(feedback)
        )
        feedback_data = data["feedback_decision"]
        for key in (
            "failure_reason",
            "assumption_gap",
            "next_scenario_change",
            "admissible_alternatives",
            "evidence_refs",
            "recommended_change",
            "change_owner_module",
            "previous_feedback_effect",
            "confidence",
            "route_destination",
            "modules_requested_for_next",
            "cause_code",
            "cause_source",
            "cause_hypothesis",
            "cause_confidence",
            "cause_evidence",
            "alternative_causes_considered",
            "allowed_restart_stages",
            "selected_restart_stage",
            "selected_feedback_branch",
            "fields_to_preserve",
            "fields_to_change",
            "concrete_repair_instruction",
            "target_reuse_allowed",
            "target_reuse_justification",
            "prohibited_prior_fingerprints",
            "feedback_provenance",
            "llm_attempted",
            "llm_succeeded",
            "parse_succeeded",
            "policy_validation",
            "fallback_reason",
        ):
            data[key] = feedback_data.get(key)
        return data


@dataclass
class EvaluationResult:
    instance_id: str
    final_score: Optional[float] = None
    fail_to_pass: bool = False
    patch_hit_rate: Optional[float] = None
    checked_coverage: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IterationState:
    instance_id: str
    iteration: int
    generated_tests: list[GeneratedTest] = field(default_factory=list)
    alignment_result: Optional[AlignmentResult] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol


ROUTE_M5 = "→M5"
ROUTE_M3 = "→M3"
ROUTE_M2 = "→M2"
ROUTE_M8 = "→M8"
ROUTE_DESTINATIONS = frozenset({ROUTE_M5, ROUTE_M3, ROUTE_M2, ROUTE_M8})
LLM_DIAGNOSIS = "LLM_DIAGNOSIS"
DETERMINISTIC_FALLBACK = "DETERMINISTIC_FALLBACK"
TELEMETRY_MARKER_SECONDS = 120.0

_ROUTE_PLANS: dict[str, tuple[str, ...]] = {
    ROUTE_M5: ("M5", "M5-A", "M6", "M7"),
    ROUTE_M3: ("M3", "M4", "M5", "M5-A", "M6", "M7"),
    ROUTE_M2: ("M2", "M3", "M4", "M5", "M5-A", "M6", "M7"),
    ROUTE_M8: ("M8",),
}
_FORBIDDEN_EVIDENCE_TERMS = (
    "golden patch",
    "golden_patch",
    "post-patch",
    "post_patch",
    "after patch",
    "after_patch",
    "fail-to-pass",
    "fail_to_pass",
    "f_to_p",
    "patch hit rate",
    "patch_hit_rate",
    "phr result",
    "m8 result",
    "m8_result",
)
_V29_ROUTE_OWNER = {
    ROUTE_M2: "M2",
    ROUTE_M3: "M3",
    ROUTE_M5: "M5",
    ROUTE_M8: "M8",
}
_V29_EVIDENCE_ROOTS = frozenset(
    {
        "m7_status",
        "m7_gate_summary",
        "score_breakdown",
        "m1_issue_evidence",
        "m2_semantic_evidence",
        "m3_scenario_evidence",
        "m5_candidate_evidence",
        "m6_execution_evidence",
        "coverage_evidence",
        "issue_api_executed",
        "assertion_executed",
        "actual_output_observed",
        "exception_observed",
        "suspected_file_covered",
        "suspected_function_covered",
        "suspected_lines_covered",
        "issue_branch_reached",
        "incorrect_behavior_observed",
        "oracle_checked_behavior",
        "fault_hypothesis_supported",
        "scenario_assumption_supported",
        "repeated_semantic_fingerprint",
        "previous_feedback",
        "previous_feedback_effect",
    }
)
_V29_UNSAFE_CHANGE_PATTERNS = (
    "change production code",
    "modify production code",
    "edit production code",
    "patch the source",
    "apply the patch",
    "weaken the assertion",
    "weaken assertions",
    "relax the assertion",
    "relax assertions",
    "remove the assertion",
    "remove assertions",
    "bypass the oracle",
    "disable the oracle",
    "lower the threshold",
    "ignore the conservative gate",
    "bypass the conservative gate",
)


class DiagnosisClient(Protocol):
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Return one JSON diagnosis response."""


@dataclass(frozen=True)
class M7Diagnosis:
    """Strict v26 non-aligned diagnosis and rerun decision."""

    failure_reason: str
    assumption_gap: str
    next_scenario_change: str
    admissible_alternatives: str
    route_destination: str
    provenance: str
    fallback_reason: str | None = None
    llm_attempted: bool = False
    llm_succeeded: bool = False
    parse_succeeded: bool = False
    model_request_elapsed_sec: float | None = None
    exceeded_120s_telemetry_marker: bool = False
    raw_response: str = ""
    llm_route: str | None = None
    route_consistency_status: str = "NOT_CHECKED_V26"
    final_route: str | None = None
    conservative_gate_assessment: str | None = None
    evidence_refs: tuple[str, ...] = ()
    recommended_change: str = ""
    change_owner_module: str = ""
    previous_feedback_effect: str = "NOT_APPLICABLE"
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.route_destination not in ROUTE_DESTINATIONS:
            raise ValueError(f"invalid v26 route_destination: {self.route_destination!r}")
        if self.provenance not in {LLM_DIAGNOSIS, DETERMINISTIC_FALLBACK}:
            raise ValueError(f"invalid v26 diagnosis provenance: {self.provenance!r}")
        for name in ("failure_reason", "assumption_gap", "next_scenario_change"):
            if not str(getattr(self, name) or "").strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not self.admissible_alternatives.strip():
            raise ValueError("admissible_alternatives must be a non-empty string")
        if self.provenance == DETERMINISTIC_FALLBACK and not self.fallback_reason:
            raise ValueError("fallback_reason is required for deterministic fallback")
        if self.provenance == LLM_DIAGNOSIS and self.fallback_reason is not None:
            raise ValueError("LLM diagnosis cannot carry a fallback_reason")
        if self.final_route is None:
            object.__setattr__(self, "final_route", self.route_destination)
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0,1]")

    @property
    def modules_requested_for_next(self) -> tuple[str, ...]:
        return route_execution_plan(self.route_destination)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.conservative_gate_assessment is None:
            payload.pop("conservative_gate_assessment", None)
        payload["modules_requested_for_next"] = list(self.modules_requested_for_next)
        payload["evidence_refs"] = list(self.evidence_refs)
        payload["schema_version"] = (
            "m7-v29-diagnosis-v1"
            if self.route_destination == ROUTE_M8 or self.conservative_gate_assessment is not None
            else "m7-v26-diagnosis-v1"
        )
        return payload


def route_execution_plan(route_destination: str) -> tuple[str, ...]:
    """Return the complete downstream execution plan for one v26 route."""
    try:
        return _ROUTE_PLANS[route_destination]
    except KeyError as exc:
        raise ValueError(f"invalid v26 route_destination: {route_destination!r}") from exc


def route_start_stage(route_destination: str) -> str:
    """Return the concrete entry module for a v26 route."""
    return route_execution_plan(route_destination)[0]


def parse_m7_diagnosis(
    raw_response: str,
    *,
    revision: str = "v26",
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly parse the legacy five-field or quality-gated v29 diagnosis."""
    try:
        payload = json.loads(_extract_json_object(raw_response))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid M7 diagnosis JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("M7 diagnosis must be a JSON object")
    if revision in {"v36", "v37"}:
        required_v36 = {
            "why_failed",
            "route_destination",
            "fix_suggestion",
            "assumption_gap",
            "cg_analysis",
        }
        if set(payload) != required_v36:
            missing = sorted(required_v36 - set(payload))
            extra = sorted(set(payload) - required_v36)
            raise ValueError(
                f"M7 diagnosis schema mismatch; missing={missing}, extra={extra}"
            )
        for key in ("why_failed", "fix_suggestion", "assumption_gap"):
            if not isinstance(payload[key], str) or not payload[key].strip():
                raise ValueError(f"{key} must be a non-empty string")
        route = payload["route_destination"]
        if route not in ROUTE_DESTINATIONS:
            raise ValueError("route_destination must be one of →M5, →M3, →M2, or →M8")
        cg_analysis = payload["cg_analysis"]
        if cg_analysis is not None and not isinstance(cg_analysis, str):
            raise ValueError("cg_analysis must be a string or null")
        if route == ROUTE_M8:
            analysis = str(cg_analysis or "").lower()
            required_context = (
                "V37_CONSERVATIVE_GATE"
                if revision == "v37"
                else "V36_CONSERVATIVE_GATE"
            )
            if str(evidence.get("m7_decision_context") or "") != required_context:
                raise ValueError(
                    f"unsafe {revision} →M8 route: conservative gate is not the sole branch"
                )
            if "(b)" not in analysis and "non-essential" not in analysis and "nonessential" not in analysis:
                raise ValueError("unsafe v36 →M8 route: cg_analysis is not non-essential")
        normalized_v36 = {
            "failure_reason": payload["why_failed"].strip(),
            "assumption_gap": payload["assumption_gap"].strip(),
            "next_scenario_change": payload["fix_suggestion"].strip(),
            "admissible_alternatives": payload["fix_suggestion"].strip(),
            "route_destination": route,
            "conservative_gate_assessment": (
                cg_analysis.strip() if isinstance(cg_analysis, str) else None
            ),
            "recommended_change": payload["fix_suggestion"].strip(),
            "change_owner_module": _V29_ROUTE_OWNER[route],
        }
        _reject_forbidden_m8_claims(normalized_v36)
        return normalized_v36

    required = {
        "failure_reason",
        "assumption_gap",
        "next_scenario_change",
        "admissible_alternatives",
        "route_destination",
    }
    if revision == "v29":
        required.update(
            {
                "evidence_refs",
                "conservative_gate_assessment",
                "recommended_change",
                "change_owner_module",
                "previous_feedback_effect",
                "confidence",
            }
        )
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        raise ValueError(f"M7 diagnosis schema mismatch; missing={missing}, extra={extra}")
    strings: dict[str, str] = {}
    for key in ("failure_reason", "assumption_gap", "next_scenario_change"):
        value = payload[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string")
        strings[key] = value.strip()
    alternatives = payload["admissible_alternatives"]
    if not isinstance(alternatives, str) or not alternatives.strip():
        raise ValueError("admissible_alternatives must be a non-empty string")
    route = payload["route_destination"]
    allowed_routes = ROUTE_DESTINATIONS if revision == "v29" else ROUTE_DESTINATIONS - {ROUTE_M8}
    if route not in allowed_routes:
        suffix = ", or →M8" if revision == "v29" else ""
        raise ValueError(f"route_destination must be one of →M5, →M3, →M2{suffix}")
    assessment = payload.get("conservative_gate_assessment")
    if revision == "v29":
        if assessment is not None and not isinstance(assessment, str):
            raise ValueError("conservative_gate_assessment must be a string or null")
        if route == ROUTE_M8:
            guard_reason = validate_v29_m8_route(evidence or {}, assessment)
            if guard_reason:
                raise ValueError(f"unsafe v29 →M8 route: {guard_reason}")
        evidence_refs = payload.get("evidence_refs")
        if not isinstance(evidence_refs, list) or not evidence_refs:
            raise ValueError("evidence_refs must be a non-empty JSON string array")
        if any(not isinstance(item, str) or not item.strip() for item in evidence_refs):
            raise ValueError("evidence_refs must contain only non-empty strings")
        for key in ("recommended_change", "change_owner_module", "previous_feedback_effect"):
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty string")
        confidence = payload.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be a number in [0,1]")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be a number in [0,1]")
    normalized = {
        **strings,
        "admissible_alternatives": alternatives.strip(),
        "route_destination": route,
    }
    if revision == "v29":
        normalized["conservative_gate_assessment"] = (
            assessment.strip() if isinstance(assessment, str) else None
        )
        normalized.update(
            {
                "evidence_refs": tuple(
                    _normalize_evidence_ref(item) for item in payload["evidence_refs"]
                ),
                "recommended_change": payload["recommended_change"].strip(),
                "change_owner_module": payload["change_owner_module"].strip().upper(),
                "previous_feedback_effect": payload["previous_feedback_effect"].strip().upper(),
                "confidence": float(payload["confidence"]),
            }
        )
        policy_error = validate_v29_feedback_policy(normalized, evidence or {})
        if policy_error:
            raise ValueError(f"unsafe v29 feedback policy: {policy_error}")
    _reject_forbidden_m8_claims(normalized)
    return normalized


def build_m7_diagnosis_prompt(
    evidence: Mapping[str, Any], *, revision: str = "v26"
) -> str:
    """Build the sole v26 M7 diagnosis prompt from pre-patch evidence."""
    _reject_forbidden_m8_claims(evidence)
    compact_evidence = _compact_evidence_for_prompt(evidence)
    if revision in {"v36", "v37"}:
        payload = {
            "schema_version": "m7-v36-diagnosis-prompt-v1",
            "role": "diagnosis analyst for a pre-patch bug-reproduction test pipeline",
            "task": "Diagnose the root cause and select exactly one repair destination.",
            "alignment_evidence": compact_evidence,
            "rules": [
                "Use only supplied M1-M7 pre-patch evidence.",
                "Choose →M5 for bug-reproduction or coverage failure.",
                "Choose →M3 for issue-test semantic misalignment.",
                "Choose →M2 only after prior M5/M3 work indicates context is wrong.",
                "Choose →M8 only when all three numeric gates pass, the Conservative Gate is the sole trigger, and cg_analysis classifies every flag as (b) non-essential.",
                "Return exactly the five required fields and no prose.",
            ],
            "required_output_schema": {
                "why_failed": "non-empty structural explanation",
                "route_destination": "→M2 | →M3 | →M5 | →M8",
                "fix_suggestion": "non-empty concrete instruction",
                "assumption_gap": "non-empty assumed-versus-observed gap",
                "cg_analysis": "(a) fundamental or (b) non-essential analysis | null",
            },
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    v29 = revision == "v29"
    payload = {
        "schema_version": "m7-v29-diagnosis-prompt-v1" if v29 else "m7-v26-diagnosis-prompt-v1",
        "role": (
            "M7 diagnostic and routing critic; never a patch generator, oracle relaxer, "
            "threshold tuner, or admission-rule bypass"
        ),
        "task": "Diagnose why the current pre-patch reproduction candidate is not aligned and select one rerun entry point.",
        "rules": [
            "Use only the supplied M1-M7 pre-patch evidence.",
            "Explain the evidence-to-assumption gap before selecting a route.",
            "Choose →M5 only when the target and scenario are supported and only the generated test stimulus/oracle must change.",
            "Choose →M3 when the target remains plausible but the scenario assumption, setup, or stimulus must change.",
            "Choose →M2 when coverage, SBFL, execution evidence, or API ownership contradicts the fault hypothesis, file, function, or target.",
            "Never recommend editing production source, applying a patch, weakening/removing assertions, bypassing the oracle, or lowering a gate.",
            "Never weaken expected behavior, semantic oracle, target verification, numeric gates, or the Conservative Gate merely to make a candidate pass.",
            "Never treat observed buggy behavior as expected behavior unless M1 issue evidence explicitly supports that interpretation.",
            "Never invent issue requirements absent from M1 evidence.",
            "Never claim a target or function is correct without supplied localization or execution evidence.",
            "change_owner_module must exactly own the requested change and match the route: →M2=M2, →M3=M3, →M5=M5, →M8=M8.",
            "Cite concrete supplied evidence paths in evidence_refs; do not cite absent fields or unsupported facts.",
            "If previous_feedback is present, state its observed effect and recommend a materially different change when it was ineffective.",
            "If evidence is uncertain, choose the safest evidence-seeking reroute and never manufacture certainty.",
            "Do not mention or infer golden, patched, after-patch, M8, F2P, or PHR results.",
            "Do not copy the input evidence object or return legacy diagnosis/diagnosis_reason fields.",
            "Every scalar output must have the exact JSON type shown in the schema.",
            "admissible_alternatives must be one semicolon-separated string, never an array or object.",
            (
                "Choose →M8 only when all quantitative gates pass, the Conservative Gate is the only branch, "
                "and conservative_gate_assessment is exactly non-critical."
                if v29
                else "Never choose →M8."
            ),
            f"Return exactly the {'eleven' if v29 else 'five'} required JSON fields and no prose.",
        ],
        "required_output_schema": {
            "failure_reason": "non-empty string grounded in observed evidence",
            "assumption_gap": "non-empty string comparing the scenario/fault assumption with evidence",
            "next_scenario_change": "non-empty concrete change consumed by the rerun",
            "admissible_alternatives": "one string containing one or more concrete alternatives",
            "route_destination": "→M5|→M3|→M2",
        },
        "example_shape_only": {
            "failure_reason": "Observed pre-patch evidence that explains the failed gate.",
            "assumption_gap": "Specific conflict between the prior assumption and observed evidence.",
            "next_scenario_change": "Concrete next-pass change.",
            "admissible_alternatives": "Alternative A; Alternative B.",
            "route_destination": "→M5",
        },
        "pre_patch_evidence": compact_evidence,
    }
    if v29:
        payload["required_output_schema"]["evidence_refs"] = (
            "non-empty array of concrete dot-path strings into pre_patch_evidence"
        )
        payload["required_output_schema"]["conservative_gate_assessment"] = (
            "non-critical|critical|null"
        )
        payload["required_output_schema"]["recommended_change"] = (
            "one concrete test-generation/scenario/localization change owned by change_owner_module"
        )
        payload["required_output_schema"]["change_owner_module"] = "M2|M3|M5|M8"
        payload["required_output_schema"]["previous_feedback_effect"] = (
            "IMPROVED|PARTIALLY_IMPROVED|NO_IMPROVEMENT|REGRESSED|NOT_APPLICABLE"
        )
        payload["required_output_schema"]["confidence"] = "JSON number in [0,1]"
        payload["required_output_schema"]["route_destination"] = "→M5|→M3|→M2|→M8"
        payload["example_shape_only"]["evidence_refs"] = [
            "m6_execution_evidence.test_results",
            "score_breakdown.issue_alignment_score",
        ]
        payload["example_shape_only"]["conservative_gate_assessment"] = None
        payload["example_shape_only"]["recommended_change"] = (
            "Regenerate the test oracle so it asserts the issue-stated expected value."
        )
        payload["example_shape_only"]["change_owner_module"] = "M5"
        payload["example_shape_only"]["previous_feedback_effect"] = "NOT_APPLICABLE"
        payload["example_shape_only"]["confidence"] = 0.8
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)


def diagnose_m7(
    *,
    evidence: Mapping[str, Any],
    client: DiagnosisClient | None,
    enabled: bool,
    revision: str = "v26",
) -> M7Diagnosis:
    """Return one strict LLM diagnosis or a deterministic evidence-based fallback.

    The 120-second marker is telemetry only. It never changes the diagnosis,
    route, pass budget, or provenance.
    """
    if not enabled:
        return deterministic_fallback(evidence, reason="diagnosis_feature_disabled", revision=revision)
    if client is None or not hasattr(client, "generate"):
        return deterministic_fallback(evidence, reason="diagnosis_client_unavailable", revision=revision)
    try:
        prompt = build_m7_diagnosis_prompt(evidence, revision=revision)
    except ValueError as exc:
        return deterministic_fallback(evidence, reason=f"invalid_pre_patch_evidence:{exc}", revision=revision)
    started_at = time.monotonic()
    raw_response = ""
    try:
        raw_response = str(
            client.generate(
                prompt,
                system_prompt=(
                    "You are the v36 M7 pre-patch diagnosis controller. "
                    "Return exactly one JSON object and no prose."
                    if revision in {"v36", "v37"}
                    else "You are the v26 M7 pre-patch diagnosis controller. "
                    "Return exactly one JSON object and no prose."
                ),
                temperature=0.0,
            )
        )
    except Exception as exc:
        elapsed = round(time.monotonic() - started_at, 3)
        return deterministic_fallback(
            evidence,
            reason=f"diagnosis_call_failed:{type(exc).__name__}:{exc}",
            llm_attempted=True,
            model_request_elapsed_sec=elapsed,
            raw_response=raw_response,
            revision=revision,
        )
    elapsed = round(time.monotonic() - started_at, 3)
    try:
        parsed = parse_m7_diagnosis(raw_response, revision=revision, evidence=evidence)
    except ValueError as exc:
        return deterministic_fallback(
            evidence,
            reason=f"diagnosis_response_invalid:{exc}",
            llm_attempted=True,
            llm_succeeded=True,
            model_request_elapsed_sec=elapsed,
            raw_response=raw_response,
            revision=revision,
        )
    if revision == "v27":
        expected_route, consistency_reason = expected_route_from_diagnosis(
            parsed, evidence
        )
        if expected_route is not None and expected_route != parsed["route_destination"]:
            return M7Diagnosis(
                **{**parsed, "route_destination": expected_route},
                provenance=DETERMINISTIC_FALLBACK,
                fallback_reason=f"route_inconsistent:{consistency_reason}",
                llm_attempted=True,
                llm_succeeded=True,
                parse_succeeded=True,
                model_request_elapsed_sec=elapsed,
                exceeded_120s_telemetry_marker=elapsed > TELEMETRY_MARKER_SECONDS,
                raw_response=raw_response,
                llm_route=parsed["route_destination"],
                route_consistency_status="REJECTED_INCONSISTENT",
                final_route=expected_route,
            )
        consistency_status = "ACCEPTED" if expected_route else "INDETERMINATE_ACCEPTED"
    else:
        consistency_status = "NOT_CHECKED_V26"
    return M7Diagnosis(
        **parsed,
        provenance=LLM_DIAGNOSIS,
        llm_attempted=True,
        llm_succeeded=True,
        parse_succeeded=True,
        model_request_elapsed_sec=elapsed,
        exceeded_120s_telemetry_marker=elapsed > TELEMETRY_MARKER_SECONDS,
        raw_response=raw_response,
        llm_route=parsed["route_destination"],
        route_consistency_status=consistency_status,
        final_route=parsed["route_destination"],
    )


def validate_v29_m8_route(
    evidence: Mapping[str, Any], assessment: Any
) -> str | None:
    """Return a rejection reason unless the narrow v29 override is safe."""
    scores = _mapping(evidence.get("score_breakdown"))
    gate_results = _mapping(evidence.get("gate_results"))
    s_b = gate_results.get("s_b", scores.get("bug_fail_score"))
    s_c = gate_results.get("s_c_prime", scores.get("coverage_score"))
    s_a = gate_results.get("s_a", scores.get("issue_alignment_score"))
    try:
        quantitative_pass = (
            float(s_b) >= 0.70 and float(s_c) >= 0.60 and float(s_a) >= 0.65
        )
    except (TypeError, ValueError):
        return "quantitative_gate_evidence_unavailable"
    if not quantitative_pass:
        return "one_or_more_quantitative_gates_failed"
    conservative_reasons = list(
        gate_results.get("conservative_gate_reasons")
        or scores.get("conservative_gate_reasons")
        or []
    )
    if not conservative_reasons:
        return "conservative_gate_not_triggered"
    if scores.get("conservative_gate_is_only_branching_reason") is not True:
        return "conservative_gate_is_not_the_only_branching_reason"
    normalized = str(assessment or "").strip().lower().replace("_", "-")
    if normalized != "non-critical":
        return "conservative_gate_assessment_not_non_critical"
    return None


def validate_v29_feedback_policy(
    diagnosis: Mapping[str, Any], evidence: Mapping[str, Any]
) -> str | None:
    """Reject unsafe, unsupported, or ownership-inconsistent v29 feedback."""
    route = str(diagnosis.get("route_destination") or "")
    owner = str(diagnosis.get("change_owner_module") or "").strip().upper()
    expected_owner = _V29_ROUTE_OWNER.get(route)
    if expected_owner is None or owner != expected_owner:
        return f"change_owner_module_{owner or 'missing'}_does_not_match_{route or 'missing_route'}"

    effect = str(diagnosis.get("previous_feedback_effect") or "").strip().upper()
    allowed_effects = {
        "NOT_APPLICABLE",
        "IMPROVED",
        "PARTIALLY_IMPROVED",
        "NO_IMPROVEMENT",
        "REGRESSED",
    }
    if effect not in allowed_effects:
        return "invalid_previous_feedback_effect"

    refs = tuple(str(item).strip() for item in diagnosis.get("evidence_refs") or ())
    if not refs:
        return "at_least_one_concrete_evidence_ref_required"
    compact = _compact_evidence_for_prompt(evidence)
    for ref in refs:
        root = ref.split(".", 1)[0]
        if root not in _V29_EVIDENCE_ROOTS or not _evidence_path_exists(compact, ref):
            return f"unsupported_evidence_ref:{ref}"

    change_text = " ".join(
        str(diagnosis.get(key) or "")
        for key in (
            "failure_reason",
            "assumption_gap",
            "recommended_change",
            "next_scenario_change",
            "admissible_alternatives",
        )
    ).lower()
    unsafe = next(
        (pattern for pattern in _V29_UNSAFE_CHANGE_PATTERNS if pattern in change_text),
        None,
    )
    if unsafe:
        return f"unsafe_change_instruction:{unsafe}"
    if (
        "change implementation" in change_text
        or "change the implementation" in change_text
        or "modify implementation" in change_text
        or "modify the implementation" in change_text
    ) and not any(
        qualifier in change_text
        for qualifier in ("test implementation", "candidate implementation", "generated test")
    ):
        return "unsafe_change_instruction:unsupported_implementation_change"

    if route == ROUTE_M8:
        change = str(diagnosis.get("recommended_change") or "").lower()
        change_words = set(_normalized_instruction(change).split())
        if change_words.intersection({"regenerate", "rewrite", "change", "revise", "refresh"}):
            return "m8_route_must_not_request_candidate_change"

    expected_route, consistency_reason = _v29_evidence_required_route(evidence)
    if expected_route is not None and expected_route != route:
        return f"route_inconsistent:{consistency_reason}"

    previous = _mapping(evidence.get("previous_feedback"))
    if previous:
        if effect == "NOT_APPLICABLE":
            return "previous_feedback_present_but_effect_not_assessed"
        prior_change = " ".join(
            str(previous.get(key) or "")
            for key in ("recommended_change", "next_scenario_change", "concrete_repair_instruction")
        )
        current_change = " ".join(
            str(diagnosis.get(key) or "")
            for key in ("recommended_change", "next_scenario_change")
        )
        if effect in {"NO_IMPROVEMENT", "PARTIALLY_IMPROVED", "REGRESSED"} and _normalized_instruction(
            prior_change
        ) == _normalized_instruction(current_change):
            return "repeated_ineffective_feedback"
    elif effect != "NOT_APPLICABLE":
        return "previous_feedback_absent_but_effect_claimed"
    return None


def _evidence_path_exists(evidence: Mapping[str, Any], path: str) -> bool:
    current: Any = evidence
    remaining = path
    while remaining:
        if not isinstance(current, Mapping):
            return False
        if remaining in current:
            return current[remaining] is not None
        part, separator, suffix = remaining.partition(".")
        if part not in current:
            return False
        current = current[part]
        remaining = suffix if separator else ""
    return current is not None


def _normalize_evidence_ref(value: str) -> str:
    normalized = str(value).strip()
    prefix = "pre_patch_evidence."
    while normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
    return normalized


def _v29_evidence_required_route(
    evidence: Mapping[str, Any]
) -> tuple[str | None, str]:
    score_breakdown = _mapping(evidence.get("score_breakdown"))
    dedicated_conservative_gate = (
        str(evidence.get("m7_decision_context") or "").upper()
        == "V29_CONSERVATIVE_GATE"
    )
    if _fact_is_false(score_breakdown.get("target_verified")):
        return ROUTE_M2, "prepatch_target_verification_failed"
    if any(
        _fact_is_false(evidence.get(key))
        for key in (
            "fault_hypothesis_supported",
            "suspected_file_covered",
            "suspected_function_covered",
        )
    ):
        return ROUTE_M2, "prepatch_localization_evidence_contradicts_target"
    if _fact_is_false(evidence.get("scenario_assumption_supported")) or _fact_is_false(
        _mapping(evidence.get("m3_scenario_evidence")).get(
            "scenario_assumption_supported"
        )
    ):
        return ROUTE_M3, "prepatch_scenario_assumption_unsupported"
    failure_detail = str(
        evidence.get("failure_type_detail")
        or score_breakdown.get("failure_type_detail")
        or ""
    ).upper()
    diagnosis = str(evidence.get("diagnosis") or "").lower()
    if failure_detail in {"ORACLE_REJECTED", "SEMANTIC_RISK"} or (
        not dedicated_conservative_gate
        and any(
            term in diagnosis
            for term in (
                "expected_behavior_oracle_not_preserved",
                "no_direct_expected_output_and_no_issue_supported_relational_oracle_candidate",
                "scenario semantics",
            )
        )
    ):
        return ROUTE_M3, "prepatch_scenario_or_oracle_contract_unsupported"
    return None, "no_conclusive_route_constraint"


def _normalized_instruction(value: str) -> str:
    return " ".join("".join(char if char.isalnum() else " " for char in value.lower()).split())


def expected_route_from_diagnosis(
    diagnosis: Mapping[str, Any], evidence: Mapping[str, Any]
) -> tuple[str | None, str]:
    """Infer the diagnosis abstraction from text plus pre-patch runtime facts."""
    score_breakdown = _mapping(evidence.get("score_breakdown"))
    if score_breakdown.get("target_verified") is False:
        return ROUTE_M2, "prepatch_target_verification_failed"
    if any(
        _fact_is_false(evidence.get(key))
        for key in (
            "fault_hypothesis_supported",
            "suspected_file_covered",
            "suspected_function_covered",
        )
    ):
        return ROUTE_M2, "prepatch_localization_evidence_contradicts_target"
    text = " ".join(
        str(diagnosis.get(key) or "")
        for key in ("failure_reason", "assumption_gap", "next_scenario_change")
    ).lower()
    localization_terms = (
        "wrong target", "incorrect target", "target function is likely wrong",
        "fault hypothesis", "localization", "wrong file", "different function",
        "context is wrong", "refresh code context", "relocaliz",
    )
    scenario_terms = (
        "scenario is wrong", "scenario specification", "scenario assumption",
        "conceptually wrong", "wrong behavior", "neighboring behavior",
        "trigger interpretation", "reproduction strategy", "revise scenario",
    )
    implementation_terms = (
        "syntax", "import", "collection", "api construction", "test implementation",
        "assertion implementation", "rewrite the candidate", "fixture", "test code",
    )
    matches = {
        ROUTE_M2: any(term in text for term in localization_terms),
        ROUTE_M3: any(term in text for term in scenario_terms),
        ROUTE_M5: any(term in text for term in implementation_terms),
    }
    selected = [route for route, matched in matches.items() if matched]
    if len(selected) == 1:
        return selected[0], f"diagnosis_abstraction_{selected[0][1:].lower()}"
    if len(selected) > 1:
        return None, "mixed_abstraction_language"
    return None, "insufficient_abstraction_evidence"


def deterministic_fallback(
    evidence: Mapping[str, Any],
    *,
    reason: str,
    llm_attempted: bool = False,
    llm_succeeded: bool = False,
    model_request_elapsed_sec: float | None = None,
    raw_response: str = "",
    revision: str = "v26",
) -> M7Diagnosis:
    """Map explicit evidence to one conservative v26 rerun route."""
    route = _fallback_route_v37(evidence) if revision == "v37" else _fallback_route(evidence)
    diagnosis = str(evidence.get("diagnosis") or evidence.get("failure_reason") or "").strip()
    failure_reason = diagnosis or "The current candidate did not satisfy deterministic M7 gates."
    assumption_gap = _fallback_assumption_gap(evidence, route)
    next_change = {
        ROUTE_M2: "Refresh code context and fault localization, then rebuild the scenario from the new target evidence.",
        ROUTE_M3: "Regenerate the single scenario with a revised assumption, stimulus, and EB-grounded oracle.",
        ROUTE_M5: "Preserve supported context and scenario evidence while rewriting the candidate stimulus or assertion.",
    }[route]
    alternatives = {
        ROUTE_M2: "Select another repository-supported target from refreshed M2 evidence.",
        ROUTE_M3: "Use another issue-grounded stimulus that reaches the supported target.",
        ROUTE_M5: "Rewrite the oracle from expected behavior without changing the supported target.",
    }[route]
    elapsed = model_request_elapsed_sec
    evidence_refs = {
        ROUTE_M2: ("m2_semantic_evidence", "coverage_evidence"),
        ROUTE_M3: ("m3_scenario_evidence", "m6_execution_evidence"),
        ROUTE_M5: ("m5_candidate_evidence", "m6_execution_evidence"),
    }[route]
    previous_effect = "NOT_APPLICABLE"
    if _mapping(evidence.get("previous_feedback")):
        effect_record = _mapping(evidence.get("previous_feedback_effect"))
        previous_effect = str(effect_record.get("assessment") or "NO_IMPROVEMENT").upper()
        if previous_effect not in {
            "IMPROVED",
            "PARTIALLY_IMPROVED",
            "NO_IMPROVEMENT",
            "REGRESSED",
        }:
            previous_effect = "NO_IMPROVEMENT"
    return M7Diagnosis(
        failure_reason=failure_reason,
        assumption_gap=assumption_gap,
        next_scenario_change=next_change,
        admissible_alternatives=alternatives,
        route_destination=route,
        provenance=DETERMINISTIC_FALLBACK,
        fallback_reason=reason,
        llm_attempted=llm_attempted,
        llm_succeeded=llm_succeeded,
        parse_succeeded=False,
        model_request_elapsed_sec=elapsed,
        exceeded_120s_telemetry_marker=(
            elapsed is not None and elapsed > TELEMETRY_MARKER_SECONDS
        ),
        raw_response=raw_response,
        evidence_refs=evidence_refs,
        recommended_change=next_change,
        change_owner_module=_V29_ROUTE_OWNER[route],
        previous_feedback_effect=previous_effect,
    )


def _fallback_route(evidence: Mapping[str, Any]) -> str:
    score_breakdown = _mapping(evidence.get("score_breakdown"))
    if _fact_is_false(score_breakdown.get("target_verified")):
        return ROUTE_M2
    if _fact_is_false(evidence.get("fault_hypothesis_supported")):
        return ROUTE_M2
    if _fact_is_false(evidence.get("suspected_file_covered")):
        return ROUTE_M2
    if _fact_is_false(evidence.get("suspected_function_covered")):
        return ROUTE_M2
    if bool(evidence.get("repeated_semantic_fingerprint")):
        return ROUTE_M2
    diagnosis = str(
        evidence.get("diagnosis") or evidence.get("failure_reason") or ""
    ).lower()
    failure_detail = str(
        evidence.get("failure_type_detail")
        or score_breakdown.get("failure_type_detail")
        or ""
    ).upper()
    if failure_detail in {"ORACLE_REJECTED", "SEMANTIC_RISK"}:
        return ROUTE_M3
    scenario_or_oracle_contract_terms = (
        "expected_behavior_oracle_not_preserved",
        "relational_equality_assertion_missing",
        "right_comparator_target_result",
        "no direct expected output",
        "no issue-supported relational oracle",
        "scenario assumption",
        "scenario semantics",
    )
    if any(term in diagnosis for term in scenario_or_oracle_contract_terms):
        return ROUTE_M3
    if _fact_is_false(evidence.get("scenario_assumption_supported")):
        return ROUTE_M3
    if _fact_is_false(evidence.get("issue_branch_reached")):
        return ROUTE_M3
    return ROUTE_M5


def _fallback_route_v37(evidence: Mapping[str, Any]) -> str:
    """Apply the exact v37 gate-only deterministic fallback table."""
    score_breakdown = _mapping(evidence.get("score_breakdown"))
    gates = _mapping(score_breakdown.get("gate_results") or evidence.get("gate_results"))
    failures = [
        name
        for name, value in (
            ("gate1", gates.get("gate1_pass")),
            ("gate2", gates.get("gate2_pass")),
            ("gate3", gates.get("gate3_pass")),
        )
        if value is not True
    ]
    if len(failures) > 1:
        return ROUTE_M5
    if failures == ["gate3"]:
        return ROUTE_M3
    return ROUTE_M5


def _fallback_assumption_gap(evidence: Mapping[str, Any], route: str) -> str:
    if route == ROUTE_M2:
        return "Pre-patch execution or coverage evidence does not support the current fault hypothesis or selected target."
    if route == ROUTE_M3:
        return "The current code context remains admissible, but execution evidence does not support the scenario assumption or stimulus path."
    return "The target and scenario remain admissible, but the generated candidate or oracle does not express the expected behavior strongly enough."


def _fact_is_false(value: Any) -> bool:
    if value is False:
        return True
    if isinstance(value, Mapping):
        return str(value.get("value") or "").upper() == "FALSE"
    return False


def _extract_json_object(raw_response: str) -> str:
    text = str(raw_response or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    decoder = json.JSONDecoder()
    _, end = decoder.raw_decode(text[start:])
    return text[start : start + end]


def _reject_forbidden_m8_claims(value: Any) -> None:
    serialized = json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True).lower()
    found = next((term for term in _FORBIDDEN_EVIDENCE_TERMS if term in serialized), None)
    if found:
        raise ValueError(f"forbidden M8/golden/post-patch claim: {found}")


def _compact_evidence_for_prompt(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Bound M7 prompt evidence while preserving every decision-bearing category."""
    source = _json_safe(evidence)
    if not isinstance(source, Mapping):
        return {}
    m2 = _mapping(source.get("m2_semantic_evidence"))
    target_source = str(m2.get("target_source") or "").replace("\\", "/").lstrip("./")
    coverage = _mapping(source.get("coverage_evidence"))
    covered_files = [str(item) for item in coverage.get("covered_files", []) if str(item)]
    target_matches = [
        item
        for item in covered_files
        if target_source and (target_source in item.replace("\\", "/") or item.replace("\\", "/") in target_source)
    ]
    bounded_files = _dedupe_strings(target_matches + covered_files)[:30]

    score = _mapping(source.get("score_breakdown"))
    compact_score = {
        key: score.get(key)
        for key in (
            "bug_fail_score",
            "coverage_score",
            "issue_alignment_score",
            "oracle_confidence_score",
            "oracle_risk_flags",
            "conservative_gate_reasons",
            "gate_warnings",
            "failure_type_detail",
            "target_verified",
            "strong_issue_evidence",
        )
        if key in score
    }
    m6 = _mapping(source.get("m6_execution_evidence"))
    compact_m6 = {
        "test_results": _bounded_mapping(_mapping(m6.get("test_results")), 20),
        "error_messages": _bounded_strings(m6.get("error_messages"), 8, 500),
        "observed_output_excerpt": str(m6.get("observed_output_excerpt") or "")[-2000:],
        "has_failure": m6.get("has_failure"),
        "has_error": m6.get("has_error"),
        "F_set": _bounded_strings(m6.get("F_set"), 20, 300),
        "P_set": _bounded_strings(m6.get("P_set"), 20, 300),
        "error_tests": _bounded_strings(m6.get("error_tests"), 20, 300),
        "covered_sut_line_count": m6.get("covered_sut_line_count"),
        "covered_sut_lines_excerpt": list(m6.get("covered_sut_lines_excerpt") or [])[:20],
        "stability": _compact_stability(m6.get("stability")),
        "sbfl_reference": m6.get("sbfl_reference"),
    }
    compact = {
        "m7_decision_context": source.get("m7_decision_context"),
        "m7_status": source.get("m7_status"),
        "m7_gate_summary": source.get("diagnosis"),
        "score_breakdown": compact_score,
        "m1_issue_evidence": _compact_issue_evidence(source.get("m1_issue_evidence")),
        "m2_semantic_evidence": m2,
        "m3_scenario_evidence": _mapping(source.get("m3_scenario_evidence")),
        "m5_candidate_evidence": _mapping(source.get("m5_candidate_evidence")),
        "m6_execution_evidence": compact_m6,
        "coverage_evidence": {
            "target_source": target_source,
            "covered_file_count": len(covered_files),
            "covered_files_excerpt": bounded_files,
            "artifact": coverage.get("artifact"),
        },
        "previous_feedback": _mapping(source.get("previous_feedback")),
        "previous_feedback_effect": _mapping(source.get("previous_feedback_effect")),
    }
    for key in (
        "issue_api_executed",
        "assertion_executed",
        "actual_output_observed",
        "exception_observed",
        "suspected_file_covered",
        "suspected_function_covered",
        "suspected_lines_covered",
        "issue_branch_reached",
        "incorrect_behavior_observed",
        "oracle_checked_behavior",
        "fault_hypothesis_supported",
        "scenario_assumption_supported",
        "repeated_semantic_fingerprint",
        "remaining_outer_iterations",
    ):
        if key in source:
            compact[key] = source[key]
    return compact


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _bounded_mapping(value: Mapping[str, Any], limit: int) -> dict[str, Any]:
    return {str(key): item for key, item in list(value.items())[:limit]}


def _bounded_strings(value: Any, limit: int, char_limit: int) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    return [str(item)[:char_limit] for item in list(value)[:limit]]


def _dedupe_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _compact_issue_evidence(value: Any) -> dict[str, Any]:
    issue = _mapping(value)
    return {
        key: _bounded_strings(issue.get(key), 8, 500)
        for key in ("observed_behavior", "expected_behavior", "steps_to_reproduce")
    }


def _compact_stability(value: Any) -> Any:
    stability = _mapping(value)
    if not stability:
        return {}
    return {
        key: stability.get(key)
        for key in ("status", "stable", "is_flaky", "attempts", "outcomes", "reason")
        if key in stability
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

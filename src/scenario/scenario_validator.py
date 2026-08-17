from __future__ import annotations

import json
import ast
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from src.scenario.code_block_roles import (
    ROLE_ACTUAL_BUGGY_OUTPUT,
    ROLE_BASELINE,
    ROLE_BUG_TRIGGER,
    ROLE_EXPECTED_OUTPUT,
    ROLE_SETUP,
    block_inferred_role,
    classify_reproduction_code_blocks,
    strict_normalized_output_equals,
)


SCORE_SCHEMA_VERSION = "scenario-dimensions-v1"
SCORE_AGGREGATION = "mean_applicable_dimensions_v1"
HIGH_CONFIDENCE = 1.0
MEDIUM_CONFIDENCE = 0.7
LOW_CONFIDENCE = 0.4
NO_EVIDENCE = 0.0
VERIFIED_DIRECT_TARGET = "VERIFIED_DIRECT_TARGET"
VERIFIED_PUBLIC_API = "VERIFIED_PUBLIC_API"
TARGET_UNRESOLVED = "TARGET_UNRESOLVED"
TARGET_CONFLICT = "TARGET_CONFLICT"
INVALID_TARGET = "INVALID_TARGET"
INVALID_SCENARIO_STRUCTURE = "INVALID_SCENARIO_STRUCTURE"
_ELIGIBLE_TARGET_CLASSIFICATIONS = {
    VERIFIED_DIRECT_TARGET,
    VERIFIED_PUBLIC_API,
    TARGET_UNRESOLVED,
}


@dataclass
class ScenarioValidationResult:
    scenario_id: str
    score: float
    decision: str
    reasons: List[str]
    normalized_scenario: Dict[str, Any] | None = None
    force_selected: bool = False
    validation_passed: bool = False
    score_breakdown: Dict[str, Any] = field(default_factory=dict)
    classification: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.setdefault("classification", "")
        normalized_status = (
            str(self.normalized_scenario.get("validation_status") or "")
            if isinstance(self.normalized_scenario, Mapping)
            else ""
        )
        data["validation_status"] = (
            "rejected_feedback_not_applied"
            if normalized_status == "rejected_feedback_not_applied"
            else _validation_status_for_decision(self.decision)
        )
        data["diagnostic_only"] = data["validation_status"] != "accepted"
        normalized = data.get("normalized_scenario")
        if isinstance(normalized, dict):
            normalized.setdefault("validation_status", data["validation_status"])
            normalized.setdefault("diagnostic_only", data["diagnostic_only"])
        return data


@dataclass
class ScenarioValidationReport:
    selected_scenarios: List[ScenarioValidationResult]
    rejected_scenarios: List[ScenarioValidationResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_scenarios": [x.to_dict() for x in self.selected_scenarios],
            "rejected_scenarios": [x.to_dict() for x in self.rejected_scenarios],
        }


_FALLBACK_MIN_SCORE = 0.0  # Always keep at least one usable scenario.

M4_STRUCTURALLY_INVALID = "STRUCTURALLY_INVALID"
M4_SEMANTICALLY_WEAK = "SEMANTICALLY_WEAK"
M4_REDUNDANT = "REDUNDANT"
M4_ISSUE_IRRELEVANT = "ISSUE_IRRELEVANT"
M4_VALID = "VALID"


def _m4_classification(
    score_breakdown: Mapping[str, Any],
    score: float,
    validation_passed: bool,
) -> str:
    if not score_breakdown.get("hard_validity_passed", False):
        return M4_STRUCTURALLY_INVALID
    dimensions = score_breakdown.get("dimensions") if isinstance(score_breakdown.get("dimensions"), Mapping) else {}
    oracle = dimensions.get("oracle") if isinstance(dimensions.get("oracle"), Mapping) else {}
    issue = dimensions.get("issue_evidence") if isinstance(dimensions.get("issue_evidence"), Mapping) else {}
    execution = dimensions.get("execution") if isinstance(dimensions.get("execution"), Mapping) else {}
    if float(issue.get("score") or 0.0) < MEDIUM_CONFIDENCE or float(execution.get("score") or 0.0) < MEDIUM_CONFIDENCE:
        return M4_ISSUE_IRRELEVANT
    if float(oracle.get("score") or 0.0) < MEDIUM_CONFIDENCE or not validation_passed:
        return M4_SEMANTICALLY_WEAK
    return M4_VALID


def _complete_deterministic_fallback_contract(
    scenario: Dict[str, Any],
    normalized: Dict[str, Any],
    score_breakdown: Dict[str, Any],
) -> bool:
    """Return whether a deterministic fallback is eligible without force-select."""

    provenance = str(
        scenario.get("generation_provenance")
        or normalized.get("generation_provenance")
        or ""
    )
    is_fallback = bool(scenario.get("fallback_used") or "fallback" in provenance)
    if not is_fallback:
        return False
    if not score_breakdown.get("hard_validity_passed", False):
        return False
    return all(
        [
            normalized.get("source_file"),
            normalized.get("target_function"),
            normalized.get("candidate_test_file"),
            normalized.get("stimulus_steps"),
            normalized.get("expected_failure"),
            normalized.get("oracle_expected") is not None,
            normalized.get("reproduction_code"),
        ]
    )


def _strong_unresolved_public_api_contract(
    normalized: Dict[str, Any],
    score_breakdown: Dict[str, Any],
) -> bool:
    target_schema = score_breakdown.get("target_location_schema")
    if not isinstance(target_schema, dict):
        return False
    if target_schema.get("target_classification") != TARGET_UNRESOLVED:
        return False
    if not target_schema.get("target_eligible_for_m5"):
        return False
    if not score_breakdown.get("hard_validity_passed", False):
        return False
    if normalized.get("target_consistency_status") not in {
        "CONSISTENT",
        "CONSISTENT_WITH_UNRESOLVED_IMPLEMENTATION",
    }:
        return False
    dimensions = score_breakdown.get("dimensions")
    if not isinstance(dimensions, dict):
        return False
    return (
        _dimension_score(dimensions.get("issue_evidence")) >= MEDIUM_CONFIDENCE
        and _dimension_score(dimensions.get("execution")) >= MEDIUM_CONFIDENCE
        and _dimension_score(dimensions.get("oracle")) >= MEDIUM_CONFIDENCE
    )


def _dimension_score(dimension: Any) -> float:
    if not isinstance(dimension, dict):
        return 0.0
    value = dimension.get("score")
    base = float(value) if isinstance(value, (int, float)) else 0.0
    if isinstance(value, (int, float)):
        base = float(value)
    evidence = dimension.get("evidence")
    if isinstance(evidence, list) and evidence:
        scores = [
            float(item.get("score", 0.0))
            for item in evidence
            if isinstance(item, dict) and isinstance(item.get("score"), (int, float))
        ]
        return max([base, *scores]) if scores else base
    return base


class ScenarioValidator:
    def __init__(
        self,
        accept_threshold: float = 0.65,
        duplicate_threshold: float = 0.60,
        max_selected: int = 2,
    ) -> None:
        self.accept_threshold = accept_threshold
        self.duplicate_threshold = duplicate_threshold
        self.max_selected = max_selected

    def validate(
        self,
        scenarios: List[Dict[str, Any]],
        clue: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ScenarioValidationReport:
        results: List[ScenarioValidationResult] = []

        for scenario in scenarios:
            input_diagnostic_only = bool(scenario.get("diagnostic_only"))
            input_validation_status = str(scenario.get("validation_status") or "")
            score, reasons, normalized, score_breakdown = self._score_scenario(
                scenario=scenario,
                clue=clue,
                context=context,
            )

            validation_passed = (
                score_breakdown.get("hard_validity_passed", False)
                and score >= self.accept_threshold
            )
            if not validation_passed and _strong_unresolved_public_api_contract(
                normalized,
                score_breakdown,
            ):
                validation_passed = True
                reasons.append(
                    "accepted unresolved public issue API because issue evidence is strong"
                )
            if not validation_passed and _complete_deterministic_fallback_contract(
                scenario,
                normalized,
                score_breakdown,
            ):
                validation_passed = True
                reasons.append(
                    "accepted deterministic fallback because complete scenario contract is present"
                )
            if input_diagnostic_only or input_validation_status == "rejected_feedback_not_applied":
                validation_passed = False
                reasons.append(
                    "feedback-bearing fallback is diagnostic-only because the requested change was not applied"
                )
            decision = "accept" if validation_passed else "reject"
            classification = _m4_classification(score_breakdown, score, validation_passed)
            score_breakdown["m4_semantic_classification"] = classification
            normalized["validation_status"] = (
                "rejected_feedback_not_applied"
                if input_diagnostic_only
                or input_validation_status == "rejected_feedback_not_applied"
                else _validation_status_for_decision(decision)
            )
            normalized["diagnostic_only"] = decision != "accept"

            results.append(
                ScenarioValidationResult(
                    scenario_id=scenario.get("scenario_id", "unknown"),
                    score=round(score, 4),
                    decision=decision,
                    reasons=reasons,
                    normalized_scenario=normalized,
                    validation_passed=validation_passed,
                    score_breakdown=score_breakdown,
                    classification=classification,
                )
            )

        deduped = self._deduplicate(results)

        accepted = [r for r in deduped if r.decision == "accept"]
        rejected = [r for r in deduped if r.decision == "reject"]

        # Rejected scenarios remain diagnostic evidence only. A deterministic
        # fallback may enter M5 only when it passed the complete scenario
        # contract above; best-effort force selection is intentionally disabled.

        accepted.sort(key=self._result_order_key)

        overflow = accepted[self.max_selected:]
        accepted = accepted[: self.max_selected]

        for extra in overflow:
            extra.decision = "reject"
            extra.validation_passed = False
            extra.reasons.append("rejected because max_selected limit exceeded")
            if isinstance(extra.normalized_scenario, dict):
                extra.normalized_scenario["validation_status"] = "rejected"
                extra.normalized_scenario["diagnostic_only"] = True
            rejected.append(extra)

        rejected.sort(key=self._result_order_key)

        return ScenarioValidationReport(
            selected_scenarios=accepted,
            rejected_scenarios=rejected,
        )

    def save(self, report: ScenarioValidationReport, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    def _score_scenario(
        self,
        scenario: Dict[str, Any],
        clue: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[float, List[str], Dict[str, Any], Dict[str, Any]]:
        target = _target_location_mapping(scenario)
        source_file = target.get("source_file", "")
        target_function = target.get("target_function") or ""
        expected_failure = scenario.get("expected_failure", "")
        # LLM이 str 대신 list를 반환할 수 있으므로 방어적 처리
        if isinstance(expected_failure, list):
            expected_failure = " ".join(str(x) for x in expected_failure)
        execution_stimulus = _ensure_list(scenario.get("execution_stimulus", []))

        source_in_context, target_exists = _target_location_exists(
            source_file,
            target_function,
            context,
        )
        target_classification = classify_scenario_target(
            scenario=scenario,
            clue=clue,
            context=context,
            source_in_context=source_in_context,
            target_exists=target_exists,
        )

        hard_validity_passed = all([
            bool(source_file),
            bool(target_function),
            len(execution_stimulus) > 0,
            bool(expected_failure),
            source_in_context,
            target_classification["eligible"],
        ])

        target_location = self._score_target_location(scenario, context)
        issue_evidence = self._score_issue_evidence(scenario, clue, context)
        execution = self._score_execution(scenario)
        oracle = self._score_oracle(scenario, clue)
        reproduction_roles = self._score_reproduction_roles(scenario, clue)
        test_placement = self._score_test_placement(scenario, context)
        dimensions = {
            "target_location": target_location,
            "issue_evidence": issue_evidence,
            "execution": execution,
            "oracle": oracle,
            "reproduction_roles": reproduction_roles,
            "test_placement": test_placement,
        }
        score = self._mean_applicable_dimensions(dimensions)

        reasons = []
        if hard_validity_passed:
            reasons.append("hard validity fields are present")
        else:
            missing = []
            if not source_file:
                missing.append("source_file")
            if not target_function:
                missing.append("target_function")
            if not execution_stimulus:
                missing.append("execution_stimulus")
            if not expected_failure:
                missing.append("expected_failure")
            if not isinstance(scenario.get("target_location"), dict):
                missing.append("target_location_schema_invalid")
            if source_file and not source_in_context:
                missing.append("source_file_not_in_context_candidates")
            if (
                source_file
                and target_function
                and source_in_context
                and not target_classification["eligible"]
            ):
                missing.append(str(target_classification["classification"]).lower())
            reasons.append(f"missing hard validity fields: {', '.join(missing)}")
        if target_classification.get("classification"):
            reasons.append(
                "target_classification: "
                f"{target_classification['classification']} "
                f"({target_classification.get('reason', '')})"
            )
        for name, dimension in dimensions.items():
            evidence = dimension.get("evidence", [])
            if evidence:
                top = max(evidence, key=lambda x: x.get("score", 0.0))
                reasons.append(
                    f"{name}: {top.get('label')} ({top.get('tier')}, score={top.get('score')})"
                )
            else:
                reasons.append(f"{name}: no applicable evidence")

        score_breakdown = {
            "score_schema_version": SCORE_SCHEMA_VERSION,
            "score_range": "0..1",
            "aggregation": SCORE_AGGREGATION,
            "hard_validity_passed": hard_validity_passed,
            "target_location_schema": {
                "source_in_context": source_in_context,
                "target_function_exists": target_exists,
                "target_classification": target_classification["classification"],
                "target_classification_reason": target_classification.get("reason", ""),
                "target_eligible_for_m5": target_classification["eligible"],
            },
            "m4_candidate_classification": target_classification["classification"],
            "dimensions": dimensions,
        }

        normalized = dict(scenario)
        preconditions = _ensure_list(scenario.get("preconditions", []))
        setup_steps = _ensure_list(scenario.get("setup_steps", []))
        # setup_steps에서 비실행형 항목 정리 (preconditions 하위호환 포함)
        all_setup = list(preconditions) + [s for s in setup_steps if s not in preconditions]
        normalized["setup_steps"] = [
            s for s in all_setup
            if not _is_non_actionable_setup(s)
        ]
        normalized.pop("preconditions", None)
        target = _target_location_mapping(normalized)
        if not isinstance(normalized.get("target_location"), dict):
            normalized["target_location"] = target
        normalized["target_function"] = str(target.get("target_function") or normalized.get("target_function") or "")
        normalized["source_file"] = str(target.get("source_file") or normalized.get("source_file") or "")
        normalized["candidate_test_file"] = (
            str(target.get("candidate_test_file") or normalized.get("candidate_test_file") or "")
            or None
        )
        normalized["stimulus_steps"] = _ensure_list(
            normalized.get("stimulus_steps") or normalized.get("execution_stimulus") or []
        )
        if "oracle_expected" not in normalized:
            expected_outputs = _ensure_list(normalized.get("expected_outputs", []))
            normalized["oracle_expected"] = expected_outputs[0] if expected_outputs else None
        normalized["reproduction_code"] = classify_reproduction_code_blocks(
            normalized.get("reproduction_code", []),
            expected_outputs=normalized.get("expected_outputs", []),
            actual_outputs=normalized.get("actual_outputs", []) or clue.get("actual_outputs", []),
            target_function=str(target.get("target_function") or ""),
        )
        normalized["issue_api_target"] = target_classification.get("issue_api_target", "")
        normalized["implementation_target"] = target_classification.get("implementation_target", "")
        normalized["setup_helper_calls"] = target_classification.get("setup_helper_calls", [])
        normalized["target_verification_status"] = target_classification["classification"]
        normalized["target_verification_provenance"] = target_classification.get("provenance", {})
        normalized["target_consistency_status"] = target_classification.get("target_consistency_status", "")
        target["issue_api_target"] = normalized["issue_api_target"]
        target["implementation_target"] = normalized["implementation_target"]
        target["setup_helper_calls"] = normalized["setup_helper_calls"]
        target["target_verification_status"] = normalized["target_verification_status"]
        target["target_verification_provenance"] = normalized["target_verification_provenance"]
        target["target_consistency_status"] = normalized["target_consistency_status"]
        normalized["target_location"] = target
        normalized.setdefault("validation_status", "pending")
        normalized.setdefault("diagnostic_only", True)

        return round(score, 4), reasons, normalized, score_breakdown

    def _score_target_location(
        self,
        scenario: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        target = _target_location_mapping(scenario)
        source_file = target.get("source_file", "")
        target_function = target.get("target_function") or ""
        context_source_entries = context.get("candidate_source_files", []) or []
        source_entry = _find_entry_by_path(context_source_entries, source_file)

        evidence: List[Dict[str, Any]] = []
        missing: List[str] = []
        source_scores: List[float] = []
        symbol_scores: List[float] = []

        if source_file:
            if source_entry:
                ev = _evidence("source_file_candidate_exact_match", MEDIUM_CONFIDENCE, "medium")
                evidence.append(ev)
                source_scores.append(ev["score"])
                if context_source_entries and context_source_entries[0].get("path") == source_file:
                    ev = _evidence("top_ranked_source_file", MEDIUM_CONFIDENCE, "medium")
                    evidence.append(ev)
                    source_scores.append(ev["score"])
            else:
                missing.append("source_file_not_in_context_candidates")

            source_name = Path(source_file).name.lower()
            if target_function:
                func_tokens = set(_split_identifier_tokens(target_function))
                source_tokens = set(_split_identifier_tokens(source_name.replace(".py", "")))
                if func_tokens & source_tokens or target_function.lower() in source_name:
                    ev = _evidence("source_filename_target_token_overlap", LOW_CONFIDENCE, "low")
                    evidence.append(ev)
                    source_scores.append(ev["score"])
        else:
            missing.append("source_file_missing")

        if target_function:
            if source_entry:
                matched_identifiers = set(source_entry.get("matched_identifiers", []) or [])
                if target_function in matched_identifiers:
                    ev = _evidence("target_function_matched_identifier", MEDIUM_CONFIDENCE, "medium")
                    evidence.append(ev)
                    symbol_scores.append(ev["score"])
                elif matched_identifiers:
                    missing.append("target_function_not_in_matched_identifiers")

                top_funcs = source_entry.get("top_level_functions") or []
                bare_name = target_function.split(".")[-1]
                if top_funcs and not (bare_name.startswith("__") and bare_name.endswith("__")):
                    in_top = any(
                        tf == target_function or tf.split(".")[-1] == bare_name
                        for tf in top_funcs
                    )
                    if in_top:
                        ev = _evidence("target_function_ast_exists", HIGH_CONFIDENCE, "high")
                        evidence.append(ev)
                        symbol_scores.append(ev["score"])
                    else:
                        symbol_scores.append(NO_EVIDENCE)
                        missing.append("target_function_not_in_ast")
        else:
            missing.append("target_function_missing")

        applicable_scores = []
        if source_file:
            applicable_scores.append(max(source_scores) if source_scores else NO_EVIDENCE)
        if target_function:
            applicable_scores.append(max(symbol_scores) if symbol_scores else NO_EVIDENCE)

        return _dimension_result(
            score=_mean(applicable_scores),
            applicable=bool(applicable_scores),
            evidence=evidence,
            missing=missing,
        )

    def _score_issue_evidence(
        self,
        scenario: Dict[str, Any],
        clue: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        target = _target_location_mapping(scenario)
        source_file = target.get("source_file", "")
        target_function = target.get("target_function") or ""
        related_classes = set(target.get("related_classes") or [])
        identifiers = clue.get("identifiers", {}) if isinstance(clue.get("identifiers"), dict) else {}
        clue_funcs = {
            fn for fn in identifiers.get("functions", []) or []
            if fn not in _NOISY_FUNCTIONS
        }
        clue_classes = set(identifiers.get("classes", []) or [])

        evidence: List[Dict[str, Any]] = []
        missing: List[str] = []
        scores: List[float] = []

        if clue_funcs:
            if target_function in clue_funcs:
                ev = _evidence("target_function_issue_identifier_exact_match", MEDIUM_CONFIDENCE, "medium")
                evidence.append(ev)
                scores.append(ev["score"])
            else:
                scores.append(NO_EVIDENCE)
                missing.append("target_function_not_in_issue_identifiers")

        if clue_classes:
            overlap = clue_classes & related_classes
            if overlap:
                ev = _evidence("related_class_issue_identifier_overlap", MEDIUM_CONFIDENCE, "medium")
                evidence.append(ev)
                scores.append(ev["score"])
            else:
                scores.append(NO_EVIDENCE)
                missing.append("related_classes_do_not_overlap_issue_classes")

        fault_locations = [
            fl for fl in clue.get("fault_locations", [])
            if fl.get("source", "traceback") == "traceback"
            and fl.get("confidence", "high") == "high"
        ]
        if fault_locations:
            fl_files: set[str] = set()
            fl_funcs: set[str] = set()
            for fl in fault_locations:
                fp = fl.get("file_path", "").replace("\\", "/")
                fn = fl.get("function_name", "")
                parts = fp.split("/")
                for k in range(1, min(5, len(parts))):
                    fl_files.add("/".join(parts[-k:]))
                if fn:
                    fl_funcs.add(fn)

            source_tail = "/".join(source_file.replace("\\", "/").split("/")[-3:])
            file_match = any(
                source_file.endswith(f) or f.endswith(source_file) or source_tail in f
                for f in fl_files
            )
            func_match = bool(target_function and target_function in fl_funcs)

            if file_match and func_match:
                ev = _evidence("traceback_fault_location_file_function_match", HIGH_CONFIDENCE, "high")
                evidence.append(ev)
                scores.append(ev["score"])
            elif file_match:
                ev = _evidence("traceback_fault_location_file_match", MEDIUM_CONFIDENCE, "medium")
                evidence.append(ev)
                scores.append(ev["score"])
            elif func_match:
                ev = _evidence("traceback_fault_location_function_match", MEDIUM_CONFIDENCE, "medium")
                evidence.append(ev)
                scores.append(ev["score"])
            else:
                scores.append(NO_EVIDENCE)
                missing.append("target_does_not_match_traceback_fault_location")

        code_text = _scenario_code_text(scenario)
        clue_code_text = _code_blocks_text(clue.get("code_examples", []) or [])
        combined_code_text = f"{code_text}\n{clue_code_text}".lower()
        if clue.get("code_examples"):
            if target_function and _contains_call_pattern(combined_code_text, target_function):
                ev = _evidence("issue_code_example_target_call_pattern", HIGH_CONFIDENCE, "high")
                evidence.append(ev)
                scores.append(ev["score"])
            else:
                scores.append(NO_EVIDENCE)
                missing.append("issue_code_example_target_call_pattern_missing")

        issue_values = _ensure_list(clue.get("expected_outputs", [])) + _ensure_list(clue.get("actual_outputs", []))
        if issue_values:
            if _outputs_match_any_candidate(issue_values, _scenario_output_candidates(scenario)):
                ev = _evidence("issue_expected_or_actual_output_reflected", MEDIUM_CONFIDENCE, "medium")
                evidence.append(ev)
                scores.append(ev["score"])
            else:
                scores.append(NO_EVIDENCE)
                missing.append("issue_expected_or_actual_output_not_reflected")

        return _dimension_result(
            score=_mean(scores),
            applicable=bool(scores),
            evidence=evidence,
            missing=missing,
        )

    def _score_execution(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        target = _target_location_mapping(scenario)
        target_function = target.get("target_function") or ""
        setup_steps = _ensure_list(scenario.get("setup_steps", []))
        execution_stimulus = _ensure_list(scenario.get("execution_stimulus", []))
        evidence: List[Dict[str, Any]] = []
        missing: List[str] = []
        scores: List[float] = []

        if execution_stimulus:
            execution_text = " ".join(execution_stimulus).lower()
            if target_function and target_function.lower() in execution_text:
                ev = _evidence("execution_stimulus_mentions_target_function", MEDIUM_CONFIDENCE, "medium")
                evidence.append(ev)
                scores.append(ev["score"])
            elif _has_actionable_text(execution_stimulus):
                ev = _evidence("execution_stimulus_actionable", MEDIUM_CONFIDENCE, "medium")
                evidence.append(ev)
                scores.append(ev["score"])
            else:
                scores.append(NO_EVIDENCE)
                missing.append("execution_stimulus_too_abstract")
        else:
            missing.append("execution_stimulus_missing")

        if setup_steps:
            actionable_setup = [s for s in setup_steps if not _is_non_actionable_setup(s)]
            if len(actionable_setup) >= 2:
                ev = _evidence("setup_steps_actionable", MEDIUM_CONFIDENCE, "medium")
                evidence.append(ev)
                scores.append(ev["score"])
            elif actionable_setup:
                ev = _evidence("setup_steps_partially_actionable", LOW_CONFIDENCE, "low")
                evidence.append(ev)
                scores.append(ev["score"])
            else:
                scores.append(NO_EVIDENCE)
                missing.append("setup_steps_non_actionable")

        repro_text = _scenario_code_text(scenario)
        if repro_text:
            ev = _evidence("reproduction_code_present", LOW_CONFIDENCE, "low")
            evidence.append(ev)
            scores.append(ev["score"])

        return _dimension_result(
            score=_mean(scores),
            applicable=bool(scores),
            evidence=evidence,
            missing=missing,
        )

    def _score_oracle(
        self,
        scenario: Dict[str, Any],
        clue: Dict[str, Any],
    ) -> Dict[str, Any]:
        expected_failure = scenario.get("expected_failure", "")
        if isinstance(expected_failure, list):
            expected_failure = " ".join(str(x) for x in expected_failure)
        expected_failure_text = str(expected_failure or "").lower()
        evidence: List[Dict[str, Any]] = []
        missing: List[str] = []
        scores: List[float] = []

        if expected_failure:
            concrete_keywords = [
                "assert", "equal", "raise", "return", "error", "exception",
                "true", "false", "none", "should", "must", "expect",
                "result", "value", "match", "correct", "fail", "pass",
                "output", "produce", "yield", "compare",
            ]
            if any(k in expected_failure_text for k in concrete_keywords):
                ev = _evidence("expected_failure_concrete_checkable", MEDIUM_CONFIDENCE, "medium")
                evidence.append(ev)
                scores.append(ev["score"])
            else:
                scores.append(NO_EVIDENCE)
                missing.append("expected_failure_not_concrete")
            if "assert" in expected_failure_text:
                ev = _evidence("expected_failure_mentions_assertion", MEDIUM_CONFIDENCE, "medium")
                evidence.append(ev)
                scores.append(ev["score"])
        else:
            missing.append("expected_failure_missing")

        issue_values = _ensure_list(clue.get("expected_outputs", [])) + _ensure_list(clue.get("actual_outputs", []))
        if issue_values:
            if _outputs_match_any_candidate(issue_values, _oracle_output_candidates(scenario)):
                ev = _evidence("oracle_reflects_issue_expected_or_actual_output", MEDIUM_CONFIDENCE, "medium")
                evidence.append(ev)
                scores.append(ev["score"])
            else:
                scores.append(NO_EVIDENCE)
                missing.append("oracle_missing_issue_expected_or_actual_output")

        oracle_contract = scenario.get("oracle_contract")
        if isinstance(oracle_contract, dict) and oracle_contract.get("rule"):
            ev = _evidence("oracle_contract_present", LOW_CONFIDENCE, "low")
            evidence.append(ev)
            scores.append(ev["score"])

        return _dimension_result(
            score=_mean(scores),
            applicable=bool(scores),
            evidence=evidence,
            missing=missing,
        )

    def _score_reproduction_roles(
        self,
        scenario: Dict[str, Any],
        clue: Dict[str, Any],
    ) -> Dict[str, Any]:
        actual_outputs = _ensure_list(clue.get("actual_outputs", []))
        issue_blocks = clue.get("code_examples", []) or []
        issue_has_trigger = _issue_has_bug_trigger_evidence(issue_blocks, actual_outputs)
        if not actual_outputs and not issue_has_trigger:
            return _dimension_result(
                score=NO_EVIDENCE,
                applicable=False,
                evidence=[],
                missing=[],
            )

        roles = _classify_reproduction_roles(scenario, actual_outputs)
        evidence: List[Dict[str, Any]] = []
        missing: List[str] = []
        scores: List[float] = []

        if roles["setup"]:
            ev = _evidence("setup_role_present", LOW_CONFIDENCE, "low")
            evidence.append(ev)
            scores.append(ev["score"])
        if roles["expected_output"]:
            ev = _evidence("expected_output_role_present", MEDIUM_CONFIDENCE, "medium")
            evidence.append(ev)
            scores.append(ev["score"])
        if roles["actual_buggy_output"]:
            ev = _evidence("actual_buggy_output_role_present", HIGH_CONFIDENCE, "high")
            evidence.append(ev)
            scores.append(ev["score"])
        if roles["bug_triggering_call"]:
            ev = _evidence("bug_triggering_call_role_present", HIGH_CONFIDENCE, "high")
            evidence.append(ev)
            scores.append(ev["score"])
        if roles["baseline"]:
            ev = _evidence("baseline_sanity_role_present", LOW_CONFIDENCE, "low")
            evidence.append(ev)
            scores.append(ev["score"])

        if actual_outputs and issue_has_trigger and roles["baseline"] and not roles["bug_triggering_call"]:
            scores.append(NO_EVIDENCE)
            missing.append("baseline_only_when_actual_buggy_output_and_bug_trigger_are_available")
        if actual_outputs and not roles["actual_buggy_output"]:
            missing.append("actual_buggy_output_not_reflected")
        if issue_has_trigger and not roles["bug_triggering_call"]:
            missing.append("bug_triggering_call_not_reflected")

        return _dimension_result(
            score=_mean(scores),
            applicable=True,
            evidence=evidence,
            missing=missing,
        )

    def _score_test_placement(
        self,
        scenario: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        target = _target_location_mapping(scenario)
        candidate_test_file = target.get("candidate_test_file") or ""
        relevant_test_files = set(scenario.get("relevant_test_files", []) or [])
        context_test_entries = context.get("candidate_test_files", []) or []
        context_test_files = {x.get("path", "") for x in context_test_entries}
        test_entry = _find_entry_by_path(context_test_entries, candidate_test_file)
        evidence: List[Dict[str, Any]] = []
        missing: List[str] = []
        scores: List[float] = []

        if candidate_test_file:
            if test_entry:
                ev = _evidence("candidate_test_file_context_exact_match", MEDIUM_CONFIDENCE, "medium")
                evidence.append(ev)
                scores.append(ev["score"])
                if test_entry.get("score", 0) >= 20:
                    ev = _evidence("candidate_test_file_strong_retrieval_score", MEDIUM_CONFIDENCE, "medium")
                    evidence.append(ev)
                    scores.append(ev["score"])
                if test_entry.get("has_module_skip"):
                    scores.append(NO_EVIDENCE)
                    missing.append("candidate_test_file_has_module_level_skip")
                else:
                    ev = _evidence("candidate_test_file_skip_free", HIGH_CONFIDENCE, "high")
                    evidence.append(ev)
                    scores.append(ev["score"])
            else:
                scores.append(NO_EVIDENCE)
                missing.append("candidate_test_file_not_in_context_candidates")
        else:
            missing.append("candidate_test_file_missing")

        if relevant_test_files:
            overlap = relevant_test_files & context_test_files
            if overlap:
                tier_score = MEDIUM_CONFIDENCE if len(overlap) >= 2 else LOW_CONFIDENCE
                tier_name = "medium" if len(overlap) >= 2 else "low"
                ev = _evidence("relevant_test_files_context_overlap", tier_score, tier_name)
                evidence.append(ev)
                scores.append(ev["score"])
            else:
                scores.append(NO_EVIDENCE)
                missing.append("relevant_test_files_do_not_overlap_context")

        return _dimension_result(
            score=_mean(scores),
            applicable=bool(scores),
            evidence=evidence,
            missing=missing,
        )

    @staticmethod
    def _mean_applicable_dimensions(dimensions: Dict[str, Dict[str, Any]]) -> float:
        scores = [
            float(d.get("score", 0.0))
            for d in dimensions.values()
            if d.get("applicable", False)
        ]
        return round(_mean(scores), 4)

    def _deduplicate(
        self,
        results: List[ScenarioValidationResult],
    ) -> List[ScenarioValidationResult]:
        kept: List[ScenarioValidationResult] = []

        for current in sorted(results, key=self._result_order_key):
            duplicate_of = None

            for prev in kept:
                sim = self._scenario_similarity(
                    current.normalized_scenario or {},
                    prev.normalized_scenario or {},
                )
                if sim >= self.duplicate_threshold:
                    duplicate_of = prev.scenario_id
                    break

            if duplicate_of is not None:
                current.decision = "reject"
                current.validation_passed = False
                current.classification = "REDUNDANT"
                current.score_breakdown["m4_semantic_classification"] = "REDUNDANT"
                current.reasons.append(f"duplicate of higher-ranked scenario {duplicate_of}")
                current.reasons.append(
                    "repair_required: change target/stimulus/oracle materially before retry"
                )
                if isinstance(current.normalized_scenario, dict):
                    current.normalized_scenario["validation_status"] = "rejected"
                    current.normalized_scenario["diagnostic_only"] = True
                    current.normalized_scenario["repair_required"] = True

            kept.append(current)

        return kept

    @staticmethod
    def _result_order_key(result: ScenarioValidationResult) -> tuple[Any, ...]:
        scenario = result.normalized_scenario or {}
        target = scenario.get("target_location") if isinstance(scenario.get("target_location"), dict) else {}
        stimulus = scenario.get("execution_stimulus") or scenario.get("stimulus_steps") or []
        oracle = scenario.get("oracle_contract") if isinstance(scenario.get("oracle_contract"), dict) else {}
        canonical = json.dumps(scenario, sort_keys=True, separators=(",", ":"), default=str)
        return (
            -float(result.score),
            str(target.get("source_file") or scenario.get("source_file") or ""),
            str(target.get("qualified_symbol") or target.get("function") or scenario.get("target_function") or ""),
            str(oracle.get("oracle_type") or scenario.get("oracle_type") or ""),
            json.dumps(stimulus, sort_keys=True, default=str),
            str(result.scenario_id),
            canonical,
        )

    def _scenario_similarity(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        if not a or not b:
            return 0.0

        a_tokens = self._collect_core_tokens(a)
        b_tokens = self._collect_core_tokens(b)

        if not a_tokens or not b_tokens:
            return 0.0

        inter = len(a_tokens & b_tokens)
        union = len(a_tokens | b_tokens)

        return inter / union if union else 0.0

    def _collect_core_tokens(self, scenario: Dict[str, Any]) -> set[str]:
        tokens: set[str] = set()

        def add_text(text: str) -> None:
            for tok in text.lower().replace("/", " ").replace("_", " ").split():
                tok = tok.strip(" ,.:;()[]{}'\"")
                if len(tok) >= 3:
                    tokens.add(tok)

        target = _target_location_mapping(scenario)
        add_text(target.get("source_file", ""))
        add_text(target.get("target_function", ""))

        for x in target.get("related_classes", []):
            add_text(x)

        for item in scenario.get("execution_stimulus", []):
            add_text(str(item))

        add_text(scenario.get("expected_failure", ""))
        add_text(scenario.get("oracle", ""))
        add_text(_scenario_code_text(scenario))
        for value in _ensure_list(scenario.get("expected_outputs", [])):
            add_text(value)
        for value in _ensure_list(scenario.get("actual_outputs", [])):
            add_text(value)

        return tokens


_NOISY_FUNCTIONS = {
    "arange", "rand", "random", "seed", "platform", "get_backend",
    "show_versions",
}


def _ensure_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    if value:
        return [str(value)]
    return []


def _validation_status_for_decision(decision: str) -> str:
    if decision == "accept":
        return "accepted"
    if decision == "fallback":
        return "fallback"
    if decision == "reject":
        return "rejected"
    return "pending"


def _target_location_mapping(scenario: Dict[str, Any]) -> Dict[str, Any]:
    target = scenario.get("target_location", {}) if isinstance(scenario, dict) else {}
    return target if isinstance(target, dict) else {}


def _classification_allows_progress(result: ScenarioValidationResult) -> bool:
    breakdown = result.score_breakdown or {}
    classification = str(
        breakdown.get("m4_candidate_classification")
        or (breakdown.get("target_location_schema") or {}).get("target_classification")
        or ""
    )
    return classification in _ELIGIBLE_TARGET_CLASSIFICATIONS


def classify_scenario_target(
    *,
    scenario: Dict[str, Any],
    clue: Dict[str, Any],
    context: Dict[str, Any],
    source_in_context: bool,
    target_exists: bool,
) -> Dict[str, Any]:
    target = _target_location_mapping(scenario)
    source_file = str(target.get("source_file") or scenario.get("source_file") or "")
    target_function = str(target.get("target_function") or scenario.get("target_function") or "")
    execution_stimulus = _ensure_list(scenario.get("execution_stimulus", []))
    expected_failure = scenario.get("expected_failure", "")
    issue_api_target, setup_helpers, call_provenance = _infer_issue_api_and_helpers(
        scenario,
        clue,
        target_function,
    )
    target_consistency = _target_consistency_status(target_function, issue_api_target)

    base = {
        "issue_api_target": issue_api_target,
        "setup_helper_calls": setup_helpers,
        "implementation_target": "",
        "target_consistency_status": target_consistency,
        "provenance": {
            "source": "m4_static_target_classifier",
            "issue_api_target_source": call_provenance,
            "source_file": source_file,
            "target_function": target_function,
            "pre_patch_only": True,
        },
    }

    if not source_file or not target_function or not execution_stimulus or not expected_failure:
        return {
            **base,
            "classification": INVALID_SCENARIO_STRUCTURE,
            "eligible": False,
            "reason": "missing_required_scenario_structure",
        }
    if target_consistency == "CONFLICT":
        return {
            **base,
            "classification": TARGET_CONFLICT,
            "eligible": False,
            "reason": "scenario_target_conflicts_with_issue_api_target",
        }
    if not source_in_context:
        return {
            **base,
            "classification": INVALID_TARGET,
            "eligible": False,
            "reason": "source_file_not_in_context_candidates",
        }

    static = _static_target_lookup(context, source_file, target_function)
    if target_exists or static["direct"]:
        return {
            **base,
            "classification": VERIFIED_DIRECT_TARGET,
            "eligible": True,
            "implementation_target": target_function,
            "reason": static["reason"] or "target_function_exists_in_source_context",
            "provenance": {
                **base["provenance"],
                **static["provenance"],
            },
        }
    if static["public_api"]:
        return {
            **base,
            "classification": VERIFIED_PUBLIC_API,
            "eligible": True,
            "implementation_target": static["implementation_target"],
            "reason": static["reason"],
            "provenance": {
                **base["provenance"],
                **static["provenance"],
            },
        }
    if issue_api_target and _has_strong_issue_target_evidence(issue_api_target, scenario, clue):
        return {
            **base,
            "classification": TARGET_UNRESOLVED,
            "eligible": True,
            "reason": "strong_issue_evidence_identifies_public_api_but_static_implementation_unresolved",
        }
    return {
        **base,
        "classification": INVALID_TARGET,
        "eligible": False,
        "reason": "target_function_not_found_and_no_strong_issue_api_evidence",
    }


def _target_consistency_status(target_function: str, issue_api_target: str) -> str:
    if not issue_api_target:
        return "CONSISTENT_WITH_UNRESOLVED_IMPLEMENTATION"
    if _same_callable_name(target_function, issue_api_target):
        return "CONSISTENT"
    return "CONFLICT"


def _infer_issue_api_and_helpers(
    scenario: Dict[str, Any],
    clue: Dict[str, Any],
    target_function: str,
) -> tuple[str, List[str], str]:
    explicit = str(scenario.get("issue_api_target") or "").strip()
    calls = _ordered_issue_calls(scenario, clue)
    if explicit:
        issue_api = explicit
        provenance = "scenario.issue_api_target"
    elif calls and any(_callable_tail(call).lower() in " ".join(_ensure_list(scenario.get("execution_stimulus", []))).lower() for call in calls):
        issue_api = _choose_issue_call(calls, scenario, clue)
        provenance = "scenario.execution_stimulus_call"
    elif target_function and any(_same_callable_name(target_function, call) for call in calls):
        issue_api = target_function
        provenance = "scenario.target_function_present_in_issue_calls"
    elif calls:
        issue_api = _choose_issue_call(calls, scenario, clue)
        provenance = "issue_reproduction_calls"
    else:
        issue_api = target_function
        provenance = "scenario.target_function"
    helpers = []
    for call in calls:
        if not _same_callable_name(call, issue_api) and call not in helpers:
            helpers.append(call)
    return issue_api, helpers, provenance


def _choose_issue_call(calls: List[str], scenario: Dict[str, Any], clue: Dict[str, Any]) -> str:
    stimulus_text = " ".join(_ensure_list(scenario.get("execution_stimulus", []))).lower()
    for call in calls:
        if _callable_tail(call).lower() in stimulus_text:
            return call
    role_blocks = (scenario.get("reproduction_code") or []) + (clue.get("code_examples") or [])
    for block in role_blocks:
        if isinstance(block, dict) and str(block.get("inferred_role") or block.get("role") or "").lower() in {
            "bug_trigger",
            "bug-trigger",
            "failing problem reproduction",
        }:
            block_calls = _extract_calls_from_text(_code_blocks_text([block]))
            if block_calls:
                return block_calls[-1]
    return calls[-1]


def _ordered_issue_calls(scenario: Dict[str, Any], clue: Dict[str, Any]) -> List[str]:
    text = "\n".join([
        _code_blocks_text(scenario.get("reproduction_code", []) or []),
        _code_blocks_text(clue.get("code_examples", []) or []),
        " ".join(_ensure_list(scenario.get("execution_stimulus", []))),
    ])
    calls: List[str] = []
    for call in _extract_calls_from_text(text):
        tail = _callable_tail(call)
        if len(tail) < 3 or tail.lower() in _NOISY_FUNCTIONS:
            continue
        if call not in calls:
            calls.append(call)
    return calls


def _extract_calls_from_text(text: str) -> List[str]:
    calls: List[str] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _ast_call_name(node.func)
                if name:
                    calls.append(name)
    if not calls:
        for dotted, bare in re.findall(r"\b((?:[A-Za-z_]\w*\.)?([A-Za-z_]\w{2,}))\s*\(", text):
            if bare.lower() not in _NOISY_FUNCTIONS:
                calls.append(dotted)
    return calls


def _ast_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _ast_call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _callable_tail(name: str) -> str:
    return str(name or "").split(".")[-1]


def _same_callable_name(left: str, right: str) -> bool:
    return bool(left and right and _callable_tail(left).lower() == _callable_tail(right).lower())


def _has_strong_issue_target_evidence(issue_api_target: str, scenario: Dict[str, Any], clue: Dict[str, Any]) -> bool:
    if not issue_api_target:
        return False
    calls = _ordered_issue_calls(scenario, clue)
    if any(_same_callable_name(issue_api_target, call) for call in calls):
        return True
    identifiers = clue.get("identifiers", {}) if isinstance(clue.get("identifiers"), dict) else {}
    functions = identifiers.get("functions", []) or []
    return any(_same_callable_name(issue_api_target, fn) for fn in functions)


def _static_target_lookup(context: Dict[str, Any], source_file: str, target_function: str) -> Dict[str, Any]:
    repo_path = str(context.get("repo_path") or "")
    source_path = Path(repo_path) / source_file if repo_path and source_file else None
    target_tail = _callable_tail(target_function)
    result = {
        "direct": False,
        "public_api": False,
        "implementation_target": "",
        "reason": "",
        "provenance": {},
    }
    if source_path is None or not source_path.exists():
        return result
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        result["reason"] = f"static_parse_unavailable:{type(exc).__name__}"
        return result
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_tail:
            result.update({
                "direct": True,
                "implementation_target": target_function,
                "reason": "ast_function_or_method_definition",
                "provenance": {"lineno": getattr(node, "lineno", None), "static_evidence": "function_def"},
            })
            return result
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [_assignment_name(target) for target in node.targets]
            if any(_same_callable_name(name, target_function) for name in names):
                result.update({
                    "public_api": True,
                    "implementation_target": _assignment_name(node.value),
                    "reason": "ast_class_or_module_assignment",
                    "provenance": {"lineno": getattr(node, "lineno", None), "static_evidence": "assignment"},
                })
                return result
    return result


def _assignment_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _assignment_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return _assignment_name(node.func)
    return ""


def _target_location_exists(
    source_file: str,
    target_function: str,
    context: Dict[str, Any],
) -> tuple[bool, bool]:
    entries = context.get("candidate_source_files", []) or []
    if not entries:
        return bool(source_file), bool(target_function)
    source_entry = _find_entry_by_path(entries, source_file)
    if source_entry is None:
        return False, False
    symbols: set[str] = set()
    # ``matched_identifiers`` is retrieval evidence, not proof that a dotted
    # receiver expression (for example ``tc.write``) is a repository
    # definition.  Only indexed definitions can establish target identity.
    for key in ("top_level_functions", "methods", "classes"):
        for value in source_entry.get(key, []) or []:
            text = str(value)
            if text:
                symbols.add(text)
                symbols.add(text.split(".")[-1])
    if not symbols:
        return True, bool(target_function)
    target_text = str(target_function or "")
    bare = target_text.split(".")[-1]
    if "." in target_text:
        receiver = target_text.rsplit(".", 1)[0].split(".")[-1]
        receiver_is_repository_type = receiver in symbols or receiver[:1].isupper()
        return True, bool(receiver_is_repository_type and bare in symbols)
    return True, bool(target_text and (target_text in symbols or bare in symbols))


def _find_entry_by_path(entries: List[Dict[str, Any]], path: str) -> Dict[str, Any] | None:
    for entry in entries:
        if entry.get("path") == path:
            return entry
    return None


def _evidence(label: str, score: float, tier: str) -> Dict[str, Any]:
    return {
        "label": label,
        "score": round(max(0.0, min(float(score), 1.0)), 4),
        "tier": tier,
    }


def _dimension_result(
    score: float,
    applicable: bool,
    evidence: List[Dict[str, Any]],
    missing: List[str],
) -> Dict[str, Any]:
    return {
        "score": round(max(0.0, min(float(score), 1.0)), 4),
        "applicable": bool(applicable),
        "evidence": evidence,
        "missing": missing,
    }


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _split_identifier_tokens(text: str) -> List[str]:
    raw = re.sub(r"(?<!^)(?=[A-Z])", "_", str(text or ""))
    return [
        tok.lower()
        for tok in re.split(r"[^A-Za-z0-9]+|_", raw)
        if len(tok) >= 3
    ]


def _is_non_actionable_setup(text: str) -> bool:
    lower = str(text or "").lower().strip()
    return lower.startswith("consider the following") or lower.startswith("if ")


def _has_actionable_text(items: List[str]) -> bool:
    action_words = (
        "call", "create", "instantiate", "run", "execute", "pass", "set",
        "assert", "import", "construct", "invoke", "render", "request",
        "trigger", "write", "load", "parse", "compare",
    )
    joined = " ".join(str(x).lower() for x in items)
    return any(word in joined for word in action_words)


def _scenario_code_text(scenario: Dict[str, Any]) -> str:
    return _code_blocks_text(scenario.get("reproduction_code", []) or [])


def _code_blocks_text(blocks: Any) -> str:
    parts: List[str] = []
    for block in blocks if isinstance(blocks, list) else _ensure_list(blocks):
        if isinstance(block, dict):
            for key in ("code", "interactive_input", "text"):
                if block.get(key):
                    parts.append(str(block.get(key)))
        else:
            parts.append(str(block))
    return "\n".join(parts)


def _contains_call_pattern(text: str, target_function: str) -> bool:
    if not text or not target_function:
        return False
    bare = re.escape(target_function.split(".")[-1].lower())
    dotted = re.escape(target_function.lower())
    return bool(
        re.search(rf"\b{bare}\s*\(", text)
        or re.search(rf"\b{dotted}\s*\(", text)
        or target_function.lower() in text
    )


def _block_text(block: Any) -> str:
    if isinstance(block, dict):
        return "\n".join(
            str(block.get(key, ""))
            for key in (
                "role",
                "label",
                "context_before",
                "text",
                "interactive_input",
                "code",
                "interactive_output",
            )
            if block.get(key)
        )
    return str(block or "")


def _contains_nested_or_complex_call(text: str) -> bool:
    if not text:
        return False
    call_count = len(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*\(", text))
    return bool(
        call_count >= 2
        or re.search(r"\b[A-Za-z_]\w*\([^()\n]*\b[A-Za-z_]\w*\s*\(", text)
        or re.search(r"\)\s*\.\s*[A-Za-z_]\w+\s*\(", text)
    )


def _issue_has_bug_trigger_evidence(blocks: Any, actual_outputs: List[str]) -> bool:
    classified = classify_reproduction_code_blocks(
        blocks,
        actual_outputs=actual_outputs,
    )
    return any(block_inferred_role(block) == ROLE_BUG_TRIGGER for block in classified)


def _classify_reproduction_roles(
    scenario: Dict[str, Any],
    actual_outputs: List[str],
) -> Dict[str, bool]:
    target = _target_location_mapping(scenario)
    blocks = classify_reproduction_code_blocks(
        scenario.get("reproduction_code", []) or [],
        expected_outputs=scenario.get("expected_outputs", []),
        actual_outputs=scenario.get("actual_outputs", []) or actual_outputs,
        target_function=str(target.get("target_function") or scenario.get("target_function") or ""),
    )
    text = "\n".join(_block_text(block) for block in blocks if block)
    stimulus_text = "\n".join(_ensure_list(scenario.get("execution_stimulus", [])) + _ensure_list(scenario.get("stimulus_steps", [])))
    evidence_text = f"{text}\n{stimulus_text}\n{scenario.get('expected_failure', '')}\n{scenario.get('oracle', '')}"
    lower = evidence_text.lower()
    actual_present = any(
        len(_normalize_value_text(output)) >= 3 and _value_appears(output, evidence_text)
        for output in actual_outputs
    )
    roles = {block_inferred_role(block) for block in blocks}
    return {
        "setup": ROLE_SETUP in roles or bool(re.search(r"\b(?:setup|precondition|fixture|import)\b", lower)),
        "baseline": ROLE_BASELINE in roles,
        "bug_triggering_call": ROLE_BUG_TRIGGER in roles,
        "expected_output": ROLE_EXPECTED_OUTPUT in roles or bool(_ensure_list(scenario.get("expected_outputs", [])) or scenario.get("oracle_expected")),
        "actual_buggy_output": ROLE_ACTUAL_BUGGY_OUTPUT in roles or actual_present or bool(_ensure_list(scenario.get("actual_outputs", []))),
    }


def _scenario_text(value: Any) -> str:
    parts: List[str] = []

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                visit(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                visit(v)
        elif obj is not None:
            parts.append(str(obj))

    visit(value)
    return "\n".join(parts)


_OUTPUT_FIELD_NAMES = {
    "oracle_expected",
    "expected_output",
    "actual_output",
    "expected_outputs",
    "actual_outputs",
    "interactive_output",
}


def _scenario_output_candidates(scenario: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    for key in _OUTPUT_FIELD_NAMES:
        candidates.extend(_ensure_list(scenario.get(key, [])))
    blocks = scenario.get("reproduction_code", [])
    if isinstance(blocks, dict):
        blocks = [blocks]
    for block in blocks if isinstance(blocks, list) else []:
        if not isinstance(block, dict):
            continue
        for key in _OUTPUT_FIELD_NAMES:
            candidates.extend(_ensure_list(block.get(key, [])))
        metadata = " ".join(str(block.get(key, "")) for key in ("role", "type", "label")).lower()
        if re.search(r"\b(?:output|actual|expected|result)\b", metadata):
            candidates.extend(_ensure_list(block.get("code", [])))
            candidates.extend(_ensure_list(block.get("text", [])))
    return _dedup_strings(candidates)


def _oracle_output_candidates(scenario: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    for key in ("oracle_expected", "expected_output", "actual_output", "expected_outputs", "actual_outputs"):
        candidates.extend(_ensure_list(scenario.get(key, [])))
    contract = scenario.get("oracle_contract")
    if isinstance(contract, dict):
        for key in ("expected_output", "actual_output", "oracle_expected"):
            candidates.extend(_ensure_list(contract.get(key, [])))
    return _dedup_strings(candidates)


def _outputs_match_any_candidate(outputs: List[str], candidates: List[str]) -> bool:
    return any(
        strict_normalized_output_equals(candidate, output)
        for output in outputs
        for candidate in candidates
    )


def _value_appears(value: Any, text: str) -> bool:
    return strict_normalized_output_equals(text, value)


def _normalize_value_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _dedup_strings(values: List[Any]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result

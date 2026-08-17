"""Deterministic M7 feedback routing payloads.

This module only packages evidence that M7 already has after its final
verdict. It does not recalculate gates, update admission, call an LLM, or
consume post-patch/M8 evidence.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

from src.contracts.v30 import FeedbackRouteV30


M7_FEEDBACK_SCHEMA_VERSION = "m7-feedback-v1"


def route_feedback_v30(
    *,
    diagnosis: str,
    verdict: str,
    error_category: str = "",
    previous_fingerprint: str = "",
    new_fingerprint: str = "",
) -> dict[str, Any]:
    """Select the earliest causal owner for a v30 feedback pass."""
    text = f"{diagnosis} {error_category}".lower()
    if any(token in text for token in ("localiz", "target", "coverage gap", "wrong source")):
        owner, modules, artifact = "M2", ["M2", "M3"], "localization_hypotheses"
    elif any(token in text for token in ("scenario", "oracle", "trigger", "issue alignment")):
        owner, modules, artifact = "M3", ["M3", "M4"], "validated_scenarios"
    elif any(token in text for token in ("syntax", "import", "fixture", "framework", "collection")):
        owner, modules, artifact = "M5", ["M5", "M5-A"], "generated_test"
    elif any(token in text for token in ("supplemental", "spectra", "sbfl", "insufficient p")):
        owner, modules, artifact = "M6", ["M6"], "supplemental_pass_evidence"
    elif verdict == "ALIGNED":
        owner, modules, artifact = "M7", [], "m7_decision"
    else:
        owner, modules, artifact = "M7", ["M7"], "m7_decision"
    material = bool(new_fingerprint and new_fingerprint != previous_fingerprint)
    route = FeedbackRouteV30(
        diagnosis=diagnosis,
        earliest_causal_owner=owner,
        requested_modules=modules,
        expected_artifact=artifact,
        previous_fingerprint=previous_fingerprint,
        new_fingerprint=new_fingerprint,
        material_change=material,
        no_effect=bool(previous_fingerprint and not material),
        escalation="ESCALATE_UPSTREAM" if previous_fingerprint and not material else "",
    )
    return route.to_dict()

_FORBIDDEN_EVIDENCE_KEYS = {
    "after_patch",
    "after_patch_outcome",
    "checked_coverage",
    "f_to_p",
    "fail_to_pass",
    "golden_patch",
    "golden_patch_lines",
    "m8",
    "m8_evaluation",
    "patch_hit",
    "patch_hit_rate",
    "phr",
    "post_patch",
    "post_patch_outcome",
}


def build_structured_feedback(
    *,
    verdict: str,
    iteration: int,
    bug_fail: float | None,
    coverage: float | None,
    issue_align: float | None,
    thresholds: Mapping[str, float],
    execution_result: Mapping[str, Any] | None = None,
    clue: Mapping[str, Any] | None = None,
    scenario: Mapping[str, Any] | None = None,
    generated_test: Mapping[str, Any] | None = None,
    score_breakdown: Mapping[str, Any] | None = None,
    legacy_feedback: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the canonical structured feedback object for a final M7 verdict.

    Scores and thresholds are copied from the caller's already-computed gate
    evidence. Missing evidence is represented as ``None`` or an ``unavailable``
    status rather than guessed text.
    """
    verdict_text = str(verdict or "")
    execution = execution_result or {}
    clue_data = clue or {}
    scenario_data = scenario or {}
    generated = generated_test or {}
    scores = {
        "s_b": _score_evidence(bug_fail, thresholds.get("s_b")),
        "s_c_prime": _score_evidence(coverage, thresholds.get("s_c_prime")),
        "s_a": _score_evidence(issue_align, thresholds.get("s_a")),
    }
    target = _target_location(scenario_data)
    coverage_evidence = _coverage_evidence(execution, target)
    issue_refs = _issue_references(clue_data, scenario_data, generated)
    avoid_list = _avoid_list_from_history(context, scenario_data, generated)

    payload = _base_payload(
        verdict=verdict_text,
        iteration=iteration,
        scores=scores,
        score_breakdown=score_breakdown or {},
        legacy_feedback=legacy_feedback or {},
        avoid_list=avoid_list,
        evidence_provenance=[
            {"field": "execution_result.test_results", "status": _availability(execution.get("test_results"))},
            {"field": "execution_result.coverage_data", "status": _availability(execution.get("coverage_data"))},
            {"field": "clue.observed_behavior", "status": _availability(clue_data.get("observed_behavior"))},
            {"field": "clue.expected_behavior", "status": _availability(clue_data.get("expected_behavior"))},
            {"field": "scenario.oracle_expected", "status": _availability(scenario_data.get("oracle_expected"))},
            {"field": "generated_test.oracle_expected", "status": _availability(generated.get("oracle_expected"))},
        ],
    )
    if str((context or {}).get("feature_profile") or "") == "v30":
        previous_fp = str((context or {}).get("previous_artifact_fingerprint") or "")
        new_fp = str((context or {}).get("current_artifact_fingerprint") or "")
        payload["v30_causal_route"] = route_feedback_v30(
            diagnosis=str(legacy_feedback.get("diagnosis") or verdict_text),
            verdict=verdict_text,
            error_category=str((score_breakdown or {}).get("failure_type_detail") or ""),
            previous_fingerprint=previous_fp,
            new_fingerprint=new_fp,
        )

    if verdict_text == "ALIGNED":
        payload.update(
            feedback_branch="none",
            target_modules=[],
            diagnosis="aligned; no regeneration requested",
            routing_reason="all final M7 gates passed",
            loop_termination_recommended=True,
        )
        return payload

    if verdict_text == "NOT_FAILED":
        tests = execution.get("test_results") if isinstance(execution.get("test_results"), Mapping) else {}
        failed = [name for name, status in sorted(tests.items()) if status == "FAILED"]
        passed = [name for name, status in sorted(tests.items()) if status == "PASSED"]
        payload.update(
            feedback_branch="M5",
            target_modules=["M5"],
            diagnosis="pre-patch execution did not reproduce the reported bug",
            routing_reason="final verdict is NOT_FAILED from M7 gate evidence",
            threshold_evidence={**payload["threshold_evidence"], "failure_execution": {
                "any_explicit_test_failed": bool(failed),
                "all_generated_tests_passed": bool(tests) and not failed and len(passed) == len(tests),
                "failure_signature_summary": _failure_signature_summary(execution),
            }},
            M5_instructions={
                "diagnose": "Determine why the generated test passed on the buggy pre-patch repository.",
                "oracle_redesign": "Redesign the oracle so it checks EB-based fixed behavior and fails on the observed buggy behavior.",
                "stimulus_redesign": "Redesign the stimulus so it executes the issue's bug-triggering path before asserting.",
            },
            loop_termination_recommended=False,
        )
        return payload

    score_data = score_breakdown if isinstance(score_breakdown, Mapping) else {}
    weighted_coverage = score_data.get("m7_sbfl_weighted_coverage")
    supplemental_collection = (
        weighted_coverage.get("supplemental_pass_collection")
        if isinstance(weighted_coverage, Mapping)
        else None
    )
    if (
        verdict_text == "NO_COVERAGE"
        and (
            score_data.get("failure_type_detail") == "SBFL_UNAVAILABLE_INSUFFICIENT_P"
            or (
                isinstance(supplemental_collection, Mapping)
                and supplemental_collection.get("stop_reason") == "SBFL_UNAVAILABLE_INSUFFICIENT_P"
            )
        )
    ):
        recoverable = iteration < 5
        payload.update(
            feedback_branch=(
                "M2+M3+M5" if recoverable else "M6_SUPPLEMENTAL_PASS_COLLECTION"
            ),
            target_modules=["M2", "M3", "M5"] if recoverable else [],
            diagnosis=(
                "current-pass PASS spectra are insufficient; change the localization target, scenario, and candidate before retrying SBFL"
                if recoverable
                else "bounded pre-patch PASS collection exhausted with fewer than three valid distinct spectra"
            ),
            routing_reason="SBFL_UNAVAILABLE_INSUFFICIENT_P",
            threshold_evidence={**payload["threshold_evidence"], "supplemental_pass_collection": supplemental_collection},
            M2_instructions=(
                {
                    "action": "select another admissible pre-patch target hypothesis",
                    "consume_target_exclusions": True,
                }
                if recoverable else None
            ),
            M3_instructions=(
                {"action": "change scenario preconditions or target-bound stimulus"}
                if recoverable else None
            ),
            M5_instructions=(
                {"action": "generate a materially different candidate and rerun M6"}
                if recoverable else None
            ),
            loop_termination_recommended=not recoverable,
        )
        return payload

    if verdict_text == "NO_COVERAGE":
        gap_functions = coverage_evidence["coverage_gap_functions"]
        payload.update(
            feedback_branch="M2+M5",
            target_modules=["M2", "M5"],
            diagnosis="pre-patch execution did not cover the suspicious target evidence enough",
            routing_reason="final verdict is NO_COVERAGE from M7 gate evidence",
            threshold_evidence={**payload["threshold_evidence"], "coverage": coverage_evidence},
            M2_instructions={
                "action": "re-include/re-evaluate under-covered suspicious functions",
                "requested_functions": gap_functions,
                "note": "Do not mutate R_func here; this is a structured request for M2 review.",
            },
            M5_instructions={
                "action": "add inputs that execute the explicit suspicious functions",
                "coverage_guided_stimulus_request": gap_functions,
            },
            loop_termination_recommended=False,
        )
        return payload

    if verdict_text == "WEAK_ALIGNMENT":
        mismatch = _explicit_mismatch_evidence(issue_refs, legacy_feedback or {}, score_breakdown or {})
        payload.update(
            feedback_branch="M3+M5",
            target_modules=["M3", "M5"],
            diagnosis="the generated test is weakly aligned with issue evidence",
            routing_reason="final verdict is WEAK_ALIGNMENT from M7 gate evidence",
            threshold_evidence={**payload["threshold_evidence"], "issue_alignment": issue_refs},
            M3_instructions={
                "action": "revise scenario against OB and EB",
                "observed_behavior_reference": issue_refs["observed_behavior"],
                "expected_behavior_reference": issue_refs["expected_behavior"],
                "current_misalignment": mismatch,
            },
            M5_instructions={
                "action": "regenerate oracle_expected against EB",
                "current_oracle_expected": issue_refs["current_oracle_expected"],
                "oracle_realignment": "Use EB as the expected assertion specification and do not use OB as oracle_expected.",
            },
            loop_termination_recommended=False,
        )
        return payload

    if verdict_text in {"ERROR", "NOT_RUN"}:
        payload.update(
            feedback_branch="M5-A+M5",
            target_modules=["M5"],
            diagnosis="pre-patch M6 did not produce an admissible complete outcome; repair and rerun the candidate",
            routing_reason="M6 ERROR/NOT_RUN is diagnostic-only and must not terminate the feedback loop",
            threshold_evidence={**payload["threshold_evidence"], "execution_status": verdict_text},
            M5A_instructions={
                "action": "apply deterministic post-processing or bounded repair for the M6 error",
                "preserve": "issue/scenario oracle semantics and candidate identity provenance",
            },
            M5_instructions={
                "action": "regenerate the candidate when M5-A cannot safely repair it",
                "rerun_m6": True,
            },
            loop_termination_recommended=False,
        )
        return payload

    payload.update(
        feedback_branch="unsupported",
        target_modules=[],
        diagnosis="non-canonical M7 verdict; no structured routing branch emitted",
        routing_reason="verdict is not one of ALIGNED, NOT_FAILED, NO_COVERAGE, WEAK_ALIGNMENT",
        loop_termination_recommended=False,
    )
    return payload


def build_iteration_feedback_summary(
    structured_feedback: Mapping[str, Any],
    *,
    iteration: int,
    verdict: str,
    bug_fail: float | None,
    coverage: float | None,
    issue_align: float | None,
) -> dict[str, Any]:
    """Return a compact iteration-history-friendly M7 feedback summary."""
    return {
        "iteration": iteration,
        "verdict": verdict,
        "feedback_branch": structured_feedback.get("feedback_branch"),
        "target_modules": list(structured_feedback.get("target_modules") or []),
        "s_b": bug_fail,
        "s_c_prime": coverage,
        "s_a": issue_align,
    }


def _base_payload(
    *,
    verdict: str,
    iteration: int,
    scores: Mapping[str, Any],
    score_breakdown: Mapping[str, Any],
    legacy_feedback: Mapping[str, Any],
    avoid_list: list[dict[str, Any]],
    evidence_provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": M7_FEEDBACK_SCHEMA_VERSION,
        "feedback_type": verdict,
        "feedback_branch": None,
        "target_modules": [],
        "diagnosis": None,
        "threshold_evidence": {
            "s_b": scores["s_b"],
            "s_c_prime": scores["s_c_prime"],
            "s_a": scores["s_a"],
            "score_definition_status": _safe_mapping(score_breakdown.get("v22_score_definition_status")),
            "coverage_score_available": score_breakdown.get("coverage_score_available"),
            "coverage_score_status": score_breakdown.get("coverage_score_status"),
            "coverage_score_unavailable_reason": score_breakdown.get("coverage_score_unavailable_reason"),
        },
        "routing_reason": None,
        "iteration": iteration,
        "M2_instructions": None,
        "M3_instructions": None,
        "M5_instructions": None,
        "avoid_list": avoid_list,
        "evidence_provenance": evidence_provenance
        + [{"field": "legacy_feedback.repair_directive", "status": _availability(legacy_feedback.get("repair_directive"))}],
        "llm_refinement_requested": False,
        "loop_termination_recommended": False,
    }


def _score_evidence(actual: float | None, threshold: float | None) -> dict[str, Any]:
    return {
        "actual": round(float(actual), 4) if actual is not None else None,
        "threshold": round(float(threshold), 4) if threshold is not None else None,
        "status": "available" if actual is not None and threshold is not None else "unavailable",
    }


def _target_location(scenario: Mapping[str, Any]) -> dict[str, str]:
    target = scenario.get("target_location") if isinstance(scenario.get("target_location"), Mapping) else {}
    return {
        "source_file": str(target.get("source_file") or ""),
        "target_function": str(target.get("target_function") or ""),
    }


def _coverage_evidence(execution: Mapping[str, Any], target: Mapping[str, str]) -> dict[str, Any]:
    coverage_data = execution.get("coverage_data")
    contributing = execution.get("contributing_functions")
    has_coverage_data = isinstance(coverage_data, Mapping) and bool(coverage_data)
    function = target.get("target_function") or ""
    source = target.get("source_file") or ""
    covered_functions = _covered_function_names(contributing)
    target_covered = bool(function and any(name.endswith(function) or name == function for name in covered_functions))
    source_coverage = _target_source_coverage(coverage_data, source) if has_coverage_data else None
    gap_functions: list[str] = []
    if has_coverage_data and function and (source_coverage is not None or covered_functions) and not target_covered:
        gap_functions.append(f"{source}:{function}" if source else function)
    return {
        "s_c_prime_available": None,
        "coverage_data_available": has_coverage_data,
        "cumulative_F_empty": _is_cumulative_f_empty(execution),
        "covered_functions": covered_functions,
        "explicit_suspicious_functions": gap_functions if has_coverage_data else [],
        "coverage_gap_functions": gap_functions,
        "target_source_coverage": source_coverage,
    }


def _covered_function_names(contributing: Any) -> list[str]:
    values: list[str] = []
    if isinstance(contributing, Mapping):
        for source, funcs in sorted(contributing.items()):
            if isinstance(funcs, list):
                values.extend(f"{source}:{fn}" for fn in funcs if str(fn).strip())
            elif funcs:
                values.append(f"{source}:{funcs}")
    elif isinstance(contributing, list):
        values.extend(str(item) for item in contributing if str(item).strip())
    return _dedupe(values)


def _target_source_coverage(coverage_data: Any, source_file: str) -> float | None:
    if not isinstance(coverage_data, Mapping) or not source_file:
        return None
    for path, info in sorted(coverage_data.items()):
        if not isinstance(info, Mapping):
            continue
        path_text = str(path).replace("\\", "/")
        source_text = source_file.replace("\\", "/")
        if path_text.endswith(source_text) or source_text.endswith(path_text) or path_text.split("/")[-1] == source_text.split("/")[-1]:
            try:
                return round(float(info.get("cover", 0.0)) / 100.0, 4)
            except (TypeError, ValueError):
                return None
    return None


def _is_cumulative_f_empty(execution: Mapping[str, Any]) -> bool | None:
    for key in ("F_set", "stable_F_set", "failing_tests"):
        value = execution.get(key)
        if isinstance(value, list):
            return len(value) == 0
    metadata = execution.get("metadata")
    if isinstance(metadata, Mapping):
        core = metadata.get("m6_core_execution_data")
        if isinstance(core, Mapping):
            for key in ("F_set", "stable_F_set"):
                value = core.get(key)
                if isinstance(value, list):
                    return len(value) == 0
    return None


def _issue_references(
    clue: Mapping[str, Any],
    scenario: Mapping[str, Any],
    generated_test: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "observed_behavior": _evidence_value(clue.get("observed_behavior")),
        "expected_behavior": _evidence_value(clue.get("expected_behavior")),
        "current_oracle_expected": _evidence_value(
            generated_test.get("oracle_expected")
            if generated_test.get("oracle_expected") is not None
            else scenario.get("oracle_expected")
        ),
    }


def _explicit_mismatch_evidence(
    issue_refs: Mapping[str, Any],
    legacy_feedback: Mapping[str, Any],
    score_breakdown: Mapping[str, Any],
) -> dict[str, Any]:
    reasons = []
    for key in ("failure_type_detail", "conservative_gate_reasons", "gate_warnings"):
        value = score_breakdown.get(key)
        if value:
            reasons.append({"source": f"score_breakdown.{key}", "value": _safe_value(value)})
    directive = legacy_feedback.get("repair_directive")
    if isinstance(directive, Mapping) and directive.get("blocking_reason"):
        reasons.append({"source": "legacy_feedback.repair_directive.blocking_reason", "value": str(directive["blocking_reason"])})
    return {
        "status": "available" if reasons else "unavailable",
        "items": reasons,
        "observed_behavior_available": issue_refs["observed_behavior"]["status"] == "available",
        "expected_behavior_available": issue_refs["expected_behavior"]["status"] == "available",
    }


def _failure_signature_summary(execution: Mapping[str, Any]) -> dict[str, Any]:
    raw_output = str(execution.get("raw_output") or "")
    failed = re.findall(r"FAILED\s+([^\s]+)", raw_output)
    assertion = bool(re.search(r"AssertionError|\bassert\b|expected|actual", raw_output, re.IGNORECASE))
    exception_match = re.search(r"\b([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))(?::\s*([^\n]+))?", raw_output)
    return {
        "status": "available" if raw_output else "unavailable",
        "failed_tests": _dedupe(failed[:5]),
        "assertion_failure_observed": assertion,
        "exception_type": exception_match.group(1).split(".")[-1] if exception_match else None,
        "exception_message": exception_match.group(2)[:200] if exception_match and exception_match.group(2) else None,
    }


def _avoid_list_from_history(
    context: Mapping[str, Any] | None,
    scenario: Mapping[str, Any],
    generated_test: Mapping[str, Any],
) -> list[dict[str, Any]]:
    histories: list[Any] = []
    for source in (context or {}, scenario, generated_test):
        for key in ("iteration_history", "previous_iterations", "previous_attempts", "prior_attempts"):
            value = source.get(key) if isinstance(source, Mapping) else None
            if isinstance(value, list):
                histories.extend(value)
    items: list[dict[str, Any]] = []
    for index, entry in enumerate(histories):
        if not isinstance(entry, Mapping):
            continue
        iteration = entry.get("iteration")
        source = f"history[{index}]"
        for key, pattern_type in (
            ("assertion_patterns", "assertion"),
            ("stimulus_patterns", "stimulus"),
            ("forbidden_patterns", "prior_forbidden"),
        ):
            patterns = entry.get(key)
            if isinstance(patterns, list):
                for pattern in patterns:
                    _append_avoid(items, pattern_type, pattern, iteration, f"{source}.{key}")
        code = entry.get("test_code") or entry.get("append_block")
        if isinstance(code, str) and code.strip():
            for assertion in _assertion_lines(code):
                _append_avoid(items, "assertion", assertion, iteration, f"{source}.test_code")
            for stimulus in _call_lines(code):
                _append_avoid(items, "stimulus", stimulus, iteration, f"{source}.test_code")
    return items


def _append_avoid(
    items: list[dict[str, Any]],
    pattern_type: str,
    pattern: Any,
    iteration: Any,
    evidence_key: str,
) -> None:
    text = str(pattern or "").strip()
    if not text:
        return
    item = {
        "pattern_type": pattern_type,
        "pattern": text[:240],
        "source_iteration": iteration if isinstance(iteration, int) else None,
        "evidence_key": evidence_key,
    }
    key = (item["pattern_type"], item["pattern"], item["source_iteration"], item["evidence_key"])
    existing = {
        (entry["pattern_type"], entry["pattern"], entry["source_iteration"], entry["evidence_key"])
        for entry in items
    }
    if key not in existing:
        items.append(item)


def _assertion_lines(code: str) -> list[str]:
    return _dedupe(
        line.strip()
        for line in code.splitlines()
        if line.strip().startswith(("assert ", "self.assert", "np.testing.assert"))
    )


def _call_lines(code: str) -> list[str]:
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if re.search(r"\b[A-Za-z_][A-Za-z0-9_.]*\s*\(", stripped) and not stripped.startswith(("def ", "class ", "assert ", "self.assert")):
            lines.append(stripped)
    return _dedupe(lines)


def _evidence_value(value: Any) -> dict[str, Any]:
    return {"status": _availability(value), "value": _safe_value(value) if _availability(value) == "available" else None}


def _availability(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, (str, list, dict, tuple, set)) and not value:
        return "unavailable"
    return "available"


def _safe_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _safe_value(item) for key, item in sorted(value.items())}


def _safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _safe_value(item)
            for key, item in sorted(value.items())
            if str(key).lower() not in _FORBIDDEN_EVIDENCE_KEYS
        }
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        norm = re.sub(r"\s+", " ", text.lower())
        if not text or norm in seen:
            continue
        seen.add(norm)
        result.append(text)
    return result

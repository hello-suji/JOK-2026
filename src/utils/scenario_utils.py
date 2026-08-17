from __future__ import annotations

import re
from typing import Any, Dict, Iterable

from src.scenario.scenario_hydrator import synthesize_oracle_contract


_NOISY_FUNCTIONS = {
    "arange", "rand", "random", "seed", "platform", "get_backend",
    "show_versions", "main", "run", "get", "set",
}


def ensure_primary_scenario(
    validation_report: Dict[str, Any],
    clue: Dict[str, Any] | None = None,
    context: Dict[str, Any] | None = None,
    reason: str = "deterministic_repair",
) -> Dict[str, Any]:
    """Ensure validation_report has one selected normalized scenario."""
    report = validation_report if isinstance(validation_report, dict) else {}
    selected_raw = report.setdefault("selected_scenarios", [])
    selected = [item for item in selected_raw if isinstance(item, dict)] if isinstance(selected_raw, list) else []
    if selected != selected_raw:
        report["selected_scenarios"] = selected
    if selected and isinstance(selected[0].get("normalized_scenario"), dict):
        return report

    repaired = _build_repaired_scenario(clue or {}, context or {}, reason=reason)
    selected.insert(0, {
        "scenario_id": repaired["scenario_id"],
        "score": 0.3,
        "decision": "pending_revalidation",
        "validation_status": "pending_revalidation",
        "diagnostic_only": True,
        "reasons": [f"repaired scenario requires M4 revalidation: {reason}"],
        "normalized_scenario": repaired,
        "force_selected": True,
        "scenario_repaired": True,
        "scenario_repair_reason": reason,
        "scenario_required_fields_filled": [
            "reproduction_code",
            "expected_outputs",
            "actual_outputs",
            "identifiers",
            "target_location",
            "oracle_contract",
        ],
    })
    report["selected_scenarios"] = selected
    return report


def select_primary_scenario(
    validation_report: Dict[str, Any],
    clue: Dict[str, Any] | None = None,
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """validation_report에서 현재 사용 중인 primary 시나리오를 가져온다.

    candidate_test_file이 있는 시나리오를 우선하고, 없으면 첫 번째 선택 시나리오를 반환한다.

    Raises:
        ValueError: selected_scenarios가 비어 있거나 normalized_scenario가 없으면 발생.
    """
    selected_raw = validation_report.get("selected_scenarios", [])
    selected = (
        [item for item in selected_raw if isinstance(item, dict)]
        if isinstance(selected_raw, list)
        else []
    )
    if selected != selected_raw:
        validation_report["selected_scenarios"] = selected
    if not selected:
        validation_report = ensure_primary_scenario(
            validation_report,
            clue=clue,
            context=context,
            reason="primary_selection_empty",
        )
        selected = validation_report.get("selected_scenarios", [])

    for item in selected:
        normalized = item.get("normalized_scenario")
        if not isinstance(normalized, dict):
            continue
        target = normalized.get("target_location")
        if isinstance(target, dict) and target.get("candidate_test_file"):
            return normalized

    # 첫 번째 선택 시나리오의 normalized_scenario 찾기
    first_normalized = selected[0].get("normalized_scenario")
    if not isinstance(first_normalized, dict):
        validation_report = ensure_primary_scenario(
            validation_report,
            clue=clue,
            context=context,
            reason="primary_selection_missing_normalized_scenario",
        )
        selected = validation_report.get("selected_scenarios", [])
        return selected[0]["normalized_scenario"]
    return first_normalized


def _build_repaired_scenario(
    clue: Dict[str, Any],
    context: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    identifiers = clue.get("identifiers", {}) if isinstance(clue.get("identifiers"), dict) else {}
    functions = identifiers.get("functions", []) or []
    classes = identifiers.get("classes", []) or []
    source_files = [
        x.get("path", "")
        for x in context.get("candidate_source_files", [])
        if isinstance(x, dict) and x.get("path")
    ]
    test_files = [
        x.get("path", "")
        for x in context.get("candidate_test_files", [])
        if isinstance(x, dict) and x.get("path")
    ]
    runner = (context.get("project_test_style") or {}).get("runner", "pytest")
    expected_outputs = clue.get("expected_outputs", []) or []
    expected_behavior = clue.get("expected_behavior", []) or []
    actual = clue.get("actual_outputs", []) or []
    oracle_contract = synthesize_oracle_contract(
        clue,
        {
            "expected_outputs": expected_outputs,
            "expected_behavior": expected_behavior,
            "actual_outputs": actual,
            "expected_failure": (
                str((clue.get("observed_behavior") or [""])[0])
            ),
        },
        repo=str(context.get("repo") or ""),
        context=context,
    )
    oracle_type = oracle_contract["oracle_type"]
    oracle_source = oracle_contract["oracle_source"]
    rule = oracle_contract["rule"]

    target_function = _canonicalize_target_function(
        _choose_target_function(clue, context, functions),
        context,
    )
    target_source_file = _source_file_for_target(
        clue,
        context,
        target_function,
        default=source_files[0] if source_files else "",
    )
    target_provenance = _target_provenance_type(
        clue, context, target_function, target_source_file, functions
    )
    candidate_invocation_expression = _candidate_invocation_expression(
        clue,
        target_function,
    )
    issue_api_target = candidate_invocation_expression or (
        target_function
        if target_provenance == "m1_issue_identifier_repository_bound"
        and not target_function.split(".")[-1].startswith("_")
        else ""
    )
    return {
        "scenario_id": "S_REPAIRED",
        "instance_id": str(context.get("instance_id") or clue.get("instance_id") or ""),
        "iteration": context.get("iteration") or context.get("outer_iteration"),
        "outer_iteration": context.get("outer_iteration") or context.get("iteration"),
        "target_location": {
            "source_file": target_source_file,
            "target_function": target_function,
            "canonical_target_identity": target_function,
            "candidate_invocation_expression": candidate_invocation_expression,
            "related_classes": classes[:3],
            "candidate_test_file": test_files[0] if test_files else "",
            "confidence": "low",
            "target_provenance": target_provenance,
            "issue_api_target": issue_api_target,
            "implementation_target": (
                target_function
                if target_provenance in {
                    "m2_pre_patch_ranked_evidence",
                    "repository_traceback_coherent",
                }
                or target_function.split(".")[-1].startswith("_")
                else ""
            ),
            "target_repair_provenance": {
                "schema_version": "v31-repaired-target-provenance-v1",
                "instance_id": str(context.get("instance_id") or clue.get("instance_id") or ""),
                "iteration": context.get("iteration") or context.get("outer_iteration"),
                "replacement_reason": reason,
                "rejected_targets": list(
                    ((context.get("metadata") or {}).get("restart_constraints") or {}).get(
                        "prohibited_targets", []
                    )
                )[-5:],
                "replacement_candidate": {
                    "source_file": target_source_file,
                    "target_function": target_function,
                },
                "evidence_type": target_provenance,
            },
        },
        "setup_steps": ["Set up the issue reproduction using existing project test helpers."],
        "execution_stimulus": [
            f"Call {target_function} with the issue reproduction conditions."
            if target_function else "Execute the issue reproduction code."
        ],
        "expected_failure": (
            str((clue.get("observed_behavior") or ["Buggy behavior should be reproduced."])[0])
        ),
        "relevant_source_files": source_files[:3],
        "relevant_test_files": test_files[:3],
        "test_environment": {"required_fixtures": [], "runner": runner},
        "reproduction_code": clue.get("code_examples", []) or [],
        "expected_outputs": expected_outputs,
        "expected_behavior": expected_behavior,
        "actual_outputs": actual,
        "error_keywords": clue.get("error_keywords", []) or [],
        "identifiers": identifiers,
        "oracle_contract": oracle_contract,
        "oracle_type": oracle_type,
        "oracle_source": oracle_source,
        "oracle_hints": [rule],
        "oracle": rule,
        "scenario_repaired": True,
        "scenario_repair_reason": reason,
        "generation_provenance": "issue_grounded_deterministic_repair",
        "canonical_target_identity": target_function,
        "candidate_invocation_expression": candidate_invocation_expression,
    }


def _candidate_invocation_expression(
    clue: Dict[str, Any],
    canonical_target_identity: str,
) -> str:
    """Return issue-spelled receiver syntax without promoting it to identity."""
    simple = str(canonical_target_identity or "").split(".")[-1]
    if not simple:
        return ""
    candidates: list[str] = []
    for block in clue.get("code_examples", []) or []:
        code = str(block.get("code") if isinstance(block, dict) else block or "")
        for match in re.finditer(
            rf"\b((?:[A-Za-z_]\w*\.)+{re.escape(simple)})\s*\(",
            code,
        ):
            expression = match.group(1)
            if expression not in candidates:
                candidates.append(expression)
    return candidates[0] if candidates else ""


def _canonicalize_target_function(value: str, context: Dict[str, Any]) -> str:
    """Prefer an M2 repository-qualified symbol over local receiver syntax."""
    simple = str(value or "").split(".")[-1]
    if not simple:
        return ""
    for entry in context.get("function_ranking", []) or []:
        if not isinstance(entry, dict):
            continue
        function = str(entry.get("qualified_name") or entry.get("function_name") or "")
        if function and function.split(".")[-1] == simple:
            return function
    return simple if "." in str(value or "") else str(value or "")


def _choose_target_function(
    clue: Dict[str, Any],
    context: Dict[str, Any],
    functions: Iterable[str],
) -> str:
    """Pick a concrete target function without inventing one."""
    restart_constraints = (
        (context.get("metadata") or {}).get("restart_constraints") or {}
        if isinstance(context.get("metadata"), dict)
        else {}
    )
    prohibited = {
        str(item.get("target_function") or item.get("function_name") or "")
        .split(".")[-1]
        .lower()
        for item in restart_constraints.get("prohibited_targets", []) or []
        if isinstance(item, dict)
    }

    def allowed(value: Any) -> bool:
        text = str(value or "")
        bare = text.split(".")[-1]
        return bool(
            text
            and all(re.fullmatch(r"[A-Za-z_]\w*", part) for part in text.split("."))
            and bare.lower() not in _NOISY_FUNCTIONS
            and bare.lower() not in prohibited
            and not (bare.startswith("__") and bare.endswith("__"))
        )

    function_list = [str(f) for f in functions if allowed(f)]
    observed_text = " ".join(
        str(value)
        for key in ("observed_behavior", "actual_outputs", "error_keywords")
        for value in (
            clue.get(key, [])
            if isinstance(clue.get(key), list)
            else [clue.get(key, "")]
        )
    ).lower()
    for fn in function_list:
        if fn.split(".")[-1].lower() in observed_text and _source_file_for_target(
            clue, context, fn, default=""
        ):
            return fn
    for fn in function_list:
        if _source_file_for_target(clue, context, fn, default=""):
            return fn
    for fault in clue.get("fault_locations", []) or []:
        if not isinstance(fault, dict):
            continue
        fn = str(fault.get("function_name") or "")
        fault_path = str(fault.get("file_path") or fault.get("source_file") or "")
        bound_path = _source_file_for_target(clue, context, fn, default="")
        if allowed(fn) and _same_repository_path(fault_path, bound_path):
            return fn
    if function_list:
        return function_list[0]
    for source in context.get("candidate_source_files", []) or []:
        for key in ("top_level_functions", "functions", "matched_identifiers"):
            values = source.get(key) or []
            if isinstance(values, dict):
                values = values.get("functions") or []
            for fn in values:
                if isinstance(fn, str) and allowed(fn):
                    return fn
    return ""


def _same_repository_path(left: str, right: str) -> bool:
    left_norm = str(left or "").replace("\\", "/").lstrip("./")
    right_norm = str(right or "").replace("\\", "/").lstrip("./")
    return bool(
        left_norm
        and right_norm
        and (
            left_norm == right_norm
            or left_norm.endswith("/" + right_norm)
            or right_norm.endswith("/" + left_norm)
        )
    )


def _target_provenance_type(
    clue: Dict[str, Any],
    context: Dict[str, Any],
    target_function: str,
    target_source_file: str,
    issue_functions: Iterable[str],
) -> str:
    if not target_function:
        return "unresolved"
    if target_function in [str(item) for item in issue_functions]:
        return "m1_issue_identifier_repository_bound"
    for fault in clue.get("fault_locations", []) or []:
        if not isinstance(fault, dict):
            continue
        if str(fault.get("function_name") or "") != target_function:
            continue
        fault_path = str(fault.get("file_path") or fault.get("source_file") or "")
        if _same_repository_path(fault_path, target_source_file):
            return "repository_traceback_coherent"
    return "m2_pre_patch_ranked_evidence"


def _source_file_for_target(
    clue: Dict[str, Any],
    context: Dict[str, Any],
    target_function: str,
    *,
    default: str,
) -> str:
    """Bind a repaired target to pre-patch evidence for the same callable."""
    if not target_function:
        return default
    bare = target_function.split(".")[-1]
    for entry in context.get("function_ranking", []) or []:
        if not isinstance(entry, dict):
            continue
        function = str(entry.get("function_name") or entry.get("qualified_name") or "")
        if function and function.split(".")[-1] == bare:
            path = str(entry.get("source_file") or entry.get("file_path") or "")
            if path:
                return path
    for fault in clue.get("fault_locations", []) or []:
        if not isinstance(fault, dict):
            continue
        function = str(fault.get("function_name") or "")
        if function and function.split(".")[-1] == bare:
            path = str(fault.get("file_path") or fault.get("source_file") or "")
            if path:
                return path
    for source in context.get("candidate_source_files", []) or []:
        if not isinstance(source, dict):
            continue
        values: list[Any] = []
        for key in ("top_level_functions", "functions", "matched_identifiers"):
            raw = source.get(key) or []
            if isinstance(raw, dict):
                raw = raw.get("functions") or []
            values.extend(raw if isinstance(raw, list) else [])
        if any(str(item).split(".")[-1] == bare for item in values):
            path = str(source.get("path") or source.get("source_file") or "")
            if path:
                return path
    return default

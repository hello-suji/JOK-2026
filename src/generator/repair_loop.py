from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.generator.m5a_llm_error_refinement import ErrorRefinementRequest
from src.utils.artifact_hash import sha256_text
from src.utils.file_io import write_json_atomic


REPAIRABLE_ERROR_CATEGORIES = {
    "SYNTAX_ERROR",
    "IMPORT_ERROR",
    "MODULE_NOT_FOUND",
    "MISSING_SYMBOL",
    "COLLECTION_NODEID",
    "MISSING_FIXTURE",
    "TEST_FRAMEWORK_STRUCTURE",
    "SIGNATURE_MISMATCH",
    "TEST_CONSTRUCTION_TYPE_ERROR",
    "TEST_CONSTRUCTION_ATTRIBUTE_ERROR",
    "TEST_CONSTRUCTION_KEY_ERROR",
    "ORACLE_REJECTED",
}

V36_COMPILATION_ERROR_CATEGORIES = {
    "SYNTAX_ERROR",
    "IMPORT_ERROR",
    "MODULE_NOT_FOUND",
    "MISSING_SYMBOL",
}

M5A_TELEMETRY_KEY = "enable_m5a_llm_error_refinement"


@dataclass
class RepairEvidence:
    error_category: str
    error_text: str
    pytest_command: str = ""
    target_test_file: str = ""
    target_nodeid: str = ""
    repository_commit: str = ""
    import_evidence: dict[str, Any] = field(default_factory=dict)
    signature_evidence: dict[str, Any] = field(default_factory=dict)
    nearby_test_examples: list[str] = field(default_factory=list)
    failed_line: str = ""
    nearby_source: str = ""
    exception_type: str = ""
    line_number: int | None = None
    offset: int | None = None
    offending_fragment: str = ""
    rejected_import: str = ""
    rejected_symbol: str = ""
    rejected_oracle_pattern: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_repairable_error(error_text: str) -> str:
    text = str(error_text or "")
    lower = text.lower()
    if "syntaxerror" in lower or "invalid syntax" in lower:
        return "SYNTAX_ERROR"
    if "semantic_risk" in lower or "semantic risk" in lower:
        return "SEMANTIC_RISK"
    if "oracle" in lower or "assertion" in lower and "post-fix behavior" in lower:
        return "ORACLE_REJECTED"
    if "modulenotfounderror" in lower or "no module named" in lower:
        return "MODULE_NOT_FOUND"
    if "importerror" in lower or "cannot import name" in lower or "invalid import" in lower:
        return "IMPORT_ERROR"
    if "nameerror" in lower and "is not defined" in lower:
        return "MISSING_SYMBOL"
    if "fixture" in lower and ("not found" in lower or "not available" in lower):
        return "MISSING_FIXTURE"
    if (
        "appregistrynotready" in lower
        or "apps aren't loaded yet" in lower
        or "apps are not loaded yet" in lower
    ):
        return "TEST_FRAMEWORK_STRUCTURE"
    if (
        "test functions/methods" in lower
        or "exactly one focused reproduction test" in lower
        or "multiple generated test" in lower
    ):
        return "TEST_FRAMEWORK_STRUCTURE"
    if "not found:" in lower or "not collected" in lower or "collected 0 items" in lower:
        return "COLLECTION_NODEID"
    if "missing 1 required positional argument" in lower or "signature" in lower:
        return "SIGNATURE_MISMATCH"
    if "django-test runner requires" in lower or "testcase" in lower and "requires" in lower:
        return "TEST_FRAMEWORK_STRUCTURE"
    if "typeerror" in lower:
        return "TEST_CONSTRUCTION_TYPE_ERROR"
    if "attributeerror" in lower:
        return "TEST_CONSTRUCTION_ATTRIBUTE_ERROR"
    if "keyerror" in lower:
        return "TEST_CONSTRUCTION_KEY_ERROR"
    return "INFRASTRUCTURE_FAILURE"


def is_repairable_category(
    category: str,
    *,
    v30: bool = False,
    v36_compile_only: bool = False,
) -> bool:
    """Return whether M5-A may see this failure under the selected method."""
    if v36_compile_only:
        return str(category or "") in V36_COMPILATION_ERROR_CATEGORIES
    # v30 treats oracle/semantic failures as upstream methodology defects;
    # legacy profiles retain the historical compatibility predicate.
    if v30 and str(category or "") in {"ORACLE_REJECTED", "SEMANTIC_RISK"}:
        return False
    return str(category or "") in REPAIRABLE_ERROR_CATEGORIES


def normalized_error_fingerprint(error_text: str) -> str:
    text = re.sub(r"\s+", " ", str(error_text or "")).strip().lower()
    text = re.sub(r"0x[0-9a-f]+", "0xADDR", text)
    text = re.sub(r"\b\d+\b", "N", text)
    return sha256_text(text[:500])


def repair_fingerprint(
    *,
    candidate_code: str,
    error_text: str,
    target_nodeid: str,
    repository_commit: str,
    repair_strategy: str,
) -> str:
    payload = {
        "candidate_sha256": sha256_text(candidate_code or ""),
        "error_fingerprint": normalized_error_fingerprint(error_text),
        "target_nodeid": str(target_nodeid or ""),
        "repository_commit": str(repository_commit or ""),
        "repair_strategy": str(repair_strategy or ""),
    }
    return sha256_text(json.dumps(payload, sort_keys=True))


def extract_generated_code(parsed_or_generated: Mapping[str, Any] | None) -> str:
    if not isinstance(parsed_or_generated, Mapping):
        return ""
    return str(
        parsed_or_generated.get("append_block")
        or parsed_or_generated.get("test_code")
        or ""
    )


def build_repair_evidence(
    *,
    error_text: str,
    context: Mapping[str, Any],
    clue: Mapping[str, Any],
    scenario: Mapping[str, Any],
    target_test_file: str = "",
    target_nodeid: str = "",
    repository_commit: str = "",
    pytest_command: str = "",
    raw_output: str = "",
) -> RepairEvidence:
    combined_error = "\n".join(x for x in [error_text, raw_output] if x)
    category = classify_repairable_error(combined_error)
    repo_path = Path(str(context.get("repo_path") or ""))
    target_file = target_test_file or _scenario_target_test_file(scenario)
    return RepairEvidence(
        error_category=category,
        error_text=combined_error,
        pytest_command=pytest_command,
        target_test_file=target_file,
        target_nodeid=target_nodeid,
        repository_commit=repository_commit,
        import_evidence=_collect_import_evidence(context, combined_error),
        signature_evidence=_collect_signature_evidence(context, combined_error),
        nearby_test_examples=_collect_nearby_tests(repo_path, target_file),
        failed_line=_extract_failed_line(combined_error),
        nearby_source=_extract_nearby_source(repo_path, target_file, combined_error),
        **_structured_failure_details(combined_error),
    )


def build_error_refinement_request(
    *,
    test_code: str,
    evidence: RepairEvidence,
    clue: Mapping[str, Any],
    context: Mapping[str, Any],
    scenario: Mapping[str, Any],
    v36_compile_only: bool = False,
) -> ErrorRefinementRequest:
    oracle_requirements = []
    oracle_contract = scenario.get("oracle_contract") if isinstance(scenario, Mapping) else {}
    if isinstance(oracle_contract, Mapping):
        oracle_requirements.extend(
            str(oracle_contract.get(key) or "")
            for key in ("oracle_type", "oracle_source", "rule")
            if oracle_contract.get(key)
        )
    if scenario.get("oracle"):
        oracle_requirements.append(str(scenario.get("oracle")))
    terms = []
    identifiers = scenario.get("identifiers") or clue.get("identifiers") or {}
    if isinstance(identifiers, Mapping):
        for key in ("functions", "classes"):
            values = identifiers.get(key)
            if isinstance(values, list):
                terms.extend(str(value) for value in values[:6])
    return ErrorRefinementRequest(
        test_code=test_code,
        error_message=evidence.error_text,
        failed_line=evidence.failed_line,
        nearby_source=evidence.nearby_source,
        scenario=scenario,
        oracle_intent="\n".join(oracle_requirements),
        target_behavior_terms=tuple(term for term in terms if term),
        pytest_command=evidence.pytest_command,
        observed_behavior=clue.get("observed_behavior", []),
        expected_behavior=clue.get("expected_behavior", []),
        repository_commit=evidence.repository_commit,
        target_test_file=evidence.target_test_file,
        target_nodeid=evidence.target_nodeid,
        import_evidence=evidence.import_evidence,
        signature_evidence=evidence.signature_evidence,
        nearby_test_examples=evidence.nearby_test_examples,
        oracle_requirements=oracle_requirements,
        issue_evidence={
            "OB": clue.get("OB") or clue.get("observed_behavior") or clue.get("actual_output"),
            "EB": clue.get("EB") or clue.get("expected_behavior") or clue.get("expected_output"),
            "S2R": clue.get("S2R") or clue.get("steps_to_reproduce") or clue.get("repro_conditions"),
        },
        m2_context_evidence=_bounded_m2_context(context),
        oracle_hint=str(context.get("oracle_hint") or ""),
        current_m7_diagnosis=_current_m7_diagnosis(scenario),
        prior_avoid_patterns=_prior_avoid_patterns(scenario),
        compile_only=v36_compile_only,
    )


def persist_quarantine_artifacts(
    artifact_dir: Path,
    *,
    raw_candidate: str,
    validation_failures: list[str],
    evidence: RepairEvidence,
    parsed_candidate: Mapping[str, Any] | None = None,
    attempt_history: list[Mapping[str, Any]] | None = None,
    raw_response: str = "",
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "candidate_raw.py").write_text(raw_candidate or "", encoding="utf-8")
    (artifact_dir / "raw_model_response.txt").write_text(
        raw_response or "",
        encoding="utf-8",
    )
    write_json_atomic(
        {
            "schema_version": "v22-invalid-candidate-quarantine-v1",
            "candidate_status": "QUARANTINED_INVALID",
            "diagnostic_only": True,
            "candidate_sha256": sha256_text(raw_candidate or ""),
            "raw_response_sha256": sha256_text(raw_response or ""),
            "raw_response_artifact": "raw_model_response.txt",
            "raw_response_available": bool(raw_response),
            "validation_failures": list(validation_failures or []),
            "attempt_history": [dict(item) for item in (attempt_history or [])],
            "repair_evidence": evidence.to_dict(),
            "parsed_candidate": dict(parsed_candidate or {}),
        },
        artifact_dir / "candidate_validation.json",
    )


def persist_repair_attempt(
    artifact_dir: Path,
    *,
    attempt_index: int,
    strategy: str,
    input_request: ErrorRefinementRequest,
    result: Mapping[str, Any],
    repaired_code: str,
    validation_status: Mapping[str, Any],
    duplicate: bool = False,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    input_payload = {
        "schema_version": "v22-m5a-repair-input-v1",
        "attempt_index": attempt_index,
        "strategy": strategy,
        "request": asdict(input_request),
    }
    write_json_atomic(input_payload, artifact_dir / f"repair_input_{attempt_index}.json")
    suffix = "llm" if strategy == "llm" else "rule"
    (artifact_dir / f"candidate_repaired_{suffix}.py").write_text(repaired_code or "", encoding="utf-8")
    attempts_path = artifact_dir / "repair_attempts.json"
    attempts = []
    if attempts_path.exists():
        try:
            loaded = json.loads(attempts_path.read_text(encoding="utf-8"))
            attempts = list(loaded.get("attempts") or [])
        except (json.JSONDecodeError, OSError):
            attempts = []
    attempts.append(
        {
            "attempt_index": attempt_index,
            "strategy": strategy,
            "duplicate_blocked": duplicate,
            "result": dict(result),
            "repaired_candidate_sha256": sha256_text(repaired_code or ""),
            "validation_status": dict(validation_status),
        }
    )
    write_json_atomic(
        {
            "schema_version": "v22-m5a-repair-attempts-v1",
            "attempts": attempts,
        },
        attempts_path,
    )


def make_m5a_telemetry(
    *,
    enabled: bool,
    eligible: bool,
    triggered: bool,
    trigger_reason: str,
    attempt_count: int,
    input_error_category: str,
    input_error_fingerprint: str,
    original_candidate_sha256: str,
    repaired_candidate_sha256: str = "",
    repair_result: str,
    post_repair_syntax_status: str = "NOT_RUN",
    post_repair_import_status: str = "NOT_RUN",
    post_repair_oracle_status: str = "NOT_RUN",
    post_repair_collection_status: str = "NOT_RUN",
    post_repair_execution_status: str = "NOT_RUN",
    terminal_reason: str = "",
    raw_response_artifact: str = "",
    response_empty: bool = False,
    parse_attempt_count: int = 0,
    parse_error: str = "",
    retry_triggered: bool = False,
    retry_prompt_hash: str = "",
    final_parse_status: str = "",
    post_repair_validation_errors: list[str] | None = None,
    post_repair_failure_fingerprint: str = "",
    no_effect_repair: bool = False,
) -> dict[str, Any]:
    status = repair_result
    return {
        M5A_TELEMETRY_KEY: {
            "enabled": enabled,
            "eligible": eligible,
            "triggered": triggered,
            "trigger_reason": trigger_reason,
            "attempt_count": attempt_count,
            "input_error_category": input_error_category,
            "input_error_fingerprint": input_error_fingerprint,
            "original_candidate_sha256": original_candidate_sha256,
            "repaired_candidate_sha256": repaired_candidate_sha256,
            "repair_result": repair_result,
            "post_repair_syntax_status": post_repair_syntax_status,
            "post_repair_import_status": post_repair_import_status,
            "post_repair_oracle_status": post_repair_oracle_status,
            "post_repair_collection_status": post_repair_collection_status,
            "post_repair_execution_status": post_repair_execution_status,
            "terminal_reason": terminal_reason,
            "raw_response_artifact": raw_response_artifact,
            "response_empty": response_empty,
            "parse_attempt_count": parse_attempt_count,
            "parse_error": parse_error,
            "retry_triggered": retry_triggered,
            "retry_prompt_hash": retry_prompt_hash,
            "final_parse_status": final_parse_status,
            "post_repair_validation_errors": list(post_repair_validation_errors or []),
            "post_repair_failure_fingerprint": post_repair_failure_fingerprint,
            "no_effect_repair": no_effect_repair,
            "status": status,
        }
    }


def validation_status_from_errors(errors: list[str]) -> dict[str, str]:
    joined = "\n".join(errors or [])
    category = classify_repairable_error(joined)
    return {
        "syntax": "FAIL" if category == "SYNTAX_ERROR" else "PASS",
        "import": "FAIL" if category in {"IMPORT_ERROR", "MODULE_NOT_FOUND", "MISSING_SYMBOL"} else "PASS",
        "oracle": "FAIL" if "oracle" in joined.lower() else "PASS",
        "semantic": "FAIL" if "semantic" in joined.lower() else "PASS",
    }


def _structured_failure_details(error_text: str) -> dict[str, Any]:
    text = str(error_text or "")
    exception_match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\b", text)
    line_match = re.search(r"\bline\s+(\d+)\b", text, re.IGNORECASE)
    offset_match = re.search(r"\boffset\s+(\d+)\b", text, re.IGNORECASE)
    import_match = re.search(r"(?:from\s+([\w.]+)\s+import\s+([\w.]+)|cannot import name ['\"]?([\w.]+))", text)
    rejected_import = ""
    rejected_symbol = ""
    if import_match:
        rejected_import = str(import_match.group(1) or "")
        rejected_symbol = str(import_match.group(2) or import_match.group(3) or "")
    fragment = _extract_failed_line(text)
    oracle_pattern = ""
    for line in text.splitlines():
        if "oracle" in line.lower() or "semantic risk" in line.lower():
            oracle_pattern = line.strip()[:500]
            break
    return {
        "exception_type": str(exception_match.group(1) if exception_match else ""),
        "line_number": int(line_match.group(1)) if line_match else None,
        "offset": int(offset_match.group(1)) if offset_match else None,
        "offending_fragment": fragment,
        "rejected_import": rejected_import,
        "rejected_symbol": rejected_symbol,
        "rejected_oracle_pattern": oracle_pattern,
    }


def _bounded_m2_context(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fault_hypothesis": context.get("fault_hypothesis"),
        "oracle_hint": context.get("oracle_hint"),
        "initial_suspicious_functions": list(context.get("initial_suspicious_functions") or [])[:5],
        "candidate_source_files": list(context.get("candidate_source_files") or [])[:5],
        "source_context": str(
            context.get("suspicious_function_source")
            or context.get("target_source_code")
            or context.get("source_context")
            or ""
        )[:6000],
    }


def _current_m7_diagnosis(scenario: Mapping[str, Any]) -> dict[str, Any]:
    diagnosis = scenario.get("v26_diagnosis") or scenario.get("m7_diagnosis") or {}
    return dict(diagnosis) if isinstance(diagnosis, Mapping) else {}


def _prior_avoid_patterns(scenario: Mapping[str, Any]) -> dict[str, Any]:
    memory = scenario.get("repair_memory") or {}
    if not isinstance(memory, Mapping):
        memory = {}
    previous = scenario.get("previous_pass_avoid_evidence") or {}
    if not isinstance(previous, Mapping):
        previous = {}
    return {
        "assertion_patterns": list(
            previous.get("previous_assertion_patterns")
            or memory.get("avoid_assertion_patterns")
            or memory.get("forbidden_patterns")
            or []
        )[:20],
        "stimulus_patterns": list(
            previous.get("previous_stimulus_patterns")
            or memory.get("avoid_stimulus")
            or memory.get("forbidden_stimulus")
            or []
        )[:20],
    }


def _scenario_target_test_file(scenario: Mapping[str, Any]) -> str:
    target = scenario.get("target_location") if isinstance(scenario, Mapping) else {}
    if isinstance(target, Mapping) and target.get("candidate_test_file"):
        return str(target.get("candidate_test_file"))
    relevant = scenario.get("relevant_test_files") if isinstance(scenario, Mapping) else []
    if isinstance(relevant, list) and relevant:
        return str(relevant[0])
    return ""


def _collect_import_evidence(context: Mapping[str, Any], error_text: str) -> dict[str, Any]:
    available = context.get("available_imports")
    evidence: dict[str, Any] = {"available_imports": {}}
    names = set(re.findall(r"'([A-Za-z_]\w*)'", error_text or ""))
    if isinstance(available, Mapping):
        for module, symbols in available.items():
            if len(evidence["available_imports"]) >= 12:
                break
            symbol_list = [str(item) for item in symbols[:20]] if isinstance(symbols, list) else []
            if not names or names & set(symbol_list) or any(name.lower() in str(module).lower() for name in names):
                evidence["available_imports"][str(module)] = symbol_list
    return evidence


def _collect_signature_evidence(context: Mapping[str, Any], error_text: str) -> dict[str, Any]:
    signatures: dict[str, Any] = {}
    for key in ("function_signatures", "callable_signatures", "api_signatures"):
        value = context.get(key)
        if isinstance(value, Mapping):
            signatures.update({str(k): v for k, v in list(value.items())[:12]})
    names = set(re.findall(r"\b([A-Za-z_]\w*)\(", error_text or ""))
    return {"signatures": signatures, "error_call_names": sorted(names)}


def _collect_nearby_tests(repo_path: Path, target_test_file: str) -> list[str]:
    if not repo_path or not target_test_file:
        return []
    path = repo_path / target_test_file
    if not path.exists():
        return []
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(content)
    except (OSError, SyntaxError):
        return []
    lines = content.splitlines()
    examples: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            start = max(1, getattr(node, "lineno", 1))
            end = min(len(lines), getattr(node, "end_lineno", start + 12))
            examples.append("\n".join(lines[start - 1:end])[:1200])
            if len(examples) >= 3:
                break
    return examples


def _extract_failed_line(error_text: str) -> str:
    for line in reversed(str(error_text or "").splitlines()):
        if line.strip().startswith(("E   ", ">")) or ".py:" in line:
            return line.strip()[:500]
    return ""


def _extract_nearby_source(repo_path: Path, target_test_file: str, error_text: str) -> str:
    if not repo_path or not target_test_file:
        return ""
    path = repo_path / target_test_file
    if not path.exists():
        return ""
    line_no = 0
    match = re.search(rf"{re.escape(target_test_file)}:(\d+)", error_text or "")
    if match:
        line_no = int(match.group(1))
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    if not line_no:
        return "\n".join(lines[:40])[:1500]
    start = max(1, line_no - 8)
    end = min(len(lines), line_no + 8)
    return "\n".join(lines[start - 1:end])[:1500]

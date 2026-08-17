"""Patch-free alignment runner — Docker SDK 직접 실행.

harness를 호출하지 않고 Docker SDK(docker-py)를 통해 직접 컨테이너에서
before-patch 테스트를 실행한다.  이미 빌드된 instance image를 재사용한다.

흐름:
  1) instance image에서 컨테이너 생성·시작
  2) eval.sh 생성(test_patch 적용 + pytest + coverage)
  3) /bin/bash /eval.sh 실행
  4) stdout 파싱 → test_results, coverage_data
  5) 컨테이너 정리
"""
from __future__ import annotations

import ast as _ast
import json
import os
import re
import signal
import tarfile
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import docker

from src.benchmark.instance_loader import BenchmarkInstance
from src.contracts.envelope import SCHEMA_VERSION, make_envelope
from src.contracts.failure import FailureCategory
from src.contracts.feature_flags import V22FeatureFlags, resolve_feature_flags
from src.contracts.instance_views import PrePatchInstanceView, make_pre_patch_view
from src.contracts.models import CoverageResult, ExecutionResult, SBFLResult
from src.contracts.status import ExecutionStatus
from src.contracts.v37_oracle_flags import validated_v37_blocking_oracle_flags
from src.utils.artifact_hash import sha256_text
from src.utils.file_io import write_json_atomic
from src.executor.supplemental_pass_collector import (
    DEFAULT_MAX_SUPPLEMENTAL_PASS_TESTS,
    DEFAULT_MIN_DISTINCT_PASSING_TESTS,
    SupplementalPassCollector,
    SupplementalTestCandidate,
    candidate_hash_from_identity,
    normalize_supplemental_exhaustion,
)

M6_STABILITY_EXCLUSION_SKIP_REASON = "PERFORMANCE_EXCLUDED_FROM_FINAL_VERIFIED_PIPELINE"


def m6_execution_stability_exclusion_telemetry() -> Dict[str, Any]:
    """Return canonical final-profile telemetry for excluded stability checks."""
    return {
        "enabled": False,
        "attempted": False,
        "executed": False,
        "succeeded": False,
        "skipped": True,
        "skip_reason": M6_STABILITY_EXCLUSION_SKIP_REASON,
        "fallback_used": False,
        "fallback_reason": None,
        "elapsed_sec": 0.0,
        "status": "DISABLED",
        "triggered": False,
    }


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class AlignmentExecutionResult:
    instance_id: str
    run_id: str
    returncode: int          # 0=성공, 1+=실패
    raw_output: str          # 컨테이너 stdout 전체
    iteration: Optional[int] = None

    # before-patch 테스트 결과 (test_name → PASSED/FAILED/ERROR)
    test_results: Dict[str, str] = field(default_factory=dict)
    has_failure: bool = False
    has_error: bool = False
    # 커버리지 (file → {"stmts", "miss", "cover", "missing", "missing_lines"})
    coverage_data: Dict[str, Dict] = field(default_factory=dict)
    # contributing test functions
    contributing_functions: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    test_execution_results: List[Dict[str, Any]] = field(default_factory=list)
    stability_results: Dict[str, Any] = field(default_factory=dict)
    failure_signature: Optional[str] = None
    covered_sut_lines: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_ms: Optional[float] = None
    execution_id: Optional[str] = None
    canonical_test_id: Optional[str] = None
    test_nodeid: Optional[str] = None
    canonical_test_nodeid: Optional[str] = None
    harness_display_name: Optional[str] = None
    observed_test_result_keys: List[str] = field(default_factory=list)
    parent_execution_id: Optional[str] = None
    phase_timings: Dict[str, Any] = field(default_factory=dict)
    generated_patch_sha256: Optional[str] = None
    execution_command: Optional[str] = None
    failure_category: Optional[str] = None
    error_stage: Optional[str] = None
    exception_type: Optional[str] = None
    exception_traceback: Optional[str] = None
    error_origin: Optional[str] = None
    blocking_oracle_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def normalize_pre_patch_execution_status(
    result: AlignmentExecutionResult | Mapping[str, Any],
) -> ExecutionStatus:
    """Normalize one M6 pre-patch execution outcome.

    Population: generated candidate execution on the pre-patch repository.
    Output range: one of PASS, FAIL, ERROR, or NOT_RUN. Post-patch outcomes are
    intentionally ignored.
    """
    if isinstance(result, Mapping):
        data = dict(result)
    elif callable(getattr(result, "to_dict", None)):
        data = dict(result.to_dict())
    else:
        data = dict(vars(result))
    test_results = data.get("test_results") if isinstance(data.get("test_results"), dict) else {}
    has_error = bool(data.get("has_error"))
    has_failure = bool(data.get("has_failure"))
    statuses = {str(status).upper() for status in test_results.values()}
    if data.get("status"):
        statuses.add(str(data.get("status")).upper())
    if has_error or "ERROR" in statuses:
        return ExecutionStatus.ERROR
    if has_failure or "FAILED" in statuses or "FAIL" in statuses:
        return ExecutionStatus.FAIL
    if statuses and statuses <= {"PASSED", "PASS"}:
        return ExecutionStatus.PASS
    return ExecutionStatus.NOT_RUN


def normalize_test_execution_result(
    result: AlignmentExecutionResult | Mapping[str, Any],
    *,
    test_name: str | None = None,
) -> Dict[str, Any]:
    """Normalize one per-test M6 execution record while preserving raw fields.

    Per-test nonbinary statuses are preserved for diagnostics.  Only PASS,
    FAIL, and ERROR are metric outcomes; SKIP/XFAIL/XPASS/NOT_RUN/UNKNOWN stay
    outside the pipeline F/P populations.
    """
    data = result.to_dict() if isinstance(result, AlignmentExecutionResult) else dict(result)
    raw_results = data.get("test_results") if isinstance(data.get("test_results"), Mapping) else {}
    if test_name is None:
        test_name = str(
            data.get("test_name")
            or data.get("test_id")
            or next(iter(raw_results.keys()), "")
            or data.get("run_id")
        )
    raw_status = str(
        data.get("status")
        or raw_results.get(test_name)
        or next(iter(raw_results.values()), "")
        or ""
    ).upper()
    status = {
        "PASSED": "PASS",
        "PASS": "PASS",
        "FAILED": "FAIL",
        "FAIL": "FAIL",
        "ERROR": "ERROR",
        "SKIPPED": "SKIP",
        "SKIP": "SKIP",
        "XFAILED": "XFAIL",
        "XFAIL": "XFAIL",
        "XPASSED": "XPASS",
        "XPASS": "XPASS",
        "DESELECTED": "NOT_RUN",
        "NOT_RUN": "NOT_RUN",
        "NOT RUN": "NOT_RUN",
    }.get(
        raw_status,
        "ERROR" if data.get("has_error") else "UNKNOWN" if raw_status else "NOT_RUN",
    )
    normalized = dict(data)
    normalized.update(
        {
            "test_name": test_name,
            "test_nodeid": data.get("canonical_test_nodeid") or data.get("test_nodeid"),
            "canonical_test_nodeid": data.get("canonical_test_nodeid") or data.get("test_nodeid"),
            "harness_display_name": data.get("harness_display_name") or test_name,
            "observed_test_result_keys": list(data.get("observed_test_result_keys") or raw_results.keys()),
            "execution_id": str(data.get("execution_id") or data.get("run_id") or ""),
            "canonical_test_id": _canonical_test_id(data, fallback=test_name),
            "parent_execution_id": data.get("parent_execution_id"),
            "status": status,
            "execution_time_ms": data.get("execution_time_ms"),
            "error_message": _first_error_message(data),
            "stack_trace": data.get("stack_trace") or _fallback_traceback(str(data.get("raw_output") or "")),
            "failure_signature": normalize_failure_signature(
                data,
                test_name=test_name,
                status=status,
            ),
        "covered_sut_lines": _covered_sut_lines_for_test(data, test_name),
        "phase_timings": dict(data.get("phase_timings") or {}),
        }
    )
    return normalized


def _select_exact_candidate_result(
    canonical_nodeid: str,
    results: Mapping[str, Any],
    *,
    allow_parameterized_children: bool = False,
) -> tuple[Dict[str, str], str | None]:
    """Select one exact runner observation for the generated test."""
    if not canonical_nodeid:
        return {}, "missing_canonical_test_nodeid"
    if canonical_nodeid in results:
        return {canonical_nodeid: str(results[canonical_nodeid])}, None
    if allow_parameterized_children:
        parameterized = [
            (str(key), str(value))
            for key, value in results.items()
            if str(key).startswith(canonical_nodeid + "[")
            and str(key).endswith("]")
        ]
        if parameterized:
            statuses = {value.upper() for _, value in parameterized}
            aggregate = "PASSED" if statuses <= {"PASS", "PASSED"} else next(
                (
                    value
                    for _, value in parameterized
                    if value.upper() not in {"PASS", "PASSED"}
                ),
                "NOT_RUN",
            )
            return {canonical_nodeid: aggregate}, None
    parts = canonical_nodeid.split("::")
    path = parts[0]
    suffix = parts[1:]
    aliases: set[str] = set()
    if suffix:
        aliases.add(f"{path}:{':'.join(suffix)}")
    if len(suffix) >= 2:
        class_name, method_name = suffix[-2], suffix[-1]
        module = path[:-3].replace("/", ".") if path.endswith(".py") else path.replace("/", ".")
        module_aliases = {module}
        if module.startswith("tests."):
            module_aliases.add(module[len("tests."):])
        for module_alias in module_aliases:
            aliases.add(f"{method_name} ({module_alias}.{class_name})")
            aliases.add(f"{method_name} ({module_alias}.{class_name}.{method_name})")
    # A bare function name is not an exact executable identity: the same name
    # can exist in multiple files or classes. Runner-qualified renderings only.
    matches = [(str(key), str(value)) for key, value in results.items() if str(key) in aliases]
    if len(matches) == 1:
        key, value = matches[0]
        return {canonical_nodeid: value}, None
    return {}, "ambiguous_candidate_observation" if matches else "candidate_not_observed"


def _is_sut_source_file(source_file: str, *, generated_test_file: str = "") -> bool:
    """Return whether a normalized coverage path belongs to pre-patch SUT code."""
    raw_path = str(source_file or "").replace("\\", "/")
    raw_parts = Path(raw_path).parts
    if ".." in raw_parts:
        return False
    path = raw_path[2:] if raw_path.startswith("./") else raw_path
    generated = str(generated_test_file or "").replace("\\", "/").lstrip("./")
    if not path or path.startswith("/") or path == generated:
        return False
    parts = tuple(part.casefold() for part in Path(path).parts)
    name = Path(path).name.casefold()
    return not (
        "tests" in parts
        or "m6_pytest_compat" in parts
        or name == "conftest.py"
    )


def _first_error_message(data: Mapping[str, Any]) -> str:
    message = data.get("error_message")
    if message:
        return str(message)
    messages = data.get("error_messages")
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)) and messages:
        return str(messages[0])
    signal = _extract_failure_signal(str(data.get("raw_output") or ""))
    return signal.get("exception_message", "")


def normalize_failure_signature(
    result: AlignmentExecutionResult | Mapping[str, Any],
    *,
    test_name: str | None = None,
    status: str | None = None,
) -> str:
    """Build a deterministic M6 stability signature.

    The signature includes status, failing test identity, exception/assertion
    type, normalized whitespace in the failure message, and one stable stack
    location. It intentionally avoids broad value-stripping regexes because no
    approved rule in this repository defines safe volatile-value removal.
    """
    data = result.to_dict() if isinstance(result, AlignmentExecutionResult) else dict(result)
    raw_output = str(data.get("raw_output") or "")
    signal = _extract_failure_signal(raw_output)
    normalized_status = (status or normalize_pre_patch_execution_status(data).value).upper()
    resolved_test = str(test_name or data.get("test_name") or data.get("test_id") or signal.get("failing_test") or "")
    exception_type = str(data.get("exception_type") or signal.get("exception_type") or "")
    if not exception_type and normalized_status == "FAIL":
        exception_type = "AssertionError"
    message = str(data.get("error_message") or signal.get("exception_message") or _first_error_message(data) or "")
    message = re.sub(r"\s+", " ", message).strip()[:300]
    stack_location = str(data.get("stack_location") or _stable_stack_location(raw_output) or "")
    parts = [
        f"status={normalized_status}",
        f"test={resolved_test}",
        f"type={exception_type}",
        f"message={message}",
        f"stack={stack_location}",
    ]
    return "|".join(parts)


def _stable_stack_location(raw_output: str) -> str:
    for line in reversed(raw_output.splitlines()):
        file_match = re.search(r'File "([^"]+)", line (\d+)', line)
        if file_match:
            return f"{file_match.group(1)}:{file_match.group(2)}"
        pytest_match = re.search(r"([\w/.\-]+\.py):(\d+):", line)
        if pytest_match:
            return f"{pytest_match.group(1)}:{pytest_match.group(2)}"
    return ""


def _covered_sut_lines_for_test(data: Mapping[str, Any], test_name: str) -> List[Dict[str, Any]]:
    coverage = data.get("coverage_data")
    if isinstance(coverage, Mapping):
        by_test = coverage.get("covered_lines_by_test")
        if isinstance(by_test, Mapping) and isinstance(by_test.get(test_name), list):
            return [dict(line) for line in by_test[test_name] if isinstance(line, Mapping)]
        sut_lines = coverage.get("SUT_lines")
        if isinstance(sut_lines, list):
            return [dict(line) for line in sut_lines if isinstance(line, Mapping)]
    lines = data.get("covered_sut_lines") or data.get("covered_lines")
    if isinstance(lines, list):
        return [dict(line) for line in lines if isinstance(line, Mapping)]
    return []


def build_pre_patch_outcome_sets(
    executions: Iterable[AlignmentExecutionResult | Mapping[str, Any]],
) -> Dict[str, List[str]]:
    """Build M6 F/P/error sets exclusively from pre-patch execution results.

    FAIL enters ``failing_tests``; PASS enters ``passing_tests``; ERROR enters
    ``error_tests``. ``post_patch_outcome`` and other M8 fields are ignored by
    design. Test IDs are deduplicated while preserving first-seen order.
    """
    sets: Dict[str, List[str]] = {
        "failing_tests": [],
        "passing_tests": [],
        "error_tests": [],
    }
    seen: Dict[str, set[str]] = {key: set() for key in sets}
    for index, execution in enumerate(executions):
        data = execution.to_dict() if isinstance(execution, AlignmentExecutionResult) else dict(execution)
        test_id = _canonical_test_id(data, fallback=f"pre_patch_execution_{index}")
        status = normalize_pre_patch_execution_status(data)
        bucket = {
            ExecutionStatus.FAIL: "failing_tests",
            ExecutionStatus.PASS: "passing_tests",
            ExecutionStatus.ERROR: "error_tests",
        }.get(status)
        if bucket and test_id not in seen[bucket]:
            sets[bucket].append(test_id)
            seen[bucket].add(test_id)
    return sets


def build_cumulative_fp_state(
    executions: Iterable[AlignmentExecutionResult | Mapping[str, Any]],
    *,
    previous_state: Mapping[str, Any] | None = None,
    iteration: int = 1,
) -> Dict[str, Any]:
    """Construct cumulative M6 F/P sets from stable pre-patch outcomes only."""
    previous_state = previous_state or {}
    f_set = _dedupe_strings(previous_state.get("F_set") or previous_state.get("stable_F_set") or [])
    p_set = _dedupe_strings(previous_state.get("P_set") or previous_state.get("stable_P_set") or [])
    stable_executions = []
    for item in executions:
        data = item.to_dict() if isinstance(item, AlignmentExecutionResult) else dict(item)
        if data.get("flaky") or data.get("is_stable") is False:
            continue
        stable_executions.append(data)
    current = build_pre_patch_outcome_sets(stable_executions)
    current_f = set(current["failing_tests"])
    current_p = set(current["passing_tests"])
    # The current observation supersedes a stale opposite outcome for the
    # same logical generated test.  F and P are always disjoint.
    f_set = [test_id for test_id in f_set if test_id not in current_p]
    p_set = [test_id for test_id in p_set if test_id not in current_f]
    new_f = [test_id for test_id in current["failing_tests"] if test_id not in f_set]
    new_p = [test_id for test_id in current["passing_tests"] if test_id not in p_set]
    f_set.extend(new_f)
    p_set.extend(new_p)
    method = "initial" if iteration == 1 else "cumulative"
    return {
        "F_set": f_set,
        "P_set": p_set,
        "F_count": len(f_set),
        "P_count": len(p_set),
        "F_count_total": len(f_set),
        "P_count_total": len(p_set),
        "new_F_this_iteration": new_f,
        "new_P_this_iteration": new_p,
        "method": method,
        "F_P_construction": {
            "method": method,
            "source": "pre_patch_only",
            "new_F_this_iteration": new_f,
            "new_P_this_iteration": new_p,
            "F_count_total": len(f_set),
            "P_count_total": len(p_set),
        },
    }


def _dedupe_strings(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _first_test_nodeid(data: Mapping[str, Any]) -> str:
    test_results = data.get("test_results") if isinstance(data.get("test_results"), Mapping) else {}
    return str(
        data.get("test_nodeid")
        or data.get("test_name")
        or next(iter(test_results.keys()), "")
        or ""
    )


def _stable_generated_test_id(data: Mapping[str, Any]) -> str:
    """Return an explicit generated-test ID without falling back to run IDs."""
    for key in ("test_id", "generated_test_id"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = data.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("test_id", "generated_test_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _explicit_generated_test_nodeid(data: Mapping[str, Any]) -> str:
    for key in ("canonical_test_nodeid", "test_nodeid", "pytest_nodeid", "nodeid"):
        value = data.get(key)
        if isinstance(value, str) and "::" in value:
            return _normalize_pytest_nodeid(value)
    metadata = data.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("canonical_test_nodeid", "test_nodeid", "pytest_nodeid", "nodeid"):
            value = metadata.get(key)
            if isinstance(value, str) and "::" in value:
                return _normalize_pytest_nodeid(value)
    return ""


def _generated_test_file(data: Mapping[str, Any]) -> str:
    for key in ("target_test_file", "test_file", "file_path"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_pytest_file(value)
    metadata = data.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("target_test_file", "test_file", "file_path"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return _normalize_pytest_file(value)
    return ""


def _normalize_pytest_file(value: str) -> str:
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _nodeid_from_code(test_file: str, code: str) -> str:
    if not test_file:
        return ""
    try:
        tree = _ast.parse(code or "")
    except SyntaxError:
        return ""
    suffixes: list[str] = []
    for node in tree.body:
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and node.name.startswith("test"):
            suffixes.append(node.name)
        elif isinstance(node, _ast.ClassDef) and _collectable_test_class(node):
            for item in node.body:
                if isinstance(item, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and item.name.startswith("test"):
                    suffixes.append(f"{node.name}::{item.name}")
    if len(suffixes) != 1:
        return ""
    return f"{test_file}::{suffixes[0]}"


def _collectable_test_class(node: _ast.ClassDef) -> bool:
    """Mirror generator-side pytest/unittest class collection semantics."""
    explicitly_disabled = any(
        isinstance(item, (_ast.Assign, _ast.AnnAssign))
        and any(
            isinstance(target, _ast.Name) and target.id == "__test__"
            for target in (
                item.targets if isinstance(item, _ast.Assign) else [item.target]
            )
        )
        and isinstance(item.value, _ast.Constant)
        and item.value.value is False
        for item in node.body
    )
    is_test_case = any(
        (isinstance(base, _ast.Name) and base.id.endswith("TestCase"))
        or (isinstance(base, _ast.Attribute) and base.attr.endswith("TestCase"))
        for base in node.bases
    )
    custom_constructor = any(
        isinstance(item, (_ast.FunctionDef, _ast.AsyncFunctionDef))
        and item.name in {"__init__", "__new__"}
        for item in node.body
    )
    return bool(
        not explicitly_disabled
        and (node.name.startswith("Test") or is_test_case)
        and not custom_constructor
    )


def _normalize_pytest_nodeid(value: str) -> str:
    parts = [
        part.strip().replace("\\", "/") if index == 0 else part.strip()
        for index, part in enumerate(str(value).strip().split("::"))
    ]
    if parts:
        while parts[0].startswith("./"):
            parts[0] = parts[0][2:]
    return "::".join(part for part in parts if part)


def _load_generated_test_identity(generated_test_json_path: str) -> Dict[str, Any]:
    try:
        raw = json.loads(Path(generated_test_json_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, Mapping):
        return {}
    identity = {
        "test_id": _stable_generated_test_id(raw),
        "test_nodeid": _explicit_generated_test_nodeid(raw),
        "target_test_file": _generated_test_file(raw),
        "generated_patch_sha256": str(
            raw.get("generated_patch_sha256") or raw.get("patch_sha256") or ""
        ),
        "blocking_oracle_flags": validated_v37_blocking_oracle_flags(
            raw.get("blocking_oracle_flags") or []
        ),
    }
    if not identity["test_nodeid"]:
        test_file = _generated_test_file(raw)
        for key in ("test_code", "append_block"):
            code = raw.get(key)
            if isinstance(code, str) and code.strip():
                identity["test_nodeid"] = _nodeid_from_code(test_file, code)
                if identity["test_nodeid"]:
                    break
    return {
        key: value
        for key, value in identity.items()
        if value or key == "blocking_oracle_flags"
    }


def _validated_generated_patch_sha256(
    generated_test_json_path: str | Path,
    generated_patch: str,
) -> str:
    """Validate and return the exact generated patch identity consumed by M6."""
    path = Path(generated_test_json_path)
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"generated candidate artifact is unreadable: {path}") from error
    if not isinstance(candidate, Mapping):
        raise ValueError(f"generated candidate artifact must be an object: {path}")
    recorded = str(
        candidate.get("generated_patch_sha256")
        or candidate.get("patch_sha256")
        or ""
    )
    if not recorded:
        raise ValueError("generated candidate artifact is missing patch SHA provenance")
    actual = sha256_text(generated_patch)
    if recorded != actual:
        raise ValueError(
            "generated candidate patch SHA does not match the patch M6 would execute: "
            f"artifact={recorded}, m6_candidate={actual}"
        )
    return actual


def _canonical_identity_from_sources(
    *,
    generated_identity: Mapping[str, str] | None,
    execution: AlignmentExecutionResult | Mapping[str, Any],
    parent_canonical_test_id: str | None = None,
    fallback: Any = "",
) -> str:
    """Resolve production canonical identity without using volatile run IDs first."""
    data = execution.to_dict() if isinstance(execution, AlignmentExecutionResult) else dict(execution)
    generated_identity = generated_identity or {}
    explicit_test_id = generated_identity.get("test_id") or _stable_generated_test_id(data)
    if explicit_test_id:
        return str(explicit_test_id)
    generated_nodeid = generated_identity.get("test_nodeid")
    execution_nodeid = _normalize_pytest_nodeid(_first_test_nodeid(data)) if _first_test_nodeid(data) else ""
    if generated_nodeid:
        return str(generated_nodeid)
    if execution_nodeid:
        return execution_nodeid
    if parent_canonical_test_id:
        return str(parent_canonical_test_id)
    return str(fallback or "")


def _canonical_test_id(data: Mapping[str, Any], *, fallback: Any = "") -> str:
    """Return the M6 canonical generated-test identity for F/P and SBFL.

    ``run_id`` remains the concrete execution ID. Stability reruns provide
    ``canonical_test_id`` explicitly so ``-stability-N`` executions do not
    inflate F/P sets or spectra counts.
    """
    return str(
        data.get("canonical_test_id")
        or data.get("test_id")
        or _first_test_nodeid(data)
        or fallback
    )


def alignment_execution_to_contract_result(
    result: AlignmentExecutionResult,
) -> ExecutionResult:
    """Convert legacy M6 execution output to the shared ExecutionResult shape."""
    failure_category = _classify_m6_execution_failure(result)
    return ExecutionResult(
        instance_id=result.instance_id,
        run_id=result.run_id,
        execution_status=normalize_pre_patch_execution_status(result),
        test_results=dict(result.test_results),
        raw_output=result.raw_output,
        metadata={
            "stage": "pre_patch_alignment",
            "returncode": result.returncode,
            "error_messages": list(result.error_messages),
            "failure_category": failure_category.value if failure_category else None,
            "error_stage": result.error_stage,
            "exception_type": result.exception_type,
            "exception_traceback": result.exception_traceback,
            "generated_patch_sha256": result.generated_patch_sha256,
            "execution_command": result.execution_command,
        },
    )


def _classify_m6_execution_failure(
    result: AlignmentExecutionResult,
) -> FailureCategory | None:
    if result.failure_category:
        try:
            return FailureCategory(str(result.failure_category))
        except ValueError:
            return FailureCategory.PIPELINE_FAILURE
    if result.returncode == 0 and not result.has_error:
        return None
    message = " ".join(result.error_messages).lower()
    if "patch sha" in message or "patch identity" in message:
        return FailureCategory.PIPELINE_FAILURE
    environment_markers = (
        "docker image",
        "image build",
        "env image",
        "dependency",
        "repository checkout",
        "model server",
        "gpu",
        "out-of-memory",
        "out of memory",
    )
    if any(marker in message for marker in environment_markers):
        return FailureCategory.ENVIRONMENT_FAILURE
    return FailureCategory.EXECUTION_FAILURE


def alignment_execution_to_coverage_result(
    result: AlignmentExecutionResult,
) -> CoverageResult:
    """Convert M6 coverage without inventing missing line-level spectra."""
    has_coverage = bool(result.coverage_data)
    covered_sut_lines = list(result.coverage_data.get("covered_sut_lines", []) or [])
    return CoverageResult(
        instance_id=result.instance_id,
        coverage_data=dict(result.coverage_data),
        checked_coverage=None,
        metadata={
            "stage": "pre_patch",
            "instrumentation_status": "SUPPORTED" if has_coverage else "UNSUPPORTED",
            "covered_sut_lines": covered_sut_lines or None,
            "covered_sut_lines_available": bool(covered_sut_lines),
            "line_level_coverage_not_fabricated": True,
        },
    )


def _line_key(record: Mapping[str, Any]) -> str:
    return f"{record.get('source_file', '')}:{record.get('line_no', '')}"


def compute_ochiai_sbfl(
    spectra: Iterable[Mapping[str, Any]],
) -> SBFLResult:
    """Compute active core M6 SBFL from explicit pre-patch line spectra only.

    Activation follows the v22 prerequisite ``|F| >= 1`` and ``|P| >= 3``.
    ``covered_lines`` entries must contain ``source_file`` and ``line_no``;
    coverage report percentages are not expanded into fabricated line records.

    ``S_ochiai`` is calculated in ``[0, 1]`` with zero denominator returning
    ``0.0``. DStar and Tarantula are left as ``None`` with explicit diagnostics
    because this repository does not define the required DStar exponent or
    Tarantula zero-denominator behavior.
    """
    records = _canonical_observation_records([dict(item) for item in spectra])
    outcome_sets = build_pre_patch_outcome_sets(records)
    failing_ids = set(outcome_sets["failing_tests"])
    passing_ids = set(outcome_sets["passing_tests"])
    error_ids = set(outcome_sets["error_tests"])
    metadata: Dict[str, Any] = {
        "failing_tests": outcome_sets["failing_tests"],
        "passing_tests": outcome_sets["passing_tests"],
        "error_tests": outcome_sets["error_tests"],
        "activation_prerequisites": {"min_failing": 1, "min_passing": 3},
        "score_fields": {
            "raw_counts": ["e_f", "n_f", "e_p"],
            "normalized_scores": ["S_ochiai"],
            "blocked_scores": ["S_dstar", "S_tarantula"],
        },
        "blocked_formulas": {
            "S_dstar": "BLOCKING: DStar exponent is not defined in repository configuration",
            "S_tarantula": "BLOCKING: Tarantula zero-denominator behavior is not defined",
        },
    }
    if len(failing_ids) < 1 or len(passing_ids) < 3:
        if len(passing_ids) < 3:
            diagnostic_classification = "INSUFFICIENT_PRE_PATCH_PASSING_POPULATION"
        elif not records:
            diagnostic_classification = "NO_PRE_PATCH_SPECTRUM"
        else:
            diagnostic_classification = "INSUFFICIENT_PRE_PATCH_FAILING_POPULATION"
        return SBFLResult(
            instance_id=str(records[0].get("instance_id", "")) if records else "",
            suspiciousness=[],
            formula="ochiai",
            metadata={
                **metadata,
                "activation_status": "inactive_insufficient_tests",
                "sbfl_active": False,
                "diagnostic_classification": diagnostic_classification,
                "spectrum_source": "pre_patch_execution_only",
                "diagnostics": [
                    f"SBFL requires |F| >= 1 and |P| >= 3; got |F|={len(failing_ids)} |P|={len(passing_ids)}"
                ],
            },
        )

    line_records: Dict[str, Mapping[str, Any]] = {}
    covered_by_fail: Dict[str, set[str]] = {}
    covered_by_pass: Dict[str, set[str]] = {}
    element_types: set[str] = set()
    for record in records:
        test_id = _canonical_test_id(record, fallback=record.get("run_id") or "")
        status = normalize_pre_patch_execution_status(record)
        if status not in {ExecutionStatus.FAIL, ExecutionStatus.PASS}:
            continue
        covered_lines = record.get("covered_lines")
        if covered_lines is None:
            covered_lines = record.get("covered_sut_lines")
        if not isinstance(covered_lines, list):
            return SBFLResult(
                instance_id=str(record.get("instance_id", "")),
                suspiciousness=[],
                formula="ochiai",
                metadata={
                    **metadata,
                    "activation_status": "unsupported",
                    "sbfl_active": False,
                    "reason": "explicit covered line spectra unavailable",
                },
            )
        for line in covered_lines:
            if not isinstance(line, Mapping):
                continue
            element_type = str(line.get("element_type") or "line")
            element_types.add(element_type)
            source_file = str(line.get("source_file", ""))
            line_no = line.get("line_no")
            if element_type != "line" or not source_file or isinstance(line_no, bool) or not isinstance(line_no, int) or line_no <= 0:
                continue
            key = _line_key(line)
            line_records[key] = {"source_file": source_file, "line_no": line_no, "element_type": "line"}
            if status == ExecutionStatus.FAIL:
                covered_by_fail.setdefault(key, set()).add(test_id)
            elif status == ExecutionStatus.PASS:
                covered_by_pass.setdefault(key, set()).add(test_id)

    if len(element_types) > 1:
        return SBFLResult(
            instance_id=str(records[0].get("instance_id", "")) if records else "",
            suspiciousness=[],
            formula="ochiai",
            metadata={
                **metadata,
                "activation_status": "unsupported_mixed_element_types",
                "sbfl_active": False,
                "element_types": sorted(element_types),
                "reason": "line-level and function-level spectra must not be mixed",
            },
        )

    locations: List[Dict[str, Any]] = []
    for key, line in line_records.items():
        e_f = len(covered_by_fail.get(key, set()))
        e_p = len(covered_by_pass.get(key, set()))
        n_f = len(failing_ids) - e_f
        denominator = ((e_f + n_f) * (e_f + e_p)) ** 0.5
        score = 0.0 if denominator == 0 else e_f / denominator
        locations.append({**line, "score": round(score, 6), "ef": e_f, "ep": e_p, "nf": n_f})
    locations.sort(key=lambda item: (-item["score"], item["source_file"], item["line_no"]))
    for rank, item in enumerate(locations, 1):
        item["rank"] = rank
        item["e_f"] = item.pop("ef")
        item["e_p"] = item.pop("ep")
        item["n_f"] = item.pop("nf")
        item["S_ochiai"] = item["score"]
        item["S_dstar"] = None
        item["S_tarantula"] = None
    top5 = locations[:5]
    boundary_ties = _top_boundary_ties(locations, "S_ochiai", limit=5)
    return SBFLResult(
        instance_id=str(records[0].get("instance_id", "")) if records else "",
        suspiciousness=locations,
        formula="ochiai",
        metadata={
            **metadata,
            "activation_status": "active",
            "sbfl_active": True,
            "sbfl_top5_ochiai": top5,
            "top5_ties_at_boundary": {"ochiai": boundary_ties},
            "deterministic_ordering": ["score_desc", "source_file_asc", "line_no_asc"],
        },
    )


def _top_boundary_ties(
    locations: List[Dict[str, Any]],
    score_key: str,
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    if len(locations) <= limit:
        return []
    boundary_score = locations[limit - 1].get(score_key)
    return [
        item for item in locations[limit:]
        if item.get(score_key) == boundary_score
    ]


def _canonical_observation_records(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse attempts with the latest stable current observation winning."""
    canonical_records: Dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(records):
        data = dict(record)
        if data.get("flaky") or data.get("is_stable") is False:
            continue
        canonical_id = _canonical_test_id(data, fallback=f"pre_patch_execution_{index}")
        data["test_id"] = canonical_id
        data["canonical_test_id"] = canonical_id
        data["execution_id"] = str(data.get("execution_id") or data.get("run_id") or canonical_id)
        data["test_nodeid"] = str(data.get("test_nodeid") or _first_test_nodeid(data))
        canonical_records[canonical_id] = data
    return list(canonical_records.values())


def _metadata_feature_flags(
    result: AlignmentExecutionResult | Mapping[str, Any],
) -> V22FeatureFlags | None:
    data = result.to_dict() if isinstance(result, AlignmentExecutionResult) else dict(result)
    metadata = data.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    values = metadata.get("feature_flags")
    if not isinstance(values, Mapping):
        return None
    return resolve_feature_flags(values)


def _make_m6_envelope(
    *,
    result: AlignmentExecutionResult,
    payload: Mapping[str, Any],
    feature_flags: V22FeatureFlags | Mapping[str, bool] | None,
) -> Dict[str, Any]:
    resolved_flags = (
        resolve_feature_flags(feature_flags)
        if isinstance(feature_flags, Mapping)
        else feature_flags
    )
    flags_unavailable = resolved_flags is None
    flag_payload = resolved_flags.to_dict() if resolved_flags else {}
    enriched_payload = dict(payload)
    metadata = dict(enriched_payload.get("metadata") or {})
    if flags_unavailable:
        metadata["feature_flags_unavailable"] = True
        metadata["feature_flags_provenance"] = "unavailable_not_fabricated"
        config_id = "v22-feature-flags-unavailable"
        enriched_payload["metadata"] = metadata
        return {
            "schema_version": SCHEMA_VERSION,
            "instance_id": result.instance_id,
            "run_id": result.run_id,
            "module": "m6",
            "config_id": config_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feature_flags": {},
            "payload": enriched_payload,
            "iteration": result.iteration,
            "seed": None,
            "model": None,
            "prompt_version": None,
        }
    else:
        metadata["feature_flags_provenance"] = "explicit_or_execution_metadata"
        config_id = resolved_flags.config_id
    enriched_payload["metadata"] = metadata
    return make_envelope(
        instance_id=result.instance_id,
        run_id=result.run_id,
        module="m6",
        payload=enriched_payload,
        feature_flags=flag_payload,
        config_id=config_id,
        iteration=result.iteration,
    ).to_dict()


def build_m6_contract_artifacts(
    result: AlignmentExecutionResult,
    *,
    feature_flags: V22FeatureFlags | Mapping[str, bool] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Build contract-compatible M6 artifacts from one pre-patch execution."""
    flags = feature_flags or _metadata_feature_flags(result)
    resolved_flags = (
        flags if isinstance(flags, V22FeatureFlags)
        else resolve_feature_flags(flags) if isinstance(flags, Mapping)
        else None
    )
    explicit_executions = _m6_explicit_executions(result)
    core_payload = getattr(result, "_m6_core_execution_data", None)
    execution = alignment_execution_to_contract_result(result).to_dict()
    coverage = alignment_execution_to_coverage_result(result).to_dict()
    if isinstance(core_payload, Mapping):
        execution["metadata"]["m6_core_execution_data"] = dict(core_payload)
        coverage_core_metadata = {
            "coverage_data": core_payload.get("coverage_data", {}),
            "F_P_construction": core_payload.get("F_P_construction", {}),
        }
        if resolved_flags and resolved_flags.m6_execution_stability:
            coverage_core_metadata["stability_results"] = core_payload.get(
                "stability_results", []
            )
        coverage["metadata"]["m6_core_execution_data"] = coverage_core_metadata
    if resolved_flags and resolved_flags.m6_sbfl:
        sbfl = compute_ochiai_sbfl(explicit_executions).to_dict()
    else:
        sbfl = _disabled_sbfl_result(
            result,
            reason="m6_sbfl feature flag disabled",
            explicit_executions=explicit_executions,
        ).to_dict()
    supplemental_collection = getattr(result, "_m6_supplemental_pass_collection", None)
    if isinstance(supplemental_collection, Mapping):
        sbfl.setdefault("metadata", {})["supplemental_pass_collection"] = dict(
            supplemental_collection
        )
        diagnostic_classification = normalize_supplemental_exhaustion(
            supplemental_collection
        )
        if (
            not sbfl.get("metadata", {}).get("sbfl_active")
            and diagnostic_classification
        ):
            sbfl["metadata"]["diagnostic_classification"] = diagnostic_classification
            sbfl["metadata"]["spectrum_source"] = "pre_patch_supplemental_collection"
    if isinstance(core_payload, Mapping):
        sbfl["metadata"]["m6_core_execution_data"] = {
            "F_set": core_payload.get("F_set", []),
            "P_set": core_payload.get("P_set", []),
            "F_count": core_payload.get("F_count", 0),
            "P_count": core_payload.get("P_count", 0),
            "F_P_construction": core_payload.get("F_P_construction", {}),
        }
    return {
        "execution_result": _make_m6_envelope(
            result=result,
            payload=execution,
            feature_flags=flags,
        ),
        "coverage_result": _make_m6_envelope(
            result=result,
            payload=coverage,
            feature_flags=flags,
        ),
        "sbfl_result": _make_m6_envelope(
            result=result,
            payload=sbfl,
            feature_flags=flags,
        ),
    }


def _m6_explicit_executions(result: AlignmentExecutionResult) -> List[Dict[str, Any]]:
    executions = getattr(result, "_m6_explicit_executions", None)
    if isinstance(executions, Sequence) and not isinstance(executions, (str, bytes)):
        return [
            dict(item.to_dict() if isinstance(item, AlignmentExecutionResult) else item)
            for item in executions
            if isinstance(item, (AlignmentExecutionResult, Mapping))
        ]
    return [
        {
            "instance_id": result.instance_id,
            "test_id": result.canonical_test_id or result.run_id,
            "run_id": result.run_id,
            "execution_id": result.execution_id or result.run_id,
            "canonical_test_id": result.canonical_test_id or result.run_id,
            "test_nodeid": result.canonical_test_nodeid or result.test_nodeid,
            "canonical_test_nodeid": result.canonical_test_nodeid or result.test_nodeid,
            "harness_display_name": result.harness_display_name,
            "observed_test_result_keys": list(result.observed_test_result_keys),
            "parent_execution_id": result.parent_execution_id,
            "test_results": result.test_results,
            "has_failure": result.has_failure,
            "has_error": result.has_error,
            "covered_lines": result.covered_sut_lines or None,
            "covered_sut_lines": result.covered_sut_lines,
            "coverage_data": result.coverage_data,
        }
    ]


def _disabled_sbfl_result(
    result: AlignmentExecutionResult,
    *,
    reason: str,
    explicit_executions: Sequence[Mapping[str, Any]],
) -> SBFLResult:
    outcome_sets = build_pre_patch_outcome_sets(explicit_executions)
    return SBFLResult(
        instance_id=result.instance_id,
        suspiciousness=[],
        formula="ochiai",
        metadata={
            "failing_tests": outcome_sets["failing_tests"],
            "passing_tests": outcome_sets["passing_tests"],
            "error_tests": outcome_sets["error_tests"],
            "activation_status": "disabled_by_feature_flag",
            "sbfl_active": False,
            "reason": reason,
            "sbfl_top5_ochiai": [],
            "sbfl_top5_dstar": [],
            "sbfl_top5_tarantula": [],
            "blocked_formulas": {
                "S_dstar": "BLOCKING: DStar exponent is not defined in repository configuration",
                "S_tarantula": "BLOCKING: Tarantula zero-denominator behavior is not defined",
            },
        },
    )


def _execution_records_for_m6_core(
    executions: Sequence[AlignmentExecutionResult],
    *,
    stability_results: Mapping[str, Any],
    use_stability: bool,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    is_stable = bool(stability_results.get("is_stable")) if use_stability else True
    flaky = bool(stability_results.get("flaky") or stability_results.get("any_flaky")) if use_stability else False
    for execution in executions:
        data = execution.to_dict()
        test_name = str(next(iter(execution.test_results.keys()), execution.run_id))
        canonical_id = _canonical_test_id(data, fallback=execution.run_id)
        data["test_id"] = canonical_id
        data["canonical_test_id"] = canonical_id
        data["test_nodeid"] = str(data.get("test_nodeid") or test_name)
        data["execution_id"] = str(data.get("execution_id") or execution.run_id)
        data["parent_execution_id"] = data.get("parent_execution_id")
        data["covered_lines"] = _covered_sut_lines_for_test(data, test_name)
        data["covered_sut_lines"] = data["covered_lines"]
        data["is_stable"] = is_stable
        data["flaky"] = flaky
        if use_stability:
            data["stability_results"] = _stability_header(stability_results)
        records.append(data)
    return records


def _compact_stability_run(
    item: Mapping[str, Any],
    *,
    run_index: int,
    canonical_test_id: str,
) -> Dict[str, Any]:
    return {
        "execution_id": str(item.get("execution_id") or item.get("run_id") or ""),
        "run_index": run_index,
        "outcome": str(item.get("status") or normalize_pre_patch_execution_status(item).value),
        "returncode": item.get("returncode"),
        "failure_signature": str(item.get("failure_signature") or normalize_failure_signature(item)),
        "canonical_test_id": str(item.get("canonical_test_id") or canonical_test_id),
        "test_nodeid": str(item.get("test_nodeid") or item.get("test_name") or _first_test_nodeid(item)),
        "parent_execution_id": item.get("parent_execution_id"),
        "phase_timings": dict(item.get("phase_timings") or {}),
    }


def _compact_stability_results(
    stability_results: Mapping[str, Any],
    *,
    canonical_test_id: str,
) -> Dict[str, Any]:
    results = stability_results.get("results")
    compact_runs = []
    if isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
        compact_runs = [
            _compact_stability_run(item, run_index=index + 1, canonical_test_id=canonical_test_id)
            for index, item in enumerate(results)
            if isinstance(item, Mapping)
        ]
    compact = {
        "runs": stability_results.get("runs", len(compact_runs)),
        "results": compact_runs,
        "signatures": list(stability_results.get("signatures") or []),
        "is_stable": bool(stability_results.get("is_stable")),
        "stable": bool(stability_results.get("stable")),
        "flaky": bool(stability_results.get("flaky")),
        "any_flaky": bool(stability_results.get("any_flaky")),
        "agreement_rate": stability_results.get("agreement_rate"),
        "runs_required": stability_results.get("runs_required"),
        "final_outcome": stability_results.get("final_outcome"),
        "stable_F_set": list(stability_results.get("stable_F_set") or []),
        "stable_P_set": list(stability_results.get("stable_P_set") or []),
        "schedule": dict(stability_results.get("schedule") or {}),
        "timing_breakdown": dict(stability_results.get("timing_breakdown") or {}),
    }
    majority = stability_results.get("majority_result")
    if isinstance(majority, Mapping):
        compact["majority_result"] = _compact_stability_run(
            majority,
            run_index=1,
            canonical_test_id=canonical_test_id,
        )
    else:
        compact["majority_result"] = None
    return compact


def _stability_header(stability_results: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "runs": stability_results.get("runs"),
        "is_stable": stability_results.get("is_stable"),
        "stable": stability_results.get("stable"),
        "flaky": stability_results.get("flaky"),
        "any_flaky": stability_results.get("any_flaky"),
        "agreement_rate": stability_results.get("agreement_rate"),
        "runs_required": stability_results.get("runs_required"),
        "final_outcome": stability_results.get("final_outcome"),
        "stable_F_set": list(stability_results.get("stable_F_set") or []),
        "stable_P_set": list(stability_results.get("stable_P_set") or []),
    }


def decide_stability(
    executions: Iterable[AlignmentExecutionResult | Mapping[str, Any]],
) -> Dict[str, Any]:
    """Apply the approved M6 stability schedule to completed executions.

    Schedule: inspect the first three runs; if outcome and failure signature
    differ, inspect up to two additional runs and majority-vote the outcome.
    This helper does not perform Docker execution.
    """
    records = [item.to_dict() if isinstance(item, AlignmentExecutionResult) else dict(item) for item in executions]
    if len(records) < 3:
        raise ValueError("stability validation requires at least three executions")
    considered = records[:5]
    normalized_results = [
        normalize_test_execution_result(item)
        for item in considered
    ]
    statuses = [str(item["status"]) for item in normalized_results]
    signatures = [str(item["failure_signature"]) for item in normalized_results]
    first_three = list(zip(statuses[:3], signatures[:3]))
    stable_initial = len(first_three) == 3 and len(set(first_three)) == 1
    test_id = _canonical_test_id(considered[0], fallback=normalized_results[0].get("test_name") or "")
    if stable_initial:
        stable_f = [test_id] if statuses[0] == "FAIL" and test_id else []
        stable_p = [test_id] if statuses[0] == "PASS" and test_id else []
        return {
            "runs": 3,
            "results": normalized_results[:3],
            "signatures": signatures[:3],
            "is_stable": True,
            "stable": True,
            "flaky": False,
            "any_flaky": False,
            "agreement_rate": 1.0,
            "runs_required": 3,
            "majority_result": normalized_results[0],
            "final_outcome": statuses[0],
            "stable_F_set": stable_f,
            "stable_P_set": stable_p,
            "schedule": {"initial_runs": 3, "additional_runs": 2, "maximum_runs": 5},
        }
    counts: Dict[Tuple[str, str], int] = {}
    for key in zip(statuses, signatures):
        counts[key] = counts.get(key, 0) + 1
    majority_key, majority_count = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )[0]
    majority_index = next(
        index for index, key in enumerate(zip(statuses, signatures))
        if key == majority_key
    )
    final_outcome = majority_key[0] if counts else ExecutionStatus.NOT_RUN.value
    total_runs = len(considered)
    return {
        "runs": total_runs,
        "results": normalized_results,
        "signatures": signatures,
        "is_stable": False,
        "stable": False,
        "flaky": bool(considered),
        "any_flaky": bool(considered),
        "agreement_rate": majority_count / total_runs if total_runs else 0.0,
        "runs_required": min(5, max(3, len(considered))),
        "majority_result": normalized_results[majority_index],
        "final_outcome": final_outcome,
        "stable_F_set": [],
        "stable_P_set": [],
        "schedule": {"initial_runs": 3, "additional_runs": 2, "maximum_runs": 5},
    }


def verify_execution_stability(
    execute_once: Callable[[int], AlignmentExecutionResult | Mapping[str, Any]],
) -> Dict[str, Any]:
    """Execute the approved M6 3+2 stability schedule with an injectable runner.

    The callable receives the zero-based isolated run index. Production Docker
    wiring is intentionally separate; unit tests can pass deterministic fake
    executors without running Docker.
    """
    executions: List[AlignmentExecutionResult | Mapping[str, Any]] = [
        execute_once(index) for index in range(3)
    ]
    first_decision = decide_stability(executions)
    if first_decision["is_stable"]:
        first_decision["timing_breakdown"] = _stability_timing_breakdown(executions)
        return first_decision
    executions.extend(execute_once(index) for index in range(3, 5))
    final_decision = decide_stability(executions)
    final_decision["timing_breakdown"] = _stability_timing_breakdown(executions)
    return final_decision


def _stability_timing_breakdown(
    executions: Sequence[AlignmentExecutionResult | Mapping[str, Any]],
) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    aggregate: Dict[str, float] = {}
    for index, execution in enumerate(executions, 1):
        data = execution.to_dict() if isinstance(execution, AlignmentExecutionResult) else dict(execution)
        timing = data.get("phase_timings") if isinstance(data.get("phase_timings"), Mapping) else {}
        phases = timing.get("phases") if isinstance(timing.get("phases"), Mapping) else {}
        runs.append({
            "run_index": index,
            "execution_id": data.get("execution_id") or data.get("run_id"),
            "phases": dict(phases),
        })
        for name, value in phases.items():
            try:
                aggregate[str(name)] = round(aggregate.get(str(name), 0.0) + float(value), 3)
            except (TypeError, ValueError):
                continue
    return {
        "schema_version": "m6-stability-timing-breakdown-v1",
        "runs": runs,
        "aggregate_phase_seconds": aggregate,
        "run_count": len(runs),
    }


def build_m6_core_execution_data(
    executions: Iterable[AlignmentExecutionResult | Mapping[str, Any]],
    *,
    previous_state: Mapping[str, Any] | None = None,
    iteration: int = 1,
) -> Dict[str, Any]:
    """Build the reusable M6 core output payload from pre-patch executions."""
    records = [item.to_dict() if isinstance(item, AlignmentExecutionResult) else dict(item) for item in executions]
    canonical_records = _canonical_observation_records(records)
    fp_state = build_cumulative_fp_state(canonical_records, previous_state=previous_state, iteration=iteration)
    sbfl = compute_ochiai_sbfl(canonical_records).to_dict()
    sbfl_metadata = dict(sbfl.get("metadata") or {})
    stability_results = _unique_stability_results(records)
    return {
        "test_results": [normalize_test_execution_result(record) for record in canonical_records],
        "execution_attempts": [_compact_execution_attempt(record, index + 1) for index, record in enumerate(records)],
        "coverage_data": _merge_coverage_payload(canonical_records),
        "F_set": fp_state["F_set"],
        "P_set": fp_state["P_set"],
        "F_count": fp_state["F_count"],
        "P_count": fp_state["P_count"],
        "sbfl_active": bool(sbfl_metadata.get("sbfl_active")),
        "sbfl_spectrum": sbfl.get("suspiciousness", []),
        "sbfl_top5_ochiai": sbfl_metadata.get("sbfl_top5_ochiai", []),
        "sbfl_top5_dstar": [],
        "sbfl_top5_tarantula": [],
        "F_P_construction": fp_state["F_P_construction"],
        "spectrum_provenance": {
            "candidate_generated_F": [
                str(record.get("canonical_test_id") or _canonical_test_id(record, fallback=""))
                for record in canonical_records
                if not record.get("supplemental_pass")
                and normalize_pre_patch_execution_status(record) == ExecutionStatus.FAIL
            ],
            "current_pass_P": [
                str(record.get("canonical_test_id") or _canonical_test_id(record, fallback=""))
                for record in canonical_records
                if not record.get("supplemental_pass")
                and normalize_pre_patch_execution_status(record) == ExecutionStatus.PASS
            ],
            "supplemental_pre_patch_P": [
                str(record.get("canonical_test_id") or _canonical_test_id(record, fallback=""))
                for record in canonical_records
                if record.get("supplemental_pass")
                and normalize_pre_patch_execution_status(record) == ExecutionStatus.PASS
            ],
        },
        "stability_results": stability_results,
        "any_flaky": any(bool(record.get("flaky")) for record in records)
        or any(bool(item.get("any_flaky")) for item in stability_results),
        "stable_F_set": fp_state["F_set"],
        "stable_P_set": fp_state["P_set"],
        "diagnostics": {
            "sbfl": sbfl_metadata,
            "activation_hook_required": (
                "Wire build_m6_core_execution_data into src.pipeline.run_single "
                "after the shared contract branch permits top-level pipeline changes."
            ),
        },
    }


def _unique_stability_results(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        stability = record.get("stability_results")
        if not isinstance(stability, Mapping):
            continue
        canonical_id = _canonical_test_id(record, fallback=f"pre_patch_execution_{index}")
        if canonical_id in seen:
            continue
        seen.add(canonical_id)
        summaries.append(dict(stability))
    return summaries


def _compact_execution_attempt(record: Mapping[str, Any], run_index: int) -> Dict[str, Any]:
    return {
        "execution_id": str(record.get("execution_id") or record.get("run_id") or ""),
        "run_index": run_index,
        "outcome": normalize_pre_patch_execution_status(record).value,
        "returncode": record.get("returncode"),
        "failure_signature": str(record.get("failure_signature") or normalize_failure_signature(record)),
        "canonical_test_id": _canonical_test_id(record, fallback=record.get("run_id") or ""),
        "test_nodeid": str(record.get("test_nodeid") or _first_test_nodeid(record)),
        "parent_execution_id": record.get("parent_execution_id"),
        "phase_timings": dict(record.get("phase_timings") or {}),
    }


def _merge_coverage_payload(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    sut_lines: List[Dict[str, Any]] = []
    covered_lines_by_test: Dict[str, List[Dict[str, Any]]] = {}
    raw_files: Dict[str, Any] = {}
    for record in records:
        test_name = _canonical_test_id(record, fallback=record.get("run_id") or "")
        lines = _covered_sut_lines_for_test(record, test_name)
        if lines:
            covered_lines_by_test[test_name] = lines
            sut_lines.extend(lines)
        coverage = record.get("coverage_data")
        if isinstance(coverage, Mapping):
            for key, value in coverage.items():
                if key not in {"SUT_lines", "covered_lines_by_test"}:
                    raw_files[str(key)] = value
    return {
        **raw_files,
        "SUT_lines": _dedupe_line_records(sut_lines),
        "covered_lines_by_test": covered_lines_by_test,
    }


def _dedupe_line_records(lines: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int]] = set()
    for line in lines:
        source_file = str(line.get("source_file", ""))
        line_no = line.get("line_no")
        if not source_file or isinstance(line_no, bool) or not isinstance(line_no, int) or line_no <= 0:
            continue
        key = (source_file, line_no)
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"source_file": source_file, "line_no": line_no, "element_type": str(line.get("element_type") or "line")})
    return sorted(deduped, key=lambda item: (item["source_file"], item["line_no"]))


def _fallback_traceback(raw_output: str, lines: int = 10) -> str:
    if not raw_output:
        return ""
    return "\n".join(raw_output.strip().splitlines()[-lines:])


def _extract_runtime_exception(raw_output: str) -> str:
    if not raw_output:
        return ""
    tb_blocks = list(re.finditer(r"Traceback \(most recent call last\):", raw_output))
    if tb_blocks:
        tb_section = raw_output[tb_blocks[-1].start():]
        lines = [line for line in tb_section.splitlines() if line.strip()]
        for line in reversed(lines):
            if re.match(r"^\s*[\w.]+(?:Error|Exception|Warning|DoesNotExist|NotFound):", line):
                return line.strip()
    m = re.search(r"\b\w+(?:Error|Exception|Warning|DoesNotExist|NotFound): [^\n]+", raw_output)
    return m.group(0).strip() if m else ""


def _extract_failure_signal(raw_output: str) -> Dict[str, str]:
    signal = {
        "exception_type": "",
        "exception_message": "",
        "failing_line": "",
        "failing_test": "",
    }
    if not raw_output:
        return signal
    failed_tests = re.findall(r"FAILED\s+([^\s]+?::[^\s]+)", raw_output)
    if failed_tests:
        signal["failing_test"] = failed_tests[-1]
    else:
        unittest_match = re.search(r"FAIL:\s+([^\n]+)", raw_output)
        if unittest_match:
            signal["failing_test"] = unittest_match.group(1).strip()
    failing_lines = [
        line.strip()[1:].strip()
        for line in raw_output.splitlines()
        if line.lstrip().startswith(">") and len(line.strip()) > 1
    ]
    if failing_lines:
        signal["failing_line"] = failing_lines[-1][:300]
    exception = _extract_runtime_exception(raw_output)
    if exception:
        m = re.match(r"([\w.]+(?:Error|Exception|Warning|DoesNotExist|NotFound|AssertionError)):\s*(.*)", exception)
        if m:
            signal["exception_type"] = m.group(1).split(".")[-1]
            signal["exception_message"] = m.group(2)[:300]
        else:
            parts = exception.split(":", 1)
            signal["exception_type"] = parts[0].strip().split(".")[-1]
            signal["exception_message"] = parts[1].strip()[:300] if len(parts) > 1 else exception[:300]
    return signal


# ---------------------------------------------------------------------------
# Docker 유틸 (harness 의존 없음)
# ---------------------------------------------------------------------------

def _copy_to_container(container, src: Path, dst: Path) -> None:
    """로컬 파일을 컨테이너에 복사한다."""
    tar_path = src.with_suffix(".tar")
    with tarfile.open(tar_path, "w") as tar:
        tar.add(str(src), arcname=dst.name)
    try:
        with open(tar_path, "rb") as f:
            data = f.read()
        container.exec_run(f"mkdir -p {dst.parent}")
        container.put_archive(str(dst.parent), data)
    finally:
        tar_path.unlink(missing_ok=True)


def _finalize_exec_stream(stream: Any) -> None:
    """Finalize a Docker exec stream through its owning HTTP response once.

    Docker's ``CancellableStream.close()`` shuts down only the underlying
    socket.  Calling it after normal exhaustion leaves requests/urllib3 to
    finalize a response whose file pointer was already torn down.  Prefer the
    idempotent owning response lifecycle; retain ``close()`` only for generic
    stream implementations that expose no response owner.
    """
    if stream is None:
        return
    response = getattr(stream, "_response", None)
    if response is not None and callable(getattr(response, "close", None)):
        response.close()
        return
    close = getattr(stream, "close", None)
    if callable(close):
        close()


def _exec_with_timeout_status(container, cmd: str, timeout: int = 600):
    """Execute a container command and preserve its real exit status."""
    exec_result = ""
    exec_id = None
    exception = None
    timed_out = False
    stream = None

    def _run():
        nonlocal exec_result, exec_id, exception, stream
        try:
            exec_id = container.client.api.exec_create(container.id, cmd)["Id"]
            stream = container.client.api.exec_start(exec_id, stream=True)
            for chunk in stream:
                exec_result += chunk.decode("utf-8", errors="replace")
        except Exception as e:
            exception = e
        finally:
            try:
                _finalize_exec_stream(stream)
            except Exception as cleanup_error:
                if exception is None:
                    exception = cleanup_error

    t = threading.Thread(target=_run, daemon=True)
    start_time = time.time()
    t.start()
    t.join(timeout)
    elapsed = time.time() - start_time
    timed_out = t.is_alive()

    if exception is not None:
        raise exception

    exit_code = None
    if exec_id is not None and not timed_out:
        inspected = container.client.api.exec_inspect(exec_id)
        raw_exit_code = inspected.get("ExitCode") if isinstance(inspected, Mapping) else None
        if isinstance(raw_exit_code, int):
            exit_code = raw_exit_code
    return exec_result, timed_out, elapsed, exit_code


def _execution_command_provenance(modified_eval: str) -> str:
    """Fingerprint the exact generated-test eval script copied to `/eval.sh`."""
    return f"eval_script_sha256:{sha256_text(modified_eval)}"


def _exec_with_timeout(container, cmd: str, timeout: int = 600):
    """Backward-compatible wrapper for callers that do not need exit status."""
    output, timed_out, elapsed, _ = _exec_with_timeout_status(container, cmd, timeout)
    return output, timed_out, elapsed


def _cleanup_container(client, container) -> None:
    """컨테이너를 정지·제거한다."""
    if container is None:
        return
    cid = container.id
    try:
        container.stop(timeout=10)
    except Exception as e:
        print(f"[cleanup] container.stop failed: {e}")
        try:
            info = client.api.inspect_container(cid)
            pid = info["State"].get("Pid", 0)
            if pid > 0:
                os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    try:
        container.remove(force=True)
    except Exception as e:
        print(f"[cleanup] container.remove failed: {e}")


# ---------------------------------------------------------------------------
# eval.sh 생성
# ---------------------------------------------------------------------------

def _build_eval_script(
    test_patch: str,
    base_commit: str,
    repo: str,
    version: str,
    *,
    test_directives_override: Sequence[str] | None = None,
    apply_test_patch: bool = True,
) -> str:
    """컨테이너 안에서 실행할 eval.sh 스크립트를 생성한다.

    harness의 make_eval_script_list() 로직을 재현하되
    instance-specific constants를 직접 참조한다.
    """
    from tddbench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    from tddbench.harness.utils import get_test_directives

    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    HEREDOC_DELIMITER = "EOF_114329324912"
    DIFF_MODIFIED_FILE_REGEX = r"--- a/(.*)"

    test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch) if apply_test_patch else []
    reset_tests = f"git checkout {base_commit} {' '.join(test_files)}"
    apply_patch = (
        f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    )

    # instance dict 형태 — get_test_directives 용
    inst_dict = {"repo": repo, "version": version, "test_patch": test_patch}
    directives = (
        list(test_directives_override)
        if test_directives_override is not None
        else get_test_directives(inst_dict)
    )
    test_command = " ".join([specs["test_cmd"], *directives])

    lines = [
        "#!/bin/bash",
        "set -uxo pipefail",
        "source /opt/miniconda3/bin/activate",
        "conda activate testbed",
        "cd /testbed",
    ]
    if "eval_commands" in specs:
        lines += specs["eval_commands"]
    lines += [
        "git config --global --add safe.directory /testbed",
        "cd /testbed",
        "git status",
        f"git diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        "conda activate testbed",
    ]
    if "install" in specs:
        lines.append(_with_bootstrap_packaging_warning_filter(specs["install"]))

    # sympy: pytest 미설치 환경이므로 coverage run -m pytest 전에 사전 설치
    if repo == "sympy/sympy":
        lines.append(_with_bootstrap_packaging_warning_filter("pip install pytest -q --disable-pip-version-check"))

    # sympy: ./bin/test → python -m pytest (coverage 수집 가능)
    if repo == "sympy/sympy" and "./bin/test" in test_command:
        test_command = re.sub(
            r"\./bin/test(?:\s+-C)?(?:\s+--verbose)?",
            "python -m pytest -x --no-header -rN",
            test_command,
        )

    # Django는 test_cmd가 coverage run을 포함하지 않으므로 래핑
    if "django" in repo.lower():
        if test_command.lstrip().startswith("coverage"):
            # test_cmd가 이미 coverage run을 포함하는 경우 중복 추가 방지
            test_command_cov = test_command
        else:
            test_command_cov = re.sub(
                r"^python(?:3)?\s+(-m\s+)?",
                lambda m: f"coverage run {m.group(1) or ''}",
                test_command,
                count=1,
            )
            if test_command_cov == test_command:
                # regex 미매칭 시 coverage run 직접 prepend
                test_command_cov = f"coverage run {test_command}"
    elif repo == "sympy/sympy":
        # sympy는 pytest로 교체됐으므로 coverage run으로 래핑
        test_command_cov = re.sub(
            r"^python\s+-m\s+pytest",
            "coverage run -m pytest",
            test_command,
            count=1,
        )
        if test_command_cov == test_command:
            test_command_cov = f"coverage run -m pytest -x --no-header -rN"
    else:
        test_command_cov = test_command

    use_pytest_compat_filter = _is_pytest_execution_command(test_command_cov)
    pytest_compat_filter_lines = (
        _pytest_legacy_nose_setup_filter_script() if use_pytest_compat_filter else []
    )
    if use_pytest_compat_filter:
        test_command_cov = _with_pytest_legacy_nose_setup_filter(test_command_cov)

    if repo.strip() == "sphinx-doc/sphinx":
        sphinx_test_command = test_command
        sphinx_uses_pytest = _is_pytest_execution_command(sphinx_test_command)
        sphinx_filter_lines = _pytest_legacy_nose_setup_filter_script() if sphinx_uses_pytest else []
        if sphinx_uses_pytest:
            sphinx_test_command = _with_pytest_legacy_nose_setup_filter(sphinx_test_command)
        lines += [
            *([reset_tests, apply_patch] if apply_test_patch else []),
            _with_bootstrap_packaging_warning_filter("python3 -m pip install coverage"),
            _with_bootstrap_packaging_warning_filter("pip install pytest-cov"),
            *sphinx_filter_lines,
            'export PYTEST_ADDOPTS="--cov=sphinx --cov-report=term-missing"',
            sphinx_test_command,
            "coverage report -m",
            "coverage json -o /tmp/m6_coverage.json >/dev/null 2>&1 || true",
            "echo M6_COVERAGE_JSON_BEGIN",
            "if [ -f /tmp/m6_coverage.json ]; then cat /tmp/m6_coverage.json; fi",
            "echo M6_COVERAGE_JSON_END",
            *([reset_tests] if apply_test_patch else [f"git reset --hard {base_commit}", "git clean -fd"]),
        ]
    else:
        lines += [
            *([reset_tests, apply_patch] if apply_test_patch else []),
            _with_bootstrap_packaging_warning_filter("python3 -m pip install coverage"),
            *pytest_compat_filter_lines,
            test_command_cov,
            "coverage report --show-missing",
            "coverage json -o /tmp/m6_coverage.json >/dev/null 2>&1 || true",
            "echo M6_COVERAGE_JSON_BEGIN",
            "if [ -f /tmp/m6_coverage.json ]; then cat /tmp/m6_coverage.json; fi",
            "echo M6_COVERAGE_JSON_END",
            *([reset_tests] if apply_test_patch else [f"git reset --hard {base_commit}", "git clean -fd"]),
        ]

    return "\n".join(lines) + "\n"


_BOOTSTRAP_PACKAGING_WARNING_FILTER = (
    "ignore:The 'wheel' package is no longer the canonical location of the "
    "'bdist_wheel' command:FutureWarning:wheel.bdist_wheel"
)


def _with_bootstrap_packaging_warning_filter(command: str) -> str:
    """Suppress one known third-party packaging warning for bootstrap commands.

    The filter is scoped to install/bootstrap commands only. It intentionally
    does not wrap pytest execution, so warnings from generated tests and target
    project code remain visible under the repository's own warning policy.
    """
    return (
        'PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}'
        f'{_BOOTSTRAP_PACKAGING_WARNING_FILTER}" {command}'
    )


_PYTEST_LEGACY_NOSE_SETUP_FILTER_MESSAGE = (
    r"(?s)Support for nose tests is deprecated and will be removed in a future "
    r"release\..*using nose-specific method: `setup\(self\)`"
)


def _pytest_legacy_nose_setup_filter_script() -> List[str]:
    """Create a pytest plugin that suppresses only legacy nose setup noise."""
    return [
        "M6_PYTEST_COMPAT_DIR=/tmp/m6_pytest_compat",
        'mkdir -p "$M6_PYTEST_COMPAT_DIR"',
        'cat > "$M6_PYTEST_COMPAT_DIR/m6_pytest_compat_filter.py" <<\'PY\'',
        "import inspect",
        "import re",
        "import warnings",
        "import pytest",
        "",
        f'_MESSAGE_RE = re.compile(r"{_PYTEST_LEGACY_NOSE_SETUP_FILTER_MESSAGE}")',
        "_ORIGINAL_WARN = warnings.warn",
        "",
        "def _is_pytest_legacy_nose_setup_warning(message, category):",
        "    warning_category = category or type(message)",
        "    expected = getattr(pytest, \"PytestRemovedIn8Warning\", None)",
        "    if expected is None or not isinstance(warning_category, type) or not issubclass(warning_category, expected):",
        "        return False",
        "    frame = inspect.currentframe()",
        "    caller = frame.f_back.f_back if frame and frame.f_back else None",
        "    caller_module = caller.f_globals.get(\"__name__\", \"\") if caller else \"\"",
        "    return caller_module == \"_pytest.python\" and bool(_MESSAGE_RE.match(str(message)))",
        "",
        "def _m6_warn(message, category=None, stacklevel=1, source=None):",
        "    if _is_pytest_legacy_nose_setup_warning(message, category):",
        "        return None",
        "    if source is None:",
        "        return _ORIGINAL_WARN(message, category=category, stacklevel=stacklevel)",
        "    return _ORIGINAL_WARN(message, category=category, stacklevel=stacklevel, source=source)",
        "",
        "def _install_filter():",
        '    category = getattr(pytest, "PytestRemovedIn8Warning", None)',
        '    if category is None:',
        '        category = getattr(pytest, "PytestWarning", Warning)',
        "    warnings.filterwarnings(",
        '        "ignore",',
        f'        message=r"{_PYTEST_LEGACY_NOSE_SETUP_FILTER_MESSAGE}",',
        "        category=category,",
        r'        module=r"_pytest\.python",',
        "    )",
        "    warnings.warn = _m6_warn",
        "",
        "def pytest_configure(config):",
        "    _install_filter()",
        "",
        "@pytest.hookimpl(tryfirst=True)",
        "def pytest_runtest_setup(item):",
        "    _install_filter()",
        "PY",
        'export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$M6_PYTEST_COMPAT_DIR"',
    ]


def _is_pytest_execution_command(command: str) -> bool:
    return bool(re.search(r"(^|\s)(?:python(?:3)?\s+-m\s+pytest|coverage\s+run\b.*\s-m\s+pytest|pytest)(\s|$)", command))


def _with_pytest_legacy_nose_setup_filter(command: str) -> str:
    """Load the M6 pytest compatibility plugin for pytest commands only."""
    if "-p m6_pytest_compat_filter" in command:
        return command
    updated = re.sub(
        r"(\b-m\s+pytest\b)",
        r"\1 -p m6_pytest_compat_filter",
        command,
        count=1,
    )
    if updated != command:
        return updated
    return re.sub(
        r"(^|\s)(pytest)(\s|$)",
        r"\1\2 -p m6_pytest_compat_filter\3",
        command,
        count=1,
    )


# ---------------------------------------------------------------------------
# 로그 파서 (harness의 MAP_REPO_TO_PARSER 재사용)
# ---------------------------------------------------------------------------

def _parse_test_output(test_output: str, repo: str) -> Dict[str, str]:
    """pytest/django/sympy 등 프레임워크별 파서로 테스트 결과를 파싱한다.

    Falls back to a simple pytest regex parser if repo-specific parser
    returns empty results.
    """
    unittest_result = _unittest_line_parse(test_output)
    if unittest_result:
        return unittest_result

    from tddbench.harness.log_parsers import MAP_REPO_TO_PARSER
    parser = MAP_REPO_TO_PARSER.get(repo)
    result: Dict[str, str] = {}
    if parser is not None:
        result = parser(test_output)
    if result:
        sanitized = _sanitize_test_results(result)
        if sanitized:
            # Repository parsers often omit nonbinary pytest outcomes. Merge
            # the line-local fallback without overriding an authoritative
            # result already emitted for the same exact node.
            fallback = _fallback_pytest_parse(test_output)
            for nodeid, status in fallback.items():
                sanitized.setdefault(nodeid, status)
            return sanitized
        # sanitization이 모든 항목을 제거한 경우 (harness parser가 오탐한 경우)
        # fallback 파서로 계속 진행
    # sympy bin/test 전용 fallback: "test_xxx F      [FAIL]" 또는 "test_xxx ok   [OK]" 패턴
    if repo == "sympy/sympy":
        sympy_result = _sympy_fallback_parse(test_output)
        if sympy_result:
            return sympy_result
    # Fallback: simple pytest PASSED/FAILED/ERROR extraction
    return _fallback_pytest_parse(test_output)


# Valid pytest node ID must contain "::" and end with a Python identifier
_VALID_NODE_RE = re.compile(r".+\.py::[\w\[\]\-]+")

# Django/unittest test ID: "test_name (module.ClassName)" or "test_name"
_DJANGO_NODE_RE = re.compile(r"^\w[\w.]* \([\w.]+\)$")

# SymPy bin/test ID: plain "test_xxx" function name (no module path, no "::")
_SYMPY_NODE_RE = re.compile(r"^test_\w+$")

# SymPy harness parser ID: "path/to/test.py:test_name" (single colon, no "::")
_SYMPY_PATH_NODE_RE = re.compile(r".+\.py:test_\w+$")


def _unittest_line_parse(output: str) -> Dict[str, str]:
    """Parse unittest/Django runner result lines.

    Django's harness parser can occasionally infer FAILED from surrounding
    traceback/diff text even when the canonical unittest result line says
    ``... ok`` or ``... ERROR``. The explicit per-test line is authoritative.
    """
    status_map = {
        "ok": "PASSED",
        "FAIL": "FAILED",
        "ERROR": "ERROR",
        "skipped": "SKIPPED",
        "expected failure": "XFAIL",
        "unexpected success": "XPASS",
    }
    results: Dict[str, str] = {}
    pending_identity: str | None = None
    pending_chatter_lines = 0
    pending_ambiguous = False
    max_pending_chatter_lines = 64
    for line in output.splitlines():
        stripped = line.strip()
        identity_only = re.match(r"^(test[\w.\-]+\s+\([^)]+\))$", stripped)
        if identity_only:
            identity = identity_only.group(1)
            if pending_identity and pending_identity != identity:
                # A detached status cannot be assigned safely after multiple
                # candidate identities have appeared without an outcome.
                pending_identity = None
                pending_ambiguous = True
            elif not pending_ambiguous:
                pending_identity = identity
                pending_chatter_lines = 0
            continue
        identity_with_chatter = re.match(
            r"^(test[\w.\-]+\s+\([^)]+\))\s+\.\.\.\s+.+$",
            stripped,
        )
        if identity_with_chatter and not re.search(
            r"\s\.\.\.\s+(?:ok|FAIL|ERROR|expected failure|unexpected success|skipped(?:\s+.*)?)\s*$",
            stripped,
        ):
            identity = identity_with_chatter.group(1)
            if pending_identity and pending_identity != identity:
                pending_identity = None
                pending_ambiguous = True
            elif not pending_ambiguous:
                pending_identity = identity
                pending_chatter_lines = 0
            continue
        m = re.match(
            r"^(?:.*?\s+)?(test[\w.\-]+\s+\([^)]+\))\s+\.\.\.\s+"
            r"(ok|FAIL|ERROR|expected failure|unexpected success|skipped(?:\s+.*)?)\s*$",
            stripped,
        )
        if not m:
            docstring_match = re.match(
                r"^.+?\s+\(([\w.]+\.(test[\w.\-]+))\)\s+\.\.\.\s+"
                r"(ok|FAIL|ERROR|expected failure|unexpected success|skipped(?:\s+.*)?)\s*$",
                stripped,
            )
            if docstring_match:
                qualified = docstring_match.group(1)
                method = docstring_match.group(2)
                raw_status = docstring_match.group(3)
                status_key = "skipped" if raw_status.startswith("skipped") else raw_status
                results[f"{method} ({qualified})"] = status_map[status_key]
                pending_identity = None
                pending_ambiguous = False
                continue
            pending_status = re.match(
                r"^.+?\s+\.\.\.\s+"
                r"(ok|FAIL|ERROR|expected failure|unexpected success|skipped(?:\s+.*)?)\s*$",
                stripped,
            )
            if pending_identity and pending_status:
                raw_status = pending_status.group(1)
                status_key = "skipped" if raw_status.startswith("skipped") else raw_status
                results[pending_identity] = status_map[status_key]
                pending_identity = None
                pending_ambiguous = False
                continue
            detached_status = re.match(
                r"^(ok|FAIL|ERROR|expected failure|unexpected success|skipped(?:\s+.*)?)$",
                stripped,
            )
            if pending_identity and detached_status and not pending_ambiguous:
                raw_status = detached_status.group(1)
                status_key = "skipped" if raw_status.startswith("skipped") else raw_status
                results[pending_identity] = status_map[status_key]
                pending_identity = None
                pending_ambiguous = False
                continue
        if m:
            raw_status = m.group(2)
            status_key = "skipped" if raw_status.startswith("skipped") else raw_status
            results[m.group(1)] = status_map[status_key]
            pending_identity = None
            pending_ambiguous = False
            continue
        if pending_identity:
            pending_chatter_lines += 1
            if pending_chatter_lines > max_pending_chatter_lines:
                pending_identity = None
    return results


def _sanitize_test_results(results: Dict[str, str]) -> Dict[str, str]:
    """Remove entries with invalid test node IDs (e.g. 'not', '[2]').

    Accepts:
    - pytest-style IDs: path.py::Class::method
    - Django/unittest-style IDs: test_name (module.ClassName)
    - SymPy bin/test IDs: test_name (plain function name)
    - SymPy harness-parser IDs: path/to/test.py:test_name (single colon)
    Some harness parsers mis-parse non-standard output lines into garbage.
    """
    return {
        k: v for k, v in results.items()
        if (_VALID_NODE_RE.match(k) or _DJANGO_NODE_RE.match(k)
            or _SYMPY_NODE_RE.match(k) or _SYMPY_PATH_NODE_RE.match(k))
    }


def _extract_error_details(raw_output: str) -> List[str]:
    """raw Docker output에서 구체적 에러 메시지를 추출한다."""
    # TDD-Bench output can echo the injected unified diff.  Diff additions are
    # candidate source, not runtime exception records.
    runtime_output = "\n".join(
        line for line in raw_output.splitlines()
        if not line.startswith(("+", "-", "@@", "diff --git", "index "))
    )
    patterns = [
        r"(ImportError:\s*.+)",
        r"(ModuleNotFoundError:\s*.+)",
        r"(SyntaxError:\s*.+)",
        r"(AttributeError:\s*.+)",
        r"(NameError:\s*.+)",
        r"(AppRegistryNotReady:\s*.+)",
        r"(E\s+ImportError:\s*.+)",
        r"(E\s+ModuleNotFoundError:\s*.+)",
    ]
    seen: set = set()
    errors: List[str] = []
    for pat in patterns:
        for m in re.finditer(pat, runtime_output):
            msg = m.group(1).strip().lstrip("E").strip()[:300]
            if msg not in seen:
                errors.append(msg)
                seen.add(msg)
    return errors[:5]


def _detect_test_not_collected(raw_output: str) -> Optional[str]:
    """Detect 'test not collected' patterns from raw Docker output.

    Returns a diagnostic message if detected, otherwise None.
    """
    # NameError (e.g. 'unittest' not defined) — more specific cause
    name_error = re.search(r"NameError: name '(\w+)' is not defined", raw_output)
    if name_error:
        name = name_error.group(1)
        return (
            f"Test not collected (NameError: '{name}' is not defined). "
            f"Missing import: add 'import {name}' to the test file."
        )

    # Django app_label RuntimeError — `from tests.xxx import Y` causes this
    if re.search(r"RuntimeError: Model class tests\.", raw_output):
        return (
            "Test not collected (RuntimeError: Django model imported via `from tests.xxx import Y`). "
            "CRITICAL: NEVER use `from tests.xxx import Y` — it triggers RuntimeError in Django's test runner. "
            "Import app modules directly without the `tests.` prefix "
            "(e.g. `from modeladmin.models import X` not `from tests.modeladmin.models import X`). "
            "Check the existing test file's import block for the correct paths."
        )

    app_registry_error = re.search(r"AppRegistryNotReady:\s*(.+)", raw_output)
    if app_registry_error:
        return (
            "Test not collected (AppRegistryNotReady: "
            f"{app_registry_error.group(1).strip()[:200]}). "
            "The generated test import or placement initialized a Django model too early."
        )

    # ImportError / ModuleNotFoundError during collection — import 오류이므로 진단 분리
    import_error = re.search(
        r"(?:E\s+)?(ImportError|ModuleNotFoundError):\s*(.+)", raw_output
    )
    if import_error:
        kind = import_error.group(1)
        detail = import_error.group(2).strip()[:200]
        return (
            f"Test not collected ({kind}: {detail}). "
            "Fix the import in the generated test."
        )

    patterns = [
        (r"ERROR:\s*not found:\s*(\S+)", "not found"),
        (r"no tests ran", "no tests ran"),
        (r"collected 0 items", "collected 0 items"),
    ]
    for pattern, label in patterns:
        m = re.search(pattern, raw_output, re.IGNORECASE)
        if m:
            detail = m.group(0)[:200]
            return (
                f"Test not collected ({label}): {detail}. "
                "Possible cause: module-level pytest.importorskip, "
                "pytest.mark.skip, or missing dependency."
            )
    return None


def _sympy_fallback_parse(output: str) -> Dict[str, str]:
    """sympy bin/test 출력 파싱.

    지원하는 출력 형식:
    1) bin/test 스타일:
        test_foo F                    [FAIL]
        test_bar ok                   [OK]
        test_baz E                    [FAIL]  (ERROR)
    2) pytest FAILURES 섹션 스타일 (sympy가 pytest로 실행될 때):
        ___ path/to/test.py:test_name ___
       → test_name = FAILED
    3) sympy bin/test 요약 스타일:
        tests finished: N passed, M failed
       → 통과/실패 카운트로 test_name 보완
    """
    results: Dict[str, str] = {}

    # Format 1: bin/test line-by-line
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("test_"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        test_name = parts[0]
        last = parts[-1]
        second = parts[1] if len(parts) >= 2 else ""
        if last in ("[FAIL]", "[ERROR]") or second in ("F", "E"):
            results[test_name] = "FAILED"
        elif last == "[OK]" or second == "ok":
            results[test_name] = "PASSED"

    # Format 2: pytest FAILURES section header
    # 두 형식 모두 지원:
    #   "___ path/to/file.py:test_name ___"  (경로+콜론 prefix)
    #   "___ test_name ___"                  (pytest -x --no-header 형식, 콜론 없음)
    for m in re.finditer(r"_{3,}\s+(?:\S+:)?(test_\w+)\s+_{3,}", output):
        test_name = m.group(1)
        results[test_name] = "FAILED"

    return results


def _fallback_pytest_parse(output: str) -> Dict[str, str]:
    """Extract test results from raw pytest output using common patterns."""
    results: Dict[str, str] = {}
    # Pattern: "PASSED tests/foo.py::TestBar::test_baz"  or
    #          "tests/foo.py::test_baz PASSED"
    for m in re.finditer(
        r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS|DESELECTED)[^\S\r\n]+"
        r"([\w/.\-]+::[^\s]+)(?:[^\r\n]*)$",
        output,
        re.MULTILINE,
    ):
        results[m.group(2)] = m.group(1)
    for m in re.finditer(
        r"^([\w/.\-]+::[^\s]+)[^\S\r\n]+"
        r"(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS|DESELECTED)(?:[^\r\n]*)$",
        output,
        re.MULTILINE,
    ):
        results[m.group(1)] = m.group(2)

    # pytest -rA short-summary SKIP form has no concrete node ID:
    # ``SKIPPED [1] tests/test_mod.py:7: reason``.  Retain a diagnostic
    # identity so the mixed execution fails closed instead of becoming PASS.
    for m in re.finditer(
        r"^SKIPPED[^\S\r\n]+\[\d+\][^\S\r\n]+([\w/.\-]+\.py):(\d+):[^\r\n]*$",
        output,
        re.MULTILINE,
    ):
        results.setdefault(
            f"{m.group(1)}::M6_SKIP_SUMMARY_LINE_{m.group(2)}",
            "SKIPPED",
        )

    # Fallback: -rN 플래그 사용 시 PASSED 형식이 없고 요약만 있는 경우
    # e.g. "======================== 1 passed, 97 warnings in 0.35s ========================"
    # e.g. "coverage run -m pytest ... path/to/test.py::test_name"
    if not results:
        summary_m = re.search(r"(\d+) passed", output)
        if summary_m and int(summary_m.group(1)) > 0:
            # coverage run 커맨드 라인에서 test node id 추출
            cmd_m = re.search(r"coverage run.*?pytest.*?([\w/.\-]+\.py::test_\w+)", output)
            if cmd_m:
                results[cmd_m.group(1)] = "PASSED"
            else:
                # coverage run 없이 pytest 직접 실행 시
                node_m = re.search(r"pytest.*?([\w/.\-]+\.py::test_\w+)", output)
                if node_m:
                    results[node_m.group(1)] = "PASSED"

    return results


# ---------------------------------------------------------------------------
# 커버리지 파싱
# ---------------------------------------------------------------------------

def _parse_coverage_text(coverage_text: str) -> Dict[str, Dict]:
    """``coverage report --show-missing`` 텍스트를 파싱한다.

    Handles both ``--show-missing`` and ``-m`` formatting and is tolerant of
    slightly different column layouts produced by different coverage versions.
    """
    data: Dict[str, Dict] = {}
    header_seen = False
    for line in coverage_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("---"):
            header_seen = True
            continue
        if line.startswith("Name"):
            header_seen = True
            continue
        if line.startswith("TOTAL"):
            continue
        if not header_seen:
            continue

        parts = line.split()
        if len(parts) < 4:
            continue
        filename = parts[0]
        if not filename.endswith(".py"):
            continue

        # Find the coverage percentage column (ends with '%')
        cover_idx = -1
        cover = 0.0
        for idx, p in enumerate(parts[1:], 1):
            if p.endswith("%"):
                try:
                    cover = float(p.rstrip("%"))
                    cover_idx = idx
                except ValueError:
                    continue
                break

        if cover_idx < 0:
            # Some versions omit % — try column 3 or 4 as a number
            for idx in (3, 4):
                if idx < len(parts):
                    try:
                        cover = float(parts[idx])
                        cover_idx = idx
                        break
                    except ValueError:
                        continue
            if cover_idx < 0:
                continue

        try:
            stmts = int(parts[1])
            miss = int(parts[2])
        except (ValueError, IndexError):
            continue

        missing_str = " ".join(parts[cover_idx + 1:]) if cover_idx + 1 < len(parts) else ""
        data[filename] = {
            "stmts": stmts,
            "miss": miss,
            "cover": cover,
            "missing": missing_str,
            "missing_lines": _parse_missing_lines(missing_str),
        }
    return data


def _parse_line_spectrum_json(
    output: str,
    *,
    test_id: str,
    generated_test_file: str = "",
) -> Dict[str, Any]:
    """Parse the optional coverage.py JSON line spectrum emitted by M6."""
    begin_marker = "M6_COVERAGE_JSON_BEGIN"
    end_marker = "M6_COVERAGE_JSON_END"
    begin = output.find(begin_marker)
    # Docker returns stdout/stderr interleaved.  A later shell trace can
    # therefore appear inside the JSON body; use the final end marker.
    end = output.rfind(end_marker) if begin >= 0 else -1
    if begin < 0 or end < 0:
        return {}
    raw_json = output[begin + len(begin_marker):end].strip()
    # Remove ``set -x`` command traces, including traces interleaved directly
    # after a JSON comma before the next newline.
    raw_json = re.sub(r"\+\s+[^\r\n]*(?:\r?\n|$)", "", raw_json)
    # ``git checkout`` writes this status line on the same multiplexed stream;
    # it can split a JSON key when stdout/stderr chunks interleave.
    raw_json = re.sub(r"Updated\s+\d+\s+path\s+from\s+[0-9a-f]+\r?\n", "", raw_json)
    try:
        # ``set -x`` prefixes the marker block with shell trace lines (for
        # example ``+ echo ...`` and ``+ cat ...``).  Decode the first complete
        # JSON object rather than requiring the marker body to be pure JSON.
        json_start = raw_json.find("{")
        if json_start < 0:
            return {}
        payload, _ = json.JSONDecoder().raw_decode(raw_json[json_start:])
    except (TypeError, json.JSONDecodeError):
        return {}
    files = payload.get("files") if isinstance(payload, Mapping) else None
    if not isinstance(files, Mapping):
        return {}
    lines: list[dict[str, Any]] = []
    sut_line_set: list[dict[str, Any]] = []
    for filename, info in files.items():
        if not isinstance(info, Mapping):
            continue
        path = str(filename).replace("\\", "/")
        if not path.endswith(".py") or not _is_sut_source_file(
            path, generated_test_file=generated_test_file
        ):
            continue
        executed_line_numbers = {
            line_no
            for line_no in (info.get("executed_lines", []) or [])
            if isinstance(line_no, int) and not isinstance(line_no, bool) and line_no > 0
        }
        missing_line_numbers = {
            line_no
            for line_no in (info.get("missing_lines", []) or [])
            if isinstance(line_no, int) and not isinstance(line_no, bool) and line_no > 0
        }
        for line_no in sorted(executed_line_numbers | missing_line_numbers):
            sut_line_set.append(
                {"source_file": path, "line_no": line_no, "element_type": "line"}
            )
        for line_no in sorted(executed_line_numbers):
            if isinstance(line_no, int) and not isinstance(line_no, bool) and line_no > 0:
                lines.append(
                    {"source_file": path, "line_no": line_no, "element_type": "line"}
                )
    if not sut_line_set:
        return {}
    return {
        "SUT_lines": lines,
        "L_SUT": sut_line_set,
        "covered_sut_lines": lines,
        "covered_lines_by_test": {test_id: lines},
        "line_spectrum_source": "coverage_json_pre_patch",
    }


def _parse_missing_lines(missing_str: str) -> List[int]:
    """'23, 45-50, 60' → [23, 45, 46, ..., 50, 60]"""
    lines: List[int] = []
    if not missing_str:
        return lines
    for part in missing_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "->" in part:
            continue
        if "-" in part:
            try:
                s, e = part.split("-", 1)
                lines.extend(range(int(s.strip()), int(e.strip()) + 1))
            except ValueError:
                continue
        else:
            try:
                lines.append(int(part))
            except ValueError:
                continue
    return lines


# ---------------------------------------------------------------------------
# contributing functions 추출
# ---------------------------------------------------------------------------

def _get_contributing_functions(test_patch: str) -> Dict[str, List[str]]:
    """test_patch에서 추가/수정된 테스트 함수 목록을 추출한다.

    Returns: {filename: [func_name, ...]}
    """
    funcs: Dict[str, List[str]] = {}
    segments = test_patch.split("+++ b")
    for seg in segments[1:]:
        filename = seg.split("\n")[0].strip()
        if filename.startswith("/"):
            filename = filename[1:]
        for part in seg.split("def test")[1:]:
            fname = "test" + part.split("(")[0].strip()
            flines = part.split("\n")
            for ln in flines:
                if ln.strip().startswith("+"):
                    cleaned = ln.replace("+", "").replace("-", "")
                    if cleaned.strip() == "":
                        continue
                    funcs.setdefault(filename, [])
                    if fname not in funcs[filename]:
                        funcs[filename].append(fname)
                    break
    return funcs


def _resolve_fun2test(container, contributing: Dict[str, List[str]], timeout: int) -> List[str]:
    """contributing functions를 Class::method 형식의 pytest 노드 ID로 변환한다."""
    fun2test: List[str] = []
    for test_file, func_names in contributing.items():
        # 컨테이너에서 테스트 파일 읽기
        output, _, _ = _exec_with_timeout(container, f"cat {test_file}", timeout)
        if not output.strip():
            for fn in func_names:
                fun2test.append(f"{test_file}::{fn}")
            continue

        # class method 매핑
        class_func = _get_class_functions(output)
        outer_func = _get_outer_functions(output)

        for fn in func_names:
            if fn in class_func:
                fun2test.append(f"{test_file}::{class_func[fn]}::{fn}")
            elif fn in outer_func:
                fun2test.append(f"{test_file}::{fn}")
            else:
                fun2test.append(f"{test_file}::{fn}")
    return fun2test


def _get_class_functions(text: str) -> Dict[str, str]:
    """test method → class name 매핑을 반환한다."""
    try:
        tree = _ast.parse(text)
    except SyntaxError:
        return {}
    mapping: Dict[str, str] = {}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef):
            for item in node.body:
                if isinstance(item, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    mapping[item.name] = node.name
    return mapping


def _get_outer_functions(text: str) -> List[str]:
    """모듈 레벨 test 함수 이름을 반환한다."""
    try:
        tree = _ast.parse(text)
    except SyntaxError:
        return []
    return [
        node.name
        for node in _ast.iter_child_nodes(tree)
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef))
        and "test" in node.name
    ]


def _modify_eval_script(script: str, repo: str, fun2test: List[str]) -> str:
    """eval.sh의 pytest/tox/sympy 테스트 명령어를 fun2test만 실행하도록 수정한다."""
    if not fun2test:
        return script

    out_lines: List[str] = []
    for ln in script.split("\n"):
        if (
            "coverage run" in ln
            or "tox --current-env -epy39 -v --" in ln
            or "./bin/test" in ln          # sympy bin/test (교체 전 원본 스크립트 매칭)
            or "python -m pytest" in ln   # plain pytest repos (astropy, matplotlib, sympy 교체 후)
            or (ln.strip().startswith("pytest ") and "#" not in ln)
            or ("runtests.py" in ln and repo == "django/django")
        ):
            parts = ln.split(" ")
            # 마지막 파일/테스트 인자 제거
            fcount = 0
            for i in range(len(parts) - 1, 0, -1):
                if ".py" in parts[i] or "." in parts[i]:
                    fcount += 1
                else:
                    break
            base_cmd = " ".join(parts[:len(parts) - fcount])

            if repo == "django/django":
                cases = []
                for item in fun2test:
                    item = item.removeprefix("tests/")
                    item = item.replace(".py", "").replace("::", ".").replace("/", ".")
                    cases.append(item)
                out_lines.append(f"{base_cmd} {' '.join(cases)}")
            elif repo == "sympy/sympy":
                # sympy는 ./bin/test → coverage run -m pytest로 교체
                for item in fun2test:
                    out_lines.append(f"coverage run -m pytest -x --no-header -rN {item}")
            else:
                out_lines.append(f"{base_cmd} {' '.join(fun2test)}")
        else:
            out_lines.append(ln)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# stdout split helper
# ---------------------------------------------------------------------------

_COVERAGE_SPLIT_PATTERNS = [
    "+ coverage report",          # set -x echo
    "+ python3 -m coverage",      # alternative invocation
    "Name    Stmts   Miss",       # coverage table header (direct)
    "Name                 Stmts",  # wider column variant
]


def _split_output(test_output: str) -> tuple:
    """Split container output into (test_text, coverage_text)."""
    for pattern in _COVERAGE_SPLIT_PATTERNS:
        if pattern in test_output:
            parts = test_output.split(pattern, 1)
            return parts[0], pattern + parts[1]
    # No coverage output found; return full output as test text
    return test_output, ""


# ---------------------------------------------------------------------------
# AlignmentRunner 메인 클래스
# ---------------------------------------------------------------------------

class AlignmentRunner:
    """Docker SDK 기반 patch-free alignment runner.

    harness를 호출하지 않고 직접 Docker 컨테이너에서 테스트를 실행한다.
    이미 빌드된 ``sweb.eval.{arch}.{instance_id}:latest`` 이미지를 재사용한다.
    """

    def __init__(
        self,
        timeout: int = 600,
        *,
        max_supplemental_pass_tests: int | None = None,
        min_distinct_passing_tests: int = DEFAULT_MIN_DISTINCT_PASSING_TESTS,
        feature_profile: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.feature_profile = feature_profile
        self._client = docker.from_env()
        self.supplemental_pass_collector = SupplementalPassCollector(
            max_supplemental_pass_tests=max_supplemental_pass_tests,
            min_distinct_passing_tests=min_distinct_passing_tests,
            feature_profile=feature_profile,
        )

    # ------------------------------------------------------------------ #
    #  public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        instance: Union[BenchmarkInstance, PrePatchInstanceView],
        generated_test_json_path: str,
        run_id: Optional[str] = None,
        iteration: Optional[int] = None,
        feature_flags: V22FeatureFlags | Mapping[str, bool] | None = None,
        supplemental_context: Mapping[str, Any] | None = None,
        supplemental_clue: Mapping[str, Any] | None = None,
    ) -> AlignmentExecutionResult:
        resolved_flags = (
            feature_flags if isinstance(feature_flags, V22FeatureFlags)
            else resolve_feature_flags(feature_flags)
        )
        supplemental_collector = getattr(
            self,
            "supplemental_pass_collector",
            SupplementalPassCollector(),
        )
        if not (
            resolved_flags.m6_execution_stability
            or resolved_flags.m6_cumulative_fp
            or resolved_flags.m6_sbfl
        ):
            result = self._run_once(
                instance=instance,
                generated_test_json_path=generated_test_json_path,
                run_id=run_id,
                iteration=iteration,
            )
            generated_identity = _load_generated_test_identity(generated_test_json_path)
            canonical_id = _canonical_identity_from_sources(
                generated_identity=generated_identity,
                execution=result,
                fallback=run_id or result.run_id,
            )
            result.execution_id = result.execution_id or result.run_id
            result.canonical_test_id = canonical_id
            result.canonical_test_nodeid = generated_identity.get("test_nodeid") or None
            result.test_nodeid = result.canonical_test_nodeid
            return result

        base_run_id = run_id
        executions: List[AlignmentExecutionResult] = []
        generated_identity = _load_generated_test_identity(generated_test_json_path)
        fallback_identity = base_run_id or f"align-{uuid.uuid4().hex[:8]}"
        parent_canonical_test_id: str | None = None

        def execute_once(index: int) -> AlignmentExecutionResult:
            nonlocal parent_canonical_test_id
            indexed_run_id = (
                base_run_id
                if index == 0
                else f"{base_run_id or fallback_identity}-stability-{index + 1}"
            )
            execution = self._run_once(
                instance=instance,
                generated_test_json_path=generated_test_json_path,
                run_id=indexed_run_id,
                iteration=iteration,
            )
            execution.execution_id = indexed_run_id
            execution.canonical_test_nodeid = generated_identity.get("test_nodeid") or None
            execution.test_nodeid = execution.canonical_test_nodeid
            canonical_id = _canonical_identity_from_sources(
                generated_identity=generated_identity,
                execution=execution,
                parent_canonical_test_id=parent_canonical_test_id,
                fallback=fallback_identity,
            )
            execution.canonical_test_id = canonical_id
            execution.parent_execution_id = parent_canonical_test_id if index > 0 else None
            if index == 0:
                parent_canonical_test_id = canonical_id
            executions.append(execution)
            return execution

        if resolved_flags.m6_execution_stability:
            stability_results = verify_execution_stability(execute_once)
            primary_result = executions[0]
            canonical_test_id = primary_result.canonical_test_id or fallback_identity
            primary_result.stability_results = _compact_stability_results(
                stability_results,
                canonical_test_id=canonical_test_id,
            )
        else:
            primary_result = execute_once(0)
            primary_result.stability_results = {}
            canonical_test_id = primary_result.canonical_test_id or fallback_identity

        contract_executions = _execution_records_for_m6_core(
            executions,
            stability_results=primary_result.stability_results,
            use_stability=resolved_flags.m6_execution_stability,
        )
        supplemental_collection: Dict[str, Any] = {
            "schema_version": "m6-supplemental-pass-collection-v1",
            "enabled": bool(resolved_flags.m6_sbfl),
            "attempted_count": 0,
            "valid_pass_count": 0,
            "rejected_count": 0,
            "max_supplemental_pass_tests": supplemental_collector.max_supplemental_pass_tests,
            "min_distinct_passing_tests": supplemental_collector.min_distinct_passing_tests,
            "stop_reason": "not_triggered",
            "candidate_records": [],
            "accepted_records": [],
        }
        cumulative_executions = contract_executions
        previous_state = (
            getattr(self, "_m6_cumulative_fp_state", None)
            if resolved_flags.m6_cumulative_fp
            else None
        )
        current_candidate_hash = str(
            generated_identity.get("generated_patch_sha256")
            or primary_result.generated_patch_sha256
            or candidate_hash_from_identity(generated_identity)
        )
        previous_candidate_hash = str((previous_state or {}).get("candidate_hash") or "")
        previous_instance_id = str((previous_state or {}).get("instance_id") or "")
        if (
            previous_state
            and (
                not previous_instance_id
                or previous_instance_id != str(primary_result.instance_id)
                or not current_candidate_hash
                or not previous_candidate_hash
                or current_candidate_hash != previous_candidate_hash
            )
        ):
            # Generated-candidate execution evidence is candidate-owned.  A
            # changed patch starts a fresh F/P and spectrum lifecycle.
            previous_state = None
            self._m6_cumulative_spectra = []
            self._m6_supplemental_spectrum_hashes = {}
            self._m6_supplemental_attempted_ids = {}
        prior_p = set(
            str(item)
            for item in (previous_state or {}).get("P_set", [])
            if str(item)
        )
        current_sets = build_pre_patch_outcome_sets(contract_executions)
        prior_p.update(current_sets.get("passing_tests", []))
        existing_distinct_spectrum_hashes: list[str] = []
        if getattr(self, "feature_profile", None) == "v37":
            distinct_ids: list[str] = []
            seen_spectra: set[str] = set()
            for record in [
                *(
                    getattr(self, "_m6_cumulative_spectra", [])
                    if isinstance(getattr(self, "_m6_cumulative_spectra", []), list)
                    else []
                ),
                *contract_executions,
            ]:
                if not isinstance(record, Mapping):
                    continue
                status = normalize_pre_patch_execution_status(record)
                if status != ExecutionStatus.PASS:
                    continue
                spectrum_hash = SupplementalPassCollector._spectrum_hash(
                    record,
                    allow_test_named_sut=True,
                )
                if not spectrum_hash or spectrum_hash in seen_spectra:
                    continue
                seen_spectra.add(spectrum_hash)
                distinct_ids.append(
                    _canonical_test_id(record, fallback=f"pass-spectrum-{len(distinct_ids) + 1}")
                )
            prior_p = set(distinct_ids)
            existing_distinct_spectrum_hashes = sorted(seen_spectra)
        if (
            resolved_flags.m6_sbfl
            and normalize_pre_patch_execution_status(primary_result) == ExecutionStatus.FAIL
            and len(prior_p) < supplemental_collector.min_distinct_passing_tests
            and isinstance(supplemental_context, Mapping)
        ):
            generated_identity = _load_generated_test_identity(generated_test_json_path)
            candidate_id = str(
                generated_identity.get("test_id")
                or generated_identity.get("test_nodeid")
                or primary_result.canonical_test_id
                or primary_result.run_id
            )
            candidate_hash = str(
                generated_identity.get("generated_patch_sha256")
                or primary_result.generated_patch_sha256
                or candidate_hash_from_identity(generated_identity)
            )

            def execute_supplemental(candidate: SupplementalTestCandidate) -> Mapping[str, Any]:
                return self._run_supplemental_candidate(
                    instance=instance,
                    candidate=candidate,
                    context=supplemental_context,
                    run_id=f"{run_id or primary_result.run_id}-supplemental-{candidate.discovery_rank}",
                )

            collection = supplemental_collector.collect(
                context=supplemental_context,
                clue=supplemental_clue or {},
                instance_id=primary_result.instance_id,
                candidate_id=candidate_id,
                candidate_hash=candidate_hash,
                outer_iteration=iteration,
                attempted_test_ids=(
                    [
                        primary_result.canonical_test_id or primary_result.run_id,
                        str(generated_identity.get("test_nodeid") or ""),
                    ]
                    + list(prior_p)
                    + list(
                        (
                            getattr(self, "_m6_supplemental_attempted_ids", {})
                            if isinstance(
                                getattr(self, "_m6_supplemental_attempted_ids", {}),
                                Mapping,
                            )
                            else {}
                        ).get(current_candidate_hash, [])
                    )
                ),
                existing_passing_ids=prior_p,
                accepted_spectrum_hashes=(
                    set(existing_distinct_spectrum_hashes)
                    | set(
                        getattr(self, "_m6_supplemental_spectrum_hashes", {}).get(
                            current_candidate_hash, []
                        )
                    )
                    if isinstance(
                        getattr(self, "_m6_supplemental_spectrum_hashes", {}), Mapping
                    )
                    else set(existing_distinct_spectrum_hashes)
                ),
                execute=execute_supplemental,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            supplemental_collection = collection.to_dict()
            spectrum_state = dict(
                getattr(self, "_m6_supplemental_spectrum_hashes", {})
                if isinstance(getattr(self, "_m6_supplemental_spectrum_hashes", {}), Mapping)
                else {}
            )
            spectrum_state[current_candidate_hash] = sorted(
                {
                    str(record.get("spectrum_hash"))
                    for record in collection.candidate_records
                    if record.get("accepted_into_P") and record.get("spectrum_hash")
                }
                | set(spectrum_state.get(current_candidate_hash, []))
            )
            self._m6_supplemental_spectrum_hashes = spectrum_state
            attempted_state = dict(
                getattr(self, "_m6_supplemental_attempted_ids", {})
                if isinstance(getattr(self, "_m6_supplemental_attempted_ids", {}), Mapping)
                else {}
            )
            attempted_state[current_candidate_hash] = sorted(
                set(attempted_state.get(current_candidate_hash, []))
                | {
                    str(record.get("supplemental_test_nodeid") or "")
                    for record in collection.candidate_records
                    if record.get("supplemental_test_nodeid")
                }
            )
            self._m6_supplemental_attempted_ids = attempted_state
            contract_executions.extend(collection.accepted_records)
            cumulative_executions = contract_executions
        elif resolved_flags.m6_sbfl and normalize_pre_patch_execution_status(primary_result) == ExecutionStatus.FAIL:
            supplemental_collection["stop_reason"] = "missing_target_aware_m2_context"
        elif len(prior_p) >= supplemental_collector.min_distinct_passing_tests:
            supplemental_collection["stop_reason"] = "already_sufficient_current_passes"
        setattr(primary_result, "_m6_supplemental_pass_collection", supplemental_collection)
        core_payload = build_m6_core_execution_data(
            contract_executions,
            previous_state=previous_state,
            iteration=iteration or 1,
        )
        if not resolved_flags.m6_cumulative_fp:
            core_payload["F_set"] = []
            core_payload["P_set"] = []
            core_payload["F_count"] = 0
            core_payload["P_count"] = 0
            core_payload["stable_F_set"] = []
            core_payload["stable_P_set"] = []
            core_payload["F_P_construction"] = {
                "method": "disabled",
                "source": "pre_patch_only",
                "reason": "m6_cumulative_fp feature flag disabled",
            }
        else:
            self._m6_cumulative_fp_state = {
                "F_set": core_payload["F_set"],
                "P_set": core_payload["P_set"],
                "stable_F_set": core_payload["stable_F_set"],
                "stable_P_set": core_payload["stable_P_set"],
                "candidate_hash": current_candidate_hash,
                "instance_id": primary_result.instance_id,
            }
            previous_spectra = getattr(self, "_m6_cumulative_spectra", [])
            cumulative_executions = [
                *[
                    dict(item)
                    for item in previous_spectra
                    if isinstance(item, Mapping)
                ],
                *contract_executions,
            ]
            self._m6_cumulative_spectra = cumulative_executions
        if not resolved_flags.m6_sbfl:
            core_payload["sbfl_active"] = False
            core_payload["sbfl_spectrum"] = []
            core_payload["sbfl_top5_ochiai"] = []
        if not resolved_flags.m6_execution_stability:
            stability_telemetry = m6_execution_stability_exclusion_telemetry()
            core_payload.pop("stability_results", None)
            core_payload.setdefault("feature_execution_telemetry", {})[
                "m6_execution_stability"
            ] = stability_telemetry
            core_payload["m6_execution_stability"] = stability_telemetry
        setattr(primary_result, "_m6_explicit_executions", cumulative_executions)
        setattr(primary_result, "_m6_core_execution_data", core_payload)
        setattr(primary_result, "metadata", {"feature_flags": resolved_flags.to_dict()})
        return primary_result

    def _run_once(
        self,
        instance: Union[BenchmarkInstance, PrePatchInstanceView],
        generated_test_json_path: str,
        run_id: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> AlignmentExecutionResult:
        pre_patch_instance = (
            instance if isinstance(instance, PrePatchInstanceView) else make_pre_patch_view(instance)
        )
        generated_path = Path(generated_test_json_path).resolve()
        if not generated_path.exists():
            raise FileNotFoundError(f"generated_test.json 없음: {generated_path}")

        patch_path = generated_path.with_name("generated_test.patch")
        if not patch_path.exists():
            raise FileNotFoundError(f"generated_test.patch 없음: {patch_path}")

        run_id = run_id or f"align-{pre_patch_instance.instance_id}-{uuid.uuid4().hex[:8]}"
        try:
            gen_test_patch = patch_path.read_text(encoding="utf-8")
            generated_patch_sha256 = _validated_generated_patch_sha256(
                generated_path,
                gen_test_patch,
            )
        except (OSError, ValueError) as error:
            return self._error_result(
                pre_patch_instance.instance_id,
                run_id,
                str(error),
                iteration=iteration,
            )
        phase_timings: Dict[str, Any] = {
            "schema_version": "m6-phase-timings-v1",
            "methodology": {
                "stability_schedule": "3 initial executions plus 2 additional only when initial signatures disagree",
                "setup_reuse": "none_per_isolated_execution",
                "reuse_rationale": "no approved immutable container snapshot contract is available",
            },
            "phases": {},
        }

        def mark_phase(name: str, started_at: float) -> None:
            phase_timings["phases"][name] = round(time.monotonic() - started_at, 3)

        current_stage = "execution_setup"
        test_output = ""
        test_results: Dict[str, str] = {}
        coverage_data: Dict[str, Dict] = {}
        line_spectrum: Dict[str, Any] = {}
        fun2test: List[str] = []
        error_msgs: List[str] = []
        execution_command: Optional[str] = None
        canonical_test_id: Optional[str] = None
        canonical_test_nodeid = ""
        harness_display_name: Optional[str] = None
        observed_test_result_keys: List[str] = []
        has_failure = False
        has_error = False

        # ── 1) instance image 확인 (없으면 자동 빌드) ──
        phase_t0 = time.monotonic()
        from tddbench.harness.test_spec import make_test_spec
        tdd_image_raw = pre_patch_instance.to_tdd_image_raw()
        spec = make_test_spec(tdd_image_raw)
        image_name = spec.instance_image_key

        try:
            self._client.images.get(image_name)
        except docker.errors.ImageNotFound:
            print(f"  [auto-build] Docker image not found: {image_name}")
            print(f"  [auto-build] Building instance image …")
            try:
                from tddbench.harness.docker_build import build_instance_images
                successful, failed = build_instance_images(
                    self._client, [tdd_image_raw],
                    force_rebuild=False, max_workers=1,
                )
                if failed:
                    return self._error_result(
                        pre_patch_instance.instance_id, run_id,
                        f"Docker image build failed: {image_name}",
                        iteration=iteration,
                        phase_timings=phase_timings,
                    )
                # env 이미지 실패 시 instance 빌드가 조용히 스킵될 수 있음
                try:
                    self._client.images.get(image_name)
                except docker.errors.ImageNotFound:
                    return self._error_result(
                        pre_patch_instance.instance_id, run_id,
                        f"Docker image build failed: {image_name} (env image dependency likely failed)",
                        iteration=iteration,
                        phase_timings=phase_timings,
                    )
                print(f"  [auto-build] Image built successfully: {image_name}")
            except Exception as build_err:
                return self._error_result(
                    pre_patch_instance.instance_id, run_id,
                    f"Docker image build error: {build_err}",
                    iteration=iteration,
                    phase_timings=phase_timings,
                )
        mark_phase("image_check_build_sec", phase_t0)

        container = None
        try:
            # ── 2) 컨테이너 생성·시작 ──
            phase_t0 = time.monotonic()
            container_name = f"sweb.align.{pre_patch_instance.instance_id}.{run_id}"
            # 기존 동명 컨테이너 제거
            try:
                old = self._client.containers.get(container_name)
                old.remove(force=True)
            except docker.errors.NotFound:
                pass
            except docker.errors.APIError as e:
                if getattr(e, "status_code", None) != 409:
                    raise
                container_name = f"{container_name}.{uuid.uuid4().hex[:8]}"

            container = self._client.containers.create(
                image_name,
                name=container_name,
                detach=True,
                tty=True,
            )
            container.start()
            mark_phase("container_create_start_sec", phase_t0)

            # ── 3) INITIAL phase: contributing functions 추출 ──
            phase_t0 = time.monotonic()
            contributing = _get_contributing_functions(gen_test_patch)
            mark_phase("collection_contributing_functions_sec", phase_t0)

            # eval.sh의 pip install coverage 이전까지만 실행 (환경 세팅)
            phase_t0 = time.monotonic()
            full_eval = _build_eval_script(
                gen_test_patch, pre_patch_instance.base_commit,
                pre_patch_instance.repo, pre_patch_instance.version,
            )
            setup_part = full_eval.split("python3 -m pip install coverage")[0].strip() + "\n"

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", delete=False, dir="/tmp",
            ) as tmp:
                tmp.write(setup_part)
                tmp_path = Path(tmp.name)

            _copy_to_container(container, tmp_path, Path("/setup.sh"))
            tmp_path.unlink(missing_ok=True)
            _exec_with_timeout(container, "/bin/bash /setup.sh", self.timeout)
            mark_phase("setup_sec", phase_t0)


            # Execute the canonical identity serialized by M5.  Reconstructing
            # it from the unpatched file loses newly generated class names and
            # can inject stale selectors into the command.
            phase_t0 = time.monotonic()
            generated_identity_for_command = _load_generated_test_identity(
                str(generated_path)
            )
            canonical_command_nodeid = str(
                generated_identity_for_command.get("test_nodeid") or ""
            )
            fun2test = (
                [canonical_command_nodeid]
                if canonical_command_nodeid
                else _resolve_fun2test(container, contributing, self.timeout)
            )
            mark_phase("collection_fun2test_resolution_sec", phase_t0)
            phase_t0 = time.monotonic()
            _exec_with_timeout(container, "git clean -fd", self.timeout)
            mark_phase("repository_clean_sec", phase_t0)

            # ── 4) BEFORE-PATCH phase: 테스트 실행 ──
            # repo를 명확히 넘기도록 수정
            phase_t0 = time.monotonic()
            modified_eval = _modify_eval_script(full_eval, pre_patch_instance.repo, fun2test)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", delete=False, dir="/tmp",
            ) as tmp:
                tmp.write(modified_eval)
                tmp_path = Path(tmp.name)

            _copy_to_container(container, tmp_path, Path("/eval.sh"))
            tmp_path.unlink(missing_ok=True)
            mark_phase("eval_script_prepare_sec", phase_t0)

            current_stage = "execution"
            phase_t0 = time.monotonic()
            test_output, timed_out, elapsed, command_exit_code = _exec_with_timeout_status(
                container, "/bin/bash /eval.sh", self.timeout,
            )
            phase_timings["phases"]["execution_coverage_command_sec"] = round(elapsed, 3)
            phase_timings["phases"]["execution_wall_sec"] = round(time.monotonic() - phase_t0, 3)

            if timed_out:
                return self._error_result(
                    pre_patch_instance.instance_id, run_id,
                    f"Timeout after {self.timeout}s",
                    raw_output=test_output,
                    iteration=iteration,
                    phase_timings=phase_timings,
                )

            # ── 5) 결과 파싱 ──
            current_stage = "post_execution_parse"
            phase_t0 = time.monotonic()
            test_text, coverage_text = _split_output(test_output)

            raw_test_results = _parse_test_output(test_text, pre_patch_instance.repo)
            coverage_data = _parse_coverage_text(coverage_text)
            generated_identity_for_spectrum = _load_generated_test_identity(str(generated_path))
            spectrum_test_id = str(
                generated_identity_for_spectrum.get("test_id")
                or generated_identity_for_spectrum.get("test_nodeid")
                or run_id
            )
            generated_identity = _load_generated_test_identity(str(generated_path))
            canonical_test_nodeid = str(generated_identity.get("test_nodeid") or "")
            test_results, identity_error = _select_exact_candidate_result(
                canonical_test_nodeid,
                raw_test_results,
                allow_parameterized_children=True,
            )
            line_spectrum = _parse_line_spectrum_json(
                test_output,
                test_id=spectrum_test_id,
                generated_test_file=str(generated_identity.get("target_test_file") or ""),
            )
            if line_spectrum:
                coverage_data.update(line_spectrum)
            mark_phase("parse_execution_and_coverage_sec", phase_t0)

            has_failure = any(v in ("FAILED", "ERROR") for v in test_results.values())
            has_error = any(v == "ERROR" for v in test_results.values())
            error_origin: str | None = None if line_spectrum else "COVERAGE"

            if identity_error:
                has_error = True
                error_origin = "ADAPTER"
                error_msgs.append(identity_error)
            runtime_errors = _extract_error_details(test_text)
            if not test_results:
                error_origin = error_origin or "PARSING"
                error_msgs.append("No test results parsed from output")
                error_msgs.extend(runtime_errors)

            if runtime_errors and any(
                re.search(r"(?:NameError|AppRegistryNotReady|ImportError|ModuleNotFoundError):", message)
                for message in runtime_errors
            ):
                has_error = True
                error_origin = error_origin or "EXECUTION"
                error_msgs.extend(message for message in runtime_errors if message not in error_msgs)

            # ── 5b) "test not collected" 감지 (module-level skip 등) ──
            not_collected_msg = _detect_test_not_collected(test_text) if not test_results else None
            if not_collected_msg:
                has_error = True
                error_origin = "COLLECTION"
                error_msgs.append(not_collected_msg)
                # Clear malformed results — they are not real test outcomes
                if test_results and not any(
                    v in ("PASSED", "FAILED") for v in test_results.values()
                ):
                    test_results = {}
                    has_failure = False

            observed_test_result_keys = [str(key) for key in raw_test_results]
            harness_display_name = observed_test_result_keys[0] if len(observed_test_result_keys) == 1 else None
            canonical_test_id = generated_identity.get("test_id") or canonical_test_nodeid or run_id
            current_stage = "post_execution_provenance"
            provenance_failure_category: Optional[str] = None
            provenance_exception_type: Optional[str] = None
            provenance_traceback: Optional[str] = None
            try:
                execution_command = _execution_command_provenance(modified_eval)
            except Exception as provenance_error:
                has_error = True
                provenance_failure_category = FailureCategory.PIPELINE_FAILURE.value
                provenance_exception_type = type(provenance_error).__name__
                provenance_traceback = traceback.format_exc()
                error_msgs.append(
                    "M6_METADATA_PROVENANCE_ERROR"
                    f"[{provenance_exception_type}]: {provenance_error}"
                )
            return AlignmentExecutionResult(
                instance_id=pre_patch_instance.instance_id,
                run_id=run_id,
                returncode=command_exit_code if command_exit_code is not None else 1,
                raw_output=test_output,
                iteration=iteration,
                test_results=test_results,
                has_failure=has_failure,
                has_error=has_error,
                coverage_data=coverage_data,
                covered_sut_lines=list(line_spectrum.get("SUT_lines", [])),
                contributing_functions=fun2test,
                error_messages=error_msgs,
                execution_id=run_id,
                canonical_test_id=canonical_test_id,
                test_nodeid=canonical_test_nodeid or None,
                canonical_test_nodeid=canonical_test_nodeid or None,
                harness_display_name=harness_display_name,
                observed_test_result_keys=observed_test_result_keys,
                phase_timings=phase_timings,
                generated_patch_sha256=generated_patch_sha256,
                execution_command=execution_command,
                failure_category=provenance_failure_category,
                error_stage=(
                    "post_execution_provenance"
                    if provenance_failure_category
                    else None
                ),
                exception_type=provenance_exception_type,
                exception_traceback=provenance_traceback,
                error_origin=(
                    "ADAPTER" if provenance_failure_category else error_origin
                ),
                blocking_oracle_flags=validated_v37_blocking_oracle_flags(
                    generated_identity.get("blocking_oracle_flags") or []
                ),
            )

        except Exception as e:
            post_execution_pipeline_error = current_stage in {
                "post_execution_parse",
                "post_execution_provenance",
            }
            return self._error_result(
                pre_patch_instance.instance_id, run_id,
                str(e),
                raw_output=test_output,
                iteration=iteration,
                phase_timings=phase_timings,
                failure_category=(
                    FailureCategory.PIPELINE_FAILURE.value
                    if post_execution_pipeline_error
                    else None
                ),
                error_stage=current_stage,
                exception_type=type(e).__name__,
                exception_traceback=traceback.format_exc(),
                test_results=test_results,
                coverage_data=coverage_data,
                covered_sut_lines=list(line_spectrum.get("SUT_lines", [])),
                contributing_functions=fun2test,
                generated_patch_sha256=generated_patch_sha256,
                execution_command=execution_command,
                canonical_test_id=canonical_test_id,
                canonical_test_nodeid=canonical_test_nodeid or None,
                harness_display_name=harness_display_name,
                observed_test_result_keys=observed_test_result_keys,
                error_origin=(
                    "PARSING"
                    if current_stage == "post_execution_parse"
                    else "ADAPTER"
                    if current_stage == "post_execution_provenance"
                    else "EXECUTION"
                ),
            )
        finally:
            cleanup_t0 = time.monotonic()
            cleanup_status = "skipped_no_container" if container is None else "attempted"
            _cleanup_container(self._client, container)
            mark_phase("cleanup_sec", cleanup_t0)
            phase_timings["cleanup_result"] = {"status": cleanup_status}

    def _run_supplemental_candidate(
        self,
        *,
        instance: Union[BenchmarkInstance, PrePatchInstanceView],
        candidate: SupplementalTestCandidate,
        context: Mapping[str, Any],
        run_id: str,
    ) -> Dict[str, Any]:
        """Execute one discovered repository test on the before-patch view."""
        pre_patch_instance = (
            instance if isinstance(instance, PrePatchInstanceView) else make_pre_patch_view(instance)
        )
        try:
            from tddbench.harness.test_spec import make_test_spec

            image_name = make_test_spec(pre_patch_instance.to_tdd_image_raw()).instance_image_key
            self._client.images.get(image_name)
        except Exception as exc:
            return {
                "execution_status": "ERROR",
                "error_message": f"supplemental image unavailable: {exc}",
                "source_checkout": "pre_patch",
                "test_results": {},
            }

        runner = str((context.get("project_test_style") or {}).get("runner") or candidate.runner)
        directive = candidate.nodeid
        if runner == "django-test":
            node_parts = candidate.nodeid.split("::")
            directive = node_parts[0][:-3].replace("/", ".")
            if directive.startswith("tests."):
                directive = directive[len("tests.") :]
            if len(node_parts) > 1:
                directive += "." + ".".join(node_parts[1:])
        elif runner not in {"pytest", "sympy-bin-test", "unittest"}:
            directive = candidate.test_file
        script = _build_eval_script(
            "",
            pre_patch_instance.base_commit,
            pre_patch_instance.repo,
            pre_patch_instance.version,
            test_directives_override=[directive],
            apply_test_patch=False,
        )
        setup_part = script.split("python3 -m pip install coverage")[0].strip() + "\n"
        container = None
        try:
            container_name = f"sweb.supplemental.{pre_patch_instance.instance_id}.{run_id}"
            try:
                old = self._client.containers.get(container_name)
                old.remove(force=True)
            except docker.errors.NotFound:
                pass
            container = self._client.containers.create(image_name, name=container_name, detach=True, tty=True)
            container.start()
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, dir="/tmp") as tmp:
                tmp.write(setup_part)
                setup_path = Path(tmp.name)
            _copy_to_container(container, setup_path, Path("/setup.sh"))
            setup_path.unlink(missing_ok=True)
            _exec_with_timeout(container, "/bin/bash /setup.sh", self.timeout)
            _exec_with_timeout(container, "git reset --hard " + pre_patch_instance.base_commit, self.timeout)
            _copy_script = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, dir="/tmp")
            try:
                _copy_script.write(script)
                _copy_script.close()
                script_path = Path(_copy_script.name)
                _copy_to_container(container, script_path, Path("/eval.sh"))
            finally:
                Path(_copy_script.name).unlink(missing_ok=True)
            output, timed_out, _ = _exec_with_timeout(container, "/bin/bash /eval.sh", self.timeout)
            if timed_out:
                return {
                    "execution_status": "NOT_RUN",
                    "error_message": "supplemental test timeout",
                    "source_checkout": "pre_patch",
                    "test_results": {},
                    "raw_output": output,
                }
            test_text, coverage_text = _split_output(output)
            parsed = _parse_test_output(test_text, pre_patch_instance.repo)
            selected, identity_error = _select_exact_candidate_result(
                candidate.nodeid,
                parsed,
                allow_parameterized_children=True,
            )
            status = str(selected.get(candidate.nodeid) or "").upper()
            line_spectrum = _parse_line_spectrum_json(
                output,
                test_id=candidate.nodeid,
                generated_test_file=candidate.test_file,
            )
            lines = [
                line for line in line_spectrum.get("SUT_lines", [])
                if _is_sut_source_file(
                    str(line.get("source_file", "")),
                    generated_test_file=candidate.test_file,
                )
            ]
            source_mapping_status = "VALID_PRE_PATCH_SOURCE" if lines else "MISSING_LINE_SPECTRUM"
            return {
                "execution_status": "PASS" if status in {"PASS", "PASSED"} else status or "NOT_RUN",
                "test_results": selected,
                "raw_output": output,
                "coverage_data": {
                    "SUT_lines": lines,
                    "covered_lines_by_test": {candidate.nodeid: lines},
                    **_parse_coverage_text(coverage_text),
                },
                "covered_sut_lines": lines,
                "source_mapping_status": source_mapping_status,
                "source_checkout": "pre_patch",
                "supplemental_test_nodeid": candidate.nodeid,
                "observed_parameterized_nodeids": sorted(
                    str(key)
                    for key in parsed
                    if str(key).startswith(candidate.nodeid + "[")
                ),
                "identity_error": identity_error,
            }
        except Exception as exc:
            return {
                "execution_status": "ERROR",
                "error_message": f"supplemental execution error: {exc}",
                "source_checkout": "pre_patch",
                "test_results": {},
            }
        finally:
            _cleanup_container(self._client, container)

    # ------------------------------------------------------------------ #

    def save(
        self,
        result: AlignmentExecutionResult,
        output_path: str,
        feature_flags: V22FeatureFlags | Mapping[str, bool] | None = None,
    ) -> Dict[str, Dict[str, Any]] | None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(result.to_dict(), path)
        if path.name == "alignment_execution.json":
            artifacts = build_m6_contract_artifacts(result, feature_flags=feature_flags)
            write_json_atomic(artifacts["execution_result"], path.with_name("m6_execution_result.json"))
            write_json_atomic(artifacts["execution_result"], path.with_name("execution_result.json"))
            write_json_atomic(artifacts["coverage_result"], path.with_name("coverage_result.json"))
            write_json_atomic(artifacts["sbfl_result"], path.with_name("sbfl_result.json"))
            return artifacts
        return None

    # ------------------------------------------------------------------ #
    #  helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _error_result(
        instance_id: str,
        run_id: str,
        msg: str,
        raw_output: str = "",
        iteration: Optional[int] = None,
        phase_timings: Mapping[str, Any] | None = None,
        failure_category: str | None = None,
        error_stage: str | None = None,
        exception_type: str | None = None,
        exception_traceback: str | None = None,
        test_results: Mapping[str, str] | None = None,
        coverage_data: Mapping[str, Dict] | None = None,
        covered_sut_lines: Sequence[Mapping[str, Any]] | None = None,
        contributing_functions: Sequence[str] | None = None,
        generated_patch_sha256: str | None = None,
        execution_command: str | None = None,
        canonical_test_id: str | None = None,
        canonical_test_nodeid: str | None = None,
        harness_display_name: str | None = None,
        observed_test_result_keys: Sequence[str] | None = None,
        error_origin: str | None = None,
    ) -> AlignmentExecutionResult:
        return AlignmentExecutionResult(
            instance_id=instance_id,
            run_id=run_id,
            returncode=1,
            raw_output=raw_output,
            iteration=iteration,
            test_results=dict(test_results or {}),
            has_error=True,
            error_messages=[msg],
            coverage_data=dict(coverage_data or {}),
            covered_sut_lines=[dict(item) for item in (covered_sut_lines or [])],
            contributing_functions=list(contributing_functions or []),
            execution_id=run_id,
            canonical_test_id=canonical_test_id or run_id,
            canonical_test_nodeid=canonical_test_nodeid,
            test_nodeid=canonical_test_nodeid,
            harness_display_name=harness_display_name,
            observed_test_result_keys=list(observed_test_result_keys or []),
            phase_timings=phase_timings if isinstance(phase_timings, dict) else dict(phase_timings or {}),
            generated_patch_sha256=generated_patch_sha256,
            execution_command=execution_command,
            failure_category=failure_category,
            error_stage=error_stage,
            exception_type=exception_type,
            exception_traceback=exception_traceback,
            error_origin=error_origin or "EXECUTION",
        )

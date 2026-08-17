"""전체 인스턴스 배치 실행기 (모델 파이프라인 전용).

alignment 루프까지 실행한다. 최종 harness 평가는 별도로
src.evaluator.final_evaluator / result_collector로 수행한다.

Usage:
    python -m src.pipeline.run_batch                             # 전체 449개 (순차)
    python -m src.pipeline.run_batch --workers 4                 # 4개 병렬
    python -m src.pipeline.run_batch --start 0 --end 10          # 인덱스 0~9
    python -m src.pipeline.run_batch --instance_ids astropy__astropy-12907,django__django-10880
    python -m src.pipeline.run_batch --force                     # 완료된 것도 재실행
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import traceback
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.benchmark.instance_loader import TDDInstanceLoader
from src.contracts.feature_flags import V22FeatureFlags, resolve_feature_flags
from src.contracts.feature_profiles import (
    FEATURE_PROFILE_NAMES,
    FeatureFlagResolution,
    V27_FEATURE_PROFILE,
    V27R1_FEATURE_PROFILE,
    V29_FEATURE_PROFILE,
    V30_FEATURE_PROFILE,
    V31_FEATURE_PROFILE,
    V36_FEATURE_PROFILE,
    V37_FEATURE_PROFILE,
    parse_feature_flag_overrides,
    resolve_feature_profile,
)
from src.contracts.final_sets import admitted_to_final_set
from src.contracts.failure import FailureCategory, FailureRecord
from src.contracts.status import legacy_failure_type_to_statuses
from src.evaluator.resolve_policy import resolve_metadata
from src.evaluator.m8_fingerprint import m8_input_fingerprint_is_fresh
from src.models.config import load_model_config
from src.pipeline.run_single import (
    DEFAULT_MAX_FEEDBACK_ITERATIONS,
    process_instance,
    resolve_feedback_iteration_budget,
    validate_v36_experiment_model_key,
)
from src.pipeline.run_provenance import build_run_provenance, provenance_sha256
from src.utils.artifact_hash import sha256_file
from src.utils.file_io import read_json_object, write_json, write_json_atomic


KST = timezone(timedelta(hours=9))

PRODUCTION_FEATURE_STATES: tuple[str, ...] = (
    "REQUESTED",
    "DISABLED",
    "INELIGIBLE",
    "INVOKED",
    "SUCCESS",
    "FALLBACK",
    "FAILED",
    "SKIPPED",
)


def _now_kst_iso() -> str:
    return datetime.now(KST).isoformat()


def _is_completed(output_dir: str) -> bool:
    """Return true only when existing artifacts prove a completed terminal row."""
    out = Path(output_dir)
    alignment = _read_json(out / "alignment_result.json")
    if alignment:
        if not _artifact_belongs_to_dir(out, alignment):
            return False
        failure_type = _normalize_failure_type(
            alignment.get("m7_alignment_status") or alignment.get("failure_type") or "UNKNOWN"
        )
        if failure_type == "ALIGNED":
            final_eval = _read_json(out / "final_evaluation.json")
            return bool(
                final_eval
                and _artifact_belongs_to_dir(out, final_eval)
                and _is_final_eval_fresh(str(out), final_eval)
            )
        if failure_type in {"NOT_FAILED", "NO_COVERAGE", "WEAK_ALIGNMENT", "NOT_VALID"}:
            return True
        if failure_type == "ERROR":
            m7_record = _read_json(out / "m7_decision_record.json")
            return bool(
                m7_record
                and _artifact_belongs_to_dir(out, m7_record)
                and str(m7_record.get("m7_decision_status") or "") == "ERROR"
                and m7_record.get("loop_terminated") is True
            )
        return False
    failure = _read_json(out / "failure.json")
    return bool(failure and _artifact_belongs_to_dir(out, failure) and failure.get("failure_category"))


def _artifact_belongs_to_dir(output_dir: Path, artifact: Mapping[str, Any]) -> bool:
    """Fail closed unless this artifact or its M7 owner proves directory identity."""
    iid = output_dir.name
    payload = artifact.get("payload") if isinstance(artifact.get("payload"), Mapping) else artifact
    artifact_iid = artifact.get("instance_id") or payload.get("instance_id")
    if artifact_iid is not None:
        return str(artifact_iid) == iid

    for companion_name in (
        "m7_decision_record.json",
        "m6_execution_result.json",
        "alignment_execution.json",
    ):
        companion = _read_json(output_dir / companion_name)
        if not companion:
            continue
        companion_payload = (
            companion.get("payload")
            if isinstance(companion.get("payload"), Mapping)
            else companion
        )
        companion_iid = companion.get("instance_id") or companion_payload.get("instance_id")
        if companion_iid is not None:
            return str(companion_iid) == iid
    return False


def _preflight_model_endpoint(model_key: str, timeout: int = 5) -> None:
    """로컬 OpenAI-compatible 서버가 살아있는지 배치 시작 전에 확인한다."""
    config = load_model_config(model_key)
    if config.provider != "local" or not config.base_url:
        return

    url = config.base_url.rstrip("/") + "/models"
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=timeout):
            return
    except HTTPError as e:
        # /models 미지원이어도 HTTP 응답이 왔으면 서버 자체는 살아있다.
        if e.code < 500:
            return
        raise RuntimeError(
            f"Model endpoint responded with HTTP {e.code}: {url}"
        ) from e
    except URLError as e:
        raise RuntimeError(
            f"Model endpoint is not reachable for '{model_key}': {url}\n"
            f"Start the local OpenAI-compatible server or update configs/models.yaml."
        ) from e


def _load_existing_summary(summary_path: str) -> Dict[str, Any]:
    """기존 batch_summary.json을 로드한다. 없으면 빈 구조 반환."""
    p = Path(summary_path)
    if p.exists():
        data = read_json_object(p)
        if isinstance(data, dict):
            data.setdefault("per_instance", [])
            return data
        print(
            f"[WARN] Ignoring unreadable batch summary and rebuilding from artifacts: {p} "
            f"({p.stat().st_size if p.exists() else 0} bytes)"
        )
    return {
        "started_at": _now_kst_iso(),
        "finished_at": None,
        "per_instance": [],
    }


def _save_summary(summary: Dict[str, Any], summary_path: str) -> None:
    """집계 정보를 계산하고 batch_summary.json을 저장.
    메모리의 summary dict도 동일하게 업데이트한다 (호출자가 summary[key]로 읽을 수 있도록).
    """
    results = summary["per_instance"]
    _normalize_summary_entries(results)
    total = len(results)

    counts: Dict[str, int] = {}
    for r in results:
        ft = r.get("failure_type", "UNKNOWN")
        counts[ft] = counts.get(ft, 0) + 1
    aligned = counts.get("ALIGNED", 0)
    resolved = sum(1 for r in results if _canonical_or_legacy_success(r))
    final_eval_count = sum(
        1
        for r in results
        if r.get("final_score") is not None
    )
    skipped = sum(1 for r in results if r.get("skipped", False))
    resolved_rows = [r for r in results if _canonical_or_legacy_success(r)]
    resolved_source_iteration_by_instance = {
        str(row["instance_id"]): int(row["resolved_source_iteration"])
        for row in resolved_rows
        if row.get("instance_id")
        and isinstance(row.get("resolved_source_iteration"), int)
        and 1 <= int(row["resolved_source_iteration"]) <= 5
    }
    resolved_source_iteration_unavailable_instances = [
        str(row.get("instance_id"))
        for row in resolved_rows
        if row.get("instance_id") not in resolved_source_iteration_by_instance
    ]
    if (
        resolved_source_iteration_unavailable_instances
        and summary.get("requested_feature_profile") == "v31"
        and summary.get("finished_at") is not None
    ):
        raise ValueError(
            "completed v31 summary has RESOLVED rows without canonical M7 source iteration: "
            + ", ".join(resolved_source_iteration_unavailable_instances)
        )
    resolved_tests_by_iteration = {
        f"iteration {iteration}": sum(
            1
            for value in resolved_source_iteration_by_instance.values()
            if value == iteration
        )
        for iteration in range(1, 6)
    }
    resolved_source_values = list(resolved_source_iteration_by_instance.values())

    # ── 배치 평균 통계 계산 ──
    token_accounting = _aggregate_token_usage(results)
    # 토큰 평균 (사용량이 양수로 관측된 케이스만; totals/counts below retain failures)
    token_entries = token_accounting["positive_entries"]
    avg_prompt_tokens = round(sum(e["prompt_tokens"] for e in token_entries) / len(token_entries)) if token_entries else None
    avg_completion_tokens = round(sum(e["completion_tokens"] for e in token_entries) / len(token_entries)) if token_entries else None
    avg_total_tokens = round(sum(e["total_tokens"] for e in token_entries) / len(token_entries)) if token_entries else None

    iteration_entries = [
        r.get("iterations")
        for r in results
        if isinstance(r.get("iterations"), (int, float))
    ]
    avg_iterations = round(sum(iteration_entries) / len(iteration_entries), 2) if iteration_entries else None

    # 메모리 dict 업데이트 (호출자가 summary['total'] 등으로 읽을 수 있게)
    summary["total"] = total
    summary["aligned"] = aligned
    summary["resolved"] = resolved
    summary["final_eval_count"] = final_eval_count
    summary["not_failed"] = counts.get("NOT_FAILED", 0)
    summary["not_valid"] = counts.get("NOT_VALID", 0)
    summary.pop("no_fail", None)
    summary["error"] = counts.get("ERROR", 0)
    summary["no_coverage"] = counts.get("NO_COVERAGE", 0)
    summary["weak_alignment"] = counts.get("WEAK_ALIGNMENT", 0)
    summary["skipped"] = skipped
    summary["aligned_rate"] = f"{aligned / total * 100:.1f}%" if total else "0.0%"
    summary["resolve_rate"] = f"{resolved / total * 100:.1f}%" if total else "0.0%"
    summary["failure_type_counts"] = counts
    summary["artifact_extra_instance_count"] = 0
    summary.pop("artifact_extra_instances", None)
    summary.pop("resolved_loss_reason_counts", None)
    summary.pop("partial_flip_after_failed", None)
    summary["avg_prompt_tokens"] = avg_prompt_tokens
    summary["avg_completion_tokens"] = avg_completion_tokens
    summary["avg_total_tokens"] = avg_total_tokens
    summary["total_prompt_tokens"] = token_accounting["total_prompt_tokens"]
    summary["total_completion_tokens"] = token_accounting["total_completion_tokens"]
    summary["total_tokens"] = token_accounting["total_tokens"]
    summary["token_usage_status_counts"] = token_accounting["status_counts"]
    feature_execution_counts = _aggregate_feature_execution_counts(results)
    feature_execution_telemetry = _aggregate_production_feature_telemetry(results)
    summary["feature_execution_counts"] = feature_execution_counts
    summary["feature_execution_telemetry"] = feature_execution_telemetry
    summary["feature_skip_reasons"] = _aggregate_feature_reasons(results, state="SKIPPED")
    summary["feature_fallback_reasons"] = _aggregate_feature_reasons(results, state="FALLBACK")
    summary.pop("max_total_tokens", None)
    summary.pop("p50_total_tokens", None)
    summary.pop("p90_total_tokens", None)
    summary.pop("token_outlier_instances", None)
    summary.pop("avg_patch_line_coverage_percent", None)
    summary.pop("avg_file_coverage", None)
    summary["avg_iterations"] = avg_iterations
    summary["completed_count"] = len({r.get("instance_id") for r in results if r.get("instance_id")})
    summary["unique_row_count"] = summary["completed_count"]
    summary["status_distribution"] = counts
    summary["aligned_count"] = aligned
    summary["m8_evaluated_count"] = final_eval_count
    summary["f_to_p_count"] = sum(1 for r in results if r.get("f_to_p") is True)
    summary["resolved_count"] = resolved
    summary["resolved_tests_by_iteration"] = resolved_tests_by_iteration
    summary["resolved_source_iteration_by_instance"] = resolved_source_iteration_by_instance
    summary["resolved_source_iteration_counts"] = {
        str(iteration): resolved_tests_by_iteration[f"iteration {iteration}"]
        for iteration in range(1, 6)
    }
    summary["resolved_source_iteration_mean"] = (
        round(statistics.mean(resolved_source_values), 4)
        if resolved_source_values
        else None
    )
    summary["resolved_source_iteration_median"] = (
        round(float(statistics.median(resolved_source_values)), 4)
        if resolved_source_values
        else None
    )
    summary["resolved_source_iteration_unavailable_instances"] = (
        resolved_source_iteration_unavailable_instances
    )
    summary["resolved_source_iteration_unavailable_count"] = len(
        resolved_source_iteration_unavailable_instances
    )

    # 집계 통계가 맨 위에 오도록 순서를 명시적으로 구성하여 파일 저장
    ordered: Dict[str, Any] = {
        "total": total,
        "aligned": aligned,
        "resolved": resolved,
        "final_eval_count": final_eval_count,
        "not_failed": counts.get("NOT_FAILED", 0),
        "not_valid": counts.get("NOT_VALID", 0),
        "error": counts.get("ERROR", 0),
        "no_coverage": counts.get("NO_COVERAGE", 0),
        "weak_alignment": counts.get("WEAK_ALIGNMENT", 0),
        "skipped": skipped,
        "aligned_rate": f"{aligned / total * 100:.1f}%" if total else "0.0%",
        "resolve_rate": f"{resolved / total * 100:.1f}%" if total else "0.0%",
        "avg_iterations": avg_iterations,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "avg_total_tokens": avg_total_tokens,
        "total_prompt_tokens": token_accounting["total_prompt_tokens"],
        "total_completion_tokens": token_accounting["total_completion_tokens"],
        "total_tokens": token_accounting["total_tokens"],
        "token_usage_status_counts": token_accounting["status_counts"],
        "feature_execution_counts": feature_execution_counts,
        "feature_execution_telemetry": feature_execution_telemetry,
        "feature_skip_reasons": summary["feature_skip_reasons"],
        "feature_fallback_reasons": summary["feature_fallback_reasons"],
        "failure_type_counts": counts,
        "requested_instance_ids": summary.get("requested_instance_ids"),
        "requested_count": summary.get("requested_count"),
        "completed_count": summary["completed_count"],
        "unique_row_count": summary["unique_row_count"],
        "missing_ids": summary.get("missing_ids"),
        "duplicate_ids": summary.get("duplicate_ids"),
        "status_distribution": summary["status_distribution"],
        "aligned_count": summary["aligned_count"],
        "m8_evaluated_count": summary["m8_evaluated_count"],
        "f_to_p_count": summary["f_to_p_count"],
        "resolved_count": summary["resolved_count"],
        "resolved_tests_by_iteration": resolved_tests_by_iteration,
        "resolved_source_iteration_by_instance": resolved_source_iteration_by_instance,
        "resolved_source_iteration_counts": summary["resolved_source_iteration_counts"],
        "resolved_source_iteration_mean": summary["resolved_source_iteration_mean"],
        "resolved_source_iteration_median": summary["resolved_source_iteration_median"],
        "resolved_source_iteration_unavailable_instances": resolved_source_iteration_unavailable_instances,
        "resolved_source_iteration_unavailable_count": len(
            resolved_source_iteration_unavailable_instances
        ),
        "model_key": summary.get("model_key"),
        "model_endpoint": summary.get("model_endpoint"),
        "model_identifier": summary.get("model_identifier"),
        "context_window": summary.get("context_window"),
        "workers": summary.get("workers"),
        "max_feedback_iterations": summary.get("max_feedback_iterations"),
        "instance_view_root": summary.get("instance_view_root"),
        "commit_sha": summary.get("commit_sha"),
        "requested_feature_profile": summary.get("requested_feature_profile"),
        "explicit_feature_overrides": summary.get("explicit_feature_overrides"),
        "effective_feature_flags": summary.get("effective_feature_flags"),
        "feature_flag_resolution_provenance": summary.get(
            "feature_flag_resolution_provenance"
        ),
        "run_provenance": summary.get("run_provenance"),
        "artifact_extra_instance_count": 0,
        "started_at": summary.get("started_at"),
        "finished_at": summary.get("finished_at"),
        "per_instance": [_compact_summary_entry(r) for r in results],
    }
    required_metadata_keys = {
        "requested_feature_profile",
        "explicit_feature_overrides",
        "effective_feature_flags",
        "feature_flag_resolution_provenance",
    }
    # None 값 필드 제거. Feature-resolution metadata is intentionally retained.
    ordered = {
        k: v
        for k, v in ordered.items()
        if v is not None or k in required_metadata_keys
    }
    summary.pop("artifact_extra_instances", None)
    # per_instance는 None이어도 유지
    ordered["per_instance"] = [_compact_summary_entry(r) for r in results]

    write_json(ordered, summary_path)


def _normalize_summary_entries(results: List[Dict[str, Any]]) -> None:
    """Keep batch summary rows compact and normalize legacy coverage fields."""
    for entry in results:
        entry["failure_type"] = _normalize_failure_type(entry.get("failure_type"))
        entry.pop("patch_sha256", None)
        entry.pop("strict_resolved", None)
        entry.pop("relaxed_resolved", None)
        entry.pop("original_failure_type", None)
        entry.pop("patch_line_coverage", None)
        entry.pop("patch_line_coverage_percent", None)
        entry.pop("patch_line_covered_lines", None)
        entry.pop("patch_line_total_lines", None)


def _aggregate_token_usage(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    status_counts = {"known": 0, "unknown": 0, "no_llm_call": 0}
    positive_entries: List[Dict[str, int]] = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for entry in results:
        usage = entry.get("token_usage")
        status = str(entry.get("token_usage_status") or "").strip()
        if not isinstance(usage, Mapping):
            status = status or "no_llm_call"
            status_counts[status if status in status_counts else "unknown"] += 1
            continue
        normalized = {
            key: int(usage.get(key, 0) or 0)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        # A terminal zero-call retry must not erase earlier model calls that
        # are present in the cumulative totals.
        if normalized["total_tokens"] > 0 and status == "no_llm_call":
            status = "known"
        if not status:
            status = "known" if normalized["total_tokens"] > 0 else "no_llm_call"
        if status not in status_counts:
            status = "unknown"
        status_counts[status] += 1
        for key in totals:
            totals[key] += normalized[key]
        if normalized["total_tokens"] > 0:
            positive_entries.append(normalized)
    return {
        "positive_entries": positive_entries,
        "status_counts": status_counts,
        "total_prompt_tokens": totals["prompt_tokens"],
        "total_completion_tokens": totals["completion_tokens"],
        "total_tokens": totals["total_tokens"],
    }


def _aggregate_feature_execution_counts(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    statuses = (
        "USED",
        "FALLBACK",
        "INACTIVE",
        "NOT_APPLICABLE",
        "ERROR",
        "DISABLED",
        "USED_SUCCESS",
        "USED_FAILED",
        "NOT_ELIGIBLE",
        "NOT_TRIGGERED",
        "DUPLICATE_BLOCKED",
        "BUDGET_EXHAUSTED",
        "MODEL_CONTEXT_OVERFLOW",
        "INFRASTRUCTURE_FAILURE",
        "NO_EFFECT_REPAIR",
    )
    counts: Dict[str, Dict[str, int]] = {}
    for entry in results:
        for feature_name, feature in _iter_feature_telemetry(entry):
            status = str(feature.get("status") or "").upper()
            if not status:
                if feature.get("used") is True:
                    status = "USED"
                elif feature.get("requested") is False or feature.get("enabled") is False:
                    status = "DISABLED"
                elif feature.get("fallback") or feature.get("fallback_used"):
                    status = "FALLBACK"
                else:
                    status = "INACTIVE"
            if status not in statuses:
                status = "USED_FAILED" if status == "FAILED" else "INACTIVE"
            feature_counts = counts.setdefault(feature_name, {key.lower(): 0 for key in statuses})
            feature_counts[status.lower()] += 1
    return counts


def _aggregate_production_feature_telemetry(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for entry in results:
        for feature_name, feature in _iter_feature_telemetry(entry):
            feature_counts = counts.setdefault(
                feature_name,
                {state.lower(): 0 for state in PRODUCTION_FEATURE_STATES},
            )
            for state in _production_states_for_feature(feature):
                feature_counts[state.lower()] += 1
    return counts


def _aggregate_feature_reasons(results: List[Dict[str, Any]], *, state: str) -> Dict[str, Dict[str, int]]:
    reasons: Dict[str, Dict[str, int]] = {}
    target = state.upper()
    for entry in results:
        for feature_name, feature in _iter_feature_telemetry(entry):
            if target not in _production_states_for_feature(feature):
                continue
            reason = _feature_reason(feature, target)
            feature_reasons = reasons.setdefault(feature_name, {})
            feature_reasons[reason] = feature_reasons.get(reason, 0) + 1
    return reasons


def _production_states_for_feature(feature: Mapping[str, Any]) -> list[str]:
    states: list[str] = []
    if feature.get("requested") is True or feature.get("enabled") is True:
        states.append("REQUESTED")
    if feature.get("requested") is False or feature.get("enabled") is False:
        states.append("DISABLED")
    if feature.get("eligible") is False or _legacy_status(feature) in {"NOT_ELIGIBLE", "NOT_APPLICABLE", "NOT_APPLICABLE"}:
        states.append("INELIGIBLE")
    if feature.get("triggered") is True or feature.get("used") is True or int(feature.get("attempt_count") or 0) > 0:
        states.append("INVOKED")
    legacy_status = _legacy_status(feature)
    repair_result = str(feature.get("repair_result") or "").upper()
    if legacy_status in {"USED_SUCCESS", "SUCCESS", "USED"} or repair_result == "USED_SUCCESS":
        states.append("SUCCESS")
    if legacy_status in {"FALLBACK", "NOT_APPLICABLE", "NOT_ELIGIBLE", "DUPLICATE_BLOCKED", "BUDGET_EXHAUSTED"} or bool(feature.get("fallback") or feature.get("fallback_used")):
        states.append("FALLBACK")
    if legacy_status in {"USED_FAILED", "FAILED", "ERROR", "MODEL_CONTEXT_OVERFLOW", "INFRASTRUCTURE_FAILURE", "NO_EFFECT_REPAIR"} or repair_result in {"USED_FAILED", "NO_EFFECT_REPAIR"}:
        states.append("FAILED")
    if feature.get("triggered") is False or legacy_status in {"INACTIVE", "NOT_TRIGGERED", "DISABLED", "NOT_ELIGIBLE", "NOT_APPLICABLE", "DUPLICATE_BLOCKED", "BUDGET_EXHAUSTED"}:
        states.append("SKIPPED")
    if not states:
        states.append("SKIPPED")
    return list(dict.fromkeys(states))


def _legacy_status(feature: Mapping[str, Any]) -> str:
    status = str(feature.get("status") or "").upper()
    if not status:
        status = str(feature.get("repair_result") or "").upper()
    if status == "FAILED":
        return "USED_FAILED"
    if status == "USED" and feature.get("used") is True:
        return "USED"
    return status


def _feature_reason(feature: Mapping[str, Any], state: str) -> str:
    if state == "FALLBACK":
        reason = feature.get("fallback") or feature.get("fallback_reason") or feature.get("reason")
    else:
        reason = (
            feature.get("skip_reason")
            or feature.get("trigger_reason")
            or feature.get("reason")
            or feature.get("terminal_reason")
        )
    return str(reason or "unspecified")


def _iter_feature_telemetry(entry: Mapping[str, Any]):
    containers = [
        entry.get("feature_execution_telemetry"),
        entry.get("optional_features"),
        entry.get("m4_ranking_metadata"),
    ]
    prompt_profile = entry.get("prompt_profile")
    if isinstance(prompt_profile, Mapping):
        containers.append(prompt_profile.get("optional_features"))
    for container in containers:
        if not isinstance(container, Mapping):
            continue
        for name, value in container.items():
            if isinstance(value, Mapping):
                yield str(name), value


def _compact_summary_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return the per-instance row shape written to batch_summary.json."""
    compact = dict(entry)
    compact.pop("avg_file_coverage", None)
    compact.pop("patch_line_coverage", None)
    compact.pop("patch_line_coverage_percent", None)
    compact.pop("patch_line_covered_lines", None)
    compact.pop("patch_line_total_lines", None)
    compact.pop("resolved_reason", None)
    compact.pop("resolved_loss_reason", None)
    compact.pop("diagnostic_flip_to_pass", None)
    compact.pop("has_final_eval", None)
    return compact


def _normalize_failure_type(value: Any) -> str:
    if value == "NO_FAIL":
        return "NOT_FAILED"
    if value == "NOT_COLLECTED":
        return "NOT_VALID"
    return str(value or "UNKNOWN")


def _status_fields(failure_type: Any) -> Dict[str, Any]:
    converted = legacy_failure_type_to_statuses(failure_type)
    return {
        "execution_status": converted["execution_status"],
        "validation_status": converted["validation_status"],
        "m7_alignment_status": converted["m7_alignment_status"],
        "diagnostic_only": converted["m7_alignment_status"] != "ALIGNED",
        "final_set_membership": {
            "in_t_final": converted["m7_alignment_status"] == "ALIGNED",
            "in_t_f2p": False,
        },
    }


def _attach_coverage(result: Dict[str, Any], output_dir: str) -> None:
    """실행 완료 후 alignment_execution.json에서 커버리지 정보를 result에 추가한다."""
    exec_path = Path(f"{output_dir}/alignment_execution.json")
    if not exec_path.exists():
        return
    try:
        ae = json.load(exec_path.open(encoding="utf-8"))
        coverage_data = ae.get("coverage_data", {})
        if not coverage_data:
            return

        # Keep file coverage available for the batch-level average only.
        file_covers = [v.get("cover", 0.0) for v in coverage_data.values() if isinstance(v, dict)]
        result["avg_file_coverage"] = round(sum(file_covers) / len(file_covers), 1) if file_covers else 0.0
    except Exception:
        pass


def _current_patch_sha(output_dir: str) -> str | None:
    patch_path = Path(output_dir) / "generated_test.patch"
    if not patch_path.exists():
        return None
    try:
        return sha256_file(patch_path)
    except Exception:
        return None


def _is_final_eval_fresh(
    output_dir: str,
    final_eval: Dict[str, Any],
    feature_flags: V22FeatureFlags | Mapping[str, Any] | None = None,
) -> bool:
    infra_failed = (
        final_eval.get("harness_returncode") not in (None, 0)
        and not final_eval.get("before_patch")
        and not final_eval.get("after_patch")
    )
    return bool(
        not infra_failed
        and m8_input_fingerprint_is_fresh(
            output_dir,
            final_eval,
            feature_flags=feature_flags,
        )
    )


def _resolution_projection(final_eval: Mapping[str, Any]) -> tuple[str, Optional[bool]]:
    """Project canonical M8 evidence without collapsing unavailable outcomes."""
    failure_record = final_eval.get("failure_record")
    failure_category = (
        str(failure_record.get("category") or "")
        if isinstance(failure_record, Mapping)
        else ""
    )
    evaluation_status = str(final_eval.get("evaluation_status") or "")
    failure_status = failure_category or evaluation_status
    if failure_status == FailureCategory.ENVIRONMENT_FAILURE.value:
        return "UNAVAILABLE", None
    if failure_status in {
        FailureCategory.EVALUATION_FAILURE.value,
        FailureCategory.PIPELINE_FAILURE.value,
        "ERROR",
    }:
        return "ERROR", None

    if "f_to_p" not in final_eval:
        legacy_resolved = final_eval.get("resolved")
        if legacy_resolved is None:
            return "NOT_EVALUATED", None
        return (
            ("RESOLVED", True)
            if legacy_resolved is True
            else ("NOT_RESOLVED", False)
        )

    if _canonical_m8_success(final_eval):
        return "RESOLVED", True
    measured = (
        final_eval.get("admitted_to_final_set") is True
        and final_eval.get("match_status") == "MATCHED"
        and final_eval.get("before_patch_outcome") in {"PASS", "FAIL"}
        and final_eval.get("after_patch_outcome") in {"PASS", "FAIL"}
        and evaluation_status == "SUCCESS"
        and final_eval.get("f_to_p") is False
    )
    if measured:
        return "NOT_RESOLVED", False
    return "NOT_EVALUATED", None


def _apply_final_eval(
    entry: Dict[str, Any], final_eval: Dict[str, Any]
) -> Optional[bool]:
    """final_evaluation.json 내용을 batch summary entry에 반영한다."""
    final_score = final_eval.get("final_score", 0.0)
    entry["final_score"] = final_score
    metadata = resolve_metadata(final_eval)
    resolution_status, is_resolved = _resolution_projection(final_eval)
    entry.update(metadata)
    if "f_to_p" in final_eval:
        for key in (
            "admitted_to_final_set",
            "match_status",
            "before_patch_outcome",
            "after_patch_outcome",
            "f_to_p",
            "final_test_count",
            "f_to_p_test_count",
            "f_to_p_rate",
            "patch_hit",
            "patch_hit_numerator",
            "patch_hit_denominator",
            "patch_hit_evidence",
            "patch_hit_rate",
            "patch_hit_rate_f2p",
            "patch_hit_population",
            "patch_hit_test_denominator",
            "patch_hit_rate_t_final_diagnostic",
            "patch_hit_t_final_diagnostic_denominator",
            "checked_coverage",
            "checked_coverage_status",
            "checked_coverage_final_mean",
            "checked_coverage_f2p_mean",
            "EXAM",
            "exam_score",
            "exam_status",
            "exam_provenance",
            "alignment_verdict",
            "admission_path",
            "CC_computed_with_flaky",
            "flaky_flag",
            "flaky_detail",
            "evaluation_status",
            "failure_record",
            "per_test",
        ):
            if key in final_eval:
                entry[key] = final_eval[key]
        entry["resolved"] = is_resolved
        entry["resolution_status"] = resolution_status
        entry["legacy_final_eval_metadata"] = {
            "resolved": final_eval.get("resolved"),
            "final_score": final_eval.get("final_score"),
            "harness_resolved": final_eval.get("harness_resolved"),
            "harness_final_score": final_eval.get("harness_final_score"),
        }
    else:
        entry["resolved"] = is_resolved
        entry["resolution_status"] = resolution_status
    entry["m8_evaluation_status"] = str(final_eval.get("evaluation_status") or "SUPPORTED")
    entry.pop("strict_resolved", None)
    entry.pop("relaxed_resolved", None)
    if final_eval.get("error"):
        entry["final_eval_error"] = final_eval.get("error")
    return is_resolved


def _evaluate_v37_candidate_set(
    *,
    final_evaluator: Any,
    instance_id: str,
    output_dir: Path,
    feature_flags: V22FeatureFlags,
    primary_final_eval: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate every independently ALIGNED v37 candidate exactly as T_final."""
    manifest = _read_json(output_dir / "v37_candidate_set.json") or {}
    records = manifest.get("candidate_records")
    if not isinstance(records, list):
        return []
    primary_sha = str(primary_final_eval.get("generated_patch_sha256") or "")
    evaluations: list[dict[str, Any]] = []
    resolved_root = output_dir.resolve()
    for record in sorted(
        (item for item in records if isinstance(item, Mapping)),
        key=lambda item: int(item.get("scenario_rank") or 0),
    ):
        if record.get("admitted_to_final_set") is not True:
            continue
        identity = record.get("m5_output_identity")
        identity = identity if isinstance(identity, Mapping) else {}
        candidate_sha = str(identity.get("generated_patch_sha256") or "")
        if candidate_sha and candidate_sha == primary_sha:
            evaluation = dict(primary_final_eval)
        else:
            candidate_dir = Path(str(record.get("candidate_dir") or ""))
            candidate_dir = (
                candidate_dir
                if candidate_dir.is_absolute()
                else output_dir / candidate_dir
            ).resolve()
            try:
                candidate_dir.relative_to(resolved_root)
            except ValueError as error:
                raise ValueError(
                    f"v37 candidate directory escapes instance output: {candidate_dir}"
                ) from error
            cached_evaluation = _read_json(candidate_dir / "final_evaluation.json")
            if cached_evaluation and _is_final_eval_fresh(
                str(candidate_dir),
                cached_evaluation,
                feature_flags,
            ):
                evaluation = cached_evaluation
            else:
                evaluation = final_evaluator.evaluate(
                    instance_id,
                    str(candidate_dir),
                    force=True,
                    feature_flags=feature_flags,
                )
        evaluations.append(
            {
                "scenario_id": record.get("scenario_id"),
                "scenario_rank": record.get("scenario_rank"),
                "candidate_dir": record.get("candidate_dir"),
                "evaluation": evaluation,
            }
        )
    write_json_atomic(
        {
            "schema_version": "v37-candidate-final-evaluations-v1",
            "instance_id": instance_id,
            "T_final_count": len(evaluations),
            "candidate_evaluations": evaluations,
        },
        output_dir / "v37_candidate_final_evaluations.json",
    )
    return evaluations


def _apply_v37_candidate_metrics(
    entry: Dict[str, Any], evaluations: Sequence[Mapping[str, Any]]
) -> None:
    """Aggregate test-level v37 metrics across independently admitted candidates."""
    final_evaluations = [
        item.get("evaluation")
        for item in evaluations
        if isinstance(item.get("evaluation"), Mapping)
    ]
    if not final_evaluations:
        return
    f2p = [item for item in final_evaluations if item.get("f_to_p") is True]
    checked = [
        float(item["checked_coverage_final_mean"])
        for item in final_evaluations
        if isinstance(item.get("checked_coverage_final_mean"), (int, float))
        and not isinstance(item.get("checked_coverage_final_mean"), bool)
    ]
    patch_hits = [item for item in f2p if item.get("patch_hit") is True]
    entry["v37_candidate_final_evaluations"] = list(evaluations)
    entry["final_test_count"] = len(final_evaluations)
    entry["f_to_p_test_count"] = len(f2p)
    entry["f_to_p_rate"] = len(f2p) / len(final_evaluations)
    entry["f_to_p"] = bool(f2p)
    entry["resolved"] = bool(f2p)
    entry["resolution_status"] = "RESOLVED" if f2p else "UNRESOLVED"
    entry["patch_hit_test_denominator"] = len(f2p)
    entry["patch_hit_rate_f2p"] = (
        len(patch_hits) / len(f2p) if f2p else None
    )
    entry["checked_coverage_final_mean"] = (
        sum(checked) / len(checked) if checked else None
    )
    entry["per_test"] = [
        test
        for item in final_evaluations
        for test in (item.get("per_test") or [])
        if isinstance(test, Mapping)
    ]


def _canonical_m8_success(entry: Mapping[str, Any]) -> bool:
    failure_record = entry.get("failure_record")
    excluded = (
        isinstance(failure_record, Mapping)
        and failure_record.get("included_in_aggregate_metrics") is False
    )
    return (
        entry.get("admitted_to_final_set") is True
        and entry.get("match_status") == "MATCHED"
        and entry.get("before_patch_outcome") == "FAIL"
        and entry.get("after_patch_outcome") == "PASS"
        and entry.get("evaluation_status") == "SUCCESS"
        and entry.get("f_to_p") is True
        and not excluded
    )


def _canonical_or_legacy_success(entry: Mapping[str, Any]) -> bool:
    if "f_to_p" in entry:
        return _canonical_m8_success(entry)
    return bool(entry.get("resolved"))


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    return read_json_object(path)


def _token_usage_from_artifacts(output_dir: Path, previous: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if previous.get("token_usage"):
        return previous.get("token_usage")
    generated = _read_json(output_dir / "generated_test.json") or {}
    token_usage = generated.get("token_usage")
    return token_usage if isinstance(token_usage, dict) else None


def _token_usage_status_from_artifacts(output_dir: Path, previous: Dict[str, Any]) -> Optional[str]:
    status = previous.get("token_usage_status")
    if isinstance(status, str) and status.strip():
        return status
    generated = _read_json(output_dir / "generated_test.json") or {}
    status = generated.get("token_usage_status")
    if isinstance(status, str) and status.strip():
        return status
    prompt_profile = generated.get("prompt_profile")
    if isinstance(prompt_profile, Mapping):
        status = prompt_profile.get("token_usage_status")
        if isinstance(status, str) and status.strip():
            return status
    return None


def _payload(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    payload = data.get("payload")
    return payload if isinstance(payload, dict) else data


def _m6_execution_status_from_artifacts(output_dir: Path, fallback: Any) -> str:
    for name in ("m6_execution_result.json", "execution_result.json"):
        data = _payload(_read_json(output_dir / name))
        status = data.get("execution_status")
        if status:
            return str(status)
        test_results = data.get("test_results")
        if isinstance(test_results, dict) and test_results:
            values = {str(value).upper() for value in test_results.values()}
            if "ERROR" in values:
                return "ERROR"
            if values & {"FAILED", "FAIL"}:
                return "FAIL"
            if values <= {"PASSED", "PASS"}:
                return "PASS"
    converted = legacy_failure_type_to_statuses(fallback)
    return str(converted["execution_status"])


def _m8_evaluation_status_from_artifacts(output_dir: Path) -> str:
    final_eval = _read_json(output_dir / "final_evaluation.json")
    if not final_eval:
        return "NOT_ADMITTED"
    return str(final_eval.get("evaluation_status") or "SUPPORTED")


def _resolved_source_iteration_from_artifacts(
    output_dir: Path,
    alignment: Mapping[str, Any],
) -> tuple[int | None, str]:
    """Validate the M7 source iteration without consulting M8 or directory order."""
    m7_record = _read_json(output_dir / "m7_decision_record.json")
    if not m7_record or not _artifact_belongs_to_dir(output_dir, m7_record):
        return None, "M7_DECISION_PROVENANCE_UNAVAILABLE"
    if str(m7_record.get("m7_decision_status") or "") != "ALIGNED":
        return None, "M7_DECISION_NOT_ALIGNED"
    raw_iteration = m7_record.get("outer_iteration")
    if isinstance(raw_iteration, bool) or not isinstance(raw_iteration, int):
        return None, "M7_OUTER_ITERATION_UNAVAILABLE"
    if raw_iteration not in range(1, 6):
        return None, "M7_OUTER_ITERATION_OUT_OF_RANGE"
    alignment_iteration = alignment.get("iteration")
    if isinstance(alignment_iteration, bool) or alignment_iteration != raw_iteration:
        return None, "M7_ALIGNMENT_ITERATION_MISMATCH"
    expected_source = f"iteration_{raw_iteration:03d}"
    m7_source = m7_record.get("source_iteration")
    alignment_source = alignment.get("source_iteration")
    if m7_source != expected_source or alignment_source != expected_source:
        return None, "M7_SOURCE_ITERATION_UNAVAILABLE_OR_INCONSISTENT"
    return raw_iteration, "AVAILABLE"


def _entry_from_artifacts(
    iid: str,
    output_dir: str | Path,
    previous: Optional[Dict[str, Any]] = None,
    skipped: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Rebuild one summary row from per-instance artifacts.

    batch_summary.json is derived state. This keeps resume/skip/final-eval paths
    from trusting stale rows after alignment_result.json or generated_test.patch
    changed.
    """
    previous = previous or {}
    out = Path(output_dir)
    alignment = _read_json(out / "alignment_result.json")
    if not alignment:
        failure = _read_json(out / "failure.json")
        if failure and _artifact_belongs_to_dir(out, failure):
            return _entry_from_failure_artifact(iid, failure, previous=previous, skipped=skipped)
        return None
    if out.name != iid or not _artifact_belongs_to_dir(out, alignment):
        return None

    failure_type = _normalize_failure_type(alignment.get("failure_type", "UNKNOWN"))
    status_fields = _status_fields(failure_type)
    m6_execution_status = _m6_execution_status_from_artifacts(out, failure_type)
    if status_fields["execution_status"] == "NOT_RUN" and m6_execution_status != "NOT_RUN":
        status_fields["execution_status"] = m6_execution_status
    entry: Dict[str, Any] = {
        "instance_id": iid,
        "failure_type": failure_type,
        **status_fields,
        "m6_execution_status": m6_execution_status,
        "m7_alignment_status": alignment.get("m7_alignment_status") or status_fields["m7_alignment_status"],
        "m8_evaluation_status": _m8_evaluation_status_from_artifacts(out),
        "failure_type_detail": (
            alignment.get("failure_type_detail")
            or (alignment.get("score_breakdown") or {}).get("failure_type_detail", "")
        ),
        "iterations": alignment.get("iterations", alignment.get("iteration", 0)),
        "error": None,
        "elapsed_sec": previous.get("elapsed_sec", 0),
        "skipped": previous.get("skipped", False) if skipped is None else skipped,
    }
    if failure_type == "ALIGNED":
        source_iteration, provenance_status = _resolved_source_iteration_from_artifacts(
            out, alignment
        )
        entry["resolved_source_iteration_status"] = provenance_status
        if source_iteration is not None:
            entry["resolved_source_iteration"] = source_iteration
    breakdown = alignment.get("score_breakdown") or {}
    for key in ("bug_fail_score", "coverage_score", "issue_alignment_score"):
        if key in breakdown:
            entry[key] = breakdown[key]

    if failure_type in {"ERROR", "NOT_VALID"}:
        entry["error"] = alignment.get("diagnosis")

    token_usage = _token_usage_from_artifacts(out, previous)
    if token_usage:
        entry["token_usage"] = token_usage
    token_usage_status = _token_usage_status_from_artifacts(out, previous)
    if token_usage_status:
        entry["token_usage_status"] = token_usage_status
    if previous.get("token_usage_scope"):
        entry["token_usage_scope"] = previous["token_usage_scope"]
    feature_telemetry = _read_json(out / "feature_execution_telemetry.json")
    if feature_telemetry:
        entry["feature_execution_telemetry"] = feature_telemetry

    _attach_coverage(entry, str(out))

    if failure_type != "ALIGNED":
        return entry

    final_eval = _read_json(out / "final_evaluation.json")
    if final_eval and _is_final_eval_fresh(str(out), final_eval):
        _apply_final_eval(entry, final_eval)
    elif final_eval:
        entry["resolved"] = None
        entry["resolution_status"] = "NOT_EVALUATED"
        entry["m8_evaluation_status"] = "STALE"
    else:
        entry["resolved"] = None
        entry["resolution_status"] = "NOT_EVALUATED"
        entry["m8_evaluation_status"] = "NOT_EVALUATED"

    return entry


def _entry_from_failure_artifact(
    iid: str,
    failure: Mapping[str, Any],
    *,
    previous: Optional[Dict[str, Any]] = None,
    skipped: Optional[bool] = None,
) -> Dict[str, Any]:
    previous = previous or {}
    failure_type = _normalize_failure_type(failure.get("failure_type") or "ERROR")
    status_fields = _status_fields(failure_type)
    return {
        "instance_id": str(failure.get("instance_id") or iid),
        "failure_type": failure_type,
        **status_fields,
        "failure_category": str(failure.get("failure_category") or "PIPELINE_FAILURE"),
        "failure_type_detail": str(failure.get("error_category") or failure.get("failure_type_detail") or ""),
        "execution_status": "ERROR",
        "validation_status": "NOT_RUN",
        "m6_execution_status": "ERROR",
        "m7_alignment_status": None,
        "m8_evaluation_status": "NOT_ADMITTED",
        "iterations": int(failure.get("iterations") or 0),
        "error": str(failure.get("message") or failure.get("error_message") or ""),
        "elapsed_sec": previous.get("elapsed_sec", failure.get("elapsed_sec", 0)),
        "skipped": previous.get("skipped", False) if skipped is None else skipped,
        "diagnostic_only": True,
        "final_set_membership": {"in_t_final": False, "in_t_f2p": False},
    }


def _upsert_summary_entry(summary: Dict[str, Any], entry: Dict[str, Any]) -> None:
    for i, existing in enumerate(summary["per_instance"]):
        if existing.get("instance_id") == entry.get("instance_id"):
            summary["per_instance"][i] = entry
            return
    summary["per_instance"].append(entry)


def _sync_summary_from_artifacts(
    summary: Dict[str, Any],
    model_output_root: str | Path,
    instance_ids: Optional[List[str]] = None,
) -> None:
    """Refresh summary rows from alignment/final-eval artifacts.

    The summary file is never allowed to be the source of truth for status.
    Existing rows keep their order and timing fields, but status/final-eval
    fields are rebuilt from current artifacts and patch hashes.
    """
    root = Path(model_output_root)
    previous_by_id = {
        r.get("instance_id"): r
        for r in summary.get("per_instance", [])
        if r.get("instance_id")
    }
    ordered_ids: List[str] = []
    seen = set()

    def add_id(iid: str) -> None:
        if iid and iid not in seen:
            seen.add(iid)
            ordered_ids.append(iid)

    requested = list(instance_ids or [])
    requested_set = set(requested)
    if requested:
        for iid in requested:
            add_id(iid)
    else:
        for row in summary.get("per_instance", []):
            add_id(row.get("instance_id", ""))

    if root.exists():
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "alignment_result.json").exists():
                if not requested_set:
                    add_id(d.name)
    summary.pop("artifact_extra_instances", None)
    summary["artifact_extra_instance_count"] = 0

    refreshed: List[Dict[str, Any]] = []
    for iid in ordered_ids:
        previous = previous_by_id.get(iid, {})
        entry = _entry_from_artifacts(iid, root / iid, previous=previous)
        if entry is not None:
            refreshed.append(entry)
    summary["per_instance"] = [r for r in refreshed if r]


def _exception_failure_result(
    iid: str,
    output_dir: str | Path,
    exc: BaseException,
    elapsed_sec: float,
) -> Dict[str, Any]:
    failure_type = "ERROR"
    failure_category = str(getattr(exc, "failure_category", FailureCategory.PIPELINE_FAILURE.value))
    error_category = str(getattr(exc, "failure_type", "") or exc.__class__.__name__)
    message = str(exc)
    failure_record = FailureRecord(
        category=FailureCategory.PIPELINE_FAILURE
        if failure_category not in FailureCategory._value2member_map_
        else FailureCategory(failure_category),
        stage="run_batch.process_instance",
        command=None,
        exit_code=None,
        error_message=message,
        retry_count=int(getattr(exc, "retry_count", 0) or 0),
        retry_safe=False,
        included_in_aggregate_metrics=False,
    ).to_dict()
    extra_record = getattr(exc, "to_failure_record", None)
    if callable(extra_record):
        failure_record.update(extra_record())
    failure_payload = {
        "schema_version": "v22",
        "instance_id": iid,
        "failure_type": failure_type,
        "failure_category": failure_category,
        "error_category": error_category,
        "message": message,
        "error_message": message,
        "execution_status": "ERROR",
        "validation_status": "NOT_RUN",
        "m7_alignment_status": None,
        "iterations": 0,
        "elapsed_sec": elapsed_sec,
        "failure_record": failure_record,
        "created_at": _now_kst_iso(),
    }
    write_json(failure_payload, Path(output_dir) / "failure.json")
    return _entry_from_failure_artifact(iid, failure_payload, previous={"elapsed_sec": elapsed_sec}, skipped=False)


def _run_one(
    iid: str,
    output_dir: str,
    model_key: str,
    loader: TDDInstanceLoader,
    feature_flags: V22FeatureFlags,
    history_window: int | None = None,
    feature_profile: str | None = None,
    max_feedback_iterations: int = DEFAULT_MAX_FEEDBACK_ITERATIONS,
    instance_view_root: str | None = None,
) -> Dict[str, Any]:
    """단일 인스턴스를 실행하고 result dict를 반환한다."""
    t0 = time.time()
    try:
        instance = loader.get_instance(iid)
        process_kwargs: Dict[str, Any] = {
            "model_key": model_key,
            "feature_flags": feature_flags,
        }
        if history_window is not None:
            process_kwargs["history_window"] = history_window
        if feature_profile in {V27_FEATURE_PROFILE, V27R1_FEATURE_PROFILE, V29_FEATURE_PROFILE, V30_FEATURE_PROFILE, V31_FEATURE_PROFILE, V36_FEATURE_PROFILE, V37_FEATURE_PROFILE}:
            process_kwargs["feature_profile"] = feature_profile
        # Keep the selected runtime budget explicit for current adapters while
        # retaining compatibility with narrow test/dry-run worker shims.
        try:
            accepts_budget = "max_feedback_iterations" in inspect.signature(process_instance).parameters
        except (TypeError, ValueError):
            accepts_budget = True
        if accepts_budget:
            process_kwargs["max_feedback_iterations"] = max_feedback_iterations
        if instance_view_root is not None:
            process_kwargs["instance_view_root"] = instance_view_root
        result = process_instance(instance, output_dir, **process_kwargs)
        result["elapsed_sec"] = round(time.time() - t0, 1)
        result["skipped"] = False
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n  ✗ ERROR [{iid}]: {e}\n{tb}")
        result = _exception_failure_result(iid, output_dir, e, round(time.time() - t0, 1))
    # 커버리지 정보 추가
    _attach_coverage(result, output_dir)
    out = Path(output_dir)
    result["m6_execution_status"] = _m6_execution_status_from_artifacts(out, result.get("failure_type"))
    if result.get("execution_status") == "NOT_RUN" and result["m6_execution_status"] != "NOT_RUN":
        result["execution_status"] = result["m6_execution_status"]
    result["m7_alignment_status"] = result.get("m7_alignment_status")
    result["m8_evaluation_status"] = _m8_evaluation_status_from_artifacts(out)
    return result


def run_batch(
    instance_ids: List[str],
    force: bool = False,
    model_key: str = "qwen",
    output_root: str = "outputs",
    workers: int = 1,
    smallbatch: bool = False,
    batch_run_id: Optional[str] = None,
    feature_flags: V22FeatureFlags | Mapping[str, Any] | None = None,
    feature_profile: str | None = None,
    explicit_feature_overrides: Mapping[str, Any] | None = None,
    history_window: int | None = None,
    execution_command: Optional[List[str]] = None,
    max_feedback_iterations: int = DEFAULT_MAX_FEEDBACK_ITERATIONS,
    instance_view_root: str | None = None,
) -> Dict[str, Any]:
    """인스턴스 목록을 실행한다. workers > 1 이면 ThreadPoolExecutor로 병렬 실행."""
    _validate_history_window(history_window)
    _validate_max_feedback_iterations(max_feedback_iterations)
    validate_v36_experiment_model_key(feature_profile, model_key)
    requested_max_feedback_iterations = max_feedback_iterations
    resolution = _resolve_batch_feature_flags(
        feature_flags=feature_flags,
        feature_profile=feature_profile,
        explicit_feature_overrides=explicit_feature_overrides,
    )
    resolved_feature_flags = resolution.effective_feature_flags
    if smallbatch:
        batch_run_id = batch_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        model_output_root = f"{output_root}/smallbatch/{batch_run_id}"
    else:
        model_output_root = f"{output_root}/{model_key}"
    summary_path = f"{model_output_root}/batch_summary.json"
    loader = TDDInstanceLoader()
    if feature_profile == V36_FEATURE_PROFILE:
        max_feedback_iterations = resolve_feedback_iteration_budget(
            feature_profile,
            max_feedback_iterations,
        )
    if feature_profile in {V27_FEATURE_PROFILE, V27R1_FEATURE_PROFILE, V29_FEATURE_PROFILE, V30_FEATURE_PROFILE, V31_FEATURE_PROFILE, V36_FEATURE_PROFILE, V37_FEATURE_PROFILE}:
        provenance = build_run_provenance(
            repository_root=Path(__file__).resolve().parents[2],
            feature_profile=feature_profile,
            explicit_feature_overrides=resolution.explicit_feature_overrides,
            model_key=model_key,
            model_config=load_model_config(model_key),
            workers=workers,
            benchmark_path=loader.dataset_path,
            execution_command=execution_command or [sys.executable, *sys.argv],
            max_feedback_iterations=max_feedback_iterations,
            requested_max_feedback_iterations=requested_max_feedback_iterations,
            instance_view_root=instance_view_root,
            effective_feature_flags=resolved_feature_flags.to_dict(),
        )
        provenance_path = Path(model_output_root) / "run_provenance.json"
        write_json_atomic(provenance, provenance_path)
    summary = _load_existing_summary(summary_path)
    summary.update(
        _batch_run_metadata(
            instance_ids,
            model_key=model_key,
            workers=workers,
            max_feedback_iterations=max_feedback_iterations,
            instance_view_root=instance_view_root,
        )
    )
    _sync_summary_from_artifacts(summary, model_output_root, instance_ids)
    summary.update(resolution.metadata())
    if feature_profile in {V27_FEATURE_PROFILE, V27R1_FEATURE_PROFILE, V29_FEATURE_PROFILE, V30_FEATURE_PROFILE, V31_FEATURE_PROFILE, V36_FEATURE_PROFILE, V37_FEATURE_PROFILE}:
        summary["run_provenance"] = {
            "schema_version": provenance["schema_version"],
            "path": str(provenance_path),
            "sha256": provenance_sha256(provenance),
        }

    # 배치를 다시 실행할 때마다 실행 시작 시각을 새로 기록
    summary["started_at"] = _now_kst_iso()
    summary["finished_at"] = None

    summary_lock = threading.Lock()
    _save_summary(summary, summary_path)

    total = len(instance_ids)

    # ── skip 처리 & 실행 대상 분리 ──
    to_run: List[str] = []
    for idx, iid in enumerate(instance_ids, 1):
        output_dir = f"{model_output_root}/{iid}"
        if not force and _is_completed(output_dir):
            print(f"[{idx}/{total}] {iid} — SKIP")
            previous = next(
                (r for r in summary["per_instance"] if r.get("instance_id") == iid),
                {},
            )
            entry = _entry_from_artifacts(iid, output_dir, previous=previous, skipped=True)
            if entry is None:
                entry = {
                    "instance_id": iid,
                    "failure_type": "UNKNOWN",
                    "iterations": 0,
                    "elapsed_sec": 0,
                    "skipped": True,
                    "error": None,
                }
            with summary_lock:
                _upsert_summary_entry(summary, entry)
                _save_summary(summary, summary_path)
        else:
            to_run.append(iid)

    run_total = len(to_run)
    if run_total > 0:
        _preflight_model_endpoint(model_key)

    def _handle_result(result: Dict[str, Any], idx: int) -> None:
        iid = result["instance_id"]
        with summary_lock:
            replaced = False
            for i, r in enumerate(summary["per_instance"]):
                if r["instance_id"] == iid:
                    summary["per_instance"][i] = result
                    replaced = True
                    break
            if not replaced:
                summary["per_instance"].append(result)
            _save_summary(summary, summary_path)
        ft = result.get("failure_type", "?")
        secs = result.get("elapsed_sec", 0)
        print(f"  [{idx}/{run_total}] {iid} → {ft} ({secs}s)")

    # 1차 실행: alignment까지
    if workers <= 1:
        # 순차 실행
        for idx, iid in enumerate(to_run, 1):
            output_dir = f"{model_output_root}/{iid}"
            print(f"\n{'#'*60}\n  [{idx}/{run_total}] {iid}\n{'#'*60}")
            result = _run_one(
                iid,
                output_dir,
                model_key,
                loader,
                resolved_feature_flags,
                history_window=history_window,
                feature_profile=feature_profile,
                max_feedback_iterations=max_feedback_iterations,
                instance_view_root=instance_view_root,
            )
            _handle_result(result, idx)
    else:
        # 병렬 실행
        print(f"\n병렬 실행: {run_total}개 인스턴스, workers={workers}")
        counter = {"n": 0}
        counter_lock = threading.Lock()

        def _task(iid: str) -> Dict[str, Any]:
            output_dir = f"{model_output_root}/{iid}"
            return _run_one(
                iid,
                output_dir,
                model_key,
                loader,
                resolved_feature_flags,
                history_window=history_window,
                feature_profile=feature_profile,
                max_feedback_iterations=max_feedback_iterations,
                instance_view_root=instance_view_root,
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_task, iid): iid for iid in to_run}
            for future in as_completed(futures):
                with counter_lock:
                    counter["n"] += 1
                    idx = counter["n"]
                result = future.result()
                _handle_result(result, idx)

    # 2차 실행: ALIGNED만 final evaluation
    _sync_summary_from_artifacts(summary, model_output_root, instance_ids)
    _save_summary(summary, summary_path)
    from src.evaluator.final_evaluator import FinalEvaluator
    final_evaluator = FinalEvaluator()

    aligned_entries = [
        e for e in summary["per_instance"]
        if admitted_to_final_set({
            "candidate_status": "GENERATED",
            "diagnostic_only": e.get("diagnostic_only", False),
            "m7_alignment_status": e.get("m7_alignment_status") or e.get("failure_type"),
        })
    ]
    aligned_total = len(aligned_entries)
    resolved = sum(1 for e in summary["per_instance"] if _canonical_or_legacy_success(e))
    needs_final_eval = []
    for entry in aligned_entries:
        output_dir = f"{model_output_root}/{entry['instance_id']}"
        existing = _read_json(Path(output_dir) / "final_evaluation.json")
        if not existing or not _is_final_eval_fresh(
            output_dir, existing, resolved_feature_flags
        ):
            needs_final_eval.append(entry["instance_id"])
    if needs_final_eval:
        final_evaluator.ensure_environment()

    print(f"\n{'='*60}")
    print(f"  Final Evaluation — {aligned_total}건 ALIGNED 평가 시작")
    print(f"{'='*60}")

    eval_count = 0
    for i, entry in enumerate(aligned_entries, 1):
        iid = entry["instance_id"]
        output_dir = f"{model_output_root}/{iid}"
        final_eval_path = Path(f"{output_dir}/final_evaluation.json")

        existing_final_eval = None
        if final_eval_path.exists():
            try:
                with open(final_eval_path, "r", encoding="utf-8") as f:
                    existing_final_eval = json.load(f)
            except Exception as e:
                entry["final_eval_error"] = f"failed to load final_evaluation.json: {e}"

        if existing_final_eval and _is_final_eval_fresh(
            output_dir, existing_final_eval, resolved_feature_flags
        ):
            _apply_final_eval(entry, existing_final_eval)
            if feature_profile == "v37":
                candidate_evaluations = _evaluate_v37_candidate_set(
                    final_evaluator=final_evaluator,
                    instance_id=iid,
                    output_dir=Path(output_dir),
                    feature_flags=resolved_feature_flags,
                    primary_final_eval=existing_final_eval,
                )
                _apply_v37_candidate_metrics(entry, candidate_evaluations)
            status = str(entry.get("resolution_status") or "NOT_EVALUATED")
            print(f"  [{i}/{aligned_total}] {iid} — LOAD ({status})")
            eval_count += 1
            resolved = sum(1 for e in summary["per_instance"] if _canonical_or_legacy_success(e))
            summary["resolved"] = resolved
            summary["final_eval_count"] = eval_count
            summary["resolve_rate"] = f"{resolved / summary['total'] * 100:.1f}%" if summary.get("total") else "0.0%"
            _save_summary(summary, summary_path)
            continue
        if existing_final_eval:
            entry["resolved"] = None
            entry["resolution_status"] = "NOT_EVALUATED"
            entry["m8_evaluation_status"] = "STALE"
            entry.pop("final_score", None)
            print(f"  [{i}/{aligned_total}] {iid} — STALE final eval, rerun")

        t0 = time.time()
        print(f"  [{i}/{aligned_total}] {iid} ...", end=" ", flush=True)
        try:
            final_eval = final_evaluator.evaluate(
                iid,
                output_dir,
                force=True,
                feature_flags=resolved_feature_flags,
            )
            elapsed = round(time.time() - t0, 1)
            is_resolved = _apply_final_eval(entry, final_eval)
            if feature_profile == "v37":
                candidate_evaluations = _evaluate_v37_candidate_set(
                    final_evaluator=final_evaluator,
                    instance_id=iid,
                    output_dir=Path(output_dir),
                    feature_flags=resolved_feature_flags,
                    primary_final_eval=final_eval,
                )
                _apply_v37_candidate_metrics(entry, candidate_evaluations)
            if is_resolved is True:
                entry["final_set_membership"] = {"in_t_final": True, "in_t_f2p": True}
            final_score = entry.get("final_score", 0.0)
            label = str(entry.get("resolution_status") or "NOT_EVALUATED")
            print(f"{label}  (score={final_score:.2f}, {elapsed}s)")
        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            entry["final_score"] = 0.0
            entry["final_eval_error"] = str(e)
            entry["resolved"] = None
            entry["resolution_status"] = "ERROR"
            entry["m8_evaluation_status"] = FailureCategory.EVALUATION_FAILURE.value
            entry["failure_record"] = FailureRecord(
                category=FailureCategory.EVALUATION_FAILURE,
                stage="run_batch.final_evaluator",
                command=None,
                exit_code=None,
                error_message=str(e),
                retry_count=0,
                retry_safe=True,
                included_in_aggregate_metrics=False,
            ).to_dict()
            print(f"ERROR: {e}  ({elapsed}s)")

        eval_count += 1
        # 매 인스턴스마다 즉시 저장
        resolved = sum(1 for e in summary["per_instance"] if _canonical_or_legacy_success(e))
        summary["resolved"] = resolved
        summary["final_eval_count"] = eval_count
        summary["resolve_rate"] = f"{resolved / summary['total'] * 100:.1f}%" if summary.get("total") else "0.0%"
        _save_summary(summary, summary_path)

    # 최종 정리 및 저장
    summary["resolved"] = resolved
    summary["final_eval_count"] = eval_count
    summary["resolve_rate"] = f"{resolved / summary['total'] * 100:.1f}%" if summary.get("total") else "0.0%"
    summary["finished_at"] = _now_kst_iso()
    _save_summary(summary, summary_path)

    # ── 최종 통계 출력 ──
    print(f"\n{'='*60}")
    print("  Batch Complete")
    print(f"{'='*60}")
    print(f"  total:                  {summary['total']}")
    print(f"  aligned:                {summary['aligned']}  ({summary['aligned_rate']})")
    print(f"  resolved:               {summary['resolved']} ({summary['resolve_rate']})")
    print(f"  not_failed:             {summary['not_failed']}")
    print(f"  not_valid:              {summary.get('not_valid', summary.get('failure_type_counts', {}).get('NOT_VALID', 0))}")
    print(f"  error:                  {summary['error']}")
    print(f"  no_coverage:            {summary['no_coverage']}")
    print(f"  weak_alignment:         {summary['weak_alignment']}")
    print(f"  skipped:                {summary['skipped']}")

    pt = summary.get('avg_prompt_tokens')
    ct = summary.get('avg_completion_tokens')
    tt = summary.get('avg_total_tokens')
    n_tok = sum(1 for r in summary["per_instance"] if r.get("token_usage", {}).get("total_tokens", 0) > 0)
    print(f"  failure_type_counts:    {summary.get('failure_type_counts', {})}")
    print(f"  avg_prompt_tokens:      {pt if pt is not None else 'N/A'}  (nonzero token_usage rows={n_tok})")
    print(f"  avg_completion_tokens:  {ct if ct is not None else 'N/A'}")
    print(f"  avg_total_tokens:       {tt if tt is not None else 'N/A'}")
    print(f"  summary → {summary_path}")

    return summary


def _batch_run_metadata(
    instance_ids: List[str],
    *,
    model_key: str,
    workers: int,
    max_feedback_iterations: int = DEFAULT_MAX_FEEDBACK_ITERATIONS,
    instance_view_root: str | None = None,
) -> Dict[str, Any]:
    duplicate_ids = sorted({iid for iid in instance_ids if instance_ids.count(iid) > 1})
    model_metadata: Dict[str, Any] = {}
    try:
        config = load_model_config(model_key)
        model_metadata = {
            "model_endpoint": config.base_url,
            "model_identifier": config.model_name,
            "context_window": config.context_window,
        }
    except Exception as exc:
        model_metadata = {"model_metadata_error": str(exc)}
    return {
        "requested_instance_ids": list(instance_ids),
        "requested_count": len(instance_ids),
        "missing_ids": [],
        "duplicate_ids": duplicate_ids,
        "model_key": model_key,
        "workers": workers,
        "max_feedback_iterations": max_feedback_iterations,
        "instance_view_root": instance_view_root,
        "commit_sha": _current_commit_sha(),
        **model_metadata,
    }


def _current_commit_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return os.environ.get("GIT_COMMIT")


def _validate_history_window(history_window: int | None) -> None:
    if history_window is None:
        return
    if not isinstance(history_window, int) or history_window <= 0:
        raise ValueError("history_window must be an explicit positive integer when provided")


def _validate_max_feedback_iterations(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_feedback_iterations must be an integer greater than or equal to 1")


def _positive_int_cli(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer greater than or equal to 1") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be an integer greater than or equal to 1")
    return parsed


def _resolve_batch_feature_flags(
    *,
    feature_flags: V22FeatureFlags | Mapping[str, Any] | None,
    feature_profile: str | None,
    explicit_feature_overrides: Mapping[str, Any] | None,
) -> FeatureFlagResolution:
    if feature_flags is not None and (feature_profile is not None or explicit_feature_overrides):
        raise ValueError(
            "feature_flags cannot be combined with feature_profile or explicit_feature_overrides"
        )
    if isinstance(feature_flags, V22FeatureFlags):
        return FeatureFlagResolution(
            requested_feature_profile=None,
            explicit_feature_overrides={},
            effective_feature_flags=resolve_feature_flags(feature_flags.to_dict(), env={}),
            feature_flag_resolution_provenance={
                "precedence": ["explicit_feature_flags"],
                "source": "run_batch_feature_flags_argument",
            },
        )
    if feature_flags is not None:
        resolved = resolve_feature_flags(feature_flags, env={})
        return FeatureFlagResolution(
            requested_feature_profile=None,
            explicit_feature_overrides=dict(feature_flags),
            effective_feature_flags=resolved,
            feature_flag_resolution_provenance={
                "precedence": ["explicit_feature_flags"],
                "source": "run_batch_feature_flags_argument",
            },
        )
    return resolve_feature_profile(feature_profile, explicit_feature_overrides)


def main():
    parser = argparse.ArgumentParser(description="전체 인스턴스 배치 실행기")
    parser.add_argument("--start", type=int, default=0, help="시작 인덱스 (default: 0)")
    parser.add_argument("--end", type=int, default=None, help="끝 인덱스, exclusive (default: 전체)")
    parser.add_argument(
        "--instance_ids",
        type=str,
        default=None,
        help="실행할 인스턴스 ID (comma-separated)",
    )
    parser.add_argument("--force", action="store_true", help="완료된 인스턴스도 재실행")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        help="사용할 모델 키 (configs/models.yaml 기준, default: qwen)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="병렬 실행 worker 수 (default: 1, 순차)",
    )
    parser.add_argument(
        "--smallbatch",
        action="store_true",
        help="outputs/smallbatch/<실행시간>/<instance_id> 아래에 별도 저장",
    )
    parser.add_argument(
        "--standard-output",
        action="store_true",
        help="부분 실행도 기존 outputs/<model>/<instance_id> 경로에 저장",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="batch output root (default: outputs)",
    )
    parser.add_argument(
        "--feature-profile",
        choices=FEATURE_PROFILE_NAMES,
        default=None,
        help="v22 feature profile (default: legacy-equivalent all-disabled flags)",
    )
    parser.add_argument(
        "--feature-flag",
        action="append",
        default=[],
        metavar="KEY=true|false",
        help="explicit canonical v22 feature flag override; repeatable",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=None,
        help="explicit positive git-history window for M2 churn/age signals",
    )
    parser.add_argument(
        "--max-feedback-iterations",
        type=_positive_int_cli,
        default=DEFAULT_MAX_FEEDBACK_ITERATIONS,
        metavar="N",
        help="parameterized total candidate/alignment passes including the initial pass",
    )
    parser.add_argument(
        "--instance-view-root",
        type=str,
        default=None,
        metavar="PATH",
        help="root for temporary per-instance detached worktrees",
    )
    args = parser.parse_args()
    feature_overrides = parse_feature_flag_overrides(args.feature_flag)

    loader = TDDInstanceLoader()
    all_ids = loader.list_instance_ids()

    if args.instance_ids:
        ids = [x.strip() for x in args.instance_ids.split(",") if x.strip()]
    else:
        ids = all_ids[args.start : args.end]

    explicit_subset = bool(args.instance_ids) or args.start != 0 or args.end is not None
    use_smallbatch = args.smallbatch or (explicit_subset and not args.standard_output)
    batch_run_id = datetime.now().strftime("%Y%m%d_%H%M%S") if use_smallbatch else None

    print(f"실행 대상: {len(ids)}개 인스턴스 (model={args.model})")
    if use_smallbatch:
        print(f"smallbatch output → {args.output_root}/smallbatch/{batch_run_id}")
    if len(ids) <= 10:
        for iid in ids:
            print(f"  - {iid}")

    run_batch(
        ids,
        force=args.force,
        model_key=args.model,
        workers=args.workers,
        smallbatch=use_smallbatch,
        batch_run_id=batch_run_id,
        output_root=args.output_root,
        feature_profile=args.feature_profile,
        explicit_feature_overrides=feature_overrides,
        history_window=args.history_window,
        max_feedback_iterations=args.max_feedback_iterations,
        instance_view_root=args.instance_view_root,
        execution_command=[sys.executable, *sys.argv],
    )


if __name__ == "__main__":
    main()

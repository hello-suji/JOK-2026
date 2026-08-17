"""
최종 재현 테스트 평가기.

alignment 루프가 끝난 후 생성된 최종 테스트를
TDD-Bench harness (full 3-stage: INITIAL→BEFORE-PATCH→AFTER-PATCH)로 실행하여
final_score를 산출한다.

Usage:
    from src.evaluator.final_evaluator import FinalEvaluator

    evaluator = FinalEvaluator()
    result = evaluator.evaluate("astropy__astropy-12907", "outputs/astropy__astropy-12907")
    print(result)  # {"instance_id": ..., "final_score": 1.0, ...}
"""
from __future__ import annotations

import ast
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.benchmark.instance_loader import TDDInstanceLoader
from src.contracts.failure import FailureCategory, FailureRecord
from src.contracts.feature_flags import V22FeatureFlags, resolve_feature_flags
from src.contracts.final_sets import admitted_to_final_set, build_t_f2p, rate
from src.contracts.instance_views import PrePatchInstanceView, make_pre_patch_view
from src.contracts.status import coerce_execution_status, legacy_failure_type_to_statuses
from src.evaluator.dynamic_slice_cc import run_before_patch_dynamic_slice_cc
from src.evaluator.m8_fingerprint import build_m8_input_fingerprint
from src.executor.m8_dynamic_slice_runner import (
    M8DynamicSliceRunner,
    make_m8_dynamic_slice_request,
)
from src.executor.test_runner import ReproductionTestRunner
from src.evaluator.resolve_policy import resolve_metadata
from src.utils.artifact_hash import sha256_file
from src.utils.file_io import read_json_object, write_json_atomic


CC_STATUSES = {
    "SUPPORTED",
    "UNSUPPORTED",
    "TIMEOUT",
    "INSTRUMENTATION_ERROR",
    "COLLECTION_ERROR",
    "EXECUTION_ERROR",
}
M8_NO_FLOW_BACK_GUARANTEE = (
    "M8 evaluation consumes admitted final-test evidence and keeps post-patch "
    "outcomes and golden patch metadata inside M8 artifacts only."
)


def _m7_admission_metadata(output_dir: Path, *, m7_status: str) -> Dict[str, Any]:
    alignment = read_json_object(output_dir / "alignment_result.json") or {}
    path = alignment.get("admission_path")
    if path not in {"DIRECT", "CONSERVATIVE_OVERRIDE"}:
        path = None
    verdict = str(
        alignment.get("alignment_verdict")
        or "NOT_ALIGNED"
    )
    return {"alignment_verdict": verdict, "admission_path": path}


def _m6_flaky_metadata(output_dir: Path) -> Dict[str, Any]:
    execution = read_json_object(output_dir / "alignment_execution.json") or {}
    stability = execution.get("stability_results")
    stability = stability if isinstance(stability, Mapping) else {}
    flaky = bool(
        execution.get("flaky")
        or execution.get("flaky_flag")
        or stability.get("flaky")
        or stability.get("any_flaky")
    )
    detail = execution.get("flaky_detail")
    if detail is None and stability:
        detail = {
            key: stability.get(key)
            for key in ("runs", "signatures", "agreement_rate", "final_outcome")
            if key in stability
        }
    return {
        "CC_computed_with_flaky": flaky,
        "flaky_flag": flaky,
        "flaky_detail": detail,
        "flaky_evidence_source": "alignment_execution.json",
    }


def _candidate_patch_identity_evidence(
    *,
    generated_test: Mapping[str, Any],
    m6_execution: Mapping[str, Any],
    current_patch_sha256: Optional[str],
    expected_instance_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare current candidate and executable-test identity with M6 evidence.

    Historical identity gaps remain readable diagnostically but cannot grant
    runtime M8 admission.
    """
    candidate_sha = str(
        generated_test.get("generated_patch_sha256")
        or generated_test.get("patch_sha256")
        or ""
    )
    m6_sha = str(m6_execution.get("generated_patch_sha256") or "")
    current_sha = str(current_patch_sha256 or "")
    candidate_identity = stable_test_identity(generated_test)
    candidate_test_id = str(candidate_identity.test_id or "")
    candidate_instance_id = str(generated_test.get("instance_id") or "")
    if not candidate_instance_id and ":" in candidate_test_id:
        candidate_instance_id = candidate_test_id.split(":", 1)[0]
    candidate_nodeid = str(candidate_identity.test_nodeid or "")
    m6_test_id = str(m6_execution.get("canonical_test_id") or m6_execution.get("test_id") or "")
    m6_nodeid = str(m6_execution.get("canonical_test_nodeid") or m6_execution.get("test_nodeid") or "")
    m6_instance_id = str(m6_execution.get("instance_id") or "")
    expected_instance = str(expected_instance_id or generated_test.get("instance_id") or "")
    legacy_without_identity = not candidate_sha and not m6_sha
    matches = bool(
        candidate_sha
        and m6_sha
        and current_sha
        and candidate_sha == m6_sha == current_sha
        and candidate_test_id
        and m6_test_id
        and candidate_test_id == m6_test_id
        and candidate_nodeid
        and m6_nodeid
        and _normalize_nodeid(candidate_nodeid) == _normalize_nodeid(m6_nodeid)
        and expected_instance
        and candidate_instance_id == expected_instance
        and m6_instance_id == expected_instance
    )
    status = "LEGACY_UNAVAILABLE" if legacy_without_identity else "MATCHED"
    if not matches:
        status = "MISMATCH"
    return {
        "status": status,
        "matches": matches,
        "candidate_generated_patch_sha256": candidate_sha or None,
        "m6_executed_patch_sha256": m6_sha or None,
        "current_generated_patch_sha256": current_sha or None,
        "candidate_test_id": candidate_test_id or None,
        "candidate_instance_id": candidate_instance_id or None,
        "m6_canonical_test_id": m6_test_id or None,
        "candidate_test_nodeid": candidate_nodeid or None,
        "m6_canonical_test_nodeid": m6_nodeid or None,
        "expected_instance_id": expected_instance or None,
        "m6_instance_id": m6_instance_id or None,
    }


def _load_m6_candidate_identity(output_dir: Path) -> Dict[str, Any]:
    """Load the M6-owned candidate SHA from canonical/compatibility artifacts."""
    for name in (
        "alignment_execution.json",
        "m6_execution_result.json",
        "execution_result.json",
    ):
        payload = read_json_object(output_dir / name) or {}
        inner = payload.get("payload") if isinstance(payload.get("payload"), Mapping) else payload
        if inner.get("generated_patch_sha256"):
            result = dict(inner)
            result.setdefault("instance_id", payload.get("instance_id"))
            return result
    return {}


@dataclass(frozen=True)
class TestIdentity:
    test_id: Optional[str]
    test_nodeid: Optional[str]
    test_file: Optional[str]
    status: str
    nodeid_provenance: Optional[str] = None


@dataclass(frozen=True)
class TestOutcomeMatch:
    status: str
    test_name: Optional[str]
    before_patch_outcome: Optional[str]
    after_patch_outcome: Optional[str]
    provenance: Optional[str] = None
    before_test_name: Optional[str] = None
    after_test_name: Optional[str] = None
    before_match_provenance: Optional[str] = None
    after_match_provenance: Optional[str] = None


class M8InputValidationError(ValueError):
    """Raised when M8 input cannot be proven to be admitted T_final evidence."""


class FinalEvaluator:
    """검증·개선이 끝난 재현 테스트를 full harness로 평가한다."""

    def __init__(
        self,
        benchmark_root: str = "benchmark/TDD-Bench-Verified",
        max_workers: int = 1,
    ) -> None:
        self.benchmark_root = Path(benchmark_root)
        self.runner = ReproductionTestRunner(
            benchmark_root=benchmark_root,
            max_workers=max_workers,
        )

    def evaluate(
        self,
        instance_id: str,
        output_dir: str,
        force: bool = False,
        m7_status: str = "ALIGNED",
        m8_view: Any | None = None,
        feature_flags: V22FeatureFlags | Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """하나의 인스턴스에 대해 full harness 평가를 실행한다.

        Args:
            instance_id: 벤치마크 인스턴스 ID
            output_dir: generated_test.json 이 있는 디렉토리
            force: True이면 기존 캐시 삭제 후 재실행

        Returns:
            {
                "instance_id": str,
                "final_score": float,
                "before_patch": dict,   # test_name → PASSED/FAILED/ERROR
                "after_patch": dict,    # test_name → PASSED/FAILED/ERROR
                "harness_returncode": int,
                "error": Optional[str],
            }
        """
        output_path = Path(output_dir)
        resolved_flags = _resolve_m8_feature_flags(feature_flags)
        generated_test_path = output_path / "generated_test.json"
        patch_path = generated_test_path.with_suffix(".patch")
        m7_status = self._m7_status_from_artifact(output_path, fallback=m7_status)
        alignment_artifact = read_json_object(output_path / "alignment_result.json") or {}
        alignment_instance = str(alignment_artifact.get("instance_id") or "")
        alignment_candidate = alignment_artifact.get("pass_provenance")
        alignment_candidate = (
            alignment_candidate.get("candidate_identity")
            if isinstance(alignment_candidate, Mapping)
            and isinstance(alignment_candidate.get("candidate_identity"), Mapping)
            else {}
        )
        admission_artifact_consistent = bool(
            alignment_artifact
            and m7_status == "ALIGNED"
            and alignment_artifact.get("alignment_verdict") == "ALIGNED"
            and alignment_artifact.get("admission_path") in {"DIRECT", "CONSERVATIVE_OVERRIDE"}
            and alignment_instance == instance_id
            and alignment_artifact.get("diagnostic_only") is not True
            and alignment_artifact.get("admitted_to_final_set") is not False
            and not (
                isinstance(alignment_artifact.get("final_set_membership"), Mapping)
                and alignment_artifact["final_set_membership"].get("in_t_final") is False
            )
        )
        admission_metadata = _m7_admission_metadata(output_path, m7_status=m7_status)
        flaky_metadata = _m6_flaky_metadata(output_path)
        generated_test = read_json_object(generated_test_path) or {}
        test_identity = stable_test_identity(generated_test)
        if alignment_candidate:
            current_sha = sha256_file(patch_path) if patch_path.exists() else None
            admission_artifact_consistent = admission_artifact_consistent and bool(
                str(alignment_candidate.get("test_id") or "") == str(test_identity.test_id or "")
                and _normalize_nodeid(str(alignment_candidate.get("canonical_test_nodeid") or ""))
                == _normalize_nodeid(str(test_identity.test_nodeid or ""))
                and str(alignment_candidate.get("generated_patch_sha256") or "")
                == str(current_sha or "")
            )
        else:
            admission_artifact_consistent = False
        admission_candidate = {
            "candidate_status": "GENERATED",
            "diagnostic_only": m7_status != "ALIGNED",
            "m7_alignment_status": m7_status,
        }
        if not admission_artifact_consistent or not admitted_to_final_set(admission_candidate):
            failure_record = (
                make_failure_record(
                    FailureCategory.PIPELINE_FAILURE,
                    "m8_candidate_admission",
                    "M7 admission provenance is missing, stale, or contradictory",
                    exit_code=-1,
                )
                if m7_status == "ALIGNED"
                else None
            )
            return {
                "instance_id": instance_id,
                "test_id": test_identity.test_id,
                "test_nodeid": test_identity.test_nodeid,
                "test_identity_status": test_identity.status,
                "m7_status": m7_status,
                "admitted_to_final_set": False,
                "diagnostic_only": True,
                **admission_metadata,
                **flaky_metadata,
                "final_test_count": 0,
                "f_to_p_test_count": 0,
                "f_to_p_rate": None,
                "f_to_p": None,
                "match_status": "UNMATCHED",
                "patch_hit_rate_f2p": None,
                "checked_coverage_mean": None,
                "checked_coverage_final_mean": None,
                "checked_coverage_f2p_mean": None,
                "evaluation_status": (
                    FailureCategory.PIPELINE_FAILURE.value
                    if failure_record
                    else "NOT_ADMITTED"
                ),
                "failure_record": failure_record,
                "error": "candidate is not admitted to T_final",
            }
        if not Path(generated_test_path).exists():
            failure_record = make_failure_record(
                FailureCategory.PIPELINE_FAILURE,
                "m8_candidate_admission",
                f"generated_test.json 없음: {generated_test_path}",
                exit_code=-1,
            )
            return {
                "instance_id": instance_id,
                "test_id": None,
                "test_identity_status": "UNAVAILABLE",
                "final_score": 0.0,
                "before_patch": {},
                "after_patch": {},
                "before_patch_outcome": None,
                "after_patch_outcome": None,
                "m7_status": m7_status,
                "admitted_to_final_set": False,
                "diagnostic_only": True,
                "f_to_p": None,
                "final_test_count": 0,
                "f_to_p_test_count": 0,
                "f_to_p_rate": None,
                "patch_hit": None,
                "patch_hit_rate_f2p": None,
                "checked_coverage": None,
                "checked_coverage_status": "INSTRUMENTATION_ERROR",
                "checked_coverage_mean": None,
                "checked_coverage_final_mean": None,
                "checked_coverage_f2p_mean": None,
                "harness_returncode": -1,
                "error": f"generated_test.json 없음: {generated_test_path}",
                "evaluation_status": failure_record["category"],
                "failure_record": failure_record,
                "patch_sha256": None,
                "generated_patch_sha256": None,
                "m8_execution_artifact": "m8_final_execution_result.json",
            }
        patch_sha256 = sha256_file(patch_path) if patch_path.exists() else None
        candidate_patch_identity = _candidate_patch_identity_evidence(
            generated_test=generated_test,
            m6_execution=_load_m6_candidate_identity(output_path),
            current_patch_sha256=patch_sha256,
            expected_instance_id=instance_id,
        )
        if not candidate_patch_identity["matches"]:
            failure_record = make_failure_record(
                FailureCategory.PIPELINE_FAILURE,
                "m8_candidate_admission",
                "generated candidate patch identity does not match M6 execution provenance",
                exit_code=-1,
            )
            return {
                "instance_id": instance_id,
                "test_id": test_identity.test_id,
                "test_identity_status": test_identity.status,
                "m7_status": m7_status,
                "admitted_to_final_set": False,
                "diagnostic_only": True,
                "f_to_p": None,
                "final_test_count": 0,
                "f_to_p_test_count": 0,
                "f_to_p_rate": None,
                "patch_hit": None,
                "patch_hit_rate_f2p": None,
                "checked_coverage": None,
                "checked_coverage_status": "INSTRUMENTATION_ERROR",
                "checked_coverage_mean": None,
                "checked_coverage_final_mean": None,
                "checked_coverage_f2p_mean": None,
                "evaluation_status": failure_record["category"],
                "failure_record": failure_record,
                "candidate_patch_identity": candidate_patch_identity,
                "error": failure_record["error_message"],
            }

        run_id = f"debug-{instance_id}"

        if force:
            self._clear_cache(instance_id, run_id)

        try:
            execution_result = self.runner.run(
                instance_id=instance_id,
                generated_test_json_path=str(generated_test_path),
                run_id=run_id,
            )
        except FileNotFoundError as e:
            failure_record = make_failure_record(
                FailureCategory.ENVIRONMENT_FAILURE,
                "m8_harness_setup",
                str(e),
                exit_code=-1,
            )
            result = self._failed_evaluation_result(
                instance_id=instance_id,
                test_id=test_identity.test_id,
                test_identity_status=test_identity.status,
                m7_status=m7_status,
                patch_sha256=patch_sha256,
                error=str(e),
                failure_record=failure_record,
            )
            write_json_atomic(result, output_path / "final_evaluation.json")
            return result

        # 실행 결과 저장
        execution_output_path = output_path / "m8_final_execution_result.json"
        self.runner.save(execution_result, str(execution_output_path))

        stdout = execution_result.harness_stdout

        internal_failure_error = (
            execution_result.internal_failure_reason
            if getattr(execution_result, "internal_failure", False)
            else None
        )

        # final_score 파싱
        final_score = self._parse_final_score(stdout)

        # before/after patch 결과 파싱
        before_patch = self._parse_stage_results(stdout, "Before Patch")
        after_patch = self._parse_stage_results(stdout, "After Patch")
        match = match_generated_test_outcomes(
            test_identity.test_id,
            before_patch,
            after_patch,
            test_nodeid=test_identity.test_nodeid,
            test_file=test_identity.test_file,
            accepted_aliases=_generated_test_identity_aliases(
                generated_test, test_identity.test_nodeid
            ),
        )
        before_patch_outcome = match.before_patch_outcome
        after_patch_outcome = match.after_patch_outcome
        measurable_outcomes = {"PASS", "FAIL"}
        f_to_p = (
            is_fail_to_pass(before_patch_outcome, after_patch_outcome)
            if before_patch_outcome in measurable_outcomes
            and after_patch_outcome in measurable_outcomes
            else None
        )
        t_final_candidate = {
            "candidate_status": "GENERATED",
            "diagnostic_only": False,
            "m7_alignment_status": "ALIGNED",
            "pre_patch_outcome": before_patch_outcome,
            "post_patch_outcome": after_patch_outcome,
        }
        t_f2p = build_t_f2p([t_final_candidate])
        f_to_p_test_count = len(t_f2p)
        f_to_p_rate = rate(f_to_p_test_count, 1) if f_to_p is not None else None

        coverage_result = self._checked_coverage_from_artifacts(
            output_path,
            instance_id=instance_id,
            generated_test=generated_test,
            generated_patch_path=patch_path,
            in_f2p=bool(f_to_p),
            feature_flags=resolved_flags,
            test_nodeid=test_identity.test_nodeid,
            m8_view=m8_view,
        )
        patch_hit_evidence = self._patch_hit_from_artifacts(
            output_path,
            instance_id=instance_id,
            m8_view=m8_view,
        )
        patch_hit = (
            patch_hit_evidence.get("patch_hit")
            if isinstance(patch_hit_evidence, Mapping)
            else None
        )
        patch_hit_rate_t_final_diagnostic = (
            1.0 if patch_hit is True else 0.0 if patch_hit is False else None
        )
        error = internal_failure_error
        if (
            match.status == "MATCHED"
            and (
                before_patch_outcome not in measurable_outcomes
                or after_patch_outcome not in measurable_outcomes
            )
        ):
            error = (
                "exact candidate has nonbinary M8 outcome: "
                f"before={before_patch_outcome} after={after_patch_outcome}"
            )
        if execution_result.harness_returncode != 0 and not (before_patch or after_patch):
            stderr = (execution_result.harness_stderr or "").strip()
            error = stderr[-4000:] if stderr else "final evaluation harness failed before stage results"
        failure_record = self._failure_record_for_evaluation(execution_result, match, error)
        if failure_record and failure_record["category"] in {
            FailureCategory.ENVIRONMENT_FAILURE.value,
            FailureCategory.EVALUATION_FAILURE.value,
        }:
            # A harness/API failure leaves the outcome unknown; it is not a
            # measured FAIL→PASS or FAIL→FAIL result.
            f_to_p = None
            f_to_p_test_count = 0
            f_to_p_rate = None

        patch_hit_rate_f2p = (
            patch_hit_rate_t_final_diagnostic if f_to_p is True else None
        )

        checked_coverage_final_mean = coverage_result["checked_coverage_final_mean"]
        localization_evidence = read_json_object(
            output_path / "m8_localization_evidence.json"
        )
        exam_score, exam_status = calculate_exam_from_evidence(
            localization_evidence
        )
        result = {
            "instance_id": instance_id,
            "test_id": test_identity.test_id,
            "test_identity_status": test_identity.status,
            "test_nodeid": test_identity.test_nodeid,
            "test_nodeid_provenance": test_identity.nodeid_provenance,
            "accepted_candidate_identity": {
                "test_id": test_identity.test_id,
                "test_nodeid": test_identity.test_nodeid,
                "test_file": test_identity.test_file,
                "nodeid_provenance": test_identity.nodeid_provenance,
                "representation_aliases": _generated_test_identity_aliases(
                    generated_test, test_identity.test_nodeid
                ),
                "generated_patch_sha256": patch_sha256,
            },
            "matched_test_name": match.test_name,
            "match_status": match.status,
            "match_provenance": match.provenance,
            "before_matched_test_name": match.before_test_name,
            "after_matched_test_name": match.after_test_name,
            "before_match_provenance": match.before_match_provenance,
            "after_match_provenance": match.after_match_provenance,
            "harness_final_score": final_score,
            "harness_resolved": final_score > 0,
            "final_score": final_score,
            "before_patch": before_patch,
            "after_patch": after_patch,
            "before_patch_outcome": before_patch_outcome,
            "after_patch_outcome": after_patch_outcome,
            "m7_status": "ALIGNED",
            "admitted_to_final_set": True,
            **admission_metadata,
            "f_to_p": f_to_p,
            "final_test_count": 1,
            "f_to_p_test_count": f_to_p_test_count,
            "f_to_p_rate": f_to_p_rate,
            "patch_hit": patch_hit,
            "patch_hit_numerator": (
                patch_hit_evidence.get("numerator")
                if isinstance(patch_hit_evidence, Mapping)
                else None
            ),
            "patch_hit_denominator": (
                patch_hit_evidence.get("denominator")
                if isinstance(patch_hit_evidence, Mapping)
                else None
            ),
            "patch_hit_evidence": patch_hit_evidence,
            "patch_hit_rate": patch_hit_rate_f2p,
            "patch_hit_rate_f2p": patch_hit_rate_f2p,
            "patch_hit_population": "T_F2P",
            "patch_hit_test_denominator": f_to_p_test_count,
            "patch_hit_rate_t_final_diagnostic": patch_hit_rate_t_final_diagnostic,
            "patch_hit_t_final_diagnostic_denominator": 1,
            "checked_coverage": coverage_result["checked_coverage"],
            "checked_coverage_status": coverage_result["checked_coverage_status"],
            "checked_coverage_mean": checked_coverage_final_mean,
            "checked_coverage_final_mean": coverage_result["checked_coverage_final_mean"],
            "checked_coverage_f2p_mean": coverage_result["checked_coverage_f2p_mean"],
            "checked_coverage_diagnostics": coverage_result["diagnostics"],
            "checked_coverage_evidence": coverage_result.get("evidence"),
            "EXAM": exam_score,
            "exam_score": exam_score,
            "exam_status": exam_status,
            "exam_provenance": {
                "metric_role": "supplementary",
                "artifact": (
                    "m8_localization_evidence.json"
                    if localization_evidence is not None
                    else None
                ),
                "missing_evidence_blocks_campaign": False,
                "affects_m7_or_admission": False,
                "affects_primary_metrics": False,
            },
            "per_test": [
                {
                    "test_id": test_identity.test_id,
                    "test_nodeid": test_identity.test_nodeid,
                    "before_patch_outcome": before_patch_outcome,
                    "after_patch_outcome": after_patch_outcome,
                    "measurement_valid": failure_record is None,
                    "f_to_p": f_to_p,
                    "patch_hit": patch_hit,
                    "patch_hit_numerator": (
                        patch_hit_evidence.get("numerator")
                        if isinstance(patch_hit_evidence, Mapping)
                        else None
                    ),
                    "patch_hit_denominator": (
                        patch_hit_evidence.get("denominator")
                        if isinstance(patch_hit_evidence, Mapping)
                        else None
                    ),
                    "checked_coverage": checked_coverage_final_mean,
                    "exclusion_reason": (
                        failure_record.get("stage") if failure_record else None
                    ),
                }
            ],
            **flaky_metadata,
            "harness_returncode": execution_result.harness_returncode,
            "error": error,
            "evaluation_status": failure_record["category"] if failure_record else "SUPPORTED",
            "failure_record": failure_record,
            "patch_sha256": patch_sha256,
            "generated_patch_sha256": patch_sha256,
            "m8_execution_artifact": execution_output_path.name,
            "m8_execution_provenance": {
                "instance_id": execution_result.instance_id,
                "run_id": execution_result.run_id,
                "benchmark_root": execution_result.benchmark_root,
                "predictions_path": execution_result.predictions_path,
                "harness_command": list(execution_result.harness_command),
                "harness_returncode": execution_result.harness_returncode,
                "raw_stdout": execution_result.harness_stdout,
                "raw_stderr": execution_result.harness_stderr,
            },
        }
        result["evaluation_status"] = "SUCCESS" if failure_record is None else failure_record["category"]
        if getattr(execution_result, "internal_failure", False):
            result["evaluation_status"] = "ERROR"
        result["internal_failure"] = bool(getattr(execution_result, "internal_failure", False))
        result["internal_failure_reason"] = getattr(execution_result, "internal_failure_reason", None)
        result["internal_failure_artifacts"] = list(
            getattr(execution_result, "internal_failure_artifacts", []) or []
        )
        result["resolved"] = (
            None
            if result["evaluation_status"] in {
                "ERROR",
                FailureCategory.ENVIRONMENT_FAILURE.value,
                FailureCategory.EVALUATION_FAILURE.value,
            }
            else _is_successful_m8_resolution(result)
        )
        result["m8_input_fingerprint"] = build_m8_input_fingerprint(
            output_path,
            feature_flags=resolved_flags,
        )

        # 평가 결과 저장
        eval_result_path = output_path / "final_evaluation.json"
        write_json_atomic(result, eval_result_path)

        return result

    @staticmethod
    def ensure_environment() -> None:
        """Fail before mutating final_evaluation artifacts if Docker is unavailable."""
        try:
            proc = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except FileNotFoundError as e:
            raise RuntimeError("Docker executable not found; final evaluation cannot run") from e
        except Exception as e:
            raise RuntimeError(f"Docker preflight failed: {e}") from e
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"Docker is not available for final evaluation: {msg}")

    @staticmethod
    def _resolve_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
        return resolve_metadata(result)

    @staticmethod
    def _m7_status_from_artifact(output_dir: Path, *, fallback: str) -> str:
        alignment = read_json_object(output_dir / "alignment_result.json")
        if not alignment:
            return "UNKNOWN"
        if alignment.get("m7_alignment_status"):
            return str(alignment["m7_alignment_status"])
        converted = legacy_failure_type_to_statuses(alignment.get("failure_type"))
        return str(converted.get("m7_alignment_status") or alignment.get("failure_type") or "UNKNOWN")

    @staticmethod
    def _checked_coverage_from_artifacts(
        output_dir: Path,
        *,
        instance_id: str = "",
        generated_test: Optional[Mapping[str, Any]] = None,
        generated_patch_path: Optional[Path] = None,
        in_f2p: bool,
        feature_flags: V22FeatureFlags | Mapping[str, Any] | None = None,
        test_nodeid: Optional[str] = None,
        m8_view: Any | None = None,
        container_runner: M8DynamicSliceRunner | None = None,
    ) -> Dict[str, Any]:
        resolved_flags = _resolve_m8_feature_flags(feature_flags)
        if not resolved_flags.m8_dynamic_slice_cc:
            return {
                "checked_coverage": None,
                "checked_coverage_status": "UNSUPPORTED",
                "checked_coverage_mean": None,
                "checked_coverage_final_mean": None,
                "checked_coverage_f2p_mean": None,
                "diagnostics": [
                    {
                        "metric": "checked_coverage",
                        "reason": "m8_dynamic_slice_cc_feature_flag_disabled",
                    }
                ],
                "evidence": None,
            }
        alignment = read_json_object(output_dir / "alignment_execution.json") or {}
        slice_result = None
        executable_nodeid = str(
            test_nodeid
            or alignment.get("canonical_test_nodeid")
            or alignment.get("test_nodeid")
            or ""
        )
        pre_patch_instance = _pre_patch_instance_for_m8_cc(instance_id, m8_view)
        patch_path = generated_patch_path or output_dir / "generated_test.patch"
        if pre_patch_instance and executable_nodeid:
            request = make_m8_dynamic_slice_request(
                instance=pre_patch_instance,
                generated_test=generated_test or {},
                generated_patch_path=patch_path,
                test_nodeid=executable_nodeid,
                output_path=output_dir / "m8_dynamic_slice_cc.json",
            )
            runner = container_runner or M8DynamicSliceRunner()
            slice_result = runner.run(request)
            if slice_result.status == "SUPPORTED":
                evidence = slice_result.to_dict()
                payload = _checked_coverage_payload_from_slice_result(evidence)
                value, status, diagnostic = checked_coverage_from_payload_with_diagnostic(payload)
            else:
                evidence = slice_result.to_dict()
                value = None
                status = slice_result.status
                diagnostic = dict(slice_result.diagnostics)
        else:
            repo_path = _before_patch_repo_path_from_artifacts(output_dir, alignment)
            if repo_path and executable_nodeid:
                slice_result = run_before_patch_dynamic_slice_cc(
                    before_patch_repo_path=repo_path,
                    test_nodeid=executable_nodeid,
                    output_path=output_dir / "m8_dynamic_slice_cc.json",
                )
                if slice_result.status == "SUPPORTED":
                    evidence = slice_result.to_dict()
                    payload = _checked_coverage_payload_from_slice_result(evidence)
                    value, status, diagnostic = checked_coverage_from_payload_with_diagnostic(payload)
                else:
                    evidence = slice_result.to_dict()
                    value = None
                    status = slice_result.status
                    diagnostic = dict(slice_result.diagnostics)
            else:
                evidence = _legacy_checked_coverage_evidence(alignment)
                value, status, diagnostic = checked_coverage_from_payload_with_diagnostic(alignment)
        diagnostics = [{"metric": "checked_coverage", **diagnostic}] if diagnostic else []
        if slice_result is not None:
            diagnostics.append(
                {
                    "metric": "checked_coverage",
                    "backend": slice_result.backend,
                    "artifact": "m8_dynamic_slice_cc.json",
                    "status": slice_result.status,
                }
            )
        checked_coverage_final_mean = mean_valid([value])
        return {
            "checked_coverage": value,
            "checked_coverage_status": status,
            "checked_coverage_mean": checked_coverage_final_mean,
            "checked_coverage_final_mean": checked_coverage_final_mean,
            "checked_coverage_f2p_mean": mean_valid([value]) if in_f2p else None,
            "diagnostics": diagnostics,
            "evidence": evidence,
        }

    @staticmethod
    def _failed_evaluation_result(
        *,
        instance_id: str,
        test_id: Optional[str],
        test_identity_status: str,
        m7_status: str,
        patch_sha256: Optional[str],
        error: str,
        failure_record: dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "instance_id": instance_id,
            "test_id": test_id,
            "test_identity_status": test_identity_status,
            "test_nodeid": None,
            "test_nodeid_provenance": None,
            "matched_test_name": None,
            "match_status": "UNAVAILABLE",
            "match_provenance": None,
            "harness_final_score": 0.0,
            "harness_resolved": False,
            "final_score": 0.0,
            "before_patch": {},
            "after_patch": {},
            "before_patch_outcome": None,
            "after_patch_outcome": None,
            "m7_status": m7_status,
            "admitted_to_final_set": True,
            "f_to_p": None,
            "final_test_count": 1,
            "f_to_p_test_count": 0,
            "f_to_p_rate": None,
            "patch_hit": None,
            "patch_hit_rate_f2p": None,
            "checked_coverage": None,
            "checked_coverage_status": "INSTRUMENTATION_ERROR",
            "checked_coverage_mean": None,
            "checked_coverage_final_mean": None,
            "checked_coverage_f2p_mean": None,
            "harness_returncode": failure_record.get("exit_code", -1),
            "error": error,
            "evaluation_status": (
                "ERROR"
                if failure_record["category"] in {
                    FailureCategory.ENVIRONMENT_FAILURE.value,
                    FailureCategory.EVALUATION_FAILURE.value,
                }
                else failure_record["category"]
            ),
            "failure_record": failure_record,
            "patch_sha256": patch_sha256,
            "generated_patch_sha256": patch_sha256,
            "m8_execution_artifact": "m8_final_execution_result.json",
            "resolved": None
            if failure_record["category"] in {
                FailureCategory.ENVIRONMENT_FAILURE.value,
                FailureCategory.EVALUATION_FAILURE.value,
            }
            else False,
        }

    @staticmethod
    def _failure_record_for_evaluation(
        execution_result: Any,
        match: "TestOutcomeMatch",
        error: Optional[str],
    ) -> Optional[dict[str, Any]]:
        if getattr(execution_result, "internal_failure", False):
            return make_failure_record(
                FailureCategory.EVALUATION_FAILURE,
                "m8_internal_harness_failure",
                str(execution_result.internal_failure_reason or "benchmark harness internal failure"),
                command=list(execution_result.harness_command),
                exit_code=execution_result.harness_returncode,
            )
        if execution_result.harness_returncode == -1 and "timed out" in (
            execution_result.harness_stderr or ""
        ).lower():
            return make_failure_record(
                FailureCategory.ENVIRONMENT_FAILURE,
                "m8_harness_timeout",
                execution_result.harness_stderr,
                command=list(execution_result.harness_command),
                exit_code=execution_result.harness_returncode,
            )
        if error:
            return make_failure_record(
                FailureCategory.EVALUATION_FAILURE,
                "m8_harness_execution",
                error,
                command=list(execution_result.harness_command),
                exit_code=execution_result.harness_returncode,
            )
        if match.status == "UNMATCHED":
            return make_failure_record(
                FailureCategory.EVALUATION_FAILURE,
                "m8_generated_test_match",
                "generated test did not match any final harness test result",
                command=list(execution_result.harness_command),
                exit_code=execution_result.harness_returncode,
            )
        if match.status == "AMBIGUOUS":
            return make_failure_record(
                FailureCategory.EVALUATION_FAILURE,
                "m8_generated_test_match",
                "generated test matched multiple final harness test results",
                command=list(execution_result.harness_command),
                exit_code=execution_result.harness_returncode,
            )
        return None

    def _patch_hit_from_artifacts(
        self,
        output_dir: Path,
        *,
        instance_id: str,
        m8_view: Any | None,
    ) -> Optional[dict[str, Any]]:
        alignment = read_json_object(output_dir / "alignment_execution.json") or {}
        coverage_data = alignment.get("coverage_data")
        if not isinstance(coverage_data, Mapping):
            return None
        patch_text = self._golden_patch_text(instance_id, m8_view)
        if not patch_text:
            return None
        golden_lines = parse_golden_patch_lines(patch_text)
        if not golden_lines:
            return None
        return patch_hit_details_from_coverage(coverage_data, golden_lines)

    def _golden_patch_text(self, instance_id: str, m8_view: Any | None) -> str:
        if m8_view is not None:
            return str(getattr(m8_view, "patch", "") or "")
        try:
            instance = TDDInstanceLoader().get_instance(instance_id)
        except Exception:
            return ""
        return instance.to_m8_evaluation_view().patch

    def _clear_cache(self, instance_id: str, run_id: str) -> None:
        import subprocess
        # Docker 컨테이너 이름 충돌 방지: 기존 debug 컨테이너 제거
        container_name = f"sweb.eval.{instance_id}.{run_id}"
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
        )
        eval_log_dir = self.benchmark_root / "logs" / "run_evaluation" / run_id
        if eval_log_dir.exists():
            shutil.rmtree(eval_log_dir, ignore_errors=True)
        for report_file in self.benchmark_root.glob(f"*{run_id}*.json"):
            report_file.unlink(missing_ok=True)

    @staticmethod
    def _parse_final_score(stdout: str) -> float:
        """harness stdout에서 Final Report의 final_score를 파싱."""
        report_match = re.search(
            r"-+Final Report-+\n(\{.*?\})\n-{10,}",
            stdout, re.DOTALL,
        )
        if report_match:
            try:
                outer = ast.literal_eval(report_match.group(1).strip())
                inner = next(iter(outer.values())) if outer else {}
                return float(inner.get("final_score", 0.0))
            except (ValueError, SyntaxError, StopIteration):
                pass
        return 0.0

    @staticmethod
    def _parse_stage_results(stdout: str, stage_name: str) -> Dict[str, str]:
        """stdout에서 특정 stage (Before Patch / After Patch) 결과를 파싱."""
        pattern = rf"-+{re.escape(stage_name)}-+\s*\n(.+?)\n-+"
        match = re.search(pattern, stdout, re.DOTALL)
        if not match:
            return {}
        text = match.group(1).strip()
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            results: Dict[str, str] = {}
            for m in re.finditer(
                r"'([^']+)':\s*'(PASSED|PASS|FAILED|FAIL|ERROR|SKIP|SKIPPED|XFAIL|XPASS|TIMEOUT|NOT_RUN)'",
                text,
            ):
                results[m.group(1)] = m.group(2)
            return results


def normalize_stage_outcome(stage_results: Mapping[str, Any]) -> str:
    """Normalize a final-evaluation harness stage to PASS, FAIL, or ERROR."""
    values = {str(value).upper() for value in stage_results.values()}
    if not values:
        return "ERROR"
    if "ERROR" in values:
        return "ERROR"
    if "FAILED" in values or "FAIL" in values:
        return "FAIL"
    try:
        statuses = {coerce_execution_status(value) for value in values}
    except ValueError:
        return "ERROR"
    if statuses == {coerce_execution_status("PASS")}:
        return "PASS"
    return "ERROR"


def normalize_single_test_outcome(status: Any) -> str:
    text = str(status or "").upper()
    if text in {"PASSED", "PASS"}:
        return "PASS"
    if text in {"FAILED", "FAIL"}:
        return "FAIL"
    if text == "ERROR":
        return "ERROR"
    if text in {"SKIP", "SKIPPED"}:
        return "SKIP"
    if text in {"XFAIL", "XPASS", "TIMEOUT", "NOT_RUN"}:
        return text
    return "UNKNOWN"


def is_fail_to_pass(before_patch_outcome: str, after_patch_outcome: str) -> bool:
    return (
        coerce_execution_status(before_patch_outcome).value == "FAIL"
        and coerce_execution_status(after_patch_outcome).value == "PASS"
    )


def validate_t_final_input(candidates: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate explicit final-set admission evidence without calling M7.

    M8 accepts only non-diagnostic candidates with canonical ALIGNED M7
    provenance. Duplicate or missing identities are rejected because per-test
    evidence and metrics must remain attributable to one final test.
    """
    accepted: list[dict[str, Any]] = []
    seen_identities: set[str] = set()
    for index, candidate in enumerate(candidates):
        identity = candidate_identity(candidate)
        if not identity:
            raise M8InputValidationError(
                f"candidate at index {index} is missing admission provenance identity"
            )
        if identity in seen_identities:
            raise M8InputValidationError(f"duplicate final-test candidate identity: {identity}")
        seen_identities.add(identity)
        if "m7_alignment_status" not in candidate:
            raise M8InputValidationError(
                f"candidate {identity} is missing m7_alignment_status admission provenance"
            )
        if candidate.get("diagnostic_only") is True:
            raise M8InputValidationError(f"candidate {identity} is diagnostic-only")
        if not admitted_to_final_set(candidate):
            raise M8InputValidationError(f"candidate {identity} is not ALIGNED for T_final")
        if candidate.get("admitted_to_final_set") is False:
            raise M8InputValidationError(
                f"candidate {identity} carries conflicting final-set admission evidence"
            )
        copied = dict(candidate)
        copied["test_id"] = identity
        copied["m7_alignment_status"] = "ALIGNED"
        copied["admitted_to_final_set"] = True
        copied["diagnostic_only"] = False
        accepted.append(copied)
    return accepted


def candidate_identity(candidate: Mapping[str, Any]) -> Optional[str]:
    for key in ("test_id", "generated_test_id", "candidate_id"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def evaluate_m8_core(
    *,
    instance_id: str,
    t_final_candidates: Iterable[Mapping[str, Any]],
    before_patch_evidence: Mapping[str, Any],
    after_patch_evidence: Mapping[str, Any],
    checked_coverage_evidence: Optional[Mapping[str, Mapping[str, Any]]] = None,
    patch_coverage_evidence: Optional[Mapping[str, Mapping[str, Any]]] = None,
    golden_patch_lines: Optional[Mapping[str, set[int]]] = None,
    before_patch_source: str = "m8_independent_pre_patch",
    after_patch_source: str = "m8_independent_post_patch",
    patch_metadata_source: Optional[str] = None,
    feature_flags: V22FeatureFlags | Mapping[str, Any] | None = None,
    localization_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate core M8 metrics from independent per-test evidence.

    The function is intentionally pure: it does not call M7, does not mutate
    candidate records, and treats M8 before/after evidence as separate inputs.
    """
    final_tests = validate_t_final_input(t_final_candidates)
    resolved_flags = _resolve_m8_feature_flags(feature_flags)
    cc_evidence = checked_coverage_evidence or {}
    patch_evidence = patch_coverage_evidence or {}
    diagnostics: list[dict[str, Any]] = []
    per_test: list[dict[str, Any]] = []
    f_to_p_tests: list[str] = []
    patch_hits_t_final: list[Optional[bool]] = []
    patch_hits_f2p: list[Optional[bool]] = []
    cc_final_values: list[Optional[float]] = []
    cc_f2p_values: list[Optional[float]] = []
    invalid_measurement_count = 0

    for candidate in final_tests:
        test_id = candidate["test_id"]
        admission_path = candidate.get("admission_path")
        candidate_instance = str(candidate.get("instance_id") or "")
        if not candidate_instance and ":" in str(test_id):
            candidate_instance = str(test_id).split(":", 1)[0]
        instance_valid = not candidate_instance or candidate_instance == instance_id
        actual_flaky_flag = bool(candidate.get("flaky_flag") or candidate.get("flaky"))
        before = normalize_single_test_outcome(_evidence_outcome(before_patch_evidence, test_id))
        after = normalize_single_test_outcome(_evidence_outcome(after_patch_evidence, test_id))
        evidence_instances = []
        for evidence in (before_patch_evidence.get(test_id), after_patch_evidence.get(test_id)):
            if isinstance(evidence, Mapping) and evidence.get("instance_id"):
                evidence_instances.append(str(evidence.get("instance_id")))
        instance_valid = instance_valid and all(value == instance_id for value in evidence_instances)
        measurement_valid = instance_valid and before in {"PASS", "FAIL"} and after in {"PASS", "FAIL"}
        f_to_p = (before == "FAIL" and after == "PASS") if measurement_valid else None
        if not measurement_valid:
            invalid_measurement_count += 1
            diagnostics.append(
                {
                    "test_id": test_id,
                    "metric": "m8_measurement",
                    "reason": "cross_instance_evidence" if not instance_valid else "nonbinary_candidate_outcome",
                    "before_patch_outcome": before,
                    "after_patch_outcome": after,
                }
            )
        if f_to_p:
            f_to_p_tests.append(test_id)

        if resolved_flags.m8_dynamic_slice_cc:
            test_cc_evidence = cc_evidence.get(test_id, {})
            cc_value, cc_status, cc_diagnostic = checked_coverage_from_payload_with_diagnostic(
                test_cc_evidence
            )
            cc_persisted_evidence = _legacy_checked_coverage_evidence(test_cc_evidence)
        else:
            cc_value = None
            cc_status = "UNSUPPORTED"
            cc_diagnostic = {"reason": "m8_dynamic_slice_cc_feature_flag_disabled"}
            cc_persisted_evidence = None
        if cc_diagnostic:
            diagnostics.append({"test_id": test_id, "metric": "checked_coverage", **cc_diagnostic})
        if cc_value is not None:
            cc_final_values.append(cc_value)
        if f_to_p:
            cc_f2p_values.append(cc_value)

        patch_hit = None
        if golden_patch_lines is None:
            diagnostics.append(
                {
                    "test_id": test_id,
                    "metric": "patch_hit_rate",
                    "reason": "missing_patch_metadata",
                }
            )
        else:
            patch_hit = patch_hit_from_coverage(patch_evidence.get(test_id, {}), golden_patch_lines)
            if patch_hit is None:
                diagnostics.append(
                    {
                        "test_id": test_id,
                        "metric": "patch_hit_rate",
                        "reason": "missing_patch_metadata",
                    }
                )
        if patch_hit is not None:
            patch_hits_t_final.append(patch_hit)
        if f_to_p:
            patch_hits_f2p.append(patch_hit)

        per_test.append(
            {
                "test_id": test_id,
                "m7_status": "ALIGNED",
                "admitted_to_final_set": True,
                "alignment_verdict": "ALIGNED",
                "admission_path": admission_path,
                "before_patch_outcome": before,
                "after_patch_outcome": after,
                "before_patch_provenance": _evidence_provenance(
                    before_patch_evidence, test_id, before_patch_source
                ),
                "after_patch_provenance": _evidence_provenance(
                    after_patch_evidence, test_id, after_patch_source
                ),
                "f_to_p": f_to_p,
                "measurement_valid": measurement_valid,
                "exclusion_reason": None if measurement_valid else (
                    "cross_instance_evidence" if not instance_valid else "nonbinary_candidate_outcome"
                ),
                "patch_hit": patch_hit,
                "patch_hit_population": "T_F2P",
                "in_patch_hit_population": f_to_p,
                "checked_coverage": cc_value,
                "checked_coverage_status": cc_status,
                "checked_coverage_evidence": cc_persisted_evidence,
                "CC_computed_with_flaky": actual_flaky_flag,
                "flaky_flag": actual_flaky_flag,
                "flaky_detail": candidate.get("flaky_detail"),
                "m7_admission_unchanged": {
                    "m7_alignment_status": "ALIGNED",
                    "admitted_to_final_set": True,
                },
            }
        )

    admitted_final_count = sum(
        1
        for candidate in final_tests
        if not str(candidate.get("instance_id") or "")
        or str(candidate.get("instance_id")) == instance_id
    )
    valid_measurement_count = sum(
        1 for item in per_test if item["measurement_valid"]
    )
    final_count = admitted_final_count
    f2p_count = len(f_to_p_tests)
    patch_hit_rate_f2p = None
    if f2p_count and all(value is not None for value in patch_hits_f2p):
        patch_hit_rate_f2p = (
            sum(1 for value in patch_hits_f2p if value is True) / f2p_count
        )
    patch_hit_rate_t_final_diagnostic = None
    if (
        final_count
        and len(patch_hits_t_final) == final_count
    ):
        patch_hit_rate_t_final_diagnostic = (
            sum(1 for value in patch_hits_t_final if value is True) / final_count
        )
    checked_coverage_final_mean = mean_valid(cc_final_values)
    exam_score, exam_status = calculate_exam_from_evidence(localization_evidence)
    return {
        "instance_id": instance_id,
        "admitted_final_test_count": admitted_final_count,
        "total_final_tests": final_count,
        "final_test_count": final_count,
        "F_to_P_tests": f_to_p_tests,
        "F_to_P_count": f2p_count,
        "f_to_p_test_count": f2p_count,
        "F_to_P_rate": rate(f2p_count, final_count),
        "f_to_p_rate": rate(f2p_count, final_count),
        "patch_hit_rate": patch_hit_rate_f2p,
        "patch_hit_rate_f2p": patch_hit_rate_f2p,
        "patch_hit_population": "T_F2P",
        "patch_hit_test_denominator": f2p_count,
        "patch_hit_rate_t_final_diagnostic": patch_hit_rate_t_final_diagnostic,
        "patch_hit_t_final_diagnostic_denominator": final_count,
        "invalid_measurement_count": invalid_measurement_count,
        "valid_measurement_count": valid_measurement_count,
        "checked_coverage_mean": checked_coverage_final_mean,
        "checked_coverage_final_mean": checked_coverage_final_mean,
        "checked_coverage_f2p_mean": mean_valid(cc_f2p_values),
        "EXAM": exam_score,
        "exam_score": exam_score,
        "exam_status": exam_status,
        "exam_population": "independent_M8_function_localization",
        "before_patch": {
            "source": before_patch_source,
            "items": _evidence_items(before_patch_evidence),
        },
        "after_patch": {
            "source": after_patch_source,
            "items": _evidence_items(after_patch_evidence),
        },
        "patch_metadata": {
            "source": patch_metadata_source,
            "available": golden_patch_lines is not None,
            "scope": "M8_ONLY",
        },
        "per_test": per_test,
        "diagnostics": diagnostics,
        "no_flow_back_guarantee": M8_NO_FLOW_BACK_GUARANTEE,
        "m8_does_not_modify_m7_admission": True,
        "ignored_m7_score_fields": ["s_b", "s_c_prime", "s_a"],
        "ignored_m7_legacy_score_fields": [
            "bug_fail_score",
            "coverage_score",
            "issue_alignment_score",
        ],
    }


def calculate_exam_from_evidence(
    evidence: Mapping[str, Any] | None,
) -> tuple[float | None, str]:
    """Calculate EXAM=rank(f*)/F from explicit independent M8 evidence."""
    if not isinstance(evidence, Mapping):
        return None, "UNAVAILABLE_MISSING_LOCALIZATION_EVIDENCE"
    rank = evidence.get("faulty_function_rank")
    total = evidence.get("total_ranked_functions")
    if isinstance(rank, bool) or isinstance(total, bool):
        return None, "UNAVAILABLE_INVALID_LOCALIZATION_EVIDENCE"
    try:
        parsed_rank = int(rank)
        parsed_total = int(total)
    except (TypeError, ValueError):
        return None, "UNAVAILABLE_INVALID_LOCALIZATION_EVIDENCE"
    if parsed_rank < 0 or parsed_total <= 0 or parsed_rank > parsed_total:
        return None, "UNAVAILABLE_INVALID_LOCALIZATION_EVIDENCE"
    return parsed_rank / parsed_total, "AVAILABLE"


def _resolve_m8_feature_flags(
    feature_flags: V22FeatureFlags | Mapping[str, Any] | None,
) -> V22FeatureFlags:
    if isinstance(feature_flags, V22FeatureFlags):
        return resolve_feature_flags(feature_flags.to_dict())
    return resolve_feature_flags(feature_flags)


def _evidence_outcome(evidence: Mapping[str, Any], test_id: str) -> Any:
    item = evidence.get(test_id)
    if isinstance(item, Mapping):
        return item.get("outcome", item.get("execution_status", item.get("status")))
    return item


def _evidence_provenance(evidence: Mapping[str, Any], test_id: str, default_source: str) -> dict[str, Any]:
    item = evidence.get(test_id)
    if isinstance(item, Mapping):
        return {
            "source": item.get("source", default_source),
            "artifact": item.get("artifact"),
            "command": item.get("command"),
        }
    return {"source": default_source, "artifact": None, "command": None}


def _evidence_items(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in evidence.items()}


def stable_test_identity(generated_test: Mapping[str, Any]) -> TestIdentity:
    """Read an explicit generated-test identity without fabricating one."""
    test_id = None
    for key in ("test_id", "generated_test_id"):
        value = generated_test.get(key)
        if isinstance(value, str) and value.strip():
            test_id = value.strip()
            break
    test_file = _generated_test_file(generated_test)
    test_nodeid, provenance = _generated_test_nodeid(generated_test, test_file)
    if test_id or test_nodeid:
        return TestIdentity(test_id, test_nodeid, test_file, "AVAILABLE", provenance)
    return TestIdentity(None, None, test_file, "UNAVAILABLE")


def _generated_test_file(generated_test: Mapping[str, Any]) -> Optional[str]:
    for key in ("target_test_file", "test_file", "file_path"):
        value = generated_test.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_file_path(value)
    metadata = generated_test.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("target_test_file", "test_file", "file_path"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return _normalize_file_path(value)
    return None


def _generated_test_nodeid(
    generated_test: Mapping[str, Any],
    test_file: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    explicit = _explicit_generated_test_nodeid(generated_test)
    if explicit:
        return _normalize_nodeid(explicit), "generated_test_metadata"
    if not test_file:
        return None, None
    explicit_function = _explicit_generated_test_function(generated_test)
    if explicit_function:
        return f"{test_file}::{explicit_function}", "generated_test_metadata"
    for key in ("test_code", "append_block"):
        code = generated_test.get(key)
        if isinstance(code, str) and code.strip():
            suffixes = _pytest_test_suffixes_from_code(code)
            if len(suffixes) == 1:
                return f"{test_file}::{suffixes[0]}", key
    patch_text = generated_test.get("test_patch")
    if isinstance(patch_text, str) and patch_text.strip():
        suffixes = _pytest_test_suffixes_from_patch(patch_text, test_file)
        if len(suffixes) == 1:
            return f"{test_file}::{suffixes[0]}", "generated_patch_text"
    return None, None


def _explicit_generated_test_nodeid(generated_test: Mapping[str, Any]) -> Optional[str]:
    for key in ("canonical_test_nodeid", "test_nodeid", "pytest_nodeid", "nodeid"):
        value = generated_test.get(key)
        if isinstance(value, str) and "::" in value:
            return value
    metadata = generated_test.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("canonical_test_nodeid", "test_nodeid", "pytest_nodeid", "nodeid"):
            value = metadata.get(key)
            if isinstance(value, str) and "::" in value:
                return value
    return None


def _explicit_generated_test_function(generated_test: Mapping[str, Any]) -> Optional[str]:
    for key in ("test_function_name", "test_name", "function_name"):
        value = generated_test.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = generated_test.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("test_function_name", "test_name", "function_name"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _pytest_test_suffixes_from_code(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    suffixes: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            suffixes.append(node.name)
        elif isinstance(node, ast.ClassDef) and _is_collectable_test_class_ast(node):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test"):
                    suffixes.append(f"{node.name}::{item.name}")
    return suffixes


def _is_collectable_test_class_ast(node: ast.ClassDef) -> bool:
    explicitly_disabled = any(
        isinstance(item, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "__test__"
            for target in (item.targets if isinstance(item, ast.Assign) else [item.target])
        )
        and isinstance(item.value, ast.Constant)
        and item.value.value is False
        for item in node.body
    )
    is_test_case = any(
        (isinstance(base, ast.Name) and base.id.endswith("TestCase"))
        or (isinstance(base, ast.Attribute) and base.attr.endswith("TestCase"))
        for base in node.bases
    )
    custom_constructor = any(
        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name in {"__init__", "__new__"}
        for item in node.body
    )
    return bool(
        not explicitly_disabled
        and (node.name.startswith("Test") or is_test_case)
        and not custom_constructor
    )


def _generated_test_identity_aliases(
    generated_test: Mapping[str, Any],
    canonical_nodeid: Optional[str],
) -> list[str]:
    """Return representation aliases owned by the accepted generated test.

    Some unittest runners replace a method name with its docstring in one
    execution stage.  Only a docstring attached to the exact canonical test
    function is eligible; aliases from sibling tests are never included.
    """
    if not canonical_nodeid:
        return []
    identity = _parse_canonical_nodeid(canonical_nodeid)
    expected_function = identity.get("function_name")
    expected_class = identity.get("class_name")
    aliases: list[str] = []
    for key in ("test_code", "append_block"):
        code = generated_test.get(key)
        if not isinstance(code, str) or not code.strip():
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in tree.body:
            candidates: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and expected_class is None:
                candidates = [node]
            elif isinstance(node, ast.ClassDef) and node.name == expected_class:
                candidates = [
                    item
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
            for function in candidates:
                if function.name != expected_function:
                    continue
                docstring = ast.get_docstring(function, clean=True)
                if docstring:
                    aliases.append(_normalize_result_key(docstring))
    return list(dict.fromkeys(aliases))


def _pytest_test_suffixes_from_patch(patch_text: str, test_file: str) -> list[str]:
    added_lines: list[str] = []
    current_file: Optional[str] = None
    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            match = re.match(r"diff --git a/(\S+) b/(\S+)", line)
            current_file = _normalize_file_path(match.group(2)) if match else None
            continue
        if current_file != test_file:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:])
    return _pytest_test_suffixes_from_code("\n".join(added_lines))


def _normalize_file_path(value: str) -> str:
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _normalize_nodeid(value: str) -> str:
    text = str(value).strip()
    if "::" not in text and ".py:" in text:
        path_part, suffix = text.split(".py:", 1)
        text = f"{path_part}.py::{suffix}"
    parts = [_normalize_file_path(part) if index == 0 else part.strip() for index, part in enumerate(text.split("::"))]
    return "::".join(part for part in parts if part)


def _nodeid_file(value: str) -> str:
    return _normalize_nodeid(value).split("::", 1)[0]


def _candidate_test_names(test_id: Optional[str], stage_results: Mapping[str, Any]) -> list[str]:
    if not test_id:
        return []
    normalized_test_id = _normalize_nodeid(test_id)
    if "::" not in normalized_test_id:
        return []
    matches: list[str] = []
    for name in stage_results:
        text = str(name)
        if _normalize_nodeid(text) == normalized_test_id:
            matches.append(text)
    return matches


def match_generated_test_outcomes(
    test_id: Optional[str],
    before_patch: Mapping[str, Any],
    after_patch: Mapping[str, Any],
    *,
    test_nodeid: Optional[str] = None,
    test_file: Optional[str] = None,
    accepted_aliases: Optional[Iterable[str]] = None,
) -> TestOutcomeMatch:
    executable_nodeid = test_nodeid or (test_id if test_id and "::" in test_id else None)
    if executable_nodeid:
        identity = _parse_canonical_nodeid(executable_nodeid)
        aliases = tuple(
            dict.fromkeys(
                _normalize_result_key(alias)
                for alias in (accepted_aliases or ())
                if _normalize_result_key(alias)
            )
        )
        before_match = _match_stage_result_key(before_patch, identity, aliases)
        after_match = _match_stage_result_key(after_patch, identity, aliases)
        if before_match["status"] == "AMBIGUOUS" or after_match["status"] == "AMBIGUOUS":
            return _outcome_match(
                "AMBIGUOUS",
                before_patch,
                after_patch,
                before_match,
                after_match,
            )
        if before_match["status"] == "MATCHED" and after_match["status"] == "MATCHED":
            if (
                before_match.get("provenance") == "parameterized_concrete_unique"
                or after_match.get("provenance") == "parameterized_concrete_unique"
            ) and _normalize_nodeid(str(before_match.get("name") or "")) != _normalize_nodeid(
                str(after_match.get("name") or "")
            ):
                return TestOutcomeMatch(
                    "UNMATCHED",
                    None,
                    None,
                    None,
                    "parameter_identity_mismatch",
                    before_test_name=str(before_match.get("name") or "") or None,
                    after_test_name=str(after_match.get("name") or "") or None,
                    before_match_provenance=before_match.get("provenance"),
                    after_match_provenance=after_match.get("provenance"),
                )
            # An alias is a representation bridge, not a standalone identity.
            # Require the other stage to prove the canonical method identity.
            if (
                before_match.get("provenance") == "accepted_candidate_docstring"
                and after_match.get("provenance") == "accepted_candidate_docstring"
            ):
                return TestOutcomeMatch(
                    "UNMATCHED",
                    None,
                    None,
                    None,
                    "accepted_candidate_docstring",
                    before_match_provenance="accepted_candidate_docstring",
                    after_match_provenance="accepted_candidate_docstring",
                )
            return _outcome_match(
                "MATCHED",
                before_patch,
                after_patch,
                before_match,
                after_match,
            )
        return _outcome_match(
            "UNMATCHED",
            before_patch,
            after_patch,
            before_match,
            after_match,
        )

    if not test_file:
        return TestOutcomeMatch("UNMATCHED", None, None, None, None)
    normalized_file = _normalize_file_path(test_file)
    before_names = _normalized_stage_names(before_patch)
    after_names = _normalized_stage_names(after_patch)
    common_names = sorted(set(before_names) & set(after_names))
    file_matches = [name for name in common_names if _nodeid_file(name) == normalized_file]
    if not file_matches:
        return TestOutcomeMatch("UNMATCHED", None, None, None, "file_constrained_unique")
    if len(file_matches) > 1:
        return TestOutcomeMatch("AMBIGUOUS", None, None, None, "file_constrained_unique")
    normalized = file_matches[0]
    if len(before_names[normalized]) > 1 or len(after_names[normalized]) > 1:
        return TestOutcomeMatch("AMBIGUOUS", None, None, None, "file_constrained_unique")
    name = before_names[normalized][0]
    after_name = after_names[normalized][0]
    return TestOutcomeMatch(
        "MATCHED",
        name,
        normalize_single_test_outcome(before_patch.get(name)),
        normalize_single_test_outcome(after_patch.get(after_name)),
        "file_constrained_unique",
        before_test_name=name,
        after_test_name=after_name,
        before_match_provenance="file_constrained_unique",
        after_match_provenance="file_constrained_unique",
    )


def _normalized_stage_names(stage_results: Mapping[str, Any]) -> dict[str, list[str]]:
    names: dict[str, list[str]] = {}
    for name in stage_results:
        normalized = _normalize_nodeid(str(name))
        names.setdefault(normalized, []).append(str(name))
    return names


def _outcome_match(
    status: str,
    before_patch: Mapping[str, Any],
    after_patch: Mapping[str, Any],
    before_match: Mapping[str, Any],
    after_match: Mapping[str, Any],
) -> TestOutcomeMatch:
    before_name = before_match.get("name") if before_match.get("status") == "MATCHED" else None
    after_name = after_match.get("name") if after_match.get("status") == "MATCHED" else None
    if before_name and after_name:
        test_name = before_name if before_name == after_name else after_name
    else:
        test_name = before_name or after_name
    before_provenance = before_match.get("provenance")
    after_provenance = after_match.get("provenance")
    provenance = before_provenance if before_provenance == after_provenance else "mixed"
    if status != "MATCHED":
        provenance = before_provenance or after_provenance
    return TestOutcomeMatch(
        status,
        test_name,
        normalize_single_test_outcome(before_patch.get(before_name)) if before_name else None,
        normalize_single_test_outcome(after_patch.get(after_name)) if after_name else None,
        provenance,
        before_test_name=before_name,
        after_test_name=after_name,
        before_match_provenance=before_provenance,
        after_match_provenance=after_provenance,
    )


def _unittest_display_matches_nodeid(display_name: str, canonical_nodeid: str) -> bool:
    identity = _parse_canonical_nodeid(canonical_nodeid)
    return _unittest_display_matches_identity(display_name, identity)


def _unittest_display_matches_identity(
    display_name: str,
    identity: Mapping[str, Any],
    accepted_aliases: Iterable[str] = (),
) -> bool:
    match = re.match(
        r"^(?P<label>.+?) \((?P<qualified>[\w.]+)\)$",
        _normalize_result_key(display_name),
    )
    if not match:
        return False
    if not identity.get("class_name"):
        return False
    qualified = match.group("qualified")
    qualified_parts = qualified.split(".")
    function_name = str(identity.get("function_name") or "")
    class_name = str(identity.get("class_name") or "")
    if qualified_parts[-2:] == [class_name, function_name]:
        module_parts = qualified_parts[:-2]
        label_is_exact = match.group("label") == function_name
        label_is_alias = _normalize_result_key(match.group("label")) in {
            _normalize_result_key(alias) for alias in accepted_aliases
        }
        if not (label_is_exact or label_is_alias):
            return False
    elif qualified_parts[-1:] == [class_name]:
        module_parts = qualified_parts[:-1]
        if match.group("label") != function_name:
            return False
    else:
        return False
    module_from_display = ".".join(module_parts)
    module_aliases = {
        str(identity.get("module") or ""),
        str(identity.get("full_module") or ""),
    } - {""}
    return any(
        module_from_display == alias or module_from_display.endswith(f".{alias}")
        for alias in module_aliases
    )


def _parse_canonical_nodeid(nodeid: str) -> dict[str, Optional[str]]:
    normalized = _normalize_nodeid(nodeid)
    parts = normalized.split("::")
    test_file = parts[0] if parts else ""
    function_name = parts[-1] if len(parts) >= 2 else ""
    class_name = parts[-2] if len(parts) >= 3 else None
    return {
        "nodeid": normalized,
        "file_path": test_file,
        "module": _module_name_from_test_file(test_file),
        "full_module": _full_module_name_from_test_file(test_file),
        "class_name": class_name,
        "function_name": function_name,
        "class_qualified": f"{class_name}::{function_name}" if class_name and function_name else None,
        "dotted_class_qualified": (
            f"{_module_name_from_test_file(test_file)}.{class_name}.{function_name}"
            if class_name and function_name
            else None
        ),
    }


def _match_stage_result_key(
    stage_results: Mapping[str, Any],
    identity: Mapping[str, Optional[str]],
    accepted_aliases: Iterable[str] = (),
) -> dict[str, Optional[str]]:
    candidates = [str(name) for name in stage_results]
    precedence = [
        ("exact_nodeid", lambda name: _normalize_nodeid(name) == identity.get("nodeid")),
        (
            "parameterized_concrete_unique",
            lambda name: _is_concrete_parameter_of_base(
                _normalize_nodeid(name), str(identity.get("nodeid") or "")
            ),
        ),
        (
            "unittest_display_name",
            lambda name: _unittest_display_matches_identity(
                name, identity, accepted_aliases
            ),
        ),
        ("class_qualified", lambda name: _class_qualified_matches_identity(name, identity)),
        ("bare_function_unique", lambda name: _normalize_result_key(name) == identity.get("function_name")),
        (
            "accepted_candidate_docstring",
            lambda name: _normalize_result_key(name) in set(accepted_aliases),
        ),
    ]
    for provenance, predicate in precedence:
        matches = [name for name in candidates if predicate(name)]
        if not matches:
            continue
        if len(matches) > 1:
            return {"status": "AMBIGUOUS", "name": None, "provenance": provenance}
        return {"status": "MATCHED", "name": matches[0], "provenance": provenance}
    return {"status": "UNMATCHED", "name": None, "provenance": None}


def _is_concrete_parameter_of_base(observed_nodeid: str, base_nodeid: str) -> bool:
    """Bridge one legacy base node to one concrete pytest parameter identity."""
    if not base_nodeid or not observed_nodeid.startswith(base_nodeid + "["):
        return False
    suffix = observed_nodeid[len(base_nodeid):]
    return suffix.startswith("[") and suffix.endswith("]") and len(suffix) > 2


def _class_qualified_matches_identity(name: str, identity: Mapping[str, Optional[str]]) -> bool:
    normalized = _normalize_result_key(name)
    class_qualified = identity.get("class_qualified")
    dotted = identity.get("dotted_class_qualified")
    if class_qualified and normalized == class_qualified:
        return True
    if dotted and normalized == dotted:
        return True
    if class_qualified and normalized.endswith(f"::{class_qualified}"):
        return _nodeid_file(normalized) == identity.get("file_path")
    return False


def _normalize_result_key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def _module_name_from_test_file(test_file: str) -> str:
    path = _normalize_file_path(test_file)
    if path.endswith(".py"):
        path = path[:-3]
    if path.startswith("tests/"):
        path = path[len("tests/"):]
    return path.replace("/", ".")


def _full_module_name_from_test_file(test_file: str) -> str:
    path = _normalize_file_path(test_file)
    if path.endswith(".py"):
        path = path[:-3]
    return path.replace("/", ".")


def _is_successful_m8_resolution(result: Mapping[str, Any]) -> bool:
    return (
        result.get("admitted_to_final_set") is True
        and result.get("match_status") == "MATCHED"
        and result.get("before_patch_outcome") == "FAIL"
        and result.get("after_patch_outcome") == "PASS"
        and result.get("evaluation_status") == "SUCCESS"
    )


def make_failure_record(
    category: FailureCategory,
    stage: str,
    message: str,
    *,
    command: Optional[list[str]] = None,
    exit_code: Optional[int] = None,
) -> dict[str, Any]:
    return FailureRecord(
        category=category,
        stage=stage,
        command=command,
        exit_code=exit_code,
        error_message=message,
        retry_count=0,
        retry_safe=False,
        included_in_aggregate_metrics=False,
    ).to_dict()


def mean_valid(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def checked_coverage_from_payload(payload: Mapping[str, Any]) -> tuple[Optional[float], str]:
    """Calculate CC strictly from pre-patch covered SUT lines and oracle slice."""
    value, status, _diagnostic = checked_coverage_from_payload_with_diagnostic(payload)
    return value, status


def checked_coverage_from_payload_with_diagnostic(
    payload: Mapping[str, Any],
) -> tuple[Optional[float], str, Optional[dict[str, Any]]]:
    """Calculate CC and explain nulls without accepting proxy scores.

    CC(test) = |covered_SUT_lines intersect dynamic_slice_of_oracle| /
    |covered_SUT_lines|. A persisted `checked_coverage`, `coverage_score`, or
    M7 `s_c_prime` value is not enough evidence for M8 checked coverage.
    """
    status = str(payload.get("checked_coverage_status") or payload.get("instrumentation_status") or "")
    if any(key in payload for key in ("checked_coverage", "coverage_score", "s_c_prime")):
        return (
            None,
            status if status in CC_STATUSES else "UNSUPPORTED",
            {"reason": "cc_proxy_score_ignored"},
        )

    covered, covered_error = _line_key_set(
        payload.get("covered_SUT_lines", payload.get("covered_sut_lines"))
    )
    dynamic_slice, dynamic_slice_error = _line_key_set(
        payload.get("dynamic_slice_of_oracle", payload.get("dynamic_slice_lines"))
    )
    if covered_error == "missing_line_records":
        return (
            None,
            status if status in CC_STATUSES else "UNSUPPORTED",
            {"reason": "missing_covered_sut_lines"},
        )
    if covered_error:
        return (
            None,
            status if status in CC_STATUSES else "UNSUPPORTED",
            {"reason": "malformed_covered_sut_lines"},
        )
    if not covered:
        return (
            None,
            status if status in CC_STATUSES else "UNSUPPORTED",
            {"reason": "empty_covered_sut_lines"},
        )
    if dynamic_slice_error == "missing_line_records":
        return (
            None,
            status if status in CC_STATUSES else "UNSUPPORTED",
            {"reason": "missing_dynamic_slice_of_oracle"},
        )
    if dynamic_slice_error:
        return (
            None,
            status if status in CC_STATUSES else "UNSUPPORTED",
            {"reason": "malformed_dynamic_slice_of_oracle"},
        )
    return len(covered & dynamic_slice) / len(covered), "SUPPORTED", None


def _checked_coverage_payload_from_slice_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "checked_coverage_status": payload.get("status"),
        "covered_sut_lines": payload.get("covered_sut_lines"),
        "dynamic_slice_lines": payload.get("checked_lines"),
        "backend": payload.get("backend"),
        "oracle_node": payload.get("oracle_node"),
        "numerator": payload.get("numerator"),
        "denominator": payload.get("denominator"),
        "diagnostics": payload.get("diagnostics"),
    }


def _legacy_checked_coverage_evidence(payload: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    covered = payload.get("covered_SUT_lines", payload.get("covered_sut_lines"))
    checked = payload.get("dynamic_slice_of_oracle", payload.get("dynamic_slice_lines"))
    if covered is None and checked is None:
        return None
    covered_set, covered_error = _line_key_set(covered)
    checked_set, checked_error = _line_key_set(checked)
    numerator = None
    denominator = None
    if covered_set is not None and checked_set is not None and covered_set:
        numerator = len(covered_set & checked_set)
        denominator = len(covered_set)
    return {
        "backend": str(payload.get("backend") or "artifact_explicit_dynamic_slice_lines"),
        "status": str(
            payload.get("checked_coverage_status")
            or payload.get("instrumentation_status")
            or ("SUPPORTED" if numerator is not None else "UNSUPPORTED")
        ),
        "oracle_node": payload.get("oracle_node") or payload.get("assertion_location"),
        "covered_sut_lines": covered,
        "checked_lines": checked,
        "numerator": numerator,
        "denominator": denominator,
        "checked_coverage": (numerator / denominator) if numerator is not None and denominator else None,
        "diagnostics": {
            "source": "pre_patch_alignment_artifact",
            "covered_error": covered_error,
            "checked_error": checked_error,
        },
    }


def _before_patch_repo_path_from_artifacts(output_dir: Path, alignment: Mapping[str, Any]) -> Optional[Path]:
    for key in ("before_patch_repo_path", "pre_patch_repo_path", "repo_path"):
        value = alignment.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value)
    context = read_json_object(output_dir / "context.json") or {}
    for key in ("before_patch_repo_path", "pre_patch_repo_path", "repo_path"):
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value)
    return None


def _pre_patch_instance_for_m8_cc(
    instance_id: str,
    m8_view: Any | None,
) -> Optional[PrePatchInstanceView]:
    """Resolve benchmark metadata for CC without carrying golden M8 fields."""
    if m8_view is not None:
        return PrePatchInstanceView(
            instance_id=str(getattr(m8_view, "instance_id", instance_id) or instance_id),
            repo=str(getattr(m8_view, "repo", "") or ""),
            base_commit=str(getattr(m8_view, "base_commit", "") or ""),
            problem_statement=str(getattr(m8_view, "problem_statement", "") or ""),
            version=str(getattr(m8_view, "version", "") or ""),
            environment_setup_commit=str(getattr(m8_view, "environment_setup_commit", "") or ""),
        )
    if not instance_id:
        return None
    try:
        return make_pre_patch_view(TDDInstanceLoader().get_instance(instance_id))
    except Exception:
        return None


def _line_key_set(value: Any) -> tuple[Optional[set[tuple[str, int]]], Optional[str]]:
    if value is None:
        return None, "missing_line_records"
    if not isinstance(value, list):
        return None, "malformed_line_records"
    lines: set[tuple[str, int]] = set()
    for item in value:
        if not isinstance(item, Mapping):
            return None, "malformed_line_records"
        file_name = str(item.get("source_file") or item.get("file") or "")
        line_no = item.get("line_no", item.get("line"))
        if not file_name or isinstance(line_no, bool) or not isinstance(line_no, int):
            return None, "malformed_line_records"
        parsed_line = line_no
        if parsed_line <= 0:
            return None, "malformed_line_records"
        lines.add((file_name, parsed_line))
    return lines, None


def parse_golden_patch_lines(patch_text: str) -> dict[str, set[int]]:
    """Parse added source lines from unified diff hunks for M8-only PHR."""
    patch_lines_by_file: dict[str, set[int]] = {}
    current_file: Optional[str] = None
    next_new_line: Optional[int] = None
    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            match = re.match(r"diff --git a/(\S+) b/(\S+)", line)
            current_file = None
            next_new_line = None
            if match:
                candidate = match.group(2)
                parts = tuple(part.casefold() for part in Path(candidate).parts)
                name = Path(candidate).name.casefold()
                if not (
                    "tests" in parts
                    or name == "conftest.py"
                    or name.startswith("test_")
                    or name.endswith("_test.py")
                ):
                    current_file = candidate
                    patch_lines_by_file.setdefault(current_file, set())
            continue
        if current_file is None:
            continue
        if line.startswith("@@"):
            match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
            next_new_line = int(match.group(1)) if match else None
            continue
        if next_new_line is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            patch_lines_by_file[current_file].add(next_new_line)
            next_new_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            continue
        else:
            next_new_line += 1
    return {file_name: lines for file_name, lines in patch_lines_by_file.items() if lines}


def patch_hit_from_coverage(
    coverage_data: Mapping[str, Any],
    golden_lines: Mapping[str, set[int]],
) -> Optional[bool]:
    details = patch_hit_details_from_coverage(coverage_data, golden_lines)
    return details.get("patch_hit") if details else None


def patch_hit_details_from_coverage(
    coverage_data: Mapping[str, Any],
    golden_lines: Mapping[str, set[int]],
) -> Optional[dict[str, Any]]:
    if not isinstance(coverage_data, Mapping) or not isinstance(golden_lines, Mapping) or not golden_lines:
        return None
    if any(not isinstance(line_numbers, (set, list, tuple)) for line_numbers in golden_lines.values()):
        return None
    denominator = sum(len(line_numbers) for line_numbers in golden_lines.values())
    if denominator <= 0:
        return None
    covered_patch_lines: list[dict[str, Any]] = []
    missing_patch_lines: list[dict[str, Any]] = []
    golden_patch_line_records: list[dict[str, Any]] = []
    used_coverage_files: set[Any] = set()
    for patch_file, line_numbers in sorted(golden_lines.items()):
        if not isinstance(line_numbers, (set, list, tuple)) or any(
            isinstance(line, bool) or not isinstance(line, int) or line <= 0
            for line in line_numbers
        ):
            return None
        normalized_patch = _normalize_file_path(patch_file)
        normalized_coverage = {
            coverage_file: _normalize_file_path(str(coverage_file))
            for coverage_file in coverage_data
        }
        exact = [key for key, value in normalized_coverage.items() if value == normalized_patch]
        suffix = [
            key
            for key, value in normalized_coverage.items()
            if value.endswith("/" + normalized_patch)
            or normalized_patch.endswith("/" + value)
        ]
        candidates = exact if exact else suffix
        matched = candidates[0] if len(candidates) == 1 else None
        if matched in used_coverage_files:
            return None
        info = coverage_data.get(matched) if matched else None
        if not matched or not isinstance(info, Mapping):
            return None
        used_coverage_files.add(matched)
        raw_missing = info.get("missing_lines")
        if not isinstance(raw_missing, (list, tuple, set)):
            return None
        missing: set[int] = set()
        for line in raw_missing:
            if isinstance(line, bool) or not isinstance(line, int):
                return None
            parsed = line
            if parsed <= 0:
                return None
            missing.add(parsed)
        for line_no in sorted(line_numbers):
            record = {"source_file": patch_file, "line_no": int(line_no)}
            golden_patch_line_records.append(record)
            if line_no not in missing:
                covered_patch_lines.append({**record, "coverage_file": str(matched)})
            else:
                missing_patch_lines.append(record)
    numerator = len(covered_patch_lines)
    return {
        "patch_hit": numerator > 0,
        "numerator": numerator,
        "denominator": denominator,
        "golden_patch_lines": golden_patch_line_records,
        "covered_patch_lines": covered_patch_lines,
        "missing_patch_lines": missing_patch_lines,
    }

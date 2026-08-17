"""
결과 수립기.

개별 인스턴스의 alignment_result + final_evaluation 결과를 수집하여
배치 단위 통계와 최종 리포트를 생성한다.

Usage:
    # 단일 인스턴스 결과 수집
    collector = ResultCollector()
    collector.collect("outputs/astropy__astropy-12907")

    # 전체 배치 결과 집계
    collector = ResultCollector("outputs")
    report = collector.aggregate()
    collector.save_report(report, "outputs/final_report.json")

    # CLI
    python -m src.evaluator.result_collector                        # outputs 전체
    python -m src.evaluator.result_collector --output_root outputs  # 명시적 경로
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluator.resolve_policy import resolve_metadata
from src.contracts.final_sets import admitted_to_final_set
from src.contracts.status import legacy_failure_type_to_statuses
from src.evaluator.m8_fingerprint import m8_input_fingerprint_is_fresh
from src.utils.file_io import read_json_object, write_json_atomic


class ResultCollector:
    """배치 결과를 수집하고 통계를 산출한다."""

    def __init__(self, output_root: str = "outputs") -> None:
        self.output_root = Path(output_root)

    def collect(self, instance_dir: str) -> Optional[Dict[str, Any]]:
        """단일 인스턴스 디렉토리에서 결과를 수집한다.

        Returns:
            {
                "instance_id": str,
                "failure_type": str,
                "iterations": int,
                "final_score": float,     # final_evaluation.json이 fresh일 때만 기록
                "before_patch": dict,
                "after_patch": dict,
            }
            또는 결과가 없으면 None
        """
        d = Path(instance_dir)
        instance_id = d.name

        # alignment 결과
        alignment_path = d / "alignment_result.json"
        if not alignment_path.exists():
            return None

        alignment = read_json_object(alignment_path)
        if alignment is None:
            return None
        if not _artifact_belongs_to_instance_dir(d, alignment):
            return None

        entry: Dict[str, Any] = {
            "instance_id": instance_id,
            "failure_type": self._normalize_failure_type(alignment.get("failure_type", "UNKNOWN")),
            "iterations": alignment.get("iterations", alignment.get("iteration", 0)),
            "score_breakdown": alignment.get("score_breakdown", {}),
            "before_patch": {},
            "after_patch": {},
        }
        entry.update(self._status_fields(entry["failure_type"]))

        # final evaluation 결과 (분리된 평가기가 생성한 파일)
        final_eval_path = d / "final_evaluation.json"
        if final_eval_path.exists():
            if not admitted_to_final_set(
                {
                    "candidate_status": "GENERATED",
                    "diagnostic_only": entry.get("diagnostic_only", False),
                    "m7_alignment_status": entry.get("m7_alignment_status")
                    or entry.get("failure_type"),
                }
            ):
                entry["diagnostic_only"] = True
                entry["final_eval_diagnostic_only"] = True
                return entry
            final_eval = read_json_object(final_eval_path)
            if final_eval is None:
                entry["final_eval_error"] = "failed to load final_evaluation.json"
                return entry
            execution_artifacts = [
                artifact
                for artifact in (
                    read_json_object(d / "m6_execution_result.json"),
                    read_json_object(d / "alignment_execution.json"),
                )
                if artifact is not None
            ]
            if not execution_artifacts or not all(
                _artifact_belongs_to_instance_dir(d, execution)
                for execution in execution_artifacts
            ):
                entry["final_eval_error"] = "M8 ownership check failed for M6 execution artifact"
                entry["resolved"] = None
                return entry
            if not _artifact_belongs_to_instance_dir(d, final_eval):
                entry["final_eval_error"] = "M8 ownership check failed for final evaluation artifact"
                entry["resolved"] = None
                return entry
            if not self._is_final_eval_fresh(d, final_eval):
                if entry["failure_type"] == "ALIGNED":
                    entry["resolved"] = None
                return entry
            self._apply_final_eval(entry, final_eval)
            entry.pop("strict_resolved", None)
            entry.pop("relaxed_resolved", None)

        return entry

    @staticmethod
    def _normalize_failure_type(value: Any) -> str:
        if value == "NO_FAIL":
            return "NOT_FAILED"
        if value == "NOT_COLLECTED":
            return "NOT_VALID"
        return str(value or "UNKNOWN")

    @staticmethod
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

    @staticmethod
    def _is_final_eval_fresh(instance_dir: Path, final_eval: Dict[str, Any]) -> bool:
        infra_failed = (
            final_eval.get("harness_returncode") not in (None, 0)
            and not final_eval.get("before_patch")
            and not final_eval.get("after_patch")
        )
        return bool(
            not infra_failed
            and m8_input_fingerprint_is_fresh(instance_dir, final_eval)
        )

    @staticmethod
    def _apply_final_eval(entry: Dict[str, Any], final_eval: Dict[str, Any]) -> None:
        entry["final_score"] = final_eval.get("final_score", 0.0)
        entry["before_patch"] = final_eval.get("before_patch", {})
        entry["after_patch"] = final_eval.get("after_patch", {})
        canonical_keys = (
            "m7_status",
            "admitted_to_final_set",
            "test_id",
            "test_identity_status",
            "matched_test_name",
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
            "checked_coverage_mean",
            "checked_coverage_final_mean",
            "checked_coverage_f2p_mean",
            "checked_coverage_diagnostics",
            "alignment_verdict",
            "admission_path",
            "CC_computed_with_flaky",
            "flaky_flag",
            "flaky_detail",
            "evaluation_status",
            "failure_record",
            "per_test",
        )
        for key in canonical_keys:
            if key in final_eval:
                entry[key] = final_eval[key]
        if "checked_coverage_mean" not in entry and "checked_coverage_final_mean" in entry:
            entry["checked_coverage_mean"] = entry["checked_coverage_final_mean"]
        if "checked_coverage_final_mean" not in entry and "checked_coverage_mean" in entry:
            entry["checked_coverage_final_mean"] = entry["checked_coverage_mean"]
        if "f_to_p" in final_eval:
            entry["resolved"] = (
                None
                if final_eval.get("evaluation_status") in {"ERROR", "EVALUATION_FAILURE", "ENVIRONMENT_FAILURE"}
                or ("resolved" in final_eval and final_eval.get("resolved") is None)
                else ResultCollector._canonical_m8_success(final_eval)
            )
            entry["legacy_final_eval_metadata"] = {
                "resolved": final_eval.get("resolved"),
                "final_score": final_eval.get("final_score"),
                "harness_resolved": final_eval.get("harness_resolved"),
                "harness_final_score": final_eval.get("harness_final_score"),
            }
        else:
            entry["resolved"] = bool(resolve_metadata({**final_eval, **entry}).get("resolved"))
        entry["final_set_membership"] = {
            "in_t_final": bool(final_eval.get("admitted_to_final_set", entry["failure_type"] == "ALIGNED")),
            "in_t_f2p": bool(final_eval.get("f_to_p", False)),
        }
        if final_eval.get("error"):
            entry["final_eval_error"] = final_eval.get("error")

    @classmethod
    def _is_resolved(cls, entry: Dict[str, Any]) -> bool:
        return bool(resolve_metadata(entry).get("resolved"))

    def aggregate(self, instance_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """전체 또는 지정된 인스턴스의 결과를 집계한다.

        Returns:
            {
                "total": int,
                "aligned": int,
                "resolved": int,
                "failure_type_counts": {...},
                "aligned_rate": str,
                "resolve_rate": str,
                "avg_iterations": float,
                "per_instance": [...]
            }
        """
        results: List[Dict[str, Any]] = []

        if instance_ids is not None:
            ordered_ids = list(dict.fromkeys(str(iid) for iid in instance_ids))
            dirs = [self.output_root / iid for iid in ordered_ids]
        else:
            dirs = sorted(
                d for d in self.output_root.iterdir()
                if d.is_dir() and (d / "alignment_result.json").exists()
            )

        for d in dirs:
            entry = self.collect(str(d))
            if entry:
                results.append(entry)

        total = len(results)

        # failure type 집계
        ft_counts: Dict[str, int] = {}
        for r in results:
            ft = r["failure_type"]
            ft_counts[ft] = ft_counts.get(ft, 0) + 1

        aligned = ft_counts.get("ALIGNED", 0)
        resolved = sum(1 for r in results if self._canonical_or_legacy_success(r))
        final_eval_count = sum(1 for r in results if r.get("final_score") is not None)
        metric_rows = [
            r
            for r in results
            if bool((r.get("final_set_membership") or {}).get("in_t_final"))
            and not r.get("diagnostic_only")
            and not self._excluded_from_aggregate_metrics(r)
        ]
        test_records: list[dict[str, Any]] = []
        seen_test_aliases: set[tuple[str, str]] = set()
        unavailable_population_rows = 0
        for row in metric_rows:
            raw_records = row.get("per_test")
            legacy_single_record = False
            if isinstance(raw_records, list) and raw_records:
                records = [dict(item) for item in raw_records if isinstance(item, dict)]
            elif int(row.get("final_test_count", 0) or 0) == 1:
                legacy_single_record = True
                records = [{
                    "test_id": row.get("test_id"),
                    "test_nodeid": row.get("test_nodeid"),
                    "f_to_p": row.get("f_to_p"),
                    "patch_hit": row.get("patch_hit"),
                    "patch_hit_numerator": row.get("patch_hit_numerator"),
                    "patch_hit_denominator": row.get("patch_hit_denominator"),
                    "checked_coverage": _primary_checked_coverage_value(row),
                    "measurement_valid": not self._excluded_from_aggregate_metrics(row),
                }]
            else:
                unavailable_population_rows += 1
                continue
            for record in records:
                logical_test_id = str(record.get("test_id") or "")
                test_nodeid = str(record.get("test_nodeid") or "")
                if legacy_single_record and not logical_test_id and not test_nodeid:
                    patch_identity = str(row.get("generated_patch_sha256") or row.get("patch_sha256") or "")
                    instance_identity = str(row.get("instance_id") or "")
                    legacy_identity = patch_identity or instance_identity
                    logical_test_id = f"legacy-single:{legacy_identity}" if legacy_identity else ""
                if not logical_test_id and not test_nodeid:
                    unavailable_population_rows += 1
                    continue
                instance_key = str(row.get("instance_id") or "")
                aliases = {
                    (instance_key, value)
                    for value in (logical_test_id, test_nodeid)
                    if value
                }
                if aliases & seen_test_aliases:
                    conflicting_indexes = [
                        index
                        for index, existing in enumerate(test_records)
                        if {
                            (instance_key, str(existing.get("test_id") or "")),
                            (instance_key, str(existing.get("test_nodeid") or "")),
                        } & aliases
                    ]
                    conflict = any(
                        any(
                            existing.get(key) != record.get(key)
                            for key in ("f_to_p", "patch_hit", "measurement_valid")
                        )
                        for existing in (test_records[index] for index in conflicting_indexes)
                    )
                    if conflict:
                        for index in reversed(conflicting_indexes):
                            test_records.pop(index)
                        unavailable_population_rows += 1
                    continue
                seen_test_aliases.update(aliases)
                if record.get("measurement_valid") is False:
                    continue
                test_records.append(record)

        final_test_count = len(test_records)
        f_to_p_records = [record for record in test_records if record.get("f_to_p") is True]
        f_to_p_test_count = len(f_to_p_records)
        patch_hit_values = [record.get("patch_hit") for record in f_to_p_records]
        patch_hit_line_numerator = sum(
            int(r.get("patch_hit_numerator", 0) or 0)
            for r in f_to_p_records
            if r.get("patch_hit_numerator") is not None
        )
        patch_hit_line_denominator = sum(
            int(r.get("patch_hit_denominator", 0) or 0)
            for r in f_to_p_records
            if r.get("patch_hit_denominator") is not None
        )
        patch_hit_unavailable_count = sum(1 for value in patch_hit_values if value is None)
        checked_coverage_values = [
            float(record["checked_coverage"])
            for record in test_records
            if record.get("checked_coverage") is not None
        ]
        checked_coverage_f2p_values = [
            float(record["checked_coverage"])
            for record in f_to_p_records
            if record.get("checked_coverage") is not None
        ]

        total_iterations = sum(r["iterations"] for r in results)
        report = {
            "total": total,
            "aligned": aligned,
            "resolved": resolved,
            "final_eval_count": final_eval_count,
            "final_test_count": final_test_count,
            "f_to_p_test_count": f_to_p_test_count,
            "f_to_p_rate": (
                f_to_p_test_count / final_test_count if final_test_count else None
            ),
            "patch_hit_rate": (
                sum(1 for value in patch_hit_values if value is True) / f_to_p_test_count
                if f_to_p_test_count
                and len(patch_hit_values) == f_to_p_test_count
                and patch_hit_unavailable_count == 0
                else None
            ),
            "patch_hit_rate_f2p": (
                sum(1 for value in patch_hit_values if value is True) / f_to_p_test_count
                if f_to_p_test_count
                and len(patch_hit_values) == f_to_p_test_count
                and patch_hit_unavailable_count == 0
                else None
            ),
            "patch_hit_population": "T_F2P",
            "patch_hit_test_denominator": f_to_p_test_count,
            "patch_hit_line_numerator": patch_hit_line_numerator,
            "patch_hit_line_denominator": patch_hit_line_denominator or None,
            "patch_hit_unavailable_count": patch_hit_unavailable_count,
            "patch_hit_available_count": (
                len(patch_hit_values) - patch_hit_unavailable_count
            ),
            "metric_population_unavailable_row_count": unavailable_population_rows,
            "checked_coverage_final_mean": (
                sum(checked_coverage_values) / len(checked_coverage_values)
                if checked_coverage_values
                else None
            ),
            "checked_coverage_mean": (
                sum(checked_coverage_values) / len(checked_coverage_values)
                if checked_coverage_values
                else None
            ),
            "checked_coverage_f2p_mean": (
                sum(checked_coverage_f2p_values) / len(checked_coverage_f2p_values)
                if checked_coverage_f2p_values
                else None
            ),
            "failure_type_counts": ft_counts,
            "aligned_rate": f"{aligned / total * 100:.1f}%" if total else "0.0%",
            "resolve_rate": f"{resolved / total * 100:.1f}%" if total else "0.0%",
            "avg_iterations": round(total_iterations / total, 2) if total else 0.0,
            "per_instance": results,
        }

        return report

    @classmethod
    def _canonical_or_legacy_success(cls, entry: Dict[str, Any]) -> bool:
        if "f_to_p" in entry:
            return cls._canonical_m8_success(entry)
        return bool(entry.get("resolved") or cls._is_resolved(entry))

    @staticmethod
    def _excluded_from_aggregate_metrics(entry: Dict[str, Any]) -> bool:
        failure_record = entry.get("failure_record")
        if not isinstance(failure_record, dict):
            return False
        return failure_record.get("included_in_aggregate_metrics") is False

    @staticmethod
    def _canonical_m8_success(entry: Dict[str, Any]) -> bool:
        return (
            entry.get("admitted_to_final_set") is True
            and entry.get("match_status") == "MATCHED"
            and entry.get("before_patch_outcome") == "FAIL"
            and entry.get("after_patch_outcome") == "PASS"
            and entry.get("evaluation_status") == "SUCCESS"
            and entry.get("f_to_p") is True
            and not ResultCollector._excluded_from_aggregate_metrics(entry)
        )

    @staticmethod
    def save_report(report: Dict[str, Any], output_path: str) -> None:
        write_json_atomic(report, output_path)

    @staticmethod
    def print_summary(report: Dict[str, Any]) -> None:
        print(f"\n{'='*60}")
        print("  Final Report")
        print(f"{'='*60}")
        print(f"  total:              {report['total']}")
        print(f"  aligned:            {report['aligned']}")
        print(f"  resolved:           {report['resolved']}")
        print(f"  final_eval_count:   {report['final_eval_count']}")
        print(f"  aligned_rate:       {report['aligned_rate']}")
        print(f"  resolve_rate:       {report['resolve_rate']}")
        print(f"  avg_iterations:     {report['avg_iterations']}")
        for key, value in sorted(report["failure_type_counts"].items()):
            print(f"  {key:20s} {value}")


def _primary_checked_coverage_value(entry: Dict[str, Any]) -> Any:
    if entry.get("checked_coverage_mean") is not None:
        return entry.get("checked_coverage_mean")
    return entry.get("checked_coverage_final_mean")


def _artifact_belongs_to_instance_dir(
    instance_dir: Path,
    artifact: Dict[str, Any],
) -> bool:
    """Apply the production resume boundary's fail-closed ownership rule."""
    payload = artifact.get("payload") if isinstance(artifact.get("payload"), dict) else artifact
    artifact_instance_id = artifact.get("instance_id") or payload.get("instance_id")
    if artifact_instance_id is not None:
        return str(artifact_instance_id) == instance_dir.name
    for companion_name in (
        "m7_decision_record.json",
        "m6_execution_result.json",
        "alignment_execution.json",
    ):
        companion = read_json_object(instance_dir / companion_name)
        if not companion:
            continue
        companion_payload = (
            companion.get("payload")
            if isinstance(companion.get("payload"), dict)
            else companion
        )
        companion_instance_id = (
            companion.get("instance_id")
            or companion_payload.get("instance_id")
        )
        if companion_instance_id is not None:
            return str(companion_instance_id) == instance_dir.name
    return False


def main():
    parser = argparse.ArgumentParser(description="결과 수립기")
    parser.add_argument(
        "--output_root", type=str, default="outputs",
        help="인스턴스 결과 디렉토리 루트 (default: outputs)",
    )
    parser.add_argument(
        "--report_path", type=str, default="outputs/final_report.json",
        help="리포트 저장 경로 (default: outputs/final_report.json)",
    )
    parser.add_argument(
        "--instance_ids", type=str, default=None,
        help="대상 인스턴스 ID (comma-separated, default: 전체)",
    )
    args = parser.parse_args()

    ids = None
    if args.instance_ids:
        ids = [x.strip() for x in args.instance_ids.split(",") if x.strip()]

    collector = ResultCollector(args.output_root)
    report = collector.aggregate(instance_ids=ids)
    collector.save_report(report, args.report_path)
    collector.print_summary(report)
    print(f"\n  report → {args.report_path}")


if __name__ == "__main__":
    main()

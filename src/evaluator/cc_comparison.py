from __future__ import annotations

import hashlib
import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from src.contracts.instance_views import PrePatchInstanceView
from src.evaluator.dynamic_slice_cc import (
    CC_COLLECTION_ERROR,
    CC_EXECUTION_ERROR,
    CC_INSTRUMENTATION_ERROR,
    CC_SUPPORTED,
    CC_TIMEOUT,
    DynamicSliceCCResult,
)
from src.executor.m8_dynamic_slice_runner import M8DynamicSliceRequest
from src.utils.artifact_hash import sha256_text


CC_AVAILABLE = "CC_AVAILABLE"
CC_UNAVAILABLE = "CC_UNAVAILABLE"

CC_UNAVAILABLE_REASONS = frozenset(
    {
        "NO_EXECUTABLE_TEST",
        "INVALID_TEST",
        "PREPATCH_TEST_PASSED",
        "MISSING_ORACLE",
        "NO_SUT_LINES",
        "UNSUPPORTED_TRACING",
        "EXECUTION_TIMEOUT",
        "EXECUTION_ERROR",
        "ORACLE_EXTRACTION_ERROR",
        "DYNAMIC_SLICE_UNAVAILABLE",
        "OTHER",
    }
)


@dataclass(frozen=True)
class NormalizedSelectedTest:
    """One method-selected test at the method-independent CC boundary."""

    instance_id: str
    method: str
    model: str
    selected_test_id: str
    test_path: str | None
    test_nodeid: str | None
    test_patch: str
    selection_provenance: str
    source_artifact: str
    source_artifact_sha256: str
    adapter_status: str = "READY"
    adapter_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_id"] = self.selected_test_id
        payload["test_code"] = None
        payload["patch_reference"] = self.source_artifact
        return payload


class DynamicSliceRunner(Protocol):
    def run(self, request: M8DynamicSliceRequest) -> DynamicSliceCCResult: ...


class BaselineTestAdapter:
    """Normalize official final-selected e-Otter++ rows without reselection."""

    method = "e-Otter++"

    def load_official_selected_manifest(
        self,
        path: str | Path,
        *,
        model: str,
        expected_sha256: str | None = None,
    ) -> list[NormalizedSelectedTest]:
        artifact = Path(path)
        raw = artifact.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(
                f"official artifact checksum mismatch for {artifact}: "
                f"expected {expected_sha256}, got {digest}"
            )
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise ValueError("official e-Otter++ selected manifest must be a JSON list")

        seen: set[str] = set()
        selected: list[NormalizedSelectedTest] = []
        for index, row in enumerate(payload):
            if not isinstance(row, Mapping):
                raise ValueError(f"manifest row {index} is not an object")
            instance_id = str(row.get("instance_id") or "")
            if not instance_id:
                raise ValueError(f"manifest row {index} has no instance_id")
            if instance_id in seen:
                raise ValueError(f"duplicate e-Otter++ instance_id: {instance_id}")
            seen.add(instance_id)
            patch = str(row.get("model_patch") or "")
            test_path, test_nodeid, adapter_reason = derive_test_identity_from_patch(patch)
            selected.append(
                NormalizedSelectedTest(
                    instance_id=instance_id,
                    method=self.method,
                    model=model,
                    selected_test_id=sha256_text(patch),
                    test_path=test_path,
                    test_nodeid=test_nodeid,
                    test_patch=patch,
                    selection_provenance=(
                        "official e-Otter++ final-selected TDD-Bench Verified manifest row; "
                        "no local candidate reselection"
                    ),
                    source_artifact=artifact.name,
                    source_artifact_sha256=digest,
                    adapter_status="READY" if patch and test_nodeid else "CC_UNAVAILABLE",
                    adapter_reason=(
                        None if patch and test_nodeid else adapter_reason or "NO_EXECUTABLE_TEST"
                    ),
                )
            )
        return selected


class OurTestAdapter:
    """Normalize a previously admitted ALIGNED test without invoking M1-M7."""

    method = "ours"

    def normalize(
        self,
        row: Mapping[str, Any],
        *,
        source_artifact: str,
        source_artifact_sha256: str,
    ) -> NormalizedSelectedTest:
        status = str(row.get("m7_status") or row.get("status") or "")
        admitted = row.get("admitted_to_final_set")
        if status != "ALIGNED" or admitted is False:
            raise ValueError("our CC adapter accepts only M7 ALIGNED final tests")
        patch = str(row.get("test_patch") or "")
        nodeid = str(row.get("test_nodeid") or row.get("canonical_test_nodeid") or "")
        instance_id = str(row.get("instance_id") or "")
        if not instance_id:
            raise ValueError("our selected test has no instance_id")
        return NormalizedSelectedTest(
            instance_id=instance_id,
            method=self.method,
            model=str(row.get("model") or "unknown"),
            selected_test_id=str(row.get("candidate_sha256") or sha256_text(patch)),
            test_path=nodeid.split("::", 1)[0] if "::" in nodeid else None,
            test_nodeid=nodeid or None,
            test_patch=patch,
            selection_provenance="canonical M7 ALIGNED admission artifact",
            source_artifact=source_artifact,
            source_artifact_sha256=source_artifact_sha256,
            adapter_status="READY" if patch and nodeid else "CC_UNAVAILABLE",
            adapter_reason=None if patch and nodeid else "NO_EXECUTABLE_TEST",
        )


class CommonPostHocCCEvaluator:
    """Evaluate any normalized selected test with the same pre-patch M8 tracer."""

    def __init__(self, runner: DynamicSliceRunner, *, timeout_seconds: int = 600) -> None:
        self._runner = runner
        self._timeout_seconds = timeout_seconds

    def evaluate(
        self,
        selected: NormalizedSelectedTest,
        *,
        instance: PrePatchInstanceView,
        output_path: str | Path,
    ) -> dict[str, Any]:
        if selected.instance_id != instance.instance_id:
            raise ValueError(
                "cross-instance CC ownership mismatch: "
                f"test={selected.instance_id}, instance={instance.instance_id}"
            )
        if not selected.test_patch.strip() or not selected.test_nodeid:
            return _unavailable_row(
                selected,
                selected.adapter_reason or "NO_EXECUTABLE_TEST",
                prepatch_execution_status="NOT_EXECUTED",
                oracle_status="NOT_EVALUATED",
            )
        request = M8DynamicSliceRequest(
            instance=instance,
            test_nodeid=selected.test_nodeid,
            output_path=Path(output_path),
            generated_test_patch=selected.test_patch,
            generated_patch_sha256=sha256_text(selected.test_patch),
            timeout_seconds=self._timeout_seconds,
        )
        result = self._runner.run(request)
        return cc_result_row(selected, result)


def build_exact_instance_mapping(
    local_instance_ids: Sequence[str],
    selected_by_model: Mapping[str, Sequence[NormalizedSelectedTest]],
) -> list[dict[str, Any]]:
    """Build exact-ID mapping rows; fuzzy/title matching is intentionally absent."""
    if len(local_instance_ids) != len(set(local_instance_ids)):
        raise ValueError("local benchmark contains duplicate instance IDs")
    local = set(local_instance_ids)
    model_maps: dict[str, dict[str, NormalizedSelectedTest]] = {}
    unexpected_by_model: dict[str, set[str]] = {}
    for model, selected in selected_by_model.items():
        index: dict[str, NormalizedSelectedTest] = {}
        for test in selected:
            if test.instance_id in index:
                raise ValueError(f"duplicate {model} instance_id: {test.instance_id}")
            index[test.instance_id] = test
        model_maps[model] = index
        unexpected_by_model[model] = set(index) - local

    rows: list[dict[str, Any]] = []
    for instance_id in local_instance_ids:
        models = {
            model: {
                "status": "mapped" if instance_id in index else "missing",
                "selected_test_id": (
                    index[instance_id].selected_test_id if instance_id in index else None
                ),
                "adapter_status": (
                    index[instance_id].adapter_status if instance_id in index else None
                ),
                "adapter_reason": (
                    index[instance_id].adapter_reason if instance_id in index else None
                ),
            }
            for model, index in sorted(model_maps.items())
        }
        rows.append(
            {
                "instance_id": instance_id,
                "mapping_key": "exact_instance_id",
                "status": (
                    "mapped" if all(v["status"] == "mapped" for v in models.values()) else "missing"
                ),
                "models": models,
            }
        )
    for model, unexpected in sorted(unexpected_by_model.items()):
        for instance_id in sorted(unexpected):
            rows.append(
                {
                    "instance_id": instance_id,
                    "mapping_key": "exact_instance_id",
                    "status": "unexpected",
                    "models": {model: {"status": "unexpected"}},
                }
            )
    return rows


def derive_test_identity_from_patch(patch: str) -> tuple[str | None, str | None, str | None]:
    """Conservatively derive one pytest node ID from a unified test patch."""
    current_file: str | None = None
    class_context: str | None = None
    header_test: str | None = None
    current_context_test: str | None = None
    identities: set[tuple[str, str | None, str]] = set()
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            if not _looks_like_python_test_path(current_file):
                current_file = None
            continue
        if line.startswith("@@"):
            hunk_context = line.split("@@", 2)[-1].strip()
            class_match = re.search(r"\bclass\s+([A-Za-z_]\w*)", hunk_context)
            class_context = class_match.group(1) if class_match else None
            test_match = re.search(
                r"\b(?:async\s+)?def\s+(test[A-Za-z0-9_]*)\s*\(", hunk_context
            )
            header_test = test_match.group(1) if test_match else None
            current_context_test = header_test
            continue
        if current_file is None:
            continue
        added_class = re.match(r"^\+(\s*)class\s+([A-Za-z_]\w*)", line)
        if added_class:
            class_context = added_class.group(2)
            continue
        surrounding_test = re.match(
            r"^[ -](\s*)(?:async\s+)?def\s+(test[A-Za-z0-9_]*)\s*\(", line
        )
        if surrounding_test:
            current_context_test = surrounding_test.group(2)
            if line.startswith("-"):
                owner = class_context if len(surrounding_test.group(1).expandtabs(4)) > 0 else None
                identities.add((current_file, owner, current_context_test))
            continue
        added_test = re.match(
            r"^\+(\s*)(?:async\s+)?def\s+(test[A-Za-z0-9_]*)\s*\(", line
        )
        if added_test:
            owner = class_context if len(added_test.group(1).expandtabs(4)) > 0 else None
            identities.add((current_file, owner, added_test.group(2)))
            current_context_test = added_test.group(2)
            continue
        if not line.startswith(("+", "-")) or not line[1:].strip():
            continue
        modified_test = current_context_test or header_test
        if modified_test:
            owner = class_context if class_context else None
            identities.add((current_file, owner, modified_test))
    if len(identities) != 1:
        paths = sorted({identity[0] for identity in identities})
        file_path: str | None = None
        if len(paths) == 1:
            file_path = paths[0]
        return file_path, None, "NO_EXECUTABLE_TEST" if not identities else "INVALID_TEST"
    path, owner, function = next(iter(identities))
    nodeid = f"{path}::{owner}::{function}" if owner else f"{path}::{function}"
    return path, nodeid, None


def cc_result_row(
    selected: NormalizedSelectedTest,
    result: DynamicSliceCCResult,
) -> dict[str, Any]:
    diagnostics = dict(result.diagnostics)
    execution_status = str(diagnostics.get("execution_status") or "UNKNOWN")
    pytest_exit_code = diagnostics.get("pytest_exit_code")
    if pytest_exit_code == 0:
        return _unavailable_row(
            selected,
            "PREPATCH_TEST_PASSED",
            prepatch_execution_status="PASS",
            oracle_status="OBSERVED" if result.oracle_node else "NOT_OBSERVED",
            result=result,
        )
    if (
        result.status == CC_SUPPORTED
        and result.checked_coverage is not None
        and result.denominator is not None
        and result.denominator > 0
    ):
        return _base_row(
            selected,
            prepatch_execution_status=_prepatch_status(pytest_exit_code, execution_status),
            oracle_status="OBSERVED",
            result=result,
            cc_available=True,
            cc_unavailable_reason=None,
            cc=float(result.checked_coverage),
        )
    return _unavailable_row(
        selected,
        _canonical_unavailable_reason(result),
        prepatch_execution_status=_prepatch_status(pytest_exit_code, execution_status),
        oracle_status="OBSERVED" if result.oracle_node else "NOT_OBSERVED",
        result=result,
    )


def aggregate_cc_results(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in materialized:
        grouped.setdefault((str(row["method"]), str(row["model"])), []).append(row)
    by_group = {
        f"{method}|{model}": _aggregate_population(group_rows)
        for (method, model), group_rows in sorted(grouped.items())
    }
    instance_sets = [
        {str(row["instance_id"]) for row in group_rows if row.get("cc_available") is True}
        for group_rows in grouped.values()
    ]
    common_ids = set.intersection(*instance_sets) if instance_sets else set()
    common = {
        f"{method}|{model}": _aggregate_population(
            row for row in group_rows if str(row["instance_id"]) in common_ids
        )
        for (method, model), group_rows in sorted(grouped.items())
    }
    return {
        "schema_version": "cc-comparison-aggregate-v1",
        "full_population": by_group,
        "common_eligible_intersection": {
            "instance_count": len(common_ids),
            "instance_ids": sorted(common_ids),
            "groups": common,
        },
    }


def _base_row(
    selected: NormalizedSelectedTest,
    *,
    prepatch_execution_status: str,
    oracle_status: str,
    result: DynamicSliceCCResult | None,
    cc_available: bool,
    cc_unavailable_reason: str | None,
    cc: float | None,
) -> dict[str, Any]:
    covered = result.covered_sut_lines if result else []
    checked = result.checked_lines if result else []
    dynamic_count = len(checked) if result else 0
    return {
        "schema_version": "cc-result-v1",
        "instance_id": selected.instance_id,
        "method": selected.method,
        "model": selected.model,
        "selected_test_id": selected.selected_test_id,
        "prepatch_execution_status": prepatch_execution_status,
        "oracle_status": oracle_status,
        "sut_covered_line_count": len(covered),
        "dynamic_slice_line_count": dynamic_count,
        "checked_line_count": len(checked),
        "cc_status": CC_AVAILABLE if cc_available else CC_UNAVAILABLE,
        "cc_available": cc_available,
        "cc_unavailable_reason": cc_unavailable_reason,
        "cc": cc,
        "f_to_p_if_later_available": None,
        "source_artifact_sha256": selected.source_artifact_sha256,
    }


def _unavailable_row(
    selected: NormalizedSelectedTest,
    reason: str,
    *,
    prepatch_execution_status: str,
    oracle_status: str,
    result: DynamicSliceCCResult | None = None,
) -> dict[str, Any]:
    canonical_reason = reason if reason in CC_UNAVAILABLE_REASONS else "OTHER"
    return _base_row(
        selected,
        prepatch_execution_status=prepatch_execution_status,
        oracle_status=oracle_status,
        result=result,
        cc_available=False,
        cc_unavailable_reason=canonical_reason,
        cc=None,
    )


def _canonical_unavailable_reason(result: DynamicSliceCCResult) -> str:
    reason = str(result.diagnostics.get("reason") or "")
    if result.status == CC_TIMEOUT:
        return "EXECUTION_TIMEOUT"
    if result.status in {CC_COLLECTION_ERROR, CC_EXECUTION_ERROR}:
        return "EXECUTION_ERROR"
    if result.status == CC_INSTRUMENTATION_ERROR:
        if "patch_apply" in reason or "collection" in reason:
            return "INVALID_TEST"
        return "EXECUTION_ERROR"
    if reason == "empty_covered_sut_lines":
        return "NO_SUT_LINES"
    if reason == "missing_executed_oracle_observation":
        return "MISSING_ORACLE"
    if reason == "oracle_observation_has_no_supported_value_names":
        return "ORACLE_EXTRACTION_ERROR"
    if reason in {"oracle_value_has_no_sut_call_window", "no_dynamic_def_use_path_from_oracle"}:
        return "DYNAMIC_SLICE_UNAVAILABLE"
    if reason in {"pytest_unavailable", "unsupported_python_version"}:
        return "UNSUPPORTED_TRACING"
    return "OTHER"


def _prepatch_status(exit_code: Any, execution_status: str) -> str:
    if exit_code == 0:
        return "PASS"
    if exit_code == 1:
        return "FAIL"
    if execution_status == "not_executed":
        return "NOT_EXECUTED"
    return "ERROR" if exit_code is not None else execution_status


def _aggregate_population(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    population = [dict(row) for row in rows]
    available = [float(row["cc"]) for row in population if row.get("cc_available") is True]
    f2p = [row for row in population if row.get("f_to_p_if_later_available") is True]
    f2p_available = [float(row["cc"]) for row in f2p if row.get("cc_available") is True]
    return {
        "total_instances": len(population),
        "selected_tests_available": sum(
            row.get("prepatch_execution_status") != "NOT_EXECUTED" for row in population
        ),
        "cc_available_count": len(available),
        "cc_available_rate": len(available) / len(population) if population else None,
        "cc_mean": statistics.fmean(available) if available else None,
        "cc_median": statistics.median(available) if available else None,
        "f2p_count": len(f2p),
        "f2p_rate": len(f2p) / len(population) if population else None,
        "f2p_subset_cc_available_count": len(f2p_available),
        "f2p_subset_cc_mean": statistics.fmean(f2p_available) if f2p_available else None,
        "f2p_subset_cc_median": statistics.median(f2p_available) if f2p_available else None,
    }


def _looks_like_python_test_path(path: str) -> bool:
    lowered = path.lower()
    return path.endswith(".py") and (
        "/test" in lowered or Path(path).name.startswith("test_") or "/tests/" in lowered
    )

from __future__ import annotations

import copy
import ast
import hashlib
import json
import logging
import re
import shutil
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Sequence

logger = logging.getLogger(__name__)

from src.benchmark.instance_loader import BenchmarkInstance, TDDInstanceLoader
from src.issue_parser.issue_clues import IssueClueExtractor
from src.context_builder.code_context import CodeContextExtractor
from src.scenario.scenario_generator import ScenarioGenerator, bind_scenarios_to_localization_hypotheses
from src.scenario.scenario_ranker import (
    MAX_SELECTED_SCENARIOS,
    rank_scenarios,
    rank_v37_scenarios,
)
from src.scenario.scenario_validator import ScenarioValidator
from src.scenario.scenario_hydrator import hydrate_validation_report
from src.generator.repro_test_generator import (
    GenerationFailureError,
    ReproductionTestGenerator,
    apply_m5a_deterministic_postprocessing,
    candidate_repair_fingerprints,
    candidate_repair_semantics,
)
from src.generator.m5a_llm_error_refinement import refine_m5a_error_with_llm
from src.generator.repair_loop import (
    build_error_refinement_request,
    build_repair_evidence,
    extract_generated_code,
    is_repairable_category,
    make_m5a_telemetry,
    normalized_error_fingerprint,
    persist_quarantine_artifacts,
    persist_repair_attempt,
    repair_fingerprint,
    validation_status_from_errors,
)
from src.executor.alignment_runner import (
    AlignmentRunner,
    m6_execution_stability_exclusion_telemetry,
    normalize_pre_patch_execution_status,
)
from src.alignment.alignment_scorer import AlignmentScorer, compute_target_coverage_evidence
from src.alignment.v26_diagnosis import (
    DETERMINISTIC_FALLBACK,
    LLM_DIAGNOSIS,
    diagnose_m7,
    route_execution_plan,
    route_start_stage,
)
from src.contracts.models import M7DecisionRecord, M7FeedbackDecision
from src.contracts.status import (
    ExecutionStatus,
    M7DecisionStatus,
    ValidationStatus,
    coerce_m7_decision_status,
    legacy_failure_type_to_statuses,
)
from src.contracts.feature_flags import V22FeatureFlags, resolve_feature_flags
from src.models.client import LLMClient
from src.models.config import load_model_config
from src.utils.file_io import write_json, write_json_atomic
from src.utils.artifact_hash import build_evidence_reference, sha256_text
from src.utils.scenario_utils import ensure_primary_scenario, select_primary_scenario

DEFAULT_MAX_FEEDBACK_ITERATIONS = 5
V36_MAX_FEEDBACK_ITERATIONS = 3
V36_INSTANCE_TIME_BUDGET_SEC = 120
V36_EXPERIMENT_MODEL_KEYS = frozenset({"qwen3next", "gpt56luna", "codellama"})
V36_STAGE_MODEL_PARAMETERS: Dict[str, tuple[float, int]] = {
    "M2": (0.0, 1024),
    "M3": (0.2, 2048),
    "M5": (0.2, 4096),
    "M5-A": (0.0, 2048),
    "M7": (0.0, 1024),
}
# Compatibility alias for external imports. Active orchestration uses the
# validated max_feedback_iterations argument instead of this module constant.
MAX_ALIGNMENT_ITERATIONS = DEFAULT_MAX_FEEDBACK_ITERATIONS
MAX_M3_VALIDATION_ATTEMPTS = 1
MAX_V31_NEGATIVE_MEMORY = 12


def resolve_feedback_iteration_budget(
    feature_profile: str | None,
    requested: int,
) -> int:
    """Return the truthful total-pass budget for this methodology profile."""
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError("max_feedback_iterations must be a positive integer")
    if feature_profile == "v36":
        return min(requested, V36_MAX_FEEDBACK_ITERATIONS)
    return requested


def validate_v36_experiment_model_key(
    feature_profile: str | None,
    model_key: str,
) -> None:
    """Keep historical GPT-4o and legacy Qwen outside v36/v37 campaigns."""
    if feature_profile in {"v36", "v37"} and model_key not in V36_EXPERIMENT_MODEL_KEYS:
        raise ValueError(
            f"{feature_profile} experiments require one of "
            f"{sorted(V36_EXPERIMENT_MODEL_KEYS)!r}; got {model_key!r}"
        )


def make_v36_stage_client(model_key: str, stage: str) -> LLMClient:
    """Build one selected-model client with the v36 stage sampling contract."""
    if stage not in V36_STAGE_MODEL_PARAMETERS:
        raise ValueError(f"unsupported v36 LLM stage: {stage}")
    temperature, max_tokens = V36_STAGE_MODEL_PARAMETERS[stage]
    selected = load_model_config(model_key)
    return LLMClient(
        replace(selected, temperature=temperature, max_tokens=max_tokens)
    )


def validate_selected_llm_stage_clients(
    model_key: str,
    stage_clients: Mapping[str, Any],
    *,
    methodology_revision: str = "v36",
) -> Dict[str, Any]:
    """Fail closed unless every LLM stage uses the selected experiment config."""
    selected = load_model_config(model_key)
    stages: Dict[str, Any] = {}
    for stage, client in stage_clients.items():
        config = getattr(client, "config", None)
        if config is None:
            raise RuntimeError(f"{stage} has no selected-model configuration")
        actual = {
            "provider": getattr(config, "provider", None),
            "model_name": getattr(config, "model_name", None),
            "base_url": getattr(config, "base_url", None),
            "temperature": getattr(config, "temperature", None),
            "max_tokens": getattr(config, "max_tokens", None),
        }
        expected_temperature, expected_max_tokens = V36_STAGE_MODEL_PARAMETERS[stage]
        expected = {
            "provider": selected.provider,
            "model_name": selected.model_name,
            "base_url": selected.base_url,
            "temperature": expected_temperature,
            "max_tokens": expected_max_tokens,
        }
        if actual != expected:
            raise RuntimeError(
                f"{stage} model configuration diverges from --model {model_key}: "
                f"actual={actual}, expected={expected}"
            )
        stages[stage] = actual
    return {
        "schema_version": f"{methodology_revision}-llm-stage-model-resolution-v1",
        "selected_model_key": model_key,
        "selected_provider": selected.provider,
        "selected_model_name": selected.model_name,
        "stages": stages,
        "automatic_gpt4o_fallback": False,
    }


def _attach_m5a_deterministic_actions(
    telemetry: Dict[str, Any],
    actions: list[str],
) -> Dict[str, Any]:
    """Attach mandatory M5-A rule actions without changing repair semantics."""
    feature = telemetry.get("enable_m5a_llm_error_refinement")
    if isinstance(feature, dict):
        feature["deterministic_postprocessing_actions"] = list(actions)
    return telemetry


def _selected_m7_client(generator: Any) -> Any:
    """Return the stage-specific v36 M7 client or the historical shared client."""
    return getattr(generator, "m7_client", getattr(generator, "client", None))


def _construct_issue_clue_extractor(llm_client: Any = None) -> Any:
    try:
        return IssueClueExtractor(llm_client=llm_client)
    except TypeError:
        return IssueClueExtractor()


def _construct_code_context_extractor(
    *,
    history_window: int | None,
    llm_client: Any = None,
    isolate_instance_checkout: bool = False,
    instance_view_root: str | Path | None = None,
    feature_profile: str | None = None,
) -> Any:
    try:
        return CodeContextExtractor(
            history_window=history_window,
            llm_client=llm_client,
            isolate_instance_checkout=isolate_instance_checkout,
        instance_view_root=instance_view_root,
            feature_profile=feature_profile,
        )
    except TypeError:
        return CodeContextExtractor(history_window=history_window)


def _fault_locations_from_m2_context(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project a deterministic location clue from M2's ranked pre-patch context."""
    candidates = context.get("candidate_source_files") or []
    if not isinstance(candidates, list):
        return []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        path = str(candidate.get("path") or "").strip()
        functions = candidate.get("top_level_functions") or []
        function_name = str(functions[0] if functions else "").strip()
        if path and function_name:
            return [{
                "file_path": path,
                "function_name": function_name,
                "line_no": 0,
                "inferred": True,
                "source": "m2_ranked_context",
                "confidence": "medium",
            }]
    return []


def _classify_generation_failure_detail(message: str) -> str:
    text = str(message or "")
    lower = text.lower()
    if "syntaxerror" in lower or "invalid syntax" in lower:
        return "SYNTAX_ERROR"
    if "importerror" in lower or "modulenotfounderror" in lower or "missing import" in lower:
        return "IMPORT_ERROR"
    if "target_test_file" in lower or "missing test file" in lower or "test file" in lower and "not found" in lower:
        return "MISSING_TEST_FILE"
    if "oracle" in lower and (
        "risk" in lower
        or "rejected" in lower
        or "blocking" in lower
        or "critical" in lower
    ):
        return "ORACLE_REJECTED"
    if "semantic risk" in lower or "semantic_risk" in lower:
        return "SEMANTIC_RISK"
    return "GENERATION_FAILED"


def _negative_memory_from_generation_failure(
    *,
    instance_id: str,
    iteration: int,
    error: BaseException,
    candidate_sha256: str = "",
) -> list[dict[str, Any]]:
    """Extract bounded, exact, pre-patch correction memory from M5 rejection."""
    error_texts = [str(error)]
    if isinstance(error, GenerationFailureError):
        error_texts = [
            str(item)
            for item in [
                *(error.validation_errors or []),
                error.last_error,
                str(error),
            ]
            if str(item or "").strip()
        ]
        if not candidate_sha256 and error.raw_candidate:
            candidate_sha256 = sha256_text(error.raw_candidate)
    combined = "\n".join(error_texts)
    entries: list[dict[str, Any]] = []

    def add(category: str, rejected: str, reason: str, prohibition: str) -> None:
        rejected = rejected.strip()[:500]
        if not rejected:
            return
        identity = sha256_text(
            f"{instance_id}\0{iteration}\0{category}\0{rejected}\0{candidate_sha256}"
        )
        entry = {
            "schema_version": "v31-negative-memory-v1",
            "memory_id": identity,
            "instance_id": instance_id,
            "source_iteration": iteration,
            "owner_module": "M5",
            "candidate_sha256": candidate_sha256 or None,
            "category": category,
            "rejected_choice": rejected,
            "reason": reason.strip()[:600],
            "prohibition": prohibition.strip()[:600],
            "repository_alternatives": [],
            "provenance": "prepatch_validation_failure",
        }
        if entry["memory_id"] not in {item["memory_id"] for item in entries}:
            entries.append(entry)

    for match in re.finditer(r"invalid import:\s*([^;\n]+)", combined, flags=re.IGNORECASE):
        rejected = match.group(1).strip()
        add(
            "REJECTED_IMPORT",
            rejected,
            f"repository import validation rejected `{rejected}`",
            f"Do not emit the exact rejected import `{rejected}`; use only repository-verified imports.",
        )
    for match in re.finditer(
        r"validated target function ['\"]([^'\"]+)['\"]",
        combined,
        flags=re.IGNORECASE,
    ):
        rejected = match.group(1).strip()
        add(
            "DISPROVEN_TARGET_INVOCATION",
            rejected,
            "the candidate did not exercise the validated target",
            f"Do not repeat `{rejected}` as a mandatory direct-invocation target without new M2/M3 evidence.",
        )
    patterns = (
        (
            "REJECTED_ORACLE_PATTERN",
            "local_object_identity_oracle",
            "Do not assert identity against a locally-created object; assert issue-supported behavior, state, or value.",
        ),
        (
            "REJECTED_ORACLE_PATTERN",
            "negative_literal_oracle",
            "Do not repeat a negative-literal-only oracle; assert the issue-supported expected behavior.",
        ),
        (
            "MISSING_EXPLICIT_ORACLE",
            "no explicit oracle",
            "Include an explicit behavior/state/value assertion derived from the M1/M3 expected behavior.",
        ),
        (
            "INVALID_FRAMEWORK_SHAPE",
            "django-test runner requires a class inheriting from django.test.TestCase or SimpleTestCase",
            "Use the repository's Django TestCase/SimpleTestCase class structure from the target test file.",
        ),
        (
            "INVALID_FRAMEWORK_SHAPE",
            "do not define Django models inside generated tests",
            "Do not define Django models locally; reuse repository test models/helpers and framework structure.",
        ),
        (
            "INVALID_TEST_SHAPE",
            "generated append_block must define at least one test function/method",
            "The append block must define exactly one collectable test function or method.",
        ),
        (
            "ORACLE_EXPECTED_BEHAVIOR_NOT_PRESERVED",
            "expected_behavior_oracle_not_preserved",
            "Preserve the explicit M1/M3 expected behavior in the assertion.",
        ),
    )
    lowered = combined.lower()
    for category, marker, prohibition in patterns:
        if marker.lower() in lowered:
            add(category, marker, marker, prohibition)
    return entries[:MAX_V31_NEGATIVE_MEMORY]


def _merge_negative_memory(
    current: list[dict[str, Any]],
    additions: list[dict[str, Any]],
    *,
    instance_id: str,
) -> list[dict[str, Any]]:
    """Merge instance-owned memory by semantic choice and keep a hard bound."""
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in [*current, *additions]:
        if str(entry.get("instance_id") or "") != instance_id:
            continue
        key = (str(entry.get("category") or ""), str(entry.get("rejected_choice") or ""))
        if all(key):
            merged[key] = copy.deepcopy(entry)
    values = sorted(
        merged.values(),
        key=lambda entry: (int(entry.get("source_iteration") or 0), str(entry.get("memory_id") or "")),
    )
    return values[-MAX_V31_NEGATIVE_MEMORY:]


def _inject_negative_memory(
    validation_dict: Mapping[str, Any],
    memory: list[dict[str, Any]],
    *,
    instance_id: str,
) -> dict[str, Any]:
    """Attach only current-instance bounded memory to every M5-eligible scenario."""
    updated = copy.deepcopy(dict(validation_dict))
    owned = [
        copy.deepcopy(entry)
        for entry in memory[-MAX_V31_NEGATIVE_MEMORY:]
        if str(entry.get("instance_id") or "") == instance_id
    ]
    for bucket in ("selected_scenarios", "rejected_scenarios"):
        for record in updated.get(bucket, []) or []:
            if not isinstance(record, dict):
                continue
            scenario = record.get("normalized_scenario")
            if isinstance(scenario, dict):
                scenario["negative_memory"] = copy.deepcopy(owned)
                scenario["negative_memory_schema_version"] = "v31-negative-memory-v1"
    return updated


def process_instance(
    instance: BenchmarkInstance,
    output_dir: str,
    model_key: str = "qwen",
    feature_flags: V22FeatureFlags | Mapping[str, Any] | None = None,
    *,
    history_window: int | None = None,
    feature_profile: str | None = None,
    max_feedback_iterations: int = DEFAULT_MAX_FEEDBACK_ITERATIONS,
    instance_view_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Run one instance and guarantee cleanup of pipeline-owned source views."""
    validate_v36_experiment_model_key(feature_profile, model_key)
    if feature_profile == "v36":
        max_feedback_iterations = resolve_feedback_iteration_budget(
            feature_profile, max_feedback_iterations
        )
    isolated_source_view_enabled = feature_profile in {"v27r1", "v29", "v30", "v31", "v36", "v37"}
    completed_without_error = False
    try:
        result = _process_instance_impl(
            instance,
            output_dir,
            model_key=model_key,
            feature_flags=feature_flags,
            history_window=history_window,
            feature_profile=feature_profile,
            max_feedback_iterations=max_feedback_iterations,
            instance_view_root=instance_view_root,
        )
        completed_without_error = True
        return result
    finally:
        if isolated_source_view_enabled:
            try:
                release_kwargs: dict[str, Any] = {
                    "repos_root": "data/repos",
                    "repo_name": instance.repo,
                    "instance_id": instance.instance_id,
                }
                if instance_view_root is not None:
                    release_kwargs["instance_view_root"] = instance_view_root
                lifecycle = CodeContextExtractor.release_instance_view(
                    **release_kwargs,
                )
            except Exception as exc:
                lifecycle = {
                    "schema_version": "kcc-instance-view-v29-v1",
                    "instance_id": instance.instance_id,
                    "repo_name": instance.repo,
                    "source_view_path": None,
                    "created_at": None,
                    "cleaned_at": datetime.now(timezone.utc).isoformat(),
                    "cleanup_status": "CLEANUP_FAILED",
                    "cleanup_error": str(exc),
                    "base_commit": instance.base_commit,
                    "disk_usage_before": None,
                    "disk_usage_after": None,
                }
            write_json_atomic(lifecycle, Path(output_dir) / "worktree_lifecycle.json")
            if completed_without_error and lifecycle.get("cleanup_status") not in {
                "CLEANED",
                "NOT_ACQUIRED",
            }:
                raise RuntimeError(
                    "pipeline-owned source-view cleanup failed: "
                    f"{lifecycle.get('cleanup_error') or lifecycle.get('cleanup_status')}"
                )


def _serialize_m2_context(context: Any, feature_profile: str | None) -> Dict[str, Any]:
    """Serialize every initial/rerun M2 context with stable profile identity."""
    payload = context.to_dict()
    payload["feature_profile"] = feature_profile
    payload["methodology_revision"] = feature_profile
    return payload


def _process_instance_impl(
    instance: BenchmarkInstance,
    output_dir: str,
    model_key: str = "qwen",
    feature_flags: V22FeatureFlags | Mapping[str, Any] | None = None,
    *,
    history_window: int | None = None,
    feature_profile: str | None = None,
    max_feedback_iterations: int = DEFAULT_MAX_FEEDBACK_ITERATIONS,
    instance_view_root: str | Path | None = None,
) -> Dict[str, Any]:
    """하나의 인스턴스에 대해 전체 파이프라인을 실행하고 결과를 반환한다.

    Returns:
        {"instance_id", "failure_type", "final_score", "iterations", "error"}
    """
    pipeline_started_at = time.monotonic()
    resolved_feature_flags = (
        feature_flags if isinstance(feature_flags, V22FeatureFlags)
        else resolve_feature_flags(feature_flags)
    )
    v27_enabled = feature_profile in {"v27", "v27r1"}
    v29_enabled = feature_profile == "v29"
    v30_enabled = feature_profile == "v30"
    v31_enabled = feature_profile == "v31"
    v36_enabled = feature_profile == "v36"
    v37_enabled = feature_profile == "v37"
    strict_v36_or_v37 = v36_enabled or v37_enabled
    if v36_enabled:
        max_feedback_iterations = resolve_feedback_iteration_budget(
            feature_profile, max_feedback_iterations
        )
    v30_or_v31 = v30_enabled or v31_enabled
    current_m7_policy_enabled = feature_profile in {"v29", "v30", "v31", "v36", "v37"}
    m7_diagnosis_revision = (
        "v37"
        if v37_enabled
        else "v36"
        if v36_enabled
        else "v29"
        if current_m7_policy_enabled
        else "v27"
        if v27_enabled
        else "v26"
    )
    isolated_source_view_enabled = feature_profile in {"v27r1", "v29", "v30", "v31", "v36", "v37"}
    _validate_history_window(history_window)
    _validate_max_feedback_iterations(max_feedback_iterations)
    pre_patch_view = instance.to_pre_patch_view()
    if strict_v36_or_v37:
        m2_llm_client = make_v36_stage_client(model_key, "M2")
        scenario_generator = ScenarioGenerator(
            client=make_v36_stage_client(model_key, "M3"),
            feature_profile=feature_profile,
        )
    else:
        # Preserve the historical constructor contract for legacy profiles and
        # their injected test doubles.  The stricter profile is a v36 concern.
        scenario_generator = ScenarioGenerator(model_key=model_key)
        m2_llm_client = getattr(scenario_generator, "client", None)
    shared_llm_client = getattr(scenario_generator, "client", None)
    if strict_v36_or_v37:
        validate_selected_llm_stage_clients(
            model_key,
            {"M2": m2_llm_client, "M3": shared_llm_client},
            methodology_revision=feature_profile or "v36",
        )

    # ── Stage 1: Issue Clue 추출 ──
    clue_extractor = _construct_issue_clue_extractor(
        shared_llm_client if resolved_feature_flags.m1_llm_refinement else None
    )
    m1_started_at = time.monotonic()
    clue = clue_extractor.extract(
        instance_id=instance.instance_id,
        issue_text=instance.problem_statement,
        feature_flags=resolved_feature_flags,
    )
    current_m1_elapsed_sec: float | None = round(time.monotonic() - m1_started_at, 3)

    clue_output_path = f"{output_dir}/clue.json"
    clue_extractor.save(clue, clue_output_path)
    clue_dict = clue.to_dict()
    if v30_or_v31:
        clue_dict.setdefault("metadata", {})["feature_profile"] = feature_profile
        write_json_atomic(clue_dict, clue_output_path)

    # ── Stage 2: Code Context 추출 ──
    context_extractor = _construct_code_context_extractor(
        history_window=history_window,
        llm_client=m2_llm_client if resolved_feature_flags.m2_llm_semantic_matching else None,
        isolate_instance_checkout=isolated_source_view_enabled,
        instance_view_root=instance_view_root,
        feature_profile=feature_profile,
    )
    m2_t0 = time.time()
    context = context_extractor.extract(
        instance=instance,
        clue=clue_dict,
        feature_flags=resolved_feature_flags,
    )
    current_context_reuse_key = _m2_context_reuse_key(
        instance_id=instance.instance_id,
        clue_dict=clue_dict,
        feature_flags=resolved_feature_flags,
        history_window=history_window,
        restart_feedback=None,
    )
    current_m2_elapsed_sec: float | None = round(time.time() - m2_t0, 3)

    context_output_path = f"{output_dir}/context.json"
    context_extractor.save(context, context_output_path)
    context_dict = _serialize_m2_context(context, feature_profile)
    if v30_or_v31:
        write_json_atomic(context_dict, context_output_path)

    # ── Stage 2.5: fault location 추론 (traceback 없는 경우) ──
    if not clue_dict.get("fault_locations"):
        inferred = _fault_locations_from_m2_context(context_dict)
        if inferred:
            clue_dict = dict(clue_dict)
            clue_dict["fault_locations"] = inferred
            logger.info("Inferred fault locations (no traceback): %s", inferred)

    scenario_output_path = f"{output_dir}/scenario.json"
    scenario_validator = ScenarioValidator()
    validation_output_path = f"{output_dir}/scenario_validation.json"
    m3_t0 = time.time()
    scenario, validation_dict = _generate_eligible_scenarios_with_retries(
        scenario_generator=scenario_generator,
        scenario_validator=scenario_validator,
        instance=instance,
        clue_dict=clue_dict,
        context_dict=context_dict,
        feature_flags=resolved_feature_flags,
        iteration=1,
        scenario_output_path=scenario_output_path,
        validation_output_path=validation_output_path,
    )
    initial_stage_timings = validation_dict.get("v26_module_timings", {})
    current_m3_elapsed_sec: float | None = initial_stage_timings.get(
        "m3_elapsed_sec",
        round(time.time() - m3_t0, 3),
    )
    current_m4_elapsed_sec: float | None = initial_stage_timings.get("m4_elapsed_sec")
    pending_m1_elapsed_sec: float | None = current_m1_elapsed_sec
    pending_m2_elapsed_sec: float | None = current_m2_elapsed_sec
    pending_m3_elapsed_sec: float | None = current_m3_elapsed_sec
    pending_m4_elapsed_sec: float | None = current_m4_elapsed_sec

    # force-selected 시나리오 경고 로그
    for sel in validation_dict.get("selected_scenarios", []):
        if sel.get("force_selected"):
            print(f"  ⚠ force-selected scenario: {sel.get('scenario_id')} (score={sel.get('score', 0):.2f})")

    _print_summary(instance, clue_output_path, context_output_path,
                   scenario_output_path, scenario, context)

    # ── Stage 5-6: generation/alignment loop with a validated total-pass budget ──
    repro_test_generator = ReproductionTestGenerator(
        client=make_v36_stage_client(model_key, "M5") if strict_v36_or_v37 else None,
        model_key=model_key,
        feature_flags=resolved_feature_flags,
        feature_profile=feature_profile,
    )
    if strict_v36_or_v37:
        repro_test_generator.m5a_client = make_v36_stage_client(model_key, "M5-A")
        repro_test_generator.m7_client = make_v36_stage_client(model_key, "M7")
    llm_stage_model_resolution = (
        validate_selected_llm_stage_clients(
            model_key,
            {
                "M2": m2_llm_client,
                "M3": shared_llm_client,
                "M5": repro_test_generator.client,
                "M5-A": repro_test_generator.m5a_client,
                "M7": repro_test_generator.m7_client,
            },
            methodology_revision=feature_profile or "v36",
        )
        if strict_v36_or_v37
        else {}
    )
    context_dict.setdefault("metadata", {})[
        "llm_stage_model_resolution"
    ] = llm_stage_model_resolution
    write_json_atomic(context_dict, context_output_path)
    try:
        alignment_runner = AlignmentRunner(feature_profile=feature_profile)
    except TypeError:
        # Narrow test/dry-run adapters may expose the historical no-argument
        # runner constructor; preserve that compatibility boundary.
        alignment_runner = AlignmentRunner()
    scorer = AlignmentScorer()
    # The scorer owns deterministic scores/gates only in v26.  Its legacy
    # advisory LLM path remains available to direct compatibility callers, but
    # active orchestration has one diagnosis authority in v26_diagnosis.py.
    scorer_feature_flags = replace(
        resolved_feature_flags,
        m7_llm_scenario_refinement=False,
    )

    # 시나리오 생성 시점에 clue 필드가 이미 포함되므로 별도 merge 불필요
    current_validation_dict = _ensure_issue_grounded_fallback_if_needed(
        validation_dict=validation_dict,
        scenario_generator=scenario_generator,
        scenario_validator=scenario_validator,
        instance=instance,
        clue_dict=clue_dict,
        context_dict=context_dict,
        feature_flags=resolved_feature_flags,
        iteration=1,
        scenario_output_path=scenario_output_path,
        validation_output_path=validation_output_path,
        reason="initial_generation_guard",
    )
    alignment_result = None
    iteration = 0
    runtime_error_for_next: str | None = None
    previous_runtime_error_fingerprint: str | None = None
    iteration_history: list[dict[str, Any]] = []
    pending_m7_feedback_application: dict[str, Any] | None = None
    active_prohibited_fingerprints: list[str] = []
    m2_prohibited_source_files: list[str] = []
    m2_prohibited_targets: list[dict[str, str]] = []
    active_negative_memory: list[dict[str, Any]] = []
    # 전체 iteration에 걸친 토큰 누적
    total_token_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    modules_for_current_pass = ["M1", "M2", "M3", "M4", "M5", "M5-A", "M6", "M7"]
    pending_pass_started_at: float | None = pipeline_started_at
    retained_m6_candidate: Any | None = None
    best_candidate_record: dict[str, Any] | None = None
    best_generated_test: Any | None = None
    best_alignment_result: Any | None = None
    best_iteration_dir: Path | None = None
    last_candidate_record: dict[str, Any] | None = None
    # v31 currently has exact M5 accounting, while M3/M7 model calls expose
    # call counts/artifacts but not provider token counts. Mark the cumulative
    # pipeline total incomplete instead of claiming that M5's subtotal is the
    # full instance cost.
    observed_token_usage_statuses: list[str] = ["unknown"] if v31_enabled else []

    for iteration in range(1, max_feedback_iterations + 1):
        pass_started_at = pending_pass_started_at or time.monotonic()
        pending_pass_started_at = None
        print(f"\n{'='*60}")
        print(f"  Alignment Loop — iteration {iteration}/{max_feedback_iterations}")
        print(f"{'='*60}")
        iteration_dir = _iteration_artifact_dir(output_dir, iteration)
        modules_actually_executed_this_pass = list(modules_for_current_pass)
        iteration_m1_elapsed_sec = pending_m1_elapsed_sec
        iteration_m2_elapsed_sec = pending_m2_elapsed_sec
        iteration_m3_elapsed_sec = pending_m3_elapsed_sec
        iteration_m4_elapsed_sec = pending_m4_elapsed_sec
        pending_m1_elapsed_sec = None
        pending_m2_elapsed_sec = None
        pending_m3_elapsed_sec = None
        pending_m4_elapsed_sec = None

        _clear_harness_cache(instance.instance_id, output_dir, alignment=True)

        # ── 테스트 생성 (Algorithm 1, line 1: t=null → NOT_VALID) ──
        generation_repair_telemetry: Dict[str, Any] = {}
        generation_usage_already_accounted = False
        v37_generated_candidate_queue: list[dict[str, Any]] = []
        v37_generation_records: list[dict[str, Any]] = []
        v37_candidate_outcomes: list[dict[str, Any]] = []
        active_v37_scenario: dict[str, Any] | None = None
        try:
            if retained_m6_candidate is not None and modules_for_current_pass == ["M6", "M7"]:
                generated_test = retained_m6_candidate
                retained_m6_candidate = None
                m5_elapsed_sec: float | None = 0.0
            else:
                current_validation_dict = _ensure_issue_grounded_fallback_if_needed(
                    validation_dict=current_validation_dict,
                    scenario_generator=scenario_generator,
                    scenario_validator=scenario_validator,
                    instance=instance,
                    clue_dict=clue_dict,
                    context_dict=context_dict,
                    feature_flags=resolved_feature_flags,
                    iteration=iteration,
                    scenario_output_path=scenario_output_path,
                    validation_output_path=validation_output_path,
                    reason=f"iteration_{iteration}_empty_selection_guard",
                )
                current_validation_dict = _limit_selected_scenarios(current_validation_dict)
                if not _has_m5_eligible_selected(current_validation_dict):
                    raise GenerationFailureError(
                        message="M5 has no M4-validated eligible scenario",
                        token_usage={},
                        attempt_count=0,
                        last_error="repaired_scenario_not_m4_validated",
                        failure_type_detail="M5_INPUT_CONTRACT",
                        token_usage_status="no_llm_call",
                        scenario=_safe_primary_scenario(current_validation_dict),
                    )
                m5_t0 = time.time()
                if v37_enabled:
                    (
                        v37_generated_candidate_queue,
                        v37_generation_records,
                    ) = _generate_v37_ranked_candidates(
                        generator=repro_test_generator,
                        instance=instance,
                        clue=clue_dict,
                        context=context_dict,
                        validation_report=current_validation_dict,
                        iteration=iteration,
                        runtime_error_hint=runtime_error_for_next,
                    )
                    if not v37_generated_candidate_queue:
                        raise GenerationFailureError(
                            message="all selected v37 scenarios failed M5 generation",
                            token_usage={},
                            attempt_count=len(v37_generation_records),
                            last_error="all_selected_scenarios_generation_failed",
                            failure_type_detail="MODEL_OUTPUT_SCHEMA",
                            token_usage_status="unknown",
                            scenario=_safe_primary_scenario(current_validation_dict),
                        )
                    active_candidate = v37_generated_candidate_queue.pop(0)
                    generated_test = active_candidate["generated_test"]
                    active_v37_scenario = dict(active_candidate["scenario"])
                else:
                    generated_test = repro_test_generator.generate(
                        instance=instance,
                        clue=clue_dict,
                        context=context_dict,
                        validation_report=current_validation_dict,
                        iteration=iteration,
                        runtime_error_hint=runtime_error_for_next,
                    )
                m5_elapsed_sec = round(time.time() - m5_t0, 3)
        except Exception as gen_err:
            logger.warning("Test generation failed (t=null): %s", gen_err)
            diagnosis = f"Test generation failed: {gen_err}"
            failed_usage = (
                dict(gen_err.token_usage)
                if isinstance(gen_err, GenerationFailureError)
                else {}
            )
            for k in total_token_usage:
                total_token_usage[k] += int(failed_usage.get(k, 0) or 0)
            generation_usage_already_accounted = True
            current_usage_status = (
                gen_err.token_usage_status
                if isinstance(gen_err, GenerationFailureError)
                else "no_llm_call"
                if total_token_usage.get("total_tokens", 0) == 0
                else "known"
            )
            observed_token_usage_statuses.append(current_usage_status)
            token_usage_status = _cumulative_token_usage_status(
                total_token_usage,
                observed_token_usage_statuses,
            )
            failure_type_detail = (
                gen_err.failure_type_detail
                if isinstance(gen_err, GenerationFailureError)
                else _classify_generation_failure_detail(diagnosis)
            )
            if failure_type_detail == "M5_INPUT_CONTRACT":
                modules_actually_executed_this_pass = [
                    module
                    for module in modules_actually_executed_this_pass
                    if module in {"M1", "M2", "M3", "M4"}
                ]
            repaired_from_generation_failure = None
            if (
                isinstance(gen_err, GenerationFailureError)
                and failure_type_detail != "M5_INPUT_CONTRACT"
            ):
                repaired_from_generation_failure, generation_repair_telemetry = (
                    _attempt_generation_failure_m5a_repair(
                        instance=instance,
                        output_dir=iteration_dir,
                        generator=repro_test_generator,
                        error=gen_err,
                        clue=clue_dict,
                        context=context_dict,
                        validation_report=current_validation_dict,
                        feature_flags=resolved_feature_flags,
                        iteration=iteration,
                        feature_profile=feature_profile,
                    )
                )
            if repaired_from_generation_failure is not None:
                generated_test = repaired_from_generation_failure
                m5_elapsed_sec = None
            else:
                repair_feature = generation_repair_telemetry.get(
                    "enable_m5a_llm_error_refinement",
                    {},
                )
                repair_validation_errors = list(
                    repair_feature.get("post_repair_validation_errors") or []
                )
                if repair_validation_errors:
                    diagnosis += (
                        "\nM5-A deterministic revalidation failed: "
                        + "; ".join(str(item) for item in repair_validation_errors[:8])
                    )
                if generation_repair_telemetry:
                    _write_feature_execution_telemetry(iteration_dir, generation_repair_telemetry)
                    _write_feature_execution_telemetry(Path(output_dir), generation_repair_telemetry)
                invalid_generated_test = None
                if isinstance(gen_err, GenerationFailureError):
                    invalid_generated_test = repro_test_generator.build_invalid_generated_test_from_failure(
                        instance=instance,
                        error=gen_err,
                        context=context_dict,
                        iteration=iteration,
                    )
                if invalid_generated_test is not None:
                    invalid_path = str(iteration_dir / "generated_test.json")
                    repro_test_generator.save(invalid_generated_test, invalid_path)
                    _sync_latest_iteration_aliases(output_dir, iteration_dir)
                failure_memory = _negative_memory_from_generation_failure(
                    instance_id=instance.instance_id,
                    iteration=iteration,
                    error=gen_err,
                    candidate_sha256=str(
                        getattr(invalid_generated_test, "generated_patch_sha256", "")
                        or getattr(invalid_generated_test, "patch_sha256", "")
                        or ""
                    ),
                )
                active_negative_memory = _merge_negative_memory(
                    active_negative_memory,
                    failure_memory,
                    instance_id=instance.instance_id,
                )
                if any(
                    entry.get("category") == "DISPROVEN_TARGET_INVOCATION"
                    for entry in failure_memory
                ):
                    failure_scenario = (
                        gen_err.scenario
                        if isinstance(gen_err, GenerationFailureError)
                        and isinstance(gen_err.scenario, Mapping)
                        else _safe_primary_scenario(current_validation_dict)
                    )
                    failed_target = _m2_target_for_exclusion(
                        failure_scenario,
                        invalid_generated_test,
                    )
                    if failed_target and failed_target not in m2_prohibited_targets:
                        m2_prohibited_targets.append(failed_target)
                write_json_atomic(
                    {
                        "schema_version": "v31-negative-memory-v1",
                        "instance_id": instance.instance_id,
                        "outer_iteration": iteration,
                        "entries": active_negative_memory,
                        "entry_count": len(active_negative_memory),
                        "max_entries": MAX_V31_NEGATIVE_MEMORY,
                    },
                    iteration_dir / "negative_memory.json",
                )
                if alignment_result is not None:
                    logger.warning(
                        "Continuing unified iteration loop after repair generation failed "
                        "(iteration=%s, previous_failure_type=%s): %s",
                        iteration,
                        alignment_result.failure_type,
                        gen_err,
                    )
                generation_failure_fingerprint = normalized_error_fingerprint(diagnosis)
                repeated_generation_failure = (
                    bool(generation_failure_fingerprint)
                    and generation_failure_fingerprint == previous_runtime_error_fingerprint
                )
                m4_m5_eligibility_signature = _m4_m5_eligibility_signature(
                    current_validation_dict
                )
                prior_identical_input_contract = bool(
                    failure_type_detail == "M5_INPUT_CONTRACT"
                    and any(
                        str(record.get("failure_type_detail") or "")
                        == "M5_INPUT_CONTRACT"
                        and record.get("m4_m5_eligibility_signature")
                        == m4_m5_eligibility_signature
                        for record in iteration_history
                    )
                )
                generation_rerun_snapshot = _rerun_effect_snapshot(
                    context=context_dict,
                    scenario=_safe_primary_scenario(current_validation_dict),
                    validation_report=current_validation_dict,
                    candidate_code=(
                        gen_err.raw_candidate
                        if isinstance(gen_err, GenerationFailureError)
                        else ""
                    ),
                )
                no_effect_rerun = _is_no_effect_rerun(
                    iteration_history,
                    generation_rerun_snapshot,
                )
                no_effect_owner = _no_effect_owner_metadata(
                    iteration_history, generation_rerun_snapshot
                )
                previous_iteration_record = iteration_history[-1] if iteration_history else {}
                previous_rerun_effect = previous_iteration_record.get("rerun_effect") or {}
                m4_relevant_state_unchanged = bool(
                    generation_rerun_snapshot.get("scenario_fingerprint")
                    and generation_rerun_snapshot.get("scenario_fingerprint")
                    == previous_rerun_effect.get("scenario_fingerprint")
                )
                unchanged_m4_ineligible_no_progress = bool(
                    failure_type_detail == "M5_INPUT_CONTRACT"
                    and not _has_m5_eligible_selected(current_validation_dict)
                    and no_effect_rerun
                    and no_effect_owner.get("owner_artifact_available") is True
                    and no_effect_owner.get("expected_change_owner") in {"M2", "M3", "M4"}
                    and m4_relevant_state_unchanged
                    and previous_iteration_record.get("failure_type_detail")
                    == "M5_INPUT_CONTRACT"
                    and previous_iteration_record.get("m4_eligibility_state")
                    == "INELIGIBLE"
                    and previous_iteration_record.get("m5_invoked") is False
                )
                no_progress_termination = bool(
                    prior_identical_input_contract
                    or unchanged_m4_ineligible_no_progress
                )
                can_continue_not_valid = bool(
                    iteration < max_feedback_iterations
                    and not no_progress_termination
                )
                generation_termination_reason = (
                    "m4_ineligible_no_effect"
                    if unchanged_m4_ineligible_no_progress
                    else "m4_ineligible_repeated_input_contract"
                    if prior_identical_input_contract
                    else None
                )
                if no_effect_rerun:
                    diagnosis += "\nNO_EFFECT_RERUN: context, scenario, and candidate fingerprints are unchanged."
                if unchanged_m4_ineligible_no_progress:
                    diagnosis += (
                        "\nNO_PROGRESS_TERMINATION: the M3/M4-owned rerun left the "
                        "M4-relevant scenario state unchanged and M5 remained ineligible; "
                        "no additional outer iteration was consumed."
                    )
                elif prior_identical_input_contract:
                    diagnosis += (
                        "\nNO_PROGRESS_TERMINATION: unchanged M4 eligibility remained false "
                        "after the bounded M3/M4 repair; M5 was not invoked again."
                    )
                recoverable_feedback_decision = _write_recoverable_iteration_result(
                    output_dir=output_dir,
                    instance_id=instance.instance_id,
                    iteration=iteration,
                    failure_type="NOT_VALID",
                    failure_type_detail=failure_type_detail,
                    diagnosis=diagnosis,
                    refined_scenario=_safe_primary_scenario(current_validation_dict),
                    should_continue=can_continue_not_valid,
                    repair_branch=_generation_failure_repair_branch(
                        failure_type_detail,
                        repeated=repeated_generation_failure,
                    ),
                    repeated_semantic_fingerprint=repeated_generation_failure,
                    semantic_fingerprint=generation_failure_fingerprint,
                    feature_execution_telemetry=generation_repair_telemetry,
                    feedback_client=_selected_m7_client(repro_test_generator),
                    diagnosis_revision=m7_diagnosis_revision,
                    max_feedback_iterations=max_feedback_iterations,
                    termination_reason=generation_termination_reason,
                )
                iteration_record = _recoverable_iteration_record(
                    iteration=iteration,
                    selected_records=_selected_scenario_records(current_validation_dict),
                    failure_type="NOT_VALID",
                    failure_type_detail=failure_type_detail,
                    feedback_branch=recoverable_feedback_decision.selected_feedback_branch,
                    rerun_targets=list(recoverable_feedback_decision.modules_requested_for_next),
                    history_window=history_window,
                    token_usage_status=token_usage_status,
                    m5_elapsed_sec=round(time.time() - m5_t0, 3) if "m5_t0" in locals() else None,
                    semantic_fingerprint=generation_failure_fingerprint,
                    repeated_semantic_fingerprint=repeated_generation_failure,
                    max_feedback_iterations=max_feedback_iterations,
                )
                requested_after_failure = (
                    list(recoverable_feedback_decision.modules_requested_for_next)
                    if can_continue_not_valid
                    else []
                )
                iteration_record.update({
                    "pass_number": iteration,
                    "modules_actually_executed_this_pass": modules_actually_executed_this_pass,
                    "modules_requested_for_next_pass": requested_after_failure,
                    "diagnosis": _v26_diagnosis_fields(recoverable_feedback_decision),
                    "route_destination": recoverable_feedback_decision.route_destination,
                    "route_provenance": recoverable_feedback_decision.feedback_provenance,
                    "candidate_identity": {
                        "test_id": getattr(invalid_generated_test, "test_id", ""),
                        "canonical_test_nodeid": getattr(
                            invalid_generated_test,
                            "canonical_test_nodeid",
                            "",
                        ),
                        "generated_patch_sha256": getattr(
                            invalid_generated_test,
                            "generated_patch_sha256",
                            "",
                        ),
                        "semantic_fingerprint": generation_failure_fingerprint,
                    },
                    "rerun_effect": {
                        **generation_rerun_snapshot,
                        **no_effect_owner,
                        "no_effect_rerun": no_effect_rerun,
                    },
                    "no_effect_rerun": no_effect_rerun,
                    "m5_invoked": failure_type_detail != "M5_INPUT_CONTRACT",
                    "m4_eligibility_state": (
                        "ELIGIBLE"
                        if _has_m5_eligible_selected(current_validation_dict)
                        else "INELIGIBLE"
                    ),
                    "eligibility_reason": (
                        "CURRENT_M4_HAS_ACCEPTED_NON_DIAGNOSTIC_SELECTION"
                        if _has_m5_eligible_selected(current_validation_dict)
                        else "CURRENT_M4_HAS_NO_ACCEPTED_NON_DIAGNOSTIC_SELECTION"
                    ),
                    "eligibility_owner": "M3/M4",
                    "changed_input_required": (
                        "scenario_or_m4_validation_result"
                        if failure_type_detail == "M5_INPUT_CONTRACT"
                        else None
                    ),
                    "m4_m5_eligibility_signature": m4_m5_eligibility_signature,
                    "no_progress_termination": no_progress_termination,
                    "no_progress_reason": (
                        "M4_INELIGIBLE_NO_EFFECT"
                        if unchanged_m4_ineligible_no_progress
                        else "M4_INELIGIBLE_REPEATED_INPUT_CONTRACT"
                        if prior_identical_input_contract
                        else None
                    ),
                    "negative_memory": {
                        "schema_version": "v31-negative-memory-v1",
                        "entry_count": len(active_negative_memory),
                        "memory_ids": [
                            entry["memory_id"] for entry in active_negative_memory
                        ],
                    },
                    "pass_elapsed_sec": round(time.monotonic() - pass_started_at, 3),
                    "total_instance_elapsed_sec": round(
                        time.monotonic() - pipeline_started_at,
                        3,
                    ),
                })
                write_json_atomic(
                    {
                        "schema_version": "v26-pass-provenance-v1",
                        "instance_id": instance.instance_id,
                        **{
                            key: iteration_record[key]
                            for key in (
                                "pass_number",
                                "modules_actually_executed_this_pass",
                                "modules_requested_for_next_pass",
                                "diagnosis",
                                "route_destination",
                                "route_provenance",
                                "candidate_identity",
                                "pass_elapsed_sec",
                                "total_instance_elapsed_sec",
                            )
                        },
                        "eligibility_state": iteration_record.get("m4_eligibility_state"),
                        "eligibility_reason": iteration_record.get("eligibility_reason"),
                        "owner": iteration_record.get("eligibility_owner"),
                        "changed_input_required": iteration_record.get("changed_input_required"),
                        "module_timings": {
                            "M1": iteration_m1_elapsed_sec,
                            "M2": iteration_m2_elapsed_sec,
                            "M3": iteration_m3_elapsed_sec,
                            "M4": iteration_m4_elapsed_sec,
                            "M5": iteration_record.get("m5_elapsed_sec"),
                            "M5-A": None,
                            "M6": None,
                            "M7": recoverable_feedback_decision.model_request_elapsed_sec,
                        },
                        "runtime_policy": {
                            "target_seconds": 120,
                            "telemetry_only": True,
                            "time_affects_control_flow": False,
                        },
                    },
                    iteration_dir / "pass_manifest.json",
                )
                _finalize_v27_pass_manifest(
                    enabled=v27_enabled or current_m7_policy_enabled,
                    iteration_dir=iteration_dir,
                    iteration_record=iteration_record,
                    previous_records=iteration_history,
                    context=context_dict,
                    scenario=_safe_primary_scenario(current_validation_dict),
                    candidate_code=(
                        gen_err.raw_candidate
                        if isinstance(gen_err, GenerationFailureError)
                        else ""
                    ),
                    schema_revision=feature_profile if current_m7_policy_enabled else "v27",
                )
                iteration_history.append(iteration_record)
                _write_iteration_history(
                    output_dir,
                    iteration_history,
                    max_feedback_iterations=max_feedback_iterations,
                )
                if can_continue_not_valid:
                    pending_pass_started_at = time.monotonic()
                    runtime_error_for_next = _runtime_hint_for_generation_failure(
                        diagnosis,
                        failure_type_detail,
                        repeated=repeated_generation_failure,
                    )
                    previous_runtime_error_fingerprint = generation_failure_fingerprint
                    recoverable_targets = list(
                        recoverable_feedback_decision.modules_requested_for_next
                    )
                    if "M2" in recoverable_targets:
                        m2_retry_started_at = time.monotonic()
                        context = context_extractor.extract(
                            instance=instance,
                            clue=clue_dict,
                            feature_flags=resolved_feature_flags,
                            restart_feedback={
                                "source": "m7_generation_failure_diagnosis",
                                "prohibited_targets": list(m2_prohibited_targets),
                                **_v26_diagnosis_fields(recoverable_feedback_decision),
                            },
                        )
                        context_dict = _serialize_m2_context(context, feature_profile)
                        pending_m2_elapsed_sec = round(
                            time.monotonic() - m2_retry_started_at,
                            3,
                        )
                        write_json_atomic(
                            context_dict,
                            iteration_dir / "context_after_not_valid_feedback.json",
                        )
                    if "M3" in recoverable_targets:
                        m3_retry_started_at = time.monotonic()
                        scenario, current_validation_dict = _generate_eligible_scenarios_with_retries(
                            scenario_generator=scenario_generator,
                            scenario_validator=scenario_validator,
                            instance=instance,
                            clue_dict=clue_dict,
                            context_dict=context_dict,
                            feature_flags=resolved_feature_flags,
                            iteration=iteration + 1,
                            scenario_output_path=scenario_output_path,
                            validation_output_path=validation_output_path,
                            initial_feedback={
                                "source": "not_valid_repeated_semantic_fingerprint",
                                "failure_type_detail": failure_type_detail,
                                "prohibit_candidate_fingerprint": generation_failure_fingerprint,
                                **_v26_diagnosis_fields(recoverable_feedback_decision),
                            },
                            attempt_artifact_dir=iteration_dir,
                        )
                        retry_stage_timings = current_validation_dict.get("v26_module_timings", {})
                        pending_m3_elapsed_sec = retry_stage_timings.get(
                            "m3_elapsed_sec",
                            round(time.monotonic() - m3_retry_started_at, 3),
                        )
                        pending_m4_elapsed_sec = retry_stage_timings.get("m4_elapsed_sec")
                    current_validation_dict = _inject_v26_diagnosis(
                        current_validation_dict,
                        recoverable_feedback_decision,
                    )
                    current_validation_dict = _inject_negative_memory(
                        current_validation_dict,
                        active_negative_memory,
                        instance_id=instance.instance_id,
                    )
                    write_json_atomic(current_validation_dict, validation_output_path)
                    modules_for_current_pass = recoverable_targets
                    continue
                _write_terminal_alignment_result(
                    output_dir=output_dir,
                    instance_id=instance.instance_id,
                    iteration=iteration,
                    failure_type="NOT_VALID",
                    failure_type_detail=failure_type_detail,
                    diagnosis=diagnosis,
                    refined_scenario=_safe_primary_scenario(current_validation_dict),
                    feature_execution_telemetry=generation_repair_telemetry,
                    termination_reason=generation_termination_reason,
                )
                _sync_latest_iteration_aliases(output_dir, iteration_dir, terminal=True)
                if (
                    best_candidate_record is not None
                    and best_alignment_result is not None
                    and best_iteration_dir is not None
                ):
                    _persist_best_candidate_selection(
                        output_dir=Path(output_dir),
                        best_record=best_candidate_record,
                        last_iteration=iteration,
                        last_status="NOT_VALID",
                    )
                    _sync_latest_iteration_aliases(
                        output_dir,
                        best_iteration_dir,
                        terminal=True,
                        terminal_alias_only=True,
                    )
                    preserved = _preserved_alignment_summary(
                        instance_id=instance.instance_id,
                        alignment_result=best_alignment_result,
                        iteration=iteration,
                        selected_iteration=int(best_candidate_record["iteration"]),
                        token_usage=total_token_usage,
                        token_usage_status=_cumulative_token_usage_status(
                            total_token_usage,
                            observed_token_usage_statuses,
                        ),
                    )
                    preserved["best_candidate_selection"] = dict(best_candidate_record)
                    if v31_enabled:
                        preserved["token_usage_scope"] = "M5_GENERATION_SUBTOTAL; M5-A/M3/M7 TOKENS UNAVAILABLE"
                    return preserved
                return {
                    "instance_id": instance.instance_id,
                    "failure_type": "NOT_VALID",
                    **_status_fields("NOT_VALID"),
                    "failure_type_detail": failure_type_detail,
                    "iterations": iteration,
                    "error": diagnosis,
                    "token_usage": total_token_usage,
                    "token_usage_status": token_usage_status,
                    "token_usage_scope": (
                        "M5_GENERATION_SUBTOTAL; M5-A/M3/M7 TOKENS UNAVAILABLE" if v31_enabled else "PIPELINE_REPORTED"
                    ),
                    "generation_attempt_count": (
                        gen_err.attempt_count if isinstance(gen_err, GenerationFailureError) else 0
                    ),
                    "feature_execution_telemetry": generation_repair_telemetry,
                }

        generated_test_output_path = str(iteration_dir / "generated_test.json")
        repro_test_generator.save(generated_test, generated_test_output_path)
        if generation_repair_telemetry:
            _write_feature_execution_telemetry(iteration_dir, generation_repair_telemetry)
            _write_feature_execution_telemetry(Path(output_dir), generation_repair_telemetry)
        _sync_latest_iteration_aliases(output_dir, iteration_dir)

        print(f"  generated test → {generated_test_output_path}")
        print(f"  scenario_id: {generated_test.scenario_id}")
        # 토큰 누적
        _accumulate_generation_token_usage(
            total_token_usage,
            generated_test.token_usage,
            already_accounted=generation_usage_already_accounted,
        )
        observed_token_usage_statuses.append(
            str(getattr(generated_test, "token_usage_status", "") or "unknown")
        )
        for queued_candidate in v37_generated_candidate_queue:
            queued_generated = queued_candidate["generated_test"]
            _accumulate_generation_token_usage(
                total_token_usage,
                queued_generated.token_usage,
                already_accounted=False,
            )
            observed_token_usage_statuses.append(
                str(
                    getattr(queued_generated, "token_usage_status", "")
                    or "unknown"
                )
            )

        evaluated_scenario = _scenario_with_generated_target(
            active_v37_scenario
            or select_primary_scenario(
                current_validation_dict, clue=clue_dict, context=context_dict
            ),
            generated_test,
        )

        pre_m6_fingerprint = _pre_m6_candidate_fingerprint(
            scenario=evaluated_scenario,
            generated_test=generated_test,
        )
        if pre_m6_fingerprint in active_prohibited_fingerprints:
            diagnosis = "Generated candidate reused an active prohibited semantic fingerprint before M6"
            can_continue_not_valid = iteration < max_feedback_iterations
            prohibited_feedback_decision = _write_recoverable_iteration_result(
                output_dir=output_dir,
                instance_id=instance.instance_id,
                iteration=iteration,
                failure_type="NOT_VALID",
                failure_type_detail="PROHIBITED_FINGERPRINT_REUSE",
                diagnosis=diagnosis,
                refined_scenario=_safe_primary_scenario(current_validation_dict),
                should_continue=can_continue_not_valid,
                repair_branch="M2+M3+M5",
                repeated_semantic_fingerprint=True,
                semantic_fingerprint=pre_m6_fingerprint,
                feedback_client=_selected_m7_client(repro_test_generator),
                diagnosis_revision=m7_diagnosis_revision,
                max_feedback_iterations=max_feedback_iterations,
            )
            iteration_record = _recoverable_iteration_record(
                iteration=iteration,
                selected_records=_selected_scenario_records(current_validation_dict),
                failure_type="NOT_VALID",
                failure_type_detail="PROHIBITED_FINGERPRINT_REUSE",
                feedback_branch=prohibited_feedback_decision.selected_feedback_branch,
                rerun_targets=list(prohibited_feedback_decision.modules_requested_for_next),
                history_window=history_window,
                token_usage_status=_token_usage_status(total_token_usage),
                m5_elapsed_sec=m5_elapsed_sec,
                semantic_fingerprint=pre_m6_fingerprint,
                repeated_semantic_fingerprint=True,
                max_feedback_iterations=max_feedback_iterations,
            )
            prohibited_requested = (
                list(prohibited_feedback_decision.modules_requested_for_next)
                if can_continue_not_valid
                else []
            )
            iteration_record.update({
                "pass_number": iteration,
                "modules_actually_executed_this_pass": modules_actually_executed_this_pass,
                "modules_requested_for_next_pass": prohibited_requested,
                "diagnosis": _v26_diagnosis_fields(prohibited_feedback_decision),
                "route_destination": prohibited_feedback_decision.route_destination,
                "route_provenance": prohibited_feedback_decision.feedback_provenance,
                "candidate_identity": {
                    "test_id": getattr(generated_test, "test_id", ""),
                    "canonical_test_nodeid": getattr(generated_test, "canonical_test_nodeid", ""),
                    "generated_patch_sha256": getattr(generated_test, "generated_patch_sha256", ""),
                    "semantic_fingerprint": pre_m6_fingerprint,
                },
                "pass_elapsed_sec": round(time.monotonic() - pass_started_at, 3),
                "total_instance_elapsed_sec": round(time.monotonic() - pipeline_started_at, 3),
            })
            write_json_atomic(
                {
                    "schema_version": "v26-pass-provenance-v1",
                    "instance_id": instance.instance_id,
                    **{
                        key: iteration_record[key]
                        for key in (
                            "pass_number",
                            "modules_actually_executed_this_pass",
                            "modules_requested_for_next_pass",
                            "diagnosis",
                            "route_destination",
                            "route_provenance",
                            "candidate_identity",
                            "pass_elapsed_sec",
                            "total_instance_elapsed_sec",
                        )
                    },
                    "runtime_policy": {
                        "target_seconds": 120,
                        "telemetry_only": True,
                        "time_affects_control_flow": False,
                    },
                },
                iteration_dir / "pass_manifest.json",
            )
            _finalize_v27_pass_manifest(
                enabled=v27_enabled or current_m7_policy_enabled,
                iteration_dir=iteration_dir,
                iteration_record=iteration_record,
                previous_records=iteration_history,
                context=context_dict,
                scenario=_safe_primary_scenario(current_validation_dict),
                validation_report=current_validation_dict,
                candidate_code=str(getattr(generated_test, "test_code", "") or ""),
                schema_revision=feature_profile if current_m7_policy_enabled else "v27",
            )
            iteration_history.append(iteration_record)
            _write_iteration_history(
                output_dir,
                iteration_history,
                max_feedback_iterations=max_feedback_iterations,
            )
            if can_continue_not_valid:
                pending_pass_started_at = time.monotonic()
                active_prohibited_fingerprints.append(pre_m6_fingerprint)
                prohibited_targets = list(prohibited_feedback_decision.modules_requested_for_next)
                if "M2" in prohibited_targets:
                    m2_retry_started_at = time.monotonic()
                    context = context_extractor.extract(
                        instance=instance,
                        clue=clue_dict,
                        feature_flags=resolved_feature_flags,
                        restart_feedback={
                            "source": "m7_prohibited_fingerprint_rejection",
                            "prohibited_source_files": list(m2_prohibited_source_files),
                            **_v26_diagnosis_fields(prohibited_feedback_decision),
                        },
                    )
                    context_dict = _serialize_m2_context(context, feature_profile)
                    pending_m2_elapsed_sec = round(time.monotonic() - m2_retry_started_at, 3)
                    write_json_atomic(
                        context_dict,
                        iteration_dir / "context_after_prohibited_fingerprint.json",
                    )
                if "M3" in prohibited_targets:
                    scenario, current_validation_dict = _generate_eligible_scenarios_with_retries(
                        scenario_generator=scenario_generator,
                        scenario_validator=scenario_validator,
                        instance=instance,
                        clue_dict=clue_dict,
                        context_dict=context_dict,
                        feature_flags=resolved_feature_flags,
                        iteration=iteration + 1,
                        scenario_output_path=scenario_output_path,
                        validation_output_path=validation_output_path,
                        initial_feedback={
                            "source": "m7_prohibited_fingerprint_rejection",
                            "prohibited_prior_fingerprints": list(active_prohibited_fingerprints),
                            "target_reuse_allowed": False,
                            **_v26_diagnosis_fields(prohibited_feedback_decision),
                        },
                        attempt_artifact_dir=iteration_dir,
                    )
                    stage_timings = current_validation_dict.get("v26_module_timings", {})
                    pending_m3_elapsed_sec = stage_timings.get("m3_elapsed_sec")
                    pending_m4_elapsed_sec = stage_timings.get("m4_elapsed_sec")
                current_validation_dict = _inject_v26_diagnosis(
                    current_validation_dict,
                    prohibited_feedback_decision,
                    generated_test=generated_test,
                )
                modules_for_current_pass = prohibited_targets
                continue
            alignment_result = SimpleNamespace(
                failure_type="NOT_VALID",
                score_breakdown={},
                diagnosis=diagnosis,
            )
            break

        # ── before-patch-only 실행 (Docker SDK 직접 실행, patch-free) ──
        if isolated_source_view_enabled:
            _verify_v27r1_source_view(context_dict, instance.base_commit)
        m6_t0 = time.time()
        supplemental_context = dict(context_dict)
        supplemental_context["feature_profile"] = feature_profile
        supplemental_context["methodology_revision"] = feature_profile
        align_result = alignment_runner.run(
            instance=pre_patch_view,
            generated_test_json_path=generated_test_output_path,
            run_id=f"align-{instance.instance_id}-it{iteration}-{int(time.time() * 1000)}",
            iteration=iteration,
            feature_flags=resolved_feature_flags,
            supplemental_context=supplemental_context,
            supplemental_clue=clue_dict,
        )
        m6_elapsed_sec = round(time.time() - m6_t0, 3)

        align_exec_path = str(iteration_dir / "alignment_execution.json")
        m6_artifacts = alignment_runner.save(
            align_result,
            align_exec_path,
            feature_flags=resolved_feature_flags,
        )
        raw_m6_artifacts = dict(m6_artifacts or {})
        m6_stability_telemetry: Dict[str, Any] = {}
        if not resolved_feature_flags.m6_execution_stability:
            m6_stability_telemetry["m6_execution_stability"] = (
                m6_execution_stability_exclusion_telemetry()
            )
        m6_repair_telemetry: Dict[str, Any] = {}
        container_execution_attempts = 1
        repaired_alignment = _attempt_m6_m5a_repair(
            instance=instance,
            pre_patch_view=pre_patch_view,
            output_dir=iteration_dir,
            generator=repro_test_generator,
            alignment_runner=alignment_runner,
            original_generated_test=generated_test,
            original_generated_test_path=generated_test_output_path,
            align_result=align_result,
            clue=clue_dict,
            context=context_dict,
            validation_report=current_validation_dict,
            feature_flags=resolved_feature_flags,
            feature_profile=feature_profile,
            iteration=iteration,
            prior_m5a_attempt_count=int(
                (generation_repair_telemetry.get("enable_m5a_llm_error_refinement") or {}).get(
                    "attempt_count",
                    0,
                )
            ),
        )
        if repaired_alignment is not None:
            generated_test, align_result, m6_artifacts, m6_repair_telemetry = repaired_alignment
            raw_m6_artifacts = dict(m6_artifacts or {})
            container_execution_attempts = 2
            generated_test_output_path = str(iteration_dir / "generated_test.json")
            align_exec_path = str(iteration_dir / "alignment_execution.json")
        elif (iteration_dir / "feature_execution_telemetry.json").exists():
            try:
                import json as _json
                m6_repair_telemetry = _json.loads(
                    (iteration_dir / "feature_execution_telemetry.json").read_text(encoding="utf-8")
                )
            except Exception:
                m6_repair_telemetry = {}
        if v30_or_v31:
            m6_artifacts = _compact_m6_iteration_artifacts(
                output_dir=Path(output_dir),
                iteration_dir=iteration_dir,
                instance_id=instance.instance_id,
                iteration=iteration,
                candidate_sha256=str(
                    getattr(align_result, "generated_patch_sha256", "") or ""
                ) or None,
                raw_artifacts=raw_m6_artifacts,
                align_result=align_result,
                feature_profile=feature_profile,
            )
        m6_repair_feature = m6_repair_telemetry.get(
            "enable_m5a_llm_error_refinement",
            {},
        )
        m6_revalidation_errors = list(
            m6_repair_feature.get("post_repair_validation_errors") or []
        )
        if m6_revalidation_errors and repaired_alignment is None:
            align_result.error_messages = list(align_result.error_messages or []) + [
                "M5-A deterministic revalidation failed: "
                + "; ".join(str(item) for item in m6_revalidation_errors[:8])
            ]
        if m6_stability_telemetry:
            m6_repair_telemetry = {
                **m6_stability_telemetry,
                **m6_repair_telemetry,
            }
            _write_feature_execution_telemetry(iteration_dir, m6_repair_telemetry)
            _write_feature_execution_telemetry(Path(output_dir), m6_repair_telemetry)
        _sync_latest_iteration_aliases(output_dir, iteration_dir)

        print(f"  returncode: {align_result.returncode}")
        print(f"  has_failure: {align_result.has_failure}")
        print(f"  test_results: {align_result.test_results}")

        # ── Docker 빌드 실패 등 복구 불가 에러 시 즉시 중단 ──
        if align_result.error_messages and any(
            "build failed" in m or "build error" in m
            for m in align_result.error_messages
        ):
            diagnosis = (
                "Unrecoverable alignment execution error: "
                + "; ".join(align_result.error_messages[:3])
            )
            print(f"  ✗ M6 build error recorded as M7 ERROR: {align_result.error_messages}")
            should_continue_error = iteration < max_feedback_iterations
            error_fingerprint = _runtime_error_fingerprint(align_result.error_messages) or normalized_error_fingerprint(diagnosis) or ""
            error_record = _build_m7_decision_record(
                instance_id=instance.instance_id,
                iteration=iteration,
                decision_status="ERROR",
                source_stage="M6",
                validation_status="VALID",
                execution_status="ERROR",
                failure_category="ENVIRONMENT_FAILURE",
                evidence=_normalize_m7_evidence(
                    {
                    "diagnosis": diagnosis,
                    "failure_type_detail": "BUILD_FAILED",
                    "error_messages": list(align_result.error_messages or []),
                    "coverage_evidence": dict(align_result.coverage_data or {}),
                    "failure_evidence_fingerprint": error_fingerprint,
                    },
                    remaining_outer_iterations=max(0, max_feedback_iterations - iteration),
                ),
                feedback_branch="M6",
                next_start_stage="M6",
                rerun_targets=["M6"],
                should_continue=should_continue_error,
                termination_reason="continue" if should_continue_error else "iteration_budget_exhausted",
                prohibited_fingerprints=[error_fingerprint] if error_fingerprint else [],
                feedback_decision=_build_m7_feedback_decision(
                    client=_selected_m7_client(repro_test_generator),
                    decision_status="ERROR",
                    iteration=iteration,
                    source_stage="M6",
                    failure_category="ENVIRONMENT_FAILURE",
                    evidence={
                        "diagnosis": diagnosis,
                        "failure_type_detail": "BUILD_FAILED",
                        "error_messages": list(align_result.error_messages or []),
                        "coverage_evidence": dict(align_result.coverage_data or {}),
                        "failure_evidence_fingerprint": error_fingerprint,
                    },
                    feedback_branch="M6",
                    next_start_stage="M6",
                    rerun_targets=["M6"],
                    prohibited_fingerprints=[error_fingerprint] if error_fingerprint else [],
                    diagnosis_revision=m7_diagnosis_revision,
                    max_feedback_iterations=max_feedback_iterations,
                ),
                max_feedback_iterations=max_feedback_iterations,
            )
            error_payload = {
                "instance_id": instance.instance_id,
                "failure_type": "ERROR",
                **_status_fields("ERROR"),
                "failure_type_detail": "BUILD_FAILED",
                "bug_fail_score": 0.0,
                "coverage_score": 0.0,
                "issue_alignment_score": 0.0,
                "iterations": iteration,
                "error": diagnosis,
                "token_usage": total_token_usage,
                "token_usage_status": _cumulative_token_usage_status(
                    total_token_usage,
                    observed_token_usage_statuses,
                ),
                "token_usage_scope": (
                    "M5_GENERATION_SUBTOTAL; M5-A/M3/M7 TOKENS UNAVAILABLE" if v31_enabled else "PIPELINE_REPORTED"
                ),
                "source_stage": "M6",
                "m7_decision_record_path": str(iteration_dir / "m7_decision_record.json"),
                "feedback_decision": error_record.feedback_decision.to_dict(),
                "should_continue": should_continue_error,
            }
            write_json(error_payload, iteration_dir / "alignment_result.json")
            _write_m7_decision_artifacts(iteration_dir, error_record)
            _sync_latest_iteration_aliases(
                output_dir, iteration_dir, terminal=not should_continue_error
            )
            error_requested = (
                list(error_record.feedback_decision.modules_requested_for_next)
                if should_continue_error
                else []
            )
            error_iteration_record = {
                "iteration": iteration,
                "pass_number": iteration,
                "modules_actually_executed_this_pass": modules_actually_executed_this_pass,
                "modules_requested_for_next_pass": error_requested,
                "selected_scenarios": _selected_scenario_records(current_validation_dict),
                "generated_scenario_id": generated_test.scenario_id,
                "m7_decision_status": "ERROR",
                "m7_alignment_status": None,
                "failure_type": "ERROR",
                "failure_type_detail": "BUILD_FAILED",
                "feedback_branch": error_record.feedback_decision.selected_feedback_branch,
                "rerun_targets": error_requested,
                "diagnosis": _v26_diagnosis_fields(error_record.feedback_decision),
                "route_destination": error_record.feedback_decision.route_destination,
                "route_provenance": error_record.feedback_decision.feedback_provenance,
                "candidate_identity": {
                    "test_id": getattr(generated_test, "test_id", ""),
                    "canonical_test_nodeid": getattr(generated_test, "canonical_test_nodeid", ""),
                    "generated_patch_sha256": getattr(generated_test, "generated_patch_sha256", ""),
                    "semantic_fingerprint": error_fingerprint,
                },
                "loop_terminated": not should_continue_error,
                "history_window": history_window,
                "semantic_progress_fingerprint": error_fingerprint,
                "repeated_semantic_fingerprint": False,
                "outer_iteration_policy": "unified_m7_decision_loop",
                "pass_elapsed_sec": round(time.monotonic() - pass_started_at, 3),
                "total_instance_elapsed_sec": round(time.monotonic() - pipeline_started_at, 3),
            }
            iteration_history.append(error_iteration_record)
            write_json_atomic(
                {
                    "schema_version": "v26-pass-provenance-v1",
                    "instance_id": instance.instance_id,
                    **{
                        key: error_iteration_record[key]
                        for key in (
                            "pass_number",
                            "modules_actually_executed_this_pass",
                            "modules_requested_for_next_pass",
                            "diagnosis",
                            "route_destination",
                            "route_provenance",
                            "candidate_identity",
                            "pass_elapsed_sec",
                            "total_instance_elapsed_sec",
                        )
                    },
                    "runtime_policy": {
                        "target_seconds": 120,
                        "telemetry_only": True,
                        "time_affects_control_flow": False,
                    },
                },
                iteration_dir / "pass_manifest.json",
            )
            _finalize_v27_pass_manifest(
                enabled=v27_enabled or current_m7_policy_enabled,
                iteration_dir=iteration_dir,
                iteration_record=error_iteration_record,
                previous_records=iteration_history[:-1],
                context=context_dict,
                scenario=_safe_primary_scenario(current_validation_dict),
                validation_report=current_validation_dict,
                candidate_code=str(getattr(generated_test, "test_code", "") or ""),
                schema_revision=feature_profile if current_m7_policy_enabled else "v27",
            )
            _write_iteration_history(
                output_dir,
                iteration_history,
                max_feedback_iterations=max_feedback_iterations,
            )
            if should_continue_error:
                pending_pass_started_at = time.monotonic()
                runtime_error_for_next = diagnosis
                previous_runtime_error_fingerprint = error_fingerprint
                if "M2" in error_requested:
                    m2_retry_started_at = time.monotonic()
                    context = context_extractor.extract(
                        instance=instance,
                        clue=clue_dict,
                        feature_flags=resolved_feature_flags,
                        restart_feedback={
                            "source": "m7_environment_failure_diagnosis",
                            **_v26_diagnosis_fields(error_record.feedback_decision),
                        },
                    )
                    context_dict = _serialize_m2_context(context, feature_profile)
                    pending_m2_elapsed_sec = round(time.monotonic() - m2_retry_started_at, 3)
                    write_json_atomic(
                        context_dict,
                        iteration_dir / "context_after_environment_feedback.json",
                    )
                if "M3" in error_requested:
                    scenario, current_validation_dict = _generate_eligible_scenarios_with_retries(
                        scenario_generator=scenario_generator,
                        scenario_validator=scenario_validator,
                        instance=instance,
                        clue_dict=clue_dict,
                        context_dict=context_dict,
                        feature_flags=resolved_feature_flags,
                        iteration=iteration + 1,
                        scenario_output_path=scenario_output_path,
                        validation_output_path=validation_output_path,
                        initial_feedback={
                            "source": "m7_environment_failure_diagnosis",
                            **_v26_diagnosis_fields(error_record.feedback_decision),
                        },
                        attempt_artifact_dir=iteration_dir,
                    )
                    stage_timings = current_validation_dict.get("v26_module_timings", {})
                    pending_m3_elapsed_sec = stage_timings.get("m3_elapsed_sec")
                    pending_m4_elapsed_sec = stage_timings.get("m4_elapsed_sec")
                current_validation_dict = _inject_v26_diagnosis(
                    current_validation_dict,
                    error_record.feedback_decision,
                    generated_test=generated_test,
                    align_result=align_result,
                )
                modules_for_current_pass = error_requested
                if error_requested == ["M6", "M7"]:
                    retained_m6_candidate = generated_test
                continue
            if (
                best_candidate_record is not None
                and best_alignment_result is not None
                and best_iteration_dir is not None
            ):
                _persist_best_candidate_selection(
                    output_dir=Path(output_dir),
                    best_record=best_candidate_record,
                    last_iteration=iteration,
                    last_status="ERROR",
                )
                _sync_latest_iteration_aliases(
                    output_dir,
                    best_iteration_dir,
                    terminal=True,
                    terminal_alias_only=True,
                )
                preserved = _preserved_alignment_summary(
                    instance_id=instance.instance_id,
                    alignment_result=best_alignment_result,
                    iteration=iteration,
                    selected_iteration=int(best_candidate_record["iteration"]),
                    token_usage=total_token_usage,
                    token_usage_status=_cumulative_token_usage_status(
                        total_token_usage,
                        observed_token_usage_statuses,
                    ),
                )
                preserved["best_candidate_selection"] = dict(best_candidate_record)
                if v31_enabled:
                    preserved["token_usage_scope"] = "M5_GENERATION_SUBTOTAL; M5-A/M3/M7 TOKENS UNAVAILABLE"
                return preserved
            return error_payload

        # ── 정합성 평가 (규칙기반, patch-free) ──
        execution_payload = align_result.to_dict()
        if m6_artifacts is not None:
            sbfl_payload = (
                raw_m6_artifacts.get("sbfl_result")
                if v30_or_v31 or strict_v36_or_v37
                else m6_artifacts.get("sbfl_result")
                if isinstance(m6_artifacts, Mapping)
                else None
            )
            if sbfl_payload is not None:
                execution_payload["canonical_m6_sbfl_result"] = sbfl_payload
                if v37_enabled:
                    payload = (
                        sbfl_payload.get("payload")
                        if isinstance(sbfl_payload, Mapping)
                        and isinstance(sbfl_payload.get("payload"), Mapping)
                        else sbfl_payload
                    )
                    metadata = (
                        payload.get("metadata")
                        if isinstance(payload, Mapping)
                        and isinstance(payload.get("metadata"), Mapping)
                        else {}
                    )
                    spectrum = (
                        list(payload.get("suspiciousness") or [])
                        if isinstance(payload, Mapping)
                        else []
                    )
                    if metadata.get("sbfl_active") is True and spectrum:
                        context_dict["prior_m6_sbfl_spectrum"] = spectrum
                        context_dict["prior_m6_covered_sut_lines"] = list(
                            getattr(align_result, "covered_sut_lines", []) or []
                        )
                        context_dict["prior_m6_sbfl_source_iteration"] = iteration
                    else:
                        context_dict.pop("prior_m6_sbfl_spectrum", None)
                        context_dict.pop("prior_m6_covered_sut_lines", None)
                        context_dict.pop("prior_m6_sbfl_source_iteration", None)
        m7_t0 = time.time()
        context_for_m7 = dict(context_dict)
        context_for_m7["max_feedback_iterations"] = max_feedback_iterations
        if v29_enabled or v30_or_v31 or strict_v36_or_v37:
            context_for_m7["methodology_revision"] = (
                feature_profile if v30_or_v31 or strict_v36_or_v37 else "v29"
            )
        alignment_result = scorer.evaluate(
            execution_result=execution_payload,
            clue=clue_dict,
            scenario=evaluated_scenario,
            generated_test=generated_test.to_dict(),
            iteration=iteration,
            validation_report=current_validation_dict,
            context=context_for_m7,
            feature_flags=scorer_feature_flags,
        )
        if v27_enabled:
            _apply_v27_admission_guard(alignment_result)
        rerun_effect_snapshot = _rerun_effect_snapshot(
            context=context_dict,
            scenario=evaluated_scenario,
            validation_report=current_validation_dict,
            candidate_code=str(getattr(generated_test, "test_code", "") or ""),
            candidate_identity={
                "target_test_file": getattr(generated_test, "target_test_file", ""),
                "canonical_test_nodeid": getattr(
                    generated_test, "canonical_test_nodeid", ""
                ),
                "execution_command": getattr(
                    align_result, "execution_command", ""
                ),
                "m6_execution_fingerprint": _m6_execution_progress_fingerprint(
                    align_result
                ),
                "imports": getattr(generated_test, "imports", []) or [],
                "oracle_identity": (
                    getattr(generated_test, "relational_oracle", None)
                    or getattr(generated_test, "candor_oracle", None)
                ),
            },
        )
        no_effect_rerun = _is_no_effect_rerun(
            iteration_history,
            rerun_effect_snapshot,
        )
        no_effect_owner = _no_effect_owner_metadata(
            iteration_history, rerun_effect_snapshot
        )
        current_rerun_effect = {
            **no_effect_owner,
            "no_effect_rerun": no_effect_rerun,
            "owner_identity_version": rerun_effect_snapshot.get("owner_identity_version"),
        }
        precomputed_m7_feedback_decision: M7FeedbackDecision | None = None
        if (
            current_m7_policy_enabled
            and _requires_v29_conservative_gate_judgment(alignment_result)
        ):
            provisional_evidence = _m7_evaluation_evidence(
                alignment_result=alignment_result,
                align_result=align_result,
                semantic_fingerprint="",
                repeated_semantic_fingerprint=False,
                selected_records=[],
                score_breakdown=alignment_result.score_breakdown,
                remaining_outer_iterations=max(0, max_feedback_iterations - iteration),
                clue=clue_dict,
                context=context_dict,
                scenario=evaluated_scenario,
                generated_test=generated_test,
                m6_artifacts=m6_artifacts,
            )
            provisional_evidence = _attach_previous_m7_feedback(
                provisional_evidence, iteration_history, current_rerun_effect
            )
            provisional_evidence["m7_decision_context"] = (
                "V37_CONSERVATIVE_GATE"
                if m7_diagnosis_revision == "v37"
                else "V36_CONSERVATIVE_GATE"
                if v36_enabled
                else "V29_CONSERVATIVE_GATE"
            )
            precomputed_m7_feedback_decision = _build_m7_feedback_decision(
                client=_selected_m7_client(repro_test_generator),
                decision_status="WEAK_ALIGNMENT",
                iteration=iteration,
                source_stage="M7",
                failure_category="NONE",
                evidence=provisional_evidence,
                feedback_branch="M7_CONSERVATIVE_DIAGNOSIS",
                next_start_stage="M5",
                rerun_targets=["M5", "M5-A", "M6", "M7"],
                prohibited_fingerprints=[],
                diagnosis_enabled=resolved_feature_flags.m7_llm_scenario_refinement,
                diagnosis_revision=m7_diagnosis_revision,
                max_feedback_iterations=max_feedback_iterations,
            )
            _apply_v29_conservative_gate_decision(
                alignment_result,
                precomputed_m7_feedback_decision,
                iteration=iteration,
                max_feedback_iterations=max_feedback_iterations,
            )
        m7_elapsed_sec = round(time.time() - m7_t0, 3)

        if v37_enabled:
            candidate_root = iteration_dir / "candidates"
            primary_rank = int(
                (getattr(generated_test, "m5_invocation_provenance", {}) or {}).get(
                    "scenario_rank", 1
                )
            )
            primary_dir = candidate_root / (
                f"rank_{primary_rank}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', generated_test.scenario_id)}"
            )
            primary_dir.mkdir(parents=True, exist_ok=True)
            repro_test_generator.save(
                generated_test, str(primary_dir / "generated_test.json")
            )
            if Path(align_exec_path).exists():
                shutil.copy2(align_exec_path, primary_dir / "alignment_execution.json")
            primary_payload = alignment_result.to_dict()
            primary_payload.update(
                {
                    "instance_id": instance.instance_id,
                    "scenario_id": generated_test.scenario_id,
                    "scenario_rank": primary_rank,
                    "pass_provenance": {
                        "candidate_identity": {
                            "test_id": getattr(generated_test, "test_id", ""),
                            "canonical_test_nodeid": generated_test.canonical_test_nodeid,
                            "generated_patch_sha256": generated_test.generated_patch_sha256,
                        }
                    },
                }
            )
            write_json_atomic(primary_payload, primary_dir / "alignment_result.json")
            v37_candidate_outcomes.append(
                {
                    "scenario_id": generated_test.scenario_id,
                    "scenario_rank": primary_rank,
                    "candidate_dir": str(primary_dir),
                    "generated_test": generated_test,
                    "execution": align_result,
                    "m6_artifacts": m6_artifacts,
                    "alignment": alignment_result,
                    "m5_output_identity": {
                        "canonical_test_nodeid": generated_test.canonical_test_nodeid,
                        "generated_patch_sha256": generated_test.generated_patch_sha256,
                    },
                    "m6_execution_identity": {
                        "canonical_test_id": align_result.canonical_test_id,
                        "canonical_test_nodeid": align_result.canonical_test_nodeid,
                        "generated_patch_sha256": align_result.generated_patch_sha256,
                    },
                    "m7_result": alignment_result.failure_type,
                    "admitted_to_final_set": alignment_result.failure_type == "ALIGNED",
                }
            )
            for queued in v37_generated_candidate_queue:
                rank = int(queued["scenario_rank"])
                scenario_id = str(queued["scenario_id"])
                candidate_dir = candidate_root / (
                    f"rank_{rank}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', scenario_id)}"
                )
                v37_candidate_outcomes.append(
                    _evaluate_v37_additional_candidate(
                        candidate=queued,
                        candidate_dir=candidate_dir,
                        generator=repro_test_generator,
                        alignment_runner=alignment_runner,
                        scorer=scorer,
                        instance=instance,
                        pre_patch_view=pre_patch_view,
                        clue=clue_dict,
                        context=context_dict,
                        validation_report=current_validation_dict,
                        feature_flags=resolved_feature_flags,
                        scorer_feature_flags=scorer_feature_flags,
                        iteration=iteration,
                        max_feedback_iterations=max_feedback_iterations,
                        diagnosis_revision=m7_diagnosis_revision,
                    )
                )
            aligned_outcomes = [
                outcome
                for outcome in v37_candidate_outcomes
                if outcome["admitted_to_final_set"]
            ]
            if alignment_result.failure_type != "ALIGNED" and aligned_outcomes:
                promoted = min(
                    aligned_outcomes, key=lambda outcome: outcome["scenario_rank"]
                )
                generated_test = promoted["generated_test"]
                align_result = promoted["execution"]
                m6_artifacts = promoted["m6_artifacts"]
                alignment_result = promoted["alignment"]
                evaluated_scenario = _scenario_with_generated_target(
                    next(
                        dict(item["scenario"])
                        for item in v37_generated_candidate_queue
                        if int(item["scenario_rank"]) == int(promoted["scenario_rank"])
                    ),
                    generated_test,
                )
                repro_test_generator.save(generated_test, generated_test_output_path)
                promoted_execution = Path(promoted["candidate_dir"]) / "alignment_execution.json"
                if promoted_execution.exists():
                    shutil.copy2(promoted_execution, align_exec_path)
            manifest_records = [
                {
                    **record,
                    "candidate_dir": None,
                    "m6_execution_identity": None,
                    "m7_result": None,
                    "admitted_to_final_set": False,
                }
                for record in v37_generation_records
                if record.get("m5_status") != "GENERATED"
            ]
            manifest_records.extend(
                {
                    key: value
                    for key, value in outcome.items()
                    if key
                    not in {
                        "generated_test",
                        "execution",
                        "m6_artifacts",
                        "alignment",
                    }
                }
                for outcome in sorted(
                    v37_candidate_outcomes,
                    key=lambda value: int(value["scenario_rank"]),
                )
            )
            candidate_manifest = {
                "schema_version": "v37-candidate-set-v1",
                "instance_id": instance.instance_id,
                "iteration": iteration,
                "feedback_owner_scenario_rank": 1,
                "candidate_records": manifest_records,
                "aligned_candidate_count": sum(
                    bool(record.get("admitted_to_final_set"))
                    for record in manifest_records
                ),
            }
            write_json_atomic(
                candidate_manifest, iteration_dir / "v37_candidate_set.json"
            )
            write_json_atomic(
                candidate_manifest, Path(output_dir) / "v37_candidate_set.json"
            )

        current_candidate_record = _candidate_evaluation_record(
            iteration=iteration,
            generated_test=generated_test,
            align_result=align_result,
            alignment_result=alignment_result,
        )
        last_candidate_record = copy.deepcopy(current_candidate_record)
        if _candidate_dominates(current_candidate_record, best_candidate_record):
            best_candidate_record = copy.deepcopy(current_candidate_record)
            best_generated_test = generated_test
            best_alignment_result = alignment_result
            best_iteration_dir = iteration_dir

        alignment_output_path = str(iteration_dir / "alignment_result.json")
        feedback_branch, rerun_targets = _feedback_route(alignment_result)
        feedback_branch, rerun_targets = _ensure_v30_continuation_route(
            failure_type=alignment_result.failure_type,
            should_continue=alignment_result.should_continue,
            iteration=iteration,
            max_feedback_iterations=max_feedback_iterations,
            feedback_branch=feedback_branch,
            rerun_targets=rerun_targets,
        ) if v30_or_v31 else (feedback_branch, rerun_targets)
        runtime_error_fingerprint = _runtime_error_fingerprint(align_result.error_messages)
        repeated_runtime_error_early_stop = (
            bool(runtime_error_fingerprint)
            and runtime_error_fingerprint == previous_runtime_error_fingerprint
            and bool(runtime_error_for_next)
        )
        no_progress_termination = bool(alignment_result.should_continue and not rerun_targets)
        semantic_fingerprint = _candidate_semantic_fingerprint(
            scenario=evaluated_scenario,
            generated_test=generated_test,
            align_result=align_result,
            alignment_result=alignment_result,
        )
        repeated_semantic_fingerprint = _is_repeated_semantic_fingerprint(
            iteration_history,
            semantic_fingerprint,
            status=alignment_result.failure_type,
        )
        if repeated_semantic_fingerprint and alignment_result.should_continue:
            rerun_targets = _escalated_rerun_targets(rerun_targets)
            feedback_branch = "+".join(rerun_targets) if rerun_targets else feedback_branch
        # A bounded retry that reproduces the same context, scenario, and
        # candidate is not a meaningful feedback pass.  Escalate ownership so
        # the next iteration must regenerate upstream evidence instead of
        # silently replaying M5/M6.  This is diagnostic control flow only; it
        # does not alter any M7 score, threshold, or admission rule.
        if no_effect_rerun and alignment_result.should_continue:
            rerun_targets = _escalated_rerun_targets(rerun_targets)
            feedback_branch = "+".join(rerun_targets) if rerun_targets else feedback_branch
        route_evidence = _m7_evaluation_evidence(
            alignment_result=alignment_result,
            align_result=align_result,
            semantic_fingerprint=semantic_fingerprint,
            repeated_semantic_fingerprint=repeated_semantic_fingerprint,
            selected_records=[],
            score_breakdown=alignment_result.score_breakdown,
            remaining_outer_iterations=max(0, max_feedback_iterations - iteration),
            clue=clue_dict,
            context=context_dict,
            scenario=evaluated_scenario,
            generated_test=generated_test,
            m6_artifacts=m6_artifacts,
        )
        route_evidence = _attach_previous_m7_feedback(
            route_evidence, iteration_history, current_rerun_effect
        )
        if alignment_result.failure_type == "NOT_FAILED" and alignment_result.should_continue:
            feedback_branch, rerun_targets = _not_failed_route_from_evidence(
                route_evidence,
                repeated=repeated_semantic_fingerprint,
            )
        if alignment_result.failure_type == "NO_COVERAGE" and repeated_semantic_fingerprint:
            no_progress_termination = True
            rerun_targets = ["M6", "M7"]
            feedback_branch = "+".join(rerun_targets)
        selected_records = _selected_scenario_records(current_validation_dict)
        combined_feature_telemetry: Dict[str, Any] = {}
        combined_feature_telemetry.update(generation_repair_telemetry or {})
        combined_feature_telemetry.update(m6_repair_telemetry or {})
        iteration_record = {
            "pass_number": iteration,
            "modules_actually_executed_this_pass": modules_actually_executed_this_pass,
            "iteration": iteration,
            "selected_scenarios": selected_records,
            "generated_scenario_id": generated_test.scenario_id,
            "m7_alignment_status": alignment_result.m7_alignment_status or alignment_result.failure_type,
            "m7_decision_status": alignment_result.failure_type,
            "failure_type": alignment_result.failure_type,
            "feedback_branch": feedback_branch,
            "rerun_targets": rerun_targets,
            "loop_terminated": not alignment_result.should_continue,
            "history_window": history_window,
            "generation_attempt_count": getattr(generated_test, "generation_attempt_count", None),
            "initial_generation_attempts": getattr(generated_test, "generation_attempt_count", None),
            "validation_retry_count": getattr(generated_test, "repair_retry_count", None),
            "deterministic_repair_attempts": len(getattr(generated_test, "postprocessing_actions", []) or []),
            "llm_repair_attempts": (
                combined_feature_telemetry.get("enable_m5a_llm_error_refinement", {}).get("attempt_count", 0)
            ),
            "container_execution_attempts": container_execution_attempts,
            "alignment_iterations": iteration,
            "repeated_validation_early_stop": False,
            "repeated_validation_fingerprint": None,
            "runtime_error_fingerprint": runtime_error_fingerprint,
            "repeated_runtime_error_early_stop": repeated_runtime_error_early_stop,
            "semantic_progress_fingerprint": semantic_fingerprint,
            "repeated_semantic_fingerprint": repeated_semantic_fingerprint,
            "rerun_effect": {
                **rerun_effect_snapshot,
                **no_effect_owner,
                "no_effect_rerun": no_effect_rerun,
            },
            "no_effect_rerun": no_effect_rerun,
            "semantic_escalation_required": repeated_semantic_fingerprint and alignment_result.should_continue,
            "no_progress_termination": no_progress_termination,
            "llm_call_count": getattr(generated_test, "generation_attempt_count", None),
            "m1_elapsed_sec": iteration_m1_elapsed_sec,
            "m2_elapsed_sec": iteration_m2_elapsed_sec,
            "m3_elapsed_sec": iteration_m3_elapsed_sec,
            "m4_elapsed_sec": iteration_m4_elapsed_sec,
            "m5_elapsed_sec": m5_elapsed_sec,
            "m5a_elapsed_sec": (
                (getattr(generated_test, "prompt_profile", {}) or {})
                .get("v26_module_timings", {})
                .get("m5a_postprocess_elapsed_sec")
            ),
            "m6_execution_coverage_elapsed_sec": m6_elapsed_sec,
            "m6_timing_breakdown": _m6_timing_breakdown(align_result),
            "m7_elapsed_sec": m7_elapsed_sec,
            "final_harness_elapsed_sec": None,
            "m8_elapsed_sec": None,
            "candidate_evaluation": current_candidate_record,
            "v37_candidate_evaluations": [
                {
                    "scenario_id": outcome.get("scenario_id"),
                    "scenario_rank": outcome.get("scenario_rank"),
                    "m5_output_identity": outcome.get("m5_output_identity"),
                    "m6_execution_identity": outcome.get("m6_execution_identity"),
                    "m7_result": outcome.get("m7_result"),
                    "admitted_to_final_set": outcome.get("admitted_to_final_set"),
                    "candidate_dir": outcome.get("candidate_dir"),
                }
                for outcome in v37_candidate_outcomes
            ],
            "best_so_far_iteration": (
                best_candidate_record.get("iteration") if best_candidate_record else None
            ),
        }
        alignment_payload = alignment_result.to_dict()
        alignment_payload["instance_id"] = instance.instance_id
        if alignment_result.failure_type in {"NOT_VALID", "ERROR"}:
            alignment_payload["m7_alignment_status"] = None
        if pending_m7_feedback_application is not None:
            pending_m7_feedback_application["after"] = _m7_feedback_effect_snapshot(
                generated_test=generated_test,
                align_result=align_result,
                alignment_result=alignment_result,
            )
            pending_m7_feedback_application["effectiveness"] = _judge_m7_feedback_effectiveness(
                pending_m7_feedback_application.get("before", {}),
                pending_m7_feedback_application["after"],
                pending_m7_feedback_application.get("verdict", ""),
            )
            if iteration_history:
                iteration_history[-1]["m7_feedback_effectiveness"] = dict(
                    pending_m7_feedback_application["effectiveness"]
                )
            write_json(
                pending_m7_feedback_application,
                Path(str(pending_m7_feedback_application["artifact_path"])),
            )
            pending_m7_feedback_application = None

        llm_application = _m7_llm_application_record(
            iteration_dir=iteration_dir,
            iteration=iteration,
            alignment_result=alignment_result,
            generated_test=generated_test,
            align_result=align_result,
        )
        if llm_application is not None:
            pending_m7_feedback_application = llm_application
            write_json(llm_application, Path(str(llm_application["artifact_path"])))
            alignment_payload["m7_feedback_application_artifact"] = llm_application["artifact_path"]
        if combined_feature_telemetry:
            alignment_payload["feature_execution_telemetry"] = combined_feature_telemetry
        alignment_payload["m7_decision_status"] = alignment_result.failure_type
        alignment_payload["source_stage"] = _source_stage_for_decision(alignment_result.failure_type)
        alignment_payload["selected_scenarios"] = selected_records
        alignment_payload["feedback_branch"] = feedback_branch
        alignment_payload["rerun_targets"] = rerun_targets
        alignment_payload["history_window"] = history_window
        alignment_payload["runtime_control"] = {
            key: iteration_record[key]
            for key in (
                "generation_attempt_count",
                "initial_generation_attempts",
                "validation_retry_count",
                "deterministic_repair_attempts",
                "llm_repair_attempts",
                "container_execution_attempts",
                "alignment_iterations",
                "repeated_validation_early_stop",
                "repeated_validation_fingerprint",
                "runtime_error_fingerprint",
                "repeated_runtime_error_early_stop",
                "semantic_progress_fingerprint",
                "repeated_semantic_fingerprint",
                "semantic_escalation_required",
                "no_progress_termination",
                "llm_call_count",
                "m2_elapsed_sec",
                "m3_elapsed_sec",
                "m5_elapsed_sec",
                "m6_execution_coverage_elapsed_sec",
                "m6_timing_breakdown",
                "m7_elapsed_sec",
                "final_harness_elapsed_sec",
                "m8_elapsed_sec",
            )
        }
        if repeated_runtime_error_early_stop and iteration >= max_feedback_iterations:
            alignment_payload["should_continue"] = False
            alignment_payload["termination_reason"] = "repeated_runtime_error"
            alignment_result.should_continue = False
        if no_progress_termination and iteration >= max_feedback_iterations:
            alignment_payload["should_continue"] = False
            alignment_payload["termination_reason"] = "no_progress"
            alignment_result.should_continue = False
        if iteration >= max_feedback_iterations and alignment_result.failure_type != "ALIGNED":
            alignment_payload["should_continue"] = False
            alignment_payload["termination_reason"] = "iteration_budget_exhausted"
            alignment_result.should_continue = False
        m7_evidence = _m7_evaluation_evidence(
            alignment_result=alignment_result,
            align_result=align_result,
            semantic_fingerprint=semantic_fingerprint,
            repeated_semantic_fingerprint=repeated_semantic_fingerprint,
            selected_records=selected_records,
            score_breakdown=alignment_result.score_breakdown,
            remaining_outer_iterations=max(0, max_feedback_iterations - iteration),
            clue=clue_dict,
            context=context_dict,
            scenario=evaluated_scenario,
            generated_test=generated_test,
            m6_artifacts=m6_artifacts,
        )
        m7_evidence = _attach_previous_m7_feedback(
            m7_evidence, iteration_history, current_rerun_effect
        )
        m7_feedback_decision = precomputed_m7_feedback_decision or _build_m7_feedback_decision(
            client=_selected_m7_client(repro_test_generator),
            decision_status=alignment_result.failure_type,
            iteration=iteration,
            source_stage=_source_stage_for_decision(alignment_result.failure_type),
            failure_category=_failure_category_for_decision(
                alignment_result.failure_type,
                align_result,
            ),
            evidence=m7_evidence,
            feedback_branch=feedback_branch,
            next_start_stage=(
                "M8"
                if alignment_result.failure_type == "ALIGNED"
                else _first_restart_stage(rerun_targets)
            ),
            rerun_targets=rerun_targets,
            prohibited_fingerprints=[semantic_fingerprint]
            if repeated_semantic_fingerprint and semantic_fingerprint
            else [],
            diagnosis_enabled=resolved_feature_flags.m7_llm_scenario_refinement,
            diagnosis_revision=m7_diagnosis_revision,
            max_feedback_iterations=max_feedback_iterations,
        )
        if alignment_result.failure_type != "ALIGNED" and alignment_result.should_continue:
            feedback_branch = m7_feedback_decision.selected_feedback_branch
            requested_rerun_targets = list(m7_feedback_decision.modules_requested_for_next)
            rerun_targets = (
                requested_rerun_targets
                if m7_diagnosis_revision in {"v36", "v37"}
                else _owner_scoped_rerun_targets(
                    alignment_result.failure_type,
                    requested_rerun_targets,
                    alignment_result.score_breakdown,
                    failure_category=_failure_category_for_decision(
                        alignment_result.failure_type,
                        align_result,
                    ),
                    execution_error_stage=str(
                        getattr(align_result, "error_stage", "") or ""
                    ),
                )
            )
            if rerun_targets != requested_rerun_targets:
                _synchronize_feedback_decision_route(
                    m7_feedback_decision,
                    rerun_targets,
                )
            iteration_record["feedback_branch"] = feedback_branch
            iteration_record["rerun_targets"] = rerun_targets
            feedback_branch = m7_feedback_decision.selected_feedback_branch
            iteration_record["feedback_branch"] = feedback_branch
            alignment_payload["feedback_branch"] = feedback_branch
            alignment_payload["rerun_targets"] = rerun_targets
        m7_record = _build_m7_decision_record(
            instance_id=instance.instance_id,
            iteration=iteration,
            decision_status=alignment_result.failure_type,
            source_stage=_source_stage_for_decision(alignment_result.failure_type),
            validation_status="VALID",
            execution_status=_execution_status_from_alignment_execution(align_result),
            failure_category=_failure_category_for_decision(
                alignment_result.failure_type,
                align_result,
            ),
            evidence=m7_evidence,
            feedback_branch=feedback_branch,
            next_start_stage=(
                "M8"
                if alignment_result.failure_type == "ALIGNED"
                else _first_restart_stage(rerun_targets)
            ),
            rerun_targets=rerun_targets,
            should_continue=alignment_result.should_continue,
            termination_reason=(
                "aligned"
                if alignment_result.failure_type == "ALIGNED"
                else "iteration_budget_exhausted"
                if iteration >= max_feedback_iterations
                else alignment_payload.get("termination_reason", "continue")
            ),
            prohibited_fingerprints=[semantic_fingerprint]
            if repeated_semantic_fingerprint and semantic_fingerprint
            else [],
            feedback_decision=m7_feedback_decision,
            max_feedback_iterations=max_feedback_iterations,
        )
        alignment_payload["m7_decision_record_path"] = str(iteration_dir / "m7_decision_record.json")
        alignment_payload["feedback_decision"] = m7_record.feedback_decision.to_dict()
        modules_requested_for_next_pass = (
            list(m7_record.feedback_decision.modules_requested_for_next)
            if alignment_result.failure_type != "ALIGNED" and alignment_result.should_continue
            else []
        )
        if alignment_result.failure_type != "ALIGNED":
            m7_record.feedback_decision.modules_requested_for_next = list(
                modules_requested_for_next_pass
            )
            alignment_payload["feedback_decision"] = m7_record.feedback_decision.to_dict()
            alignment_payload["loop_terminated"] = m7_record.loop_terminated
            alignment_payload["feedback_branch"] = (
                m7_record.feedback_decision.selected_feedback_branch
            )
            alignment_payload["rerun_targets"] = list(modules_requested_for_next_pass)
        diagnosis_record = _v26_diagnosis_fields(m7_record.feedback_decision)
        candidate_identity = {
            "test_id": getattr(generated_test, "test_id", ""),
            "canonical_test_nodeid": getattr(generated_test, "canonical_test_nodeid", ""),
            "generated_patch_sha256": getattr(generated_test, "generated_patch_sha256", "")
            or getattr(generated_test, "patch_sha256", ""),
            "semantic_fingerprint": semantic_fingerprint,
        }
        diagnosis_elapsed = m7_record.feedback_decision.model_request_elapsed_sec or 0.0
        iteration_record["m7_elapsed_sec"] = round(m7_elapsed_sec + diagnosis_elapsed, 3)
        iteration_record["loop_terminated"] = m7_record.loop_terminated
        if alignment_result.failure_type != "ALIGNED":
            iteration_record["feedback_branch"] = (
                m7_record.feedback_decision.selected_feedback_branch
            )
            iteration_record["rerun_targets"] = modules_requested_for_next_pass
        iteration_record["modules_requested_for_next_pass"] = modules_requested_for_next_pass
        iteration_record["diagnosis"] = diagnosis_record
        iteration_record["route_destination"] = m7_record.feedback_decision.route_destination
        iteration_record["route_provenance"] = m7_record.feedback_decision.feedback_provenance
        iteration_record["candidate_identity"] = candidate_identity
        iteration_record["pass_elapsed_sec"] = round(time.monotonic() - pass_started_at, 3)
        iteration_record["total_instance_elapsed_sec"] = round(
            time.monotonic() - pipeline_started_at,
            3,
        )
        alignment_payload["runtime_control"]["m7_elapsed_sec"] = iteration_record["m7_elapsed_sec"]
        alignment_payload["runtime_control"]["pass_elapsed_sec"] = iteration_record["pass_elapsed_sec"]
        alignment_payload["runtime_control"]["total_instance_elapsed_sec"] = iteration_record[
            "total_instance_elapsed_sec"
        ]
        alignment_payload["pass_provenance"] = {
            key: iteration_record[key]
            for key in (
                "pass_number",
                "modules_actually_executed_this_pass",
                "modules_requested_for_next_pass",
                "diagnosis",
                "route_destination",
                "route_provenance",
                "candidate_identity",
                "pass_elapsed_sec",
                "total_instance_elapsed_sec",
            )
        }
        for fingerprint in m7_record.feedback_decision.prohibited_prior_fingerprints:
            if fingerprint and fingerprint not in active_prohibited_fingerprints:
                active_prohibited_fingerprints.append(fingerprint)
        _write_m7_decision_artifacts(iteration_dir, m7_record)
        write_json(alignment_payload, alignment_output_path)
        write_json_atomic(
            {
                "schema_version": "v26-pass-provenance-v1",
                "instance_id": instance.instance_id,
                **alignment_payload["pass_provenance"],
                "module_timings": {
                    "M1": iteration_record.get("m1_elapsed_sec"),
                    "M2": iteration_record.get("m2_elapsed_sec"),
                    "M3": iteration_record.get("m3_elapsed_sec"),
                    "M4": iteration_record.get("m4_elapsed_sec"),
                    "M5": iteration_record.get("m5_elapsed_sec"),
                    "M5-A": iteration_record.get("m5a_elapsed_sec"),
                    "M6": iteration_record.get("m6_execution_coverage_elapsed_sec"),
                    "M7": iteration_record.get("m7_elapsed_sec"),
                },
                "artifact_references": {
                    "clue": clue_output_path,
                    "context": context_output_path,
                    "scenario": scenario_output_path,
                    "scenario_validation": validation_output_path,
                    "generated_test": generated_test_output_path,
                    "alignment_execution": align_exec_path,
                    "alignment_result": alignment_output_path,
                    "m7_decision_record": str(iteration_dir / "m7_decision_record.json"),
                },
                "runtime_policy": {
                    "target_seconds": 120,
                    "telemetry_only": True,
                    "time_affects_control_flow": False,
                    "max_feedback_iterations": max_feedback_iterations,
                },
            },
            iteration_dir / "pass_manifest.json",
        )
        _finalize_v27_pass_manifest(
            enabled=v27_enabled or current_m7_policy_enabled,
            iteration_dir=iteration_dir,
            iteration_record=iteration_record,
            previous_records=iteration_history,
            context=context_dict,
            scenario=_safe_primary_scenario(current_validation_dict),
            validation_report=current_validation_dict,
            candidate_code=str(getattr(generated_test, "test_code", "") or ""),
            schema_revision=feature_profile if current_m7_policy_enabled else "v27",
        )
        if combined_feature_telemetry:
            _write_feature_execution_telemetry(iteration_dir, combined_feature_telemetry)
            _write_feature_execution_telemetry(Path(output_dir), combined_feature_telemetry)
        iteration_history.append(iteration_record)
        _write_iteration_history(
            output_dir,
            iteration_history,
            max_feedback_iterations=max_feedback_iterations,
        )
        _sync_latest_iteration_aliases(output_dir, iteration_dir)

        print(f"  failure_type: {alignment_result.failure_type}")
        breakdown = alignment_result.score_breakdown
        print(f"  bug_fail_score: {breakdown.get('bug_fail_score')}")
        print(f"  raw_target_coverage: {breakdown.get('raw_target_coverage')}")
        print(f"  s_c_prime: {breakdown.get('s_c_prime')}")
        print(f"  s_c_prime_status: {breakdown.get('s_c_prime_status')}")
        print(f"  issue_alignment_score: {breakdown.get('issue_alignment_score')}")
        print(f"  diagnosis: {alignment_result.diagnosis}")

        if not alignment_result.should_continue:
            if alignment_result.failure_type == "ALIGNED":
                print(f"\n  ✓ ALIGNED at iteration {iteration}")
            else:
                print(f"\n  ✗ terminal {alignment_result.failure_type} at iteration {iteration}")
            _sync_latest_iteration_aliases(output_dir, iteration_dir, terminal=True)
            break

        if iteration == max_feedback_iterations:
            print(f"\n  ✗ 최대 {max_feedback_iterations}회 도달, 루프 종료")
            _sync_latest_iteration_aliases(output_dir, iteration_dir, terminal=True)
            break

        print(f"  → 시나리오 보강 후 재시도 "
              f"(failure_type={alignment_result.failure_type}, rerun_targets={rerun_targets})")

        # 다음 iteration을 위해 런타임 에러 수집
        runtime_error_for_next = None
        if align_result.error_messages:
            relevant = [
                m for m in align_result.error_messages
                if any(kw in m for kw in [
                    "TypeError", "AttributeError", "NameError",
                    "ImportError", "ModuleNotFoundError", "SyntaxError",
                    "RuntimeError", "missing 1 required", "self",
                    "OperationalError", "IntegrityError",
                    "not collected", "not found",
                ])
            ]
            if relevant:
                runtime_error_for_next = "; ".join(relevant[:2])
        previous_runtime_error_fingerprint = runtime_error_fingerprint

        pending_pass_started_at = time.monotonic()

        if "M2" in rerun_targets:
            m2_t0 = time.time()
            selected_scenario = select_primary_scenario(
                current_validation_dict,
                clue=clue_dict,
                context=context_dict,
            )
            target_verification = (
                alignment_result.score_breakdown.get("target_verification_evidence") or {}
                if isinstance(alignment_result.score_breakdown, Mapping)
                else {}
            )
            runtime_target = (
                target_verification.get("runtime_target_coverage") or {}
                if isinstance(target_verification, Mapping)
                else {}
            )
            target_disproven = bool(
                target_verification.get("localization_disproven") is True
                or (
                    target_verification.get("verification_path") == "STATIC_REJECTION"
                    and target_verification.get("reason")
                    == "selected_source_missing_canonical_target"
                )
            )
            selected_target = _m2_target_for_exclusion(
                selected_scenario,
                generated_test,
            )
            if target_disproven and selected_target and selected_target not in m2_prohibited_targets:
                m2_prohibited_targets.append(selected_target)
                target_memory = {
                    "schema_version": "v31-negative-memory-v1",
                    "memory_id": sha256_text(
                        f"{instance.instance_id}\0{iteration}\0DISPROVEN_TARGET\0"
                        f"{selected_target['source_file']}\0{selected_target['target_function']}"
                    ),
                    "instance_id": instance.instance_id,
                    "source_iteration": iteration,
                    "owner_module": "M2",
                    "candidate_sha256": getattr(generated_test, "generated_patch_sha256", None),
                    "category": "DISPROVEN_TARGET",
                    "rejected_choice": selected_target["target_function"],
                    "reason": str(target_verification.get("reason") or "target not executed"),
                    "prohibition": (
                        "Do not reuse this disproven target without new runtime-grounded evidence: "
                        + selected_target["target_function"]
                    ),
                    "repository_alternatives": [],
                    "provenance": "prepatch_target_verification",
                }
                active_negative_memory = _merge_negative_memory(
                    active_negative_memory,
                    [target_memory],
                    instance_id=instance.instance_id,
                )
            m2_restart_feedback = {
                "source": "m7_restart_decision",
                "outer_iteration": iteration,
                "m7_decision_status": alignment_result.failure_type,
                "target_reuse_allowed": not target_disproven,
                "require_alternative_target": target_disproven,
                "prohibited_source_files": list(m2_prohibited_source_files),
                "prohibited_targets": list(m2_prohibited_targets),
                "no_effect_rerun": no_effect_rerun,
                "no_effect_owner": no_effect_owner.get("expected_change_owner"),
                "no_effect_owner_fingerprint": rerun_effect_snapshot.get(
                    str(no_effect_owner.get("owner_fingerprint_key") or "")
                ),
                "concrete_repair_instruction": m7_feedback_decision.concrete_repair_instruction,
                "m6_runtime_coverage_evidence": {
                    key: m7_feedback_decision.evidence_used.get(key)
                    for key in (
                        "suspected_file_covered",
                        "suspected_function_covered",
                        "suspected_lines_covered",
                        "issue_branch_reached",
                        "fault_hypothesis_supported",
                        "coverage_evidence",
                        "m6_execution_evidence",
                    )
                    if key in m7_feedback_decision.evidence_used
                },
                **_v26_diagnosis_fields(m7_feedback_decision),
            }
            next_context_reuse_key = _m2_context_reuse_key(
                instance_id=instance.instance_id,
                clue_dict=clue_dict,
                feature_flags=resolved_feature_flags,
                history_window=history_window,
                restart_feedback=m2_restart_feedback,
            )
            if next_context_reuse_key == current_context_reuse_key:
                pending_m2_elapsed_sec = 0.0
            else:
                context = context_extractor.extract(
                    instance=instance,
                    clue=clue_dict,
                    feature_flags=resolved_feature_flags,
                    restart_feedback=m2_restart_feedback,
                )
                context_dict = _serialize_m2_context(context, feature_profile)
                current_context_reuse_key = next_context_reuse_key
                pending_m2_elapsed_sec = round(time.time() - m2_t0, 3)
            context_iteration_path = str(iteration_dir / "context_after_feedback.json")
            write_json_atomic(context_dict, context_iteration_path)
            write_json_atomic(context_dict, context_output_path)

        if "M3" in rerun_targets:
            m3_feedback = _m3_feedback_payload(
                alignment_result=alignment_result,
                generated_test=generated_test,
                align_result=align_result,
                rerun_targets=rerun_targets,
                feedback_decision=m7_feedback_decision,
            )
            m3_t0 = time.time()
            scenario, current_validation_dict = _generate_eligible_scenarios_with_retries(
                scenario_generator=scenario_generator,
                scenario_validator=scenario_validator,
                instance=instance,
                clue_dict=clue_dict,
                context_dict=context_dict,
                feature_flags=resolved_feature_flags,
                iteration=iteration + 1,
                scenario_output_path=scenario_output_path,
                validation_output_path=validation_output_path,
                initial_feedback=m3_feedback,
                attempt_artifact_dir=iteration_dir,
            )
            rerun_stage_timings = current_validation_dict.get("v26_module_timings", {})
            pending_m3_elapsed_sec = rerun_stage_timings.get(
                "m3_elapsed_sec",
                round(time.time() - m3_t0, 3),
            )
            pending_m4_elapsed_sec = rerun_stage_timings.get("m4_elapsed_sec")
        current_validation_dict = _ensure_issue_grounded_fallback_if_needed(
            validation_dict=current_validation_dict,
            scenario_generator=scenario_generator,
            scenario_validator=scenario_validator,
            instance=instance,
            clue_dict=clue_dict,
            context_dict=context_dict,
            feature_flags=resolved_feature_flags,
            iteration=iteration + 1,
            scenario_output_path=scenario_output_path,
            validation_output_path=validation_output_path,
            reason=f"iteration_{iteration}_refinement_guard",
        )
        current_validation_dict = _limit_selected_scenarios(current_validation_dict)
        current_validation_dict = _inject_v26_diagnosis(
            current_validation_dict,
            m7_feedback_decision,
            generated_test=generated_test,
            align_result=align_result,
        )
        current_validation_dict = _inject_negative_memory(
            current_validation_dict,
            active_negative_memory,
            instance_id=instance.instance_id,
        )
        write_json_atomic(
            {
                "schema_version": "v31-negative-memory-v1",
                "instance_id": instance.instance_id,
                "outer_iteration": iteration,
                "entries": active_negative_memory,
                "entry_count": len(active_negative_memory),
                "max_entries": MAX_V31_NEGATIVE_MEMORY,
            },
            iteration_dir / "negative_memory.json",
        )
        write_json(current_validation_dict, validation_output_path)
        if rerun_targets == ["M6", "M7"]:
            retained_m6_candidate = generated_test
        modules_for_current_pass = list(rerun_targets)

    # At the bounded terminal point, expose the strongest already-evaluated
    # candidate rather than blindly treating iteration 5 as authoritative.
    # This reuses its original M7 decision; it never recalculates admission.
    if best_candidate_record is not None:
        last_status = str(getattr(alignment_result, "failure_type", "") or "UNKNOWN")
        _persist_best_candidate_selection(
            output_dir=Path(output_dir),
            best_record=best_candidate_record,
            last_iteration=iteration,
            last_status=last_status,
        )
    if (
        best_candidate_record is not None
        and best_alignment_result is not None
        and best_iteration_dir is not None
        and int(best_candidate_record.get("iteration") or 0) != iteration
        and (
            last_candidate_record is None
            or _candidate_dominance_key(best_candidate_record)
            > _candidate_dominance_key(last_candidate_record)
        )
        and getattr(alignment_result, "failure_type", "") != "ALIGNED"
    ):
        _sync_latest_iteration_aliases(
            output_dir,
            best_iteration_dir,
            terminal=True,
            terminal_alias_only=True,
        )
        alignment_result = best_alignment_result

    # ── 모델 로직 결과 반환 (alignment 루프까지) ──
    failure_type = alignment_result.failure_type if alignment_result else "ERROR"
    score_breakdown = alignment_result.score_breakdown if alignment_result else {}

    print(f"\n{'='*60}")
    print("  Model Pipeline Result")
    print(f"{'='*60}")
    print(f"  failure_type: {failure_type}")
    print(f"  bug_fail_score: {score_breakdown.get('bug_fail_score')}")
    print(f"  raw_target_coverage: {score_breakdown.get('raw_target_coverage')}")
    print(f"  s_c_prime: {score_breakdown.get('s_c_prime')}")
    print(f"  s_c_prime_status: {score_breakdown.get('s_c_prime_status')}")
    print(f"  issue_alignment_score: {score_breakdown.get('issue_alignment_score')}")
    print(f"  iterations: {iteration}")

    error_msg = None
    if alignment_result is None:
        error_msg = "alignment_result is None: loop did not complete"
    elif alignment_result.failure_type == "ERROR":
        error_msg = alignment_result.diagnosis

    return {
        "instance_id": instance.instance_id,
        "failure_type": failure_type,
        **_status_fields(failure_type),
        "failure_type_detail": getattr(alignment_result, "failure_type_detail", "") if alignment_result else "",
        "bug_fail_score": score_breakdown.get("bug_fail_score"),
        "coverage_score": score_breakdown.get("coverage_score"),
        "issue_alignment_score": score_breakdown.get("issue_alignment_score"),
        "iterations": iteration,
        "best_candidate_iteration": (
            best_candidate_record.get("iteration") if best_candidate_record else None
        ),
        "best_candidate_selection": dict(best_candidate_record or {}),
        "error": error_msg,
        "token_usage": total_token_usage,
        "token_usage_status": _cumulative_token_usage_status(
            total_token_usage,
            observed_token_usage_statuses,
        ),
        "token_usage_scope": (
            "M5_GENERATION_SUBTOTAL; M5-A/M3/M7 TOKENS UNAVAILABLE" if v31_enabled else "PIPELINE_REPORTED"
        ),
        "iteration_history": iteration_history,
    }


def _validate_history_window(history_window: int | None) -> None:
    if history_window is None:
        return
    if not isinstance(history_window, int) or history_window <= 0:
        raise ValueError("history_window must be an explicit positive integer when provided")


def _ensure_v30_continuation_route(
    *,
    failure_type: str,
    should_continue: bool,
    iteration: int,
    max_feedback_iterations: int,
    feedback_branch: str,
    rerun_targets: list[str],
) -> tuple[str, list[str]]:
    """Enforce the v30 outer-loop continuation invariant at one boundary."""
    if (
        should_continue
        and iteration < max_feedback_iterations
        and failure_type != "ALIGNED"
        and not rerun_targets
    ):
        # ERROR and NOT_RUN remain non-admissible but are recoverable while the
        # budget remains.  M5 is the smallest safe regeneration boundary.
        return feedback_branch or "M5", ["M5"]
    return feedback_branch, list(rerun_targets)


def _validate_max_feedback_iterations(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_feedback_iterations must be an integer greater than or equal to 1")


def _verify_v27r1_source_view(context: Mapping[str, Any], base_commit: str) -> None:
    """Fail closed when the isolated pre-patch source view moved before M6."""
    repo_path = Path(str(context.get("repo_path") or ""))
    source_view = (
        (context.get("metadata") or {}).get("source_view")
        if isinstance(context.get("metadata"), Mapping)
        else {}
    )
    if not repo_path.is_dir() or (source_view or {}).get("isolation") != "per_instance_detached_worktree":
        raise RuntimeError("v27r1 requires a per-instance detached source worktree")

    def git_output(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"v27r1 source verification failed: git {' '.join(args)}: {result.stderr.strip()}"
            )
        return result.stdout.strip()

    expected = git_output("rev-parse", f"{base_commit}^{{commit}}")
    actual = git_output("rev-parse", "HEAD")
    if actual != expected:
        raise RuntimeError(
            f"v27r1 source view HEAD mismatch: expected={expected} actual={actual} path={repo_path}"
        )


def _apply_v27_admission_guard(alignment_result: Any) -> None:
    """Require mutually consistent pre-patch semantic evidence for admission."""
    if getattr(alignment_result, "failure_type", None) != "ALIGNED":
        return
    breakdown = getattr(alignment_result, "score_breakdown", {}) or {}
    oracle = breakdown.get("oracle_consistency") or {}
    reasons: list[str] = []
    if breakdown.get("target_verified") is not True:
        reasons.append("v27_target_execution_not_verified")
    if str(breakdown.get("oracle_type") or "") == "last_resort_structural":
        reasons.append("v27_last_resort_structural_oracle")
    oracle_status = str(oracle.get("status") or "")
    strong_issue = breakdown.get("strong_issue_evidence") is True
    issue_expected_oracle = str(breakdown.get("oracle_source") or "") == "issue_expected"
    issue_reported_invariant = (
        breakdown.get("issue_reported_semantic_invariant") is True
        and str(breakdown.get("oracle_source") or "") == "issue_reported_semantic_invariant"
    )
    symptom_score = float((breakdown.get("bug_fail_features") or {}).get("f_symptom") or 0.0)
    legacy_oracle_is_independently_grounded = (
        strong_issue
        and (
            issue_reported_invariant
            or (issue_expected_oracle and symptom_score == 1.0)
        )
    )
    if oracle_status in {"", "legacy_provenance_unavailable", "not_evaluated"} and not (
        oracle_status == "legacy_provenance_unavailable"
        and legacy_oracle_is_independently_grounded
    ):
        reasons.append("v27_oracle_semantic_provenance_unavailable")
    trigger_present = oracle.get("trigger_present") is True
    if not strong_issue and not trigger_present:
        reasons.append("v27_issue_trigger_not_strongly_grounded")
    if not reasons:
        breakdown["v27_admission_guard"] = {
            "status": "PASS",
            "pre_patch_only": True,
            "reasons": [],
        }
        return
    gate_reasons = list(breakdown.get("conservative_gate_reasons") or [])
    gate_reasons.extend(reason for reason in reasons if reason not in gate_reasons)
    breakdown["conservative_gate_reasons"] = gate_reasons
    breakdown["v27_admission_guard"] = {
        "status": "REJECT",
        "pre_patch_only": True,
        "reasons": reasons,
    }
    breakdown["failure_type_detail"] = "V27_SEMANTIC_ADMISSION_GUARD"
    alignment_result.failure_type = "WEAK_ALIGNMENT"
    alignment_result.failure_type_detail = "V27_SEMANTIC_ADMISSION_GUARD"
    alignment_result.should_continue = True
    alignment_result.m7_alignment_status = "WEAK_ALIGNMENT"
    alignment_result.admitted_to_final_set = False
    alignment_result.diagnostic_only = True
    alignment_result.diagnosis = (
        "V27 semantic admission guard rejected ALIGNED: " + ", ".join(reasons)
    )


def _m2_context_reuse_key(
    *,
    instance_id: str,
    clue_dict: Mapping[str, Any],
    feature_flags: V22FeatureFlags,
    history_window: int | None,
    restart_feedback: Mapping[str, Any] | None,
) -> str:
    """Return a deterministic key for M2 inputs currently consumed by source."""
    payload = {
        "instance_id": instance_id,
        "clue": clue_dict,
        "feature_flags": feature_flags.to_dict(),
        "history_window": history_window,
        "m7_feedback_consumed_by_m2": bool(restart_feedback),
        "restart_feedback": dict(restart_feedback or {}),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _generate_eligible_scenarios_with_retries(
    *,
    scenario_generator: ScenarioGenerator,
    scenario_validator: ScenarioValidator,
    instance: BenchmarkInstance,
    clue_dict: Dict[str, Any],
    context_dict: Dict[str, Any],
    feature_flags: V22FeatureFlags,
    iteration: int,
    scenario_output_path: str,
    validation_output_path: str,
    initial_feedback: Mapping[str, Any] | None = None,
    attempt_artifact_dir: Path | None = None,
) -> tuple[list[Any], Dict[str, Any]]:
    initial_m7_feedback = dict(initial_feedback or {})
    feedback = dict(initial_m7_feedback)
    attempts: list[dict[str, Any]] = []
    last_scenarios: list[Any] = []
    last_validation: Dict[str, Any] = {}
    m3_elapsed = 0.0
    m4_elapsed = 0.0
    is_v36 = str(
        context_dict.get("feature_profile")
        or context_dict.get("methodology_revision")
        or ""
    ) in {"v36", "v37"}
    attempt_limit = 1 if is_v36 else MAX_M3_VALIDATION_ATTEMPTS
    for attempt in range(1, attempt_limit + 1):
        m3_started_at = time.monotonic()
        scenarios = scenario_generator.extract(
            instance=instance,
            clue=clue_dict,
            context=context_dict,
            feature_flags=feature_flags,
            feedback=feedback or None,
        )
        if context_dict.get("localization_hypotheses") and context_dict.get("feature_profile") in {"v30", "v31"}:
            scenarios = bind_scenarios_to_localization_hypotheses(
                scenarios,
                context_dict.get("localization_hypotheses") or [],
            )
        m3_elapsed += time.monotonic() - m3_started_at
        _write_m3_nonadaptive_diagnostics(
            scenario_generator,
            scenario_output_path=scenario_output_path,
            attempt_artifact_dir=attempt_artifact_dir,
            attempt=attempt,
        )
        _annotate_m3_scenarios(
            scenarios,
            instance_id=instance.instance_id,
            iteration=iteration,
            attempt=attempt,
            model_call_count=1,
            fallback_used=_m3_last_fallback_used(scenario_generator),
            fallback_reason=_m3_last_fallback_reason(scenario_generator),
        )
        m4_started_at = time.monotonic()
        validation_dict = _validate_and_rank_scenarios(
            scenario_validator=scenario_validator,
            scenarios=scenarios,
            instance=instance,
            clue_dict=clue_dict,
            context_dict=context_dict,
            feature_flags=feature_flags,
            iteration=iteration,
        )
        m4_elapsed += time.monotonic() - m4_started_at
        validation_dict["v26_module_timings"] = {
            "m3_elapsed_sec": round(m3_elapsed, 3),
            "m4_elapsed_sec": round(m4_elapsed, 3),
            "time_affects_control_flow": False,
        }
        attempts.append(_m3_validation_attempt_record(attempt, validation_dict))
        last_scenarios = scenarios
        last_validation = validation_dict
        if _has_m5_eligible_selected(validation_dict):
            scenario_generator.save(scenarios, scenario_output_path)
            if attempt_artifact_dir is not None:
                scenario_generator.save(scenarios, str(attempt_artifact_dir / "scenario_after_feedback.json"))
            validation_dict["m3_validation_attempts"] = attempts
            write_json(validation_dict, validation_output_path)
            return scenarios, validation_dict
        rejection_feedback = _structured_m3_rejection_feedback(
            attempt=attempt,
            validation_dict=validation_dict,
            clue_dict=clue_dict,
        )
        feedback = dict(initial_m7_feedback)
        feedback["m4_validation_rejection"] = rejection_feedback

    if is_v36:
        scenario_generator.save(last_scenarios, scenario_output_path)
        last_validation["m3_validation_attempts"] = attempts
        last_validation["v36_no_synthetic_fallback"] = True
        write_json(last_validation, validation_output_path)
        return last_scenarios, last_validation

    fallback_started_at = time.monotonic()
    if initial_m7_feedback:
        fallback_scenarios = scenario_generator._build_fallback_scenarios(
            clue_dict, context_dict, feedback=initial_m7_feedback
        )
    else:
        fallback_scenarios = scenario_generator._build_fallback_scenarios(
            clue_dict, context_dict
        )
    m3_elapsed += time.monotonic() - fallback_started_at
    _annotate_m3_scenarios(
        fallback_scenarios,
        instance_id=instance.instance_id,
        iteration=iteration,
        attempt=MAX_M3_VALIDATION_ATTEMPTS + 1,
        model_call_count=0,
        fallback_used=True,
        fallback_reason="bounded_m3_validation_attempts_exhausted",
    )
    m4_started_at = time.monotonic()
    fallback_validation = _validate_and_rank_scenarios(
        scenario_validator=scenario_validator,
        scenarios=fallback_scenarios,
        instance=instance,
        clue_dict=clue_dict,
        context_dict=context_dict,
        feature_flags=feature_flags,
        iteration=iteration,
        allow_unresolved_fallback=True,
    )
    m4_elapsed += time.monotonic() - m4_started_at
    fallback_validation["v26_module_timings"] = {
        "m3_elapsed_sec": round(m3_elapsed, 3),
        "m4_elapsed_sec": round(m4_elapsed, 3),
        "time_affects_control_flow": False,
    }
    fallback_validation["m3_validation_attempts"] = attempts
    fallback_validation["fallback_used"] = True
    fallback_validation["fallback_reason"] = "bounded_m3_validation_attempts_exhausted"
    scenario_generator.save(fallback_scenarios or last_scenarios, scenario_output_path)
    if attempt_artifact_dir is not None:
        scenario_generator.save(fallback_scenarios or last_scenarios, str(attempt_artifact_dir / "scenario_after_feedback.json"))
    write_json(fallback_validation or last_validation, validation_output_path)
    return fallback_scenarios or last_scenarios, fallback_validation or last_validation


def _validate_and_rank_scenarios(
    *,
    scenario_validator: ScenarioValidator,
    scenarios: list[Any],
    instance: BenchmarkInstance,
    clue_dict: Dict[str, Any],
    context_dict: Dict[str, Any],
    feature_flags: V22FeatureFlags,
    iteration: int,
    allow_unresolved_fallback: bool = False,
) -> Dict[str, Any]:
    feature_profile = str(
        context_dict.get("feature_profile")
        or context_dict.get("methodology_revision")
        or ""
    )
    if feature_profile == "v37":
        validation_dict = _v37_structural_validation_dict(scenarios)
    else:
        validation_report = scenario_validator.validate(
            scenarios=[s.to_dict() for s in scenarios],
            clue=clue_dict,
            context=context_dict,
        )
        validation_report = hydrate_validation_report(
            validation_report,
            clue_dict,
            repo=instance.repo,
            context=context_dict,
        )
        validation_dict = validation_report.to_dict()
    if allow_unresolved_fallback and not validation_dict.get("selected_scenarios"):
        pending_repair = ensure_primary_scenario(
            {},
            clue=clue_dict,
            context={
                **context_dict,
                "iteration": iteration,
                "outer_iteration": iteration,
            },
            reason="issue_grounded_fallback_after_m4_no_selection",
        )
        repaired = pending_repair["selected_scenarios"][0]["normalized_scenario"]
        repaired_report = scenario_validator.validate(
            scenarios=[repaired],
            clue=clue_dict,
            context=context_dict,
        )
        repaired_report = hydrate_validation_report(
            repaired_report,
            clue_dict,
            repo=instance.repo,
            context=context_dict,
        )
        repaired_validation = repaired_report.to_dict()
        for bucket in ("selected_scenarios", "rejected_scenarios"):
            for record in repaired_validation.get(bucket, []) or []:
                if not isinstance(record, dict):
                    continue
                record["scenario_repaired"] = True
                record["scenario_repair_reason"] = "issue_grounded_fallback_after_m4_no_selection"
                record["recovery_validated"] = bucket == "selected_scenarios"
                record["force_selected"] = False
                normalized = record.get("normalized_scenario")
                if isinstance(normalized, dict):
                    normalized["scenario_repaired"] = True
                    normalized["scenario_repair_reason"] = "issue_grounded_fallback_after_m4_no_selection"
                    normalized["recovery_validated"] = bucket == "selected_scenarios"
                    if bucket == "rejected_scenarios":
                        normalized["diagnostic_only"] = True
        repaired_validation["rejected_scenarios"] = list(
            validation_dict.get("rejected_scenarios", []) or []
        ) + list(repaired_validation.get("rejected_scenarios", []) or [])
        repaired_validation["scenario_recovery"] = {
            "attempted": True,
            "validated": bool(repaired_validation.get("selected_scenarios")),
            "synthetic_score_used_for_admission": False,
            "reason": "issue_grounded_fallback_after_m4_no_selection",
        }
        validation_dict = repaired_validation
    validation_dict = _apply_m4_scenario_ranking(
        validation_dict,
        repo_root=_m4_repo_root(instance, context_dict),
        iteration=iteration,
        feature_flags=feature_flags,
        feature_profile=str(
            context_dict.get("feature_profile")
            or context_dict.get("methodology_revision")
            or ""
        ),
        clue_dict=clue_dict,
        context_dict=context_dict,
    )
    validation_dict = _limit_selected_scenarios(
        validation_dict,
        max_selected=(
            2
            if str(
                context_dict.get("feature_profile")
                or context_dict.get("methodology_revision")
                or ""
            ) in {"v36", "v37"}
            else 1
        ),
    )
    _mark_selected_m4_policy(validation_dict)
    return validation_dict


def _v37_structural_validation_dict(scenarios: Sequence[Any]) -> Dict[str, Any]:
    """Retain v37 structural validity without legacy semantic admission scores."""
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        payload = scenario.to_dict() if hasattr(scenario, "to_dict") else dict(scenario)
        scenario_id = str(payload.get("scenario_id") or f"S{index + 1}")
        missing = [
            name
            for name in (
                "target_function",
                "source_file",
                "oracle_type",
                "oracle_expected",
                "stimulus_steps",
            )
            if payload.get(name) in (None, "", [])
        ]
        record = {
            "scenario_id": scenario_id,
            "normalized_scenario": payload,
            "score": None,
            "score_breakdown": {"policy": "v37_structural_only"},
            "reasons": [],
            "force_selected": False,
        }
        if missing:
            record.update(
                {
                    "decision": "reject",
                    "validation_status": "rejected",
                    "diagnostic_only": True,
                    "reasons": ["missing required v37 field(s): " + ", ".join(missing)],
                }
            )
            rejected.append(record)
        else:
            record.update(
                {
                    "decision": "accept",
                    "validation_status": "accepted",
                    "diagnostic_only": False,
                }
            )
            selected.append(record)
    return {
        "schema_version": "m4-v37-structural-validation-v1",
        "selected_scenarios": selected,
        "rejected_scenarios": rejected,
        "validation_policy": "structural_fields_then_v37_formula_ranking",
    }


def _has_m5_eligible_selected(validation_dict: Mapping[str, Any]) -> bool:
    for item in validation_dict.get("selected_scenarios", []) or []:
        if not isinstance(item, Mapping):
            continue
        normalized = item.get("normalized_scenario")
        status = str(item.get("validation_status") or "")
        diagnostic = bool(item.get("diagnostic_only"))
        if isinstance(normalized, Mapping):
            status = status or str(normalized.get("validation_status") or "")
            diagnostic = diagnostic or bool(normalized.get("diagnostic_only"))
        if status == "accepted" and not diagnostic and not item.get("force_selected"):
            return True
    return False


def _scenario_with_generated_target(
    scenario: Mapping[str, Any],
    generated_test: Any,
) -> dict[str, Any]:
    """Overlay M5's repository-verified target on the scenario consumed by M7.

    M3 owns the issue stimulus, but M5 may resolve a receiver expression to a
    concrete implementation callable while inspecting the pre-patch source.
    Scoring the stale M3 target discards that stronger evidence and can report
    ``target_verified=False`` despite execution of the resolved function.
    """
    updated = copy.deepcopy(dict(scenario or {}))
    target = updated.get("target_location")
    target = dict(target) if isinstance(target, Mapping) else {}
    source_file = str(getattr(generated_test, "target_source_file", "") or "").strip()
    implementation = str(
        getattr(generated_test, "selected_implementation_target", "") or ""
    ).strip()
    issue_api = str(
        getattr(generated_test, "selected_issue_api_target", "")
        or getattr(generated_test, "m5_target_used", "")
        or ""
    ).strip()
    if source_file:
        target["source_file"] = source_file
        updated["source_file"] = source_file
    if implementation:
        target["canonical_target_identity"] = implementation
        target["target_function"] = implementation
        target["implementation_target"] = implementation
        updated["canonical_target_identity"] = implementation
        updated["target_function"] = implementation
        updated["implementation_target"] = implementation
    if issue_api:
        target["issue_api_target"] = issue_api
        target["candidate_invocation_expression"] = issue_api
        updated["issue_api_target"] = issue_api
        updated["candidate_invocation_expression"] = issue_api
    status = str(getattr(generated_test, "target_verification_status", "") or "")
    provenance = getattr(generated_test, "target_verification_provenance", {}) or {}
    if status:
        target["target_verification_status"] = status
        updated["target_verification_status"] = status
    if isinstance(provenance, Mapping) and provenance:
        target["target_verification_provenance"] = dict(provenance)
        updated["target_verification_provenance"] = dict(provenance)
    updated["target_location"] = target
    updated["m7_target_source"] = "M5_VERIFIED_TARGET_OVERLAY"
    return updated


def _m4_m5_eligibility_signature(validation_dict: Mapping[str, Any]) -> str:
    """Fingerprint only the M4 state that controls admission into M5."""
    records: list[dict[str, Any]] = []
    for bucket in ("selected_scenarios", "rejected_scenarios"):
        for item in validation_dict.get(bucket, []) or []:
            if not isinstance(item, Mapping):
                continue
            normalized = item.get("normalized_scenario")
            normalized = normalized if isinstance(normalized, Mapping) else {}
            target = normalized.get("target_location")
            target = target if isinstance(target, Mapping) else {}
            records.append({
                "bucket": bucket,
                "scenario_id": item.get("scenario_id") or normalized.get("scenario_id"),
                "validation_status": item.get("validation_status") or normalized.get("validation_status"),
                "diagnostic_only": bool(item.get("diagnostic_only") or normalized.get("diagnostic_only")),
                "force_selected": bool(item.get("force_selected")),
                "source_file": target.get("source_file") or normalized.get("source_file"),
                "target_function": (
                    target.get("canonical_target_identity")
                    or target.get("target_function")
                    or normalized.get("target_function")
                ),
                "execution_stimulus": normalized.get("execution_stimulus") or [],
                "expected_behavior": normalized.get("expected_behavior") or "",
                "oracle_intent": normalized.get("oracle_intent") or normalized.get("oracle") or "",
            })
    payload = {
        "m5_eligible": _has_m5_eligible_selected(validation_dict),
        "records": records,
    }
    return sha256_text(json.dumps(payload, sort_keys=True, default=str))


def _ensure_issue_grounded_fallback_if_needed(
    *,
    validation_dict: Dict[str, Any],
    scenario_generator: ScenarioGenerator,
    scenario_validator: ScenarioValidator,
    instance: BenchmarkInstance,
    clue_dict: Dict[str, Any],
    context_dict: Dict[str, Any],
    feature_flags: V22FeatureFlags,
    iteration: int,
    scenario_output_path: str,
    validation_output_path: str,
    reason: str,
) -> Dict[str, Any]:
    if _has_m5_eligible_selected(validation_dict):
        return validation_dict
    existing_records = list(validation_dict.get("selected_scenarios", []) or []) + list(
        validation_dict.get("rejected_scenarios", []) or []
    )
    for record in existing_records:
        if not isinstance(record, Mapping):
            continue
        normalized = record.get("normalized_scenario")
        statuses = {
            str(record.get("validation_status") or ""),
            str(
                normalized.get("validation_status")
                if isinstance(normalized, Mapping)
                else ""
            ),
        }
        if "rejected_feedback_not_applied" in statuses:
            # A feedback-owned rerun that could not safely apply its requested
            # change must remain diagnostic. Rebuilding a no-feedback fallback
            # here would silently discard the instruction and restore M5
            # eligibility for the unchanged scenario.
            return validation_dict
    fallback_scenarios = scenario_generator._build_fallback_scenarios(clue_dict, context_dict)
    for scenario in fallback_scenarios:
        scenario.generation_provenance = "ISSUE_GROUNDED_FALLBACK"
        scenario.fallback_used = True
        scenario.fallback_reason = reason
        scenario.diagnostic_only = False
    fallback_validation = _validate_and_rank_scenarios(
        scenario_validator=scenario_validator,
        scenarios=fallback_scenarios,
        instance=instance,
        clue_dict=clue_dict,
        context_dict=context_dict,
        feature_flags=feature_flags,
        iteration=iteration,
        allow_unresolved_fallback=False,
    )
    fallback_validation["fallback_used"] = True
    fallback_validation["fallback_reason"] = reason
    fallback_validation["fallback_policy"] = "validated_issue_grounded_fallback"
    if _has_m5_eligible_selected(fallback_validation):
        scenario_generator.save(fallback_scenarios, scenario_output_path)
        write_json(fallback_validation, validation_output_path)
        return fallback_validation
    return validation_dict


def _m3_last_fallback_used(scenario_generator: ScenarioGenerator) -> bool:
    metadata = getattr(scenario_generator, "_last_nonadaptive_metadata", None)
    if isinstance(metadata, Mapping):
        return bool(metadata.get("fallback_used"))
    return False


def _m3_last_fallback_reason(scenario_generator: ScenarioGenerator) -> str:
    metadata = getattr(scenario_generator, "_last_nonadaptive_metadata", None)
    if not isinstance(metadata, Mapping):
        return ""
    parse = metadata.get("parse_diagnostics")
    if isinstance(parse, Mapping) and parse.get("failure_kind"):
        return str(parse.get("failure_kind"))
    return "parse_fallback" if metadata.get("fallback_used") else ""


def _write_m3_nonadaptive_diagnostics(
    scenario_generator: ScenarioGenerator,
    *,
    scenario_output_path: str,
    attempt_artifact_dir: Path | None,
    attempt: int,
) -> None:
    metadata = getattr(scenario_generator, "_last_nonadaptive_metadata", None)
    if not isinstance(metadata, Mapping):
        return
    payload = dict(metadata)
    raw_response = str(payload.pop("raw_response", "") or "")
    payload["attempt"] = attempt
    payload["raw_response_artifact"] = ""
    root = Path(scenario_output_path).parent
    artifact_dirs = [root / "m3_diagnostics" / f"attempt_{attempt:03d}"]
    if attempt_artifact_dir is not None:
        artifact_dirs.append(attempt_artifact_dir / "m3_diagnostics" / f"attempt_{attempt:03d}")
    for artifact_dir in artifact_dirs:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        raw_path = artifact_dir / "raw_response.txt"
        raw_path.write_text(raw_response, encoding="utf-8")
        local_payload = dict(payload)
        local_payload["raw_response_artifact"] = str(raw_path)
        write_json(local_payload, artifact_dir / "parse_status.json")


def _annotate_m3_scenarios(
    scenarios: list[Any],
    *,
    instance_id: str,
    iteration: int,
    attempt: int,
    model_call_count: int,
    fallback_used: bool,
    fallback_reason: str,
    ) -> None:
    for scenario in scenarios:
        scenario.instance_id = instance_id
        scenario.iteration = iteration
        scenario.outer_iteration = iteration
        scenario.scenario_generation_attempt = attempt
        scenario.generation_provenance = (
            "issue_grounded_fallback"
            if fallback_used
            else getattr(scenario, "generation_provenance", "model_generated")
        )
        scenario.m3_model_call_count = model_call_count
        scenario.fallback_used = fallback_used
        scenario.fallback_reason = fallback_reason


def _mark_selected_m4_policy(validation_dict: Dict[str, Any]) -> None:
    for item in validation_dict.get("selected_scenarios", []) or []:
        if not isinstance(item, dict):
            continue
        item["m4_selection_policy"] = (
            "eligible_ranked_selection"
            if item.get("m4_rank") is not None
            else "eligible_base_order"
        )
        normalized = item.get("normalized_scenario")
        if isinstance(normalized, dict):
            normalized["m4_selection_policy"] = item["m4_selection_policy"]
            normalized["m4_candidate_classification"] = normalized.get("target_verification_status", "")


def _m3_validation_attempt_record(attempt: int, validation_dict: Mapping[str, Any]) -> dict[str, Any]:
    rejected = validation_dict.get("rejected_scenarios", []) if isinstance(validation_dict, Mapping) else []
    selected = validation_dict.get("selected_scenarios", []) if isinstance(validation_dict, Mapping) else []
    return {
        "attempt": attempt,
        "model_call_count": 1,
        "selected_count": len(selected or []),
        "rejected": [
            {
                "scenario_id": item.get("scenario_id"),
                "classification": _record_classification(item),
                "source_file": ((item.get("normalized_scenario") or {}).get("source_file") if isinstance(item.get("normalized_scenario"), dict) else None),
                "target_function": ((item.get("normalized_scenario") or {}).get("target_function") if isinstance(item.get("normalized_scenario"), dict) else None),
                "diagnostics": item.get("reasons", []),
            }
            for item in rejected or []
            if isinstance(item, Mapping)
        ],
    }


def _structured_m3_rejection_feedback(
    *,
    attempt: int,
    validation_dict: Mapping[str, Any],
    clue_dict: Mapping[str, Any],
) -> dict[str, Any]:
    rejected = _m3_validation_attempt_record(attempt, validation_dict).get("rejected", [])
    issue_api = ""
    for item in rejected:
        target = str(item.get("target_function") or "")
        if target:
            issue_api = target
            break
    return {
        "source": "m4_validation_rejection",
        "attempt": attempt,
        "rejected_scenarios": rejected,
        "exact_correction_constraints": [
            "preserve the public issue API that triggers the bug",
            "do not replace the issue API target with setup/helper calls",
            "use an existing candidate source file and test file",
            "include executable stimulus and an EB-grounded oracle",
        ],
        "previous_target_pair": {
            "issue_api": issue_api,
            "implementation_target": "",
        },
        "issue_api_that_must_be_preserved": issue_api,
        "pre_patch_only": True,
        "golden_patch_used": False,
        "post_patch_result_used": False,
        "issue_identifiers": clue_dict.get("identifiers", {}),
    }


def _record_classification(record: Mapping[str, Any]) -> str:
    normalized = record.get("normalized_scenario")
    if isinstance(normalized, Mapping):
        return str(normalized.get("target_verification_status") or normalized.get("m4_candidate_classification") or "")
    breakdown = record.get("score_breakdown") if isinstance(record.get("score_breakdown"), Mapping) else {}
    return str(breakdown.get("m4_candidate_classification") or "")


def _limit_selected_scenarios(
    validation_dict: Dict[str, Any],
    *,
    max_selected: int = 2,
) -> Dict[str, Any]:
    """Ensure M4 forwards at most the approved number of scenarios to M5."""
    selected = validation_dict.get("selected_scenarios")
    if not isinstance(selected, list) or len(selected) <= max_selected:
        return validation_dict
    trimmed = copy.deepcopy(validation_dict)
    overflow = trimmed["selected_scenarios"][max_selected:]
    trimmed["selected_scenarios"] = trimmed["selected_scenarios"][:max_selected]
    rejected = trimmed.setdefault("rejected_scenarios", [])
    for item in overflow:
        if isinstance(item, dict):
            item = copy.deepcopy(item)
            item["decision"] = "reject"
            item["validation_status"] = "rejected"
            item["diagnostic_only"] = True
            item["rejection_reason"] = "not_forwarded_to_m5_max_selected_limit"
            item.setdefault("reasons", []).append(
                "not forwarded to M5 because max selected scenario limit is 2"
            )
        rejected.append(item)
    return trimmed


def _generate_v37_ranked_candidates(
    *,
    generator: ReproductionTestGenerator,
    instance: Any,
    clue: Mapping[str, Any],
    context: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    iteration: int,
    runtime_error_hint: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Invoke M5 exactly once for every v37-selected scenario in rank order."""
    selected = [
        item
        for item in (validation_report.get("selected_scenarios") or [])[:2]
        if isinstance(item, Mapping)
    ]
    successful: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for scenario_rank, selected_record in enumerate(selected, 1):
        scenario = selected_record.get("normalized_scenario")
        scenario = dict(scenario if isinstance(scenario, Mapping) else selected_record)
        scenario_id = str(scenario.get("scenario_id") or f"S{scenario_rank}")
        restricted_report = {
            "selected_scenarios": [copy.deepcopy(dict(selected_record))],
            "rejected_scenarios": [],
        }
        try:
            generated = generator.generate(
                instance=instance,
                clue=dict(clue),
                context=dict(context),
                validation_report=restricted_report,
                iteration=iteration,
                runtime_error_hint=runtime_error_hint,
            )
            provenance = dict(generated.m5_invocation_provenance or {})
            provenance.update(
                {
                    "selected_scenario_id": scenario_id,
                    "scenario_rank": scenario_rank,
                    "model_call_index": scenario_rank,
                    "model_call_count": len(selected),
                }
            )
            generated.m5_invocation_provenance = provenance
            successful.append(
                {
                    "scenario_rank": scenario_rank,
                    "scenario_id": scenario_id,
                    "scenario": scenario,
                    "generated_test": generated,
                }
            )
            records.append(
                {
                    "scenario_rank": scenario_rank,
                    "scenario_id": scenario_id,
                    "m5_status": "GENERATED",
                    "m5_output_identity": {
                        "canonical_test_nodeid": generated.canonical_test_nodeid,
                        "generated_patch_sha256": generated.generated_patch_sha256,
                    },
                }
            )
        except Exception as error:
            records.append(
                {
                    "scenario_rank": scenario_rank,
                    "scenario_id": scenario_id,
                    "m5_status": "FAILED",
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                    "m5_output_identity": None,
                }
            )
    return successful, records


def _evaluate_v37_additional_candidate(
    *,
    candidate: Mapping[str, Any],
    candidate_dir: Path,
    generator: ReproductionTestGenerator,
    alignment_runner: AlignmentRunner,
    scorer: AlignmentScorer,
    instance: Any,
    pre_patch_view: Any,
    clue: Mapping[str, Any],
    context: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    feature_flags: V22FeatureFlags,
    scorer_feature_flags: V22FeatureFlags,
    iteration: int,
    max_feedback_iterations: int,
    diagnosis_revision: str,
) -> dict[str, Any]:
    """Run one additional ranked v37 candidate through M5-A, M6, and M7."""
    candidate_dir.mkdir(parents=True, exist_ok=True)
    generated = candidate["generated_test"]
    scenario = _scenario_with_generated_target(
        dict(candidate["scenario"]), generated
    )
    generated_path = candidate_dir / "generated_test.json"
    generator.save(generated, str(generated_path))
    supplemental_context = dict(context)
    supplemental_context.update(
        {"feature_profile": "v37", "methodology_revision": "v37"}
    )
    execution = alignment_runner.run(
        instance=pre_patch_view,
        generated_test_json_path=str(generated_path),
        run_id=(
            f"align-{getattr(instance, 'instance_id', '')}-it{iteration}"
            f"-scenario-rank-{candidate['scenario_rank']}"
        ),
        iteration=iteration,
        feature_flags=feature_flags,
        supplemental_context=supplemental_context,
        supplemental_clue=dict(clue),
    )
    artifacts = alignment_runner.save(
        execution,
        str(candidate_dir / "alignment_execution.json"),
        feature_flags=feature_flags,
    )
    restricted_validation = {
        "selected_scenarios": [
            {
                "scenario_id": candidate["scenario_id"],
                "rank": candidate["scenario_rank"],
                "normalized_scenario": dict(candidate["scenario"]),
            }
        ],
        "rejected_scenarios": [],
    }
    repaired = _attempt_m6_m5a_repair(
        instance=instance,
        pre_patch_view=pre_patch_view,
        output_dir=candidate_dir,
        generator=generator,
        alignment_runner=alignment_runner,
        original_generated_test=generated,
        original_generated_test_path=str(generated_path),
        align_result=execution,
        clue=dict(clue),
        context=dict(context),
        validation_report=restricted_validation,
        feature_flags=feature_flags,
        feature_profile="v37",
        iteration=iteration,
        prior_m5a_attempt_count=0,
    )
    if repaired is not None:
        generated, execution, artifacts, _repair_telemetry = repaired
    execution_payload = execution.to_dict()
    if isinstance(artifacts, Mapping):
        sbfl_payload = artifacts.get("sbfl_result")
        if sbfl_payload is not None:
            execution_payload["canonical_m6_sbfl_result"] = sbfl_payload
    m7_context = dict(context)
    m7_context.update(
        {
            "methodology_revision": "v37",
            "max_feedback_iterations": max_feedback_iterations,
        }
    )
    alignment = scorer.evaluate(
        execution_result=execution_payload,
        clue=dict(clue),
        scenario=scenario,
        generated_test=generated.to_dict(),
        iteration=iteration,
        validation_report=restricted_validation,
        context=m7_context,
        feature_flags=scorer_feature_flags,
    )
    if _requires_v29_conservative_gate_judgment(alignment):
        evidence = _m7_evaluation_evidence(
            alignment_result=alignment,
            align_result=execution,
            semantic_fingerprint="",
            repeated_semantic_fingerprint=False,
            selected_records=_selected_scenario_records(restricted_validation),
            score_breakdown=alignment.score_breakdown,
            remaining_outer_iterations=max(0, max_feedback_iterations - iteration),
            clue=clue,
            context=context,
            scenario=scenario,
            generated_test=generated,
            m6_artifacts=artifacts,
        )
        evidence["m7_decision_context"] = "V37_CONSERVATIVE_GATE"
        decision = _build_m7_feedback_decision(
            client=_selected_m7_client(generator),
            decision_status="WEAK_ALIGNMENT",
            iteration=iteration,
            source_stage="M7",
            failure_category="NONE",
            evidence=evidence,
            feedback_branch="M7_CONSERVATIVE_DIAGNOSIS",
            next_start_stage="M5",
            rerun_targets=["M5", "M5-A", "M6", "M7"],
            prohibited_fingerprints=[],
            diagnosis_enabled=feature_flags.m7_llm_scenario_refinement,
            diagnosis_revision=diagnosis_revision,
            max_feedback_iterations=max_feedback_iterations,
        )
        _apply_v29_conservative_gate_decision(
            alignment,
            decision,
            iteration=iteration,
            max_feedback_iterations=max_feedback_iterations,
        )
    payload = alignment.to_dict()
    payload.update(
        {
            "instance_id": getattr(instance, "instance_id", ""),
            "scenario_id": candidate["scenario_id"],
            "scenario_rank": candidate["scenario_rank"],
            "pass_provenance": {
                "candidate_identity": {
                    "test_id": getattr(generated, "test_id", ""),
                    "canonical_test_nodeid": generated.canonical_test_nodeid,
                    "generated_patch_sha256": generated.generated_patch_sha256,
                }
            },
        }
    )
    write_json_atomic(payload, candidate_dir / "alignment_result.json")
    return {
        "scenario_id": candidate["scenario_id"],
        "scenario_rank": candidate["scenario_rank"],
        "candidate_dir": str(candidate_dir),
        "generated_test": generated,
        "execution": execution,
        "m6_artifacts": artifacts,
        "alignment": alignment,
        "m5_output_identity": {
            "canonical_test_nodeid": generated.canonical_test_nodeid,
            "generated_patch_sha256": generated.generated_patch_sha256,
        },
        "m6_execution_identity": {
            "canonical_test_id": execution.canonical_test_id,
            "canonical_test_nodeid": execution.canonical_test_nodeid,
            "generated_patch_sha256": execution.generated_patch_sha256,
        },
        "m7_result": alignment.failure_type,
        "admitted_to_final_set": alignment.failure_type == "ALIGNED",
    }


def _apply_m4_scenario_ranking(
    validation_dict: Dict[str, Any],
    *,
    repo_root: Path,
    iteration: int,
    feature_flags: V22FeatureFlags,
    feature_profile: str | None = None,
    clue_dict: Mapping[str, Any] | None = None,
    context_dict: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Optionally rank validated M4 scenarios without fabricating evidence."""
    feature_metadata = _m4_ranking_feature_metadata(feature_flags)
    if not any(item["requested"] for item in feature_metadata.values()):
        ranked = copy.deepcopy(validation_dict)
        ranked["m4_ranking_metadata"] = feature_metadata
        return ranked

    ranked = copy.deepcopy(validation_dict)
    original_selected = [
        item
        for item in ranked.get("selected_scenarios", []) or []
        if isinstance(item, Mapping)
    ]
    candidate_scenarios = [
        _scenario_for_m4_ranking(item)
        for item in original_selected
    ]
    candidate_scenarios = [item for item in candidate_scenarios if item]
    if not candidate_scenarios:
        feature_metadata = _mark_m4_features_not_applicable(
            feature_metadata,
            reason="insufficient_evidence",
            provenance={"scenario_count": 0},
        )
        ranked["m4_ranking_metadata"] = feature_metadata
        return ranked

    if feature_profile == "v37":
        rel_llm_by_source = {
            str(item.get("file_path") or item.get("path") or ""): float(
                item.get("llm_relevance_norm") or item.get("rel_llm") or 0.0
            )
            for item in (context_dict or {}).get("file_ranking", []) or []
            if isinstance(item, Mapping)
        }
        report = rank_v37_scenarios(
            candidate_scenarios,
            repo_root,
            clue=clue_dict or {},
            rel_llm_by_source=rel_llm_by_source,
            iteration=iteration,
            sbfl_norm_by_id=(context_dict or {}).get("prior_m6_sbfl_norm_by_scenario"),
            prior_sbfl_spectrum=(context_dict or {}).get("prior_m6_sbfl_spectrum"),
            prior_covered_sut_lines=(context_dict or {}).get("prior_m6_covered_sut_lines"),
            max_selected=2,
        )
    else:
        report = rank_scenarios(
            candidate_scenarios,
            repo_root,
            iteration=iteration,
            max_selected=2 if feature_profile == "v36" else MAX_SELECTED_SCENARIOS,
            input_provenance={
                "scenario_source": "scenario_validation.selected_scenarios.normalized_scenario",
                "repo_root": str(repo_root),
                "allowed_evidence": "explicit scenario fields and canonical pre-patch artifacts only",
            },
        )
    ranked["m4_ranking_report"] = report
    ranked["m4_ranking_metadata"] = _m4_feature_metadata_from_report(
        feature_metadata,
        report,
        scenario_count=len(candidate_scenarios),
    )
    if feature_profile == "v36" and not report.get("selected_scenarios"):
        ranked["selected_scenarios"] = []
        ranked.setdefault("rejected_scenarios", [])
        for item in original_selected:
            rejected = _m4_rejected_from_validation_record(item)
            rejected["rejection_reason"] = "BLOCKING_V36_M4_FORMULA_EVIDENCE_UNAVAILABLE"
            rejected["diagnostic_only"] = True
            ranked["rejected_scenarios"].append(rejected)
        ranked["m4_v36_blocking"] = {
            "status": "BLOCKING",
            "reason": (
                "BaseScore component acquisition, scenario C_f provenance, "
                "and RRA tie handling are not approved"
            ),
        }
        return ranked
    if not feature_flags.m4_scenario_score or not report.get("selected_scenarios"):
        return ranked

    by_id = {
        str(item.get("scenario_id")): item
        for item in original_selected
        if item.get("scenario_id") is not None
    }
    selected_records: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for ranked_item in report.get("selected_scenarios", []) or []:
        scenario_id = str(ranked_item.get("scenario_id") or "")
        if not scenario_id or scenario_id not in by_id:
            continue
        selected_ids.add(scenario_id)
        selected_records.append(_merge_m4_ranking_record(by_id[scenario_id], ranked_item))
    if selected_records:
        overflow = [
            _m4_rejected_from_validation_record(item)
            for item in original_selected
            if str(item.get("scenario_id")) not in selected_ids
        ]
        ranked["selected_scenarios"] = selected_records
        ranked.setdefault("rejected_scenarios", [])
        ranked["rejected_scenarios"].extend(overflow)
    return ranked


def _m4_ranking_feature_metadata(feature_flags: V22FeatureFlags) -> dict[str, dict[str, Any]]:
    return {
        "m4_scenario_score": _m4_feature_entry(feature_flags.m4_scenario_score),
        "m4_rra": _m4_feature_entry(feature_flags.m4_rra),
        "m4_multi_formula_consensus": _m4_feature_entry(feature_flags.m4_multi_formula_consensus),
    }


def _m4_feature_entry(requested: bool) -> dict[str, Any]:
    return {
        "requested": requested,
        "enabled": requested,
        "used": False,
        "status": "disabled" if not requested else "inactive",
        "fallback": "base_scenario_selection" if not requested else "",
        "reason": "feature_flag_disabled" if not requested else None,
        "evidence": {},
        "input_evidence_provenance": {},
    }


def _mark_m4_features_not_applicable(
    metadata: dict[str, dict[str, Any]],
    *,
    reason: str,
    provenance: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    updated = copy.deepcopy(metadata)
    for item in updated.values():
        if item["requested"]:
            item.update({
                "used": False,
                "status": "not_applicable",
                "fallback": "base_scenario_selection",
                "reason": reason,
                "evidence": {"unmet_precondition": reason},
                "input_evidence_provenance": dict(provenance),
            })
    return updated


def _m4_feature_metadata_from_report(
    metadata: dict[str, dict[str, Any]],
    report: Mapping[str, Any],
    *,
    scenario_count: int,
) -> dict[str, dict[str, Any]]:
    updated = copy.deepcopy(metadata)
    available_scores = [
        item for item in report.get("scenario_scores", []) or []
        if isinstance(item, Mapping) and item.get("status") == "AVAILABLE"
    ]
    if updated["m4_scenario_score"]["requested"]:
        used = bool(available_scores)
        updated["m4_scenario_score"].update({
            "used": used,
            "status": "used" if used else "not_applicable",
            "fallback": "" if used else "base_scenario_selection",
            "reason": None if used else "insufficient_evidence",
            "evidence": {"scenario_scores_available": len(available_scores)},
            "input_evidence_provenance": {
                "scenario_scores": [
                    {
                        "scenario_id": item.get("scenario_id"),
                        "components": item.get("components"),
                    }
                    for item in report.get("scenario_scores", []) or []
                    if isinstance(item, Mapping)
                ],
            },
        })
    if updated["m4_rra"]["requested"]:
        updated["m4_rra"].update({
            "used": False,
            "status": "not_applicable",
            "fallback": "base_scenario_selection",
            "reason": (
                "insufficient_scenarios"
                if scenario_count < 2
                else "insufficient_evidence"
            ),
            "evidence": {"scenario_count": scenario_count},
            "input_evidence_provenance": {
                "rank_aggregation_status": report.get("rank_aggregation_status"),
                "rank_aggregation_method": report.get("rank_aggregation_method"),
            },
        })
    if updated["m4_multi_formula_consensus"]["requested"]:
        consensus = report.get("multi_formula_consensus") or {}
        top5 = consensus.get("top5_intersection") if isinstance(consensus, Mapping) else []
        used = scenario_count >= 2 and bool(top5)
        updated["m4_multi_formula_consensus"].update({
            "used": used,
            "status": "used" if used else "not_applicable",
            "fallback": "" if used else "base_scenario_selection",
            "reason": (
                None
                if used
                else ("insufficient_scenarios" if scenario_count < 2 else "insufficient_evidence")
            ),
            "evidence": {"scenario_count": scenario_count, "top5_intersection_count": len(top5 or [])},
            "input_evidence_provenance": consensus if isinstance(consensus, Mapping) else {},
        })
    return updated


def _scenario_for_m4_ranking(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = record.get("normalized_scenario")
    scenario = copy.deepcopy(normalized) if isinstance(normalized, Mapping) else {}
    if not scenario:
        return {}
    scenario.setdefault("scenario_id", record.get("scenario_id"))
    return scenario


def _merge_m4_ranking_record(
    original: Mapping[str, Any],
    ranked_item: Mapping[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(original))
    scoring = ranked_item.get("scoring") if isinstance(ranked_item.get("scoring"), Mapping) else {}
    merged["m4_rank"] = ranked_item.get("rank")
    merged["m4_ranking"] = {
        "selected": True,
        "stable_identity": ranked_item.get("stable_identity"),
        "scoring": dict(scoring),
    }
    if scoring.get("ScenarioScore") is not None:
        merged["score"] = scoring["ScenarioScore"]
    return merged


def _m4_rejected_from_validation_record(original: Mapping[str, Any]) -> dict[str, Any]:
    rejected = copy.deepcopy(dict(original))
    rejected["decision"] = "reject"
    rejected["validation_status"] = "rejected"
    rejected["diagnostic_only"] = True
    rejected["rejection_reason"] = "not_selected_by_m4_ranking"
    rejected.setdefault("reasons", []).append("not selected by enabled M4 scenario ranking")
    return rejected


def _m4_repo_root(instance: Any, context: Mapping[str, Any]) -> Path:
    raw = context.get("repo_path") if isinstance(context, Mapping) else None
    if raw:
        return Path(str(raw))
    before_patch = getattr(instance, "before_patch_repo_path", None)
    if before_patch:
        return Path(str(before_patch))
    return Path.cwd()


def _feedback_route(alignment_result: Any) -> tuple[str, list[str]]:
    structured = getattr(alignment_result, "structured_feedback", {}) or {}
    branch = structured.get("feedback_branch")
    targets = structured.get("target_modules")
    if isinstance(branch, str) and isinstance(targets, list):
        return branch, [str(target) for target in targets]
    status = str(
        getattr(alignment_result, "m7_alignment_status", None)
        or getattr(alignment_result, "failure_type", "")
    )
    fallback = {
        "ALIGNED": ("none", []),
        "NOT_VALID": ("M5", ["M5"]),
        "ERROR": ("M6", ["M6"]),
        "NOT_FAILED": ("M5", ["M5"]),
        "NO_COVERAGE": ("M2+M5", ["M2", "M5"]),
        "WEAK_ALIGNMENT": ("M3+M5", ["M3", "M5"]),
    }
    return fallback.get(status, ("unsupported", []))


def _m3_feedback_payload(
    *,
    alignment_result: Any,
    generated_test: Any,
    align_result: Any,
    rerun_targets: list[str],
    feedback_decision: M7FeedbackDecision,
) -> dict[str, Any]:
    """Build the pre-patch M7 feedback payload for a new M3 invocation."""
    payload = {
        "feedback_branch": feedback_decision.selected_feedback_branch,
        "target_modules": [str(target) for target in rerun_targets],
        "m7_alignment_status": getattr(alignment_result, "m7_alignment_status", None)
        or getattr(alignment_result, "failure_type", ""),
        "previous_generated_scenario_id": getattr(generated_test, "scenario_id", ""),
        "pre_patch_execution": {
            "returncode": getattr(align_result, "returncode", None),
            "has_failure": getattr(align_result, "has_failure", None),
            "error_messages": list(getattr(align_result, "error_messages", []) or [])[:5],
            "test_results": getattr(align_result, "test_results", {}),
        },
        **_v26_diagnosis_fields(feedback_decision),
    }
    return payload


def _v26_diagnosis_fields(feedback_decision: M7FeedbackDecision) -> dict[str, Any]:
    """Return the exact M7 diagnosis fields consumed by downstream modules."""
    return {
        "failure_reason": feedback_decision.failure_reason,
        "assumption_gap": feedback_decision.assumption_gap,
        "next_scenario_change": feedback_decision.next_scenario_change,
        "admissible_alternatives": feedback_decision.admissible_alternatives,
        "evidence_refs": list(feedback_decision.evidence_refs),
        "recommended_change": feedback_decision.recommended_change,
        "change_owner_module": feedback_decision.change_owner_module,
        "previous_feedback_effect": feedback_decision.previous_feedback_effect,
        "confidence": feedback_decision.confidence,
        "route_destination": feedback_decision.route_destination,
    }


def _inject_v26_diagnosis(
    validation_dict: Mapping[str, Any],
    feedback_decision: M7FeedbackDecision,
    *,
    generated_test: Any | None = None,
    align_result: Any | None = None,
) -> dict[str, Any]:
    """Persist the exact M7 diagnosis on the scenario consumed by M5."""
    updated = copy.deepcopy(dict(validation_dict))
    diagnosis = _v26_diagnosis_fields(feedback_decision)
    generated_payload = (
        generated_test.to_dict()
        if generated_test is not None and hasattr(generated_test, "to_dict")
        else {}
    )
    code = str(generated_payload.get("test_code") or generated_payload.get("append_block") or "")
    repair_contract = _repair_contract_from_feedback(
        feedback_decision,
        previous_candidate_code=code,
    )
    avoid_evidence = {
        "candidate_sha256": generated_payload.get("generated_patch_sha256")
        or generated_payload.get("patch_sha256"),
        "canonical_test_nodeid": generated_payload.get("canonical_test_nodeid"),
        "previous_assertion_patterns": [
            line.strip()
            for line in code.splitlines()
            if "assert" in line and line.strip()
        ][:8],
        "previous_stimulus_patterns": [],
    }
    runtime_evidence = {
        "test_results": dict(getattr(align_result, "test_results", {}) or {}),
        "error_messages": list(getattr(align_result, "error_messages", []) or [])[:8],
        "observed_output_excerpt": str(getattr(align_result, "raw_output", "") or "")[-2000:],
    }
    for selected in updated.get("selected_scenarios", []) or []:
        if not isinstance(selected, dict):
            continue
        scenario = selected.get("normalized_scenario")
        if not isinstance(scenario, dict):
            continue
        avoid_evidence["previous_stimulus_patterns"] = list(
            scenario.get("execution_stimulus") or scenario.get("stimulus_steps") or []
        )[:8]
        scenario["feedback_consumed"] = diagnosis
        scenario["m7_diagnosis"] = diagnosis
        scenario["previous_pass_avoid_evidence"] = copy.deepcopy(avoid_evidence)
        scenario["previous_pass_runtime_evidence"] = copy.deepcopy(runtime_evidence)
        scenario["repair_directive"] = copy.deepcopy(repair_contract)
    return updated


def _extract_first_json_object(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("M7 Qwen refiner response was empty")
    fence = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("M7 Qwen refiner response did not contain a JSON object")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise ValueError("M7 Qwen refiner JSON object was incomplete")


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, Mapping):
            return {str(k): _json_safe(v) for k, v in value.items() if not callable(v)}
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [_json_safe(item) for item in value]
        return str(value)


def _m7_llm_application_record(
    *,
    iteration_dir: Path,
    iteration: int,
    alignment_result: Any,
    generated_test: Any,
    align_result: Any,
) -> dict[str, Any] | None:
    structured = getattr(alignment_result, "structured_feedback", {}) or {}
    llm = structured.get("llm_scenario_refinement") if isinstance(structured, Mapping) else {}
    if not isinstance(llm, Mapping) or not llm.get("used_llm"):
        return None
    refined = getattr(alignment_result, "refined_scenario", {}) or {}
    application = refined.get("m7_llm_feedback_application") if isinstance(refined, Mapping) else {}
    artifact_path = iteration_dir / "m7_feedback_application.json"
    return {
        "schema_version": "m7-feedback-application-v1",
        "iteration": iteration,
        "verdict": getattr(alignment_result, "failure_type", ""),
        "artifact_path": str(artifact_path),
        "llm_feedback": _json_safe(llm),
        "feedback_applied": bool(application),
        "scenario_hash_before": (application or {}).get("scenario_hash_before"),
        "scenario_hash_after": (application or {}).get("scenario_hash_after"),
        "before": _m7_feedback_effect_snapshot(
            generated_test=generated_test,
            align_result=align_result,
            alignment_result=alignment_result,
        ),
        "after": None,
        "effectiveness": {"status": "PENDING_NEXT_ITERATION"},
    }


def _m7_feedback_effect_snapshot(
    *,
    generated_test: Any,
    align_result: Any,
    alignment_result: Any,
) -> dict[str, Any]:
    code = str(getattr(generated_test, "test_code", "") or "")
    score = getattr(alignment_result, "score_breakdown", {}) or {}
    return {
        "test_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "target_test_file": str(getattr(generated_test, "target_test_file", "") or ""),
        "canonical_test_nodeid": str(getattr(generated_test, "canonical_test_nodeid", "") or ""),
        "execution_result": {
            "has_failure": bool(getattr(align_result, "has_failure", False)),
            "has_error": bool(getattr(align_result, "has_error", False)),
            "test_results": _json_safe(getattr(align_result, "test_results", {}) or {}),
            "error_messages": _json_safe(getattr(align_result, "error_messages", []) or []),
        },
        "coverage": score.get("coverage_score"),
        "alignment_scores": {
            "bug_fail_score": score.get("bug_fail_score"),
            "coverage_score": score.get("coverage_score"),
            "issue_alignment_score": score.get("issue_alignment_score"),
        },
        "m7_status": getattr(alignment_result, "m7_alignment_status", None)
        or getattr(alignment_result, "failure_type", ""),
    }


def _judge_m7_feedback_effectiveness(before: Mapping[str, Any], after: Mapping[str, Any], verdict: str) -> dict[str, Any]:
    changed = {
        "test_sha_changed": before.get("test_sha256") != after.get("test_sha256"),
        "execution_changed": before.get("execution_result") != after.get("execution_result"),
        "coverage_changed": before.get("coverage") != after.get("coverage"),
        "status_changed": before.get("m7_status") != after.get("m7_status"),
    }
    before_scores = before.get("alignment_scores") if isinstance(before.get("alignment_scores"), Mapping) else {}
    after_scores = after.get("alignment_scores") if isinstance(after.get("alignment_scores"), Mapping) else {}
    score_improved = any(
        (after_scores.get(key) or 0) > (before_scores.get(key) or 0)
        for key in ("bug_fail_score", "coverage_score", "issue_alignment_score")
    )
    intended = False
    if verdict == "NOT_FAILED":
        intended = bool(after_scores.get("bug_fail_score", 0) > before_scores.get("bug_fail_score", 0))
    elif verdict == "NO_COVERAGE":
        intended = bool(after_scores.get("coverage_score", 0) > before_scores.get("coverage_score", 0))
    elif verdict == "WEAK_ALIGNMENT":
        intended = score_improved or after.get("m7_status") == "ALIGNED"
    effective = changed["test_sha_changed"] and (intended or changed["status_changed"])
    return {
        "status": "EFFECTIVE" if effective else "INEFFECTIVE",
        "measurable_changes": changed,
        "score_improved": score_improved,
        "intended_direction_improved": intended,
    }


def _selected_scenario_records(validation_dict: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in validation_dict.get("selected_scenarios", []) or []:
        if not isinstance(item, Mapping):
            continue
        normalized = item.get("normalized_scenario")
        records.append({
            "scenario_id": item.get("scenario_id") or (
                normalized.get("scenario_id") if isinstance(normalized, Mapping) else None
            ),
            "score": item.get("score"),
            "decision": item.get("decision"),
            "validation_status": item.get("validation_status"),
            "diagnostic_only": item.get("diagnostic_only"),
            "force_selected": item.get("force_selected", False),
        })
    return records


def _candidate_semantic_fingerprint(
    *,
    scenario: Mapping[str, Any],
    generated_test: Any,
    align_result: Any,
    alignment_result: Any,
) -> str:
    code = str(getattr(generated_test, "test_code", "") or "")
    try:
        ast_fingerprint = ast.dump(ast.parse(code), include_attributes=False)
    except SyntaxError:
        ast_fingerprint = re.sub(r"\s+", " ", code).strip()
    target = scenario.get("target_location") if isinstance(scenario.get("target_location"), Mapping) else {}
    coverage = getattr(align_result, "coverage_data", {}) or {}
    covered_files = sorted(str(key) for key in coverage.keys()) if isinstance(coverage, Mapping) else []
    payload = {
        "issue_api_target": scenario.get("issue_api_target") or target.get("issue_api_target") or target.get("target_function"),
        "implementation_target": scenario.get("implementation_target") or target.get("implementation_target") or target.get("target_function"),
        "target_test_file": getattr(generated_test, "target_test_file", "") or "",
        "setup": scenario.get("setup_steps") or [],
        "stimulus": scenario.get("execution_stimulus") or scenario.get("stimulus_steps") or [],
        "assertion_ast": ast_fingerprint,
        "execution_command": getattr(align_result, "test_nodeid", None)
        or getattr(generated_test, "canonical_test_nodeid", ""),
        "covered_files": covered_files,
        "failure_reason": getattr(alignment_result, "failure_type", ""),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _pre_m6_candidate_fingerprint(
    *,
    scenario: Mapping[str, Any],
    generated_test: Any,
) -> str:
    code = str(getattr(generated_test, "test_code", "") or "")
    try:
        ast_fingerprint = ast.dump(ast.parse(code), include_attributes=False)
    except SyntaxError:
        ast_fingerprint = re.sub(r"\s+", " ", code).strip()
    target = scenario.get("target_location") if isinstance(scenario.get("target_location"), Mapping) else {}
    payload = {
        "issue_api_target": scenario.get("issue_api_target") or target.get("issue_api_target") or target.get("target_function"),
        "implementation_target": scenario.get("implementation_target") or target.get("implementation_target") or target.get("target_function"),
        "target_test_file": getattr(generated_test, "target_test_file", "") or "",
        "setup": scenario.get("setup_steps") or [],
        "stimulus": scenario.get("execution_stimulus") or scenario.get("stimulus_steps") or [],
        "oracle": scenario.get("oracle") or scenario.get("oracle_expected") or "",
        "assertion_ast": ast_fingerprint,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_repeated_semantic_fingerprint(
    iteration_history: list[dict[str, Any]],
    semantic_fingerprint: str,
    *,
    status: str,
) -> bool:
    if not semantic_fingerprint:
        return False
    for record in iteration_history:
        if (
            record.get("semantic_progress_fingerprint") == semantic_fingerprint
            and record.get("failure_type") == status
        ):
            return True
    return False


def _rerun_effect_snapshot(
    *,
    context: Mapping[str, Any],
    scenario: Mapping[str, Any],
    candidate_code: str,
    validation_report: Mapping[str, Any] | None = None,
    candidate_identity: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Fingerprint behaviorally relevant rerun inputs without patch-only data."""
    context_payload = {
        "fault_hypothesis": context.get("fault_hypothesis"),
        "oracle_hint": context.get("oracle_hint"),
        "localization_hypotheses": context.get("localization_hypotheses") or [],
        "target_hypotheses": context.get("target_hypotheses") or [],
        "canonical_fault_candidates": context.get("canonical_fault_candidates") or [],
        "file_ranking": context.get("file_ranking") or [],
        "function_ranking": context.get("function_ranking") or [],
        "initial_suspicious_functions": context.get("initial_suspicious_functions") or [],
        "candidate_source_files": context.get("candidate_source_files") or [],
    }
    scenario_payload = {
        "target_location": scenario.get("target_location") or {},
        "target_function": scenario.get("target_function"),
        "source_file": scenario.get("source_file"),
        "oracle": scenario.get("oracle") or scenario.get("oracle_contract") or scenario.get("oracle_expected"),
        "stimulus": scenario.get("execution_stimulus") or scenario.get("stimulus_steps") or [],
        "setup_steps": scenario.get("setup_steps") or [],
        "preconditions": scenario.get("preconditions") or [],
        "setup_helper_calls": scenario.get("setup_helper_calls") or [],
        "test_environment": scenario.get("test_environment") or {},
        "expected_failure": scenario.get("expected_failure"),
        "coverage_intent": scenario.get("coverage_intent") or {},
        "relevant_source_files": scenario.get("relevant_source_files") or [],
    }
    try:
        candidate_tree = ast.parse(candidate_code)
        imports = sorted(
            ast.dump(node, include_attributes=False)
            for node in candidate_tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        )
        statements = [
            ast.dump(node, include_attributes=False)
            for node in candidate_tree.body
            if not isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        candidate_code_identity = {"imports": imports, "statements": statements}
    except SyntaxError:
        candidate_code_identity = {"raw": candidate_code}
    material_candidate_identity = {
        "code": candidate_code_identity,
        "target_test_file": (candidate_identity or {}).get("target_test_file"),
        "canonical_test_nodeid": (candidate_identity or {}).get("canonical_test_nodeid"),
        "execution_command": (candidate_identity or {}).get("execution_command"),
        "imports": (candidate_identity or {}).get("imports") or [],
        "oracle_identity": (candidate_identity or {}).get("oracle_identity"),
    }
    return {
        "context_fingerprint": sha256_text(
            json.dumps(context_payload, ensure_ascii=False, sort_keys=True, default=str)
        ),
        "scenario_fingerprint": sha256_text(
            json.dumps(scenario_payload, ensure_ascii=False, sort_keys=True, default=str)
        ),
        "candidate_fingerprint": sha256_text(
            json.dumps(material_candidate_identity, sort_keys=True, separators=(",", ":"), default=str)
        ) if candidate_code else "",
        "m6_execution_fingerprint": str(
            (candidate_identity or {}).get("m6_execution_fingerprint") or ""
        ),
        "owner_identity_version": "material-v31-v3",
    }


def _m6_execution_progress_fingerprint(align_result: Any) -> str:
    """Fingerprint only the M6-owned execution/postprocessing evidence."""
    material = {
        "test_results": getattr(align_result, "test_results", {}) or {},
        "raw_output_sha256": sha256_text(
            str(getattr(align_result, "raw_output", "") or "")
        ),
        "coverage_data": getattr(align_result, "coverage_data", {}) or {},
        "generated_patch_sha256": getattr(
            align_result, "generated_patch_sha256", None
        ),
        "execution_command": getattr(align_result, "execution_command", None),
        "failure_category": getattr(align_result, "failure_category", None),
        "error_stage": getattr(align_result, "error_stage", None),
        "error_origin": getattr(align_result, "error_origin", None),
        "exception_type": getattr(align_result, "exception_type", None),
        "error_messages": getattr(align_result, "error_messages", []) or [],
    }
    if not any(
        value
        for key, value in material.items()
        if key != "raw_output_sha256"
    ) and not str(getattr(align_result, "raw_output", "") or ""):
        return ""
    return sha256_text(
        json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    )


def _finalize_v27_pass_manifest(
    *,
    enabled: bool,
    iteration_dir: Path,
    iteration_record: Mapping[str, Any],
    previous_records: list[dict[str, Any]],
    context: Mapping[str, Any],
    scenario: Mapping[str, Any],
    candidate_code: str,
    validation_report: Mapping[str, Any] | None = None,
    schema_revision: str = "v27",
) -> None:
    """Upgrade one just-written pass manifest to the complete v27 contract."""
    if not enabled:
        return
    manifest_path = iteration_dir / "pass_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot finalize v27 pass provenance: {manifest_path}") from exc
    previous = previous_records[-1] if previous_records else {}
    diagnosis = iteration_record.get("diagnosis") or {}
    rerun_identities = _rerun_effect_snapshot(
        context=context,
        scenario=scenario,
        validation_report=(
            validation_report
            if isinstance(validation_report, Mapping)
            else {"selected_scenarios": [{"normalized_scenario": dict(scenario)}]}
        ),
        candidate_code=candidate_code,
    )
    telemetry = _read_mapping_artifact(iteration_dir / "feature_execution_telemetry.json")
    raw_path = iteration_dir / "m5_invalid_candidate" / "candidate_raw.py"
    postprocessed_path = iteration_dir / "generated_test_rendered.py"
    repaired = _m5a_repair_was_used(telemetry)
    artifact_references = dict(manifest.get("artifact_references") or {})
    instance_dir = iteration_dir.parent.parent
    context_snapshot = iteration_dir / "context_snapshot.json"
    scenario_snapshot = iteration_dir / "scenario_snapshot.json"
    scenario_validation_snapshot = iteration_dir / "scenario_validation_snapshot.json"
    write_json_atomic(dict(context), context_snapshot)
    write_json_atomic(dict(scenario), scenario_snapshot)
    validation_snapshot_payload = (
        dict(validation_report)
        if isinstance(validation_report, Mapping)
        else {"selected_scenarios": [{"normalized_scenario": dict(scenario)}]}
    )
    write_json_atomic(validation_snapshot_payload, scenario_validation_snapshot)
    artifact_references.update(
        {
            "pass_manifest": str(manifest_path),
            "feature_execution_telemetry": str(iteration_dir / "feature_execution_telemetry.json"),
            "raw_candidate": str(raw_path) if raw_path.exists() else None,
            "postprocessed_candidate": (
                str(postprocessed_path) if postprocessed_path.exists() else None
            ),
            "m6_evidence": _first_existing_reference(
                iteration_dir, ("m6_execution_result.json", "alignment_execution.json")
            ),
            "m7_decision": _first_existing_reference(
                iteration_dir, ("m7_decision_record.json", "alignment_result.json")
            ),
            "clue": _existing_reference(instance_dir / "clue.json"),
            "context": str(context_snapshot),
            "scenario": str(scenario_snapshot),
            "scenario_validation": str(scenario_validation_snapshot),
            "generated_test": _existing_reference(
                iteration_dir / "generated_test.json"
            ),
        }
    )
    if schema_revision not in {"v27", "v29", "v30", "v31", "v36", "v37"}:
        raise ValueError(f"unsupported pass provenance revision: {schema_revision!r}")
    manifest.update(
        {
            "schema_version": f"{schema_revision}-pass-provenance-v1",
            "route_selected_from_previous_pass": previous.get("route_destination"),
            "route_provenance_from_previous_pass": previous.get("route_provenance"),
            "modules_requested_for_current_pass": (
                list(previous.get("modules_requested_for_next_pass") or [])
                if previous
                else list(iteration_record.get("modules_actually_executed_this_pass") or [])
            ),
            "diagnosis_identity_sha256": _canonical_identity_hash(diagnosis),
            "context_identity_sha256": rerun_identities["context_fingerprint"],
            "scenario_identity_sha256": rerun_identities["scenario_fingerprint"],
            "context_snapshot_sha256": _binary_file_sha256(context_snapshot),
            "scenario_snapshot_sha256": _binary_file_sha256(scenario_snapshot),
            "scenario_validation_snapshot_sha256": _binary_file_sha256(
                scenario_validation_snapshot
            ),
            "owner_identity_version": rerun_identities["owner_identity_version"],
            "raw_candidate_identity_sha256": (
                _binary_file_sha256(raw_path)
                if raw_path.exists()
                else None
            ),
            "postprocessed_candidate_identity_sha256": (
                _binary_file_sha256(postprocessed_path)
                if postprocessed_path.exists()
                else rerun_identities["candidate_fingerprint"] or None
            ),
            "repaired_candidate_identity_sha256": (
                rerun_identities["candidate_fingerprint"]
                if repaired and candidate_code
                else None
            ),
            "m6_evidence_identity_sha256": _artifact_identity_hash(
                iteration_dir, ("m6_execution_result.json", "alignment_execution.json")
            ),
            "m7_decision_identity_sha256": _artifact_identity_hash(
                iteration_dir, ("m7_decision_record.json", "alignment_result.json")
            ),
            "requested_next_route": iteration_record.get("route_destination"),
            "requested_next_modules": list(
                iteration_record.get("modules_requested_for_next_pass") or []
            ),
            "no_effect_repair": _m5a_no_effect_repair(telemetry),
            "no_effect_rerun": bool(iteration_record.get("no_effect_rerun")),
            "artifact_references": artifact_references,
        }
    )
    write_json_atomic(manifest, manifest_path)


def _canonical_identity_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _binary_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_mapping_artifact(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _first_existing_reference(iteration_dir: Path, names: tuple[str, ...]) -> str | None:
    for name in names:
        path = iteration_dir / name
        if path.exists():
            return str(path)
    return None


def _existing_reference(path: Path) -> str | None:
    return str(path) if path.exists() else None


def _artifact_identity_hash(iteration_dir: Path, names: tuple[str, ...]) -> str | None:
    for name in names:
        path = iteration_dir / name
        if path.exists():
            return _binary_file_sha256(path)
    return None


def _m5a_records(telemetry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for key, value in telemetry.items():
        if "m5a" not in str(key).lower() and "m5-a" not in str(key).lower():
            continue
        if isinstance(value, Mapping):
            records.append(value)
    return records


def _m5a_repair_was_used(telemetry: Mapping[str, Any]) -> bool:
    for record in _m5a_records(telemetry):
        result = str(record.get("repair_result") or "").upper()
        if result in {"USED_SUCCESS", "SUCCESS", "PARTIAL"}:
            return True
        original = str(record.get("original_candidate_sha256") or "")
        repaired = str(record.get("repaired_candidate_sha256") or "")
        if original and repaired and original != repaired:
            return True
    return False


def _m5a_no_effect_repair(telemetry: Mapping[str, Any]) -> bool:
    return any(
        record.get("no_effect_repair") is True
        or str(record.get("repair_result") or "").upper() == "NO_EFFECT_REPAIR"
        for record in _m5a_records(telemetry)
    )


def _is_no_effect_rerun(
    iteration_history: list[dict[str, Any]],
    snapshot: Mapping[str, str],
) -> bool:
    """Compare the artifact owned by the previous pass's requested change."""
    if not iteration_history:
        return False
    previous_record = iteration_history[-1]
    previous = previous_record.get("rerun_effect") or {}
    diagnosis = previous_record.get("diagnosis") or {}
    owner = str(diagnosis.get("change_owner_module") or "") if isinstance(diagnosis, Mapping) else ""
    if not owner:
        owner = str(previous_record.get("route_destination") or "").split("+")[0]
    if not owner:
        requested = previous_record.get("modules_requested_for_next_pass") or previous_record.get("rerun_targets") or []
        owner = str(requested[0]) if requested else ""
    owner_key = {
        "M2": "context_fingerprint",
        "M3": "scenario_fingerprint",
        "M4": "scenario_fingerprint",
        "M5": "candidate_fingerprint",
        "M5-A": "candidate_fingerprint",
        "M6": "m6_execution_fingerprint",
    }.get(owner)
    if owner_key:
        current_value = snapshot.get(owner_key)
        previous_value = previous.get(owner_key)
        return bool(current_value and previous_value and current_value == previous_value)
    if not all(snapshot.get(key) for key in (
        "context_fingerprint", "scenario_fingerprint", "candidate_fingerprint"
    )):
        return False
    return all(
        previous.get(key) == snapshot.get(key)
        for key in (
            "context_fingerprint",
            "scenario_fingerprint",
            "candidate_fingerprint",
        )
    )


def _m2_source_file_for_exclusion(
    scenario: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None,
) -> str:
    """Return the exact M2-owned source target that a rerun must not repeat."""
    scenario = scenario or {}
    context = context or {}
    target = scenario.get("target_location")
    candidates: list[Any] = []
    if isinstance(target, Mapping):
        candidates.append(target.get("source_file"))
    candidates.extend((scenario.get("source_file"), scenario.get("target_source_file")))
    for key in ("candidate_source_files", "ranked_source_files", "R_init"):
        values = context.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, Mapping):
                candidates.append(
                    value.get("path")
                    or value.get("source_file")
                    or value.get("file_path")
                )
            elif isinstance(value, str):
                candidates.append(value)
    for candidate in candidates:
        normalized = str(candidate or "").replace("\\", "/").lstrip("./")
        if normalized:
            return normalized
    return ""


def _m2_target_for_exclusion(
    scenario: Mapping[str, Any] | None,
    generated_test: Any | None,
) -> dict[str, str] | None:
    """Return the exact target identity rejected by current pre-patch evidence."""
    scenario = scenario or {}
    target = scenario.get("target_location")
    source_file = ""
    target_function = ""
    if isinstance(target, Mapping):
        source_file = str(target.get("source_file") or "")
        target_function = str(target.get("target_function") or "")
    if generated_test is not None:
        source_file = source_file or str(
            getattr(generated_test, "target_source_file", "") or ""
        )
        target_function = target_function or str(
            getattr(generated_test, "selected_implementation_target", "")
            or getattr(generated_test, "selected_issue_api_target", "")
            or getattr(generated_test, "m5_target_used", "")
            or ""
        )
    source_file = source_file.replace("\\", "/").lstrip("./")
    target_function = target_function.strip()
    if not source_file or not target_function:
        return None
    return {"source_file": source_file, "target_function": target_function}


def _no_effect_owner_metadata(
    iteration_history: list[dict[str, Any]],
    snapshot: Mapping[str, str],
) -> dict[str, Any]:
    """Describe which prior owner artifact the no-effect decision examined."""
    if not iteration_history:
        return {
            "expected_change_owner": None,
            "owner_fingerprint_key": None,
            "owner_artifact_available": False,
        }
    previous_record = iteration_history[-1]
    diagnosis = previous_record.get("diagnosis") or {}
    owner = str(diagnosis.get("change_owner_module") or "") if isinstance(diagnosis, Mapping) else ""
    if not owner:
        owner = str(previous_record.get("route_destination") or "").lstrip("→").split("+")[0]
    owner_key = {
        "M2": "context_fingerprint",
        "M3": "scenario_fingerprint",
        "M4": "scenario_fingerprint",
        "M5": "candidate_fingerprint",
        "M5-A": "candidate_fingerprint",
        "M6": "m6_execution_fingerprint",
    }.get(owner)
    return {
        "expected_change_owner": owner or None,
        "owner_fingerprint_key": owner_key,
        "owner_artifact_available": bool(owner_key and snapshot.get(owner_key)),
    }


def _escalated_rerun_targets(rerun_targets: list[str]) -> list[str]:
    ordered = ["M2", "M3", "M5"]
    existing = set(rerun_targets or [])
    if "M3" not in existing:
        existing.add("M3")
    if "M5" not in existing:
        existing.add("M5")
    # Repetition/no-effect is not evidence that localization is wrong.  Keep
    # M2 only when the originating route already assigned ownership there.
    return [target for target in ordered if target in existing]


def _owner_scoped_rerun_targets(
    failure_type: str,
    rerun_targets: list[str],
    score_breakdown: Mapping[str, Any] | None,
    *,
    failure_category: str | None = None,
    execution_error_stage: str | None = None,
) -> list[str]:
    """Remove M2 from retries unless pre-patch evidence challenges identity."""
    ordered = [str(target) for target in rerun_targets]
    breakdown = score_breakdown if isinstance(score_breakdown, Mapping) else {}
    target_evidence = breakdown.get("target_verification_evidence")
    target_evidence = target_evidence if isinstance(target_evidence, Mapping) else {}
    runtime_target = target_evidence.get("runtime_target_coverage")
    runtime_target = runtime_target if isinstance(runtime_target, Mapping) else {}
    reason = str(target_evidence.get("reason") or "")
    explicit_static_disproof = bool(
        target_evidence.get("localization_disproven") is True
        or (
            target_evidence.get("verification_path") == "STATIC_REJECTION"
            and reason == "selected_source_missing_canonical_target"
        )
    )
    # Runtime non-coverage challenges this candidate's call path, not M2's
    # localization. Receiver binding and unavailable spans likewise remain
    # M3/M5 ownership until repository source explicitly contradicts M2.
    target_disproven = explicit_static_disproof
    if target_disproven:
        return ordered
    if failure_type == "ERROR":
        if (
            ordered == ["M6", "M7"]
            or failure_category == "ENVIRONMENT_FAILURE"
            or execution_error_stage == "post_execution_provenance"
        ):
            # A rule-conclusive M6 evidence/environment retry retains the
            # candidate; it is not an early candidate-construction failure.
            return ["M6", "M7"]
        # Construction, fixture, missing-method, and signature errors commonly
        # occur before the target can execute.  Absence of runtime target
        # evidence is not localization disproof; restart at candidate repair.
        return ["M5", "M5-A", "M6", "M7"]
    if "M2" not in ordered:
        return ordered
    scoped = [target for target in ordered if target != "M2"]
    if failure_type == "NO_COVERAGE" and "M3" not in scoped:
        scoped.insert(0, "M3")
    if "M5" not in scoped:
        scoped.append("M5")
    return scoped


def _synchronize_feedback_decision_route(
    decision: M7FeedbackDecision,
    rerun_targets: list[str],
) -> None:
    """Make every persisted owner field match the scoped execution route."""
    targets = [str(target) for target in rerun_targets]
    restart_stage = _first_restart_stage(targets)
    route_stage = restart_stage if restart_stage in {"M2", "M3", "M5", "M8"} else "M5"
    destination = f"→{route_stage}"
    allowed = [
        stage
        for stage in targets
        if stage in {"M2", "M3", "M4", "M5", "M6", "M8"}
    ]
    if restart_stage not in allowed:
        allowed.insert(0, restart_stage)
    decision.modules_requested_for_next = targets
    decision.selected_feedback_branch = destination
    decision.next_start_stage = restart_stage
    decision.route_destination = destination
    decision.allowed_restart_stages = allowed
    decision.selected_restart_stage = restart_stage
    decision.change_owner_module = restart_stage
    decision.final_route = destination


def _m6_timing_breakdown(align_result: Any) -> Dict[str, Any]:
    timing = getattr(align_result, "phase_timings", {}) or {}
    stability = getattr(align_result, "stability_results", {}) or {}
    return {
        "phase_timings": dict(timing) if isinstance(timing, Mapping) else {},
        "stability_timing_breakdown": (
            dict(stability.get("timing_breakdown") or {})
            if isinstance(stability, Mapping)
            else {}
        ),
    }


def _write_iteration_history(
    output_dir: str,
    iteration_history: list[dict[str, Any]],
    *,
    max_feedback_iterations: int = DEFAULT_MAX_FEEDBACK_ITERATIONS,
) -> None:
    write_json(
        {
            "max_alignment_iterations": max_feedback_iterations,
            "max_feedback_iterations": max_feedback_iterations,
            "iterations": iteration_history,
        },
        Path(output_dir) / "iteration_history.json",
    )


def _candidate_evaluation_record(
    *,
    iteration: int,
    generated_test: Any | None,
    align_result: Any | None,
    alignment_result: Any | None,
    validity: str = "VALID",
) -> dict[str, Any]:
    """Serialize the truthful evidence used for best-so-far retention."""
    breakdown = (
        dict(getattr(alignment_result, "score_breakdown", {}) or {})
        if alignment_result is not None
        else {}
    )
    gate_results = breakdown.get("gate_results")
    gate_results = dict(gate_results) if isinstance(gate_results, Mapping) else {}
    target_evidence = breakdown.get("target_verification_evidence")
    target_evidence = dict(target_evidence) if isinstance(target_evidence, Mapping) else {}
    oracle_consistency = breakdown.get("oracle_consistency")
    oracle_consistency = dict(oracle_consistency) if isinstance(oracle_consistency, Mapping) else {}
    execution_status = (
        _execution_status_from_alignment_execution(align_result)
        if align_result is not None
        else "NOT_RUN"
    )
    coverage_available = bool(
        breakdown.get("coverage_score_available")
        or gate_results.get("s_c_prime") is not None
        or breakdown.get("s_c_prime") is not None
    )
    conservative_assessment = breakdown.get("conservative_gate_assessment")
    conservative_pass = bool(
        getattr(alignment_result, "failure_type", "") == "ALIGNED"
        or conservative_assessment in {"PASS", "ALIGNED", "SUPPORTED"}
    )
    oracle_valid = bool(
        oracle_consistency.get("usable_oracle") is True
        and not (breakdown.get("oracle_risk_flags") or [])
    )
    passed_dimensions = {
        "s_b": bool(gate_results.get("gate1_pass")),
        "s_c_prime": bool(gate_results.get("gate2_pass")),
        "s_a": bool(gate_results.get("gate3_pass")),
        "target_verification": bool(breakdown.get("target_verified")),
        "oracle_validity": oracle_valid,
        "conservative_gate": conservative_pass,
    }
    return {
        "schema_version": "v31-candidate-evaluation-v1",
        "iteration": int(iteration),
        "candidate_sha256": str(
            getattr(generated_test, "generated_patch_sha256", "")
            or getattr(generated_test, "patch_sha256", "")
            or ""
        ),
        "test_id": str(getattr(generated_test, "test_id", "") or ""),
        "validity": str(validity or "NOT_VALID"),
        "execution_state": execution_status,
        "s_b": breakdown.get("bug_fail_score"),
        "s_c_prime_available": coverage_available,
        "s_c_prime": breakdown.get("s_c_prime") if coverage_available else None,
        "s_a": breakdown.get("issue_alignment_score"),
        "target_verification": bool(breakdown.get("target_verified")),
        "runtime_verified_target": target_evidence.get("runtime_verified_target"),
        "oracle_validity": oracle_valid,
        "conservative_gate_result": (
            "PASS" if conservative_pass else conservative_assessment or "NOT_PASSED"
        ),
        "m7_decision_status": str(getattr(alignment_result, "failure_type", "") or "NOT_VALID"),
        "admitted_to_final_set": bool(
            getattr(alignment_result, "failure_type", "") == "ALIGNED"
        ),
        "passed_dimensions": passed_dimensions,
    }


def _candidate_dominance_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Rank candidates without weakening or replacing any M7 gate."""
    passed = record.get("passed_dimensions")
    passed = passed if isinstance(passed, Mapping) else {}
    execution_state = str(record.get("execution_state") or "NOT_RUN").upper()
    valid_execution = execution_state not in {"ERROR", "NOT_RUN", "UNKNOWN"}
    score_values = [record.get(key) for key in ("s_b", "s_c_prime", "s_a")]
    score_sum = sum(float(value) for value in score_values if isinstance(value, (int, float)))
    return (
        str(record.get("validity") or "").upper() == "VALID",
        valid_execution,
        bool(record.get("admitted_to_final_set")),
        sum(bool(value) for value in passed.values()),
        bool(record.get("target_verification")),
        bool(record.get("oracle_validity")),
        str(record.get("conservative_gate_result") or "") == "PASS",
        bool(record.get("s_c_prime_available")),
        score_sum,
    )


def _candidate_dominates(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any] | None,
) -> bool:
    """Return true only when ``candidate`` deterministically outranks incumbent."""
    return incumbent is None or _candidate_dominance_key(candidate) > _candidate_dominance_key(incumbent)


def _persist_best_candidate_selection(
    *,
    output_dir: Path,
    best_record: Mapping[str, Any],
    last_iteration: int,
    last_status: str,
) -> None:
    """Persist terminal selection without changing the selected M7 verdict."""
    write_json_atomic(
        {
            "schema_version": "v31-best-candidate-selection-v1",
            "best_candidate": dict(best_record),
            "best_iteration": int(best_record.get("iteration") or 0),
            "last_iteration": int(last_iteration),
            "last_status": str(last_status),
            "last_candidate_regressed": int(best_record.get("iteration") or 0) != int(last_iteration),
            "admission_recomputed": False,
            "admission_bypass": False,
            "selection_rule": "deterministic_candidate_dominance",
        },
        output_dir / "best_candidate_selection.json",
    )


def _repair_contract_from_feedback(
    feedback_decision: M7FeedbackDecision,
    *,
    previous_candidate_code: str,
) -> dict[str, Any]:
    """Translate one feedback decision into a bounded dimension-local repair."""
    status = feedback_decision.m7_decision_status.value
    evidence_text = " ".join(
        str(value)
        for value in (
            feedback_decision.failure_reason,
            feedback_decision.cause_code,
            feedback_decision.cause_hypothesis,
            feedback_decision.concrete_repair_instruction,
        )
        if value
    ).lower()
    if "syntax" in evidence_text:
        dimension = "SYNTAX_IMPORT_FRAMEWORK"
        preserve = ["stimulus", "oracle"]
        modify = ["syntax", "executable_structure"]
    elif "import" in evidence_text:
        dimension = "INVALID_IMPORT"
        # Repository preflight and the unchanged oracle protect behavior here.
        # AST call-shape fingerprints cannot equate ``pkg.call()`` with a
        # corrected ``from pkg import call; call()`` and caused the live hard
        # lock this branch is intended to repair.
        preserve = ["stimulus", "oracle"]
        modify = ["imports", "fully_qualified_references"]
    elif any(token in evidence_text for token in ("framework", "collect", "testcase", "simpletestcase")):
        dimension = "DJANGO_FRAMEWORK_SHAPE"
        preserve = ["stimulus", "oracle"]
        modify = ["test_framework", "class_wrapper", "fixture_access_style"]
    elif status == "ERROR":
        dimension = "EXECUTION_SETUP"
        preserve = ["oracle", "issue_behavior"]
        modify = [
            "setup", "fixture_or_model_construction", "call_signature",
            "imports", "test_framework", "necessary_stimulus_details",
        ]
    elif status == "NOT_VALID" and any(
        token in evidence_text for token in ("no m4", "eligible scenario", "scenario")
    ):
        dimension = "SCENARIO_CONTRACT"
        preserve = ["issue_evidence", "validated_target_when_available"]
        modify = ["scenario", "stimulus", "oracle_contract"]
    elif status == "NO_COVERAGE":
        dimension = "TARGET_REACHING_STIMULUS"
        preserve = ["oracle", "issue_behavior"]
        modify = ["stimulus", "call_path", "setup", "imports", "test_framework"]
    elif status == "NOT_FAILED":
        dimension = "BUG_REPRODUCTION"
        preserve = []
        modify = ["stimulus", "oracle"]
    elif "target" in evidence_text and "oracle" not in evidence_text:
        dimension = "TARGET_EXERCISE"
        preserve = ["valid_oracle"]
        modify = ["stimulus", "call_path"]
    else:
        dimension = "ORACLE_ALIGNMENT"
        preserve = ["stimulus", "canonical_target"]
        modify = ["oracle"]
    preservation_policy = {
        "MUST_PRESERVE_SEMANTICS": list(preserve),
        "MAY_CHANGE_FOR_REPAIR": list(modify),
        "MUST_NOT_CHANGE": [
            "pre_patch_only_evidence_boundary",
            "m7_thresholds",
            "aligned_only_admission",
        ],
    }
    return {
        "schema_version": "v31-dimension-local-repair-v2",
        "mode": "dimension_local",
        "dimension": dimension,
        "preserved_fields": preserve,
        "modified_fields": modify,
        "must_keep": preserve,
        "must_change": modify,
        "blocking_reason": feedback_decision.failure_reason
        or feedback_decision.concrete_repair_instruction
        or status,
        "reason": feedback_decision.concrete_repair_instruction or status,
        "previous_candidate": str(previous_candidate_code or "")[:4000],
        "preservation_fingerprints": candidate_repair_fingerprints(previous_candidate_code),
        "semantic_preservation_fingerprints": candidate_repair_semantics(
            previous_candidate_code
        ),
        "preservation_policy": preservation_policy,
    }


def _write_feature_execution_telemetry(path: Path, telemetry: Mapping[str, Any]) -> None:
    if not telemetry:
        return
    write_json(dict(telemetry), path / "feature_execution_telemetry.json")


def _attempt_generation_failure_m5a_repair(
    *,
    instance: BenchmarkInstance,
    output_dir: Path,
    generator: ReproductionTestGenerator,
    error: GenerationFailureError,
    clue: Dict[str, Any],
    context: Dict[str, Any],
    validation_report: Dict[str, Any],
    feature_flags: V22FeatureFlags,
    iteration: int,
    feature_profile: str | None = None,
) -> tuple[Any | None, Dict[str, Any]]:
    scenario = error.scenario or select_primary_scenario(
        validation_report,
        clue=clue,
        context=context,
    )
    raw_candidate = error.raw_candidate or extract_generated_code(error.parsed_candidate)
    validation_errors = list(error.validation_errors or ([error.last_error] if error.last_error else []))
    evidence = build_repair_evidence(
        error_text=error.last_error or str(error),
        context=context,
        clue=clue,
        scenario=scenario,
        target_test_file=str((error.parsed_candidate or {}).get("target_test_file") or ""),
        target_nodeid="",
        repository_commit=str(getattr(instance, "base_commit", "") or ""),
    )
    quarantine_dir = output_dir / "m5_invalid_candidate"
    persist_quarantine_artifacts(
        quarantine_dir,
        raw_candidate=raw_candidate,
        validation_failures=validation_errors,
        evidence=evidence,
        parsed_candidate=error.parsed_candidate,
        attempt_history=getattr(error, "attempt_history", []),
        raw_response=error.raw_response,
    )

    enabled = bool(feature_flags.enable_m5a_llm_error_refinement)
    eligible = bool(
        raw_candidate
        and is_repairable_category(
            evidence.error_category,
            v30=feature_profile in {"v30", "v31"},
            v36_compile_only=feature_profile in {"v36", "v37"},
        )
    )
    original_hash = hashlib.sha256(raw_candidate.encode("utf-8")).hexdigest()
    error_fp = normalized_error_fingerprint(evidence.error_text)
    if not enabled:
        return None, make_m5a_telemetry(
            enabled=False,
            eligible=eligible,
            triggered=False,
            trigger_reason="feature_disabled",
            attempt_count=0,
            input_error_category=evidence.error_category,
            input_error_fingerprint=error_fp,
            original_candidate_sha256=original_hash,
            repair_result="DISABLED",
            terminal_reason="m5_generation_failure",
        )
    if not eligible:
        return None, make_m5a_telemetry(
            enabled=True,
            eligible=False,
            triggered=False,
            trigger_reason="not_repairable_or_no_candidate",
            attempt_count=0,
            input_error_category=evidence.error_category,
            input_error_fingerprint=error_fp,
            original_candidate_sha256=original_hash,
            repair_result="NOT_ELIGIBLE",
            terminal_reason="m5_generation_failure",
        )

    request = build_error_refinement_request(
        test_code=raw_candidate,
        evidence=evidence,
        clue=clue,
        context=context,
        scenario=scenario,
        v36_compile_only=feature_profile in {"v36", "v37"},
    )
    processed_parsed, deterministic_actions = apply_m5a_deterministic_postprocessing(
        dict(error.parsed_candidate or {
            "append_block": raw_candidate,
            "test_code": raw_candidate,
        }),
        clue=clue,
        repo_path=str(context.get("repo_path") or ""),
        context=context,
        runner=str((context.get("project_test_style") or {}).get("runner") or "pytest"),
        import_checker=generator._check_import_validity if hasattr(generator, "_check_import_validity") else None,
        preserve_test_semantics=feature_profile in {"v36", "v37"},
    )
    processed_code = str(processed_parsed.get("append_block") or processed_parsed.get("test_code") or raw_candidate)
    if processed_code != raw_candidate:
        raw_candidate = processed_code
        request = build_error_refinement_request(
            test_code=processed_code,
            evidence=evidence,
            clue=clue,
            context=context,
            scenario=scenario,
            v36_compile_only=feature_profile in {"v36", "v37"},
        )
    result = refine_m5a_error_with_llm(
        request,
        enabled=True,
        rule_based_error_remaining=True,
        provider=getattr(generator, "m5a_client", generator.client),
        artifact_dir=quarantine_dir,
    )
    repaired_code = result.refined_code
    repaired_hash = hashlib.sha256(repaired_code.encode("utf-8")).hexdigest()
    candidate_unchanged = bool(
        repaired_hash == original_hash
        and (result.used or result.repair_result == "NO_EFFECT_REPAIR")
    )
    parsed = dict(processed_parsed)
    parsed.setdefault("target_test_file", evidence.target_test_file)
    parsed["append_block"] = repaired_code
    parsed["test_code"] = repaired_code
    parsed.setdefault("insert_mode", "append_block")
    parsed.setdefault("imports", [])
    if candidate_unchanged:
        validation = SimpleNamespace(
            is_valid=False,
            errors=["NO_EFFECT_REPAIR: candidate fingerprint unchanged"],
        )
        status = validation_status_from_errors(validation.errors)
    elif result.final_parse_status in {"M5A_INVALID_RESPONSE", "M5A_PROVIDER_ERROR"}:
        validation = SimpleNamespace(is_valid=False, errors=[result.parse_error])
        status = {"syntax": "NOT_RUN", "import": "NOT_RUN", "oracle": "NOT_RUN", "semantic": "NOT_RUN"}
    else:
        validation = generator.validate_repair_candidate(
            parsed=parsed,
            repo_path=str(context.get("repo_path") or ""),
            context=context,
            clue=clue,
            scenario=scenario,
        )
        status = validation_status_from_errors(validation.errors)
    persist_repair_attempt(
        quarantine_dir,
        attempt_index=1,
        strategy="llm",
        input_request=request,
        result=result.to_dict(),
        repaired_code=repaired_code,
        validation_status={"is_valid": validation.is_valid, "errors": validation.errors, **status},
    )
    post_failure_fingerprint = normalized_error_fingerprint("\n".join(validation.errors))
    same_failure = bool(
        not validation.is_valid
        and post_failure_fingerprint
        and post_failure_fingerprint == error_fp
    )
    no_effect_repair = candidate_unchanged or same_failure
    if not result.used or not validation.is_valid:
        return None, make_m5a_telemetry(
            enabled=True,
            eligible=True,
            triggered=True,
            trigger_reason="m5_generation_static_validation_failure",
            attempt_count=result.llm_call_count,
            input_error_category=evidence.error_category,
            input_error_fingerprint=error_fp,
            original_candidate_sha256=original_hash,
            repaired_candidate_sha256=repaired_hash,
            repair_result="NO_EFFECT_REPAIR" if no_effect_repair else "USED_FAILED",
            post_repair_syntax_status=status["syntax"],
            post_repair_import_status=status["import"],
            post_repair_oracle_status=status["oracle"],
            post_repair_collection_status="NOT_RUN",
            post_repair_execution_status="NOT_RUN",
            terminal_reason=(
                "NO_EFFECT_REPAIR"
                if no_effect_repair
                else result.final_parse_status
                if result.final_parse_status in {"M5A_INVALID_RESPONSE", "M5A_PROVIDER_ERROR"}
                else "post_repair_static_validation_failed"
            ),
            raw_response_artifact=result.raw_response_artifact,
            response_empty=result.response_empty,
            parse_attempt_count=result.parse_attempt_count,
            parse_error=result.parse_error,
            retry_triggered=result.retry_triggered,
            retry_prompt_hash=result.retry_prompt_hash,
            final_parse_status=result.final_parse_status,
            post_repair_validation_errors=list(validation.errors),
            post_repair_failure_fingerprint=post_failure_fingerprint,
            no_effect_repair=no_effect_repair,
        )
    repaired_test = generator.build_repaired_generated_test(
        instance=instance,
        original=None,
        parsed=parsed,
        repaired_code=repaired_code,
        clue=clue,
        context=context,
        scenario=scenario,
        iteration=iteration,
        prompt=error.prompt,
        raw_response="",
        llm_error_refinement=result.to_dict(),
        generation_attempt_count=error.attempt_count + result.llm_call_count,
        token_usage=error.token_usage,
        token_usage_status=error.token_usage_status,
    )
    return repaired_test, make_m5a_telemetry(
        enabled=True,
        eligible=True,
        triggered=True,
        trigger_reason="m5_generation_static_validation_failure",
        attempt_count=result.llm_call_count,
        input_error_category=evidence.error_category,
        input_error_fingerprint=error_fp,
        original_candidate_sha256=original_hash,
        repaired_candidate_sha256=repaired_hash,
        repair_result="USED_SUCCESS",
        post_repair_syntax_status="PASS",
        post_repair_import_status="PASS",
        post_repair_oracle_status="PASS",
        post_repair_collection_status="PENDING",
        post_repair_execution_status="PENDING",
        terminal_reason="promoted_after_static_validation",
        raw_response_artifact=result.raw_response_artifact,
        response_empty=result.response_empty,
        parse_attempt_count=result.parse_attempt_count,
        parse_error=result.parse_error,
        retry_triggered=result.retry_triggered,
        retry_prompt_hash=result.retry_prompt_hash,
        final_parse_status=result.final_parse_status,
    )


def _attempt_m6_m5a_repair(
    *,
    instance: BenchmarkInstance,
    pre_patch_view: Any,
    output_dir: Path,
    generator: ReproductionTestGenerator,
    alignment_runner: AlignmentRunner,
    original_generated_test: Any,
    original_generated_test_path: str,
    align_result: Any,
    clue: Dict[str, Any],
    context: Dict[str, Any],
    validation_report: Dict[str, Any],
    feature_flags: V22FeatureFlags,
    iteration: int,
    feature_profile: str | None = None,
    prior_m5a_attempt_count: int = 0,
) -> tuple[Any, Any, Any, Dict[str, Any]] | None:
    if not (align_result.has_error or align_result.error_messages):
        return None
    scenario = select_primary_scenario(validation_report, clue=clue, context=context)
    error_text = "\n".join(str(item) for item in (align_result.error_messages or []))
    raw_output = str(getattr(align_result, "raw_output", "") or "")
    target_nodeid = (
        getattr(align_result, "canonical_test_nodeid", None)
        or getattr(align_result, "test_nodeid", None)
        or getattr(original_generated_test, "canonical_test_nodeid", "")
    )
    pytest_command = _pytest_command_for_repair(original_generated_test, target_nodeid)
    evidence = build_repair_evidence(
        error_text=error_text,
        context=context,
        clue=clue,
        scenario=scenario,
        target_test_file=getattr(original_generated_test, "target_test_file", ""),
        target_nodeid=str(target_nodeid or ""),
        repository_commit=str(getattr(instance, "base_commit", "") or ""),
        pytest_command=pytest_command,
        raw_output=raw_output,
    )
    code = str(getattr(original_generated_test, "test_code", "") or "")
    # A2: deterministic post-processing is mandatory and its action list must
    # exist on every branch, including disabled, duplicate, and failed paths.
    deterministic_actions: list[dict[str, Any]] = []
    original_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    processed_parsed, deterministic_actions = apply_m5a_deterministic_postprocessing(
        {
            "target_test_file": getattr(original_generated_test, "target_test_file", ""),
            "insert_mode": getattr(original_generated_test, "insert_mode", "append_block"),
            "insertion_hint": getattr(original_generated_test, "insertion_hint", "append to file"),
            "append_block": code,
            "test_code": code,
            "imports": list(getattr(original_generated_test, "imports", []) or []),
        },
        clue=clue,
        repo_path=str(context.get("repo_path") or ""),
        context=context,
        runner=str((context.get("project_test_style") or {}).get("runner") or "pytest"),
        import_checker=generator._check_import_validity if hasattr(generator, "_check_import_validity") else None,
        preserve_test_semantics=feature_profile in {"v36", "v37"},
    )
    processed_code = str(processed_parsed.get("append_block") or processed_parsed.get("test_code") or code)
    if processed_code != code:
        code = processed_code
        original_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    error_fp = normalized_error_fingerprint(evidence.error_text)
    enabled = bool(feature_flags.enable_m5a_llm_error_refinement)
    eligible = bool(code and is_repairable_category(
        evidence.error_category,
        v30=feature_profile in {"v30", "v31"},
        v36_compile_only=feature_profile in {"v36", "v37"},
    ))
    repair_dir = output_dir / "m5a_container_repair"
    persist_quarantine_artifacts(
        repair_dir,
        raw_candidate=code,
        validation_failures=[error_text],
        evidence=evidence,
        parsed_candidate={
            "target_test_file": getattr(original_generated_test, "target_test_file", ""),
            "insert_mode": getattr(original_generated_test, "insert_mode", "append_block"),
            "append_block": code,
            "test_code": code,
            "imports": getattr(original_generated_test, "imports", []),
        },
    )
    if prior_m5a_attempt_count >= 1:
        logger.info(
            "Skipping second M5-A repair in pass %s; the pass already used %s repair call(s)",
            iteration,
            prior_m5a_attempt_count,
        )
        return None
    if not enabled or not eligible:
        telemetry = _attach_m5a_deterministic_actions(make_m5a_telemetry(
            enabled=enabled,
            eligible=eligible,
            triggered=False,
            trigger_reason="feature_disabled" if not enabled else "not_repairable_or_no_candidate",
            attempt_count=0,
            input_error_category=evidence.error_category,
            input_error_fingerprint=error_fp,
            original_candidate_sha256=original_hash,
            repair_result="DISABLED" if not enabled else "NOT_ELIGIBLE",
            post_repair_collection_status="NOT_RUN",
            post_repair_execution_status="NOT_RUN",
            terminal_reason="m6_error_not_repaired",
        ), deterministic_actions)
        _write_feature_execution_telemetry(output_dir, telemetry)
        return None

    fingerprint = repair_fingerprint(
        candidate_code=code,
        error_text=evidence.error_text,
        target_nodeid=str(target_nodeid or ""),
        repository_commit=str(getattr(instance, "base_commit", "") or ""),
        repair_strategy="llm",
    )
    duplicate_marker = repair_dir / f"{fingerprint}.seen"
    if duplicate_marker.exists():
        telemetry = _attach_m5a_deterministic_actions(make_m5a_telemetry(
            enabled=True,
            eligible=True,
            triggered=False,
            trigger_reason="duplicate_candidate_error_strategy",
            attempt_count=0,
            input_error_category=evidence.error_category,
            input_error_fingerprint=error_fp,
            original_candidate_sha256=original_hash,
            repair_result="DUPLICATE_BLOCKED",
            terminal_reason="duplicate_blocked",
        ), deterministic_actions)
        _write_feature_execution_telemetry(output_dir, telemetry)
        return None
    duplicate_marker.parent.mkdir(parents=True, exist_ok=True)
    duplicate_marker.write_text(fingerprint, encoding="utf-8")

    request = build_error_refinement_request(
        test_code=code,
        evidence=evidence,
        clue=clue,
        context=context,
        scenario=scenario,
        v36_compile_only=feature_profile in {"v36", "v37"},
    )
    result = refine_m5a_error_with_llm(
        request,
        enabled=True,
        rule_based_error_remaining=True,
        provider=getattr(generator, "m5a_client", generator.client),
        artifact_dir=repair_dir,
    )
    repaired_code = result.refined_code
    repaired_hash = hashlib.sha256(repaired_code.encode("utf-8")).hexdigest()
    candidate_unchanged = bool(
        repaired_hash == original_hash
        and (result.used or result.repair_result == "NO_EFFECT_REPAIR")
    )
    parsed = dict(processed_parsed)
    parsed["append_block"] = repaired_code
    parsed["test_code"] = repaired_code
    if candidate_unchanged:
        validation = SimpleNamespace(
            is_valid=False,
            errors=["NO_EFFECT_REPAIR: candidate fingerprint unchanged"],
        )
        status = validation_status_from_errors(validation.errors)
    elif result.final_parse_status in {"M5A_INVALID_RESPONSE", "M5A_PROVIDER_ERROR"}:
        validation = SimpleNamespace(is_valid=False, errors=[result.parse_error])
        status = {"syntax": "NOT_RUN", "import": "NOT_RUN", "oracle": "NOT_RUN", "semantic": "NOT_RUN"}
    else:
        validation = generator.validate_repair_candidate(
            parsed=parsed,
            repo_path=str(context.get("repo_path") or ""),
            context=context,
            clue=clue,
            scenario=scenario,
        )
        status = validation_status_from_errors(validation.errors)
    persist_repair_attempt(
        repair_dir,
        attempt_index=1,
        strategy="llm",
        input_request=request,
        result=result.to_dict(),
        repaired_code=repaired_code,
        validation_status={"is_valid": validation.is_valid, "errors": validation.errors, **status},
    )
    post_failure_fingerprint = normalized_error_fingerprint("\n".join(validation.errors))
    same_failure = bool(
        not validation.is_valid
        and post_failure_fingerprint
        and post_failure_fingerprint == error_fp
    )
    no_effect_repair = candidate_unchanged or same_failure
    if not result.used or not validation.is_valid:
        telemetry = _attach_m5a_deterministic_actions(make_m5a_telemetry(
            enabled=True,
            eligible=True,
            triggered=True,
            trigger_reason="m6_container_repairable_error",
            attempt_count=result.llm_call_count,
            input_error_category=evidence.error_category,
            input_error_fingerprint=error_fp,
            original_candidate_sha256=original_hash,
            repaired_candidate_sha256=repaired_hash,
            repair_result="NO_EFFECT_REPAIR" if no_effect_repair else "USED_FAILED",
            post_repair_syntax_status=status["syntax"],
            post_repair_import_status=status["import"],
            post_repair_oracle_status=status["oracle"],
            post_repair_collection_status="NOT_RUN",
            post_repair_execution_status="NOT_RUN",
            terminal_reason=(
                "NO_EFFECT_REPAIR"
                if no_effect_repair
                else result.final_parse_status
                if result.final_parse_status in {"M5A_INVALID_RESPONSE", "M5A_PROVIDER_ERROR"}
                else "post_repair_static_validation_failed"
            ),
            raw_response_artifact=result.raw_response_artifact,
            response_empty=result.response_empty,
            parse_attempt_count=result.parse_attempt_count,
            parse_error=result.parse_error,
            retry_triggered=result.retry_triggered,
            retry_prompt_hash=result.retry_prompt_hash,
            final_parse_status=result.final_parse_status,
            post_repair_validation_errors=list(validation.errors),
            post_repair_failure_fingerprint=post_failure_fingerprint,
            no_effect_repair=no_effect_repair,
        ), deterministic_actions)
        _write_feature_execution_telemetry(output_dir, telemetry)
        return None

    repaired_test = generator.build_repaired_generated_test(
        instance=instance,
        original=original_generated_test,
        parsed=parsed,
        repaired_code=repaired_code,
        clue=clue,
        context=context,
        scenario=scenario,
        iteration=iteration,
        prompt=getattr(original_generated_test, "prompt", ""),
        raw_response=getattr(original_generated_test, "raw_response", ""),
        llm_error_refinement=result.to_dict(),
        generation_attempt_count=getattr(original_generated_test, "generation_attempt_count", 0) + result.llm_call_count,
        token_usage=getattr(original_generated_test, "token_usage", {}),
        token_usage_status=getattr(original_generated_test, "token_usage_status", "known"),
    )
    repaired_path = output_dir / "generated_test.repaired.json"
    generator.save(repaired_test, str(repaired_path))
    final_path = Path(original_generated_test_path)
    shutil.copyfile(repaired_path, final_path)
    shutil.copyfile(repaired_path.with_suffix(".patch"), final_path.with_suffix(".patch"))
    rendered_source = repaired_path.with_name(repaired_path.stem + "_rendered.py")
    rendered_target = final_path.with_name(final_path.stem + "_rendered.py")
    if rendered_source.exists():
        shutil.copyfile(rendered_source, rendered_target)
    repaired_align = alignment_runner.run(
        instance=pre_patch_view,
        generated_test_json_path=str(final_path),
        run_id=f"align-{instance.instance_id}-it{iteration}-m5a-repair-{int(time.time() * 1000)}",
        iteration=iteration,
        feature_flags=feature_flags,
        supplemental_context=context,
        supplemental_clue=clue,
    )
    repaired_artifacts = alignment_runner.save(
        repaired_align,
        str(output_dir / "alignment_execution.json"),
        feature_flags=feature_flags,
    )
    execution_status = normalize_pre_patch_execution_status(repaired_align).value
    metric_execution = execution_status in {"PASS", "FAIL"}
    collection_status = "COLLECTED" if repaired_align.test_results and metric_execution else "ERROR_NOT_COLLECTED"
    telemetry = _attach_m5a_deterministic_actions(make_m5a_telemetry(
        enabled=True,
        eligible=True,
        triggered=True,
        trigger_reason="m6_container_repairable_error",
        attempt_count=result.llm_call_count,
        input_error_category=evidence.error_category,
        input_error_fingerprint=error_fp,
        original_candidate_sha256=original_hash,
        repaired_candidate_sha256=repaired_hash,
        repair_result=(
            "NO_EFFECT_REPAIR"
            if normalized_error_fingerprint(
                "\n".join(str(item) for item in (repaired_align.error_messages or []))
            ) == error_fp
            and bool(repaired_align.error_messages)
            else "USED_SUCCESS"
            if metric_execution and bool(repaired_align.test_results) and not repaired_align.has_error
            else "USED_FAILED"
        ),
        post_repair_syntax_status="PASS",
        post_repair_import_status="PASS",
        post_repair_oracle_status="PASS",
        post_repair_collection_status=collection_status,
        post_repair_execution_status=execution_status,
        terminal_reason="reran_collection_execution_after_repair",
        raw_response_artifact=result.raw_response_artifact,
        response_empty=result.response_empty,
        parse_attempt_count=result.parse_attempt_count,
        parse_error=result.parse_error,
        retry_triggered=result.retry_triggered,
        retry_prompt_hash=result.retry_prompt_hash,
        final_parse_status=result.final_parse_status,
        post_repair_validation_errors=list(repaired_align.error_messages or []),
        post_repair_failure_fingerprint=normalized_error_fingerprint(
            "\n".join(str(item) for item in (repaired_align.error_messages or []))
        ),
        no_effect_repair=(
            bool(repaired_align.error_messages)
            and normalized_error_fingerprint(
                "\n".join(str(item) for item in (repaired_align.error_messages or []))
            ) == error_fp
        ),
    ), deterministic_actions)
    _write_feature_execution_telemetry(output_dir, telemetry)
    return repaired_test, repaired_align, repaired_artifacts, telemetry


def _pytest_command_for_repair(generated_test: Any, target_nodeid: Any) -> str:
    nodeid = str(target_nodeid or getattr(generated_test, "canonical_test_nodeid", "") or "")
    if nodeid:
        return f"pytest -q {nodeid}"
    target_file = str(getattr(generated_test, "target_test_file", "") or "")
    return f"pytest -q {target_file}" if target_file else "pytest -q"


def _safe_primary_scenario(validation_report: Mapping[str, Any]) -> Dict[str, Any]:
    for item in validation_report.get("selected_scenarios", []) or []:
        if isinstance(item, Mapping) and isinstance(item.get("normalized_scenario"), Mapping):
            return dict(item["normalized_scenario"])
    return {}


def _generation_failure_repair_branch(failure_type_detail: str, *, repeated: bool) -> str:
    if failure_type_detail == "M5_INPUT_CONTRACT":
        return "M3"
    if repeated:
        return "M3+M5"
    if failure_type_detail in {"SYNTAX_ERROR", "IMPORT_ERROR", "MISSING_TEST_FILE"}:
        return "M5"
    if failure_type_detail in {"ORACLE_REJECTED", "SEMANTIC_RISK"}:
        return "M3+M5"
    return "M5"


def _runtime_hint_for_generation_failure(
    diagnosis: str,
    failure_type_detail: str,
    *,
    repeated: bool,
) -> str:
    hint = (
        f"Previous candidate was NOT_VALID ({failure_type_detail}). "
        f"Repair the specific cause before producing the next candidate: {diagnosis}"
    )
    if repeated:
        hint += (
            " The same semantic fingerprint repeated; use a different scenario, "
            "target test file, stimulus, and oracle path."
        )
    return hint


def _recoverable_iteration_record(
    *,
    iteration: int,
    selected_records: list[dict[str, Any]],
    failure_type: str,
    failure_type_detail: str,
    feedback_branch: str,
    rerun_targets: list[str],
    history_window: int | None,
    token_usage_status: str,
    m5_elapsed_sec: float | None,
    semantic_fingerprint: str,
    repeated_semantic_fingerprint: bool,
    max_feedback_iterations: int = DEFAULT_MAX_FEEDBACK_ITERATIONS,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "selected_scenarios": selected_records,
        "generated_scenario_id": None,
        "m7_decision_status": failure_type,
        "m7_alignment_status": None if failure_type in {"NOT_VALID", "ERROR"} else failure_type,
        "failure_type": failure_type,
        "failure_type_detail": failure_type_detail,
        "feedback_branch": feedback_branch,
        "rerun_targets": rerun_targets,
        "loop_terminated": iteration >= max_feedback_iterations,
        "history_window": history_window,
        "generation_attempt_count": None,
        "initial_generation_attempts": None,
        "validation_retry_count": None,
        "deterministic_repair_attempts": None,
        "llm_repair_attempts": None,
        "container_execution_attempts": 0,
        "alignment_iterations": iteration,
        "repeated_validation_early_stop": False,
        "repeated_validation_fingerprint": None,
        "runtime_error_fingerprint": semantic_fingerprint,
        "repeated_runtime_error_early_stop": False,
        "semantic_progress_fingerprint": semantic_fingerprint,
        "repeated_semantic_fingerprint": repeated_semantic_fingerprint,
        "semantic_escalation_required": repeated_semantic_fingerprint,
        "no_progress_termination": False,
        "llm_call_count": None,
        "m3_elapsed_sec": None,
        "m5_elapsed_sec": m5_elapsed_sec,
        "m6_execution_coverage_elapsed_sec": None,
        "m6_timing_breakdown": {},
        "m7_elapsed_sec": None,
        "final_harness_elapsed_sec": None,
        "m8_elapsed_sec": None,
        "token_usage_status": token_usage_status,
        "outer_iteration_policy": "unified_candidate_lifecycle",
    }


def _write_recoverable_iteration_result(
    *,
    output_dir: str,
    instance_id: str,
    iteration: int,
    failure_type: str,
    failure_type_detail: str,
    diagnosis: str,
    refined_scenario: Dict[str, Any],
    should_continue: bool,
    repair_branch: str,
    repeated_semantic_fingerprint: bool,
    semantic_fingerprint: str,
    feature_execution_telemetry: Mapping[str, Any] | None = None,
    feedback_client: Any = None,
    diagnosis_revision: str = "v26",
    max_feedback_iterations: int = DEFAULT_MAX_FEEDBACK_ITERATIONS,
    termination_reason: str | None = None,
) -> M7FeedbackDecision:
    input_contract_owned_by_m4 = failure_type_detail == "M5_INPUT_CONTRACT"
    source_stage = "M4" if input_contract_owned_by_m4 else (
        "M5" if failure_type == "NOT_VALID" else "UNKNOWN"
    )
    payload = {
        "instance_id": instance_id,
        "iteration": iteration,
        "iterations": iteration,
        "failure_type": failure_type,
        **_status_fields(failure_type),
        "m7_decision_status": failure_type,
        "source_stage": source_stage,
        "evaluation_performed": False,
        "score_breakdown": {
            "bug_fail_score": 0.0,
            "issue_alignment_score": 0.0,
            "coverage_score": 0.0,
            "failure_type_detail": failure_type_detail,
        },
        "diagnosis": diagnosis,
        "feedback": {
            "repair_branch": repair_branch,
            "target_modules": ["M5"] if repair_branch == "M5" else repair_branch.split("+"),
            "cause_specific_repair": failure_type_detail,
            "prohibit_candidate_fingerprint": semantic_fingerprint,
            "escalation_required": repeated_semantic_fingerprint,
        },
        "refined_scenario": refined_scenario,
        "should_continue": should_continue,
        "termination_reason": (
            "continue"
            if should_continue
            else termination_reason or "iteration_budget_exhausted"
        ),
        "test_results": {},
        "coverage_summary": {},
        "failure_type_detail": failure_type_detail,
        "recoverable": True,
        "outer_iteration_policy": "unified_candidate_lifecycle",
        "semantic_progress_fingerprint": semantic_fingerprint,
        "repeated_semantic_fingerprint": repeated_semantic_fingerprint,
    }
    if feature_execution_telemetry:
        payload["feature_execution_telemetry"] = dict(feature_execution_telemetry)
    iteration_dir = _iteration_artifact_dir(output_dir, iteration)
    write_json(payload, iteration_dir / "alignment_result.json")
    evidence = _normalize_m7_evidence(
        {
            "diagnosis": diagnosis,
            "failure_type_detail": failure_type_detail,
            "semantic_progress_fingerprint": semantic_fingerprint,
            "repeated_semantic_fingerprint": repeated_semantic_fingerprint,
            "coverage_evidence": {},
        },
        remaining_outer_iterations=max(0, max_feedback_iterations - iteration),
    )
    rerun_targets = ["M5"] if repair_branch == "M5" else repair_branch.split("+")
    if input_contract_owned_by_m4:
        rerun_targets = ["M3", "M4", "M5", "M5-A", "M6", "M7"]
        feedback_decision = _deterministic_m7_feedback_decision(
            decision=M7DecisionStatus.NOT_VALID,
            iteration=iteration,
            source_stage="M4",
            failure_category="GENERATION_FAILURE",
            evidence=evidence,
            feedback_branch="→M3",
            next_start_stage="M3",
            rerun_targets=rerun_targets,
            prohibited_fingerprints=(
                [semantic_fingerprint] if semantic_fingerprint else []
            ),
            fallback_reason="rule_conclusive_m5_input_contract",
            max_feedback_iterations=max_feedback_iterations,
        )
        feedback_decision.selected_feedback_branch = "→M3"
        feedback_decision.next_start_stage = "M3"
        feedback_decision.route_destination = "→M3"
        feedback_decision.modules_requested_for_next = list(rerun_targets)
        feedback_decision.allowed_restart_stages = ["M3", "M4", "M5"]
        feedback_decision.selected_restart_stage = "M3"
        feedback_decision.change_owner_module = "M3"
        feedback_decision.feedback_provenance = "RULE_CONCLUSIVE"
        feedback_decision.cause_source = "RULE"
        feedback_decision.cause_code = "M5_INPUT_CONTRACT_M4_OWNED"
        feedback_decision.deterministic_fallback_used = False
        feedback_decision.llm_attempted = False
        feedback_decision.llm_succeeded = False
        feedback_decision.final_route = "→M3"
    else:
        feedback_decision = _build_m7_feedback_decision(
            client=feedback_client,
            decision_status=failure_type,
            iteration=iteration,
            source_stage=source_stage,
            failure_category=(
                "GENERATION_FAILURE"
                if failure_type == "NOT_VALID"
                else "PIPELINE_FAILURE"
            ),
            evidence=evidence,
            feedback_branch=repair_branch,
            next_start_stage=_first_restart_stage(rerun_targets),
            rerun_targets=rerun_targets,
            prohibited_fingerprints=(
                [semantic_fingerprint] if semantic_fingerprint else []
            ),
            diagnosis_revision=diagnosis_revision,
            max_feedback_iterations=max_feedback_iterations,
        )
    record = _build_m7_decision_record(
        instance_id=instance_id,
        iteration=iteration,
        decision_status=failure_type,
        source_stage=source_stage,
        validation_status="INVALID" if failure_type == "NOT_VALID" else "NOT_RUN",
        execution_status="NOT_RUN",
        failure_category="GENERATION_FAILURE" if failure_type == "NOT_VALID" else "PIPELINE_FAILURE",
        evidence=evidence,
        feedback_branch=repair_branch,
        next_start_stage=_first_restart_stage(rerun_targets),
        rerun_targets=rerun_targets,
        should_continue=should_continue,
        termination_reason=(
            "continue"
            if should_continue
            else termination_reason or "iteration_budget_exhausted"
        ),
        prohibited_fingerprints=[semantic_fingerprint] if semantic_fingerprint else [],
        feedback_decision=feedback_decision,
        max_feedback_iterations=max_feedback_iterations,
    )
    _write_m7_decision_artifacts(iteration_dir, record)
    if feature_execution_telemetry:
        _write_feature_execution_telemetry(iteration_dir, feature_execution_telemetry)
        _write_feature_execution_telemetry(Path(output_dir), feature_execution_telemetry)
    _sync_latest_iteration_aliases(
        output_dir, iteration_dir, terminal=not should_continue
    )
    return feedback_decision


def _write_terminal_alignment_result(
    output_dir: str,
    instance_id: str,
    iteration: int,
    failure_type: str,
    failure_type_detail: str,
    diagnosis: str,
    refined_scenario: Dict[str, Any],
    test_results: Dict[str, str] | None = None,
    coverage_summary: Dict[str, Any] | None = None,
    feature_execution_telemetry: Mapping[str, Any] | None = None,
    termination_reason: str | None = None,
) -> None:
    """Persist a terminal alignment_result.json for early-exit failures."""
    failure = {
        "instance_id": instance_id,
        "iteration": iteration,
        "iterations": iteration,
        "failure_type": failure_type,
        **_status_fields(failure_type),
        "score_breakdown": {
            "bug_fail_score": 0.0,
            "issue_alignment_score": 0.0,
            "coverage_score": 0.0,
            "failure_type_detail": failure_type_detail,
        },
        "diagnosis": diagnosis,
        "feedback": {},
        "refined_scenario": refined_scenario,
        "should_continue": False,
        "termination_reason": termination_reason or "terminal_failure",
        "test_results": test_results or {},
        "coverage_summary": coverage_summary or {},
        "failure_type_detail": failure_type_detail,
    }
    if feature_execution_telemetry:
        failure["feature_execution_telemetry"] = dict(feature_execution_telemetry)
    iteration_dir = _iteration_artifact_dir(output_dir, iteration)
    write_json(failure, iteration_dir / "alignment_result.json")
    if feature_execution_telemetry:
        _write_feature_execution_telemetry(iteration_dir, feature_execution_telemetry)
        _write_feature_execution_telemetry(Path(output_dir), feature_execution_telemetry)
    _sync_latest_iteration_aliases(output_dir, iteration_dir)


def _build_m7_decision_record(
    *,
    instance_id: str,
    iteration: int,
    decision_status: str,
    source_stage: str,
    validation_status: str,
    execution_status: str,
    failure_category: str,
    evidence: Mapping[str, Any],
    feedback_branch: str,
    next_start_stage: str,
    rerun_targets: list[str],
    should_continue: bool,
    termination_reason: str,
    prohibited_fingerprints: list[str] | None = None,
    feedback_decision: M7FeedbackDecision | None = None,
    max_feedback_iterations: int = DEFAULT_MAX_FEEDBACK_ITERATIONS,
) -> M7DecisionRecord:
    decision = coerce_m7_decision_status(decision_status)
    if decision is None:
        decision = M7DecisionStatus.ERROR
    remaining = max(0, max_feedback_iterations - iteration)
    normalized_next = next_start_stage if next_start_stage in {"M2", "M3", "M4", "M5", "M6", "M8"} else "M5"
    if decision == M7DecisionStatus.ALIGNED:
        normalized_next = "M8"
    feedback = feedback_decision or _deterministic_m7_feedback_decision(
        decision=decision,
        iteration=iteration,
        source_stage=source_stage,
        failure_category=failure_category,
        evidence=evidence,
        feedback_branch=feedback_branch,
        next_start_stage=normalized_next,
        rerun_targets=rerun_targets,
        prohibited_fingerprints=list(prohibited_fingerprints or []),
        fallback_reason="model_invocation_failure: feedback_client_unavailable",
        max_feedback_iterations=max_feedback_iterations,
    )
    return M7DecisionRecord(
        instance_id=instance_id,
        outer_iteration=iteration,
        m7_decision_status=decision,
        source_stage=source_stage,
        validation_status=validation_status,
        execution_status=execution_status,
        failure_category=failure_category,
        evaluation_evidence=dict(evidence),
        feedback_decision=feedback,
        previous_iteration_reference={
            "previous_iteration": iteration - 1 if iteration > 1 else None,
            "rerun_targets": list(rerun_targets or []),
        },
        remaining_outer_iterations=remaining,
        loop_terminated=not should_continue,
        termination_reason=termination_reason,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _m7_fact(value: Any, *, unknown_reason: str, sources: list[str], unavailable: list[str] | None = None) -> dict[str, Any]:
    if isinstance(value, Mapping) and "value" in value:
        label = str(value.get("value") or "UNKNOWN").upper()
        if label in {"TRUE", "FALSE", "UNKNOWN"}:
            result = dict(value)
            result["value"] = label
            if label == "UNKNOWN":
                result.setdefault("unknown_reason", unknown_reason)
                result.setdefault("evidence_sources_checked", list(sources))
                result.setdefault("unavailable_artifacts_or_signals", list(unavailable or []))
            return result
    if value is True:
        return {"value": "TRUE", "evidence_sources_checked": list(sources)}
    if value is False:
        return {"value": "FALSE", "evidence_sources_checked": list(sources)}
    return {
        "value": "UNKNOWN",
        "unknown_reason": unknown_reason,
        "evidence_sources_checked": list(sources),
        "unavailable_artifacts_or_signals": list(unavailable or []),
    }


def _legacy_fact_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        label = str(value.get("value") or "UNKNOWN").upper()
        if label == "TRUE":
            return True
        if label == "FALSE":
            return False
        return "UNKNOWN"
    return value


def _normalize_m7_evidence(
    evidence: Mapping[str, Any],
    *,
    remaining_outer_iterations: int,
) -> dict[str, Any]:
    normalized = dict(evidence)
    default_sources = [
        "alignment_execution.json",
        "m6_execution_result.json",
        "coverage_result.json",
        "alignment_result.json",
    ]
    normalized["issue_api_executed"] = _m7_fact(
        normalized.get("issue_api_executed"),
        unknown_reason="issue API execution signal is not directly available in current M6 artifacts",
        sources=default_sources,
        unavailable=["issue_api_probe"],
    )
    normalized["suspected_file_covered"] = _m7_fact(
        normalized.get("suspected_file_covered"),
        unknown_reason="file coverage signal is unavailable without parsed coverage data",
        sources=["coverage_result.json"],
        unavailable=["coverage_result.json"] if "suspected_file_covered" not in normalized else [],
    )
    normalized["assertion_executed"] = _m7_fact(
        normalized.get("assertion_executed"),
        unknown_reason="assertion execution is inferred only when parsed test results expose assertion-level evidence",
        sources=["alignment_execution.json"],
        unavailable=["assertion_trace"],
    )
    normalized["actual_output_observed"] = _m7_fact(
        normalized.get("actual_output_observed"),
        unknown_reason="actual output observation is unavailable unless execution output exposes issue-specific values",
        sources=["alignment_execution.json"],
        unavailable=["issue_output_probe"],
    )
    normalized["exception_observed"] = _m7_fact(
        normalized.get("exception_observed"),
        unknown_reason="exception observation is unavailable unless parsed error messages identify it",
        sources=["alignment_execution.json"],
        unavailable=[],
    )
    normalized["suspected_function_covered"] = _m7_fact(
        normalized.get("suspected_function_covered"),
        unknown_reason="function-level coverage is unavailable without resolved suspected function spectra",
        sources=["coverage_result.json"],
        unavailable=["function_coverage_signal"],
    )
    normalized["suspected_lines_covered"] = _m7_fact(
        normalized.get("suspected_lines_covered"),
        unknown_reason="line-level suspected-location coverage is unavailable without resolved issue lines",
        sources=["coverage_result.json"],
        unavailable=["issue_line_mapping"],
    )
    normalized["issue_branch_reached"] = _m7_fact(
        normalized.get("issue_branch_reached"),
        unknown_reason="branch/state evidence is unavailable in current M6 artifacts",
        sources=default_sources,
        unavailable=["branch_probe", "state_probe"],
    )
    normalized["incorrect_behavior_observed"] = _m7_fact(
        normalized.get("incorrect_behavior_observed"),
        unknown_reason="incorrect behavior observation is unavailable without issue-specific output/state probe",
        sources=default_sources,
        unavailable=["behavior_probe"],
    )
    normalized["oracle_checked_behavior"] = _m7_fact(
        normalized.get("oracle_checked_behavior"),
        unknown_reason="oracle behavior check cannot be proven from current parsed outcome alone",
        sources=default_sources,
        unavailable=["oracle_trace"],
    )
    normalized["remaining_outer_iterations"] = remaining_outer_iterations
    normalized.setdefault("attempted_evidence_sources", default_sources)
    normalized.setdefault("unavailable_artifacts_or_signals", [])
    return normalized


def _deterministic_m7_feedback_decision(
    *,
    decision: M7DecisionStatus,
    iteration: int,
    source_stage: str,
    failure_category: str,
    evidence: Mapping[str, Any],
    feedback_branch: str,
    next_start_stage: str,
    rerun_targets: list[str],
    prohibited_fingerprints: list[str],
    fallback_reason: str,
    llm_succeeded: bool = False,
    parse_succeeded: bool = False,
    raw_response: str = "",
    model_request_elapsed_sec: float | None = None,
    max_feedback_iterations: int = DEFAULT_MAX_FEEDBACK_ITERATIONS,
) -> M7FeedbackDecision:
    allowed = _allowed_restart_stages_for_decision(decision.value, source_stage)
    selected_stage = next_start_stage if next_start_stage in allowed else allowed[0]
    return M7FeedbackDecision(
        m7_decision_status=decision,
        outer_iteration=iteration,
        source_stage=source_stage,
        failure_category=failure_category,
        feedback_summary=str(evidence.get("diagnosis") or decision.value),
        selected_feedback_branch=feedback_branch or next_start_stage,
        next_start_stage=selected_stage,
        cause_code="M7_FEEDBACK_FALLBACK",
        cause_source="DETERMINISTIC_FALLBACK",
        cause_hypothesis=str(evidence.get("diagnosis") or decision.value),
        cause_confidence=None,
        cause_evidence=dict(evidence),
        alternative_causes_considered=[],
        allowed_restart_stages=allowed,
        selected_restart_stage=selected_stage,
        concrete_repair_instruction=str(evidence.get("diagnosis") or fallback_reason),
        policy_validation={
            "status": "fallback",
            "fallback_reason": fallback_reason,
        },
        fields_to_preserve=_fields_to_preserve_for_decision(decision.value, rerun_targets),
        fields_to_change=_fields_to_change_for_decision(decision.value, rerun_targets),
        prohibited_prior_fingerprints=list(prohibited_fingerprints or []),
        target_reuse_allowed=_target_reuse_allowed(decision.value, next_start_stage),
        target_reuse_justification=_target_reuse_justification(decision.value, next_start_stage),
        evidence_used=dict(evidence),
        remaining_outer_iterations=max(0, max_feedback_iterations - iteration),
        deterministic_fallback_used=True,
        feedback_provenance="DETERMINISTIC_FALLBACK",
        llm_attempted=True,
        llm_succeeded=llm_succeeded,
        parse_succeeded=parse_succeeded,
        fallback_reason=fallback_reason,
        model_request_elapsed_sec=model_request_elapsed_sec,
        raw_response=raw_response,
    )


def _allowed_restart_stages_for_decision(decision: str, source_stage: str) -> list[str]:
    if decision == "ALIGNED":
        return ["M8"]
    if decision == "NOT_VALID":
        stages = ["M3", "M4", "M5"]
        if source_stage in {"M2", "UNKNOWN"}:
            stages.insert(0, "M2")
        return stages
    if decision == "ERROR":
        return ["M3", "M5", "M6", "M2"]
    if decision == "NOT_FAILED":
        return ["M2", "M3", "M5"]
    if decision in {"NO_COVERAGE", "WEAK_ALIGNMENT"}:
        return ["M2", "M3", "M5"]
    return ["M5"]


def _requires_v29_conservative_gate_judgment(alignment_result: Any) -> bool:
    """Return whether quantitative admission awaits the dedicated v29 LLM gate.

    Persisted gate facts, rather than ``failure_type``, control this branch so
    a compatibility label cannot preempt Conservative Gate review.
    """
    score_breakdown = getattr(alignment_result, "score_breakdown", {}) or {}
    if not isinstance(score_breakdown, Mapping):
        return False
    gate_results = score_breakdown.get("gate_results") or {}
    if not isinstance(gate_results, Mapping):
        return False
    all_numeric_gates_pass = gate_results.get("all_numeric_gates_pass")
    if all_numeric_gates_pass is None:
        all_numeric_gates_pass = all(
            gate_results.get(key) is True
            for key in ("gate1_pass", "gate2_pass", "gate3_pass")
        )
    return bool(
        all_numeric_gates_pass is True
        and score_breakdown.get("conservative_gate_triggered") is True
        and score_breakdown.get("conservative_gate_is_only_branching_reason") is True
    )


def _apply_v29_conservative_gate_decision(
    alignment_result: Any,
    decision: M7FeedbackDecision,
    *,
    iteration: int,
    max_feedback_iterations: int,
) -> None:
    """Canonicalize one dedicated Conservative Gate decision before M8 routing."""
    score_breakdown = alignment_result.score_breakdown
    score_breakdown["conservative_gate_assessment"] = (
        decision.conservative_gate_assessment
    )
    score_breakdown["conservative_gate_pending_llm"] = False
    if decision.m7_decision_status == M7DecisionStatus.ALIGNED:
        alignment_result.failure_type = "ALIGNED"
        alignment_result.failure_type_detail = ""
        alignment_result.should_continue = False
        alignment_result.alignment_verdict = "ALIGNED"
        alignment_result.admission_path = "CONSERVATIVE_OVERRIDE"
        alignment_result.m7_alignment_status = "ALIGNED"
        alignment_result.admitted_to_final_set = True
        alignment_result.diagnostic_only = False
        alignment_result.legacy_failure_type = "ALIGNED"
        score_breakdown["admission_path"] = "CONSERVATIVE_OVERRIDE"
        return

    alignment_result.failure_type = "WEAK_ALIGNMENT"
    alignment_result.failure_type_detail = "V29_CONSERVATIVE_GATE_REROUTE"
    alignment_result.should_continue = iteration < max_feedback_iterations
    alignment_result.alignment_verdict = "NOT_ALIGNED"
    alignment_result.admission_path = None
    alignment_result.m7_alignment_status = "WEAK_ALIGNMENT"
    alignment_result.admitted_to_final_set = False
    alignment_result.diagnostic_only = True
    alignment_result.legacy_failure_type = "WEAK_ALIGNMENT"
    alignment_result.diagnosis = decision.failure_reason
    route = str(getattr(decision, "route_destination", None) or getattr(
        decision, "selected_feedback_branch", ""
    ))
    modules = list(getattr(decision, "modules_requested_for_next", []) or [])
    if isinstance(getattr(alignment_result, "structured_feedback", None), dict):
        alignment_result.structured_feedback.update(
            feedback_type="WEAK_ALIGNMENT",
            feedback_branch=route,
            target_modules=modules,
            loop_termination_recommended=False,
        )
    if isinstance(getattr(alignment_result, "iteration_feedback_summary", None), dict):
        alignment_result.iteration_feedback_summary.update(
            verdict="WEAK_ALIGNMENT",
            feedback_branch=route,
            target_modules=modules,
        )
    score_breakdown.pop("admission_path", None)


def _build_m7_feedback_decision(
    *,
    client: Any,
    decision_status: str,
    iteration: int,
    source_stage: str,
    failure_category: str,
    evidence: Mapping[str, Any],
    feedback_branch: str,
    next_start_stage: str,
    rerun_targets: list[str],
    prohibited_fingerprints: list[str],
    diagnosis_enabled: bool = True,
    diagnosis_revision: str = "v26",
    max_feedback_iterations: int = DEFAULT_MAX_FEEDBACK_ITERATIONS,
) -> M7FeedbackDecision:
    decision = coerce_m7_decision_status(decision_status) or M7DecisionStatus.ERROR
    normalized_evidence = _normalize_m7_evidence(
        evidence,
        remaining_outer_iterations=max(0, max_feedback_iterations - iteration),
    )
    score_evidence = normalized_evidence.get("score_breakdown")
    failure_detail = (
        str(score_evidence.get("failure_type_detail") or "")
        if isinstance(score_evidence, Mapping)
        else ""
    )
    if (
        diagnosis_revision != "v37"
        and
        decision == M7DecisionStatus.NO_COVERAGE
        and failure_detail == "SBFL_UNAVAILABLE_INSUFFICIENT_P"
        and iteration < max_feedback_iterations
    ):
        return M7FeedbackDecision(
            m7_decision_status=decision,
            outer_iteration=iteration,
            source_stage=source_stage,
            failure_category=failure_category,
            feedback_summary="change localization target and scenario/candidate before recollecting PASS spectra",
            selected_feedback_branch="M2+M3+M5",
            next_start_stage="M2",
            recommended_change="select an alternative grounded target, regenerate the scenario and candidate, then rerun M6 SBFL collection",
            change_owner_module="M2",
            modules_requested_for_next=["M2", "M3", "M5"],
            cause_code="SBFL_UNAVAILABLE_INSUFFICIENT_P_RECOVERABLE",
            cause_source="RULE",
            cause_hypothesis="the current candidate did not yield three valid distinct PASS spectra; later owner changes can alter the collection population",
            cause_confidence=1.0,
            cause_evidence=dict(normalized_evidence),
            allowed_restart_stages=["M2", "M3", "M5"],
            selected_restart_stage="M2",
            concrete_repair_instruction="exclude the current M2 target, choose another grounded hypothesis, rebuild M3/M5, and retry current-pass pre-patch SBFL",
            policy_validation={
                "status": "PASS",
                "guard": "rule_conclusive_insufficient_p_recovery_before_max",
            },
            fields_to_preserve=["issue_clues", "pre_patch_only_boundary"],
            fields_to_change=["target", "context", "scenario", "candidate", "m6_pass_spectra"],
            target_reuse_allowed=False,
            target_reuse_justification="unchanged target/candidate evidence already exhausted the bounded PASS pool",
            evidence_used=dict(normalized_evidence),
            remaining_outer_iterations=max_feedback_iterations - iteration,
            feedback_provenance="RULE_CONCLUSIVE",
            llm_attempted=False,
        )
    if (
        decision == M7DecisionStatus.ERROR
        and source_stage == "M6"
        and failure_category in {"ENVIRONMENT_FAILURE", "PIPELINE_FAILURE"}
    ):
        is_pipeline_failure = failure_category == "PIPELINE_FAILURE"
        failure_detail = str(
            normalized_evidence.get("failure_type_detail") or failure_category
        )
        return M7FeedbackDecision(
            m7_decision_status=decision,
            outer_iteration=iteration,
            source_stage=source_stage,
            failure_category=failure_category,
            feedback_summary=(
                "retry retained candidate after repairing M6 post-execution processing"
                if is_pipeline_failure
                else "retry retained candidate after repairing M6 execution setup"
            ),
            selected_feedback_branch="M6",
            next_start_stage="M6",
            recommended_change=(
                "repair only M6 post-execution processing"
                if is_pipeline_failure
                else "repair only M6 execution setup"
            ),
            change_owner_module="M6",
            modules_requested_for_next=["M6", "M7"],
            cause_code=f"M6_{failure_category}_{failure_detail}",
            cause_source="RULE",
            cause_hypothesis=(
                "M6 host postprocessing failed after candidate execution"
                if is_pipeline_failure
                else "M6 setup failed before candidate behavior could be measured"
            ),
            cause_confidence=1.0,
            cause_evidence=dict(normalized_evidence),
            allowed_restart_stages=["M6"],
            selected_restart_stage="M6",
            concrete_repair_instruction=(
                "preserve collected execution evidence and candidate identity; retry M6 only"
                if is_pipeline_failure
                else "preserve candidate identity and retry M6 setup/execution"
            ),
            policy_validation={
                "status": "PASS",
                "guard": (
                    "rule_conclusive_m6_pipeline_retry"
                    if is_pipeline_failure
                    else "rule_conclusive_m6_environment_retry"
                ),
            },
            fields_to_preserve=["candidate", "context", "scenario", "target", "oracle"],
            fields_to_change=[
                "m6_post_execution_processing"
                if is_pipeline_failure
                else "execution_setup"
            ],
            target_reuse_allowed=True,
            target_reuse_justification=(
                "M6 host failure is not evidence against the retained candidate"
                if is_pipeline_failure
                else "environment failure yielded no candidate behavior evidence"
            ),
            evidence_used=dict(normalized_evidence),
            remaining_outer_iterations=max(0, max_feedback_iterations - iteration),
            feedback_provenance="RULE_CONCLUSIVE",
            llm_attempted=False,
        )
    if decision == M7DecisionStatus.ALIGNED:
        return M7FeedbackDecision(
            m7_decision_status=decision,
            outer_iteration=iteration,
            source_stage=source_stage,
            failure_category=failure_category,
            feedback_summary="candidate admitted to final set",
            selected_feedback_branch="M8",
            next_start_stage="M8",
            cause_code="ALIGNED_ADMISSION",
            cause_source="RULE",
            cause_hypothesis="all deterministic M7 admission requirements passed",
            cause_confidence=1.0,
            cause_evidence=dict(normalized_evidence),
            alternative_causes_considered=[],
            allowed_restart_stages=["M8"],
            selected_restart_stage="M8",
            concrete_repair_instruction="admit candidate to T_final and proceed to M8",
            policy_validation={"status": "PASS", "guard": "aligned_routes_to_m8"},
            admission_path="DIRECT",
            fields_to_preserve=["candidate", "scenario", "target", "oracle"],
            fields_to_change=[],
            prohibited_prior_fingerprints=[],
            target_reuse_allowed=True,
            target_reuse_justification="candidate admitted to final set",
            evidence_used=dict(normalized_evidence),
            remaining_outer_iterations=max(0, max_feedback_iterations - iteration),
            feedback_provenance="NOT_APPLICABLE",
            llm_attempted=False,
            llm_succeeded=False,
            parse_succeeded=False,
            fallback_reason=None,
            model_request_elapsed_sec=None,
        )
    diagnosis = diagnose_m7(
        evidence=normalized_evidence,
        client=client,
        enabled=diagnosis_enabled,
        revision=diagnosis_revision,
    )
    conservative_override = (
        diagnosis_revision in {"v29", "v36", "v37"}
        and diagnosis.route_destination == "→M8"
    )
    if conservative_override:
        decision = M7DecisionStatus.ALIGNED
    selected_stage = route_start_stage(diagnosis.route_destination)
    complete_plan = list(route_execution_plan(diagnosis.route_destination))
    preserve = (
        ["candidate", "context", "scenario", "target", "oracle"]
        if selected_stage == "M8"
        else
        ["context", "scenario", "target"]
        if selected_stage == "M5"
        else ["context", "target"]
        if selected_stage == "M3"
        else ["issue_clue"]
    )
    change = (
        []
        if selected_stage == "M8"
        else
        ["generated_candidate", "oracle"]
        if selected_stage == "M5"
        else ["scenario", "generated_candidate", "oracle"]
        if selected_stage == "M3"
        else ["context", "scenario", "generated_candidate", "oracle"]
    )
    return M7FeedbackDecision(
        m7_decision_status=decision,
        outer_iteration=iteration,
        source_stage=source_stage,
        failure_category=failure_category,
        feedback_summary=diagnosis.failure_reason,
        selected_feedback_branch=diagnosis.route_destination,
        next_start_stage=selected_stage,
        failure_reason=diagnosis.failure_reason,
        assumption_gap=diagnosis.assumption_gap,
        next_scenario_change=diagnosis.next_scenario_change,
        admissible_alternatives=diagnosis.admissible_alternatives,
        evidence_refs=list(diagnosis.evidence_refs),
        conservative_gate_assessment=diagnosis.conservative_gate_assessment,
        recommended_change=diagnosis.recommended_change or diagnosis.next_scenario_change,
        change_owner_module=diagnosis.change_owner_module or selected_stage,
        previous_feedback_effect=diagnosis.previous_feedback_effect,
        confidence=diagnosis.confidence,
        admission_path="CONSERVATIVE_OVERRIDE" if conservative_override else None,
        route_destination=diagnosis.route_destination,
        modules_requested_for_next=complete_plan,
        cause_code=(
            "V29_CONSERVATIVE_OVERRIDE"
            if conservative_override
            else "V29_HOLISTIC_DIAGNOSIS"
            if diagnosis_revision == "v29"
            else "V26_HOLISTIC_DIAGNOSIS"
        ),
        cause_source=diagnosis.provenance,
        cause_hypothesis=diagnosis.assumption_gap,
        cause_confidence=diagnosis.confidence,
        cause_evidence=dict(normalized_evidence),
        alternative_causes_considered=[diagnosis.admissible_alternatives],
        allowed_restart_stages=(
            ["M8"] if conservative_override else ["M2", "M3", "M5"]
        ),
        selected_restart_stage=selected_stage,
        concrete_repair_instruction=diagnosis.next_scenario_change,
        policy_validation={
            "status": "PASS" if diagnosis.provenance == LLM_DIAGNOSIS else "fallback",
            "guard": (
                "v29_six_field_schema_plus_hard_m8_guard"
                if diagnosis_revision == "v29"
                else
                "v27_five_field_schema_plus_route_consistency"
                if diagnosis_revision == "v27"
                else "v26_exact_five_field_schema"
            ),
            "fallback_reason": diagnosis.fallback_reason,
            "llm_route": diagnosis.llm_route,
            "route_consistency_status": diagnosis.route_consistency_status,
            "final_route": diagnosis.final_route,
            "exceeded_120s_telemetry_marker": diagnosis.exceeded_120s_telemetry_marker,
            "time_affects_control_flow": False,
        },
        fields_to_preserve=preserve,
        fields_to_change=change,
        prohibited_prior_fingerprints=list(prohibited_fingerprints or []),
        target_reuse_allowed=selected_stage in {"M5", "M8"},
        target_reuse_justification=(
            "M8 override preserves the quantitatively aligned candidate"
            if selected_stage == "M8"
            else "M5 route preserves supported context and scenario evidence"
            if selected_stage == "M5"
            else "upstream reroute requires refreshed target or scenario evidence"
        ),
        evidence_used=dict(normalized_evidence),
        remaining_outer_iterations=max(0, max_feedback_iterations - iteration),
        deterministic_fallback_used=diagnosis.provenance == DETERMINISTIC_FALLBACK,
        feedback_provenance=diagnosis.provenance,
        llm_attempted=diagnosis.llm_attempted,
        llm_succeeded=diagnosis.llm_succeeded,
        parse_succeeded=diagnosis.parse_succeeded,
        fallback_reason=diagnosis.fallback_reason,
        model_request_elapsed_sec=diagnosis.model_request_elapsed_sec,
        raw_response=diagnosis.raw_response,
        llm_route=diagnosis.llm_route,
        route_consistency_status=diagnosis.route_consistency_status,
        final_route=diagnosis.final_route,
    )


def _write_m7_decision_artifacts(iteration_dir: Path, record: M7DecisionRecord) -> None:
    data = record.to_dict()
    instance_root = iteration_dir.parent.parent
    candidate_evidence = record.evaluation_evidence.get("m5_candidate_evidence")
    candidate_sha256 = (
        str(candidate_evidence.get("candidate_sha") or "") or None
        if isinstance(candidate_evidence, Mapping)
        else None
    )
    evidence_path = iteration_dir / "m7_evaluation_evidence.json"
    write_json_atomic(
        {
            "schema_version": "m7-evaluation-evidence-v1",
            "instance_id": record.instance_id,
            "outer_iteration": record.outer_iteration,
            "candidate_sha256": candidate_sha256,
            "artifact_type": "m7_evaluation_evidence",
            "evidence": record.evaluation_evidence,
        },
        evidence_path,
    )
    evidence_ref = build_evidence_reference(
        evidence_path,
        instance_root,
        instance_id=record.instance_id,
        outer_iteration=record.outer_iteration,
        candidate_sha256=candidate_sha256,
        artifact_type="m7_evaluation_evidence",
    )
    evidence_summary = {
        "m7_status": record.evaluation_evidence.get("m7_status"),
        "diagnosis": record.evaluation_evidence.get("diagnosis"),
        "failure_type_detail": (
            record.evaluation_evidence.get("score_breakdown") or {}
        ).get("failure_type_detail"),
        "target_fingerprint": record.evaluation_evidence.get("target_fingerprint"),
    }
    evidence_pointer = {
        "schema_version": "m7-evidence-pointer-v1",
        "artifact_ref": evidence_ref,
        "summary": evidence_summary,
    }
    data["evaluation_evidence"] = evidence_pointer
    feedback_data = dict(data["feedback_decision"])
    for field in ("cause_evidence", "evidence_used"):
        if feedback_data.get(field):
            feedback_data[field] = evidence_pointer
    data["feedback_decision"] = feedback_data
    write_json(data, iteration_dir / "m7_decision_record.json")
    feedback_dir = iteration_dir / "m7_feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    prompt = {
        "schema_version": "m7-feedback-prompt-v1",
        "outer_iteration": record.outer_iteration,
        "m7_decision_status": record.m7_decision_status.value,
        "source_stage": record.source_stage,
        "evaluation_evidence": evidence_pointer,
        "previous_iteration_reference": record.previous_iteration_reference,
        "remaining_outer_iterations": record.remaining_outer_iterations,
    }
    write_json(prompt, feedback_dir / "prompt.json")
    (feedback_dir / "raw_response.txt").write_text(
        str(feedback_data.get("raw_response") or json.dumps(feedback_data, ensure_ascii=False, sort_keys=True)),
        encoding="utf-8",
    )
    write_json(feedback_data, feedback_dir / "parsed_response.json")
    write_json(
        {
            "schema_version": "m7-feedback-parse-status-v1",
            "parse_status": (
                "SUCCESS"
                if feedback_data.get("feedback_provenance") in {"LLM", "LLM_DIAGNOSIS"}
                else feedback_data.get("feedback_provenance")
            ),
            "deterministic_fallback_used": bool(feedback_data.get("deterministic_fallback_used")),
            "feedback_provenance": feedback_data.get("feedback_provenance"),
            "llm_attempted": bool(feedback_data.get("llm_attempted")),
            "llm_succeeded": bool(feedback_data.get("llm_succeeded")),
            "parse_succeeded": bool(feedback_data.get("parse_succeeded")),
            "fallback_reason": feedback_data.get("fallback_reason"),
            "model_timing": {"model_request_sec": feedback_data.get("model_request_elapsed_sec")},
        },
        feedback_dir / "parse_status.json",
    )
    write_json(
        {
            "schema_version": "m7-feedback-timing-v1",
            "model_request_elapsed_sec": feedback_data.get("model_request_elapsed_sec"),
        },
        feedback_dir / "timing.json",
    )


def _first_restart_stage(rerun_targets: list[str]) -> str:
    for stage in ("M2", "M3", "M4", "M5", "M6", "M8"):
        if stage in rerun_targets:
            return stage
    return "M5"


def _rerun_targets_from_next_start_stage(next_start_stage: str) -> list[str]:
    if next_start_stage == "M2":
        return list(route_execution_plan("→M2"))
    if next_start_stage == "M3":
        return list(route_execution_plan("→M3"))
    if next_start_stage == "M5":
        return list(route_execution_plan("→M5"))
    return []


def _fields_to_preserve_for_decision(decision: str, rerun_targets: list[str]) -> list[str]:
    if decision == "ALIGNED":
        return ["candidate", "scenario", "target", "oracle"]
    if decision == "NOT_FAILED" and rerun_targets == ["M5"]:
        return ["target", "setup", "stimulus"]
    if "M2" in rerun_targets:
        return ["issue_evidence"]
    if "M3" in rerun_targets:
        return ["issue_evidence", "validated_target_when_justified"]
    return ["issue_evidence", "scenario"]


def _fields_to_change_for_decision(decision: str, rerun_targets: list[str]) -> list[str]:
    if decision == "ALIGNED":
        return []
    if decision == "NOT_VALID":
        return ["generated_test"] if "M5" in rerun_targets else ["scenario"]
    if decision == "ERROR":
        return ["execution_setup"] if "M6" in rerun_targets else ["candidate"]
    if decision == "NOT_FAILED" and rerun_targets == ["M5"]:
        return ["oracle"]
    if "M2" in rerun_targets:
        return ["target", "scenario", "generated_test"]
    if "M3" in rerun_targets:
        return ["scenario", "setup", "stimulus"]
    return ["generated_test", "oracle"]


def _target_reuse_allowed(decision: str, next_start_stage: str) -> bool:
    if decision == "ALIGNED":
        return True
    if next_start_stage == "M2":
        return False
    return decision in {"NOT_VALID", "WEAK_ALIGNMENT", "NOT_FAILED"}


def _target_reuse_justification(decision: str, next_start_stage: str) -> str:
    if decision == "ALIGNED":
        return "candidate admitted to final set"
    if next_start_stage == "M2":
        return ""
    if decision == "NOT_FAILED":
        return "target may be retained only when evidence shows missed branch/precondition or missed oracle"
    if decision == "NOT_VALID":
        return "valid scenario target may be preserved while repairing candidate validity"
    return "target reuse requires changed scenario, stimulus, or oracle evidence"


def _source_stage_for_decision(decision: str) -> str:
    if decision == "ERROR":
        return "M6"
    if decision in {"NOT_FAILED", "NO_COVERAGE", "WEAK_ALIGNMENT", "ALIGNED"}:
        return "M7"
    if decision == "NOT_VALID":
        return "M5"
    return "UNKNOWN"


def _failure_category_for_decision(decision: str, align_result: Any = None) -> str:
    if decision == "ERROR":
        error_origin = (
            align_result.get("error_origin")
            if isinstance(align_result, Mapping)
            else getattr(align_result, "error_origin", None)
        )
        if error_origin in {"ADAPTER", "PARSING", "COVERAGE"}:
            return "PIPELINE_FAILURE"
        explicit = (
            align_result.get("failure_category")
            if isinstance(align_result, Mapping)
            else getattr(align_result, "failure_category", None)
        )
        if explicit in {
            "ENVIRONMENT_FAILURE",
            "PIPELINE_FAILURE",
            "EXECUTION_FAILURE",
            "EVALUATION_FAILURE",
        }:
            return str(explicit)
        return "EXECUTION_FAILURE"
    if decision == "NOT_VALID":
        return "GENERATION_FAILURE"
    return "PIPELINE_FAILURE" if decision not in {"ALIGNED", "NOT_FAILED", "NO_COVERAGE", "WEAK_ALIGNMENT"} else "NONE"


def _execution_status_from_alignment_execution(align_result: Any) -> str:
    if hasattr(align_result, "to_dict"):
        payload = align_result.to_dict()
    elif isinstance(align_result, Mapping):
        payload = dict(align_result)
    else:
        payload = {
            "has_error": getattr(align_result, "has_error", False),
            "has_failure": getattr(align_result, "has_failure", False),
            "test_results": getattr(align_result, "test_results", {}) or {},
        }
    return normalize_pre_patch_execution_status(payload).value


def _m7_evaluation_evidence(
    *,
    alignment_result: Any,
    align_result: Any,
    semantic_fingerprint: str,
    repeated_semantic_fingerprint: bool,
    selected_records: list[dict[str, Any]],
    score_breakdown: Mapping[str, Any],
    remaining_outer_iterations: int = 0,
    clue: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    scenario: Mapping[str, Any] | None = None,
    generated_test: Any | None = None,
    m6_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    clue = dict(clue or {})
    context = dict(context or {})
    scenario = dict(scenario or {})
    generated_payload = (
        generated_test.to_dict()
        if generated_test is not None and hasattr(generated_test, "to_dict")
        else dict(generated_test or {})
        if isinstance(generated_test, Mapping)
        else {}
    )
    coverage_data = getattr(align_result, "coverage_data", {}) or {}
    covered_files = sorted(str(key) for key in coverage_data.keys()) if isinstance(coverage_data, Mapping) else []
    error_messages = list(getattr(align_result, "error_messages", []) or [])
    test_results = dict(getattr(align_result, "test_results", {}) or {})
    raw_output = str(getattr(align_result, "raw_output", "") or "")
    target = scenario.get("target_location") if isinstance(scenario.get("target_location"), Mapping) else {}
    target_source = str(target.get("source_file") or scenario.get("source_file") or "")
    target_function = str(target.get("target_function") or scenario.get("target_function") or "")
    target_coverage = compute_target_coverage_evidence(
        coverage_data if isinstance(coverage_data, Mapping) else {},
        scenario,
        context,
    )
    covered_sut_lines = list(target_coverage.get("covered_target_lines") or [])
    fault_hypothesis = context.get("fault_hypothesis")
    fault_hypothesis_supported: Any = "UNKNOWN"
    if fault_hypothesis and target_coverage["target_file_covered"] != "UNKNOWN":
        fault_hypothesis_supported = target_coverage["target_file_covered"]
    coverage_score = score_breakdown.get("coverage_score")
    scenario_assumption_supported: Any = "UNKNOWN"
    if isinstance(coverage_score, (int, float)):
        scenario_assumption_supported = bool(coverage_score > 0 and not error_messages)
    code = str(generated_payload.get("test_code") or generated_payload.get("append_block") or "")
    assertions = [
        line.strip()
        for line in code.splitlines()
        if "assert" in line and line.strip()
    ][:8]
    assertion_executed: Any = "UNKNOWN"
    if re.search(r"\bAssertionError\b|^\s*>\s*assert\b", raw_output, re.MULTILINE):
        assertion_executed = True
    m6_core = getattr(align_result, "_m6_core_execution_data", {}) or {}
    stability = getattr(align_result, "stability_results", {}) or {}
    sbfl_payload = dict(m6_artifacts.get("sbfl_result") or {}) if isinstance(m6_artifacts, Mapping) else {}
    v30_storage = str(context.get("feature_profile") or context.get("methodology_revision") or "") in {"v30", "v31"}
    evidence = {
        "schema_version": "m7-v26-evidence-v1",
        "outer_iteration_policy": "v26_three_pass_diagnosis_loop",
        "diagnosis": getattr(alignment_result, "diagnosis", ""),
        "m7_status": getattr(alignment_result, "failure_type", ""),
        "score_breakdown": dict(score_breakdown or {}),
        "m1_issue_evidence": {
            "observed_behavior": list(clue.get("observed_behavior") or []),
            "expected_behavior": list(clue.get("expected_behavior") or []),
            "steps_to_reproduce": list(
                clue.get("steps_to_reproduce") or clue.get("repro_conditions") or []
            ),
        },
        "m2_semantic_evidence": {
            "fault_hypothesis": fault_hypothesis,
            "oracle_hint": context.get("oracle_hint"),
            "target_source": target_source,
            "target_function": target_function,
            "fault_hypothesis_supported": fault_hypothesis_supported,
        },
        "m3_scenario_evidence": {
            "scenario_id": scenario.get("scenario_id"),
            "assumption": scenario.get("assumption") or scenario.get("expected_failure"),
            "stimulus": scenario.get("execution_stimulus") or scenario.get("stimulus_steps") or [],
            "oracle": scenario.get("oracle_contract") or scenario.get("oracle"),
            "scenario_assumption_supported": scenario_assumption_supported,
        },
        "m5_candidate_evidence": {
            "scenario_id": generated_payload.get("scenario_id"),
            "candidate_sha": generated_payload.get("generated_patch_sha256") or generated_payload.get("patch_sha256"),
            "canonical_test_nodeid": generated_payload.get("canonical_test_nodeid"),
            "target_test_file": generated_payload.get("target_test_file"),
            "assertions": assertions,
        },
        "m6_execution_evidence": {
            "test_results": test_results,
            "error_messages": error_messages[:12],
            "failure_category": getattr(align_result, "failure_category", None),
            "error_stage": getattr(align_result, "error_stage", None),
            "error_origin": getattr(align_result, "error_origin", None),
            "exception_type": getattr(align_result, "exception_type", None),
            "observed_output_excerpt": raw_output[-6000:],
            "has_failure": bool(getattr(align_result, "has_failure", False)),
            "has_error": bool(getattr(align_result, "has_error", False)),
            "F_set": list(m6_core.get("F_set") or []),
            "P_set": list(m6_core.get("P_set") or []),
            "error_tests": list(m6_core.get("error_tests") or []),
            "covered_sut_line_count": len(covered_sut_lines),
            "covered_sut_lines_excerpt": covered_sut_lines[:20],
            "target_coverage": target_coverage,
            "stability": _json_safe(stability),
            "sbfl_reference": {
                "schema_version": sbfl_payload.get("schema_version"),
                "artifact": "sbfl_result.json" if sbfl_payload else None,
                "canonical_evidence_ref": (m6_artifacts.get("canonical_evidence_ref") if isinstance(m6_artifacts, Mapping) else None),
            },
        },
        "coverage_evidence": {
            "covered_files": covered_files,
            "artifact": "coverage_result.json",
        },
        "issue_api_executed": "UNKNOWN",
        "assertion_executed": assertion_executed,
        "actual_output_observed": bool(raw_output.strip()),
        "exception_observed": bool(error_messages) if error_messages else False,
        "suspected_file_covered": target_coverage["target_file_covered"],
        "suspected_function_covered": target_coverage["target_function_covered"],
        "suspected_lines_covered": (
            bool(covered_sut_lines)
            if target_coverage["target_function_covered"] != "UNKNOWN"
            else "UNKNOWN"
        ),
        "issue_branch_reached": scenario_assumption_supported,
        "incorrect_behavior_observed": "UNKNOWN",
        "oracle_checked_behavior": "UNKNOWN",
        "fault_hypothesis_supported": fault_hypothesis_supported,
        "scenario_assumption_supported": scenario_assumption_supported,
        "target_fingerprint": semantic_fingerprint,
        "coverage_fingerprint": hashlib.sha256(
            json.dumps(covered_files, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "failure_evidence_fingerprint": semantic_fingerprint,
        "selected_scenarios": selected_records,
        "repeated_semantic_fingerprint": repeated_semantic_fingerprint,
    }
    if v30_storage:
        evidence["m6_execution_evidence"]["F_set"] = [str(item) for item in m6_core.get("F_set") or []][:32]
        evidence["m6_execution_evidence"]["P_set"] = [str(item) for item in m6_core.get("P_set") or []][:32]
        evidence["m6_execution_evidence"]["raw_evidence_storage"] = "canonical_reference_only"
        evidence["m6_execution_evidence"]["covered_sut_lines"] = "canonical_reference_only"
    return _normalize_m7_evidence(
        evidence,
        remaining_outer_iterations=remaining_outer_iterations,
    )


def _attach_previous_m7_feedback(
    evidence: Mapping[str, Any],
    iteration_history: list[dict[str, Any]],
    current_rerun_effect: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach the prior instruction and its observed pre-patch effect."""
    updated = dict(evidence)
    if not iteration_history:
        return updated
    previous = iteration_history[-1]
    diagnosis = previous.get("diagnosis") or {}
    if not isinstance(diagnosis, Mapping):
        diagnosis = {}
    updated["previous_feedback"] = {
        "pass_number": previous.get("pass_number") or previous.get("iteration"),
        "route_destination": previous.get("route_destination"),
        "recommended_change": diagnosis.get("recommended_change"),
        "next_scenario_change": diagnosis.get("next_scenario_change"),
        "change_owner_module": diagnosis.get("change_owner_module"),
        "confidence": diagnosis.get("confidence"),
    }
    effectiveness = previous.get("m7_feedback_effectiveness") or {}
    rerun_effect = dict(current_rerun_effect or {})
    if current_rerun_effect is not None:
        assessment = (
            "NO_IMPROVEMENT"
            if rerun_effect.get("no_effect_rerun") is True
            else "IMPROVED"
        )
    else:
        assessment = (
            effectiveness.get("effectiveness")
            or effectiveness.get("status")
            or "NOT_APPLICABLE"
        )
    updated["previous_feedback_effect"] = {
        "assessment": assessment,
        "no_effect_rerun": bool(rerun_effect.get("no_effect_rerun")),
        "observed_in_pass": (
            (previous.get("pass_number") or previous.get("iteration") or 0) + 1
            if current_rerun_effect is not None
            else None
        ),
        "semantic_fingerprint_repeated": bool(
            previous.get("repeated_semantic_fingerprint")
        ),
    }
    return updated


def _not_failed_route_from_evidence(
    evidence: Mapping[str, Any],
    *,
    repeated: bool = False,
) -> tuple[str, list[str]]:
    if repeated:
        return "M3+M5", ["M3", "M5"]

    suspected_file_covered = _legacy_fact_value(evidence.get("suspected_file_covered"))
    suspected_function_covered = _legacy_fact_value(evidence.get("suspected_function_covered"))
    issue_branch_reached = _legacy_fact_value(evidence.get("issue_branch_reached"))
    incorrect_behavior_observed = _legacy_fact_value(evidence.get("incorrect_behavior_observed"))
    oracle_checked_behavior = _legacy_fact_value(evidence.get("oracle_checked_behavior"))

    if suspected_file_covered is False or suspected_function_covered is False:
        return "M3+M5", ["M3", "M5"]

    if incorrect_behavior_observed is True and oracle_checked_behavior is False:
        return "M5", ["M5"]

    if issue_branch_reached is True and incorrect_behavior_observed is False:
        return "M3+M5", ["M3", "M5"]

    if suspected_function_covered is True and issue_branch_reached in {False, "UNKNOWN", None}:
        return "M3+M5", ["M3", "M5"]

    return "M3+M5", ["M3", "M5"]


def _iteration_artifact_dir(output_dir: str, iteration: int) -> Path:
    path = Path(output_dir) / "iterations" / f"iteration_{iteration:03d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _compact_m6_iteration_artifacts(
    *,
    output_dir: Path,
    iteration_dir: Path,
    instance_id: str,
    iteration: int,
    candidate_sha256: str | None,
    raw_artifacts: Mapping[str, Any],
    align_result: Any,
    feature_profile: str | None,
) -> dict[str, Any]:
    """Replace redundant M6 projections with bounded, traceable manifests.

    The full pre-patch execution is retained once in
    ``alignment_execution.json`` and the full SBFL evidence once in
    ``sbfl_result.json``.  Live scoring continues to consume ``raw_artifacts``
    held by the caller, so this serialization-only step cannot change scores.
    """
    alignment_path = iteration_dir / "alignment_execution.json"
    sbfl_path = iteration_dir / "sbfl_result.json"
    alignment_ref = build_evidence_reference(
        alignment_path,
        output_dir,
        instance_id=instance_id,
        outer_iteration=iteration,
        candidate_sha256=candidate_sha256,
        artifact_type="alignment_execution",
    )
    sbfl_ref = build_evidence_reference(
        sbfl_path,
        output_dir,
        instance_id=instance_id,
        outer_iteration=iteration,
        candidate_sha256=candidate_sha256,
        artifact_type="sbfl_result",
    )
    test_results = dict(getattr(align_result, "test_results", {}) or {})
    execution_summary = {
        "execution_status": _execution_status_from_alignment_execution(align_result),
        "test_results": test_results,
        "has_failure": bool(getattr(align_result, "has_failure", False)),
        "has_error": bool(getattr(align_result, "has_error", False)),
        "generated_patch_sha256": candidate_sha256,
        "canonical_test_id": getattr(align_result, "canonical_test_id", None),
        "canonical_test_nodeid": getattr(align_result, "canonical_test_nodeid", None),
    }
    projection = {
        "schema_version": f"{feature_profile or 'v31'}-m6-compact-projection-v1",
        "instance_id": instance_id,
        "iteration": iteration,
        "outer_iteration": iteration,
        "candidate_sha256": candidate_sha256,
        "payload": execution_summary,
        "artifact_refs": {
            "alignment_execution": alignment_ref,
            "sbfl_result": sbfl_ref,
        },
        "summary": execution_summary,
    }
    write_json_atomic(projection, iteration_dir / "m6_execution_result.json")
    write_json_atomic(projection, iteration_dir / "execution_result.json")
    coverage_projection = {
        "schema_version": f"{feature_profile or 'v31'}-coverage-compact-projection-v1",
        "instance_id": instance_id,
        "iteration": iteration,
        "outer_iteration": iteration,
        "candidate_sha256": candidate_sha256,
        "artifact_ref": alignment_ref,
        "summary": {
            "covered_file_count": len(getattr(align_result, "coverage_data", {}) or {}),
            "covered_sut_line_count": len(getattr(align_result, "covered_sut_lines", []) or []),
            "source_checkout": "pre_patch",
        },
    }
    write_json_atomic(coverage_projection, iteration_dir / "coverage_result.json")
    return {
        "schema_version": f"{feature_profile or 'v31'}-m6-evidence-reference-v1",
        "instance_id": instance_id,
        "outer_iteration": iteration,
        "candidate_sha256": candidate_sha256,
        "canonical_evidence_refs": {
            "alignment_execution": alignment_ref,
            "sbfl_result": sbfl_ref,
        },
        "sbfl_result": {
            "schema_version": "compact-sbfl-reference-v1",
            "artifact_ref": sbfl_ref,
        },
        "summary": execution_summary,
        "serialization_only": True,
        "raw_artifact_keys": sorted(str(key) for key in raw_artifacts),
    }


def _sync_latest_iteration_aliases(
    output_dir: str,
    iteration_dir: Path,
    *,
    terminal: bool = False,
    terminal_alias_only: bool = False,
) -> None:
    """Synchronize root aliases, clearing stale files at terminal completion."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if terminal and not terminal_alias_only:
        _canonicalize_terminal_iteration_artifacts(root, iteration_dir)
    alias_names = (
        "generated_test.json",
        "generated_test.patch",
        "generated_test_rendered.py",
        "alignment_execution.json",
        "execution_result.json",
        "m6_execution_result.json",
        "coverage_result.json",
        "sbfl_result.json",
        "m7_decision_record.json",
        "alignment_result.json",
    )
    manifest: Dict[str, Any] = {
        "schema_version": "root-artifact-sync-v29-v1",
        "terminal": terminal,
        "source_iteration": iteration_dir.name,
        "artifacts": {},
    }
    candidate_id = None
    candidate_hash = None
    generated_source = iteration_dir / "generated_test.json"
    if generated_source.exists():
        try:
            generated_payload = json.loads(generated_source.read_text(encoding="utf-8"))
            candidate_id = generated_payload.get("test_id") or generated_payload.get("canonical_test_nodeid")
            candidate_hash = generated_payload.get("generated_patch_sha256") or generated_payload.get("patch_sha256")
        except (OSError, json.JSONDecodeError):
            pass
    manifest.update(candidate_id=candidate_id, candidate_hash=candidate_hash)
    referenced_json_aliases = {
        "alignment_execution.json": "alignment_execution",
        "execution_result.json": "execution_result",
        "m6_execution_result.json": "m6_execution_result",
        "coverage_result.json": "coverage_result",
        "sbfl_result.json": "sbfl_result",
    }
    for name in alias_names:
        source = iteration_dir / name
        if source.exists():
            if name in referenced_json_aliases:
                try:
                    source_payload = json.loads(source.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    source_payload = {}
                source_iteration = int(
                    source_payload.get("outer_iteration")
                    or source_payload.get("iteration")
                    or iteration_dir.name.rsplit("_", 1)[-1]
                )
                instance_id = str(source_payload.get("instance_id") or root.name)
                artifact_type = referenced_json_aliases[name]
                artifact_ref = build_evidence_reference(
                    source,
                    root,
                    instance_id=instance_id,
                    outer_iteration=source_iteration,
                    candidate_sha256=candidate_hash,
                    artifact_type=artifact_type,
                )
                write_json_atomic(
                    {
                        "schema_version": "root-artifact-reference-v1",
                        "instance_id": instance_id,
                        "iteration": source_iteration,
                        "outer_iteration": source_iteration,
                        "source_iteration": iteration_dir.name,
                        "candidate_id": candidate_id,
                        "candidate_sha256": candidate_hash,
                        "artifact_type": artifact_type,
                        "artifact_ref": artifact_ref,
                    },
                    root / name,
                )
            elif name.endswith(".json"):
                try:
                    payload = json.loads(source.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = None
                if isinstance(payload, dict):
                    if terminal and terminal_alias_only and name in {
                        "alignment_result.json",
                        "m7_decision_record.json",
                    }:
                        payload = _canonicalize_terminal_payload(payload)
                    payload.update(
                        source_iteration=iteration_dir.name,
                        candidate_id=candidate_id,
                        candidate_hash=candidate_hash,
                    )
                    write_json_atomic(payload, root / name)
                else:
                    shutil.copyfile(source, root / name)
            else:
                shutil.copyfile(source, root / name)
            manifest["artifacts"][name] = {
                "status": (
                    "REFERENCED" if name in referenced_json_aliases else "COPIED"
                ),
                "source_iteration": iteration_dir.name,
                "candidate_id": candidate_id,
                "candidate_hash": candidate_hash,
            }
        elif terminal:
            target = root / name
            if name.endswith(".json"):
                write_json_atomic(
                    {
                        "schema_version": "root-artifact-absence-v29-v1",
                        "status": "NOT_AVAILABLE",
                        "source_iteration": iteration_dir.name,
                        "candidate_id": None,
                        "candidate_hash": None,
                        "artifact": name,
                    },
                    target,
                )
            else:
                target.write_text("", encoding="utf-8")
            manifest["artifacts"][name] = {
                "status": "NOT_AVAILABLE",
                "source_iteration": iteration_dir.name,
                "candidate_id": None,
                "candidate_hash": None,
            }
    if terminal:
        write_json_atomic(manifest, root / "root_artifact_sync.json")


_TERMINAL_LIST_FIELDS = {
    "rerun_targets",
    "modules_requested_for_next",
    "modules_requested_for_next_pass",
    "requested_next_modules",
    "requested_modules",
    "target_modules",
    "allowed_restart_stages",
    "fields_to_change",
}
_TERMINAL_NULL_FIELDS = {
    "feedback_branch",
    "selected_feedback_branch",
    "route_destination",
    "next_start_stage",
    "selected_restart_stage",
    "requested_next_route",
    "change_owner_module",
    "next_scenario_change",
    "recommended_change",
    "concrete_repair_instruction",
    "final_route",
    "llm_route",
}
_TERMINAL_FALSE_FIELDS = {
    "should_continue",
    "continuation_required",
    "continue_feedback_loop",
}


def _canonicalize_terminal_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove executable continuation state while retaining labeled history."""
    diagnostics: Dict[str, Any] = {}

    def visit(value: Any, path: str = "") -> Any:
        if isinstance(value, Mapping):
            normalized: Dict[str, Any] = {}
            for raw_key, raw_value in value.items():
                key = str(raw_key)
                field_path = f"{path}.{key}" if path else key
                if key in _TERMINAL_LIST_FIELDS:
                    if raw_value:
                        diagnostics[field_path] = raw_value
                    normalized[key] = []
                elif key in _TERMINAL_NULL_FIELDS:
                    if raw_value not in (None, "", []):
                        diagnostics[field_path] = raw_value
                    normalized[key] = None
                elif key in _TERMINAL_FALSE_FIELDS:
                    if raw_value is not False:
                        diagnostics[field_path] = raw_value
                    normalized[key] = False
                elif key.endswith("_instructions"):
                    if raw_value not in (None, "", [], {}):
                        diagnostics[field_path] = raw_value
                    normalized[key] = None
                elif key in {"remaining_outer_iterations", "remaining_iterations"}:
                    if raw_value not in (None, 0):
                        diagnostics[field_path] = raw_value
                    normalized[key] = 0
                else:
                    normalized[key] = visit(raw_value, field_path)
            return normalized
        if isinstance(value, list):
            return [visit(item, f"{path}[]") for item in value]
        return value

    result = visit(dict(payload))
    result["loop_terminated"] = True
    result["should_continue"] = False
    if diagnostics:
        result["terminal_continuation_diagnostic"] = {
            "diagnostic_only": True,
            "previous_actionable_values": diagnostics,
        }
    return result


def _canonicalize_terminal_iteration_artifacts(root: Path, iteration_dir: Path) -> None:
    """Apply the terminal invariant to every canonical orchestration alias."""
    for path in (
        iteration_dir / "alignment_result.json",
        iteration_dir / "m7_decision_record.json",
        iteration_dir / "pass_manifest.json",
    ):
        payload = _read_mapping_artifact(path)
        if payload:
            write_json_atomic(_canonicalize_terminal_payload(payload), path)
    history_path = root / "iteration_history.json"
    history = _read_mapping_artifact(history_path)
    iterations = history.get("iterations") if isinstance(history, Mapping) else None
    if isinstance(iterations, list):
        normalized_history = dict(history)
        normalized_history["iterations"] = [
            _canonicalize_terminal_payload(item)
            if isinstance(item, Mapping)
            and int(item.get("iteration") or item.get("pass_number") or 0)
            == int(iteration_dir.name.rsplit("_", 1)[-1])
            else item
            for item in iterations
        ]
        write_json_atomic(normalized_history, history_path)


def _preserved_alignment_summary(
    instance_id: str,
    alignment_result: Any,
    iteration: int,
    token_usage: Dict[str, Any],
    token_usage_status: str = "known",
    selected_iteration: int | None = None,
) -> Dict[str, Any]:
    """Return the selected executed candidate when later generation regresses."""
    score_breakdown = alignment_result.score_breakdown
    return {
        "instance_id": instance_id,
        "failure_type": alignment_result.failure_type,
        **_status_fields(alignment_result.failure_type),
        "admitted_to_final_set": alignment_result.failure_type == "ALIGNED",
        "diagnostic_only": alignment_result.failure_type != "ALIGNED",
        "failure_type_detail": getattr(alignment_result, "failure_type_detail", ""),
        "bug_fail_score": score_breakdown.get("bug_fail_score"),
        "coverage_score": score_breakdown.get("coverage_score"),
        "issue_alignment_score": score_breakdown.get("issue_alignment_score"),
        "iterations": iteration if selected_iteration is not None else max(0, iteration - 1),
        "best_candidate_iteration": selected_iteration or max(1, iteration - 1),
        "terminal_attempt_iteration": iteration if selected_iteration is not None else None,
        "error": (
            alignment_result.diagnosis
            if alignment_result.failure_type == "ERROR"
            else None
        ),
        "token_usage": token_usage,
        "token_usage_status": token_usage_status,
    }


def _status_fields(failure_type: str) -> Dict[str, Any]:
    converted = legacy_failure_type_to_statuses(failure_type)
    return {
        "execution_status": converted["execution_status"],
        "validation_status": converted["validation_status"],
        "m7_decision_status": converted["m7_decision_status"],
        "m7_alignment_status": converted["m7_alignment_status"],
    }


def _token_usage_status(token_usage: Mapping[str, Any]) -> str:
    total = token_usage.get("total_tokens") if isinstance(token_usage, Mapping) else None
    if isinstance(total, int) and total > 0:
        return "known"
    return "no_llm_call"


def _accumulate_generation_token_usage(
    cumulative: Dict[str, Any],
    usage: Mapping[str, Any],
    *,
    already_accounted: bool,
) -> None:
    """Add one M5 generation charge unless its failure path already did so."""
    if already_accounted:
        return
    for key in cumulative:
        cumulative[key] += int(usage.get(key, 0) or 0)


def _cumulative_token_usage_status(
    token_usage: Mapping[str, Any],
    observed_statuses: Sequence[str],
) -> str:
    """Report cumulative accounting without letting a zero-call retry erase calls."""
    statuses = {str(status or "").strip() for status in observed_statuses}
    if "unknown" in statuses:
        return "unknown"
    if _token_usage_status(token_usage) == "known" or "known" in statuses:
        return "known"
    return "no_llm_call"


def _runtime_error_fingerprint(error_messages: Any) -> str | None:
    if not error_messages:
        return None
    if isinstance(error_messages, str):
        messages = [error_messages]
    elif isinstance(error_messages, list):
        messages = [str(item) for item in error_messages if str(item).strip()]
    else:
        messages = [str(error_messages)]
    normalized = " ".join(messages[:3]).lower()
    normalized = re.sub(r"/[^\\s:]+", "<path>", normalized)
    normalized = re.sub(r"line \d+", "line <n>", normalized)
    normalized = re.sub(r"\d+", "<n>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def main():
    loader = TDDInstanceLoader()
    instance = loader.get_by_index(0)
    output_dir = f"outputs/{instance.instance_id}"
    process_instance(instance, output_dir)


def _inject_refined_scenario(
    validation_dict: dict,
    refined_scenario: dict,
    current_scenario_id: str | None = None,
) -> dict:
    """보강된 시나리오를 validation_report에 주입하여 다음 generate()에서 사용하게 한다.
    
    Args:
        validation_dict: validation_report dict
        refined_scenario: 보강된 시나리오
        current_scenario_id: 현재 사용 중인 시나리오 ID (정확한 교체 대상 찾기 용)
    """
    import copy
    new_dict = copy.deepcopy(validation_dict)
    selected = new_dict.get("selected_scenarios", [])
    
    if selected:
        # current_scenario_id가 지정되면 해당 시나리오를 찾아서 교체
        if current_scenario_id:
            for sel in selected:
                normalized = sel.get("normalized_scenario", {})
                if normalized.get("scenario_id") == current_scenario_id:
                    sel["normalized_scenario"] = refined_scenario
                    return new_dict
        # ID 지정 없거나 찾지 못한 경우 첫 번째(=primary) 시나리오 교체
        selected[0]["normalized_scenario"] = refined_scenario
    else:
        new_dict["selected_scenarios"] = [{
            "scenario_id": refined_scenario.get("scenario_id", "S_REFINED"),
            "score": 0.3,
            "decision": "accept",
            "reasons": ["selected refined scenario because selected_scenarios was empty"],
            "normalized_scenario": refined_scenario,
            "force_selected": True,
            "scenario_repaired": True,
            "scenario_repair_reason": "inject_refined_empty_selection",
        }]
    return new_dict


def _clear_harness_cache(instance_id: str, output_dir: str, alignment: bool = False) -> None:
    """이전 실행의 하네스 캐시를 삭제한다."""
    benchmark_root = Path("benchmark/TDD-Bench-Verified")

    if alignment:
        run_id = f"align-{instance_id}"
    else:
        run_id = f"debug-{instance_id}"

    # 평가 로그 삭제
    eval_log_dir = benchmark_root / "logs" / "run_evaluation" / run_id
    if eval_log_dir.exists():
        shutil.rmtree(eval_log_dir, ignore_errors=True)

    # 리포트 파일 삭제
    for report_file in benchmark_root.glob(f"*{run_id}*.json"):
        report_file.unlink(missing_ok=True)


def _print_summary(instance, clue_path, context_path, scenario_path, scenarios, context):
    """초기 파이프라인 단계 요약 출력."""
    print("instance_id:", instance.instance_id)
    print("repo:", instance.repo)
    print("clue saved to:", clue_path)
    print("context saved to:", context_path)
    print("scenario saved to:", scenario_path)

    print(f"\nnum_scenarios: {len(scenarios)}")
    for s in scenarios:
        print("-", s.scenario_id)

    print("\ncandidate_source_files")
    for x in context.candidate_source_files:
        print("-", x["path"], "| score =", x["score"])

    print("\ncandidate_test_files")
    for x in context.candidate_test_files:
        print("-", x["path"], "| score =", x["score"])

    print("\nproject_test_style")
    print(context.project_test_style)


if __name__ == "__main__":
    main()

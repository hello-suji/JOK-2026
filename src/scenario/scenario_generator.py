from __future__ import annotations

import json
import copy
import logging
import math
import os
import re
import hashlib
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from src.models.client import LLMClient, ModelStageTimeoutError, ModelTimeoutError, estimate_prompt_tokens
from src.models.config import load_model_config
from src.contracts.feature_flags import V22FeatureFlags, resolve_feature_flags
from src.contracts.instance_views import make_pre_patch_view
from src.contracts.v37_oracle_flags import ORACLE_EXPECTED_MISSING
from src.scenario.code_block_roles import ensure_reproduction_code_blocks
from src.scenario.scenario_hydrator import hydrate_scenario_dict

logger = logging.getLogger(__name__)

M3_N_MIN = 3
M3_N_MAX = 10
M3_CONSECUTIVE_CLUSTER_STOP = 3
M3_SHARPENING_BETA = 2
M3_STABLE_THRESHOLD = 0.6
M3_LOW_CONSENSUS_THRESHOLD = 0.4
M3_MAX_ROLLBACK_ATTEMPTS = 2
M3_ADAPTIVE_STATUS_CONSENSUS_VALID = "CONSENSUS_VALID"
M3_ADAPTIVE_STATUS_INSUFFICIENT_VALID_SAMPLES = "INSUFFICIENT_VALID_SAMPLES"
M3_ADAPTIVE_STATUS_SKIPPED_FALLBACK = "SKIPPED_FALLBACK"
M3_STAGE_TIMEOUT_ENV = "V22_M3_STAGE_TIMEOUT_SEC"
M3_STAGE_MAX_CALLS_ENV = "V22_M3_STAGE_MAX_MODEL_CALLS"
M3_NONADAPTIVE_MAX_TOKENS_ENV = "V22_M3_NONADAPTIVE_MAX_TOKENS"
M3_NONADAPTIVE_DEFAULT_MAX_TOKENS = 2048
M3_OPTIMIZED_MODEL_CALL_LIMIT = M3_N_MAX
M3_STOP_EARLY_STOPPED = "EARLY_STOPPED"
M3_STOP_ATTEMPT_BUDGET_EXHAUSTED = "ATTEMPT_BUDGET_EXHAUSTED"
M3_STOP_MALFORMED_JSON_RETRY_EXHAUSTED = "MALFORMED_JSON_RETRY_EXHAUSTED"
M3_STOP_ORACLE_ROLLBACK_RETRY_EXHAUSTED = "ORACLE_ROLLBACK_RETRY_EXHAUSTED"
M3_STOP_MODEL_STAGE_TIMEOUT = "MODEL_STAGE_TIMEOUT"
M3_STOP_CALL_BUDGET_EXHAUSTED = "CALL_BUDGET_EXHAUSTED"
M3_BLOCKING_ORACLE_FLAGS = {
    "oracle_is_actual_behavior",
    "oracle_tests_wrong_property",
    "oracle_is_tautology",
    "oracle_has_no_assertion",
}


def normalize_v37_confidence_score(value: Any) -> tuple[int, str]:
    """Apply the exact v37 confidence cast/default contract."""
    if isinstance(value, bool):
        return 3, "default_malformed"
    if value is None:
        return 3, "default_missing"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 3, "default_malformed"
    if not math.isfinite(numeric) or not numeric.is_integer():
        return 3, "default_malformed"
    parsed = int(numeric)
    if not 1 <= parsed <= 5:
        return 3, "default_out_of_range"
    return parsed, "model_value_integer_like" if not isinstance(value, int) else "model_value_integer"


def normalize_v37_json_transport(raw_response: str) -> tuple[str, str]:
    """Remove one surrounding JSON code fence without recovering prose."""
    text = str(raw_response or "").strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced is None:
        return text, "NONE"
    return fenced.group(1).strip(), "SINGLE_JSON_CODE_FENCE_REMOVED"


@dataclass
class TestScenario:
    scenario_id: str
    target_location: Dict[str, Any]
    setup_steps: List[str]
    execution_stimulus: List[str]
    expected_failure: str
    relevant_source_files: List[str]
    relevant_test_files: List[str]
    test_environment: Dict[str, Any] = None  # {"required_fixtures": [...], "runner": "pytest"}
    # clue에서 merge되는 필드 (run_single.py에서 validation 후 채워짐)
    reproduction_code: List[Dict[str, str]] = None   # clue.code_examples
    expected_outputs: List[str] = None               # clue.expected_outputs
    actual_outputs: List[str] = None                 # clue.actual_outputs
    error_keywords: List[str] = None                 # clue.error_keywords
    identifiers: Dict[str, List[str]] = None         # clue.identifiers
    oracle_hints: List[str] = None                   # synthesized oracle guidance
    oracle: str = ""                                 # compact oracle guidance string
    oracle_contract: Dict[str, str] = None           # {"oracle_type": ..., "oracle_source": ..., "rule": ...}
    oracle_type: str = ""                            # compact top-level copy for prompts
    oracle_source: str = ""                          # compact top-level copy for prompts
    validation_status: str = "pending"
    diagnostic_only: bool = False
    generation_provenance: str = "model_generated"
    issue_api_target: str = ""
    implementation_target: str = ""
    setup_helper_calls: List[str] = None
    target_verification_status: str = ""
    target_verification_provenance: Dict[str, Any] = None
    target_consistency_status: str = ""
    scenario_generation_attempt: int = 1
    m3_model_call_count: int = 0
    fallback_used: bool = False
    fallback_reason: str = ""
    fault_hypothesis: str = ""
    m2_oracle_hint: str = ""
    feedback_consumed: Dict[str, Any] = None
    hypothesis_id: str = ""
    uncertainty: str = ""
    issue_evidence: List[str] = None
    instance_id: str = ""
    iteration: int | None = None
    outer_iteration: int | None = None
    confidence_score: int = 3
    confidence_score_provenance: str = "default_missing"
    blocking_oracle_flags: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("reproduction_code") is None:
            d["reproduction_code"] = []
        if d.get("expected_outputs") is None:
            d["expected_outputs"] = []
        if d.get("actual_outputs") is None:
            d["actual_outputs"] = []
        if d.get("error_keywords") is None:
            d["error_keywords"] = []
        if d.get("identifiers") is None:
            d["identifiers"] = {}
        if d.get("test_environment") is None:
            d["test_environment"] = {}
        if d.get("oracle_hints") is None:
            d["oracle_hints"] = []
        if d.get("oracle") is None:
            d["oracle"] = ""
        if d.get("oracle_contract") is None:
            d["oracle_contract"] = {}
        if d.get("oracle_type") is None:
            d["oracle_type"] = ""
        if d.get("oracle_source") is None:
            d["oracle_source"] = ""
        if not d.get("validation_status"):
            d["validation_status"] = "pending"
        if d.get("setup_helper_calls") is None:
            d["setup_helper_calls"] = []
        if d.get("target_verification_provenance") is None:
            d["target_verification_provenance"] = {}
        if d.get("feedback_consumed") is None:
            d["feedback_consumed"] = {}
        if d.get("issue_evidence") is None:
            d["issue_evidence"] = []
        if d.get("blocking_oracle_flags") is None:
            d["blocking_oracle_flags"] = []
        target = d.get("target_location") if isinstance(d.get("target_location"), dict) else {}
        contract_view = canonical_scenario_projection(d)
        d.update({
            "target_function": contract_view["target_function"],
            "source_file": contract_view["source_file"],
            "oracle_expected": contract_view["oracle_expected"],
            "stimulus_steps": contract_view["stimulus_steps"],
            "candidate_test_file": contract_view["candidate_test_file"],
            "validation_status": contract_view["validation_status"],
            "diagnostic_only": bool(d.get("diagnostic_only", False)),
            "issue_api_target": str(d.get("issue_api_target") or target.get("issue_api_target") or ""),
            "implementation_target": str(d.get("implementation_target") or target.get("implementation_target") or ""),
            "setup_helper_calls": list(d.get("setup_helper_calls") or target.get("setup_helper_calls") or []),
            "target_verification_status": str(d.get("target_verification_status") or target.get("target_verification_status") or ""),
            "target_verification_provenance": dict(d.get("target_verification_provenance") or target.get("target_verification_provenance") or {}),
            "target_consistency_status": str(d.get("target_consistency_status") or target.get("target_consistency_status") or ""),
        })
        d["target_location"] = target
        return d


def bind_scenarios_to_localization_hypotheses(
    scenarios: Sequence["TestScenario"],
    hypotheses: Sequence[Mapping[str, Any]],
) -> List["TestScenario"]:
    """Create bounded hypothesis-specific scenario candidates for v30."""
    if not hypotheses:
        return list(scenarios)
    output: list[TestScenario] = []
    for base in scenarios:
        base_data = base.to_dict() if hasattr(base, "to_dict") else dict(base)
        for index, hypothesis in enumerate(hypotheses[:3]):
            if not isinstance(hypothesis, Mapping):
                continue
            item = copy.deepcopy(base_data)
            hypothesis_id = str(hypothesis.get("hypothesis_id") or f"h{index + 1}")
            target = dict(item.get("target_location") or {})
            target["source_file"] = str(hypothesis.get("source_file") or target.get("source_file") or "")
            qualified_symbol = str(
                hypothesis.get("qualified_name")
                or hypothesis.get("qualified_symbol")
                or hypothesis.get("function_name")
                or target.get("target_function")
                or ""
            )
            target["target_function"] = qualified_symbol
            target["qualified_symbol"] = qualified_symbol
            target["line_range"] = list(hypothesis.get("line_range") or [])
            target["hypothesis_id"] = hypothesis_id
            item["target_location"] = target
            item["scenario_id"] = f"{item.get('scenario_id') or 'S1'}-{hypothesis_id}"
            item["hypothesis_id"] = hypothesis_id
            item["uncertainty"] = str(hypothesis.get("uncertainty") or "")
            item["issue_evidence"] = list(hypothesis.get("evidence") or [])
            item["generation_provenance"] = "v30_hypothesis_bound"
            output.append(TestScenario(**{
                key: value for key, value in item.items()
                if key in TestScenario.__dataclass_fields__
            }))
    return output


@dataclass
class ScenarioSamplingAttempt:
    attempt_index: int
    raw_response: str
    parsed: Any = None
    validated: Dict[str, Any] | None = None
    valid: bool = False
    rollback_reason: str | None = None
    blocking_oracle_flags: List[str] | None = None
    answer_key: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if data.get("blocking_oracle_flags") is None:
            data["blocking_oracle_flags"] = []
        return data


@dataclass
class AdaptiveScenarioSamplingResult:
    scenarios: List[Dict[str, Any]]
    attempts: List[ScenarioSamplingAttempt]
    sample_count: int
    answer_key: str | None
    consensus: Dict[str, Any]
    validated_scenarios: List[Dict[str, Any]]
    rollback_count: int
    rollback_reasons: List[str]
    early_stopped: bool
    valid_sample_count: int = 0
    failed_parse_count: int = 0
    total_attempt_count: int = 0
    termination_reason: str = ""
    adaptive_sampling_status: str = ""
    fallback_used: bool = False
    metadata: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenarios": self.scenarios,
            "sample_count": self.sample_count,
            "answer_key": self.answer_key,
            "consensus": self.consensus,
            "validated_scenarios": self.validated_scenarios,
            "rollback_count": self.rollback_count,
            "rollback_reasons": self.rollback_reasons,
            "early_stopped": self.early_stopped,
            "valid_sample_count": self.valid_sample_count,
            "failed_parse_count": self.failed_parse_count,
            "total_attempt_count": self.total_attempt_count,
            "termination_reason": self.termination_reason,
            "adaptive_sampling_status": self.adaptive_sampling_status,
            "fallback_used": self.fallback_used,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            # Legacy M3 names from the implementation spec.
            "samples_generated": self.sample_count,
            "cluster_counts": {
                key: int(round(value * self.sample_count))
                for key, value in self.consensus.get("answer_marginals", {}).items()
            },
            "consensus_norm": self.consensus.get("Consensus_norm"),
            "validation_failures": [
                attempt.to_dict()
                for attempt in self.attempts
                if not attempt.valid
            ],
            "blocking_oracle_flags": sorted({
                flag
                for attempt in self.attempts
                for flag in (attempt.blocking_oracle_flags or [])
            }),
            "metadata": self.metadata or {},
        }


@dataclass
class M3StageCallBudget:
    stage: str
    timeout_sec: float
    max_model_calls: int
    started_at: float = 0.0
    model_call_count: int = 0
    completed_call_count: int = 0
    per_call_latency: List[float] = None
    call_records: List[Dict[str, Any]] = None
    last_sample_index: int = 0

    def __post_init__(self) -> None:
        if self.started_at <= 0:
            self.started_at = time.monotonic()
        if self.per_call_latency is None:
            self.per_call_latency = []
        if self.call_records is None:
            self.call_records = []

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining(self) -> float:
        return max(0.0, float(self.timeout_sec) - self.elapsed)

    def before_call(self, sample_index: int, configured_timeout: float) -> float:
        self.last_sample_index = sample_index
        if self.model_call_count >= self.max_model_calls:
            self.raise_timeout("M3 adaptive model call budget exhausted")
        remaining = self.remaining
        if remaining <= 0:
            self.raise_timeout("M3 adaptive stage deadline exhausted")
        self.model_call_count += 1
        return max(1.0, min(float(configured_timeout), remaining))

    def record_completed_call(self, latency: float) -> None:
        self.completed_call_count += 1
        self.per_call_latency.append(round(float(latency), 3))

    def record_call(self, record: Mapping[str, Any]) -> None:
        self.call_records.append(dict(record))

    def raise_timeout(self, message: str) -> None:
        raise ModelStageTimeoutError(
            message,
            stage=self.stage,
            model_call_count=self.model_call_count,
            completed_call_count=self.completed_call_count,
            per_call_latency=list(self.per_call_latency),
            total_stage_latency=round(self.elapsed, 3),
            timeout_sec=float(self.timeout_sec),
            last_sample_index=self.last_sample_index,
            call_records=list(self.call_records),
            stop_reason=M3_STOP_MODEL_STAGE_TIMEOUT,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "model_call_count": self.model_call_count,
            "completed_call_count": self.completed_call_count,
            "per_call_latency": list(self.per_call_latency),
            "total_stage_latency": round(self.elapsed, 3),
            "timeout_sec": float(self.timeout_sec),
            "last_sample_index": self.last_sample_index,
            "max_model_calls": self.max_model_calls,
            "call_records": list(self.call_records),
        }


ScenarioSampleAdapter = Callable[[int], str]
OracleExpectedRegenerator = Callable[[Dict[str, Any], str, int], Any]


def canonical_scenario_projection(
    scenario: Mapping[str, Any],
    *,
    instance_id: str = "",
) -> Dict[str, Any]:
    """Return the v22 Scenario contract projection without dropping legacy fields."""
    target = scenario.get("target_location", {})
    if not isinstance(target, Mapping):
        target = {}
    oracle_contract = scenario.get("oracle_contract", {})
    if not isinstance(oracle_contract, Mapping):
        oracle_contract = {}
    expected_outputs = _ensure_projection_list(scenario.get("expected_outputs", []))
    oracle_expected: Any = scenario.get("oracle_expected")
    if oracle_expected is None and expected_outputs:
        oracle_expected = expected_outputs[0]
    stimulus_steps = _ensure_projection_list(
        scenario.get("stimulus_steps")
        or scenario.get("execution_stimulus")
        or []
    )
    validation_status = str(scenario.get("validation_status") or "pending")
    return {
        "instance_id": str(instance_id or scenario.get("instance_id", "")),
        "scenario_id": str(scenario.get("scenario_id") or "unknown"),
        "target_function": str(
            scenario.get("target_function") or target.get("target_function") or ""
        ),
        "source_file": str(scenario.get("source_file") or target.get("source_file") or ""),
        "oracle_type": str(
            scenario.get("oracle_type") or oracle_contract.get("oracle_type") or ""
        ),
        "oracle_expected": oracle_expected,
        "stimulus_steps": stimulus_steps,
        "candidate_test_file": (
            str(scenario.get("candidate_test_file") or target.get("candidate_test_file") or "")
            or None
        ),
        "validation_status": validation_status,
    }


def _ensure_projection_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    if value:
        return [str(value)]
    return []


def normalize_answer_key_value(value: Any) -> str:
    """Normalize answer-key fields without dropping module identity."""
    text = " ".join(str(value or "").replace("\\", "/").split())
    if not text:
        return ""
    if "." in text:
        parts = [part.strip() for part in text.split(".") if part.strip()]
        return ".".join(part.lower() for part in parts)
    return text.lower()


def stimulus_summary_from_steps(stimulus_steps: Sequence[Any]) -> str:
    """Return an order-sensitive, whitespace-normalized stimulus summary."""
    steps = [" ".join(str(step).split()) for step in stimulus_steps if str(step).strip()]
    return json.dumps(steps, ensure_ascii=False, separators=(",", ":"))


def build_answer_key(scenario: Mapping[str, Any]) -> str:
    """Build the deterministic M3 clustering key.

    The key is a canonical serialization of target_function, oracle_type, and
    an order-preserving stimulus summary. Free-form reasoning fields are not
    included.
    """
    projected = canonical_scenario_projection(scenario)
    key_tuple = {
        "target_function": normalize_answer_key_value(projected.get("target_function")),
        "oracle_type": normalize_answer_key_value(projected.get("oracle_type")),
        "stimulus_summary": stimulus_summary_from_steps(
            projected.get("stimulus_steps") or []
        ),
    }
    return json.dumps(key_tuple, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_answer_marginals(answer_keys: Sequence[str]) -> Dict[str, float]:
    """Return answer marginals in deterministic key order.

    Empty input returns an empty mapping rather than fabricating a zero-valued
    population.
    """
    total = len(answer_keys)
    if total == 0:
        return {}
    counts: Dict[str, int] = {}
    for answer_key in answer_keys:
        counts[answer_key] = counts.get(answer_key, 0) + 1
    return {
        answer_key: counts[answer_key] / total
        for answer_key in sorted(counts)
    }


def sharpen_answer_marginals(
    answer_marginals: Mapping[str, float],
    *,
    beta_s: int = M3_SHARPENING_BETA,
) -> Dict[str, float]:
    """Apply beta=2 answer-marginal sharpening with deterministic ordering."""
    if not answer_marginals:
        return {}
    unnormalized = {
        key: max(0.0, min(1.0, float(value))) ** beta_s
        for key, value in answer_marginals.items()
    }
    denominator = sum(unnormalized.values())
    if denominator <= 0:
        return {key: 0.0 for key in sorted(unnormalized)}
    return {
        key: unnormalized[key] / denominator
        for key in sorted(unnormalized)
    }


def build_consensus(answer_keys: Sequence[str]) -> Dict[str, Any]:
    """Build answer marginals, sharpened marginals, and stable/unstable status."""
    answer_marginals = compute_answer_marginals(answer_keys)
    sharp_marginals = sharpen_answer_marginals(answer_marginals)
    consensus_available = len(answer_keys) >= M3_N_MIN
    consensus_norm = max(sharp_marginals.values()) if sharp_marginals and consensus_available else None
    stability = (
        "stable"
        if consensus_norm is not None and consensus_norm >= M3_STABLE_THRESHOLD
        else "unstable"
    )
    if not consensus_available:
        stability = "unavailable"
    diagnostic_band = None
    if consensus_norm is not None:
        if consensus_norm < M3_LOW_CONSENSUS_THRESHOLD:
            diagnostic_band = "low_consensus"
        elif consensus_norm < M3_STABLE_THRESHOLD:
            diagnostic_band = "borderline_consensus"
    return {
        "answer_marginals": answer_marginals,
        "sharp_marginals": sharp_marginals,
        "Consensus_norm": consensus_norm,
        "stability": stability,
        "diagnostic_band": diagnostic_band,
        "total_samples": len(answer_keys),
        "valid_sample_count": len(answer_keys),
        "early_stopped": False,
    }


def validate_m3_oracle_expected(
    scenario: Mapping[str, Any],
    clue: Mapping[str, Any],
) -> List[str]:
    """Return M3 blocking oracle flags for EB/OB alignment checks."""
    projected = canonical_scenario_projection(scenario)
    oracle_expected = projected.get("oracle_expected")
    oracle_text = " ".join(
        str(value)
        for value in [
            oracle_expected,
            scenario.get("expected_failure", ""),
            scenario.get("oracle", ""),
            " ".join(_ensure_projection_list(scenario.get("oracle_hints", []))),
        ]
        if value is not None
    )
    normalized_oracle = _semantic_text(oracle_text)
    expected_signals = _semantic_values(
        _ensure_projection_list(clue.get("expected_outputs", []))
        + _ensure_projection_list(clue.get("expected_behavior", []))
    )
    actual_signals = _semantic_values(
        _ensure_projection_list(clue.get("actual_outputs", []))
        + _ensure_projection_list(clue.get("observed_behavior", []))
    )
    flags: List[str] = []
    if not normalized_oracle or normalized_oracle in {"assert", "assert()"}:
        flags.append("oracle_has_no_assertion")
    if _is_tautological_oracle(normalized_oracle):
        flags.append("oracle_is_tautology")
    actual_match = bool(normalized_oracle and any(value and value in normalized_oracle for value in actual_signals))
    expected_match = bool(normalized_oracle and any(value and value in normalized_oracle for value in expected_signals))
    if actual_match and not expected_match:
        flags.append("oracle_is_actual_behavior")
    if expected_signals and not expected_match:
        flags.append("oracle_tests_wrong_property")
    return [flag for flag in flags if flag in M3_BLOCKING_ORACLE_FLAGS]


def _semantic_values(values: Sequence[Any]) -> List[str]:
    return [text for text in (_semantic_text(value) for value in values) if text]


def _semantic_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _is_tautological_oracle(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"\b(assert\s+true|asserttrue\s*\(\s*true|always\s+true|tautolog)", text))


def _parse_strict_scenario_response(raw_response: str) -> List[Dict[str, Any]]:
    text = raw_response.strip()
    fence_match = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    data = json.loads(text)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not data:
        raise ValueError("scenario response must be a non-empty JSON object or array")
    scenarios = [item for item in data if isinstance(item, dict)]
    if not scenarios:
        raise ValueError("scenario response did not contain scenario objects")
    return scenarios


def _validate_m3_schema(scenario: Mapping[str, Any]) -> Dict[str, Any]:
    projected = canonical_scenario_projection(scenario)
    missing = [
        field_name
        for field_name in ("target_function", "source_file", "oracle_type", "stimulus_steps")
        if not projected.get(field_name)
    ]
    if "oracle_expected" not in projected or projected.get("oracle_expected") is None:
        missing.append("oracle_expected")
    if missing:
        raise ValueError(f"scenario schema missing required fields: {', '.join(missing)}")
    normalized = dict(scenario)
    normalized.update(projected)
    target = normalized.get("target_location") if isinstance(normalized.get("target_location"), dict) else {}
    target.setdefault("target_function", projected["target_function"])
    target.setdefault("source_file", projected["source_file"])
    if projected.get("candidate_test_file"):
        target.setdefault("candidate_test_file", projected["candidate_test_file"])
    normalized["target_location"] = target
    normalized["stimulus_steps"] = projected["stimulus_steps"]
    normalized.setdefault("execution_stimulus", projected["stimulus_steps"])
    normalized["oracle_type"] = projected["oracle_type"]
    normalized["oracle_expected"] = projected["oracle_expected"]
    return normalized


class M3AdaptiveSamplingController:
    """Reusable adaptive self-consistency controller for M3 scenario samples."""

    def __init__(
        self,
        *,
        n_min: int = M3_N_MIN,
        n_max: int = M3_N_MAX,
        consecutive_cluster_stop: int = M3_CONSECUTIVE_CLUSTER_STOP,
        max_rollback_attempts: int = M3_MAX_ROLLBACK_ATTEMPTS,
    ) -> None:
        if n_min < 1 or n_max < n_min:
            raise ValueError("M3 adaptive sampling requires 1 <= n_min <= n_max")
        if consecutive_cluster_stop < 1:
            raise ValueError("consecutive_cluster_stop must be positive")
        self.n_min = n_min
        self.n_max = n_max
        self.consecutive_cluster_stop = consecutive_cluster_stop
        self.max_rollback_attempts = max_rollback_attempts

    def run(
        self,
        *,
        sample_adapter: ScenarioSampleAdapter,
        clue: Mapping[str, Any],
        oracle_expected_regenerator: OracleExpectedRegenerator | None = None,
        fallback_scenarios: Sequence[Mapping[str, Any]] | None = None,
    ) -> AdaptiveScenarioSamplingResult:
        attempts: List[ScenarioSamplingAttempt] = []
        valid_scenarios: List[Dict[str, Any]] = []
        answer_keys: List[str] = []
        rollback_count = 0
        rollback_reasons: List[str] = []
        early_stopped = False
        sample_attempt_index = 0
        failed_parse_count = 0
        blocking_oracle_count = 0
        termination_reason = "ATTEMPT_BUDGET_EXHAUSTED"

        while sample_attempt_index < self.n_max:
            sample_attempt_index += 1
            raw_response = sample_adapter(sample_attempt_index)
            attempt = ScenarioSamplingAttempt(
                attempt_index=sample_attempt_index,
                raw_response=raw_response,
            )
            attempts.append(attempt)

            try:
                parsed = _parse_strict_scenario_response(raw_response)
                attempt.parsed = parsed
                validated = _validate_m3_schema(parsed[0])
            except (json.JSONDecodeError, ValueError) as exc:
                reason = f"schema_or_json_violation: {exc}"
                attempt.rollback_reason = reason
                rollback_count += 1
                failed_parse_count += 1
                rollback_reasons.append(reason)
                if failed_parse_count > self.max_rollback_attempts:
                    termination_reason = M3_STOP_MALFORMED_JSON_RETRY_EXHAUSTED
                    break
                continue

            flags = validate_m3_oracle_expected(validated, clue)
            if flags and oracle_expected_regenerator is not None:
                oracle_rollbacks_for_sample = 0
                for flag in flags[:]:
                    if oracle_rollbacks_for_sample >= self.max_rollback_attempts:
                        break
                    oracle_rollbacks_for_sample += 1
                    rollback_count += 1
                    reason = f"oracle_expected_regenerated: {flag}"
                    rollback_reasons.append(reason)
                    validated["oracle_expected"] = oracle_expected_regenerator(
                        dict(validated),
                        flag,
                        rollback_count,
                    )
                    flags = validate_m3_oracle_expected(validated, clue)
                    if not flags:
                        break

            attempt.blocking_oracle_flags = flags
            if flags:
                reason = "blocking_oracle_flags: " + ",".join(flags)
                attempt.rollback_reason = reason
                if oracle_expected_regenerator is None:
                    rollback_count += 1
                rollback_reasons.append(reason)
                blocking_oracle_count += 1
                if blocking_oracle_count > self.max_rollback_attempts:
                    termination_reason = M3_STOP_ORACLE_ROLLBACK_RETRY_EXHAUSTED
                    break
                continue

            answer_key = build_answer_key(validated)
            validated["answer_key"] = answer_key
            attempt.validated = validated
            attempt.valid = True
            attempt.answer_key = answer_key
            valid_scenarios.append(validated)
            answer_keys.append(answer_key)

            if (
                len(valid_scenarios) >= self.n_min
                and len(answer_keys) >= self.consecutive_cluster_stop
                and len(set(answer_keys[-self.consecutive_cluster_stop:])) == 1
            ):
                early_stopped = True
                termination_reason = "EARLY_STOPPED"
                break

        valid_sample_count = len(valid_scenarios)
        fallback_used = False
        if valid_sample_count < self.n_min and fallback_scenarios:
            for fallback in fallback_scenarios:
                try:
                    validated = _validate_m3_schema(fallback)
                except ValueError as exc:
                    rollback_reasons.append(f"fallback_schema_violation: {exc}")
                    continue
                answer_key = build_answer_key(validated)
                validated["answer_key"] = answer_key
                valid_scenarios.append(validated)
                fallback_used = True
                break

        consensus = build_consensus(answer_keys)
        if valid_sample_count < self.n_min:
            consensus["Consensus_norm"] = None
            consensus["stability"] = "unavailable"
            consensus["diagnostic_band"] = "insufficient_valid_samples"
        consensus["early_stopped"] = early_stopped
        if valid_sample_count >= self.n_min:
            adaptive_sampling_status = M3_ADAPTIVE_STATUS_CONSENSUS_VALID
        elif fallback_used:
            adaptive_sampling_status = M3_ADAPTIVE_STATUS_INSUFFICIENT_VALID_SAMPLES
        else:
            adaptive_sampling_status = M3_ADAPTIVE_STATUS_SKIPPED_FALLBACK
        selected_answer_key = _select_consensus_answer_key(consensus.get("sharp_marginals", {}))
        return AdaptiveScenarioSamplingResult(
            scenarios=valid_scenarios,
            attempts=attempts,
            sample_count=valid_sample_count,
            answer_key=selected_answer_key if valid_sample_count >= self.n_min else None,
            consensus=consensus,
            validated_scenarios=valid_scenarios,
            rollback_count=rollback_count,
            rollback_reasons=rollback_reasons,
            early_stopped=early_stopped,
            valid_sample_count=valid_sample_count,
            failed_parse_count=failed_parse_count,
            total_attempt_count=len(attempts),
            termination_reason=termination_reason,
            adaptive_sampling_status=adaptive_sampling_status,
            fallback_used=fallback_used,
        )


def _choose_issue_api_target_for_fallback(
    clue: Mapping[str, Any],
    functions: Sequence[str],
    prohibited_targets: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    helper_names = {
        "copy", "getvalue", "read", "write_text", "read_text", "open", "close",
        "flush", "seek", "tell", "append", "extend",
        "print", "repr", "str", "format", "len", "range", "list", "dict",
        "set", "tuple", "bool", "int", "float", "bytes", "isinstance",
        "issubclass", "hasattr", "getattr", "setattr", "super",
        "platform", "get_backend", "bit", "enforced",
    }
    prohibited_tails = {
        str(item.get("target_function") or item.get("function_name") or "")
        .split(".")[-1]
        .lower()
        for item in (prohibited_targets or [])
        if isinstance(item, Mapping)
    }
    function_tails = {
        str(function).split(".")[-1].lower()
        for function in functions
        if str(function).strip()
    }
    calls: List[str] = []
    for block in clue.get("code_examples", []) or []:
        if not isinstance(block, Mapping):
            continue
        text = "\n".join(
            str(block.get(key) or "")
            for key in ("code", "interactive_input", "text")
        )
        for call in re.findall(r"\b((?:[A-Za-z_]\w*\.)?([A-Za-z_]\w{2,}))\s*\(", text):
            full, bare = call
            if bare.lower() in prohibited_tails:
                continue
            if bare.lower() in helper_names:
                continue
            if full not in calls:
                calls.append(full)
    issue_text = " ".join(
        str(value)
        for key in ("raw_issue_text", "observed_behavior", "expected_behavior", "repro_conditions")
        for value in (
            clue.get(key, []) if isinstance(clue.get(key), list) else [clue.get(key, "")]
        )
    ).lower()
    ranked: List[tuple[int, str]] = []
    for index, call in enumerate(calls):
        tail = call.split(".")[-1]
        score = index
        if tail.lower() in issue_text or call.lower() in issue_text:
            score += 10
        if tail.lower() in function_tails:
            score += 25
        if tail.lower() in helper_names:
            score -= 50
        ranked.append((score, call))
    if ranked:
        return sorted(ranked, key=lambda item: item[0], reverse=True)[0][1]
    function_candidates: List[tuple[int, str]] = []
    for index, fn in enumerate(functions):
        name = str(fn)
        tail = name.split(".")[-1].lower()
        score = -index
        if tail in issue_text or name.lower() in issue_text:
            score += 10
        if tail in helper_names or tail in prohibited_tails:
            score -= 50
        function_candidates.append((score, name))
    for _score, fn in sorted(function_candidates, key=lambda item: item[0], reverse=True):
        if str(fn).split(".")[-1].lower() not in helper_names | prohibited_tails:
            return str(fn)
    return next(
        (
            str(function)
            for function in functions
            if str(function).split(".")[-1].lower() not in prohibited_tails
        ),
        "",
    )


def _select_consensus_answer_key(sharp_marginals: Mapping[str, float]) -> str | None:
    if not sharp_marginals:
        return None
    return sorted(
        sharp_marginals,
        key=lambda key: (-float(sharp_marginals[key]), key),
    )[0]


def _m3_stage_timeout_sec(client: LLMClient) -> float:
    raw = os.environ.get(M3_STAGE_TIMEOUT_ENV)
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            logger.warning("Ignoring invalid %s=%r", M3_STAGE_TIMEOUT_ENV, raw)
    config = getattr(client, "config", None)
    return float(getattr(config, "timeout", 120) or 120)


def _m3_stage_max_model_calls(default_n_max: int) -> int:
    raw = os.environ.get(M3_STAGE_MAX_CALLS_ENV)
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            logger.warning("Ignoring invalid %s=%r", M3_STAGE_MAX_CALLS_ENV, raw)
    return int(default_n_max)


def _m3_nonadaptive_max_tokens(client: LLMClient) -> int:
    raw = os.environ.get(M3_NONADAPTIVE_MAX_TOKENS_ENV)
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            logger.warning("Ignoring invalid %s=%r", M3_NONADAPTIVE_MAX_TOKENS_ENV, raw)
    configured = int(getattr(getattr(client, "config", None), "max_tokens", 1024) or 1024)
    return max(configured, M3_NONADAPTIVE_DEFAULT_MAX_TOKENS)


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _m3_validation_summary(attempt: ScenarioSamplingAttempt) -> Dict[str, Any]:
    return {
        "valid": attempt.valid,
        "rollback_reason": attempt.rollback_reason,
        "blocking_oracle_flags": list(attempt.blocking_oracle_flags or []),
        "has_required_fields": attempt.validated is not None,
        "answer_key": attempt.answer_key,
    }


def _split_ranked_source_snippets(snippet_section: str) -> List[str]:
    chunks = re.split(r"(?m)(?=^####\s+)", snippet_section)
    snippets = [chunk for chunk in chunks if chunk.strip()]
    return snippets or [snippet_section]


def _truncate_source_snippet(
    snippet: str,
    *,
    keep_top_context: bool,
    reduction: float = 0.35,
) -> str:
    if "```" not in snippet:
        return snippet if keep_top_context else ""
    lines = snippet.splitlines()
    kept: List[str] = []
    in_code = False
    code_lines: List[str] = []
    for line in lines:
        if line.startswith("```"):
            if in_code:
                limit = max(12 if keep_top_context else 0, int(len(code_lines) * reduction))
                kept.extend(code_lines[:limit])
                if len(code_lines) > limit:
                    kept.append("# ... truncated lower-ranked repository context ...")
                code_lines = []
                kept.append(line)
                in_code = False
            else:
                kept.append(line)
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
        else:
            kept.append(line)
    if in_code:
        limit = max(12 if keep_top_context else 0, int(len(code_lines) * reduction))
        kept.extend(code_lines[:limit])
        if len(code_lines) > limit:
            kept.append("# ... truncated lower-ranked repository context ...")
    return "\n".join(kept).rstrip() + "\n"


def _compact_text_tail_first(text: str, safe_tokens: int) -> str:
    if estimate_prompt_tokens(text) <= safe_tokens:
        return text
    lines = text.splitlines()
    kept: List[str] = []
    for line in lines:
        candidate = "\n".join(kept + [line])
        if estimate_prompt_tokens(candidate) > safe_tokens:
            break
        kept.append(line)
    return "\n".join(kept).rstrip()


class ScenarioGenerator:

    SYSTEM_PROMPT = (
        "You are a software test scenario planner. "
        "Analyze the issue and generate structured test scenarios. "
        "Return JSON only."
    )

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        model_key: str = "qwen",
        feature_profile: str | None = None,
    ) -> None:
        self.client = client or LLMClient(load_model_config(model_key))
        self.feature_profile = feature_profile
        self._last_adaptive_result: AdaptiveScenarioSamplingResult | None = None
        self._last_nonadaptive_metadata: Dict[str, Any] | None = None
        self._last_parse_diagnostics: Dict[str, Any] = {}

    def extract(
        self,
        instance: Any,
        clue: Dict[str, Any],
        context: Dict[str, Any],
        feature_flags: V22FeatureFlags | Mapping[str, Any] | None = None,
        feedback: Mapping[str, Any] | None = None,
    ) -> List[TestScenario]:
        flags = (
            feature_flags if isinstance(feature_flags, V22FeatureFlags)
            else resolve_feature_flags(feature_flags)
        )
        self._last_adaptive_result = None
        self._last_nonadaptive_metadata = None
        self._last_parse_diagnostics = {}
        self._active_v26_feedback = dict(feedback or {})
        if self.feature_profile in {"v36", "v37"}:
            return self._extract_v36(instance, clue, context, feedback=feedback)
        timings: Dict[str, float] = {}
        prompt_t0 = time.monotonic()
        prompt = self._build_v26_prompt(instance, clue, context, feedback=feedback)
        timings["prompt_build_sec"] = round(time.monotonic() - prompt_t0, 3)
        raw_response = ""
        model_t0 = time.monotonic()
        original_max_tokens = getattr(getattr(self.client, "config", None), "max_tokens", None)
        requested_max_tokens = _m3_nonadaptive_max_tokens(self.client)
        scenarios: List[TestScenario] = []
        schema_errors: List[str] = []
        model_call_count = 0
        oracle_regeneration_count = 0
        blocking_oracle_flags: List[str] = []
        try:
            if hasattr(getattr(self.client, "config", None), "max_tokens"):
                self.client.config.max_tokens = requested_max_tokens
            for schema_attempt in range(1, 4):
                current_prompt = prompt
                if schema_errors:
                    current_prompt += (
                        "\n\n[Schema rollback]\n"
                        f"Previous response was invalid: {schema_errors[-1]}. "
                        "Return exactly one object matching the five-field schema."
                    )
                raw_response = self.client.generate(
                    current_prompt,
                    system_prompt=self.SYSTEM_PROMPT,
                    prompt_compactor=self._compact_m3_prompt,
                )
                model_call_count += 1
                scenarios, schema_error = self._parse_v26_response(raw_response, clue, context)
                if not schema_error:
                    break
                schema_errors.append(schema_error)
            if scenarios:
                blocking_oracle_flags = validate_m3_oracle_expected(
                    scenarios[0].to_dict(),
                    clue,
                )
                fixed_fields = self._v26_fixed_scenario_fields(scenarios[0])
                for oracle_attempt in range(1, 3):
                    if not blocking_oracle_flags:
                        break
                    oracle_prompt = (
                        prompt
                        + "\n\n[Oracle-only rollback]\n"
                        + "The prior oracle failed these deterministic checks: "
                        + json.dumps(blocking_oracle_flags, ensure_ascii=False)
                        + ". Return the same exact five-field object, changing ONLY "
                        + "oracle_expected so it is grounded in EB. Preserve target_function, "
                        + "source_file, oracle_type, and stimulus_steps exactly.\n"
                        + json.dumps(fixed_fields, ensure_ascii=False, sort_keys=True)
                    )
                    raw_response = self.client.generate(
                        oracle_prompt,
                        system_prompt=self.SYSTEM_PROMPT,
                        prompt_compactor=self._compact_m3_prompt,
                    )
                    model_call_count += 1
                    oracle_regeneration_count += 1
                    revised, revision_error = self._parse_v26_response(raw_response, clue, context)
                    if revision_error or not revised:
                        blocking_oracle_flags = [revision_error or "oracle_regeneration_empty"]
                        continue
                    revised_fixed = self._v26_fixed_scenario_fields(revised[0])
                    if revised_fixed != fixed_fields:
                        blocking_oracle_flags = ["oracle_only_regeneration_changed_fixed_fields"]
                        continue
                    revised_flags = validate_m3_oracle_expected(revised[0].to_dict(), clue)
                    scenarios = revised
                    blocking_oracle_flags = revised_flags
        finally:
            if original_max_tokens is not None and hasattr(getattr(self.client, "config", None), "max_tokens"):
                self.client.config.max_tokens = original_max_tokens
        timings["model_request_sec"] = round(time.monotonic() - model_t0, 3)
        parse_t0 = time.monotonic()
        if not scenarios or blocking_oracle_flags:
            scenarios = self._build_fallback_scenarios(clue, context, feedback)[:1]
            for scenario in scenarios:
                scenario.fallback_used = True
                scenario.fallback_reason = (
                    "v26_oracle_regeneration_exhausted"
                    if blocking_oracle_flags
                    else "v26_schema_retries_exhausted"
                )
            self._last_parse_diagnostics = {
                "parse_status": "FAILED",
                "fallback_used": True,
                "failure_kind": (
                    "v26_oracle_regeneration_exhausted"
                    if blocking_oracle_flags
                    else "v26_schema_retries_exhausted"
                ),
                "schema_errors": schema_errors,
                "blocking_oracle_flags": blocking_oracle_flags,
            }
        timings["parse_and_fallback_sec"] = round(time.monotonic() - parse_t0, 3)
        timings["deterministic_repair_probe_sec"] = 0.0
        probe_status = {
            "status": "SKIPPED",
            "reason": "nonadaptive_m3_stage_deadline_protection",
            "method_available": True,
        }
        scenarios = [
            self._dict_to_hydrated_scenario(
                s.to_dict(),
                clue=clue,
                context=context,
                repo=getattr(instance, "repo", ""),
            )
            for s in scenarios[:1]
        ]
        total_elapsed = round(sum(timings.values()), 3)
        self._last_nonadaptive_metadata = {
            "schema_version": "m3-v26-single-scenario-diagnostics-v1",
            "stage": "M3",
            "prompt_hash": _prompt_hash(prompt),
            "prompt_token_estimate": estimate_prompt_tokens(self.SYSTEM_PROMPT)
            + estimate_prompt_tokens(prompt),
            "raw_response": raw_response,
            "raw_response_chars": len(raw_response or ""),
            "finish_reason": getattr(self.client, "last_finish_reason", "") or "",
            "token_usage": dict(getattr(self.client, "last_usage", {}) or {}),
            "stage_budget": {
                "max_model_calls": 5,
                "model_call_count": model_call_count,
                "schema_rollback_limit": 2,
                "oracle_only_rollback_limit": 2,
                "oracle_regeneration_count": oracle_regeneration_count,
                "time_affects_control_flow": False,
                "legacy_adaptive_flag_ignored": bool(flags.m3_adaptive_self_consistency),
            },
            "requested_max_output_tokens": requested_max_tokens,
            "configured_max_output_tokens": original_max_tokens,
            "durations": {
                **timings,
                "fallback_construction_sec": float(
                    self._last_parse_diagnostics.get("fallback_construction_sec", 0.0)
                ),
                "total_nonadaptive_sec": total_elapsed,
            },
            "deterministic_probe": probe_status,
            "parse_diagnostics": dict(self._last_parse_diagnostics),
            "fallback_used": any(bool(getattr(scenario, "fallback_used", False)) for scenario in scenarios),
            "model_call_count": model_call_count,
            "schema_errors": schema_errors,
            "blocking_oracle_flags": blocking_oracle_flags,
            "duplicate_or_overlapping_requests_prevented": True,
        }
        return scenarios[:1]

    def _extract_v36(
        self,
        instance: Any,
        clue: Dict[str, Any],
        context: Dict[str, Any],
        *,
        feedback: Mapping[str, Any] | None,
    ) -> List[TestScenario]:
        """Generate one strict response containing two or three v36 scenarios."""
        prompt = self._build_v36_prompt(instance, clue, context, feedback=feedback)
        raw_response = self.client.generate(
            prompt,
            system_prompt=self.SYSTEM_PROMPT,
            prompt_compactor=self._compact_m3_prompt,
        )
        scenarios, error = self._parse_v36_response(raw_response, clue, context)
        self._last_parse_diagnostics = {
            "parse_status": "PASS" if not error else "FAILED",
            "failure_kind": error or None,
            "fallback_used": False,
            "schema_errors": [error] if error else [],
            "blocking_oracle_flags": [],
            "transport_normalization": getattr(
                self, "_last_v37_transport_normalization", "NONE"
            ),
        }
        self._last_nonadaptive_metadata = {
            "schema_version": "m3-v36-scenario-set-diagnostics-v1",
            "stage": "M3",
            "prompt_hash": _prompt_hash(prompt),
            "raw_response": raw_response,
            "model_call_count": 1,
            "fallback_used": False,
            "parse_diagnostics": dict(self._last_parse_diagnostics),
        }
        return [
            self._dict_to_hydrated_scenario(
                scenario.to_dict(),
                clue=clue,
                context=context,
                repo=getattr(instance, "repo", ""),
            )
            for scenario in scenarios
        ]

    def _build_v36_prompt(
        self,
        instance: Any,
        clue: Mapping[str, Any],
        context: Mapping[str, Any],
        *,
        feedback: Mapping[str, Any] | None,
    ) -> str:
        suspicious = list(
            context.get("suspicious_functions")
            or context.get("top5_functions")
            or context.get("initial_suspicious_functions")
            or []
        )[:5]
        v36_feedback = {
            key: (feedback or {}).get(key)
            for key in (
                "why_failed",
                "fix_suggestion",
                "assumption_gap",
                "route_destination",
            )
            if (feedback or {}).get(key) not in (None, "", [])
        }
        is_v37 = self.feature_profile == "v37"
        payload = {
            "schema_version": (
                "m3-v37-scenario-set-v1" if is_v37 else "m3-v36-scenario-set-v1"
            ),
            "task": "Generate 2 or 3 diverse issue-reproducing test scenarios in one JSON array.",
            "repository": getattr(instance, "repo", ""),
            "OB_do_not_use_as_expected": clue.get("OB") or clue.get("observed_behavior"),
            "EB_oracle_grounding": clue.get("EB") or clue.get("expected_behavior") or clue.get("expected_outputs"),
            "S2R_stimulus_grounding": clue.get("S2R") or clue.get("steps_to_reproduce") or clue.get("repro_conditions"),
            "allowed_m2_suspicious_functions": suspicious,
            "fault_hypothesis": context.get("fault_hypothesis"),
            "oracle_hint": context.get("oracle_hint"),
            "rerun_feedback": v36_feedback,
            "required_item_keys": [
                "target_function",
                "source_file",
                "oracle_type",
                "oracle_expected",
                "stimulus_steps",
                *(["confidence_score"] if is_v37 else []),
            ],
            "rules": [
                "Return JSON only: an array of exactly 2 or 3 objects.",
                "Every target_function/source_file pair must be one of allowed_m2_suspicious_functions.",
                "Ground oracle_expected only in EB and stimulus_steps only in S2R.",
                "Make the scenarios behaviorally distinct.",
                *(
                    [
                        "confidence_score is an integer from 1 through 5 indicating likelihood of reproducing the bug."
                    ]
                    if is_v37
                    else []
                ),
            ],
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)

    def _parse_v36_response(
        self,
        raw_response: str,
        clue: Dict[str, Any],
        context: Dict[str, Any],
    ) -> tuple[List[TestScenario], str]:
        text = str(raw_response or "").strip()
        transport_normalization = "NONE"
        if self.feature_profile == "v37":
            text, transport_normalization = normalize_v37_json_transport(text)
        self._last_v37_transport_normalization = transport_normalization
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return [], f"malformed_json:{exc.msg}"
        if not isinstance(data, list) or len(data) not in {2, 3}:
            return [], "v36_requires_two_or_three_scenarios"
        allowed_pairs: set[tuple[str, str]] = set()
        for item in (
            context.get("suspicious_functions")
            or context.get("top5_functions")
            or context.get("initial_suspicious_functions")
            or []
        ):
            if not isinstance(item, Mapping):
                continue
            path = str(item.get("file_path") or item.get("path") or item.get("source_file") or "")
            if not path:
                continue
            normalized_path = path.replace("\\", "/")
            names = {
                str(item.get("function_name") or ""),
                str(item.get("qualified_name") or ""),
                str(item.get("target_function") or ""),
            }
            names.discard("")
            for name in list(names):
                if "." in name:
                    names.add(name.rsplit(".", 1)[-1])
            for name in names:
                allowed_pairs.add((normalized_path, name))
        if not allowed_pairs:
            return [], "v36_m2_suspicious_function_set_unavailable"
        parsed: List[TestScenario] = []
        seen: set[str] = set()
        for index, item in enumerate(data, start=1):
            if not isinstance(item, Mapping):
                return [], f"scenario_{index}_must_be_object"
            path = str(item.get("source_file") or "").replace("\\", "/")
            name = str(item.get("target_function") or "")
            if (path, name) not in allowed_pairs:
                return [], f"scenario_{index}_target_not_in_m2_top5"
            parse_item = dict(item)
            if self.feature_profile == "v37":
                parse_item.pop("confidence_score", None)
            hydrated, error = self._parse_v26_response(
                json.dumps(parse_item, ensure_ascii=False),
                clue,
                context,
                restrict_oracle_type=False,
            )
            if error or len(hydrated) != 1:
                return [], f"scenario_{index}:{error or 'hydration_failed'}"
            scenario = hydrated[0]
            if self.feature_profile == "v37":
                confidence_score, confidence_provenance = normalize_v37_confidence_score(
                    item.get("confidence_score")
                )
                scenario.confidence_score = confidence_score
                scenario.confidence_score_provenance = confidence_provenance
                scenario.blocking_oracle_flags = (
                    []
                    if str(item.get("oracle_expected") or "").strip()
                    else [ORACLE_EXPECTED_MISSING]
                )
            target = scenario.target_location or {}
            if (
                str(target.get("source_file") or "").replace("\\", "/") != path
                or str(target.get("target_function") or "") != name
            ):
                return [], f"scenario_{index}_target_was_mutated"
            fingerprint_payload = dict(item)
            if self.feature_profile == "v37":
                fingerprint_payload.pop("confidence_score", None)
            fingerprint = json.dumps(
                fingerprint_payload, ensure_ascii=False, sort_keys=True
            )
            if fingerprint in seen:
                return [], "v36_scenarios_must_be_distinct"
            seen.add(fingerprint)
            scenario.scenario_id = f"S{index}"
            parsed.append(scenario)
        return parsed, ""

    @staticmethod
    def _v26_fixed_scenario_fields(scenario: TestScenario) -> Dict[str, Any]:
        """Return the four M3 fields that an oracle-only retry cannot change."""
        payload = scenario.to_dict()
        target = payload.get("target_location") or {}
        return {
            "target_function": payload.get("target_function")
            or target.get("target_function", ""),
            "source_file": payload.get("source_file") or target.get("source_file", ""),
            "oracle_type": payload.get("oracle_type", ""),
            "stimulus_steps": list(
                payload.get("stimulus_steps")
                or payload.get("execution_stimulus")
                or []
            ),
        }

    def _extract_adaptive(
        self,
        instance: Any,
        clue: Dict[str, Any],
        context: Dict[str, Any],
        flags: V22FeatureFlags,
        *,
        feedback: Mapping[str, Any] | None = None,
    ) -> List[TestScenario]:
        result = self.sample_adaptive(instance, clue, context, feedback=feedback)
        metadata = dict(result.metadata or {})
        metadata.update({
            "canonical_feature_flags": flags.to_dict(),
            "config_id": flags.config_id,
        })
        result.metadata = metadata
        scenarios = [
            self._dict_to_hydrated_scenario(
                {**scenario, "generation_provenance": scenario.get("generation_provenance", "adaptive_model_generated")},
                clue=clue,
                context=context,
                repo=getattr(instance, "repo", ""),
            )
            for scenario in result.scenarios
        ]
        self._last_adaptive_result = result
        return scenarios[:3]

    def sample_adaptive(
        self,
        instance: Any,
        clue: Dict[str, Any],
        context: Dict[str, Any],
        *,
        oracle_expected_regenerator: OracleExpectedRegenerator | None = None,
        controller: M3AdaptiveSamplingController | None = None,
        feedback: Mapping[str, Any] | None = None,
    ) -> AdaptiveScenarioSamplingResult:
        """Opt-in M3 adaptive self-consistency hook.

        The default pipeline still calls `extract()`. Integration can activate
        this method once the shared M3 output contract and feature routing land.
        """
        prompt = self._build_prompt(instance, clue, context, feedback=feedback)
        active_controller = controller or M3AdaptiveSamplingController()
        max_model_calls = min(
            M3_OPTIMIZED_MODEL_CALL_LIMIT,
            active_controller.n_max,
            _m3_stage_max_model_calls(M3_OPTIMIZED_MODEL_CALL_LIMIT),
        )
        stage_budget = M3StageCallBudget(
            stage="M3",
            timeout_sec=_m3_stage_timeout_sec(self.client),
            max_model_calls=max_model_calls,
        )
        attempts: List[ScenarioSamplingAttempt] = []
        valid_scenarios: List[Dict[str, Any]] = []
        answer_keys: List[str] = []
        rollback_count = 0
        rollback_reasons: List[str] = []
        failed_parse_count = 0
        blocking_oracle_count = 0
        seen_prompt_hashes: set[str] = set()
        stop_reason = M3_STOP_ATTEMPT_BUDGET_EXHAUSTED

        def run_model_call(call_index: int, call_purpose: str, call_prompt: str) -> ScenarioSamplingAttempt | None:
            nonlocal rollback_count, failed_parse_count, blocking_oracle_count, stop_reason
            prompt_hash = _prompt_hash(call_prompt)
            if prompt_hash in seen_prompt_hashes:
                return None
            seen_prompt_hashes.add(prompt_hash)
            per_call_timeout = stage_budget.before_call(
                call_index,
                configured_timeout=float(
                    getattr(getattr(self.client, "config", None), "timeout", stage_budget.timeout_sec)
                ),
            )
            t0 = time.monotonic()
            try:
                raw = self.client.generate(
                    call_prompt,
                    system_prompt=self.SYSTEM_PROMPT,
                    prompt_compactor=self._compact_m3_prompt,
                    timeout=per_call_timeout,
                )
            except ModelTimeoutError:
                stage_budget.raise_timeout("M3 adaptive model call timed out")
            latency_sec = time.monotonic() - t0
            stage_budget.record_completed_call(latency_sec)
            attempt = self._validate_m3_attempt(
                call_index,
                raw,
                clue,
                oracle_expected_regenerator=oracle_expected_regenerator,
            )
            attempts.append(attempt)
            if attempt.valid:
                valid_scenarios.append(dict(attempt.validated or {}))
                if attempt.answer_key:
                    answer_keys.append(attempt.answer_key)
            else:
                rollback_count += 1
                failed_parse_count += 1 if attempt.parsed is None else 0
                if attempt.rollback_reason:
                    rollback_reasons.append(attempt.rollback_reason)
                if (
                    attempt.parsed is None
                    and failed_parse_count > active_controller.max_rollback_attempts
                ):
                    stop_reason = M3_STOP_MALFORMED_JSON_RETRY_EXHAUSTED
                if attempt.blocking_oracle_flags:
                    blocking_oracle_count += 1
                    if blocking_oracle_count > active_controller.max_rollback_attempts:
                        stop_reason = M3_STOP_ORACLE_ROLLBACK_RETRY_EXHAUSTED

            call_record = {
                "call_index": call_index,
                "call_purpose": call_purpose,
                "prompt_hash": prompt_hash,
                "prompt_token_estimate": estimate_prompt_tokens(self.SYSTEM_PROMPT)
                + estimate_prompt_tokens(call_prompt),
                "latency_sec": round(latency_sec, 3),
                "completion_tokens": int(getattr(self.client, "last_usage", {}).get("completion_tokens", 0)),
                "finish_reason": getattr(self.client, "last_finish_reason", "") or "",
                "validation_result": _m3_validation_summary(attempt),
            }
            stage_budget.record_call(call_record)
            if stage_budget.remaining <= 0:
                stage_budget.raise_timeout("M3 adaptive stage deadline exhausted after model call")
            return attempt

        fallback = [
            scenario.to_dict()
            for scenario in self._build_fallback_scenarios(
                clue, context, getattr(self, "_active_v26_feedback", None)
            )
        ]
        try:
            sample_index = 0
            while sample_index < active_controller.n_max:
                if stage_budget.model_call_count >= stage_budget.max_model_calls:
                    stop_reason = M3_STOP_CALL_BUDGET_EXHAUSTED
                    break
                sample_index += 1
                if sample_index == 1:
                    call_purpose = "PRIMARY"
                    call_prompt = prompt
                elif attempts and not attempts[-1].valid and sample_index == 2:
                    call_purpose = "REFINEMENT"
                    call_prompt = self._build_m3_refinement_prompt(
                        prompt,
                        attempts[-1],
                        raw_response=attempts[-1].raw_response,
                    )
                else:
                    call_purpose = "ADAPTIVE_SAMPLE"
                    call_prompt = self._build_m3_adaptive_sample_prompt(prompt, sample_index)
                attempt = run_model_call(sample_index, call_purpose, call_prompt)
                if attempt is None:
                    stop_reason = M3_STOP_CALL_BUDGET_EXHAUSTED
                    break
                if stop_reason == M3_STOP_MALFORMED_JSON_RETRY_EXHAUSTED:
                    break
                if stop_reason == M3_STOP_ORACLE_ROLLBACK_RETRY_EXHAUSTED:
                    break
                if (
                    len(valid_scenarios) >= active_controller.n_min
                    and len(answer_keys) >= active_controller.consecutive_cluster_stop
                    and len(set(answer_keys[-active_controller.consecutive_cluster_stop:])) == 1
                ):
                    stop_reason = M3_STOP_EARLY_STOPPED
                    break
        except ModelStageTimeoutError:
            raise
        fallback_used = False
        if not valid_scenarios:
            for fallback_scenario in fallback:
                try:
                    validated = _validate_m3_schema(fallback_scenario)
                except ValueError as exc:
                    rollback_reasons.append(f"fallback_schema_violation: {exc}")
                    continue
                answer_key = build_answer_key(validated)
                validated["answer_key"] = answer_key
                valid_scenarios.append(validated)
                fallback_used = True
                break

        consensus = build_consensus(answer_keys)
        if len(valid_scenarios) < active_controller.n_min:
            consensus["Consensus_norm"] = None
            consensus["stability"] = "unavailable"
            consensus["diagnostic_band"] = "insufficient_valid_samples"
        consensus["early_stopped"] = stop_reason == M3_STOP_EARLY_STOPPED
        result = AdaptiveScenarioSamplingResult(
            scenarios=valid_scenarios,
            attempts=attempts,
            sample_count=len(answer_keys),
            answer_key=answer_keys[0] if len(answer_keys) >= active_controller.n_min else None,
            consensus=consensus,
            validated_scenarios=valid_scenarios,
            rollback_count=rollback_count,
            rollback_reasons=rollback_reasons,
            early_stopped=consensus["early_stopped"],
            valid_sample_count=len(answer_keys),
            failed_parse_count=failed_parse_count,
            total_attempt_count=len(attempts),
            termination_reason=stop_reason,
            adaptive_sampling_status=(
                M3_ADAPTIVE_STATUS_CONSENSUS_VALID
                if len(answer_keys) >= active_controller.n_min
                else M3_ADAPTIVE_STATUS_INSUFFICIENT_VALID_SAMPLES
                if fallback_used
                else M3_ADAPTIVE_STATUS_SKIPPED_FALLBACK
            ),
            fallback_used=fallback_used,
        )
        result.metadata = dict(result.metadata or {})
        result.metadata["m3_stage_budget"] = stage_budget.to_dict()
        result.metadata["m3_call_policy"] = {
            "max_model_calls": stage_budget.max_model_calls,
            "stop_reason": stop_reason,
            "calls": list(stage_budget.call_records),
        }
        return result

    @staticmethod
    def _build_m3_adaptive_sample_prompt(primary_prompt: str, sample_index: int) -> str:
        return (
            primary_prompt
            + "\n\nAdaptive self-consistency sample "
            + str(sample_index)
            + ": produce an independent JSON scenario using the same issue evidence. "
            + "Do not reuse observed buggy output as the expected oracle."
        )

    def _validate_m3_attempt(
        self,
        attempt_index: int,
        raw_response: str,
        clue: Mapping[str, Any],
        *,
        oracle_expected_regenerator: OracleExpectedRegenerator | None = None,
    ) -> ScenarioSamplingAttempt:
        attempt = ScenarioSamplingAttempt(
            attempt_index=attempt_index,
            raw_response=raw_response,
        )
        try:
            parsed = _parse_strict_scenario_response(raw_response)
            attempt.parsed = parsed
            validated = _validate_m3_schema(parsed[0])
        except (json.JSONDecodeError, ValueError) as exc:
            attempt.rollback_reason = f"schema_or_json_violation: {exc}"
            return attempt

        flags = validate_m3_oracle_expected(validated, clue)
        if flags and oracle_expected_regenerator is not None:
            for flag in flags[:M3_MAX_ROLLBACK_ATTEMPTS]:
                validated["oracle_expected"] = oracle_expected_regenerator(dict(validated), flag, attempt_index)
                flags = validate_m3_oracle_expected(validated, clue)
                if not flags:
                    break
        attempt.blocking_oracle_flags = flags
        if flags:
            attempt.rollback_reason = "blocking_oracle_flags: " + ",".join(flags)
            return attempt

        answer_key = build_answer_key(validated)
        validated["answer_key"] = answer_key
        attempt.validated = validated
        attempt.valid = True
        attempt.answer_key = answer_key
        return attempt

    def _build_m3_refinement_prompt(
        self,
        primary_prompt: str,
        primary_attempt: ScenarioSamplingAttempt | None,
        *,
        raw_response: str,
    ) -> str:
        validation = _m3_validation_summary(primary_attempt) if primary_attempt else {
            "valid": False,
            "rollback_reason": "primary call did not produce a validation attempt",
            "blocking_oracle_flags": [],
            "has_required_fields": False,
            "answer_key": None,
        }
        response_excerpt = raw_response[:4000]
        return (
            f"{primary_prompt}\n\n"
            "Refinement required for the previous response.\n"
            "Return JSON only using the exact same scenario schema.\n"
            f"Validation result: {json.dumps(validation, ensure_ascii=False, sort_keys=True)}\n"
            f"Previous response excerpt:\n{response_excerpt}"
        )

    # ── Probe execution ──

    def _enrich_with_probe(
        self,
        scenarios: List[TestScenario],
        instance: Any,
        context: Dict[str, Any],
    ) -> List[TestScenario]:
        """Top 시나리오에 대해 probe test를 실행하여 actual_outputs를 채운다."""
        if not scenarios:
            return scenarios
        top = scenarios[0]
        if top.actual_outputs:
            return scenarios  # 이미 있으면 skip

        repro = top.reproduction_code or []
        if not repro:
            return scenarios
        code = repro[0].get("code", "") if isinstance(repro[0], dict) else str(repro[0])
        if not code.strip():
            return scenarios

        probe_code = self._transform_to_probe(code)
        if not probe_code:
            return scenarios

        target_test_file = top.target_location.get("candidate_test_file", "")
        if not target_test_file:
            return scenarios

        # repo_path 결정: data/repos/{repo_owner}/{repo_name}
        # instance.instance_id 형식: "owner__repo-number" or instance.raw["repo"] = "owner/repo"
        try:
            repo_slug = getattr(instance, "repo", "") or ""
            if repo_slug:
                repo_path = Path("data/repos") / repo_slug.replace("/", "__")
                if not repo_path.exists():
                    repo_path = Path("data/repos") / repo_slug
            else:
                repo_path = Path("data/repos") / instance.instance_id.rsplit("-", 1)[0]
        except Exception:
            return scenarios

        if not repo_path.exists():
            logger.debug("probe: repo_path not found: %s", repo_path)
            return scenarios

        try:
            probe_result = self._run_probe(probe_code, instance, target_test_file, repo_path)
        except Exception as e:
            logger.debug("probe run failed: %s", e)
            return scenarios

        raw_output = probe_result.get("raw_output", "") if isinstance(probe_result, dict) else (probe_result or "")
        probe_cov = probe_result.get("coverage_data", {}) if isinstance(probe_result, dict) else {}

        if not raw_output:
            return scenarios

        # actual_outputs 채우기
        actual = self._parse_probe_output(raw_output)
        if actual:
            top.actual_outputs = [actual]
            logger.info("probe enriched actual_outputs for %s: %s", top.scenario_id, actual[:120])

        # Fault location 검증: target_source_file이 probe에서 실행됐는지 확인
        target_src = top.target_location.get("source_file", "")
        if target_src and probe_cov:
            src_covered = self._check_source_covered(probe_cov, target_src)
            if not src_covered:
                logger.warning(
                    "probe: source file '%s' NOT covered — fault localization may be wrong. "
                    "Trying alternative scenario.",
                    target_src,
                )
                # 다른 시나리오 중 더 나은 것으로 교체 시도
                for i, alt in enumerate(scenarios[1:], 1):
                    alt_src = alt.target_location.get("source_file", "")
                    if alt_src and alt_src != target_src:
                        # 대안 시나리오가 다른 파일을 타겟한다면 순서 교체
                        scenarios[0], scenarios[i] = scenarios[i], scenarios[0]
                        logger.info("probe: switched to scenario %s (source: %s)", alt.scenario_id, alt_src)
                        break

        return scenarios

    def _check_source_covered(self, coverage_data: Dict[str, Any], source_file: str) -> bool:
        """probe coverage에서 target source file이 실행됐는지 확인."""
        for fname, info in coverage_data.items():
            if not isinstance(info, dict):
                continue
            if fname.endswith(source_file) or source_file.endswith(fname):
                return info.get("cover", 0) > 5  # 5% 이상이면 실행됐다고 판단
        return False

    def _transform_to_probe(self, code: str) -> str:
        """reproduction_code에서 assertion을 제거하고 결과 캡처 코드로 변환."""
        lines = code.strip().splitlines()
        if not lines:
            return ""

        new_lines: List[str] = []
        func_indent = "    "

        for line in lines:
            # 함수명 변경
            m_def = re.match(r'(\s*def\s+)(test_\w+)(.*)', line)
            if m_def:
                line = m_def.group(1) + "test_probe_capture" + m_def.group(3)
                new_lines.append(line)
                continue

            stripped = line.strip()

            # assert 문 및 assert_*() 헬퍼 호출 제거 (probe가 결과를 캡처하려면 예외가 나면 안 됨)
            if (stripped.startswith("assert ")
                    or re.match(r'assert_\w+\s*\(', stripped)
                    or re.match(r'self\.assert\w*\s*\(', stripped)
                    or re.match(r'np\.testing\.assert\w*\s*\(', stripped)):
                indent = len(line) - len(line.lstrip())
                new_lines.append(" " * indent + "# [probe: assertion removed]")
                continue

            # pytest.raises 블록 제거
            if stripped.startswith("with pytest.raises") or stripped.startswith("with self.assertRaises"):
                indent = len(line) - len(line.lstrip())
                new_lines.append(" " * indent + "# [probe: pytest.raises removed]")
                continue

            # result = func(...) 패턴: __probe_result에 캡처
            m_assign = re.match(r'^(\s*)(\w+)\s*=\s*(.+\(.*)', stripped)
            if (
                m_assign
                and not stripped.startswith(("import ", "from ", "def ", "class "))
            ):
                var_name = m_assign.group(2)
                indent = len(line) - len(line.lstrip())
                new_lines.append(line)
                new_lines.append(" " * indent + f"__probe_result = {var_name}")
                continue

            new_lines.append(line)

        # 함수 마지막에 출력 캡처 블록 추가
        new_lines += [
            f"{func_indent}# --- probe output capture ---",
            f"{func_indent}import sys as __sys",
            f"{func_indent}if '__probe_result' in dir():",
            f"{func_indent}    print('__PROBE_RESULT__:' + repr(__probe_result), file=__sys.stderr)",
        ]
        return "\n".join(new_lines)

    def _run_probe(
        self,
        probe_code: str,
        instance: Any,
        target_test_file: str,
        repo_path: Path,
    ) -> Dict[str, Any]:
        """probe test를 Docker에서 실행하고 raw_output + coverage_data를 반환."""
        import json as _json
        import os
        import tempfile

        from src.executor.alignment_runner import AlignmentRunner
        from src.generator.repro_test_generator import ReproductionTestGenerator

        runner = AlignmentRunner()

        abs_path = repo_path / target_test_file
        original = ""
        if abs_path.exists():
            original = abs_path.read_text(encoding="utf-8", errors="ignore")

        modified = original.rstrip() + "\n\n" + probe_code + "\n"

        # ReproductionTestGenerator의 _build_unified_patch 재사용
        rtg = ReproductionTestGenerator.__new__(ReproductionTestGenerator)
        test_patch = rtg._build_unified_patch(original, modified, target_test_file)

        probe_dict = {
            "instance_id": instance.instance_id,
            "scenario_id": "probe",
            "model_name": "probe",
            "target_test_file": target_test_file,
            "test_patch": test_patch,
            "modified_test_file_content": modified,
            "test_code": probe_code,
            "insert_mode": "append_block",
            "append_block": probe_code,
            "imports": [],
            "original_test_file_content": original,
            "insertion_hint": "",
            "raw_response": "",
            "prompt": "",
            "repo_path": str(repo_path),
            "target_test_file_abspath": str(abs_path),
            "target_source_file": "",
            "token_usage": {},
        }

        tmp_dir = tempfile.TemporaryDirectory(prefix="m3_probe_")
        probe_json_path = os.path.join(tmp_dir.name, "generated_test.json")
        probe_patch_path = os.path.join(tmp_dir.name, "generated_test.patch")
        pre_patch_instance = make_pre_patch_view(instance)

        try:
            with open(probe_json_path, "w", encoding="utf-8") as f:
                _json.dump(probe_dict, f)
            with open(probe_patch_path, "w", encoding="utf-8") as f:
                f.write(test_patch)

            result = runner.run(
                instance=pre_patch_instance,
                generated_test_json_path=probe_json_path,
                run_id=f"probe-{pre_patch_instance.instance_id}",
            )
            return {
                "raw_output": result.raw_output or "",
                "coverage_data": result.coverage_data or {},
            }
        finally:
            tmp_dir.cleanup()

    def _parse_probe_output(self, raw_output: str) -> str:
        """raw Docker output에서 __PROBE_RESULT__: 값 추출."""
        m = re.search(r"__PROBE_RESULT__:(.+)", raw_output)
        if m:
            return m.group(1).strip()[:300]
        return ""

    def save(self, scenarios: List[TestScenario], output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            scenario_dicts = [s.to_dict() for s in scenarios]
            instance_id = str(next((item.get("instance_id") for item in scenario_dicts if item.get("instance_id")), ""))
            outer_iteration = next(
                (item.get("outer_iteration") for item in scenario_dicts if item.get("outer_iteration") is not None),
                None,
            )
            if self._last_adaptive_result is not None:
                artifact = self._last_adaptive_result.to_dict()
                artifact["scenarios"] = scenario_dicts
                artifact.update(
                    instance_id=instance_id,
                    iteration=outer_iteration,
                    outer_iteration=outer_iteration,
                )
                json.dump(artifact, f, ensure_ascii=False, indent=2)
                if path.name == "scenario.json":
                    with open(path.with_name("scenarios.json"), "w", encoding="utf-8") as alias:
                        json.dump(artifact, alias, ensure_ascii=False, indent=2)
                return
            json.dump(scenario_dicts, f, ensure_ascii=False, indent=2)
            if path.name == "scenario.json":
                with open(path.with_name("scenarios.json"), "w", encoding="utf-8") as alias:
                    json.dump(
                        {
                            "schema_version": "m3-nonadaptive-scenarios-v1",
                            "module": "M3",
                            "instance_id": instance_id,
                            "iteration": outer_iteration,
                            "outer_iteration": outer_iteration,
                            "adaptive_self_consistency": False,
                            "model_call_count": 1,
                            "scenarios": scenario_dicts,
                            "metadata": {
                                "scenario_count": len(scenario_dicts),
                                "generation_path": "non_adaptive_extract",
                            },
                        },
                        alias,
                        ensure_ascii=False,
                        indent=2,
                    )

    # ── LLM 기반 fault location 추론 (traceback 없는 경우) ──

    def infer_fault_locations(
        self,
        clue: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """traceback이 없는 이슈에서 소스 코드 스니펫을 분석하여 버그 위치를 추론한다.

        Returns:
            fault_locations 형식의 리스트:
            [{"file_path": "...", "function_name": "...", "line_no": 0, "inferred": True}, ...]
        """
        observed = clue.get("observed_behavior", [])
        expected = clue.get("expected_behavior", [])
        noisy_functions = {
            "arange", "rand", "random", "seed", "platform", "get_backend",
            "show_versions",
        }
        functions = [
            fn for fn in clue.get("identifiers", {}).get("functions", [])
            if fn not in noisy_functions
        ]
        classes = clue.get("identifiers", {}).get("classes", [])
        error_keywords = clue.get("error_keywords", [])

        # 소스 코드 스니펫 수집 (최대 5개 파일)
        snippet_parts = []
        for sf in context.get("candidate_source_files", [])[:5]:
            snippets = sf.get("code_snippets") or {}
            if not snippets:
                continue
            parts = [f"File: {sf['path']}"]
            for ident, snippet in list(snippets.items())[:3]:
                parts.append(f"```python\n{snippet[:800]}\n```")
            snippet_parts.append("\n".join(parts))

        if not snippet_parts:
            return []

        snippet_section = "\n\n".join(snippet_parts)

        prompt = f"""You are analyzing a GitHub issue to identify which function is most likely buggy.

Issue observed behavior: {json.dumps(observed[:3], ensure_ascii=False)}
Issue expected behavior: {json.dumps(expected[:3], ensure_ascii=False)}
Related identifiers — functions: {functions[:8]}, classes: {classes[:8]}
{f"Error keywords: {error_keywords[:5]}" if error_keywords else ""}

Source code snippets from the repository:
{snippet_section}

Based on the issue description and code above, identify the most likely buggy function(s).

Return JSON only:
{{"fault_locations": [{{"file_path": "relative/path/to/file.py", "function_name": "function_name", "line_no": 0}}]}}

Rules:
- Return at most 3 candidates, most likely first.
- Use the exact file paths shown above.
- If you are uncertain, return {{"fault_locations": []}}
- Do NOT include test files.
"""
        try:
            raw = self.client.generate(
                prompt,
                system_prompt="You are a bug localization assistant. Return JSON only.",
                prompt_compactor=self._compact_m3_prompt,
            )
        except Exception as e:
            logger.warning("infer_fault_locations LLM call failed: %s", e)
            return []

        # JSON 파싱
        text = raw.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            obj_match = re.search(r"(\{.*\})", text, re.DOTALL)
            if not obj_match:
                logger.warning("infer_fault_locations: failed to parse LLM response")
                return []
            try:
                data = json.loads(obj_match.group(1))
            except json.JSONDecodeError:
                logger.warning("infer_fault_locations: JSON parse error in extracted object")
                return []

        locations = data.get("fault_locations", [])
        if not isinstance(locations, list):
            return []

        result = []
        for loc in locations[:3]:
            if not isinstance(loc, dict):
                continue
            fp = loc.get("file_path", "")
            fn = loc.get("function_name", "")
            if fp and fn:
                result.append({
                    "file_path": fp,
                    "function_name": fn,
                    "line_no": loc.get("line_no", 0),
                    "inferred": True,
                    "source": "inferred_llm",
                    "confidence": "medium",
                })

        if result:
            logger.info("infer_fault_locations: found %d location(s): %s", len(result), result)
        return result

    # ── LLM 기반 시나리오 생성 ──

    def _build_v26_prompt(
        self,
        instance: Any,
        clue: Dict[str, Any],
        context: Dict[str, Any],
        feedback: Mapping[str, Any] | None = None,
    ) -> str:
        """Build the deterministic v26 one-scenario prompt contract."""
        source_candidates = context.get("candidate_source_files") or []
        selected_source = source_candidates[0] if source_candidates else {}
        source_file = str(selected_source.get("path") or "")
        top_functions = list(selected_source.get("top_level_functions") or [])
        target_function = str(
            (context.get("top_functions") or [{}])[0].get("function_name")
            if isinstance(context.get("top_functions"), list) and context.get("top_functions")
            else top_functions[0]
            if top_functions
            else ""
        )
        snippet_lines: List[str] = []
        for snippet in (selected_source.get("code_snippets") or {}).values():
            snippet_lines.extend(str(snippet).splitlines())
            if len(snippet_lines) >= 50:
                break
        target_source_code = "\n".join(snippet_lines[:50]) or "(source excerpt unavailable)"
        expected = list(clue.get("expected_behavior") or [])
        repro = list(clue.get("steps_to_reproduce") or clue.get("repro_conditions") or [])
        observed = list(clue.get("observed_behavior") or [])
        diagnosis = self._v26_feedback_fields(feedback)
        payload = {
            "schema_version": "m3-v26-prompt-v1",
            "task": "Generate exactly one issue-reproducing test scenario.",
            "repository": getattr(instance, "repo", ""),
            "oracle_specification_expected_behavior": expected,
            "observed_buggy_behavior_do_not_use_as_expected": observed,
            "stimulus_specification_steps_to_reproduce": repro,
            "target_function_specification": {
                "target_function": target_function,
                "source_file": source_file,
                "source_code_max_50_lines": target_source_code,
                "fault_hypothesis": context.get("fault_hypothesis"),
                "relevance": selected_source.get("rel_LLM", selected_source.get("score")),
            },
            "m2_oracle_hint_for_downstream_consistency": context.get("oracle_hint"),
            "m7_feedback_to_consume": diagnosis,
            "rules": [
                "Return one JSON object, not an array.",
                "Use EB for oracle_expected; never use OB as the expected value.",
                "Transform S2R into concrete ordered stimulus_steps.",
                "Use the supplied source and fault hypothesis; do not invent an unrelated target.",
                "When M7 feedback is present, behaviorally apply all four diagnosis fields.",
            ],
            "required_output_schema": {
                "target_function": "string",
                "source_file": "string",
                "oracle_type": "assert_equals|assert_true|assert_throws|assert_false|assert_null|assert_not_null",
                "oracle_expected": "string",
                "stimulus_steps": ["string"],
            },
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)

    @staticmethod
    def _v26_feedback_fields(feedback: Mapping[str, Any] | None) -> Dict[str, Any]:
        if not isinstance(feedback, Mapping):
            return {}
        nested = feedback.get("feedback_decision")
        source = nested if isinstance(nested, Mapping) else feedback
        return {
            key: source.get(key)
            for key in (
                "failure_reason",
                "assumption_gap",
                "next_scenario_change",
                "admissible_alternatives",
            )
            if source.get(key) not in (None, "", [])
        }

    def _parse_v26_response(
        self,
        raw_response: str,
        clue: Dict[str, Any],
        context: Dict[str, Any],
        *,
        restrict_oracle_type: bool = True,
    ) -> tuple[List[TestScenario], str]:
        text = str(raw_response or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return [], f"malformed_json:{exc.msg}"
        required = {
            "target_function",
            "source_file",
            "oracle_type",
            "oracle_expected",
            "stimulus_steps",
        }
        if not isinstance(data, Mapping) or set(data) != required:
            return [], "schema_keys_must_match_exact_v26_contract"
        for key in ("target_function", "source_file", "oracle_expected"):
            if not isinstance(data.get(key), str) or not str(data[key]).strip():
                return [], f"{key}_must_be_nonempty_string"
        allowed_oracles = {
            "assert_equals",
            "assert_true",
            "assert_throws",
            "assert_false",
            "assert_null",
            "assert_not_null",
        }
        if restrict_oracle_type and data.get("oracle_type") not in allowed_oracles:
            return [], "oracle_type_is_not_supported"
        stimulus = data.get("stimulus_steps")
        if (
            not isinstance(stimulus, list)
            or not stimulus
            or any(not isinstance(item, str) or not item.strip() for item in stimulus)
        ):
            return [], "stimulus_steps_must_be_nonempty_string_list"
        test_candidates = context.get("candidate_test_files") or []
        legacy = {
            "scenario_id": "S1",
            "target_location": {
                "source_file": data["source_file"].strip(),
                "target_function": data["target_function"].strip(),
                "related_classes": [],
                "candidate_test_file": str(test_candidates[0].get("path") or "") if test_candidates else "",
                "confidence": "medium",
            },
            "setup_steps": [],
            "execution_stimulus": [item.strip() for item in stimulus],
            "expected_failure": f"Pre-patch behavior must differ from EB oracle: {data['oracle_expected'].strip()}",
            "test_environment": {
                "required_fixtures": [],
                "runner": context.get("project_test_style", {}).get("runner", "pytest"),
            },
            "reproduction_code": list(clue.get("code_examples") or []),
            "identifiers": dict(clue.get("identifiers") or {}),
            "expected_outputs": [data["oracle_expected"].strip()],
            "actual_outputs": list(clue.get("actual_outputs") or []),
            "error_keywords": list(clue.get("error_keywords") or []),
            "oracle_type": data["oracle_type"],
            "oracle_source": "M1_EB",
            "oracle": data["oracle_expected"].strip(),
            "oracle_contract": {
                "oracle_type": data["oracle_type"],
                "oracle_source": "M1_EB",
                "rule": data["oracle_expected"].strip(),
            },
        }
        parsed = self._parse_response(json.dumps(legacy), clue, context)[:1]
        if not parsed:
            return [], "v26_scenario_hydration_failed"
        for scenario in parsed:
            scenario.fault_hypothesis = str(context.get("fault_hypothesis") or "")
            scenario.m2_oracle_hint = str(context.get("oracle_hint") or "")
            scenario.feedback_consumed = self._v26_feedback_fields(
                getattr(self, "_active_v26_feedback", None)
            )
        return parsed, ""

    def _build_prompt(
        self,
        instance: Any,
        clue: Dict[str, Any],
        context: Dict[str, Any],
        feedback: Mapping[str, Any] | None = None,
    ) -> str:
        noisy_functions = {
            "arange", "rand", "random", "seed", "platform", "get_backend",
            "show_versions",
        }
        functions = [
            fn for fn in clue.get("identifiers", {}).get("functions", [])
            if fn not in noisy_functions
        ]
        classes = clue.get("identifiers", {}).get("classes", [])
        error_keywords = clue.get("error_keywords", [])
        observed = clue.get("observed_behavior", [])
        expected = clue.get("expected_behavior", [])
        repro = clue.get("repro_conditions", [])
        raw_issue_text = clue.get("raw_issue_text", "")
        code_examples = clue.get("code_examples", [])
        expected_outputs = clue.get("expected_outputs", [])
        if not expected_outputs and expected:
            expected_outputs = [str(expected[0])]
        actual_outputs = clue.get("actual_outputs", [])
        fault_locations = clue.get("fault_locations", [])

        source_files = [x["path"] for x in context.get("candidate_source_files", [])]
        test_files = [x["path"] for x in context.get("candidate_test_files", [])]
        framework = context.get("project_test_style", {}).get("framework", "unknown")
        runner = context.get("project_test_style", {}).get("runner", "pytest")
        conftest_fixtures = context.get("conftest_fixtures", {})

        # 소스 파일별 실제 공개 함수 목록 (AST 추출, 환각 방지용)
        func_list_lines = []
        for sf in context.get("candidate_source_files", [])[:3]:
            funcs = sf.get("top_level_functions") or []
            if funcs:
                func_list_lines.append(f"  {sf['path']}: {funcs[:20]}")
        func_list_section = (
            "\n[Available Functions per Source File — from AST]\n"
            + "\n".join(func_list_lines)
            + "\n"
        ) if func_list_lines else ""

        # 소스 코드 스니펫 섹션 (matched identifiers의 실제 함수/클래스 정의)
        source_snippet_parts = []
        for sf in context.get("candidate_source_files", [])[:2]:
            snippets = sf.get("code_snippets") or {}
            if not snippets:
                continue
            parts = [f"#### {sf['path']}"]
            for ident, snippet in list(snippets.items())[:2]:
                parts.append(f"```python\n{snippet[:600]}\n```")
            source_snippet_parts.append("\n".join(parts))
        source_snippet_section = "\n\n".join(source_snippet_parts) if source_snippet_parts else "(none available)"

        # 코드 예시 섹션
        code_section = ""
        if code_examples:
            parts = []
            for i, block in enumerate(code_examples):
                if block.get("is_system_or_output"):
                    continue
                code = block.get("code", "")
                ctx = block.get("context_before", "")
                label = f"Code Block {i + 1}"
                if ctx:
                    label += f' (context: "{ctx[:100]}")'
                parts.append(f"### {label}\n```python\n{code}\n```")
            code_section = "\n\n".join(parts)

        expected_output_section = ""
        if expected_outputs:
            expected_output_section = "\n[Expected Correct Output]\n" + "\n".join(
                f"```\n{out[:500]}\n```" for out in expected_outputs[:3]
            )

        actual_output_section = ""
        if actual_outputs:
            actual_output_section = "\n[Actual Buggy Output]\n" + "\n".join(
                f"```\n{out[:500]}\n```" for out in actual_outputs[:3]
            )

        feedback_section = self._build_m7_feedback_section(feedback)

        # 이슈 원문 (truncated)
        raw_issue_section = ""
        if raw_issue_text:
            truncated = raw_issue_text[:800]
            if len(raw_issue_text) > 800:
                truncated += "\n... (truncated)"
            raw_issue_section = f"\n[Issue Description]\n{truncated}\n"

        # fault locations 섹션: real traceback and inferred candidates are separate
        fault_location_section = ""
        if fault_locations:
            traceback_lines = []
            inferred_lines = []
            for fl in fault_locations[:5]:
                fp = fl.get("file_path", "")
                fn = fl.get("function_name", "")
                ln = fl.get("line_no", "?")
                source = fl.get("source", "traceback")
                confidence = fl.get("confidence", "high" if source == "traceback" else "medium")
                # 절대 경로에서 repo 내 상대 경로 추정 (마지막 의미 있는 부분)
                # e.g. /home/user/.../astropy/coordinates/sky_coordinate.py → astropy/coordinates/sky_coordinate.py
                parts = fp.replace("\\", "/").split("/")
                # site-packages 이후 경로는 제외했으므로 그냥 마지막 3~4 segments 사용
                rel_guess = "/".join(parts[-4:]) if len(parts) >= 4 else fp
                line = f"  - {rel_guess}  line {ln}  in {fn}"
                if source == "traceback" and confidence == "high":
                    traceback_lines.append(line)
                else:
                    inferred_lines.append(line)
            sections = []
            if traceback_lines:
                sections.append(
                    "\n[CRITICAL: Fault Locations from Issue Traceback]\n"
                    "These locations were explicitly identified in the issue's stack trace.\n"
                    "They are HIGH-CONFIDENCE indicators of where the bug lives.\n"
                    + "\n".join(traceback_lines)
                    + "\n"
                    "→ S1 MUST use one of these as target_location (source_file + target_function).\n"
                    "→ If the function listed here is a private helper (starts with _), use the\n"
                    "  closest public caller function visible in [Source Code Snippets] instead.\n"
                )
            if inferred_lines:
                sections.append(
                    "\n[Inferred Fault Location Candidates]\n"
                    "These are MEDIUM-CONFIDENCE guesses from issue text and source snippets.\n"
                    "Use them only if they agree with the issue behavior and available functions.\n"
                    + "\n".join(inferred_lines)
                    + "\n"
                )
            fault_location_section = "".join(sections)

        prompt = f"""Generate 1-3 bug reproduction test scenarios for the issue below. Return JSON array only.

Repository: {instance.repo}

[Issue Clue]
Observed: {json.dumps(observed[:3], ensure_ascii=False)}
Expected: {json.dumps(expected[:3], ensure_ascii=False)}
Repro: {json.dumps(repro[:3], ensure_ascii=False)}
Functions: {functions[:8]}  Classes: {classes[:8]}
{f"Errors: {error_keywords[:5]}" if error_keywords else ""}{raw_issue_section}
{fault_location_section}
{f"[Issue Code Examples]{chr(10)}{code_section}" if code_section else ""}
{expected_output_section}
{actual_output_section}
{feedback_section}

[Code Context]
Test framework: {framework}
Test runner: {runner}
Candidate source files: {source_files[:5]}
Candidate test files: {test_files[:5]}
{self._build_conftest_section(conftest_fixtures)}
[Source Code Snippets - Use these to write concrete execution_stimulus and setup_steps]
{source_snippet_section}

[Output Format]
Return a JSON array of 1-3 meaningfully different scenario objects. Each object must have exactly these fields:
{{
  "scenario_id": "S1",
  "target_location": {{
    "source_file": "path/to/source.py",
    "target_function": "function_name",
    "related_classes": ["ClassName1", "ClassName2"],
    "candidate_test_file": "path/to/test_file.py",
    "confidence": "high|medium|low"
  }},
  "setup_steps": ["step 1 (imports/preconditions)", "step 2 (object/fixture setup)"],
  "execution_stimulus": ["action 1", "action 2"],
  "expected_failure": "Description of how the test should fail on buggy code",
  "test_environment": {{
    "required_fixtures": ["fixture_name_if_needed"],
    "runner": "{runner}"
  }},
  "reproduction_code": [{{"language": "python", "code": "def test_<name>():\n    # imports\n    # setup from setup_steps\n    # execution from execution_stimulus\n    assert <expected_failure condition>"}}],
  "identifiers": {{"functions": ["target_func"], "classes": ["RelatedClass"]}},
  "expected_outputs": [],
  "actual_outputs": [],
  "error_keywords": []
}}

[Target Function Selection Guide]
- PREFER: domain-specific functions explicitly named in the issue (e.g., `separability_matrix`, `register_cmap`, `get_cmap`)
- PREFER: public API functions that a user would call directly to trigger the bug
- AVOID: dunder methods (`__str__`, `__hash__`, `__init__`, `__setitem__`, etc.) unless the issue is explicitly about that dunder's behavior AND it appears in the identified functions list
- If dunder is the only option, make sure execution_stimulus calls it indirectly (e.g., `str(obj)` not `obj.__str__()`) so it actually appears in the test
- If functions list is empty: infer the most specific public function from the issue description and source file name

{func_list_section}[Constraints]
0. Generate behavioral diversity, not wording diversity. If the repository evidence supports multiple routes, each scenario should use a different plausible reproduction route, target behavior, fixture/helper, or assertion perspective. Do not return paraphrases of the same setup/action/oracle.
1. target_function selection priority:
   a. IF [Fault Locations from Issue Traceback] exists → use that function/file for S1 (HIGHEST PRIORITY)
   b. ELSE IF [Inferred Fault Location Candidates] exists → treat them as hints, not mandatory truth
   c. ELSE use the issue's identified functions: {functions[:10]}
   d. If both are empty, pick from [Available Functions per Source File] above
   CRITICAL: target_function MUST appear in [Available Functions per Source File] for the chosen
   source_file. NEVER invent a function name not listed there or visible in [Source Code Snippets].
2. source_file MUST be one of the candidate source files: {source_files[:5]}
3. candidate_test_file SHOULD be one of the candidate test files: {test_files[:5]}
4. execution_stimulus must describe concrete actions, not abstract descriptions
5. expected_failure must explicitly separate observed buggy behavior from expected fixed behavior; do not merge them into one ambiguous sentence.
6. setup_steps should include both preconditions (environment/imports) and concrete setup actions
7. At least one scenario (preferably S3) should include a candidate_test_file
8. CRITICAL — actual_outputs / expected_outputs: extract from [Issue Description] text.
   actual_outputs — the BUGGY value the code currently produces:
     Search [Issue Description] for: "currently returns X" / "outputs Y" / "produces Z" /
     "Actual: Z" / numbers or arrays described as WRONG / error messages shown.
     Extract VERBATIM. Even a single token like ["1"] or ["None"] is valuable.
     Example: issue says "distance is calculated as 1" → actual_outputs: ["1"]
     Leave [] ONLY if the issue gives absolutely NO observable output or value.
   expected_outputs — the CORRECT value after the fix:
     ONLY fill if issue EXPLICITLY states: "should return X" / "correct output: Y" / "Expected: Z".
     DO NOT guess or calculate. If not stated, leave [].
9. reproduction_code MUST be a runnable Python test function (def test_<name>():) that:
   - Imports necessary modules
   - Executes the setup_steps and execution_stimulus as code
   - Contains a concrete assert statement or pytest.raises oracle that will FAIL on the buggy code
   - Is self-contained (no fixtures, no class required unless runner=django-test)
10. Do not use placeholder symbols such as xxx, foo, bar, Dummy*, MockModel, FooModel, or BarModel.
11. Do not define Django model classes in scenarios. Reuse candidate test file models/helpers only.
12. Do not invent APIs, fixtures, imports, models, helpers, or dependencies that are absent from [Code Context], [Available Pytest Fixtures], [Available Functions], [Source Code Snippets], or existing target tests.
13. Required fixtures must be drawn from [Available Pytest Fixtures]; otherwise use [].
14. The oracle must be executable and must assert the expected fixed behavior from the issue, not the observed buggy value.
15. Return JSON array only. No explanation.
""".strip()

        return prompt

    @staticmethod
    def _build_m7_feedback_section(feedback: Mapping[str, Any] | None) -> str:
        if not isinstance(feedback, Mapping) or not feedback:
            return ""
        safe_feedback = {
            key: value
            for key, value in feedback.items()
            if key
            not in {
                "golden_patch",
                "golden_patch_lines",
                "patch_hit_rate",
                "fail_to_pass",
                "m8_results",
                "post_patch_outcome",
            }
        }
        if not safe_feedback:
            return ""
        return (
            "\n[M7 Feedback For This New M3 Invocation]\n"
            "Use this pre-patch feedback to revise the next scenarios. "
            "Keep one JSON response with 1-3 distinct scenarios; do not perform adaptive sampling.\n"
            f"{json.dumps(safe_feedback, ensure_ascii=False, sort_keys=True)[:3000]}\n"
        )

    @staticmethod
    def _compact_m3_prompt(prompt: str, safe_user_tokens: int) -> str:
        """Compact M3 prompts while preserving issue evidence and top context."""
        if estimate_prompt_tokens(prompt) <= safe_user_tokens:
            return prompt
        start_marker = "[Source Code Snippets - Use these to write concrete execution_stimulus and setup_steps]"
        end_marker = "\n\n[Output Format]"
        if start_marker not in prompt or end_marker not in prompt:
            return _compact_text_tail_first(prompt, safe_user_tokens)

        prefix, rest = prompt.split(start_marker, 1)
        snippet_section, suffix = rest.split(end_marker, 1)
        header = start_marker
        snippets = _split_ranked_source_snippets(snippet_section)

        current_snippets = snippets[:]
        current = prefix + header + "".join(current_snippets) + end_marker + suffix
        if estimate_prompt_tokens(current) <= safe_user_tokens:
            return current

        for index in range(len(current_snippets) - 1, -1, -1):
            current_snippets[index] = _truncate_source_snippet(current_snippets[index], keep_top_context=index == 0)
            current = prefix + header + "".join(current_snippets) + end_marker + suffix
            if estimate_prompt_tokens(current) <= safe_user_tokens:
                return current

        while len(current_snippets) > 1:
            current_snippets.pop()
            current = prefix + header + "".join(current_snippets) + end_marker + suffix
            if estimate_prompt_tokens(current) <= safe_user_tokens:
                return current

        if current_snippets:
            compacted_first = current_snippets[0]
            while estimate_prompt_tokens(prefix + header + compacted_first + end_marker + suffix) > safe_user_tokens:
                next_compacted = _truncate_source_snippet(compacted_first, keep_top_context=True, reduction=0.5)
                if next_compacted == compacted_first:
                    break
                compacted_first = next_compacted
            current = prefix + header + compacted_first + end_marker + suffix
            if estimate_prompt_tokens(current) <= safe_user_tokens:
                return current

        protected_prompt = prefix + header + "\n(none available)\n" + end_marker + suffix
        return _compact_text_tail_first(protected_prompt, safe_user_tokens)

    def prompt_budget_metadata(self) -> Dict[str, Any]:
        """Return prompt-budget accounting from the last model call."""
        return dict(getattr(self.client, "last_prompt_budget", {}) or {})

    def _parse_response(
        self,
        raw_response: str,
        clue: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[TestScenario]:
        text = raw_response.strip()
        self._last_parse_diagnostics = {
            "parse_status": "SUCCESS",
            "fallback_used": False,
            "failure_kind": "",
            "json_error": "",
            "raw_response_chars": len(raw_response or ""),
            "looks_truncated": bool(raw_response and raw_response.rstrip()[-1:] not in {"]", "}"}),
            "fallback_construction_sec": 0.0,
        }

        # 코드 펜스 안의 JSON 추출
        fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 배열 패턴 추출 시도
            arr_match = re.search(r"(\[.*\])", text, re.DOTALL)
            obj_match = re.search(r"(\{.*\})", text, re.DOTALL)
            if arr_match:
                try:
                    data = json.loads(arr_match.group(1))
                except json.JSONDecodeError:
                    # 배열 파싱도 실패 → 단일 객체 시도
                    if obj_match:
                        try:
                            data = [json.loads(obj_match.group(1))]
                        except json.JSONDecodeError as exc:
                            logger.error("LLM scenario response parsing failed, using fallback scenarios")
                            return self._fallback_after_parse_failure(clue, context, "malformed_json", exc)
                    else:
                        logger.error("LLM scenario response parsing failed, using fallback scenarios")
                        return self._fallback_after_parse_failure(clue, context, "malformed_json", None)
            elif obj_match:
                # 단일 객체 → 배열로 래핑
                try:
                    data = [json.loads(obj_match.group(1))]
                except json.JSONDecodeError as exc:
                    logger.error("LLM scenario response parsing failed, using fallback scenarios")
                    return self._fallback_after_parse_failure(clue, context, "malformed_json", exc)
            else:
                logger.error("LLM scenario response parsing failed, using fallback scenarios")
                return self._fallback_after_parse_failure(clue, context, "no_json_array_or_object", None)

        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list) or not data:
            logger.warning("LLM scenario response is empty, using fallback scenarios")
            return self._fallback_after_parse_failure(clue, context, "empty_or_non_list", None)

        source_files = [x["path"] for x in context.get("candidate_source_files", [])]
        test_files = [x["path"] for x in context.get("candidate_test_files", [])]
        functions = clue.get("identifiers", {}).get("functions", [])
        classes = clue.get("identifiers", {}).get("classes", [])
        code_examples = clue.get("code_examples", [])
        expected_outputs = clue.get("expected_outputs", [])
        actual_outputs = clue.get("actual_outputs", [])
        error_keywords = clue.get("error_keywords", [])
        clue_identifiers = clue.get("identifiers", {})
        runner = context.get("project_test_style", {}).get("runner", "pytest")

        # source_file → 실제/검색된 symbol 맵 (target_function 실존 검증용)
        candidate_symbol_map: Dict[str, List[str]] = {}
        for sf in context.get("candidate_source_files", []):
            symbols: List[str] = []
            for key in ("matched_identifiers", "top_level_functions"):
                for value in sf.get(key, []) or []:
                    text = str(value)
                    if text and text not in symbols:
                        symbols.append(text)
                    bare = text.split(".")[-1]
                    if bare and bare not in symbols:
                        symbols.append(bare)
            candidate_symbol_map[sf["path"]] = symbols

        items: List[Dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                items.append(item)
            elif isinstance(item, list):
                nested = [x for x in item if isinstance(x, dict)]
                if nested:
                    logger.warning("LLM scenario response contained nested list; flattening it")
                    items.extend(nested)
                else:
                    logger.warning("skipping non-dict scenario list item: %s", type(item).__name__)
            else:
                logger.warning("skipping non-dict scenario item: %s", type(item).__name__)

        if not items:
            logger.warning("LLM scenario response has no dict items, using fallback scenarios")
            return self._fallback_after_parse_failure(clue, context, "no_dict_items", None)

        scenarios: List[TestScenario] = []
        for i, item in enumerate(items):
            # target_function 실존 검증 + 자동 교체
            tl = item.get("target_location", {})
            if not isinstance(tl, dict):
                tl = {}
            src = tl.get("source_file", "")
            tfunc = tl.get("target_function", "")
            actual_symbols = candidate_symbol_map.get(src, [])
            if not tfunc and actual_symbols:
                issue_funcs = functions or []
                valid_alts = [
                    f for f in issue_funcs
                    if f in actual_symbols or f.split(".")[-1] in actual_symbols
                ]
                tl["target_function"] = valid_alts[0] if valid_alts else actual_symbols[0]
                item["target_location"] = tl
            elif tfunc and actual_symbols and tfunc not in actual_symbols and tfunc.split(".")[-1] not in actual_symbols:
                item.setdefault("target_verification_status", "TARGET_UNRESOLVED")
                item.setdefault("target_verification_provenance", {
                    "source": "m3_nonadaptive_parser",
                    "reason": "target_not_in_context_symbols_preserved_for_m4",
                    "target_function": tfunc,
                    "source_file": src,
                })

            try:
                scenario = self._dict_to_scenario(
                    item,
                    fallback_id=f"S{i + 1}",
                    source_files=source_files,
                    test_files=test_files,
                    functions=functions,
                    classes=classes,
                    code_examples=code_examples,
                    expected_outputs=expected_outputs,
                    actual_outputs=actual_outputs,
                    error_keywords=error_keywords,
                    clue_identifiers=clue_identifiers,
                    runner=runner,
                )
                scenarios.append(scenario)
            except Exception as e:
                logger.warning("scenario parsing failed (index %d): %s", i, e)

        if not scenarios:
            logger.warning("all LLM scenario parsing failed, using fallback scenarios")
            return self._fallback_after_parse_failure(clue, context, "schema_incompatible_items", None)

        return scenarios

    def _fallback_after_parse_failure(
        self,
        clue: Dict[str, Any],
        context: Dict[str, Any],
        failure_kind: str,
        exc: Exception | None,
    ) -> List[TestScenario]:
        fallback_t0 = time.monotonic()
        scenarios = self._build_fallback_scenarios(
            clue, context, getattr(self, "_active_v26_feedback", None)
        )
        elapsed = round(time.monotonic() - fallback_t0, 3)
        self._last_parse_diagnostics.update(
            {
                "parse_status": "FAILED",
                "fallback_used": True,
                "failure_kind": failure_kind,
                "json_error": str(exc or ""),
                "fallback_construction_sec": elapsed,
            }
        )
        return scenarios

    def _filter_distinct_scenarios(
        self,
        scenarios: Sequence[TestScenario],
        *,
        max_count: int = 3,
    ) -> List[TestScenario]:
        kept: List[TestScenario] = []
        seen_exact: set[str] = set()
        duplicate_count = 0
        incomplete_count = 0
        for scenario in scenarios:
            data = scenario.to_dict()
            if not self._has_required_nonadaptive_fields(data):
                incomplete_count += 1
                continue
            exact_key = self._scenario_exact_fingerprint(data)
            if exact_key in seen_exact:
                duplicate_count += 1
                continue
            if any(self._is_near_duplicate_scenario(data, previous.to_dict()) for previous in kept):
                duplicate_count += 1
                continue
            seen_exact.add(exact_key)
            kept.append(scenario)
            if len(kept) >= max_count:
                break
        if kept:
            for index, scenario in enumerate(kept, 1):
                original_id = str(scenario.scenario_id or f"S{index}")
                scenario.scenario_id = f"S{index}"
                scenario.generation_provenance = "model_generated"
                if original_id != scenario.scenario_id:
                    hints = scenario.oracle_hints or []
                    hints.append(f"original_scenario_id:{original_id}")
                    scenario.oracle_hints = hints
            if duplicate_count or incomplete_count:
                logger.info(
                    "M3 non-adaptive scenario filtering kept=%d duplicate=%d incomplete=%d",
                    len(kept),
                    duplicate_count,
                    incomplete_count,
                )
            return kept
        return list(scenarios[:max_count])

    @staticmethod
    def _has_required_nonadaptive_fields(scenario: Mapping[str, Any]) -> bool:
        projected = canonical_scenario_projection(scenario)
        return all(
            [
                projected.get("target_function"),
                projected.get("source_file"),
                projected.get("stimulus_steps"),
                scenario.get("expected_failure"),
            ]
        )

    @staticmethod
    def _scenario_exact_fingerprint(scenario: Mapping[str, Any]) -> str:
        projected = canonical_scenario_projection(scenario)
        payload = {
            "target_function": normalize_answer_key_value(projected.get("target_function")),
            "source_file": normalize_answer_key_value(projected.get("source_file")),
            "oracle_type": normalize_answer_key_value(projected.get("oracle_type")),
            "oracle_expected": _semantic_text(projected.get("oracle_expected")),
            "stimulus_steps": stimulus_summary_from_steps(projected.get("stimulus_steps") or []),
            "expected_failure": _semantic_text(scenario.get("expected_failure", "")),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _is_near_duplicate_scenario(
        scenario: Mapping[str, Any],
        previous: Mapping[str, Any],
    ) -> bool:
        a = canonical_scenario_projection(scenario)
        b = canonical_scenario_projection(previous)
        if normalize_answer_key_value(a.get("target_function")) != normalize_answer_key_value(b.get("target_function")):
            return False
        if normalize_answer_key_value(a.get("source_file")) != normalize_answer_key_value(b.get("source_file")):
            return False
        if stimulus_summary_from_steps(a.get("stimulus_steps") or []) == stimulus_summary_from_steps(b.get("stimulus_steps") or []):
            return True
        a_tokens = ScenarioGenerator._behavior_tokens(scenario)
        b_tokens = ScenarioGenerator._behavior_tokens(previous)
        if not a_tokens or not b_tokens:
            return False
        return len(a_tokens & b_tokens) / len(a_tokens | b_tokens) >= 0.92

    @staticmethod
    def _behavior_tokens(scenario: Mapping[str, Any]) -> set[str]:
        projected = canonical_scenario_projection(scenario)
        text = " ".join(
            [
                " ".join(projected.get("stimulus_steps") or []),
                str(scenario.get("expected_failure") or ""),
                str(projected.get("oracle_expected") or ""),
            ]
        )
        return {
            token.strip(" ,.:;()[]{}'\"").lower()
            for token in re.split(r"\s+|/|_", text)
            if len(token.strip(" ,.:;()[]{}'\"")) >= 3
        }

    def _dict_to_scenario(
        self,
        item: Dict[str, Any],
        fallback_id: str,
        source_files: List[str],
        test_files: List[str],
        functions: List[str],
        classes: List[str],
        code_examples: List[Dict[str, str]],
        expected_outputs: List[str],
        actual_outputs: List[str],
        error_keywords: Optional[List[str]] = None,
        clue_identifiers: Optional[Dict[str, Any]] = None,
        runner: str = "pytest",
    ) -> TestScenario:
        target = item.get("target_location", {})
        if not isinstance(target, dict):
            target = {}

        # source_file / target_function 보정
        src_file = target.get("source_file", "")
        if not src_file and source_files:
            src_file = source_files[0]

        tgt_func = target.get("target_function", "")
        if not tgt_func and functions:
            tgt_func = functions[0]

        related = self._ensure_list(target.get("related_classes", classes[:3]))
        candidate_test = target.get("candidate_test_file", "")
        if not candidate_test and test_files:
            candidate_test = test_files[0]

        confidence = target.get("confidence", "medium")
        test_environment = item.get("test_environment", {})
        if not isinstance(test_environment, dict):
            test_environment = {}
        reproduction_code = self._ensure_reproduction_code(
            item.get("reproduction_code") or code_examples
        )
        identifiers = item.get("identifiers") or clue_identifiers or {}
        if not isinstance(identifiers, dict):
            identifiers = clue_identifiers or {}

        # preconditions가 있으면 setup_steps 앞에 병합 (하위호환)
        preconditions = self._ensure_list(item.get("preconditions", []))
        setup_steps = self._ensure_list(item.get("setup_steps", []))
        merged_setup = preconditions + [s for s in setup_steps if s not in preconditions]

        return TestScenario(
            scenario_id=item.get("scenario_id", fallback_id),
            target_location={
                "source_file": src_file,
                "target_function": tgt_func,
                "related_classes": related[:5],
                "candidate_test_file": candidate_test,
                "confidence": confidence,
            },
            setup_steps=merged_setup,
            execution_stimulus=self._ensure_list(item.get("execution_stimulus", [])),
            expected_failure=self._ensure_str(
                item.get("expected_failure"), "The test should fail due to the reported bug."
            ),
            relevant_source_files=source_files[:3],
            relevant_test_files=test_files[:3],
            test_environment={
                "required_fixtures": self._ensure_list(
                    test_environment.get("required_fixtures", [])
                ),
                "runner": test_environment.get("runner", runner),
            },
            # clue 파생 필드: LLM이 생성했으면 그대로, 없으면 clue 데이터로 fallback
            reproduction_code=reproduction_code,
            expected_outputs=item.get("expected_outputs") or expected_outputs,
            actual_outputs=item.get("actual_outputs") or actual_outputs,
            error_keywords=item.get("error_keywords") or error_keywords or [],
            identifiers=identifiers,
            oracle_hints=self._ensure_list(item.get("oracle_hints", [])),
            oracle=self._ensure_str(item.get("oracle"), ""),
            oracle_contract=item.get("oracle_contract") if isinstance(item.get("oracle_contract"), dict) else {},
                oracle_type=self._ensure_str(item.get("oracle_type"), ""),
                oracle_source=self._ensure_str(item.get("oracle_source"), ""),
                generation_provenance=self._ensure_str(item.get("generation_provenance"), "model_generated"),
                issue_api_target=self._ensure_str(item.get("issue_api_target"), ""),
                implementation_target=self._ensure_str(item.get("implementation_target"), ""),
                setup_helper_calls=self._ensure_list(item.get("setup_helper_calls", [])),
                target_verification_status=self._ensure_str(item.get("target_verification_status"), ""),
                target_verification_provenance=item.get("target_verification_provenance")
                if isinstance(item.get("target_verification_provenance"), dict)
                else {},
                target_consistency_status=self._ensure_str(item.get("target_consistency_status"), ""),
                scenario_generation_attempt=int(item.get("scenario_generation_attempt", 1) or 1),
                m3_model_call_count=int(item.get("m3_model_call_count", 0) or 0),
                fallback_used=bool(item.get("fallback_used", False)),
                fallback_reason=self._ensure_str(item.get("fallback_reason"), ""),
            )

    def _dict_to_hydrated_scenario(
        self,
        item: Dict[str, Any],
        clue: Dict[str, Any],
        context: Dict[str, Any],
        repo: str = "",
    ) -> TestScenario:
        hydrated = hydrate_scenario_dict(item, clue, repo=repo, context=context)
        target = hydrated.get("target_location", {}) if isinstance(hydrated.get("target_location"), dict) else {}
        test_env = hydrated.get("test_environment", {}) if isinstance(hydrated.get("test_environment"), dict) else {}
        return TestScenario(
            scenario_id=hydrated.get("scenario_id", "S1"),
            target_location=target,
            setup_steps=self._ensure_list(hydrated.get("setup_steps", [])),
            execution_stimulus=self._ensure_list(hydrated.get("execution_stimulus", [])),
            expected_failure=self._ensure_str(hydrated.get("expected_failure"), ""),
            relevant_source_files=self._ensure_list(hydrated.get("relevant_source_files", [])),
            relevant_test_files=self._ensure_list(hydrated.get("relevant_test_files", [])),
            test_environment=test_env,
            reproduction_code=self._ensure_reproduction_code(hydrated.get("reproduction_code", [])),
            expected_outputs=self._ensure_list(hydrated.get("expected_outputs", [])),
            actual_outputs=self._ensure_list(hydrated.get("actual_outputs", [])),
            error_keywords=self._ensure_list(hydrated.get("error_keywords", [])),
            identifiers=hydrated.get("identifiers", {}) if isinstance(hydrated.get("identifiers"), dict) else {},
            oracle_hints=self._ensure_list(hydrated.get("oracle_hints", [])),
            oracle=self._ensure_str(hydrated.get("oracle"), ""),
            oracle_contract=hydrated.get("oracle_contract", {}) if isinstance(hydrated.get("oracle_contract"), dict) else {},
            oracle_type=self._ensure_str(hydrated.get("oracle_type"), ""),
            oracle_source=self._ensure_str(hydrated.get("oracle_source"), ""),
            validation_status=self._ensure_str(hydrated.get("validation_status"), "pending"),
            diagnostic_only=bool(hydrated.get("diagnostic_only", False)),
            generation_provenance=self._ensure_str(hydrated.get("generation_provenance"), "model_generated"),
            issue_api_target=self._ensure_str(hydrated.get("issue_api_target"), ""),
            implementation_target=self._ensure_str(hydrated.get("implementation_target"), ""),
            setup_helper_calls=self._ensure_list(hydrated.get("setup_helper_calls", [])),
            target_verification_status=self._ensure_str(hydrated.get("target_verification_status"), ""),
            target_verification_provenance=hydrated.get("target_verification_provenance", {})
            if isinstance(hydrated.get("target_verification_provenance"), dict)
            else {},
            target_consistency_status=self._ensure_str(hydrated.get("target_consistency_status"), ""),
            scenario_generation_attempt=int(hydrated.get("scenario_generation_attempt", 1) or 1),
            m3_model_call_count=int(hydrated.get("m3_model_call_count", 0) or 0),
            fallback_used=bool(hydrated.get("fallback_used", False)),
            fallback_reason=self._ensure_str(hydrated.get("fallback_reason"), ""),
            fault_hypothesis=self._ensure_str(
                hydrated.get("fault_hypothesis") or context.get("fault_hypothesis"),
                "",
            ),
            m2_oracle_hint=self._ensure_str(
                hydrated.get("m2_oracle_hint") or context.get("oracle_hint"),
                "",
            ),
            confidence_score=int(hydrated.get("confidence_score", 3) or 3),
            confidence_score_provenance=self._ensure_str(
                hydrated.get("confidence_score_provenance"), "default_missing"
            ),
        )

    def _build_fallback_scenarios(
        self,
        clue: Dict[str, Any],
        context: Dict[str, Any],
        feedback: Mapping[str, Any] | None = None,
    ) -> List[TestScenario]:
        """LLM 호출 실패 시 clue/context 기반 최소 시나리오 생성."""
        feedback_present = bool(feedback)
        functions = clue.get("identifiers", {}).get("functions", [])
        classes = clue.get("identifiers", {}).get("classes", [])
        observed = clue.get("observed_behavior", [])
        expected = clue.get("expected_behavior", [])

        source_files = [x["path"] for x in context.get("candidate_source_files", [])]
        test_files = [x["path"] for x in context.get("candidate_test_files", [])]
        framework = context.get("project_test_style", {}).get("framework", "unknown")
        runner = context.get("project_test_style", {}).get("runner", "pytest")

        # Prefer an issue-grounded traceback/file hint over the first ranked
        # retrieval result.  Retrieval is intentionally broad, but a fallback
        # scenario must not silently move the reproduction to an unrelated
        # implementation file when the issue itself names a source location.
        primary_source = ""
        for location in clue.get("fault_locations", []) or []:
            if not isinstance(location, Mapping):
                continue
            hinted = str(
                location.get("file_path")
                or location.get("source_file")
                or location.get("file")
                or ""
            ).replace("\\", "/").lstrip("./")
            if hinted:
                matches = [
                    path for path in source_files
                    if path == hinted or path.endswith("/" + hinted) or hinted.endswith("/" + path)
                ]
                if matches:
                    primary_source = matches[0]
                    break
        if not primary_source:
            primary_source = source_files[0] if source_files else ""
        primary_test = test_files[0] if test_files else ""
        restart_constraints = (
            (context.get("metadata") or {}).get("restart_constraints") or {}
            if isinstance(context.get("metadata"), Mapping)
            else {}
        )
        prohibited_targets = restart_constraints.get("prohibited_targets") or []
        primary_func = _choose_issue_api_target_for_fallback(
            clue,
            functions,
            prohibited_targets=prohibited_targets,
        )
        for fl in clue.get("fault_locations", []) or []:
            if not primary_func and fl.get("function_name"):
                primary_func = fl["function_name"]
                break
        prohibited_tails = {
            str(item.get("target_function") or item.get("function_name") or "")
            .split(".")[-1]
            .lower()
            for item in prohibited_targets
            if isinstance(item, Mapping)
        }
        if not primary_func:
            primary_func = next(
                (
                    str(function)
                    for function in functions
                    if str(function).split(".")[-1].lower() not in prohibited_tails
                ),
                "",
            )
        if not primary_func:
            for sf in context.get("candidate_source_files", []) or []:
                symbols = sf.get("matched_identifiers", []) or []
                symbols += sf.get("top_level_functions", []) or []
                for symbol in symbols:
                    bare = str(symbol).split(".")[-1]
                    if (
                        bare
                        and bare.lower() not in prohibited_tails
                        and not (bare.startswith("__") and bare.endswith("__"))
                    ):
                        primary_func = str(symbol)
                        break
                if primary_func:
                    break

        expected_outputs = clue.get("expected_outputs", [])
        actual_outputs = clue.get("actual_outputs", [])
        if expected_outputs:
            oracle_type = "positive_value"
            oracle_source = "issue_expected"
            oracle_rule = "Assert the fixed behavior stated in expected_outputs."
        elif actual_outputs:
            oracle_type = "semantic_invariant"
            oracle_source = "actual_buggy_output"
            oracle_rule = "Call the target API and assert a public invariant that excludes the buggy output."
        else:
            oracle_type = "semantic_invariant"
            oracle_source = "inferred_semantic"
            oracle_rule = "Call the target API and assert public state/value behavior, not local constants."

        failure_text = observed[0] if observed else "The assertion should fail due to the bug described in the issue."

        framework_step = [f"Import the relevant module (framework: {framework})."] if framework != "unknown" else ["Import the relevant module."]

        scenarios = [
            TestScenario(
                scenario_id="S1",
                target_location={
                    "source_file": primary_source,
                    "target_function": primary_func,
                    "related_classes": classes[:3],
                    "candidate_test_file": primary_test,
                    "confidence": "medium",
                },
                setup_steps=framework_step + [
                    f"Set up an environment where {primary_func} can be called." if primary_func else "Reproduce the input conditions described in the issue.",
                ],
                execution_stimulus=[
                    f"Call {primary_func} with the reproduction conditions from the issue." if primary_func else "Execute the reproduction code from the issue.",
                    "Compare the result with the expected value.",
                ],
                expected_failure=failure_text,
                relevant_source_files=source_files[:3],
                relevant_test_files=test_files[:3],
                test_environment={"required_fixtures": [], "runner": runner},
                reproduction_code=clue.get("code_examples", []),
                expected_outputs=expected_outputs,
                actual_outputs=actual_outputs,
                error_keywords=clue.get("error_keywords", []),
                identifiers=clue.get("identifiers", {}),
                oracle_hints=[oracle_rule],
                oracle=oracle_rule,
                oracle_contract={
                    "oracle_type": oracle_type,
                    "oracle_source": oracle_source,
                    "rule": oracle_rule,
                },
                oracle_type=oracle_type,
                oracle_source=oracle_source,
                validation_status=(
                    "rejected_feedback_not_applied"
                    if feedback_present
                    else "pending"
                ),
                diagnostic_only=feedback_present,
                generation_provenance="fallback_generated",
                issue_api_target=primary_func,
                implementation_target="",
                setup_helper_calls=[],
                target_verification_status="TARGET_UNRESOLVED",
                target_verification_provenance={
                    "source": "m3_fallback",
                    "reason": "fallback preserves issue-grounded public API when implementation is unresolved",
                    "m7_feedback": dict(feedback or {}),
                },
                target_consistency_status="CONSISTENT_WITH_UNRESOLVED_IMPLEMENTATION",
                scenario_generation_attempt=1,
                m3_model_call_count=0,
                fallback_used=True,
                fallback_reason="fallback_generated",
                fault_hypothesis=self._ensure_str(
                    context.get("fault_hypothesis"),
                    "",
                ),
                m2_oracle_hint=self._ensure_str(
                    context.get("oracle_hint"),
                    "",
                ),
                feedback_consumed={},
            ),
        ]

        return scenarios


    @staticmethod
    def _build_conftest_section(conftest_fixtures: Dict[str, List[str]]) -> str:
        if not conftest_fixtures:
            return ""
        lines = ["[Available Pytest Fixtures (from conftest.py)]"]
        for path, names in conftest_fixtures.items():
            lines.append(f"  {path}: {', '.join(names)}")
        lines.append("")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _ensure_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(x) for x in value if x]
        if isinstance(value, str):
            return [value]
        return []

    @staticmethod
    def _ensure_str(value: Any, default: str = "") -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return " ".join(str(x) for x in value)
        return str(value) if value else default

    @staticmethod
    def _ensure_reproduction_code(value: Any) -> List[Dict[str, Any]]:
        return ensure_reproduction_code_blocks(value)

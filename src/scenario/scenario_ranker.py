from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


M4_RANKER_SCHEMA_VERSION = "m4-scenario-ranker-v26"
MAX_SELECTED_SCENARIOS = 1
V36_MAX_SELECTED_SCENARIOS = 2
V37_RRA_FIXED_SEED = 37


def normalize_v37_confidence(confidence_score: Any) -> float:
    """Return q_oracle using the v37 parse/default rule."""
    if isinstance(confidence_score, bool):
        parsed = 3
    else:
        try:
            numeric = float(confidence_score)
            parsed = int(numeric) if numeric.is_integer() else 3
        except (TypeError, ValueError):
            parsed = 3
    if not 1 <= parsed <= 5:
        parsed = 3
    return (parsed - 1) / 4.0


def _v37_tokens(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value or "")
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", text)}


def compute_v37_q_s2r(s2r: Any, stimulus_steps: Any) -> tuple[float, dict[str, Any]]:
    """Compute binary q_s2r using deterministic no-stemming tokenization."""
    s2r_tokens = _v37_tokens(s2r)
    stimulus_tokens = _v37_tokens(stimulus_steps)
    if not s2r_tokens:
        raise ValueError(
            "BLOCKING SPEC_AMBIGUITY: q_s2r empty S2R_keyword_count is undefined"
        )
    intersection = s2r_tokens & stimulus_tokens
    ratio = len(intersection) / len(s2r_tokens)
    return (
        1.0 if ratio >= 0.5 else 0.0,
        {
            "tokenization": "regex_alphanumeric_underscore_lowercase_no_stemming",
            "s2r_keyword_count": len(s2r_tokens),
            "intersection_count": len(intersection),
            "overlap_ratio": ratio,
            "threshold": 0.5,
        },
    )


def _v37_issue_target_classes(clue: Mapping[str, Any]) -> set[str]:
    identifiers = clue.get("identifiers")
    classes: set[str] = set()
    if isinstance(identifiers, Mapping):
        for value in identifiers.get("classes", []) or []:
            name = str(value).split(".")[-1]
            if name:
                classes.add(name)
    for hint in clue.get("defect_location_hints", []) or clue.get("fault_locations", []) or []:
        text = json.dumps(hint, ensure_ascii=False) if isinstance(hint, Mapping) else str(hint)
        classes.update(re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", text))
    return classes


def _v37_scenario_target_class(target_function: Any) -> str:
    parts = [part for part in str(target_function or "").split(".") if part]
    return parts[-2] if len(parts) >= 2 else ""


def compute_v37_base_score(
    scenario: Mapping[str, Any],
    clue: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute formula (7) without an M4 model call."""
    q_oracle = normalize_v37_confidence(scenario.get("confidence_score"))
    q_s2r, q_s2r_evidence = compute_v37_q_s2r(
        clue.get("S2R")
        or clue.get("steps_to_reproduce")
        or clue.get("repro_conditions"),
        scenario.get("stimulus_steps") or scenario.get("execution_stimulus"),
    )
    issue_classes = _v37_issue_target_classes(clue)
    target_function = scenario.get("target_function")
    if not target_function and isinstance(scenario.get("target_location"), Mapping):
        target_function = scenario["target_location"].get("target_function")
    scenario_class = _v37_scenario_target_class(target_function)
    q_target = 1.0 if scenario_class and scenario_class in issue_classes else 0.0
    score = 0.5 * q_oracle + 0.3 * q_s2r + 0.2 * q_target
    return {
        "BaseScore": score,
        "q_oracle": q_oracle,
        "q_s2r": q_s2r,
        "q_target": q_target,
        "weights": {"q_oracle": 0.5, "q_s2r": 0.3, "q_target": 0.2},
        "provenance": {
            "q_oracle": "ScenarioSet.confidence_score_normalized_(s-1)/4",
            "q_s2r": q_s2r_evidence,
            "q_target": {
                "issue_target_classes": sorted(issue_classes),
                "scenario_target_class": scenario_class,
                "comparison": "exact_simple_class_name",
            },
            "m4_llm_call_count": 0,
        },
    }


def compute_v37_rra(
    rank_sources: Mapping[str, Mapping[str, float]],
    *,
    generation_indices: Mapping[str, int],
    scenario_scores: Mapping[str, float],
) -> dict[str, Any]:
    """Compute reciprocal-rank aggregation over an explicit closed source set."""
    if not rank_sources:
        raise ValueError("at least one explicit v37 RRA rank source is required")
    identities = sorted(generation_indices, key=lambda key: generation_indices[key])
    reciprocal: dict[str, list[float]] = {identity: [] for identity in identities}
    source_rankings: dict[str, list[str]] = {}
    for source_name, values in rank_sources.items():
        missing = [identity for identity in identities if identity not in values]
        if missing:
            raise ValueError(f"{source_name} missing identities: {missing}")
        ordered = sorted(
            identities,
            key=lambda identity: (-float(values[identity]), generation_indices[identity]),
        )
        source_rankings[source_name] = ordered
        for rank, identity in enumerate(ordered, 1):
            reciprocal[identity].append(1.0 / rank)
    raw = {
        identity: sum(values) / len(values)
        for identity, values in reciprocal.items()
    }
    minimum = min(raw.values())
    maximum = max(raw.values())
    normalization_available = maximum != minimum
    normalized = {
        identity: (raw[identity] - minimum) / (maximum - minimum)
        if normalization_available
        else 1.0
        for identity in identities
    }
    ordered = sorted(
        identities,
        key=lambda identity: (
            -raw[identity],
            -float(scenario_scores[identity]),
            generation_indices[identity],
            _fixed_seed_tie_key(identity, V37_RRA_FIXED_SEED),
        ),
    )
    return {
        "formula": "(1/|R|) * sum(1/rank_r(f))",
        "rank_sources": list(rank_sources),
        "source_rankings": source_rankings,
        "raw_scores": raw,
        "normalized_scores": normalized,
        "normalization_status": (
            "AVAILABLE" if normalization_available else "ALL_RAW_VALUES_EQUAL_NORMALIZED_TO_ONE"
        ),
        "tie_break": [
            "higher_ScenarioScore",
            "earlier_M3_generation_index",
            f"deterministic_fixed_seed_{V37_RRA_FIXED_SEED}",
        ],
        "ordering": ordered,
    }


def _fixed_seed_tie_key(identity: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()
BLOCKING_UNSPECIFIED = "BLOCKING_UNSPECIFIED"
UNAVAILABLE = "UNAVAILABLE"
AVAILABLE = "AVAILABLE"


def rank_v37_scenarios(
    scenarios: Sequence[Mapping[str, Any]],
    repo_root: Path,
    *,
    clue: Mapping[str, Any],
    rel_llm_by_source: Mapping[str, float],
    iteration: int,
    sbfl_norm_by_id: Mapping[str, float] | None = None,
    prior_sbfl_spectrum: Sequence[Mapping[str, Any]] | None = None,
    prior_covered_sut_lines: Sequence[Mapping[str, Any]] | None = None,
    max_selected: int = 2,
) -> dict[str, Any]:
    """Rank the complete structurally valid v37 population with formulas 4–7."""
    score_records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    generation_indices: dict[str, int] = {}
    scenario_scores: dict[str, float] = {}
    sbfl_scores: dict[str, float] = {}
    by_identity: dict[str, dict[str, Any]] = {}
    for index, raw_scenario in enumerate(scenarios):
        scenario = dict(raw_scenario)
        scenario_id = _scenario_id(scenario, index)
        identity = _stable_identity(scenario, scenario_id)
        existence = check_target_exists(repo_root, scenario, scenario_id=scenario_id)
        if not existence.exists:
            rejected.append(
                {
                    "scenario_id": scenario_id,
                    "scenario": scenario,
                    "rejection_reason": "invalid_target_location",
                    "diagnostics": existence.diagnostics,
                }
            )
            continue
        base = compute_v37_base_score(scenario, clue)
        source_file = _scenario_source_file(scenario)
        rel_llm = float(rel_llm_by_source.get(source_file, 0.0))
        if not 0.0 <= rel_llm <= 1.0:
            raise ValueError(f"rel_llm must be in [0,1] for {source_file}")
        if iteration == 1:
            sbfl_norm = None
            score = 0.9 * base["BaseScore"] + 0.1 * rel_llm
            weights = {"BaseScore": 0.9, "SBFL_norm": 0.0, "rel_llm": 0.1}
        else:
            supplied = sbfl_norm_by_id or {}
            supplied_value = supplied.get(scenario_id, supplied.get(identity))
            if supplied_value is None and prior_sbfl_spectrum is not None:
                supplied_value = compute_v37_sbfl_norm(
                    prior_sbfl_spectrum,
                    prior_covered_sut_lines or [],
                    source_file=source_file,
                )
            if supplied_value is None:
                sbfl_norm = None
                score = 0.9 * base["BaseScore"] + 0.1 * rel_llm
                weights = {"BaseScore": 0.9, "SBFL_norm": 0.0, "rel_llm": 0.1}
            else:
                sbfl_norm = float(supplied_value)
                if not 0.0 <= sbfl_norm <= 1.0:
                    raise ValueError(f"SBFL_norm must be in [0,1] for {scenario_id}")
                score = 0.7 * base["BaseScore"] + 0.2 * sbfl_norm + 0.1 * rel_llm
                weights = {"BaseScore": 0.7, "SBFL_norm": 0.2, "rel_llm": 0.1}
                sbfl_scores[identity] = sbfl_norm
        generation_indices[identity] = index
        scenario_scores[identity] = score
        record = {
            "scenario_id": scenario_id,
            "stable_identity": identity,
            "scenario": scenario,
            "status": AVAILABLE,
            "ScenarioScore": score,
            "weights": weights,
            "components": {
                "BaseScore": base,
                "SBFL_norm": {
                    "value": sbfl_norm,
                    "status": (
                        "NOT_CONSUMED"
                        if iteration == 1
                        else AVAILABLE
                        if sbfl_norm is not None
                        else UNAVAILABLE
                    ),
                    "provenance": "iteration_1_weight_zero"
                    if iteration == 1
                    else "prior_M6_positive_ochiai_covered_mass"
                    if sbfl_norm is not None
                    else "prior_SBFL_unavailable",
                },
                "rel_llm": {
                    "value": rel_llm,
                    "status": AVAILABLE,
                    "provenance": "M2_file_ranking.llm_relevance_norm",
                },
            },
            "diagnostics": [],
            "score_range": "0..1",
            "higher_is_better": True,
            "generation_index": index,
        }
        by_identity[identity] = record
        score_records.append(record)
    if not score_records:
        return {
            "schema_version": "m4-v37-ranking-v1",
            "selected_scenarios": [],
            "rejected_scenarios": rejected,
            "scenario_scores": [],
            "S_final_ranking": {"status": UNAVAILABLE},
            "rank_aggregation_status": UNAVAILABLE,
            "input_provenance": {"m4_llm_call_count": 0},
        }
    rank_sources: dict[str, Mapping[str, float]] = {"ScenarioScore": scenario_scores}
    if iteration >= 2 and sbfl_scores and len(sbfl_scores) == len(scenario_scores):
        rank_sources["SBFL_norm"] = sbfl_scores
    rra = compute_v37_rra(
        rank_sources,
        generation_indices=generation_indices,
        scenario_scores=scenario_scores,
    )
    selected: list[dict[str, Any]] = []
    for rank, identity in enumerate(rra["ordering"], 1):
        record = by_identity[identity]
        record["rank"] = rank
        record["selected"] = rank <= min(2, max(0, int(max_selected)))
        record["S_final_raw"] = rra["raw_scores"][identity]
        record["S_final"] = rra["normalized_scores"][identity]
        if record["selected"]:
            selected.append(_selected_record(record))
        else:
            rejected.append(
                {
                    "scenario_id": record["scenario_id"],
                    "scenario": record["scenario"],
                    "rejection_reason": "not_selected",
                    "rank": rank,
                    "diagnostic_only": True,
                }
            )
    return {
        "schema_version": "m4-v37-ranking-v1",
        "selected_scenarios": selected,
        "rejected_scenarios": rejected,
        "scenario_scores": [_score_output_record(record) for record in score_records],
        "S_final_ranking": rra,
        "rank_aggregation_status": rra["normalization_status"],
        "rank_aggregation_method": "v37_reciprocal_rank_aggregation",
        "input_provenance": {
            "m4_llm_call_count": 0,
            "rank_sources_closed_set": list(rank_sources),
        },
    }


def compute_v37_sbfl_norm(
    spectrum: Sequence[Mapping[str, Any]],
    covered_sut_lines: Sequence[Mapping[str, Any]],
    *,
    source_file: str,
) -> float | None:
    """Compute formula (6) from prior positive Ochiai and covered SUT lines."""
    positive: dict[tuple[str, int], float] = {}
    for item in spectrum:
        path = str(item.get("source_file") or "").replace("\\", "/")
        line_no = item.get("line_no")
        score = item.get("S_ochiai", item.get("score"))
        if not path or isinstance(line_no, bool) or not isinstance(line_no, int):
            continue
        try:
            numeric = float(score)
        except (TypeError, ValueError):
            continue
        if numeric > 0.0:
            positive[(path, line_no)] = numeric
    denominator = sum(positive.values())
    if denominator <= 0.0:
        return None
    normalized_source = str(source_file).replace("\\", "/")
    covered = {
        (str(item.get("source_file") or "").replace("\\", "/"), item.get("line_no"))
        for item in covered_sut_lines
        if isinstance(item, Mapping)
    }
    numerator = sum(
        score
        for identity, score in positive.items()
        if identity[0] == normalized_source and identity in covered
    )
    return numerator / denominator


@dataclass(frozen=True)
class EvidenceValue:
    value: float | None
    status: str
    provenance: str
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "status": self.status,
            "provenance": self.provenance,
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class TargetDefinition:
    qualified_name: str
    kind: str
    lineno: int
    end_lineno: int | None


@dataclass
class TargetExistenceResult:
    scenario_id: str
    source_file: str
    target_function: str
    exists: bool
    status: str
    diagnostics: list[str] = field(default_factory=list)
    matches: list[TargetDefinition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "source_file": self.source_file,
            "target_function": self.target_function,
            "exists": self.exists,
            "status": self.status,
            "diagnostics": list(self.diagnostics),
            "matches": [
                {
                    "qualified_name": match.qualified_name,
                    "kind": match.kind,
                    "lineno": match.lineno,
                    "end_lineno": match.end_lineno,
                }
                for match in self.matches
            ],
        }


def rank_scenarios(
    scenarios: Sequence[Mapping[str, Any]],
    repo_root: str | Path,
    *,
    iteration: int,
    base_scores: Mapping[str, Any] | None = None,
    consensus_scores: Mapping[str, Any] | None = None,
    rel_llm_scores: Mapping[str, Any] | None = None,
    ochiai_scores: Mapping[str, Any] | None = None,
    s_final_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    input_provenance: Mapping[str, Any] | None = None,
    max_selected: int = MAX_SELECTED_SCENARIOS,
) -> dict[str, Any]:
    """Rank M4 scenarios deterministically from explicit pre-patch evidence.

    Input component scores are expected in ``[0, 1]`` and are clamped into that
    range when numeric. Missing values remain unavailable; the scorer does not
    fabricate ``BaseScore``, ``Consensus_norm``, ``SBFL_norm``, or ``rel_llm``.
    Higher ``ScenarioScore`` and ``S_final`` values are better. Missing
    evidence yields an unavailable scenario score rather than a synthetic zero.
    The evaluation population is the valid-target scenario list supplied to
    this function.
    """
    if iteration < 1:
        raise ValueError("iteration must be >= 1")
    weights = scenario_score_weights(iteration)
    # V26 requires this independent fallback ranking to be calculated before
    # ScenarioScore. It never supplies or fabricates a missing BaseScore.
    s_final = compute_s_final_ranking(s_final_evidence or {})
    repo_path = Path(repo_root)
    selected_limit = min(max(0, int(max_selected)), V36_MAX_SELECTED_SCENARIOS)
    input_provenance_dict = dict(input_provenance or {})
    if "base_score_source" not in input_provenance_dict:
        input_provenance_dict["base_score_source"] = (
            "explicit evidence required; repository has no approved hidden BaseScore formula"
        )

    existence_results: list[TargetExistenceResult] = []
    rejected_scenarios: list[dict[str, Any]] = []
    score_records: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    accepted_records: list[dict[str, Any]] = []

    for index, scenario in enumerate(scenarios):
        scenario_dict = dict(scenario)
        scenario_id = _scenario_id(scenario_dict, index)
        identity = _stable_identity(scenario_dict, scenario_id)
        existence = check_target_exists(repo_path, scenario_dict, scenario_id=scenario_id)
        existence_results.append(existence)
        if not existence.exists:
            rejected_scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "scenario": scenario_dict,
                    "rejection_reason": "invalid_target_location",
                    "diagnostics": list(existence.diagnostics),
                    "existence_check": existence.to_dict(),
                    "stable_identity": identity,
                }
            )
            continue

        score_record = score_scenario(
            scenario_dict,
            scenario_id=scenario_id,
            stable_identity=identity,
            iteration=iteration,
            base_scores=base_scores,
            consensus_scores=consensus_scores,
            rel_llm_scores=rel_llm_scores,
            ochiai_scores=ochiai_scores,
        )
        score_records.append(score_record)
        if score_record["status"] == AVAILABLE:
            accepted_records.append({"scenario": scenario_dict, **score_record})
        else:
            rejected_scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "scenario": scenario_dict,
                    "rejection_reason": "scenario_score_unavailable",
                    "diagnostics": list(score_record["diagnostics"]),
                    "existence_check": existence.to_dict(),
                    "stable_identity": identity,
                    "scoring": score_record,
                }
            )

    accepted_records.sort(key=_ranking_key)
    selected_scenarios: list[dict[str, Any]] = []
    for rank, record in enumerate(accepted_records, 1):
        record["rank"] = rank
        record["selected"] = rank <= selected_limit
        _mark_score_record(
            score_records,
            stable_identity=str(record["stable_identity"]),
            rank=rank,
            selected=bool(record["selected"]),
        )
        if record["selected"]:
            selected_scenarios.append(_selected_record(record))
        else:
            rejected_scenarios.append(
                {
                    "scenario_id": record["scenario_id"],
                    "scenario": record["scenario"],
                    "rejection_reason": "not_selected",
                    "diagnostics": ["scenario ranked below selected scenario limit"],
                    "stable_identity": record["stable_identity"],
                    "rank": rank,
                    "scoring": _score_output_record(record),
                }
            )

    multi_formula = build_multi_formula_consensus(ochiai_scores or {})
    scoring_fallback_used = True
    if s_final["ranking"]:
        diagnostics.append("S_final fallback ranking computed from explicit S_ochiai and rel_llm evidence")
    else:
        diagnostics.append("S_final fallback unavailable without both S_ochiai and rel_llm evidence")
    if not score_records and any(item.exists for item in existence_results):
        diagnostics.append("no scenario scores available; explicit BaseScore may be missing")

    return {
        "schema_version": M4_RANKER_SCHEMA_VERSION,
        "module": "M4",
        "iteration": iteration,
        "existence_check": [
            item.to_dict()
            for item in sorted(existence_results, key=_existence_sort_key)
        ],
        "rejected_scenarios": sorted(rejected_scenarios, key=_rejected_sort_key),
        "scenario_scores": [
            _score_output_record(record)
            for record in sorted(score_records, key=_score_record_sort_key)
        ],
        "selected_scenarios": selected_scenarios,
        "S_final_ranking": s_final,
        "rank_aggregation_status": BLOCKING_UNSPECIFIED,
        "rank_aggregation_method": "ScenarioScore ordering fallback; exact RRA algorithm is unspecified",
        "scoring_fallback_used": scoring_fallback_used,
        "multi_formula_consensus": multi_formula,
        "diagnostics": diagnostics,
        "input_provenance": input_provenance_dict,
    }


def scenario_score_weights(iteration: int) -> dict[str, float]:
    """Return the approved v26 M4 ScenarioScore weights for a pass."""
    if iteration < 1:
        raise ValueError("iteration must be >= 1")
    weights = (
        {"BaseScore": 0.9, "Consensus_norm": 0.0, "SBFL_norm": 0.0, "rel_llm": 0.1}
        if iteration == 1
        else {"BaseScore": 0.7, "Consensus_norm": 0.0, "SBFL_norm": 0.2, "rel_llm": 0.1}
    )
    total = sum(weights.values())
    if round(total, 10) != 1.0:
        raise ValueError(f"ScenarioScore weights must sum to 1.0; got {total}")
    return weights


def score_scenario(
    scenario: Mapping[str, Any],
    *,
    scenario_id: str,
    stable_identity: str,
    iteration: int,
    base_scores: Mapping[str, Any] | None = None,
    consensus_scores: Mapping[str, Any] | None = None,
    rel_llm_scores: Mapping[str, Any] | None = None,
    ochiai_scores: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    weights = scenario_score_weights(iteration)
    components = {
        "BaseScore": _evidence_for(scenario, scenario_id, base_scores, ("BaseScore", "base_score")),
        "Consensus_norm": _evidence_for(
            scenario,
            scenario_id,
            consensus_scores,
            ("Consensus_norm", "consensus_norm"),
        ),
        "rel_llm": _evidence_for(scenario, scenario_id, rel_llm_scores, ("rel_llm", "rel_LLM")),
    }
    if iteration == 1:
        components["SBFL_norm"] = EvidenceValue(
            value=None,
            status="NOT_CONSUMED",
            provenance="iteration_1_weight_zero",
            diagnostics=("iteration 1 does not consume supplied SBFL evidence",),
        )
    else:
        components["SBFL_norm"] = _evidence_for(
            scenario,
            scenario_id,
            ochiai_scores,
            ("SBFL_norm", "S_ochiai", "s_ochiai"),
        )

    diagnostics: list[str] = []
    unavailable = [
        name
        for name, evidence in components.items()
        if weights[name] > 0 and evidence.status != AVAILABLE
    ]
    for evidence in components.values():
        diagnostics.extend(evidence.diagnostics)
    if unavailable:
        diagnostics.append("ScenarioScore unavailable because required evidence is missing: " + ", ".join(unavailable))
        score = None
        status = UNAVAILABLE
    else:
        score = round(
            sum(
                weights[name] * (evidence.value or 0.0)
                for name, evidence in components.items()
            ),
            6,
        )
        status = AVAILABLE

    return {
        "scenario_id": scenario_id,
        "stable_identity": stable_identity,
        "status": status,
        "ScenarioScore": score,
        "weights": weights,
        "components": {name: value.to_dict() for name, value in components.items()},
        "diagnostics": diagnostics,
        "score_range": "0..1",
        "higher_is_better": True,
    }


def compute_s_final_ranking(
    evidence_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute the approved fallback ``S_final = 0.6*S_ochiai + 0.4*rel_llm``.

    Missing evidence keeps the item unavailable. The zero-denominator behavior
    is not applicable because this fallback is a weighted sum over explicit
    normalized inputs.
    """
    ranking: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    for identity, evidence in evidence_by_id.items():
        ochiai = _clamp_optional(evidence.get("S_ochiai"), name=f"{identity}.S_ochiai")
        rel_llm = _clamp_optional(evidence.get("rel_llm"), name=f"{identity}.rel_llm")
        diagnostics = list(ochiai.diagnostics) + list(rel_llm.diagnostics)
        if ochiai.status != AVAILABLE or rel_llm.status != AVAILABLE:
            unavailable.append(
                {
                    "identity": identity,
                    "status": UNAVAILABLE,
                    "S_final": None,
                    "diagnostics": diagnostics
                    + ["S_final requires both explicit S_ochiai and rel_llm evidence"],
                }
            )
            continue
        ranking.append(
            {
                "identity": identity,
                "status": AVAILABLE,
                "S_ochiai": ochiai.value,
                "rel_llm": rel_llm.value,
                "S_final": round(0.6 * (ochiai.value or 0.0) + 0.4 * (rel_llm.value or 0.0), 6),
                "formula": "0.6 * S_ochiai + 0.4 * rel_llm",
                "diagnostics": diagnostics,
            }
        )
    ranking.sort(key=lambda item: (-float(item["S_final"]), item["identity"]))
    for rank, item in enumerate(ranking, 1):
        item["rank"] = rank
    return {
        "status": AVAILABLE if ranking else UNAVAILABLE,
        "method": "explicit_fallback_S_final",
        "formula": "0.6 * S_ochiai + 0.4 * rel_llm",
        "ranking": ranking,
        "unavailable": unavailable,
    }


def build_multi_formula_consensus(ochiai_scores: Mapping[str, Any]) -> dict[str, Any]:
    ochiai_rank = _rank_score_mapping(ochiai_scores, score_name="S_ochiai")
    return {
        "status": BLOCKING_UNSPECIFIED,
        "ochiai_top5": ochiai_rank[:5],
        "dstar_top5": [],
        "tarantula_top5": [],
        "top5_intersection": [],
        "formula_status": {
            "ochiai": AVAILABLE if ochiai_rank else UNAVAILABLE,
            "dstar": "BLOCKING: DStar exponent is not defined in repository configuration",
            "tarantula": "BLOCKING: Tarantula zero-denominator behavior is not defined",
        },
        "diagnostics": [
            "Only Ochiai has approved executable support; DStar and Tarantula values were not fabricated",
            "Three-formula top-5 intersection was not computed",
        ],
    }


def check_target_exists(
    repo_root: Path,
    scenario: Mapping[str, Any],
    *,
    scenario_id: str,
) -> TargetExistenceResult:
    source_file = _scenario_source_file(scenario)
    target_function = _scenario_target_function(scenario)
    normalized, path_diagnostics = _normalize_repo_relative_path(source_file)
    diagnostics = list(path_diagnostics)
    if not normalized:
        return TargetExistenceResult(
            scenario_id=scenario_id,
            source_file=source_file,
            target_function=target_function,
            exists=False,
            status="REJECTED_INVALID_SOURCE_FILE",
            diagnostics=diagnostics,
        )
    if not target_function:
        return TargetExistenceResult(
            scenario_id=scenario_id,
            source_file=normalized,
            target_function=target_function,
            exists=False,
            status="REJECTED_MISSING_TARGET_FUNCTION",
            diagnostics=diagnostics + ["target_function is required"],
        )
    source_path = repo_root / normalized
    if not source_path.is_file():
        return TargetExistenceResult(
            scenario_id=scenario_id,
            source_file=normalized,
            target_function=target_function,
            exists=False,
            status="REJECTED_SOURCE_FILE_NOT_FOUND",
            diagnostics=diagnostics + [f"source_file does not exist: {normalized}"],
        )
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=normalized)
    except SyntaxError as exc:
        return TargetExistenceResult(
            scenario_id=scenario_id,
            source_file=normalized,
            target_function=target_function,
            exists=False,
            status="REJECTED_SOURCE_PARSE_ERROR",
            diagnostics=diagnostics + [f"source_file could not be parsed: {exc.msg}"],
        )
    definitions = _collect_definitions(tree)
    matches = _definition_matches(definitions, target_function)
    if len(matches) == 1:
        return TargetExistenceResult(
            scenario_id=scenario_id,
            source_file=normalized,
            target_function=target_function,
            exists=True,
            status="VALID",
            diagnostics=diagnostics + ["target location resolved exactly"],
            matches=matches,
        )
    if not matches:
        return TargetExistenceResult(
            scenario_id=scenario_id,
            source_file=normalized,
            target_function=target_function,
            exists=False,
            status="REJECTED_TARGET_NOT_FOUND",
            diagnostics=diagnostics + ["target_function was not found in source_file"],
        )
    return TargetExistenceResult(
        scenario_id=scenario_id,
        source_file=normalized,
        target_function=target_function,
        exists=False,
        status="REJECTED_AMBIGUOUS_TARGET",
        diagnostics=diagnostics + ["target_function matched multiple definitions; use a qualified class method"],
        matches=matches,
    )


def _collect_definitions(tree: ast.AST) -> list[TargetDefinition]:
    definitions: list[TargetDefinition] = []

    def visit_body(body: Sequence[ast.stmt], parents: tuple[str, ...]) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit_body(node.body, (*parents, node.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = ".".join((*parents, node.name))
                kind = "method" if parents else "function"
                definitions.append(
                    TargetDefinition(
                        qualified_name=qualified,
                        kind=kind,
                        lineno=node.lineno,
                        end_lineno=getattr(node, "end_lineno", None),
                    )
                )

    visit_body(getattr(tree, "body", []), ())
    return definitions


def _definition_matches(
    definitions: Sequence[TargetDefinition],
    target_function: str,
) -> list[TargetDefinition]:
    if "." in target_function:
        return [definition for definition in definitions if definition.qualified_name == target_function]
    exact_top_level = [
        definition
        for definition in definitions
        if definition.kind == "function" and definition.qualified_name == target_function
    ]
    if exact_top_level:
        return exact_top_level
    return [
        definition
        for definition in definitions
        if definition.kind == "method" and definition.qualified_name.split(".")[-1] == target_function
    ]


def _normalize_repo_relative_path(value: str) -> tuple[str | None, list[str]]:
    diagnostics: list[str] = []
    if not value:
        return None, ["source_file is required"]
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute():
        return None, ["source_file must be repository-relative, not absolute"]
    if any(part in {"", ".", ".."} for part in path.parts):
        return None, ["source_file must be normalized and must not contain '.', '..', or empty segments"]
    if normalized != path.as_posix():
        diagnostics.append(f"source_file normalized from {value!r} to {path.as_posix()!r}")
    return path.as_posix(), diagnostics


def _evidence_for(
    scenario: Mapping[str, Any],
    scenario_id: str,
    provided: Mapping[str, Any] | None,
    scenario_keys: Sequence[str],
) -> EvidenceValue:
    identities = _identity_candidates(scenario, scenario_id)
    if provided:
        for identity in identities:
            if identity in provided:
                return _clamp_optional(provided[identity], name=identity, provenance=f"explicit_map:{identity}")
    for key in scenario_keys:
        if key in scenario:
            return _clamp_optional(scenario.get(key), name=key, provenance=f"scenario_field:{key}")
    target = scenario.get("target_location")
    if isinstance(target, Mapping):
        for key in scenario_keys:
            if key in target:
                return _clamp_optional(target.get(key), name=key, provenance=f"target_location_field:{key}")
    return EvidenceValue(None, UNAVAILABLE, "missing", ("explicit evidence is unavailable",))


def _clamp_optional(value: Any, *, name: str, provenance: str = "explicit") -> EvidenceValue:
    if value is None:
        return EvidenceValue(None, UNAVAILABLE, provenance, (f"{name} is unavailable",))
    if isinstance(value, bool):
        return EvidenceValue(None, UNAVAILABLE, provenance, (f"{name} must be numeric, not bool",))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return EvidenceValue(None, UNAVAILABLE, provenance, (f"{name} must be numeric",))
    clamped = min(1.0, max(0.0, number))
    diagnostics: tuple[str, ...] = ()
    if clamped != number:
        diagnostics = (f"{name} clamped from {number} to {clamped}",)
    return EvidenceValue(round(clamped, 6), AVAILABLE, provenance, diagnostics)


def _rank_score_mapping(scores: Mapping[str, Any], *, score_name: str) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for identity, value in scores.items():
        evidence = _clamp_optional(value, name=f"{identity}.{score_name}")
        if evidence.status == AVAILABLE:
            ranked.append(
                {
                    "identity": identity,
                    score_name: evidence.value,
                    "diagnostics": list(evidence.diagnostics),
                }
            )
    ranked.sort(key=lambda item: (-float(item[score_name]), item["identity"]))
    for rank, item in enumerate(ranked, 1):
        item["rank"] = rank
    return ranked


def _ranking_key(record: Mapping[str, Any]) -> tuple[float, str, str, str]:
    scenario_score = record.get("ScenarioScore")
    score = float(scenario_score if scenario_score is not None else -1.0)
    scenario = record.get("scenario", {})
    source_file = _scenario_source_file(scenario if isinstance(scenario, Mapping) else {})
    target_function = _scenario_target_function(scenario if isinstance(scenario, Mapping) else {})
    return (-score, str(record.get("stable_identity", "")), source_file, target_function)


def _score_record_sort_key(record: Mapping[str, Any]) -> tuple[int, int, str]:
    rank = record.get("rank")
    if isinstance(rank, int):
        return (0, rank, str(record.get("stable_identity", "")))
    return (1, 0, str(record.get("stable_identity", "")))


def _mark_score_record(
    score_records: list[dict[str, Any]],
    *,
    stable_identity: str,
    rank: int,
    selected: bool,
) -> None:
    for score_record in score_records:
        if score_record.get("stable_identity") == stable_identity:
            score_record["rank"] = rank
            score_record["selected"] = selected
            return


def _existence_sort_key(result: TargetExistenceResult) -> tuple[str, str, str]:
    return (result.scenario_id, result.source_file, result.target_function)


def _rejected_sort_key(record: Mapping[str, Any]) -> tuple[str, str, int]:
    rank = record.get("rank")
    rank_value = rank if isinstance(rank, int) else 0
    return (str(record.get("scenario_id", "")), str(record.get("rejection_reason", "")), rank_value)


def _selected_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": record["scenario_id"],
        "scenario": record["scenario"],
        "rank": record["rank"],
        "selected": True,
        "stable_identity": record["stable_identity"],
        "scoring": _score_output_record(record),
    }


def _score_output_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": record["scenario_id"],
        "stable_identity": record["stable_identity"],
        "rank": record.get("rank"),
        "selected": record.get("selected", False),
        "status": record["status"],
        "ScenarioScore": record["ScenarioScore"],
        "weights": record["weights"],
        "components": record["components"],
        "diagnostics": record["diagnostics"],
        "score_range": record["score_range"],
        "higher_is_better": record["higher_is_better"],
    }


def _scenario_id(scenario: Mapping[str, Any], index: int) -> str:
    raw = scenario.get("scenario_id") or scenario.get("id")
    return str(raw) if raw else f"scenario-{index + 1}"


def _stable_identity(scenario: Mapping[str, Any], scenario_id: str) -> str:
    stimulus = scenario.get("stimulus_steps") or scenario.get("execution_stimulus") or []
    if isinstance(stimulus, (list, tuple)):
        stimulus_text = "|".join(str(item) for item in stimulus)
    else:
        stimulus_text = str(stimulus)
    return "::".join(
        [
            scenario_id,
            _scenario_source_file(scenario),
            _scenario_target_function(scenario),
            str(scenario.get("oracle_type") or ""),
            stimulus_text,
        ]
    )


def _identity_candidates(scenario: Mapping[str, Any], scenario_id: str) -> list[str]:
    source_file = _scenario_source_file(scenario)
    target_function = _scenario_target_function(scenario)
    return [
        scenario_id,
        _stable_identity(scenario, scenario_id),
        f"{source_file}:{target_function}",
        target_function,
        source_file,
    ]


def _scenario_source_file(scenario: Mapping[str, Any]) -> str:
    target = scenario.get("target_location")
    if isinstance(target, Mapping) and target.get("source_file"):
        return str(target.get("source_file") or "")
    return str(scenario.get("source_file") or "")


def _scenario_target_function(scenario: Mapping[str, Any]) -> str:
    target = scenario.get("target_location")
    if isinstance(target, Mapping) and target.get("target_function"):
        return str(target.get("target_function") or "")
    return str(scenario.get("target_function") or "")

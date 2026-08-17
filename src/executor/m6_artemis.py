from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from src.contracts.feature_flags import V22FeatureFlags, resolve_feature_flags
from src.executor.m6_flitsr import BLOCKING_UNSPECIFIED, DISABLED


@dataclass(frozen=True)
class ArtemisResult:
    status: str
    ranking: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_artemis_bubble_up(
    flitsr_output: Mapping[str, Any],
    *,
    feature_flags: V22FeatureFlags | Mapping[str, Any] | None = None,
) -> ArtemisResult:
    """Consume explicit FLITSR output and preserve rank fields.

    The approved repository specification does not define ARTEMIS bubble-up
    semantics or a distinct-fault evidence schema. This core therefore never
    fabricates multi-fault evidence. It preserves FLITSR base rank as
    ``base_rank`` and mirrors it to ``adjusted_rank`` only when disabled or
    blocked.
    """
    flags = (
        feature_flags
        if isinstance(feature_flags, V22FeatureFlags)
        else resolve_feature_flags(feature_flags)
    )
    base_ranking = _base_ranking(flitsr_output)
    metadata = {
        "feature_flag": "m6_artemis",
        "consumes_flitsr_output": True,
        "base_status": flitsr_output.get("status"),
    }
    if not flags.m6_artemis:
        return ArtemisResult(
            status=DISABLED,
            ranking=base_ranking,
            diagnostics=["m6_artemis disabled by feature flag"],
            metadata={**metadata, "enabled": False},
        )
    if flitsr_output.get("status") != "SUPPORTED":
        return ArtemisResult(
            status=BLOCKING_UNSPECIFIED,
            ranking=base_ranking,
            diagnostics=[
                "BLOCKING: ARTEMIS requires explicit supported FLITSR output",
                "BLOCKING: ARTEMIS bubble-up procedure is not defined by the approved specification",
            ],
            metadata={**metadata, "enabled": True},
        )
    return ArtemisResult(
        status=BLOCKING_UNSPECIFIED,
        ranking=base_ranking,
        diagnostics=[
            "BLOCKING: ARTEMIS distinct-fault evidence schema is not defined by the approved specification",
            "BLOCKING: ARTEMIS bubble-up procedure is not defined by the approved specification",
        ],
        metadata={**metadata, "enabled": True, "distinct_fault_evidence_used": False},
    )


def _base_ranking(flitsr_output: Mapping[str, Any]) -> list[dict[str, Any]]:
    final_order = flitsr_output.get("final_order")
    if not isinstance(final_order, list):
        return []
    ranking: list[dict[str, Any]] = []
    for index, item in enumerate(final_order, 1):
        if not isinstance(item, Mapping):
            continue
        base_rank = item.get("rank", index)
        ranking.append(
            {
                **dict(item),
                "base_rank": base_rank,
                "adjusted_rank": base_rank,
                "artemis_bubbled": False,
                "distinct_fault_evidence": None,
            }
        )
    ranking.sort(
        key=lambda item: (
            item["adjusted_rank"],
            str(item.get("source_file", "")),
            item.get("line_no", 0),
        )
    )
    return ranking

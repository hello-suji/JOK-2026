from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence


_PATCH_LEAKAGE_RE = re.compile(
    r"\b(golden[-_ ]?patch|post[-_ ]?patch|after[-_ ]?patch|fail[-_ ]?to[-_ ]?pass|"
    r"patch hit rate|phr|m8)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CandorOracleProposal:
    """Provider-independent CANDOR oracle proposal."""

    proposal_id: str
    oracle_type: str
    expected_behavior: str
    supporting_evidence: list[str]
    issue_evidence_behavior_alignment: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandorProposalValidation:
    proposal_id: str
    is_valid: bool
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandorDisagreementRecord:
    signature: str
    proposal_ids: list[str]
    oracle_type: str
    expected_behavior: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandorConsensusResult:
    enabled: bool
    proposals: list[CandorOracleProposal]
    validations: list[CandorProposalValidation]
    agreement_rate: float
    approved_threshold: float | None
    consensus_reached: bool
    consensus_signature: str | None
    consensus_proposal_ids: list[str]
    disagreements: list[CandorDisagreementRecord]
    fallback_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "proposals": [p.to_dict() for p in self.proposals],
            "validations": [v.to_dict() for v in self.validations],
            "agreement_rate": self.agreement_rate,
            "approved_threshold": self.approved_threshold,
            "consensus_reached": self.consensus_reached,
            "consensus_signature": self.consensus_signature,
            "consensus_proposal_ids": list(self.consensus_proposal_ids),
            "disagreements": [d.to_dict() for d in self.disagreements],
            "fallback_reason": self.fallback_reason,
        }


def build_candor_oracle_prompt(
    *,
    issue_evidence: Mapping[str, Any],
    scenario: Mapping[str, Any],
    oracle_hint: str = "",
) -> str:
    """Build a provider-independent prompt for independent oracle proposals."""

    payload = {
        "issue_evidence": dict(issue_evidence),
        "scenario": dict(scenario),
        "oracle_hint": oracle_hint,
        "required_json_shape": {
            "proposals": [
                {
                    "proposal_id": "stable unique id",
                    "oracle_type": "oracle category",
                    "expected_behavior": "expected issue behavior to assert",
                    "supporting_evidence": ["issue/scenario evidence"],
                    "issue_evidence_behavior_alignment": "why evidence supports behavior",
                }
            ]
        },
        "constraints": [
            "Generate independent oracle proposals.",
            "Use only issue evidence, scenario evidence, and pre-patch context.",
            "Do not use golden patch, post-patch, M8, Fail-to-Pass, or PHR information.",
        ],
    }
    return json.dumps(payload, sort_keys=True, indent=2)


def parse_candor_oracle_proposals(raw_response: str) -> list[CandorOracleProposal]:
    """Parse CANDOR proposal JSON without provider-specific assumptions."""

    data = json.loads(raw_response)
    proposals = data.get("proposals", data if isinstance(data, list) else [])
    if not isinstance(proposals, list):
        raise ValueError("CANDOR response must contain a proposals list")
    parsed: list[CandorOracleProposal] = []
    for index, item in enumerate(proposals, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"CANDOR proposal {index} must be an object")
        evidence = item.get("supporting_evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        parsed.append(
            CandorOracleProposal(
                proposal_id=str(item.get("proposal_id") or f"proposal-{index}"),
                oracle_type=str(item.get("oracle_type") or ""),
                expected_behavior=str(item.get("expected_behavior") or item.get("expected") or ""),
                supporting_evidence=[str(value) for value in evidence if str(value).strip()],
                issue_evidence_behavior_alignment=str(
                    item.get("issue_evidence_behavior_alignment")
                    or item.get("alignment")
                    or ""
                ),
            )
        )
    return parsed


def validate_candor_proposal(
    proposal: CandorOracleProposal,
    *,
    issue_evidence_terms: Iterable[str] = (),
) -> CandorProposalValidation:
    """Validate one proposal deterministically with patch-isolation guards."""

    errors: list[str] = []
    if not proposal.oracle_type.strip():
        errors.append("missing_oracle_type")
    if not proposal.expected_behavior.strip():
        errors.append("missing_expected_behavior")
    if not proposal.supporting_evidence:
        errors.append("missing_supporting_evidence")
    if not proposal.issue_evidence_behavior_alignment.strip():
        errors.append("missing_issue_evidence_behavior_alignment")

    combined = " ".join(
        [
            proposal.oracle_type,
            proposal.expected_behavior,
            *proposal.supporting_evidence,
            proposal.issue_evidence_behavior_alignment,
        ]
    )
    if _PATCH_LEAKAGE_RE.search(combined):
        errors.append("patch_or_post_patch_reference")

    terms = [str(term).strip().lower() for term in issue_evidence_terms if str(term).strip()]
    if terms:
        proposal_text = combined.lower()
        if not any(term in proposal_text for term in terms):
            errors.append("missing_issue_evidence_overlap")

    return CandorProposalValidation(
        proposal_id=proposal.proposal_id,
        is_valid=not errors,
        errors=errors,
    )


def evaluate_candor_consensus(
    proposals: Sequence[CandorOracleProposal],
    *,
    enabled: bool,
    approved_threshold: float | None,
    issue_evidence_terms: Iterable[str] = (),
) -> CandorConsensusResult:
    """Validate proposals and calculate deterministic majority agreement.

    Agreement rate is the largest valid oracle-signature group divided by the
    number of valid proposals. Empty valid populations have agreement 0.0.
    """

    if not enabled:
        return CandorConsensusResult(
            enabled=False,
            proposals=list(proposals),
            validations=[],
            agreement_rate=0.0,
            approved_threshold=approved_threshold,
            consensus_reached=False,
            consensus_signature=None,
            consensus_proposal_ids=[],
            disagreements=[],
            fallback_reason="feature_disabled_single_oracle_fallback",
        )

    validations = [
        validate_candor_proposal(proposal, issue_evidence_terms=issue_evidence_terms)
        for proposal in proposals
    ]
    valid_ids = {validation.proposal_id for validation in validations if validation.is_valid}
    valid_proposals = [proposal for proposal in proposals if proposal.proposal_id in valid_ids]
    if not valid_proposals:
        return CandorConsensusResult(
            enabled=True,
            proposals=list(proposals),
            validations=validations,
            agreement_rate=0.0,
            approved_threshold=approved_threshold,
            consensus_reached=False,
            consensus_signature=None,
            consensus_proposal_ids=[],
            disagreements=[],
            fallback_reason="no_valid_candor_proposals_single_oracle_fallback",
        )

    grouped: dict[str, list[CandorOracleProposal]] = {}
    for proposal in valid_proposals:
        grouped.setdefault(_oracle_signature(proposal), []).append(proposal)

    winner_signature, winner_group = sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )[0]
    agreement_rate = round(len(winner_group) / len(valid_proposals), 6)
    disagreements = [
        CandorDisagreementRecord(
            signature=signature,
            proposal_ids=[proposal.proposal_id for proposal in group],
            oracle_type=group[0].oracle_type,
            expected_behavior=group[0].expected_behavior,
        )
        for signature, group in sorted(grouped.items())
        if signature != winner_signature
    ]

    threshold_missing = approved_threshold is None
    threshold_met = False if threshold_missing else agreement_rate >= float(approved_threshold)
    return CandorConsensusResult(
        enabled=True,
        proposals=list(proposals),
        validations=validations,
        agreement_rate=agreement_rate,
        approved_threshold=approved_threshold,
        consensus_reached=threshold_met,
        consensus_signature=winner_signature if threshold_met else None,
        consensus_proposal_ids=[proposal.proposal_id for proposal in winner_group] if threshold_met else [],
        disagreements=disagreements,
        fallback_reason="" if threshold_met else (
            "missing_approved_consensus_threshold_single_oracle_fallback"
            if threshold_missing
            else "agreement_below_threshold_single_oracle_fallback"
        ),
    )


def _oracle_signature(proposal: CandorOracleProposal) -> str:
    return (
        f"{_normalize_signature_text(proposal.oracle_type)}::"
        f"{_normalize_signature_text(proposal.expected_behavior)}"
    )


def _normalize_signature_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())

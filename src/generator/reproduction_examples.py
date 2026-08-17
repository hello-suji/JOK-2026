from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from src.scenario.code_block_roles import (
    ROLE_ACTUAL_BUGGY_OUTPUT,
    ROLE_BUG_TRIGGER,
    ROLE_EXPECTED_OUTPUT,
    ROLE_SETUP,
    block_inferred_role,
    classify_reproduction_code_blocks,
    is_setup_only_block,
    strict_normalized_output_equals,
)


ORACLE_REGENERATION_STATUS = "requires_oracle_regeneration"
ORACLE_COMPLETE_STATUS = "complete"


@dataclass
class ReproductionExampleGroup:
    """A source-local reproduction example with directly paired outputs."""

    setup_blocks: List[Dict[str, Any]]
    stimulus_block: Dict[str, Any]
    expected_outputs: List[str]
    actual_outputs: List[str]
    source_index: int
    role: str
    role_evidence: str
    target_function_call: str
    oracle_requires_regeneration: bool = False
    oracle_pairing_status: str = ORACLE_COMPLETE_STATUS
    selected_example_id: str = ""
    stimulus_provenance: str = "issue_reproduction_code"
    expected_output_provenance: str = "direct_issue_expected_output"
    actual_output_provenance: str = "direct_issue_actual_output"
    structural_compatibility_status: str = "not_evaluated"
    structural_compatibility_detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def blocks(self) -> List[Dict[str, Any]]:
        return [*self.setup_blocks, self.stimulus_block]

    def metadata(self) -> Dict[str, Any]:
        return {
            "selected_example_id": self.selected_example_id,
            "source_index": self.source_index,
            "role": self.role,
            "role_evidence": self.role_evidence,
            "target_function_call": self.target_function_call,
            "stimulus_block_source": self.stimulus_provenance,
            "stimulus_provenance": self.stimulus_provenance,
            "expected_output_provenance": self.expected_output_provenance,
            "actual_output_provenance": self.actual_output_provenance,
            "oracle_pairing_status": self.oracle_pairing_status,
            "oracle_requires_regeneration": self.oracle_requires_regeneration,
            "requires_oracle_regeneration": self.oracle_requires_regeneration,
            "structural_compatibility_status": self.structural_compatibility_status,
            "structural_compatibility_detail": dict(self.structural_compatibility_detail),
        }


def coerce_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def dedup_text_items(values: Sequence[Any], *, limit: int = 8) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        norm = re.sub(r"\s+", " ", text.lower())
        if not text or norm in seen:
            continue
        seen.add(norm)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _dedup_group_outputs(values: Sequence[Any], *, limit: int = 4) -> List[str]:
    return dedup_text_items(values, limit=limit)


def _source_index(block: Dict[str, Any], fallback: int) -> int:
    try:
        return int(block.get("source_index", fallback))
    except (TypeError, ValueError):
        return fallback


def _block_input_text(block: Dict[str, Any]) -> str:
    return str(block.get("interactive_input") or block.get("code") or block.get("text") or "").strip()


def _scenario_target_function(scenario: Dict[str, Any]) -> str:
    target = scenario.get("target_location") if isinstance(scenario.get("target_location"), dict) else {}
    return str(scenario.get("target_function") or target.get("target_function") or "")


def _reproduction_block_output_candidates(block: Any) -> List[str]:
    candidates: List[str] = []
    if isinstance(block, dict):
        for key in ("interactive_output", "actual_output", "expected_output"):
            candidates.extend(str(value) for value in coerce_list(block.get(key)) if str(value).strip())
        for key in ("actual_outputs", "expected_outputs"):
            candidates.extend(str(value) for value in coerce_list(block.get(key)) if str(value).strip())
        for key in ("code", "text"):
            for line in str(block.get(key) or "").splitlines():
                cleaned = re.sub(r"^\s*(?:>>>|\.\.\.)\s?", "", line).strip()
                if cleaned:
                    candidates.append(cleaned)
        return dedup_text_items(candidates, limit=16)
    return dedup_text_items([str(block or "")], limit=1)


def _direct_block_outputs(
    block: Dict[str, Any],
    *,
    expected_outputs: Sequence[Any],
    actual_outputs: Sequence[Any],
) -> tuple[List[str], List[str]]:
    candidates = _reproduction_block_output_candidates(block)
    direct_expected: List[str] = []
    direct_actual: List[str] = []
    for value in coerce_list(block.get("expected_output")) + coerce_list(block.get("expected_outputs")):
        if str(value).strip():
            direct_expected.append(str(value).strip())
    for value in coerce_list(block.get("actual_output")) + coerce_list(block.get("actual_outputs")):
        if str(value).strip():
            direct_actual.append(str(value).strip())
    for expected in expected_outputs:
        if any(strict_normalized_output_equals(candidate, expected) for candidate in candidates):
            direct_expected.append(str(expected))
    for actual in actual_outputs:
        if any(strict_normalized_output_equals(candidate, actual) for candidate in candidates):
            direct_actual.append(str(actual))
    return _dedup_group_outputs(direct_expected), _dedup_group_outputs(direct_actual)


def shape_from_output(value: Any) -> tuple[int, ...] | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"(?:array|matrix)\s*\((\[.*\])\s*(?:,\s*dtype=.*)?\)\s*$", text, re.DOTALL)
    literal = match.group(1) if match else text
    if not literal.lstrip().startswith("["):
        return None
    try:
        parsed = ast.literal_eval(literal)
    except (SyntaxError, ValueError):
        normalized = re.sub(r"\bTrue\b", "True", literal)
        normalized = re.sub(r"\bFalse\b", "False", normalized)
        try:
            parsed = ast.literal_eval(normalized)
        except (SyntaxError, ValueError):
            return None

    def shape(obj: Any) -> tuple[int, ...] | None:
        if not isinstance(obj, list):
            return ()
        if not obj:
            return (0,)
        child_shapes = [shape(item) for item in obj]
        if any(child is None for child in child_shapes):
            return None
        first = child_shapes[0]
        if any(child != first for child in child_shapes):
            return None
        return (len(obj), *first)

    result = shape(parsed)
    return result if result else None


def outputs_structurally_compatible(
    selected_actual_outputs: Sequence[str],
    expected_output: Any,
) -> bool:
    expected_shape = shape_from_output(expected_output)
    if expected_shape is None:
        return True
    known_actual_shapes = [
        shape
        for actual in selected_actual_outputs
        for shape in [shape_from_output(actual)]
        if shape is not None
    ]
    if not known_actual_shapes:
        return True
    return expected_shape in known_actual_shapes


def structural_compatibility_status(
    selected_actual_outputs: Sequence[str],
    expected_outputs: Sequence[Any],
) -> tuple[str, Dict[str, Any]]:
    expected_shapes = [shape_from_output(value) for value in expected_outputs]
    actual_shapes = [shape_from_output(value) for value in selected_actual_outputs]
    known_expected = [shape for shape in expected_shapes if shape is not None]
    known_actual = [shape for shape in actual_shapes if shape is not None]
    detail = {
        "actual_shapes": [list(shape) for shape in known_actual],
        "expected_shapes": [list(shape) for shape in known_expected],
    }
    if not known_expected or not known_actual:
        return "not_determinable", detail
    if any(shape in known_actual for shape in known_expected):
        return "compatible", detail
    return "incompatible", detail


def _associated_following_output(
    block: Dict[str, Any],
    *,
    role: str,
    expected_outputs: Sequence[Any],
    actual_outputs: Sequence[Any],
) -> tuple[List[str], List[str]]:
    expected, actual = _direct_block_outputs(
        block,
        expected_outputs=expected_outputs,
        actual_outputs=actual_outputs,
    )
    if role == ROLE_ACTUAL_BUGGY_OUTPUT:
        actual.extend(
            str(value)
            for value in actual_outputs
            if strict_normalized_output_equals(block.get("code") or block.get("text"), value)
        )
    elif role == ROLE_EXPECTED_OUTPUT:
        expected.extend(
            str(value)
            for value in expected_outputs
            if strict_normalized_output_equals(block.get("code") or block.get("text"), value)
        )
    return _dedup_group_outputs(expected), _dedup_group_outputs(actual)


def _selected_example_id(block: Dict[str, Any], source_index: int, stimulus: str) -> str:
    explicit = str(block.get("selected_example_id") or block.get("example_id") or "").strip()
    if explicit:
        return explicit
    digest = hashlib.sha256(stimulus.encode("utf-8")).hexdigest()[:12]
    return f"issue-repro:{source_index}:{digest}"


def build_reproduction_example_groups(
    blocks: List[Any],
    scenario: Dict[str, Any],
) -> List[ReproductionExampleGroup]:
    expected_outputs = scenario.get("expected_outputs", []) or []
    actual_outputs = scenario.get("actual_outputs", []) or []
    classified = classify_reproduction_code_blocks(
        blocks,
        expected_outputs=expected_outputs,
        actual_outputs=actual_outputs,
        target_function=_scenario_target_function(scenario),
    )
    target_function = _scenario_target_function(scenario)
    groups: List[ReproductionExampleGroup] = []
    setup_context: List[Dict[str, Any]] = []

    for index, block in enumerate(classified):
        if not isinstance(block, dict):
            continue
        role = block_inferred_role(block)
        if role == ROLE_SETUP or is_setup_reproduction_block(block):
            setup_context.append(block)
            continue
        if role in {"expected_output", "actual_buggy_output"} and not _block_input_text(block):
            continue
        if not _block_input_text(block):
            continue

        direct_expected, direct_actual = _direct_block_outputs(
            block,
            expected_outputs=expected_outputs,
            actual_outputs=actual_outputs,
        )
        if index + 1 < len(classified) and isinstance(classified[index + 1], dict):
            following = classified[index + 1]
            following_role = block_inferred_role(following)
            if following_role in {"expected_output", "actual_buggy_output"}:
                following_expected, following_actual = _associated_following_output(
                    following,
                    role=following_role,
                    expected_outputs=expected_outputs,
                    actual_outputs=actual_outputs,
                )
                direct_expected.extend(following_expected)
                direct_actual.extend(following_actual)

        direct_actual = _dedup_group_outputs(direct_actual)
        direct_expected = [
            value
            for value in _dedup_group_outputs(direct_expected)
            if outputs_structurally_compatible(direct_actual, value)
        ]
        compatibility, compatibility_detail = structural_compatibility_status(
            direct_actual,
            direct_expected,
        )
        pairing_status = ORACLE_COMPLETE_STATUS if direct_expected else ORACLE_REGENERATION_STATUS
        stimulus = _block_input_text(block)
        source_index = _source_index(block, index)
        groups.append(
            ReproductionExampleGroup(
                setup_blocks=list(setup_context),
                stimulus_block=block,
                expected_outputs=direct_expected,
                actual_outputs=direct_actual,
                source_index=source_index,
                role=role,
                role_evidence=str(block.get("role_evidence") or ""),
                target_function_call=stimulus
                if target_function and re.search(rf"\b{re.escape(target_function.split('.')[-1])}\s*\(", stimulus)
                else "",
                oracle_requires_regeneration=not direct_expected,
                oracle_pairing_status=pairing_status,
                selected_example_id=_selected_example_id(block, source_index, stimulus),
                expected_output_provenance=(
                    "direct_issue_expected_output" if direct_expected else "unpaired_requires_regeneration"
                ),
                actual_output_provenance=(
                    "direct_issue_actual_output" if direct_actual else "not_available"
                ),
                structural_compatibility_status=compatibility,
                structural_compatibility_detail=compatibility_detail,
            )
        )
    return groups


def _role_evidence_strength(text: str) -> int:
    evidence = str(text or "")
    if "actual" in evidence:
        return 4
    if "bug" in evidence or "failing" in evidence:
        return 3
    if "expected" in evidence:
        return 2
    if evidence:
        return 1
    return 0


def trigger_group_rank(group: ReproductionExampleGroup) -> tuple[int, int, int, int, int, int, int, int]:
    block = group.stimulus_block
    role_text = " ".join(
        str(block.get(key, ""))
        for key in ("role", "label", "context_before", "text")
        if block.get(key)
    ).lower()
    problem_context = int(bool(re.search(r"\b(?:bug|fail|failing|problem|wrong|incorrect|repro)\b", role_text)))
    setup_dependency = int(bool(group.setup_blocks))
    target_call = int(bool(group.target_function_call))
    specificity = min(
        6,
        len(dedup_text_items(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\b", _block_input_text(block)), limit=12)),
    )
    return (
        int(group.role == ROLE_BUG_TRIGGER),
        int(bool(group.actual_outputs)),
        problem_context,
        setup_dependency,
        target_call,
        _role_evidence_strength(group.role_evidence),
        specificity,
        group.source_index,
    )


def select_reproduction_example_group(
    blocks: List[Any],
    scenario: Dict[str, Any],
) -> ReproductionExampleGroup | None:
    groups = build_reproduction_example_groups(blocks, scenario)
    trigger_groups = [group for group in groups if group.role == ROLE_BUG_TRIGGER]
    if not trigger_groups:
        return None
    return max(trigger_groups, key=trigger_group_rank)


def is_setup_reproduction_block(block: Any) -> bool:
    return block_inferred_role(block) == ROLE_SETUP or is_setup_only_block(block)


def selected_example_requires_oracle_regeneration(scenario: Dict[str, Any]) -> bool:
    selected = scenario.get("selected_reproduction_example")
    if not isinstance(selected, dict):
        return bool(
            scenario.get("oracle_requires_regeneration")
            or scenario.get("requires_oracle_regeneration")
            or scenario.get("oracle_pairing_status") == ORACLE_REGENERATION_STATUS
        )
    return bool(
        selected.get("requires_oracle_regeneration")
        or selected.get("oracle_requires_regeneration")
        or selected.get("oracle_pairing_status") == ORACLE_REGENERATION_STATUS
    )


def sanitize_oracle_regeneration_payload(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Remove unpaired expected-output hints for selected examples needing regeneration."""
    if not selected_example_requires_oracle_regeneration(scenario):
        return scenario
    sanitized = dict(scenario)
    unpaired_expected = [str(value) for value in sanitized.get("expected_outputs", []) or [] if str(value).strip()]
    sanitized["expected_outputs"] = []
    sanitized.pop("oracle_expected", None)
    sanitized["oracle_requires_regeneration"] = True
    sanitized["requires_oracle_regeneration"] = True
    sanitized["oracle_pairing_status"] = ORACLE_REGENERATION_STATUS
    sanitized["oracle_source"] = "requires_regeneration"
    contract = sanitized.get("oracle_contract")
    if isinstance(contract, dict):
        updated = dict(contract)
        updated["oracle_source"] = "requires_regeneration"
        updated["rule"] = (
            "No safely associated expected output is available for the selected reproduction stimulus. "
            "Regenerate an EB-grounded oracle; do not borrow unrelated expected outputs."
        )
        sanitized["oracle_contract"] = updated
    directive = sanitized.get("repair_directive")
    if isinstance(directive, dict):
        sanitized["repair_directive"] = sanitize_repair_directive(directive)
    for key in ("oracle", "expected_failure"):
        value = sanitized.get(key)
        if isinstance(value, str):
            sanitized[key] = _remove_fixed_expected_hint_text(value, unpaired_expected)
    return sanitized


def sanitize_repair_directive(directive: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(directive)
    sanitized["must_keep"] = [
        item
        for item in sanitized.get("must_keep", []) or []
        if not _looks_like_unpaired_expected_hint(item)
    ]
    sanitized["replacement_hints"] = [
        item
        for item in sanitized.get("replacement_hints", []) or []
        if not _looks_like_unpaired_expected_hint(item)
    ]
    return sanitized


def _looks_like_unpaired_expected_hint(value: Any) -> bool:
    text = str(value or "").lower()
    return bool(
        re.search(r"\b(?:fixed expected output|assert the fixed expected output|expected correct output)\b", text)
        or re.search(r"\bexpected[_ -]?outputs?\b", text)
    )


def _remove_fixed_expected_hint_text(value: str, unpaired_expected: Sequence[str] | None = None) -> str:
    lines = []
    for line in value.splitlines():
        if _looks_like_unpaired_expected_hint(line):
            continue
        normalized = re.sub(r"\s+", "", line.lower())
        if any(re.sub(r"\s+", "", expected.lower()) in normalized for expected in unpaired_expected or []):
            continue
        lines.append(line)
    return "\n".join(lines).strip()

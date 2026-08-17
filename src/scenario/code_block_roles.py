from __future__ import annotations

import ast
import copy
import re
from typing import Any, Dict, List, Mapping, Sequence


ROLE_SETUP = "setup/precondition"
ROLE_BASELINE = "baseline/sanity"
ROLE_BUG_TRIGGER = "bug_trigger"
ROLE_EXPECTED_OUTPUT = "expected_output"
ROLE_ACTUAL_BUGGY_OUTPUT = "actual_buggy_output"
ROLE_UNKNOWN = "unknown"

def ensure_reproduction_code_blocks(value: Any) -> List[Dict[str, Any]]:
    """Return reproduction-code block dictionaries without dropping metadata."""
    if isinstance(value, list):
        return [
            block
            for item in value
            for block in _coerce_reproduction_block(item)
            if block.get("code") or block.get("interactive_input") or block.get("interactive_output") or block.get("text")
        ]
    return _coerce_reproduction_block(value)


def classify_reproduction_code_blocks(
    blocks: Any,
    *,
    expected_outputs: Sequence[Any] | None = None,
    actual_outputs: Sequence[Any] | None = None,
    target_function: str = "",
) -> List[Dict[str, Any]]:
    """Add deterministic code-block roles derived from issue-local evidence.

    Existing role/type/label fields are preserved. The derived classification
    is additive (`inferred_role`, `role_evidence`) so older `{language, code}`
    blocks remain compatible.
    """
    normalized = ensure_reproduction_code_blocks(blocks)
    if not normalized:
        return []

    expected = [str(value) for value in expected_outputs or [] if str(value).strip()]
    actual = [str(value) for value in actual_outputs or [] if str(value).strip()]
    target = str(target_function or "").strip()

    pending_input_for_output: int | None = None
    for index, block in enumerate(normalized):
        role, evidence = _classify_single_block(
            block,
            expected_outputs=expected,
            actual_outputs=actual,
            target_function=target,
        )
        _set_inferred_role(block, role, evidence)

        if pending_input_for_output is not None and index != pending_input_for_output + 1:
            pending_input_for_output = None

        if (
            pending_input_for_output is not None
            and index == pending_input_for_output + 1
            and role in {ROLE_EXPECTED_OUTPUT, ROLE_ACTUAL_BUGGY_OUTPUT}
            and _is_output_only_block(block)
        ):
            paired_role = ROLE_BASELINE if role == ROLE_EXPECTED_OUTPUT else ROLE_BUG_TRIGGER
            paired_evidence = (
                "paired_following_expected_output"
                if role == ROLE_EXPECTED_OUTPUT
                else "paired_following_actual_buggy_output"
            )
            _upgrade_input_role(normalized[pending_input_for_output], paired_role, paired_evidence)
            pending_input_for_output = None
        elif _has_interactive_input(block):
            pending_input_for_output = (
                index
                if role not in {ROLE_EXPECTED_OUTPUT, ROLE_ACTUAL_BUGGY_OUTPUT, ROLE_SETUP}
                else None
            )
        else:
            pending_input_for_output = None

        output = str(block.get("interactive_output") or "")
        if output:
            if _matches_any_output(output, actual):
                _upgrade_input_role(block, ROLE_BUG_TRIGGER, "interactive_output_matches_actual_outputs")
            elif _matches_any_output(output, expected):
                _upgrade_input_role(block, ROLE_BASELINE, "interactive_output_matches_expected_outputs")

    return normalized


def block_inferred_role(block: Any) -> str:
    if not isinstance(block, Mapping):
        return ROLE_UNKNOWN
    return str(block.get("inferred_role") or ROLE_UNKNOWN)


def block_has_semantic_role_evidence(block: Any) -> bool:
    if not isinstance(block, Mapping):
        return False
    evidence = block.get("role_evidence")
    return bool(evidence and block_inferred_role(block) != ROLE_UNKNOWN)


def block_text(block: Any) -> str:
    if isinstance(block, Mapping):
        return "\n".join(
            str(block.get(key, ""))
            for key in (
                "role",
                "type",
                "label",
                "context_before",
                "text",
                "interactive_input",
                "code",
                "interactive_output",
            )
            if block.get(key)
        )
    return str(block or "")


def output_matches(value: Any, outputs: Sequence[Any]) -> bool:
    return _matches_any_output(str(value or ""), [str(output) for output in outputs if str(output).strip()])


def strict_normalized_output_equals(value: Any, expected: Any) -> bool:
    """Return true only when outputs are equal after deterministic normalization."""
    needle = _normalize_output(expected)
    if not needle:
        return False
    return needle in _normalized_output_candidates(value)


def contains_target_call(text: str, target_function: str) -> bool:
    target = str(target_function or "").strip()
    if not target:
        return False
    bare = re.escape(target.split(".")[-1])
    dotted = re.escape(target)
    return bool(
        re.search(rf"\b{bare}\s*\(", text or "")
        or re.search(rf"\b{dotted}\s*\(", text or "")
    )


def is_setup_only_block(block: Any, *, target_function: str = "") -> bool:
    text = block_text(block)
    code = ""
    if isinstance(block, Mapping):
        code = str(block.get("interactive_input") or block.get("code") or "")
    lower = text.lower()
    if re.search(r"\b(?:setup|precondition|fixture|import)\b", lower):
        return True
    if _is_import_only(code):
        return True
    if code and not contains_target_call(code, target_function) and _looks_object_construction_only(code):
        return True
    return False


def _coerce_reproduction_block(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [copy.deepcopy(dict(value))]
    if isinstance(value, str):
        return [{"language": "python", "code": value}]
    if value:
        return [{"language": "python", "code": str(value)}]
    return []


def _classify_single_block(
    block: Dict[str, Any],
    *,
    expected_outputs: Sequence[str],
    actual_outputs: Sequence[str],
    target_function: str,
) -> tuple[str, str]:
    text = block_text(block)
    metadata = " ".join(
        str(block.get(key, ""))
        for key in ("role", "type", "label", "context_before", "text")
        if block.get(key)
    ).lower()
    code = str(block.get("interactive_input") or block.get("code") or "")
    output = str(block.get("interactive_output") or "")

    if output and _matches_any_output(output, actual_outputs):
        return ROLE_ACTUAL_BUGGY_OUTPUT, "interactive_output_matches_actual_outputs"
    if output and _matches_any_output(output, expected_outputs):
        return ROLE_EXPECTED_OUTPUT, "interactive_output_matches_expected_outputs"
    if code and contains_target_call(code, target_function) and _matches_specific_output(text, actual_outputs):
        return ROLE_BUG_TRIGGER, "block_text_matches_actual_outputs_with_target_call"
    if code and contains_target_call(code, target_function) and _matches_specific_output(text, expected_outputs):
        return ROLE_BASELINE, "block_text_matches_expected_outputs_with_target_call"
    if _is_output_only_block(block) and _matches_any_output(text, actual_outputs):
        return ROLE_ACTUAL_BUGGY_OUTPUT, "block_text_matches_actual_outputs"
    if _is_output_only_block(block) and _matches_any_output(text, expected_outputs):
        return ROLE_EXPECTED_OUTPUT, "block_text_matches_expected_outputs"

    if re.search(r"\b(?:baseline|sanity|passes|works|control example)\b", metadata):
        return ROLE_BASELINE, "explicit_baseline_marker"
    if re.search(r"\b(?:setup|precondition|fixture|import)\b", metadata) or is_setup_only_block(block, target_function=target_function):
        return ROLE_SETUP, "setup_or_precondition_marker"

    has_bug_marker = bool(re.search(r"\b(?:bug|fail|failing|problem|wrong|incorrect|repro|actual)\b", metadata))
    if has_bug_marker and contains_target_call(code, target_function):
        return ROLE_BUG_TRIGGER, "explicit_bug_marker_with_target_call"

    return ROLE_UNKNOWN, ""


def _set_inferred_role(block: Dict[str, Any], role: str, evidence: str) -> None:
    block["inferred_role"] = role
    if evidence:
        block["role_evidence"] = evidence
    else:
        block.pop("role_evidence", None)


def _upgrade_input_role(block: Dict[str, Any], role: str, evidence: str) -> None:
    current = block_inferred_role(block)
    if current == ROLE_BUG_TRIGGER:
        return
    if current == ROLE_SETUP and role == ROLE_BASELINE:
        return
    _set_inferred_role(block, role, evidence)


def _has_interactive_input(block: Mapping[str, Any]) -> bool:
    if _is_output_only_block(block):
        return False
    return bool(str(block.get("interactive_input") or block.get("code") or "").strip())


def _is_output_only_block(block: Mapping[str, Any]) -> bool:
    if block.get("interactive_input"):
        return False
    code = str(block.get("code") or "").strip()
    text = str(block.get("text") or "").strip()
    metadata = " ".join(str(block.get(key, "")) for key in ("role", "type", "label")).lower()
    if re.search(r"\b(?:output|actual|expected|result)\b", metadata) and not _contains_any_call(code):
        return True
    return bool((code or text) and not _contains_any_call(code or text) and not re.search(r"^\s*(?:from|import)\b", code))


def _matches_any_output(text: str, outputs: Sequence[str]) -> bool:
    haystacks = _normalized_output_candidates(text)
    if not haystacks:
        return False
    for output in outputs:
        needle = _normalize_output(output)
        if not needle:
            continue
        if needle in haystacks:
            return True
    return False


def _matches_specific_output(text: str, outputs: Sequence[str]) -> bool:
    return any(
        len(_normalize_output(output)) >= 3 and _matches_any_output(text, [output])
        for output in outputs
    )


def _normalize_output(text: Any) -> str:
    lines = []
    for line in str(text or "").splitlines():
        cleaned = re.sub(r"^\s*(?:>>>|\.\.\.)\s?", "", line)
        if cleaned.strip():
            lines.append(cleaned.strip())
    value = re.sub(r"\s+", " ", " ".join(lines)).strip()
    value = re.sub(r"\s*([()\[\]{},:=+\-*/<>])\s*", r"\1", value)
    return value


def _normalized_output_candidates(text: Any) -> set[str]:
    raw = str(text or "")
    candidates = {_normalize_output(raw)}
    lines = raw.splitlines()
    non_prompt_lines = [
        line
        for line in lines
        if not re.match(r"^\s*(?:>>>|\.\.\.)\s+", line)
    ]
    if non_prompt_lines and len(non_prompt_lines) != len(lines):
        candidates.add(_normalize_output("\n".join(non_prompt_lines)))
    for line in lines:
        candidates.add(_normalize_output(line))
    return {candidate for candidate in candidates if candidate}


def _contains_any_call(text: str) -> bool:
    return bool(re.search(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*\(", text or ""))


def _is_import_only(code: str) -> bool:
    lines = [line.strip() for line in str(code or "").splitlines() if line.strip()]
    return bool(lines) and all(line.startswith("import ") or line.startswith("from ") for line in lines)


def _looks_object_construction_only(code: str) -> bool:
    stripped = str(code or "").strip()
    if not stripped:
        return False
    if re.search(r"\bassert\b|==|!=|pytest\.raises|self\.assert", stripped):
        return False
    try:
        tree = ast.parse(stripped)
    except SyntaxError:
        return bool(re.search(r"=", stripped) and re.search(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*\(", stripped))
    if not tree.body:
        return False
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            return False
        return False
    return True

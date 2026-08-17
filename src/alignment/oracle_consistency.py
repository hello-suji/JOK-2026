from __future__ import annotations

import ast
import re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Sequence

from src.generator.reproduction_examples import (
    ORACLE_REGENERATION_STATUS,
    outputs_structurally_compatible,
    selected_example_requires_oracle_regeneration,
)
from src.generator.relational_oracles import RELATIONAL_ORACLE_PROVENANCE
from src.scenario.code_block_roles import (
    ROLE_BUG_TRIGGER,
    block_inferred_role,
    classify_reproduction_code_blocks,
    contains_target_call,
    is_setup_only_block,
    strict_normalized_output_equals,
)


APPROVED_ORACLE_SOURCES = {
    "direct_issue_expected_output",
    "selected_example_expected_output",
    "validated_semantic_invariant",
    "expected_outputs",
    "issue_expected_outputs",
    "inferred_semantic",
    RELATIONAL_ORACLE_PROVENANCE,
}


@dataclass(frozen=True)
class OracleConsistencyResult:
    trigger_present: bool
    usable_oracle: bool
    status: str
    feedback_action: str = "REWRITE_ORACLE"
    reasons: List[str] = field(default_factory=list)
    structural_compatibility_status: str = "not_evaluated"
    selected_example_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def selected_example_metadata(
    scenario: Dict[str, Any],
    generated_test: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    for source in (
        (generated_test or {}).get("selected_reproduction_example"),
        ((generated_test or {}).get("metadata") or {}).get("selected_reproduction_example")
        if isinstance((generated_test or {}).get("metadata"), dict)
        else None,
        ((generated_test or {}).get("prompt_profile") or {}).get("selected_reproduction_example")
        if isinstance((generated_test or {}).get("prompt_profile"), dict)
        else None,
        scenario.get("selected_reproduction_example"),
    ):
        if isinstance(source, dict):
            return dict(source)
    return {}


def trigger_present_with_local_alias(
    issue_trigger: str,
    generated_code: str,
    *,
    target_function: str = "",
) -> bool:
    """Detect target(expr) when generated code uses local aliases like x = expr; target(x)."""
    expected_calls = _target_calls(issue_trigger, target_function=target_function)
    if not expected_calls:
        return False
    try:
        generated_tree = ast.parse(generated_code or "")
    except SyntaxError:
        return False
    for body in _straight_line_bodies(generated_tree):
        if _body_contains_expected_call(body, expected_calls):
            return True
    return False


def issue_bug_trigger_patterns(clue: Dict[str, Any], *, limit: int = 3) -> List[str]:
    actual_outputs = clue.get("actual_outputs", []) or []
    target_function = _clue_target_function(clue)
    if not target_function:
        return []
    classified_blocks = classify_reproduction_code_blocks(
        clue.get("code_examples", []) or [],
        expected_outputs=clue.get("expected_outputs", []) or [],
        actual_outputs=actual_outputs,
        target_function=target_function,
    )
    patterns: List[tuple[float, str]] = []
    for block in classified_blocks:
        if not isinstance(block, dict) or block.get("is_system_or_output"):
            continue
        code = str(block.get("interactive_input") or block.get("code") or "").strip()
        if not code:
            continue
        if block_inferred_role(block) != ROLE_BUG_TRIGGER:
            continue
        if is_setup_only_block(block, target_function=target_function):
            continue
        if target_function and not contains_target_call(code, target_function):
            continue
        metadata = " ".join(
            str(block.get(key, ""))
            for key in ("role", "label", "context_before", "text", "interactive_output")
        ).lower()
        score = 10.0
        if any(marker in metadata for marker in ("bug", "fail", "failing", "problem", "repro", "actual")):
            score += 4.0
        if any(str(output or "").strip() and str(output).strip().lower()[:80] in metadata for output in actual_outputs):
            score += 4.0
        call_count = len(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*\(", code))
        if call_count >= 2:
            score += 2.0
        if re.search(r"\b[A-Za-z_]\w*\([^()\n]*\b[A-Za-z_]\w*\s*\(", code):
            score += 2.0
        if re.search(r"\)\s*\.\s*[A-Za-z_]\w+\s*\(", code):
            score += 1.0
        if re.search(r"\b(?:baseline|sanity|passes|works)\b", metadata):
            score -= 2.0
        patterns.append((score, code[:240]))
    ordered = [pattern for _, pattern in sorted(patterns, key=lambda item: item[0], reverse=True)]
    return _dedupe_strs(ordered, limit=limit)


def generated_test_contains_bug_trigger(
    clue: Dict[str, Any],
    generated_test: Dict[str, Any],
) -> bool:
    patterns = issue_bug_trigger_patterns(clue)
    if not patterns:
        return False
    code = str(generated_test.get("test_code") or generated_test.get("append_block") or "")
    target_function = _clue_target_function(clue)
    for pattern in patterns:
        if trigger_present_with_local_alias(pattern, code, target_function=target_function):
            return True
    return False


def evaluate_oracle_consistency(
    scenario: Dict[str, Any],
    clue: Dict[str, Any],
    generated_test: Dict[str, Any],
) -> OracleConsistencyResult:
    validated_relational = has_repository_validated_relational_oracle(generated_test)
    metadata = selected_example_metadata(scenario, generated_test)
    if not metadata:
        return OracleConsistencyResult(
            trigger_present=generated_test_contains_bug_trigger(clue, generated_test),
            usable_oracle=True,
            status="legacy_provenance_unavailable",
        )

    code = str(generated_test.get("test_code") or generated_test.get("append_block") or "")
    trigger_present = generated_test_contains_bug_trigger(clue, generated_test)
    reasons: List[str] = []
    selected_id = str(metadata.get("selected_example_id") or "")
    expected_outputs = [str(value) for value in scenario.get("expected_outputs", []) or [] if str(value).strip()]
    actual_outputs = [str(value) for value in scenario.get("actual_outputs", []) or [] if str(value).strip()]
    clue_expected_outputs = [str(value) for value in clue.get("expected_outputs", []) or [] if str(value).strip()]
    requires_regeneration = selected_example_requires_oracle_regeneration(scenario)
    pairing_status = str(metadata.get("oracle_pairing_status") or scenario.get("oracle_pairing_status") or "")
    oracle_source = _oracle_source(scenario, generated_test)
    borrowed = _matched_values(code, clue_expected_outputs)
    direct_expected = _matched_values(code, expected_outputs)
    structural_status = str(metadata.get("structural_compatibility_status") or "not_evaluated")

    for value in borrowed:
        if value not in expected_outputs:
            reasons.append("expected_output_from_other_example")
            if not outputs_structurally_compatible(actual_outputs, value):
                reasons.append("deterministic_structural_incompatibility")
                structural_status = "incompatible"

    for value in direct_expected or borrowed:
        if not outputs_structurally_compatible(actual_outputs, value):
            reasons.append("deterministic_structural_incompatibility")
            structural_status = "incompatible"

    regeneration_pending = (
        requires_regeneration
        or pairing_status == ORACLE_REGENERATION_STATUS
    )
    if regeneration_pending and not validated_relational:
        reasons.append("oracle_pairing_status_requires_regeneration")
    if requires_regeneration and borrowed:
        reasons.append("unpaired_expected_output_used")
    if structural_status == "incompatible":
        reasons.append("deterministic_structural_incompatibility")
    if expected_outputs and not direct_expected and not _has_validated_invariant(scenario, generated_test):
        reasons.append("selected_expected_output_not_asserted")
    if not oracle_source:
        reasons.append("oracle_provenance_unknown")
    elif oracle_source not in APPROVED_ORACLE_SOURCES:
        reasons.append(f"oracle_provenance_unapproved:{oracle_source}")

    usable = not reasons or reasons == ["selected_expected_output_not_asserted"]
    if "selected_expected_output_not_asserted" in reasons and not _has_validated_invariant(scenario, generated_test):
        usable = False
    status = "usable" if usable else _primary_status(reasons)
    return OracleConsistencyResult(
        trigger_present=trigger_present,
        usable_oracle=usable,
        status=status,
        reasons=_dedupe_strs(reasons, limit=10),
        structural_compatibility_status=structural_status,
        selected_example_id=selected_id,
    )


def _primary_status(reasons: Sequence[str]) -> str:
    for preferred in (
        "deterministic_structural_incompatibility",
        "expected_output_from_other_example",
        "oracle_pairing_status_requires_regeneration",
        "oracle_provenance_unknown",
        "selected_expected_output_not_asserted",
    ):
        if preferred in reasons:
            return preferred
    return str(reasons[0]) if reasons else "usable"


def _oracle_source(scenario: Dict[str, Any], generated_test: Dict[str, Any]) -> str:
    contract = scenario.get("oracle_contract") if isinstance(scenario.get("oracle_contract"), dict) else {}
    metadata = generated_test.get("metadata") if isinstance(generated_test.get("metadata"), dict) else {}
    if has_repository_validated_relational_oracle(generated_test):
        return RELATIONAL_ORACLE_PROVENANCE
    for value in (
        generated_test.get("oracle_source"),
        metadata.get("oracle_source"),
        contract.get("oracle_source"),
        scenario.get("oracle_source"),
    ):
        text = str(value or "").strip()
        if text == RELATIONAL_ORACLE_PROVENANCE:
            continue
        if text and text != "requires_regeneration":
            return text
    return ""


def _matched_values(code: str, values: Sequence[str]) -> List[str]:
    return [
        value
        for value in values
        if value and _normal_value(value) in _normal_value(code)
    ]


def _normal_value(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").lower())


def _has_validated_invariant(scenario: Dict[str, Any], generated_test: Dict[str, Any]) -> bool:
    contract = scenario.get("oracle_contract") if isinstance(scenario.get("oracle_contract"), dict) else {}
    source = str(contract.get("oracle_source") or generated_test.get("oracle_source") or "")
    if has_repository_validated_relational_oracle(generated_test):
        return True
    if source == "validated_semantic_invariant":
        return True
    return bool(generated_test.get("validated_semantic_invariant"))


def has_repository_validated_relational_oracle(generated_test: Dict[str, Any]) -> bool:
    """Return true only for relational metadata assigned by the local validator."""
    if not isinstance(generated_test, dict):
        return False
    relational = generated_test.get("relational_oracle")
    if not isinstance(relational, dict):
        return False
    if relational.get("validated_provenance") != RELATIONAL_ORACLE_PROVENANCE:
        return False
    if relational.get("deterministic_validator_status") != "validated":
        return False
    if relational.get("relation_kind") != "equivalent_construction":
        return False
    if relational.get("relation_operator") != "equality":
        return False
    for key in ("target_function", "left_stimulus_expression", "right_comparator_expression"):
        if not str(relational.get(key) or "").strip():
            return False
    evidence = relational.get("evidence_source_identifiers")
    if not isinstance(evidence, list) or not any(str(item or "").strip() for item in evidence):
        return False
    reasons = relational.get("validation_reasons")
    if not isinstance(reasons, list) or "validated_issue_supported_relational_oracle" not in reasons:
        return False
    return True


def _target_calls(text: str, *, target_function: str = "") -> List[ast.Call]:
    try:
        tree = ast.parse(str(text or ""), mode="exec")
    except SyntaxError:
        try:
            expr = ast.parse(str(text or ""), mode="eval")
        except SyntaxError:
            return []
        tree = expr
    calls = _outer_calls(tree)
    if not target_function:
        return calls
    bare = target_function.split(".")[-1]
    return [node for node in calls if _call_name(node.func).split(".")[-1] == bare]


def _outer_calls(tree: ast.AST) -> List[ast.Call]:
    calls: List[ast.Call] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            calls.append(node)

    for child in ast.iter_child_nodes(tree):
        if isinstance(child, ast.Expr) and isinstance(child.value, ast.Call):
            calls.append(child.value)
        else:
            Visitor().visit(child)
    return calls


def _straight_line_bodies(tree: ast.AST) -> List[List[ast.stmt]]:
    if not isinstance(tree, ast.Module):
        return []
    module_body = [
        statement
        for statement in tree.body
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    bodies: List[List[ast.stmt]] = [module_body] if module_body else []
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_test_function(statement):
            bodies.append(statement.body)
        elif isinstance(statement, ast.ClassDef) and _is_test_class(statement):
            bodies.extend(
                item.body
                for item in statement.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_test_function(item)
            )
    return bodies


def _is_test_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return node.name.startswith("test_")


def _is_test_class(node: ast.ClassDef) -> bool:
    if node.name.startswith("Test"):
        return True
    return any(_base_name(base).endswith("TestCase") for base in node.bases)


def _base_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _base_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    return ""


def _body_contains_expected_call(
    body: Sequence[ast.stmt],
    expected_calls: Sequence[ast.Call],
) -> bool:
    aliases: Dict[str, ast.AST] = {}
    for statement in body:
        if _statement_matches_expected_call(statement, aliases, expected_calls):
            return True
        _update_aliases_after_statement(statement, aliases)
    return False


def _statement_matches_expected_call(
    statement: ast.stmt,
    aliases: Dict[str, ast.AST],
    expected_calls: Sequence[ast.Call],
) -> bool:
    if _is_unsupported_control_flow(statement):
        return False

    class ExecutedCallVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.matched = False

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_ListComp(self, node: ast.ListComp) -> None:
            return

        def visit_SetComp(self, node: ast.SetComp) -> None:
            return

        def visit_DictComp(self, node: ast.DictComp) -> None:
            return

        def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
            return

        def visit_Call(self, node: ast.Call) -> None:
            if self.matched:
                return
            expanded = _expand_aliases(node, aliases)
            if any(_same_call(expected, expanded) for expected in expected_calls):
                self.matched = True
                return
            self.generic_visit(node)

    visitor = ExecutedCallVisitor()
    visitor.visit(statement)
    return visitor.matched


def _update_aliases_after_statement(statement: ast.stmt, aliases: Dict[str, ast.AST]) -> None:
    if _is_unsupported_control_flow(statement):
        aliases.clear()
        return
    if isinstance(statement, ast.Assign):
        for target in statement.targets:
            for name in _assigned_names(target):
                aliases.pop(name, None)
        if len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name) and _safe_alias_value(statement.value):
            aliases[statement.targets[0].id] = statement.value
        return
    if isinstance(statement, ast.AnnAssign):
        for name in _assigned_names(statement.target):
            aliases.pop(name, None)
        if isinstance(statement.target, ast.Name) and statement.value is not None and _safe_alias_value(statement.value):
            aliases[statement.target.id] = statement.value
        return
    if isinstance(statement, ast.AugAssign):
        for name in _assigned_names(statement.target):
            aliases.pop(name, None)
        return
    for name in _assigned_names(statement):
        aliases.pop(name, None)


def _is_unsupported_control_flow(statement: ast.stmt) -> bool:
    return isinstance(
        statement,
        (
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.With,
            ast.AsyncWith,
            ast.Match,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
        ),
    )


def _assigned_names(node: ast.AST) -> List[str]:
    names: List[str] = []
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        names.append(node.id)
    for child in ast.iter_child_nodes(node):
        names.extend(_assigned_names(child))
    return names


def _safe_alias_value(node: ast.AST) -> bool:
    unsupported = (
        ast.Lambda,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.NamedExpr,
        ast.Yield,
        ast.YieldFrom,
        ast.Await,
    )
    return not any(isinstance(child, unsupported) for child in ast.walk(node))


def _expand_aliases(node: ast.AST, aliases: Dict[str, ast.AST]) -> ast.AST:
    class AliasExpander(ast.NodeTransformer):
        def visit_Name(self, name: ast.Name) -> ast.AST:
            replacement = aliases.get(name.id)
            if replacement is None:
                return name
            return ast.copy_location(replacement, name)

    return AliasExpander().visit(ast.fix_missing_locations(ast.parse(ast.unparse(node), mode="eval").body))


def _same_call(expected: ast.Call, actual: ast.AST) -> bool:
    if not isinstance(actual, ast.Call):
        return False
    if _call_name(expected.func).split(".")[-1] != _call_name(actual.func).split(".")[-1]:
        return False
    return ast.dump(_strip_context(expected), include_attributes=False) == ast.dump(
        _strip_context(actual),
        include_attributes=False,
    )


def _strip_context(node: ast.AST) -> ast.AST:
    return ast.fix_missing_locations(ast.parse(ast.unparse(node), mode="eval").body)


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        prefix = _call_name(func.value)
        return f"{prefix}.{func.attr}" if prefix else func.attr
    return ""


def _normal_form_for_call_presence(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").lower())


def _clue_target_function(clue: Dict[str, Any]) -> str:
    identifiers = clue.get("identifiers", {}) if isinstance(clue.get("identifiers"), dict) else {}
    for key in ("functions", "methods"):
        values = identifiers.get(key, []) if isinstance(identifiers.get(key, []), list) else []
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
    for location in clue.get("fault_locations", []) or []:
        if isinstance(location, dict) and location.get("function_name"):
            return str(location.get("function_name") or "").strip()
    return ""


def _dedupe_strs(items: Sequence[Any], *, limit: int = 8) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        norm = re.sub(r"\s+", " ", text.lower())
        if not text or norm in seen:
            continue
        seen.add(norm)
        result.append(text)
        if len(result) >= limit:
            break
    return result

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.generator.reproduction_examples import (
    ORACLE_REGENERATION_STATUS,
    selected_example_requires_oracle_regeneration,
)


RELATIONAL_ORACLE_PROVENANCE = "issue_supported_relational_oracle"

_EQUIVALENCE_EVIDENCE_RE = re.compile(
    r"\b(?:equivalent|same|again\s+as\s+expected|as\s+expected|independent|"
    r"separable|without\s+mixing|parallel|nest(?:ed|ing)?|flatten(?:ed|ing)?)\b",
    re.IGNORECASE,
)
_ASSERT_HELPERS = {
    "assert_array_equal",
    "assert_allclose",
    "assert_equal",
    "np.testing.assert_array_equal",
    "np.testing.assert_allclose",
    "numpy.testing.assert_array_equal",
    "numpy.testing.assert_allclose",
    "self.assertEqual",
    "self.assertListEqual",
    "self.assertTupleEqual",
}


@dataclass(frozen=True)
class RelationalOracleCandidate:
    relation_kind: str
    left_stimulus_expression: str
    right_comparator_expression: str
    target_function: str
    relation_operator: str
    evidence_source_identifiers: List[str] = field(default_factory=list)
    left_stimulus_provenance: str = "issue_reproduction_code"
    right_comparator_provenance: str = "issue_reproduction_code"
    relation_provenance: str = RELATIONAL_ORACLE_PROVENANCE
    deterministic_validator_status: str = "candidate"
    validation_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelationalOracleValidation:
    is_valid: bool
    candidate: Optional[RelationalOracleCandidate] = None
    provenance: str = ""
    reasons: List[str] = field(default_factory=list)

    def to_metadata(self) -> Dict[str, Any]:
        if not self.is_valid or self.candidate is None:
            return {
                "deterministic_validator_status": "invalid",
                "validation_reasons": list(self.reasons),
            }
        data = self.candidate.to_dict()
        data["deterministic_validator_status"] = "validated"
        data["validation_reasons"] = list(self.reasons)
        data["validated_provenance"] = RELATIONAL_ORACLE_PROVENANCE
        return data


@dataclass(frozen=True)
class _RelationalValidationScan:
    left_matches: List[ast.Call] = field(default_factory=list)
    right_matches: List[ast.Call] = field(default_factory=list)
    has_equality_assertion: bool = False


def build_issue_supported_relational_oracle_candidate(
    scenario: Mapping[str, Any],
    clue: Optional[Mapping[str, Any]] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> Optional[RelationalOracleCandidate]:
    """Find an issue-supported equivalent-construction oracle candidate.

    This is intentionally generic and patch-free.  It only recognizes a narrow
    structural case: a selected target call containing a setup alias whose
    expression is flattened in another issue code example with the same target
    function.  The relation is accepted only when issue/context text contains
    explicit equivalence-style evidence.
    """
    if not selected_example_requires_oracle_regeneration(dict(scenario)):
        return None
    selected = scenario.get("selected_reproduction_example")
    if not isinstance(selected, Mapping):
        return None
    left_call = _clean_code(str(selected.get("target_function_call") or ""))
    target = _scenario_target_function(scenario)
    if not left_call or not target:
        return None
    if not _call_uses_target(left_call, target):
        return None
    if not _has_equivalence_evidence(scenario, clue, context):
        return None

    setup_assignments = _setup_assignments(scenario)
    if not setup_assignments:
        return None
    left_arg = _single_target_call_argument(left_call, target)
    if left_arg is None:
        return None
    expanded_left = _expand_names(left_arg, setup_assignments)
    left_operands = _flatten_single_operator(expanded_left)
    if left_operands is None:
        return None

    for block in _issue_blocks(scenario, clue):
        call_text = _block_call_text(block)
        if not call_text or _normalize_src(call_text) == _normalize_src(left_call):
            continue
        if not _call_uses_target(call_text, target):
            continue
        right_arg = _single_target_call_argument(call_text, target)
        if right_arg is None:
            continue
        right_operands = _flatten_single_operator(right_arg)
        if right_operands is None:
            continue
        if [_normalize_ast(node) for node in left_operands] != [_normalize_ast(node) for node in right_operands]:
            continue
        return RelationalOracleCandidate(
            relation_kind="equivalent_construction",
            left_stimulus_expression=left_call,
            right_comparator_expression=call_text,
            target_function=target,
            relation_operator="equality",
            evidence_source_identifiers=_evidence_source_identifiers(scenario, clue, block),
        )
    return None


def validate_relational_oracle_candidate(
    code: str,
    candidate: Optional[RelationalOracleCandidate],
) -> RelationalOracleValidation:
    """Validate generated code against a repository-synthesized relation."""
    if candidate is None:
        return RelationalOracleValidation(False, reasons=["relational_oracle_candidate_absent"])
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        return RelationalOracleValidation(False, candidate, reasons=[f"syntax_error:{exc}"])

    reasons: List[str] = []
    if candidate.relation_provenance != RELATIONAL_ORACLE_PROVENANCE:
        reasons.append("relational_candidate_provenance_not_repository_assigned")
    if candidate.relation_kind != "equivalent_construction":
        reasons.append("unsupported_relation_kind")
    if candidate.relation_operator != "equality":
        reasons.append("unsupported_relation_operator")

    target = candidate.target_function
    expected_left = _single_target_call_argument(candidate.left_stimulus_expression, target)
    expected_right = _single_target_call_argument(candidate.right_comparator_expression, target)
    if expected_left is None or expected_right is None:
        reasons.append("candidate_target_calls_not_parseable")
        return RelationalOracleValidation(False, candidate, reasons=_dedupe(reasons))

    scan = _scan_relational_validation(tree, target, expected_left, expected_right)
    left_matches = scan.left_matches
    right_matches = scan.right_matches
    if not left_matches:
        reasons.append("left_stimulus_target_result_missing")
    if not right_matches:
        reasons.append("right_comparator_target_result_missing")
    if any(left is right for left in left_matches for right in right_matches):
        reasons.append("both_sides_not_independently_evaluated")

    if not scan.has_equality_assertion:
        reasons.append("relational_equality_assertion_missing")
    if _has_literal_expected_comparison(tree):
        reasons.append("literal_expected_value_or_array_used")
    if _has_negative_only_assertions(tree):
        reasons.append("negative_only_assertion")
    if _has_structural_only_assertions(tree):
        reasons.append("structural_only_assertion")

    valid = not reasons
    return RelationalOracleValidation(
        valid,
        candidate,
        provenance=RELATIONAL_ORACLE_PROVENANCE if valid else "",
        reasons=["validated_issue_supported_relational_oracle"] if valid else _dedupe(reasons),
    )


def relational_oracle_prompt_section(candidate: Optional[RelationalOracleCandidate]) -> str:
    if candidate is None:
        return ""
    return (
        "\n[Issue-Supported Relational Oracle]\n"
        "The selected reproduction stimulus has no directly paired expected output, but the issue/context "
        "supports an equivalent-construction oracle. Compare both target results directly; do not invent "
        "a literal expected value.\n"
        f"- relation_kind: {candidate.relation_kind}\n"
        f"- target_function: {candidate.target_function}\n"
        f"- left stimulus: {candidate.left_stimulus_expression}\n"
        f"- right comparator: {candidate.right_comparator_expression}\n"
        f"- relation_operator: {candidate.relation_operator}\n"
        f"- relation_provenance: {candidate.relation_provenance}\n"
        "Required assertion pattern: store the result of each target call in separate variables, then assert equality "
        "with the project's normal equality helper, such as assert left == right or assert_array_equal(left, right).\n"
    )


def _scenario_target_function(scenario: Mapping[str, Any]) -> str:
    target = scenario.get("target_location") if isinstance(scenario.get("target_location"), Mapping) else {}
    return str(scenario.get("target_function") or target.get("target_function") or "").strip()


def _issue_blocks(scenario: Mapping[str, Any], clue: Optional[Mapping[str, Any]]) -> List[Any]:
    blocks: List[Any] = []
    for source in (scenario.get("reproduction_code"), (clue or {}).get("code_examples")):
        if isinstance(source, list):
            blocks.extend(source)
    return blocks


def _block_call_text(block: Any) -> str:
    if isinstance(block, Mapping):
        return _clean_code(str(block.get("interactive_input") or block.get("code") or ""))
    return _clean_code(str(block or ""))


def _clean_code(value: str) -> str:
    lines = []
    for line in str(value or "").splitlines():
        cleaned = re.sub(r"^\s*(?:>>>|\.\.\.)\s?", "", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines).strip()


def _setup_assignments(scenario: Mapping[str, Any]) -> Dict[str, ast.AST]:
    assignments: Dict[str, ast.AST] = {}
    for block in scenario.get("reproduction_code") or []:
        if not isinstance(block, Mapping):
            continue
        code = _clean_code(str(block.get("code") or ""))
        if not code:
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                assignments[node.targets[0].id] = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                assignments[node.target.id] = node.value
    return assignments


def _single_target_call_argument(call_text: str, target_function: str) -> Optional[ast.AST]:
    try:
        tree = ast.parse(_clean_code(call_text), mode="exec")
    except SyntaxError:
        return None
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name_matches(node.func, target_function) and len(node.args) == 1
    ]
    return calls[0].args[0] if len(calls) == 1 else None


def _call_uses_target(call_text: str, target_function: str) -> bool:
    return _single_target_call_argument(call_text, target_function) is not None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _call_name_matches(node: ast.AST, target_function: str) -> bool:
    name = _call_name(node)
    bare = target_function.split(".")[-1]
    return bool(name and (name == target_function or name.split(".")[-1] == bare))


def _expand_names(node: ast.AST, assignments: Mapping[str, ast.AST]) -> ast.AST:
    class Expander(ast.NodeTransformer):
        def visit_Name(self, name_node: ast.Name) -> ast.AST:  # noqa: N802
            replacement = assignments.get(name_node.id)
            if replacement is None:
                return name_node
            return ast.copy_location(ast.fix_missing_locations(replacement), name_node)

    return ast.fix_missing_locations(Expander().visit(ast.fix_missing_locations(node)))


def _flatten_single_operator(node: ast.AST) -> Optional[List[ast.AST]]:
    if not isinstance(node, ast.BinOp):
        return None
    op_type = type(node.op)
    operands: List[ast.AST] = []

    def visit(expr: ast.AST) -> bool:
        if isinstance(expr, ast.BinOp):
            if type(expr.op) is not op_type:
                return False
            return visit(expr.left) and visit(expr.right)
        operands.append(expr)
        return True

    return operands if visit(node) and len(operands) >= 2 else None


def _normalize_ast(node: ast.AST) -> str:
    return re.sub(r"\s+", "", ast.unparse(ast.fix_missing_locations(node)))


def _normalize_src(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _has_equivalence_evidence(
    scenario: Mapping[str, Any],
    clue: Optional[Mapping[str, Any]],
    context: Optional[Mapping[str, Any]],
) -> bool:
    values: List[str] = []
    for source in (scenario, clue or {}, context or {}):
        for key in ("expected_behavior", "observed_behavior", "repro_conditions", "raw_issue_text", "oracle", "oracle_hints"):
            value = source.get(key) if isinstance(source, Mapping) else None
            if isinstance(value, list):
                values.extend(str(item) for item in value)
            elif value:
                values.append(str(value))
    return bool(_EQUIVALENCE_EVIDENCE_RE.search("\n".join(values)))


def _evidence_source_identifiers(
    scenario: Mapping[str, Any],
    clue: Optional[Mapping[str, Any]],
    block: Any,
) -> List[str]:
    identifiers = ["selected_reproduction_example"]
    selected = scenario.get("selected_reproduction_example")
    if isinstance(selected, Mapping) and selected.get("selected_example_id"):
        identifiers.append(str(selected["selected_example_id"]))
    if isinstance(block, Mapping):
        if block.get("source_index") is not None:
            identifiers.append(f"issue_code_block:{block.get('source_index')}")
        elif block.get("interactive_input"):
            identifiers.append("issue_code_block:interactive_input")
    if (clue or {}).get("raw_issue_text"):
        identifiers.append("raw_issue_text")
    return _dedupe(identifiers)


def _straight_line_bodies(tree: ast.AST) -> List[List[ast.stmt]]:
    bodies: List[List[ast.stmt]] = []
    if isinstance(tree, ast.Module):
        module_body = [
            statement
            for statement in tree.body
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if module_body:
            bodies.append(module_body)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                bodies.append(list(node.body))
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test"):
                        bodies.append(list(child.body))
    return bodies


def _scan_relational_validation(
    tree: ast.AST,
    target_function: str,
    expected_left: ast.AST,
    expected_right: ast.AST,
) -> _RelationalValidationScan:
    left_matches: List[ast.Call] = []
    right_matches: List[ast.Call] = []
    has_equality_assertion = False

    for body in _straight_line_bodies(tree):
        aliases: Dict[str, ast.AST] = {}
        body_target_assignments: Dict[str, ast.Call] = {}
        body_left_matches: List[ast.Call] = []
        body_right_matches: List[ast.Call] = []
        for statement in body:
            if _is_unsupported_control_flow(statement):
                aliases.clear()
                body_target_assignments.clear()
                continue

            statement_calls = _target_calls_in_statement(statement, target_function)
            for call in statement_calls:
                if not call.args:
                    continue
                if _expr_matches_exact_or_alias(call.args[0], expected_left, aliases):
                    body_left_matches.append(call)
                    left_matches.append(call)
                if _expr_matches_exact_or_alias(call.args[0], expected_right, aliases):
                    body_right_matches.append(call)
                    right_matches.append(call)

            if _statement_has_direct_equality_assertion(
                statement,
                body_target_assignments,
                body_left_matches,
                body_right_matches,
                target_function,
            ):
                has_equality_assertion = True

            _record_target_assignments(statement, target_function, body_target_assignments)
            _update_aliases_after_statement(statement, aliases)

    return _RelationalValidationScan(
        left_matches=left_matches,
        right_matches=right_matches,
        has_equality_assertion=has_equality_assertion,
    )


def _target_calls_in_statement(statement: ast.stmt, target_function: str) -> List[ast.Call]:
    return [
        node
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and _call_name_matches(node.func, target_function)
    ]


def _record_target_assignments(
    statement: ast.stmt,
    target_function: str,
    target_assignments: Dict[str, ast.Call],
) -> None:
    if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call) and _call_name_matches(statement.value.func, target_function):
        for target_node in statement.targets:
            if isinstance(target_node, ast.Name):
                target_assignments[target_node.id] = statement.value
    elif (
        isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and isinstance(statement.value, ast.Call)
        and _call_name_matches(statement.value.func, target_function)
    ):
        target_assignments[statement.target.id] = statement.value


def _update_aliases_after_statement(statement: ast.stmt, aliases: Dict[str, ast.AST]) -> None:
    if isinstance(statement, ast.Assign):
        if len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            aliases[statement.targets[0].id] = statement.value
        else:
            aliases.clear()
        return
    if isinstance(statement, ast.AnnAssign):
        if isinstance(statement.target, ast.Name) and statement.value is not None:
            aliases[statement.target.id] = statement.value
        elif isinstance(statement.target, ast.Name):
            aliases.pop(statement.target.id, None)
        else:
            aliases.clear()
        return
    if isinstance(statement, ast.AugAssign):
        for name in _assigned_names(statement.target):
            aliases.pop(name, None)
        return
    if _is_unsupported_control_flow(statement):
        aliases.clear()


def _is_unsupported_control_flow(statement: ast.stmt) -> bool:
    control_flow_types = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)
    match_type = getattr(ast, "Match", None)
    if match_type is not None:
        control_flow_types = (*control_flow_types, match_type)
    return isinstance(statement, control_flow_types)


def _assigned_names(target: ast.AST) -> List[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: List[str] = []
        for item in target.elts:
            names.extend(_assigned_names(item))
        return names
    return []


def _statement_has_direct_equality_assertion(
    statement: ast.stmt,
    target_assignments: Mapping[str, ast.Call],
    left_matches: Sequence[ast.Call],
    right_matches: Sequence[ast.Call],
    target_function: str,
) -> bool:
    left_ids = {name for name, call in target_assignments.items() if call in left_matches}
    right_ids = {name for name, call in target_assignments.items() if call in right_matches}

    def side_kind(expr: ast.AST) -> str:
        if isinstance(expr, ast.Name):
            if expr.id in left_ids:
                return "left"
            if expr.id in right_ids:
                return "right"
        if isinstance(expr, ast.Call) and _call_name_matches(expr.func, target_function):
            if expr in left_matches:
                return "left"
            if expr in right_matches:
                return "right"
        return ""

    for node in ast.walk(statement):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            operands = [node.test.left, *node.test.comparators]
            if any(not isinstance(op, (ast.Eq, ast.Is)) for op in node.test.ops):
                continue
            kinds = {side_kind(operand) for operand in operands}
            if {"left", "right"} <= kinds:
                return True
        if isinstance(node, ast.Call) and _call_name(node.func) in _ASSERT_HELPERS and len(node.args) >= 2:
            kinds = {side_kind(node.args[0]), side_kind(node.args[1])}
            if {"left", "right"} <= kinds:
                return True
    return False


def _expr_matches_with_aliases(expr: ast.AST, expected: ast.AST, aliases: Mapping[str, ast.AST]) -> bool:
    if _normalize_ast(expr) == _normalize_ast(expected):
        return True
    expanded = _expand_names(expr, aliases)
    if _normalize_ast(expanded) == _normalize_ast(expected):
        return True
    left = _flatten_single_operator(expanded)
    right = _flatten_single_operator(expected)
    return bool(left and right and [_normalize_ast(x) for x in left] == [_normalize_ast(x) for x in right])


def _expr_matches_exact_or_alias(expr: ast.AST, expected: ast.AST, aliases: Mapping[str, ast.AST]) -> bool:
    if _normalize_ast(expr) == _normalize_ast(expected):
        return True
    if isinstance(expr, ast.Name):
        replacement = aliases.get(expr.id)
        return bool(replacement is not None and _normalize_ast(replacement) == _normalize_ast(expected))
    return False


def _has_literal_expected_comparison(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            if any(isinstance(item, (ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Constant)) for item in [node.test.left, *node.test.comparators]):
                return True
        if isinstance(node, ast.Call) and _call_name(node.func) in _ASSERT_HELPERS:
            if any(isinstance(arg, (ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Constant)) for arg in node.args[:2]):
                return True
    return False


def _has_negative_only_assertions(tree: ast.AST) -> bool:
    assertions: List[ast.AST] = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    assertions.extend(node for node in ast.walk(tree) if isinstance(node, ast.Call) and re.search(r"assert(Not|False|NotEqual|IsNot)", _call_name(node.func)))
    if not assertions:
        return False
    negative = 0
    for node in assertions:
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            if any(isinstance(op, (ast.NotEq, ast.IsNot, ast.NotIn)) for op in node.test.ops):
                negative += 1
        elif isinstance(node, ast.Call) and re.search(r"assert(Not|False|NotEqual|IsNot)", _call_name(node.func)):
            negative += 1
    return negative == len(assertions)


def _has_structural_only_assertions(tree: ast.AST) -> bool:
    assertions = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    if not assertions:
        return False
    structural = 0
    for node in assertions:
        text = ast.unparse(node.test).lower()
        if re.search(r"\.shape\b|\.dtype\b|isinstance\s*\(|len\s*\(|is\s+not\s+none|np\.isfinite|sum\s*\(", text):
            structural += 1
    return structural == len(assertions)


def _dedupe(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result

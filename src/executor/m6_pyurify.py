from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from src.contracts.feature_flags import V22FeatureFlags, resolve_feature_flags


SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"
DISABLED = "DISABLED"


@dataclass(frozen=True)
class AtomizedTestVariant:
    canonical_test_id: str
    variant_id: str
    variant_suffix: str
    source: str
    oracle: dict[str, Any]
    preserved_statement_lines: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AtomizationResult:
    status: str
    canonical_test_id: str
    variants: list[AtomizedTestVariant] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["variants"] = [variant.to_dict() for variant in self.variants]
        return data


def atomize_assertions(
    source: str,
    *,
    canonical_test_id: str,
    feature_flags: V22FeatureFlags | Mapping[str, Any] | None = None,
) -> AtomizationResult:
    """Atomize one Python test source into assertion-focused variants.

    Supported population: top-level pytest-style test functions with top-level
    ``assert`` statements. Each variant contains imports, fixtures/helper
    definitions, the original test signature, prerequisite non-oracle
    statements that appear before the selected oracle, and exactly one oracle.

    Dynamic slicing is not inferred here; metadata explicitly reports it as
    ``UNSUPPORTED`` because no dynamic dependency evidence is collected by this
    deterministic core.
    """
    flags = (
        feature_flags
        if isinstance(feature_flags, V22FeatureFlags)
        else resolve_feature_flags(feature_flags)
    )
    if not flags.m6_pyurify:
        return AtomizationResult(
            status=DISABLED,
            canonical_test_id=canonical_test_id,
            diagnostics=["m6_pyurify disabled by feature flag"],
            metadata={"feature_flag": "m6_pyurify", "enabled": False},
        )
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        return _unsupported(canonical_test_id, f"syntax error during atomization: {exc.msg}")

    test_functions = [
        node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")
    ]
    if len(test_functions) != 1:
        return _unsupported(
            canonical_test_id,
            f"expected exactly one test function, found {len(test_functions)}",
        )
    test_function = test_functions[0]
    oracle_indexes = [
        index for index, statement in enumerate(test_function.body)
        if _is_supported_oracle(statement)
    ]
    nested_oracles = [
        node for statement in test_function.body
        for node in ast.walk(statement)
        if node is not statement and _is_supported_oracle(node)
    ]
    if nested_oracles:
        return _unsupported(
            canonical_test_id,
            "nested oracle statements require dynamic dependency evidence",
        )
    if not oracle_indexes:
        return _unsupported(canonical_test_id, "no supported top-level oracle statements found")

    variants: list[AtomizedTestVariant] = []
    for variant_number, oracle_index in enumerate(oracle_indexes, 1):
        variant_suffix = f"__oracle_{variant_number:03d}"
        variant_function = _copy_function_with_single_oracle(
            test_function,
            oracle_index=oracle_index,
            variant_suffix=variant_suffix,
        )
        variant_module = ast.Module(
            body=[
                _clone_statement(statement)
                if statement is not test_function else variant_function
                for statement in module.body
            ],
            type_ignores=list(module.type_ignores),
        )
        ast.fix_missing_locations(variant_module)
        oracle = test_function.body[oracle_index]
        variants.append(
            AtomizedTestVariant(
                canonical_test_id=canonical_test_id,
                variant_id=f"{canonical_test_id}{variant_suffix}",
                variant_suffix=variant_suffix,
                source=ast.unparse(variant_module) + "\n",
                oracle={
                    "oracle_index": variant_number,
                    "oracle_kind": _oracle_kind(oracle),
                    "lineno": getattr(oracle, "lineno", None),
                    "source": ast.get_source_segment(source, oracle) or ast.unparse(oracle),
                },
                preserved_statement_lines=[
                    getattr(statement, "lineno", 0)
                    for statement in test_function.body[:oracle_index]
                    if not _is_supported_oracle(statement)
                ],
                metadata={
                    "dynamic_slicing_status": UNSUPPORTED,
                    "dynamic_slicing_reason": "no dynamic dependency evidence collected",
                    "oracle_count": 1,
                },
            )
        )
    return AtomizationResult(
        status=SUPPORTED,
        canonical_test_id=canonical_test_id,
        variants=variants,
        metadata={
            "original_oracle_count": len(oracle_indexes),
            "variant_count": len(variants),
            "identity_policy": "canonical_test_id plus deterministic oracle suffix",
        },
    )


def _unsupported(canonical_test_id: str, reason: str) -> AtomizationResult:
    return AtomizationResult(
        status=UNSUPPORTED,
        canonical_test_id=canonical_test_id,
        diagnostics=[reason],
        metadata={"dynamic_slicing_status": UNSUPPORTED},
    )


def _is_supported_oracle(node: ast.AST) -> bool:
    if isinstance(node, ast.Assert):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        return _is_unittest_assert_call(node.value)
    return False


def _is_unittest_assert_call(call: ast.Call) -> bool:
    func = call.func
    return isinstance(func, ast.Attribute) and func.attr.startswith("assert")


def _oracle_kind(node: ast.AST) -> str:
    if isinstance(node, ast.Assert):
        return "python_assert"
    return "unittest_assert_call"


def _copy_function_with_single_oracle(
    test_function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    oracle_index: int,
    variant_suffix: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    copied = _clone_statement(test_function)
    copied.name = f"{test_function.name}{variant_suffix}"
    copied.body = [
        _clone_statement(statement)
        for index, statement in enumerate(test_function.body[: oracle_index + 1])
        if index == oracle_index or not _is_supported_oracle(statement)
    ]
    return copied


def _clone_statement(statement: Any) -> Any:
    return ast.fix_missing_locations(ast.parse(ast.unparse(statement)).body[0])

from __future__ import annotations

import ast
import copy
import difflib
import importlib.util
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.models.client import LLMClient, estimate_prompt_tokens
from src.models.config import load_model_config
from src.contracts.feature_flags import V22FeatureFlags, core_only_feature_flags
from src.contracts.models import FinalSetMembership
from src.contracts.status import CandidateStatus
from src.contracts.v37_oracle_flags import (
    ORACLE_ASSERTION_MISSING,
    ORACLE_EXPECTED_MISSING,
    ORACLE_SEMANTICS_CHANGED_BY_REPAIR,
    ORACLE_SPEC_MISMATCH,
    validated_v37_blocking_oracle_flags,
)
from src.contracts.v31 import (
    ImportManifestEntryV31,
    M5TelemetryV31,
    OracleContractV31,
    OracleTraceV31,
    TestGenerationContractV31,
    normalize_target_hypotheses,
    V31_SCHEMA_VERSION,
)
from src.generator.reproduction_examples import (
    ReproductionExampleGroup,
    build_reproduction_example_groups,
    outputs_structurally_compatible,
    sanitize_oracle_regeneration_payload,
    sanitize_repair_directive,
    select_reproduction_example_group,
    selected_example_requires_oracle_regeneration,
    shape_from_output,
    trigger_group_rank,
)
from src.generator.relational_oracles import (
    RELATIONAL_ORACLE_PROVENANCE,
    RelationalOracleValidation,
    build_issue_supported_relational_oracle_candidate,
    relational_oracle_prompt_section,
    validate_relational_oracle_candidate,
)
from src.generator.m5_candor import CandorConsensusResult
from src.generator.m5a_llm_error_refinement import ErrorRefinementResult
from src.generator.repair_loop import validation_status_from_errors
from src.scenario.code_block_roles import (
    ROLE_BASELINE,
    ROLE_BUG_TRIGGER,
    ROLE_ACTUAL_BUGGY_OUTPUT,
    ROLE_EXPECTED_OUTPUT,
    ROLE_SETUP,
    block_has_semantic_role_evidence,
    block_inferred_role,
    classify_reproduction_code_blocks,
    is_setup_only_block,
    strict_normalized_output_equals,
)
from src.utils.artifact_hash import sha256_text
from src.utils.file_io import read_text
from src.utils.scenario_utils import select_primary_scenario

logger = logging.getLogger(__name__)

_PROMPT_RAW_ISSUE_CHARS = 500
_PROMPT_CODE_EXAMPLES_MAX = 2
_PROMPT_CODE_CHARS = 350
_PROMPT_INTERACTIVE_CHARS = 180
_PROMPT_OUTPUTS_MAX = 2
_PROMPT_OUTPUT_CHARS = 180
_PROMPT_TEST_EXAMPLE_CHARS = 300
_PROMPT_IMPORT_MODULES = 8
_PROMPT_IMPORT_SYMBOLS = 8
_PROMPT_EXISTING_IMPORTS_CHARS = 600
_PROMPT_EXISTING_SYMBOLS = 25
_PROMPT_ORACLE_TEXT_CHARS = 450
_PROMPT_ORACLE_HINTS_MAX = 4
_PROMPT_ORACLE_HINT_CHARS = 180
_PROMPT_CONFTEST_PATHS = 4
_PROMPT_CONFTEST_FIXTURES = 8
_PROMPT_CORRECTION_MEMORY_MAX = 4
_RETRY_PREVIOUS_RESPONSE_CHARS = 600
_RETRY_PREVIOUS_CODE_CHARS = 1200
_RETRY_ERROR_ITEMS_MAX = 6
_RETRY_ERROR_CHARS = 180
_RETRY_TASK_SUMMARY_CHARS = 1200


def _normalize_v37_oracle_value(value: Any) -> str:
    """Return stable text used by the v37 oracle-preservation checks."""
    if value is None:
        return ""
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _v37_oracle_assertion_present(
    code: str,
    oracle_spec: Mapping[str, Any],
) -> bool:
    """Check for a deterministic assertion or explicit expected exception."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    rendered_assertions = "\n".join(
        ast.unparse(node) if hasattr(ast, "unparse") else ast.dump(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
    )
    expected = _normalize_v37_oracle_value(oracle_spec.get("expected"))
    if rendered_assertions and (
        not expected
        or expected in _normalize_v37_oracle_value(rendered_assertions)
    ):
        return True
    oracle_type = _normalize_v37_oracle_value(oracle_spec.get("type"))
    expected_exception = "exception" in oracle_type or "raises" in oracle_type
    if not expected_exception:
        return False
    return any(
        isinstance(node, (ast.Call, ast.Attribute))
        and (
            (isinstance(node, ast.Attribute) and node.attr in {"raises", "assertRaises"})
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"raises", "assertRaises"}
            )
        )
        for node in ast.walk(tree)
    )


def _v37_oracle_semantic_fingerprint(code: str) -> tuple[str, ...] | None:
    """Return deterministic assertion/expected-exception semantics for M5-A."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    oracle_nodes: list[str] = []
    for node in ast.walk(tree):
        include = isinstance(node, ast.Assert)
        if isinstance(node, (ast.With, ast.AsyncWith)):
            include = any(
                isinstance(item.context_expr, ast.Call)
                and _call_name(item.context_expr.func).endswith("raises")
                for item in node.items
            )
        if isinstance(node, ast.Call):
            name = _call_name(node.func).lower()
            include = include or name.endswith("raises") or name.split(".")[-1].startswith(
                "assert"
            )
        if include:
            oracle_nodes.append(ast.dump(node, include_attributes=False))
    return tuple(sorted(oracle_nodes))


def _v37_m5a_changed_oracle_semantics(
    before_code: str,
    after_code: str,
    *,
    before_oracle_spec: Mapping[str, Any] | None = None,
    after_oracle_spec: Mapping[str, Any] | None = None,
) -> bool:
    """Detect the closed-taxonomy M5-A semantic-change condition."""
    before_fingerprint = _v37_oracle_semantic_fingerprint(before_code)
    after_fingerprint = _v37_oracle_semantic_fingerprint(after_code)
    if before_fingerprint is not None and after_fingerprint is not None:
        if before_fingerprint != after_fingerprint:
            return True
    if before_oracle_spec is not None or after_oracle_spec is not None:
        return _normalize_v37_oracle_value(before_oracle_spec or {}) != (
            _normalize_v37_oracle_value(after_oracle_spec or {})
        )
    return False


def _take_prompt_prefix(text: str, token_budget: int) -> str:
    kept: List[str] = []
    for line in text.splitlines():
        candidate = "\n".join(kept + [line])
        if estimate_prompt_tokens(candidate) > token_budget:
            break
        kept.append(line)
    return "\n".join(kept).rstrip()


def _take_prompt_suffix(text: str, token_budget: int) -> str:
    kept: List[str] = []
    for line in reversed(text.splitlines()):
        candidate = "\n".join(reversed([line] + kept))
        if estimate_prompt_tokens(candidate) > token_budget:
            break
        kept.insert(0, line)
    return "\n".join(kept).lstrip()


def compact_m5_prompt(prompt: str, safe_user_tokens: int) -> str:
    """Preserve M5 issue evidence and its scenario/schema contract on overflow."""
    if estimate_prompt_tokens(prompt) <= safe_user_tokens:
        return prompt

    context_marker = "[Code Context]"
    imports_marker = "[Available Imports from Repository]"
    scenario_marker = "[Validated Scenario]"
    schema_marker = "[Required Output JSON Schema]"
    omission = "\n[M5 prompt middle omitted to fit the model context]\n"
    omission_tokens = estimate_prompt_tokens(omission)

    if all(marker in prompt for marker in (context_marker, scenario_marker, schema_marker)):
        prefix = prompt.split(context_marker, 1)[0].rstrip()
        imports_block = (
            imports_marker + prompt.split(imports_marker, 1)[1].split(scenario_marker, 1)[0]
            if imports_marker in prompt
            else ""
        )
        scenario_and_contract = prompt.split(scenario_marker, 1)[1]
        scenario_block = scenario_marker + scenario_and_contract.split(schema_marker, 1)[0]
        schema_block = schema_marker + prompt.split(schema_marker, 1)[1]
        essential = (
            prefix
            + omission
            + (imports_block.rstrip() + "\n" if imports_block else "")
            + scenario_block.rstrip()
            + "\n"
            + schema_block
        )
        if estimate_prompt_tokens(essential) <= safe_user_tokens:
            return essential

        schema_tokens = min(
            estimate_prompt_tokens(schema_block),
            max(80, safe_user_tokens // 5),
        )
        imports_tokens = max(80, safe_user_tokens // 6) if imports_block else 0
        scenario_tokens = max(100, safe_user_tokens // 4)
        prefix_tokens = max(
            1,
            safe_user_tokens - schema_tokens - imports_tokens - scenario_tokens - omission_tokens,
        )
        compacted = (
            _take_prompt_prefix(prefix, prefix_tokens)
            + omission
            + (_take_prompt_prefix(imports_block, imports_tokens) + "\n" if imports_block else "")
            + _take_prompt_prefix(scenario_block, scenario_tokens)
            + "\n"
            + _take_prompt_prefix(schema_block, schema_tokens)
        ).strip()
        if estimate_prompt_tokens(compacted) <= safe_user_tokens:
            return compacted

    head_budget = max(1, int(safe_user_tokens * 0.6) - omission_tokens)
    tail_budget = max(1, safe_user_tokens - head_budget - omission_tokens)
    return (
        _take_prompt_prefix(prompt, head_budget)
        + omission
        + _take_prompt_suffix(prompt, tail_budget)
    ).strip()

_STDLIB_IMPORT_ROOTS = set(getattr(sys, "stdlib_module_names", set())) | {
    "os", "sys", "re", "json", "math", "collections", "itertools",
    "functools", "pathlib", "typing", "abc", "copy", "io", "datetime",
    "logging", "unittest", "dataclasses", "contextlib", "textwrap",
    "warnings", "traceback", "inspect", "importlib", "operator",
    "tempfile", "uuid",
}
_EXTERNAL_IMPORT_ROOTS = {
    "pytest", "numpy", "np", "scipy", "matplotlib", "pandas",
    "requests", "yaml", "toml", "setuptools", "pkg_resources",
    "flask", "sqlalchemy", "celery", "redis", "cv2", "PIL",
}
_REPO_OWNED_IMPORT_ROOTS = {
    "django", "sphinx", "sympy", "sklearn", "astropy", "_pytest", "testing",
    "pylint", "requests",
}
_PYTEST_DEV_ALLOWED_IMPORT_ROOTS = _STDLIB_IMPORT_ROOTS | {"pytest", "_pytest", "testing"}


def _build_v31_generation_contract(
    *,
    scenario: Mapping[str, Any],
    clue: Mapping[str, Any],
    context: Mapping[str, Any],
    target_test_file: str,
    target_source_file: str,
    runner: str,
    existing_test_imports: str,
    target_test_example: str,
) -> TestGenerationContractV31:
    """Construct a deterministic M5 contract from pre-patch evidence only."""
    style = context.get("project_test_style") if isinstance(context.get("project_test_style"), Mapping) else {}
    framework = str(style.get("framework") or style.get("style") or "pytest")
    shape = "class_method" if "class" in target_test_example.lower() and runner != "pytest" else "top_level_function"
    target = scenario.get("target_location") if isinstance(scenario.get("target_location"), Mapping) else {}
    target_symbol = str(
        target.get("canonical_target_identity")
        if "canonical_target_identity" in target
        else scenario.get("canonical_target_identity")
        if "canonical_target_identity" in scenario
        else target.get("target_function") or target.get("symbol") or ""
    )
    candidate_invocation = str(
        scenario.get("candidate_invocation_expression")
        or scenario.get("issue_api_target")
        or target.get("candidate_invocation_expression")
        or target.get("issue_api_target")
        or target_symbol
    )
    # Canonical implementation identity is M6 runtime evidence, not a literal
    # M5 source-string requirement.  Keep only a distinct issue-facing/public
    # invocation in the static contract.
    if candidate_invocation == target_symbol:
        candidate_invocation = ""
    imports: list[ImportManifestEntryV31] = []
    available = context.get("available_imports") if isinstance(context.get("available_imports"), Mapping) else {}
    for module, symbols in list(available.items())[:64]:
        module_text = str(module or "").strip()
        if not module_text:
            # A malformed/empty context entry must not make construction of the
            # whole v31 contract fail.  Empty manifest entries carry no useful
            # provenance and are intentionally omitted.
            continue
        values = [str(item) for item in symbols] if isinstance(symbols, Sequence) and not isinstance(symbols, (str, bytes)) else []
        if values:
            imports.append(ImportManifestEntryV31(
                import_line=f"from {module_text} import {', '.join(values[:8])}",
                module=module_text, symbol=", ".join(values[:8]),
                provenance="pre_patch_source",
                verified_module=True, verified_symbol=True,
            ))
        else:
            imports.append(ImportManifestEntryV31(
                import_line=f"import {module_text}", module=module_text,
                provenance="pre_patch_source", verified_module=True,
            ))
    for line in _extract_top_level_import_lines(existing_test_imports):
        if not any(entry.import_line == line for entry in imports):
            match = re.match(r"(?:from|import)\s+([\w.]+)", line.strip())
            root = match.group(1) if match else ""
            if not root:
                continue
            imports.append(ImportManifestEntryV31(
                import_line=line, module=root, provenance="pre_patch_test",
                verified_module=True,
            ))
    observed = " ".join(str(item) for item in (scenario.get("observed_buggy_behavior") or clue.get("observed_behavior") or []) if item)
    expected = " ".join(str(item) for item in (scenario.get("expected_behavior") or clue.get("expected_behavior") or []) if item)
    oracle_text = str(scenario.get("oracle") or context.get("oracle_hint") or expected or observed or "issue-visible behavior")
    identifiers = []
    clue_ids = clue.get("identifiers")
    if isinstance(clue_ids, Mapping):
        identifiers = [str(value) for values in clue_ids.values() for value in (values if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) else [values])]
    oracle = OracleContractV31(
        oracle_type="issue_behavior",
        property=oracle_text[:450],
        evidence=[observed[:240], expected[:240]] if observed or expected else ["validated scenario issue evidence"],
        allowed_forms=["assertion on target behavior", "exception or return-value assertion"],
        forbidden_forms=["assert True", "unrelated constant assertion", "private implementation-only assertion"],
        issue_identifiers=identifiers[:16],
        target_relation=target_symbol,
    )
    metadata = context.get("metadata") if isinstance(context.get("metadata"), Mapping) else {}
    target_hypotheses = normalize_target_hypotheses(context.get("localization_hypotheses") or metadata.get("v31_target_hypotheses") or [])
    return TestGenerationContractV31(
        schema_version=V31_SCHEMA_VERSION,
        framework=framework, runner=runner, shape=shape,
        target_test_file=target_test_file,
        target_source_file=target_source_file,
        target_symbol=target_symbol,
        allowed_imports=imports[:64],
        nearby_patterns=[target_test_example[:600]] if target_test_example else [],
        fixture_candidates=[str(item) for item in (context.get("conftest_fixtures") or {}).keys()][:16],
        target_invocation_candidates=[candidate_invocation] if candidate_invocation else [],
        observed_behavior=observed[:450], expected_behavior=expected[:450], oracle=oracle,
        forbidden_patterns=["golden patch", "post-patch", "assert True"],
        skeleton_source=("def test_reproduction():\n    # setup and target invocation required\n    pass" if shape == "top_level_function" else "class TestReproduction:\n    def test_reproduction(self):\n        pass"),
        target_hypotheses=target_hypotheses,
    )


def _v31_contract_errors(
    parsed: Mapping[str, Any],
    contract: TestGenerationContractV31,
    *,
    repo: Optional[Path] = None,
    available_imports: Optional[Mapping[str, Sequence[str]]] = None,
    import_checker: Any = None,
    import_context: Optional[Mapping[str, Any]] = None,
) -> list[str]:
    """Apply bounded v31 checks after deterministic M5-A repairs.

    The manifest is provenance, not a closed-world allow-list.  Imports that
    were not present in the truncated manifest are validated against the
    pre-patch repository/environment checker.  Only an explicit ``invalid``
    result is rejected here; ``unknown`` is left to normal preflight/runtime
    validation so a missing manifest row cannot create a false negative.
    """
    code = str(parsed.get("append_block") or parsed.get("test_code") or "")
    errors: list[str] = []
    allowed = {entry.import_line for entry in contract.allowed_imports}
    for line in _extract_top_level_import_lines(code):
        if line in allowed:
            continue
        if import_checker is None or repo is None:
            # Preserve the legacy safety boundary for callers that do not have
            # repository context; the generation path always supplies it.
            errors.append(f"v31 import validation unavailable: {line}")
            continue
        result = import_checker(
            line,
            repo,
            dict(available_imports or {}),
            dict(import_context or {}),
        )
        if getattr(result, "is_invalid", False):
            errors.append(f"v31 import validation rejected: {line} ({result.reason})")

    invocation_candidates = list(contract.target_invocation_candidates or [])
    # Canonical identity is proved from pre-patch runtime coverage in M6/M7.
    # Only a distinct issue-facing invocation is a literal M5 requirement.
    required_invocation = str(
        invocation_candidates[0] if invocation_candidates else ""
    ).strip()
    if required_invocation and not required_invocation.startswith("_"):
        target = required_invocation
        terminal = target.rsplit(".", 1)[-1]
        # A dotted target may be exercised through a bound receiver whose
        # variable name differs from the localization label.  Require the
        # terminal attribute/call, but do not require the full textual path.
        target_present = bool(
            re.search(rf"(?:\.\s*){re.escape(terminal)}\b", code)
            or re.search(rf"\b{re.escape(terminal)}\s*\(", code)
            or target in code
        )
        if not target_present:
            errors.append(f"v31 target invocation contract violation: {target}")
    if repo is None:
        # Compatibility for direct contract callers that do not have the
        # normal validation context.  The generation path supplies ``repo``
        # and delegates oracle checks to _validate_generated_code/M5-A.
        if re.search(r"\bassert\s+True\b|assert\s+1\s*==\s*1", code):
            errors.append("v31 oracle contract violation: trivial assertion")
        if not re.search(r"\bassert\b|pytest\.raises|self\.assert", code):
            errors.append("v31 oracle contract violation: no executable assertion")
    return errors


def _coerce_v31_generation_contract(
    raw: Mapping[str, Any],
) -> TestGenerationContractV31:
    """Rehydrate the persisted contract used by initial and repair validation."""
    oracle_raw = raw.get("oracle") if isinstance(raw.get("oracle"), Mapping) else {}
    oracle = OracleContractV31(
        oracle_type=str(oracle_raw.get("oracle_type") or "issue_behavior"),
        property=str(oracle_raw.get("property") or "issue-visible behavior"),
        evidence=[str(item) for item in oracle_raw.get("evidence", []) or []],
        allowed_forms=[str(item) for item in oracle_raw.get("allowed_forms", []) or []],
        forbidden_forms=[str(item) for item in oracle_raw.get("forbidden_forms", []) or []],
        issue_identifiers=[str(item) for item in oracle_raw.get("issue_identifiers", []) or []],
        target_relation=str(oracle_raw.get("target_relation") or ""),
    )
    imports = [
        ImportManifestEntryV31(
            import_line=str(item.get("import_line") or ""),
            module=str(item.get("module") or ""),
            symbol=str(item.get("symbol") or ""),
            provenance=str(item.get("provenance") or "pre_patch_source"),
            verified_module=bool(item.get("verified_module", True)),
            verified_symbol=bool(item.get("verified_symbol", False)),
        )
        for item in raw.get("allowed_imports", []) or []
        if isinstance(item, Mapping)
        and str(item.get("import_line") or "").strip()
        and str(item.get("module") or "").strip()
    ]
    return TestGenerationContractV31(
        schema_version=str(raw.get("schema_version") or V31_SCHEMA_VERSION),
        framework=str(raw.get("framework") or "pytest"),
        runner=str(raw.get("runner") or "pytest"),
        shape=str(raw.get("shape") or "top_level_function"),
        target_test_file=str(raw.get("target_test_file") or "tests/test_reproduction.py"),
        target_source_file=str(raw.get("target_source_file") or "unknown.py"),
        target_symbol=str(raw.get("target_symbol") or ""),
        allowed_imports=imports,
        nearby_patterns=[str(item) for item in raw.get("nearby_patterns", []) or []],
        fixture_candidates=[str(item) for item in raw.get("fixture_candidates", []) or []],
        target_invocation_candidates=[
            str(item) for item in raw.get("target_invocation_candidates", []) or []
        ],
        observed_behavior=str(raw.get("observed_behavior") or ""),
        expected_behavior=str(raw.get("expected_behavior") or ""),
        oracle=oracle,
        forbidden_patterns=[str(item) for item in raw.get("forbidden_patterns", []) or []],
        skeleton_source=str(raw.get("skeleton_source") or ""),
        target_hypotheses=normalize_target_hypotheses(raw.get("target_hypotheses") or []),
    )


@dataclass
class GeneratedReproductionTest:
    instance_id: str
    scenario_id: str
    model_name: str
    repo_path: str
    target_test_file: str
    target_test_file_abspath: str
    target_source_file: str
    insert_mode: str
    insertion_hint: str
    imports: List[str]
    test_code: str
    original_test_file_content: str
    modified_test_file_content: str
    test_patch: str
    raw_response: str
    prompt: str
    canonical_test_nodeid: str = ""
    patch_sha256: str = ""
    generated_patch_path: str = ""
    generated_patch_sha256: str = ""
    candidate_status: str = CandidateStatus.POSTPROCESSED.value
    diagnostic_only: bool = False
    final_set_membership: Dict[str, bool] = None
    postprocessing_actions: List[Dict[str, Any]] = None
    repair_attempted: bool = False
    repair_actions: List[str] = None
    repair_failed_reason: str = ""
    repair_retry_count: int = 0
    retry_required_oracle_risks: List[str] = None
    semantic_risk_flags: List[str] = None
    selected_reproduction_example: Dict[str, Any] = None
    relational_oracle: Dict[str, Any] = None
    candor_oracle: Dict[str, Any] = None
    llm_error_refinement: Dict[str, Any] = None
    prompt_profile: Dict[str, Any] = None
    iteration: int = 0
    generation_attempt_count: int = 0
    token_usage_status: str = "no_llm_call"
    generated_scenario_id: str = ""
    scenario_generation_attempt: int = 1
    scenario_generation_provenance: str = ""
    selected_issue_api_target: str = ""
    selected_implementation_target: str = ""
    setup_helper_calls: List[str] = None
    target_verification_status: str = ""
    target_verification_provenance: Dict[str, Any] = None
    target_consistency_status: str = ""
    m4_candidate_classification: str = ""
    m4_selection_policy: str = ""
    m5_target_used: str = ""
    m3_model_call_count: int = 0
    m5_attempt_count: int = 0
    fallback_used: bool = False
    fallback_reason: str = ""
    validation_diagnostics: Dict[str, Any] = None
    # LLM 토큰 사용량 (API 응답 기준, 누적)
    token_usage: Dict[str, int] = None
    v31_generation_contract: Dict[str, Any] = None
    v31_oracle_trace: Dict[str, Any] = None
    v31_telemetry: Dict[str, Any] = None
    language: str = "python"
    test_methods: List[str] = None
    oracle_spec: Dict[str, Any] = None
    m5_invocation_provenance: Dict[str, Any] = None
    blocking_oracle_flags: List[str] = None
    diagnostic_oracle_flags: List[str] = None

    def __post_init__(self):
        if self.token_usage is None:
            self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not self.canonical_test_nodeid:
            self.canonical_test_nodeid = _canonical_test_nodeid_from_generated(
                self.target_test_file,
                self.test_code,
            )
        if self.repair_actions is None:
            self.repair_actions = []
        if self.retry_required_oracle_risks is None:
            self.retry_required_oracle_risks = []
        if self.semantic_risk_flags is None:
            self.semantic_risk_flags = []
        if self.blocking_oracle_flags is None:
            self.blocking_oracle_flags = []
        if self.diagnostic_oracle_flags is None:
            self.diagnostic_oracle_flags = []
        if self.selected_reproduction_example is None:
            self.selected_reproduction_example = {}
        if self.relational_oracle is None:
            self.relational_oracle = {}
        if self.candor_oracle is None:
            self.candor_oracle = {}
        if self.llm_error_refinement is None:
            self.llm_error_refinement = {}
        if self.prompt_profile is None:
            self.prompt_profile = {}
        if not self.token_usage_status:
            self.token_usage_status = "known" if self.token_usage and self.token_usage.get("total_tokens", 0) > 0 else "no_llm_call"
        self.prompt_profile.setdefault("generation_attempt_count", self.generation_attempt_count)
        self.prompt_profile.setdefault("token_usage_status", self.token_usage_status)
        if not self.patch_sha256 and self.test_patch is not None:
            self.patch_sha256 = sha256_text(self.test_patch)
        if not self.generated_patch_sha256 and self.test_patch is not None:
            self.generated_patch_sha256 = sha256_text(self.test_patch)
        if self.patch_sha256 and not self.generated_patch_sha256:
            self.generated_patch_sha256 = self.patch_sha256
        if not self.patch_sha256 and self.generated_patch_sha256:
            self.patch_sha256 = self.generated_patch_sha256
        if self.final_set_membership is None:
            self.final_set_membership = FinalSetMembership().to_dict()
        if self.postprocessing_actions is None:
            self.postprocessing_actions = []
        if not self.generated_scenario_id:
            self.generated_scenario_id = self.scenario_id
        if self.setup_helper_calls is None:
            self.setup_helper_calls = []
        if self.target_verification_provenance is None:
            self.target_verification_provenance = {}
        if self.validation_diagnostics is None:
            self.validation_diagnostics = {}
        if self.v31_generation_contract is None:
            self.v31_generation_contract = {}
        if self.v31_oracle_trace is None:
            self.v31_oracle_trace = {}
        if self.v31_telemetry is None:
            self.v31_telemetry = {}
        if self.test_methods is None:
            self.test_methods = []
        if self.oracle_spec is None:
            self.oracle_spec = {}
        if self.m5_invocation_provenance is None:
            self.m5_invocation_provenance = {}

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["test_id"] = self.test_id
        data["canonical_test_nodeid"] = self.canonical_test_nodeid
        data["test_nodeid"] = self.canonical_test_nodeid
        data["generated_patch_sha256"] = self.generated_patch_sha256 or self.patch_sha256
        data["candidate_status"] = str(self.candidate_status or CandidateStatus.POSTPROCESSED.value)
        data["diagnostic_only"] = bool(self.diagnostic_only)
        data["final_set_membership"] = dict(self.final_set_membership or FinalSetMembership().to_dict())
        data["postprocessing_actions"] = list(self.postprocessing_actions or [])
        data["relational_oracle"] = dict(self.relational_oracle or {})
        data["candor_oracle"] = dict(self.candor_oracle or {})
        data["llm_error_refinement"] = dict(self.llm_error_refinement or {})
        data["generated_scenario_id"] = self.generated_scenario_id or self.scenario_id
        return data

    @property
    def test_id(self) -> str:
        return f"{self.instance_id}:{self.scenario_id}:{self.target_test_file}"


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fixed_imports: Optional[List[str]] = None


@dataclass(frozen=True)
class ImportCheckResult:
    status: str
    corrected: str = ""
    reason: str = ""

    @property
    def is_valid(self) -> bool:
        return self.status == "valid"

    @property
    def is_correctable(self) -> bool:
        return self.status == "correctable" and bool(self.corrected)

    @property
    def is_unknown(self) -> bool:
        return self.status == "unknown"

    @property
    def is_invalid(self) -> bool:
        return self.status == "invalid"


def _valid_import(reason: str = "") -> ImportCheckResult:
    return ImportCheckResult("valid", reason=reason)


def _invalid_import(reason: str = "") -> ImportCheckResult:
    return ImportCheckResult("invalid", reason=reason)


def _unknown_import(reason: str = "") -> ImportCheckResult:
    return ImportCheckResult("unknown", reason=reason)


def _correctable_import(corrected: str, reason: str = "") -> ImportCheckResult:
    return ImportCheckResult("correctable", corrected=corrected, reason=reason)


@dataclass
class GenerationFailureError(RuntimeError):
    """Raised when generation fails after preserving all observed token usage."""

    message: str
    token_usage: Dict[str, int]
    attempt_count: int
    last_error: str
    failure_type_detail: str
    token_usage_status: str
    raw_candidate: str = ""
    raw_response: str = ""
    parsed_candidate: Dict[str, Any] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    validation_status: Dict[str, str] = field(default_factory=dict)
    prompt: str = ""
    scenario: Dict[str, Any] = field(default_factory=dict)
    attempt_history: List[Dict[str, Any]] = field(default_factory=list)

    def __str__(self) -> str:
        return self.message


def _coerce_import_check_result(value: Any) -> ImportCheckResult:
    """Accept the old bool/string contract while call sites migrate."""
    if isinstance(value, ImportCheckResult):
        return value
    if value is True:
        return _valid_import("legacy_true")
    if value is False:
        return _invalid_import("legacy_false")
    if isinstance(value, str) and value.strip():
        return _correctable_import(value.strip(), "legacy_corrected")
    return _unknown_import("legacy_unknown")


def _fix_django_imports(imports: List[str]) -> List[str]:
    """Convert unittest base imports while preserving package-relative imports."""
    result = []
    has_django_test = any("from django.test import" in imp for imp in imports)
    for imp in imports:
        if "from unittest import" in imp or imp.strip() in ("import unittest", "import unittest.TestCase"):
            if not has_django_test:
                result.append("from django.test import TestCase, SimpleTestCase")
                has_django_test = True
            continue
        result.append(imp)
    return result


def _fix_sphinx_test_code(test_code: str) -> str:
    """Preserve verified Sphinx fixture dependencies without semantic mutation."""
    return test_code


def _clip_prompt_text(
    text: Any,
    limit: int,
    section: str = "",
    prompt_profile: Optional[Dict[str, Any]] = None,
) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    if prompt_profile is not None and section:
        prompt_profile.setdefault("truncated_sections", []).append(section)
    return value[:limit].rstrip() + "\n... (truncated)"


def _mark_prompt_section(
    prompt_profile: Optional[Dict[str, Any]],
    name: str,
) -> None:
    if prompt_profile is not None:
        prompt_profile.setdefault("sections_included", []).append(name)


def _compact_list(values: Any, max_items: int, char_limit: int) -> List[str]:
    compacted: List[str] = []
    if not isinstance(values, list):
        return compacted
    for value in values:
        if len(compacted) >= max_items:
            break
        text = str(value).strip()
        if text:
            compacted.append(text[:char_limit])
    return compacted


def _dedup_text_items(items: List[str], limit: Optional[int] = None) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        norm = re.sub(r"\s+", " ", text.lower())
        if not text or norm in seen:
            continue
        seen.add(norm)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def _compact_validation_errors(error_message: str) -> str:
    raw_parts = re.split(r";|\n", str(error_message or ""))
    parts = _dedup_text_items(
        [p[:_RETRY_ERROR_CHARS] for p in raw_parts if p.strip()],
        limit=_RETRY_ERROR_ITEMS_MAX,
    )
    return "\n".join(f"- {p}" for p in parts) if parts else "- Unknown validation error"


def _compact_previous_attempt_code(
    previous_response: str,
    previous_parsed: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the most useful failed code excerpt for retry prompting."""
    code = ""
    if previous_parsed:
        code = str(
            previous_parsed.get("append_block")
            or previous_parsed.get("test_code")
            or ""
        ).strip()
    if not code:
        code = str(previous_response or "").strip()
    return _clip_prompt_text(code, _RETRY_PREVIOUS_CODE_CHARS)


def _build_retry_repair_directive_hint(
    error_message: str,
    scenario: Optional[Dict[str, Any]] = None,
) -> str:
    directive = (scenario or {}).get("repair_directive")
    if not isinstance(directive, dict):
        return ""
    if selected_example_requires_oracle_regeneration(scenario or {}):
        directive = sanitize_repair_directive(directive)

    parts = [
        "[REPAIR DIRECTIVE RETRY REQUIREMENT]",
        "Your next JSON must satisfy this directive while preserving the issue reproduction.",
    ]
    mode = directive.get("mode")
    reason = directive.get("blocking_reason")
    if mode:
        parts.append(f"- mode: {mode}")
    if reason:
        parts.append(f"- blocking_reason: {reason}")

    must_change = [str(x) for x in (directive.get("must_change") or []) if str(x).strip()]
    must_keep = [str(x) for x in (directive.get("must_keep") or []) if str(x).strip()]
    forbidden = [str(x) for x in (directive.get("forbidden_patterns") or []) if str(x).strip()]
    replacement_hints = [str(x) for x in (directive.get("replacement_hints") or []) if str(x).strip()]
    if must_change:
        parts.append("- must_change:")
        parts.extend(f"  * {item}" for item in must_change[:5])
    if must_keep:
        parts.append("- must_keep:")
        parts.extend(f"  * {item}" for item in must_keep[:4])
    if forbidden:
        parts.append("- forbidden_patterns that must not appear again:")
        parts.extend(f"  * {item}" for item in forbidden[:6])
    if replacement_hints:
        parts.append("- replacement_hints to use instead:")
        parts.extend(f"  * {item}" for item in replacement_hints[:6])

    evidence = directive.get("evidence") if isinstance(directive.get("evidence"), dict) else {}
    target_source = evidence.get("target_source") or ((scenario or {}).get("target_location") or {}).get("source_file")
    target_function = evidence.get("target_function") or ((scenario or {}).get("target_location") or {}).get("target_function")
    candidate_test_file = evidence.get("candidate_test_file") or ((scenario or {}).get("target_location") or {}).get("candidate_test_file")
    if target_source or target_function or candidate_test_file:
        parts.append("- retarget using:")
        if target_source:
            parts.append(f"  * source_file: {target_source}")
        if target_function:
            parts.append(f"  * target_function/API: {target_function}")
        if candidate_test_file:
            parts.append(f"  * target_test_file: {candidate_test_file}")

    if "repair directive forbidden pattern repeated" in error_message:
        parts.extend([
            "- Do not make a tiny edit around the forbidden line. Rewrite the stimulus and assertion so the forbidden API/pattern is absent.",
            "- The new assertion must inspect the target function result or a public state change caused by the target API.",
        ])

    return "\n".join(parts) + "\n"


def _build_m5_feedback_constraints(
    *,
    instance_repo: str,
    clue: Dict[str, Any],
    context: Dict[str, Any],
    scenario: Dict[str, Any],
    validation_errors: Optional[List[str]] = None,
    rejected_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convert accepted M7 feedback and validation evidence into hard M5 rules."""
    repo_lower = str(instance_repo or "").lower()
    combined = "\n".join(
        [
            json.dumps(clue or {}, ensure_ascii=False),
            json.dumps(scenario or {}, ensure_ascii=False),
            "\n".join(validation_errors or []),
        ]
    ).lower()
    is_requests_http = "requests" in repo_lower or "httpbin" in combined or "requests." in combined
    mandatory: List[str] = []
    forbidden: List[str] = []
    alternatives: List[str] = []

    negative_memory = scenario.get("negative_memory") if isinstance(scenario, dict) else []
    oracle_recovery_requested = False
    if isinstance(negative_memory, list):
        for entry in negative_memory[-_PROMPT_CORRECTION_MEMORY_MAX:]:
            if not isinstance(entry, Mapping):
                continue
            category = str(entry.get("category") or "")
            if category in {
                "MISSING_EXPLICIT_ORACLE",
                "ORACLE_EXPECTED_BEHAVIOR_NOT_PRESERVED",
                "REJECTED_ORACLE_PATTERN",
            }:
                oracle_recovery_requested = True
            rejected = str(entry.get("rejected_choice") or "").strip()
            reason = str(entry.get("reason") or "").strip()
            prohibition = str(entry.get("prohibition") or "").strip()
            if prohibition:
                mandatory.append(prohibition)
            elif reason:
                mandatory.append(f"Do not repeat the prior {category} violation: {reason}")
            if rejected:
                forbidden.append(rejected)
            alternatives.extend(
                str(item).strip()
                for item in (entry.get("repository_alternatives") or [])[:5]
                if str(item).strip()
            )
    if oracle_recovery_requested:
        expected_behavior = [
            str(item).strip()
            for item in (
                scenario.get("expected_outputs")
                or scenario.get("expected_behavior")
                or clue.get("expected_behavior")
                or []
            )
            if str(item).strip()
        ][:5]
        if expected_behavior:
            mandatory.append(
                "The replacement oracle must explicitly preserve this M1/M3 expected behavior: "
                + json.dumps(expected_behavior, ensure_ascii=False)
            )

    directive = scenario.get("repair_directive") if isinstance(scenario, dict) else {}
    if isinstance(directive, dict):
        mandatory.extend(str(item).strip() for item in (directive.get("must_change") or []) if str(item).strip())
        forbidden.extend(str(item).strip() for item in (directive.get("forbidden_patterns") or []) if str(item).strip())
        structured = directive.get("m7_llm_structured_hints")
        if isinstance(structured, dict) and structured:
            structured_summary = json.dumps(
                _m5_constraint_json_safe(structured),
                ensure_ascii=False,
                sort_keys=True,
            )[:1200]
            mandatory.append(
                "Apply accepted M7 structured feedback: "
                + structured_summary
            )

    verified_target = context.get("verified_target_evidence") if isinstance(context, dict) else {}
    if isinstance(verified_target, dict) and verified_target:
        candidate_invocation = str(
            verified_target.get("candidate_invocation_expression")
            or verified_target.get("issue_api_target")
            or ""
        )
        canonical_identity = str(
            verified_target.get("canonical_target_identity")
            or verified_target.get("target_callable")
            or ""
        )
        if candidate_invocation:
            mandatory.append(
                "Use this issue-grounded invocation as stimulus evidence, but the receiver name is not a canonical "
                "repository identity and may be replaced by a repository-valid receiver: "
                + candidate_invocation
            )
        if verified_target.get("target_test_file_exists"):
            mandatory.append(
                "Use this verified existing target test file unless validation rejects it: "
                + str(verified_target.get("target_test_file"))
            )
        if verified_target.get("source_file_exists"):
            mandatory.append(
                "Use this verified target source file as repository-local evidence: "
                + str(verified_target.get("source_file"))
            )
        if verified_target.get("target_callable_exists") and canonical_identity:
            mandatory.append(
                "Directly exercise this verified callable or an explicit public wrapper that reaches it: "
                + canonical_identity
                + (
                    f" with signature {verified_target.get('signature')}"
                    if verified_target.get("signature")
                    else ""
                )
            )

    if is_requests_http:
        mandatory.extend(
            [
                "No real network access: the generated test must not depend on DNS, internet, localhost services, or network availability.",
                "No external URLs in executable requests calls; do not call httpbin.org or any http(s) URL through requests.get/post/put/delete/request.",
                "When the issue reproduction uses an external URL, preserve the method and payload semantics with repository-local mocks, adapters, PreparedRequest, Request.prepare(), fake responses, or local helper objects.",
                "Use nearby existing requests tests as style evidence, especially PreparedRequest/request preparation or adapter/session fake-response patterns.",
                "Reproduce the issue-specific binary payload behavior and keep the non-ASCII binary payload stimulus visible.",
                "Use a narrow issue-specific oracle about request preparation/body/encoding or the documented exception behavior.",
                "For binary payload issues, assert the prepared request body equals the exact encoded non-ASCII payload; do not only assert isinstance(body, bytes).",
                "Do not fail because a network endpoint is unavailable.",
                "Do not use response.ok as the primary oracle unless the issue explicitly states response.ok is the fixed behavior.",
            ]
        )
        forbidden.extend(
            [
                "http://httpbin.org",
                "https://httpbin.org",
                "httpbin.org",
                "requests.put('http",
                'requests.put("http',
                "requests.get('http",
                'requests.get("http',
                "requests.post('http",
                'requests.post("http',
                "requests.request('",
                'requests.request("',
                "response.ok",
                "isinstance(prepared_request.body, bytes)",
                "isinstance(req.body, bytes)",
                "isinstance(request.body, bytes)",
            ]
        )
        alternatives.extend(
            [
                "PreparedRequest().prepare(method='PUT', url='http://www.example.com', data=...)",
                "requests.Request('PUT', 'http://www.example.com', data=...).prepare()",
                "assert prepared_request.body == u'ööö'.encode('utf-8')",
                "HTTPAdapter/Session fake send path using repository-local objects, without contacting the URL",
                "Existing local helper patterns from the target test file",
            ]
        )

    if validation_errors:
        mandatory.append("Fix these exact validation errors before returning a new candidate: " + " | ".join(validation_errors[:5]))
    if rejected_patterns:
        forbidden.extend(str(item).strip() for item in rejected_patterns if str(item).strip())

    nearby = context.get("nearby_test_examples") or context.get("test_example_snippet") or []
    nearby_examples = [nearby] if isinstance(nearby, str) else [str(item) for item in list(nearby)[:3]]
    return {
        "schema_version": "m5-feedback-constraints-v1",
        "mandatory_constraints": _dedupe_preserve_order(mandatory),
        "forbidden_patterns": _dedupe_preserve_order(forbidden),
        "repository_local_alternatives": _dedupe_preserve_order(alternatives),
        # Imports and full examples have dedicated bounded prompt sections.
        "nearby_test_examples": [example[:600] for example in nearby_examples[:1]],
    }


def _m5_constraint_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _m5_constraint_json_safe(item)
            for key, item in value.items()
            if not any(term in str(key).lower() for term in ("golden", "post_patch", "after_patch", "m8", "f_to_p"))
        }
    if isinstance(value, list):
        return [_m5_constraint_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_m5_constraint_json_safe(item) for item in value]
    return value


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _network_rejected_patterns(code: str) -> List[str]:
    patterns: List[str] = []
    if "httpbin.org" in code:
        patterns.append("httpbin.org")
    for match in re.finditer(r"requests\.(?:get|post|put|delete|request)\s*\(\s*['\"]https?://[^'\"]+", code):
        patterns.append(match.group(0))
    if "response.ok" in code:
        patterns.append("response.ok")
    if "unittest.mock" in code:
        patterns.append("unverified import: unittest.mock")
    return _dedupe_preserve_order(patterns)


def _build_syntax_retry_hint(
    error_message: str,
    attempt: int = 0,
    original_prompt: str = "",
) -> str:
    lower = str(error_message or "").lower()
    if "syntaxerror" not in lower:
        return ""

    hints = [
        "[PYTHON SYNTAX RETRY REQUIREMENT]",
        "- mode=REWRITE_MINIMAL_TOP_LEVEL_TEST unless the project is Django's test runner.",
        "- Return a complete, parseable append_block. Do not return partial snippets.",
        "- Every if/for/while/with/try/except/finally/class/function block must contain an indented body.",
        "- Prefer a straight-line test with one assertion over nested try/except control flow.",
    ]
    if "expected an indented block after 'except'" in lower:
        hints.extend([
            "- Remove empty except blocks. If the fixed behavior should not raise, do not catch the exception; call the API and assert the returned value/state.",
            "- If the fixed behavior should raise, use `with pytest.raises(ExpectedException):` around the triggering call.",
        ])
    if "unterminated string" in lower or "eol while scanning string" in lower:
        hints.append("- Use short string literals or triple-quoted strings with balanced quotes.")
    if "unexpected indent" in lower or "unindent" in lower:
        hints.append("- Match the style of the target file: top-level pytest functions start at column 0; class methods are indented exactly once.")
    if "invalid syntax" in lower:
        hints.append("- Remove explanatory text, shell prompts, ellipses, and partial expressions from append_block; return Python code only.")
    if "expected an indented block after function definition" in lower:
        hints.append("- Every generated test function must contain executable statements and at least one explicit assertion.")
    if attempt >= 2:
        prompt_lower = str(original_prompt or "").lower()
        hints.extend([
            "[TEMPLATE FALLBACK REQUIRED]",
            "- Do not freely restructure the test. Use a minimal one-test skeleton and fill only the target call plus one public assertion.",
            "- Preferred JSON shape: {\"target_test_file\": \"<existing target file>\", \"insert_mode\": \"append_block\", \"append_block\": \"<complete Python test>\"}.",
        ])
        if "pytest-dev" in prompt_lower or "pytester" in prompt_lower or "testdir" in prompt_lower:
            hints.extend([
                "- For pytest-dev, create generated files with textwrap.dedent and balanced triple quotes.",
                "- Skeleton: import textwrap; def test_issue_repro(pytester): pytester.makepyfile(textwrap.dedent(\"\"\"\\n        def test_inner():\\n            assert True\\n    \"\"\")); result = pytester.runpytest(); assert result.ret != 0",
            ])
        if "sympy" in prompt_lower or "raises_only_no_body_assertion" in lower:
            hints.extend([
                "- For success-path issues, do not use pytest.raises as the fallback oracle.",
                "- Skeleton: def test_issue_repro(): result = target_call(); assert result is not None",
            ])
    return "\n".join(hints) + "\n"


def _repo_contains_top_module(repo: Path, top_module: str) -> bool:
    if not top_module:
        return False
    return (
        (repo / top_module).is_dir()
        or (repo / f"{top_module}.py").exists()
        or (repo / "src" / top_module).is_dir()
        or (repo / "src" / f"{top_module}.py").exists()
    )


def _is_pytest_dev_repo(repo: Path) -> bool:
    return (repo / "src" / "_pytest").exists() or (repo / "_pytest").exists()


def _module_file_for_import(repo: Path, module_path: str) -> Optional[Path]:
    rel = module_path.replace(".", "/")
    candidates = [
        repo / f"{rel}.py",
        repo / rel / "__init__.py",
        repo / "src" / f"{rel}.py",
        repo / "src" / rel / "__init__.py",
    ]
    return next((p for p in candidates if p.exists()), None)


def _literal_str_sequence(node: ast.AST) -> set[str]:
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return set()
    return {
        elt.value for elt in node.elts
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    }


def _resolve_imported_module_file(path: Path, node: ast.ImportFrom) -> Optional[Path]:
    """Resolve a statically imported module relative to ``path``.

    The resolver is intentionally filesystem-only.  It never imports the
    benchmark package, which keeps preflight deterministic and avoids running
    repository module side effects.
    """
    module = str(node.module or "")
    if node.level:
        package_dir = path.parent
        for _ in range(max(0, node.level - 1)):
            package_dir = package_dir.parent
        module_parts = [part for part in module.split(".") if part]
        candidate_base = package_dir.joinpath(*module_parts)
        candidates = [candidate_base.with_suffix(".py"), candidate_base / "__init__.py"]
        return next((candidate for candidate in candidates if candidate.is_file()), None)

    if not module:
        return None
    for root in path.parents:
        candidate = _module_file_for_import(root, module)
        if candidate is not None:
            return candidate
    return None


def _collect_module_exported_names(
    path: Path,
    *,
    _seen: Optional[set[Path]] = None,
) -> set[str]:
    """Collect public names, including statically resolvable re-exports."""
    try:
        resolved_path = path.resolve()
    except OSError:
        resolved_path = path
    seen = set(_seen or set())
    if resolved_path in seen:
        return set()
    seen.add(resolved_path)
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
    except Exception:
        return set()

    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            imported_file = _resolve_imported_module_file(path, node)
            for alias in node.names:
                if alias.name == "*":
                    if imported_file is not None:
                        names.update(
                            name
                            for name in _collect_module_exported_names(
                                imported_file, _seen=seen
                            )
                            if not name.startswith("_")
                        )
                    continue
                names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    names.update(elt.id for elt in target.elts if isinstance(elt, ast.Name))
            value = node.value
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                names.update(_literal_str_sequence(value))
        elif isinstance(node, ast.AugAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "__all__"
                and isinstance(node.op, ast.Add)
            ):
                names.update(_literal_str_sequence(node.value))
    return names


def _module_has_dynamic_exports(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return True
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__getattr__":
            return True
        if isinstance(node, ast.ImportFrom) and node.names:
            if any(alias.name == "*" for alias in node.names):
                return True
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                if not _literal_str_sequence(node.value):
                    return True
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            if not _literal_str_sequence(node.value):
                return True
    return False


def _submodule_exists_for_from_import(repo: Path, package: str, name: str) -> bool:
    if not package or not name or "." in name:
        return False
    return _module_file_for_import(repo, f"{package}.{name}") is not None


def _canonical_test_nodeid_from_generated(target_test_file: str, code: str) -> str:
    test_file = str(target_test_file or "").strip().replace("\\", "/")
    if not test_file:
        return ""
    while test_file.startswith("./"):
        test_file = test_file[2:]
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return ""
    suffixes: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            suffixes.append(node.name)
        elif isinstance(node, ast.ClassDef) and _is_collectable_test_class_ast(node):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test"):
                    suffixes.append(f"{node.name}::{item.name}")
    if len(suffixes) != 1:
        return ""
    return f"{test_file}::{suffixes[0]}"


def _is_collectable_test_class_ast(node: ast.ClassDef) -> bool:
    """Return whether a generated class has pytest/unittest collection shape.

    Pytest collects ``Test*`` classes without a custom constructor, while
    unittest and Django collect subclasses of ``*TestCase`` regardless of the
    class-name prefix.  ``__test__ = False`` explicitly disables collection.
    """
    explicitly_disabled = any(
        isinstance(item, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "__test__"
            for target in (
                item.targets if isinstance(item, ast.Assign) else [item.target]
            )
        )
        and isinstance(item.value, ast.Constant)
        and item.value.value is False
        for item in node.body
    )
    is_test_case = any(
        (isinstance(base, ast.Name) and base.id.endswith("TestCase"))
        or (isinstance(base, ast.Attribute) and base.attr.endswith("TestCase"))
        for base in node.bases
    )
    custom_constructor = any(
        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name in {"__init__", "__new__"}
        for item in node.body
    )
    return bool(
        not explicitly_disabled
        and (node.name.startswith("Test") or is_test_case)
        and not custom_constructor
    )


def _empty_token_usage() -> Dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _accumulate_token_usage(accumulated: Dict[str, int], usage: Any) -> str:
    if not isinstance(usage, dict):
        return "unknown"
    if not any(key in usage for key in accumulated):
        return "unknown"
    for key in accumulated:
        value = usage.get(key)
        if isinstance(value, int):
            accumulated[key] += value
        elif value is None:
            return "unknown"
    return "known"


def _generation_failure_type_detail(message: str) -> str:
    lower = str(message or "").lower()
    if "syntaxerror" in lower or "invalid syntax" in lower:
        return "SYNTAX_ERROR"
    if "semantic risk" in lower or "semantic_risk" in lower:
        return "SEMANTIC_RISK"
    if "oracle" in lower or "relational oracle" in lower:
        return "ORACLE_REJECTED"
    if "validation" in lower or "valid test" in lower:
        return "VALIDATION_REJECTED"
    return "GENERATION_FAILED"


def _find_verified_import_alternative(
    repo: Path,
    import_names: List[str],
    available_imports: Dict[str, List[str]],
    original_module: str = "",
) -> Optional[str]:
    """Return an alternative import only when the file proves the export exists."""
    if not import_names or import_names == ["*"]:
        return None
    for alt_module, alt_symbols in sorted((available_imports or {}).items()):
        if alt_module == original_module:
            continue
        if not all(n in (alt_symbols or []) for n in import_names):
            continue
        alt_file = _module_file_for_import(repo, alt_module)
        if alt_file is None:
            continue
        exported = _collect_module_exported_names(alt_file)
        if exported and all(n in exported for n in import_names):
            return f"from {alt_module} import {', '.join(import_names)}"
    return None


def _extract_top_level_import_lines(code: str) -> List[str]:
    lines = code.splitlines()
    result: List[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return result
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", start)
            if start:
                result.append("\n".join(lines[start - 1:end]).strip())
    return result


def _extract_existing_import_lines(file_content: str) -> set[str]:
    imports: set[str] = set()
    try:
        tree = ast.parse(file_content or "")
    except SyntaxError:
        return imports
    lines = file_content.splitlines()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", start)
            if start:
                imports.add("\n".join(lines[start - 1:end]).strip())
    return imports


def _target_file_has_test_classes(file_content: str) -> bool:
    try:
        tree = ast.parse(file_content or "")
    except SyntaxError:
        return False
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        has_test_method = any(
            isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name.startswith("test")
            for child in node.body
        )
        base_names = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(base.attr)
        if has_test_method or any("TestCase" in name for name in base_names):
            return True
    return False


def _scenario_repair_memory(scenario: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(scenario, dict):
        return {}
    memory = scenario.get("repair_memory")
    return memory if isinstance(memory, dict) else {}


def _scenario_forbidden_test_files(scenario: Optional[Dict[str, Any]]) -> set[str]:
    memory = _scenario_repair_memory(scenario)
    forbidden = {
        str(path).strip()
        for path in memory.get("forbidden_test_files", []) or []
        if str(path).strip()
    }
    directive = (scenario or {}).get("repair_directive") if isinstance(scenario, dict) else {}
    if isinstance(directive, dict) and directive.get("mode") in {
        "CHANGE_TEST_FILE",
        "SWITCH_SCENARIO_OR_TEST_FILE",
    }:
        evidence = directive.get("evidence") if isinstance(directive.get("evidence"), dict) else {}
        old_file = str(evidence.get("candidate_test_file") or "").strip()
        if old_file:
            forbidden.add(old_file)
    return forbidden


def _scenario_required_target_file(scenario: Optional[Dict[str, Any]]) -> str:
    memory = _scenario_repair_memory(scenario)
    return str(memory.get("required_target_file") or "").strip()


def _target_file_constraint_errors(
    target_test_file: str,
    scenario: Optional[Dict[str, Any]],
    repo: Optional[Path] = None,
) -> List[str]:
    target = str(target_test_file or "").strip()
    errors: List[str] = []
    target_path = Path(target)
    if target and (target_path.is_absolute() or ".." in target_path.parts):
        return [f"CRITICAL: target_test_file must be repository-relative: {target}."]
    required = _scenario_required_target_file(scenario)
    forbidden = _scenario_forbidden_test_files(scenario)
    if required and target != required:
        errors.append(
            "CRITICAL: target_test_file violates repair memory: "
            f"must use {required}, got {target or '<empty>'}."
        )
    if target and target in forbidden:
        errors.append(
            "CRITICAL: target_test_file violates repair memory: "
            f"{target} is forbidden after CHANGE_TEST_FILE feedback."
        )
    if target and repo is not None and not _target_file_has_collection_evidence(
        repo / target
    ):
        errors.append(
            "CRITICAL: target_test_file lacks runner-grounded collection evidence: "
            f"{target}."
        )
    return errors


def _target_file_has_collection_evidence(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return False
    if _module_is_unconditionally_skipped(tree):
        return False
    name = path.name.lower()
    if (
        (name.startswith("test") and name.endswith(".py"))
        or name.endswith("_test.py")
        or (name.startswith("unittest_") and name.endswith(".py"))
    ):
        return True
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            return True
        if isinstance(node, ast.ClassDef) and _is_collectable_test_class(node):
            if any(
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name.startswith("test")
                for item in node.body
            ):
                return True
    return False


def _module_is_unconditionally_skipped(tree: ast.Module) -> bool:
    pytest_aliases = {"pytest"}
    importorskip_aliases: set[str] = set()
    skip_aliases: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pytest":
                    pytest_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "pytest":
            for alias in node.names:
                if alias.name == "importorskip":
                    importorskip_aliases.add(alias.asname or alias.name)
                elif alias.name == "skip":
                    skip_aliases.add(alias.asname or alias.name)
    def call_name(call: ast.Call) -> str:
        name = _ast_dotted_name(call.func)
        if name in importorskip_aliases:
            return "pytest.importorskip"
        if name in skip_aliases:
            return "pytest.skip"
        if "." in name and name.split(".", 1)[0] in pytest_aliases:
            return "pytest." + name.split(".", 1)[1]
        return name
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            name = call_name(node.value)
            if name in {"pytest.importorskip", "pytest.skip"}:
                return True
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if "pytestmark" in names:
                values = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
                if any(
                    call_name(value) in {"pytest.mark.skip", "pytest.mark.skipif"}
                    and (
                        call_name(value) == "pytest.mark.skip"
                        or not value.args
                        or not isinstance(value.args[0], ast.Constant)
                        or value.args[0].value is not False
                    )
                    for value in values if isinstance(value, ast.Call)
                ):
                    return True
            if isinstance(node.value, ast.Call) and call_name(node.value) == "pytest.importorskip":
                return True
    return False


def _ast_dotted_name(node: ast.AST) -> str:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _is_collectable_test_class(node: ast.ClassDef) -> bool:
    """Return whether pytest/unittest can collect test methods from ``node``."""
    for item in node.body:
        if isinstance(item, (ast.Assign, ast.AnnAssign)):
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            value = item.value
            if any(isinstance(target, ast.Name) and target.id == "__test__" for target in targets):
                if isinstance(value, ast.Constant) and value.value is False:
                    return False
    is_test_case = any(
        (isinstance(base, ast.Name) and base.id.endswith("TestCase"))
        or (isinstance(base, ast.Attribute) and base.attr.endswith("TestCase"))
        for base in node.bases
    )
    if any(
        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name in {"__init__", "__new__"}
        for item in node.body
    ):
        return False
    return node.name.startswith("Test") or is_test_case


def _disabled_test_function_names(tree: ast.Module) -> set[str]:
    disabled: set[str] = set()
    aliases: Dict[str, str] = {}
    # Resolve the simple top-level aliases pytest itself observes when a
    # function object is subsequently marked ``__test__ = False``.
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if isinstance(value, ast.Name):
            for target in targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = aliases.get(value.id, value.id)

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if not isinstance(value, ast.Constant) or value.value is not False:
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "__test__"
                    and isinstance(target.value, ast.Name)
                ):
                    disabled.add(aliases.get(target.value.id, target.value.id))
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "setattr"
            and len(node.value.args) >= 3
            and isinstance(node.value.args[0], ast.Name)
            and isinstance(node.value.args[1], ast.Constant)
            and node.value.args[1].value == "__test__"
            and isinstance(node.value.args[2], ast.Constant)
            and node.value.args[2].value is False
        ):
            disabled.add(aliases.get(node.value.args[0].id, node.value.args[0].id))
    return disabled


def _append_block_preflight_errors(
    append_block: str,
    original_content: str,
    repo: Path,
    context: Dict[str, Any],
    available_imports: Dict[str, List[str]],
    import_checker,
) -> List[str]:
    errors: List[str] = []
    runner = ((context or {}).get("project_test_style") or {}).get("runner", "pytest")
    existing_imports = _extract_existing_import_lines(original_content)
    allow_new_class = runner == "django-test" or _target_file_has_test_classes(original_content)

    try:
        tree = ast.parse(append_block)
    except SyntaxError:
        return errors

    for node in ast.iter_child_nodes(tree):
        class_like_test = (
            isinstance(node, ast.ClassDef)
            and (
                node.name.startswith("Test")
                or any(
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name.startswith("test")
                    for child in node.body
                )
            )
        )
        if class_like_test and not allow_new_class:
            errors.append(
                "CRITICAL: new test class definitions are not allowed for this target file. "
                "Rewrite as one top-level test_* function matching the existing pytest style."
            )
            break

    if runner == "django-test":
        valid_testcase_names: set[str] = set()
        valid_testcase_modules: set[str] = {"django.test"}
        try:
            combined_tree = ast.parse(original_content + "\n" + append_block)
        except SyntaxError:
            combined_tree = tree
        for imported in ast.walk(combined_tree):
            if isinstance(imported, ast.ImportFrom) and imported.module in {
                "django.test",
                "django.test.testcases",
            }:
                for alias in imported.names:
                    if alias.name in {"TestCase", "SimpleTestCase"}:
                        valid_testcase_names.add(alias.asname or alias.name)
            elif isinstance(imported, ast.Import):
                for alias in imported.names:
                    if alias.name == "django.test":
                        valid_testcase_modules.add(alias.asname or alias.name)
        has_testcase_class = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_path = _call_name(base)
                    base_leaf = base_path.split(".")[-1]
                    module_path = ".".join(base_path.split(".")[:-1])
                    if (
                        base_path in valid_testcase_names
                        or (
                            base_leaf in {"TestCase", "SimpleTestCase"}
                            and module_path in valid_testcase_modules
                        )
                    ):
                        has_testcase_class = True
                        break
        if not has_testcase_class:
            errors.append(
                "CRITICAL: django-test runner requires a class inheriting from "
                "django.test.TestCase or SimpleTestCase. Wrap the test method inside that class."
            )

        errors.extend(
            _django_repository_preflight_errors(
                append_block=append_block,
                original_content=original_content,
                repo=repo,
                target_test_file=str(context.get("_target_test_file") or ""),
            )
        )

    for import_line in _extract_top_level_import_lines(append_block):
        if import_line in existing_imports:
            continue
        check = _coerce_import_check_result(
            import_checker(import_line, repo, available_imports, context)
        )
        if check.is_valid or check.is_unknown:
            continue
        if check.is_correctable:
            errors.append(f"invalid import: {import_line}; use {check.corrected} instead.")
        elif check.reason:
            errors.append(
                f"invalid import: {import_line}; {check.reason}."
            )
        else:
            errors.append(
                f"invalid import: {import_line}; module or symbol not found in repository context."
            )

    return errors


def _repository_import_bindings(
    code: str,
    *,
    repo: Path,
    target_test_file: str,
) -> Dict[str, tuple[Path, str]]:
    """Resolve imported aliases to pre-patch repository modules/symbols."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    bindings: Dict[str, tuple[Path, str]] = {}
    target_path = Path(target_test_file) if str(target_test_file or "").strip() else None
    target_parts = (
        list(target_path.with_suffix("").parts[:-1])
        if target_path is not None and target_path.name
        else []
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_file = _module_file_for_import(repo, alias.name)
                if module_file is not None:
                    bindings[alias.asname or alias.name.split(".")[0]] = (
                        module_file,
                        "",
                    )
        elif isinstance(node, ast.ImportFrom):
            module = str(node.module or "")
            if node.level:
                keep = max(0, len(target_parts) - node.level + 1)
                module_parts = target_parts[:keep] + ([*module.split(".")] if module else [])
                module_file = _module_file_for_import(repo, ".".join(module_parts))
            else:
                module_file = _module_file_for_import(repo, module)
            for alias in node.names:
                if alias.name == "*":
                    continue
                symbol_file = module_file
                if node.level == 0:
                    submodule = _module_file_for_import(repo, f"{module}.{alias.name}")
                    if submodule is not None:
                        symbol_file = submodule
                        symbol = ""
                    else:
                        symbol = alias.name
                else:
                    symbol = alias.name
                if symbol_file is not None:
                    bindings[alias.asname or alias.name] = (symbol_file, symbol)
    return bindings


def _module_definition_index(path: Path) -> tuple[set[str], Dict[str, ast.ClassDef], bool]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, UnicodeError):
        return set(), {}, True
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    dynamic = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "__getattr__"
        for node in tree.body
    )
    return functions, classes, dynamic


def _django_model_field_contract(
    path: Path, class_name: str
) -> tuple[set[str], set[str], bool]:
    _, classes, _ = _module_definition_index(path)
    node = classes.get(class_name)
    if node is None:
        return set(), set(), False
    bases = {_call_name(base).split(".")[-1] for base in node.bases}
    closed_world = "Model" in bases
    fields: set[str] = {"id", "pk"}
    required: set[str] = set()
    for statement in node.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            value = statement.value
            if isinstance(value, ast.Call) and _call_name(value.func).split(".")[-1].endswith(("Field", "Key", "Relation")):
                field_names = {
                    target.id for target in targets if isinstance(target, ast.Name)
                }
                fields.update(field_names)
                field_type = _call_name(value.func).split(".")[-1]
                keyword_values = {keyword.arg: keyword.value for keyword in value.keywords if keyword.arg}
                has_default = "default" in keyword_values or "db_default" in keyword_values
                nullable = (
                    isinstance(keyword_values.get("null"), ast.Constant)
                    and keyword_values["null"].value is True
                )
                primary_key = (
                    isinstance(keyword_values.get("primary_key"), ast.Constant)
                    and keyword_values["primary_key"].value is True
                )
                auto_field = field_type in {"AutoField", "BigAutoField", "SmallAutoField"}
                automatic_date = any(
                    isinstance(keyword_values.get(option), ast.Constant)
                    and keyword_values[option].value is True
                    for option in ("auto_now", "auto_now_add")
                )
                implicit_empty_string = field_type in {
                    "CharField",
                    "TextField",
                    "SlugField",
                    "EmailField",
                    "URLField",
                    "FilePathField",
                }
                constructor_managed_relation = field_type == "ManyToManyField"
                if not (
                    has_default
                    or nullable
                    or primary_key
                    or auto_field
                    or automatic_date
                    or implicit_empty_string
                    or constructor_managed_relation
                ):
                    required.update(field_names)
    return fields, required, closed_world


def _django_model_fields(path: Path, class_name: str) -> tuple[set[str], bool]:
    fields, _, closed_world = _django_model_field_contract(path, class_name)
    return fields, closed_world


def _django_model_relation_attnames(path: Path, class_name: str) -> set[str]:
    """Return constructor attnames Django exposes for single-valued relations."""
    _, classes, _ = _module_definition_index(path)
    node = classes.get(class_name)
    if node is None:
        return set()
    attnames: set[str] = set()
    for statement in node.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if not isinstance(value, ast.Call):
            continue
        field_type = _call_name(value.func).split(".")[-1]
        if field_type not in {"ForeignKey", "OneToOneField"}:
            continue
        attnames.update(
            f"{target.id}_id" for target in targets if isinstance(target, ast.Name)
        )
    return attnames


def _django_repository_preflight_errors(
    *,
    append_block: str,
    original_content: str,
    repo: Path,
    target_test_file: str,
) -> List[str]:
    """Reject repository-disprovable Django API/setup guesses before M6."""
    combined = original_content + "\n" + append_block
    bindings = _repository_import_bindings(
        combined,
        repo=repo,
        target_test_file=target_test_file,
    )
    try:
        tree = ast.parse(append_block)
    except SyntaxError:
        return []
    errors: List[str] = []
    assigned_classes: Dict[str, tuple[Path, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                binding = bindings.get(value.func.id)
                if binding and binding[1]:
                    for target in targets:
                        if isinstance(target, ast.Name):
                            assigned_classes[target.id] = binding

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node.func)
        parts = call_name.split(".")
        # Imported module API: timezone.get_timezone(...)
        if len(parts) == 2 and parts[0] in bindings and not bindings[parts[0]][1]:
            module_path, _ = bindings[parts[0]]
            functions, classes, dynamic = _module_definition_index(module_path)
            public_exports = _collect_module_exported_names(module_path)
            if (
                parts[1] not in functions
                and parts[1] not in classes
                and parts[1] not in public_exports
                and not dynamic
            ):
                errors.append(
                    f"CRITICAL: repository API does not exist: {call_name} "
                    f"({module_path.relative_to(repo)})."
                )
        # Method on an instance of an imported concrete class.
        if len(parts) == 2 and parts[0] in assigned_classes:
            module_path, class_name = assigned_classes[parts[0]]
            _, classes, _ = _module_definition_index(module_path)
            class_node = classes.get(class_name)
            if class_node is not None:
                methods = {
                    item.name
                    for item in class_node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                # Fail closed only for a private method guess on a concrete
                # repository class; public methods may be inherited/dynamic.
                if parts[1].startswith("_") and parts[1] not in methods:
                    errors.append(
                        f"CRITICAL: repository method does not exist on {class_name}: {parts[1]}."
                    )
        # Model(...) / Model.objects.create(...) keyword grounding.
        root = parts[0] if parts else ""
        binding = bindings.get(root)
        is_constructor = len(parts) == 1
        is_manager_create = len(parts) >= 3 and parts[1:3] == ["objects", "create"]
        if binding and binding[1] and (is_constructor or is_manager_create):
            fields, required, closed_world = _django_model_field_contract(
                binding[0], binding[1]
            )
            if closed_world:
                has_expansion = any(keyword.arg is None for keyword in node.keywords)
                relation_attnames = _django_model_relation_attnames(
                    binding[0], binding[1]
                )
                invalid = sorted(
                    keyword.arg
                    for keyword in node.keywords
                    if keyword.arg
                    and keyword.arg not in fields
                    and keyword.arg not in relation_attnames
                )
                if invalid:
                    errors.append(
                        "CRITICAL: repository model constructor keyword does not exist: "
                        f"{binding[1]}({', '.join(invalid)}). Available fields: "
                        + ", ".join(sorted(fields)[:24])
                    )
                if not has_expansion and not node.args:
                    supplied = {keyword.arg for keyword in node.keywords if keyword.arg}
                    missing = sorted(
                        field
                        for field in required
                        if field not in supplied and f"{field}_id" not in supplied
                    )
                    if missing:
                        errors.append(
                            "CRITICAL: repository model constructor is missing required "
                            f"field(s): {binding[1]}({', '.join(missing)})."
                        )
    return _dedup_text_items(errors, limit=8)


def _django_repository_grounding(
    *, repo: Path, target_test_file: str, original_content: str
) -> Dict[str, Any]:
    """Summarize inspectable Django test/model/API facts for the M5 prompt."""
    bindings = _repository_import_bindings(
        original_content,
        repo=repo,
        target_test_file=target_test_file,
    )
    try:
        tree = ast.parse(original_content)
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])
    bases = sorted({
        _call_name(base).split(".")[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        for base in node.bases
        if _call_name(base).split(".")[-1].endswith("TestCase")
    })
    models: Dict[str, List[str]] = {}
    required_models: Dict[str, List[str]] = {}
    model_details: Dict[str, Dict[str, Any]] = {}
    apis: Dict[str, List[str]] = {}
    for alias, (path, symbol) in sorted(bindings.items()):
        functions, classes, _ = _module_definition_index(path)
        if symbol and symbol in classes:
            fields, required, closed_world = _django_model_field_contract(path, symbol)
            if closed_world:
                models[alias] = sorted(fields)
                required_models[alias] = sorted(required)
                field_types: Dict[str, str] = {}
                class_node = classes[symbol]
                for statement in class_node.body:
                    if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                        continue
                    value = statement.value
                    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                    if isinstance(value, ast.Call):
                        field_type = _call_name(value.func).split(".")[-1]
                        for target in targets:
                            if isinstance(target, ast.Name) and field_type:
                                field_types[target.id] = field_type
                nearby_examples = [
                    line.strip()
                    for line in original_content.splitlines()
                    if re.search(
                        rf"\b{re.escape(alias)}(?:\.objects\.create)?\s*\(", line
                    )
                ][:2]
                model_details[alias] = {
                    "model_path": str(path.relative_to(repo)),
                    "model_class": symbol,
                    "required_fields": sorted(required),
                    "field_types": field_types,
                    "nearby_creation_examples": nearby_examples,
                }
            apis[alias] = sorted(
                item.name
                for item in classes[symbol].body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            )[:32]
        elif not symbol:
            apis[alias] = sorted(functions)[:32]
    return {
        "source": "pre_patch_repository",
        "target_test_file": target_test_file,
        "test_case_bases": bases or ["TestCase", "SimpleTestCase"],
        "model_constructor_fields": models,
        "required_model_constructor_fields": required_models,
        "model_constructor_grounding": model_details,
        "available_repository_apis": apis,
    }


def _select_existing_target_test_file(
    repo_path: str,
    preferred: str,
    context: Dict[str, Any],
    scenario: Optional[Dict[str, Any]] = None,
) -> str:
    repo = Path(repo_path)
    required = _scenario_required_target_file(scenario)
    if required and (repo / required).exists():
        return required
    candidates: List[tuple[int, str]] = []
    skip_set = {
        str(candidate.get("path") or "")
        for candidate in context.get("candidate_test_files", []) or []
        if isinstance(candidate, dict) and candidate.get("has_module_skip")
    }
    directive = (scenario or {}).get("repair_directive") if isinstance(scenario, dict) else {}
    switch_requested = isinstance(directive, dict) and directive.get("mode") == "SWITCH_SCENARIO_OR_TEST_FILE"
    avoid: set[str] = set(_scenario_forbidden_test_files(scenario))
    if switch_requested:
        avoid.add(str(preferred or ""))
        target = (scenario or {}).get("target_location", {}) if isinstance((scenario or {}).get("target_location"), dict) else {}
        avoid.add(str(target.get("candidate_test_file") or ""))

    def add(path: Any, priority: int) -> None:
        text = str(path or "").strip()
        if text and text not in avoid:
            if text in skip_set:
                priority += 20
            if Path(text).name == "__init__.py":
                priority += 100
            candidates.append((priority, text))

    if not switch_requested:
        add(preferred, 0)
    target = (scenario or {}).get("target_location", {}) if isinstance((scenario or {}).get("target_location"), dict) else {}
    if not switch_requested:
        add(target.get("candidate_test_file", ""), 1)
        for path in (scenario or {}).get("relevant_test_files", []) or []:
            add(path, 2)
    for idx, candidate in enumerate(context.get("candidate_test_files", []) or []):
        if isinstance(candidate, dict):
            skip_penalty = 20 if candidate.get("has_module_skip") else 0
            add(candidate.get("path", ""), 3 + skip_penalty + idx)
        else:
            add(candidate, 3 + idx)

    seen: set[str] = set()
    for _, path in sorted(candidates, key=lambda item: item[0]):
        if path in seen:
            continue
        seen.add(path)
        if (repo / path).exists():
            return path
    source_file = ""
    target = (scenario or {}).get("target_location", {}) if isinstance((scenario or {}).get("target_location"), dict) else {}
    source_file = str(target.get("source_file") or "")
    if not source_file:
        for candidate in _context_source_candidates(context):
            source_file = str(candidate.get("path") or "")
            if source_file:
                break
    for new_path in _new_test_file_candidates_from_source(repo_path, source_file, context):
        if new_path not in avoid:
            return new_path
    preferred_path = Path(str(preferred or ""))
    if (
        preferred
        and preferred not in avoid
        and (repo / preferred_path.parent).is_dir()
        and preferred_path.suffix == ".py"
    ):
        return preferred
    return ""


def _new_test_file_candidates_from_source(
    repo_path: str,
    source_file: str,
    context: Dict[str, Any],
) -> List[str]:
    repo = Path(repo_path or "")
    source = Path(str(source_file or ""))
    if not repo or not source.name:
        return []
    stem = source.stem
    existing_dirs: List[Path] = []
    for candidate in context.get("candidate_test_files", []) or []:
        path = candidate.get("path") if isinstance(candidate, dict) else str(candidate)
        if not path:
            continue
        abs_path = repo / str(path)
        if abs_path.exists():
            parent = Path(str(path)).parent
            if parent not in existing_dirs:
                existing_dirs.append(parent)
    results: List[str] = []
    for parent in existing_dirs:
        if not (repo / parent).is_dir():
            continue
        candidate = parent / f"test_{stem}.py"
        text = str(candidate).replace("\\", "/")
        if text not in results:
            results.append(text)
    return results


def _context_source_candidates(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for key in ("candidate_source_files", "candidate_files"):
        for item in context.get(key, []) or []:
            if isinstance(item, dict):
                path = str(item.get("path") or item.get("file_path") or item.get("source_file") or "")
                if path:
                    candidate = dict(item)
                    candidate["path"] = path
                    candidates.append(candidate)
    return candidates


def _context_function_candidates(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for key in ("top5_functions", "initial_suspicious_functions", "function_ranking"):
        for item in context.get(key, []) or []:
            if isinstance(item, dict):
                source = str(item.get("source_file") or item.get("file_path") or "")
                name = str(item.get("qualified_name") or item.get("function_name") or "")
                if source and name:
                    candidates.append({**item, "source_file": source, "qualified_name": name})
    for source in _context_source_candidates(context):
        source_path = str(source.get("path") or "")
        for name in source.get("top_level_functions") or []:
            if source_path and name:
                candidates.append({"source_file": source_path, "qualified_name": str(name)})
    return candidates


def _callable_signature_from_ast(repo_path: str, source_file: str, callable_name: str) -> Dict[str, Any]:
    repo = Path(repo_path or "")
    source = str(source_file or "")
    callable_text = str(callable_name or "")
    if not repo or not source or not callable_text:
        return {"exists": False}
    source_path = repo / source
    if not source_path.exists():
        return {"exists": False, "source_exists": False}
    try:
        tree = ast.parse(read_text(source_path))
    except SyntaxError:
        return {"exists": False, "source_exists": True}

    named_nodes: List[tuple[str, ast.AST]] = []

    def collect(body: List[ast.stmt], parents: tuple[str, ...] = ()) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                qualified = (*parents, node.name)
                named_nodes.append((".".join(qualified), node))
                collect(node.body, qualified)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                named_nodes.append((".".join((*parents, node.name)), node))

    collect(tree.body)
    simple_name = callable_text.split(".")[-1]
    if "." in callable_text:
        matches = [
            (qualified, node)
            for qualified, node in named_nodes
            if qualified == callable_text or callable_text.endswith("." + qualified)
        ]
    else:
        matches = [
            (qualified, node)
            for qualified, node in named_nodes
            if qualified == callable_text
        ]
        if not matches:
            matches = [
                (qualified, node)
                for qualified, node in named_nodes
                if qualified.split(".")[-1] == simple_name
            ]
    for qualified, node in matches:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [arg.arg for arg in node.args.args]
            return {
                "exists": True,
                "source_exists": True,
                "name": callable_text,
                "qualified_name": qualified,
                "simple_name": simple_name,
                "signature": f"{simple_name}({', '.join(args)})",
                "lineno": getattr(node, "lineno", None),
            }
        if isinstance(node, ast.ClassDef):
            return {
                "exists": True,
                "source_exists": True,
                "name": callable_text,
                "qualified_name": qualified,
                "simple_name": simple_name,
                "signature": f"class {simple_name}",
                "lineno": getattr(node, "lineno", None),
            }
    return {"exists": False, "source_exists": True}


def _issue_target_terms(clue: Dict[str, Any], scenario: Dict[str, Any]) -> set[str]:
    return {term.lower() for term in _extract_issue_function_evidence(_issue_evidence_text(clue, scenario))}


def _select_verified_target_callable(
    repo_path: str,
    source_file: str,
    clue: Dict[str, Any],
    context: Dict[str, Any],
    scenario: Dict[str, Any],
) -> str:
    target = scenario.get("target_location", {}) if isinstance(scenario.get("target_location"), dict) else {}
    explicit = str(target.get("target_function") or scenario.get("target_function") or "")
    terms = _issue_target_terms(clue, scenario)

    def score_name(name: str) -> int:
        simple = name.split(".")[-1]
        if explicit and name == explicit:
            return 100
        if explicit and simple == explicit.split(".")[-1]:
            return 95
        if simple.lower() in terms:
            return 80
        if name.lower() in terms:
            return 75
        if simple.startswith("__"):
            return 5
        return 20

    candidates: List[tuple[int, str]] = []
    for item in _context_function_candidates(context):
        item_source = str(item.get("source_file") or item.get("file_path") or "")
        if item_source != source_file:
            continue
        name = str(item.get("qualified_name") or item.get("function_name") or "")
        if name:
            candidates.append((score_name(name), name))
    if explicit:
        candidates.append((score_name(explicit), explicit))
    seen: set[str] = set()
    for _, name in sorted(candidates, key=lambda item: item[0], reverse=True):
        if name in seen:
            continue
        seen.add(name)
        check = _callable_signature_from_ast(repo_path, source_file, name)
        if check.get("exists"):
            return name
    return ""


def _select_canonical_target_identity(
    repo_path: str,
    source_file: str,
    clue: Dict[str, Any],
    context: Dict[str, Any],
    scenario: Dict[str, Any],
    candidate_invocation_expression: str,
) -> tuple[str, str]:
    """Resolve a repository callable independently of a receiver expression.

    The issue may spell a call as ``self.method`` or ``local.method``. Those
    expressions are useful M5 stimulus evidence, but only an AST-backed M2
    callable and its defining source file can become the canonical identity.
    """
    target = scenario.get("target_location", {}) if isinstance(scenario.get("target_location"), dict) else {}
    explicit_canonical = str(
        target.get("canonical_target_identity")
        or scenario.get("canonical_target_identity")
        or ""
    )
    legacy_target = str(
        target.get("target_function")
        or scenario.get("target_function")
        or ""
    )
    explicit = explicit_canonical or legacy_target
    # An explicit canonical hint owns repository lookup.  The issue-facing
    # invocation may intentionally end in a different public wrapper name.
    simple = str(explicit or candidate_invocation_expression).split(".")[-1]
    issue_terms = _issue_target_terms(clue, scenario)
    issue_text = _issue_evidence_text(clue, scenario).lower()

    ranked: list[tuple[int, str, str]] = []
    for order, item in enumerate(_context_function_candidates(context)):
        item_source = str(item.get("source_file") or item.get("file_path") or "")
        name = str(item.get("qualified_name") or item.get("function_name") or "")
        candidate_simple = name.split(".")[-1]
        if not item_source or not name:
            continue
        if simple and candidate_simple != simple:
            continue
        if (
            not simple
            and candidate_simple.lower() not in issue_terms
            and name.lower() not in issue_terms
            and not re.search(rf"(?<![A-Za-z0-9_]){re.escape(candidate_simple.lower())}(?![A-Za-z0-9_])", issue_text)
        ):
            continue
        score = 1000 - order
        if item_source == source_file:
            score += 100
        if explicit and name == explicit:
            score += 80
        if "." in name:
            score += 10
        ranked.append((score, name, item_source))

    for _, name, item_source in sorted(ranked, reverse=True):
        if _callable_signature_from_ast(repo_path, item_source, name).get("exists"):
            return name, item_source

    # A plain repository symbol already bound to the selected source remains
    # usable. Receiver-bound/local expressions do not get this fallback.
    if explicit and "." not in explicit:
        if _callable_signature_from_ast(repo_path, source_file, explicit).get("exists"):
            return explicit, source_file
    return "", source_file


def _verified_target_evidence(
    repo_path: str,
    clue: Dict[str, Any],
    context: Dict[str, Any],
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    repo = Path(repo_path or "")
    scenario = scenario or {}
    target = scenario.get("target_location", {}) if isinstance(scenario.get("target_location"), dict) else {}
    candidate_invocation_expression = _selected_issue_api_target(clue, scenario)
    setup_helpers = _setup_helper_calls(clue, scenario, candidate_invocation_expression)
    source_file = str(target.get("source_file") or scenario.get("source_file") or "")
    if not source_file or not (repo / source_file).exists():
        for candidate in _context_source_candidates(context):
            path = str(candidate.get("path") or "")
            if path and (repo / path).exists():
                source_file = path
                break
    test_file = _select_existing_target_test_file(
        repo_path,
        str(target.get("candidate_test_file") or ""),
        context,
        scenario,
    )
    callable_name, canonical_source_file = _select_canonical_target_identity(
        repo_path,
        source_file,
        clue,
        context,
        scenario,
        candidate_invocation_expression,
    )
    if callable_name:
        source_file = canonical_source_file
    signature = _callable_signature_from_ast(repo_path, source_file, callable_name)
    implementation_target = callable_name if signature.get("exists") else ""
    consistency = (
        "CONSISTENT"
        if candidate_invocation_expression and callable_name and callable_name.split(".")[-1] == candidate_invocation_expression.split(".")[-1]
        else "CONSISTENT_WITH_UNRESOLVED_IMPLEMENTATION"
        if candidate_invocation_expression
        else "CONSISTENT_WITH_UNRESOLVED_IMPLEMENTATION"
    )
    verification_status = (
        "VERIFIED_DIRECT_TARGET"
        if signature.get("exists")
        else "TARGET_UNRESOLVED"
        if candidate_invocation_expression
        else "INVALID_TARGET"
    )
    nearby_usages: List[str] = []
    for candidate in context.get("candidate_test_files", []) or []:
        path = candidate.get("path") if isinstance(candidate, dict) else str(candidate)
        abs_path = repo / str(path or "")
        if abs_path.exists() and callable_name:
            text = read_text(abs_path)
            simple = callable_name.split(".")[-1]
            if simple in text:
                idx = text.find(simple)
                nearby_usages.append(text[max(0, idx - 180): idx + 420])
        if len(nearby_usages) >= 3:
            break
    return {
        "schema_version": "m5-verified-target-evidence-v1",
        "source_file": source_file,
        "source_file_exists": bool(source_file and (repo / source_file).exists()),
        "target_test_file": test_file,
        "target_test_file_exists": bool(test_file and (repo / test_file).exists()),
        "target_callable": callable_name,
        "target_callable_exists": bool(signature.get("exists")),
        "canonical_target_identity": callable_name,
        "candidate_invocation_expression": candidate_invocation_expression,
        "issue_api_target": candidate_invocation_expression,
        "implementation_target": implementation_target,
        "setup_helper_calls": setup_helpers,
        "target_verification_status": verification_status,
        "target_verification_provenance": {
            "source": "m5_verified_target_evidence",
            "pre_patch_only": True,
            "rule": "issue_api_target_is_never_replaced_by_setup_helper",
        },
        "target_consistency_status": consistency,
        "signature": signature.get("signature", ""),
        "lineno": signature.get("lineno"),
        "nearby_usages": nearby_usages,
    }


def _apply_verified_target_to_scenario(
    scenario: Dict[str, Any],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    if not evidence:
        return scenario
    updated = copy.deepcopy(scenario or {})
    target = dict(updated.get("target_location") or {})
    if evidence.get("source_file_exists"):
        target["source_file"] = evidence.get("source_file", "")
    if evidence.get("target_test_file_exists"):
        target["candidate_test_file"] = evidence.get("target_test_file", "")
        updated["relevant_test_files"] = [evidence.get("target_test_file")] + [
            p for p in updated.get("relevant_test_files", []) if p != evidence.get("target_test_file")
        ]
    candidate_invocation = str(
        evidence.get("candidate_invocation_expression")
        or evidence.get("issue_api_target")
        or ""
    )
    canonical_identity = str(
        evidence.get("canonical_target_identity")
        or evidence.get("target_callable")
        or ""
    )
    # Persist the canonical field even when AST resolution failed.  This keeps
    # downstream code from promoting a local receiver expression into a
    # repository callable identity through a legacy fallback.
    target["canonical_target_identity"] = canonical_identity
    updated["canonical_target_identity"] = canonical_identity
    if canonical_identity:
        target["target_function"] = canonical_identity
        updated["target_function"] = canonical_identity
    if candidate_invocation:
        target["issue_api_target"] = candidate_invocation
        target["candidate_invocation_expression"] = candidate_invocation
        updated["issue_api_target"] = candidate_invocation
        updated["candidate_invocation_expression"] = candidate_invocation
    target["implementation_target"] = str(evidence.get("implementation_target") or "")
    target["setup_helper_calls"] = list(evidence.get("setup_helper_calls") or [])
    target["target_verification_status"] = str(evidence.get("target_verification_status") or "")
    target["target_verification_provenance"] = dict(evidence.get("target_verification_provenance") or {})
    target["target_consistency_status"] = str(evidence.get("target_consistency_status") or "")
    updated["implementation_target"] = target["implementation_target"]
    updated["setup_helper_calls"] = target["setup_helper_calls"]
    updated["target_verification_status"] = target["target_verification_status"]
    updated["target_verification_provenance"] = target["target_verification_provenance"]
    updated["target_consistency_status"] = target["target_consistency_status"]
    updated["target_location"] = target
    updated["verified_target_evidence"] = evidence
    return updated


def _selected_issue_api_target(clue: Dict[str, Any], scenario: Dict[str, Any]) -> str:
    explicit = str(scenario.get("issue_api_target") or "").strip()
    target = scenario.get("target_location", {}) if isinstance(scenario.get("target_location"), dict) else {}
    target_function = str(target.get("target_function") or scenario.get("target_function") or "")
    if explicit:
        return explicit
    calls = _ordered_issue_calls_for_m5(clue, scenario)
    observed_text = " ".join(
        str(value)
        for key in ("observed_behavior", "actual_outputs", "error_keywords")
        for value in (
            clue.get(key, [])
            if isinstance(clue.get(key), list)
            else [clue.get(key, "")]
        )
    ).lower()
    observed_calls = [
        call for call in calls
        if call.split(".")[-1].lower() in observed_text
    ]
    if observed_calls:
        # The reported failing/actual path owns the reproduction stimulus.
        # Calls mentioned only as expected-working contrasts remain helpers.
        return observed_calls[0]
    stimulus_text = " ".join(str(x) for x in scenario.get("execution_stimulus", []) or []).lower()
    for call in calls:
        if call.split(".")[-1].lower() in stimulus_text:
            return call
    if target_function and any(call.split(".")[-1] == target_function.split(".")[-1] for call in calls):
        return target_function
    return calls[-1] if calls else target_function


def _setup_helper_calls(clue: Dict[str, Any], scenario: Dict[str, Any], issue_api_target: str) -> List[str]:
    helpers: List[str] = []
    for call in _ordered_issue_calls_for_m5(clue, scenario):
        if issue_api_target and call.split(".")[-1] == issue_api_target.split(".")[-1]:
            continue
        if call not in helpers:
            helpers.append(call)
    return helpers


def _ordered_issue_calls_for_m5(clue: Dict[str, Any], scenario: Dict[str, Any]) -> List[str]:
    text = "\n".join([
        _issue_evidence_text(clue, scenario),
        " ".join(str(x) for x in scenario.get("execution_stimulus", []) or []),
    ])
    calls: List[str] = []
    for call in re.findall(r"\b((?:[A-Za-z_]\w*\.)?([A-Za-z_]\w{2,}))\s*\(", text):
        full, bare = call
        if bare.lower() in _REPRO_TERM_STOPWORDS:
            continue
        if full not in calls:
            calls.append(full)
    return calls


def _issue_evidence_text(
    clue: Optional[Dict[str, Any]],
    scenario: Optional[Dict[str, Any]],
) -> str:
    chunks: List[str] = []
    clue = clue or {}
    scenario = scenario or {}
    for key in ("observed_behavior", "expected_behavior", "repro_conditions", "expected_outputs", "actual_outputs"):
        value = clue.get(key)
        if isinstance(value, list):
            chunks.extend(str(x) for x in value)
        elif value:
            chunks.append(str(value))
    chunks.append(str(clue.get("raw_issue_text", "")))
    chunks.extend(_scenario_reproduction_blocks(clue, scenario))
    return "\n".join(chunks)


def _extract_issue_function_evidence(text: str) -> set[str]:
    funcs: set[str] = set()
    for call in re.findall(r"\b([A-Za-z_]\w{2,})\s*\(", text):
        lower = call.lower()
        if lower not in _REPRO_TERM_STOPWORDS and not lower.startswith("test_"):
            funcs.add(call)
    for dotted in re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b", text):
        tail = dotted.rsplit(".", 1)[-1]
        if len(tail) >= 3 and tail.lower() not in _REPRO_TERM_STOPWORDS:
            funcs.add(tail)
    return funcs


def _infer_test_file_for_source(repo_path: str, source_file: str) -> str:
    if not repo_path or not source_file:
        return ""
    repo = Path(repo_path)
    source = Path(source_file)
    stem = source.stem
    parent = source.parent
    candidates = [
        parent / "tests" / f"test_{stem}.py",
        parent.parent / "tests" / f"test_{stem}.py",
        parent.parent / "tests" / f"test_{parent.name}_{stem}.py",
        parent / f"test_{stem}.py",
    ]
    for candidate in candidates:
        text = str(candidate)
        if text and (repo / text).exists():
            return text
    return ""


def _retarget_scenario_from_explicit_issue_evidence(
    scenario: Dict[str, Any],
    clue: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Retarget only when issue text explicitly names a candidate source/function."""
    text = _issue_evidence_text(clue, scenario)
    if not text:
        return scenario
    norm_text = text.replace("\\", "/").lower()
    explicit_funcs = {f.lower() for f in _extract_issue_function_evidence(text)}
    if not explicit_funcs and ".py" not in norm_text:
        return scenario

    current = scenario.get("target_location", {}) if isinstance(scenario.get("target_location"), dict) else {}
    current_source = str(current.get("source_file") or "").lower()
    current_func = str(current.get("target_function") or "").lower()
    best: tuple[int, Dict[str, Any], str] | None = None
    for candidate in context.get("candidate_source_files", []) or []:
        if not isinstance(candidate, dict):
            continue
        path = str(candidate.get("path") or "")
        path_l = path.lower()
        symbols = {
            str(x)
            for key in ("top_level_functions", "matched_identifiers")
            for x in (candidate.get(key) or [])
        }
        symbol_l = {s.lower().split(".")[-1] for s in symbols}
        func_hits = explicit_funcs & symbol_l
        path_hit = bool(path_l and path_l in norm_text)
        if not path_hit and not func_hits:
            continue
        score = (5 if path_hit else 0) + min(4, len(func_hits))
        if path_l == current_source and (not func_hits or current_func in func_hits):
            continue
        if best is None or score > best[0]:
            top_functions = [str(x) for x in (candidate.get("top_level_functions") or [])]
            if current_func and current_func in func_hits:
                target_func = current.get("target_function", "")
            elif current.get("issue_api_target") and str(current["issue_api_target"]).lower().split(".")[-1] in func_hits:
                target_func = str(current["issue_api_target"])
            else:
                target_func = sorted(func_hits)[0] if func_hits else (top_functions[0] if top_functions else "")
            best = (score, candidate, target_func)

    if best is None or best[0] < 5:
        return scenario
    _, candidate, target_func = best
    updated = copy.deepcopy(scenario)
    target = dict(updated.get("target_location") or {})
    source_path = candidate.get("path", target.get("source_file", ""))
    target["source_file"] = source_path
    if target_func:
        target["target_function"] = target_func
    inferred_test = _infer_test_file_for_source(str(context.get("repo_path") or ""), str(source_path or ""))
    if inferred_test:
        target["candidate_test_file"] = inferred_test
        updated["relevant_test_files"] = [inferred_test] + [
            p for p in updated.get("relevant_test_files", []) if p != inferred_test
        ]
    updated["target_location"] = target
    updated.setdefault("repair_notes", []).append(
        "retargeted_from_explicit_issue_source_evidence"
    )
    return updated


def _repair_target_test_file_selection(
    parsed: Dict[str, Any],
    repo_path: str,
    context: Dict[str, Any],
    scenario: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    current = str(parsed.get("target_test_file") or "")
    selected = _select_existing_target_test_file(repo_path, current, context, scenario)
    if selected and selected != current:
        parsed = dict(parsed)
        parsed["target_test_file"] = selected
    return parsed


def _extract_retry_task_summary(original_prompt: str) -> str:
    lines: List[str] = []
    keep_prefixes = (
        "Repository:",
        "Instance ID:",
        "Base Commit:",
        "Observed behavior:",
        "Expected behavior:",
        "Reproduction conditions:",
        "- functions:",
        "- classes:",
        "- error/exception keywords:",
        "Candidate source files:",
        "Candidate test files:",
        "oracle_type:",
        "oracle_source:",
        "rule:",
    )
    for line in str(original_prompt or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(keep_prefixes) or stripped in {
            "[Issue Clue]",
            "[Code Context]",
            "[Oracle Contract — follow this before writing assertions]",
        }:
            lines.append(line)
        if len("\n".join(lines)) >= _RETRY_TASK_SUMMARY_CHARS:
            break
    summary = "\n".join(lines).strip()
    if not summary:
        summary = str(original_prompt or "")[:_RETRY_TASK_SUMMARY_CHARS].strip()
    return summary[:_RETRY_TASK_SUMMARY_CHARS]


def _prioritized_available_imports(
    available_imports: Dict[str, Any],
    clue: Dict[str, Any],
    context: Dict[str, Any],
    scenario: Dict[str, Any],
) -> List[tuple[str, List[str]]]:
    identifiers = (scenario.get("identifiers") or clue.get("identifiers") or {})
    target_location = scenario.get("target_location", {}) or {}
    source_paths = [
        str(target_location.get("source_file") or ""),
        *[str(x.get("path", "")) for x in context.get("candidate_source_files", [])[:3]],
        *[str(x.get("path", "")) for x in context.get("candidate_test_files", [])[:2]],
    ]
    target_terms = {
        str(target_location.get("target_function") or "").lower(),
        *[str(x).lower() for x in identifiers.get("functions", [])[:8]],
        *[str(x).lower() for x in identifiers.get("classes", [])[:8]],
    }
    target_terms = {x for x in target_terms if x}
    path_terms = set()
    for path in source_paths:
        for part in re.split(r"[/.\\_-]+", path.lower()):
            if len(part) >= 3:
                path_terms.add(part)

    ranked: List[tuple[int, str, List[str]]] = []
    for module, symbols in sorted((available_imports or {}).items()):
        if not isinstance(symbols, list):
            symbols = []
        module_l = str(module).lower()
        score = 0
        if any(term and term in module_l for term in path_terms):
            score += 4
        if any(term and term in module_l for term in target_terms):
            score += 3
        symbol_l = {str(s).lower() for s in symbols}
        score += min(4, len(symbol_l & target_terms))
        ranked.append((score, str(module), [str(s) for s in symbols]))

    ranked.sort(key=lambda x: (-x[0], x[1]))
    selected = [item for item in ranked if item[0] > 0][:_PROMPT_IMPORT_MODULES]
    if len(selected) < _PROMPT_IMPORT_MODULES:
        selected.extend(
            item for item in ranked
            if item not in selected
        )
    return [(module, symbols) for _, module, symbols in selected[:_PROMPT_IMPORT_MODULES]]


def _detect_blocking_oracle_risks(code: str, clue: Optional[Dict[str, Any]] = None) -> List[str]:
    """Reject high-risk oracles before spending an alignment iteration."""
    lower = code.lower()
    errors: List[str] = []

    def _has_issue_expected_signal() -> bool:
        return _has_issue_expected_signal_in_oracle(code, clue)
    has_oracle = bool(re.search(
        r"^\s*(assert\b|self\.assert|with\s+.*raises|pytest\.raises|"
        r".*assert_(?:allclose|array|equal|raises))",
        code,
        re.MULTILINE,
    ))
    if not has_oracle:
        errors.append(
            "CRITICAL: no explicit oracle remains. Add an assertion for the post-fix behavior."
        )
    if re.search(
        r"^(?:\s*)(?:self\.)?assertTrue\s*\(\s*(?:True|1)\s*(?:,\s*[^)]*)?\)\s*(?:#.*)?$"
        r"|^\s*assert\s+(?:True|1)\s*(?:#.*)?$",
        code,
        re.MULTILINE | re.IGNORECASE,
    ):
        errors.append(
            "CRITICAL: trivial oracle detected. Assert the post-fix return value or state change, not True."
        )
    if re.search(r"requests\.(get|post|put|delete|request)\s*\(\s*['\"]https?://", code):
        errors.append(
            "CRITICAL: real network calls are not allowed. Use PreparedRequest, mocks, or local helpers."
        )
    if re.search(r"\bresponse\.ok\b|\.\bok\b", code) and "response.ok" not in json.dumps(clue or {}).lower():
        errors.append(
            "CRITICAL: response.ok is not an issue-specific oracle. Assert request preparation/body/encoding behavior or an issue-stated public result."
        )
    if (
        re.search(r"isinstance\s*\(\s*(?:prepared_request|req|request)\.body\s*,\s*bytes\s*\)", code)
        and re.search(r"ööö|binary payload|to_native_string", json.dumps(clue or {}, ensure_ascii=False), re.IGNORECASE)
    ):
        errors.append(
            "CRITICAL: retry required: weak_prepared_request_body_type_oracle. Assert the exact encoded binary payload body, not only isinstance(body, bytes)."
        )
    if re.search(r"class\s+\w+\s*\([^)]*models\.Model[^)]*\)", code):
        errors.append(
            "CRITICAL: do not define Django models inside generated tests. Reuse existing test models/imports."
        )
    if re.search(r"float\s*\(\s*['\"]nan['\"]\s*\)|\bnp\.nan\b", lower) and re.search(r"!=|==|assertnot", lower):
        errors.append(
            "CRITICAL: do not compare NaN directly. Use np.isnan(...) or warning behavior."
        )
    if re.search(r"^\s*assert\s+.+==\s*np\.array\s*\(", code, re.MULTILINE):
        errors.append(
            "CRITICAL: do not compare numpy arrays with plain assert ==. Use np.testing.assert_array_equal or assert_allclose."
        )
    if re.search(r"get_[xy]lim\(\)\s*\[\s*[01]\s*\]\s*==", code):
        errors.append(
            "CRITICAL: raw Matplotlib limit equality is brittle. Use ax.xaxis_inverted()/ax.yaxis_inverted() or semantic tick/bin assertions."
        )
    if re.search(
        r"assert\s+str\s*\(\s*[\w.]+\s*\)\s*!=\s*['\"]|"
        r"\w+(?:\.value)?\.args\[\d+\]\s*!=\s*['\"]|"
        r"assert\s+['\"].+['\"]\s+not\s+in\s+str\s*\(|"
        r"self\.assert(?:NotIn|NotRegex)\s*\([^,\n]+,\s*str\s*\(|"
        r"self\.assertNotEqual\s*\(\s*str\s*\(",
        code,
        re.IGNORECASE,
    ):
        errors.append(
            "CRITICAL: do not assert exception message absence/change. Assert the success path or exception type."
        )
    if re.search(
        r"(?:expected|baseline|correct|desired)_(?:matrix|array|result|values?)\s*=.*\n"
        r"(?s:.*?)(?:assert_array_equal|assert_allclose|assert_equal)\s*\([^,\n]+,\s*"
        r"(?:expected|baseline|correct|desired)_(?:matrix|array|result|values?)",
        code,
        re.IGNORECASE,
    ) and not _has_issue_expected_signal():
        errors.append(
            "CRITICAL: guessed expected arrays are brittle. Use issue-stated expected output or a semantic invariant."
        )
    if re.search(
        r"assert(?:In|NotIn)\s*\(\s*['\"][^'\"]{80,}['\"]|"
        r"assert(?:In|NotIn)\s*\(\s*['\"][^'\"]*(?:\\PYG|\\sphinx|<[^>]+>)[^'\"]*['\"]",
        code,
        re.IGNORECASE,
    ):
        errors.append(
            "CRITICAL: raw rendered HTML/LaTeX/Sphinx string oracle is brittle. Use a small semantic marker/invariant."
        )
    retry_risks = _detect_retry_required_oracle_risks(code, clue=clue)
    for risk in retry_risks:
        errors.append(f"CRITICAL: retry required: {risk}")
    return errors


def _issue_says_success_path(clue: Optional[Dict[str, Any]]) -> bool:
    clue = clue or {}
    text = " ".join(
        str(x)
        for x in (
            clue.get("observed_behavior", [])
            + clue.get("expected_behavior", [])
            + clue.get("repro_conditions", [])
            + [clue.get("raw_issue_text", "")]
        )
    ).lower()
    return bool(re.search(
        r"should\s+not\s+(?:raise|error|fail|crash|warn)|"
        r"must\s+not\s+(?:raise|error|fail|crash|warn)|"
        r"does\s+not\s+(?:raise|error|fail|crash|warn)|"
        r"doesn't\s+(?:raise|error|fail|crash|warn)|"
        r"without\s+(?:raising|error|failing|crashing|warning)|"
        r"no\s+(?:exception|error|warning)|"
        r"no\s+longer\s+(?:raises|errors|fails|crashes|warns)",
        text,
    ))


def _issue_says_exception_expected(clue: Optional[Dict[str, Any]]) -> bool:
    clue = clue or {}
    text = " ".join(
        str(x)
        for x in (
            clue.get("observed_behavior", [])
            + clue.get("expected_behavior", [])
            + clue.get("repro_conditions", [])
            + [clue.get("raw_issue_text", "")]
        )
    ).lower()
    return bool(re.search(
        r"should\s+raise|must\s+raise|expected\s+(?:error|exception)|"
        r"should\s+(?:error|fail)\b|raises?\s+(?:a\s+)?(?:typeerror|valueerror|attributeerror|runtimeerror)",
        text,
    ))


def _issue_says_exception_validation_or_message(clue: Optional[Dict[str, Any]]) -> bool:
    text = _issue_text_blob(clue).lower()
    return bool(
        re.search(r"\b(?:exception|error|valueerror|typeerror|attributeerror|validation|message)\b", text)
        and re.search(r"\b(?:confusing|misleading|wrong|incorrect|expected|actual|message|validation|required)\b", text)
    )


def _has_issue_expected_signal(code: str, clue: Optional[Dict[str, Any]]) -> bool:
    return _has_issue_expected_signal_in_oracle(code, clue)


def _has_issue_expected_signal_in_oracle(code: str, clue: Optional[Dict[str, Any]]) -> bool:
    expected_outputs = [str(out) for out in (clue or {}).get("expected_outputs", [])[:3] if str(out).strip()]
    candidates = _extract_complete_oracle_output_candidates(code)
    return any(
        strict_normalized_output_equals(candidate, expected)
        for expected in expected_outputs
        for candidate in candidates
    )


def _extract_complete_oracle_output_candidates(code: str) -> List[str]:
    candidates: List[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    assignments: Dict[str, List[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            values = _oracle_expr_candidates(node.value, code)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = values
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assignments[node.target.id] = _oracle_expr_candidates(node.value, code)

    def add_expr(expr: ast.AST) -> None:
        if isinstance(expr, ast.Name) and expr.id in assignments:
            candidates.extend(assignments[expr.id])
        else:
            candidates.extend(_oracle_expr_candidates(expr, code))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            test = node.test
            if isinstance(test, ast.Compare):
                for comparator in test.comparators:
                    add_expr(comparator)
            else:
                add_expr(test)
        elif isinstance(node, ast.Call):
            func_name = _call_name(node.func)
            if re.search(r"(?:assert|assertEqual|assertAlmostEqual|assert_allclose|assert_array_equal|assert_equal)$", func_name):
                for arg in node.args[1:]:
                    add_expr(arg)

    return _dedup_text_items(candidates, limit=32)


def _oracle_expr_candidates(expr: Optional[ast.AST], source: str) -> List[str]:
    if expr is None:
        return []
    candidates: List[str] = []
    segment = ast.get_source_segment(source, expr)
    if segment:
        candidates.append(segment.strip())
    if isinstance(expr, ast.Constant):
        candidates.append(str(expr.value))
    unparsed = ast.unparse(expr).strip()
    if unparsed:
        candidates.append(unparsed)
    if isinstance(expr, ast.Call):
        func_name = _call_name(expr.func)
        if "." in func_name:
            arg_texts = []
            for arg in expr.args:
                arg_candidates = _oracle_expr_candidates(arg, source)
                if arg_candidates:
                    arg_texts.append(arg_candidates[0])
            bare_call = f"{func_name.rsplit('.', 1)[-1]}({', '.join(arg_texts)})"
            candidates.append(bare_call)
    return _dedup_text_items(candidates, limit=8)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _issue_text_blob(clue: Optional[Dict[str, Any]]) -> str:
    clue = clue or {}
    return " ".join(
        str(x)
        for x in (
            clue.get("observed_behavior", [])
            + clue.get("expected_behavior", [])
            + clue.get("repro_conditions", [])
            + clue.get("expected_outputs", [])
            + [clue.get("raw_issue_text", "")]
        )
    )


def _extract_issue_literals(clue: Optional[Dict[str, Any]]) -> List[str]:
    text = _issue_text_blob(clue)
    literals: List[str] = []
    for value in re.findall(r"['`]([^'`]{3,80})['`]", text):
        cleaned = value.strip()
        if cleaned and not cleaned.startswith(("#", "http://", "https://")):
            literals.append(cleaned)
    for value in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*-[A-Za-z0-9_-]+\b", text):
        literals.append(value)
    return _dedup_text_items(literals, limit=8)


def _extract_concise_expected_values(clue: Optional[Dict[str, Any]]) -> List[str]:
    text = _issue_text_blob(clue)
    values: List[str] = []
    patterns = [
        r"expected\s+(?:to\s+be|is|:)\s+['`\"]?([^'`\"\n.;,]{1,80})",
        r"['`\"]([^'`\"]{1,80})['`\"]\s+is\s+expected",
        r"expected\s+['`\"]([^'`\"]{1,80})['`\"]",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            cleaned = str(match).strip()
            if cleaned and not cleaned.lower().startswith(("behavior", "results")):
                values.append(cleaned)
    return _dedup_text_items(values, limit=5)


def _concise_expected_values_for_prompt(
    prompt_scenario: Dict[str, Any],
    clue: Optional[Dict[str, Any]],
) -> List[str]:
    selected = prompt_scenario.get("selected_reproduction_example")
    if isinstance(selected, dict):
        if selected_example_requires_oracle_regeneration(prompt_scenario):
            return []
        provenance = str(selected.get("expected_output_provenance") or "").strip()
        if provenance not in {"direct_issue_expected_output", "selected_example_expected_output"}:
            return []
        return _dedup_text_items(prompt_scenario.get("expected_outputs", []) or [], limit=5)
    if selected_example_requires_oracle_regeneration(prompt_scenario):
        return []
    return _extract_concise_expected_values(clue)


def _sanitize_raw_issue_text_for_unpaired_oracle(
    text: str,
    clue: Optional[Dict[str, Any]],
) -> str:
    expected_values = [
        str(value).strip()
        for value in (clue or {}).get("expected_outputs", []) or []
        if str(value).strip()
    ]
    sanitized_lines: List[str] = []
    for line in str(text or "").splitlines():
        lowered = line.lower()
        normalized = re.sub(r"\s+", "", lowered)
        if any(re.sub(r"\s+", "", expected.lower()) in normalized for expected in expected_values):
            continue
        if "fixed expected output" in lowered or "expected:" in lowered:
            continue
        sanitized_lines.append(line)
    return "\n".join(sanitized_lines).strip()


def _has_positive_issue_oracle_signal(clue: Optional[Dict[str, Any]]) -> bool:
    text = _issue_text_blob(clue).lower()
    return bool(
        (clue or {}).get("expected_outputs")
        or _extract_concise_expected_values(clue)
        or _issue_says_warning_expected(clue)
        or re.search(r"\b(?:expected|correct|finite|return(?:s)?|should\s+be|be able to)\b", text)
    )


def _issue_expected_warning_type(clue: Optional[Dict[str, Any]]) -> str:
    text = _issue_text_blob(clue)
    m = re.search(r"\b([A-Z][A-Za-z0-9_]*Warning)\b", text)
    if m:
        return m.group(1)
    if re.search(r"\bwarn(?:ing|s)?\b", text, re.IGNORECASE):
        return "Warning"
    return ""


def _issue_says_warning_expected(clue: Optional[Dict[str, Any]]) -> bool:
    text = _issue_text_blob(clue).lower()
    if not re.search(r"\bwarn(?:ing|s)?\b", text):
        return False
    expected_text = " ".join(str(x) for x in (clue or {}).get("expected_behavior", [])).lower()
    if re.search(r"\bwarn(?:ing|s)?\b|[a-z0-9_]*warning\b", expected_text):
        return not bool(re.search(
            r"should\s+not\s+warn|must\s+not\s+warn|without\s+warning|no\s+warning|"
            r"no\s+longer\s+warns?|does\s+not\s+warn|doesn't\s+warn",
            expected_text,
        ))
    return not bool(re.search(
        r"should\s+not\s+warn|must\s+not\s+warn|without\s+warning|no\s+warning|"
        r"no\s+longer\s+warns?|does\s+not\s+warn|doesn't\s+warn",
        text,
    ))


def _scenario_reproduction_blocks(
    clue: Optional[Dict[str, Any]],
    scenario: Optional[Dict[str, Any]],
) -> List[str]:
    scenario_blocks: List[str] = []
    clue_blocks: List[str] = []

    def looks_concrete_repro(text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if re.match(
            r"^(?:Call|Compare|Run)\s+.+(?:reproduction conditions|expected value|expected result|with --pdb flag)\.?$",
            stripped,
            re.IGNORECASE,
        ):
            return False
        return bool(
            "\n" in stripped
            or re.search(r"[A-Za-z_]\w*\s*\(", stripped)
            or re.search(r"\w+\s*=", stripped)
        )

    def add(value: Any, dest: List[str]) -> None:
        if isinstance(value, dict):
            if value.get("is_system_or_output"):
                return
            value = value.get("interactive_input") or value.get("code") or value.get("text")
        elif isinstance(value, list):
            for item in value:
                add(item, dest)
            return
        text = str(value or "").strip()
        if text and looks_concrete_repro(text):
            dest.append(text)

    scenario = scenario or {}
    clue = clue or {}
    add(scenario.get("reproduction_code"), scenario_blocks)
    if not scenario_blocks:
        add(scenario.get("execution_stimulus"), scenario_blocks)
    if scenario_blocks:
        return _dedup_text_items(scenario_blocks)[:5]
    add(clue.get("code_examples"), clue_blocks)
    return _dedup_text_items(clue_blocks)[:5]


_REPRO_TERM_STOPWORDS = {
    "assert", "pytest", "self", "result", "expected", "actual", "import",
    "from", "with", "return", "print", "test", "none", "true", "false",
    "array", "list", "dict", "str", "int", "float", "len", "range",
    "call", "compare", "compute", "run", "context", "exception",
    "assertraises", "assertin", "asserttrue", "assertfalse", "assert_equal",
    "main", "skip", "raises", "warns", "instance",
}
_REPRO_ASSERTION_HELPERS = {
    "assertRaises", "assertIn", "assertTrue", "assertFalse", "assertEqual",
    "assertNotEqual", "assertIs", "assertIsNot", "assertIsNone",
    "assertIsNotNone", "assert_array_equal", "assert_allclose",
}
_REPRO_ALIAS_EQUIVALENTS = {
    "Q": {"Q", "Query"},
    "Query": {"Q", "Query"},
    "models": {"models", "m"},
    "m": {"models", "m"},
}


def _issue_reproduction_required_terms(
    clue: Optional[Dict[str, Any]],
    scenario: Optional[Dict[str, Any]],
) -> List[str]:
    text = "\n".join(_scenario_reproduction_blocks(clue, scenario))
    if not text:
        return []
    terms: List[str] = []

    for dotted in re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b", text):
        lower = dotted.lower()
        if (
            not lower.startswith(("np.", "pytest.", "self.", "context."))
            and not lower.startswith(("tests.", "testing."))
            and ".py" not in lower
            and not lower.endswith(".exception")
        ):
            terms.append(dotted)
    for call in re.findall(r"\b([A-Za-z_]\w*)\s*\(", text):
        lower = call.lower()
        if (
            len(call) >= 4
            and lower not in _REPRO_TERM_STOPWORDS
            and not lower.startswith("test_")
            and call not in _REPRO_ASSERTION_HELPERS
        ):
            terms.append(call)
    for ident in re.findall(r"\b[A-Z][A-Za-z0-9_]{3,}\b", text):
        lower = ident.lower()
        if (
            lower not in _REPRO_TERM_STOPWORDS
            and not ident.startswith("Test")
            and ident not in _REPRO_ASSERTION_HELPERS
        ):
            terms.append(ident)
    for op, label in (("&", "operator:&"), ("**", "operator:**")):
        if op in text:
            terms.append(label)
    return _dedup_text_items(terms, limit=10)


def _alias_variants_for_term(term: str) -> List[str]:
    if "." not in term:
        return [term]
    head, rest = term.split(".", 1)
    aliases = _REPRO_ALIAS_EQUIVALENTS.get(head)
    if not aliases:
        return [term]
    return [f"{alias}.{rest}" for alias in aliases]


def _term_present_in_code(term: str, code: str) -> bool:
    if term.startswith("operator:"):
        return term.split(":", 1)[1] in code
    if "." in term:
        return any(variant in code for variant in _alias_variants_for_term(term))
    aliases = _REPRO_ALIAS_EQUIVALENTS.get(term)
    if aliases:
        return any(re.search(rf"\b{re.escape(alias)}\b", code) for alias in aliases)
    return bool(re.search(rf"\b{re.escape(term)}\b", code))


def _detect_issue_reproduction_drift(
    code: str,
    clue: Optional[Dict[str, Any]],
    scenario: Optional[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    required = _issue_reproduction_required_terms(clue, scenario)
    if len(required) < 2:
        return []
    missing = [term for term in required if not _term_present_in_code(term, code)]
    if len(missing) >= max(2, (len(required) + 1) // 2):
        verified = (context or {}).get("verified_target_evidence") if isinstance(context, dict) else {}
        target_callable = str((verified or {}).get("target_callable") or "")
        has_explicit_assertion = bool(
            re.search(
                r"^\s*(assert\s+|self\.assert(?:Equal|True|False|Is|In|NotIn|Almost)|np\.testing\.assert)",
                code,
                re.MULTILINE,
            )
        )
        if (
            isinstance(verified, dict)
            and verified.get("target_callable_exists")
            and target_callable
            and ReproductionTestGenerator._check_target_function_presence(target_callable, code)
            and has_explicit_assertion
        ):
            return []
        kept = [term for term in required if term not in missing]
        return [
            "issue_reproduction_code_not_followed: "
            f"missing canonical issue terms/operators {missing[:5]} "
            f"(kept {kept[:3]}). Reuse the original reproduction stimulus instead of a generic example."
        ]
    return []


def _semantic_anchor_violations(
    code: str,
    clue: Optional[Dict[str, Any]],
    scenario: Optional[Dict[str, Any]],
) -> List[str]:
    """Check only explicit issue/scenario anchors; do not infer new semantics."""
    clue = clue or {}
    scenario = scenario or {}
    violations: List[str] = []
    issue_api = _selected_issue_api_target(clue, scenario)
    if issue_api and not _term_present_in_code(issue_api, code):
        violations.append(f"issue_api_not_preserved:{issue_api}")

    stimulus = "\n".join(str(item) for item in scenario.get("execution_stimulus", []) or [])
    stimulus_calls = [
        full
        for full, bare in re.findall(
            r"\b((?:[A-Za-z_]\w*\.)?([A-Za-z_]\w{2,}))\s*\(",
            stimulus,
        )
        if bare.lower() not in _REPRO_TERM_STOPWORDS
    ]
    missing_calls = [call for call in stimulus_calls if not _term_present_in_code(call, code)]
    if stimulus_calls and len(missing_calls) == len(stimulus_calls):
        violations.append(
            "key_stimulus_calls_not_preserved:" + ",".join(missing_calls[:4])
        )

    expected_outputs = [
        str(item).strip()
        for item in clue.get("expected_outputs", []) or []
        if str(item).strip()
    ]
    if expected_outputs and "assert" in code and not any(
        output.lower() in code.lower() for output in expected_outputs
    ):
        violations.append("expected_behavior_oracle_not_preserved")
    issue_text = _issue_evidence_text(clue, scenario)
    issue_contract_comparisons = set(
        re.findall(
            r"(?:should|must)[^\n]{0,100}?(>=|<=|==|!=|>|<)\s*"
            r"(-?\d+(?:\.\d+)?|None|True|False)",
            issue_text,
            re.IGNORECASE,
        )
    )
    if issue_contract_comparisons and not expected_outputs:
        try:
            assertion_count = sum(isinstance(node, ast.Assert) for node in ast.walk(ast.parse(code)))
        except SyntaxError:
            assertion_count = 0
        if assertion_count > 1:
            violations.append("unsupported_additional_oracles_masked_by_issue_exception")
    return violations


def _detect_retry_required_oracle_risks(
    code: str,
    clue: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Detect oracle patterns that should trigger repair/retry, not final eval.

    This is intentionally patch-free: it uses only generated code and issue clue.
    """
    risks: List[str] = []

    def add(flag: str) -> None:
        if flag not in risks:
            risks.append(flag)

    if re.search(r"@image_comparison", code):
        add("image_comparison_decorator")

    has_raises = bool(re.search(r"pytest\.raises|assertRaises|assert_raises|with\s+.*raises", code))
    has_body_assertion = bool(re.search(
        r"^\s*(assert\s+(?!.*raises)|self\.assert(?:Equal|True|False|Is|In|NotIn|Almost)|np\.testing\.assert)",
        code,
        re.MULTILINE,
    ))
    if has_raises and (
        _issue_says_success_path(clue)
        or (not _issue_says_exception_expected(clue) and re.search(r"post[- ]fix|should\s+accept|fit\s+success|succeed", code, re.IGNORECASE))
    ):
        add("fix_disappearing_exception_oracle")
    if has_raises and not has_body_assertion and not _issue_says_exception_validation_or_message(clue):
        add("raises_only_no_body_assertion")

    warning_expected = _issue_says_warning_expected(clue)
    if re.search(
        r"len\s*\(\s*w\s*\)\s*==|"
        r"issubclass\s*\([^)]*(?:Warning|RuntimeWarning)|"
        r"\.category\s*,\s*(?:Warning|RuntimeWarning)|"
        r"assertWarns|pytest\.warns",
        code,
        re.IGNORECASE | re.DOTALL,
    ) and not warning_expected:
        add("warning_presence_oracle")

    if re.search(
        r"(?:self\.)?assert(?:In|NotIn)\s*\(\s*['\"][^'\"]{80,}['\"]|"
        r"(?:self\.)?assert(?:In|NotIn)\s*\(\s*['\"][^'\"]*(?:\\PYG|\\sphinx|<[^>]+>|latex|html)[^'\"]*['\"]",
        code,
        re.IGNORECASE,
    ):
        add("raw_rendered_output_exact_match")

    # Private attribute reads are often fragile, but treating every read as a
    # hard retry gate causes many otherwise executable regression tests to die
    # at generation time.  Direct private-state assignments are removed by
    # _fix_private_attr_access(); reads are left to alignment/final eval.

    if re.search(
        r"(?:expected|baseline|correct|desired)_(?:matrix|array|result|values?)\s*=.*\n"
        r"(?s:.*?)(?:assert_array_equal|assert_allclose|assert_equal)\s*\([^,\n]+,\s*"
        r"(?:expected|baseline|correct|desired)_(?:matrix|array|result|values?)",
        code,
        re.IGNORECASE,
    ) and not _has_issue_expected_signal(code, clue):
        add("guessed_expected_array")

    if re.search(
        r"(?:expected|baseline|correct|desired)_(?:value|output|result)\s*=.*\n"
        r"(?s:.*?)(?:assert\s+[^=\n]+==\s*|self\.assertEqual\s*\([^,\n]+,\s*)"
        r"(?:expected|baseline|correct|desired)_(?:value|output|result)",
        code,
        re.IGNORECASE,
    ) and not _has_issue_expected_signal(code, clue):
        add("guessed_expected_value")

    if re.search(
        r"(?:expected|baseline|correct|desired|known)_[A-Za-z0-9_]*\s*!=|"
        r"assert\s+repr\s*\(\s*(?:expected|baseline|correct|desired|known)_[A-Za-z0-9_]*\s*\)\s*!=",
        code,
        re.IGNORECASE,
    ):
        add("constant_negative_oracle")

    assertion_lines = _assertion_lines(code)
    if assertion_lines and _has_positive_issue_oracle_signal(clue):
        negative_lines = [
            line for line in assertion_lines
            if (
                "!=" in line
                or re.search(r"\bnot\s+in\b|\bis\s+not\b", line, re.IGNORECASE)
                or re.search(r"assert(?:Not|False)", line)
            )
        ]
        if len(negative_lines) == len(assertion_lines):
            add("negative_literal_oracle")

    return risks


def _identifier_terms(clue: Optional[Dict[str, Any]]) -> set[str]:
    clue = clue or {}
    identifiers = clue.get("identifiers", {}) if isinstance(clue.get("identifiers"), dict) else {}
    terms: set[str] = set()
    for key in ("functions", "classes", "exceptions", "files"):
        for value in identifiers.get(key, []) or []:
            text = str(value).lower()
            if len(text) >= 3:
                terms.add(text)
                terms.update(t for t in re.split(r"[_\W]+", text) if len(t) >= 4)
    return terms


def _target_terms(context: Optional[Dict[str, Any]], scenario: Optional[Dict[str, Any]] = None) -> set[str]:
    terms: set[str] = set()
    scenario = scenario or {}
    target = scenario.get("target_location", {}) if isinstance(scenario.get("target_location"), dict) else {}
    paths = [
        target.get("source_file", ""),
        target.get("target_function", ""),
    ]
    for item in (context or {}).get("candidate_source_files", [])[:3]:
        if isinstance(item, dict):
            paths.append(item.get("path", ""))
            paths.extend(item.get("matched_identifiers", []) or [])
    for path in paths:
        text = str(path).lower()
        terms.update(t for t in re.split(r"[/_.\W]+", text) if len(t) >= 4)
    return terms


def _detect_semantic_risk_flags(
    code: str,
    clue: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    scenario: Optional[Dict[str, Any]] = None,
    original_content: str = "",
) -> List[str]:
    """Detect issue/context-inconsistent generated tests without using patches."""
    flags: List[str] = []

    def add(flag: str) -> None:
        if flag not in flags:
            flags.append(flag)

    lower = code.lower()
    clue_terms = _identifier_terms(clue)
    target_terms = _target_terms(context, scenario)
    allowed_terms = clue_terms | target_terms

    unrelated_sklearn = {
        "countvectorizer",
        "tfidfvectorizer",
        "latentdirichletallocation",
        "lda",
        "kmeans",
        "randomforestclassifier",
        "svc",
    }
    if "sklearn" in lower:
        for symbol in unrelated_sklearn:
            if symbol in lower and symbol not in allowed_terms:
                add("unrelated_api_invention")
                break

    if re.search(r"@unittest\.skip|unittest\.skip\s*\(", code) and re.search(r"\bxxx\b", code):
        add("inline_skipped_class_reproduction")

    if "httpdigestauth" in lower and "www-authenticate" not in lower and "chal" not in lower:
        add("requests_digest_without_challenge")

    issue_literals = _extract_issue_literals(clue)
    for literal in issue_literals:
        literal_l = literal.lower()
        if literal_l == "content-length" and literal_l not in lower:
            add("issue_literal_not_used=Content-Length")
            break

    # Placeholder checks must be semantic-identifier exact.  Searching the
    # entire source text made ordinary prose such as "unexpected keyword"
    # look like an invented ``Unexpected-Key`` symbol.
    try:
        parsed_tree = ast.parse(code)
    except SyntaxError:
        parsed_tree = None
    if parsed_tree is not None:
        semantic_identifiers: set[str] = set()
        for node in ast.walk(parsed_tree):
            if isinstance(node, ast.Name):
                semantic_identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                semantic_identifiers.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                semantic_identifiers.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                semantic_identifiers.update(alias.asname or alias.name.split(".")[-1] for alias in node.names)
        normalized_identifiers = {
            re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
            for value in semantic_identifiers
        }
        if (
            normalized_identifiers & {
                "unexpected_key", "x_unrelated", "dummy_header", "xxx",
                "mockmodel", "foomodel", "barmodel",
            }
            or any(value.startswith("dummy") for value in normalized_identifiers)
        ):
            add("placeholder_symbol")

    if "author.objects" in lower and ".annotate(" in lower and ".order_by(" not in lower:
        add("django_query_warning_unhandled")

    target = (scenario or {}).get("target_location", {}) if isinstance((scenario or {}).get("target_location"), dict) else {}
    target_function = str(target.get("target_function") or "")
    # target_function_not_called gets an explicit rewrite chance in generate().
    # Do not hard-block it here, or valid public-wrapper reproductions become
    # NOT_VALID before alignment can judge them.

    runner = ((context or {}).get("project_test_style") or {}).get("runner", "")
    if runner == "django-test":
        if re.search(r"class\s+\w+\s*\([^)]*models\.Model[^)]*\)", code):
            add("django_inline_model")
        if re.search(r"\b(?:models\.Model|MockModel)\.objects\b|MockModel\._meta\b|\bself\.apps\b|\bapp_label\s*=", code):
            add("django_invalid_model_api")
        target_file = str(((scenario or {}).get("target_location") or {}).get("candidate_test_file") or "")
        issue_text = _issue_text_blob(clue).lower()
        if "tests/queries" in target_file and "user.objects" in lower and "user" not in issue_text and "auth" not in issue_text:
            add("django_existing_query_model_not_reused")
        known_symbols = _imported_or_existing_symbols(code + "\n" + original_content)
        for model_name in re.findall(r"\b([A-Z][A-Za-z0-9_]+)\.objects\.", code):
            if model_name not in known_symbols:
                add(f"unknown_django_model={model_name}")
                break

    if "sphinx.testing.fixtures" in lower or "pytest_plugins" in lower and "sphinx.testing" in lower:
        add("sphinx_testing_fixture_import")

    if ".get_legend(" in code and (
        "seaborn" in lower
        or "plotter" in lower
        or re.search(r"\bPlot\s*\(", code)
        or "plot.plot" in lower
    ):
        add("seaborn_plotter_matplotlib_axes_oracle")

    if re.search(r"\bisinstance\s*\([^,\n]+,\s*HasValues\s*\)", code):
        add("xarray_scalar_dataarray_raw_object_oracle")

    if re.search(r"assert\s+\w+\s+is\s+\w+\b", code):
        add("local_object_identity_oracle")

    return flags


def _detect_repair_directive_violations(
    code: str,
    scenario: Optional[Dict[str, Any]] = None,
) -> List[str]:
    directive = (scenario or {}).get("repair_directive")
    if not isinstance(directive, dict):
        return []
    patterns = directive.get("forbidden_patterns") or []
    violations: List[str] = []
    for pattern in patterns:
        text = str(pattern or "").strip()
        if not text:
            continue
        if text in code:
            violations.append(text[:160])
            continue
        # Some directives intentionally carry regex-like fragments. Try them
        # cautiously after literal matching, but ignore malformed patterns.
        if len(text) <= 160:
            try:
                if re.search(text, code):
                    violations.append(text[:160])
            except re.error:
                pass
    policy = directive.get("preservation_policy")
    expected = directive.get("semantic_preservation_fingerprints")
    previous_candidate = str(directive.get("previous_candidate") or "")
    # There is no semantic baseline to preserve after an M4/M5-null failure.
    # Hashes of the empty program previously turned the first real repair into
    # a false destructive-change rejection.
    if (
        previous_candidate.strip()
        and isinstance(policy, Mapping)
        and isinstance(expected, Mapping)
    ):
        actual = candidate_repair_semantics(code)
        field_to_partition = {
            "stimulus": "stimulus_semantics",
            "issue_stimulus": "stimulus_semantics",
            "stimulus_semantics": "stimulus_semantics",
            "semantic_body": "stimulus_semantics",
            "test_behavior": "stimulus_semantics",
            "oracle": "oracle_semantics",
            "oracle_semantics": "oracle_semantics",
            "valid_oracle": "oracle_semantics",
            # Import and framework shape are independently validated against
            # the repository.  They are never fingerprint locks: both must be
            # free to change for import/framework/setup repairs.
        }
        for field in policy.get("MUST_PRESERVE_SEMANTICS") or []:
            partition = field_to_partition.get(str(field))
            if (
                partition
                and expected.get(partition)
                and expected.get(partition) != actual.get(partition)
            ):
                stable_label = {
                    "stimulus_semantics": "semantic_body",
                    "oracle_semantics": "oracle",
                }.get(partition, partition)
                violations.append(f"preserved_{stable_label}_changed")
    return _dedup_text_items(violations)[:5]


def candidate_repair_fingerprints(code: str) -> Dict[str, str]:
    """Fingerprint repair dimensions so unchanged behavior is enforceable."""
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return {}

    partitions: Dict[str, List[str]] = {
        "imports": [],
        "oracle": [],
        "semantic_body": [],
        "test_framework": [],
    }

    def is_oracle(statement: ast.stmt) -> bool:
        if isinstance(statement, ast.Assert):
            return True
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            return any(
                isinstance(node, ast.Call)
                and (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"raises", "warns"}
                    or isinstance(node.func, ast.Name)
                    and node.func.id in {"raises", "warns"}
                )
                for node in ast.walk(statement)
            )
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            function = statement.value.func
            name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else function.id
                if isinstance(function, ast.Name)
                else ""
            )
            return name.startswith("assert") or name in {"fail", "raises", "warns"}
        return False

    def collect(statements: Sequence[ast.stmt]) -> None:
        for statement in statements:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                partitions["imports"].append(ast.dump(statement, include_attributes=False))
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name.startswith("test_"):
                partitions["test_framework"].append(
                    json.dumps(
                        {
                            "kind": type(statement).__name__,
                            "name": statement.name,
                            "args": ast.dump(statement.args, include_attributes=False),
                            "decorators": [ast.dump(item, include_attributes=False) for item in statement.decorator_list],
                        },
                        sort_keys=True,
                    )
                )
                collect(statement.body)
            elif isinstance(statement, ast.ClassDef) and statement.name.startswith("Test"):
                partitions["test_framework"].append(
                    json.dumps(
                        {
                            "kind": "ClassDef",
                            "name": statement.name,
                            "bases": [ast.dump(item, include_attributes=False) for item in statement.bases],
                        },
                        sort_keys=True,
                    )
                )
                collect(statement.body)
            elif is_oracle(statement):
                partitions["oracle"].append(ast.dump(statement, include_attributes=False))
            else:
                partitions["semantic_body"].append(ast.dump(statement, include_attributes=False))

    collect(tree.body)
    return {
        name: sha256_text(json.dumps(values, sort_keys=True, separators=(",", ":")))
        for name, values in partitions.items()
    }


def candidate_repair_semantics(code: str) -> Dict[str, str]:
    """Return AST semantic partitions used by dimension-local repair.

    These fingerprints deliberately omit source formatting and local variable
    names.  They are applied only to dimensions explicitly classified as
    ``MUST_PRESERVE_SEMANTICS``; setup/import/framework dimensions marked
    ``MAY_CHANGE_FOR_REPAIR`` are never byte- or AST-locked.
    """
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return {}

    local_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            local_names.update(
                inner.id
                for inner in ast.walk(node)
                if isinstance(inner, ast.Name)
                and isinstance(inner.ctx, ast.Store)
            )
            arguments = node.args
            local_names.update(
                argument.arg
                for argument in (
                    list(arguments.posonlyargs)
                    + list(arguments.args)
                    + list(arguments.kwonlyargs)
                )
            )
            if arguments.vararg is not None:
                local_names.add(arguments.vararg.arg)
            if arguments.kwarg is not None:
                local_names.add(arguments.kwarg.arg)

    candidate_bindings: Dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = node.value
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value = node.value
            targets = [node.target]
        else:
            continue
        if any(
            isinstance(inner, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom, ast.Lambda))
            for inner in ast.walk(value)
        ):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                candidate_bindings.setdefault(target.id, []).append(value)
    static_bindings = {
        name: values[0]
        for name, values in candidate_bindings.items()
        if len(values) == 1
    }

    def copied_expression(value: ast.AST) -> ast.AST:
        return ast.parse(ast.unparse(value), mode="eval").body

    def resolved_static_expression(name: str, seen: set[str] | None = None) -> ast.AST:
        active = set(seen or set())
        active.add(name)

        class StaticBindingResolver(ast.NodeTransformer):
            def visit_Name(self, inner: ast.Name) -> ast.AST:
                if inner.id in static_bindings and inner.id not in active:
                    return ast.copy_location(
                        resolved_static_expression(inner.id, active), inner
                    )
                return inner

        return StaticBindingResolver().visit(copied_expression(static_bindings[name]))

    def is_oracle_node(node: ast.AST) -> bool:
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Call):
            name = _call_name(node.func).split(".")[-1]
            return name.startswith("assert") or name in {"raises", "warns", "fail"}
        return False

    def oracle_token(node: ast.AST) -> str:
        class LocalNameNormalizer(ast.NodeTransformer):
            def visit_Name(self, inner: ast.Name) -> ast.AST:
                if inner.id in static_bindings:
                    return ast.copy_location(
                        resolved_static_expression(inner.id), inner
                    )
                if inner.id in local_names:
                    return ast.copy_location(ast.Name(id="LOCAL", ctx=inner.ctx), inner)
                return inner

        def normalized_dump(value: ast.AST) -> str:
            normalized = LocalNameNormalizer().visit(ast.fix_missing_locations(ast.parse(
                ast.unparse(value), mode="eval"
            ).body))
            return ast.dump(normalized, include_attributes=False)

        if isinstance(node, ast.Assert):
            test = node.test
            if isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1:
                return (
                    type(test.ops[0]).__name__ + ":"
                    + normalized_dump(test.left) + ":"
                    + normalized_dump(test.comparators[0])
                )
            return "Truth:" + normalized_dump(test)
        if isinstance(node, ast.Call):
            name = _call_name(node.func).split(".")[-1]
            aliases = {
                "assertEqual": "Eq", "assertNotEqual": "NotEq",
                "assertTrue": "Truth", "assertFalse": "Falsehood",
                "assertIs": "Is", "assertIsNot": "IsNot",
                "assertIn": "In", "assertNotIn": "NotIn",
            }
            return aliases.get(name, name) + ":" + ":".join(
                normalized_dump(arg) for arg in node.args
            )
        return ast.dump(node, include_attributes=False)

    imports: list[str] = []
    framework: list[str] = []
    stimulus_calls: list[str] = []
    oracle_nodes: list[str] = []
    import_call_aliases: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_call_aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name != "*":
                    import_call_aliases[alias.asname or alias.name] = (
                        f"{node.module}.{alias.name}"
                    )
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.dump(node, include_attributes=False))
        elif isinstance(node, ast.ClassDef):
            framework.append(
                "class:" + ",".join(
                    _call_name(base).split(".")[-1] for base in node.bases
                )
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            framework.append("async_test" if isinstance(node, ast.AsyncFunctionDef) else "test")
        elif isinstance(node, ast.Call):
            if is_oracle_node(node):
                oracle_nodes.append(oracle_token(node))
            else:
                name = _call_name(node.func).split(".")[-1]
                if name and name not in {
                    "len", "str", "int", "float", "list", "dict", "set", "tuple",
                    "range", "print", "repr", "super",
                }:
                    class ArgumentNameNormalizer(ast.NodeTransformer):
                        def visit_Name(self, inner: ast.Name) -> ast.AST:
                            if inner.id in static_bindings:
                                return ast.copy_location(
                                    resolved_static_expression(inner.id), inner
                                )
                            if inner.id in local_names:
                                return ast.copy_location(
                                    ast.Name(id="LOCAL", ctx=inner.ctx), inner
                                )
                            return inner

                    normalized_args = [
                        ast.dump(
                            ArgumentNameNormalizer().visit(
                                ast.parse(ast.unparse(arg), mode="eval").body
                            ),
                            include_attributes=False,
                        )
                        for arg in node.args
                    ]
                    normalized_keywords = [
                        (
                            keyword.arg,
                            ast.dump(
                                ArgumentNameNormalizer().visit(
                                    ast.parse(
                                        ast.unparse(keyword.value), mode="eval"
                                    ).body
                                ),
                                include_attributes=False,
                            ),
                        )
                        for keyword in node.keywords
                    ]
                    callable_name = _call_name(node.func)
                    if isinstance(node.func, ast.Attribute):
                        root = node.func.value
                        if isinstance(root, ast.Name) and root.id in local_names:
                            callable_name = "LOCAL." + node.func.attr
                    elif isinstance(node.func, ast.Name) and node.func.id in local_names:
                        callable_name = "LOCAL"
                    if callable_name and callable_name != "LOCAL":
                        root, separator, remainder = callable_name.partition(".")
                        if root in import_call_aliases:
                            callable_name = import_call_aliases[root] + (
                                separator + remainder if separator else ""
                            )
                    stimulus_calls.append(
                        json.dumps(
                            [callable_name, normalized_args, normalized_keywords],
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
        elif isinstance(node, ast.Assert):
            oracle_nodes.append(oracle_token(node))

    def digest(values: Iterable[str]) -> str:
        return sha256_text(json.dumps(sorted(set(values)), separators=(",", ":")))

    return {
        "stimulus_semantics": digest(stimulus_calls),
        "oracle_semantics": digest(oracle_nodes),
        "import_semantics": digest(imports),
        "framework_semantics": digest(framework),
    }


def _would_violate_repair_directive(
    code: str,
    scenario: Optional[Dict[str, Any]] = None,
) -> bool:
    return bool(_detect_repair_directive_violations(code, scenario))


def _has_explicit_expected_output(
    clue: Optional[Dict[str, Any]],
    scenario: Optional[Dict[str, Any]],
) -> bool:
    for source in (scenario or {}, clue or {}):
        outputs = source.get("expected_outputs") or []
        if any(str(out).strip() for out in outputs):
            return True
    return False


def _assertion_lines(code: str) -> List[str]:
    return [
        line.strip()
        for line in code.splitlines()
        if re.match(
            r"\s*(assert\s+|self\.assert(?:Equal|True|False|Is|IsNot|IsInstance|In|NotIn|Greater|Less)|np\.testing\.assert)",
            line,
        )
    ]


def _has_structural_only_assertion(code: str) -> bool:
    assertions = _assertion_lines(code)
    if not assertions:
        return False
    structural = (
        r"\bis\s+not\s+None\b",
        r"\bisinstance\s*\(",
        r"\blen\s*\([^)]+\)\s*(?:>|>=|!=)\s*0\b",
        r"assert(?:IsNotNone|IsInstance|True)\s*\(",
        r"\.shape\b",
        r"\.ndim\b",
    )
    return all(any(re.search(pat, line, re.IGNORECASE) for pat in structural) for line in assertions)


def _tier2_result_assignment_matches_target(
    code: str,
    scenario: Optional[Dict[str, Any]],
) -> bool:
    target = (scenario or {}).get("target_location", {})
    target_func = ""
    if isinstance(target, dict):
        target_func = str(target.get("target_function") or "").strip()
    assertions = _assertion_lines(code)
    if not assertions:
        return False
    first_assert = code.find(assertions[0])
    prefix = code if first_assert < 0 else code[:first_assert]
    assignments = [
        line.strip()
        for line in prefix.splitlines()
        if re.match(r"^\w+\s*=\s*.+\(", line.strip())
    ]
    if not assignments:
        return False
    if not target_func:
        return True
    last_assignment = assignments[-1]
    return bool(re.search(rf"\b{re.escape(target_func)}\s*\(", last_assignment))


def _should_inject_tier2_assertion(
    code: str,
    clue: Optional[Dict[str, Any]],
    scenario: Optional[Dict[str, Any]],
) -> bool:
    if _has_explicit_expected_output(clue, scenario):
        return False
    return (
        _has_structural_only_assertion(code)
        and _tier2_result_assignment_matches_target(code, scenario)
    )


def _imported_or_existing_symbols(code: str) -> set[str]:
    symbols: set[str] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return symbols
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                symbols.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    symbols.add(alias.asname or alias.name)
        elif isinstance(node, ast.ClassDef):
            symbols.add(node.name)
    return symbols


def _remove_warning_presence_assertions(code: str) -> str:
    """Remove warning-count/type assertions when another value oracle remains."""
    patterns = [
        r"^\s*assert\s+len\s*\(\s*w\s*\)\s*==\s*\d+\s*$",
        r"^\s*self\.assertEqual\s*\(\s*len\s*\(\s*w\s*\)\s*,\s*\d+\s*\)\s*$",
        r"^\s*assert\s+issubclass\s*\([^)]*(?:Warning|RuntimeWarning)[^)]*\)\s*$",
        r"^\s*self\.assertTrue\s*\(\s*issubclass\s*\([^)]*(?:Warning|RuntimeWarning)[^)]*\)\s*\)\s*$",
    ]
    new_code = code
    for pat in patterns:
        new_code = re.sub(
            pat,
            lambda m: " " * (len(m.group(0)) - len(m.group(0).lstrip())) + "# [removed: warning presence oracle — assert fixed value/state instead]",
            new_code,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    return new_code


def _remove_image_comparison_decorators(code: str) -> str:
    return re.sub(
        r"^\s*@image_comparison\s*\([^)]*\)\s*\n",
        "",
        code,
        flags=re.MULTILINE | re.DOTALL,
    )


def _apply_oracle_repairs(
    code: str,
    clue: Optional[Dict[str, Any]] = None,
) -> tuple[str, List[str]]:
    """Apply deterministic, patch-free oracle repairs to generated code."""
    actions: List[str] = []
    repaired = code
    original = code

    def changed(name: str, new_code: str) -> None:
        nonlocal repaired
        if new_code != repaired:
            repaired = new_code
            actions.append(name)

    changed("remove_image_comparison_decorator", _remove_image_comparison_decorators(repaired))
    changed("remove_trivial_assertions", _remove_trivial_assertions(repaired))
    changed("remove_exception_message_matching", _fix_exception_message_matching(repaired))
    changed("remove_private_attribute_assignment", _fix_private_attr_access(repaired))
    changed("remove_warning_presence_assertions", _remove_warning_presence_assertions(repaired))
    changed("prune_to_best_generated_test", _prune_to_best_generated_test(repaired, clue))

    def assertion_count(value: str) -> int:
        try:
            tree = ast.parse(value)
        except SyntaxError:
            return len(re.findall(r"^\s*assert\b|pytest\.raises|self\.assert[A-Z]", value, re.MULTILINE))
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                # ``assert True``/constant self-equality is intentionally not
                # considered a valid issue oracle for preservation purposes.
                trivial = isinstance(node.test, ast.Constant) and node.test.value is True
                trivial = trivial or (
                    isinstance(node.test, ast.Compare)
                    and isinstance(node.test.left, ast.Constant)
                    and len(node.test.comparators) == 1
                    and isinstance(node.test.comparators[0], ast.Constant)
                    and node.test.left.value == node.test.comparators[0].value
                )
                rendered = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
                semantic_risk = "str(" in rendered or "warning" in rendered.lower()
                if not trivial and not semantic_risk:
                    count += 1
        count += sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr.startswith("assert") for node in ast.walk(tree))
        return int(count)
    if assertion_count(original) > 0 and assertion_count(repaired) == 0:
        repaired = original
        actions.append("preserve_last_valid_oracle")

    return repaired, actions


def apply_m5a_deterministic_postprocessing(
    parsed: Dict[str, Any],
    *,
    clue: Optional[Dict[str, Any]] = None,
    repo_path: str = "",
    context: Optional[Dict[str, Any]] = None,
    runner: str = "pytest",
    import_checker=None,
    preserve_test_semantics: bool = False,
) -> tuple[Dict[str, Any], List[str]]:
    """Apply the mandatory M5-A rules before any optional LLM refinement.

    This is intentionally patch-free: it normalizes generated test structure
    and imports while preserving the candidate's target and oracle semantics.
    ``import_checker`` is optional so fixture callers can exercise the rules
    without constructing a full generator client.
    """
    current = dict(parsed or {})
    actions: List[str] = []
    for key in ("append_block", "test_code"):
        code = current.get(key)
        if not code:
            continue
        repaired, oracle_actions = (
            (str(code), [])
            if preserve_test_semantics
            else _apply_oracle_repairs(str(code), clue)
        )
        if repaired != code:
            current[key] = repaired
        actions.extend(f"{key}:{item}" for item in oracle_actions)
    if runner == "django-test" and not preserve_test_semantics:
        imports = _fix_django_imports(list(current.get("imports", []) or []))
        if imports != current.get("imports", []):
            current["imports"] = imports
            actions.append("imports:django_normalization")
        for key in ("append_block", "test_code"):
            if current.get(key):
                repaired = _fix_django_test_code(str(current[key]))
                if repaired != current[key]:
                    current[key] = repaired
                    actions.append(f"{key}:django_structural_normalization")
    if not preserve_test_semantics and "sphinx" in str((context or {}).get("repo") or "").lower():
        for key in ("append_block", "test_code"):
            if current.get(key):
                repaired = _fix_sphinx_test_code(str(current[key]))
                if repaired != current[key]:
                    current[key] = repaired
                    actions.append(f"{key}:sphinx_structural_normalization")
    before = current.get("append_block") or current.get("test_code") or ""
    normalized = _fix_append_block_imports(current, repo_path=repo_path)
    if normalized != current:
        actions.append("append_block:stdlib_import_normalization")
    current = normalized
    if import_checker is not None and repo_path and current.get("append_block"):
        normalized = _fix_append_block_repo_imports(
            current,
            repo_path=repo_path,
            context=context or {},
            import_checker=import_checker,
        )
        if normalized != current:
            actions.append("append_block:repository_import_normalization")
        current = normalized
    if before and current.get("test_code") != before and not current.get("append_block"):
        current["test_code"] = current.get("append_block") or before
    return current, list(dict.fromkeys(actions))


def _postprocessing_action_records(actions: List[str]) -> List[Dict[str, Any]]:
    """Represent deterministic M5-A actions without adding execution results."""
    records: List[Dict[str, Any]] = []
    for action in _dedup_text_items(actions):
        records.append({
            "stage": "m5a_deterministic_postprocessing",
            "action": action,
            "source": "static_generated_test_processing",
        })
    return records


def _m3_m5_optional_feature_metadata(
    flags: V22FeatureFlags,
    candor_result: Optional[CandorConsensusResult] = None,
    llm_error_refinement: Optional[ErrorRefinementResult] = None,
) -> Dict[str, Any]:
    """Record optional-feature usage and deterministic fallback status."""
    candor_used = bool(candor_result and candor_result.consensus_reached)
    candor_fallback = (
        candor_result.fallback_reason
        if candor_result is not None
        else "non_candor_oracle_validation"
    )
    llm_used = bool(llm_error_refinement and llm_error_refinement.used)
    llm_fallback = (
        llm_error_refinement.fallback_reason
        if llm_error_refinement is not None
        else "deterministic_m5a_postprocessing"
    )
    return {
        "enable_m5_candor": {
            "requested": bool(flags.enable_m5_candor),
            "enabled": bool(flags.enable_m5_candor),
            "used": candor_used,
            "status": "USED" if candor_used else ("FALLBACK" if flags.enable_m5_candor else "DISABLED"),
            "unavailable": False,
            "fallback": "" if candor_used else candor_fallback,
            "reason": (
                "deterministic_candor_consensus"
                if candor_used
                else (candor_fallback if flags.enable_m5_candor else "feature_disabled")
            ),
            "evidence": (
                candor_result.to_dict()
                if candor_result is not None and hasattr(candor_result, "to_dict")
                else {}
            ),
        },
        "enable_m5a_llm_error_refinement": {
            "requested": bool(flags.enable_m5a_llm_error_refinement),
            "enabled": bool(flags.enable_m5a_llm_error_refinement),
            "used": llm_used,
            "status": "USED" if llm_used else ("FALLBACK" if flags.enable_m5a_llm_error_refinement else "DISABLED"),
            "unavailable": False,
            "fallback": "" if llm_used else llm_fallback,
            "reason": (
                "guarded_llm_error_refinement"
                if llm_used
                else (llm_fallback if flags.enable_m5a_llm_error_refinement else "feature_disabled")
            ),
            "evidence": (
                llm_error_refinement.to_dict()
                if llm_error_refinement is not None and hasattr(llm_error_refinement, "to_dict")
                else {}
            ),
        },
    }


def _inject_tier2_assertion(code: str, actual_output: str) -> str:
    """probe로 수집한 buggy_value를 코드 레벨 Tier 2 assertion으로 주입.

    LLM의 assertion 품질과 무관하게 `assert repr(result) != BUGGY_REPR` 구문을
    결정적으로 삽입한다.
    """
    if not actual_output or not code.strip():
        return code

    lines = code.splitlines()

    # except/with 블록 범위 계산 (이 안에서는 result_var가 미정의일 수 있음)
    except_ranges: list = []
    current_except_start: int = -1
    current_except_indent: int = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if re.match(r'except\b', stripped) or re.match(r'with\s+.*raises', stripped):
            current_except_start = i
            current_except_indent = indent
        elif current_except_start >= 0 and stripped and indent <= current_except_indent:
            except_ranges.append((current_except_start, i - 1))
            current_except_start = -1
    if current_except_start >= 0:
        except_ranges.append((current_except_start, len(lines) - 1))

    def _in_except_block(idx: int) -> bool:
        return any(s <= idx <= e for s, e in except_ranges)

    # 마지막 assert 줄 찾기 — except 블록 바깥에 있는 것만 유효
    last_assert_idx = -1
    for i, line in enumerate(lines):
        if re.match(r'\s*(assert|self\.assert)', line.strip()) and not _in_except_block(i):
            last_assert_idx = i

    if last_assert_idx == -1:
        return code

    # assert 직전까지에서 마지막 함수 호출 결과 변수 찾기 (except 블록 밖에서만)
    # expected_*, buggy_*, baseline_* 같은 상수 변수는 제외 (함수 결과가 아님)
    _SKIP_VAR_PREFIXES = ("expected", "buggy", "baseline", "correct", "desired", "known")
    result_var = None
    for i, line in enumerate(lines[:last_assert_idx]):
        if _in_except_block(i):
            continue
        stripped = line.strip()
        m = re.match(r'^(\w+)\s*=\s*\S+\(', stripped)
        if m and not stripped.startswith(("import ", "from ", "def ", "class ")):
            var_name = m.group(1)
            if not any(var_name.lower().startswith(p) for p in _SKIP_VAR_PREFIXES):
                result_var = var_name

    if result_var is None:
        return code

    indent = len(lines[last_assert_idx]) - len(lines[last_assert_idx].lstrip())
    buggy_repr = repr(actual_output)
    tier2 = [
        f"{' ' * indent}# [Tier 2: probe-verified buggy repr — must differ after fix]",
        f"{' ' * indent}assert repr({result_var}) != {buggy_repr}",
    ]
    lines = lines[:last_assert_idx + 1] + tier2 + lines[last_assert_idx + 1:]
    return "\n".join(lines)


def _fix_exception_message_matching(code: str) -> str:
    """예외 메시지 exact/containment matching 제거.

    패치 전후로 에러 메시지 포맷이 달라지면 before AND after 모두 실패한다.
    메시지 비교를 제거하고 예외 타입 체크만 남긴다.
    """
    # 1. assert "msg" in str(exc) / assert "msg" not in str(exc)
    code = re.sub(
        r'^(\s*)assert\s+["\'].+["\']\s+(?:not\s+)?in\s+str\s*\(.+\)\s*$',
        r'\1# [removed: exception message matching — version-dependent]',
        code,
        flags=re.MULTILINE,
    )
    # 2. assert str(exc) ==/!= "exact message"  /  assert str(cm.exception) ==/!= "..."
    code = re.sub(
        r'^(\s*)assert\s+str\s*\(\s*[\w.]+\s*\)\s*(?:==|!=)\s*["\'].+["\']\s*$',
        r'\1# [removed: exception message exact match — version-dependent]',
        code,
        flags=re.MULTILINE,
    )
    # 3. assert exc.value.args[0] ==/!= "exact message" / assert exc.args[0] ==/!= "..."
    code = re.sub(
        r'^(\s*)assert\s+\w+(?:\.value)?\.args\[\d+\]\s*(?:==|!=)\s*["\'].+["\']\s*$',
        r'\1# [removed: exception args exact match — version-dependent]',
        code,
        flags=re.MULTILINE,
    )
    # 4. self.assertEqual/NotEqual(str(exc), "exact message")
    code = re.sub(
        r'^(\s*)self\.assert(?:Equal|NotEqual)\s*\(\s*str\s*\(\s*[\w.]+\s*\)\s*,\s*["\'].+["\']\s*\)\s*$',
        r'\1# [removed: exception message assertEqual — version-dependent]',
        code,
        flags=re.MULTILINE,
    )
    # 5. self.assertIn/NotIn("msg", str(exc))
    code = re.sub(
        r'^(\s*)self\.assert(?:In|NotIn|Regex|NotRegex)\s*\(\s*[^,\n]+,\s*str\s*\(\s*[\w.]+\s*\).*\)\s*$',
        r'\1# [removed: exception message containment — version-dependent]',
        code,
        flags=re.MULTILINE,
    )
    return code


def _remove_trivial_assertions(code: str) -> str:
    """Remove assertions that do not check repository behavior."""
    code = re.sub(
        r'^\s*(?:self\.)?assertTrue\s*\(\s*(?:True|1)\s*(?:,\s*[^)]*)?\)\s*(?:#.*)?$',
        '# [removed: trivial oracle — assert post-fix behavior instead]',
        code,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    code = re.sub(
        r'^\s*assert\s+(?:True|1)\s*(?:#.*)?$',
        '# [removed: trivial oracle — assert post-fix behavior instead]',
        code,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    return code


def _fix_private_attr_access(code: str) -> str:
    """obj._private_attr = value 직접 설정 패턴 제거.

    이 패턴은 내부 상태를 우회 설정하므로 실제 API 동작을 트리거하지 않아
    테스트가 패치 전/후 모두 실패하게 만든다.
    """
    return re.sub(
        r'^(\s*)\w[\w.]*\._\w+\s*=\s*.+$',
        r'\1# [removed: private attribute assignment — use public API instead]',
        code,
        flags=re.MULTILINE,
    )


def _has_private_attr_read(code: str) -> bool:
    """Detect private attribute reads that should get a rewrite chance."""
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.search(r"\._[A-Za-z]\w*\s*=", line):
            continue
        if re.search(r"\._[A-Za-z]\w*", line):
            return True
    return False


def _fix_django_test_code(test_code: str) -> str:
    """test_code 내 unittest.TestCase / unittest.SimpleTestCase 상속 교정.

    추가로:
    - TestCase 상속 없는 class TestXxx → class TestXxx(TestCase)로 교정
    - TestCase 사용하는데 import 없으면 상단에 주입
    - repository-valid package-relative imports are preserved
    """
    # unittest.TestCase → TestCase 교정
    test_code = re.sub(r'\(unittest\.TestCase\)', '(TestCase)', test_code)
    test_code = re.sub(r'\(unittest\.SimpleTestCase\)', '(SimpleTestCase)', test_code)

    # class TestXxx: (상속 없음) → class TestXxx(TestCase):
    test_code = re.sub(
        r'^(class\s+Test\w+)\s*:',
        r'\1(TestCase):',
        test_code,
        flags=re.MULTILINE,
    )

    # The Django runner requires a TestCase/SimpleTestCase container.  When a
    # candidate contains exactly one top-level test function, wrapping that
    # function is a structure-only repair: imports, setup, target call, and
    # assertions are preserved byte-for-byte (apart from indentation).  Do
    # not guess when multiple test functions or an existing class are present.
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        tree = None
    if tree is not None and not any(isinstance(node, ast.ClassDef) for node in tree.body):
        functions = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        ]
        if len(functions) == 1:
            function = functions[0]
            lines = test_code.splitlines()
            start = min(
                [decorator.lineno for decorator in function.decorator_list]
                + [function.lineno]
            ) - 1
            end = function.end_lineno or function.lineno
            block = lines[start:end]
            wrapped = ["class TestGeneratedReproduction(TestCase):"]
            wrapped.extend("    " + line if line else "" for line in block)
            lines[start:end] = wrapped
            test_code = "\n".join(lines)

    # TestCase가 코드에 쓰이는데 import가 없으면 상단에 주입
    uses_testcase = bool(re.search(r'\(TestCase\)|\(SimpleTestCase\)', test_code))
    has_import = bool(re.search(r'from django\.test import', test_code))
    if uses_testcase and not has_import:
        test_code = "from django.test import TestCase, SimpleTestCase\n" + test_code

    return test_code


# stdlib / 공통 모듈 자동 주입 목록: (사용 패턴, import 구문)
_AUTO_INJECT_IMPORTS: List[tuple] = [
    ("unittest.",        "import unittest"),
    ("uuid.",            "import uuid"),
    ("datetime.",        "import datetime"),
    ("os.path",          "import os"),
    ("os.",              "import os"),
    ("sys.",             "import sys"),
    ("json.",            "import json"),
    ("re.",              "import re"),
    ("io.",              "import io"),
    ("math.",            "import math"),
    ("copy.",            "import copy"),
    ("collections.",     "import collections"),
    ("itertools.",       "import itertools"),
    ("functools.",       "import functools"),
    ("pathlib.",         "from pathlib import Path"),
    ("tempfile.",        "import tempfile"),
    ("textwrap.",        "import textwrap"),
    ("inspect.",         "import inspect"),
    ("typing.",          "from typing import Any, Dict, List, Optional, Tuple"),
]


# 관용 alias → import 매핑: 이 alias가 block에서 사용되는데 import가 없으면 실패 처리
_COMMON_ALIAS_IMPORTS: List[tuple] = [
    ("np.",    "import numpy as np"),
    ("pd.",    "import pandas as pd"),
    ("plt.",   "import matplotlib.pyplot as plt"),
    ("scipy.", "import scipy"),
    ("sk.",    "import sklearn"),
    ("tf.",    "import tensorflow as tf"),
    ("torch.", "import torch"),
]


def _detect_missing_common_aliases(append_block: str, existing_file_content: str = "") -> List[str]:
    """append_block에서 관용 alias(np., pd. 등)가 import 없이 쓰이는 경우를 감지한다.

    기존 파일의 imports도 함께 확인하여 이미 import된 것은 제외한다.
    Returns: 누락된 import 구문 목록
    """
    combined_imports = existing_file_content + "\n" + append_block
    missing = []
    for alias, import_stmt in _COMMON_ALIAS_IMPORTS:
        if alias not in append_block:
            continue
        module = import_stmt.split()[1]  # "numpy", "pandas", etc.
        already = (
            f"import {module}" in combined_imports
            or f"as {alias.rstrip('.')}" in combined_imports
        )
        if not already:
            missing.append(import_stmt)
    return missing


def _ensure_repro_suffix(append_block: str) -> str:
    """생성된 테스트 함수/메서드명에 _repro 접미사를 보장한다.

    기존 레포에 같은 이름의 테스트가 있으면 하네스가 잘못된 테스트를 실행하기 때문에
    반드시 유일한 이름을 사용해야 한다. _repro로 끝나지 않는 test_* 이름은 모두 교체한다.
    """
    def add_suffix(m: re.Match) -> str:
        name = m.group(1)
        if name.endswith("_repro"):
            return m.group(0)
        return m.group(0).replace(name, name + "_repro", 1)

    # def test_xxx(...): 패턴
    result = re.sub(r'\bdef (test_\w+)(?=\s*\()', add_suffix, append_block)
    return result


def _count_generated_tests(code: str) -> int:
    """Count test functions/methods introduced by an append block."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    count = 0
    disabled_functions = _disabled_test_function_names(tree)
    custom_constructor_classes = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name in {"__init__", "__new__"}
            for item in node.body
        )
    }
    class_bases = {
        node.name: {
            base.id for base in node.bases if isinstance(base, ast.Name)
        }
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    changed = True
    while changed:
        changed = False
        for class_name, bases in class_bases.items():
            if class_name not in custom_constructor_classes and bases & custom_constructor_classes:
                custom_constructor_classes.add(class_name)
                changed = True
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test") and node.name not in disabled_functions:
            count += 1
        elif (
            isinstance(node, ast.ClassDef)
            and _is_collectable_test_class(node)
            and node.name not in custom_constructor_classes
        ):
            count += sum(
                1
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name.startswith("test")
            )
    return count


def _has_unsupported_parameterized_test(code: str) -> bool:
    """Return true when a generated test expands beyond one concrete item."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    pytest_aliases = {"pytest"}
    mark_aliases: set[str] = set()
    parametrize_aliases: set[str] = set()
    fixture_aliases: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pytest":
                    pytest_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "pytest":
            for alias in node.names:
                if alias.name == "mark":
                    mark_aliases.add(alias.asname or alias.name)
                elif alias.name == "fixture":
                    fixture_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "pytest.mark":
            for alias in node.names:
                if alias.name == "parametrize":
                    parametrize_aliases.add(alias.asname or alias.name)

    def is_mark_expression(expression: ast.expr) -> bool:
        return (
            isinstance(expression, ast.Name)
            and expression.id in mark_aliases
        ) or (
            isinstance(expression, ast.Attribute)
            and expression.attr == "mark"
            and isinstance(expression.value, ast.Name)
            and expression.value.id in pytest_aliases
        )

    def is_parametrize_expression(expression: ast.expr) -> bool:
        if isinstance(expression, ast.Name):
            return expression.id in parametrize_aliases
        if (
            isinstance(expression, ast.Attribute)
            and expression.attr == "parametrize"
            and is_mark_expression(expression.value)
        ):
            return True
        return (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id == "getattr"
            and len(expression.args) >= 2
            and is_mark_expression(expression.args[0])
            and isinstance(expression.args[1], ast.Constant)
            and expression.args[1].value == "parametrize"
        )

    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if is_mark_expression(value):
            mark_aliases.update(names)
        if is_parametrize_expression(value):
            parametrize_aliases.update(names)
        if "pytestmark" in names and value is not None:
            expressions = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
            if any(
                is_parametrize_expression(
                    expression.func if isinstance(expression, ast.Call) else expression
                )
                for expression in expressions
            ):
                return True

    def is_parametrize(decorator: ast.expr) -> bool:
        expression = decorator.func if isinstance(decorator, ast.Call) else decorator
        return is_parametrize_expression(expression)

    def is_parameterized_fixture(decorator: ast.expr) -> bool:
        if not isinstance(decorator, ast.Call):
            return False
        expression = decorator.func
        is_fixture = (
            isinstance(expression, ast.Name)
            and expression.id in fixture_aliases
        ) or (
            isinstance(expression, ast.Attribute)
            and expression.attr == "fixture"
            and isinstance(expression.value, ast.Name)
            and expression.value.id in pytest_aliases
        )
        return is_fixture and any(keyword.arg == "params" for keyword in decorator.keywords)

    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and (
                (
                    (
                        isinstance(node, ast.ClassDef)
                        or node.name.startswith("test")
                    )
                    and any(is_parametrize(decorator) for decorator in node.decorator_list)
                )
                or any(is_parameterized_fixture(decorator) for decorator in node.decorator_list)
            )
        ):
            return True
    return False


_PYTEST_BUILTIN_FIXTURES = {
    "request", "pytestconfig", "cache", "capsys", "capsysbinary", "capfd",
    "capfdbinary", "doctest_namespace", "monkeypatch", "recwarn", "tmp_path",
    "tmp_path_factory", "tmpdir", "tmpdir_factory", "caplog", "record_property",
    "record_testsuite_property", "record_xml_attribute", "pytester", "testdir",
    "worker_id",
}
_DJANGO_BUILTIN_FIXTURES = {
    "db", "transactional_db", "django_db_setup", "django_db_blocker", "settings",
    "client", "admin_client", "rf", "admin_user", "django_user_model",
    "live_server", "mailoutbox",
}


def _fixture_definitions(code: str) -> tuple[set[str], set[str]]:
    """Return local fixture names and the subset that expand through params."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(), set()
    fixtures: set[str] = set()
    parameterized: set[str] = set()
    pytest_aliases = {"pytest"}
    fixture_aliases: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pytest":
                    pytest_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "pytest":
            for alias in node.names:
                if alias.name == "fixture":
                    fixture_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            expression = node.value
            if (
                isinstance(expression, ast.Attribute)
                and expression.attr == "fixture"
                and isinstance(expression.value, ast.Name)
                and expression.value.id in pytest_aliases
            ):
                fixture_aliases.update(
                    target.id for target in node.targets if isinstance(target, ast.Name)
                )
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            expression = decorator.func if isinstance(decorator, ast.Call) else decorator
            is_fixture = (
                isinstance(expression, ast.Name) and expression.id in fixture_aliases
            ) or (
                isinstance(expression, ast.Attribute)
                and expression.attr == "fixture"
                and isinstance(expression.value, ast.Name)
                and expression.value.id in pytest_aliases
            )
            if not is_fixture:
                continue
            fixtures.add(node.name)
            if isinstance(decorator, ast.Call) and any(
                keyword.arg == "params" for keyword in decorator.keywords
            ):
                parameterized.add(node.name)
    return fixtures, parameterized


def _fixture_parameter_errors(
    code: str,
    *,
    original_content: str,
    context: Mapping[str, Any],
    target_test_file: str = "",
) -> List[str]:
    """Fail closed when generated pytest arguments lack grounded fixture evidence."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    local_fixtures, local_parameterized = _fixture_definitions(code)
    original_fixtures, original_parameterized = _fixture_definitions(original_content)
    parameterized = local_parameterized | original_parameterized
    inventory = set(_PYTEST_BUILTIN_FIXTURES) | set(_DJANGO_BUILTIN_FIXTURES)
    inventory.update(local_fixtures)
    inventory.update(original_fixtures)
    try:
        original_tree = ast.parse(original_content)
    except SyntaxError:
        original_tree = ast.Module(body=[], type_ignores=[])
    parameter_names: set[str] = set()
    for node in ast.walk(original_tree):
        decorators = getattr(node, "decorator_list", [])
        for decorator in decorators:
            if not isinstance(decorator, ast.Call) or not decorator.args:
                continue
            decorator_name = _ast_dotted_name(decorator.func)
            if not decorator_name.endswith("parametrize"):
                continue
            raw_names = decorator.args[0]
            if isinstance(raw_names, ast.Constant) and isinstance(raw_names.value, str):
                parameter_names.update(
                    name.strip() for name in raw_names.value.split(",") if name.strip()
                )
            elif isinstance(raw_names, (ast.List, ast.Tuple)):
                parameter_names.update(
                    str(item.value).strip()
                    for item in raw_names.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                )
    for node in ast.walk(original_tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            positional = [*node.args.posonlyargs, *node.args.args]
            defaulted = {
                argument.arg
                for argument in positional[len(positional) - len(node.args.defaults):]
            } if node.args.defaults else set()
            defaulted.update(
                argument.arg
                for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                if default is not None
            )
            inventory.update(
                argument.arg
                for argument in (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                )
                if argument.arg not in {"self", "cls"}
                and argument.arg not in parameter_names
                and argument.arg not in defaulted
            )
    conftest = context.get("conftest_fixtures")
    if isinstance(conftest, Mapping):
        target_dir = Path(target_test_file).parent
        applicable_dirs = {target_dir, *target_dir.parents, Path(".")}
        for conftest_path, values in conftest.items():
            conftest_dir = Path(str(conftest_path)).parent
            if target_test_file and conftest_dir not in applicable_dirs:
                continue
            if isinstance(values, Mapping):
                inventory.update(str(name) for name in values)
            elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                inventory.update(str(name) for name in values)
    repo_path = Path(str(context.get("repo_path") or ""))
    if target_test_file and repo_path:
        target_dir = Path(target_test_file).parent
        for directory in (target_dir, *target_dir.parents, Path(".")):
            conftest_path = repo_path / directory / "conftest.py"
            if not conftest_path.is_file():
                continue
            try:
                conftest_source = conftest_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            fixtures, expanding_fixtures = _fixture_definitions(conftest_source)
            inventory.update(fixtures)
            parameterized.update(expanding_fixtures)
    errors: List[str] = []
    for node in tree.body:
        functions: List[ast.FunctionDef | ast.AsyncFunctionDef] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            functions = [node]
        elif isinstance(node, ast.ClassDef) and _is_collectable_test_class(node):
            functions = [
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name.startswith("test")
            ]
        for function in functions:
            arguments = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
            positional = [*function.args.posonlyargs, *function.args.args]
            defaulted = {
                argument.arg
                for argument in positional[len(positional) - len(function.args.defaults):]
            } if function.args.defaults else set()
            defaulted.update(
                argument.arg
                for argument, default in zip(function.args.kwonlyargs, function.args.kw_defaults)
                if default is not None
            )
            names = ({argument.arg for argument in arguments} - {"self", "cls"}) - defaulted
            missing = sorted(names - inventory)
            expanding = sorted(names & parameterized)
            if missing:
                errors.append(
                    "UNGROUNDED_FIXTURE: generated test requires unavailable fixture(s): "
                    + ", ".join(missing)
                )
            if expanding:
                errors.append(
                    "PARAMETERIZED_TEST_UNSUPPORTED: fixture params expand the generated test: "
                    + ", ".join(expanding)
                )
    return errors


def _prune_to_best_generated_test(code: str, clue: Optional[Dict[str, Any]] = None) -> str:
    """Keep the strongest generated test when the model emits several.

    This is a pre-patch/static repair only: it uses issue clue text and the
    generated code, never final-eval or post-patch results.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        and getattr(node, "end_lineno", None)
    ]
    if len(candidates) <= 1:
        return code

    clue = clue or {}
    identifiers = clue.get("identifiers", {}) if isinstance(clue.get("identifiers"), dict) else {}
    terms: List[str] = []
    for key in ("functions", "classes", "exceptions", "files"):
        terms.extend(str(x).lower() for x in identifiers.get(key, []) if x)
    terms = [t for t in terms if len(t) >= 3]
    issue_outputs = [
        str(x)
        for field in ("expected_outputs", "actual_outputs")
        for x in clue.get(field, [])[:3]
        if str(x).strip()
    ]

    def score(node: ast.AST) -> float:
        segment = ast.get_source_segment(code, node) or ""
        lower = segment.lower()
        value = 0.0
        value += min(sum(1 for term in terms if term in lower), 8) * 2.0
        segment_candidates = _extract_complete_oracle_output_candidates(segment)
        value += min(
            sum(
                1
                for output in issue_outputs
                if any(strict_normalized_output_equals(candidate, output) for candidate in segment_candidates)
            ),
            4,
        ) * 2.0
        value += len(re.findall(r"\bassert\b|self\.assert|pytest\.raises|assert_(?:allclose|array|equal)", segment)) * 1.5
        if re.search(r"assert(?:True)?\s*\(\s*(?:True|1)\s*\)|^\s*assert\s+(?:True|1)\b", segment, re.MULTILINE):
            value -= 8.0
        if re.search(r"str\s*\(|args\[\d+\]|assert(?:In|NotIn)\s*\(\s*['\"]", segment):
            value -= 1.5
        if re.search(r"is\s+not\s+None|assertIsInstance|len\s*\([^)]*\)\s*>", segment):
            value -= 0.5
        # Earlier tests are usually the primary reproduction, all else equal.
        value -= getattr(node, "lineno", 0) * 0.001
        return value

    best = max(candidates, key=score)
    remove_ranges = {
        line_no
        for node in candidates
        if node is not best
        for line_no in range(node.lineno, node.end_lineno + 1)
    }
    lines = code.splitlines()
    pruned = "\n".join(
        line for idx, line in enumerate(lines, start=1)
        if idx not in remove_ranges
    ).rstrip() + "\n"
    try:
        ast.parse(pruned)
    except SyntaxError:
        return code
    return pruned if _count_generated_tests(pruned) == 1 else code


def _block_text_for_prompt_selection(block: Any) -> str:
    if isinstance(block, dict):
        parts = [
            block.get("context_before", ""),
            block.get("role", ""),
            block.get("label", ""),
            block.get("text", ""),
            block.get("interactive_input", ""),
            block.get("code", ""),
            block.get("interactive_output", ""),
        ]
        return "\n".join(str(part) for part in parts if part)
    return str(block or "")


def _has_reproduction_role_evidence(blocks: List[Any], scenario: Dict[str, Any]) -> bool:
    classified = classify_reproduction_code_blocks(
        blocks,
        expected_outputs=scenario.get("expected_outputs", []) or [],
        actual_outputs=scenario.get("actual_outputs", []) or [],
        target_function=_scenario_target_function(scenario),
    )
    return any(block_has_semantic_role_evidence(block) for block in classified)


def _contains_complex_reproduction_call(text: str) -> bool:
    return bool(
        re.search(r"\b[A-Za-z_]\w*\([^()\n]*\b[A-Za-z_]\w*\s*\(", text)
        or re.search(r"\)\s*\.\s*[A-Za-z_]\w+\s*\(", text)
    )


def _reproduction_block_score(block: Any, scenario: Dict[str, Any]) -> float:
    text = _block_text_for_prompt_selection(block)
    lower = text.lower()
    score = 0.0
    role = block_inferred_role(block)
    if role == ROLE_BUG_TRIGGER:
        score += 20.0
    elif role == ROLE_BASELINE:
        score -= 8.0
    elif role == ROLE_SETUP:
        score -= 1.0
    if isinstance(block, dict):
        role_text = " ".join(
            str(block.get(key, ""))
            for key in ("role", "label", "context_before", "text")
        ).lower()
        if any(marker in role_text for marker in ("bug", "fail", "failing", "problem", "repro")):
            score += 6.0
        if "actual" in role_text or block.get("actual_outputs"):
            score += 5.0
        if "baseline" in role_text or "sanity" in role_text:
            score -= 4.0
        if "setup" in role_text:
            score -= 1.0
        if block.get("interactive_output"):
            score += 2.0

    output_candidates = _reproduction_block_output_candidates(block)
    for output in scenario.get("actual_outputs", []) or []:
        if any(strict_normalized_output_equals(candidate, output) for candidate in output_candidates):
            score += 5.0
            break
    for output in scenario.get("expected_outputs", []) or []:
        if any(strict_normalized_output_equals(candidate, output) for candidate in output_candidates):
            score += 1.0
            break

    call_count = len(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*\(", text))
    if call_count >= 2:
        score += 2.0
    if _contains_complex_reproduction_call(text):
        score += 2.0
    if re.search(r"\b(?:fail|fails|failing|wrong|incorrect|bug|problem|reproduce|regression)\b", lower):
        score += 2.0
    if re.search(r"\b(?:pass|passes|works|baseline|sanity)\b", lower):
        score -= 1.0
    return score


def _reproduction_block_output_candidates(block: Any) -> List[str]:
    candidates: List[str] = []
    if isinstance(block, dict):
        for key in ("interactive_output", "actual_output", "expected_output"):
            candidates.extend(str(value) for value in _coerce_list(block.get(key)) if str(value).strip())
        for key in ("actual_outputs", "expected_outputs"):
            candidates.extend(str(value) for value in _coerce_list(block.get(key)) if str(value).strip())
        for key in ("code", "text"):
            for line in str(block.get(key) or "").splitlines():
                cleaned = re.sub(r"^\s*(?:>>>|\.\.\.)\s?", "", line).strip()
                if cleaned:
                    candidates.append(cleaned)
        return _dedup_text_items(candidates, limit=16)
    return _dedup_text_items([str(block or "")], limit=1)


def _coerce_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _dedup_group_outputs(values: Sequence[Any], *, limit: int = 4) -> List[str]:
    result: List[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _source_index(block: Dict[str, Any], fallback: int) -> int:
    try:
        return int(block.get("source_index", fallback))
    except (TypeError, ValueError):
        return fallback


def _block_input_text(block: Dict[str, Any]) -> str:
    return str(block.get("interactive_input") or block.get("code") or block.get("text") or "").strip()


def _direct_block_outputs(
    block: Dict[str, Any],
    *,
    expected_outputs: Sequence[Any],
    actual_outputs: Sequence[Any],
) -> tuple[List[str], List[str]]:
    candidates = _reproduction_block_output_candidates(block)
    direct_expected: List[str] = []
    direct_actual: List[str] = []
    for value in _coerce_list(block.get("expected_output")) + _coerce_list(block.get("expected_outputs")):
        if str(value).strip():
            direct_expected.append(str(value).strip())
    for value in _coerce_list(block.get("actual_output")) + _coerce_list(block.get("actual_outputs")):
        if str(value).strip():
            direct_actual.append(str(value).strip())
    for expected in expected_outputs:
        if any(strict_normalized_output_equals(candidate, expected) for candidate in candidates):
            direct_expected.append(str(expected))
    for actual in actual_outputs:
        if any(strict_normalized_output_equals(candidate, actual) for candidate in candidates):
            direct_actual.append(str(actual))
    return _dedup_group_outputs(direct_expected), _dedup_group_outputs(direct_actual)


def _shape_from_output(value: Any) -> tuple[int, ...] | None:
    return shape_from_output(value)


def _outputs_structurally_compatible(
    selected_actual_outputs: Sequence[str],
    expected_output: Any,
) -> bool:
    return outputs_structurally_compatible(selected_actual_outputs, expected_output)


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
        actual.extend(str(value) for value in actual_outputs if strict_normalized_output_equals(block.get("code") or block.get("text"), value))
    elif role == ROLE_EXPECTED_OUTPUT:
        expected.extend(str(value) for value in expected_outputs if strict_normalized_output_equals(block.get("code") or block.get("text"), value))
    return _dedup_group_outputs(expected), _dedup_group_outputs(actual)


def _build_reproduction_example_groups(
    blocks: List[Any],
    scenario: Dict[str, Any],
) -> List[ReproductionExampleGroup]:
    return build_reproduction_example_groups(blocks, scenario)


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


def _trigger_group_rank(group: ReproductionExampleGroup) -> tuple[int, int, int, int, int, int, int, int]:
    return trigger_group_rank(group)


def _select_reproduction_example_group(
    blocks: List[Any],
    scenario: Dict[str, Any],
) -> ReproductionExampleGroup | None:
    return select_reproduction_example_group(blocks, scenario)


def _is_setup_reproduction_block(block: Any) -> bool:
    return block_inferred_role(block) == ROLE_SETUP or is_setup_only_block(block)


def _scenario_target_function(scenario: Dict[str, Any]) -> str:
    target = scenario.get("target_location") if isinstance(scenario.get("target_location"), dict) else {}
    return str(scenario.get("target_function") or target.get("target_function") or "")


def _select_reproduction_blocks_for_prompt(
    blocks: List[Any],
    scenario: Dict[str, Any],
    *,
    max_blocks: int = _PROMPT_CODE_EXAMPLES_MAX,
) -> List[Any]:
    """Choose bounded reproduction blocks without assuming raw list order.

    When role/evidence metadata is absent, preserve the legacy first-N behavior.
    """
    if not blocks:
        return []
    if not _has_reproduction_role_evidence(blocks, scenario):
        return blocks[:max_blocks]
    selected_group = _select_reproduction_example_group(blocks, scenario)
    if selected_group is None:
        return blocks[:max_blocks]

    selected: List[Any] = []
    for prior in selected_group.setup_blocks:
        if len(selected) >= max_blocks - 1:
            break
        if _is_setup_reproduction_block(prior):
            selected.append(prior)
    selected.append(selected_group.stimulus_block)
    if len(selected) < max_blocks:
        groups = _build_reproduction_example_groups(blocks, scenario)
        for group in sorted(groups, key=_trigger_group_rank, reverse=True):
            block = group.stimulus_block
            if block in selected or group.role != ROLE_BUG_TRIGGER:
                continue
            selected.append(block)
            if len(selected) >= max_blocks:
                break
    return selected[:max_blocks]


def _scenario_for_selected_reproduction_example(scenario: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(scenario)
    repro = result.get("reproduction_code")
    if not isinstance(repro, list) or not _has_reproduction_role_evidence(repro, result):
        return result
    selected_group = _select_reproduction_example_group(repro, result)
    if selected_group is None:
        return result

    result["reproduction_code"] = selected_group.blocks
    result["expected_outputs"] = selected_group.expected_outputs
    result["actual_outputs"] = selected_group.actual_outputs
    result["selected_reproduction_example"] = selected_group.metadata()
    result["selected_example_id"] = selected_group.selected_example_id
    if selected_group.oracle_requires_regeneration:
        result["oracle_requires_regeneration"] = True
        result["oracle_pairing_status"] = selected_group.oracle_pairing_status
        result.pop("oracle_expected", None)
        contract = result.get("oracle_contract")
        if isinstance(contract, dict):
            updated_contract = dict(contract)
            updated_contract["oracle_source"] = "requires_regeneration"
            updated_contract["rule"] = (
                "No safely associated expected output is available for the selected reproduction stimulus. "
                "Regenerate an EB-grounded oracle or switch scenario; do not borrow an unrelated baseline output."
            )
            result["oracle_contract"] = updated_contract
    elif selected_group.expected_outputs:
        result["oracle_expected"] = selected_group.expected_outputs[0]
    return sanitize_oracle_regeneration_payload(result)


def _attach_relational_oracle_candidate(
    scenario: Dict[str, Any],
    clue: Optional[Dict[str, Any]],
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    candidate = build_issue_supported_relational_oracle_candidate(scenario, clue, context)
    if candidate is None:
        return scenario
    result = dict(scenario)
    result["relational_oracle_candidate"] = candidate.to_dict()
    result["oracle_type"] = "relational"
    contract = result.get("oracle_contract")
    if isinstance(contract, dict):
        updated = dict(contract)
    else:
        updated = {}
    updated["oracle_type"] = "relational"
    updated["oracle_source"] = "requires_regeneration"
    updated["rule"] = (
        "No directly paired expected output is available. Use the repository-provided "
        "issue-supported relational oracle candidate and compare both independently "
        "constructed target results; do not invent a literal expected value."
    )
    result["oracle_contract"] = updated
    return result


def _truncate_scenario_for_prompt(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """시나리오 JSON을 프롬프트에 포함하기 전에 긴 필드를 잘라낸다.

    actual_outputs / expected_outputs / reproduction_code 등이 길면 토큰 초과 원인이 된다.
    """
    result = _scenario_for_selected_reproduction_example(scenario)
    # These fields have dedicated compact sections. Keeping them in the
    # scenario JSON duplicated entire M2/M7 artifacts and stale history.
    for diagnostic_field in (
        "feedback_consumed",
        "m7_diagnosis",
        "previous_pass_avoid_evidence",
        "previous_pass_runtime_evidence",
        "negative_memory",
        "verified_target_evidence",
        "repair_directive",
        "v31_generation_contract",
        "localization_hypotheses",
    ):
        result.pop(diagnostic_field, None)
    for field in ("actual_outputs", "expected_outputs"):
        items = result.get(field)
        if isinstance(items, list):
            truncated = []
            for item in items[:2]:
                s = item if isinstance(item, str) else str(item)
                truncated.append(s[:300] + "…" if len(s) > 300 else s)
            result[field] = truncated
    # reproduction_code: code 필드만 300자로 제한
    repro = result.get("reproduction_code")
    if isinstance(repro, list):
        selected_repro = _select_reproduction_blocks_for_prompt(repro, result)
        result["reproduction_code"] = [
            (
                {**b, "code": b["code"][:300] + "…"}
                if isinstance(b, dict) and isinstance(b.get("code"), str) and len(b["code"]) > 300
                else b
            )
            for b in selected_repro
        ]
    contract = result.get("oracle_contract")
    if isinstance(contract, dict):
        result["oracle_contract"] = {
            k: contract.get(k, "")
            for k in ("oracle_type", "oracle_source", "rule")
            if contract.get(k)
        }
    return result


def _fix_append_block_imports(
    parsed: Dict[str, Any],
    repo_path: Optional[str] = None,
) -> Dict[str, Any]:
    """append_block에서 사용 중인데 import가 누락된 stdlib 모듈을 자동 주입한다.

    - append_block이 없으면 no-op.
    Import path validity is handled by _validate_generated_code(); this helper
    only injects safe missing stdlib/Django root imports.
    """
    block = parsed.get("append_block", "")
    if not block:
        return parsed

    # append_block의 기존 import 라인 수집
    existing = set()
    lines = block.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            existing.add(stripped)

    # stdlib 누락 import 감지 및 주입
    to_inject = []
    for pattern, import_stmt in _AUTO_INJECT_IMPORTS:
        if pattern in block:
            # 이미 import 있는지 확인 (import_stmt의 모듈명으로)
            module = import_stmt.split()[-1].split(".")[0]
            already = any(
                (f"import {module}" in ex or f"from {module}" in ex)
                for ex in existing
            )
            if not already and import_stmt not in to_inject:
                to_inject.append(import_stmt)

    if "django.test." in block and not any(
        "import django" in ex or "from django import" in ex
        for ex in existing
    ):
        to_inject.insert(0, "import django")

    if to_inject:
        new_block = "\n".join(to_inject) + ("\n" if to_inject else "") + block
        parsed = dict(parsed)
        parsed["append_block"] = new_block
        parsed["test_code"] = new_block
        if to_inject:
            logger.info("Auto-injected imports into append_block: %s", to_inject)

    return parsed


def _fix_append_block_repo_imports(
    parsed: Dict[str, Any],
    repo_path: str,
    context: Dict[str, Any],
    import_checker,
) -> Dict[str, Any]:
    """Rewrite import lines when the static checker can prove a better path.

    This handles cases like `from astropy.table import Table` where the repo
    context says the importable symbol is exposed from `astropy` instead.  Hard
    failures still go through validation and become retry feedback.
    """
    block = parsed.get("append_block", "")
    if not block:
        return parsed

    repo = Path(repo_path)
    available_imports = (context or {}).get("available_imports", {})
    replacements: Dict[str, str] = {}
    for import_line in _extract_top_level_import_lines(block):
        check = _coerce_import_check_result(
            import_checker(import_line, repo, available_imports, context)
        )
        if check.is_correctable and check.corrected.strip() != import_line:
            replacements[import_line] = check.corrected.strip()

    if not replacements:
        return parsed

    new_block = block
    for old, new in replacements.items():
        new_block = new_block.replace(old, new, 1)

    parsed = dict(parsed)
    parsed["append_block"] = new_block
    parsed["test_code"] = new_block
    logger.info("Auto-corrected append_block imports: %s", replacements)
    return parsed


def _fix_append_block_imports_against_file(
    parsed: Dict[str, Any],
    file_content: str,
) -> Dict[str, Any]:
    """실제 삽입될 파일의 기존 imports를 기준으로 append_block의 누락 import를 추가 주입한다.

    LLM이 hint와 다른 파일을 선택했을 때, 해당 파일에 없는 모듈이 append_block에서
    사용되면 NameError가 발생한다. 이를 방지한다.
    """
    block = parsed.get("append_block", "")
    if not block:
        return parsed

    # 실제 파일의 기존 import 수집
    file_imports: set = set()
    for line in file_content.splitlines():
        stripped = line.strip()
        if (stripped.startswith("import ") or stripped.startswith("from ")) and not line.startswith((" ", "\t")):
            file_imports.add(stripped)

    # append_block의 import 수집
    block_imports: set = set()
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            block_imports.add(stripped)

    # 파일 + block에 있는 imports 합집합
    all_available = file_imports | block_imports

    # stdlib 자동 주입: block에서 사용되는데 어디에도 없는 것
    to_inject = []
    for pattern, import_stmt in _AUTO_INJECT_IMPORTS:
        if pattern in block:
            module = import_stmt.split()[-1].split(".")[0]
            already = any(
                (f"import {module}" in ex or f"from {module}" in ex)
                for ex in all_available
            )
            if not already and import_stmt not in to_inject:
                to_inject.append(import_stmt)

    if to_inject:
        new_block = "\n".join(to_inject) + "\n" + block
        parsed = dict(parsed)
        parsed["append_block"] = new_block
        parsed["test_code"] = new_block
        logger.info("File-aware import injection into append_block: %s", to_inject)

    return parsed


def _generated_code_from_parsed(parsed: Mapping[str, Any] | None) -> str:
    if not isinstance(parsed, Mapping):
        return ""
    return str(parsed.get("append_block") or parsed.get("test_code") or "")


def _scenario_target_location(scenario: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Return a mapping target location without dereferencing malformed input."""
    if not isinstance(scenario, Mapping):
        return {}
    target = scenario.get("target_location")
    return dict(target) if isinstance(target, Mapping) else {}


def _validate_regeneration_oracle_gate(
    parsed: Dict[str, Any],
    *,
    scenario: Dict[str, Any],
    clue: Optional[Dict[str, Any]],
    context: Optional[Dict[str, Any]],
) -> RelationalOracleValidation:
    if not selected_example_requires_oracle_regeneration(scenario):
        return RelationalOracleValidation(
            True,
            candidate=None,
            reasons=["oracle_regeneration_not_required"],
        )
    candidate = build_issue_supported_relational_oracle_candidate(scenario, clue, context)
    validation = validate_relational_oracle_candidate(_generated_code_from_parsed(parsed), candidate)
    if validation.is_valid:
        return validation
    reasons = list(validation.reasons)
    if candidate is None:
        reasons = [
            "no_direct_expected_output_and_no_issue_supported_relational_oracle_candidate",
        ]
    return RelationalOracleValidation(
        False,
        candidate=candidate,
        reasons=reasons,
    )


class ReproductionTestGenerator:

    SYSTEM_PROMPT = "You are a careful software test generation assistant. Return JSON only."

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        max_retries: int = 3,
        model_key: str = "qwen",
        feature_flags: Optional[V22FeatureFlags] = None,
        feature_profile: str | None = None,
    ) -> None:
        self.client = client or LLMClient(load_model_config(model_key))
        self.max_retries = max_retries
        self.feature_flags = feature_flags or core_only_feature_flags()
        self.feature_profile = feature_profile

    def generate(
        self,
        instance: Any,
        clue: Dict[str, Any],
        context: Dict[str, Any],
        validation_report: Dict[str, Any],
        iteration: int = 1,
        runtime_error_hint: Optional[str] = None,
    ) -> GeneratedReproductionTest:
        generation_started_at = time.monotonic()
        # A3: normalize optional producer fields once at the M5 boundary.
        # ``None`` is a typed contract failure, never a reason to dereference
        # nested mappings or fabricate repository context.
        if not isinstance(clue, dict):
            raise GenerationFailureError(
                message="M5 clue contract is unavailable or malformed",
                token_usage={}, attempt_count=0, last_error="clue_not_mapping",
                failure_type_detail="M5_INPUT_CONTRACT", token_usage_status="no_llm_call",
            )
        if not isinstance(context, dict):
            raise GenerationFailureError(
                message="M5 context contract is unavailable or malformed",
                token_usage={}, attempt_count=0, last_error="context_not_mapping",
                failure_type_detail="M5_INPUT_CONTRACT", token_usage_status="no_llm_call",
            )
        if not isinstance(validation_report, dict):
            raise GenerationFailureError(
                message="M5 validation report contract is unavailable or malformed",
                token_usage={}, attempt_count=0, last_error="validation_report_not_mapping",
                failure_type_detail="M5_INPUT_CONTRACT", token_usage_status="no_llm_call",
            )
        expected_identity = {
            "instance_id": str(getattr(instance, "instance_id", "") or ""),
            "repo": str(getattr(instance, "repo", "") or ""),
            "base_commit": str(getattr(instance, "base_commit", "") or ""),
        }
        for owner_name, payload in (
            ("clue", clue),
            ("context", context),
            ("validation_report", validation_report),
        ):
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
            for key, expected in expected_identity.items():
                observed = payload.get(key, metadata.get(key))
                if observed not in (None, "") and expected and str(observed) != expected:
                    raise GenerationFailureError(
                        message=f"M5 {owner_name} {key} does not match the active instance",
                        token_usage={},
                        attempt_count=0,
                        last_error=f"cross_stage_{key}_mismatch",
                        failure_type_detail="M5_INPUT_IDENTITY",
                        token_usage_status="no_llm_call",
                    )
        if str(context.get("feature_profile") or context.get("methodology_revision") or "") == "v31":
            for owner_name, payload in (("clue", clue), ("context", context)):
                metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
                observed = payload.get("instance_id", metadata.get("instance_id"))
                if not observed:
                    raise GenerationFailureError(
                        message=f"M5 {owner_name} instance_id is required for v31",
                        token_usage={}, attempt_count=0, last_error="missing_cross_stage_instance_id",
                        failure_type_detail="M5_INPUT_IDENTITY", token_usage_status="no_llm_call",
                    )
        for selected in validation_report.get("selected_scenarios", []) or []:
            if not isinstance(selected, Mapping):
                continue
            normalized = selected.get("normalized_scenario")
            normalized = normalized if isinstance(normalized, Mapping) else selected
            nested_instance = normalized.get("instance_id")
            if nested_instance and str(nested_instance) != expected_identity["instance_id"]:
                raise GenerationFailureError(
                    message="M5 selected scenario instance_id does not match the active instance",
                    token_usage={}, attempt_count=0, last_error="cross_stage_instance_id_mismatch",
                    failure_type_detail="M5_INPUT_IDENTITY", token_usage_status="no_llm_call",
                )
        context["project_test_style"] = (
            dict(context.get("project_test_style"))
            if isinstance(context.get("project_test_style"), Mapping)
            else {}
        )
        context["candidate_test_files"] = (
            list(context.get("candidate_test_files"))
            if isinstance(context.get("candidate_test_files"), list)
            else []
        )
        context["available_imports"] = (
            dict(context.get("available_imports"))
            if isinstance(context.get("available_imports"), Mapping)
            else {}
        )
        context["conftest_fixtures"] = (
            dict(context.get("conftest_fixtures"))
            if isinstance(context.get("conftest_fixtures"), Mapping)
            else {}
        )
        if getattr(self, "feature_profile", None) == "v37":
            scenario = select_primary_scenario(validation_report, clue=clue, context=context)
            return self._generate_v37_single(
                instance=instance,
                clue=clue,
                context=context,
                scenario=scenario,
                iteration=iteration,
                runtime_error_hint=runtime_error_hint,
            )
        model_request_elapsed_sec = 0.0
        postprocess_started_at: float | None = None
        scenario = select_primary_scenario(validation_report, clue=clue, context=context)
        scenario = _retarget_scenario_from_explicit_issue_evidence(
            scenario,
            clue,
            context,
        )

        repo_path = context.get("repo_path", "")
        if not repo_path:
            raise ValueError("context.json is missing repo_path.")

        runner = str(context["project_test_style"].get("runner") or "pytest")
        verified_target = _verified_target_evidence(repo_path, clue, context, scenario)
        scenario = _apply_verified_target_to_scenario(scenario, verified_target)
        context = dict(context)
        context["verified_target_evidence"] = verified_target

        target_location = scenario.get("target_location")
        if not isinstance(target_location, Mapping):
            raise GenerationFailureError(
                message="M5 scenario target contract is unavailable",
                token_usage={}, attempt_count=0, last_error="target_location_not_mapping",
                failure_type_detail="M5_INPUT_CONTRACT", token_usage_status="no_llm_call",
                scenario=dict(scenario),
            )
        target_location = dict(target_location)
        target_test_file_hint = target_location.get("candidate_test_file") or (
            scenario.get("relevant_test_files", [""])[0] if scenario.get("relevant_test_files") else ""
        )
        hint_path = Path(str(target_test_file_hint or ""))
        if hint_path.is_absolute() or ".." in hint_path.parts:
            raise GenerationFailureError(
                message="M5 target test file must remain repository-relative",
                token_usage={}, attempt_count=0, last_error="target_test_file_outside_repository",
                failure_type_detail="M5_INPUT_IDENTITY", token_usage_status="no_llm_call",
            )
        repaired_target_test_file = _select_existing_target_test_file(
            repo_path,
            target_test_file_hint,
            context,
            scenario,
        )
        if repaired_target_test_file and repaired_target_test_file != target_test_file_hint:
            scenario = copy.deepcopy(scenario)
            target_location = dict(scenario.get("target_location") or {})
            target_location["candidate_test_file"] = repaired_target_test_file
            scenario["target_location"] = target_location
            scenario["relevant_test_files"] = [repaired_target_test_file] + [
                p for p in scenario.get("relevant_test_files", []) if p != repaired_target_test_file
            ]
            target_test_file_hint = repaired_target_test_file
        if not target_test_file_hint:
            raise FileNotFoundError(
                "No verified target test file is available for M5 generation; "
                "invalid M7/test-file reroute was rejected."
            )
        scenario = _scenario_for_selected_reproduction_example(scenario)
        scenario = _attach_relational_oracle_candidate(scenario, clue, context)

        # 대상 테스트 파일의 기존 import 블록 + 실제 test 메서드 예시 추출 (프롬프트용)
        existing_test_imports = ""
        target_test_example = ""
        if target_test_file_hint:
            test_file_abs = Path(repo_path) / target_test_file_hint
            if test_file_abs.exists():
                existing_test_imports = self._extract_import_block(test_file_abs)
                preferred_example_terms = [
                    str(_scenario_target_location(scenario).get("target_function") or "").split(".")[-1],
                    *[
                        str(item)
                        for values in (clue.get("identifiers") or {}).values()
                        for item in (
                            values
                            if isinstance(values, Sequence) and not isinstance(values, (str, bytes))
                            else [values]
                        )
                    ],
                ] if isinstance(clue.get("identifiers"), Mapping) else [
                    str(_scenario_target_location(scenario).get("target_function") or "").split(".")[-1]
                ]
                target_test_example = self._extract_test_examples_from_file(
                    str(test_file_abs),
                    preferred_terms=preferred_example_terms,
                )
                if runner == "django-test":
                    context["django_repository_grounding"] = _django_repository_grounding(
                        repo=Path(repo_path),
                        target_test_file=str(target_test_file_hint),
                        original_content=read_text(test_file_abs),
                    )

        v31_enabled = str(context.get("feature_profile") or context.get("methodology_revision") or "") == "v31"
        v31_contract: TestGenerationContractV31 | None = None
        if v31_enabled:
            v31_contract = _build_v31_generation_contract(
                scenario=scenario,
                clue=clue,
                context=context,
                target_test_file=target_test_file_hint,
                target_source_file=str(_scenario_target_location(scenario).get("source_file") or ""),
                runner=runner,
                existing_test_imports=existing_test_imports,
                target_test_example=target_test_example,
            )
            context["v31_generation_contract"] = v31_contract.to_dict()

        prompt_profile: Dict[str, Any] = {
            "budget_mode": "compact",
            "sections_included": [],
            "truncated_sections": [],
            "retry_prompt_chars": [],
        }
        prompt = self._build_prompt(
            instance=instance,
            clue=clue,
            context=context,
            scenario=scenario,
            existing_test_imports=existing_test_imports,
            target_test_example=target_test_example,
            runtime_error_hint=runtime_error_hint,
            prompt_profile=prompt_profile,
        )
        prompt_profile["prompt_chars"] = len(prompt)
        prompt_profile["initial_prompt_chars"] = len(prompt)

        last_error_msg = ""
        last_raw_response = ""
        last_parsed = None
        last_validation_errors: List[str] = []
        repair_actions_accum: List[str] = []
        repair_retry_count = 0
        soft_retry_count = 0
        retry_required_oracle_risks: List[str] = []
        semantic_risk_flags: List[str] = []
        # 이 generate() 호출에서 누적된 토큰 사용량
        accumulated_tokens: Dict[str, int] = _empty_token_usage()
        llm_call_count = 0
        usage_missing = False
        attempt_history: List[Dict[str, Any]] = []
        seen_prompt_hashes: set[str] = set()
        seen_candidate_error_pairs: set[str] = set()

        # V26 permits exactly one oracle-first M5 model call per outer pass.
        # Validation failures are routed by M7; M5 does not run an internal
        # prompt/temperature refinement loop.
        max_attempts = 1
        validation_passed = False
        stable_valid_parsed: Optional[Dict[str, Any]] = None
        for attempt in range(max_attempts):
            current_temperature = self.client.config.temperature

            if attempt == 0:
                current_prompt = prompt
            else:
                current_prompt = self._build_fix_prompt(
                    original_prompt=prompt,
                    previous_response=last_raw_response,
                    previous_parsed=last_parsed,
                    error_message=last_error_msg,
                    attempt=attempt,
                    scenario=scenario,
                    context=context,
                )
                prompt_profile.setdefault("retry_prompt_chars", []).append(len(current_prompt))
            prompt_hash = sha256_text(current_prompt)
            duplicate_prompt = prompt_hash in seen_prompt_hashes and attempt > 0
            seen_prompt_hashes.add(prompt_hash)

            try:
                request_started_at = time.monotonic()
                raw_response = self.client.generate(
                    current_prompt,
                    system_prompt=self.SYSTEM_PROMPT,
                    temperature=current_temperature,
                    prompt_compactor=compact_m5_prompt,
                )
                model_request_elapsed_sec += time.monotonic() - request_started_at
                llm_call_count += 1
                if _accumulate_token_usage(accumulated_tokens, getattr(self.client, "last_usage", None)) == "unknown":
                    usage_missing = True
            except Exception as e:
                last_error_msg = f"LLM call failed: {e}"
                last_raw_response = ""
                attempt_history.append({
                    "attempt": attempt + 1,
                    "prompt_hash": prompt_hash,
                    "candidate_sha256": "",
                    "validation_errors": [last_error_msg],
                    "rejected_patterns": [],
                    "duplicate_prompt_blocked": duplicate_prompt,
                    "duplicate_candidate_blocked": False,
                })
                logger.warning("[attempt %d/%d] %s", attempt + 1, max_attempts, last_error_msg)
                continue

            last_raw_response = raw_response
            postprocess_started_at = time.monotonic()

            try:
                parsed = self._parse_model_output(raw_response, scenario, context)
            except (ValueError, TypeError, AttributeError) as e:
                last_error_msg = f"Model output parsing failed: {e}"
                attempt_history.append({
                    "attempt": attempt + 1,
                    "prompt_hash": prompt_hash,
                    "candidate_sha256": "",
                    "validation_errors": [last_error_msg],
                    "rejected_patterns": [],
                    "duplicate_prompt_blocked": duplicate_prompt,
                    "duplicate_candidate_blocked": False,
                })
                logger.warning("[attempt %d/%d] %s", attempt + 1, max_attempts, last_error_msg)
                continue
            parsed = _repair_target_test_file_selection(parsed, repo_path, context, scenario)

            if getattr(self, "feature_profile", None) != "v37":
                for key in ("append_block", "test_code"):
                    if parsed.get(key):
                        parsed[key], repair_actions = _apply_oracle_repairs(parsed[key], clue)
                        repair_actions_accum.extend(f"{key}:{a}" for a in repair_actions)

            if "sphinx" in getattr(instance, "repo", "").lower():
                for key in ("append_block", "test_code"):
                    if parsed.get(key):
                        parsed[key] = _fix_sphinx_test_code(parsed[key])

            if runner == "django-test":
                parsed["imports"] = _fix_django_imports(parsed.get("imports", []))
                for key in ("append_block", "test_code"):
                    if parsed.get(key):
                        parsed[key] = _fix_django_test_code(parsed[key])

            # stdlib 누락 import 자동 주입 (검증 전에 수행해야 재시도 시 반영됨)
            parsed = _fix_append_block_imports(parsed, repo_path=repo_path)
            parsed = _fix_append_block_repo_imports(
                parsed,
                repo_path=repo_path,
                context=context,
                import_checker=self._check_import_validity,
            )

            # v31's manifest is bounded provenance, not a closed-world import
            # allow-list.  Run the normal deterministic import repair first,
            # then reject only imports proven invalid by the pre-patch checker.
            # Oracle and syntax checks remain centralized in
            # _validate_generated_code so M5-A receives every repairable
            # candidate instead of being bypassed by a duplicate gate.
            contract_errors: List[str] = []

            verified = context.get("verified_target_evidence") if isinstance(context, dict) else {}
            target_function = self._required_generation_invocation(
                scenario=scenario,
                verified_target=verified,
            )
            target_missing = (
                target_function
                and not target_function.startswith("_")
                and len(target_function) >= 3
                and target_function not in {"path", "main", "run", "get", "set"}
                and not any(
                    parsed.get(key) and self._check_target_function_presence(target_function, parsed[key])
                    for key in ("append_block", "test_code")
                )
            )
            if target_missing and attempt < max_attempts - 1:
                last_parsed = parsed
                last_validation_errors = [
                    "CRITICAL: semantic risk: target_function_public_api_rewrite"
                ]
                repair_retry_count += 1
                last_error_msg = (
                    "CRITICAL: semantic risk: target_function_public_api_rewrite. "
                    "Rewrite the test to call the target function from the scenario, "
                    "or a public wrapper that visibly exercises that target behavior."
                )
                logger.warning("[attempt %d/%d] validation failed: %s", attempt + 1, max_attempts, last_error_msg)
                continue

            private_read_keys = [
                key
                for key in ("append_block", "test_code")
                if parsed.get(key) and _has_private_attr_read(parsed[key])
            ]
            if private_read_keys and attempt < max_attempts - 1:
                last_parsed = parsed
                last_validation_errors = [
                    "CRITICAL: semantic risk: private_attribute_public_api_rewrite"
                ]
                repair_retry_count += 1
                last_error_msg = (
                    "CRITICAL: semantic risk: private_attribute_public_api_rewrite. "
                    "Rewrite assertions to inspect public API return values, public state, "
                    "artist/axis/legend accessors, or issue-visible behavior instead of _private attributes."
                )
                logger.warning("[attempt %d/%d] validation failed: %s", attempt + 1, max_attempts, last_error_msg)
                continue

            # 사전 검증
            validation = self._validate_generated_code(
                parsed=parsed,
                repo_path=repo_path,
                context=context,
                clue=clue,
                scenario=scenario,
            )

            # Apply the v31 contract only after complete syntax/import/oracle
            # validation.  This preserves the M5 -> M5-A handoff for every
            # invalid candidate; the contract augments normal diagnostics and
            # never short-circuits repair/revalidation.
            if v31_contract is not None:
                contract_errors = _v31_contract_errors(
                    parsed,
                    v31_contract,
                    repo=Path(repo_path),
                    available_imports=context.get("available_imports", {}),
                    import_checker=self._check_import_validity,
                    import_context=context,
                )
                if contract_errors:
                    validation = ValidationResult(
                        is_valid=False,
                        errors=list(validation.errors) + contract_errors,
                        warnings=list(validation.warnings),
                        fixed_imports=validation.fixed_imports,
                    )

            if validation.fixed_imports is not None:
                parsed["imports"] = validation.fixed_imports

            if validation.is_valid:
                relational_validation = _validate_regeneration_oracle_gate(
                    parsed,
                    scenario=scenario,
                    clue=clue,
                    context=context,
                )
                if getattr(self, "feature_profile", None) != "v37" and not relational_validation.is_valid:
                    last_parsed = parsed
                    last_validation_errors = [
                        "CRITICAL: relational oracle regeneration gate: " + reason
                        for reason in relational_validation.reasons
                    ]
                    repair_retry_count += 1
                    last_error_msg = "; ".join(last_validation_errors)
                    logger.warning("[attempt %d/%d] validation failed: %s", attempt + 1, max_attempts, last_error_msg)
                    if attempt < max_attempts - 1:
                        continue
                    validation_passed = False
                    break
                if getattr(self, "feature_profile", None) != "v37" and relational_validation.candidate is not None:
                    parsed["relational_oracle"] = relational_validation.to_metadata()
                    parsed["oracle_source"] = RELATIONAL_ORACLE_PROVENANCE
                last_parsed = parsed
                stable_valid_parsed = copy.deepcopy(parsed)
                if validation.warnings:
                    prompt_profile.setdefault("soft_validation_warnings", []).extend(validation.warnings)
                    repair_actions_accum.extend(
                        f"soft_validation_warning:{warning[:120]}"
                        for warning in validation.warnings
                    )
                if (
                    validation.warnings
                    and soft_retry_count < 1
                    and attempt < max_attempts - 1
                    and any("issue_reproduction_code_not_followed" in w for w in validation.warnings)
                ):
                    soft_retry_count += 1
                    repair_retry_count += 1
                    last_validation_errors = list(validation.warnings)
                    last_error_msg = "; ".join(validation.warnings)
                    logger.warning(
                        "[attempt %d/%d] soft validation warning, retrying once: %s",
                        attempt + 1,
                        max_attempts,
                        last_error_msg,
                    )
                    continue
                last_validation_errors = []
                last_error_msg = ""
                validation_passed = True
                logger.info("[attempt %d/%d] validation passed", attempt + 1, max_attempts)
                break
            else:
                last_parsed = parsed
                last_validation_errors = list(validation.errors)
                candidate_code = _generated_code_from_parsed(parsed)
                candidate_sha = sha256_text(candidate_code)
                rejected_patterns = _network_rejected_patterns(candidate_code)
                pair_hash = sha256_text(json.dumps({
                    "candidate_sha256": candidate_sha,
                    "errors": last_validation_errors,
                    "patterns": rejected_patterns,
                }, sort_keys=True))
                duplicate_candidate = pair_hash in seen_candidate_error_pairs
                seen_candidate_error_pairs.add(pair_hash)
                attempt_history.append({
                    "attempt": attempt + 1,
                    "prompt_hash": prompt_hash,
                    "candidate_sha256": candidate_sha,
                    "validation_errors": last_validation_errors,
                    "rejected_patterns": rejected_patterns,
                    "duplicate_prompt_blocked": duplicate_prompt,
                    "duplicate_candidate_blocked": duplicate_candidate,
                })
                if any("retry required" in e or "semantic risk" in e for e in validation.errors):
                    repair_retry_count += 1
                last_error_msg = "; ".join(validation.errors)
                logger.warning("[attempt %d/%d] validation failed: %s", attempt + 1, max_attempts, last_error_msg)
                if duplicate_candidate and any("real network calls are not allowed" in e for e in validation.errors):
                    last_error_msg = (
                        "CRITICAL: duplicate real-network candidate/error pattern blocked. "
                        + last_error_msg
                    )
                    break

        # Django runner: unittest.TestCase → django.test.TestCase 자동 교정
        if last_parsed is not None and runner == "django-test":
            last_parsed["imports"] = _fix_django_imports(last_parsed.get("imports", []))
            last_parsed["test_code"] = _fix_django_test_code(last_parsed.get("test_code", ""))
            # append_block 방식에도 동일하게 적용
            if last_parsed.get("append_block"):
                last_parsed["append_block"] = _fix_django_test_code(last_parsed["append_block"])

        # sphinx: pytest.mark.sphinx 데코레이터 제거 (sphinx.testing.fixtures 의존성 회피)
        if last_parsed is not None and "sphinx" in getattr(instance, "repo", "").lower():
            if last_parsed.get("append_block"):
                last_parsed["append_block"] = _fix_sphinx_test_code(last_parsed["append_block"])
            if last_parsed.get("test_code"):
                last_parsed["test_code"] = _fix_sphinx_test_code(last_parsed["test_code"])

        # private attribute 직접 설정 제거 (fragile, public API 로직 우회)
        if last_parsed is not None:
            if last_parsed.get("append_block"):
                last_parsed["append_block"] = _fix_private_attr_access(last_parsed["append_block"])
            if last_parsed.get("test_code"):
                last_parsed["test_code"] = _fix_private_attr_access(last_parsed["test_code"])

        # 예외 메시지 exact matching 제거 (버전 의존, after_patch에서 항상 실패)
        if last_parsed is not None:
            for key in ("append_block", "test_code"):
                if last_parsed.get(key):
                    last_parsed[key], repair_actions = _apply_oracle_repairs(last_parsed[key], clue)
                    repair_actions_accum.extend(f"final_{key}:{a}" for a in repair_actions)

        # Tier 2 assertion 주입 (probe actual_outputs가 있을 때만, 코드 레벨 강제)
        if last_parsed is not None:
            actual_outs = (scenario or {}).get("actual_outputs") or []
            if actual_outs:
                for key in ("append_block", "test_code"):
                    if last_parsed.get(key):
                        if not _should_inject_tier2_assertion(last_parsed[key], clue, scenario):
                            repair_actions_accum.append(
                                f"{key}:skip_tier2_assertion_not_structural_or_has_expected_output"
                            )
                            continue
                        injected = _inject_tier2_assertion(
                            last_parsed[key], actual_outs[0]
                        )
                        if _would_violate_repair_directive(injected, scenario):
                            repair_actions_accum.append(
                                f"{key}:skip_tier2_assertion_due_to_repair_directive"
                            )
                        else:
                            last_parsed[key] = injected

        # 모든 retry 소진 후에도 정적/의미 검증을 통과하지 못한 결과는 사용하지 않는다.
        # Invalid tests flowing into alignment create misleading ALIGNED rows and poor final eval.
        if last_parsed is None or not validation_passed:
            raise GenerationFailureError(
                message=(
                    f"Failed to generate a valid test after {max_attempts} attempts. "
                    f"Last error: {last_error_msg}"
                ),
                token_usage=dict(accumulated_tokens),
                attempt_count=llm_call_count,
                last_error=last_error_msg,
                failure_type_detail=_generation_failure_type_detail(last_error_msg),
                token_usage_status=(
                    "no_llm_call"
                    if llm_call_count == 0
                    else "unknown"
                    if usage_missing
                    else "known"
                ),
                raw_candidate=_generated_code_from_parsed(last_parsed),
                raw_response=last_raw_response,
                parsed_candidate=dict(last_parsed or {}),
                validation_errors=list(last_validation_errors or ([last_error_msg] if last_error_msg else [])),
                validation_status=validation_status_from_errors(
                    list(last_validation_errors or ([last_error_msg] if last_error_msg else []))
                ),
                prompt=prompt,
                scenario=dict(scenario or {}),
                attempt_history=list(attempt_history),
            )

        final_validation = self._validate_generated_code(
            parsed=last_parsed,
            repo_path=repo_path,
            context=context,
            clue=clue,
            scenario=scenario,
        )
        if final_validation.fixed_imports is not None:
            last_parsed["imports"] = final_validation.fixed_imports
        if not final_validation.is_valid:
            fallback_validation = None
            if stable_valid_parsed is not None:
                stable_valid_parsed = _repair_target_test_file_selection(
                    stable_valid_parsed,
                    repo_path,
                    context,
                    scenario,
                )
                fallback_validation = self._validate_generated_code(
                    parsed=stable_valid_parsed,
                    repo_path=repo_path,
                    context=context,
                    clue=clue,
                    scenario=scenario,
                )
            if fallback_validation is not None and fallback_validation.is_valid:
                last_parsed = stable_valid_parsed
                repair_actions_accum.append("fallback_to_pre_auto_repair_valid_candidate")
            else:
                message = (
                    "Generated test became invalid after automatic repair: "
                    + "; ".join(final_validation.errors)
                )
                raise GenerationFailureError(
                    message=message,
                    token_usage=dict(accumulated_tokens),
                    attempt_count=llm_call_count,
                    last_error=message,
                    failure_type_detail=_generation_failure_type_detail(message),
                    token_usage_status="unknown" if usage_missing else "known",
                    raw_candidate=_generated_code_from_parsed(last_parsed),
                    raw_response=last_raw_response,
                    parsed_candidate=dict(last_parsed or {}),
                    validation_errors=list(final_validation.errors),
                    validation_status=validation_status_from_errors(list(final_validation.errors)),
                    prompt=prompt,
                    scenario=dict(scenario or {}),
            )

        parsed = last_parsed
        repair_failed_reason = ""
        if getattr(self, "feature_profile", None) != "v37":
            for key in ("append_block", "test_code"):
                if parsed.get(key):
                    remaining_retry_risks = _detect_retry_required_oracle_risks(parsed[key], clue=clue)
                    retry_required_oracle_risks = sorted(set(remaining_retry_risks))
                    if remaining_retry_risks:
                        repair_failed_reason = "retry_required_oracle_risk=" + ",".join(remaining_retry_risks)
                        break
            for key in ("append_block", "test_code"):
                if parsed.get(key):
                    semantic_risk_flags = sorted(set(_detect_semantic_risk_flags(parsed[key], clue, context, scenario)))
                    semantic_risk_flags.extend(
                        flag
                        for flag in _semantic_anchor_violations(parsed[key], clue, scenario)
                        if flag not in semantic_risk_flags
                    )
                    if semantic_risk_flags and not repair_failed_reason:
                        repair_failed_reason = "semantic_risk=" + ",".join(semantic_risk_flags)
                    break
        if not repair_failed_reason and last_validation_errors:
            retry_errors = [e for e in last_validation_errors if "retry required" in e]
            if retry_errors:
                repair_failed_reason = "; ".join(retry_errors[:3])

        retry_chars = prompt_profile.get("retry_prompt_chars", [])
        if retry_chars:
            prompt_profile["max_retry_prompt_chars"] = max(retry_chars)
            prompt_profile["avg_retry_prompt_chars"] = round(sum(retry_chars) / len(retry_chars))
            prompt_profile["retry_to_initial_char_ratio"] = round(
                prompt_profile["max_retry_prompt_chars"] / max(1, prompt_profile.get("initial_prompt_chars", 1)),
                3,
            )

        target_test_file = parsed["target_test_file"]
        parsed = _repair_target_test_file_selection(parsed, repo_path, context, scenario)
        target_test_file = parsed["target_test_file"]

        # ── __init__.py guard: LLM이 __init__.py를 선택한 경우, 시나리오 힌트로 교체 ──
        if target_test_file.endswith("__init__.py") and target_test_file_hint and not target_test_file_hint.endswith("__init__.py"):
            logger.warning(
                "LLM chose __init__.py as test file %s; overriding to %s",
                target_test_file, target_test_file_hint,
            )
            target_test_file = target_test_file_hint
            parsed["target_test_file"] = target_test_file

        # ── skip-guard: LLM이 module-level skip 파일을 선택한 경우, 시나리오 힌트로 교체 ──
        _skip_set = {
            cf["path"]
            for cf in context.get("candidate_test_files", [])
            if cf.get("has_module_skip")
        }
        if target_test_file in _skip_set and target_test_file_hint and target_test_file_hint not in _skip_set:
            logger.warning(
                "LLM chose skip-flagged file %s; overriding to %s",
                target_test_file, target_test_file_hint,
            )
            target_test_file = target_test_file_hint
            parsed["target_test_file"] = target_test_file

        target_test_abspath = Path(repo_path) / target_test_file

        if not target_test_abspath.exists() and not target_test_abspath.parent.exists():
            raise FileNotFoundError(f"Target test file does not exist: {target_test_abspath}")

        original_content = read_text(target_test_abspath) if target_test_abspath.exists() else ""

        # ── base_commit 버전의 파일 내용 가져오기 (patch context 불일치 방지) ──
        # 로컬 파일은 최신 커밋 기준이지만 Docker 컨테이너는 base_commit에서 실행됨.
        # patch를 base_commit 버전 파일 기준으로 생성해야 git apply가 성공함.
        base_commit = getattr(instance, "base_commit", None)
        content_for_patch = original_content  # fallback
        if base_commit and repo_path:
            try:
                r = subprocess.run(
                    ["git", "show", f"{base_commit}:{target_test_file}"],
                    capture_output=True, text=True, cwd=repo_path,
                )
                if r.returncode == 0 and r.stdout.strip():
                    content_for_patch = r.stdout
                    logger.debug(
                        "Using base_commit content for patch: %s@%s",
                        target_test_file, base_commit[:8],
                    )
            except Exception as e:
                logger.warning("git show failed for %s@%s: %s", target_test_file, base_commit, e)

        # ── 실제 선택된 파일 기준 import 보완 ──
        # LLM이 hint와 다른 파일을 선택했을 수 있으므로,
        # 실제 파일의 기존 imports를 확인하여 append_block에 필요한 import를 추가 주입한다.
        parsed = _fix_append_block_imports_against_file(parsed, original_content)
        final_post_file_validation = self._validate_generated_code(
            parsed=parsed,
            repo_path=repo_path,
            context=context,
            clue=clue,
            scenario=scenario,
        )
        if final_post_file_validation.fixed_imports is not None:
            parsed["imports"] = final_post_file_validation.fixed_imports
        if not final_post_file_validation.is_valid:
            message = (
                "Generated test became invalid after file-aware repair: "
                + "; ".join(final_post_file_validation.errors)
            )
            raise GenerationFailureError(
                message=message,
                token_usage=dict(accumulated_tokens),
                attempt_count=llm_call_count,
                last_error=message,
                failure_type_detail=_generation_failure_type_detail(message),
                token_usage_status="unknown" if usage_missing else "known",
                raw_candidate=_generated_code_from_parsed(parsed),
                raw_response=last_raw_response,
                parsed_candidate=dict(parsed or {}),
                validation_errors=list(final_post_file_validation.errors),
                validation_status=validation_status_from_errors(list(final_post_file_validation.errors)),
                prompt=prompt,
                scenario=dict(scenario or {}),
            )
        final_relational_validation = _validate_regeneration_oracle_gate(
            parsed,
            scenario=scenario,
            clue=clue,
            context=context,
        )
        if getattr(self, "feature_profile", None) != "v37" and not final_relational_validation.is_valid:
            message = (
                "Rejected generated test before patch construction: relational oracle regeneration gate failed: "
                + "; ".join(final_relational_validation.reasons)
            )
            raise GenerationFailureError(
                message=message,
                token_usage=dict(accumulated_tokens),
                attempt_count=llm_call_count,
                last_error=message,
                failure_type_detail=_generation_failure_type_detail(message),
                token_usage_status="unknown" if usage_missing else "known",
                raw_candidate=_generated_code_from_parsed(parsed),
                raw_response=last_raw_response,
                parsed_candidate=dict(parsed or {}),
                validation_errors=list(final_relational_validation.reasons),
                validation_status=validation_status_from_errors(list(final_relational_validation.reasons)),
                prompt=prompt,
                scenario=dict(scenario or {}),
            )
        if getattr(self, "feature_profile", None) != "v37" and final_relational_validation.candidate is not None:
            parsed["relational_oracle"] = final_relational_validation.to_metadata()
            parsed["oracle_source"] = RELATIONAL_ORACLE_PROVENANCE

        if parsed.get("insert_mode") == "append_block":
            # 단순 append — base_commit 버전 파일에 붙임
            modified_content = content_for_patch.rstrip() + "\n\n" + parsed["append_block"] + "\n"
        else:
            # 구 방식 (하위 호환)
            modified_content = self._build_modified_test_file_content(
                original_content=content_for_patch,
                imports=parsed["imports"],
                test_code=parsed["test_code"],
            )
        test_patch = self._build_unified_patch(
            original_content=content_for_patch,
            modified_content=modified_content,
            relative_path=target_test_file,
        )
        canonical_test_nodeid = _canonical_test_nodeid_from_generated(
            target_test_file,
            parsed.get("append_block") or parsed.get("test_code") or "",
        )

        prompt_profile["optional_features"] = _m3_m5_optional_feature_metadata(
            self.feature_flags
        )
        selected_reproduction_example = (
            scenario.get("selected_reproduction_example")
            if isinstance(scenario.get("selected_reproduction_example"), dict)
            else {}
        )
        if selected_reproduction_example:
            prompt_profile["selected_reproduction_example"] = dict(selected_reproduction_example)
        relational_oracle = parsed.get("relational_oracle") if isinstance(parsed.get("relational_oracle"), dict) else {}
        if relational_oracle:
            prompt_profile["relational_oracle"] = dict(relational_oracle)
        total_generation_elapsed_sec = time.monotonic() - generation_started_at
        prompt_profile["v26_module_timings"] = {
            "m5_total_elapsed_sec": round(total_generation_elapsed_sec, 3),
            "m5_model_request_elapsed_sec": round(model_request_elapsed_sec, 3),
            "m5a_postprocess_elapsed_sec": round(
                time.monotonic() - postprocess_started_at,
                3,
            ) if postprocess_started_at is not None else None,
            "time_affects_control_flow": False,
            "target_120s_is_telemetry_only": True,
        }
        v31_contract_dict = v31_contract.to_dict() if v31_contract is not None else {}
        v31_trace = {}
        v31_telemetry = {}
        if v31_contract is not None:
            final_code = str(parsed.get("append_block") or parsed.get("test_code") or "")
            trace = OracleTraceV31(
                assertion=next((line.strip() for line in final_code.splitlines() if "assert" in line or "pytest.raises" in line or "self.assert" in line), "oracle assertion"),
                behavior_contract_field="expected_behavior",
                issue_evidence=list(v31_contract.oracle.evidence),
                target_relation=v31_contract.target_symbol,
            )
            v31_trace = trace.to_dict()
            v31_telemetry = M5TelemetryV31(
                generation_attempt_id=f"{instance.instance_id}-it{iteration}-a{llm_call_count}",
                candidate_hash=sha256_text(final_code),
                framework_valid=True,
                imports_valid=not any("import manifest violation" in error for error in last_validation_errors),
                target_invocation_valid=not any("target invocation" in error for error in last_validation_errors),
                oracle_valid=True,
                syntax_valid=True,
                repair_actions=sorted(set(repair_actions_accum)),
                rejection_reason="",
            ).to_dict()

        return GeneratedReproductionTest(
            instance_id=instance.instance_id,
            scenario_id=scenario.get("scenario_id", "unknown"),
            model_name=self.client.config.model_name,
            repo_path=repo_path,
            target_test_file=target_test_file,
            target_test_file_abspath=str(target_test_abspath),
            target_source_file=_scenario_target_location(scenario).get("source_file", ""),
            insert_mode=parsed["insert_mode"],
            insertion_hint=parsed["insertion_hint"],
            imports=parsed["imports"],
            test_code=parsed["test_code"],
            original_test_file_content=original_content,
            modified_test_file_content=modified_content,
            test_patch=test_patch,
            raw_response=last_raw_response,
            prompt=prompt,
            canonical_test_nodeid=canonical_test_nodeid,
            generated_patch_path="generated_test.patch",
            generated_patch_sha256=sha256_text(test_patch),
            candidate_status=CandidateStatus.POSTPROCESSED.value,
            diagnostic_only=False,
            final_set_membership=FinalSetMembership().to_dict(),
            postprocessing_actions=_postprocessing_action_records(repair_actions_accum),
            repair_attempted=bool(repair_actions_accum or repair_failed_reason),
            repair_actions=sorted(set(repair_actions_accum)),
            repair_failed_reason=repair_failed_reason,
            repair_retry_count=repair_retry_count,
            retry_required_oracle_risks=retry_required_oracle_risks,
            semantic_risk_flags=semantic_risk_flags,
            selected_reproduction_example=dict(selected_reproduction_example),
            relational_oracle=dict(relational_oracle),
            candor_oracle={},
            llm_error_refinement={},
            prompt_profile=prompt_profile,
            iteration=iteration,
            generation_attempt_count=llm_call_count,
            token_usage_status="unknown" if usage_missing else "known",
            token_usage=accumulated_tokens,
            generated_scenario_id=scenario.get("scenario_id", "unknown"),
            scenario_generation_attempt=int(scenario.get("scenario_generation_attempt", 1) or 1),
            scenario_generation_provenance=str(scenario.get("generation_provenance") or ""),
            selected_issue_api_target=str(scenario.get("issue_api_target") or target_location.get("target_function") or ""),
            selected_implementation_target=str(scenario.get("implementation_target") or ""),
            setup_helper_calls=list(scenario.get("setup_helper_calls") or []),
            target_verification_status=str(scenario.get("target_verification_status") or ""),
            target_verification_provenance=dict(scenario.get("target_verification_provenance") or {}),
            target_consistency_status=str(scenario.get("target_consistency_status") or ""),
            m4_candidate_classification=str(scenario.get("target_verification_status") or ""),
            m4_selection_policy=str(scenario.get("m4_selection_policy") or "eligible_base_order"),
            m5_target_used=str(scenario.get("issue_api_target") or target_location.get("target_function") or ""),
            m3_model_call_count=int(scenario.get("m3_model_call_count", 0) or 0),
            m5_attempt_count=llm_call_count,
            fallback_used=bool(str(scenario.get("generation_provenance") or "").endswith("fallback")),
            fallback_reason=str(scenario.get("fallback_reason") or ""),
            v31_generation_contract=v31_contract_dict,
            v31_oracle_trace=v31_trace,
            v31_telemetry=v31_telemetry,
        )

    def _generate_v37_single(
        self,
        *,
        instance: Any,
        clue: Mapping[str, Any],
        context: Mapping[str, Any],
        scenario: Mapping[str, Any],
        iteration: int,
        runtime_error_hint: str | None,
    ) -> GeneratedReproductionTest:
        """Generate and parse one exact v37 TestCode artifact for one scenario."""
        target_location = _scenario_target_location(scenario)
        target_test_file = str(
            target_location.get("candidate_test_file")
            or (scenario.get("relevant_test_files") or [""])[0]
            or ""
        )
        if not target_test_file:
            raise GenerationFailureError(
                message="v37 M5 requires a repository-relative candidate test file",
                token_usage={}, attempt_count=0, last_error="missing_target_test_file",
                failure_type_detail="M5_INPUT_CONTRACT", token_usage_status="no_llm_call",
                scenario=dict(scenario),
            )
        relative = Path(target_test_file)
        if relative.is_absolute() or ".." in relative.parts:
            raise GenerationFailureError(
                message="v37 M5 target test file must remain repository-relative",
                token_usage={}, attempt_count=0, last_error="target_test_file_outside_repository",
                failure_type_detail="M5_INPUT_IDENTITY", token_usage_status="no_llm_call",
                scenario=dict(scenario),
            )
        repo_path = str(context.get("repo_path") or "")
        target_path = Path(repo_path) / relative
        if not target_path.exists() and not target_path.parent.exists():
            raise FileNotFoundError(f"Target test file does not exist: {target_path}")
        prior_spectrum = list(context.get("prior_m6_sbfl_spectrum") or [])[:50]
        feedback = scenario.get("diagnostic_feedback") or {}
        prompt_payload = {
            "schema_version": "m5-v37-test-code-request-v1",
            "task": "Generate one issue-reproducing test for exactly this selected scenario.",
            "scenario": {
                key: scenario.get(key) or target_location.get(key)
                for key in (
                    "scenario_id",
                    "target_function",
                    "source_file",
                    "oracle_type",
                    "oracle_expected",
                    "stimulus_steps",
                )
            },
            "target_test_file": target_test_file,
            "code_context": {
                "suspicious_functions": list(context.get("suspicious_functions") or [])[:5],
                "fault_hypothesis": context.get("fault_hypothesis"),
                "oracle_hint": context.get("oracle_hint"),
            },
            "prior_sbfl_spectrum": prior_spectrum,
            "prior_sbfl_source_limit": 50,
            "diagnostic_feedback": {
                key: feedback.get(key)
                for key in ("why_failed", "fix_suggestion", "assumption_gap")
                if isinstance(feedback, Mapping) and feedback.get(key) not in (None, "", [])
            },
            "avoid_list": list(scenario.get("previous_assertion_pattern_avoid_list") or []),
            "runtime_error_hint": runtime_error_hint,
            "required_output": {
                "code": "complete test code",
                "language": "python|java|...",
                "test_methods": ["method_names"],
                "oracle_spec": {"type": "oracle type", "expected": "expected behavior"},
            },
            "rules": [
                "Follow stimulus_steps and verify oracle_expected using oracle_type.",
                "Import only symbols present in the pre-patch repository.",
                "Return exactly one JSON object with the four required fields.",
            ],
        }
        prompt = json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True, indent=2)
        raw_response = self.client.generate(
            prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=self.client.config.temperature,
            prompt_compactor=compact_m5_prompt,
        )
        try:
            parsed = json.loads(str(raw_response).strip())
        except json.JSONDecodeError as exc:
            raise GenerationFailureError(
                message=f"v37 M5 exact TestCode JSON parse failed: {exc.msg}",
                token_usage=dict(getattr(self.client, "last_usage", {}) or {}),
                attempt_count=1, last_error="malformed_exact_testcode_json",
                failure_type_detail="MODEL_OUTPUT_SCHEMA", token_usage_status="known",
                raw_response=raw_response, prompt=prompt, scenario=dict(scenario),
            ) from exc
        required_keys = {"code", "language", "test_methods", "oracle_spec"}
        if not isinstance(parsed, Mapping) or set(parsed) != required_keys:
            raise GenerationFailureError(
                message="v37 M5 output must contain exactly code, language, test_methods, oracle_spec",
                token_usage=dict(getattr(self.client, "last_usage", {}) or {}),
                attempt_count=1, last_error="invalid_exact_testcode_shape",
                failure_type_detail="MODEL_OUTPUT_SCHEMA", token_usage_status="known",
                raw_response=raw_response, prompt=prompt, scenario=dict(scenario),
            )
        code = str(parsed.get("code") or "")
        methods = parsed.get("test_methods")
        oracle_spec = parsed.get("oracle_spec")
        if not code or not isinstance(methods, list) or not methods or not isinstance(oracle_spec, Mapping):
            raise GenerationFailureError(
                message="v37 M5 TestCode fields have invalid types or empty required values",
                token_usage=dict(getattr(self.client, "last_usage", {}) or {}),
                attempt_count=1, last_error="invalid_exact_testcode_fields",
                failure_type_detail="MODEL_OUTPUT_SCHEMA", token_usage_status="known",
                raw_response=raw_response, prompt=prompt, scenario=dict(scenario),
            )
        blocking_oracle_flags = list(scenario.get("blocking_oracle_flags") or [])
        scenario_expected = scenario.get("oracle_expected")
        if not _normalize_v37_oracle_value(scenario_expected):
            blocking_oracle_flags.append(ORACLE_EXPECTED_MISSING)
        if _normalize_v37_oracle_value(oracle_spec.get("expected")) != (
            _normalize_v37_oracle_value(scenario_expected)
        ):
            blocking_oracle_flags.append(ORACLE_SPEC_MISMATCH)
        m5_code_before_postprocessing = code
        postprocessed, m5a_actions = apply_m5a_deterministic_postprocessing(
            {
                "append_block": code,
                "test_code": code,
                "imports": [],
            },
            clue=dict(clue),
            repo_path=repo_path,
            context=dict(context),
            runner=str(
                (context.get("project_test_style") or {}).get("runner") or "pytest"
            ),
            import_checker=self._check_import_validity,
            preserve_test_semantics=True,
        )
        code = str(
            postprocessed.get("append_block")
            or postprocessed.get("test_code")
            or code
        )
        if _v37_m5a_changed_oracle_semantics(
            m5_code_before_postprocessing,
            code,
            before_oracle_spec=oracle_spec,
            after_oracle_spec=oracle_spec,
        ):
            blocking_oracle_flags.append(ORACLE_SEMANTICS_CHANGED_BY_REPAIR)
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise GenerationFailureError(
                message=f"v37 M5 TestCode does not compile: {exc.msg}",
                token_usage=dict(getattr(self.client, "last_usage", {}) or {}),
                attempt_count=1, last_error="testcode_syntax_error",
                failure_type_detail="SYNTAX_ERROR", token_usage_status="known",
                raw_candidate=code, raw_response=raw_response, prompt=prompt,
                scenario=dict(scenario),
            ) from exc
        actual_methods = {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test")
        }
        if set(str(name) for name in methods) != actual_methods:
            raise GenerationFailureError(
                message="v37 M5 test_methods must exactly name generated test methods",
                token_usage=dict(getattr(self.client, "last_usage", {}) or {}),
                attempt_count=1, last_error="test_methods_mismatch",
                failure_type_detail="MODEL_OUTPUT_SCHEMA", token_usage_status="known",
                raw_candidate=code, raw_response=raw_response, prompt=prompt,
                scenario=dict(scenario),
            )
        if not _v37_oracle_assertion_present(code, oracle_spec):
            blocking_oracle_flags.append(ORACLE_ASSERTION_MISSING)
        original = read_text(target_path) if target_path.exists() else ""
        modified = original.rstrip() + "\n\n" + code.rstrip() + "\n"
        try:
            ast.parse(modified)
        except SyntaxError as exc:
            raise GenerationFailureError(
                message=f"v37 M5 candidate does not compile in target file: {exc.msg}",
                token_usage=dict(getattr(self.client, "last_usage", {}) or {}),
                attempt_count=1, last_error="target_file_compile_error",
                failure_type_detail="SYNTAX_ERROR", token_usage_status="known",
                raw_candidate=code, raw_response=raw_response, prompt=prompt,
                scenario=dict(scenario),
            ) from exc
        patch = self._build_unified_patch(
            original_content=original,
            modified_content=modified,
            relative_path=target_test_file,
        )
        usage = dict(getattr(self.client, "last_usage", {}) or {})
        return GeneratedReproductionTest(
            instance_id=str(getattr(instance, "instance_id", "")),
            scenario_id=str(scenario.get("scenario_id") or "unknown"),
            model_name=self.client.config.model_name,
            repo_path=repo_path,
            target_test_file=target_test_file,
            target_test_file_abspath=str(target_path),
            target_source_file=str(target_location.get("source_file") or ""),
            insert_mode="append_block",
            insertion_hint="append exact v37 TestCode",
            imports=[],
            test_code=code,
            original_test_file_content=original,
            modified_test_file_content=modified,
            test_patch=patch,
            raw_response=raw_response,
            prompt=prompt,
            generated_patch_path="generated_test.patch",
            generated_patch_sha256=sha256_text(patch),
            postprocessing_actions=_postprocessing_action_records(m5a_actions),
            prompt_profile={
                "schema_version": "m5-v37-test-code-provenance-v1",
                "parse_mode": "exact_json",
                "raw_response_sha256": sha256_text(raw_response),
                "source_scenario_id": str(scenario.get("scenario_id") or "unknown"),
                "source_generation_index": scenario.get("m3_generation_index"),
                "m5_model_call_count": 1,
                "prior_sbfl_line_count": len(prior_spectrum),
                "m5a_deterministic_postprocessing": {
                    "executed": True,
                    "preserve_test_semantics": True,
                    "actions": list(m5a_actions),
                },
            },
            iteration=iteration,
            generation_attempt_count=1,
            token_usage_status="known" if usage else "unknown",
            token_usage=usage,
            generated_scenario_id=str(scenario.get("scenario_id") or "unknown"),
            m5_attempt_count=1,
            language=str(parsed["language"]),
            test_methods=[str(name) for name in methods],
            oracle_spec=dict(oracle_spec),
            m5_invocation_provenance={
                "selected_scenario_id": str(scenario.get("scenario_id") or "unknown"),
                "model_call_index": 1,
                "model_call_count": 1,
            },
            blocking_oracle_flags=validated_v37_blocking_oracle_flags(
                blocking_oracle_flags
            ),
        )

    def save(self, result: GeneratedReproductionTest, output_path: str) -> None:
        """
        output_path 예:
        outputs/<instance_id>/generated_test.json

        같이 저장되는 파일:
        - generated_test.json
        - generated_test.patch
        - generated_test_rendered.py
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.patch_sha256 = sha256_text(result.test_patch)
        result.generated_patch_sha256 = result.patch_sha256

        patch_path = path.with_suffix(".patch")
        result.generated_patch_path = patch_path.name

        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(result.test_patch)

        rendered_path = path.with_name(path.stem + "_rendered.py")
        with open(rendered_path, "w", encoding="utf-8") as f:
            f.write(result.modified_test_file_content)

    def validate_repair_candidate(
        self,
        *,
        parsed: Dict[str, Any],
        repo_path: str,
        context: Dict[str, Any],
        clue: Dict[str, Any],
        scenario: Dict[str, Any],
    ) -> ValidationResult:
        """Run the mandatory M5 static gates for a repaired candidate."""
        validation = self._validate_generated_code(
            parsed=parsed,
            repo_path=repo_path,
            context=context,
            clue=clue,
            scenario=scenario,
        )
        if not validation.is_valid:
            return validation
        raw_contract = context.get("v31_generation_contract")
        if not isinstance(raw_contract, Mapping):
            return validation
        contract = _coerce_v31_generation_contract(raw_contract)
        contract_errors = _v31_contract_errors(
            parsed,
            contract,
            repo=Path(repo_path),
            available_imports=(
                context.get("available_imports")
                if isinstance(context.get("available_imports"), Mapping)
                else {}
            ),
            import_checker=self._check_import_validity,
            import_context=context,
        )
        if not contract_errors:
            return validation
        return ValidationResult(
            is_valid=False,
            errors=[*validation.errors, *contract_errors],
            warnings=list(validation.warnings),
            fixed_imports=validation.fixed_imports,
        )

    def build_repaired_generated_test(
        self,
        *,
        instance: Any,
        original: GeneratedReproductionTest | None,
        parsed: Dict[str, Any],
        repaired_code: str,
        clue: Dict[str, Any],
        context: Dict[str, Any],
        scenario: Dict[str, Any],
        iteration: int,
        prompt: str = "",
        raw_response: str = "",
        llm_error_refinement: Dict[str, Any] | None = None,
        generation_attempt_count: int = 0,
        token_usage: Dict[str, int] | None = None,
        token_usage_status: str = "known",
    ) -> GeneratedReproductionTest:
        """Promote a repaired candidate only after the caller revalidates it."""
        repo_path = str(context.get("repo_path") or (original.repo_path if original else ""))
        if not repo_path:
            raise ValueError("cannot build repaired generated test without repo_path")
        scenario = dict(scenario or {})
        parsed = dict(parsed or {})
        parsed.setdefault("insert_mode", "append_block")
        parsed.setdefault("imports", [])
        parsed.setdefault("insertion_hint", "append to file")
        parsed["append_block"] = repaired_code
        parsed["test_code"] = repaired_code
        target_test_file = str(
            parsed.get("target_test_file")
            or (original.target_test_file if original else "")
            or ((scenario.get("target_location") or {}).get("candidate_test_file") if isinstance(scenario.get("target_location"), dict) else "")
            or ""
        )
        if not target_test_file:
            raise ValueError("cannot build repaired generated test without target_test_file")
        parsed["target_test_file"] = target_test_file
        validation = self._validate_generated_code(
            parsed=parsed,
            repo_path=repo_path,
            context=context,
            clue=clue,
            scenario=scenario,
        )
        if validation.fixed_imports is not None:
            parsed["imports"] = validation.fixed_imports
        if not validation.is_valid:
            raise ValueError("repaired candidate failed validation: " + "; ".join(validation.errors))

        target_test_abspath = Path(repo_path) / target_test_file
        original_content = (
            original.original_test_file_content
            if original is not None and original.original_test_file_content
            else read_text(target_test_abspath)
            if target_test_abspath.exists()
            else ""
        )
        base_content = original_content
        base_commit = getattr(instance, "base_commit", None)
        if base_commit:
            try:
                shown = subprocess.run(
                    ["git", "show", f"{base_commit}:{target_test_file}"],
                    capture_output=True,
                    text=True,
                    cwd=repo_path,
                )
                if shown.returncode == 0 and shown.stdout.strip():
                    base_content = shown.stdout
            except Exception:
                base_content = original_content

        if parsed.get("insert_mode") == "append_block":
            modified_content = base_content.rstrip() + "\n\n" + repaired_code.rstrip() + "\n"
        else:
            modified_content = self._build_modified_test_file_content(
                original_content=base_content,
                imports=parsed.get("imports", []),
                test_code=repaired_code,
            )
        test_patch = self._build_unified_patch(
            original_content=base_content,
            modified_content=modified_content,
            relative_path=target_test_file,
        )
        canonical_test_nodeid = _canonical_test_nodeid_from_generated(
            target_test_file,
            repaired_code,
        )
        repaired_oracle_spec = (
            dict(parsed.get("oracle_spec"))
            if isinstance(parsed.get("oracle_spec"), Mapping)
            else dict(original.oracle_spec or {})
            if original is not None
            else {}
        )
        blocking_oracle_flags = list(
            getattr(original, "blocking_oracle_flags", []) or []
        )
        if self.feature_profile == "v37":
            if original is not None and _v37_m5a_changed_oracle_semantics(
                original.test_code,
                repaired_code,
                before_oracle_spec=original.oracle_spec,
                after_oracle_spec=repaired_oracle_spec,
            ):
                blocking_oracle_flags.append(
                    ORACLE_SEMANTICS_CHANGED_BY_REPAIR
                )
            if not _v37_oracle_assertion_present(
                repaired_code,
                repaired_oracle_spec,
            ):
                blocking_oracle_flags.append(ORACLE_ASSERTION_MISSING)
        return GeneratedReproductionTest(
            instance_id=instance.instance_id,
            scenario_id=scenario.get("scenario_id", "unknown"),
            model_name=self.client.config.model_name,
            repo_path=repo_path,
            target_test_file=target_test_file,
            target_test_file_abspath=str(target_test_abspath),
            target_source_file=(scenario.get("target_location") or {}).get("source_file", "")
            if isinstance(scenario.get("target_location"), dict)
            else "",
            insert_mode=parsed["insert_mode"],
            insertion_hint=parsed["insertion_hint"],
            imports=parsed.get("imports", []),
            test_code=repaired_code,
            original_test_file_content=original_content,
            modified_test_file_content=modified_content,
            test_patch=test_patch,
            raw_response=raw_response,
            prompt=prompt,
            canonical_test_nodeid=canonical_test_nodeid,
            generated_patch_path="generated_test.patch",
            generated_patch_sha256=sha256_text(test_patch),
            candidate_status=CandidateStatus.POSTPROCESSED.value,
            diagnostic_only=False,
            final_set_membership=FinalSetMembership().to_dict(),
            postprocessing_actions=[
                {
                    "stage": "m5a_llm_error_refinement",
                    "action": "promoted_repaired_candidate_after_static_validation",
                    "source": "repair_loop",
                }
            ],
            repair_attempted=True,
            repair_actions=["m5a_llm_error_refinement"],
            repair_failed_reason="",
            repair_retry_count=1,
            retry_required_oracle_risks=[],
            semantic_risk_flags=[],
            selected_reproduction_example={},
            relational_oracle={},
            candor_oracle={},
            llm_error_refinement=dict(llm_error_refinement or {}),
            prompt_profile={
                "optional_features": _m3_m5_optional_feature_metadata(
                    self.feature_flags,
                    llm_error_refinement=None,
                ),
                "repair_promoted": True,
            },
            iteration=iteration,
            generation_attempt_count=generation_attempt_count,
            token_usage_status=token_usage_status,
            token_usage=token_usage or _empty_token_usage(),
            language=str(
                parsed.get("language")
                or getattr(original, "language", "python")
                or "python"
            ),
            test_methods=[
                str(name)
                for name in (
                    parsed.get("test_methods")
                    or getattr(original, "test_methods", [])
                    or []
                )
            ],
            oracle_spec=repaired_oracle_spec,
            blocking_oracle_flags=validated_v37_blocking_oracle_flags(
                blocking_oracle_flags
            ),
            diagnostic_oracle_flags=list(
                getattr(original, "diagnostic_oracle_flags", []) or []
            ),
        )

    def build_invalid_generated_test_from_failure(
        self,
        *,
        instance: Any,
        error: GenerationFailureError,
        context: Dict[str, Any],
        iteration: int,
    ) -> GeneratedReproductionTest | None:
        """Persist the strongest observed invalid M5 candidate for diagnostics.

        This does not make the candidate executable or eligible for M6/M7; callers
        route it to NOT_VALID accounting after saving the artifact.
        """
        scenario = dict(error.scenario or {})
        parsed = dict(error.parsed_candidate or {})
        rendered_code = (
            str(parsed.get("append_block") or parsed.get("test_code") or error.raw_candidate or "")
        )
        if not rendered_code.strip():
            return None
        repo_path = str(context.get("repo_path") or "")
        target = scenario.get("target_location") if isinstance(scenario.get("target_location"), dict) else {}
        target_test_file = str(
            parsed.get("target_test_file")
            or target.get("candidate_test_file")
            or ""
        )
        if not target_test_file:
            return None
        target_test_abspath = Path(repo_path) / target_test_file if repo_path else Path(target_test_file)
        original_content = ""
        if target_test_abspath.exists():
            try:
                original_content = read_text(target_test_abspath)
            except OSError:
                original_content = ""
        modified_content = original_content.rstrip() + "\n\n" + rendered_code.rstrip() + "\n"
        test_patch = self._build_unified_patch(
            original_content=original_content,
            modified_content=modified_content,
            relative_path=target_test_file,
        )
        return GeneratedReproductionTest(
            instance_id=instance.instance_id,
            scenario_id=scenario.get("scenario_id", "unknown"),
            model_name=self.client.config.model_name,
            repo_path=repo_path,
            target_test_file=target_test_file,
            target_test_file_abspath=str(target_test_abspath),
            target_source_file=str(target.get("source_file") or ""),
            insert_mode=str(parsed.get("insert_mode") or "append_block"),
            insertion_hint=str(parsed.get("insertion_hint") or "diagnostic_invalid_candidate"),
            imports=list(parsed.get("imports") or []),
            test_code=rendered_code,
            original_test_file_content=original_content,
            modified_test_file_content=modified_content,
            test_patch=test_patch,
            raw_response=error.raw_response,
            prompt=error.prompt,
            canonical_test_nodeid=_canonical_test_nodeid_from_generated(target_test_file, rendered_code),
            generated_patch_path="generated_test.patch",
            generated_patch_sha256=sha256_text(test_patch),
            candidate_status=CandidateStatus.INVALID.value,
            diagnostic_only=True,
            final_set_membership=FinalSetMembership().to_dict(),
            postprocessing_actions=[],
            repair_attempted=True,
            repair_actions=[],
            repair_failed_reason=error.last_error,
            repair_retry_count=0,
            retry_required_oracle_risks=[],
            semantic_risk_flags=[],
            selected_reproduction_example={},
            relational_oracle={},
            candor_oracle={},
            llm_error_refinement={},
            prompt_profile={
                "invalid_candidate_persisted": True,
                "generation_attempt_count": error.attempt_count,
                "attempt_history": list(error.attempt_history or []),
                "token_usage_status": error.token_usage_status,
            },
            iteration=iteration,
            generation_attempt_count=error.attempt_count,
            token_usage_status=error.token_usage_status,
            token_usage=dict(error.token_usage or _empty_token_usage()),
            generated_scenario_id=scenario.get("scenario_id", "unknown"),
            scenario_generation_attempt=int(scenario.get("scenario_generation_attempt", 1) or 1),
            scenario_generation_provenance=str(scenario.get("generation_provenance") or ""),
            selected_issue_api_target=str(scenario.get("issue_api_target") or target.get("target_function") or ""),
            selected_implementation_target=str(scenario.get("implementation_target") or ""),
            setup_helper_calls=list(scenario.get("setup_helper_calls") or []),
            target_verification_status=str(scenario.get("target_verification_status") or ""),
            target_verification_provenance=dict(scenario.get("target_verification_provenance") or {}),
            target_consistency_status=str(scenario.get("target_consistency_status") or ""),
            m4_candidate_classification=str(scenario.get("target_verification_status") or ""),
            m4_selection_policy=str(scenario.get("m4_selection_policy") or "eligible_base_order"),
            m5_target_used=str(scenario.get("issue_api_target") or target.get("target_function") or ""),
            m3_model_call_count=int(scenario.get("m3_model_call_count", 0) or 0),
            m5_attempt_count=error.attempt_count,
            fallback_used=bool(str(scenario.get("generation_provenance") or "").endswith("fallback")),
            fallback_reason=str(scenario.get("fallback_reason") or ""),
            validation_diagnostics={
                "validation_errors": list(error.validation_errors or []),
                "validation_status": dict(error.validation_status or {}),
                "failure_type_detail": error.failure_type_detail,
                "last_error": error.last_error,
                "attempt_history": list(error.attempt_history or []),
            },
        )

    @staticmethod
    def _build_fault_location_section(clue: Dict[str, Any]) -> str:
        """clue의 fault_locations를 프롬프트 섹션으로 변환한다."""
        fault_locations = clue.get("fault_locations", [])
        if not fault_locations:
            return ""
        traceback_lines = []
        inferred_lines = []
        for fl in fault_locations[:5]:
            fp = fl.get("file_path", "").replace("\\", "/")
            fn = fl.get("function_name", "")
            ln = fl.get("line_no", "?")
            source = fl.get("source", "traceback")
            confidence = fl.get("confidence", "high" if source == "traceback" else "medium")
            parts = fp.split("/")
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
                "The issue's stack trace explicitly points to these code locations.\n"
                "Your test MUST directly call the function identified here (or its public wrapper):\n"
                + "\n".join(traceback_lines)
                + "\n"
                "If this function is private (starts with _), call its nearest public caller instead.\n"
            )
        if inferred_lines:
            sections.append(
                "\n[Inferred Fault Location Candidates]\n"
                "These are medium-confidence hints, not mandatory traceback locations:\n"
                + "\n".join(inferred_lines)
                + "\n"
            )
        return "".join(sections)

    @staticmethod
    def _build_repair_directive_section(
        scenario: Dict[str, Any],
        prompt_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        directive = scenario.get("repair_directive") if isinstance(scenario, dict) else None
        if not isinstance(directive, dict) or not directive.get("mode"):
            return ""
        directive = sanitize_repair_directive(directive) if scenario.get("oracle_requires_regeneration") else directive
        evidence = directive.get("evidence") if isinstance(directive.get("evidence"), dict) else {}
        compact = {
            "mode": directive.get("mode", ""),
            "dimension": directive.get("dimension", ""),
            "blocking_reason": directive.get("blocking_reason", ""),
            "preserved_fields": (directive.get("preserved_fields") or [])[:8],
            "modified_fields": (directive.get("modified_fields") or [])[:8],
            "preservation_policy": directive.get("preservation_policy") or {},
            "must_change": (directive.get("must_change") or [])[:6],
            "must_keep": (directive.get("must_keep") or [])[:5],
            "forbidden_patterns": (directive.get("forbidden_patterns") or [])[:8],
            "replacement_hints": (directive.get("replacement_hints") or [])[:8],
            "repair_memory": {
                "forbidden_test_files": (
                    (scenario.get("repair_memory") or {}).get("forbidden_test_files") or []
                )[:8],
                "required_target_file": (
                    (scenario.get("repair_memory") or {}).get("required_target_file") or ""
                ),
                "forbidden_imports": (
                    (scenario.get("repair_memory") or {}).get("forbidden_imports") or []
                )[:8],
            },
            "evidence": {
                "target_source": evidence.get("target_source", ""),
                "target_function": evidence.get("target_function", ""),
                "candidate_test_file": evidence.get("candidate_test_file", ""),
                "covered_functions": (evidence.get("covered_functions") or [])[:8],
                "exception_type": evidence.get("exception_type", ""),
                "failing_line": evidence.get("failing_line", ""),
                "failing_test": evidence.get("failing_test", ""),
                "coverage_score": evidence.get("coverage_score", ""),
                "issue_call_patterns": (evidence.get("issue_call_patterns") or [])[:3],
            },
            "previous_candidate": _clip_prompt_text(
                str(directive.get("previous_candidate") or ""),
                4000,
                "repair_previous_candidate",
                prompt_profile,
            ),
        }
        _mark_prompt_section(prompt_profile, "repair_directive")
        return (
            "\n[REPAIR DIRECTIVE — MUST OBEY]\n"
            "This directive supersedes older free-text feedback. Do not repeat forbidden patterns.\n"
            f"{json.dumps(compact, ensure_ascii=False, indent=2)}\n"
        )

    def _build_prompt(
        self,
        instance: Any,
        clue: Dict[str, Any],
        context: Dict[str, Any],
        scenario: Dict[str, Any],
        existing_test_imports: str = "",
        target_test_example: str = "",
        runtime_error_hint: Optional[str] = None,
        prompt_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        if prompt_profile is not None:
            prompt_profile.setdefault("budget_mode", "compact")
            prompt_profile.setdefault("sections_included", [])
            prompt_profile.setdefault("truncated_sections", [])

        prompt_scenario = _scenario_for_selected_reproduction_example(scenario)
        prompt_scenario = _attach_relational_oracle_candidate(prompt_scenario, clue, context)
        _identifiers = scenario.get("identifiers") or clue.get("identifiers", {})
        noisy_functions = {
            "arange", "rand", "random", "seed", "platform", "get_backend",
            "show_versions",
        }
        issue_functions = [
            fn for fn in _identifiers.get("functions", [])
            if fn not in noisy_functions
        ]
        issue_classes = _identifiers.get("classes", [])
        issue_error_keywords = scenario.get("error_keywords") or clue.get("error_keywords", [])
        target_location = _scenario_target_location(scenario)
        target_test_file = target_location.get("candidate_test_file") or (
            scenario.get("relevant_test_files", [""])[0] if scenario.get("relevant_test_files") else ""
        )
        verified_target = context.get("verified_target_evidence") or scenario.get("verified_target_evidence") or {}
        verified_target_section = ""
        if isinstance(verified_target, dict) and verified_target:
            compact_verified_target = {
                key: verified_target.get(key)
                for key in (
                    "source_file",
                    "target_test_file",
                    "canonical_target_identity",
                    "candidate_invocation_expression",
                    "target_verification_status",
                    "signature",
                )
                if verified_target.get(key) not in (None, "", [], {})
            }
            _mark_prompt_section(prompt_profile, "verified_target_evidence")
            verified_target_section = (
                "\n[Verified Repository Target Evidence — MANDATORY]\n"
                "This evidence was checked against the pre-patch repository before generation. "
                "Invalid M7 reroutes and nonexistent files have lower priority than this section.\n"
                f"{json.dumps(_m5_constraint_json_safe(compact_verified_target), ensure_ascii=False, indent=2)}\n"
            )

        project_framework = context.get("project_test_style", {}).get("framework", "unknown")
        runner = context.get("project_test_style", {}).get("runner", "pytest")
        test_example = context.get("test_example_snippet", "")
        source_candidates = [x.get("path", "") for x in context.get("candidate_source_files", [])[:3]]
        test_candidates = [x.get("path", "") for x in context.get("candidate_test_files", [])[:3]]
        hypotheses = context.get("localization_hypotheses") or []
        v30_hypothesis_section = ""
        if context.get("feature_profile") == "v30" and isinstance(hypotheses, list):
            v30_hypothesis_section = (
                "\n[V30 Localization Hypotheses — SELECT ONE AND PRESERVE IDENTITY]\n"
                + json.dumps(_m5_constraint_json_safe(hypotheses[:3]), ensure_ascii=False, indent=2)
                + "\n"
            )
            _mark_prompt_section(prompt_profile, "v30_localization_hypotheses")

        v31_contract_section = ""
        if context.get("feature_profile") == "v31" and isinstance(context.get("v31_generation_contract"), Mapping):
            full_contract = context["v31_generation_contract"]
            contract_payload = {
                key: full_contract.get(key)
                for key in (
                    "framework",
                    "runner",
                    "shape",
                    "target_test_file",
                    "target_source_file",
                    "forbidden_patterns",
                    "skeleton_source",
                )
                if full_contract.get(key) not in (None, "", [], {})
            }
            v31_contract_section = (
                "\n[V31 Pre-Generation Structure — MANDATORY]\n"
                "Use this repository-verified runner, file, and test shape. Imports and oracle evidence are listed once in their dedicated sections below.\n"
                + json.dumps(_m5_constraint_json_safe(contract_payload), ensure_ascii=False, indent=2)
                + "\n"
            )
            _mark_prompt_section(prompt_profile, "v31_generation_contract")

        django_grounding_section = ""
        django_grounding = context.get("django_repository_grounding")
        if runner == "django-test" and isinstance(django_grounding, Mapping):
            django_grounding_section = (
                "\n[Django Pre-Patch Repository Grounding — DO NOT GUESS]\n"
                "Use only these inspectable class shapes, model fields, and APIs. "
                "Adapt the nearby working test rather than inventing a model/API.\n"
                + json.dumps(
                    _m5_constraint_json_safe(django_grounding),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
            _mark_prompt_section(prompt_profile, "django_repository_grounding")

        # available_imports 정보를 compact budget에 맞춰 변환
        available_imports = context.get("available_imports", {})
        import_map_lines = []
        for module, symbols in _prioritized_available_imports(available_imports, clue, context, scenario):
            if symbols:
                import_map_lines.append(f"  {module}: {', '.join(symbols[:_PROMPT_IMPORT_SYMBOLS])}")
        import_map_text = "\n".join(import_map_lines) if import_map_lines else "  (not available)"
        if import_map_lines:
            _mark_prompt_section(prompt_profile, "available_imports_compact")
        if len(available_imports or {}) > len(import_map_lines):
            if prompt_profile is not None:
                prompt_profile.setdefault("truncated_sections", []).append("available_imports")

        # conftest fixtures 섹션
        conftest_fixtures = context.get("conftest_fixtures", {})
        conftest_section = ""
        if conftest_fixtures:
            lines = ["[Available Pytest Fixtures (from conftest.py)]"]
            for path, names in list(conftest_fixtures.items())[:_PROMPT_CONFTEST_PATHS]:
                lines.append(f"  {path}: {', '.join(names[:_PROMPT_CONFTEST_FIXTURES])}")
            conftest_section = "\n".join(lines) + "\n"
            _mark_prompt_section(prompt_profile, "conftest_fixtures")
            if len(conftest_fixtures) > _PROMPT_CONFTEST_PATHS and prompt_profile is not None:
                prompt_profile.setdefault("truncated_sections", []).append("conftest_fixtures")

        # scenario의 required_fixtures 섹션
        test_env = scenario.get("test_environment", {})
        required_fixtures = test_env.get("required_fixtures", []) if test_env else []
        required_fixtures_section = ""
        if required_fixtures:
            required_fixtures_section = (
                f"[Required Fixtures for This Scenario]\n"
                f"  {', '.join(required_fixtures)}\n"
            )

        # 기존 테스트 파일 import 블록 — 가져온 심볼 이름 추출
        existing_imports_section = ""
        if existing_test_imports:
            # import 라인에서 이름 파싱 (전체 블록에서 파싱하되, 출력은 cap)
            _imported_names: list[str] = []
            for _line in existing_test_imports.splitlines():
                _line = _line.strip()
                if _line.startswith("from ") and " import " in _line:
                    _names_part = _line.split(" import ", 1)[1].split("#")[0]
                    for _n in _names_part.split(","):
                        _n = _n.strip().split(" as ")[-1].strip()
                        if _n and _n.isidentifier():
                            _imported_names.append(_n)
                elif _line.startswith("import "):
                    _n = _line[7:].strip().split(" as ")[-1].split("#")[0].strip()
                    if _n and _n.isidentifier():
                        _imported_names.append(_n)
            _imported_names_str = ", ".join(_imported_names[:_PROMPT_EXISTING_SYMBOLS]) if _imported_names else "(none)"

            # 토큰 예산 보호: import 블록이 길면 잘라냄
            _imports_display = _clip_prompt_text(
                existing_test_imports,
                _PROMPT_EXISTING_IMPORTS_CHARS,
                "existing_test_imports",
                prompt_profile,
            )
            _mark_prompt_section(prompt_profile, "existing_test_imports")

            existing_imports_section = f"""
[Existing Imports in Target Test File]
The following imports already exist in {target_test_file}.
```
{_imports_display}
```
Available symbols (already imported — use them DIRECTLY, do NOT re-import):
{_imported_names_str}

CRITICAL:
- Do NOT define new class or model definitions (e.g., class MyModel(models.Model): ...).
- Do NOT import symbols that are not listed in [Available Imports from Repository] or above.
- Use only the classes/functions that are already imported in this file.
"""

        # === 이슈 원문의 코드 예시 섹션 구축 ===
        # scenario에 merge된 값 우선 사용 (run_single._merge_clue_into_scenarios로 주입됨)
        code_examples = prompt_scenario.get("reproduction_code") or clue.get("code_examples", [])
        expected_outputs = prompt_scenario.get("expected_outputs") or (
            clue.get("expected_outputs", []) if not prompt_scenario.get("oracle_requires_regeneration") else []
        )
        actual_outputs = prompt_scenario.get("actual_outputs") or clue.get("actual_outputs", [])
        concise_expected_values = _concise_expected_values_for_prompt(prompt_scenario, clue)
        oracle_hints = prompt_scenario.get("oracle_hints") or []
        oracle_text = prompt_scenario.get("oracle", "")

        oracle_hint_section = ""
        if oracle_hints or oracle_text:
            lines = ["[Synthesized Oracle Hints — use before raw issue text]"]
            if oracle_text:
                lines.append(_clip_prompt_text(oracle_text, _PROMPT_ORACLE_TEXT_CHARS, "oracle_text", prompt_profile))
            for hint in oracle_hints[:_PROMPT_ORACLE_HINTS_MAX]:
                hint_str = str(hint).strip()
                if hint_str:
                    lines.append(f"- {_clip_prompt_text(hint_str, _PROMPT_ORACLE_HINT_CHARS, 'oracle_hints', prompt_profile)}")
            oracle_hint_section = "\n".join(lines) + "\n"
            _mark_prompt_section(prompt_profile, "oracle_hints")

        issue_code_section = ""
        if code_examples:
            code_parts = []
            included_blocks = 0
            selected_code_examples = (
                _select_reproduction_blocks_for_prompt(code_examples, scenario)
                if isinstance(code_examples, list)
                else code_examples
            )
            for i, block in enumerate(selected_code_examples):
                if included_blocks >= _PROMPT_CODE_EXAMPLES_MAX:
                    break
                if isinstance(block, dict) and block.get("is_system_or_output"):
                    continue
                if not isinstance(block, dict):
                    block = {"code": str(block)}
                ctx = block.get("context_before", "")
                code = block.get("code", "")
                interactive_in = block.get("interactive_input", "")
                interactive_out = block.get("interactive_output", "")

                label = f"Code Block {i + 1}"
                if ctx:
                    label += f" (context: \"{ctx[:100]}\")"

                code_trunc = _clip_prompt_text(code, _PROMPT_CODE_CHARS, "issue_code_examples", prompt_profile)
                code_parts.append(f"### {label}\n```python\n{code_trunc}\n```")
                if interactive_in:
                    code_parts.append(
                        "Interactive input:\n```python\n"
                        f"{_clip_prompt_text(interactive_in, _PROMPT_INTERACTIVE_CHARS, 'interactive_input', prompt_profile)}\n```"
                    )
                if interactive_out:
                    code_parts.append(
                        "Output:\n```\n"
                        f"{_clip_prompt_text(interactive_out, _PROMPT_INTERACTIVE_CHARS, 'interactive_output', prompt_profile)}\n```"
                    )
                included_blocks += 1

            if code_parts:
                issue_code_section += "\n[Issue Reproduction Code from Original Issue]\n"
                issue_code_section += "The following code blocks are extracted from the original GitHub issue.\n"
                issue_code_section += (
                    "Treat these blocks as the canonical stimulus. Preserve the same object construction, "
                    "operators, call signatures, and warning/exception context whenever possible. "
                    "Do NOT replace them with a simpler generic example. "
                    "If multiple blocks show a simple baseline and a more complex failing/problem variant, "
                    "build the test from the failing/problem variant, not the baseline sanity check.\n\n"
                )
                issue_code_section += "\n\n".join(code_parts)
                _mark_prompt_section(prompt_profile, "issue_code_examples")

            if concise_expected_values:
                issue_code_section += "\n\n[Concise Expected Value Markers]\n"
                issue_code_section += (
                    "These short values were extracted from the issue text. Prefer a positive assertion against one of them when applicable:\n"
                )
                for value in concise_expected_values:
                    issue_code_section += f"- {value}\n"
                _mark_prompt_section(prompt_profile, "concise_expected_values")

            if expected_outputs:
                issue_code_section += "\n\n[Expected Correct Output (from issue)]\n"
                issue_code_section += "These outputs represent the CORRECT behavior (what the code should produce after the fix):\n"
                for out in expected_outputs[:_PROMPT_OUTPUTS_MAX]:
                    out_str = out if isinstance(out, str) else str(out)
                    issue_code_section += (
                        "```\n"
                        f"{_clip_prompt_text(out_str, _PROMPT_OUTPUT_CHARS, 'expected_outputs', prompt_profile)}\n"
                        "```\n"
                    )
                _mark_prompt_section(prompt_profile, "expected_outputs")

            if actual_outputs:
                issue_code_section += "\n[Actual Buggy Output (from issue)]\n"
                issue_code_section += "These outputs represent the BUGGY behavior (what the code currently produces):\n"
                for out in actual_outputs[:_PROMPT_OUTPUTS_MAX]:
                    out_str = out if isinstance(out, str) else str(out)
                    issue_code_section += (
                        "```\n"
                        f"{_clip_prompt_text(out_str, _PROMPT_OUTPUT_CHARS, 'actual_outputs', prompt_profile)}\n"
                        "```\n"
                    )
                _mark_prompt_section(prompt_profile, "actual_outputs")

            issue_code_section += "\n"

        if prompt_scenario.get("oracle_requires_regeneration"):
            issue_code_section += "\n[Stimulus-Oracle Pairing]\n"
            issue_code_section += (
                "The selected reproduction stimulus has no safely associated expected output. "
                "Do not reuse expected outputs from other examples; derive an EB-grounded oracle "
                "from the selected stimulus or choose a different scenario.\n"
            )
            candidate = build_issue_supported_relational_oracle_candidate(prompt_scenario, clue, context)
            relational_section = relational_oracle_prompt_section(candidate)
            if relational_section:
                issue_code_section += relational_section
                _mark_prompt_section(prompt_profile, "relational_oracle_candidate")
            _mark_prompt_section(prompt_profile, "stimulus_oracle_pairing")

        if not code_examples and (expected_outputs or actual_outputs):
            if concise_expected_values:
                issue_code_section += "\n[Concise Expected Value Markers]\n"
                for value in concise_expected_values:
                    issue_code_section += f"- {value}\n"
                _mark_prompt_section(prompt_profile, "concise_expected_values")
            if expected_outputs:
                issue_code_section += "\n[Expected Correct Output (from issue)]\n"
                for out in expected_outputs[:_PROMPT_OUTPUTS_MAX]:
                    issue_code_section += (
                        "```\n"
                        f"{_clip_prompt_text(out, _PROMPT_OUTPUT_CHARS, 'expected_outputs', prompt_profile)}\n"
                        "```\n"
                    )
                _mark_prompt_section(prompt_profile, "expected_outputs")
            if actual_outputs:
                issue_code_section += "\n[Actual Buggy Output (from issue)]\n"
                for out in actual_outputs[:_PROMPT_OUTPUTS_MAX]:
                    issue_code_section += (
                        "```\n"
                        f"{_clip_prompt_text(out, _PROMPT_OUTPUT_CHARS, 'actual_outputs', prompt_profile)}\n"
                        "```\n"
                    )
                _mark_prompt_section(prompt_profile, "actual_outputs")

        # === raw_issue_text 섹션 (코드 예시가 없을 때의 fallback이자 보충) ===
        raw_issue_text = clue.get("raw_issue_text", "")
        if prompt_scenario.get("oracle_requires_regeneration"):
            raw_issue_text = _sanitize_raw_issue_text_for_unpaired_oracle(raw_issue_text, clue)
        raw_issue_section = ""
        if raw_issue_text:
            truncated = _clip_prompt_text(raw_issue_text, _PROMPT_RAW_ISSUE_CHARS, "raw_issue_text", prompt_profile)
            raw_issue_section = f"""
[Full Issue Description]
{truncated}
"""
            _mark_prompt_section(prompt_profile, "raw_issue_text")

        # runner별 테스트 구조 힌트 (모든 레포에 대해 적응적으로 생성)
        framework_constraint = ""
        if runner == "django-test":
            framework_constraint = """
[CRITICAL: Test Structure — Django Test Runner]
This project runs tests via Django's test runner (not raw pytest).
- MUST use `from django.test import TestCase` (NOT `from unittest import TestCase`)
- MUST inherit from django.test.TestCase (or SimpleTestCase if no DB needed)
- Test method MUST be inside the class (standalone `def test_xxx():` is NOT discovered by Django runner)
- Class name MUST start with "Test", method name must start with "test_"
- Package-relative imports are allowed only when they resolve from the existing
  target test module to a repository module exporting the requested symbol.
- Do not invent or assume imports from the `tests` package. A `tests.*` import is
  allowed only when the exact module and every imported symbol are present in
  [Available Imports] or [Existing Imports in Target Test File]; otherwise use
  a repository-verified application import or report the ambiguity.
- Example structure:
  from django.test import TestCase

  class TestMyFeature(TestCase):
      def test_behavior(self):
          self.assertEqual(expected, actual)

FORBIDDEN (these will cause OperationalError, LookupError, or test not collected):
- Defining new Django models inline: `class MyModel(models.Model): ...` → no migration, no table
- Using models NOT imported in [Existing Imports in Target Test File]
- Accessing the database without inheriting from django.test.TestCase
- Creating custom app labels or INSTALLED_APPS entries
- Standalone test functions with `self` parameter: `def test_foo(self):` outside a class
"""
        elif runner == "unittest":
            framework_constraint = """
[Test Structure — unittest/pytest-compatible Repository]
This repository has unittest-style tests, but many files are still collected by pytest.
- Mirror the exact style shown in [Example: Actual Test Methods] when available.
- If the target file mostly uses top-level test functions, write one top-level `def test_*():`.
- If the target file uses TestCase classes, add one method inside a compatible TestCase subclass.
- Use `self.assert*()` only inside TestCase methods; use plain `assert` in top-level functions.
- Do NOT force a new unittest.TestCase class when the target file's existing tests are top-level functions.
"""
        elif runner == "sympy-bin-test":
            framework_constraint = """
[CRITICAL: Test Structure — SymPy (pytest)]
This project runs tests via pytest (sympy test files are pytest-compatible).
- Tests are top-level functions starting with "test_"
- Use plain `assert` statements directly (NOT self.assert*)
- pytest fixtures (e.g. tmp_path, monkeypatch) are allowed if needed
- Keep the test simple and self-contained

[CRITICAL: SymPy Import Rules]
- Import ONLY from modules listed in [Available Imports from Repository]
- NEVER import from deep sub-modules like `sympy.sets.sets`, `sympy.core.core`, etc.
  unless they are explicitly listed in [Available Imports]
- ALWAYS prefer top-level imports: `from sympy import symbols, Function, Lambda, Eq, solve, S`
- If unsure whether a symbol exists in a sub-module, use `from sympy import XYZ` (top-level namespace)
- Common safe top-level imports: symbols, Function, Lambda, Eq, solve, S, I, oo, pi,
  Rational, Integer, Float, Matrix, Symbol, Expr, Add, Mul, Pow, Number
"""

        # matplotlib 특수 처리: Agg backend 필수 지시
        if "matplotlib" in instance.repo.lower():
            framework_constraint += """
[CRITICAL: matplotlib Backend]
REQUIRED at the very TOP of append_block (before any other matplotlib/pyplot import):
  import matplotlib
  matplotlib.use('Agg')  # MUST be set before importing matplotlib.pyplot
Do NOT call plt.show() — the test environment has no display.
"""

        # Sphinx fixtures are valid when grounded in the selected target.
        if "sphinx" in instance.repo.lower():
            framework_constraint += """
[CRITICAL: sphinx Test Constraints]
- Preserve repository-verified @pytest.mark.sphinx decorators and app/status/warning
  fixtures when they already exist in the target style or fixture inventory.
- Do not invent an unavailable Sphinx fixture. If no verified fixture exists,
  use a repository-local helper or a minimal standalone public API construction.
- Never delete a fixture parameter or decorator while retaining body references.
"""

        # pytest-dev 특수 처리: 외부 패키지 import 금지
        if "pytest" in instance.repo.lower():
            framework_constraint += """
[CRITICAL: pytest-dev Test Constraints]
- NEVER import external packages unrelated to pytest (e.g. youtube_dl, roman, requests, etc.)
- Only import from: pytest, _pytest.*, testing.*, and Python standard library
- Use pytester fixture or tmp_path for creating temporary test files
- For skip/PDB issues, create a temporary test file and run pytester; do NOT append a skipped class with undefined names into the repository test file.
- Ensure all f-strings, brackets, and quotes are properly closed (no SyntaxError)
"""

        # django 추가 import 제약 (기존 django-test 블록 이후에 추가)
        if runner == "django-test":
            framework_constraint += """
[CRITICAL: Django Import Rules]
- NEVER import from app names that are NOT listed in [Available Imports from Repository]
  (e.g. `from app import X`, `from myapp import X`, `from myapp1 import X` are FORBIDDEN
   unless that exact module path appears in [Available Imports])
- ONLY import from: django.*, and the exact module paths shown in [Available Imports]
- If a symbol you need is not in [Available Imports], do NOT use it
- For query tests that touch Author.objects or Meta.ordering, mirror existing tests and add explicit .order_by(...) when warnings would become errors.
"""

        if "requests" in instance.repo.lower():
            framework_constraint += """
[CRITICAL: Requests HTTP/Auth Test Constraints]
- Do not make external network calls.
- For digest auth issues, use existing local helper/mock challenge flow when Authorization depends on WWW-Authenticate challenge data.
- A bare PreparedRequest with HTTPDigestAuth but no challenge is usually not enough to test digest header behavior.
"""

        if any(name in instance.repo.lower() for name in ("sphinx", "matplotlib", "seaborn")):
            framework_constraint += """
[CRITICAL: Public Semantic Oracle Constraints]
- Do not inspect private attributes or long raw rendered strings.
- Prefer public artists, axes, legend/text objects, rendered node properties, or minimal semantic markers.
"""

        # 타겟 파일의 실제 test 메서드 예시 (타겟 파일 우선, fallback은 context snippet)
        display_example = target_test_example or test_example
        if display_example:
            display_example_trunc = _clip_prompt_text(
                display_example,
                _PROMPT_TEST_EXAMPLE_CHARS,
                "target_test_example",
                prompt_profile,
            )
            example_source = f"from {target_test_file}" if target_test_example else "from this repository"
            framework_constraint += f"""
[Example: Actual Test Methods {example_source} — Mirror This Style Exactly]
These are REAL test methods from the exact file where your test will be appended.
Follow the same class hierarchy, decorator usage, and assertion style:
```python
{display_example_trunc}
```
"""
            _mark_prompt_section(prompt_profile, "target_test_example")

        runtime_error_section = ""
        if runtime_error_hint:
            hint_trunc = runtime_error_hint[:500] + "…" if len(runtime_error_hint) > 500 else runtime_error_hint
            self_fix = (
                '"missing 1 required positional argument: self" → for Django/TestCase files, put the method inside a TestCase class; '
                'for pytest-style files, remove the self parameter from the top-level test function'
            )
            runtime_error_section = f"""
[Previous Execution Error — MUST FIX]
The previous attempt ran in the test environment but produced this runtime error:
  {hint_trunc}
You MUST fix this error in your new attempt. Common fixes:
- {self_fix}
- "RuntimeError: Model class ... INSTALLED_APPS" → only use model classes already imported from the existing test file; do NOT define new models
- "no such table" → use SimpleTestCase instead of TestCase, or mock DB calls
- "AttributeError" / "ImportError" → verify the attribute/module actually exists in this repo version
"""
        repair_directive_section = self._build_repair_directive_section(prompt_scenario, prompt_profile)
        m5_constraints = _build_m5_feedback_constraints(
            instance_repo=getattr(instance, "repo", ""),
            clue=clue,
            context=context,
            scenario=prompt_scenario,
        )
        mandatory_constraint_section = ""
        if (
            m5_constraints["mandatory_constraints"]
            or m5_constraints["forbidden_patterns"]
            or m5_constraints["repository_local_alternatives"]
        ):
            _mark_prompt_section(prompt_profile, "m5_mandatory_constraints")
            mandatory_constraint_section = (
                "\n[M7/M5 Mandatory Generation Constraints — HIGHEST PRIORITY]\n"
                "These constraints supersede the raw issue example when the raw example would violate repository validation.\n"
                "You MUST obey every mandatory constraint and avoid every forbidden pattern.\n"
                f"{json.dumps(m5_constraints, ensure_ascii=False, indent=2)}\n"
            )

        m2_semantic_evidence = {
            "fault_hypothesis": _clip_prompt_text(
                str(context.get("fault_hypothesis") or prompt_scenario.get("fault_hypothesis", "")),
                900,
                "m2_fault_hypothesis",
                prompt_profile,
            ),
            "oracle_hint": _clip_prompt_text(
                str(context.get("oracle_hint") or prompt_scenario.get("m2_oracle_hint", "")),
                600,
                "m2_oracle_hint",
                prompt_profile,
            ),
        }
        m2_semantic_section = (
            "\n[M2 Semantic Evidence — CONSUME EXPLICITLY]\n"
            f"{json.dumps(_m5_constraint_json_safe(m2_semantic_evidence), ensure_ascii=False, indent=2)}\n"
        )
        _mark_prompt_section(prompt_profile, "m2_semantic_evidence")

        v26_feedback = prompt_scenario.get("feedback_consumed") or prompt_scenario.get("m7_diagnosis")
        v26_feedback_section = ""
        if isinstance(v26_feedback, Mapping) and v26_feedback:
            v26_feedback_payload = {
                "diagnosis": v26_feedback,
                "actual_runtime_evidence": prompt_scenario.get("previous_pass_runtime_evidence") or {},
                "avoid_evidence": prompt_scenario.get("previous_pass_avoid_evidence") or {},
            }
            v26_feedback_section = (
                "\n[M7 Diagnosis From Previous Pass — CONSUME EXPLICITLY]\n"
                "Change the new candidate in accordance with these exact pre-patch diagnosis fields. "
                "Do not repeat prior assertion or stimulus patterns identified in avoid_evidence unless "
                "the diagnosis explicitly proves that preserving one is necessary.\n"
                f"{json.dumps(_m5_constraint_json_safe(v26_feedback_payload), ensure_ascii=False, indent=2)}\n"
            )
            _mark_prompt_section(prompt_profile, "m7_v26_diagnosis")

        prompt = f"""
You are generating a Python issue-reproducing test for a real repository.

Task:
Generate one focused reproduction test for the issue below.
The test must be designed to fail on the pre-patch code because of the issue behavior described.
The test must PASS after the bug is fixed (i.e., the assertion reflects correct/expected behavior).

[Oracle-First Generation Requirement]
Before choosing imports, setup, or helper structure, decide the assertion from
the fixed behavior described by the issue's Expected behavior, expected_outputs,
oracle hints, and oracle contract below. Do not use the observed buggy behavior
as the expected value.
{runtime_error_section}{mandatory_constraint_section}{verified_target_section}{django_grounding_section}{m2_semantic_section}{v26_feedback_section}{repair_directive_section}{framework_constraint}
Repository: {instance.repo}
Instance ID: {instance.instance_id}
Base Commit: {instance.base_commit}

[Issue Clue]
Observed behavior:
{json.dumps(clue.get("observed_behavior", []), ensure_ascii=False, indent=2)}

Expected behavior:
{json.dumps(clue.get("expected_behavior", []), ensure_ascii=False, indent=2)}

Reproduction conditions:
{json.dumps(clue.get("repro_conditions", []), ensure_ascii=False, indent=2)}

Related identifiers:
- functions: {issue_functions}
- classes: {issue_classes}
{f"- error/exception keywords: {issue_error_keywords}" + chr(10) if issue_error_keywords else ""}{self._build_fault_location_section(clue)}{oracle_hint_section}{raw_issue_section}
{issue_code_section}
[Code Context]
Framework: {project_framework} (runner: {runner})
Candidate source files: {source_candidates}
Candidate test files: {test_candidates}
{v30_hypothesis_section}
{v31_contract_section}
{conftest_section}{required_fixtures_section}
[Available Imports from Repository]
CRITICAL: You MUST explicitly import every symbol you use. Do NOT assume anything is pre-imported.
If you use a symbol listed below, you MUST include the corresponding import in your append_block.
Only use imports from the following verified module paths:
{import_map_text}
{existing_imports_section}
[Validated Scenario]
{json.dumps(_truncate_scenario_for_prompt(prompt_scenario), ensure_ascii=False, separators=(',', ':'))}

[Oracle Contract — follow this before writing assertions]
oracle_type: {(prompt_scenario.get("oracle_contract") or {}).get("oracle_type") or prompt_scenario.get("oracle_type", "")}
oracle_source: {(prompt_scenario.get("oracle_contract") or {}).get("oracle_source") or prompt_scenario.get("oracle_source", "")}
rule: {(prompt_scenario.get("oracle_contract") or {}).get("rule", "")}

[CRITICAL: Bug Reproduction Contract]
The test must FAIL on buggy pre-patch code and PASS on fixed post-patch code.
Write assertions from the fixed behavior perspective:
- If [Expected Correct Output] exists, assert that exact value/output. Use np.testing for arrays and pytest.approx for floats.
- If [Expected Correct Output] exists, do not use a fallback repr(result) != buggy assertion; strengthen the expected-output assertion instead.
- If only buggy output is known, assert the function return value or public state is not that buggy value.
- If the fix should remove an exception, assert the success path; do not use pytest.raises for that case.
- If the fix should introduce/correct an exception, pytest.raises must wrap the actual triggering call.
- If the issue explicitly expects a warning, pytest.warns(ExpectedWarning) around the target call is allowed.
- Last resort only: type/non-None/length structural assertions.

Forbidden oracle patterns:
- bug symptom assertions: BUG_STRING in str(exc), exact exception message, raw rendered HTML/LaTeX/Sphinx strings
- negative assertion on a local constant such as expected_matrix/baseline_value/correct_output
- guessed exact expected arrays/values not stated by the issue
- numpy direct equality, @image_comparison, external network calls, private attribute reads, Django inline models
- warning count/type alone unless the issue explicitly expects that warning; otherwise assert returned value/public state

[Generation Constraints]
1. Do not invent issue-irrelevant APIs or identifiers.
2. Prefer using the validated target function and target source file.
3. Prefer inserting into this test file if suitable: {target_test_file}
4. Prefer ONE focused reproduction test function or method. If helpers are needed, helper names must not start with test_.
5. Use appropriate assertions: plain `assert` for pytest/sympy, `self.assert*()` for unittest/Django.
6. The test should reproduce the issue described by the scenario, not a generic failure.
7. NEVER make real network calls. Use PreparedRequest, mock responses, or existing local HTTP helper classes.
8. CRITICAL: Copy the EXACT API call pattern from [Issue Reproduction Code]. If the issue shows
   code like `func(a, b)`, use that exact signature. Do NOT invent call patterns.
   If [Issue Reproduction Code] shows a class/object, instantiate it the same way.
   If it shows nested operators (e.g. &, @, **), matrix assumptions, or warning contexts, preserve them.
   The assertion target MUST be the return value or state change of the function under test —
   never a local constant you defined (e.g., expected_matrix, baseline_value, correct_output).
9. Return JSON only. No explanation.
10. CRITICAL: Only import symbols that actually exist in [Available Imports] or [Existing Imports].
11. NEVER define Django model classes inside tests. Use existing models/imports only.
12. Prefer reusing existing imports from the target test file over adding new ones.
13. CRITICAL: Use the EXACT patterns from [Issue Reproduction Code], including operators and class instantiation.
14. For file-based tools, write content to a tempfile and pass the path.
15. NEVER access private attributes (names starting with _) of library objects. Test through public API.
16. For requests/http issues, NEVER call external URLs. Build and inspect PreparedRequest objects or use existing local test helpers.

[Required Output JSON Schema]
{{
  "target_test_file": "string (relative path to test file)",
  "append_block": "complete Python code to append at the END of the target file"
}}

The append_block will be placed verbatim at the end of the target file.
- Include any NEW imports at the top of append_block (only imports NOT already in [Existing Imports] above)
- Then include the complete focused reproduction test function or class
- Do NOT duplicate imports already shown in [Existing Imports]
- The function/method name must start with: test_
- Prefer one test_* function/method total; if multiple are returned, the pipeline may keep only the strongest reproduction candidate.
- The code must be valid Python
- The code must naturally exercise: {str((verified_target or {}).get("candidate_invocation_expression") or (verified_target or {}).get("issue_api_target") or target_location.get("candidate_invocation_expression") or target_location.get("issue_api_target") or target_location.get("canonical_target_identity") or target_location.get("target_function", ""))}
- The canonical repository identity is M6/M7 runtime-verification context. Do not copy an internal qualified-name string solely to satisfy this prompt.
""".strip()

        if prompt_profile is not None:
            prompt_profile["prompt_chars"] = len(prompt)
            prompt_profile["sections_included"] = _dedup_text_items(
                prompt_profile.get("sections_included", [])
            )
            prompt_profile["truncated_sections"] = _dedup_text_items(
                prompt_profile.get("truncated_sections", [])
            )

        return prompt

    def _parse_model_output(
        self,
        raw_response: str,
        scenario: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        text = raw_response.strip() if isinstance(raw_response, str) else str(raw_response or "")

        # ── 1단계: 코드 펜스 안의 JSON 추출 ──
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        # ── 2단계: JSON 파싱 시도 ──
        data = None
        json_error = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            json_error = str(e)
            data = self._extract_outermost_json(text)

        # ── 3단계: JSON 파싱 실패 → 절단된 JSON 복구 시도 ──
        if data is None:
            data = self._try_repair_truncated_json(raw_response)

        # ── 4단계: 여전히 실패 → Python 코드 직접 추출 ──
        if data is None:
            fallback = self._extract_python_code_fallback(raw_response, scenario)
            if fallback is not None:
                logger.warning("JSON parsing failed (%s); recovered test code via Python fallback.", json_error)
                return fallback
            raise ValueError(f"Model output is not valid JSON and contains no extractable test code. JSON error: {json_error}")

        # ── 이하: data dict 처리 ──
        if not isinstance(data, dict):
            raise ValueError(f"Model output JSON must be an object, got {type(data).__name__}")

        target_test_file = self._coerce_model_string(data.get("target_test_file", ""))
        if not target_test_file:
            target = scenario.get("target_location", {})
            if not isinstance(target, dict):
                target = {}
            target_test_file = self._coerce_model_string(target.get("candidate_test_file")) or (
                scenario.get("relevant_test_files", [""])[0] if scenario.get("relevant_test_files") else ""
            )

        # 새 방식: append_block
        if "append_block" in data:
            append_block = self._coerce_model_code(data["append_block"])
            if not append_block:
                raise ValueError("append_block is empty.")
            # def test_ 없으면 Python fallback으로 재시도
            if "def test_" not in append_block:
                fallback = self._extract_python_code_fallback(raw_response, scenario)
                if fallback is not None:
                    logger.warning("append_block has no test function; recovered via Python fallback.")
                    return fallback
                raise ValueError("append_block has no valid test function.")
            # 기존 테스트와 이름 충돌 방지: test 함수명에 _repro 접미사 보장
            append_block = _ensure_repro_suffix(append_block)
            self._require_parseable_test_block(append_block)
            verified = context.get("verified_target_evidence") if isinstance(context, dict) else {}
            required_invocation = self._required_generation_invocation(
                scenario=scenario,
                verified_target=verified,
            )
            if required_invocation and not self._check_target_function_presence(required_invocation, append_block):
                raise ValueError(
                    "Generated test does not exercise the issue-grounded invocation "
                    f"'{required_invocation}'. Regenerate the stimulus against the issue API."
                )
            return {
                "target_test_file": target_test_file,
                "insert_mode": "append_block",
                "insertion_hint": "end_of_file",
                "imports": [],
                "test_code": append_block,  # 하위 호환용
                "append_block": append_block,
            }

        # 구 방식: imports + test_code → append_block으로 통합 처리
        imports = data.get("imports", [])
        test_code = self._coerce_model_code(data.get("test_code", "")).rstrip()

        if not isinstance(imports, list):
            imports = []
        imports = [x.strip() for x in imports if isinstance(x, str) and x.strip()]

        if not test_code or "def test_" not in test_code:
            raise ValueError("Generated test_code has no valid test function.")
        self._require_parseable_test_block(test_code)

        verified = context.get("verified_target_evidence") if isinstance(context, dict) else {}
        required_invocation = self._required_generation_invocation(
            scenario=scenario,
            verified_target=verified,
        )
        if required_invocation and not self._check_target_function_presence(required_invocation, test_code):
            raise ValueError(
                "Generated test does not exercise the issue-grounded invocation "
                f"'{required_invocation}'. Regenerate the stimulus against the issue API."
            )

        # imports + test_code를 하나의 append_block으로 합침
        parts = []
        if imports:
            parts.extend(imports)
            parts.append("")
        parts.append(test_code)
        append_block = "\n".join(parts)

        return {
            "target_test_file": target_test_file,
            "insert_mode": "append_block",
            "insertion_hint": "end_of_file",
            "imports": imports,
            "test_code": test_code,
            "append_block": append_block,
        }

    @staticmethod
    def _coerce_model_string(value: Any) -> str:
        """LLM이 string 필드에 dict/list를 넣어도 가능한 문자열만 뽑는다."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for key in ("path", "file", "target_test_file", "value", "text", "code"):
                if key in value:
                    coerced = ReproductionTestGenerator._coerce_model_string(value[key])
                    if coerced:
                        return coerced
            return ""
        if isinstance(value, list):
            parts = [
                ReproductionTestGenerator._coerce_model_string(item)
                for item in value
            ]
            return "\n".join(part for part in parts if part).strip()
        return str(value).strip()


    @staticmethod
    def _coerce_model_code(value: Any) -> str:
        """append_block/test_code 필드를 Python 코드 문자열로 정규화한다."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for key in ("code", "append_block", "test_code", "content", "text", "value"):
                if key in value:
                    coerced = ReproductionTestGenerator._coerce_model_code(value[key])
                    if coerced:
                        return coerced
            return ""
        if isinstance(value, list):
            parts = [
                ReproductionTestGenerator._coerce_model_code(item)
                for item in value
            ]
            return "\n".join(part for part in parts if part).strip()
        return str(value).strip()

    @staticmethod
    def _extract_outermost_json(text: str) -> Optional[dict]:
        """텍스트에서 가장 바깥쪽 { ... } 블록을 brace-depth counting으로 추출 후 JSON 파싱."""
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    @staticmethod
    def _try_repair_truncated_json(text: str) -> Optional[dict]:
        """max_tokens 초과로 JSON이 중간에 잘린 경우 복구 시도.

        target_test_file과 append_block의 시작을 찾아, 잘린 Python 코드라도 추출한다.
        """
        file_match = re.search(r'"target_test_file"\s*:\s*"([^"]+)"', text)
        if not file_match:
            return None
        target_file = file_match.group(1)

        block_start = re.search(r'"append_block"\s*:\s*"', text)
        if not block_start:
            return None

        raw = text[block_start.end():]
        decoded: List[str] = []
        i = 0
        while i < len(raw):
            ch = raw[i]
            if ch == "\\" and i + 1 < len(raw):
                nc = raw[i + 1]
                mapping = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "'": "'", "r": "\r"}
                decoded.append(mapping.get(nc, nc))
                i += 2
            elif ch == '"':
                break  # JSON 문자열 종료
            else:
                decoded.append(ch)
                i += 1

        code = "".join(decoded).strip()
        if not code or "def test_" not in code:
            return None

        try:
            ast.parse(code)
        except SyntaxError:
            logger.warning(
                "Rejected truncated JSON recovery because append_block is syntactically incomplete."
            )
            return None

        logger.warning("Repaired truncated JSON: recovered %d chars of append_block.", len(code))
        return {
            "target_test_file": target_file,
            "insert_mode": "append_block",
            "insertion_hint": "end_of_file",
            "imports": [],
            "test_code": code,
            "append_block": code,
        }

    @staticmethod
    def _extract_python_code_fallback(text: str, scenario: Dict[str, Any]) -> Optional[dict]:
        """LLM이 JSON 대신 Python 코드를 직접 출력했거나 코드 펜스만 있는 경우 처리.

        우선순위:
        1. ```python ... ``` 코드 펜스 안의 test 코드
        2. 응답 전체에서 def test_ / class Test 로 시작하는 블록
        """
        target_location = _scenario_target_location(scenario)
        target_test_file = target_location.get("candidate_test_file") or (
            scenario.get("relevant_test_files", [""])[0] if scenario.get("relevant_test_files") else ""
        )

        def _make_result(code: str) -> dict:
            return {
                "target_test_file": target_test_file,
                "insert_mode": "append_block",
                "insertion_hint": "end_of_file",
                "imports": [],
                "test_code": code,
                "append_block": code,
            }

        # 1) Python 코드 펜스
        for m in re.finditer(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL):
            code = m.group(1).strip()
            if "def test_" in code:
                return _make_result(code)

        # 2) 텍스트 전체에서 def test_ / class Test 블록 추출
        m = re.search(r"^(?:def test_|class Test)", text, re.MULTILINE)
        if m:
            code = text[m.start():].strip()
            # 이후 불필요한 설명 텍스트 제거: 연속된 non-indented non-def/class 줄이 나오면 자름
            lines = code.splitlines()
            kept: List[str] = []
            for line in lines:
                if kept and line and not line[0].isspace() and not line.startswith(("def ", "class ", "@", "#")):
                    break
                kept.append(line)
            code = "\n".join(kept).strip()
            if "def test_" in code:
                return _make_result(code)

        return None

    @staticmethod
    def _require_parseable_test_block(test_code: str) -> None:
        """Reject truncated recovery unless Python and a real test both parse."""
        try:
            tree = ast.parse(str(test_code or ""))
        except SyntaxError as exc:
            raise ValueError(f"Recovered append_block is not valid Python: {exc.msg}") from exc
        if not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
            for node in ast.walk(tree)
        ):
            raise ValueError("Recovered append_block contains no parsed test function.")

    @staticmethod
    def _required_generation_invocation(
        *,
        scenario: Mapping[str, Any],
        verified_target: Mapping[str, Any] | None,
    ) -> str:
        """Return the natural issue-facing call expression M5 must exercise.

        M5 validates the stimulus spelling. M6/M7, not source-string matching,
        owns proof that this call reached the canonical repository callable.
        """
        verified = verified_target if isinstance(verified_target, Mapping) else {}
        target = _scenario_target_location(dict(scenario))
        candidate = str(
            verified.get("candidate_invocation_expression")
            or scenario.get("candidate_invocation_expression")
            or target.get("candidate_invocation_expression")
            or ""
        ).strip()
        issue_api = str(
            verified.get("issue_api_target")
            or scenario.get("issue_api_target")
            or target.get("issue_api_target")
            or ""
        ).strip()
        canonical = str(
            verified.get("canonical_target_identity")
            or verified.get("target_callable")
            or target.get("canonical_target_identity")
            or target.get("target_function")
            or ""
        ).strip()
        for value in (candidate, issue_api):
            if value and value != canonical:
                return value
        return ""

    @staticmethod
    def _check_target_function_presence(target_function: str, test_code: str) -> bool:
        """Check if the target function appears in test code (flexible).

        Dunder methods are invoked indirectly (e.g. obj() → __call__), so we
        only skip those.  Everything else is checked via string containment or
        a call-pattern regex.
        """
        target_function = str(target_function or "").strip()
        simple_target = target_function.split(".")[-1]

        # dunder methods: used indirectly (e.g. obj() → __call__)
        if simple_target.startswith("__") and simple_target.endswith("__"):
            return True

        # very short names (<=2 chars) cause too many false positives
        if len(simple_target) <= 2:
            return True

        # A repository-valid local receiver can differ from the issue spelling;
        # requiring the callable component is enough at generation time.
        if target_function in test_code or simple_target in test_code:
            return True

        # for private methods, also check the bare name without leading underscores
        if simple_target.startswith("_"):
            bare = simple_target.lstrip("_")
            if bare and bare in test_code:
                return True

        # function call pattern: .func_name( or func_name(
        call_pattern = re.compile(
            r'(?:^|[.\s(,=])' + re.escape(simple_target) + r'\s*\(',
            re.MULTILINE,
        )
        if call_pattern.search(test_code):
            return True

        return False

    def _validate_generated_code(
        self,
        parsed: Dict[str, Any],
        repo_path: str,
        context: Dict[str, Any],
        clue: Optional[Dict[str, Any]] = None,
        scenario: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        생성된 코드에 대해 정적 검증을 수행한다.
        1) 구문 검증 (ast.parse)
        2) import 경로가 repo 내에 존재하는지 확인
        3) test_code에서 사용하지 않는 import 제거
        4) 이슈 코드 예시의 핵심 식별자가 테스트에 포함되는지 soft 검증
        잘못된 import를 교정할 수 있으면 fixed_imports를 반환한다.
        """
        errors: List[str] = []
        warnings: List[str] = []
        imports = list(parsed.get("imports", []))
        test_code = parsed.get("test_code", "")
        target_test_file = parsed.get("target_test_file", "")

        repo = Path(repo_path)
        available_imports = context.get("available_imports", {})
        target_file_errors = _target_file_constraint_errors(
            target_test_file, scenario, repo
        )
        if target_file_errors:
            return ValidationResult(is_valid=False, errors=target_file_errors)

        # append_block 방식: 단순 append 후 구문 검증
        if parsed.get("insert_mode") == "append_block":
            append_block = parsed.get("append_block", "")
            try:
                ast.parse(append_block)
            except SyntaxError as e:
                errors.append(f"append_block SyntaxError: {e}")
                return ValidationResult(is_valid=False, errors=errors)
            test_count = _count_generated_tests(append_block)
            if test_count == 0:
                errors.append(
                    "CRITICAL: generated append_block must define at least one test function/method."
                )
                return ValidationResult(is_valid=False, errors=errors)
            if test_count > 1:
                errors.append(
                    f"CRITICAL: generated append_block defines {test_count} test functions/methods; "
                    "keep exactly one focused reproduction test."
                )
                return ValidationResult(is_valid=False, errors=errors)
            if _has_unsupported_parameterized_test(append_block):
                errors.append(
                    "PARAMETERIZED_TEST_UNSUPPORTED: v31 requires exactly one concrete test item."
                )
                return ValidationResult(is_valid=False, errors=errors)
            directive_violations = _detect_repair_directive_violations(
                append_block,
                scenario,
            )
            if directive_violations:
                errors.extend(
                    "CRITICAL: repair directive forbidden pattern repeated: " + v
                    for v in directive_violations
                )
                return ValidationResult(is_valid=False, errors=errors)
            test_file_abs = repo / target_test_file if target_test_file else None
            original_content = ""
            if test_file_abs and test_file_abs.exists():
                original_content = read_text(test_file_abs)
                trial_content = original_content.rstrip() + "\n\n" + append_block + "\n"
                try:
                    ast.parse(trial_content)
                except SyntaxError as e:
                    errors.append(f"appended file SyntaxError: {e}")
                    return ValidationResult(is_valid=False, errors=errors)
            fixture_errors = _fixture_parameter_errors(
                append_block,
                original_content=original_content,
                context=context,
                target_test_file=target_test_file,
            )
            if fixture_errors:
                return ValidationResult(is_valid=False, errors=fixture_errors)
            preflight_errors = _append_block_preflight_errors(
                append_block,
                original_content,
                repo,
                {**context, "_target_test_file": target_test_file},
                available_imports,
                self._check_import_validity,
            )
            if preflight_errors:
                errors.extend(preflight_errors)
                return ValidationResult(is_valid=False, errors=errors)
            # 관용 alias가 import 없이 쓰이면 invalid 처리
            undefined = _detect_missing_common_aliases(append_block, original_content)
            if undefined:
                errors.append(
                    f"Missing imports for common aliases: {undefined}. "
                    "Add explicit import statements at the top of append_block."
                )
                return ValidationResult(is_valid=False, errors=errors)
            oracle_risks = _detect_blocking_oracle_risks(append_block, clue=clue)
            if oracle_risks and getattr(self, "feature_profile", None) != "v37":
                errors.extend(oracle_risks)
                return ValidationResult(is_valid=False, errors=errors)
            if oracle_risks:
                warnings.extend(f"DIAGNOSTIC_ONLY: {risk}" for risk in oracle_risks)
            semantic_risks = _detect_semantic_risk_flags(
                append_block,
                clue,
                context,
                scenario,
                original_content=original_content,
            )
            if semantic_risks and getattr(self, "feature_profile", None) != "v37":
                errors.extend(f"CRITICAL: semantic risk: {risk}" for risk in semantic_risks)
                return ValidationResult(is_valid=False, errors=errors)
            if semantic_risks:
                warnings.extend(
                    f"DIAGNOSTIC_ONLY: semantic risk: {risk}" for risk in semantic_risks
                )
            reproduction_drifts = _detect_issue_reproduction_drift(
                append_block,
                clue,
                scenario,
                context,
            )
            if reproduction_drifts:
                warnings.extend(f"SOFT: {risk}" for risk in reproduction_drifts)
            return ValidationResult(is_valid=True, errors=[], warnings=warnings, fixed_imports=None)

        # 구 방식: 1) 구문 검증 — test_code 자체
        try:
            ast.parse(test_code)
        except SyntaxError as e:
            errors.append(f"test_code SyntaxError: {e}")
            return ValidationResult(is_valid=False, errors=errors)
        test_count = _count_generated_tests(test_code)
        if test_count == 0:
            errors.append(
                "CRITICAL: generated test_code must define at least one test function/method."
            )
            return ValidationResult(is_valid=False, errors=errors)
        if test_count > 1 and getattr(self, "feature_profile", None) != "v37":
            errors.append(
                f"CRITICAL: generated test_code defines {test_count} test functions/methods; "
                "keep exactly one focused reproduction test."
            )
            return ValidationResult(is_valid=False, errors=errors)
        if getattr(self, "feature_profile", None) != "v37" and _has_unsupported_parameterized_test(test_code):
            errors.append(
                "PARAMETERIZED_TEST_UNSUPPORTED: v31 requires exactly one concrete test item."
            )
            return ValidationResult(is_valid=False, errors=errors)

        # 2) 전체 파일 구문 검증 (imports + test_code)
        test_file_abs = repo / target_test_file if target_test_file else None
        original_content = ""
        if test_file_abs and test_file_abs.exists():
            original_content = read_text(test_file_abs)
            trial_content = self._build_modified_test_file_content(
                original_content=original_content,
                imports=imports,
                test_code=test_code,
            )
            try:
                ast.parse(trial_content)
            except SyntaxError as e:
                errors.append(f"modified file SyntaxError: {e}")
                return ValidationResult(is_valid=False, errors=errors)
        fixture_errors = _fixture_parameter_errors(
            test_code,
            original_content=original_content,
            context=context,
            target_test_file=target_test_file,
        )
        if fixture_errors:
            return ValidationResult(is_valid=False, errors=fixture_errors)

        # 3) import 검증 — 각 import 문의 심볼이 repo에 존재하는지
        validated_imports: List[str] = []
        import_errors: List[str] = []

        # 기존 테스트 파일의 import를 수집 (이미 있는 건 검증 skip)
        existing_imports: set = set()
        if test_file_abs and test_file_abs.exists():
            original_content = read_text(test_file_abs)
            for line in original_content.splitlines():
                if (line.startswith("import ") or line.startswith("from ")) and not line.startswith((" ", "\t")):
                    existing_imports.add(line.strip())

        # unittest 자동 주입: test_code에서 unittest.를 사용하는데 import가 없으면 추가
        if "unittest." in test_code or "(unittest.TestCase)" in test_code:
            has_unittest = (
                "import unittest" in existing_imports
                or any("import unittest" in imp for imp in imports)
            )
            if not has_unittest:
                imports = ["import unittest"] + imports

        for imp in imports:
            # 기존 파일에 이미 있는 import는 중복이므로 제거
            if imp in existing_imports:
                continue

            import_context = dict(context or {})
            import_context["_target_test_file"] = target_test_file
            check = _coerce_import_check_result(
                self._check_import_validity(
                    imp, repo, available_imports, import_context
                )
            )
            if check.is_valid or check.is_unknown:
                validated_imports.append(imp)
                if check.is_unknown:
                    import_errors.append(f"import path unverifiable (keeping, in use): {imp}")
            elif check.is_correctable:
                # 교정된 import
                validated_imports.append(check.corrected)
                import_errors.append(f"corrected: '{imp}' -> '{check.corrected}'")
            else:
                # 사용 여부 확인: test_code에서 실제 사용하는 심볼인지
                symbols = self._extract_imported_symbols(imp)
                used = any(sym in test_code for sym in symbols)
                if used:
                    import_errors.append(f"import path unverifiable (keeping, in use): {imp}")
                    validated_imports.append(imp)
                else:
                    import_errors.append(f"removed unused/unverifiable import: {imp}")

        if import_errors:
            errors.extend(import_errors)

        # 4) TestCase 상속 검증.  Only Django's runner strictly requires
        # class-based test methods here; several pytest-collected repositories
        # are classified as "unittest" because they contain TestCase classes
        # while still accepting top-level test functions.
        runner = context.get("project_test_style", {}).get("runner", "pytest")
        if runner == "django-test":
            try:
                tree = ast.parse(test_code)
            except SyntaxError:
                pass  # 이미 위에서 SyntaxError로 처리됨
            else:
                has_testcase_class = False
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                base_name = base.id
                            elif isinstance(base, ast.Attribute):
                                base_name = base.attr
                            else:
                                base_name = ""
                            if "TestCase" in base_name or "SimpleTestCase" in base_name:
                                has_testcase_class = True
                                break

                standalone_test_funcs = [
                    node.name
                    for node in ast.iter_child_nodes(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name.startswith("test")
                ]

                if not has_testcase_class:
                    func_hint = (
                        f" (found standalone functions: {', '.join(standalone_test_funcs[:2])})"
                        if standalone_test_funcs else ""
                    )
                    errors.append(
                        f"CRITICAL: {runner} runner requires a class inheriting from "
                        f"django.test.TestCase or SimpleTestCase.{func_hint} "
                        "Wrap all test methods inside a TestCase subclass."
                    )

        # 5) 이슈 코드 예시 alignment soft 검증
        if clue:
            self._soft_validate_issue_alignment(test_code, clue, errors)

        blocking_oracle_risks = _detect_blocking_oracle_risks(test_code, clue=clue)
        semantic_risks = _detect_semantic_risk_flags(
            test_code,
            clue,
            context,
            scenario,
            original_content=original_content if test_file_abs and test_file_abs.exists() else "",
        )
        if getattr(self, "feature_profile", None) == "v37":
            warnings.extend(
                f"DIAGNOSTIC_ONLY: {risk}" for risk in blocking_oracle_risks
            )
            warnings.extend(
                f"DIAGNOSTIC_ONLY: semantic risk: {risk}" for risk in semantic_risks
            )
        else:
            errors.extend(blocking_oracle_risks)
            errors.extend(
                f"CRITICAL: semantic risk: {risk}" for risk in semantic_risks
            )
        warnings.extend(
            f"SOFT: {risk}"
            for risk in _detect_issue_reproduction_drift(test_code, clue, scenario, context)
        )

        # critical error가 있으면 실패 (SyntaxError, TestCase 미상속 등)
        has_critical = any("SyntaxError" in e or "CRITICAL:" in e for e in errors)

        return ValidationResult(
            is_valid=not has_critical,
            errors=errors,
            warnings=warnings,
            fixed_imports=validated_imports,
        )

    def _soft_validate_issue_alignment(
        self,
        test_code: str,
        clue: Dict[str, Any],
        errors: List[str],
    ) -> None:
        """이슈 원문의 핵심 식별자가 생성된 테스트에 포함되는지 soft 검증 (경고만)."""
        code_examples = clue.get("code_examples", [])
        if not code_examples:
            return

        # 코드 예시에서 class/function 호출 패턴 추출
        issue_identifiers: set = set()
        for block in code_examples:
            code = block.get("code", "") + " " + block.get("interactive_input", "")
            # ClassName( 패턴
            issue_identifiers.update(re.findall(r'\b([A-Z][A-Za-z0-9_]+)\s*\(', code))
            # function_call( 패턴
            issue_identifiers.update(re.findall(r'\b([a-z_][a-z0-9_]+)\s*\(', code))

        # 너무 일반적인 식별자 제외
        generic = {"print", "range", "len", "type", "str", "int", "float", "list",
                   "dict", "set", "tuple", "isinstance", "assert", "True", "False",
                   "None", "array", "import", "from", "def", "class", "return"}
        issue_identifiers -= generic

        if not issue_identifiers:
            return

        found = {ident for ident in issue_identifiers if ident in test_code}
        missing = issue_identifiers - found

        if found:
            hit_ratio = len(found) / len(issue_identifiers)
            if hit_ratio < 0.3 and len(missing) > 2:
                logger.warning(
                    "[soft-validation] issue code identifier alignment low: %.0f%% (%d/%d). "
                    "missing: %s",
                    hit_ratio * 100, len(found), len(issue_identifiers),
                    ", ".join(sorted(missing)[:10]),
                )
                errors.append(
                    f"[warning] issue code identifier alignment low: "
                    f"{len(found)}/{len(issue_identifiers)} matched. "
                    f"missing: {', '.join(sorted(missing)[:5])}"
                )
        else:
            logger.warning(
                "[soft-validation] no issue code identifiers found in test: %s",
                ", ".join(sorted(issue_identifiers)[:10]),
            )
            errors.append(
                f"[warning] no issue code identifiers found in test: "
                f"{', '.join(sorted(issue_identifiers)[:5])}"
            )

    def _check_import_validity(
        self,
        import_line: str,
        repo: Path,
        available_imports: Dict[str, List[str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> ImportCheckResult:
        """
        import 문이 repo 내에서 유효한지 확인한다.
        Returns:
            ImportCheckResult(status):
              valid: 정적으로 통과
              invalid: 실행 전 차단해야 하는 명백한 실패
              correctable: 실제 파일 export로 안전한 교정 가능
              unknown: 정적으로 증명 불가하므로 실행 단계로 보냄
        """
        stripped = import_line.strip()

        # Generated tests are appended to a repository package module, so a
        # relative import is resolved against that exact target package.
        if stripped.startswith("from ."):
            target_test_file = str((context or {}).get("_target_test_file") or "")
            if not target_test_file:
                return _unknown_import("relative import requires target package context")
            match = re.match(
                r"from\s+(?P<dots>\.+)(?P<module>[\w.]*)\s+import\s+(?P<names>.*)",
                stripped,
            )
            if not match:
                return _invalid_import("malformed relative import")
            package_parts = list(Path(target_test_file).parent.parts)
            ascend = len(match.group("dots")) - 1
            if not package_parts or any(
                not repo.joinpath(*package_parts[:index], "__init__.py").is_file()
                for index in range(1, len(package_parts) + 1)
            ):
                return _invalid_import("target test module is not in a repository package")
            if ascend >= len(package_parts):
                return _invalid_import("relative import traverses above repository package")
            base_parts = package_parts[: len(package_parts) - ascend]
            module_suffix = [
                part for part in match.group("module").split(".") if part
            ]
            module_parts = base_parts + module_suffix
            module_file = repo.joinpath(*module_parts).with_suffix(".py")
            package_init = repo.joinpath(*module_parts, "__init__.py")
            target_file = module_file if module_file.is_file() else package_init
            if not target_file.is_file():
                return _invalid_import("relative repository module does not exist")
            names = [
                name.strip().split(" as ")[0].strip()
                for name in match.group("names").split(",")
                if name.strip()
            ]
            if names == ["*"]:
                return _valid_import("relative repository module exists")
            exported = _collect_module_exported_names(target_file)
            missing = [name for name in names if name not in exported]
            if missing:
                return _invalid_import(
                    "relative repository symbols missing: " + ", ".join(missing)
                )
            return _valid_import("relative repository module and symbols exist")

        # "import X" or "from X import Y" 에서 최상위 모듈 추출
        if stripped.startswith("from "):
            match = re.match(r"from\s+([\w.]+)\s+import\s+(.*)", stripped)
            if not match:
                return _unknown_import("could not parse from-import")
            module_path = match.group(1)
            import_names = [n.strip().split(" as ")[0].strip() for n in match.group(2).split(",")]
        elif stripped.startswith("import "):
            match = re.match(r"import\s+([\w.]+)", stripped)
            if not match:
                return _unknown_import("could not parse import")
            module_path = match.group(1)
            import_names = []
        else:
            return _valid_import("not an import statement")

        top_module = module_path.split(".")[0]

        if top_module in _STDLIB_IMPORT_ROOTS:
            return _valid_import("stdlib import")
        if _is_pytest_dev_repo(repo) and top_module not in _PYTEST_DEV_ALLOWED_IMPORT_ROOTS:
            return _invalid_import(
                f"pytest-dev repository only allows stdlib, pytest, _pytest, and testing imports; got {top_module}"
            )
        repo_owns_top = (
            top_module in _REPO_OWNED_IMPORT_ROOTS
            or _repo_contains_top_module(repo, top_module)
        )
        if top_module in _EXTERNAL_IMPORT_ROOTS and not repo_owns_top:
            return _valid_import("external package import")
        # ``packaging`` is a declared project dependency rather than a
        # repository-owned module.  Ground it with the current environment
        # contract instead of allowing arbitrary installed roots.
        if top_module == "packaging" and not repo_owns_top:
            if importlib.util.find_spec(top_module) is not None:
                return _valid_import("declared project dependency import")
            return _unknown_import("declared dependency is not importable in the current environment")

        target_file = _module_file_for_import(repo, module_path)

        # available_imports에서 확인.  Repo-owned modules with a visible file
        # still need file-level export confirmation before from-imports are
        # treated as proven; available_imports can be broad context rather than
        # a real re-export contract.
        if module_path in available_imports:
            available_symbols = set(available_imports[module_path])
            if not import_names:
                return _valid_import("module listed in available_imports")
            submodule_names = [
                n for n in import_names
                if _submodule_exists_for_from_import(repo, module_path, n)
            ]
            missing = [n for n in import_names if n not in available_symbols and n not in submodule_names]
            file_exports: set[str] = set()
            if missing and repo_owns_top and target_file is not None:
                file_exports = _collect_module_exported_names(target_file)
                missing = [n for n in missing if n not in file_exports]
            if not missing:
                if submodule_names:
                    return _valid_import("from-imported repository submodule exists")
                if repo_owns_top and target_file is not None:
                    defined = file_exports or _collect_module_exported_names(target_file)
                    if defined and all(n in defined for n in import_names):
                        return _valid_import("symbol exported by repository file")
                    if defined and all(
                        n in available_symbols or n in defined for n in import_names
                    ):
                        return _valid_import(
                            "symbols proven by available imports and repository file"
                        )
                    if defined:
                        return _unknown_import(
                            f"{module_path} is listed in available_imports, but file exports do not prove {', '.join(import_names)}"
                        )
                return _valid_import("symbol listed in available_imports")
            corrected = _find_verified_import_alternative(
                repo,
                import_names,
                available_imports,
                original_module=module_path,
            )
            if corrected:
                return _correctable_import(corrected, "verified alternative module export")
            # 사용된 심볼만 유효한 것으로 필터
            valid_names = [
                n for n in import_names
                if n in available_symbols or n in file_exports
            ]
            if valid_names and len(valid_names) < len(import_names):
                corrected = f"from {module_path} import {', '.join(valid_names)}"
                return _correctable_import(corrected, "drop symbols absent from available_imports")
            if repo_owns_top and target_file is not None:
                defined = file_exports or _collect_module_exported_names(target_file)
                if defined:
                    if _module_has_dynamic_exports(target_file):
                        return _unknown_import(
                            f"{module_path} has dynamic exports; missing symbols are not proven invalid"
                        )
                    return _invalid_import(
                        f"symbol not exported by {module_path}: {', '.join(missing)}"
                    )
                return _unknown_import(
                    f"{module_path} exists, but symbol export is not statically proven"
                )
            return _invalid_import(f"symbol not found in available imports: {', '.join(missing)}")

        # repo 파일 시스템에서 모듈 존재 여부
        if target_file is not None:
            # 모듈 파일은 존재 — from X import Y 인 경우 심볼도 검증
            if import_names:
                if import_names == ["*"]:
                    return _valid_import("star import from existing module")
                if all(_submodule_exists_for_from_import(repo, module_path, n) for n in import_names):
                    return _valid_import("from-imported repository submodule exists")
                defined = _collect_module_exported_names(target_file)
                if defined:
                    missing = [n for n in import_names if n not in defined]
                    if missing:
                        submodule_missing = [
                            n for n in missing
                            if _submodule_exists_for_from_import(repo, module_path, n)
                        ]
                        if len(submodule_missing) == len(missing):
                            return _valid_import("from-imported repository submodule exists")
                        valid = [n for n in import_names if n in defined]
                        if valid:
                            corrected = f"from {module_path} import {', '.join(valid)}"
                            return _correctable_import(corrected, "drop missing symbols from existing module")
                        corrected = _find_verified_import_alternative(
                            repo,
                            import_names,
                            available_imports,
                            original_module=module_path,
                        )
                        if corrected:
                            return _correctable_import(corrected, "verified alternative module export")
                        if _module_has_dynamic_exports(target_file):
                            return _unknown_import(
                                f"{module_path} has dynamic exports; missing symbols are not proven invalid"
                            )
                        return _invalid_import(
                            f"symbol not exported by {module_path}: {', '.join(missing)}"
                        )
                elif repo_owns_top:
                    corrected = _find_verified_import_alternative(
                        repo,
                        import_names,
                        available_imports,
                        original_module=module_path,
                    )
                    if corrected:
                        return _correctable_import(corrected, "verified alternative module export")
                    return _unknown_import(
                        f"{module_path} exists, but exported names could not be determined"
                    )
            return _valid_import("repository module exists")

        # 부모 모듈에서 찾기
        if "." in module_path:
            parent = module_path.rsplit(".", 1)[0]
            parent_file = _module_file_for_import(repo, parent)
            if parent_file is not None and import_names:
                parent_exports = _collect_module_exported_names(parent_file)
                if parent_exports and all(n in parent_exports for n in import_names):
                    corrected = f"from {parent} import {', '.join(import_names)}"
                    return _correctable_import(corrected, "verified parent module export")

        if repo_owns_top:
            return _invalid_import(f"module not found in repository: {module_path}")
        return _unknown_import(f"module not found in static context: {module_path}")

    def _extract_imported_symbols(self, import_line: str) -> List[str]:
        """import 문에서 가져오는 심볼 이름 추출"""
        stripped = import_line.strip()
        if stripped.startswith("from "):
            match = re.match(r"from\s+[\w.]+\s+import\s+(.*)", stripped)
            if match:
                parts = match.group(1).split(",")
                symbols = []
                for p in parts:
                    p = p.strip()
                    if " as " in p:
                        symbols.append(p.split(" as ")[-1].strip())
                    else:
                        symbols.append(p.strip())
                return symbols
        elif stripped.startswith("import "):
            match = re.match(r"import\s+([\w.]+)(?:\s+as\s+(\w+))?", stripped)
            if match:
                alias = match.group(2) or match.group(1).split(".")[-1]
                return [alias]
        return []

    def _extract_test_examples_from_file(
        self,
        file_path: str,
        n: int = 2,
        max_lines_per: int = 30,
        preferred_terms: Optional[Sequence[str]] = None,
    ) -> str:
        """타겟 테스트 파일에서 완성된 test_ 메서드 n개를 추출 (decorator 포함).

        LLM이 타겟 파일의 실제 클래스 구조, decorator 패턴, assert 스타일을 직접 볼 수 있도록
        한다. 이를 통해 import 삽입 위치 오류(decorator 앞 삽입 등)를 방지한다.
        """
        path = Path(file_path)
        if not path.exists():
            return ""
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except Exception:
            return ""

        lines = src.splitlines()
        ranked_snippets: List[tuple[int, int, str]] = []
        terms = [str(term).lower() for term in (preferred_terms or []) if str(term).strip()]

        def add_snippet(snippet: str, order: int) -> None:
            lower = snippet.lower()
            score = sum(1 for term in terms if term in lower)
            ranked_snippets.append((score, order, snippet))

        # TestClass 내부 test_ 메서드 우선 (ast.walk는 순서 보장 안 됨 → 직접 순회)
        # "Test"로 시작하거나 "Tests"로 끝나는 클래스 포함
        order = 0
        for node in tree.body:  # top-level 문장만
            if isinstance(node, ast.ClassDef) and (
                node.name.startswith("Test") or node.name.endswith("Tests") or node.name.endswith("TestCase")
            ):
                class_header = lines[node.lineno - 1]  # e.g. "class TestFoo(TestCase):"
                for item in node.body:
                    if (
                        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and item.name.startswith("test_")
                    ):
                        # decorator부터 포함
                        start = (
                            item.decorator_list[0].lineno - 1
                            if item.decorator_list
                            else item.lineno - 1
                        )
                        end = min(start + max_lines_per, len(lines))
                        method_lines = "\n".join(lines[start:end])
                        # 클래스 컨텍스트(헤더 한 줄)도 포함
                        snippet = f"{class_header}\n    ...\n{method_lines}"
                        add_snippet(snippet, order)
                        order += 1

        # TestClass 없으면 standalone test_ 함수
        if not ranked_snippets:
            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name.startswith("test_")
                ):
                    start = (
                        node.decorator_list[0].lineno - 1
                        if node.decorator_list
                        else node.lineno - 1
                    )
                    end = min(start + max_lines_per, len(lines))
                    add_snippet("\n".join(lines[start:end]), order)
                    order += 1

        ranked_snippets.sort(key=lambda item: (-item[0], item[1]))
        return "\n\n".join(item[2] for item in ranked_snippets[:n])

    def _extract_import_block(self, file_path: Path) -> str:
        """파일에서 top-level import 블록만 추출"""
        try:
            content = read_text(file_path)
        except Exception:
            return ""

        lines = content.splitlines()
        import_lines: List[str] = []
        seen_import = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and not line.startswith((" ", "\t")):
                import_lines.append(line)
                seen_import = True
            elif stripped == "" or stripped.startswith("#"):
                if seen_import:
                    import_lines.append(line)
            else:
                if seen_import:
                    break

        return "\n".join(import_lines).strip()

    def _build_fix_prompt(
        self,
        original_prompt: str,
        previous_response: str,
        previous_parsed: Optional[Dict[str, Any]],
        error_message: str,
        attempt: int,
        scenario: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """이전 시도의 에러를 포함한 compact 수정 요청 프롬프트.

        Retry에서 original prompt 전체를 반복하면 토큰이 기하급수적으로 증가한다.
        여기서는 task summary, oracle contract, 핵심 에러, 이전 응답 일부만 전달한다.
        """
        # JSON 파싱 실패인지 감지 → 더 구체적인 포맷 지침 추가
        is_json_fail = (
            "not valid JSON" in error_message
            or "JSONDecodeError" in error_message
            or "Model output parsing" in error_message
        )
        if is_json_fail:
            format_hint = """
[CRITICAL FORMAT REQUIREMENT]
Your previous response could not be parsed as JSON. You MUST return ONLY a valid JSON object.
- Do NOT write any explanation before or after the JSON.
- Do NOT use markdown outside the JSON value.
- The "append_block" value must be a JSON string: escape newlines as \\n and quotes as \\".
- Example of correct response format:
  {"target_test_file": "path/to/test_file.py", "append_block": "def test_foo():\\n    assert bar() == 1\\n"}
- If your test code is long, keep it concise to avoid truncation.
"""
        else:
            format_hint = ""

        oracle_retry_hint = ""
        if (
            "retry required" in error_message
            or "oracle" in error_message.lower()
            or "issue_reproduction_code_not_followed" in error_message
        ):
            oracle_retry_hint = self._oracle_rewrite_hint(error_message)

        directive_retry_hint = _build_retry_repair_directive_hint(
            error_message,
            scenario,
        )
        syntax_retry_hint = _build_syntax_retry_hint(
            error_message,
            attempt=attempt,
            original_prompt=original_prompt,
        )

        semantic_retry_hint = ""
        if "semantic risk" in error_message:
            semantic_retry_hint = """
[SEMANTIC REWRITE REQUIREMENT]
Your previous test used APIs or setup that are not grounded in the issue target.
- Do not introduce unrelated high-level APIs just to manufacture data.
- Directly exercise the target/source API from [Validated Scenario] and [Issue Reproduction Code].
- For pytest skip/PDB issues, create a temporary test file with pytester/tmp_path instead of appending a skipped class with undefined names.
- For Django query issues, avoid ordering warnings by mirroring existing query tests and adding order_by(...) when needed.
- For Django model/query tests, use only models/helpers already imported by the target test file or explicitly available in repository imports.
- If django_existing_query_model_not_reused appears, replace unrelated auth/User models with the existing query test model shown by the target file or issue.
- Do not use placeholder symbols such as xxx, MockModel, Dummy*, FooModel, or BarModel.
- If issue_literal_not_used appears, rebuild the assertion/stimulus around that exact issue literal, such as a required HTTP header name.
- If target_function_not_called appears, rewrite the stimulus to directly call the target/source API from the scenario.
- If target_function_public_api_rewrite appears, call the target function from [Validated Scenario],
  or call the public wrapper shown in the issue that reaches the same behavior. Do not switch to an unrelated API.
- If private_attribute_public_api_rewrite appears, replace any `._private` assertion with a public API assertion:
  use return values, public object state, Matplotlib axis/legend accessors, Seaborn artist/axis properties,
  or a small issue-visible semantic invariant.
- For Requests digest auth, use the repository's digest auth helper/mock challenge flow instead of a bare PreparedRequest with no challenge.
- For Sphinx tests, do not import sphinx.testing.fixtures directly; mirror existing local test helpers or use a minimal parser/app pattern.
"""

        task_summary = _extract_retry_task_summary(original_prompt)
        compact_errors = _compact_validation_errors(error_message)
        compact_code = _compact_previous_attempt_code(previous_response, previous_parsed)
        retry_constraints = _build_m5_feedback_constraints(
            instance_repo="",
            clue={},
            context=context or {},
            scenario=scenario or {},
            validation_errors=[part.strip() for part in re.split(r";|\n", str(error_message or "")) if part.strip()],
            rejected_patterns=_network_rejected_patterns(compact_code),
        )
        retry_constraint_section = ""
        if (
            retry_constraints["mandatory_constraints"]
            or retry_constraints["forbidden_patterns"]
            or retry_constraints["repository_local_alternatives"]
        ):
            retry_constraint_section = (
                "\n[M5 RETRY MANDATORY CONSTRAINTS — MUST FIX BEFORE RETURNING JSON]\n"
                f"{json.dumps(retry_constraints, ensure_ascii=False, indent=2)}\n"
            )
        compact_response = _clip_prompt_text(
            previous_response,
            _RETRY_PREVIOUS_RESPONSE_CHARS,
        )

        return f"""You are fixing a previously generated reproduction test. Return corrected JSON only.

[Compact Task Summary]
{task_summary}

[Oracle Contract]
- The assertion must define fixed/post-patch behavior, not the bug symptom.
- Use issue-stated expected output when available.
- If only buggy output is known, compare the function return value/state against the buggy value.
- Do not invent exact expected arrays/values without issue evidence.
- Do not use private attributes, external network calls, raw rendered exact strings, or warning count/type alone.

[Previous Attempt #{attempt} - FAILED]
Key validation errors:
{compact_errors}

Previous response snippet:
```
{compact_response}
```

Previous append_block/test_code to rewrite:
```python
{compact_code}
```
{format_hint}
{syntax_retry_hint}
{directive_retry_hint}
{retry_constraint_section}
{oracle_retry_hint}
{semantic_retry_hint}
[Fix Instructions]
1. Fix all errors listed above.
2. Rewrite the failing stimulus/assertion instead of repeating forbidden or risky patterns.
3. Only import symbols that actually exist in the repository.
4. Ensure the test code is syntactically valid Python.
5. Keep one focused test_* function/method and use the same target/source API as the issue.
6. Return corrected JSON only — no explanation, no markdown outside the JSON.
""".strip()

    @staticmethod
    def _oracle_rewrite_hint(error_message: str) -> str:
        hints = [
            "[ORACLE REWRITE REQUIREMENT]",
            "Your previous test used an oracle that is likely to fail on both buggy and fixed code.",
        ]
        if "guessed_expected_array" in error_message or "guessed_expected_value" in error_message:
            hints.append("- Do NOT invent exact expected arrays/values. Use issue-stated expected output only, or assert shape/type/finite/range/semantic invariant.")
        if "negative_literal_oracle" in error_message or "negative_oracle_only" in error_message:
            hints.append("- Do NOT use a negative-only assertion such as score != 1 or header not in headers when the issue states a positive expected value/warning/state.")
            hints.append("- Replace it with a positive oracle: expected value, pytest.warns(ExpectedWarning), np.isfinite/np.isnan as stated by the issue, or required public state.")
        if "raises_only_no_body_assertion" in error_message or "fix_disappearing_exception_oracle" in error_message:
            hints.append("- Use pytest.raises/assertRaises only when the issue explicitly says the fixed behavior should raise. Otherwise assert the success path and post-call state/value.")
            hints.append("- Replace raises-only tests with: call the target API, store its return value/public state, then assert the fixed expected value or a small issue-visible invariant.")
        if "private_attribute_oracle" in error_message:
            hints.append("- Do NOT read private attributes such as _legend_data, _gridOnMajor, or _legend_labels. Use public axis/artist/legend APIs.")
        if "raw_rendered_output_exact_match" in error_message:
            hints.append("- Do NOT assert long raw HTML/LaTeX/Sphinx strings. Use a minimal public semantic marker or whitespace invariant.")
        if "warning_presence_oracle" in error_message or "warning_catch_only" in error_message:
            hints.append("- Do NOT assert warning count/type alone. Assert the returned value or public state after the call.")
        if "issue_reproduction_code_not_followed" in error_message:
            hints.append("- Rebuild the stimulus from [Issue Reproduction Code]. Preserve its key classes/functions, operators, assumptions, and warning context instead of using a generic example.")
        if "relational oracle regeneration gate" in error_message:
            hints.append("- The selected stimulus has no paired expected output. Use the issue-supported relational oracle candidate when present.")
            hints.append("- Evaluate the target function twice with independently constructed left/right equivalent inputs, store both target results, and directly assert equality between those two results.")
            hints.append("- Do NOT compare against literal arrays/values, shape, dtype, finite-ness, counts, no-exception, or a negative buggy output assertion.")
        if len(hints) == 2:
            hints.append("- Do NOT use @image_comparison, private attributes, raw rendered strings, warning-count assertions, or guessed expected arrays/values.")
            hints.append("- The assertion must describe the fixed behavior using issue expected output, public API state, or a small semantic invariant.")
        return "\n".join(hints) + "\n"

    def _build_modified_test_file_content(
        self,
        original_content: str,
        imports: List[str],
        test_code: str,
    ) -> str:
        lines = original_content.splitlines()

        # 기존 top-level import만 수집
        existing_top_imports = set()
        for line in lines:
            stripped = line.strip()
            if (line.startswith("import ") or line.startswith("from ")) and not line.startswith((" ", "\t")):
                existing_top_imports.add(stripped)

        new_imports = [imp.strip() for imp in imports if imp.strip() and imp.strip() not in existing_top_imports]

        # top-level import block 끝 찾기
        insert_idx = 0
        seen_top_import = False
        past_decorator = False  # @decorator를 이미 지났으면 import 삽입 금지
        in_paren = 0  # 괄호 depth (multi-line import 추적)

        for i, line in enumerate(lines):
            stripped = line.strip()

            # 괄호 depth 추적
            in_paren += line.count("(") - line.count(")")
            if in_paren < 0:
                in_paren = 0

            # @decorator를 지난 이후에는 빈 줄 포함 모든 줄 무시 → import block 종료
            if past_decorator:
                break

            # 빈 줄은 import block 안에서는 허용
            if not seen_top_import and stripped == "":
                continue
            # import block을 이미 봤고 빈 줄 → 아직 계속 허용 (import 사이 빈 줄)
            if seen_top_import and stripped == "" and in_paren == 0:
                continue

            # top-level import — 괄호 안에 있으면 무시
            if (line.startswith("import ") or line.startswith("from ")) and not line.startswith((" ", "\t")):
                seen_top_import = True
                # 괄호가 이 줄에서 닫히면 다음 줄, 아니면 괄호 닫힐 때까지 대기
                if in_paren == 0:
                    insert_idx = i + 1
                continue

            # multi-line import 괄호 닫힘
            if in_paren == 0 and seen_top_import and stripped.endswith(")"):
                insert_idx = i + 1
                continue

            # 괄호 안에 있으면 import block 계속
            if in_paren > 0:
                continue

            # 첫 번째 top-level 비-import 문장을 만나면 종료
            if not line.startswith((" ", "\t")) and stripped != "":
                if stripped.startswith("@"):
                    # decorator → import block 종료, 이후 줄은 무시
                    past_decorator = True
                    continue
                if seen_top_import:
                    break
                else:
                    insert_idx = i
                    break
        else:
            insert_idx = len(lines)

        updated_lines = lines[:]

        if new_imports:
            block = []
            if insert_idx > 0 and updated_lines[insert_idx - 1].strip() != "":
                block.append("")
            block.extend(new_imports)
            block.append("")
            updated_lines[insert_idx:insert_idx] = block

        rendered_test_code = test_code.rstrip()
        updated_content = "\n".join(updated_lines).rstrip() + "\n\n" + rendered_test_code + "\n"
        return updated_content

    def _build_unified_patch(
        self,
        original_content: str,
        modified_content: str,
        relative_path: str,
    ) -> str:
        old_lines = original_content.splitlines(keepends=True)
        new_lines = modified_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
        )
        return "".join(diff)

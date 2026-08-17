from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from src.utils.artifact_hash import sha256_text


MAX_LLM_REFINEMENT_ROUNDS = 1
M5A_INVALID_RESPONSE = "M5A_INVALID_RESPONSE"
M5A_PROVIDER_ERROR = "M5A_PROVIDER_ERROR"
M5A_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["test_code"],
    "properties": {
        "test_code": "complete revised Python test code",
        "expected_behavior_evidence": "issue/scenario evidence justifying unchanged oracle intent",
    },
}

_PATCH_LEAKAGE_RE = re.compile(
    r"\b(golden[-_ ]?patch|post[-_ ]?patch|after[-_ ]?patch|fail[-_ ]?to[-_ ]?pass|"
    r"patch hit rate|phr|m8)\b",
    re.IGNORECASE,
)


class LLMRepairProvider(Protocol):
    def generate(self, prompt: str, **kwargs: Any) -> str:
        ...


@dataclass(frozen=True)
class ErrorRefinementRequest:
    test_code: str
    error_message: str
    failed_line: str = ""
    nearby_source: str = ""
    scenario: Mapping[str, Any] = field(default_factory=dict)
    oracle_intent: str = ""
    target_behavior_terms: tuple[str, ...] = ()
    pytest_command: str = ""
    observed_behavior: Any = field(default_factory=list)
    expected_behavior: Any = field(default_factory=list)
    repository_commit: str = ""
    target_test_file: str = ""
    target_nodeid: str = ""
    import_evidence: Any = field(default_factory=dict)
    signature_evidence: Any = field(default_factory=dict)
    nearby_test_examples: Any = field(default_factory=list)
    oracle_requirements: Any = field(default_factory=list)
    issue_evidence: Any = field(default_factory=dict)
    m2_context_evidence: Any = field(default_factory=dict)
    oracle_hint: str = ""
    current_m7_diagnosis: Any = field(default_factory=dict)
    prior_avoid_patterns: Any = field(default_factory=dict)
    compile_only: bool = False


@dataclass(frozen=True)
class ErrorRefinementRoundRecord:
    round_index: int
    before_sha256: str
    after_sha256: str
    validation_errors: list[str]
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ErrorRefinementResult:
    enabled: bool
    used: bool
    refined_code: str
    before_sha256: str
    after_sha256: str
    round_count: int
    validations: list[ErrorRefinementRoundRecord]
    fallback_reason: str
    llm_call_count: int
    raw_response_artifact: str = ""
    raw_response_artifacts: list[str] = field(default_factory=list)
    response_empty: bool = False
    parse_attempt_count: int = 0
    parse_error: str = ""
    retry_triggered: bool = False
    retry_prompt_hash: str = ""
    final_parse_status: str = "NOT_RUN"
    repair_result: str = "USED_FAILED"
    response_parse_mode: str = "NOT_RUN"

    def to_dict(self) -> dict[str, Any]:
        if self.repair_result == "USED_SUCCESS" and self.used:
            telemetry_status = "REPAIR_SUCCESS"
        elif self.repair_result in {"NO_EFFECT_REPAIR", "NOT_TRIGGERED", "DISABLED"}:
            telemetry_status = "SAFE_REJECTION" if self.repair_result == "NO_EFFECT_REPAIR" else "NOT_INVOKED"
        elif self.repair_result == "USED_FAILED" or self.final_parse_status in {
            M5A_INVALID_RESPONSE,
            "FAILED",
        }:
            telemetry_status = "REPAIR_FAILED"
        else:
            telemetry_status = "NOT_ELIGIBLE"
        return {
            "enabled": self.enabled,
            "used": self.used,
            "refined_code": self.refined_code,
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
            "round_count": self.round_count,
            "validations": [record.to_dict() for record in self.validations],
            "fallback_reason": self.fallback_reason,
            "llm_call_count": self.llm_call_count,
            "raw_response_artifact": self.raw_response_artifact,
            "raw_response_artifacts": list(self.raw_response_artifacts),
            "response_empty": self.response_empty,
            "parse_attempt_count": self.parse_attempt_count,
            "parse_error": self.parse_error,
            "retry_triggered": self.retry_triggered,
            "retry_prompt_hash": self.retry_prompt_hash,
            "final_parse_status": self.final_parse_status,
            "repair_result": self.repair_result,
            "response_parse_mode": self.response_parse_mode,
            "telemetry_status": telemetry_status,
        }


def build_m5a_error_refinement_prompt(request: ErrorRefinementRequest) -> str:
    """Build a provider-independent M5-A repair prompt."""

    payload = {
        "task": "repair generated reproduction test only",
        "test_code": request.test_code,
        "compilation_error": request.error_message,
        "exact_traceback": request.error_message,
        "pytest_command": request.pytest_command,
        "failed_line": request.failed_line,
        "nearby_source": request.nearby_source,
        "observed_behavior": request.observed_behavior,
        "expected_behavior": request.expected_behavior,
        "repository_commit": request.repository_commit,
        "target_test_file": request.target_test_file,
        "target_nodeid": request.target_nodeid,
        "repository_local_import_evidence": request.import_evidence,
        "callable_signature_evidence": request.signature_evidence,
        "nearby_existing_test_examples": request.nearby_test_examples,
        "scenario": dict(request.scenario),
        "oracle_intent": request.oracle_intent,
        "oracle_requirements": request.oracle_requirements,
        "issue_evidence": request.issue_evidence,
        "m2_context_evidence": request.m2_context_evidence,
        "oracle_hint": request.oracle_hint,
        "current_m7_diagnosis": request.current_m7_diagnosis,
        "prior_avoid_patterns": request.prior_avoid_patterns,
        "target_behavior_terms": list(request.target_behavior_terms),
        "required_json_shape": {
            "test_code": "complete revised Python test code",
            "expected_behavior_evidence": "issue/scenario evidence justifying unchanged oracle intent",
        },
        "constraints": [
            "Modify generated test code only.",
            "Preserve the test scenario and oracle intent.",
            "Make the smallest sufficient change to the current candidate.",
            "Do not remove assertions.",
            "Do not change expected behavior without issue or scenario evidence.",
            "Do not target unrelated production behavior.",
            "Do not use golden patch, post-patch, M8, Fail-to-Pass, or PHR information.",
        ],
    }
    if request.compile_only:
        payload["task"] = "repair only the residual Python syntax, import, or name/type compilation error"
        payload["constraints"] = [
            "Fix only the reported syntax, import, missing-name, or type-compilation error.",
            "Preserve every assertion and its AST exactly.",
            "Preserve all existing test method names; do not add or remove a test method.",
            "Do not change setup, stimulus, calls, control flow, oracle logic, or expected values.",
            "Return the complete revised Python test in the required JSON object.",
            "Do not use golden patch, post-patch, M8, Fail-to-Pass, or PHR information.",
        ]
    return json.dumps(payload, sort_keys=True, indent=2)


def refine_m5a_error_with_llm(
    request: ErrorRefinementRequest,
    *,
    enabled: bool,
    rule_based_error_remaining: bool,
    provider: LLMRepairProvider | None,
    artifact_dir: Path | str | None = None,
) -> ErrorRefinementResult:
    """Run at most one guarded LLM M5-A repair call after rule repair fails."""

    before_hash = sha256_text(request.test_code)
    if not enabled:
        return _fallback_result(
            request.test_code,
            before_hash,
            enabled=False,
            fallback_reason="feature_disabled_deterministic_postprocessing_fallback",
        )
    if not rule_based_error_remaining:
        return _fallback_result(
            request.test_code,
            before_hash,
            enabled=True,
            fallback_reason="rule_based_processing_succeeded_no_llm_call",
        )
    if provider is None:
        return _fallback_result(
            request.test_code,
            before_hash,
            enabled=True,
            fallback_reason="llm_provider_unavailable_deterministic_postprocessing_fallback",
        )

    current = request.test_code
    records: list[ErrorRefinementRoundRecord] = []
    llm_calls = 0
    parse_attempt_count = 0
    response_empty = False
    parse_error = ""
    retry_triggered = False
    retry_prompt_hash = ""
    raw_response_artifacts: list[str] = []
    seen_prompt_hashes: set[str] = set()
    previous_validation_errors: list[str] = []
    for round_index in range(1, MAX_LLM_REFINEMENT_ROUNDS + 1):
        error_message = request.error_message
        if previous_validation_errors:
            error_message = (
                f"{request.error_message}\n\n"
                "Previous M5-A response validation errors: "
                f"{json.dumps(previous_validation_errors, sort_keys=True)}"
            )
        prompt = build_m5a_error_refinement_prompt(
            ErrorRefinementRequest(
                test_code=current,
                error_message=error_message,
                failed_line=request.failed_line,
                nearby_source=request.nearby_source,
                scenario=request.scenario,
                oracle_intent=request.oracle_intent,
                target_behavior_terms=request.target_behavior_terms,
                pytest_command=request.pytest_command,
                observed_behavior=request.observed_behavior,
                expected_behavior=request.expected_behavior,
                repository_commit=request.repository_commit,
                target_test_file=request.target_test_file,
                target_nodeid=request.target_nodeid,
                import_evidence=request.import_evidence,
                signature_evidence=request.signature_evidence,
                nearby_test_examples=request.nearby_test_examples,
                oracle_requirements=request.oracle_requirements,
                issue_evidence=request.issue_evidence,
                m2_context_evidence=request.m2_context_evidence,
                oracle_hint=request.oracle_hint,
                current_m7_diagnosis=request.current_m7_diagnosis,
                prior_avoid_patterns=request.prior_avoid_patterns,
                compile_only=request.compile_only,
            )
        )
        prompt_hash = sha256_text(prompt)
        if prompt_hash in seen_prompt_hashes:
            break
        seen_prompt_hashes.add(prompt_hash)
        llm_calls += 1
        try:
            raw = provider.generate(prompt)
        except Exception as exc:
            return ErrorRefinementResult(
                enabled=True,
                used=False,
                refined_code=request.test_code,
                before_sha256=before_hash,
                after_sha256=before_hash,
                round_count=0,
                validations=[],
                fallback_reason=M5A_PROVIDER_ERROR,
                llm_call_count=llm_calls,
                parse_error=f"{type(exc).__name__}: {exc}",
                final_parse_status=M5A_PROVIDER_ERROR,
                repair_result="USED_FAILED",
            )
        artifact = _persist_raw_m5a_response(artifact_dir, round_index, parse_attempt_count + 1, raw)
        if artifact:
            raw_response_artifacts.append(artifact)
        parse_attempt_count += 1
        try:
            revised, response_parse_mode = _parse_m5a_error_refinement_response(raw)
            parse_error = ""
        except (json.JSONDecodeError, ValueError) as exc:
            response_empty = response_empty or not str(raw or "").strip()
            parse_error = str(exc)
            return _invalid_response_result(
                request=request,
                before_hash=before_hash,
                llm_calls=llm_calls,
                raw_response_artifacts=raw_response_artifacts,
                response_empty=response_empty,
                parse_attempt_count=parse_attempt_count,
                parse_error=parse_error,
                retry_triggered=False,
                retry_prompt_hash="",
            )
        validation_errors = validate_m5a_error_refinement_revision(
            before_code=current,
            revised_code=revised,
            request=request,
        )
        record = ErrorRefinementRoundRecord(
            round_index=round_index,
            before_sha256=sha256_text(current),
            after_sha256=sha256_text(revised),
            validation_errors=validation_errors,
            accepted=not validation_errors,
        )
        records.append(record)
        if not validation_errors:
            return ErrorRefinementResult(
                enabled=True,
                used=True,
                refined_code=revised,
                before_sha256=before_hash,
                after_sha256=sha256_text(revised),
                round_count=round_index,
                validations=records,
                fallback_reason="",
                llm_call_count=llm_calls,
                raw_response_artifact=raw_response_artifacts[-1] if raw_response_artifacts else "",
                raw_response_artifacts=raw_response_artifacts,
                response_empty=response_empty,
                parse_attempt_count=parse_attempt_count,
                parse_error=parse_error,
                retry_triggered=retry_triggered,
                retry_prompt_hash=retry_prompt_hash,
                final_parse_status="SUCCESS",
                repair_result="USED_SUCCESS",
                response_parse_mode=response_parse_mode,
            )
        previous_validation_errors = list(validation_errors)

    no_effect_repair = any(
        "unchanged_candidate_identity" in record.validation_errors
        for record in records
    )
    return ErrorRefinementResult(
        enabled=True,
        used=False,
        refined_code=request.test_code,
        before_sha256=before_hash,
        after_sha256=before_hash,
        round_count=len(records),
        validations=records,
        fallback_reason=(
            "NO_EFFECT_REPAIR"
            if no_effect_repair
            else "llm_refinement_rejected_deterministic_postprocessing_fallback"
        ),
        llm_call_count=llm_calls,
        raw_response_artifact=raw_response_artifacts[-1] if raw_response_artifacts else "",
        raw_response_artifacts=raw_response_artifacts,
        response_empty=response_empty,
        parse_attempt_count=parse_attempt_count,
        parse_error=parse_error,
        retry_triggered=retry_triggered,
        retry_prompt_hash=retry_prompt_hash,
        final_parse_status="SUCCESS" if parse_attempt_count else "NOT_RUN",
        repair_result="NO_EFFECT_REPAIR" if no_effect_repair else "USED_FAILED",
    )


def parse_m5a_error_refinement_response(raw_response: str) -> str:
    """Parse exact JSON first, then one unambiguous complete-code fallback."""
    code, _ = _parse_m5a_error_refinement_response(raw_response)
    return code


def _parse_m5a_error_refinement_response(raw_response: str) -> tuple[str, str]:
    if not str(raw_response or "").strip():
        raise ValueError("M5-A refinement response was empty")
    json_error: Exception | None = None
    try:
        candidate_json = _extract_m5a_json_object(raw_response)
        try:
            data = json.loads(candidate_json)
        except json.JSONDecodeError:
            # Providers occasionally emit a strict-schema object with a
            # trailing comma.  Removing only commas immediately before a
            # closing delimiter is a lossless syntax recovery; semantic and
            # identity validation still runs below.
            repaired_json = re.sub(r",\s*([}\]])", r"\1", candidate_json)
            data = json.loads(repaired_json)
        if not isinstance(data, Mapping):
            raise ValueError("M5-A refinement response must be a JSON object")
        # ``test_code`` is the canonical field.  A small, explicit recovery
        # set handles providers that used an equivalent field name while
        # preserving the same complete-code validation below.  This does not
        # synthesize code or accept prose.
        code = str(
            data.get("test_code")
            or data.get("append_block")
            or data.get("revised_code")
            or data.get("refined_code")
            or data.get("code")
            or ""
        )
        if not code.strip():
            raise ValueError("M5-A refinement response missing test_code")
        return code, "JSON_OBJECT"
    except (json.JSONDecodeError, ValueError) as exc:
        json_error = exc

    fallback_code = _extract_unambiguous_python_response(raw_response)
    if fallback_code is not None:
        return fallback_code, "UNAMBIGUOUS_PYTHON_FALLBACK"
    raise ValueError(str(json_error)) from json_error


def _extract_unambiguous_python_response(raw_response: str) -> str | None:
    """Return complete code from a code-only or single fenced response.

    M5-A providers sometimes prepend an explanation and then emit one fenced
    Python module.  Recovering exactly one syntactically complete test block is
    deterministic; prose, multiple competing blocks, and fragments remain
    rejected.
    """
    text = str(raw_response or "").strip()
    fenced_blocks = re.findall(
        r"```(?:python|py)?\s*\n?(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if len(fenced_blocks) > 1:
        return None
    candidate = fenced_blocks[0].strip() if fenced_blocks else text
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return None
    has_test = any(
        (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"))
        or (isinstance(node, ast.ClassDef) and node.name.startswith("Test"))
        for node in ast.walk(tree)
    )
    if not has_test:
        return None
    # A code-only response may contain imports, helpers, and the complete test,
    # but explanatory prose would not parse as a Python module.
    return candidate + ("\n" if candidate and not candidate.endswith("\n") else "")


def build_m5a_json_repair_prompt(invalid_raw_response: str, parse_error: str) -> str:
    payload = {
        "task": "repair invalid M5-A provider response JSON",
        "invalid_raw_response": str(invalid_raw_response or ""),
        "parse_error": str(parse_error or ""),
        "expected_json_schema": M5A_RESPONSE_SCHEMA,
        "instruction": "Return JSON only. Do not include Markdown fences or explanatory text.",
    }
    return json.dumps(payload, sort_keys=True, indent=2)


def _extract_m5a_json_object(raw_response: str) -> str:
    text = _strip_markdown_json_fence(str(raw_response or "").strip())
    start = text.find("{")
    if start < 0:
        raise ValueError("M5-A refinement response did not contain a JSON object")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise ValueError("M5-A refinement response JSON object was incomplete")


def _strip_markdown_json_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def validate_m5a_error_refinement_revision(
    *,
    before_code: str,
    revised_code: str,
    request: ErrorRefinementRequest,
) -> list[str]:
    """Reject revisions that violate M5-A oracle and isolation constraints."""

    errors: list[str] = []
    if sha256_text(revised_code) == sha256_text(before_code):
        errors.append("unchanged_candidate_identity")
    try:
        ast.parse(revised_code)
    except SyntaxError as exc:
        errors.append(f"syntax_error:{exc}")

    before_assertions = _assertion_count(before_code)
    revised_assertions = _assertion_count(revised_code)
    structural_prune_allowed = any(
        marker in str(request.error_message or "").lower()
        for marker in (
            "test functions/methods",
            "exactly one focused reproduction test",
            "multiple generated test",
        )
    )
    if revised_assertions < before_assertions and not structural_prune_allowed:
        errors.append("removed_assertions")
    if revised_assertions == 0:
        errors.append("missing_assertions")

    if request.compile_only:
        if _test_method_names(revised_code) != _test_method_names(before_code):
            errors.append("changed_test_method_identity")
        if _assertion_fingerprints(revised_code) != _assertion_fingerprints(before_code):
            errors.append("changed_assertion_or_oracle_logic")
        before_semantics = _compile_only_semantic_fingerprint(before_code)
        revised_semantics = _compile_only_semantic_fingerprint(revised_code)
        if (
            before_semantics is not None
            and revised_semantics is not None
            and before_semantics != revised_semantics
        ):
            errors.append("changed_setup_stimulus_or_control_flow")

    combined = "\n".join([revised_code, request.oracle_intent, str(dict(request.scenario))])
    if _PATCH_LEAKAGE_RE.search(combined):
        errors.append("patch_or_post_patch_reference")

    before_behavior = _expected_behavior_literals(before_code)
    revised_behavior = _expected_behavior_literals(revised_code)
    if before_behavior and revised_behavior and revised_behavior != before_behavior:
        evidence_text = _evidence_text(request)
        changed_literals = revised_behavior - before_behavior
        if changed_literals and not any(value.lower() in evidence_text for value in changed_literals):
            errors.append("changed_expected_behavior_without_evidence")

    terms = {term.lower() for term in request.target_behavior_terms if term.strip()}
    if terms:
        if not _revision_targets_expected_behavior(revised_code, terms):
            errors.append("targets_unrelated_production_behavior")

    return errors


def _compile_only_semantic_fingerprint(code: str) -> str | None:
    """Fingerprint all non-import, non-type-annotation executable semantics."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    class CompileSurfaceNormalizer(ast.NodeTransformer):
        def visit_Import(self, node: ast.Import) -> None:
            return None

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            return None

        def visit_arg(self, node: ast.arg) -> ast.arg:
            node.annotation = None
            node.type_comment = None
            return node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
            self.generic_visit(node)
            node.returns = None
            node.type_comment = None
            return node

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
            self.generic_visit(node)
            node.returns = None
            node.type_comment = None
            return node

        def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
            self.generic_visit(node)
            if node.value is None:
                return ast.Pass()
            return ast.Assign(targets=[node.target], value=node.value)

    normalized = CompileSurfaceNormalizer().visit(tree)
    normalized = ast.fix_missing_locations(normalized)
    return sha256_text(ast.dump(normalized, include_attributes=False))


def _test_method_names(code: str) -> tuple[str, ...]:
    """Return declared test methods without requiring the input to compile."""
    return tuple(
        re.findall(
            r"^\s*(?:async\s+)?def\s+(test_[A-Za-z_]\w*)\s*\(",
            code or "",
            flags=re.MULTILINE,
        )
    )


def _assertion_fingerprints(code: str) -> tuple[str, ...]:
    """Return assertion ASTs, with a line fallback for invalid source."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return tuple(
            re.sub(r"\s+", " ", line.strip())
            for line in (code or "").splitlines()
            if re.search(
                r"\bassert\b|self\.assert|pytest\.raises|assert_(?:allclose|array|equal|raises)",
                line,
            )
        )
    assertions: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            assertions.append(ast.dump(node, include_attributes=False))
        elif isinstance(node, ast.With):
            for item in node.items:
                expression = item.context_expr
                if (
                    isinstance(expression, ast.Call)
                    and isinstance(expression.func, ast.Attribute)
                    and expression.func.attr in {"raises", "warns"}
                ):
                    assertions.append(ast.dump(node, include_attributes=False))
                    break
        elif isinstance(node, ast.Call):
            function = node.func
            name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else function.id
                if isinstance(function, ast.Name)
                else ""
            )
            if name.startswith("assert"):
                assertions.append(ast.dump(node, include_attributes=False))
    return tuple(assertions)


def _fallback_result(
    code: str,
    before_hash: str,
    *,
    enabled: bool,
    fallback_reason: str,
) -> ErrorRefinementResult:
    return ErrorRefinementResult(
        enabled=enabled,
        used=False,
        refined_code=code,
        before_sha256=before_hash,
        after_sha256=before_hash,
        round_count=0,
        validations=[],
        fallback_reason=fallback_reason,
        llm_call_count=0,
        repair_result=(
            "DISABLED"
            if not enabled
            else "NOT_TRIGGERED"
            if "no_llm_call" in fallback_reason or "rule_based_processing_succeeded" in fallback_reason
            else "USED_FAILED"
        ),
    )


def _invalid_response_result(
    *,
    request: ErrorRefinementRequest,
    before_hash: str,
    llm_calls: int,
    raw_response_artifacts: list[str],
    response_empty: bool,
    parse_attempt_count: int,
    parse_error: str,
    retry_triggered: bool,
    retry_prompt_hash: str,
) -> ErrorRefinementResult:
    return ErrorRefinementResult(
        enabled=True,
        used=False,
        refined_code=request.test_code,
        before_sha256=before_hash,
        after_sha256=before_hash,
        round_count=0,
        validations=[],
        fallback_reason=M5A_INVALID_RESPONSE,
        llm_call_count=llm_calls,
        raw_response_artifact=raw_response_artifacts[-1] if raw_response_artifacts else "",
        raw_response_artifacts=raw_response_artifacts,
        response_empty=response_empty,
        parse_attempt_count=parse_attempt_count,
        parse_error=parse_error,
        retry_triggered=retry_triggered,
        retry_prompt_hash=retry_prompt_hash,
        final_parse_status=M5A_INVALID_RESPONSE,
        repair_result="USED_FAILED",
    )


def _persist_raw_m5a_response(
    artifact_dir: Path | str | None,
    round_index: int,
    parse_attempt_index: int,
    raw_response: str,
) -> str:
    if artifact_dir is None:
        return ""
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    response_path = path / f"m5a_raw_response_round_{round_index}_parse_{parse_attempt_index}.txt"
    response_path.write_text(str(raw_response or ""), encoding="utf-8")
    return str(response_path)


def _assertion_count(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return len(re.findall(r"^\s*assert\b|self\.assert\w+\s*\(", code, re.MULTILINE))
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            count += 1
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr.startswith("assert"):
                count += 1
    return count


def _expected_behavior_literals(code: str) -> set[str]:
    literals: set[str] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return literals
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            literals.update(_literal_strings(node.test))
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr.startswith("assert"):
                for arg in node.args:
                    literals.update(_literal_strings(arg))
    return {value for value in literals if value}


def _literal_strings(node: ast.AST) -> set[str]:
    values: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, (str, int, float, bool)):
            values.add(str(child.value).lower())
    return values


def _evidence_text(request: ErrorRefinementRequest) -> str:
    chunks = [request.oracle_intent, str(dict(request.scenario))]
    chunks.extend(request.target_behavior_terms)
    return "\n".join(chunks).lower()


def _revision_targets_expected_behavior(code: str, terms: set[str]) -> bool:
    call_terms = {term[:-2] for term in terms if term.endswith("()") and term[:-2]}
    text_terms = terms - {f"{term}()" for term in call_terms}
    if call_terms:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            tree = None
        if tree is not None:
            called_names: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        called_names.add(node.func.id.lower())
                    elif isinstance(node.func, ast.Attribute):
                        called_names.add(node.func.attr.lower())
            if call_terms & called_names:
                return True
    revised_lower = code.lower()
    return any(term in revised_lower for term in text_terms)

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Protocol

from src.contracts.envelope import make_envelope
from src.contracts.feature_flags import V22FeatureFlags, core_only_feature_flags, resolve_feature_flags
from src.contracts.models import IssueClue
from src.utils.file_io import write_json_atomic


def _resolve_flags(
    feature_flags: V22FeatureFlags | Dict[str, bool] | None,
) -> V22FeatureFlags:
    if isinstance(feature_flags, V22FeatureFlags):
        return feature_flags
    if feature_flags is None:
        return core_only_feature_flags()
    return resolve_feature_flags(feature_flags)


def _resolve_artifact_flags(
    feature_flags: V22FeatureFlags | Dict[str, bool] | None,
    payload: Dict[str, Any],
) -> V22FeatureFlags:
    if feature_flags is not None:
        return _resolve_flags(feature_flags)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        raw_flags = metadata.get("feature_flags")
        if isinstance(raw_flags, dict):
            try:
                return resolve_feature_flags(raw_flags)
            except (TypeError, ValueError):
                pass
    return core_only_feature_flags()


@dataclass
class IssueClueFile:
    instance_id: str
    observed_behavior: List[str]
    expected_behavior: List[str]
    repro_conditions: List[str]
    environment: List[str]
    identifiers: Dict[str, List[str]]
    raw_issue_text: str
    code_examples: List[Dict[str, str]] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    actual_outputs: List[str] = field(default_factory=list)
    error_keywords: List[str] = field(default_factory=list)
    # 이슈 traceback에서 파싱된 고신뢰도 fault location 후보
    # 각 항목: {"file_path": "...", "line_no": N, "function_name": "..."}
    fault_locations: List[Dict[str, Any]] = field(default_factory=list)
    defect_location_hints: List[Dict[str, Any]] = field(default_factory=list)
    implicit_fault_locations: List[Dict[str, Any]] = field(default_factory=list)
    inferred_EB: List[str] = field(default_factory=list)
    inferred_fault_location_clues: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float | None = None
    confidence_norm: float | None = None
    llm_refinement_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["fact_ids"] = {
            "OB": [f"OB-{index + 1}" for index, _ in enumerate(self.observed_behavior)],
            "EB": [f"EB-{index + 1}" for index, _ in enumerate(self.expected_behavior)],
            "S2R": [f"S2R-{index + 1}" for index, _ in enumerate(self.repro_conditions)],
            "IDENTIFIER": [f"ID-{index + 1}" for index, _ in enumerate(
                value for values in self.identifiers.values() for value in values
            )],
        }
        payload["facts"] = [
            {"fact_id": fact_id, "kind": kind, "text": text}
            for kind, values in (("OB", self.observed_behavior), ("EB", self.expected_behavior), ("S2R", self.repro_conditions))
            for index, text in enumerate(values)
            for fact_id in [f"{kind}-{index + 1}"]
        ]
        payload["steps_to_reproduce"] = list(self.repro_conditions)
        payload["expected_output"] = list(self.expected_outputs)
        payload["actual_output"] = list(self.actual_outputs)
        payload["code_snippets"] = list(self.code_examples)
        payload["fault_location_clues"] = [
            *self.fault_locations,
            *self.inferred_fault_location_clues,
        ]
        payload["defect_location_hints"] = list(self.defect_location_hints)
        adequacy = self.metadata.get("extraction_adequacy") if isinstance(self.metadata, dict) else {}
        if not isinstance(adequacy, dict):
            adequacy = {}
        payload["evidence_status"] = {
            "observed_behavior": "available" if self.observed_behavior else "missing",
            "expected_behavior": "available" if self.expected_behavior else "missing",
            "reproduction_or_code": "available" if (self.repro_conditions or self.code_examples) else "missing",
            "api_identifier": "available" if any(self.identifiers.get(k) for k in ("functions", "classes", "dotted_apis")) else "missing",
            "missing_fields": list(adequacy.get("missing_evidence") or []),
            "grounded": not bool(adequacy.get("missing_evidence")),
        }
        payload["OB"] = list(self.observed_behavior)
        payload["EB"] = list(self.expected_behavior)
        payload["S2R"] = list(self.repro_conditions)
        payload["schema_version"] = "m1-v30-fact-contract-v1" if self.metadata.get("feature_profile") == "v30" else "m1-v26-rule-clues-v1"
        return payload

    def to_issue_clue(self) -> IssueClue:
        return IssueClue(
            instance_id=self.instance_id,
            observed_behavior="\n".join(self.observed_behavior),
            expected_behavior="\n".join(self.expected_behavior),
            steps_to_reproduce=list(self.repro_conditions),
            identifiers={key: list(value) for key, value in self.identifiers.items()},
            defect_location_hints=[dict(value) for value in self.defect_location_hints],
            raw_issue_text=self.raw_issue_text,
            metadata={
                "legacy_observed_behavior": list(self.observed_behavior),
                "legacy_expected_behavior": list(self.expected_behavior),
                "environment": list(self.environment),
                "code_examples": list(self.code_examples),
                "expected_output_clues": list(self.expected_outputs),
                "actual_output_clues": list(self.actual_outputs),
                "error_keywords": list(self.error_keywords),
                "fault_location_clues": list(self.fault_locations),
                "implicit_fault_locations": list(self.implicit_fault_locations),
                "inferred_EB": list(self.inferred_EB),
                "inferred_fault_location_clues": list(self.inferred_fault_location_clues),
                "confidence": self.confidence,
                "confidence_norm": self.confidence_norm,
                "llm_refinement_used": self.llm_refinement_used,
                **self.metadata,
            },
        )


class IssueClueRefinementClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class IssueClueExtractor:
    PROMPT_VERSION = "m1_llm_refinement_v1"

    def __init__(self, llm_client: IssueClueRefinementClient | None = None) -> None:
        self.llm_client = llm_client
        self.class_pattern = re.compile(r"\b[A-Z][A-Za-z0-9_]+\b")
        self.file_pattern = re.compile(r"\b[\w\-/\\]+\.(py|txt|json|yaml|yml|ini|cfg)\b")
        self.exception_pattern = re.compile(r"\b[A-Z][A-Za-z0-9_]*(Error|Exception)\b")
        self.func_pattern = re.compile(r"\b[a-z_][a-zA-Z0-9_]*\(")
        self.class_stopwords = {
            # Original
            "Consider", "If", "It", "The", "This", "That", "Suddenly",
            "True", "False", "Modeling", "Expected", "Actual",
            # Markdown headers / section titles
            "Description", "Example", "Reproduction", "Note", "Warning",
            "See", "Also", "Summary", "Details", "Steps", "Problem",
            "Solution", "Result", "Output", "Input", "Background",
            "Context", "Workaround", "Suggestion", "Resolution",
            # English common words that match CamelCase pattern
            "For", "So", "Use", "Using", "Fix", "Fixes", "Fixed",
            "Thanks", "After", "Before", "When", "Where", "How",
            "What", "About", "Between", "Into", "With", "Without",
            "From", "Based", "During", "Through", "However", "Since",
            "Because", "Although", "While", "Each", "Every", "Some",
            "Any", "Other", "Another", "Both", "Here", "There",
            # Python literals / builtins
            "None", "True", "False", "NotImplemented",
            # Common framework names (not identifier targets)
            "Django", "Astropy", "Flask", "React", "Rails",
            "Python", "Pytest", "Numpy", "Scipy", "Matplotlib",
            "GitHub", "Windows", "Linux", "MacOS",
            # IPython/Jupyter notebook style
            "In", "Out",
            # Generic prose words that match CamelCase
            "File", "New", "Old", "Line", "Code", "Test", "Class",
            "Module", "Calling", "Documents", "Looking", "Users", "Correct",
            "Traceback", "Error", "Exception",
        }
        self.function_stopwords = {
            "array", "print", "len",
            # Reproduction/environment helpers that often appear in issue snippets
            # but are poor root-cause candidates.
            "arange", "rand", "random", "seed", "platform", "get_backend",
            "show_versions",
            # Python builtins
            "range", "type", "str", "int", "list", "dict", "set",
            "format", "open", "input", "super", "isinstance",
            "getattr", "setattr", "hasattr", "delattr",
            "repr", "hash", "iter", "next", "sorted", "reversed",
            "enumerate", "zip", "map", "filter", "any", "all",
            "min", "max", "sum", "abs", "round", "bool", "float",
            "tuple", "bytes", "bytearray", "object", "property",
            "staticmethod", "classmethod", "vars", "dir", "id",
        }

    def extract(
        self,
        instance_id: str,
        issue_text: str,
        *,
        feature_flags: V22FeatureFlags | Dict[str, bool] | None = None,
        repository_path: str | Path | None = None,
    ) -> IssueClueFile:
        text = issue_text.strip()
        signal_text = self._strip_template_noise(text)
        flags = _resolve_flags(feature_flags)

        sections = self._extract_labeled_sections(text)
        observed = sections.get("observed") or self._extract_observed_behavior(text)
        expected = sections.get("expected") or self._extract_expected_behavior(text)
        repro = sections.get("repro") or self._extract_repro_conditions(text)
        env = self._extract_environment(text)
        identifiers = self._extract_identifiers(signal_text)
        code_examples = self._extract_code_blocks(text)
        expected_outputs, actual_outputs = self._extract_output_examples(text, code_examples)
        error_keywords = self._extract_error_keywords(signal_text, identifiers, code_examples)

        fault_locations = self._extract_fault_locations(text, code_examples)
        refinement = {
            "implicit_fault_locations": [],
            "inferred_EB": [],
            "inferred_fault_location_clues": [],
            "confidence": None,
            "confidence_norm": None,
            "used": False,
            "fallback_used": False,
            "fallback_reason": None,
            "trigger_reason": "v26_rule_only_extraction",
            "prompt_provenance": None,
            "parser_status": "not_applicable",
            "status": "removed_v26_rule_only",
            "client_configured": self.llm_client is not None,
            "repository_evidence_configured": repository_path is not None,
            "extraction_adequacy": self._extraction_adequacy(
                observed=observed,
                expected=expected,
                repro=repro,
                identifiers=identifiers,
                code_examples=code_examples,
            ),
            "labeled_sections": {
                key: bool(value) for key, value in sections.items()
            },
        }

        metadata = self._feature_metadata(flags, refinement=refinement)
        if isinstance(feature_flags, Mapping) and feature_flags.get("feature_profile") == "v30":
            metadata["feature_profile"] = "v30"
        return IssueClueFile(
            instance_id=instance_id,
            observed_behavior=observed,
            expected_behavior=expected,
            repro_conditions=repro,
            environment=env,
            identifiers=identifiers,
            raw_issue_text=text,
            code_examples=code_examples,
            expected_outputs=expected_outputs,
            actual_outputs=actual_outputs,
            error_keywords=error_keywords,
            fault_locations=fault_locations,
            defect_location_hints=[dict(value) for value in fault_locations],
            implicit_fault_locations=refinement["implicit_fault_locations"],
            inferred_EB=refinement["inferred_EB"],
            inferred_fault_location_clues=refinement["inferred_fault_location_clues"],
            confidence=refinement["confidence"],
            confidence_norm=refinement["confidence_norm"],
            llm_refinement_used=refinement["used"],
            metadata=metadata,
        )

    @staticmethod
    def _strip_template_noise(text: str) -> str:
        """Remove issue-template comments and bulky environment dumps for identifier extraction."""
        text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
        text = re.sub(r"<details>.*?</details>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(
            r"(?is)installed versions\s*-+\s*.*?(?:\n\s*\n|$)",
            " ",
            text,
        )
        return text

    def _extract_labeled_sections(self, text: str) -> Dict[str, List[str]]:
        """Extract case-insensitive behavior/S2R sections without traceback frames."""
        labels = {
            "observed": re.compile(
                r"^(?:actual(?: behavior| results?| outputs?)?|observed(?: behavior| results?)?|current behavior)$",
                re.IGNORECASE,
            ),
            "expected": re.compile(
                r"^(?:expected(?: behavior| results?| outputs?)?|desired behavior|should happen)$",
                re.IGNORECASE,
            ),
            "repro": re.compile(
                r"^(?:steps?(?: to reproduce)?|steps?\s*/\s*code to reproduce|code to reproduce|reproduction|how to reproduce|minimal example)$",
                re.IGNORECASE,
            ),
        }
        results: Dict[str, List[str]] = {"observed": [], "expected": [], "repro": []}
        current: str | None = None
        in_traceback = False
        in_fence = False
        raw_lines = text.splitlines()
        for index, raw_line in enumerate(raw_lines):
            stripped = raw_line.strip()
            bold_heading = bool(re.fullmatch(r"\*\*[^*]+\*\*:?", stripped))
            if bold_heading:
                stripped = stripped.strip(":").strip("*").strip()
            is_setext_title = (
                index + 1 < len(raw_lines)
                and bool(stripped)
                and bool(re.fullmatch(r"\s*(?:=+|-+)\s*", raw_lines[index + 1]))
            )
            is_setext_underline = bool(re.fullmatch(r"\s*(?:=+|-+)\s*", raw_line))
            if is_setext_underline:
                continue
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            heading = re.match(r"^(?:#{1,6}\s*)?([^:]{2,40}?)(?::\s*(.*))?$", stripped)
            matched_label: str | None = None
            inline = ""
            if heading:
                title = re.sub(r"\s+#+\s*$", "", heading.group(1)).strip().strip("*_`")
                for key, pattern in labels.items():
                    if pattern.fullmatch(title):
                        matched_label = key
                        inline = (heading.group(2) or "").strip()
                        break
            if matched_label:
                current = matched_label
                in_traceback = False
                if inline:
                    results[current].append(f"{title}: {inline}")
                continue
            if is_setext_title:
                current = None
                in_traceback = False
                continue
            if bold_heading:
                current = None
                in_traceback = False
                continue
            if re.match(r"^#{1,6}(?:\s+|$)", stripped):
                current = None
                in_traceback = False
                continue
            if not current or not stripped or stripped.startswith("```"):
                continue
            if stripped.startswith("Traceback (most recent call last)"):
                in_traceback = True
                continue
            if in_traceback:
                if re.match(r"^(?:File\s+|\^+$|~+$|During handling|The above exception)", stripped):
                    continue
                if re.match(r"^[A-Z][A-Za-z0-9_.]*(?:Error|Exception):", stripped):
                    results[current].append(stripped)
                continue
            results[current].append(stripped)
        return {key: self._dedup(value) for key, value in results.items()}

    @staticmethod
    def _extraction_adequacy(
        *,
        observed: List[str],
        expected: List[str],
        repro: List[str],
        identifiers: Mapping[str, List[str]],
        code_examples: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        has_api = bool(
            identifiers.get("functions")
            or identifiers.get("classes")
            or identifiers.get("dotted_apis")
        )
        has_stimulus = bool(repro or code_examples)
        missing = [
            name
            for name, present in (
                ("observed_behavior", bool(observed)),
                ("expected_behavior", bool(expected)),
                ("reproduction_or_code", has_stimulus),
                ("api_identifier", has_api),
            )
            if not present
        ]
        status = "ADEQUATE" if not missing else "PARTIAL" if len(missing) <= 2 else "INSUFFICIENT"
        return {
            "status": status,
            "missing_evidence": missing,
            "expected_behavior_inferred": False,
        }

    def save(
        self,
        clue: IssueClueFile,
        output_path: str,
        *,
        run_id: str | None = None,
        feature_flags: V22FeatureFlags | Dict[str, bool] | None = None,
        enveloped: bool = False,
    ) -> None:
        payload = clue.to_dict()
        if enveloped:
            flags = _resolve_artifact_flags(feature_flags, payload)
            payload = make_envelope(
                instance_id=clue.instance_id,
                run_id=run_id or clue.instance_id,
                module="m1",
                payload=payload,
                feature_flags=flags,
            ).to_dict()
        write_json_atomic(payload, output_path)

    def save_canonical(
        self,
        clue: IssueClueFile,
        output_path: str,
        *,
        run_id: str,
        feature_flags: V22FeatureFlags | Dict[str, bool] | None = None,
    ) -> None:
        flags = _resolve_flags(feature_flags)
        envelope = make_envelope(
            instance_id=clue.instance_id,
            run_id=run_id,
            module="m1",
            payload=clue.to_issue_clue().to_dict(),
            feature_flags=flags,
        )
        write_json_atomic(envelope.to_dict(), output_path)

    @staticmethod
    def _feature_metadata(
        flags: V22FeatureFlags,
        *,
        refinement: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        requested_legacy = flags.enable_m1_llm_clue_refinement
        enabled = False
        used = bool((refinement or {}).get("used"))
        fallback_used = bool((refinement or {}).get("fallback_used", enabled and not used))
        status = str(
            (refinement or {}).get(
                "status",
                "disabled" if not enabled else "fallback_no_client",
            )
        )
        return {
            "feature_flags": {**flags.to_dict(), **flags.to_legacy_alias_dict()},
            "canonical_feature_flags": flags.to_dict(),
            "rejected_references": list((refinement or {}).get("rejected_references") or []),
            "extraction_adequacy": dict(
                (refinement or {}).get("extraction_adequacy") or {}
            ),
            "labeled_sections": dict((refinement or {}).get("labeled_sections") or {}),
            "optional_features": {
                "m1_llm_clue_refinement": {
                    "enabled": enabled,
                    "requested_legacy": requested_legacy,
                    "used": used,
                    "fallback_used": fallback_used,
                    "status": status,
                    "trigger_reason": (refinement or {}).get("trigger_reason"),
                    "prompt_provenance": (refinement or {}).get("prompt_provenance"),
                    "parser_status": (refinement or {}).get("parser_status"),
                    "fallback_reason": (refinement or {}).get("fallback_reason"),
                    "availability": {
                        "client_configured": bool((refinement or {}).get("client_configured")),
                        "repository_evidence_configured": bool(
                            (refinement or {}).get("repository_evidence_configured")
                        ),
                    },
                }
            },
        }

    def _maybe_refine_with_llm(
        self,
        *,
        issue_text: str,
        observed_behavior: List[str],
        expected_behavior: List[str],
        expected_outputs: List[str],
        fault_locations: List[Dict[str, Any]],
        identifiers: Dict[str, List[str]],
        repository_path: Path | None,
        flags: V22FeatureFlags,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "implicit_fault_locations": [],
            "inferred_EB": [],
            "inferred_fault_location_clues": [],
            "confidence": None,
            "confidence_norm": None,
            "used": False,
            "fallback_used": False,
            "fallback_reason": None,
            "trigger_reason": None,
            "prompt_provenance": None,
            "parser_status": "not_invoked",
            "status": "disabled",
            "client_configured": self.llm_client is not None,
            "repository_evidence_configured": repository_path is not None,
        }
        if not flags.enable_m1_llm_clue_refinement:
            return result

        trigger_reason = self._m1_refinement_trigger_reason(
            expected_behavior=expected_behavior,
            expected_outputs=expected_outputs,
            fault_locations=fault_locations,
            identifiers=identifiers,
        )
        result["trigger_reason"] = trigger_reason
        if trigger_reason == "explicit_evidence_sufficient":
            result.update(
                {
                    "fallback_used": False,
                    "status": "skipped_explicit_evidence_sufficient",
                    "fallback_reason": None,
                }
            )
            return result
        if self.llm_client is None:
            result.update(
                {
                    "fallback_used": True,
                    "fallback_reason": "llm_client_unavailable",
                    "status": "fallback_no_client",
                }
            )
            return result

        prompt = self.build_m1_refinement_prompt(
            issue_text=issue_text,
            observed_behavior=observed_behavior,
            expected_behavior=expected_behavior,
            expected_outputs=expected_outputs,
            fault_locations=fault_locations,
            identifiers=identifiers,
        )
        result["prompt_provenance"] = {
            "module": "m1",
            "prompt_version": self.PROMPT_VERSION,
            "builder": "IssueClueExtractor.build_m1_refinement_prompt",
        }
        try:
            raw = self.llm_client.complete(prompt)
        except (RuntimeError, ValueError, TypeError) as exc:
            result.update(
                {
                    "fallback_used": True,
                    "fallback_reason": f"llm_client_error:{type(exc).__name__}",
                    "parser_status": "not_parsed_client_error",
                    "status": "fallback_client_error",
                }
            )
            return result

        parsed = self.parse_m1_refinement_response(raw)
        result["parser_status"] = parsed["parser_status"]
        if not parsed["valid"]:
            result.update(
                {
                    "fallback_used": True,
                    "fallback_reason": parsed["fallback_reason"],
                    "status": "fallback_invalid_response",
                }
            )
            return result

        validated_locations, rejected_locations = self._validate_m1_inferred_locations(
            parsed["implicit_fault_locations"],
            repository_path=repository_path,
        )
        if rejected_locations:
            result["rejected_references"] = rejected_locations
        result.update(
            {
                "implicit_fault_locations": validated_locations,
                "inferred_fault_location_clues": validated_locations,
                "inferred_EB": parsed["inferred_EB"],
                "confidence": parsed["confidence"],
                "confidence_norm": parsed["confidence_norm"],
                "used": True,
                "fallback_used": False,
                "status": "used",
            }
        )
        return result

    @staticmethod
    def _m1_refinement_trigger_reason(
        *,
        expected_behavior: List[str],
        expected_outputs: List[str],
        fault_locations: List[Dict[str, Any]],
        identifiers: Dict[str, List[str]],
    ) -> str:
        has_expected = bool([x for x in expected_behavior + expected_outputs if str(x).strip()])
        has_fault = bool(fault_locations) or bool(identifiers.get("files")) or bool(
            identifiers.get("functions")
        )
        if has_expected and has_fault:
            return "explicit_evidence_sufficient"
        missing = []
        if not has_expected:
            missing.append("expected_behavior")
        if not has_fault:
            missing.append("fault_location_clues")
        return "missing_" + "_and_".join(missing)

    @classmethod
    def build_m1_refinement_prompt(
        cls,
        *,
        issue_text: str,
        observed_behavior: List[str],
        expected_behavior: List[str],
        expected_outputs: List[str],
        fault_locations: List[Dict[str, Any]],
        identifiers: Dict[str, List[str]],
    ) -> str:
        payload = {
            "prompt_version": cls.PROMPT_VERSION,
            "task": "Infer only missing M1 issue clues from the pre-patch issue text.",
            "constraints": [
                "Return strict JSON only.",
                "Do not use patch, post-patch, golden tests, golden patch lines, M8, Fail-to-Pass, or Patch Hit Rate data.",
                "Do not invent file or function names without repository evidence.",
            ],
            "required_schema": {
                "implicit_fault_locations": [
                    {"file_path": "repo/relative.py", "function_name": "optional"}
                ],
                "inferred_EB": ["expected behavior text"],
                "inferred_fault_location_clues": [
                    {"file_path": "repo/relative.py", "function_name": "optional"}
                ],
                "confidence": 0.0,
            },
            "issue_text": issue_text,
            "existing_rule_based_evidence": {
                "observed_behavior": observed_behavior,
                "expected_behavior": expected_behavior,
                "expected_outputs": expected_outputs,
                "fault_locations": fault_locations,
                "identifiers": identifiers,
            },
        }
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def parse_m1_refinement_response(raw: str) -> Dict[str, Any]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "valid": False,
                "parser_status": "malformed_json",
                "fallback_reason": "malformed_json",
            }
        if not isinstance(payload, dict):
            return {
                "valid": False,
                "parser_status": "non_object_json",
                "fallback_reason": "non_object_json",
            }
        if IssueClueExtractor._contains_prohibited_data(payload):
            return {
                "valid": False,
                "parser_status": "prohibited_data_rejected",
                "fallback_reason": "prohibited_data",
            }
        required = {
            "implicit_fault_locations",
            "inferred_EB",
            "inferred_fault_location_clues",
            "confidence",
        }
        if any(key not in payload for key in required):
            return {
                "valid": False,
                "parser_status": "missing_required_fields",
                "fallback_reason": "missing_required_fields",
            }
        if not isinstance(payload["implicit_fault_locations"], list):
            return {
                "valid": False,
                "parser_status": "invalid_implicit_fault_locations",
                "fallback_reason": "invalid_implicit_fault_locations",
            }
        if not isinstance(payload["inferred_fault_location_clues"], list):
            return {
                "valid": False,
                "parser_status": "invalid_inferred_fault_location_clues",
                "fallback_reason": "invalid_inferred_fault_location_clues",
            }
        if not isinstance(payload["inferred_EB"], list) or not all(
            isinstance(item, str) for item in payload["inferred_EB"]
        ):
            return {
                "valid": False,
                "parser_status": "invalid_inferred_EB",
                "fallback_reason": "invalid_inferred_EB",
            }
        try:
            confidence = float(payload["confidence"])
        except (TypeError, ValueError):
            return {
                "valid": False,
                "parser_status": "invalid_confidence",
                "fallback_reason": "invalid_confidence",
            }
        confidence_norm = max(0.0, min(1.0, confidence))
        locations = payload["implicit_fault_locations"] + payload["inferred_fault_location_clues"]
        if not all(isinstance(item, dict) for item in locations):
            return {
                "valid": False,
                "parser_status": "invalid_fault_location_item",
                "fallback_reason": "invalid_fault_location_item",
            }
        return {
            "valid": True,
            "parser_status": "parsed",
            "fallback_reason": None,
            "implicit_fault_locations": payload["implicit_fault_locations"],
            "inferred_fault_location_clues": payload["inferred_fault_location_clues"],
            "inferred_EB": [item.strip() for item in payload["inferred_EB"] if item.strip()],
            "confidence": confidence,
            "confidence_norm": confidence_norm,
        }

    @staticmethod
    def _contains_prohibited_data(value: Any) -> bool:
        prohibited = {
            "patch",
            "golden_patch",
            "golden_patch_lines",
            "test_patch",
            "patched_source",
            "patched_repo",
            "post_patch",
            "post_patch_outcome",
            "post_patch_results",
            "after_patch",
            "fail_to_pass",
            "patch_hit_rate",
            "m8_results",
        }
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key).lower()
                if key_text in prohibited:
                    return True
                if IssueClueExtractor._contains_prohibited_data(item):
                    return True
        elif isinstance(value, list):
            return any(IssueClueExtractor._contains_prohibited_data(item) for item in value)
        elif isinstance(value, str):
            lowered = value.lower()
            return any(token in lowered for token in prohibited)
        return False

    def _validate_m1_inferred_locations(
        self,
        locations: List[Dict[str, Any]],
        *,
        repository_path: Path | None,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if repository_path is None or not repository_path.exists():
            return [], [
                {"location": dict(location), "reason": "repository_evidence_unavailable"}
                for location in locations
            ]
        evidence = self._repository_python_symbol_evidence(repository_path)
        valid: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for location in locations:
            raw_path = str(location.get("file_path", "")).replace("\\", "/").strip()
            raw_function = str(location.get("function_name", "") or "").strip()
            matched_path = self._match_repository_path(raw_path, evidence)
            if matched_path is None:
                rejected.append({"location": dict(location), "reason": "unknown_file"})
                continue
            if raw_function and raw_function not in evidence.get(matched_path, set()):
                rejected.append({"location": dict(location), "reason": "unknown_function"})
                continue
            key = (matched_path, raw_function)
            if key in seen:
                continue
            seen.add(key)
            valid.append(
                {
                    "file_path": matched_path,
                    "function_name": raw_function,
                    "source": "m1_llm_refinement",
                    "confidence": "medium",
                }
            )
        return valid, rejected

    @staticmethod
    def _repository_python_symbol_evidence(repository_path: Path) -> Dict[str, set[str]]:
        evidence: Dict[str, set[str]] = {}
        for path in repository_path.rglob("*.py"):
            rel_path = str(path.relative_to(repository_path)).replace("\\", "/")
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except Exception:
                evidence[rel_path] = set()
                continue
            names: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(node.name)
            evidence[rel_path] = names
        return evidence

    @staticmethod
    def _match_repository_path(raw_path: str, evidence: Mapping[str, set[str]]) -> str | None:
        normalized = raw_path.lstrip("./")
        if normalized in evidence:
            return normalized
        matches = [path for path in evidence if path.endswith(normalized) or normalized.endswith(path)]
        if len(matches) == 1:
            return matches[0]
        return None

    def _split_lines(self, text: str) -> List[str]:
        lines = [line.strip() for line in text.splitlines()]
        return [line for line in lines if line]

    def _extract_expected_behavior(self, text: str) -> List[str]:
        lines = self._split_lines(text)
        results = []

        normative = re.compile(
            r"\b(?:expected|should|must|ought\s+to|be\s+able\s+to|"
            r"is\s+required\s+to|needs?\s+to)\b",
            re.IGNORECASE,
        )
        in_traceback = False
        in_fence = False
        for line in lines:
            lower = line.lower()
            if line.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            if line.startswith("Traceback (most recent call last)"):
                in_traceback = True
                continue
            if in_traceback:
                if re.search(r"\b[A-Z][A-Za-z0-9_.]*(?:Error|Exception):", line):
                    in_traceback = False
                continue
            if line.startswith("- [") or line.startswith("* ["):
                continue
            if re.search(r"\b[A-Z][A-Za-z0-9_.]*(?:Error|Exception):", line):
                continue
            if normative.search(lower):
                results.append(line)

        return self._dedup(results)

    def _extract_observed_behavior(self, text: str) -> List[str]:
        lines = self._split_lines(text)
        results = []

        keywords = [
            "actual",
            "instead",
            "error",
            "wrong",
            "fails",
            "failure",
            "returns",
            "raised",
            "traceback",
            "does not",
            "cannot",
        ]

        for line in lines:
            lower = line.lower()
            if any(k in lower for k in keywords):
                results.append(line)

        return self._dedup(results)

    def _extract_repro_conditions(self, text: str) -> List[str]:
        lines = self._split_lines(text)
        results = []

        keywords = [
            "consider the following",
            "reproduce",
            "steps",
            "example",
            "when",
            "if",
            "using",
        ]

        for line in lines:
            lower = line.lower()
            if any(k in lower for k in keywords):
                results.append(line)

        return self._dedup(results)

    def _extract_environment(self, text: str) -> List[str]:
        lines = self._split_lines(text)
        results = []

        keywords = [
            "version",
            "python ",
            "python:",
            "ubuntu",
            "linux",
            "windows",
            "macos",
            "os:",
        ]

        for line in lines:
            lower = line.lower()

            # 코드블록/코드라인 제외
            if line.startswith("```"):
                continue
            if line.startswith("from ") or line.startswith("import "):
                continue
            if "=" in line and "(" in line:
                continue

            if any(k in lower for k in keywords):
                results.append(line)

        return self._dedup(results)

    # 식별자 추출 상한 (LLM 프롬프트 노이즈 방지)
    _MAX_FUNCTIONS = 10
    _MAX_CLASSES = 10

    _SYSTEM_DETAIL_KEYWORDS = {
        "system details", "matplotlib version", "python version",
        "operating system", "jupyter version", "numpy", "scipy",
        "pyerfa", "platform", "windows", "linux", "macos",
    }

    def _is_system_or_output_block(
        self,
        code: str,
        language: str = "",
        context_before: str = "",
    ) -> bool:
        """Return True for environment dumps or plain outputs, not repro code."""
        language = (language or "").lower()
        context = (context_before or "").lower()
        stripped = code.strip()
        lower = stripped.lower()
        if not stripped:
            return True
        if any(k in context for k in self._SYSTEM_DETAIL_KEYWORDS):
            return True
        if any(k in context for k in ("how to reproduce", "reproduce", "file ", "code example", "minimal example")):
            return False
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if not lines:
            return True
        code_like = sum(
            1
            for line in lines
            if (
                line.startswith(("from ", "import ", "def ", "class ", "with ", "for ", "if ", "assert ", ">>> "))
                or "=" in line
                or re.search(r"\w+\s*\(", line)
            )
        )
        version_like = sum(
            1
            for line in lines
            if re.search(r"\b(version|python|numpy|scipy|matplotlib|windows|linux|macos|jupyter|pyerfa)\b", line.lower())
        )
        array_output_like = bool(re.fullmatch(r"\[?[-+0-9eE.,\s]+\]?", stripped))
        return (
            code_like == 0
            or (code_like <= 1 and version_like >= max(1, len(lines) // 2))
            or array_output_like
        )

    def _extract_identifiers(self, text: str) -> Dict[str, List[str]]:
        """이슈 텍스트에서 Python 식별자를 추출한다.

        전략:
          1) 코드 영역(fence/inline) 우선 추출 — 신뢰도 높음
          2) 코드 영역에서 충분히 얻지 못한 경우에만 산문 보완
             (단, 산문 클래스는 더 엄격한 필터 적용)
          3) 함수/클래스 각각 상한(_MAX_FUNCTIONS/_MAX_CLASSES) 적용
        """
        functions_code: set = set()
        functions_prose: set = set()
        classes_code: set = set()
        classes_prose: set = set()
        files: set = set()
        exceptions: set = set()
        dotted_apis: set = set()

        # ── 코드 영역 위치 계산 ──
        code_spans: list = []
        for m in re.finditer(r"```\s*(\w*)\s*\n(.*?)```", text, re.DOTALL):
            preceding = text[max(0, m.start() - 200):m.start()].strip()
            preceding_lines = [l.strip() for l in preceding.splitlines() if l.strip()]
            context_before = preceding_lines[-1] if preceding_lines else ""
            if self._is_system_or_output_block(m.group(2), m.group(1), context_before):
                continue
            code_spans.append((m.start(2), m.end(2)))
        for m in re.finditer(r"`([^`\n]+)`", text):
            code_spans.append((m.start(1), m.end(1)))

        def _is_in_code(pos: int) -> bool:
            return any(s <= pos < e for s, e in code_spans)

        # ── 함수 패턴: 코드/산문 분리 추출 ──
        for match in self.func_pattern.finditer(text):
            fn = match.group(0)[:-1]
            if fn in self.function_stopwords:
                continue
            if _is_in_code(match.start()):
                functions_code.add(fn)
            else:
                functions_prose.add(fn)

        # Preserve explicit dotted API paths from inline/fenced code.  The
        # public API and its tail are both useful; neither is treated as a
        # proven implementation target.
        for match in re.finditer(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b", text):
            if not _is_in_code(match.start()):
                continue
            dotted = match.group(0)
            if dotted.lower().endswith((".py", ".json", ".yaml", ".yml")):
                continue
            dotted_apis.add(dotted)
            tail = dotted.split(".")[-1]
            if tail not in self.function_stopwords and not tail[:1].isupper():
                functions_code.add(tail)

        # ── 클래스 패턴: 코드/산문 분리 추출 ──
        for match in self.class_pattern.finditer(text):
            name = match.group(0)
            if name.endswith(("Error", "Exception")):
                continue
            if name in self.class_stopwords:
                continue
            if name.isupper():
                continue
            has_underscore = "_" in name
            # 내부 대문자 개수: Python 복합 클래스명(FileSystemStorage)은 1개 이상
            # 단순 영단어(Browser, Additional, Enabling)는 0개
            interior_upper = sum(1 for c in name[1:] if c.isupper())
            is_compound_pascal = interior_upper >= 1 or has_underscore

            if _is_in_code(match.start()):
                # 코드 영역: 소문자 포함이면 허용 (주석/docstring 내 단순 단어 제외)
                has_lower = any(c.islower() for c in name[1:])
                if has_lower or has_underscore:
                    classes_code.add(name)
            else:
                # 산문: 복합 PascalCase(FileSystemStorage)만 허용, 단순 영단어 제외
                if is_compound_pascal:
                    classes_prose.add(name)

        # ── 파일/예외 ──
        files.update(self._extract_file_references(text))
        for m in self.exception_pattern.finditer(text):
            exceptions.add(m.group(0))

        # ── 최종 병합: 코드 우선, 필요 시 산문 보완, 상한 적용 ──
        final_functions = sorted(functions_code)
        if len(final_functions) < self._MAX_FUNCTIONS:
            for fn in sorted(functions_prose):
                if fn not in functions_code:
                    final_functions.append(fn)
                if len(final_functions) >= self._MAX_FUNCTIONS:
                    break

        final_classes = sorted(classes_code)
        if len(final_classes) < self._MAX_CLASSES:
            for cls in sorted(classes_prose):
                if cls not in classes_code:
                    final_classes.append(cls)
                if len(final_classes) >= self._MAX_CLASSES:
                    break

        return {
            "functions": sorted(final_functions[:self._MAX_FUNCTIONS]),
            "classes": sorted(final_classes[:self._MAX_CLASSES]),
            "files": sorted(files),
            "exceptions": sorted(exceptions),
            "dotted_apis": sorted(dotted_apis)[: self._MAX_FUNCTIONS],
        }

    def _extract_file_references(self, text: str) -> List[str]:
        """Extract repo-relative file hints from prose, line refs, and GitHub URLs."""
        files: set[str] = set()

        def add_path(value: str) -> None:
            path = (value or "").strip().strip("`'\"()[]{}.,;:")
            path = path.replace("\\", "/")
            path = re.sub(r"^(?:\./|a/|b/)+", "", path)
            if "/blob/" in path or path.startswith(("http", "www.", "github.com/", "com/")):
                return
            if path and "." in Path(path).name:
                files.add(path)

        for m in self.file_pattern.finditer(text):
            add_path(m.group(0))

        # path/to/file.py:859 and Windows variants.
        for m in re.finditer(r"(?<!\w)([\w./\\-]+\.(?:py|txt|json|yaml|yml|ini|cfg)):(\d+)", text):
            add_path(m.group(1))

        # GitHub blob URL:
        # https://github.com/org/repo/blob/<sha-or-branch>/path/to/file.py#L10
        blob_re = re.compile(
            r"https?://github\.com/[^/\s]+/[^/\s]+/blob/[^/\s]+/"
            r"([^\s#?]+?\.(?:py|txt|json|yaml|yml|ini|cfg))(?:[#?][^\s]*)?",
            re.IGNORECASE,
        )
        for m in blob_re.finditer(text):
            add_path(m.group(1))

        return sorted(files)

    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """
        이슈 텍스트에서 코드 블록을 추출한다.
        마크다운 ``` 코드 펜스와 >>> 인터랙티브 예시를 모두 처리한다.
        각 블록에 대해 앞쪽 문맥(context_before)도 함께 저장한다.
        """
        blocks: List[Dict[str, str]] = []

        # 1) 마크다운 코드 펜스 추출: ```python ... ``` 또는 ``` ... ```
        fence_pattern = re.compile(
            r"```\s*(\w*)\s*\n(.*?)```",
            re.DOTALL,
        )
        for match in fence_pattern.finditer(text):
            language = match.group(1) or "python"
            code = match.group(2).strip()
            if not code:
                continue

            # 코드 블록 바로 앞의 텍스트(최대 200자)를 context로 저장
            start = match.start()
            preceding = text[max(0, start - 200):start].strip()
            # 마지막 문장만 추출
            preceding_lines = [l.strip() for l in preceding.splitlines() if l.strip()]
            context_before = preceding_lines[-1] if preceding_lines else ""

            blocks.append({
                "language": language,
                "code": code,
                "context_before": context_before,
                "is_system_or_output": self._is_system_or_output_block(
                    code, language, context_before
                ),
            })

        # 2) 코드 펜스 내의 >>> 인터랙티브 블록에서 코드+출력 분리
        #    (이미 위에서 추출한 블록 안에서 >>> 패턴을 해석)
        for block in blocks:
            code = block["code"]
            if ">>>" not in code:
                continue

            input_lines = []
            output_lines = []
            for line in code.splitlines():
                stripped = line.strip()
                if stripped.startswith(">>> "):
                    input_lines.append(stripped[4:])
                elif stripped.startswith("..."):
                    input_lines.append(stripped[4:] if len(stripped) > 4 else "")
                else:
                    output_lines.append(stripped)

            if input_lines:
                block["interactive_input"] = "\n".join(input_lines)
            if output_lines:
                block["interactive_output"] = "\n".join(output_lines)

        return blocks

    def _extract_output_examples(
        self, text: str, code_blocks: List[Dict[str, str]]
    ) -> tuple:
        """
        코드 블록의 context_before를 분석하여 기대 출력과 실제(버그) 출력을 분류한다.
        Returns: (expected_outputs, actual_outputs)
        """
        # "expect" 포함 변형도 매칭 (might expect, as you expect, etc.)
        expected_keywords = {
            "expected", "as expected", "correct", "should", "works", "expect",
            "ideally", "want", "desired", "suppose",
        }
        actual_keywords = {
            "however", "suddenly", "bug", "wrong", "instead", "no longer",
            "does not", "doesn't", "broken", "fail", "issue", "problem",
            "currently", "actually", "incorrectly", "unexpected",
        }

        expected_outputs: List[str] = []
        actual_outputs: List[str] = []

        blocks_with_output = []
        for block in code_blocks:
            output = block.get("interactive_output", "").strip()

            if not output:
                # interactive 출력이 없어도 traceback 블록이면 actual_output으로 추가
                code = block.get("code", "")
                if re.search(
                    r"Traceback \(most recent call last\)|^\w+Error:",
                    code,
                    re.MULTILINE,
                ):
                    lines = code.strip().splitlines()
                    last_err = next(
                        (
                            l.strip()
                            for l in reversed(lines)
                            if re.match(r"\w+Error:", l.strip())
                        ),
                        None,
                    )
                    if last_err:
                        actual_outputs.append(last_err)
                continue

            blocks_with_output.append((block, output))

        for i, (block, output) in enumerate(blocks_with_output):
            context = block.get("context_before", "").lower()

            is_expected = any(kw in context for kw in expected_keywords)
            is_actual = any(kw in context for kw in actual_keywords)

            if is_actual:
                actual_outputs.append(output)
            elif is_expected:
                expected_outputs.append(output)
            else:
                # 순서 기반 fallback: 마지막 블록 → actual(버그 시연), 첫 블록 → expected(정상 동작)
                if i == 0 and len(blocks_with_output) > 1:
                    expected_outputs.append(output)
                else:
                    actual_outputs.append(output)

        # Inline comment patterns in minimal examples, e.g.
        # "# correct ... => array([...])" / "# incorrect ... => array([...])".
        for block in code_blocks:
            comment_label = ""
            for line in block.get("code", "").splitlines():
                stripped = line.strip()
                if not stripped.startswith("#"):
                    continue
                comment = stripped[1:].strip()
                comment_lower = comment.lower()
                if "=>" not in comment:
                    if any(k in comment_lower for k in ("incorrect", "actual", "bug", "wrong")):
                        comment_label = "actual"
                    elif any(k in comment_lower for k in ("correct", "expected", "ideally", "desired")):
                        comment_label = "expected"
                    continue
                label, value = comment.split("=>", 1)
                label_lower = label.lower()
                value = value.strip()
                if not value:
                    continue
                if comment_label == "expected" or any(k in label_lower for k in ("correct", "expected", "ideally", "desired")):
                    expected_outputs.append(value)
                if comment_label == "actual" or any(k in label_lower for k in ("incorrect", "actual", "bug", "wrong")):
                    actual_outputs.append(value)

        return self._dedup(expected_outputs), self._dedup(actual_outputs)

    def _extract_error_keywords(
        self,
        text: str,
        identifiers: Dict[str, List[str]],
        code_blocks: List[Dict[str, str]],
    ) -> List[str]:
        """이슈에서 에러/예외 관련 핵심 키워드를 추출한다.

        1) identifiers.exceptions에서 이미 추출된 예외 타입 포함
        2) code_blocks의 interactive_output에서 "ExcType: message" 패턴 추출
        3) 이슈 텍스트 전체에서 예외 타입명 보완
        """
        keywords: List[str] = []

        # 1) identifiers에서 이미 추출된 예외 타입 그대로 포함
        keywords.extend(identifiers.get("exceptions", []))

        # 2) interactive_output에서 "ExcType: message" 한 줄짜리 에러 메시지 추출
        for block in code_blocks:
            output = block.get("interactive_output", "")
            for m in re.finditer(
                r"\b([A-Z]\w*(?:Error|Exception)): ([^\n]{5,100})",
                output,
            ):
                entry = m.group(0)[:120]
                if entry not in keywords:
                    keywords.append(entry)

        # 3) 이슈 텍스트 전체에서 예외 타입명 보완 (identifiers에 없는 것만)
        existing_exc = set(identifiers.get("exceptions", []))
        for m in re.finditer(r"\b([A-Z]\w*(?:Error|Exception))\b", text):
            exc = m.group(0)
            if exc not in existing_exc and exc not in keywords:
                keywords.append(exc)
                existing_exc.add(exc)

        return self._dedup(keywords)[:10]

    # traceback frame 패턴: File "path", line N, in func_name
    _TRACEBACK_FRAME = re.compile(
        r'File\s+"([^"]+\.py)",\s+line\s+(\d+),\s+in\s+(\S+)'
    )
    # site-packages / stdlib 등 제외할 경로 패턴
    _SKIP_PATH_PATTERNS = re.compile(
        r"site-packages|dist-packages|lib/python\d+\.\d+/(?!site)|"
        r"\.tox/|/tmp/|\\tmp\\|<.*>"
    )

    def _extract_fault_locations(
        self,
        text: str,
        code_examples: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """이슈 텍스트의 traceback에서 fault location 후보를 추출한다.

        `File "path/to/file.py", line N, in func_name` 패턴을 파싱하여
        repo 내부 파일에 해당하는 항목만 반환한다. (site-packages / stdlib 제외)

        Returns:
            [{"file_path": "...", "line_no": N, "function_name": "..."}, ...]
            파일 경로는 원문 절대 경로 그대로 — CodeContextExtractor가 suffix 매칭으로 처리.
        """
        seen: set[str] = set()
        results: List[Dict[str, Any]] = []

        # Only actual traceback text is high-confidence. Reproduction snippets
        # and environment dumps must not be promoted to CRITICAL fault locations.
        search_texts = [text]
        search_texts.extend(
            b.get("code", "")
            for b in code_examples
            if not b.get("is_system_or_output")
            and "Traceback (most recent call last)" in b.get("code", "")
        )

        for src in search_texts:
            if "Traceback (most recent call last)" not in src:
                continue
            for m in self._TRACEBACK_FRAME.finditer(src):
                file_path = m.group(1)
                line_no = int(m.group(2))
                func_name = m.group(3)

                # stdlib / venv / tox 경로 제외
                if self._SKIP_PATH_PATTERNS.search(file_path):
                    continue

                key = f"{file_path}:{func_name}"
                if key in seen:
                    continue
                seen.add(key)

                results.append({
                    "file_path": file_path,
                    "line_no": line_no,
                    "function_name": func_name,
                    "source": "traceback",
                    "confidence": "high",
                })

        # Non-traceback file/line references are weaker than stack frames but
        # still valuable for IR-style fault localization.
        for m in re.finditer(
            r"(?<!\w)([\w./\\-]+\.py):(\d+)(?!\d)",
            text,
        ):
            file_path = m.group(1).replace("\\", "/")
            line_no = int(m.group(2))
            if self._SKIP_PATH_PATTERNS.search(file_path):
                continue
            key = f"{file_path}:line:{line_no}"
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "file_path": file_path,
                "line_no": line_no,
                "function_name": "",
                "source": "file_reference",
                "confidence": "medium",
            })

        github_line_re = re.compile(
            r"https?://github\.com/[^/\s]+/[^/\s]+/blob/[^/\s]+/"
            r"([^\s#?]+?\.py)#L(\d+)",
            re.IGNORECASE,
        )
        for m in github_line_re.finditer(text):
            file_path = m.group(1).replace("\\", "/")
            line_no = int(m.group(2))
            key = f"{file_path}:line:{line_no}"
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "file_path": file_path,
                "line_no": line_no,
                "function_name": "",
                "source": "file_reference",
                "confidence": "medium",
            })

        # 상한 10건
        return results[:10]

    def _dedup(self, items: List[str]) -> List[str]:
        seen = set()
        results = []

        for item in items:
            normalized = item.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                results.append(normalized)

        return results

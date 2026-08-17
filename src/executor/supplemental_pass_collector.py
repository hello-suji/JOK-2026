"""Bounded, target-aware pre-patch PASS-spectrum acquisition for M6.

The collector deliberately owns discovery/ranking/acceptance policy separately
from Docker execution.  M6 supplies an executor callback so unit tests can use
deterministic spectra while the production runner executes each node in the
same pre-patch source view.
"""

from __future__ import annotations

import ast
import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


DEFAULT_MAX_SUPPLEMENTAL_PASS_TESTS = 6
V37_MAX_SUPPLEMENTAL_PASS_TESTS = 10
DEFAULT_MIN_DISTINCT_PASSING_TESTS = 3
_INSUFFICIENT_PASS_STOP_REASONS = {
    "SBFL_UNAVAILABLE_INSUFFICIENT_P",
    "execution_budget_exhausted",
    "candidate_pool_exhausted",
    "no_collectable_candidates",
    "no_passing_candidates",
}


def normalize_supplemental_exhaustion(payload: Mapping[str, Any]) -> str | None:
    """Classify exhausted insufficient-P telemetry while retaining raw reason."""
    minimum = int(
        payload.get("min_distinct_passing_tests", DEFAULT_MIN_DISTINCT_PASSING_TESTS)
        or DEFAULT_MIN_DISTINCT_PASSING_TESTS
    )
    if (
        str(payload.get("stop_reason") or "") in _INSUFFICIENT_PASS_STOP_REASONS
        and int(payload.get("valid_pass_count", 0) or 0) < minimum
    ):
        return "SBFL_UNAVAILABLE_INSUFFICIENT_P"
    return None
MAX_STATIC_TEST_DISCOVERY_FILES = 512
V30_MAX_SOURCE_FILES = 16
V30_MAX_TEST_FILES = 32
V30_MAX_TEST_NODES = 256
V31_MAX_TEST_FILES = V30_MAX_TEST_FILES
V31_MAX_TEST_NODES = V30_MAX_TEST_NODES
# Repository test suites use more than the pytest ``test_*.py`` convention;
# Pylint's live G1 artifacts, for example, expose ``unittest_*.py`` files.
# Keep this a filename/layout filter only—node discovery remains AST-based and
# bounded by the v30 budgets.
_TEST_FILE_RE = re.compile(
    r"(?:^|/)(?:test|unittest)[^/]*\.py$"
    r"|(?:^|/)tests?/[^/]+\.py$"
    r"|(?:^|/)[^/]+_test\.py$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SupplementalTestCandidate:
    """One statically discovered repository test node."""

    nodeid: str
    test_file: str
    discovery_reason: str
    discovery_rank: int
    score: float
    runner: str = "pytest"

    @property
    def identity(self) -> str:
        return self.nodeid


@dataclass
class SupplementalPassCollectionResult:
    """Bounded collection outcome and complete candidate-level provenance."""

    attempted_count: int = 0
    valid_pass_count: int = 0
    rejected_count: int = 0
    candidate_pool_size: int = 0
    max_supplemental_pass_tests: int = DEFAULT_MAX_SUPPLEMENTAL_PASS_TESTS
    min_distinct_passing_tests: int = DEFAULT_MIN_DISTINCT_PASSING_TESTS
    accepted_records: list[dict[str, Any]] = field(default_factory=list)
    candidate_records: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "not_started"
    candidate_pool_before_pruning: int = 0
    candidate_pool_after_pruning: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": "m6-supplemental-pass-collection-v1",
            "attempted_count": self.attempted_count,
            "valid_pass_count": self.valid_pass_count,
            "rejected_count": self.rejected_count,
            "candidate_pool_size": self.candidate_pool_size,
            "max_supplemental_pass_tests": self.max_supplemental_pass_tests,
            "min_distinct_passing_tests": self.min_distinct_passing_tests,
            "accepted_records": list(self.accepted_records),
            "candidate_records": list(self.candidate_records),
            "stop_reason": self.stop_reason,
            "candidate_pool_before_pruning": self.candidate_pool_before_pruning,
            "candidate_pool_after_pruning": self.candidate_pool_after_pruning,
        }
        classification = normalize_supplemental_exhaustion(payload)
        if classification:
            payload["m7_diagnostic_classification"] = classification
        return payload


ExecutionCallback = Callable[[SupplementalTestCandidate], Mapping[str, Any]]


class SupplementalPassCollector:
    """Discover and validate a bounded set of pre-patch repository PASS tests."""

    def __init__(
        self,
        *,
        max_supplemental_pass_tests: int | None = None,
        min_distinct_passing_tests: int = DEFAULT_MIN_DISTINCT_PASSING_TESTS,
        feature_profile: str | None = None,
        max_source_files: int = V30_MAX_SOURCE_FILES,
        max_test_files: int = V30_MAX_TEST_FILES,
        max_test_nodes: int = V30_MAX_TEST_NODES,
    ) -> None:
        if max_supplemental_pass_tests is None:
            max_supplemental_pass_tests = (
                V37_MAX_SUPPLEMENTAL_PASS_TESTS
                if feature_profile == "v37"
                else DEFAULT_MAX_SUPPLEMENTAL_PASS_TESTS
            )
        if max_supplemental_pass_tests < 0:
            raise ValueError("max_supplemental_pass_tests must be non-negative")
        if min_distinct_passing_tests < 1:
            raise ValueError("min_distinct_passing_tests must be positive")
        self.max_supplemental_pass_tests = int(max_supplemental_pass_tests)
        self.min_distinct_passing_tests = int(min_distinct_passing_tests)
        self.feature_profile = feature_profile
        self.max_source_files = int(max_source_files)
        self.max_test_files = int(max_test_files)
        self.max_test_nodes = int(max_test_nodes)
        self.last_discovery_stats: dict[str, int] = {
            "candidate_pool_before_pruning": 0,
            "candidate_pool_after_pruning": 0,
            "node_pool_before_pruning": 0,
            "node_pool_after_pruning": 0,
        }
        if min(self.max_source_files, self.max_test_files, self.max_test_nodes) < 1:
            raise ValueError("v30 collector bounds must be positive")

    def collect(
        self,
        *,
        context: Mapping[str, Any],
        clue: Mapping[str, Any],
        instance_id: str,
        candidate_id: str,
        candidate_hash: str,
        outer_iteration: int | None,
        attempted_test_ids: Iterable[str] = (),
        existing_passing_ids: Iterable[str] = (),
        accepted_spectrum_hashes: Iterable[str] = (),
        execute: ExecutionCallback,
        timestamp: str | None = None,
    ) -> SupplementalPassCollectionResult:
        """Collect valid PASS records until the threshold or fixed budget.

        ``execute`` must run one candidate against the intended before-patch
        checkout and return a mapping containing ``test_results``,
        ``covered_sut_lines`` (or ``coverage_data.SUT_lines``), and provenance.
        No post-patch or benchmark answer fields are accepted by this class.
        """
        result = SupplementalPassCollectionResult(
            max_supplemental_pass_tests=self.max_supplemental_pass_tests,
            min_distinct_passing_tests=self.min_distinct_passing_tests,
        )
        attempted = {str(item) for item in attempted_test_ids if str(item)}
        accepted_ids = {str(item) for item in existing_passing_ids if str(item)}
        accepted_ids = set(accepted_ids)
        distinct_spectra_required = self.feature_profile in {"v31", "v37"}
        v31 = self.feature_profile == "v31"
        accepted_spectra: set[str] = {
            str(value) for value in accepted_spectrum_hashes if str(value)
        }
        if len(accepted_ids) >= self.min_distinct_passing_tests:
            result.stop_reason = "reached_required_distinct_pass" if v31 else "already_sufficient_current_passes"
            return result

        candidates = self.discover(
            context=context,
            clue=clue,
            attempted_test_ids=attempted,
            candidate_id=candidate_id,
        )
        result.candidate_pool_size = len(candidates)
        result.candidate_pool_before_pruning = int(self.last_discovery_stats.get("candidate_pool_before_pruning", len(candidates)))
        result.candidate_pool_after_pruning = len(candidates)
        if not candidates:
            # An empty bounded M2 pool is an exhausted evidence search, not a
            # successful collection. Keep the M7-facing diagnostic stable.
            result.stop_reason = "no_collectable_candidates" if v31 else "SBFL_UNAVAILABLE_INSUFFICIENT_P"
            return result

        for candidate in candidates:
            if result.attempted_count >= self.max_supplemental_pass_tests:
                result.stop_reason = "execution_budget_exhausted" if v31 else "supplemental_execution_budget_exhausted"
                break
            if len(accepted_ids) >= self.min_distinct_passing_tests:
                result.stop_reason = "reached_required_distinct_pass" if v31 else "minimum_distinct_passes_reached"
                break

            result.attempted_count += 1
            attempted.add(candidate.identity)
            started = time.time()
            try:
                raw = dict(execute(candidate))
            except Exception as exc:  # executor failures are diagnostic records
                raw = {
                    "execution_status": "ERROR",
                    "error_message": f"supplemental executor error: {exc}",
                    "test_results": {},
                }
            elapsed = round(time.time() - started, 3)
            record = self._provenance_record(
                raw,
                candidate=candidate,
                instance_id=instance_id,
                candidate_id=candidate_id,
                candidate_hash=candidate_hash,
                outer_iteration=outer_iteration,
                timestamp=timestamp,
                elapsed_sec=elapsed,
            )
            accepted, reason = self._validate_pass_record(
                raw,
                candidate=candidate,
                accepted_ids=accepted_ids,
                context=context,
                allow_test_named_sut=self.feature_profile == "v37",
            )
            record["accepted_into_P"] = accepted
            record["rejection_reason"] = reason
            spectrum_hash = self._spectrum_hash(
                raw,
                allow_test_named_sut=self.feature_profile == "v37",
            )
            if distinct_spectra_required and accepted and not spectrum_hash:
                accepted = False
                reason = "malformed_or_empty_spectrum"
                record["accepted_into_P"] = False
                record["rejection_reason"] = reason
                record["collection_status"] = "REJECTED"
            record["spectrum_hash"] = spectrum_hash
            record["candidate_rank"] = candidate.discovery_rank
            record["collection_status"] = "PASS" if accepted else "REJECTED"
            record["accepted_distinct"] = False
            if distinct_spectra_required and accepted and spectrum_hash in accepted_spectra:
                accepted = False
                reason = "duplicate_spectrum"
                record["accepted_into_P"] = False
                record["rejection_reason"] = reason
                record["collection_status"] = "REJECTED"
            result.candidate_records.append(record)
            if accepted:
                canonical_id = str(record["supplemental_test_nodeid"])
                accepted_ids.add(canonical_id)
                accepted_spectra.add(spectrum_hash)
                record["accepted_distinct"] = True
                result.valid_pass_count += 1
                accepted_record = dict(raw)
                accepted_record.update(
                    {
                        "instance_id": instance_id,
                        "candidate_id": candidate_id,
                        "candidate_hash": candidate_hash,
                        "outer_iteration": outer_iteration,
                        "supplemental_pass": True,
                        "supplemental_test_nodeid": canonical_id,
                        "supplemental_test_file": candidate.test_file,
                        "discovery_reason": candidate.discovery_reason,
                        "discovery_rank": candidate.discovery_rank,
                        "accepted_into_P": True,
                        "source_mapping_status": "VALID_PRE_PATCH_SOURCE",
                        "provenance_timestamp": timestamp,
                    }
                )
                accepted_record.setdefault("test_id", canonical_id)
                accepted_record.setdefault("canonical_test_id", canonical_id)
                accepted_record.setdefault("test_nodeid", canonical_id)
                accepted_record.setdefault("canonical_test_nodeid", canonical_id)
                accepted_record.setdefault("test_results", {canonical_id: "PASSED"})
                result.accepted_records.append(accepted_record)
            else:
                result.rejected_count += 1

        if not result.stop_reason or result.stop_reason == "not_started":
            result.stop_reason = (
                "reached_required_distinct_pass" if v31 and len(accepted_ids) >= self.min_distinct_passing_tests else
                "minimum_distinct_passes_reached" if len(accepted_ids) >= self.min_distinct_passing_tests else
                "candidate_pool_exhausted" if v31 and result.attempted_count < self.max_supplemental_pass_tests else
                "execution_budget_exhausted" if v31 else "SBFL_UNAVAILABLE_INSUFFICIENT_P"
            )
        if len(accepted_ids) < self.min_distinct_passing_tests and result.attempted_count >= self.max_supplemental_pass_tests:
            result.stop_reason = "execution_budget_exhausted" if v31 else "SBFL_UNAVAILABLE_INSUFFICIENT_P"
        if v31 and result.attempted_count and result.valid_pass_count == 0 and result.stop_reason == "candidate_pool_exhausted":
            result.stop_reason = "no_passing_candidates"
        return result

    def discover(
        self,
        *,
        context: Mapping[str, Any],
        clue: Mapping[str, Any],
        attempted_test_ids: Iterable[str] = (),
        candidate_id: str = "",
    ) -> list[SupplementalTestCandidate]:
        """Rank nodes from the bounded M2 candidate-test pool only."""
        attempted = {str(item) for item in attempted_test_ids if str(item)}
        generated_node = str(candidate_id or "")
        target = self._target_location(clue, context)
        target_file = str(target.get("source_file") or "")
        target_function = str(target.get("target_function") or "")
        target_class = str(target.get("target_class") or "")
        target_module = target_file.rsplit("/", 1)[0] if "/" in target_file else ""
        runner = str((context.get("project_test_style") or {}).get("runner") or "pytest")
        repo_path = Path(str(context.get("repo_path") or ""))
        try:
            resolved_repo_path = repo_path.resolve(strict=True)
        except (OSError, RuntimeError):
            return []
        ranked: list[tuple[float, str, str, str]] = []
        seen: set[str] = set()
        entries = self._candidate_entries(
            context=context,
            clue=clue,
            target_file=target_file,
            target_module=target_module,
            target_function=target_function,
            target_class=target_class,
        )
        before_pruning = len(entries)
        self.last_discovery_stats["candidate_pool_before_pruning"] = before_pruning
        if self.feature_profile in {"v30", "v31"}:
            entries = entries[: self.max_test_files]
        self.last_discovery_stats["candidate_pool_after_pruning"] = len(entries)
        node_budget = self.max_test_nodes if self.feature_profile in {"v30", "v31"} else None
        node_count_before = 0
        for entry_index, entry in enumerate(entries):
            test_file = str(entry.get("path") or "").strip()
            if not self._valid_test_file(test_file):
                continue
            if test_file.endswith("/__init__.py") or test_file.endswith("/__init__.py"):
                continue
            path = repo_path / test_file
            try:
                resolved_path = path.resolve(strict=True)
                resolved_path.relative_to(resolved_repo_path)
            except (OSError, RuntimeError, ValueError):
                continue
            path = resolved_path
            nodes = (
                self._discover_v37_nodes(path, test_file)
                if self.feature_profile == "v37"
                else self._discover_nodes(path, test_file)
            )
            node_count_before += len(nodes)
            if not nodes:
                continue
            source_text = ""
            try:
                source_text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                pass
            score, reason = self._rank_file(
                test_file=test_file,
                source_text=source_text,
                entry_index=entry_index,
                target_file=target_file,
                target_module=target_module,
                target_function=target_function,
                target_class=target_class,
                clue=clue,
            )
            if entry.get("discovery_scope"):
                reason = f"{entry['discovery_scope']};{reason}"
            for node in nodes:
                if node_budget is not None and len(ranked) >= node_budget:
                    break
                if node in seen or node in attempted or node == generated_node:
                    continue
                seen.add(node)
                ranked.append((score, node, test_file, reason))
            if node_budget is not None and len(ranked) >= node_budget:
                break
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        if self.feature_profile == "v37":
            # Relevance determines only the preferred group.  Within it, the
            # exact nodeid is the deterministic order required by v37.
            ranked.sort(key=lambda item: (-int(item[0] > 1.0), item[1], item[2]))
        # A single high-ranked file can contain dozens of nodes.  Taking the
        # global prefix therefore spent the complete six-test budget in one
        # file in live v31 runs (and repeated the same fixture/collection
        # failure).  Preserve file relevance ordering, but round-robin nodes
        # across the ranked relevant files before returning to a second node
        # from any file.  This is deterministic and does not increase either
        # the discovery or execution budget.
        if self.feature_profile != "v37":
            by_file: dict[str, list[tuple[float, str, str, str]]] = {}
            file_order: list[str] = []
            for item in ranked:
                test_file = item[2]
                if test_file not in by_file:
                    by_file[test_file] = []
                    file_order.append(test_file)
                by_file[test_file].append(item)
            diversified: list[tuple[float, str, str, str]] = []
            depth = 0
            while len(diversified) < len(ranked):
                added = False
                for test_file in file_order:
                    nodes = by_file[test_file]
                    if depth < len(nodes):
                        diversified.append(nodes[depth])
                        added = True
                if not added:
                    break
                depth += 1
            ranked = diversified
        candidates = [
            SupplementalTestCandidate(
                nodeid=node,
                test_file=test_file,
                discovery_reason=reason,
                discovery_rank=index + 1,
                score=round(score, 4),
                runner=runner,
            )
            for index, (score, node, test_file, reason) in enumerate(ranked)
        ]
        self.last_discovery_stats["node_pool_before_pruning"] = node_count_before
        self.last_discovery_stats["node_pool_after_pruning"] = len(candidates)
        if self.feature_profile in {"v30", "v31"}:
            candidates = candidates[: self.max_test_nodes]
        return candidates

    @classmethod
    def _candidate_entries(
        cls,
        *,
        context: Mapping[str, Any],
        clue: Mapping[str, Any],
        target_file: str,
        target_module: str,
        target_function: str,
        target_class: str,
    ) -> list[dict[str, Any]]:
        """Build a bounded static candidate pool beyond M2's top-k files.

        The expansion is deliberately limited to repository test layout and
        static reference evidence. It never executes a repository-wide test
        sweep and never reads benchmark/golden/post-patch artifacts.
        """
        entries_by_path: dict[str, dict[str, Any]] = {}
        supplied = context.get("candidate_test_files") or []
        if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)):
            for index, entry in enumerate(supplied):
                if not isinstance(entry, Mapping):
                    continue
                path = str(entry.get("path") or "").strip()
                if path and path not in entries_by_path:
                    entries_by_path[path] = {
                        "path": path,
                        "discovery_scope": "M2_candidate_test_file",
                        "seed_index": index,
                    }

        repo_path = Path(str(context.get("repo_path") or ""))
        if cls._is_v30_context(context):
            # v30 discovery is strictly M2-owned.  Do not perform a recursive
            # repository sweep when the upstream candidate pool is empty.
            if str(context.get("feature_profile") or context.get("methodology_revision") or "") == "v30":
                return list(entries_by_path.values())
            # v31 permits a bounded local/static expansion while retaining
            # the same source-view and traversal caps.
        if not repo_path.exists() or not repo_path.is_dir():
            return list(entries_by_path.values())

        target_parts = [part for part in target_file.replace("\\", "/").split("/") if part]
        target_dir = repo_path.joinpath(*target_parts[:-1]) if target_parts else repo_path
        search_roots: list[Path] = []
        current = target_dir
        for _ in range(4):
            if current == repo_path.parent or not str(current).startswith(str(repo_path)):
                break
            for name in ("tests", "test"):
                candidate_root = current / name
                if candidate_root.is_dir():
                    search_roots.append(candidate_root)
            if current == repo_path:
                break
            current = current.parent
        for name in ("tests", "test"):
            root = repo_path / name
            if root.is_dir():
                search_roots.append(root)

        # Include package-local test directories discovered from repository
        # layout, but cap both directory and file traversal.
        try:
            layout_roots = sorted(
                (path for path in repo_path.rglob("tests") if path.is_dir()),
                key=lambda path: (cls._path_distance(path, target_dir), str(path)),
            )[:64]
            search_roots.extend(layout_roots)
        except OSError:
            pass

        static_tokens = {
            token.lower()
            for token in (
                Path(target_file).stem,
                target_function,
                *(str(item) for item in (context.get("candidate_source_files") or []) if isinstance(item, str)),
            )
            if token and len(token) >= 3
        }
        static_tokens.update(
            str(value).lower()
            for value in cls._clue_identifiers(context)
            if len(str(value)) >= 3
        )
        static_tokens.update(
            str(value).lower()
            for value in cls._mapping_identifiers(clue)
            if len(str(value)) >= 3
        )
        seen_paths: set[str] = set(entries_by_path)
        files: list[Path] = []
        for root in search_roots:
            try:
                files.extend(path for path in root.rglob("test*.py") if path.is_file())
            except OSError:
                continue
        unique_files = sorted(
            set(files),
            key=lambda path: (cls._path_distance(path, target_dir), str(path)),
        )[:MAX_STATIC_TEST_DISCOVERY_FILES]
        for path in unique_files:
            relative = path.relative_to(repo_path).as_posix()
            if not cls._valid_test_file(relative) or relative.endswith("/__init__.py"):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            lower = text.lower()
            matches = [token for token in static_tokens if token in lower]
            same_area = target_module and relative.startswith(target_module)
            if not matches and not same_area:
                continue
            if relative not in seen_paths:
                seen_paths.add(relative)
                entries_by_path[relative] = {
                    "path": relative,
                    "discovery_scope": (
                        "same_package_test_layout" if same_area else "static_reference_scan"
                    ),
                }
        return list(entries_by_path.values())

    @staticmethod
    def _is_v30_context(context: Mapping[str, Any]) -> bool:
        return str(context.get("feature_profile") or context.get("methodology_revision") or "") in {"v30", "v31"}

    @staticmethod
    def _spectrum_hash(
        raw: Mapping[str, Any], *, allow_test_named_sut: bool = False
    ) -> str:
        lines = raw.get("covered_sut_lines") or (raw.get("coverage_data") or {}).get("SUT_lines", [])
        normalized: set[tuple[str, int, str]] = set()
        malformed = False
        if isinstance(lines, Sequence) and not isinstance(lines, (str, bytes)):
            for item in lines:
                if isinstance(item, Mapping):
                    source_file = str(item.get("source_file") or item.get("file") or item.get("path") or "").replace("\\", "/")
                    if not SupplementalPassCollector._valid_sut_spectrum_path(
                        source_file,
                        allow_test_named_sut=allow_test_named_sut,
                    ):
                        malformed = True
                        continue
                    raw_line_no = item.get("line_no") or item.get("line") or item.get("lineno") or 0
                    if isinstance(raw_line_no, bool):
                        malformed = True
                        continue
                    if isinstance(raw_line_no, bool) or not isinstance(raw_line_no, int):
                        malformed = True
                        continue
                    line_no = raw_line_no
                    if not source_file or line_no <= 0:
                        malformed = True
                        continue
                    normalized.add((source_file, line_no, str(item.get("element_type") or "line")))
                else:
                    malformed = True
        if malformed or not normalized:
            return ""
        return hashlib.sha256(repr(sorted(normalized, key=str)).encode("utf-8")).hexdigest()

    @staticmethod
    def _valid_sut_spectrum_path(
        source_file: str, *, allow_test_named_sut: bool = False
    ) -> bool:
        normalized = str(source_file or "").replace("\\", "/")
        path = Path(normalized)
        parts = {part.lower() for part in path.parts}
        name = path.name.lower()
        return bool(
            normalized
            and not path.is_absolute()
            and ".." not in path.parts
            and "tests" not in parts
            and (
                allow_test_named_sut
                or not name.startswith("test_")
                and not name.endswith("_test.py")
            )
        )

    @staticmethod
    def _path_distance(left: Path, right: Path) -> int:
        left_parts, right_parts = left.parts, right.parts
        common = 0
        for a, b in zip(left_parts, right_parts):
            if a != b:
                break
            common += 1
        return (len(left_parts) - common) + (len(right_parts) - common)

    @staticmethod
    def _clue_identifiers(context: Mapping[str, Any]) -> list[str]:
        values: list[str] = []
        for key in ("identifiers", "matched_identifiers"):
            item = context.get(key)
            if isinstance(item, Mapping):
                for value in item.values():
                    values.extend(str(v) for v in value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else values.append(str(value))
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                values.extend(str(value) for value in item)
        return values

    @staticmethod
    def _mapping_identifiers(mapping: Mapping[str, Any]) -> list[str]:
        values: list[str] = []
        for key in ("identifiers", "functions", "classes", "target_identifiers"):
            item = mapping.get(key)
            if isinstance(item, Mapping):
                for value in item.values():
                    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                        values.extend(str(v) for v in value)
                    else:
                        values.append(str(value))
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                values.extend(str(value) for value in item)
            elif item:
                values.append(str(item))
        return values

    @staticmethod
    def _target_location(clue: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
        locations = clue.get("fault_locations") if isinstance(clue, Mapping) else None
        if isinstance(locations, Sequence) and locations and isinstance(locations[0], Mapping):
            first = locations[0]
            return {
                "source_file": first.get("file_path") or first.get("source_file") or "",
                "target_function": first.get("function_name") or first.get("target_function") or "",
                "target_class": first.get("class_name") or first.get("target_class") or "",
            }
        for entry in context.get("candidate_source_files") or []:
            if isinstance(entry, Mapping):
                return {
                    "source_file": entry.get("path") or "",
                    "target_function": (entry.get("top_level_functions") or [""])[0],
                    "target_class": (entry.get("top_level_classes") or [""])[0],
                }
        return {}

    @staticmethod
    def _valid_test_file(test_file: str) -> bool:
        if not test_file or not test_file.endswith(".py"):
            return False
        normalized = test_file.replace("\\", "/")
        path = Path(normalized)
        return bool(
            not path.is_absolute()
            and ".." not in path.parts
            and _TEST_FILE_RE.search(normalized)
        )

    @staticmethod
    def _discover_nodes(path: Path, test_file: str) -> list[str]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError, UnicodeError):
            return []
        nodes: list[str] = []
        for item in ast.walk(tree):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test"):
                parent = SupplementalPassCollector._parent_test_class(tree, item)
                nodes.append(
                    f"{test_file}::{parent}::{item.name}" if parent else f"{test_file}::{item.name}"
                )
        return sorted(set(nodes))

    @staticmethod
    def _discover_v37_nodes(path: Path, test_file: str) -> list[str]:
        """Return only statically collectable top-level/class test nodeids."""
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError, UnicodeError):
            return []
        nodes: list[str] = []
        for item in tree.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name.startswith("test"):
                    nodes.append(f"{test_file}::{item.name}")
                continue
            if not isinstance(item, ast.ClassDef):
                continue
            collectable = item.name.startswith("Test") or any(
                (
                    isinstance(base, ast.Name)
                    and base.id in {"TestCase", "SimpleTestCase"}
                )
                or (
                    isinstance(base, ast.Attribute)
                    and base.attr in {"TestCase", "SimpleTestCase"}
                )
                for base in item.bases
            )
            if not collectable:
                continue
            for child in item.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test"):
                    nodes.append(f"{test_file}::{item.name}::{child.name}")
        return sorted(set(nodes))

    @staticmethod
    def _parent_test_class(tree: ast.AST, target: ast.AST) -> str:
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef) and any(child is target for child in parent.body):
                return parent.name
        return ""

    @staticmethod
    def _rank_file(
        *,
        test_file: str,
        source_text: str,
        entry_index: int,
        target_file: str,
        target_module: str,
        target_function: str,
        target_class: str = "",
        clue: Mapping[str, Any],
    ) -> tuple[float, str]:
        score = max(0.0, 1.0 - entry_index * 0.05)
        reasons: list[str] = ["M2_candidate_test_file"]
        if target_module and test_file.startswith(target_module):
            score += 4.0
            reasons.insert(0, "same_target_test_area")
        target_stem = Path(target_file).stem.lower()
        lower_text = source_text.lower()
        if target_stem and target_stem in lower_text:
            score += 2.0
            reasons.append("references_target_module")
        if target_function and re.search(rf"\b{re.escape(target_function)}\b", source_text):
            score += 3.0
            reasons.append("references_target_function")
        if target_class and re.search(rf"\b{re.escape(target_class)}\b", source_text):
            score += 2.0
            reasons.append("references_target_class")
        identifiers = clue.get("identifiers") if isinstance(clue, Mapping) else {}
        for value in (identifiers or {}).get("functions", []) if isinstance(identifiers, Mapping) else []:
            if str(value) and re.search(rf"\b{re.escape(str(value))}\b", source_text):
                score += 1.0
                reasons.append("references_issue_identifier")
                break
        return score, ";".join(dict.fromkeys(reasons))

    @staticmethod
    def _provenance_record(
        raw: Mapping[str, Any],
        *,
        candidate: SupplementalTestCandidate,
        instance_id: str,
        candidate_id: str,
        candidate_hash: str,
        outer_iteration: int | None,
        timestamp: str | None,
        elapsed_sec: float,
    ) -> dict[str, Any]:
        return {
            "instance_id": instance_id,
            "candidate_id": candidate_id,
            "candidate_hash": candidate_hash,
            "outer_iteration": outer_iteration,
            "source_checkout": str(raw.get("source_checkout") or "pre_patch"),
            "supplemental_test_nodeid": candidate.nodeid,
            "supplemental_test_file": candidate.test_file,
            "discovery_reason": candidate.discovery_reason,
            "discovery_rank": candidate.discovery_rank,
            "execution_status": str(raw.get("execution_status") or raw.get("status") or "NOT_RUN"),
            "line_spectrum": raw.get("covered_sut_lines") or (raw.get("coverage_data") or {}).get("SUT_lines", []),
            "source_mapping_status": raw.get("source_mapping_status") or "UNKNOWN",
            "provenance_timestamp": timestamp,
            "elapsed_sec": elapsed_sec,
        }

    @staticmethod
    def _validate_pass_record(
        raw: Mapping[str, Any],
        *,
        candidate: SupplementalTestCandidate,
        accepted_ids: set[str],
        context: Mapping[str, Any],
        allow_test_named_sut: bool = False,
    ) -> tuple[bool, str | None]:
        if candidate.identity in accepted_ids:
            return False, "duplicate_test_node"
        status = str(raw.get("execution_status") or raw.get("status") or "").upper()
        results = raw.get("test_results") if isinstance(raw.get("test_results"), Mapping) else {}
        if candidate.nodeid not in results:
            return False, "exact_test_node_not_observed"
        observed = str(results.get(candidate.nodeid) or "").upper()
        if status not in {"PASS", "PASSED"}:
            return False, f"execution_status_{status or 'UNKNOWN'}"
        if observed not in {"PASS", "PASSED"}:
            return False, f"observed_status_{observed or 'UNKNOWN'}"
        if any(str(value).upper() in {"FAIL", "FAILED", "ERROR", "NOT_RUN"} for value in results.values()):
            return False, "mixed_or_non_pass_results"
        lines = raw.get("covered_sut_lines")
        if not isinstance(lines, list):
            coverage = raw.get("coverage_data")
            lines = coverage.get("SUT_lines") if isinstance(coverage, Mapping) else None
        if not isinstance(lines, list) or not lines:
            return False, "missing_explicit_line_spectrum"
        repo_path = Path(str(context.get("repo_path") or ""))
        for line in lines:
            if not isinstance(line, Mapping):
                return False, "malformed_line_spectrum"
            source_file = str(line.get("source_file") or line.get("file") or line.get("path") or "")
            if not SupplementalPassCollector._valid_sut_spectrum_path(
                source_file,
                allow_test_named_sut=allow_test_named_sut,
            ):
                return False, "invalid_sut_source_path"
            if repo_path:
                try:
                    resolved_source = (repo_path / source_file).resolve()
                    resolved_source.relative_to(repo_path.resolve())
                except (OSError, RuntimeError, ValueError):
                    return False, "invalid_sut_source_path"
            line_no = line.get("line_no", line.get("line", line.get("lineno")))
            if isinstance(line_no, bool) or not isinstance(line_no, int) or line_no <= 0:
                return False, "malformed_line_spectrum"
        if str(raw.get("source_mapping_status") or "") not in {"VALID_PRE_PATCH_SOURCE", "VALID"}:
            return False, "invalid_source_mapping"
        forbidden = {"golden_patch", "test_patch", "post_patch_outcomes", "m8_results", "phr"}
        if forbidden.intersection(str(key).lower() for key in raw):
            return False, "forbidden_post_patch_provenance"
        return True, None


def candidate_hash_from_identity(identity: Mapping[str, Any]) -> str:
    """Return a stable candidate hash without reading any post-patch field."""
    payload = "|".join(
        str(identity.get(key) or "")
        for key in ("test_id", "canonical_test_nodeid", "generated_patch_sha256", "candidate_id")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

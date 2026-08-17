from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from src.utils.file_io import write_json_atomic


CC_SUPPORTED = "SUPPORTED"
CC_UNSUPPORTED = "UNSUPPORTED"
CC_TIMEOUT = "TIMEOUT"
CC_INSTRUMENTATION_ERROR = "INSTRUMENTATION_ERROR"
CC_COLLECTION_ERROR = "COLLECTION_ERROR"
CC_EXECUTION_ERROR = "EXECUTION_ERROR"


@dataclass(frozen=True, order=True)
class LineKey:
    source_file: str
    line_no: int

    def to_dict(self) -> dict[str, Any]:
        return {"source_file": self.source_file, "line_no": self.line_no}


@dataclass(frozen=True)
class OracleNode:
    source_file: str
    line_no: int
    oracle_type: str
    observed_names: tuple[str, ...] = ()
    expression: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "line_no": self.line_no,
            "oracle_type": self.oracle_type,
            "observed_names": list(self.observed_names),
            "expression": self.expression,
        }


@dataclass
class DynamicSliceCCResult:
    backend: str
    status: str
    oracle_node: Optional[OracleNode] = None
    covered_sut_lines: list[LineKey] = field(default_factory=list)
    checked_lines: list[LineKey] = field(default_factory=list)
    numerator: Optional[int] = None
    denominator: Optional[int] = None
    checked_coverage: Optional[float] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "backend": self.backend,
            "status": self.status,
            "oracle_node": self.oracle_node.to_dict() if self.oracle_node else None,
            "covered_sut_lines": [line.to_dict() for line in self.covered_sut_lines],
            "checked_lines": [line.to_dict() for line in self.checked_lines],
            "numerator": self.numerator,
            "denominator": self.denominator,
            "checked_coverage": self.checked_coverage,
            "diagnostics": dict(self.diagnostics),
        }
        for key in (
            "reason",
            "instance_id",
            "container_image",
            "execution_environment",
            "testbed_path",
            "test_nodeid",
            "generated_test_patch_provenance",
            "generated_patch_sha256",
            "patch_applied",
            "patch_application_status",
            "container_python_version",
            "worker_compatibility_mode",
            "pytest_command",
            "pytest_exit_code",
            "collection_status",
            "execution_status",
            "captured_stdout",
            "captured_stderr",
        ):
            if key in self.diagnostics:
                payload[key] = self.diagnostics[key]
        return payload


BACKEND_NAME = "python_trace_ast_def_use_v1"
NUMPY_ASSERTION_HELPERS = {
    "assert_array_equal",
    "assert_allclose",
    "assert_equal",
}
UNITTEST_ASSERTION_PREFIX = "assert"


def run_before_patch_dynamic_slice_cc(
    *,
    before_patch_repo_path: str | Path,
    test_nodeid: str,
    output_path: str | Path | None = None,
    timeout_seconds: int = 30,
    python_executable: str = sys.executable,
) -> DynamicSliceCCResult:
    """Run a deterministic before-patch pytest trace for M8 checked coverage.

    This backend is intentionally conservative. It supports same-process pytest
    tests where an executed assertion observes a value that can be tied to a
    SUT call window in the test frame. It does not claim full dynamic slicing:
    within that dynamic call window it uses AST def-use evidence over executed
    Python lines to identify the covered SUT lines that affected the observed
    assertion value. Unsupported shapes return ``UNSUPPORTED``.
    """
    repo_path = Path(before_patch_repo_path).resolve()
    result_path = Path(output_path).resolve() if output_path else None
    if not repo_path.exists():
        result = _unsupported("before_patch_repo_path_missing", before_patch_repo_path=str(repo_path))
        _write_dynamic_slice_artifact_if_requested(result, result_path)
        return result
    if not test_nodeid or "::" not in test_nodeid:
        result = _unsupported("missing_pytest_nodeid")
        _write_dynamic_slice_artifact_if_requested(result, result_path)
        return result

    if result_path is None:
        result_path = repo_path / ".m8_dynamic_slice_cc.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        "-m",
        "src.evaluator.dynamic_slice_cc",
        "--worker",
        "--repo",
        str(repo_path),
        "--nodeid",
        test_nodeid,
        "--output",
        str(result_path),
    ]
    workspace_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(workspace_root)
        if not existing_pythonpath
        else f"{workspace_root}{os.pathsep}{existing_pythonpath}"
    )
    try:
        proc = subprocess.run(
            command,
            cwd=str(repo_path),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return DynamicSliceCCResult(
            backend=BACKEND_NAME,
            status=CC_TIMEOUT,
            diagnostics={"reason": "dynamic_slice_timeout", "timeout_seconds": timeout_seconds},
        )
    if result_path.exists():
        try:
            return dynamic_slice_result_from_dict(json.loads(result_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return DynamicSliceCCResult(
                backend=BACKEND_NAME,
                status=CC_INSTRUMENTATION_ERROR,
                diagnostics={"reason": "malformed_dynamic_slice_artifact", "error": str(exc)},
            )
    return DynamicSliceCCResult(
        backend=BACKEND_NAME,
        status=CC_INSTRUMENTATION_ERROR if proc.returncode else CC_UNSUPPORTED,
        diagnostics={
            "reason": "missing_dynamic_slice_artifact",
            "returncode": proc.returncode,
            "stderr": proc.stderr[-1000:],
        },
    )


def _write_dynamic_slice_artifact_if_requested(
    result: DynamicSliceCCResult,
    result_path: Path | None,
) -> None:
    if result_path is None:
        return
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(result.to_dict(), result_path)


def dynamic_slice_result_from_dict(payload: Mapping[str, Any]) -> DynamicSliceCCResult:
    oracle_payload = payload.get("oracle_node")
    oracle = None
    if isinstance(oracle_payload, Mapping):
        observed = oracle_payload.get("observed_names", [])
        oracle = OracleNode(
            source_file=str(oracle_payload.get("source_file") or ""),
            line_no=int(oracle_payload.get("line_no") or 0),
            oracle_type=str(oracle_payload.get("oracle_type") or "assert"),
            observed_names=tuple(str(name) for name in observed if isinstance(name, str)),
            expression=(
                str(oracle_payload.get("expression"))
                if oracle_payload.get("expression") is not None
                else None
            ),
        )
    return DynamicSliceCCResult(
        backend=str(payload.get("backend") or BACKEND_NAME),
        status=str(payload.get("status") or CC_UNSUPPORTED),
        oracle_node=oracle,
        covered_sut_lines=_line_keys_from_payload(payload.get("covered_sut_lines")),
        checked_lines=_line_keys_from_payload(payload.get("checked_lines")),
        numerator=_optional_int(payload.get("numerator")),
        denominator=_optional_int(payload.get("denominator")),
        checked_coverage=_optional_float(payload.get("checked_coverage")),
        diagnostics=dict(payload.get("diagnostics") or {}),
    )


def compute_dynamic_slice_from_trace(
    *,
    repo_path: str | Path,
    test_nodeid: str,
    events: Iterable[Mapping[str, Any]],
    oracle_events: Iterable[Mapping[str, Any]],
) -> DynamicSliceCCResult:
    """Compute checked coverage from one isolated before-patch trace."""
    repo = Path(repo_path).resolve()
    normalized_events = [_normalize_event(repo, event) for event in events]
    covered = sorted(
        {
            LineKey(event["rel_file"], int(event["line_no"]))
            for event in normalized_events
            if event["kind"] == "line" and _is_sut_file(event["rel_file"], _nodeid_file(test_nodeid))
        }
    )
    if not covered:
        return _unsupported("empty_covered_sut_lines")

    oracle = _first_oracle_node(repo, oracle_events)
    if oracle is None:
        return DynamicSliceCCResult(
            backend=BACKEND_NAME,
            status=CC_UNSUPPORTED,
            covered_sut_lines=covered,
            diagnostics={"reason": "missing_executed_oracle_observation"},
        )
    if not oracle.observed_names:
        return DynamicSliceCCResult(
            backend=BACKEND_NAME,
            status=CC_UNSUPPORTED,
            oracle_node=oracle,
            covered_sut_lines=covered,
            diagnostics={"reason": "oracle_observation_has_no_supported_value_names"},
        )

    call_windows = _sut_call_windows(normalized_events, _nodeid_file(test_nodeid))
    relevant_windows = [
        window for window in call_windows if window["test_line_no"] <= oracle.line_no
    ]
    if not relevant_windows:
        return DynamicSliceCCResult(
            backend=BACKEND_NAME,
            status=CC_UNSUPPORTED,
            oracle_node=oracle,
            covered_sut_lines=covered,
            diagnostics={"reason": "oracle_value_has_no_sut_call_window"},
        )

    sut_event_indexes = {index for window in relevant_windows for index in window["sut_indexes"]}
    checked = _backward_ast_def_use_slice(
        repo=repo,
        events=[
            event
            for index, event in enumerate(normalized_events)
            if index in sut_event_indexes and event["kind"] == "line"
        ],
        initial_names=set(oracle.observed_names),
    )
    if not checked:
        return DynamicSliceCCResult(
            backend=BACKEND_NAME,
            status=CC_UNSUPPORTED,
            oracle_node=oracle,
            covered_sut_lines=covered,
            diagnostics={"reason": "no_dynamic_def_use_path_from_oracle"},
        )
    covered_set = set(covered)
    checked_lines = sorted(covered_set & checked)
    numerator = len(checked_lines)
    denominator = len(covered)
    return DynamicSliceCCResult(
        backend=BACKEND_NAME,
        status=CC_SUPPORTED,
        oracle_node=oracle,
        covered_sut_lines=covered,
        checked_lines=checked_lines,
        numerator=numerator,
        denominator=denominator,
        checked_coverage=numerator / denominator,
        diagnostics={
            "algorithm": (
                "single-test sys.settrace line trace; executed pytest assertion "
                "oracle; dynamic SUT call windows before the oracle; AST def-use "
                "backward slice within those executed SUT lines"
            ),
            "limitations": [
                "Python pytest only",
                "same-process execution only",
                "dynamic control dependence is not modeled",
                "fixtures, subprocesses, native extensions, and opaque framework oracles may be unsupported",
            ],
        },
    )


def _unsupported(reason: str, **diagnostics: Any) -> DynamicSliceCCResult:
    payload = {"reason": reason}
    payload.update(diagnostics)
    return DynamicSliceCCResult(backend=BACKEND_NAME, status=CC_UNSUPPORTED, diagnostics=payload)


def _normalize_event(repo: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    filename = Path(str(event.get("file") or "")).resolve()
    try:
        rel_file = filename.relative_to(repo).as_posix()
    except ValueError:
        rel_file = filename.as_posix()
    return {
        "kind": str(event.get("kind") or "line"),
        "rel_file": rel_file,
        "line_no": int(event.get("line_no") or 0),
        "function": str(event.get("function") or ""),
        "depth": int(event.get("depth") or 0),
    }


def _line_keys_from_payload(value: Any) -> list[LineKey]:
    if not isinstance(value, list):
        return []
    lines: list[LineKey] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        try:
            lines.append(LineKey(str(item.get("source_file") or ""), int(item.get("line_no") or 0)))
        except (TypeError, ValueError):
            continue
    return sorted(line for line in lines if line.source_file and line.line_no > 0)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _nodeid_file(test_nodeid: str) -> str:
    return test_nodeid.split("::", 1)[0].replace("\\", "/")


def _is_sut_file(rel_file: str, test_file: str) -> bool:
    if not rel_file.endswith(".py"):
        return False
    normalized = rel_file.replace("\\", "/")
    if normalized == test_file:
        return False
    parts = normalized.split("/")
    return not (Path(normalized).name.startswith("test_") or "tests" in parts or "test" in parts)


def _first_oracle_node(repo: Path, oracle_events: Iterable[Mapping[str, Any]]) -> Optional[OracleNode]:
    for event in oracle_events:
        filename = Path(str(event.get("file") or "")).resolve()
        try:
            rel_file = filename.relative_to(repo).as_posix()
        except ValueError:
            rel_file = filename.as_posix()
        line_no = int(event.get("line_no") or 0)
        expression = _assert_expression(filename, line_no)
        observed = tuple(sorted(_supported_observed_names(filename, line_no, event.get("locals", {}))))
        return OracleNode(
            source_file=rel_file,
            line_no=line_no,
            oracle_type="assert",
            observed_names=observed,
            expression=expression,
        )
    return None


def _assert_expression(path: Path, line_no: int) -> Optional[str]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and node.lineno == line_no:
            return ast.unparse(node.test)
        if isinstance(node, ast.Expr) and node.lineno == line_no:
            if (
                _is_allowlisted_numpy_assertion_call(node.value)
                or _is_allowlisted_unittest_assertion_call(node.value)
            ):
                return ast.unparse(node.value)
    return None


def _supported_observed_names(path: Path, line_no: int, locals_payload: Any) -> set[str]:
    if not isinstance(locals_payload, Mapping):
        locals_payload = {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and node.lineno == line_no:
            return {
                item.id
                for item in ast.walk(node.test)
                if isinstance(item, ast.Name) and item.id in locals_payload
            }
        if isinstance(node, ast.Expr) and node.lineno == line_no:
            call = node.value
            if not (
                _is_allowlisted_numpy_assertion_call(call)
                or _is_allowlisted_unittest_assertion_call(call)
            ):
                continue
            return {
                item.id
                for argument in call.args
                for item in ast.walk(argument)
                if isinstance(item, ast.Name) and item.id in locals_payload
            }
    return set()


def _is_allowlisted_numpy_assertion_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    parts = _call_parts(node.func)
    if not parts or parts[-1] not in NUMPY_ASSERTION_HELPERS:
        return False
    return len(parts) == 1 or "testing" in parts[:-1]


def _is_allowlisted_unittest_assertion_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    parts = _call_parts(node.func)
    return len(parts) == 2 and parts[0] == "self" and parts[1].startswith(UNITTEST_ASSERTION_PREFIX)


def _call_parts(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [*_call_parts(node.value), node.attr]
    return []


def _sut_call_windows(events: list[Mapping[str, Any]], test_file: str) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    active: Optional[dict[str, Any]] = None
    for index, event in enumerate(events):
        if event["kind"] != "line":
            continue
        is_test_line = event["rel_file"] == test_file
        if is_test_line:
            if active and active["sut_indexes"]:
                windows.append(active)
            active = {"test_line_no": int(event["line_no"]), "sut_indexes": set()}
            continue
        if active is not None and _is_sut_file(event["rel_file"], test_file):
            active["sut_indexes"].add(index)
    if active and active["sut_indexes"]:
        windows.append(active)
    return windows


def _backward_ast_def_use_slice(
    *,
    repo: Path,
    events: list[Mapping[str, Any]],
    initial_names: set[str],
) -> set[LineKey]:
    tainted = set(initial_names)
    checked: set[LineKey] = set()
    line_facts: dict[tuple[str, int], tuple[set[str], set[str]]] = {}
    for event in events:
        key = (str(event["rel_file"]), int(event["line_no"]))
        if key not in line_facts:
            line_facts[key] = _line_def_use(repo / key[0], key[1])
    for event in reversed(events):
        rel_file = str(event["rel_file"])
        line_no = int(event["line_no"])
        defs, uses = line_facts[(rel_file, line_no)]
        is_return = _line_has_return(repo / rel_file, line_no)
        if defs & tainted or (is_return and uses):
            checked.add(LineKey(rel_file, line_no))
            tainted.update(uses)
    return checked


def _line_def_use(path: Path, line_no: int) -> tuple[set[str], set[str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set(), set()
    defs: set[str] = set()
    uses: set[str] = set()
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) != line_no:
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            defs.update(_target_names(targets))
            value = node.value if not isinstance(node, ast.AugAssign) else node
            uses.update(_load_names(value))
        elif isinstance(node, ast.Return):
            uses.update(_load_names(node.value))
        elif isinstance(node, ast.Expr):
            uses.update(_load_names(node))
    return defs, uses - defs


def _line_has_return(path: Path, line_no: int) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    return any(isinstance(node, ast.Return) and node.lineno == line_no for node in ast.walk(tree))


def _target_names(targets: Iterable[ast.AST]) -> set[str]:
    names: set[str] = set()
    for target in targets:
        for node in ast.walk(target):
            if isinstance(node, ast.Name):
                names.add(node.id)
    return names


def _load_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)}


class _TracePlugin:
    def __init__(self, repo: Path) -> None:
        self.repo = repo.resolve()
        self.events: list[dict[str, Any]] = []
        self.oracle_events: list[dict[str, Any]] = []
        self.depth = 0
        self.collected_count = 0
        self.call_reports = 0
        self.setup_failures = 0
        self.collection_failures = 0

    def pytest_collection_modifyitems(self, session: Any, config: Any, items: list[Any]) -> None:
        self.collected_count = len(items)

    def pytest_collectreport(self, report: Any) -> None:
        if getattr(report, "failed", False):
            self.collection_failures += 1

    def pytest_runtest_logreport(self, report: Any) -> None:
        if report.when == "setup" and report.failed:
            self.setup_failures += 1
        if report.when == "call":
            self.call_reports += 1

    def pytest_runtest_call(self, item: Any) -> None:
        sys.settrace(self._trace)

    def pytest_runtest_makereport(self, item: Any, call: Any) -> None:
        sys.settrace(None)
        if call.when != "call" or call.excinfo is None:
            return
        if not call.excinfo.errisinstance(AssertionError):
            return
        traceback = call.excinfo.traceback
        for entry in reversed(traceback):
            path = Path(str(entry.path)).resolve()
            if _is_under(path, self.repo):
                self.oracle_events.append(
                    {
                        "file": str(path),
                        "line_no": int(entry.lineno) + 1,
                        "locals": _jsonable_locals(entry.frame.f_locals),
                    }
                )
                return

    def _trace(self, frame: Any, event: str, arg: Any) -> Any:
        if event == "call":
            self.depth += 1
            return self._trace
        if event == "return":
            self.depth = max(0, self.depth - 1)
            return self._trace
        if event != "line":
            return self._trace
        path = Path(frame.f_code.co_filename).resolve()
        if _is_under(path, self.repo):
            self.events.append(
                {
                    "kind": "line",
                    "file": str(path),
                    "line_no": int(frame.f_lineno),
                    "function": frame.f_code.co_name,
                    "depth": self.depth,
                    "locals": _jsonable_locals(frame.f_locals),
                }
            )
        return self._trace


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _jsonable_locals(values: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): type(value).__name__ for key, value in values.items()}


def _run_worker(repo: Path, nodeid: str, output: Path) -> int:
    try:
        import pytest
    except ImportError:
        write_json_atomic(_unsupported("pytest_unavailable").to_dict(), output)
        return 0
    plugin = _TracePlugin(repo)
    code = pytest.main(
        [
            "-q",
            nodeid,
        ],
        plugins=[plugin],
    )
    failure_result = _pytest_failure_result(
        repo=repo,
        nodeid=nodeid,
        code=int(code),
        plugin=plugin,
    )
    if failure_result is not None:
        write_json_atomic(failure_result.to_dict(), output)
        return 0
    result = compute_dynamic_slice_from_trace(
        repo_path=repo,
        test_nodeid=nodeid,
        events=plugin.events,
        oracle_events=plugin.oracle_events,
    )
    result.diagnostics.setdefault("pytest_exit_code", int(code))
    result.diagnostics.setdefault("collection_status", "collected")
    result.diagnostics.setdefault("execution_status", "executed")
    result.diagnostics.setdefault("pytest_collected_count", plugin.collected_count)
    result.diagnostics.setdefault("pytest_call_reports", plugin.call_reports)
    write_json_atomic(result.to_dict(), output)
    return 0


def _pytest_failure_result(
    *,
    repo: Path,
    nodeid: str,
    code: int,
    plugin: _TracePlugin,
) -> Optional[DynamicSliceCCResult]:
    if code == 4 or plugin.collection_failures or plugin.collected_count == 0:
        return DynamicSliceCCResult(
            backend=BACKEND_NAME,
            status=CC_COLLECTION_ERROR,
            diagnostics={
                "reason": "pytest_collection_failed",
                "pytest_exit_code": code,
                "collection_status": "failed",
                "execution_status": "not_executed",
                "pytest_collected_count": plugin.collected_count,
                "pytest_collection_failures": plugin.collection_failures,
            },
        )
    if plugin.call_reports == 0:
        return DynamicSliceCCResult(
            backend=BACKEND_NAME,
            status=CC_EXECUTION_ERROR,
            diagnostics={
                "reason": (
                    "benchmark_environment_unavailable"
                    if plugin.setup_failures
                    else "pytest_execution_failed"
                ),
                "pytest_exit_code": code,
                "collection_status": "collected",
                "execution_status": "not_executed",
                "pytest_collected_count": plugin.collected_count,
                "pytest_setup_failures": plugin.setup_failures,
            },
        )
    return None


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--nodeid", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if not args.worker:
        parser.error("--worker is required")
    return _run_worker(Path(args.repo).resolve(), args.nodeid, Path(args.output).resolve())


if __name__ == "__main__":
    raise SystemExit(_main())

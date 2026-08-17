import argparse
import ast
import json
import os
import sys
import tempfile


BACKEND_NAME = "python_trace_ast_def_use_v1"
CC_SUPPORTED = "SUPPORTED"
CC_UNSUPPORTED = "UNSUPPORTED"
CC_INSTRUMENTATION_ERROR = "INSTRUMENTATION_ERROR"
CC_COLLECTION_ERROR = "COLLECTION_ERROR"
CC_EXECUTION_ERROR = "EXECUTION_ERROR"
NUMPY_ASSERTION_HELPERS = set(["assert_array_equal", "assert_allclose", "assert_equal"])
UNITTEST_ASSERTION_PREFIX = "assert"


def _line(source_file, line_no):
    return {"source_file": source_file, "line_no": int(line_no)}


def _result(status, oracle_node=None, covered_sut_lines=None, checked_lines=None,
            numerator=None, denominator=None, checked_coverage=None, diagnostics=None):
    return {
        "backend": BACKEND_NAME,
        "status": status,
        "oracle_node": oracle_node,
        "covered_sut_lines": covered_sut_lines or [],
        "checked_lines": checked_lines or [],
        "numerator": numerator,
        "denominator": denominator,
        "checked_coverage": checked_coverage,
        "diagnostics": diagnostics or {},
    }


def _unsupported(reason, diagnostics=None):
    payload = dict(diagnostics or {})
    payload["reason"] = reason
    return _result(CC_UNSUPPORTED, diagnostics=payload)


def _write_json_atomic(payload, output_path):
    parent = os.path.dirname(output_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    fd, tmp_path = tempfile.mkstemp(prefix=".m8cc-", suffix=".tmp", dir=parent or None)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _relpath(path, repo):
    abs_path = os.path.abspath(path)
    repo = os.path.abspath(repo)
    try:
        return os.path.relpath(abs_path, repo).replace(os.sep, "/")
    except ValueError:
        return abs_path.replace(os.sep, "/")


def _nodeid_file(test_nodeid):
    return test_nodeid.split("::", 1)[0].replace("\\", "/")


def _is_sut_file(rel_file, test_file):
    if not rel_file.endswith(".py"):
        return False
    normalized = rel_file.replace("\\", "/")
    if normalized == test_file:
        return False
    parts = normalized.split("/")
    name = os.path.basename(normalized)
    return not (name.startswith("test_") or "tests" in parts or "test" in parts)


def _jsonable_locals(values):
    return dict((str(key), type(value).__name__) for key, value in values.items())


def _call_parts(node):
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return _call_parts(node.value) + [node.attr]
    return []


def _is_allowlisted_numpy_assertion_call(node):
    if not isinstance(node, ast.Call):
        return False
    parts = _call_parts(node.func)
    if not parts or parts[-1] not in NUMPY_ASSERTION_HELPERS:
        return False
    return len(parts) == 1 or "testing" in parts[:-1]


def _is_allowlisted_unittest_assertion_call(node):
    if not isinstance(node, ast.Call):
        return False
    parts = _call_parts(node.func)
    return len(parts) == 2 and parts[0] == "self" and parts[1].startswith(UNITTEST_ASSERTION_PREFIX)


def _read_tree(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return ast.parse(handle.read())
    except (IOError, OSError, SyntaxError, UnicodeDecodeError):
        return None


def _supported_observed_names(path, line_no, locals_payload):
    if not isinstance(locals_payload, dict):
        locals_payload = {}
    tree = _read_tree(path)
    if tree is None:
        return set()
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) != line_no:
            continue
        if isinstance(node, ast.Assert):
            return set(
                item.id for item in ast.walk(node.test)
                if isinstance(item, ast.Name) and item.id in locals_payload
            )
        if isinstance(node, ast.Expr):
            call = node.value
            if not (
                _is_allowlisted_numpy_assertion_call(call)
                or _is_allowlisted_unittest_assertion_call(call)
            ):
                continue
            names = set()
            for argument in call.args:
                for item in ast.walk(argument):
                    if isinstance(item, ast.Name) and item.id in locals_payload:
                        names.add(item.id)
            return names
    return set()


def _first_oracle_node(repo, oracle_events):
    for event in oracle_events:
        path = os.path.abspath(str(event.get("file") or ""))
        line_no = int(event.get("line_no") or 0)
        observed = sorted(_supported_observed_names(path, line_no, event.get("locals", {})))
        return {
            "source_file": _relpath(path, repo),
            "line_no": line_no,
            "oracle_type": "assert",
            "observed_names": observed,
            "expression": None,
        }
    return None


def _normalize_events(repo, events):
    normalized = []
    for event in events:
        normalized.append({
            "kind": str(event.get("kind") or "line"),
            "rel_file": _relpath(str(event.get("file") or ""), repo),
            "line_no": int(event.get("line_no") or 0),
            "function": str(event.get("function") or ""),
            "depth": int(event.get("depth") or 0),
        })
    return normalized


def _sut_call_windows(events, test_file):
    windows = []
    active = None
    for index, event in enumerate(events):
        if event["kind"] != "line":
            continue
        if event["rel_file"] == test_file:
            if active and active["sut_indexes"]:
                windows.append(active)
            active = {"test_line_no": int(event["line_no"]), "sut_indexes": set()}
            continue
        if active is not None and _is_sut_file(event["rel_file"], test_file):
            active["sut_indexes"].add(index)
    if active and active["sut_indexes"]:
        windows.append(active)
    return windows


def _target_names(targets):
    names = set()
    for target in targets:
        for node in ast.walk(target):
            if isinstance(node, ast.Name):
                names.add(node.id)
    return names


def _load_names(node):
    if node is None:
        return set()
    return set(
        item.id for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    )


def _line_def_use(path, line_no):
    tree = _read_tree(path)
    if tree is None:
        return set(), set()
    defs = set()
    uses = set()
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


def _line_has_return(path, line_no):
    tree = _read_tree(path)
    if tree is None:
        return False
    return any(isinstance(node, ast.Return) and node.lineno == line_no for node in ast.walk(tree))


def _backward_ast_def_use_slice(repo, events, initial_names):
    tainted = set(initial_names)
    checked = set()
    line_facts = {}
    for event in events:
        key = (str(event["rel_file"]), int(event["line_no"]))
        if key not in line_facts:
            line_facts[key] = _line_def_use(os.path.join(repo, key[0]), key[1])
    for event in reversed(events):
        rel_file = str(event["rel_file"])
        line_no = int(event["line_no"])
        defs, uses = line_facts[(rel_file, line_no)]
        if defs & tainted or (_line_has_return(os.path.join(repo, rel_file), line_no) and uses):
            checked.add((rel_file, line_no))
            tainted.update(uses)
    return checked


def compute_dynamic_slice_from_trace(repo, test_nodeid, events, oracle_events):
    normalized_events = _normalize_events(repo, events)
    test_file = _nodeid_file(test_nodeid)
    covered_pairs = sorted(set(
        (event["rel_file"], int(event["line_no"]))
        for event in normalized_events
        if event["kind"] == "line" and _is_sut_file(event["rel_file"], test_file)
    ))
    covered = [_line(path, line_no) for path, line_no in covered_pairs]
    if not covered:
        return _unsupported("empty_covered_sut_lines")
    oracle = _first_oracle_node(repo, oracle_events)
    if oracle is None:
        return _result(
            CC_UNSUPPORTED,
            covered_sut_lines=covered,
            diagnostics={"reason": "missing_executed_oracle_observation"},
        )
    if not oracle["observed_names"]:
        return _result(
            CC_UNSUPPORTED,
            oracle_node=oracle,
            covered_sut_lines=covered,
            diagnostics={"reason": "oracle_observation_has_no_supported_value_names"},
        )
    windows = [
        window for window in _sut_call_windows(normalized_events, test_file)
        if window["test_line_no"] <= oracle["line_no"]
    ]
    if not windows:
        return _result(
            CC_UNSUPPORTED,
            oracle_node=oracle,
            covered_sut_lines=covered,
            diagnostics={"reason": "oracle_value_has_no_sut_call_window"},
        )
    sut_indexes = set()
    for window in windows:
        sut_indexes.update(window["sut_indexes"])
    checked_pairs = _backward_ast_def_use_slice(
        repo,
        [
            event for index, event in enumerate(normalized_events)
            if index in sut_indexes and event["kind"] == "line"
        ],
        set(oracle["observed_names"]),
    )
    if not checked_pairs:
        return _result(
            CC_UNSUPPORTED,
            oracle_node=oracle,
            covered_sut_lines=covered,
            diagnostics={"reason": "no_dynamic_def_use_path_from_oracle"},
        )
    covered_set = set(covered_pairs)
    checked = [_line(path, line_no) for path, line_no in sorted(covered_set & checked_pairs)]
    numerator = len(checked)
    denominator = len(covered)
    return _result(
        CC_SUPPORTED,
        oracle_node=oracle,
        covered_sut_lines=covered,
        checked_lines=checked,
        numerator=numerator,
        denominator=denominator,
        checked_coverage=float(numerator) / float(denominator),
        diagnostics={
            "algorithm": (
                "single-test sys.settrace line trace; executed pytest assertion oracle; "
                "dynamic SUT call windows before the oracle; AST def-use backward slice "
                "within those executed SUT lines"
            ),
            "limitations": [
                "Python pytest only",
                "same-process execution only",
                "dynamic control dependence is not modeled",
                "fixtures, subprocesses, native extensions, and opaque framework oracles may be unsupported",
            ],
        },
    )


def _is_under(path, root):
    path = os.path.abspath(path)
    root = os.path.abspath(root)
    return path == root or path.startswith(root + os.sep)


class TracePlugin(object):
    def __init__(self, repo):
        self.repo = os.path.abspath(repo)
        self.events = []
        self.oracle_events = []
        self.depth = 0
        self.collected_count = 0
        self.call_reports = 0
        self.setup_failures = 0
        self.collection_failures = 0

    def pytest_collection_modifyitems(self, session, config, items):
        self.collected_count = len(items)

    def pytest_collectreport(self, report):
        if getattr(report, "failed", False):
            self.collection_failures += 1

    def pytest_runtest_logreport(self, report):
        if report.when == "setup" and report.failed:
            self.setup_failures += 1
        if report.when == "call":
            self.call_reports += 1

    def pytest_runtest_call(self, item):
        sys.settrace(self._trace)

    def pytest_runtest_makereport(self, item, call):
        sys.settrace(None)
        if call.when != "call" or call.excinfo is None:
            return
        if not call.excinfo.errisinstance(AssertionError):
            return
        traceback = call.excinfo.traceback
        for entry in reversed(traceback):
            path = os.path.abspath(str(entry.path))
            if _is_under(path, self.repo):
                self.oracle_events.append({
                    "file": path,
                    "line_no": int(entry.lineno) + 1,
                    "locals": _jsonable_locals(entry.frame.f_locals),
                })
                return

    def _trace(self, frame, event, arg):
        if event == "call":
            self.depth += 1
            return self._trace
        if event == "return":
            self.depth = max(0, self.depth - 1)
            return self._trace
        if event != "line":
            return self._trace
        path = os.path.abspath(frame.f_code.co_filename)
        if _is_under(path, self.repo):
            self.events.append({
                "kind": "line",
                "file": path,
                "line_no": int(frame.f_lineno),
                "function": frame.f_code.co_name,
                "depth": self.depth,
                "locals": _jsonable_locals(frame.f_locals),
            })
        return self._trace


def _pytest_failure_result(repo, nodeid, code, plugin, base_diagnostics):
    if code == 4 or plugin.collection_failures or plugin.collected_count == 0:
        diagnostics = dict(base_diagnostics)
        diagnostics.update({
            "reason": "pytest_collection_failed",
            "pytest_exit_code": code,
            "collection_status": "failed",
            "execution_status": "not_executed",
            "pytest_collected_count": plugin.collected_count,
            "pytest_collection_failures": plugin.collection_failures,
        })
        return _result(CC_COLLECTION_ERROR, diagnostics=diagnostics)
    if plugin.call_reports == 0:
        diagnostics = dict(base_diagnostics)
        diagnostics.update({
            "reason": "benchmark_environment_unavailable" if plugin.setup_failures else "pytest_execution_failed",
            "pytest_exit_code": code,
            "collection_status": "collected",
            "execution_status": "not_executed",
            "pytest_collected_count": plugin.collected_count,
            "pytest_setup_failures": plugin.setup_failures,
        })
        return _result(CC_EXECUTION_ERROR, diagnostics=diagnostics)
    return None


def run_worker(repo, nodeid, output):
    if repo not in sys.path:
        sys.path.insert(0, repo)
    base_diagnostics = {
        "container_python_version": sys.version.split()[0],
        "worker_compatibility_mode": "standalone_py36",
        "test_nodeid": nodeid,
        "collection_status": None,
        "execution_status": None,
        "patch_application_status": "applied_before_worker",
    }
    try:
        import pytest
    except ImportError:
        _write_json_atomic(_unsupported("pytest_unavailable", base_diagnostics), output)
        return 0
    plugin = TracePlugin(repo)
    code = pytest.main([
        "-q",
        nodeid,
    ], plugins=[plugin])
    failure = _pytest_failure_result(repo, nodeid, int(code), plugin, base_diagnostics)
    if failure is not None:
        _write_json_atomic(failure, output)
        return 0
    result = compute_dynamic_slice_from_trace(repo, nodeid, plugin.events, plugin.oracle_events)
    diagnostics = dict(base_diagnostics)
    diagnostics.update(result.get("diagnostics") or {})
    diagnostics.setdefault("pytest_exit_code", int(code))
    diagnostics.setdefault("collection_status", "collected")
    diagnostics.setdefault("execution_status", "executed")
    diagnostics.setdefault("pytest_collected_count", plugin.collected_count)
    diagnostics.setdefault("pytest_call_reports", plugin.call_reports)
    result["diagnostics"] = diagnostics
    _write_json_atomic(result, output)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--nodeid", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    return run_worker(os.path.abspath(args.repo), args.nodeid, os.path.abspath(args.output))


if __name__ == "__main__":
    raise SystemExit(main())

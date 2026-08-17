from __future__ import annotations

import ast
import contextlib
import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import threading
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from src.contracts.envelope import make_envelope
from src.contracts.feature_flags import V22FeatureFlags, core_only_feature_flags, resolve_feature_flags
from src.contracts.instance_views import (
    PrePatchInstanceView,
    assert_no_patch_access,
    make_pre_patch_view,
)
from src.contracts.models import FaultCandidate
from src.contracts.v30 import normalize_localization_hypotheses
from src.contracts.v31 import normalize_target_hypotheses
from src.contracts.validation import validate_score_range
from src.utils.file_io import read_text
from src.utils.file_io import write_json_atomic

# 같은 repository에 대한 git 작업을 직렬화하기 위한 전역 lock
_REPO_LOCKS: dict[str, threading.Lock] = defaultdict(threading.Lock)

_INSTANCE_VIEW_LIFECYCLE_DIR = ".lifecycle"
_INSTANCE_VIEW_MARKER_SCHEMA = "kcc-instance-view-v29-v1"

V37_CODEBERT_CHECKPOINT = "microsoft/codebert-base"
V37_CODEBERT_MAX_TOKENS = 512
V37_CODEBERT_EMBEDDING_DIM = 768
_V37_CODEBERT_LOAD_LOCK = threading.RLock()
_V37_CODEBERT_ENCODER_SINGLETON: Any = None


def clamp_cosine_similarity(value: float) -> float:
    """Clamp cosine similarity to the v37 interval, including negatives."""
    if not math.isfinite(float(value)):
        raise ValueError("cosine similarity must be finite")
    return max(0.0, min(float(value), 1.0))


def mean_pool_last_hidden_state(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Return attention-mask-aware mean pooling over a final hidden state."""
    expanded_mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    denominator = expanded_mask.sum(dim=1).clamp(min=1.0)
    return (last_hidden_state * expanded_mask).sum(dim=1) / denominator


class V37CodeBertMethodEncoder:
    """Exact v37 method-source encoder with lazy optional dependencies."""

    def __init__(self, tokenizer: Any = None, model: Any = None) -> None:
        if tokenizer is None or model is None:
            # Transformers exposes Auto* through a lazy module namespace. The
            # full449 runner uses threads, so import and first construction must
            # be serialized inside this process.
            with _V37_CODEBERT_LOAD_LOCK:
                try:
                    from transformers import AutoModel, AutoTokenizer
                except ImportError as exc:
                    raise RuntimeError(
                        "V37 CodeBERT requires the approved transformers runtime dependency"
                    ) from exc
                tokenizer = tokenizer or AutoTokenizer.from_pretrained(
                    V37_CODEBERT_CHECKPOINT
                )
                model = model or AutoModel.from_pretrained(V37_CODEBERT_CHECKPOINT)
        self.tokenizer = tokenizer
        self.model = model
        eval_model = getattr(self.model, "eval", None)
        if callable(eval_model):
            eval_model()

    def encode(self, source_text: str) -> Any:
        """Encode source using tail truncation, retaining tokens 0 through 511."""
        encoded = self.tokenizer(
            str(source_text),
            truncation=True,
            max_length=V37_CODEBERT_MAX_TOKENS,
            return_tensors="pt",
        )
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("V37 CodeBERT requires PyTorch") from exc
        with torch.inference_mode():
            output = self.model(**encoded)
        vector = mean_pool_last_hidden_state(
            output.last_hidden_state,
            encoded["attention_mask"],
        )[0]
        if int(vector.shape[-1]) != V37_CODEBERT_EMBEDDING_DIM:
            raise ValueError(
                "V37 CodeBERT embedding must be 768-dimensional; "
                f"got {int(vector.shape[-1])}"
            )
        return vector

    @staticmethod
    def cosine(left: Any, right: Any) -> float:
        """Return cosine similarity with the v37 negative clamp."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("V37 CodeBERT requires PyTorch") from exc
        raw = torch.nn.functional.cosine_similarity(
            left.unsqueeze(0), right.unsqueeze(0), dim=1
        ).item()
        return clamp_cosine_similarity(float(raw))


def get_v37_codebert_method_encoder() -> V37CodeBertMethodEncoder:
    """Return one lazily initialized CodeBERT encoder per Python process."""
    global _V37_CODEBERT_ENCODER_SINGLETON
    if _V37_CODEBERT_ENCODER_SINGLETON is None:
        with _V37_CODEBERT_LOAD_LOCK:
            if _V37_CODEBERT_ENCODER_SINGLETON is None:
                _V37_CODEBERT_ENCODER_SINGLETON = V37CodeBertMethodEncoder()
    return _V37_CODEBERT_ENCODER_SINGLETON


def extract_python_method_sources(source: str) -> list[dict[str, Any]]:
    """Extract complete function/method source spans in deterministic AST order."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    extracted: list[dict[str, Any]] = []
    parents: list[str] = []

    def visit(nodes: Sequence[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                parents.append(node.name)
                visit(node.body)
                parents.pop()
                continue
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            end_lineno = int(getattr(node, "end_lineno", node.lineno))
            extracted.append(
                {
                    "qualified_name": ".".join([*parents, node.name]),
                    "start_line": int(node.lineno),
                    "end_line": end_lineno,
                    "source": "\n".join(lines[node.lineno - 1 : end_lineno]),
                }
            )
            parents.append(node.name)
            visit(node.body)
            parents.pop()

    visit(tree.body)
    return extracted


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _disk_usage_payload(path: Path) -> Dict[str, int]:
    usage = shutil.disk_usage(path)
    return {
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def _pid_is_alive(pid: Any) -> bool:
    try:
        parsed = int(pid)
    except (TypeError, ValueError):
        return False
    if parsed <= 0:
        return False
    try:
        os.kill(parsed, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_start_token(pid: Any) -> str | None:
    """Return a token identifying *this specific process instance* for ``pid``.

    Built from the kernel-reported start time in ``/proc/<pid>/stat`` (field
    22, jiffies since boot). Two different OS processes can share the same
    numeric PID over time once the kernel recycles it; this token lets stale
    ownership metadata be told apart from a live, unrelated process that
    happens to reuse the old owner's PID number.
    """
    try:
        parsed = int(pid)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    try:
        raw = Path(f"/proc/{parsed}/stat").read_text()
    except OSError:
        return None
    try:
        after_comm = raw.rsplit(")", 1)[1]
    except IndexError:
        return None
    fields = after_comm.split()
    # After stripping "pid (comm)", fields[0] is state (stat field 3), so
    # starttime (stat field 22) is fields[19].
    if len(fields) <= 19:
        return None
    return fields[19]


def _owner_is_live(pid: Any, recorded_start_token: Any = None) -> bool:
    """Return whether a recorded lease owner is still the same live process.

    A dead PID is never live. A live PID whose current start token diverges
    from the token recorded at lease-creation time indicates PID reuse by an
    unrelated process, and is also treated as not-live so the stale lease can
    be reclaimed. Missing/unreadable identity data falls back to plain PID
    liveness rather than blocking reclamation on unverifiable evidence.
    """
    if not _pid_is_alive(pid):
        return False
    if recorded_start_token is None:
        return True
    current_token = _pid_start_token(pid)
    if current_token is None:
        return True
    return current_token == recorded_start_token


@contextlib.contextmanager
def _cross_process_repo_lock(lock_path: Path):
    """Serialize instance-view claim/reconcile across separate OS processes.

    ``_REPO_LOCKS`` only guards threads inside a single interpreter; the
    full449 batch runner can have multiple independent worker processes
    touching the same shared ``.instance_views`` state, and only a real
    kernel-level lock closes the check-then-act gap between reconciling a
    stale lease and claiming it.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _process_referencing_view(view_dir: Path) -> int | None:
    """Return a live PID that visibly references ``view_dir``, if any.

    This is deliberately conservative: it only provides positive evidence of
    an active owner.  A missing owner marker is never treated as safe merely
    because a path is old; the registered Git worktree and process evidence
    must both support stale-state reclamation.
    """
    target = str(view_dir.resolve())
    proc_root = Path("/proc")
    if not proc_root.exists():
        return None
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            cwd = str((entry / "cwd").resolve())
        except (OSError, RuntimeError):
            cwd = ""
        if cwd == target or cwd.startswith(target + os.sep):
            return pid
        try:
            raw_cmdline = (entry / "cmdline").read_bytes()
            cmdline = raw_cmdline.replace(b"\x00", b" ").decode(errors="replace")
        except OSError:
            cmdline = ""
        if target in cmdline:
            return pid
    return None


class SemanticMatchingClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...


def _resolve_flags(
    feature_flags: V22FeatureFlags | Dict[str, bool] | None,
    *,
    base: V22FeatureFlags | None = None,
) -> V22FeatureFlags:
    if isinstance(feature_flags, V22FeatureFlags):
        return resolve_feature_flags(feature_flags.to_dict(), base=base)
    if feature_flags is None:
        return base or core_only_feature_flags()
    return resolve_feature_flags(_normalize_m2_formula_legacy_alias(feature_flags), base=base)


def _normalize_m2_formula_legacy_alias(values: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(values)
    if "enable_formula_ranking" not in normalized:
        return normalized
    legacy_value = normalized.pop("enable_formula_ranking")
    canonical_value = normalized.get("m2_formula_ranking")
    if canonical_value is not None and canonical_value != legacy_value:
        raise ValueError("conflicting values for v22 feature flag: m2_formula_ranking")
    normalized["m2_formula_ranking"] = legacy_value
    return normalized


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


def _get_call_fullname(call: ast.Call) -> str:
    """Extract dotted name from a Call node, e.g. 'pytest.importorskip'."""
    func = call.func
    parts: List[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts)) if parts else ""


def _get_ast_dotted_name(node: ast.AST) -> str:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


@dataclass
class IndexedFile:
    path: str
    is_test_file: bool
    classes: List[str]
    functions: List[str]
    methods: List[str]
    test_functions: List[str]
    imports: List[str]
    call_names: List[str]
    attribute_names: List[str]
    parse_error: Optional[str] = None
    has_module_skip: bool = False


@dataclass
class CandidateFile:
    path: str
    score: int
    matched_identifiers: List[str]
    reasons: List[str]
    has_module_skip: bool = False
    code_snippets: Optional[Dict[str, str]] = None  # {identifier_name: "def foo(x):\n    ..."}
    top_level_functions: Optional[List[str]] = None  # AST로 추출한 public 함수 목록 (환각 방지)
    deterministic_text_similarity: Optional[float] = None
    metadata: Dict[str, Any] = None

    def to_fault_candidate(self, instance_id: str) -> FaultCandidate:
        score = self.deterministic_text_similarity
        validate_score_range(score, name="deterministic_text_similarity")
        return FaultCandidate(
            instance_id=instance_id,
            file_path=self.path,
            score=score,
            source="m2_deterministic_context",
            metadata={
                "entity_type": "file",
                "candidate_id": f"{instance_id}:file:{self.path}",
                "qualified_name": self.path,
                "rank_score": self.score,
                "matched_identifiers": list(self.matched_identifiers),
                "reasons": list(self.reasons),
                "selection_sources": list((self.metadata or {}).get("selection_sources") or []),
                "scores": {
                    "deterministic_text_similarity": score,
                    "tfidf_similarity": None,
                    "codebert_similarity": None,
                    "llm_relevance_raw": None,
                    "llm_relevance_norm": None,
                    "r_init": None,
                    "r_func": None,
                    "churn_norm": None,
                    "age_norm": None,
                },
                "unavailable_scores": [
                    "tfidf_similarity",
                    "codebert_similarity",
                    "llm_relevance_raw",
                    "llm_relevance_norm",
                    "r_init",
                    "r_func",
                    "churn_norm",
                    "age_norm",
                ],
                "top_level_functions": list(self.top_level_functions or []),
                "code_snippets": dict(self.code_snippets or {}),
                **dict(self.metadata or {}),
            },
        )


@dataclass(frozen=True)
class M2RankingWeights:
    """Weights for v22 M2 ranking.

    All input components and output scores are in [0, 1], where higher is
    better. The formula path clamps score components into range and rejects
    invalid weight sums.
    """

    gamma: float = 0.4
    alpha: float = 0.5
    beta: float = 0.5
    delta0: float = 0.7
    delta1: float = 0.1
    delta2: float = 0.1
    delta3: float = 0.1

    def validate(self) -> None:
        for name in ("gamma", "alpha", "beta", "delta0", "delta1", "delta2", "delta3"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite: {value!r}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]: {value!r}")
        if not math.isclose(self.alpha + self.beta, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("alpha + beta must equal 1")
        delta_sum = self.delta0 + self.delta1 + self.delta2 + self.delta3
        if not math.isclose(delta_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("delta weights must sum to 1")


@dataclass(frozen=True)
class FunctionMetricRecord:
    file_path: str
    function_name: str
    qualified_name: str
    start_line: int
    end_line: int
    size: int
    churn: Optional[int] = None
    age: Optional[int] = None
    churn_scope: str = "unavailable"
    git_history_available: bool = False
    provenance: Optional[Dict[str, Any]] = None
    unavailable_metric_diagnostics: Optional[List[str]] = None


@dataclass(frozen=True)
class _ResolvedImport:
    local_name: str
    module: str
    rel_path: str
    imported_name: Optional[str] = None


@dataclass(frozen=True)
class _GitFileMetrics:
    churn: Optional[int]
    age: Optional[int]
    git_history_available: bool
    provenance: Dict[str, Any]
    diagnostics: List[str]


@dataclass(frozen=True)
class M2RankingResult:
    candidate_files: List[Dict[str, Any]]
    file_ranking: List[Dict[str, Any]]
    function_ranking: List[Dict[str, Any]]
    initial_suspicious_functions: List[Dict[str, Any]]
    top5_functions: List[Dict[str, Any]]
    fault_hypothesis: Optional[str]
    oracle_hint: Optional[str]
    diagnostics: List[str]
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProjectTestStyle:
    framework: str
    evidence: List[str]
    assert_style: List[str]
    runner: str = "pytest"  # "pytest" | "django-test" | "sympy-bin-test" | "unittest" | "unknown"


@dataclass
class CodeContextFile:
    instance_id: str
    repo: str
    base_commit: str
    repo_path: str
    candidate_source_files: List[Dict[str, Any]]
    candidate_test_files: List[Dict[str, Any]]
    project_test_style: Dict[str, Any]
    indexed_file_count: int
    indexed_test_file_count: int
    available_imports: Optional[Dict[str, List[str]]] = None
    test_example_snippet: str = ""  # 상위 테스트 파일의 첫 번째 Test 클래스/함수 스니펫
    conftest_fixtures: Optional[Dict[str, List[str]]] = None  # {conftest_path: [fixture_names]}
    canonical_fault_candidates: Optional[List[Dict[str, Any]]] = None
    candidate_files: Optional[List[Dict[str, Any]]] = None
    file_ranking: Optional[List[Dict[str, Any]]] = None
    function_ranking: Optional[List[Dict[str, Any]]] = None
    initial_suspicious_functions: Optional[List[Dict[str, Any]]] = None
    top5_functions: Optional[List[Dict[str, Any]]] = None
    fault_hypothesis: Optional[str] = None
    oracle_hint: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    localization_hypotheses: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("available_imports") is None:
            d["available_imports"] = {}
        if d.get("conftest_fixtures") is None:
            d["conftest_fixtures"] = {}
        if d.get("canonical_fault_candidates") is None:
            d["canonical_fault_candidates"] = []
        for key in (
            "candidate_files",
            "file_ranking",
            "function_ranking",
            "initial_suspicious_functions",
            "top5_functions",
        ):
            if d.get(key) is None:
                d[key] = []
        if d.get("metadata") is None:
            d["metadata"] = {}
        if d.get("localization_hypotheses") is None:
            d["localization_hypotheses"] = []
        # V36 canonical boundary names. Historical field names remain as
        # compatibility aliases with the same ordered objects.
        d["suspicious_functions"] = list(d["top5_functions"])
        d["top_functions"] = list(d["top5_functions"])
        return d


class CodeContextExtractor:
    PROMPT_VERSION = "m2_llm_semantic_matching_v1"

    def __init__(
        self,
        repos_root: str = "data/repos",
        top_k_source: int = 5,
        top_k_test: int = 5,
        *,
        enable_formula_ranking: bool = False,
        ranking_weights: M2RankingWeights | None = None,
        history_window: int | None = None,
        llm_client: SemanticMatchingClient | None = None,
        isolate_instance_checkout: bool = False,
        instance_view_root: str | Path | None = None,
        feature_profile: str | None = None,
        codebert_encoder: V37CodeBertMethodEncoder | None = None,
    ) -> None:
        self.repos_root = Path(repos_root)
        self.instance_view_root = (
            Path(instance_view_root)
            if instance_view_root is not None
            else self.repos_root / ".instance_views"
        )
        self.top_k_source = top_k_source
        self.top_k_test = top_k_test
        self._constructor_feature_flags = resolve_feature_flags(
            {"m2_formula_ranking": enable_formula_ranking}
        )
        self.ranking_weights = ranking_weights or M2RankingWeights()
        self.history_window = history_window
        self.llm_client = llm_client
        self.isolate_instance_checkout = isolate_instance_checkout
        self.feature_profile = feature_profile
        self.codebert_encoder = codebert_encoder
        self._last_restart_constraints: Dict[str, Any] = {}
        self._last_api_function_evidence: Dict[str, Any] = {}

    def extract(
        self,
        instance: Any,
        clue: Dict[str, Any],
        *,
        feature_flags: V22FeatureFlags | Dict[str, bool] | None = None,
        restart_feedback: Mapping[str, Any] | None = None,
    ) -> CodeContextFile:
        view = self._coerce_pre_patch_view(instance)
        flags = _resolve_flags(feature_flags, base=self._constructor_feature_flags)
        if self.isolate_instance_checkout:
            repo_path = self._prepare_repo(
                view.repo,
                view.base_commit,
                instance_id=view.instance_id,
            )
        else:
            repo_path = self._prepare_repo(view.repo, view.base_commit)
        indexed_files = self._index_repository(repo_path)
        source_candidates, test_candidates = self._rank_files(
            indexed_files,
            clue,
            repo_path,
            use_formula_signals=flags.m2_formula_ranking,
            restart_feedback=restart_feedback,
        )
        restart_constraints = dict(self._last_restart_constraints)
        test_style = self._infer_test_style(indexed_files, test_candidates, repo_name=view.repo)
        available_imports = self._collect_available_imports(
            indexed_files, source_candidates, test_candidates, clue, repo_path,
        )
        test_example_snippet = self._extract_test_example(test_candidates, repo_path)
        conftest_fixtures = self._extract_conftest_fixtures(test_candidates, repo_path)
        llm_semantic = self._maybe_apply_m2_llm_semantic_matching(
            source_candidates,
            repo_path=repo_path,
            issue_text=str(clue.get("raw_issue_text") or view.problem_statement or ""),
            flags=flags,
            restart_feedback=restart_feedback,
        )
        ranking_result = None
        if flags.m2_formula_ranking or flags.enable_m2_llm_semantic_matching:
            ranking_result = self.build_m2_ranking(
                source_candidates,
                repo_path,
                flags=flags,
                history_window=self.history_window,
                weights=self.ranking_weights,
                clue=clue,
            )
            ranking_result, function_constraints = self._apply_function_restart_constraints(
                ranking_result,
                restart_feedback,
            )
            if function_constraints.get("function_exclusion_requested"):
                self._last_restart_constraints.update(function_constraints)
                restart_constraints = dict(self._last_restart_constraints)
        hypotheses = self._build_v30_hypotheses(
            ranking_result=ranking_result,
            candidate_test_files=[asdict(x) for x in test_candidates],
        ) if self.feature_profile in {"v30", "v31"} else []
        target_hypotheses = normalize_target_hypotheses(hypotheses) if self.feature_profile == "v31" else []

        context_metadata = self._feature_metadata(
            flags,
            history_window=self.history_window,
            llm_semantic=llm_semantic,
            restart_constraints=restart_constraints,
            isolated_source_view=self.isolate_instance_checkout,
            api_function_evidence=self._last_api_function_evidence,
        )
        if self.feature_profile == "v37":
            context_metadata["optional_features"]["codebert_similarity"] = {
                "implemented": True,
                "status": "method_scores_measured_file_aggregation_unavailable",
                "checkpoint": V37_CODEBERT_CHECKPOINT,
                "embedding_unit": "method_source",
                "representation": "last_hidden_state_attention_masked_mean_pooling",
                "embedding_dimension": V37_CODEBERT_EMBEDDING_DIM,
                "max_tokens": V37_CODEBERT_MAX_TOKENS,
                "truncation": "tail_truncation_prefix_retained",
                "file_aggregation": "SPEC_AMBIGUITY",
            }
        context_metadata["source_view"]["root"] = str(self.instance_view_root.resolve())
        if self.feature_profile == "v31":
            context_metadata["v31_target_hypotheses"] = [item.to_dict() for item in target_hypotheses]
            context_metadata["target_identity_provenance"] = "m2_pre_patch_ranked_evidence"
        if ranking_result is not None:
            llm_grounded = bool((llm_semantic or {}).get("used"))
            context_metadata["grounded_output_provenance"] = {
                "fault_hypothesis": (
                    "validated_m2_llm_semantic_matching"
                    if llm_grounded
                    else "deterministic_pre_patch_ranking_fallback"
                ),
                "oracle_hint": (
                    "validated_m2_llm_semantic_matching"
                    if llm_grounded
                    else "deterministic_m1_eb_s2r_fallback"
                ),
            }

        return CodeContextFile(
            instance_id=view.instance_id,
            repo=view.repo,
            base_commit=view.base_commit,
            repo_path=str(repo_path),
            candidate_source_files=[asdict(x) for x in source_candidates],
            candidate_test_files=[asdict(x) for x in test_candidates],
            project_test_style=asdict(test_style),
            indexed_file_count=len(indexed_files),
            indexed_test_file_count=sum(1 for x in indexed_files if x.is_test_file),
            available_imports=available_imports,
            test_example_snippet=test_example_snippet,
            conftest_fixtures=conftest_fixtures,
            canonical_fault_candidates=[
                candidate.to_fault_candidate(view.instance_id).to_dict()
                for candidate in source_candidates
            ],
            candidate_files=ranking_result.candidate_files if ranking_result else None,
            file_ranking=ranking_result.file_ranking if ranking_result else None,
            function_ranking=ranking_result.function_ranking if ranking_result else None,
            initial_suspicious_functions=(
                ranking_result.initial_suspicious_functions if ranking_result else None
            ),
            top5_functions=ranking_result.top5_functions if ranking_result else None,
            fault_hypothesis=ranking_result.fault_hypothesis if ranking_result else None,
            oracle_hint=ranking_result.oracle_hint if ranking_result else None,
            metadata=context_metadata,
            localization_hypotheses=hypotheses,
        )

    @staticmethod
    def _build_v30_hypotheses(
        *,
        ranking_result: M2RankingResult | None,
        candidate_test_files: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if ranking_result is None:
            return []
        raw: list[dict[str, Any]] = []
        entries = ranking_result.function_ranking or ranking_result.file_ranking
        for index, entry in enumerate(entries):
            source_file = str(entry.get("source_file") or entry.get("file_path") or entry.get("path") or "").strip()
            if not source_file:
                continue
            score = float(entry.get("R_func") or entry.get("R_init") or entry.get("score") or 0.0)
            component = entry.get("component_evidence") if isinstance(entry.get("component_evidence"), Mapping) else {}
            identifiers = component.get("identifier") if isinstance(component.get("identifier"), Mapping) else {}
            raw.append({
                "hypothesis_id": f"h{index + 1}",
                "source_file": source_file.replace("\\", "/"),
                "function_name": str(entry.get("function_name") or ""),
                "class_name": str(entry.get("class_name") or ""),
                "qualified_name": str(
                    entry.get("qualified_name")
                    or entry.get("canonical_symbol")
                    or entry.get("function_name")
                    or ""
                ),
                "line_range": [
                    int(entry.get("start_line")),
                    int(entry.get("end_line") or entry.get("start_line")),
                ] if entry.get("start_line") is not None else [],
                "confidence": max(0.0, min(1.0, score)),
                "evidence": list(entry.get("selection_sources") or []) + list(entry.get("reasons") or []),
                "issue_clue_support": list(identifiers.get("matched_identifiers") or []),
                "static_source_support": list(entry.get("selection_sources") or []),
                "candidate_test_files": [str(item.get("path") or "") for item in candidate_test_files if isinstance(item, Mapping)],
                "provenance": {"source": "m2_pre_patch_ranked_evidence", "rank": index + 1},
            })
        return normalize_localization_hypotheses(raw)

    def save(
        self,
        context: CodeContextFile,
        output_path: str,
        *,
        run_id: str | None = None,
        feature_flags: V22FeatureFlags | Dict[str, bool] | None = None,
        enveloped: bool = False,
    ) -> None:
        payload = context.to_dict()
        if enveloped:
            flags = _resolve_artifact_flags(feature_flags, payload)
            payload = make_envelope(
                instance_id=context.instance_id,
                run_id=run_id or context.instance_id,
                module="m2",
                payload=payload,
                feature_flags=flags,
            ).to_dict()
        write_json_atomic(payload, output_path)

    @staticmethod
    def _coerce_pre_patch_view(instance: Any) -> PrePatchInstanceView:
        if isinstance(instance, PrePatchInstanceView):
            assert_no_patch_access(instance.to_dict())
            return instance
        from src.benchmark.instance_loader import BenchmarkInstance

        if isinstance(instance, BenchmarkInstance):
            view = make_pre_patch_view(instance)
            assert_no_patch_access(view.to_dict())
            return view
        if hasattr(instance, "to_pre_patch_view"):
            view = instance.to_pre_patch_view()
            if not isinstance(view, PrePatchInstanceView):
                raise TypeError("to_pre_patch_view() must return PrePatchInstanceView")
            assert_no_patch_access(view.to_dict())
            return view
        raise TypeError("M2 requires PrePatchInstanceView or BenchmarkInstance")

    @staticmethod
    def _feature_metadata(
        flags: V22FeatureFlags,
        *,
        history_window: int | None = None,
        llm_semantic: Mapping[str, Any] | None = None,
        restart_constraints: Mapping[str, Any] | None = None,
        isolated_source_view: bool = False,
        api_function_evidence: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        llm_enabled = flags.enable_m2_llm_semantic_matching
        used = bool((llm_semantic or {}).get("used"))
        fallback_used = bool((llm_semantic or {}).get("fallback_used", llm_enabled and not used))
        status = str(
            (llm_semantic or {}).get(
                "status",
                "disabled" if not llm_enabled else "fallback_no_client",
            )
        )
        return {
            "feature_flags": {**flags.to_dict(), **flags.to_legacy_alias_dict()},
            "canonical_feature_flags": flags.to_dict(),
            "optional_features": {
                "m2_llm_semantic_matching": {
                    "enabled": llm_enabled,
                    "used": used,
                    "fallback_used": fallback_used,
                    "status": status,
                    "prompt_provenance": (llm_semantic or {}).get("prompt_provenance"),
                    "output_provenance": (llm_semantic or {}).get("output_provenance"),
                    "parser_status": (llm_semantic or {}).get("parser_status"),
                    "fallback_reason": (llm_semantic or {}).get("fallback_reason"),
                    "availability": {
                        "client_configured": bool((llm_semantic or {}).get("client_configured")),
                        "candidate_count": int((llm_semantic or {}).get("candidate_count") or 0),
                    },
                },
                "codebert_similarity": {
                    "implemented": False,
                    "status": "blocked_missing_checkpoint_and_pooling",
                },
                "churn_aware_ranking": {
                    "implemented": history_window is not None,
                    "status": "file_level_git_history_configured"
                    if history_window is not None
                    else "blocked_missing_churn_window",
                    "history_window": history_window,
                    "churn_scope": "file_level_inherited_by_function"
                    if history_window is not None
                    else "unavailable",
                },
            },
            "restart_constraints": dict(restart_constraints or {
                "requested": False,
                "applied": False,
                "prohibited_source_files": [],
                "excluded_source_files": [],
                "fallback_reason": "initial_m2_execution",
            }),
            "source_view": {
                "isolation": "per_instance_detached_worktree"
                if isolated_source_view
                else "shared_checkout",
            },
            "api_function_evidence": dict(api_function_evidence or {}),
        }

    def _maybe_apply_m2_llm_semantic_matching(
        self,
        candidates: List[CandidateFile],
        *,
        repo_path: Path,
        issue_text: str,
        flags: V22FeatureFlags,
        restart_feedback: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "used": False,
            "fallback_used": False,
            "status": "disabled",
            "fallback_reason": None,
            "parser_status": "not_invoked",
            "prompt_provenance": None,
            "output_provenance": None,
            "client_configured": self.llm_client is not None,
            "candidate_count": len(candidates),
        }
        if not flags.enable_m2_llm_semantic_matching:
            return result
        if not candidates:
            result.update(
                {
                    "fallback_used": True,
                    "status": "fallback_no_candidates",
                    "fallback_reason": "no_candidate_files",
                }
            )
            return result
        if self.llm_client is None:
            result.update(
                {
                    "fallback_used": True,
                    "status": "fallback_no_client",
                    "fallback_reason": "llm_client_unavailable",
                }
            )
            return result

        prompt = self.build_m2_semantic_matching_prompt(
            issue_text=issue_text,
            candidates=candidates,
            repo_path=repo_path,
            restart_feedback=restart_feedback,
        )
        result["prompt_provenance"] = {
            "module": "m2",
            "prompt_version": self.PROMPT_VERSION,
            "builder": "CodeContextExtractor.build_m2_semantic_matching_prompt",
            "candidate_count": len(candidates),
            "single_batch_call": True,
            "m7_diagnosis_consumed": bool(restart_feedback),
        }
        try:
            raw = self.llm_client.complete(prompt)
        except (RuntimeError, ValueError, TypeError) as exc:
            result.update(
                {
                    "fallback_used": True,
                    "status": "fallback_client_error",
                    "fallback_reason": f"llm_client_error:{type(exc).__name__}",
                    "parser_status": "not_parsed_client_error",
                }
            )
            return result

        allowed_paths = {candidate.path for candidate in candidates}
        parsed = self.parse_m2_semantic_matching_response(raw, allowed_paths=allowed_paths)
        result["parser_status"] = parsed["parser_status"]
        result["output_provenance"] = {
            "parser": "CodeContextExtractor.parse_m2_semantic_matching_response",
            "candidate_count": len(candidates),
            "accepted_count": len(parsed.get("items") or []),
        }
        if not parsed["valid"]:
            result.update(
                {
                    "fallback_used": True,
                    "status": "fallback_invalid_response",
                    "fallback_reason": parsed["fallback_reason"],
                }
            )
            return result

        by_path = {item["file_path"]: item for item in parsed["items"]}
        for candidate in candidates:
            item = by_path[candidate.path]
            metadata = dict(candidate.metadata or {})
            scores = dict(metadata.get("scores") or {})
            scores["llm_relevance_raw"] = item["relevance_score"]
            scores["llm_relevance_norm"] = item["relevance_norm"]
            metadata["scores"] = scores
            metadata["llm_relevance_raw"] = item["relevance_score"]
            metadata["fault_hypothesis"] = item["fault_hypothesis"]
            metadata["oracle_hint"] = item["oracle_hint"]
            metadata["llm_semantic_matching"] = {
                "prompt_version": self.PROMPT_VERSION,
                "source": "m2_llm_semantic_matching",
                "validated_against_candidate_file": True,
            }
            candidate.metadata = metadata
        result.update({"used": True, "status": "used"})
        return result

    @classmethod
    def build_m2_semantic_matching_prompt(
        cls,
        *,
        issue_text: str,
        candidates: List[CandidateFile],
        repo_path: Path,
        restart_feedback: Mapping[str, Any] | None = None,
    ) -> str:
        payload_candidates = []
        for candidate in candidates:
            payload_candidates.append(
                {
                    "file_path": candidate.path,
                    "snippet": cls._read_candidate_prompt_snippet(repo_path / candidate.path),
                }
            )
        payload = {
            "prompt_version": cls.PROMPT_VERSION,
            "task": "Score each allowed pre-patch candidate file for issue relevance.",
            "constraints": [
                "Return strict JSON only.",
                "Return exactly one item per allowed candidate file.",
                "Do not add unknown files.",
                "Do not use patch, post-patch, golden tests, golden patch lines, M8, Fail-to-Pass, or Patch Hit Rate data.",
            ],
            "required_schema": {
                "files": [
                    {
                        "file_path": "allowed/candidate.py",
                        "relevance_score": 1,
                        "fault_hypothesis": "brief hypothesis",
                        "oracle_hint": "brief oracle hint",
                    }
                ]
            },
            "issue_text": issue_text,
            "m7_diagnosis_to_consume": {
                key: restart_feedback.get(key)
                for key in (
                    "why_failed",
                    "fix_suggestion",
                    "failure_reason",
                    "assumption_gap",
                    "next_scenario_change",
                    "admissible_alternatives",
                    "route_destination",
                )
                if isinstance(restart_feedback, Mapping)
                and restart_feedback.get(key) not in (None, "", [])
            },
            "candidate_files": payload_candidates,
        }
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def _read_candidate_prompt_snippet(path: Path, max_lines: int = 30) -> str:
        try:
            lines = read_text(path).splitlines()
        except Exception:
            return ""
        return "\n".join(lines[:max_lines])

    @staticmethod
    def parse_m2_semantic_matching_response(
        raw: str,
        *,
        allowed_paths: set[str],
    ) -> Dict[str, Any]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "valid": False,
                "parser_status": "malformed_json",
                "fallback_reason": "malformed_json",
            }
        if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
            return {
                "valid": False,
                "parser_status": "missing_files_list",
                "fallback_reason": "missing_files_list",
            }
        if CodeContextExtractor._contains_prohibited_data(payload):
            return {
                "valid": False,
                "parser_status": "prohibited_data_rejected",
                "fallback_reason": "prohibited_data",
            }
        items: List[Dict[str, Any]] = []
        seen_paths: set[str] = set()
        for item in payload["files"]:
            if not isinstance(item, dict):
                return {
                    "valid": False,
                    "parser_status": "invalid_file_item",
                    "fallback_reason": "invalid_file_item",
                }
            path = str(item.get("file_path", "")).replace("\\", "/").strip()
            if path not in allowed_paths:
                return {
                    "valid": False,
                    "parser_status": "unknown_or_non_candidate_file",
                    "fallback_reason": "unknown_or_non_candidate_file",
                }
            if path in seen_paths:
                return {
                    "valid": False,
                    "parser_status": "duplicate_candidate_file",
                    "fallback_reason": "duplicate_candidate_file",
                }
            seen_paths.add(path)
            missing = [
                key
                for key in ("relevance_score", "fault_hypothesis", "oracle_hint")
                if key not in item
            ]
            if missing:
                return {
                    "valid": False,
                    "parser_status": "missing_required_fields",
                    "fallback_reason": "missing_required_fields",
                }
            try:
                relevance_score = int(item["relevance_score"])
            except (TypeError, ValueError):
                return {
                    "valid": False,
                    "parser_status": "invalid_relevance_score",
                    "fallback_reason": "invalid_relevance_score",
                }
            if relevance_score < 1 or relevance_score > 5:
                return {
                    "valid": False,
                    "parser_status": "invalid_relevance_score",
                    "fallback_reason": "invalid_relevance_score",
                }
            fault_hypothesis = item["fault_hypothesis"]
            oracle_hint = item["oracle_hint"]
            if not isinstance(fault_hypothesis, str) or not fault_hypothesis.strip():
                return {
                    "valid": False,
                    "parser_status": "invalid_fault_hypothesis",
                    "fallback_reason": "invalid_fault_hypothesis",
                }
            if not isinstance(oracle_hint, str) or not oracle_hint.strip():
                return {
                    "valid": False,
                    "parser_status": "invalid_oracle_hint",
                    "fallback_reason": "invalid_oracle_hint",
                }
            items.append(
                {
                    "file_path": path,
                    "relevance_score": relevance_score,
                    "relevance_norm": (relevance_score - 1) / 4.0,
                    "fault_hypothesis": fault_hypothesis.strip(),
                    "oracle_hint": oracle_hint.strip(),
                }
            )
        if seen_paths != allowed_paths:
            return {
                "valid": False,
                "parser_status": "missing_candidate_files",
                "fallback_reason": "missing_candidate_files",
            }
        return {
            "valid": True,
            "parser_status": "parsed",
            "fallback_reason": None,
            "items": items,
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
                if CodeContextExtractor._contains_prohibited_data(item):
                    return True
        elif isinstance(value, list):
            return any(CodeContextExtractor._contains_prohibited_data(item) for item in value)
        elif isinstance(value, str):
            lowered = value.lower()
            normalized = re.sub(r"[-\s]+", "_", lowered)
            return any(
                re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", normalized)
                for token in prohibited
            )
        return False

    def _prepare_repo(
        self,
        repo_name: str,
        base_commit: str,
        *,
        instance_id: str | None = None,
    ) -> Path:
        parts = repo_name.split("/")
        if len(parts) != 2:
            raise ValueError(f"repo_name 형식이 잘못됨 (expected 'owner/repo'): {repo_name!r}")
        owner, name = parts
        repo_dir = self.repos_root / f"{owner}__{name}"
        clone_url = f"https://github.com/{repo_name}.git"

        self.repos_root.mkdir(parents=True, exist_ok=True)

        repo_key = str(repo_dir.resolve())
        repo_lock = _REPO_LOCKS[repo_key]

        # 같은 repo에 대해서는 git 작업을 한 번에 하나의 스레드만 수행
        with repo_lock:
            if not repo_dir.exists():
                self._run_git(["clone", clone_url, str(repo_dir)], cwd=Path("."))

            self._run_git(["fetch", "--all", "--tags"], cwd=repo_dir)
            if self.isolate_instance_checkout:
                if not instance_id:
                    raise ValueError("instance_id is required for isolated source views")
                views_root = self.instance_view_root.resolve()
                views_root.mkdir(parents=True, exist_ok=True)
                repo_view_lock_path = views_root / _INSTANCE_VIEW_LIFECYCLE_DIR / f".lock.{owner}__{name}"
                # threading.Lock (repo_lock, above) only excludes other threads
                # in this interpreter. The full449 batch runner can have
                # independent worker processes sharing this same
                # .instance_views state, so the reconcile-then-claim sequence
                # below also needs a real cross-process lock to close the
                # check-then-act race between two separate PIDs.
                with _cross_process_repo_lock(repo_view_lock_path):
                    self.reconcile_stale_instance_views(
                        repos_root=self.repos_root,
                        repo_name=repo_name,
                        instance_view_root=self.instance_view_root,
                    )
                    resolved_commit = self._git_stdout(
                        ["rev-parse", f"{base_commit}^{{commit}}"], cwd=repo_dir
                    )
                    safe_instance_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", instance_id)
                    view_name = f"{safe_instance_id}__{resolved_commit[:12]}"
                    view_dir = (views_root / view_name).resolve()
                    lifecycle_dir = views_root / _INSTANCE_VIEW_LIFECYCLE_DIR
                    marker_path = lifecycle_dir / f"{view_name}.json"
                    view_dir.parent.mkdir(parents=True, exist_ok=True)
                    lifecycle_dir.mkdir(parents=True, exist_ok=True)
                    reclaimed_unmarked = False
                    if view_dir.exists():
                        if not (view_dir / ".git").exists():
                            raise RuntimeError(
                                f"isolated source view exists but is not a Git worktree: {view_dir}"
                            )
                        marker = self._read_lifecycle_marker(marker_path)
                        if not marker and self._can_reclaim_unmarked_view(repo_dir, view_dir):
                            self._run_git_checked(
                                ["worktree", "remove", "--force", str(view_dir)],
                                cwd=repo_dir,
                            )
                            self._run_git_checked(["worktree", "prune"], cwd=repo_dir)
                            marker = {}
                            reclaimed_unmarked = True
                        if not reclaimed_unmarked and not self._marker_matches_view(
                            marker,
                            repo_name=repo_name,
                            instance_id=instance_id,
                            view_dir=view_dir,
                        ):
                            raise RuntimeError(
                                "refusing to reuse an isolated source view not owned by this "
                                f"pipeline invocation: {view_dir}"
                            )
                        if not reclaimed_unmarked and int(marker.get("owner_pid") or -1) != os.getpid():
                            # reconcile_stale_instance_views (above, under the
                            # same cross-process lock) already removed any
                            # dead-or-PID-reused owner's marker+worktree for
                            # this repo, so a marker still present here and
                            # not ours belongs to a genuinely live owner.
                            raise RuntimeError(
                                f"isolated source view has another active owner: {view_dir}"
                            )
                    else:
                        marker = {
                            "schema_version": _INSTANCE_VIEW_MARKER_SCHEMA,
                            "repo_name": repo_name,
                            "instance_id": instance_id,
                            "source_view_path": str(view_dir),
                            "base_commit": resolved_commit,
                            "owner_pid": os.getpid(),
                            "owner_pid_start_token": _pid_start_token(os.getpid()),
                            "created_at": _utc_now_iso(),
                            "cleaned_at": None,
                            "cleanup_status": "ACTIVE",
                            "cleanup_error": None,
                            "disk_usage_before": _disk_usage_payload(views_root),
                        }
                        write_json_atomic(marker, marker_path)
                        self._run_git(
                            ["worktree", "add", "--detach", str(view_dir), resolved_commit],
                            cwd=repo_dir,
                        )
                    # A safely reclaimed unmarked worktree, or a stale-owner
                    # view removed above, is now absent and must follow the
                    # normal marker-before-worktree creation protocol.
                    if not view_dir.exists():
                        marker = {
                            "schema_version": _INSTANCE_VIEW_MARKER_SCHEMA,
                            "repo_name": repo_name,
                            "instance_id": instance_id,
                            "source_view_path": str(view_dir),
                            "base_commit": resolved_commit,
                            "owner_pid": os.getpid(),
                            "owner_pid_start_token": _pid_start_token(os.getpid()),
                            "created_at": _utc_now_iso(),
                            "cleaned_at": None,
                            "cleanup_status": "ACTIVE",
                            "cleanup_error": None,
                            "disk_usage_before": _disk_usage_payload(views_root),
                        }
                        write_json_atomic(marker, marker_path)
                        self._run_git(
                            ["worktree", "add", "--detach", str(view_dir), resolved_commit],
                            cwd=repo_dir,
                        )
                    actual_commit = self._git_stdout(["rev-parse", "HEAD"], cwd=view_dir)
                    if actual_commit != resolved_commit:
                        raise RuntimeError(
                            "isolated source view commit mismatch: "
                            f"expected={resolved_commit} actual={actual_commit} path={view_dir}"
                        )
                    dirty = self._git_stdout(["status", "--porcelain"], cwd=view_dir)
                    if dirty:
                        raise RuntimeError(
                            "isolated source view is dirty; refusing destructive reset/clean: "
                            f"{view_dir}"
                        )
                    return view_dir

            self._run_git(["reset", "--hard"], cwd=repo_dir)
            self._run_git(["clean", "-fd"], cwd=repo_dir)
            self._run_git(["checkout", base_commit], cwd=repo_dir)

        return repo_dir

    @staticmethod
    def _can_reclaim_unmarked_view(repo_dir: Path, view_dir: Path) -> bool:
        """Allow recreation only for a registered, owner-less Git worktree.

        Unmarked paths are otherwise refused.  A registered worktree with no
        live process referencing it is the narrow stale-state case left by an
        interrupted pre-marker invocation; active or unregistered paths remain
        protected to preserve source-view isolation.
        """
        if _process_referencing_view(view_dir) is not None:
            return False
        try:
            listing = subprocess.run(
                ["git", "worktree", "list", "--porcelain"],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return False
        if listing.returncode != 0:
            return False
        target = str(view_dir.resolve())
        return any(
            line.strip() == f"worktree {target}"
            for line in listing.stdout.splitlines()
        )

    @staticmethod
    def _read_lifecycle_marker(marker_path: Path) -> Dict[str, Any]:
        try:
            data = json.loads(marker_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid instance-view lifecycle marker: {marker_path}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"instance-view lifecycle marker is not an object: {marker_path}")
        return data

    @staticmethod
    def _marker_matches_view(
        marker: Mapping[str, Any],
        *,
        repo_name: str,
        instance_id: str,
        view_dir: Path,
    ) -> bool:
        return bool(
            marker.get("schema_version") == _INSTANCE_VIEW_MARKER_SCHEMA
            and marker.get("repo_name") == repo_name
            and marker.get("instance_id") == instance_id
            and Path(str(marker.get("source_view_path") or "")).resolve() == view_dir.resolve()
        )

    @classmethod
    def release_instance_view(
        cls,
        *,
        repos_root: str | Path,
        repo_name: str,
        instance_id: str,
        instance_view_root: str | Path | None = None,
    ) -> Dict[str, Any]:
        """Release this process's pipeline-owned view and prune Git metadata.

        Result artifacts are deliberately not stored in the worktree.  Cleanup
        only acts on a marker bearing the pipeline schema, exact instance/repo,
        and current owner PID; unrelated or active user worktrees are untouched.
        """
        root = Path(repos_root)
        parts = repo_name.split("/")
        if len(parts) != 2:
            raise ValueError(f"repo_name must be 'owner/repo': {repo_name!r}")
        repo_dir = root / f"{parts[0]}__{parts[1]}"
        views_root = (
            Path(instance_view_root).resolve()
            if instance_view_root is not None
            else (root / ".instance_views").resolve()
        )
        lifecycle_dir = views_root / _INSTANCE_VIEW_LIFECYCLE_DIR
        usage_path = views_root if views_root.exists() else views_root.parent
        safe_instance_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", instance_id)
        marker_paths = sorted(lifecycle_dir.glob(f"{safe_instance_id}__*.json"))
        if not marker_paths:
            return {
                "schema_version": _INSTANCE_VIEW_MARKER_SCHEMA,
                "instance_id": instance_id,
                "repo_name": repo_name,
                "source_view_path": None,
                "created_at": None,
                "cleaned_at": _utc_now_iso(),
                "cleanup_status": "NOT_ACQUIRED",
                "cleanup_error": None,
                "base_commit": None,
                "disk_usage_before": None,
                "disk_usage_after": _disk_usage_payload(usage_path),
            }
        records: List[Dict[str, Any]] = []
        repo_key = str(repo_dir.resolve())
        with _REPO_LOCKS[repo_key]:
            for marker_path in marker_paths:
                marker = cls._read_lifecycle_marker(marker_path)
                view_dir = Path(str(marker.get("source_view_path") or "")).resolve()
                valid = cls._marker_matches_view(
                    marker,
                    repo_name=repo_name,
                    instance_id=instance_id,
                    view_dir=view_dir,
                )
                valid = valid and view_dir.parent == views_root
                if not valid:
                    records.append({
                        **marker,
                        "cleaned_at": _utc_now_iso(),
                        "cleanup_status": "SKIPPED_INVALID_MARKER",
                        "cleanup_error": "marker ownership/path validation failed",
                        "disk_usage_after": _disk_usage_payload(usage_path),
                    })
                    continue
                if int(marker.get("owner_pid") or -1) != os.getpid():
                    records.append({
                        **marker,
                        "cleaned_at": _utc_now_iso(),
                        "cleanup_status": "SKIPPED_ACTIVE_OWNER",
                        "cleanup_error": "view is not owned by the current process",
                        "disk_usage_after": _disk_usage_payload(usage_path),
                    })
                    continue
                record = dict(marker)
                try:
                    if view_dir.exists():
                        cls._run_git_checked(
                            ["worktree", "remove", "--force", str(view_dir)],
                            cwd=repo_dir,
                        )
                    cls._run_git_checked(["worktree", "prune"], cwd=repo_dir)
                    marker_path.unlink(missing_ok=True)
                    record["cleanup_status"] = "CLEANED"
                    record["cleanup_error"] = None
                except (OSError, RuntimeError) as exc:
                    record["cleanup_status"] = "CLEANUP_FAILED"
                    record["cleanup_error"] = str(exc)
                    write_json_atomic(record, marker_path)
                record["cleaned_at"] = _utc_now_iso()
                record["disk_usage_after"] = _disk_usage_payload(usage_path)
                records.append(record)
        if len(records) == 1:
            return records[0]
        return {
            "schema_version": _INSTANCE_VIEW_MARKER_SCHEMA,
            "instance_id": instance_id,
            "repo_name": repo_name,
            "cleanup_status": (
                "CLEANED" if all(r.get("cleanup_status") == "CLEANED" for r in records)
                else "PARTIAL_FAILURE"
            ),
            "cleanup_error": [r.get("cleanup_error") for r in records if r.get("cleanup_error")],
            "views": records,
            "cleaned_at": _utc_now_iso(),
            "disk_usage_after": _disk_usage_payload(usage_path),
        }

    @classmethod
    def reconcile_stale_instance_views(
        cls,
        *,
        repos_root: str | Path,
        repo_name: str,
        instance_view_root: str | Path | None = None,
    ) -> List[Dict[str, Any]]:
        """Remove only dead-owner views carrying this pipeline's marker."""
        root = Path(repos_root)
        parts = repo_name.split("/")
        if len(parts) != 2:
            raise ValueError(f"repo_name must be 'owner/repo': {repo_name!r}")
        repo_dir = root / f"{parts[0]}__{parts[1]}"
        views_root = (
            Path(instance_view_root).resolve()
            if instance_view_root is not None
            else (root / ".instance_views").resolve()
        )
        lifecycle_dir = views_root / _INSTANCE_VIEW_LIFECYCLE_DIR
        reconciled: List[Dict[str, Any]] = []
        if not lifecycle_dir.exists() or not repo_dir.exists():
            return reconciled
        for marker_path in sorted(lifecycle_dir.glob("*.json")):
            marker = cls._read_lifecycle_marker(marker_path)
            if marker.get("schema_version") != _INSTANCE_VIEW_MARKER_SCHEMA:
                continue
            if marker.get("repo_name") != repo_name or _owner_is_live(
                marker.get("owner_pid"), marker.get("owner_pid_start_token")
            ):
                continue
            view_dir = Path(str(marker.get("source_view_path") or "")).resolve()
            instance_id = str(marker.get("instance_id") or "")
            if not instance_id or not cls._marker_matches_view(
                marker,
                repo_name=repo_name,
                instance_id=instance_id,
                view_dir=view_dir,
            ) or view_dir.parent != views_root:
                continue
            record = dict(marker)
            try:
                if view_dir.exists():
                    cls._run_git_checked(
                        ["worktree", "remove", "--force", str(view_dir)],
                        cwd=repo_dir,
                    )
                cls._run_git_checked(["worktree", "prune"], cwd=repo_dir)
                marker_path.unlink(missing_ok=True)
                record["cleanup_status"] = "RECONCILED"
                record["cleanup_error"] = None
            except (OSError, RuntimeError) as exc:
                record["cleanup_status"] = "RECONCILE_FAILED"
                record["cleanup_error"] = str(exc)
                write_json_atomic(record, marker_path)
            record["cleaned_at"] = _utc_now_iso()
            record["disk_usage_after"] = _disk_usage_payload(views_root)
            reconciled.append(record)
        return reconciled

    @staticmethod
    def _run_git_checked(args: List[str], *, cwd: Path) -> None:
        command = ["git", *args]
        result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Git command failed: {' '.join(command)}\n"
                f"cwd={cwd}\nstdout={result.stdout}\nstderr={result.stderr}"
            )

    def _git_stdout(self, args: List[str], cwd: Path) -> str:
        cmd = ["git"] + args
        result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Git command failed: {' '.join(cmd)}\n"
                f"cwd={cwd}\nstdout={result.stdout}\nstderr={result.stderr}"
            )
        return result.stdout.strip()

    def _run_git(self, args: List[str], cwd: Path) -> None:
        cmd = ["git"] + args
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Git command failed: {' '.join(cmd)}\n"
                f"cwd={cwd}\n"
                f"stdout={result.stdout}\n"
                f"stderr={result.stderr}"
            )

    def _index_repository(self, repo_path: Path) -> List[IndexedFile]:
        indexed: List[IndexedFile] = []

        for file_path in repo_path.rglob("*.py"):
            if self._should_skip(file_path):
                continue

            rel_path = str(file_path.relative_to(repo_path))
            is_test_file = self._is_test_file(rel_path)

            try:
                source = read_text(file_path)
            except UnicodeDecodeError:
                try:
                    source = file_path.read_text(encoding="latin-1")
                except Exception as e:
                    indexed.append(
                        IndexedFile(
                            path=rel_path,
                            is_test_file=is_test_file,
                            classes=[],
                            functions=[],
                            methods=[],
                            test_functions=[],
                            imports=[],
                            call_names=[],
                            attribute_names=[],
                            parse_error=f"read_error: {e}",
                        )
                    )
                    continue
            except Exception as e:
                indexed.append(
                    IndexedFile(
                        path=rel_path,
                        is_test_file=is_test_file,
                        classes=[],
                        functions=[],
                        methods=[],
                        test_functions=[],
                        imports=[],
                        call_names=[],
                        attribute_names=[],
                        parse_error=f"read_error: {e}",
                    )
                )
                continue

            indexed.append(self._parse_python_file(rel_path, source, is_test_file))

        return indexed

    def _parse_python_file(self, rel_path: str, source: str, is_test_file: bool) -> IndexedFile:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source)
        except Exception as e:
            return IndexedFile(
                path=rel_path,
                is_test_file=is_test_file,
                classes=[],
                functions=[],
                methods=[],
                test_functions=[],
                imports=[],
                call_names=[],
                attribute_names=[],
                parse_error=f"ast_error: {e}",
            )

        classes: set[str] = set()
        functions: set[str] = set()
        methods: set[str] = set()
        test_functions: set[str] = set()
        imports: set[str] = set()
        call_names: set[str] = set()
        attribute_names: set[str] = set()

        has_module_skip = False

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.add(node.name)
                is_unittest_case = any(
                    (
                        isinstance(base, ast.Name)
                        and base.id.endswith("TestCase")
                    )
                    or (
                        isinstance(base, ast.Attribute)
                        and base.attr.endswith("TestCase")
                    )
                    for base in node.bases
                )
                explicitly_disabled = any(
                    isinstance(item, (ast.Assign, ast.AnnAssign))
                    and any(
                        isinstance(target, ast.Name) and target.id == "__test__"
                        for target in (item.targets if isinstance(item, ast.Assign) else [item.target])
                    )
                    and isinstance(item.value, ast.Constant)
                    and item.value.value is False
                    for item in node.body
                )
                custom_constructor = any(
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name in {"__init__", "__new__"}
                    for item in node.body
                )
                collectable_test_class = (
                    not explicitly_disabled
                    and (node.name.startswith("Test") or is_unittest_case)
                    and not custom_constructor
                )
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.add(item.name)
                        if collectable_test_class and item.name.startswith("test"):
                            test_functions.add(item.name)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.add(node.name)
                if node.name.startswith("test"):
                    test_functions.add(node.name)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)

            # Detect module-level skip patterns (only for test files)
            if is_test_file and not has_module_skip:
                has_module_skip = self._is_module_level_skip(node)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = _get_call_fullname(node)
                if call_name:
                    call_names.add(call_name)
                    call_names.add(call_name.split(".")[-1])
            elif isinstance(node, ast.Attribute):
                attribute_names.add(node.attr)

        test_module_name = Path(rel_path).name.lower()
        disabled_functions = {
            target.value.id
            for assignment in tree.body
            if isinstance(assignment, (ast.Assign, ast.AnnAssign))
            for target in (assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target])
            if isinstance(target, ast.Attribute)
            and target.attr == "__test__"
            and isinstance(target.value, ast.Name)
            and isinstance(assignment.value, ast.Constant)
            and assignment.value.value is False
        }
        test_functions.difference_update(disabled_functions)
        conventionally_collectable = (
            test_module_name.startswith("test") and test_module_name.endswith(".py")
        ) or test_module_name.endswith("_test.py") or (
            test_module_name.startswith("unittest_")
            and test_module_name.endswith(".py")
        )
        is_test_file = is_test_file and (
            conventionally_collectable or bool(test_functions)
        )

        # GIS/환경 의존 경로는 module_skip으로 마킹 (GIS 미설치 환경에서 항상 실패)
        _ENV_SKIP_PATTERNS = ("gis_tests/", "contrib/gis/")
        if is_test_file and not has_module_skip:
            if any(pat in rel_path.replace("\\", "/") for pat in _ENV_SKIP_PATTERNS):
                has_module_skip = True

        return IndexedFile(
            path=rel_path,
            is_test_file=is_test_file,
            classes=sorted(classes),
            functions=sorted(functions),
            methods=sorted(methods),
            test_functions=sorted(test_functions),
            imports=sorted(imports),
            call_names=sorted(call_names),
            attribute_names=sorted(attribute_names),
            parse_error=None,
            has_module_skip=has_module_skip,
        )

    @staticmethod
    def _is_module_level_skip(node: ast.AST) -> bool:
        """Detect module-level skip patterns that prevent test collection.

        Patterns detected:
          - pytest.importorskip("pkg")
          - var = pytest.importorskip("pkg")
          - pytest.skip("reason")
          - pytestmark = pytest.mark.skip(...)
          - pytestmark = pytest.mark.skipif(...)
          - pytestmark = [pytest.mark.skip(...)]
        """
        # --- Expr: bare pytest.importorskip(...) or pytest.skip(...) ---
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            func_name = _get_call_fullname(call)
            if func_name in ("pytest.importorskip", "pytest.skip"):
                return True

        # --- Assign ---
        if isinstance(node, ast.Assign):
            # var = pytest.importorskip(...)
            if isinstance(node.value, ast.Call):
                func_name = _get_call_fullname(node.value)
                if func_name == "pytest.importorskip":
                    return True

                # pytestmark = pytest.mark.skip(...) / skipif(...)
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "pytestmark":
                        if func_name and ("pytest.mark.skip" in func_name):
                            return True

            # pytestmark = [pytest.mark.skip(...), ...]
            if isinstance(node.value, (ast.List, ast.Tuple)):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "pytestmark":
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Call):
                                elt_name = _get_call_fullname(elt)
                                if elt_name and "pytest.mark.skip" in elt_name:
                                    return True

        # --- try-except block: pytest.importorskip / pytest.skip inside try ---
        if isinstance(node, ast.Try):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func_name = _get_call_fullname(child)
                    if func_name in ("pytest.importorskip", "pytest.skip"):
                        return True

        # --- if block: if condition: pytest.skip() / pytest.importorskip() ---
        if isinstance(node, ast.If):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func_name = _get_call_fullname(child)
                    if func_name in ("pytest.importorskip", "pytest.skip"):
                        return True

        return False

    def _should_skip(self, file_path: Path) -> bool:
        skip_parts = {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            "site-packages",
            "node_modules",
            ".mypy_cache",
            ".pytest_cache",
            "build",
            "dist",
        }
        return any(part in skip_parts for part in file_path.parts)

    def _is_test_file(self, rel_path: str) -> bool:
        p = rel_path.replace("\\", "/").lower()
        name = Path(p).name
        return (
            "/tests/" in p
            or p.startswith("tests/")
            or name.startswith("test_")
            or name.endswith("_test.py")
            or name == "conftest.py"
        )

    def _extract_code_snippets(
        self,
        file_path: Path,
        identifiers: List[str],
        max_lines: int = 15,
    ) -> Dict[str, str]:
        """matched identifiers에 해당하는 함수/클래스 시그니처+본문 앞부분 추출."""
        if not identifiers or not file_path.exists():
            return {}
        try:
            source = read_text(file_path)
            tree = ast.parse(source)
        except Exception:
            return {}
        lines = source.splitlines()
        ident_set = set(identifiers)
        snippets: Dict[str, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if node.name not in ident_set:
                continue
            start = node.lineno - 1
            end = min(start + max_lines, len(lines))
            snippets[node.name] = "\n".join(lines[start:end])
        return snippets

    def _extract_top_level_functions(self, file_path: Path, max_results: int = 40) -> List[str]:
        """소스 파일에서 top-level public 함수 및 클래스의 public 메서드 이름 추출.

        ScenarioGenerator가 target_function 선택 시 실제 존재하는 함수명만
        사용하도록 돕기 위한 힌트 목록. 환각 방지 목적.
        """
        if not file_path.exists():
            return []
        try:
            source = read_text(file_path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source)
        except Exception:
            return []

        seen: set = set()
        names: List[str] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_") and node.name not in seen:
                    seen.add(node.name)
                    names.append(node.name)
            elif isinstance(node, ast.ClassDef):
                for item in ast.iter_child_nodes(node):
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        qualified = f"{node.name}.{item.name}"
                        if not item.name.startswith("_") and qualified not in seen:
                            seen.add(qualified)
                            names.append(qualified)
        return names[:max_results]

    def _extract_code_example_symbols(self, clue: Dict[str, Any]) -> Tuple[set[str], set[str]]:
        """Return call names and imported modules from issue reproduction code."""
        call_names: set[str] = set()
        import_modules: set[str] = set()
        for block in clue.get("code_examples", []) or []:
            if block.get("is_system_or_output"):
                continue
            code = block.get("interactive_input") or block.get("code", "") or ""
            if not code.strip():
                continue
            try:
                tree = ast.parse(code)
            except SyntaxError:
                tree = None
            if tree is not None:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        call_name = _get_call_fullname(node)
                        if call_name:
                            call_names.add(call_name)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            import_modules.add(alias.name)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        import_modules.add(node.module)
                        for alias in node.names:
                            if alias.name != "*":
                                call_names.add(alias.name)

            for m in re.finditer(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(", code):
                name = m.group(1)
                if name not in {"if", "for", "while", "with", "return"}:
                    call_names.add(name)
            for m in re.finditer(r"^\s*from\s+([A-Za-z_][\w.]*)\s+import\s+(.+)$", code, re.MULTILINE):
                import_modules.add(m.group(1))
                for symbol in re.split(r"\s*,\s*", m.group(2).strip("() ")):
                    symbol = symbol.split(" as ", 1)[0].strip()
                    if re.match(r"^[A-Za-z_]\w*$", symbol):
                        call_names.add(symbol)
            for m in re.finditer(r"^\s*import\s+([A-Za-z_][\w.]*)", code, re.MULTILINE):
                import_modules.add(m.group(1))
        return call_names, import_modules

    def _source_document_tokens(self, item: IndexedFile) -> set[str]:
        tokens: set[str] = set()
        path = item.path.replace("\\", "/")
        for part in re.split(r"[/_.\W]+", path):
            if len(part) >= 2:
                tokens.add(part.lower())
        for name in (
            item.classes
            + item.functions
            + item.methods
            + item.imports
            + item.call_names
            + item.attribute_names
        ):
            lowered = str(name).lower()
            if lowered:
                tokens.add(lowered)
            tokens.update(self._split_identifier_tokens(str(name)))
            if "." in str(name):
                tokens.update(
                    part.lower()
                    for part in str(name).split(".")
                    if len(part) >= 2
                )
        return {t for t in tokens if len(t) >= 2}

    def _build_weighted_issue_query(
        self,
        clue: Dict[str, Any],
        clue_functions: set[str],
        clue_classes: set[str],
        clue_files: set[str],
        salient_tokens: List[Tuple[str, int]],
        issue_call_names: set[str],
        issue_import_modules: set[str],
    ) -> Counter[str]:
        query: Counter[str] = Counter()

        def add_token(token: str, weight: int) -> None:
            token = (token or "").lower().strip()
            if len(token) >= 2:
                query[token] += weight

        def add_identifier(value: str, weight: int) -> None:
            value = str(value or "")
            add_token(value, weight)
            for token in self._split_identifier_tokens(value):
                add_token(token, max(1, weight - 1))

        for fn in clue_functions:
            add_identifier(fn, 6)
        for cls in clue_classes:
            add_identifier(cls, 5)
        for path in clue_files:
            add_identifier(Path(path).stem, 7)
            for part in Path(path.replace("\\", "/")).parts:
                add_identifier(part.replace(".py", ""), 4)
        for call_name in issue_call_names:
            add_identifier(call_name, 6)
            add_identifier(call_name.split(".")[-1], 6)
        for module in issue_import_modules:
            add_identifier(module, 5)
            add_identifier(module.split(".")[-1], 5)
        for token, weight in salient_tokens:
            add_token(token, weight)
        for value in (
            clue.get("expected_outputs", [])
            + clue.get("actual_outputs", [])
            + clue.get("error_keywords", [])
        ):
            for token in self._split_salient_text(str(value).lower()):
                add_token(token, 4)
        return query

    @staticmethod
    def _score_weighted_issue_overlap(
        query: Counter[str],
        doc_tokens: set[str],
        doc_freq: Counter[str],
        doc_count: int,
    ) -> Tuple[List[str], float]:
        if not query or not doc_tokens:
            return [], 0.0
        hits: List[Tuple[str, float]] = []
        for token, weight in query.items():
            if token not in doc_tokens:
                continue
            df = max(doc_freq.get(token, 0), 1)
            idf = math.log((doc_count + 1) / df)
            hits.append((token, float(weight) * (1.0 + idf)))
        hits.sort(key=lambda x: (-x[1], x[0]))
        score = sum(value for _token, value in hits[:8]) / 4.0
        return [token for token, _value in hits[:8]], score

    def _rank_files(
        self,
        indexed_files: List[IndexedFile],
        clue: Dict[str, Any],
        repo_path: Path,
        *,
        use_formula_signals: bool = False,
        restart_feedback: Mapping[str, Any] | None = None,
    ) -> Tuple[List[CandidateFile], List[CandidateFile]]:
        noisy_functions = {
            "arange", "rand", "random", "seed", "platform", "get_backend",
            "show_versions",
        }
        clue_functions = {
            fn for fn in clue.get("identifiers", {}).get("functions", [])
            if fn not in noisy_functions
        }
        clue_classes = set(clue.get("identifiers", {}).get("classes", []))
        clue_files = {
            self._normalize_issue_file_hint(f)
            for f in clue.get("identifiers", {}).get("files", [])
            if isinstance(f, str)
        }
        clue_files = {f for f in clue_files if f}

        # code_examples(이슈 코드 블록)에서 추가 함수 호출 추출 — public API 경유 내부 구현 탐색
        _stopwords = frozenset({
            "if", "for", "while", "def", "class", "return", "import",
            "from", "with", "as", "in", "not", "and", "or", "is", "True", "False",
            "None", "print", "len", "range", "type", "str", "int", "list", "dict",
            "set", "tuple", "super", "self", "cls",
            "arange", "rand", "random", "seed", "platform", "get_backend",
            "show_versions",
        })
        for block in clue.get("code_examples", []):
            if block.get("is_system_or_output"):
                continue
            code = block.get("code", "") or ""
            for m in re.finditer(r"\b([a-z_][a-zA-Z0-9_]{2,})\s*\(", code):
                fn = m.group(1)
                if fn not in _stopwords:
                    clue_functions.add(fn)
        issue_call_names, issue_import_modules = self._extract_code_example_symbols(clue)
        for call_name in issue_call_names:
            bare = call_name.split(".")[-1]
            if bare and bare not in _stopwords:
                clue_functions.add(bare)

        # traceback에서 추출한 fault location 후보 — suffix 매칭으로 파일 식별
        fault_locations: List[Dict[str, Any]] = clue.get("fault_locations", [])
        # {rel_path → fault_location} 매핑 (suffix 매칭 후 채워짐)
        fault_file_info: Dict[str, Dict[str, Any]] = {}
        for fl in fault_locations:
            fl_path = fl.get("file_path", "").replace("\\", "/")
            for ifile in indexed_files:
                if not ifile.is_test_file and fl_path.endswith(ifile.path.replace("\\", "/")):
                    fault_file_info[ifile.path] = fl
                    break

        observed = " ".join(clue.get("observed_behavior", []))
        expected = " ".join(clue.get("expected_behavior", []))
        repro = " ".join(clue.get("repro_conditions", []))
        raw_issue_text = clue.get("raw_issue_text", "")
        restart_diagnosis_text = " ".join(
            str(restart_feedback.get(key) or "")
            for key in (
                "failure_reason",
                "assumption_gap",
                "next_scenario_change",
                "admissible_alternatives",
            )
        ) if isinstance(restart_feedback, Mapping) else ""
        clue_text = (
            f"{observed} {expected} {repro} {raw_issue_text} {restart_diagnosis_text}"
        ).lower()
        salient_tokens = self._extract_salient_issue_tokens(clue, clue_text)
        source_doc_tokens = {
            item.path: self._source_document_tokens(item)
            for item in indexed_files
            if not item.is_test_file
        }
        source_doc_freq: Counter[str] = Counter()
        for tokens in source_doc_tokens.values():
            source_doc_freq.update(tokens)
        weighted_query = self._build_weighted_issue_query(
            clue=clue,
            clue_functions=clue_functions,
            clue_classes=clue_classes,
            clue_files=clue_files,
            salient_tokens=salient_tokens,
            issue_call_names=issue_call_names,
            issue_import_modules=issue_import_modules,
        )

        source_candidates: List[CandidateFile] = []
        test_candidates: List[CandidateFile] = []

        # 1단계: source 파일 우선 점수화
        for item in indexed_files:
            if item.is_test_file:
                continue

            score = 0
            matched_identifiers: set[str] = set()
            reasons: List[str] = []

            file_name = Path(item.path).name.lower()
            path_lower = item.path.lower()
            stem = Path(item.path).stem.lower()
            file_funcs = set(item.functions) | set(item.methods)

            for fn in clue_functions:
                if fn in file_funcs:
                    score += 5
                    matched_identifiers.add(fn)
                    reasons.append(f"function_match:{fn}")

            for cls in clue_classes:
                # ALL_CAPS identifiers are likely constants, not classes
                if cls.isupper():
                    continue
                if cls in item.classes:
                    score += 4
                    matched_identifiers.add(cls)
                    reasons.append(f"class_match:{cls}")

            for clue_file in clue_files:
                clue_file_name = Path(clue_file).name.lower()
                clue_path = clue_file.lower().lstrip("./")
                if clue_path and self._path_suffix_matches(path_lower, clue_path):
                    score += 20
                    matched_identifiers.add(clue_file)
                    reasons.append(f"explicit_file_path_hint:{clue_file}")
                elif clue_file_name and clue_file_name == file_name:
                    score += 10
                    matched_identifiers.add(clue_file)
                    reasons.append(f"file_hint_match:{clue_file}")

            for fn in clue_functions:
                fn_lower = fn.lower()
                if fn_lower in path_lower:
                    score += 2
                    reasons.append(f"path_contains_function:{fn}")

            for cls in clue_classes:
                cls_lower = cls.lower()
                if cls_lower in path_lower:
                    score += 2
                    reasons.append(f"path_contains_class:{cls}")

            if stem and stem in clue_text and len(stem) >= 4:
                score += 1
                reasons.append(f"stem_in_issue_text:{stem}")

            doc_tokens = source_doc_tokens.get(item.path, set())
            ir_hits, ir_score = self._score_weighted_issue_overlap(
                weighted_query,
                doc_tokens,
                source_doc_freq,
                max(len(source_doc_tokens), 1),
            )
            if ir_hits:
                bonus = min(int(round(ir_score)), 18)
                if bonus > 0:
                    score += bonus
                    matched_identifiers.update(ir_hits[:5])
                    reasons.append(f"weighted_ir_overlap:{','.join(ir_hits[:6])}")

            for module in issue_import_modules:
                module_path = module.replace(".", "/").lower()
                if module_path and (
                    path_lower.endswith(module_path + ".py")
                    or path_lower.endswith(module_path + "/__init__.py")
                    or module_path in path_lower
                ):
                    score += 10
                    matched_identifiers.add(module)
                    reasons.append(f"issue_code_import_module:{module}")

            issue_call_bares = {name.split(".")[-1] for name in issue_call_names}
            call_overlap = sorted(
                name for name in issue_call_bares
                if name and (name in file_funcs or name in item.call_names or name in item.attribute_names)
            )
            if call_overlap:
                score += min(3 * len(call_overlap), 12)
                matched_identifiers.update(call_overlap[:4])
                reasons.append(f"issue_code_call_overlap:{','.join(call_overlap[:5])}")

            content_hits = self._score_source_content_tokens(
                repo_path / item.path,
                salient_tokens,
            )
            if content_hits:
                hit_tokens = [tok for tok, _weight in content_hits]
                bonus = min(sum(weight for _tok, weight in content_hits), 16)
                score += bonus
                matched_identifiers.update(hit_tokens[:4])
                reasons.append(f"source_content_tokens:{','.join(hit_tokens[:5])}")

            # High-confidence traceback gets a strong boost; LLM-inferred
            # locations get only a weak hint so they do not dominate ranking.
            if item.path in fault_file_info:
                fault = fault_file_info[item.path]
                fault_fn = fault.get("function_name", "")
                source = fault.get("source", "traceback")
                confidence = fault.get("confidence", "high" if source == "traceback" else "medium")
                if source == "traceback" and confidence == "high":
                    score += 12
                    reasons.append(f"traceback_fault_location:{fault_fn}")
                else:
                    score += 3
                    reasons.append(f"inferred_fault_location:{fault_fn}")
                if fault_fn:
                    matched_identifiers.add(fault_fn)

            if score <= 0:
                continue

            source_candidates.append(
                CandidateFile(
                    path=item.path,
                    score=score,
                    matched_identifiers=sorted(matched_identifiers),
                    reasons=reasons[:10],
                )
            )

        source_candidates.sort(key=lambda x: (-x.score, x.path))
        seed_candidates = source_candidates[: self.top_k_source]
        if use_formula_signals:
            source_candidates = self._expand_one_hop_source_candidates(
                seed_candidates,
                indexed_files,
                repo_path=repo_path,
                max_new=max(self.top_k_source, 1),
            )
        else:
            source_candidates = self._expand_legacy_one_hop_source_candidates(
                seed_candidates,
                indexed_files,
                max_new=max(self.top_k_source, 1),
            )
        source_candidates, self._last_restart_constraints = (
            self._apply_restart_constraints(source_candidates, restart_feedback)
        )
        self._attach_api_function_evidence(source_candidates, indexed_files, clue)
        self._assign_deterministic_text_similarity(
            source_candidates,
            clue=clue,
            repo_path=repo_path,
        )

        # top-K 소스 파일의 matched identifier 스니펫 + 함수 목록 추출
        for candidate in source_candidates:
            file_path = repo_path / candidate.path
            candidate.code_snippets = self._extract_code_snippets(
                file_path, candidate.matched_identifiers
            )
            candidate.top_level_functions = self._extract_top_level_functions(file_path)

        # source 후보 기반 힌트 생성
        source_paths = [Path(x.path) for x in source_candidates]
        source_dirs = {str(p.parent).replace("\\", "/").lower() for p in source_paths}
        source_stems = {p.stem.lower() for p in source_paths}

        # source directory → expected test directory 매핑 (추론)
        inferred_test_dirs: set[str] = set()
        for src_dir in source_dirs:
            parts = src_dir.split("/")
            # e.g. django/db/backends/postgresql → tests/backends/postgresql
            if len(parts) >= 2:
                inferred_test_dirs.add(f"tests/{'/'.join(parts[1:])}")
            # e.g. django/db/backends → tests/db_backends (flat variant)
            inferred_test_dirs.add(f"tests/{'_'.join(parts)}")
            # e.g. astropy/modeling → astropy/modeling/tests
            inferred_test_dirs.add(f"{src_dir}/tests")

        # issue/clue에서 test file retrieval용 토큰 추출
        clue_tokens: set[str] = set()

        for fn in clue_functions:
            clue_tokens.update(self._split_identifier_tokens(fn))

        for cls in clue_classes:
            clue_tokens.update(self._split_identifier_tokens(cls))

        for p in source_paths:
            clue_tokens.update(self._split_identifier_tokens(p.stem))
            for part in p.parts:
                clue_tokens.update(self._split_identifier_tokens(part))

        # 너무 일반적인 토큰 제거
        weak_tokens = {
            "test", "tests", "py", "model", "models", "file", "files",
            "class", "classes", "function", "functions", "astropy",
            # Framework / project names
            "django", "flask", "numpy", "scipy", "matplotlib",
            # Common generic module names
            "utils", "helpers", "base", "core", "common", "generic",
            "mixins", "compat", "conf", "config", "settings", "views",
            "urls", "admin", "apps", "managers", "signals", "middleware",
            "serializers", "validators", "decorators", "exceptions",
        }
        clue_tokens = {t for t in clue_tokens if len(t) >= 4 and t not in weak_tokens}

        # 2단계: test 파일 점수화 (강화 버전)
        for item in indexed_files:
            if not item.is_test_file:
                continue
            if item.has_module_skip:
                continue

            # Empty package initializers are not executable test placement
            # evidence.  Ranking them above a real test module can trigger
            # framework initialization errors (notably Django app loading).
            if Path(item.path).name == "__init__.py":
                try:
                    if not (repo_path / item.path).read_text(
                        encoding="utf-8", errors="ignore"
                    ).strip():
                        continue
                except OSError:
                    continue

            score = 0
            matched_identifiers: set[str] = set()
            reasons: List[str] = []

            test_path = item.path.replace("\\", "/")
            test_path_lower = test_path.lower()
            test_name = Path(test_path).name.lower()
            test_stem = Path(test_path).stem.lower()

            file_funcs = set(item.functions) | set(item.methods)

            # A. 직접 식별자 매칭
            for fn in clue_functions:
                if fn in file_funcs:
                    score += 5
                    matched_identifiers.add(fn)
                    reasons.append(f"test_function_match:{fn}")

                fn_lower = fn.lower()
                if fn_lower in test_path_lower:
                    score += 4
                    matched_identifiers.add(fn)
                    reasons.append(f"test_path_contains_function:{fn}")

            for cls in clue_classes:
                if cls in item.classes:
                    score += 4
                    matched_identifiers.add(cls)
                    reasons.append(f"test_class_match:{cls}")

                cls_tokens = self._split_identifier_tokens(cls)
                if any(tok in test_path_lower for tok in cls_tokens if len(tok) >= 4):
                    score += 3
                    matched_identifiers.add(cls)
                    reasons.append(f"test_path_related_to_class:{cls}")

            # B. source 후보와 같은 디렉토리 계열이면 강한 보너스
            for src_dir in source_dirs:
                src_dir_parts = src_dir.split("/")
                if len(src_dir_parts) >= 2:
                    anchor = "/".join(src_dir_parts[:-1])  # e.g. astropy/modeling
                    if anchor and anchor in test_path_lower:
                        score += 5
                        reasons.append(f"same_module_area:{anchor}")

            # B2. 추론된 테스트 디렉토리 매칭
            for inferred_dir in inferred_test_dirs:
                if test_path_lower.startswith(inferred_dir + "/") or inferred_dir + "/" in test_path_lower:
                    score += 4
                    reasons.append(f"inferred_test_dir_match:{inferred_dir}")
                    break

            # C. source 파일 stem과 test 파일 이름 유사
            test_stem_tokens = set(re.split(r"[_\W]+", test_stem))
            for src_stem in source_stems:
                if src_stem and (src_stem == test_stem or src_stem in test_stem_tokens):
                    score += 6
                    reasons.append(f"test_name_matches_source_stem:{src_stem}")

            # D. token overlap 기반 점수
            overlap_tokens = [tok for tok in clue_tokens if tok in test_path_lower]
            if overlap_tokens:
                bonus = min(len(overlap_tokens) * 2, 8)
                score += bonus
                reasons.append(f"token_overlap:{','.join(sorted(overlap_tokens)[:5])}")

            # E. 테스트 파일 내부 함수/클래스명도 활용
            internal_names = (
                [x.lower() for x in item.functions]
                + [x.lower() for x in item.methods]
                + [x.lower() for x in item.classes]
            )

            for tok in clue_tokens:
                if any(tok in name for name in internal_names):
                    score += 2
                    reasons.append(f"internal_name_overlap:{tok}")
                    break

            # F. 일반적인 test 파일 보너스는 아주 약하게만
            if item.test_functions:
                score += 1
                reasons.append("has_test_functions")

            # G. module-level skip 페널티 (importorskip 등)
            if item.has_module_skip:
                score -= 10
                reasons.append("module_level_skip_penalty")

            # H. Strongly prefer tests from the same repo area when source is explicit.
            if source_candidates:
                best_source = source_candidates[0].path.replace("\\", "/")
                best_parts = best_source.split("/")
                if len(best_parts) >= 2:
                    source_area = "/".join(best_parts[:2]).lower()
                    if source_area in test_path_lower:
                        score += 6
                        reasons.append(f"explicit_source_area_match:{source_area}")
                    elif not any(tok in test_path_lower for tok in source_area.split("/") if len(tok) >= 4):
                        score -= 3
                        reasons.append(f"unrelated_to_explicit_source_area:{source_area}")

            if score <= 0:
                continue

            test_candidates.append(
                CandidateFile(
                    path=item.path,
                    score=score,
                    matched_identifiers=sorted(matched_identifiers),
                    reasons=reasons[:12],
                    has_module_skip=item.has_module_skip,
                )
            )

        test_candidates.sort(key=lambda x: (-x.score, x.path))
        test_candidates = test_candidates[: self.top_k_test]

        return source_candidates, test_candidates

    def _attach_api_function_evidence(
        self,
        candidates: List[CandidateFile],
        indexed_files: List[IndexedFile],
        clue: Mapping[str, Any],
    ) -> None:
        """Keep issue APIs separate from repository-grounded implementation choices."""
        identifiers = clue.get("identifiers") if isinstance(clue.get("identifiers"), Mapping) else {}
        issue_functions = {
            str(value).split(".")[-1]
            for value in identifiers.get("functions", []) or []
            if str(value).strip()
        }
        public_apis = [
            str(value)
            for value in identifiers.get("dotted_apis", []) or []
            if str(value).strip()
        ]
        indexed_by_path = {item.path: item for item in indexed_files}
        all_alternatives: List[Dict[str, Any]] = []
        for candidate in candidates:
            indexed = indexed_by_path.get(candidate.path)
            if indexed is None:
                continue
            definitions = sorted(
                issue_functions & (set(indexed.functions) | set(indexed.methods))
            )
            references = sorted(
                issue_functions & (set(indexed.call_names) | set(indexed.attribute_names))
            )
            plausible_names = definitions + [
                name
                for name in [*indexed.functions, *indexed.methods]
                if name not in definitions
            ]
            alternatives = [
                {
                    "file_path": candidate.path,
                    "function_name": name,
                    "evidence": (
                        "exact_issue_definition"
                        if name in definitions
                        else "repository_definition"
                    ),
                    "public_api": next(
                        (api for api in public_apis if api.split(".")[-1] == name),
                        None,
                    ),
                }
                for name in plausible_names[:5]
            ]
            metadata = dict(candidate.metadata or {})
            metadata["symbol_evidence"] = {
                "exact_issue_definitions": definitions,
                "exact_issue_references": references,
                "public_issue_apis": public_apis,
                "target_resolution_status": (
                    "EXACT_ISSUE_DEFINITION"
                    if definitions
                    else "ISSUE_API_REFERENCE_ONLY"
                    if references
                    else "ALTERNATIVE_IMPLEMENTATION_CANDIDATE"
                ),
                "plausible_function_alternatives": alternatives,
            }
            candidate.metadata = metadata
            all_alternatives.extend(alternatives)
        self._last_api_function_evidence = {
            "public_issue_apis": public_apis,
            "issue_function_identifiers": sorted(issue_functions),
            "plausible_implementation_targets": all_alternatives[:15],
            "provenance": "pre_patch_ast_definition_reference",
        }

    @staticmethod
    def _apply_restart_constraints(
        candidates: List[CandidateFile],
        restart_feedback: Mapping[str, Any] | None,
    ) -> Tuple[List[CandidateFile], Dict[str, Any]]:
        """Exclude disproven targets and fail closed when reuse is prohibited."""
        raw_paths = (
            restart_feedback.get("prohibited_source_files", [])
            if isinstance(restart_feedback, Mapping)
            else []
        )
        prohibited = {
            str(path).replace("\\", "/").lstrip("./")
            for path in raw_paths
            if str(path).strip()
        }
        metadata: Dict[str, Any] = {
            "requested": bool(restart_feedback),
            "applied": False,
            "prohibited_source_files": sorted(prohibited),
            "excluded_source_files": [],
            "fallback_reason": None,
            "constraint_status": "NOT_APPLICABLE",
        }
        if not restart_feedback:
            metadata["fallback_reason"] = "initial_m2_execution"
            return list(candidates), metadata
        if not prohibited:
            metadata["fallback_reason"] = "no_prohibited_source_file"
            return list(candidates), metadata

        retained = [
            candidate
            for candidate in candidates
            if candidate.path.replace("\\", "/").lstrip("./") not in prohibited
        ]
        excluded = [candidate.path for candidate in candidates if candidate not in retained]
        if not excluded:
            metadata["fallback_reason"] = "prohibited_target_not_in_candidates"
            return list(candidates), metadata
        if not retained:
            metadata["fallback_reason"] = "no_grounded_alternative_candidate"
            metadata["constraint_status"] = "UNSATISFIED"
            metadata["target_reuse_allowed"] = False
            metadata["excluded_source_files"] = excluded
            return [], metadata

        metadata.update(
            {
                "applied": True,
                "excluded_source_files": excluded,
                "fallback_reason": None,
                "constraint_status": "SATISFIED",
            }
        )
        return retained, metadata

    @staticmethod
    def _apply_function_restart_constraints(
        ranking: M2RankingResult,
        restart_feedback: Mapping[str, Any] | None,
    ) -> Tuple[M2RankingResult, Dict[str, Any]]:
        """Remove only disproven target identities while retaining their files."""
        raw = (
            restart_feedback.get("prohibited_targets", [])
            if isinstance(restart_feedback, Mapping)
            else []
        )
        prohibited = {
            (
                str(item.get("source_file") or "").replace("\\", "/").lstrip("./"),
                str(item.get("target_function") or item.get("function_name") or "").split(".")[-1],
            )
            for item in raw
            if isinstance(item, Mapping)
            and str(item.get("source_file") or "").strip()
            and str(item.get("target_function") or item.get("function_name") or "").strip()
        }
        metadata: Dict[str, Any] = {
            "function_exclusion_requested": bool(prohibited),
            "prohibited_targets": [
                {"source_file": source, "target_function": function}
                for source, function in sorted(prohibited)
            ],
            "excluded_targets": [],
            "function_constraint_status": "NOT_APPLICABLE",
        }
        if not prohibited:
            return ranking, metadata

        def retained(entry: Mapping[str, Any]) -> bool:
            identity = (
                str(entry.get("source_file") or entry.get("file_path") or "")
                .replace("\\", "/")
                .lstrip("./"),
                str(entry.get("function_name") or entry.get("qualified_name") or "")
                .split(".")[-1],
            )
            if identity in prohibited:
                metadata["excluded_targets"].append(
                    {"source_file": identity[0], "target_function": identity[1]}
                )
                return False
            return True

        original_top5_count = len(ranking.top5_functions)
        filtered = [entry for entry in ranking.function_ranking if retained(entry)]
        if len(filtered) == len(ranking.function_ranking):
            metadata["function_constraint_status"] = "UNSATISFIED_NOT_IN_RANKING"
            return ranking, metadata
        initial_suspicious_functions = [
            entry for entry in ranking.initial_suspicious_functions if retained(entry)
        ]
        top5_functions = [entry for entry in ranking.top5_functions if retained(entry)]
        refill = [
            entry
            for entry in filtered
            if entry not in top5_functions
        ]
        desired = max(1, original_top5_count)
        top5_functions = (top5_functions + refill)[:desired]
        initial_suspicious_functions = list(top5_functions)
        metadata["excluded_targets"] = [
            dict(item)
            for item in {
                (entry["source_file"], entry["target_function"]): entry
                for entry in metadata["excluded_targets"]
            }.values()
        ]
        metadata["function_constraint_status"] = (
            "SATISFIED" if filtered else "UNSATISFIED_EXHAUSTED"
        )
        return replace(
            ranking,
            function_ranking=filtered,
            initial_suspicious_functions=initial_suspicious_functions,
            top5_functions=top5_functions,
        ), metadata

    def _expand_legacy_one_hop_source_candidates(
        self,
        candidates: List[CandidateFile],
        indexed_files: List[IndexedFile],
        *,
        max_new: int,
    ) -> List[CandidateFile]:
        """Preserve the pre-formula symbol-overlap expansion path."""
        if not candidates or max_new <= 0:
            return candidates
        by_path = {item.path: item for item in indexed_files if not item.is_test_file}
        existing_paths = {candidate.path for candidate in candidates}
        seed_paths = [candidate.path for candidate in candidates[:max_new]]
        seed_symbols: set[str] = set()
        seed_import_tails: set[str] = set()
        for path in seed_paths:
            item = by_path.get(path)
            if item is None:
                continue
            seed_symbols.update(item.classes)
            seed_symbols.update(item.functions)
            seed_symbols.update(item.methods)
            seed_symbols.update(item.call_names)
            seed_symbols.update(item.attribute_names)
            seed_import_tails.update(module.split(".")[-1] for module in item.imports)

        added: List[CandidateFile] = []
        for item in indexed_files:
            if item.is_test_file or item.path in existing_paths:
                continue
            own_symbols = set(item.classes + item.functions + item.methods)
            import_tails = {module.split(".")[-1] for module in item.imports}
            item_stem = Path(item.path).stem
            symbol_overlap = sorted((own_symbols | set(item.call_names)) & seed_symbols)
            import_overlap = sorted((import_tails | {item_stem}) & seed_import_tails)
            if not symbol_overlap and not import_overlap:
                continue
            reason_parts = []
            if symbol_overlap:
                reason_parts.append(f"one_hop_symbol_overlap:{','.join(symbol_overlap[:5])}")
            if import_overlap:
                reason_parts.append(f"one_hop_import_overlap:{','.join(import_overlap[:5])}")
            added.append(
                CandidateFile(
                    path=item.path,
                    score=1,
                    matched_identifiers=(symbol_overlap + import_overlap)[:8],
                    reasons=reason_parts,
                    metadata={"one_hop_expansion": True},
                )
            )
            if len(added) >= max_new:
                break
        return candidates + added

    def _expand_one_hop_source_candidates(
        self,
        candidates: List[CandidateFile],
        indexed_files: List[IndexedFile],
        *,
        repo_path: Path | None = None,
        max_new: int,
    ) -> List[CandidateFile]:
        """Add direct local import/call neighbors of seed source candidates.

        Expansion is intentionally one-hop: only imports and explicitly
        resolved imported callables from the original seed files are examined.
        Added files are not traversed.
        """
        if not candidates or max_new <= 0:
            return candidates
        for candidate in candidates:
            metadata = dict(candidate.metadata or {})
            selection_sources = set(metadata.get("selection_sources") or [])
            if self._has_identifier_selection_reason(candidate.reasons):
                selection_sources.add("identifier_match")
            metadata["selection_sources"] = sorted(selection_sources)
            candidate.metadata = metadata

        if repo_path is None:
            return candidates

        local_paths = {
            item.path.replace("\\", "/")
            for item in indexed_files
            if not item.is_test_file
        }
        existing_paths = {candidate.path.replace("\\", "/") for candidate in candidates}
        selected: Dict[str, Dict[str, Any]] = {}

        for candidate in candidates:
            seed_path = candidate.path.replace("\\", "/")
            direct_edges = self._resolve_direct_local_edges(repo_path, seed_path, local_paths)
            for rel_path, sources in direct_edges.items():
                if rel_path in existing_paths or rel_path == seed_path:
                    continue
                entry = selected.setdefault(
                    rel_path,
                    {"sources": set(), "matched": set(), "reasons": set()},
                )
                entry["sources"].update(sources)
                entry["matched"].update(sources)
                for source in sorted(sources):
                    entry["reasons"].add(f"{source}:{seed_path}")

        added: List[CandidateFile] = []
        for rel_path in sorted(selected):
            details = selected[rel_path]
            added.append(
                CandidateFile(
                    path=rel_path,
                    score=1,
                    matched_identifiers=sorted(details["matched"]),
                    reasons=sorted(details["reasons"]),
                    metadata={
                        "one_hop_expansion": True,
                        "selection_sources": sorted(details["sources"]),
                    },
                )
            )
            if len(added) >= max_new:
                break
        return candidates + added

    @staticmethod
    def _has_identifier_selection_reason(reasons: Sequence[str]) -> bool:
        prefixes = (
            "function_match:",
            "class_match:",
            "explicit_file_path_hint:",
            "file_hint_match:",
            "path_contains_function:",
            "path_contains_class:",
            "traceback_fault_location:",
            "inferred_fault_location:",
        )
        return any(str(reason).startswith(prefixes) for reason in reasons)

    def _resolve_direct_local_edges(
        self,
        repo_path: Path,
        seed_path: str,
        local_paths: set[str],
    ) -> Dict[str, set[str]]:
        path = repo_path / seed_path
        try:
            source = read_text(path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source)
        except Exception:
            return {}

        imports = self._resolved_local_imports(seed_path, tree, local_paths)
        edges: Dict[str, set[str]] = defaultdict(set)
        for resolved in imports:
            edges[resolved.rel_path].add("import_one_hop")

        import_by_local_name = {resolved.local_name: resolved for resolved in imports}
        module_imports = {
            resolved.local_name: resolved
            for resolved in imports
            if resolved.imported_name is None
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            resolved = self._resolve_explicit_call_import(node, import_by_local_name, module_imports)
            if resolved is not None:
                edges[resolved.rel_path].add("call_one_hop")
        return edges

    def _resolved_local_imports(
        self,
        seed_path: str,
        tree: ast.AST,
        local_paths: set[str],
    ) -> List[_ResolvedImport]:
        resolved: List[_ResolvedImport] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    rel_path = self._module_to_local_path(alias.name, local_paths)
                    if rel_path is None:
                        continue
                    local_name = alias.asname or alias.name.split(".")[0]
                    resolved.append(_ResolvedImport(local_name, alias.name, rel_path))
            elif isinstance(node, ast.ImportFrom):
                module = self._absolute_import_from_module(seed_path, node)
                if not module:
                    continue
                module_path = self._module_to_local_path(module, local_paths)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    local_name = alias.asname or alias.name
                    imported_module = f"{module}.{alias.name}" if module else alias.name
                    imported_module_path = self._module_to_local_path(imported_module, local_paths)
                    if imported_module_path is not None:
                        resolved.append(
                            _ResolvedImport(local_name, imported_module, imported_module_path)
                        )
                    elif module_path is not None:
                        resolved.append(
                            _ResolvedImport(local_name, module, module_path, imported_name=alias.name)
                        )
        return resolved

    @staticmethod
    def _absolute_import_from_module(seed_path: str, node: ast.ImportFrom) -> str:
        module = node.module or ""
        if node.level <= 0:
            return module
        parts = seed_path.replace("\\", "/").split("/")[:-1]
        if Path(seed_path).name == "__init__.py" and parts:
            parts = parts[:-1]
        prefix_count = max(len(parts) - node.level + 1, 0)
        prefix = ".".join(parts[:prefix_count])
        if prefix and module:
            return f"{prefix}.{module}"
        return prefix or module

    @staticmethod
    def _module_to_local_path(module: str, local_paths: set[str]) -> Optional[str]:
        if not module:
            return None
        module_path = module.replace(".", "/")
        file_path = f"{module_path}.py"
        package_path = f"{module_path}/__init__.py"
        if file_path in local_paths:
            return file_path
        if package_path in local_paths:
            return package_path
        return None

    @staticmethod
    def _resolve_explicit_call_import(
        call: ast.Call,
        import_by_local_name: Mapping[str, _ResolvedImport],
        module_imports: Mapping[str, _ResolvedImport],
    ) -> Optional[_ResolvedImport]:
        func = call.func
        if isinstance(func, ast.Name):
            resolved = import_by_local_name.get(func.id)
            if resolved is not None and resolved.imported_name is not None:
                return resolved
            return None
        if not isinstance(func, ast.Attribute):
            return None
        root = func
        while isinstance(root, ast.Attribute):
            parent = root.value
            if isinstance(parent, ast.Name):
                return module_imports.get(parent.id) or import_by_local_name.get(parent.id)
            root = parent
        return None

    def _assign_deterministic_text_similarity(
        self,
        candidates: List[CandidateFile],
        *,
        clue: Mapping[str, Any],
        repo_path: Path,
    ) -> None:
        if self.feature_profile == "v37":
            self._assign_v37_tfidf_cosine(candidates, clue=clue, repo_path=repo_path)
            self._assign_v37_codebert_method_cosines(
                candidates,
                clue=clue,
                repo_path=repo_path,
            )
            return
        if self.feature_profile == "v36":
            self._assign_v36_tfidf_cosine(candidates, clue=clue, repo_path=repo_path)
            return
        max_score = max((candidate.score for candidate in candidates), default=0)
        if max_score <= 0:
            return
        for candidate in candidates:
            candidate.deterministic_text_similarity = candidate.score / max_score

    @staticmethod
    def _assign_v37_tfidf_cosine(
        candidates: List[CandidateFile],
        *,
        clue: Mapping[str, Any],
        repo_path: Path,
    ) -> None:
        """Assign exact v37 file-level sklearn TF-IDF cosine similarities."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError as exc:
            raise RuntimeError(
                "V37 TF-IDF requires the approved scikit-learn runtime dependency"
            ) from exc
        query = str(clue.get("raw_issue_text") or "")
        documents: list[str] = []
        for candidate in candidates:
            try:
                documents.append(
                    (repo_path / candidate.path).read_text(
                        encoding="utf-8", errors="ignore"
                    )
                )
            except OSError:
                documents.append("")
        if not candidates:
            return
        matrix = TfidfVectorizer(stop_words="english").fit_transform(
            [query, *documents]
        )
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        for candidate, raw_similarity in zip(candidates, similarities):
            similarity = clamp_cosine_similarity(float(raw_similarity))
            candidate.deterministic_text_similarity = similarity
            if not isinstance(candidate.metadata, dict):
                candidate.metadata = {}
            candidate.metadata.setdefault("scores", {})["tfidf_similarity"] = similarity
            candidate.metadata.setdefault("evidence_availability", {})[
                "tfidf_similarity"
            ] = True
            candidate.metadata["tfidf_provenance"] = {
                "implementation": "sklearn.feature_extraction.text.TfidfVectorizer",
                "stop_words": "english",
                "embedding_unit": "complete_file_source",
                "query_source": "IssueClue.raw_issue_text",
                "similarity": "cosine_negative_clamped_to_zero",
            }

    def _assign_v37_codebert_method_cosines(
        self,
        candidates: List[CandidateFile],
        *,
        clue: Mapping[str, Any],
        repo_path: Path,
    ) -> None:
        """Measure v37 method similarities and aggregate each file by maximum."""
        encoder = self.codebert_encoder or get_v37_codebert_method_encoder()
        self.codebert_encoder = encoder
        try:
            query_embedding = encoder.encode(str(clue.get("raw_issue_text") or ""))
            query_embedding_error: str | None = None
        except (RuntimeError, TypeError, ValueError) as error:
            query_embedding = None
            query_embedding_error = f"{type(error).__name__}: {error}"
        for candidate in candidates:
            try:
                source = (repo_path / candidate.path).read_text(
                    encoding="utf-8", errors="ignore"
                )
            except OSError:
                source = ""
            method_scores: list[dict[str, Any]] = []
            for method in extract_python_method_sources(source):
                try:
                    similarity = (
                        encoder.cosine(
                            query_embedding,
                            encoder.encode(str(method["source"])),
                        )
                        if query_embedding is not None
                        else 0.0
                    )
                    embedding_status = (
                        "AVAILABLE" if query_embedding is not None else "UNAVAILABLE"
                    )
                except (RuntimeError, TypeError, ValueError) as error:
                    similarity = 0.0
                    embedding_status = f"UNAVAILABLE_{type(error).__name__}"
                method_scores.append(
                    {
                        "qualified_name": method["qualified_name"],
                        "start_line": method["start_line"],
                        "end_line": method["end_line"],
                        "similarity": similarity,
                        "embedding_status": embedding_status,
                    }
                )
            if not isinstance(candidate.metadata, dict):
                candidate.metadata = {}
            candidate.metadata["method_codebert_similarities"] = method_scores
            file_similarity = max(
                (float(item["similarity"]) for item in method_scores),
                default=0.0,
            )
            candidate.metadata.setdefault("scores", {})[
                "codebert_similarity"
            ] = file_similarity
            candidate.metadata["codebert_provenance"] = {
                "checkpoint": V37_CODEBERT_CHECKPOINT,
                "embedding_unit": "method_source",
                "representation": "last_hidden_state_attention_masked_mean_pooling",
                "embedding_dimension": V37_CODEBERT_EMBEDDING_DIM,
                "max_tokens": V37_CODEBERT_MAX_TOKENS,
                "truncation": "tail_truncation_prefix_retained",
                "similarity": "cosine_negative_clamped_to_zero",
                "file_aggregation": "max_method_similarity",
                "empty_or_invalid_method_set": 0.0,
                "query_embedding_error": query_embedding_error,
            }
            candidate.metadata.setdefault("evidence_availability", {})[
                "codebert_similarity"
            ] = True

    @staticmethod
    def _assign_v36_tfidf_cosine(
        candidates: List[CandidateFile],
        *,
        clue: Mapping[str, Any],
        repo_path: Path,
    ) -> None:
        """Assign genuine TF-IDF cosine similarity without proxy bonuses."""
        token_pattern = re.compile(r"[A-Za-z_]\w+")

        def tokens(text: str) -> List[str]:
            return [value.lower() for value in token_pattern.findall(text or "")]

        query_text = json.dumps(dict(clue), ensure_ascii=False, sort_keys=True)
        query_counts = Counter(tokens(query_text))
        document_counts: list[Counter[str]] = []
        for candidate in candidates:
            try:
                source = (repo_path / candidate.path).read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except OSError:
                source = ""
            document_counts.append(Counter(tokens(candidate.path + "\n" + source)))
        document_frequency: Counter[str] = Counter()
        for counts in document_counts:
            document_frequency.update(counts.keys())
        document_count = len(document_counts)

        def vector(counts: Counter[str]) -> Dict[str, float]:
            return {
                token: frequency
                * (math.log((document_count + 1) / (document_frequency.get(token, 0) + 1)) + 1.0)
                for token, frequency in counts.items()
            }

        query_vector = vector(query_counts)
        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        for candidate, counts in zip(candidates, document_counts):
            document_vector = vector(counts)
            document_norm = math.sqrt(sum(value * value for value in document_vector.values()))
            denominator = query_norm * document_norm
            similarity = (
                sum(
                    query_vector[token] * document_vector.get(token, 0.0)
                    for token in query_vector
                )
                / denominator
                if denominator
                else 0.0
            )
            candidate.deterministic_text_similarity = max(0.0, min(float(similarity), 1.0))
            if not isinstance(candidate.metadata, dict):
                candidate.metadata = {}
            candidate.metadata.setdefault("scores", {})["tfidf_similarity"] = (
                candidate.deterministic_text_similarity
            )
            candidate.metadata.setdefault("evidence_availability", {})[
                "tfidf_similarity"
            ] = True

    @staticmethod
    def _clamp_score(value: Any, *, name: str, diagnostics: List[str]) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            diagnostics.append(f"{name}: unavailable; using 0.0")
            return 0.0
        if not math.isfinite(numeric):
            diagnostics.append(f"{name}: non-finite; using 0.0")
            return 0.0
        if numeric < 0.0:
            diagnostics.append(f"{name}: clamped below range to 0.0")
            return 0.0
        if numeric > 1.0:
            diagnostics.append(f"{name}: clamped above range to 1.0")
            return 1.0
        return numeric

    @staticmethod
    def _normalize_inverse(
        value: Optional[int],
        max_value: int,
        *,
        metric_name: str,
        diagnostics: List[str],
    ) -> float:
        """Return 1 - value / max_value with deterministic zero-denominator handling.

        Inputs are raw non-negative counts. Output is clamped to [0, 1], where
        higher is better. If max_value is zero, no relative ordering is defined,
        so the normalized contribution is 0.0 and a diagnostic is recorded.
        """
        if value is None:
            diagnostics.append(f"{metric_name}: unavailable; using 0.0")
            return 0.0
        if max_value <= 0:
            diagnostics.append(f"{metric_name}: max denominator is 0; using 0.0")
            return 0.0
        return max(0.0, min(1.0, 1.0 - (float(value) / float(max_value))))

    @staticmethod
    def _normalize_ratio(
        value: Optional[int],
        max_value: int,
        *,
        metric_name: str,
        diagnostics: List[str],
    ) -> float:
        """Return value / max_value with deterministic zero-denominator handling.

        Inputs are raw non-negative counts. Output is clamped to [0, 1], where
        higher is better. If max_value is zero, no relative ordering is defined,
        so the normalized contribution is 0.0 and a diagnostic is recorded.
        """
        if value is None:
            diagnostics.append(f"{metric_name}: unavailable; using 0.0")
            return 0.0
        if max_value <= 0:
            diagnostics.append(f"{metric_name}: max denominator is 0; using 0.0")
            return 0.0
        return max(0.0, min(1.0, float(value) / float(max_value)))

    @staticmethod
    def calculate_r_init(
        *,
        rel_llm: float,
        sim_tfidf: float,
        sim_codebert: float,
        llm_enabled: bool,
        weights: M2RankingWeights | None = None,
    ) -> float:
        """Calculate file-level R_init for one candidate file.

        Inputs and output are in [0, 1], where higher is better. Components are
        clamped into range. When LLM matching is disabled, rel_llm and gamma are
        forced to 0.0. The task-defined formula is:

        R_init = gamma * rel_LLM + (1 - gamma) * (alpha * TF-IDF + beta * CodeBERT)

        Values below or above [0, 1] after formula application are clamped.
        """
        selected = weights or M2RankingWeights()
        if not llm_enabled:
            selected = M2RankingWeights(
                gamma=0.0,
                alpha=0.5,
                beta=0.5,
                delta0=selected.delta0,
                delta1=selected.delta1,
                delta2=selected.delta2,
                delta3=selected.delta3,
            )
            rel_llm = 0.0
        selected.validate()
        diagnostics: List[str] = []
        rel = CodeContextExtractor._clamp_score(rel_llm, name="rel_llm", diagnostics=diagnostics)
        tfidf = CodeContextExtractor._clamp_score(
            sim_tfidf, name="sim_tfidf", diagnostics=diagnostics
        )
        codebert = CodeContextExtractor._clamp_score(
            sim_codebert, name="sim_codebert", diagnostics=diagnostics
        )
        raw = selected.gamma * rel + (
            (1.0 - selected.gamma) * ((selected.alpha * tfidf) + (selected.beta * codebert))
        )
        return max(0.0, min(1.0, raw))

    @staticmethod
    def calculate_r_func(
        *,
        r_init: float,
        size_norm: float,
        churn_norm: float,
        age_norm: float,
        weights: M2RankingWeights | None = None,
    ) -> float:
        """Calculate function-level R_func for one function.

        Inputs and output are in [0, 1], where higher is better. Components are
        clamped into range. The task-defined formula is:

        R_func = R_init * (delta0 + delta1*size_norm + delta2*churn_norm + delta3*age_norm)
        """
        selected = weights or M2RankingWeights()
        selected.validate()
        diagnostics: List[str] = []
        base = CodeContextExtractor._clamp_score(r_init, name="r_init", diagnostics=diagnostics)
        size = CodeContextExtractor._clamp_score(size_norm, name="size_norm", diagnostics=diagnostics)
        churn = CodeContextExtractor._clamp_score(
            churn_norm, name="churn_norm", diagnostics=diagnostics
        )
        age = CodeContextExtractor._clamp_score(age_norm, name="age_norm", diagnostics=diagnostics)
        raw = base * (
            selected.delta0
            + (selected.delta1 * size)
            + (selected.delta2 * churn)
            + (selected.delta3 * age)
        )
        return max(0.0, min(1.0, raw))

    @staticmethod
    def _llm_relevance_norm(raw_score: Any, diagnostics: List[str]) -> float:
        try:
            score = int(raw_score)
        except (TypeError, ValueError):
            diagnostics.append("llm_relevance_raw: unavailable; using rel_llm=0.0")
            return 0.0
        if score < 1 or score > 5:
            diagnostics.append("llm_relevance_raw: outside 1..5; clamped before normalization")
        score = max(1, min(5, score))
        return (score - 1) / 4.0

    @staticmethod
    def _score_available(scores: Mapping[str, Any], key: str) -> bool:
        value = scores.get(key)
        if value is None:
            return False
        try:
            float(value)
        except (TypeError, ValueError):
            return False
        return True

    def _extract_function_metric_records(
        self,
        repo_path: Path,
        file_paths: List[str],
        *,
        history_window: int | None,
        diagnostics: Optional[List[str]] = None,
    ) -> List[FunctionMetricRecord]:
        records: List[FunctionMetricRecord] = []
        for rel_path in file_paths:
            path = repo_path / rel_path
            git_metrics = self._collect_git_file_metrics(
                repo_path,
                rel_path,
                history_window=history_window,
            )
            if diagnostics is not None:
                diagnostics.extend(git_metrics.diagnostics)
            try:
                source = read_text(path)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(source)
            except Exception as exc:
                if diagnostics is not None:
                    diagnostics.append(f"{rel_path}: ast_parse_failed; no functions extracted: {exc}")
                continue

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    records.append(
                        self._function_record_from_node(
                            rel_path,
                            node,
                            qualified_name=node.name,
                            git_metrics=git_metrics,
                        )
                    )
                elif isinstance(node, ast.ClassDef):
                    for item in ast.iter_child_nodes(node):
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            records.append(
                                self._function_record_from_node(
                                    rel_path,
                                    item,
                                    qualified_name=f"{node.name}.{item.name}",
                                    git_metrics=git_metrics,
                                )
                            )
        return records

    @staticmethod
    def _function_record_from_node(
        rel_path: str,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        qualified_name: str,
        git_metrics: _GitFileMetrics,
    ) -> FunctionMetricRecord:
        end_line = int(getattr(node, "end_lineno", node.lineno))
        size = max(0, end_line - int(node.lineno) + 1)
        provenance: Dict[str, Any] = {
            "size": "python_ast_function_span",
            **git_metrics.provenance,
        }
        return FunctionMetricRecord(
            file_path=rel_path,
            function_name=node.name,
            qualified_name=qualified_name,
            start_line=int(node.lineno),
            end_line=end_line,
            size=size,
            churn=git_metrics.churn,
            age=git_metrics.age,
            churn_scope="file_level_inherited_by_function"
            if git_metrics.git_history_available
            else "unavailable",
            git_history_available=git_metrics.git_history_available,
            provenance=provenance,
            unavailable_metric_diagnostics=git_metrics.diagnostics,
        )

    def _collect_git_file_metrics(
        self,
        repo_path: Path,
        rel_path: str,
        *,
        history_window: int | None,
    ) -> _GitFileMetrics:
        if history_window is None:
            return _GitFileMetrics(
                churn=None,
                age=None,
                git_history_available=False,
                provenance={
                    "churn": "unavailable_no_history_window",
                    "age": "unavailable_no_history_window",
                    "churn_scope": "unavailable",
                },
                diagnostics=[f"{rel_path}: churn/age unavailable; no configured history_window"],
            )
        if history_window <= 0:
            return _GitFileMetrics(
                churn=None,
                age=None,
                git_history_available=False,
                provenance={
                    "churn": "unavailable_invalid_history_window",
                    "age": "unavailable_invalid_history_window",
                    "history_window": history_window,
                    "churn_scope": "unavailable",
                },
                diagnostics=[f"{rel_path}: churn/age unavailable; history_window must be positive"],
            )
        if not (repo_path / ".git").exists():
            return _GitFileMetrics(
                churn=None,
                age=None,
                git_history_available=False,
                provenance={
                    "churn": "unavailable_no_git_repository",
                    "age": "unavailable_no_git_repository",
                    "history_window": history_window,
                    "churn_scope": "unavailable",
                },
                diagnostics=[f"{rel_path}: git history unavailable; no .git directory"],
            )

        if not self._git_command_succeeds(repo_path, ["rev-parse", "--is-inside-work-tree"]):
            return _GitFileMetrics(
                churn=None,
                age=None,
                git_history_available=False,
                provenance={
                    "churn": "unavailable_git_rev_parse_failed",
                    "age": "unavailable_git_rev_parse_failed",
                    "history_window": history_window,
                    "churn_scope": "unavailable",
                },
                diagnostics=[f"{rel_path}: git history unavailable; rev-parse failed"],
            )

        churn = self._git_file_churn(repo_path, rel_path, history_window)
        age = self._git_file_age(repo_path, rel_path)
        diagnostics: List[str] = []
        if churn is None:
            diagnostics.append(f"{rel_path}: churn unavailable from git log")
        if age is None:
            diagnostics.append(f"{rel_path}: age unavailable from git log")
        available = churn is not None and age is not None
        return _GitFileMetrics(
            churn=churn,
            age=age,
            git_history_available=available,
            provenance={
                "churn": "git_log_numstat_file_level" if churn is not None else "unavailable_git_log",
                "age": "git_rev_list_commits_since_file_touch" if age is not None else "unavailable_git_log",
                "history_window": history_window,
                "churn_scope": "file_level_inherited_by_function" if available else "unavailable",
            },
            diagnostics=diagnostics,
        )

    @staticmethod
    def _git_command_succeeds(repo_path: Path, args: List[str]) -> bool:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    @staticmethod
    def _git_file_churn(repo_path: Path, rel_path: str, history_window: int) -> Optional[int]:
        result = subprocess.run(
            [
                "git",
                "log",
                "--follow",
                f"-n{history_window}",
                "--numstat",
                "--format=",
                "--",
                rel_path,
            ],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        churn = 0
        saw_record = False
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            added, deleted = parts[0], parts[1]
            if added == "-" or deleted == "-":
                saw_record = True
                continue
            try:
                churn += int(added) + int(deleted)
                saw_record = True
            except ValueError:
                continue
        return churn if saw_record else None

    @staticmethod
    def _git_file_age(repo_path: Path, rel_path: str) -> Optional[int]:
        file_log = subprocess.run(
            ["git", "log", "--follow", "--format=%H", "--", rel_path],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
        )
        if file_log.returncode != 0:
            return None
        latest_touch = next((line.strip() for line in file_log.stdout.splitlines() if line.strip()), "")
        if not latest_touch:
            return None
        rev_list = subprocess.run(
            ["git", "rev-list", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
        )
        if rev_list.returncode != 0:
            return None
        commits = [line.strip() for line in rev_list.stdout.splitlines() if line.strip()]
        try:
            return commits.index(latest_touch)
        except ValueError:
            return None

    @staticmethod
    def _select_top_k(
        ranked_functions: List[Dict[str, Any]],
        *,
        k: int,
    ) -> List[Dict[str, Any]]:
        if k <= 0 or len(ranked_functions) <= k:
            return list(ranked_functions)
        return list(ranked_functions[:k])

    @staticmethod
    def _count_cutoff_ties_beyond_k(
        ranked_functions: List[Dict[str, Any]],
        *,
        k: int,
    ) -> int:
        if k <= 0 or len(ranked_functions) <= k:
            return 0
        cutoff = ranked_functions[k - 1]["R_func"]
        return sum(1 for entry in ranked_functions[k:] if entry["R_func"] == cutoff)

    def build_m2_ranking(
        self,
        candidates: List[CandidateFile],
        repo_path: Path,
        *,
        flags: V22FeatureFlags | Dict[str, bool] | None = None,
        weights: M2RankingWeights | None = None,
        function_metrics: List[FunctionMetricRecord] | None = None,
        history_window: int | None = None,
        k_init: int = 5,
        clue: Mapping[str, Any] | None = None,
    ) -> M2RankingResult:
        """Build deterministic v22 M2 file and function ranking artifacts.

        The engine uses only pre-patch candidate files and local AST data. It
        does not call an LLM or read golden/post-patch data. Zero denominators
        in function metric normalization yield 0.0 with diagnostics.
        """
        selected_weights = weights or self.ranking_weights
        selected_weights.validate()
        resolved_flags = _resolve_flags(flags)
        llm_enabled = bool(resolved_flags.enable_m2_llm_semantic_matching)
        if not llm_enabled:
            selected_weights = M2RankingWeights(
                gamma=0.0,
                alpha=0.5,
                beta=0.5,
                delta0=selected_weights.delta0,
                delta1=selected_weights.delta1,
                delta2=selected_weights.delta2,
                delta3=selected_weights.delta3,
            )
            selected_weights.validate()

        diagnostics: List[str] = []
        if history_window is None:
            diagnostics.append("history_window: unresolved default; churn and age unavailable")

        file_ranking: List[Dict[str, Any]] = []
        for candidate in candidates:
            metadata = candidate.metadata or {}
            scores = metadata.get("scores") if isinstance(metadata.get("scores"), Mapping) else {}
            file_diagnostics: List[str] = []
            tfidf_available = self._score_available(scores, "tfidf_similarity")
            tfidf = self._clamp_score(
                scores.get("tfidf_similarity", candidate.deterministic_text_similarity),
                name=f"{candidate.path}:tfidf_similarity",
                diagnostics=file_diagnostics,
            )
            codebert_available = self._score_available(scores, "codebert_similarity")
            if codebert_available:
                codebert = self._clamp_score(
                    scores.get("codebert_similarity"),
                    name=f"{candidate.path}:codebert_similarity",
                    diagnostics=file_diagnostics,
                )
                codebert_provenance = "upstream_adapter_supplied"
            else:
                if self.feature_profile in {"v36", "v37"}:
                    raise ValueError(
                        f"BLOCKING: {self.feature_profile} requires independently measured "
                        "CodeBERT similarity; no proxy is permitted"
                    )
                codebert = tfidf
                codebert_provenance = "fallback_to_tfidf_ir_only_not_measured"
                file_diagnostics.append(
                    f"{candidate.path}:codebert_similarity unavailable; using TF-IDF fallback"
                )
            llm_raw = scores.get("llm_relevance_raw", metadata.get("llm_relevance_raw"))
            llm_available = llm_enabled and llm_raw is not None
            rel_llm = self._llm_relevance_norm(llm_raw, file_diagnostics) if llm_enabled else 0.0
            r_init = self.calculate_r_init(
                rel_llm=rel_llm,
                sim_tfidf=tfidf,
                sim_codebert=codebert,
                llm_enabled=llm_enabled,
                weights=selected_weights,
            )
            entry = {
                "path": candidate.path,
                "file_path": candidate.path,
                "tfidf_similarity": tfidf,
                "codebert_similarity": codebert,
                "llm_relevance_raw": llm_raw if llm_enabled else None,
                "llm_relevance_norm": rel_llm if llm_enabled else None,
                "rel_llm": rel_llm,
                "r_init": r_init,
                "R_init": r_init,
                "legacy_rank_score": candidate.score,
                "matched_identifiers": list(candidate.matched_identifiers),
                "reasons": list(candidate.reasons),
                "component_scores": {
                    "rel_llm": rel_llm,
                    "sim_TF_IDF": tfidf,
                    "sim_CodeBERT": codebert,
                },
                "evidence_availability": {
                    "tfidf_similarity": tfidf_available
                    or candidate.deterministic_text_similarity is not None,
                    "codebert_similarity": codebert_available,
                    "llm_relevance": llm_available,
                },
                "R_init_input_provenance": {
                    "rel_LLM": "upstream_adapter_supplied" if llm_available else (
                        "disabled" if not llm_enabled else "unavailable"
                    ),
                    "sim_TF_IDF": "candidate_deterministic_text_similarity"
                    if not tfidf_available
                    else "upstream_adapter_supplied",
                    "sim_CodeBERT": codebert_provenance,
                    "formula": "approved_v22_R_init",
                },
                "unavailable_metric_diagnostics": file_diagnostics,
                "selection_sources": list(
                    (candidate.metadata or {}).get("selection_sources") or []
                ),
                "metadata": {
                    "fault_hypothesis": metadata.get("fault_hypothesis"),
                    "oracle_hint": metadata.get("oracle_hint"),
                },
            }
            file_ranking.append(entry)
            diagnostics.extend(file_diagnostics)

        file_ranking.sort(key=lambda entry: (-entry["R_init"], entry["file_path"]))
        file_by_path = {entry["file_path"]: entry for entry in file_ranking}
        metric_records = function_metrics
        if metric_records is None:
            metric_records = self._extract_function_metric_records(
                repo_path,
                [entry["file_path"] for entry in file_ranking],
                history_window=history_window,
                diagnostics=diagnostics,
            )

        max_size = max((record.size for record in metric_records), default=0)
        max_churn = max((record.churn or 0 for record in metric_records), default=0)
        max_age = max((record.age or 0 for record in metric_records), default=0)

        function_ranking: List[Dict[str, Any]] = []
        for record in metric_records:
            parent = file_by_path.get(record.file_path)
            if parent is None:
                continue
            function_diagnostics = list(record.unavailable_metric_diagnostics or [])
            size_norm = self._normalize_inverse(
                record.size,
                max_size,
                metric_name=f"{record.qualified_name}:size_norm",
                diagnostics=function_diagnostics,
            )
            churn_norm = self._normalize_ratio(
                record.churn,
                max_churn,
                metric_name=f"{record.qualified_name}:churn_norm",
                diagnostics=function_diagnostics,
            )
            age_norm = self._normalize_inverse(
                record.age,
                max_age,
                metric_name=f"{record.qualified_name}:age_norm",
                diagnostics=function_diagnostics,
            )
            r_func = self.calculate_r_func(
                r_init=parent["R_init"],
                size_norm=size_norm,
                churn_norm=churn_norm,
                age_norm=age_norm,
                weights=selected_weights,
            )
            entry = {
                "function_name": record.function_name,
                "qualified_name": record.qualified_name,
                "file_path": record.file_path,
                "source_file": record.file_path,
                "start_line": record.start_line,
                "end_line": record.end_line,
                "size": record.size,
                "churn": record.churn,
                "age": record.age,
                "size_norm": size_norm,
                "churn_norm": churn_norm,
                "age_norm": age_norm,
                "churn_scope": record.churn_scope,
                "git_history_available": record.git_history_available,
                "R_init": parent["R_init"],
                "r_func": r_func,
                "R_func": r_func,
                "metric_provenance": dict(record.provenance or {}),
                "R_init_input_provenance": dict(parent.get("R_init_input_provenance") or {}),
                "R_func_input_provenance": {
                    "size_norm": "python_ast_function_span",
                    "churn_norm": record.churn_scope,
                    "age_norm": "git_rev_list_commits_since_file_touch"
                    if record.git_history_available
                    else "unavailable",
                    "formula": "approved_v22_R_func",
                },
                "selection_sources": list(parent.get("selection_sources") or []),
                "component_evidence": {
                    "lexical": {
                        "tfidf_similarity": parent.get("tfidf_similarity"),
                        "reasons": list(parent.get("reasons") or []),
                    },
                    "semantic": {
                        "codebert_similarity": parent.get("codebert_similarity"),
                        "llm_relevance_norm": parent.get("llm_relevance_norm"),
                    },
                    "identifier": {
                        "matched_identifiers": list(parent.get("matched_identifiers") or []),
                        "function_name_match": any(
                            str(identifier).split(".")[-1] == record.function_name
                            for identifier in parent.get("matched_identifiers") or []
                        ),
                        "qualified_name_match": record.qualified_name in set(
                            map(str, parent.get("matched_identifiers") or [])
                        ),
                    },
                    "structural": {
                        "selection_sources": list(parent.get("selection_sources") or []),
                        "size_norm": size_norm,
                        "churn_norm": churn_norm,
                        "age_norm": age_norm,
                    },
                    "runtime": {
                        "available": False,
                        "reason": "initial_ranking_or_no_function_level_rerun_spectrum",
                    },
                    "hypothesis": {
                        "fault_hypothesis": (parent.get("metadata") or {}).get("fault_hypothesis")
                        if isinstance(parent.get("metadata"), Mapping)
                        else None,
                    },
                },
                "unavailable_metric_diagnostics": function_diagnostics,
            }
            function_ranking.append(entry)
            diagnostics.extend(function_diagnostics)

        function_ranking.sort(
            key=lambda entry: (
                -entry["R_func"],
                entry["file_path"].replace("\\", "/"),
                entry["qualified_name"],
            )
        )
        top_functions = self._select_top_k(function_ranking, k=k_init)
        cutoff_ties = self._count_cutoff_ties_beyond_k(function_ranking, k=k_init)
        if cutoff_ties:
            diagnostics.append(
                f"top{k_init}_functions: {cutoff_ties} tied candidates beyond cutoff omitted"
            )
        guidance_candidates = candidates
        if self.feature_profile in {"v36", "v37"}:
            by_path = {candidate.path: candidate for candidate in candidates}
            guidance_candidates = [
                by_path[str(entry.get("path") or entry.get("file_path") or "")]
                for entry in file_ranking
                if str(entry.get("path") or entry.get("file_path") or "") in by_path
            ]
        fault_hypothesis = None if not llm_enabled else self._first_string_metadata(
            guidance_candidates, "fault_hypothesis"
        )
        oracle_hint = None if not llm_enabled else self._first_string_metadata(
            guidance_candidates, "oracle_hint"
        )
        if fault_hypothesis is None and llm_enabled and clue:
            fault_hypothesis = self._deterministic_fault_hypothesis(
                top_functions=top_functions,
                file_ranking=file_ranking,
            )
            if fault_hypothesis:
                diagnostics.append(
                    "fault_hypothesis: deterministic pre-patch ranking fallback"
                )
        if oracle_hint is None and llm_enabled and clue:
            oracle_hint = self._deterministic_oracle_hint(clue or {})
            if oracle_hint:
                diagnostics.append("oracle_hint: deterministic M1 EB/S2R fallback")
        return M2RankingResult(
            candidate_files=list(file_ranking),
            file_ranking=file_ranking,
            function_ranking=function_ranking,
            initial_suspicious_functions=top_functions,
            top5_functions=top_functions,
            fault_hypothesis=fault_hypothesis,
            oracle_hint=oracle_hint,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _first_string_metadata(candidates: List[CandidateFile], key: str) -> Optional[str]:
        for candidate in candidates:
            metadata = candidate.metadata or {}
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None

    @staticmethod
    def _deterministic_fault_hypothesis(
        *,
        top_functions: Sequence[Mapping[str, Any]],
        file_ranking: Sequence[Mapping[str, Any]],
    ) -> Optional[str]:
        if top_functions:
            top = top_functions[0]
            file_path = str(top.get("file_path") or "").strip()
            function_name = str(
                top.get("qualified_name") or top.get("function_name") or ""
            ).strip()
            if file_path and function_name:
                return (
                    "Pre-patch identifier and lexical ranking localizes the issue to "
                    f"{function_name} in {file_path}; verify this function against the "
                    "issue reproduction stimulus."
                )
        if file_ranking:
            file_path = str(file_ranking[0].get("file_path") or "").strip()
            if file_path:
                return (
                    "Pre-patch identifier and lexical ranking localizes the issue to "
                    f"{file_path}; verify its issue-matched definitions against the "
                    "reproduction stimulus."
                )
        return None

    @staticmethod
    def _deterministic_oracle_hint(clue: Mapping[str, Any]) -> Optional[str]:
        expected_values = clue.get("expected_behavior") or clue.get("expected_outputs") or []
        expected = next(
            (str(value).strip() for value in expected_values if str(value).strip()),
            "",
        )
        conditions = clue.get("repro_conditions") or []
        stimulus = next(
            (str(value).strip() for value in conditions if str(value).strip()),
            "",
        )
        if not stimulus:
            for example in clue.get("code_examples") or []:
                if not isinstance(example, Mapping):
                    continue
                stimulus = str(
                    example.get("interactive_input") or example.get("code") or ""
                ).strip()
                if stimulus:
                    break
        if not expected:
            return None
        if stimulus:
            return (
                f"Exercise the issue reproduction stimulus ({stimulus}) and assert the "
                f"issue-reported expected behavior: {expected}"
            )
        return f"Assert the issue-reported expected behavior: {expected}"

    def _infer_test_style(
        self,
        indexed_files: List[IndexedFile],
        top_test_candidates: List[CandidateFile],
        repo_name: str = "",
    ) -> ProjectTestStyle:
        candidate_paths = {x.path for x in top_test_candidates}
        pool = [x for x in indexed_files if x.path in candidate_paths]

        if not pool:
            pool = [x for x in indexed_files if x.is_test_file][:20]

        pytest_score = 0
        unittest_score = 0
        evidence: List[str] = []
        assert_style: set[str] = set()
        has_django_test_import = False

        for item in pool:
            joined_imports = " ".join(item.imports)
            has_pytest_file_signal = (
                "pytest" in joined_imports
                or Path(item.path).name == "conftest.py"
                or any(name.startswith("test_") for name in item.functions)
            )

            if "pytest" in joined_imports or Path(item.path).name == "conftest.py":
                pytest_score += 2
                evidence.append(f"{item.path}:pytest_import_or_conftest")
                assert_style.add("plain assert")

            if "unittest" in joined_imports or "django.test" in joined_imports:
                unittest_score += 2
                evidence.append(f"{item.path}:unittest_or_django_test_import")
                assert_style.add("self.assert*")
                if "django.test" in joined_imports:
                    has_django_test_import = True

            if any(name.startswith("test_") for name in item.functions):
                pytest_score += 1
                evidence.append(f"{item.path}:top_level_test_functions")
                assert_style.add("plain assert")

            for cls in item.classes:
                # A class named ``Test*`` is common in pytest files and does
                # not establish unittest semantics by itself.  Count class
                # evidence only when the file has no stronger pytest signal;
                # explicit unittest/django imports above remain authoritative.
                if (
                    (cls.endswith("TestCase") or cls.startswith("Test"))
                    and not has_pytest_file_signal
                ):
                    unittest_score += 1
                    evidence.append(f"{item.path}:testcase_class:{cls}")
                    assert_style.add("self.assert*")

        if pytest_score > unittest_score:
            framework = "pytest"
        elif unittest_score > pytest_score:
            framework = "unittest"
        elif pytest_score == 0 and unittest_score == 0:
            framework = "unknown"
        else:
            framework = "mixed"

        if not assert_style:
            assert_style.add("unknown")

        # runner 결정: repo 이름 기반으로 감지 (파일 패턴은 오탐 가능성 높아 제외)
        repo_lower = repo_name.lower()

        if has_django_test_import or "django" in repo_lower:
            runner = "django-test"
        elif "sympy" in repo_lower:
            runner = "sympy-bin-test"
        elif framework == "unknown":
            runner = "unknown"
        elif framework in ("unittest",):
            runner = "unittest"
        else:
            # pytest, mixed → default pytest runner
            runner = "pytest"

        return ProjectTestStyle(
            framework=framework,
            evidence=evidence[:20],
            assert_style=sorted(assert_style),
            runner=runner,
        )
    
    def _extract_test_example(
        self,
        test_candidates: List[CandidateFile],
        repo_path: Path,
        max_lines: int = 35,
    ) -> str:
        """상위 테스트 파일에서 첫 번째 Test 클래스 또는 test_ 함수를 스니펫으로 추출한다."""
        for candidate in test_candidates[:3]:
            full_path = repo_path / candidate.path
            if not full_path.exists():
                continue
            try:
                src = full_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"[context] 파일 읽기 실패 {full_path}: {e}")
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            lines = src.splitlines()
            # Test 클래스 우선, 없으면 test_ 함수
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                    start = node.lineno - 1
                    return "\n".join(lines[start : start + max_lines])
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("test_"):
                        start = node.lineno - 1
                        return "\n".join(lines[start : start + max_lines])
        return ""

    def _extract_conftest_fixtures(
        self,
        test_candidates: List[CandidateFile],
        repo_path: Path,
    ) -> Dict[str, List[str]]:
        """candidate test file 경로 계층의 conftest.py에서 fixture 이름 수집."""
        result: Dict[str, List[str]] = {}
        seen: set[str] = set()
        for candidate in test_candidates[:3]:
            parts = Path(candidate.path).parts[:-1]  # 파일명 제외, 디렉토리만
            for i in range(len(parts), -1, -1):
                conftest_rel = str(Path(*parts[:i]) / "conftest.py") if i > 0 else "conftest.py"
                if conftest_rel in seen:
                    continue
                seen.add(conftest_rel)
                conftest_abs = repo_path / conftest_rel
                if conftest_abs.exists():
                    fixtures = self._parse_conftest_fixtures(conftest_abs)
                    if fixtures:
                        result[conftest_rel] = fixtures
        return result

    def _parse_conftest_fixtures(self, conftest_path: Path) -> List[str]:
        """conftest.py에서 @pytest.fixture 데코레이터가 붙은 함수 이름 추출."""
        try:
            source = conftest_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except Exception:
            return []
        fixtures: List[str] = []
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
                value_name = _get_ast_dotted_name(node.value)
                if value_name == "pytest.fixture":
                    fixture_aliases.update(
                        target.id for target in node.targets if isinstance(target, ast.Name)
                    )
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Call):
                        dec_name = _get_call_fullname(dec)
                    elif isinstance(dec, ast.Name):
                        dec_name = dec.id
                    elif isinstance(dec, ast.Attribute):
                        dec_name = dec.attr
                    else:
                        dec_name = ""
                    if dec_name in fixture_aliases or (
                        dec_name.endswith(".fixture")
                        and dec_name.split(".", 1)[0] in pytest_aliases
                    ):
                        fixtures.append(node.name)
                        break
        return fixtures

    def _collect_available_imports(
        self,
        indexed_files: List[IndexedFile],
        source_candidates: List[CandidateFile],
        test_candidates: List[CandidateFile],
        clue: Dict[str, Any],
        repo_path: Path,
    ) -> Dict[str, List[str]]:
        """
        candidate 파일들과 clue 식별자를 기반으로,
        각 모듈 경로에서 실제로 import 가능한 심볼 목록을 수집한다.
        예: {"astropy.modeling.models": ["Linear1D", "Gaussian1D"], ...}
        """
        result: Dict[str, List[str]] = {}

        # 관심 있는 파일 경로 수집 (source + test 후보)
        interest_paths: set[str] = set()
        for c in source_candidates:
            interest_paths.add(c.path)
        for c in test_candidates:
            interest_paths.add(c.path)

        # 각 candidate 파일의 조상 패키지 __init__.py도 포함
        # 예: django/db/models/expressions.py → django/__init__.py, django/db/__init__.py,
        #     django/db/models/__init__.py
        indexed_paths = {item.path for item in indexed_files}
        ancestor_inits: set[str] = set()
        for p in list(interest_paths):
            parts = p.replace("\\", "/").split("/")
            for depth in range(1, len(parts)):
                init_candidate = "/".join(parts[:depth]) + "/__init__.py"
                if init_candidate in indexed_paths:
                    ancestor_inits.add(init_candidate)
        interest_paths.update(ancestor_inits)

        # clue의 식별자가 정의된 모듈도 찾기
        clue_classes = set(clue.get("identifiers", {}).get("classes", []))
        clue_functions = set(clue.get("identifiers", {}).get("functions", []))
        all_clue_symbols = clue_classes | clue_functions

        for item in indexed_files:
            # candidate 파일이거나, clue 심볼을 정의하고 있는 파일
            has_clue_symbol = bool(
                (set(item.classes) | set(item.functions)) & all_clue_symbols
            )
            if item.path not in interest_paths and not has_clue_symbol:
                continue

            module_path = self._file_path_to_module(item.path)
            if not module_path:
                continue

            exported = sorted(set(item.classes + item.functions))
            if exported:
                result[module_path] = exported

            # __init__.py의 re-export도 수집
            if item.path.endswith("__init__.py"):
                init_file = repo_path / item.path
                reexports = self._collect_init_reexports(init_file)
                if reexports:
                    parent_module = module_path.rsplit(".", 1)[0] if "." in module_path else module_path
                    existing = set(result.get(parent_module, []))
                    existing.update(reexports)
                    result[parent_module] = sorted(existing)

        return result

    def _file_path_to_module(self, rel_path: str) -> str:
        """파일 경로를 Python 모듈 경로로 변환 (예: astropy/modeling/models.py -> astropy.modeling.models)"""
        p = rel_path.replace("\\", "/")
        if p.endswith("/__init__.py"):
            p = p[: -len("/__init__.py")]
        elif p.endswith(".py"):
            p = p[:-3]
        else:
            return ""
        return p.replace("/", ".")

    def _collect_init_reexports(self, init_path: Path) -> List[str]:
        """__init__.py에서 re-export되는 심볼 목록 수집"""
        if not init_path.exists():
            return []
        try:
            source = read_text(init_path)
        except Exception:
            return []
        try:
            tree = ast.parse(source)
        except Exception:
            return []

        symbols: set[str] = set()

        for node in tree.body:
            # __all__ 정의
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    symbols.add(elt.value)

            # from .X import Y 형태
            if isinstance(node, ast.ImportFrom):
                if node.names:
                    for alias in node.names:
                        if alias.name != "*":
                            symbols.add(alias.asname or alias.name)

        return sorted(symbols)

    def _split_identifier_tokens(self, name: str) -> set[str]:
        import re

        if not name:
            return set()

        s = name.replace("-", "_").replace("/", "_").replace(".", "_")
        parts = [p for p in s.split("_") if p]

        tokens: set[str] = set()
        for part in parts:
            camel = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", part)
            if camel:
                tokens.update(x.lower() for x in camel if x)
            else:
                tokens.add(part.lower())

        return {t for t in tokens if t}

    @staticmethod
    def _normalize_issue_file_hint(value: str) -> str:
        path = (value or "").replace("\\", "/").strip().strip("`'\"()[]{}.,;:")
        if "/blob/" in path:
            # Old clue artifacts may have captured part of a GitHub blob URL.
            tail = path.split("/blob/", 1)[1]
            parts = tail.split("/")
            if len(parts) >= 2:
                path = "/".join(parts[1:])
        path = re.sub(r"^(?:\./|a/|b/)+", "", path)
        if path.startswith(("http", "www.", "github.com/", "com/")):
            return ""
        return path

    @staticmethod
    def _path_suffix_matches(candidate_path: str, clue_path: str) -> bool:
        candidate_parts = [p for p in candidate_path.lower().split("/") if p]
        clue_parts = [p for p in clue_path.lower().split("/") if p]
        if not candidate_parts or not clue_parts:
            return False
        if len(clue_parts) == 1:
            return candidate_parts[-1] == clue_parts[0]
        if len(clue_parts) > len(candidate_parts):
            return clue_parts[-len(candidate_parts):] == candidate_parts
        return candidate_parts[-len(clue_parts):] == clue_parts

    def _extract_salient_issue_tokens(
        self,
        clue: Dict[str, Any],
        clue_text_lower: str,
    ) -> List[Tuple[str, int]]:
        """Tokens whose presence in source text is stronger than generic API names."""
        stop = {
            "about", "after", "again", "against", "already", "also", "because",
            "before", "being", "between", "class", "code", "correct", "current",
            "currently", "error", "expected", "false", "file", "files", "from",
            "function", "github", "have", "issue", "line", "model", "module",
            "none", "object", "output", "python", "return", "returns", "should",
            "test", "tests", "that", "this", "true", "using", "value", "when",
            "with", "without", "wrong",
            "sklearn", "pylint", "requests", "astropy", "seaborn", "matplotlib",
            "django", "numpy", "scipy",
        }
        high_value_phrases = [
            "content-length", "preparedrequest", "scalarformatter",
            "required columns", "required column", "pyreverse", "umls",
            "invalid value", "runtimewarning",
        ]
        tokens: Dict[str, int] = {}
        for phrase in high_value_phrases:
            if phrase in clue_text_lower:
                tokens[phrase] = 6

        for raw in re.findall(r"`([^`]{3,80})`|['\"]([^'\"]{3,80})['\"]", clue_text_lower):
            value = raw[0] or raw[1]
            for token in self._split_salient_text(value):
                if token not in stop:
                    tokens[token] = max(tokens.get(token, 0), 4)

        for value in (
            clue.get("expected_outputs", [])
            + clue.get("actual_outputs", [])
            + clue.get("error_keywords", [])
        ):
            for token in self._split_salient_text(str(value).lower()):
                if token not in stop:
                    tokens[token] = max(tokens.get(token, 0), 4)

        for token in self._split_salient_text(clue_text_lower):
            if token not in stop and len(token) >= 5:
                tokens[token] = max(tokens.get(token, 0), 2)

        return sorted(tokens.items(), key=lambda x: (-x[1], x[0]))[:40]

    def _split_salient_text(self, text: str) -> set[str]:
        parts = re.findall(r"[a-zA-Z_][a-zA-Z0-9_-]{2,}", text)
        result: set[str] = set()
        for part in parts:
            result.add(part.lower())
            if "-" in part or "_" in part:
                result.update(x for x in re.split(r"[-_]+", part.lower()) if len(x) >= 3)
        return result

    def _score_source_content_tokens(
        self,
        file_path: Path,
        salient_tokens: List[Tuple[str, int]],
    ) -> List[Tuple[str, int]]:
        if not salient_tokens or not file_path.exists():
            return []
        try:
            source = read_text(file_path)
        except Exception:
            return []
        source_lower = source[:200_000].lower()
        hits: List[Tuple[str, int]] = []
        for token, weight in salient_tokens:
            if not token or len(token) < 4:
                continue
            if token in source_lower:
                hits.append((token, weight))
        # One weak prose token is too noisy; a single high-value phrase is useful.
        if len(hits) == 1 and hits[0][1] < 4:
            return []
        return hits[:8]

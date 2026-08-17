from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.models.client import ModelConfig


PROVENANCE_SCHEMA_VERSION = "v27-run-provenance-v1"
RELEVANT_ROOTS = ("src", "scripts", "configs", "tests", "docs")


def build_run_provenance(
    *,
    repository_root: Path,
    feature_profile: str,
    explicit_feature_overrides: Mapping[str, bool],
    model_key: str,
    model_config: ModelConfig,
    workers: int,
    benchmark_path: Path,
    execution_command: Sequence[str],
    max_feedback_iterations: int = 5,
    requested_max_feedback_iterations: int | None = None,
    instance_view_root: str | Path | None = None,
    effective_feature_flags: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Describe the exact dirty source/config snapshot used by a v27 run.

    The record contains hashes and inventories, never file contents or model
    credentials.  It is intentionally independent from Git commit creation.
    """
    root = repository_root.resolve()
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    tracked_diff = _git(root, "diff", "--binary", "--no-ext-diff")
    relevant_files = list(_relevant_files(root))
    untracked = _untracked_relevant_files(root)
    return {
        "schema_version": (
            "v37-run-provenance-v1"
            if feature_profile == "v37"
            else "v36-run-provenance-v1"
            if feature_profile == "v36"
            else
            "v31-run-provenance-v1"
            if feature_profile == "v31"
            else
            "v30-run-provenance-v1"
            if feature_profile == "v30"
            else
            "v29-run-provenance-v1"
            if feature_profile == "v29"
            else PROVENANCE_SCHEMA_VERSION
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(root),
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        "head": _git(root, "rev-parse", "HEAD").strip(),
        "git_status": status.splitlines(),
        "git_status_sha256": _sha256_text(status),
        "tracked_diff_sha256": _sha256_text(tracked_diff),
        "untracked_relevant_files": untracked,
        "untracked_relevant_inventory_sha256": _inventory_hash(root, untracked),
        "source_config_content_sha256": _inventory_hash(root, relevant_files),
        "source_config_file_count": len(relevant_files),
        "feature_profile": feature_profile,
        "explicit_feature_overrides": dict(sorted(explicit_feature_overrides.items())),
        "effective_feature_flags": dict(sorted((effective_feature_flags or {}).items())),
        "model_key": model_key,
        "served_model_identifier": model_config.model_name,
        "base_url": model_config.base_url,
        "generation_parameters": {
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
            "timeout": model_config.timeout,
            "context_window": model_config.context_window,
            "context_safety_margin": model_config.context_safety_margin,
            "reasoning_effort": model_config.reasoning_effort,
            "output_token_parameter": (
                "max_completion_tokens"
                if model_config.reasoning_effort is not None
                else "max_tokens"
            ),
        },
        "worker_count": workers,
        "runtime_controls": {
            **(
                {
                    "feedback_iteration_budget": "PARAMETERIZED",
                    "requested_max_feedback_iterations": (
                        max_feedback_iterations
                        if requested_max_feedback_iterations is None
                        else requested_max_feedback_iterations
                    ),
                }
                if feature_profile == "v37"
                else {}
            ),
            "max_feedback_iterations": max_feedback_iterations,
            "instance_time_budget_sec": 120 if feature_profile == "v36" else None,
            "instance_view_root": str(Path(instance_view_root).resolve())
            if instance_view_root is not None
            else None,
        },
        "benchmark": {
            "path": str(benchmark_path.resolve()),
            "sha256": _sha256_file(benchmark_path),
        },
        "execution_command": sanitize_execution_command(execution_command),
    }


def sanitize_execution_command(command: Sequence[str]) -> list[str]:
    """Redact values of credential-looking CLI options."""
    sanitized: list[str] = []
    redact_next = False
    secret_names = ("api-key", "api_key", "token", "password", "secret")
    for raw in command:
        value = str(raw)
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        lowered = value.lower()
        if value.startswith("-") and any(name in lowered for name in secret_names):
            if "=" in value:
                sanitized.append(value.split("=", 1)[0] + "=<redacted>")
            else:
                sanitized.append(value)
                redact_next = True
            continue
        sanitized.append(value)
    return sanitized


def provenance_sha256(payload: Mapping[str, Any]) -> str:
    return _sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _relevant_files(root: Path) -> Iterable[str]:
    for relative_root in RELEVANT_ROOTS:
        directory = root / relative_root
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                yield path.relative_to(root).as_posix()


def _untracked_relevant_files(root: Path) -> list[str]:
    output = _git(root, "ls-files", "--others", "--exclude-standard", "--", *RELEVANT_ROOTS)
    return sorted(line for line in output.splitlines() if line)


def _inventory_hash(root: Path, relative_paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(set(relative_paths)):
        path = root / relative
        if not path.is_file():
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
    )

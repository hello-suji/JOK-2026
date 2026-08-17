from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


PROHIBITED_PRE_PATCH_KEYS = {
    "patch",
    "golden_patch",
    "patched_source",
    "post_patch_outcomes",
    "after_patch",
    "final_evaluation",
    "m8_results",
    "phr",
    "patch_hit_rate",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
}


@dataclass(frozen=True)
class PrePatchInstanceView:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    created_at: str = ""
    version: str = ""
    environment_setup_commit: str = ""
    difficulty: str = ""
    hints_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_tdd_image_raw(self) -> dict[str, Any]:
        return sanitize_raw_for_tdd_image_spec(self.to_dict())


@dataclass(frozen=True)
class M8EvaluationInstanceView:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str = ""
    test_patch: str = ""
    version: str = ""
    environment_setup_commit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_pre_patch_view(instance: Any) -> PrePatchInstanceView:
    raw = getattr(instance, "raw", None) or {}
    view = PrePatchInstanceView(
        instance_id=str(getattr(instance, "instance_id", raw.get("instance_id", ""))),
        repo=str(getattr(instance, "repo", raw.get("repo", ""))),
        base_commit=str(getattr(instance, "base_commit", raw.get("base_commit", ""))),
        problem_statement=str(
            getattr(instance, "problem_statement", raw.get("problem_statement", ""))
        ),
        created_at=str(getattr(instance, "created_at", raw.get("created_at", ""))),
        version=str(getattr(instance, "version", raw.get("version", ""))),
        environment_setup_commit=str(
            getattr(instance, "environment_setup_commit", raw.get("environment_setup_commit", ""))
        ),
        difficulty=str(getattr(instance, "difficulty", raw.get("difficulty", ""))),
        hints_text=str(raw.get("hints_text", getattr(instance, "hints_text", ""))),
    )
    assert_no_patch_access(view.to_dict())
    return view


def make_m8_evaluation_view(instance: Any) -> M8EvaluationInstanceView:
    return M8EvaluationInstanceView(
        instance_id=str(getattr(instance, "instance_id", "")),
        repo=str(getattr(instance, "repo", "")),
        base_commit=str(getattr(instance, "base_commit", "")),
        problem_statement=str(getattr(instance, "problem_statement", "")),
        patch=str(getattr(instance, "patch", "")),
        test_patch=str(getattr(instance, "test_patch", "")),
        version=str(getattr(instance, "version", "")),
        environment_setup_commit=str(getattr(instance, "environment_setup_commit", "")),
    )


def sanitize_raw_for_tdd_image_spec(raw: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = {
        "instance_id": str(raw.get("instance_id", "")),
        "repo": str(raw.get("repo", "")),
        "version": str(raw.get("version", "")),
        "base_commit": str(raw.get("base_commit", "")),
        "problem_statement": str(raw.get("problem_statement", "")),
        "hints_text": str(raw.get("hints_text", "")),
        "created_at": str(raw.get("created_at", "")),
        "environment_setup_commit": str(
            raw.get("environment_setup_commit", raw.get("base_commit", ""))
        ),
        "test_patch": "",
    }
    assert_no_patch_access({k: v for k, v in sanitized.items() if k != "test_patch"})
    return sanitized


def assert_no_patch_access(data: Mapping[str, Any]) -> None:
    present = sorted(key for key in PROHIBITED_PRE_PATCH_KEYS if key in data)
    if present:
        raise ValueError(f"pre-patch view contains prohibited keys: {', '.join(present)}")

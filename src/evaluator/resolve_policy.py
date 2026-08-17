"""Shared final-evaluation resolve policy."""
from __future__ import annotations

import re
from typing import Any, Dict


BAD_STATUSES = {"FAILED", "ERROR"}


def normalize_test_name(name: str) -> str:
    """Collapse harness-specific prefixes so the same generated test can match."""
    text = str(name or "").strip()
    if "::" in text:
        text = text.split("::")[-1]
    if ":" in text:
        text = text.split(":")[-1]
    paren = re.search(r"\(([^)]+)\)", text)
    if paren:
        left = text[: paren.start()].strip()
        text = left or text
    text = text.split(".")[-1]
    return re.sub(r"\W+", "_", text).strip("_").lower()


def is_generated_test_name(name: str) -> bool:
    return "_repro" in normalize_test_name(name)


def resolve_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    before_patch = result.get("before_patch", {}) or {}
    after_patch = result.get("after_patch", {}) or {}
    final_score = float(result.get("final_score", 0.0) or 0.0)
    harness_ran = bool(before_patch or after_patch)

    before_bad_exact = {
        t for t, status in before_patch.items()
        if status in BAD_STATUSES and is_generated_test_name(t)
    }
    after_pass_exact = {
        t for t, status in after_patch.items()
        if status == "PASSED" and is_generated_test_name(t)
    }
    after_bad_exact = {
        t for t, status in after_patch.items()
        if status in BAD_STATUSES and is_generated_test_name(t)
    }

    exact_flip = any(t in after_pass_exact for t in before_bad_exact)
    before_bad_norm = {normalize_test_name(t) for t in before_bad_exact}
    after_pass_norm = {normalize_test_name(t) for t in after_pass_exact}
    after_bad_norm = {normalize_test_name(t) for t in after_bad_exact}
    normalized_flip = bool(before_bad_norm & after_pass_norm)
    failure_disappeared = bool(
        before_bad_norm
        and after_pass_norm
        and not (before_bad_norm & after_bad_norm)
    )

    resolved = harness_ran and (
        final_score > 0
        or exact_flip
        or normalized_flip
        or failure_disappeared
    )
    return {"resolved": resolved}

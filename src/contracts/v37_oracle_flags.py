"""Closed deterministic Conservative-Gate flag contract for v37."""

from __future__ import annotations

from collections.abc import Iterable


ORACLE_EXPECTED_MISSING = "ORACLE_EXPECTED_MISSING"
ORACLE_SPEC_MISMATCH = "ORACLE_SPEC_MISMATCH"
ORACLE_ASSERTION_MISSING = "ORACLE_ASSERTION_MISSING"
ORACLE_SEMANTICS_CHANGED_BY_REPAIR = "ORACLE_SEMANTICS_CHANGED_BY_REPAIR"

V37_BLOCKING_ORACLE_FLAGS = frozenset(
    {
        ORACLE_EXPECTED_MISSING,
        ORACLE_SPEC_MISMATCH,
        ORACLE_ASSERTION_MISSING,
        ORACLE_SEMANTICS_CHANGED_BY_REPAIR,
    }
)


def validated_v37_blocking_oracle_flags(values: Iterable[object]) -> list[str]:
    """Return the sorted, deduplicated closed-taxonomy subset of ``values``."""
    return sorted(
        {
            str(value)
            for value in values
            if str(value) in V37_BLOCKING_ORACLE_FLAGS
        }
    )

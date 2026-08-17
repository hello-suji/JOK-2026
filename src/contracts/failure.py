from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional


class FailureCategory(str, Enum):
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
    PIPELINE_FAILURE = "PIPELINE_FAILURE"
    GENERATION_FAILURE = "GENERATION_FAILURE"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    EVALUATION_FAILURE = "EVALUATION_FAILURE"


@dataclass
class FailureRecord:
    category: FailureCategory
    stage: str
    command: Optional[list[str]] = None
    exit_code: Optional[int] = None
    error_message: str = ""
    retry_count: int = 0
    retry_safe: bool = False
    included_in_aggregate_metrics: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.category, FailureCategory):
            self.category = FailureCategory(str(self.category))
        validate_failure_record(self)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["category"] = self.category.value
        return data


def validate_failure_record(record: FailureRecord) -> None:
    if not record.stage:
        raise ValueError("failure record requires stage")
    if record.retry_count < 0:
        raise ValueError("retry_count must be non-negative")

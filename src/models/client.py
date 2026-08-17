from __future__ import annotations

import os
import re
import signal
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI


MODEL_CONTEXT_OVERFLOW = "MODEL_CONTEXT_OVERFLOW"
MODEL_TIMEOUT = "MODEL_TIMEOUT"
MODEL_STAGE_TIMEOUT = "MODEL_STAGE_TIMEOUT"
DEFAULT_CONTEXT_WINDOW = 8192
DEFAULT_CONTEXT_SAFETY_MARGIN = 256
PROMPT_ESTIMATE_HEADROOM = 0.85
RETRY_PROMPT_ESTIMATE_HEADROOM = 0.75
SUPPORTED_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)


@dataclass
class ModelConfig:
    provider: str
    model_name: str
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: int = 120
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    context_window: int = DEFAULT_CONTEXT_WINDOW
    context_safety_margin: int = DEFAULT_CONTEXT_SAFETY_MARGIN
    reasoning_effort: Optional[str] = None

    def __post_init__(self) -> None:
        if (
            self.reasoning_effort is not None
            and self.reasoning_effort not in SUPPORTED_REASONING_EFFORTS
        ):
            raise ValueError(
                "reasoning_effort must be one of "
                f"{sorted(SUPPORTED_REASONING_EFFORTS)!r}"
            )


class ModelContextOverflowError(RuntimeError):
    """Raised after deterministic prompt compaction cannot fit the model window."""

    def __init__(
        self,
        message: str,
        *,
        original_prompt_tokens: int,
        final_prompt_tokens: int,
        safe_input_tokens: int,
        context_window: int,
        max_output_tokens: int,
        retry_count: int,
    ) -> None:
        super().__init__(message)
        self.failure_type = MODEL_CONTEXT_OVERFLOW
        self.failure_category = "GENERATION_FAILURE"
        self.original_prompt_tokens = original_prompt_tokens
        self.final_prompt_tokens = final_prompt_tokens
        self.safe_input_tokens = safe_input_tokens
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.retry_count = retry_count

    def to_failure_record(self) -> dict:
        return {
            "failure_type": self.failure_type,
            "failure_category": self.failure_category,
            "stage": "model_call",
            "error_message": str(self),
            "retry_count": self.retry_count,
            "retry_safe": False,
            "included_in_aggregate_metrics": False,
            "original_prompt_tokens": self.original_prompt_tokens,
            "final_prompt_tokens": self.final_prompt_tokens,
            "safe_input_tokens": self.safe_input_tokens,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
        }


class ModelTimeoutError(RuntimeError):
    """Raised when a bounded model request times out."""

    def __init__(self, message: str, *, timeout: int, retry_count: int = 0) -> None:
        super().__init__(message)
        self.failure_type = MODEL_TIMEOUT
        self.failure_category = "GENERATION_FAILURE"
        self.timeout = timeout
        self.retry_count = retry_count

    def to_failure_record(self) -> dict:
        return {
            "failure_type": self.failure_type,
            "failure_category": self.failure_category,
            "stage": "model_call_timeout",
            "error_message": str(self),
            "retry_count": self.retry_count,
            "retry_safe": True,
            "included_in_aggregate_metrics": False,
            "timeout": self.timeout,
        }


class ModelStageTimeoutError(RuntimeError):
    """Raised when an entire model-using pipeline stage exceeds its deadline."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        model_call_count: int,
        completed_call_count: int,
        per_call_latency: list[float],
        total_stage_latency: float,
        timeout_sec: float,
        last_sample_index: int,
        call_records: list[dict] | None = None,
        stop_reason: str = "MODEL_STAGE_TIMEOUT",
    ) -> None:
        super().__init__(message)
        self.failure_type = MODEL_STAGE_TIMEOUT
        self.failure_category = "GENERATION_FAILURE"
        self.stage = stage
        self.model_call_count = model_call_count
        self.completed_call_count = completed_call_count
        self.per_call_latency = list(per_call_latency)
        self.total_stage_latency = total_stage_latency
        self.timeout_sec = timeout_sec
        self.last_sample_index = last_sample_index
        self.call_records = list(call_records or [])
        self.stop_reason = stop_reason
        self.retry_count = 0

    def to_failure_record(self) -> dict:
        return {
            "failure_type": self.failure_type,
            "failure_category": self.failure_category,
            "stage": self.stage,
            "error_message": str(self),
            "retry_count": self.retry_count,
            "retry_safe": True,
            "included_in_aggregate_metrics": False,
            "model_call_count": self.model_call_count,
            "completed_call_count": self.completed_call_count,
            "per_call_latency": list(self.per_call_latency),
            "total_stage_latency": self.total_stage_latency,
            "timeout_sec": self.timeout_sec,
            "last_sample_index": self.last_sample_index,
            "call_records": list(self.call_records),
            "stop_reason": self.stop_reason,
        }


def estimate_prompt_tokens(text: str) -> int:
    """Return a deterministic conservative token estimate for prompt budgeting."""
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def safe_input_token_budget(config: ModelConfig, max_output_tokens: Optional[int] = None) -> int:
    """Derive the usable input-token budget from model context settings."""
    output_tokens = int(max_output_tokens if max_output_tokens is not None else config.max_tokens)
    margin = max(DEFAULT_CONTEXT_SAFETY_MARGIN, int(config.context_safety_margin))
    return max(1, int(config.context_window) - output_tokens - margin)


def compact_prompt_to_token_budget(prompt: str, safe_input_tokens: int) -> str:
    """Deterministically compact a prompt by preserving the prefix first."""
    if estimate_prompt_tokens(prompt) <= safe_input_tokens:
        return prompt
    lines = prompt.splitlines()
    kept: list[str] = []
    for line in lines:
        candidate = "\n".join(kept + [line])
        if estimate_prompt_tokens(candidate) > safe_input_tokens:
            break
        kept.append(line)
    if kept:
        return "\n".join(kept).rstrip()
    tokens = re.findall(r"\w+|[^\w\s]", prompt, flags=re.UNICODE)
    return " ".join(tokens[:safe_input_tokens]).strip()


class LLMClient:
    def __init__(self, config: ModelConfig):
        self.config = config
        # 마지막 API 호출의 토큰 사용량 (누적 추적용)
        self.last_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.last_prompt_budget: dict = {}
        self.last_finish_reason: str = ""

        if config.provider == "local":
            self.client = OpenAI(
                base_url=config.base_url,
                api_key=config.api_key or "EMPTY",
                timeout=config.timeout,
                max_retries=0,
            )
        elif config.provider == "openai":
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                timeout=config.timeout,
                max_retries=0,
            )
        else:
            raise ValueError(f"지원하지 않는 provider: {config.provider}")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant",
        temperature: Optional[float] = None,
        prompt_compactor: Optional[Callable[[str, int], str]] = None,
        timeout: Optional[float] = None,
    ) -> str:
        safe_input_tokens = safe_input_token_budget(self.config)
        original_prompt_tokens = estimate_prompt_tokens(system_prompt) + estimate_prompt_tokens(prompt)
        budgeted_prompt = prompt
        compactor = prompt_compactor or compact_prompt_to_token_budget
        if original_prompt_tokens > safe_input_tokens:
            user_budget = max(
                1,
                int((safe_input_tokens - estimate_prompt_tokens(system_prompt)) * PROMPT_ESTIMATE_HEADROOM),
            )
            budgeted_prompt = compactor(prompt, user_budget)
        final_prompt_tokens = estimate_prompt_tokens(system_prompt) + estimate_prompt_tokens(budgeted_prompt)
        self.last_prompt_budget = {
            "context_window": self.config.context_window,
            "max_output_tokens": self.config.max_tokens,
            "safety_margin": max(DEFAULT_CONTEXT_SAFETY_MARGIN, self.config.context_safety_margin),
            "safe_input_tokens": safe_input_tokens,
            "original_prompt_tokens": original_prompt_tokens,
            "final_prompt_tokens": final_prompt_tokens,
            "compacted": final_prompt_tokens < original_prompt_tokens,
            "retry_count": 0,
        }
        try:
            response = self._create_completion(budgeted_prompt, system_prompt, temperature, timeout=timeout)
        except APITimeoutError as exc:
            raise ModelTimeoutError(
                f"model request timed out after {self.config.timeout}s",
                timeout=int(self.config.timeout),
                retry_count=0,
            ) from exc
        except APIConnectionError as exc:
            timeout_exc = _find_timeout_cause(exc)
            if timeout_exc is not None:
                raise timeout_exc
            raise
        except BadRequestError as exc:
            if not _is_context_length_error(exc):
                raise
            retry_prompt = compactor(
                budgeted_prompt,
                max(
                    1,
                    int(
                        (safe_input_tokens - estimate_prompt_tokens(system_prompt))
                        * RETRY_PROMPT_ESTIMATE_HEADROOM
                    ),
                ),
            )
            self.last_prompt_budget["retry_count"] = 1
            self.last_prompt_budget["retry_reason"] = MODEL_CONTEXT_OVERFLOW
            self.last_prompt_budget["final_prompt_tokens"] = (
                estimate_prompt_tokens(system_prompt) + estimate_prompt_tokens(retry_prompt)
            )
            try:
                response = self._create_completion(retry_prompt, system_prompt, temperature, timeout=timeout)
            except APITimeoutError as timeout_exc:
                raise ModelTimeoutError(
                    f"model request timed out after {self.config.timeout}s during context-overflow retry",
                    timeout=int(self.config.timeout),
                    retry_count=1,
                ) from timeout_exc
            except APIConnectionError as conn_exc:
                timeout_exc = _find_timeout_cause(conn_exc)
                if timeout_exc is not None:
                    raise ModelTimeoutError(
                        f"model request timed out after {self.config.timeout}s during context-overflow retry",
                        timeout=int(self.config.timeout),
                        retry_count=1,
                    ) from timeout_exc
                raise
            except ModelTimeoutError as timeout_exc:
                raise ModelTimeoutError(
                    f"model request timed out after {self.config.timeout}s during context-overflow retry",
                    timeout=int(self.config.timeout),
                    retry_count=1,
                ) from timeout_exc
            except BadRequestError as retry_exc:
                if _is_context_length_error(retry_exc):
                    raise ModelContextOverflowError(
                        "model context window exceeded after deterministic compacting retry",
                        original_prompt_tokens=original_prompt_tokens,
                        final_prompt_tokens=int(self.last_prompt_budget["final_prompt_tokens"]),
                        safe_input_tokens=safe_input_tokens,
                        context_window=self.config.context_window,
                        max_output_tokens=self.config.max_tokens,
                        retry_count=1,
                    ) from retry_exc
                raise
        # API 응답에서 실제 토큰 사용량 저장
        if response.usage:
            self.last_usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
        self.last_finish_reason = ""
        if response.choices:
            self.last_finish_reason = getattr(response.choices[0], "finish_reason", "") or ""
            return response.choices[0].message.content
        return ""

    def complete(self, prompt: str) -> str:
        """Compatibility method for M1/M2 single-prompt refinement protocols."""
        return self.generate(prompt, system_prompt="You are a helpful assistant")

    def _create_completion(
        self,
        prompt: str,
        system_prompt: str,
        temperature: Optional[float],
        timeout: Optional[float] = None,
    ):
        effective_timeout = int(max(1, timeout if timeout is not None else self.config.timeout))

        def call_model():
            request: dict[str, Any] = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature if temperature is not None else self.config.temperature,
                "timeout": effective_timeout,
            }
            if self.config.reasoning_effort is None:
                request["max_tokens"] = self.config.max_tokens
            else:
                request["reasoning_effort"] = self.config.reasoning_effort
                request["max_completion_tokens"] = self.config.max_tokens
            return self.client.chat.completions.create(
                **request,
            )

        return _run_with_wall_timeout(
            call_model,
            timeout_seconds=effective_timeout,
            timeout_message=f"model request timed out after {effective_timeout}s",
        )


def _is_context_length_error(exc: BadRequestError) -> bool:
    text = str(exc).lower()
    return (
        "context" in text
        and any(marker in text for marker in ("length", "window", "limit", "maximum", "8192"))
    )


def _find_timeout_cause(exc: BaseException) -> ModelTimeoutError | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ModelTimeoutError):
            return current
        current = current.__cause__ or current.__context__
    return None


def _run_with_wall_timeout(
    fn: Callable[[], Any],
    *,
    timeout_seconds: int,
    timeout_message: str,
) -> Any:
    """Run one model request with a hard wall-clock guard.

    The OpenAI-compatible server used in local runs can leave socket reads
    blocked beyond the SDK timeout. On the main thread, SIGALRM interrupts the
    blocking read so the pipeline records a structured MODEL_TIMEOUT instead
    of waiting indefinitely.
    """
    if threading.current_thread() is not threading.main_thread():
        return fn()

    def _handle_timeout(_signum: int, _frame: Any) -> None:
        raise ModelTimeoutError(
            timeout_message,
            timeout=timeout_seconds,
            retry_count=0,
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, max(1, timeout_seconds))
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isfinite

from agent.models import Agent, Message, ProviderMessage, Run, Session
from providers.base import ProviderRequest


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Request limits for the selected model, independent of stored history."""

    context_window_tokens: int
    max_output_tokens: int
    safety_margin_tokens: int = 1024
    trigger_ratio: float = 0.85
    target_ratio: float = 0.60

    def __post_init__(self) -> None:
        if self.context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.safety_margin_tokens < 0:
            raise ValueError("safety_margin_tokens cannot be negative")
        if self.input_limit_tokens <= 0:
            raise ValueError("Output reserve and safety margin leave no input space")
        if not (
            isfinite(self.target_ratio)
            and isfinite(self.trigger_ratio)
            and 0 < self.target_ratio < self.trigger_ratio <= 1
        ):
            raise ValueError("Ratios must satisfy 0 < target_ratio < trigger_ratio <= 1")
        if not 0 < self.target_tokens < self.trigger_tokens:
            raise ValueError("Budget is too small for distinct target and trigger tokens")

    @property
    def input_limit_tokens(self) -> int:
        return (
            self.context_window_tokens
            - self.max_output_tokens
            - self.safety_margin_tokens
        )

    @property
    def trigger_tokens(self) -> int:
        return int(self.input_limit_tokens * self.trigger_ratio)

    @property
    def target_tokens(self) -> int:
        return int(self.input_limit_tokens * self.target_ratio)

    def should_compact(self, input_tokens: int) -> bool:
        if input_tokens < 0:
            raise ValueError("input_tokens cannot be negative")
        return input_tokens >= self.trigger_tokens


@dataclass(frozen=True, slots=True)
class ContextUsage:
    """Approximate input tokens, not billing usage or a provider token count."""

    system_tokens: int
    history_tokens: int
    tool_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.system_tokens + self.history_tokens + self.tool_tokens


def _estimate_json_tokens(value: object) -> int:
    # Include structured fields and UTF-8 bytes; this is still a heuristic,
    # not an upper bound on any provider's tokenizer or wire-format overhead.
    serialized = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    )
    return (len(serialized.encode("utf-8")) + 3) // 4


def estimate_request_usage(request: ProviderRequest) -> ContextUsage:
    """Estimate the current request without modifying messages or counting run metadata."""
    system_tokens = 0
    history_tokens = 0
    for message in request.messages:
        tokens = _estimate_json_tokens(asdict(message))
        if message.role == "system":
            system_tokens += tokens
        else:
            history_tokens += tokens

    tool_tokens = (
        _estimate_json_tokens([asdict(tool) for tool in request.tools])
        if request.tools
        else 0
    )
    return ContextUsage(
        system_tokens=system_tokens,
        history_tokens=history_tokens,
        tool_tokens=tool_tokens,
    )


def _to_provider_message(message: Message) -> ProviderMessage:
    return ProviderMessage(
        role=message.role,
        content=message.content,
        name=message.name,
        tool_calls=list(message.tool_calls),
        tool_call_id=message.tool_call_id,
        is_error=(
            message.role == "tool"
            and message.metadata.get("ok") is False
        ),
    )


def build_provider_messages(
    agent: Agent,
    messages: list[Message],
    *,
    session: Session | None = None,
    run: Run | None = None,
) -> list[ProviderMessage]:
    if (session is None) != (run is None):
        raise ValueError("Session and run temporal context must be provided together")

    provider_messages: list[ProviderMessage] = []

    if agent.instructions:
        provider_messages.append(
            ProviderMessage(
                role="system",
                content=agent.instructions,
            )
        )

    if session is not None and run is not None:
        provider_messages.append(
            ProviderMessage(
                role="system",
                content=(
                    "Temporal context (UTC):\n"
                    f"- Session created at: {session.created_at.isoformat()}\n"
                    f"- Current run started at: {run.created_at.isoformat()}"
                ),
            )
        )

    provider_messages.extend(
        _to_provider_message(message)
        for message in messages
    )
    return provider_messages

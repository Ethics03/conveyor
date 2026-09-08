from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Literal

from agent.models import (
    Agent,
    Message,
    ProviderMessage,
    ProviderReplayState,
    Run,
    Session,
)
from providers.base import ProviderRequest, ToolSchema

PROTECTED_RECENT_TURNS = 2
MIN_CLEARABLE_TOOL_RESULT_CHARS = 4_096


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
            raise ValueError(
                "Ratios must satisfy 0 < target_ratio < trigger_ratio <= 1"
            )
        if not 0 < self.target_tokens < self.trigger_tokens:
            raise ValueError(
                "Budget is too small for distinct target and trigger tokens"
            )

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


TokenCountSource = Literal["provider", "heuristic"]


@dataclass(frozen=True, slots=True)
class ContextMeasurement:
    """Input-token measurement and the source that produced it."""

    input_tokens: int
    source: TokenCountSource
    usage: ContextUsage | None = None

    def __post_init__(self) -> None:
        if self.input_tokens < 0:
            raise ValueError("input_tokens cannot be negative")


@dataclass(frozen=True, slots=True)
class ContextPlan:
    """A sendable provider request and the decisions used to produce it."""

    request: ProviderRequest
    before: ContextMeasurement
    after: ContextMeasurement
    compaction_triggered: bool
    cleared_tool_results: int
    summary_required: bool

    def __post_init__(self) -> None:
        if self.cleared_tool_results < 0:
            raise ValueError("cleared_tool_results cannot be negative")
        if not self.compaction_triggered and (
            self.cleared_tool_results > 0 or self.summary_required
        ):
            raise ValueError(
                "An untriggered context plan cannot clear or summarize history"
            )


def group_conversation_turns(messages: list[Message]) -> list[tuple[Message, ...]]:
    """Group each user message with every response produced before the next user."""
    groups: list[tuple[Message, ...]] = []
    current: list[Message] = []

    for message in messages:
        if message.role == "user" and current:
            groups.append(tuple(current))
            current = []
        current.append(message)

    if current:
        groups.append(tuple(current))

    return groups


def clear_tool_results(messages: list[Message]) -> list[Message]:
    """Clear bulky old tool payloads while preserving transcript structure."""
    groups = group_conversation_turns(messages)
    protected_start = max(0, len(groups) - PROTECTED_RECENT_TURNS)
    reduced: list[Message] = []

    for index, group in enumerate(groups):
        for message in group:
            if (
                index < protected_start
                and message.role == "tool"
                and len(message.content) >= MIN_CLEARABLE_TOOL_RESULT_CHARS
            ):
                tool_name = message.name or "tool"
                reduced.append(
                    replace(
                        message,
                        content=(
                            f"[{tool_name} result omitted from active context: "
                            f"{len(message.content)} characters. "
                            "The original result remains in the session transcript.]"
                        ),
                    )
                )
            else:
                reduced.append(message)

    return reduced


def _estimate_json_tokens(value: object) -> int:
    # Include structured fields and UTF-8 bytes; this is still a heuristic,
    # not an upper bound on any provider's tokenizer or wire-format overhead.
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
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
        is_error=(message.role == "tool" and message.metadata.get("ok") is False),
        replay_state=ProviderReplayState.from_metadata(
            message.metadata.get("provider_replay_state")
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

    provider_messages.extend(_to_provider_message(message) for message in messages)
    return provider_messages


def _build_context_request(
    *,
    agent: Agent,
    messages: list[Message],
    tools: list[ToolSchema],
    budget: ContextBudget,
    session: Session | None,
    run: Run | None,
) -> ProviderRequest:
    return ProviderRequest(
        messages=build_provider_messages(
            agent,
            messages,
            session=session,
            run=run,
        ),
        tools=list(tools),
        model=agent.model,
        max_tokens=budget.max_output_tokens,
    )


def _measure_request(request: ProviderRequest) -> ContextMeasurement:
    usage = estimate_request_usage(request)
    return ContextMeasurement(
        input_tokens=usage.total_tokens,
        source="heuristic",
        usage=usage,
    )


def plan_context(
    *,
    agent: Agent,
    messages: list[Message],
    tools: list[ToolSchema],
    budget: ContextBudget,
    session: Session | None = None,
    run: Run | None = None,
) -> ContextPlan:
    """Build a provider request and apply deterministic cleanup when required."""
    request = _build_context_request(
        agent=agent,
        messages=messages,
        tools=tools,
        budget=budget,
        session=session,
        run=run,
    )
    before = _measure_request(request)
    if not budget.should_compact(before.input_tokens):
        return ContextPlan(
            request=request,
            before=before,
            after=before,
            compaction_triggered=False,
            cleared_tool_results=0,
            summary_required=False,
        )

    reduced_messages = clear_tool_results(messages)
    cleared_tool_results = sum(
        original is not reduced
        for original, reduced in zip(messages, reduced_messages, strict=True)
    )
    if cleared_tool_results == 0:
        return ContextPlan(
            request=request,
            before=before,
            after=before,
            compaction_triggered=True,
            cleared_tool_results=0,
            summary_required=True,
        )

    reduced_request = _build_context_request(
        agent=agent,
        messages=reduced_messages,
        tools=tools,
        budget=budget,
        session=session,
        run=run,
    )
    after = _measure_request(reduced_request)
    return ContextPlan(
        request=reduced_request,
        before=before,
        after=after,
        compaction_triggered=True,
        cleared_tool_results=cleared_tool_results,
        summary_required=budget.should_compact(after.input_tokens),
    )

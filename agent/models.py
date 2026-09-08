from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid7

SessionStatus = Literal["active", "archived"]
SessionTitleSource = Literal["default", "auto", "user"]
RunStatus = Literal["pending", "running", "blocked", "finished", "cancelled", "failed"]
MessageRole = Literal["user", "assistant", "system", "tool"]
EventType = Literal[
    "session.created",
    "message.created",
    "context.compaction_planned",
    "context.compacted",
    "run.started",
    "run.blocked",
    "run.resumed",
    "run.finished",
    "tool.started",
    "approval.requested",
    "approval.resolved",
    "tool.finished",
    "run.failed",
]
ApprovalStatus = Literal["pending", "approved", "denied"]
ApprovalDecision = Literal["approved", "denied"]
ToolPermission = Literal["read", "write", "dangerous"]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid7().hex}"


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class Agent:
    id: str = field(default_factory=lambda: new_id("agent"))
    name: str = "Assistant"
    instructions: str = ""
    model: str | None = None
    tools: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Session:
    id: str = field(default_factory=lambda: new_id("ses"))
    title: str = "New session"
    title_source: SessionTitleSource = "default"
    status: SessionStatus = "active"
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class Run:
    id: str = field(default_factory=lambda: new_id("run"))
    session_id: str = ""
    agent_id: str = ""
    parent_run_id: str | None = None
    status: RunStatus = "pending"
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    error: str | None = None


@dataclass(slots=True)
class Message:
    id: str = field(default_factory=lambda: new_id("msg"))
    session_id: str = ""
    role: MessageRole = "user"
    content: str = ""
    run_id: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunOutcome:
    run: Run
    final_message: Message | None
    iterations: int


@dataclass(slots=True)
class Event:
    id: str = field(default_factory=lambda: new_id("evt"))
    type: EventType = "message.created"
    session_id: str | None = None
    run_id: str | None = None
    message_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class ToolCall:
    id: str = field(default_factory=lambda: new_id("call"))
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    tool_call_id: str
    name: str
    ok: bool
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ApprovalRequest:
    session_id: str
    run_id: str
    tool_call: ToolCall
    reason: str = ""
    id: str = field(default_factory=lambda: new_id("appr"))
    status: ApprovalStatus = "pending"
    created_at: datetime = field(default_factory=utc_now)
    resolved_at: datetime | None = None


@dataclass(slots=True)
class ProviderMessage:
    role: MessageRole
    content: str
    name: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    is_error: bool = False
    replay_state: ProviderReplayState | None = None


@dataclass(frozen=True, slots=True)
class ProviderReplayState:
    """Opaque provider-owned items needed to continue a conversation."""

    provider: str
    items: tuple[dict[str, object], ...]

    def to_metadata(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "items": [dict(item) for item in self.items],
        }

    @classmethod
    def from_metadata(cls, value: object) -> ProviderReplayState | None:
        if not isinstance(value, dict):
            return None

        provider = value.get("provider")
        raw_items = value.get("items")
        if not isinstance(provider, str) or not isinstance(raw_items, list):
            return None

        items: list[dict[str, object]] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                return None
            items.append(dict(raw_item))
        return cls(provider=provider, items=tuple(items))


@dataclass(slots=True)
class ProviderResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    replay_state: ProviderReplayState | None = None

    @classmethod
    def message(
        cls,
        content: str,
        raw: dict[str, Any] | None = None,
        finish_reason: str | None = "stop",
    ) -> ProviderResponse:
        return cls(content=content, finish_reason=finish_reason, raw=raw or {})

    @classmethod
    def tool(
        cls,
        name: str,
        arguments: dict[str, Any],
        raw: dict[str, Any] | None = None,
        *,
        tool_call_id: str | None = None,
        content: str = "",
        finish_reason: str | None = "tool_use",
    ) -> ProviderResponse:
        tool_call = ToolCall(name=name, arguments=arguments)
        if tool_call_id is not None:
            tool_call.id = tool_call_id
        return cls(
            content=content,
            tool_calls=[tool_call],
            finish_reason=finish_reason,
            raw=raw or {},
        )

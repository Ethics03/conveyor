from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid7


SessionStatus = Literal["active", "archived"]
RunStatus = Literal["pending", "running", "blocked", "finished", "cancelled", "failed"]
MessageRole = Literal["user", "assistant", "system", "tool"]
EventType = Literal[
    "session.created",
    "message.created",
    "run.started",
    "run.finished",
    "tool.started",
    "approval.requested",
    "tool.finished",
    "run.failed",
]

ToolPermission = Literal["read", "write", "dangerous"]
ProviderResponseType = Literal["message", "tool_call"]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid7().hex}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


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
    id: str = field(default_factory=lambda: new_id("appr"))
    session_id: str = ""
    run_id: str = ""
    tool_call: ToolCall | None = None
    reason: str = ""
    status: Literal["pending", "approved", "denied"] = "pending"
    created_at: datetime = field(default_factory=utc_now)
    resolved_at: datetime | None = None


@dataclass(slots=True)
class ProviderMessage:
    role: MessageRole
    content: str
    name: str | None = None


@dataclass(slots=True)
class ProviderResponse:
    type: ProviderResponseType
    content: str = ""
    tool_call: ToolCall | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def message(
        cls, content: str, raw: dict[str, Any] | None = None
    ) -> "ProviderResponse":
        return cls(type="message", content=content, raw=raw or {})

    @classmethod
    def tool(
        cls,
        name: str,
        arguments: dict[str, Any],
        raw: dict[str, Any] | None = None,
    ) -> "ProviderResponse":
        return cls(
            type="tool_call",
            tool_call=ToolCall(name=name, arguments=arguments),
            raw=raw or {},
        )

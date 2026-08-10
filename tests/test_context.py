from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent.context import build_provider_messages
from agent.models import Agent, Message, ProviderMessage, Run, Session, ToolCall


def test_build_provider_messages_prepends_agent_instructions() -> None:
    agent = Agent(instructions="Be precise.")
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi"),
    ]

    assert build_provider_messages(agent, messages) == [
        ProviderMessage(role="system", content="Be precise."),
        ProviderMessage(role="user", content="Hello"),
        ProviderMessage(role="assistant", content="Hi"),
    ]


def test_build_provider_messages_omits_empty_instructions() -> None:
    messages = [Message(role="user", content="Hello")]

    assert build_provider_messages(Agent(), messages) == [
        ProviderMessage(role="user", content="Hello"),
    ]


def test_build_provider_messages_includes_temporal_context() -> None:
    session = Session(
        created_at=datetime(2026, 8, 10, 8, 0, tzinfo=timezone.utc),
    )
    run = Run(
        session_id=session.id,
        created_at=datetime(2026, 8, 10, 8, 5, tzinfo=timezone.utc),
    )

    provider_messages = build_provider_messages(
        Agent(),
        [Message(role="user", content="What happened?")],
        session=session,
        run=run,
    )

    assert provider_messages == [
        ProviderMessage(
            role="system",
            content=(
                "Temporal context (UTC):\n"
                "- Session created at: 2026-08-10T08:00:00+00:00\n"
                "- Current run started at: 2026-08-10T08:05:00+00:00"
            ),
        ),
        ProviderMessage(role="user", content="What happened?"),
    ]


def test_build_provider_messages_rejects_partial_temporal_context() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        _ = build_provider_messages(Agent(), [], session=Session())


def test_build_provider_messages_preserves_tool_call_relationships() -> None:
    tool_call = ToolCall(
        id="call_readme",
        name="read_file",
        arguments={"path": "README.md"},
    )
    assistant = Message(
        role="assistant",
        content="I will inspect the file.",
        tool_calls=[tool_call],
    )
    tool_result = Message(
        role="tool",
        content='{"content": "Conveyor"}',
        name="read_file",
        tool_call_id=tool_call.id,
    )

    provider_messages = build_provider_messages(
        Agent(),
        [assistant, tool_result],
    )

    assert provider_messages == [
        ProviderMessage(
            role="assistant",
            content="I will inspect the file.",
            tool_calls=[tool_call],
        ),
        ProviderMessage(
            role="tool",
            content='{"content": "Conveyor"}',
            name="read_file",
            tool_call_id="call_readme",
        ),
    ]
    assert provider_messages[0].tool_calls is not assistant.tool_calls

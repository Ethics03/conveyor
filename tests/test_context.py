from __future__ import annotations

from agent.context import build_provider_messages
from agent.models import Agent, Message, ProviderMessage, ToolCall


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

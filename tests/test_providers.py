from types import SimpleNamespace

import pytest

from agent.models import ProviderMessage, ProviderResponse, ToolCall
from providers.anthropic_provider import AnthropicProvider, _anthropic_messages
from providers.base import ProviderRequest


def test_provider_response_can_contain_text_and_tool_calls() -> None:
    response = ProviderResponse.tool(
        "read_file",
        {"path": "README.md"},
        tool_call_id="call_readme",
        content="I will inspect the file.",
    )

    assert response.content == "I will inspect the file."
    assert response.finish_reason == "tool_use"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_readme"
    assert response.tool_calls[0].name == "read_file"


def test_anthropic_messages_preserve_tool_call_relationships() -> None:
    messages = [
        ProviderMessage(role="user", content="Read both files."),
        ProviderMessage(
            role="assistant",
            content="I will inspect them.",
            tool_calls=[
                ToolCall(
                    id="call_one",
                    name="read_file",
                    arguments={"path": "README.md"},
                ),
                ToolCall(
                    id="call_two",
                    name="read_file",
                    arguments={"path": "pyproject.toml"},
                ),
            ],
        ),
        ProviderMessage(
            role="tool",
            content='{"content": "README"}',
            name="read_file",
            tool_call_id="call_one",
        ),
        ProviderMessage(
            role="tool",
            content='{"content": "pyproject"}',
            name="read_file",
            tool_call_id="call_two",
        ),
    ]

    assert _anthropic_messages(messages) == [
        {
            "role": "user",
            "content": "Read both files.",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I will inspect them.",
                },
                {
                    "type": "tool_use",
                    "id": "call_one",
                    "name": "read_file",
                    "input": {"path": "README.md"},
                },
                {
                    "type": "tool_use",
                    "id": "call_two",
                    "name": "read_file",
                    "input": {"path": "pyproject.toml"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_one",
                    "content": '{"content": "README"}',
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "call_two",
                    "content": '{"content": "pyproject"}',
                },
            ],
        },
    ]


def test_anthropic_tool_result_requires_tool_call_id() -> None:
    with pytest.raises(ValueError, match="requires tool_call_id"):
        _anthropic_messages([
            ProviderMessage(role="tool", content="missing id"),
        ])


def test_anthropic_provider_collects_multiple_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anthropic_response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="I will inspect both files."),
            SimpleNamespace(
                type="tool_use",
                id="call_one",
                name="read_file",
                input={"path": "README.md"},
            ),
            SimpleNamespace(
                type="tool_use",
                id="call_two",
                name="read_file",
                input={"path": "pyproject.toml"},
            ),
        ],
        stop_reason="tool_use",
    )

    class FakeMessages:
        def create(self, **params: object) -> object:
            return anthropic_response

    class FakeAnthropic:
        def __init__(self, **params: object) -> None:
            self.messages = FakeMessages()

    monkeypatch.setattr("providers.anthropic_provider.Anthropic", FakeAnthropic)

    response = AnthropicProvider(api_key="test-key").generate(
        ProviderRequest(messages=[])
    )

    assert response.content == "I will inspect both files."
    assert response.finish_reason == "tool_use"
    assert [tool_call.id for tool_call in response.tool_calls] == [
        "call_one",
        "call_two",
    ]
    assert [tool_call.arguments for tool_call in response.tool_calls] == [
        {"path": "README.md"},
        {"path": "pyproject.toml"},
    ]

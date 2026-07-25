from types import SimpleNamespace

import pytest

from agent.models import ProviderResponse
from providers.anthropic import AnthropicProvider
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

    fake_module = SimpleNamespace(Anthropic=FakeAnthropic)
    monkeypatch.setattr("providers.anthropic.import_module", lambda name: fake_module)

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

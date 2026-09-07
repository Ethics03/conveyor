import pytest

from agent.models import ProviderResponse
from agent.titles import (
    MAX_DERIVED_TITLE_CHARS,
    MAX_SESSION_TITLE_CHARS,
    MAX_TITLE_INPUT_CHARS,
    clean_session_title,
    derive_session_title,
    generate_session_title,
)
from providers.fake import FakeProvider


@pytest.mark.parametrize("message", ["", "   ", "\n\t\n"])
def test_derive_session_title_rejects_empty_input(message: str) -> None:
    assert derive_session_title(message) is None


def test_derive_session_title_uses_first_meaningful_line() -> None:
    title = derive_session_title(
        "\n  Review   the Anthropic provider  \nIgnore this later detail",
    )

    assert title == "Review the Anthropic provider"


def test_derive_session_title_preserves_short_input() -> None:
    assert derive_session_title("Fix API key handling") == "Fix API key handling"


def test_derive_session_title_truncates_at_a_word_boundary() -> None:
    title = derive_session_title(
        "Investigate the Anthropic provider request timeout configuration",
    )

    assert title == "Investigate the Anthropic provider request..."
    assert len(title) <= MAX_DERIVED_TITLE_CHARS


def test_derive_session_title_bounds_a_single_long_word() -> None:
    title = derive_session_title("x" * 100)

    assert title == "x" * 45 + "..."
    assert len(title) == MAX_DERIVED_TITLE_CHARS


def test_derive_session_title_ignores_content_beyond_input_cap() -> None:
    message = " " * MAX_TITLE_INPUT_CHARS + "Use this as the title"

    assert derive_session_title(message) is None


def test_clean_session_title_normalizes_whitespace() -> None:
    assert clean_session_title("  Fix\n  provider   timeout  ") == "Fix provider timeout"


@pytest.mark.parametrize("title", ["", " ", "\n\t"])
def test_clean_session_title_rejects_empty_title(title: str) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        clean_session_title(title)


def test_clean_session_title_rejects_long_title() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        clean_session_title("x" * (MAX_SESSION_TITLE_CHARS + 1))


def test_generate_session_title_uses_a_bounded_no_tools_request() -> None:
    provider = FakeProvider(['"Interview Today"'])

    title = generate_session_title(
        provider,
        user_message="Do I have an interview today?",
        assistant_response="Yes, at 3 PM.",
        model="title-model",
    )

    assert title == "Interview Today"
    request = provider.requests[0]
    assert request.model == "title-model"
    assert request.temperature == 0
    assert request.max_tokens == 32
    assert request.tools == []
    assert request.metadata == {"purpose": "session_title"}
    assert [message.role for message in request.messages] == ["system", "user"]


def test_generate_session_title_rejects_tool_calls() -> None:
    provider = FakeProvider([ProviderResponse.tool("read_file", {"path": "README.md"})])

    with pytest.raises(ValueError, match="returned tool calls"):
        generate_session_title(
            provider,
            user_message="Read the project",
            assistant_response="I will inspect it.",
        )

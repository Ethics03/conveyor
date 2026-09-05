from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

import pytest

from agent.context import ContextBudget, build_provider_messages, estimate_request_usage
from agent.models import Agent, Message, ProviderMessage, Run, Session, ToolCall
from providers.base import ProviderRequest, ToolSchema


def test_context_budget_reserves_output_and_compacts_before_input_limit() -> None:
    budget = ContextBudget(
        context_window_tokens=10_000,
        max_output_tokens=1_000,
        safety_margin_tokens=1_000,
        trigger_ratio=0.75,
        target_ratio=0.50,
    )

    assert budget.input_limit_tokens == 8_000
    assert budget.trigger_tokens == 6_000
    assert budget.target_tokens == 4_000
    assert budget.should_compact(5_999) is False
    assert budget.should_compact(6_000) is True
    assert budget.should_compact(budget.target_tokens) is False
    assert budget.should_compact(10_000) is True
    with pytest.raises(ValueError, match="input_tokens cannot be negative"):
        budget.should_compact(-1)


@pytest.mark.parametrize(
    ("window", "output", "margin", "trigger", "target"),
    [
        (0, 100, 0, 0.85, 0.60),
        (1_000, 0, 0, 0.85, 0.60),
        (1_000, 100, -1, 0.85, 0.60),
        (1_000, 900, 100, 0.85, 0.60),
        (1_000, 100, 0, 1.1, 0.60),
        (1_000, 100, 0, 0.85, 0.85),
        (1_000, 100, 0, 0.85, 0.0),
        (1_000, 100, 0, float("nan"), 0.60),
        (1_000, 100, 0, 0.85, float("inf")),
        (2, 1, 0, 0.85, 0.60),
    ],
)
def test_context_budget_rejects_invalid_configuration(
    window: int, output: int, margin: int, trigger: float, target: float
) -> None:
    with pytest.raises(ValueError):
        ContextBudget(window, output, margin, trigger, target)


def test_estimate_request_usage_includes_system_history_and_tool_schemas() -> None:
    session = Session()
    request = ProviderRequest(
        messages=build_provider_messages(
            Agent(instructions="policy " * 1_000),
            [Message(role="user", content="question " * 1_000)],
            session=session,
            run=Run(session_id=session.id),
        ),
        tools=[
            ToolSchema(
                name="read_file",
                description="Read workspace text",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "schema " * 1_000},
                    },
                },
            ),
        ],
    )

    usage = estimate_request_usage(request)

    assert usage.system_tokens > 1_000
    assert usage.history_tokens > 1_000
    assert usage.tool_tokens > 1_000
    assert usage.total_tokens == (
        usage.system_tokens + usage.history_tokens + usage.tool_tokens
    )


def test_estimate_request_usage_accounts_for_tool_arguments_and_new_results() -> None:
    call = ToolCall(
        id="call_write", name="write_file", arguments={"content": "x" * 20_000},
    )
    request = ProviderRequest(
        messages=[ProviderMessage(role="assistant", content="", tool_calls=[call])],
    )
    before = estimate_request_usage(request)
    assert before.history_tokens >= 5_000

    request.messages.append(
        ProviderMessage(
            role="tool", content="y" * 20_000, tool_call_id=call.id, is_error=True,
        )
    )
    snapshot = deepcopy(request)
    after = estimate_request_usage(request)

    assert after.history_tokens - before.history_tokens >= 5_000
    assert request == snapshot


def test_estimate_request_usage_excludes_metadata_and_output_reserve() -> None:
    request = ProviderRequest(messages=[ProviderMessage(role="user", content="Hello")])
    before = estimate_request_usage(request)
    request.metadata["debug"] = "x" * 20_000
    request.max_tokens = 8_000

    assert estimate_request_usage(request) == before
    assert estimate_request_usage(ProviderRequest(messages=[])).total_tokens == 0


def test_estimate_request_usage_accounts_for_multibyte_text() -> None:
    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="\u4e2d" * 4_000)],
    )

    assert estimate_request_usage(request).history_tokens >= 3_000


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
        created_at=datetime(2026, 8, 10, 8, 0, tzinfo=UTC),
    )
    run = Run(
        session_id=session.id,
        created_at=datetime(2026, 8, 10, 8, 5, tzinfo=UTC),
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


def test_build_provider_messages_marks_failed_tool_results() -> None:
    provider_messages = build_provider_messages(
        Agent(),
        [
            Message(
                role="tool",
                content="Tool execution failed",
                tool_call_id="call_failed",
                metadata={"ok": False},
            )
        ],
    )

    assert provider_messages == [
        ProviderMessage(
            role="tool",
            content="Tool execution failed",
            tool_call_id="call_failed",
            is_error=True,
        )
    ]

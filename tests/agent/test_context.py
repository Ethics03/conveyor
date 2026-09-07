from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

import pytest

from agent.context import (
    MIN_CLEARABLE_TOOL_RESULT_CHARS,
    ContextBudget,
    ContextMeasurement,
    ContextPlan,
    ContextUsage,
    build_provider_messages,
    clear_tool_results,
    estimate_request_usage,
    group_conversation_turns,
    plan_context,
)
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


def test_context_measurement_records_source_and_optional_breakdown() -> None:
    usage = ContextUsage(system_tokens=10, history_tokens=20, tool_tokens=5)

    measurement = ContextMeasurement(
        input_tokens=usage.total_tokens,
        source="heuristic",
        usage=usage,
    )

    assert measurement.input_tokens == 35
    assert measurement.source == "heuristic"
    assert measurement.usage == usage


def test_context_measurement_rejects_negative_input_tokens() -> None:
    with pytest.raises(ValueError, match="input_tokens cannot be negative"):
        ContextMeasurement(input_tokens=-1, source="provider")


def test_context_plan_records_sendable_request_and_decisions() -> None:
    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="Continue")],
    )
    before = ContextMeasurement(input_tokens=8_500, source="heuristic")
    after = ContextMeasurement(input_tokens=6_500, source="heuristic")

    plan = ContextPlan(
        request=request,
        before=before,
        after=after,
        compaction_triggered=True,
        cleared_tool_results=3,
        summary_required=False,
    )

    assert plan.request is request
    assert plan.before is before
    assert plan.after is after
    assert plan.compaction_triggered is True
    assert plan.cleared_tool_results == 3
    assert plan.summary_required is False


def test_context_plan_rejects_negative_clear_count() -> None:
    measurement = ContextMeasurement(input_tokens=100, source="heuristic")

    with pytest.raises(ValueError, match="cleared_tool_results cannot be negative"):
        ContextPlan(
            request=ProviderRequest(messages=[]),
            before=measurement,
            after=measurement,
            compaction_triggered=True,
            cleared_tool_results=-1,
            summary_required=False,
        )


def test_context_plan_rejects_work_without_trigger() -> None:
    measurement = ContextMeasurement(input_tokens=100, source="heuristic")

    with pytest.raises(ValueError, match="untriggered context plan"):
        ContextPlan(
            request=ProviderRequest(messages=[]),
            before=measurement,
            after=measurement,
            compaction_triggered=False,
            cleared_tool_results=1,
            summary_required=False,
        )


def test_group_conversation_turns_keeps_tool_exchange_with_its_user_turn() -> None:
    first_user = Message(role="user", content="Inspect the file")
    tool_call = ToolCall(id="call_read", name="read_file")
    assistant_call = Message(role="assistant", tool_calls=[tool_call])
    tool_result = Message(
        role="tool",
        name="read_file",
        tool_call_id=tool_call.id,
        content="contents",
    )
    first_answer = Message(role="assistant", content="Done")
    second_user = Message(role="user", content="What changed?")
    second_answer = Message(role="assistant", content="Nothing")

    groups = group_conversation_turns(
        [
            first_user,
            assistant_call,
            tool_result,
            first_answer,
            second_user,
            second_answer,
        ]
    )

    assert groups == [
        (first_user, assistant_call, tool_result, first_answer),
        (second_user, second_answer),
    ]


def test_clear_tool_results_preserves_pairing_and_stored_messages() -> None:
    old_payload = "x" * MIN_CLEARABLE_TOOL_RESULT_CHARS
    old_call = ToolCall(id="call_old", name="read_file")
    old_assistant = Message(role="assistant", tool_calls=[old_call])
    old_result = Message(
        role="tool",
        name="read_file",
        tool_call_id=old_call.id,
        content=old_payload,
    )
    messages = [
        Message(role="user", content="Old turn"),
        old_assistant,
        old_result,
        Message(role="assistant", content="Old answer"),
        Message(role="user", content="Recent turn one"),
        Message(role="assistant", content="Recent answer one"),
        Message(role="user", content="Recent turn two"),
        Message(role="assistant", content="Recent answer two"),
    ]

    reduced = clear_tool_results(messages)

    assert old_result.content == old_payload
    assert reduced[1] is old_assistant
    assert reduced[2] is not old_result
    assert reduced[2].tool_call_id == old_call.id
    assert reduced[2].name == "read_file"
    assert reduced[2].content == (
        "[read_file result omitted from active context: 4096 characters. "
        "The original result remains in the session transcript.]"
    )


def test_clear_tool_results_keeps_small_and_recent_results() -> None:
    old_small = Message(role="tool", content="small", name="search_files")
    recent_large = Message(
        role="tool",
        content="y" * MIN_CLEARABLE_TOOL_RESULT_CHARS,
        name="read_file",
    )
    messages = [
        Message(role="user", content="Old turn"),
        old_small,
        Message(role="user", content="Recent turn one"),
        recent_large,
        Message(role="user", content="Recent turn two"),
        Message(role="assistant", content="Recent answer"),
    ]

    reduced = clear_tool_results(messages)

    assert reduced == messages


def test_plan_context_returns_unchanged_request_below_trigger() -> None:
    agent = Agent(model="test-model", instructions="Be concise.")
    message = Message(role="user", content="Hello")
    tools = [ToolSchema(name="read_file", description="Read a file")]
    budget = ContextBudget(
        context_window_tokens=100_000,
        max_output_tokens=4_000,
    )

    plan = plan_context(
        agent=agent,
        messages=[message],
        tools=tools,
        budget=budget,
    )

    assert plan.compaction_triggered is False
    assert plan.cleared_tool_results == 0
    assert plan.summary_required is False
    assert plan.before is plan.after
    assert plan.request.model == "test-model"
    assert plan.request.max_tokens == 4_000
    assert plan.request.tools == tools
    assert plan.request.tools is not tools
    assert plan.request.messages[-1] == ProviderMessage(
        role="user",
        content="Hello",
    )


def test_plan_context_clears_old_results_and_remeasures() -> None:
    old_payload = "x" * 40_000
    tool_call = ToolCall(id="call_old", name="read_file")
    messages = [
        Message(role="user", content="Old turn"),
        Message(role="assistant", tool_calls=[tool_call]),
        Message(
            role="tool",
            name="read_file",
            tool_call_id=tool_call.id,
            content=old_payload,
        ),
        Message(role="assistant", content="Old answer"),
        Message(role="user", content="Recent turn one"),
        Message(role="assistant", content="Recent answer one"),
        Message(role="user", content="Recent turn two"),
    ]
    budget = ContextBudget(
        context_window_tokens=12_000,
        max_output_tokens=1_000,
        safety_margin_tokens=1_000,
    )

    plan = plan_context(
        agent=Agent(),
        messages=messages,
        tools=[],
        budget=budget,
    )

    compacted_result = next(
        message
        for message in plan.request.messages
        if message.tool_call_id == tool_call.id
    )
    assert plan.before.input_tokens >= budget.trigger_tokens
    assert plan.after.input_tokens < plan.before.input_tokens
    assert plan.after.input_tokens < budget.trigger_tokens
    assert plan.compaction_triggered is True
    assert plan.cleared_tool_results == 1
    assert plan.summary_required is False
    assert "result omitted from active context" in compacted_result.content
    assert messages[2].content == old_payload


def test_plan_context_requests_summary_when_cleanup_cannot_reach_trigger() -> None:
    budget = ContextBudget(
        context_window_tokens=12_000,
        max_output_tokens=1_000,
        safety_margin_tokens=1_000,
    )

    plan = plan_context(
        agent=Agent(),
        messages=[Message(role="user", content="x" * 40_000)],
        tools=[],
        budget=budget,
    )

    assert plan.before.input_tokens >= budget.trigger_tokens
    assert plan.after is plan.before
    assert plan.compaction_triggered is True
    assert plan.cleared_tool_results == 0
    assert plan.summary_required is True


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

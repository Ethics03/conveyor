from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from threading import Event as ThreadEvent

import pytest

from agent.approvals import DefaultApprovalPolicy, PolicyDecision, ToolCallDecision
from agent.context import MIN_CLEARABLE_TOOL_RESULT_CHARS
from agent.loop import (
    _block_run,
    _preflight_tool_calls,
    _start_run,
    run_agent,
)
from agent.models import (
    Agent,
    ApprovalDecision,
    ApprovalRequest,
    Message,
    ProviderReplayState,
    ProviderResponse,
    Session,
    ToolCall,
)
from providers.base import ModelLimits, ProviderRequest
from providers.fake import FakeProvider
from storage.store import Store
from tools.base import ExecutionContext, tool
from tools.registry import ToolRegistry
from tools.workspace import write_file


def _session_with_user_message(store: Store) -> Session:
    session = Session()
    store.save_session(session)
    store.save_message(
        Message(
            session_id=session.id,
            role="user",
            content="Complete the task.",
        )
    )
    return session


def test_preflight_tool_calls_preserves_batch_order(tmp_path) -> None:
    @tool(permission="read")
    def inspect_workspace() -> str:
        return ""

    @tool(permission="write")
    def update_workspace() -> str:
        return ""

    tool_calls = [
        ToolCall(id="call_read", name="inspect_workspace"),
        ToolCall(id="call_write", name="update_workspace"),
    ]

    decisions = _preflight_tool_calls(
        tool_calls=tool_calls,
        registry=ToolRegistry([inspect_workspace, update_workspace]),
        context=ExecutionContext(workspace=tmp_path),
        policy=DefaultApprovalPolicy(),
    )

    assert [item.tool_call.id for item in decisions] == ["call_read", "call_write"]
    assert [item.decision.action for item in decisions] == ["allow", "ask"]


def test_preflight_tool_calls_denies_unknown_tool(tmp_path) -> None:
    decisions = _preflight_tool_calls(
        tool_calls=[ToolCall(name="missing")],
        registry=ToolRegistry(),
        context=ExecutionContext(workspace=tmp_path),
        policy=DefaultApprovalPolicy(),
    )

    assert decisions[0].decision.action == "deny"
    assert decisions[0].decision.reason == "Unknown tool: missing"


def test_block_run_persists_pending_approvals_and_events() -> None:
    store = Store(":memory:")
    session = _session_with_user_message(store)
    run = _start_run(agent=Agent(), session=session, store=store)
    read_call = ToolCall(id="call_read", name="read_file")
    denied_call = ToolCall(id="call_denied", name="run_command")
    write_call = ToolCall(id="call_write", name="write_file")
    assistant_message = Message(
        session_id=session.id,
        run_id=run.id,
        role="assistant",
        tool_calls=[read_call, denied_call, write_call],
    )
    store.save_message(assistant_message)

    approvals = _block_run(
        run=run,
        final_message=assistant_message,
        decisions=[
            ToolCallDecision(read_call, PolicyDecision("allow")),
            ToolCallDecision(
                denied_call,
                PolicyDecision("deny", "command is forbidden by policy"),
            ),
            ToolCallDecision(
                write_call,
                PolicyDecision("ask", "write_file can modify workspace files"),
            ),
        ],
        iterations=1,
        store=store,
    )

    assert run.status == "blocked"
    assert len(approvals) == 1
    approval = approvals[0]
    assert approval.tool_call == write_call
    assert store.get_approval(approval.id) == approval
    assert [event.type for event in store.list_events(run_id=run.id)] == [
        "run.started",
        "approval.requested",
        "run.blocked",
    ]
    blocked_event = store.list_events(run_id=run.id)[-1]
    assert blocked_event.payload == {
        "approval_ids": [approval.id],
        "approval_count": 1,
        "iterations": 1,
    }


def test_block_run_requires_pending_approval() -> None:
    store = Store(":memory:")
    session = _session_with_user_message(store)
    run = _start_run(agent=Agent(), session=session, store=store)
    tool_call = ToolCall(name="read_file")
    assistant_message = Message(
        session_id=session.id,
        run_id=run.id,
        role="assistant",
        tool_calls=[tool_call],
    )

    with pytest.raises(ValueError, match="without pending approvals"):
        _block_run(
            run=run,
            final_message=assistant_message,
            decisions=[ToolCallDecision(tool_call, PolicyDecision("allow"))],
            iterations=1,
            store=store,
        )

    assert run.status == "running"


def test_run_agent_finishes_with_plain_response(tmp_path) -> None:
    store = Store(":memory:")
    session = _session_with_user_message(store)
    provider = FakeProvider([ProviderResponse.message("Done.")])

    outcome = run_agent(
        agent=Agent(),
        session=session,
        provider=provider,
        registry=ToolRegistry(),
        context=ExecutionContext(workspace=tmp_path),
        store=store,
    )

    assert outcome.run.status == "finished"
    assert outcome.iterations == 1
    assert outcome.final_message is not None
    assert outcome.final_message.content == "Done."
    assert [message.role for message in store.list_messages(session.id)] == [
        "user",
        "assistant",
    ]
    assert [event.type for event in store.list_events(run_id=outcome.run.id)] == [
        "run.started",
        "message.created",
        "run.finished",
    ]


def test_run_agent_sends_compacted_context_and_records_telemetry(tmp_path) -> None:
    store = Store(":memory:")
    session = Session()
    store.save_session(session)
    old_call = ToolCall(id="call_old", name="read_file")
    old_payload = "x" * (MIN_CLEARABLE_TOOL_RESULT_CHARS * 10)
    history = [
        Message(session_id=session.id, role="user", content="Old turn"),
        Message(
            session_id=session.id,
            role="assistant",
            tool_calls=[old_call],
        ),
        Message(
            session_id=session.id,
            role="tool",
            name="read_file",
            tool_call_id=old_call.id,
            content=old_payload,
        ),
        Message(session_id=session.id, role="assistant", content="Old answer"),
        Message(session_id=session.id, role="user", content="Recent turn one"),
        Message(session_id=session.id, role="assistant", content="Recent answer"),
        Message(session_id=session.id, role="user", content="Recent turn two"),
    ]
    for message in history:
        store.save_message(message)

    provider = FakeProvider(
        [ProviderResponse.message("Done with compacted context.")],
        model_limits=ModelLimits(12_000, 1_000),
    )
    outcome = run_agent(
        agent=Agent(),
        session=session,
        provider=provider,
        registry=ToolRegistry(),
        context=ExecutionContext(workspace=tmp_path),
        store=store,
    )

    request = provider.requests[0]
    compacted_result = next(
        message for message in request.messages if message.tool_call_id == old_call.id
    )
    telemetry = request.metadata["context_plan"]
    assert "result omitted from active context" in compacted_result.content
    assert telemetry["compaction_triggered"] is True
    assert telemetry["cleared_tool_results"] == 1
    assert telemetry["summary_required"] is False
    assert telemetry["input_tokens_after"] < telemetry["input_tokens_before"]
    assert request.max_tokens == 1_000
    assert store.list_messages(session.id)[2].content == old_payload
    assert [event.type for event in store.list_events(run_id=outcome.run.id)] == [
        "run.started",
        "context.compaction_planned",
        "message.created",
        "run.finished",
    ]


def test_run_agent_persists_native_compaction_checkpoint(tmp_path) -> None:
    store = Store(":memory:")
    session = _session_with_user_message(store)
    state = ProviderReplayState(
        provider="anthropic",
        items=(
            {
                "type": "compaction",
                "content": "<summary>Current task state.</summary>",
                "encrypted_content": "opaque-checkpoint",
            },
        ),
    )
    provider = FakeProvider(
        [
            ProviderResponse(
                content="Continued.",
                finish_reason="end_turn",
                replay_state=state,
            )
        ]
    )

    outcome = run_agent(
        agent=Agent(),
        session=session,
        provider=provider,
        registry=ToolRegistry(),
        context=ExecutionContext(workspace=tmp_path),
        store=store,
    )

    assert outcome.final_message is not None
    assert outcome.final_message.metadata["provider_replay_state"] == (
        state.to_metadata()
    )
    reopened = store.list_messages(session.id)
    assert reopened[-1].metadata["provider_replay_state"] == state.to_metadata()
    events = store.list_events(run_id=outcome.run.id)
    assert [event.type for event in events] == [
        "run.started",
        "message.created",
        "context.compacted",
        "run.finished",
    ]
    assert events[2].payload == {
        "provider": "anthropic",
        "item_count": 1,
    }


def test_run_agent_can_use_main_thread_store_from_worker(tmp_path) -> None:
    store = Store(":memory:")
    session = _session_with_user_message(store)
    provider = FakeProvider([ProviderResponse.message("Done from worker.")])

    with ThreadPoolExecutor(max_workers=1) as executor:
        outcome = executor.submit(
            run_agent,
            agent=Agent(),
            session=session,
            provider=provider,
            registry=ToolRegistry(),
            context=ExecutionContext(workspace=tmp_path),
            store=store,
        ).result()

    assert outcome.run.status == "finished"
    assert store.get_run(outcome.run.id) == outcome.run
    assert [event.type for event in store.list_events(run_id=outcome.run.id)] == [
        "run.started",
        "message.created",
        "run.finished",
    ]


def test_run_agent_executes_tool_and_continues(tmp_path) -> None:
    @tool(permission="read")
    def echo(value: str) -> str:
        return value

    store = Store(":memory:")
    session = _session_with_user_message(store)
    provider = FakeProvider(
        [
            ProviderResponse.tool(
                "echo",
                {"value": "hello"},
                tool_call_id="call_echo",
            ),
            ProviderResponse.message("Tool complete."),
        ]
    )

    outcome = run_agent(
        agent=Agent(tools=["echo"]),
        session=session,
        provider=provider,
        registry=ToolRegistry([echo]),
        context=ExecutionContext(workspace=tmp_path),
        store=store,
    )

    assert outcome.run.status == "finished"
    assert outcome.iterations == 2
    messages = store.list_messages(session.id)
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[2].content == "hello"
    assert messages[2].tool_call_id == "call_echo"
    assert provider.requests[1].messages[-1].role == "tool"
    assert provider.requests[1].messages[-1].tool_call_id == "call_echo"


def test_run_agent_executes_parallel_safe_tools_concurrently_in_call_order(
    tmp_path,
) -> None:
    first_started = ThreadEvent()
    second_started = ThreadEvent()
    release_first = ThreadEvent()

    @tool(permission="read", parallel_safe=True)
    def first() -> str:
        first_started.set()
        if not second_started.wait(timeout=1):
            raise RuntimeError("second tool did not start concurrently")
        if not release_first.wait(timeout=1):
            raise RuntimeError("first tool was not released")
        return "first"

    @tool(permission="read", parallel_safe=True)
    def second() -> str:
        second_started.set()
        if not first_started.wait(timeout=1):
            raise RuntimeError("first tool did not start concurrently")
        release_first.set()
        return "second"

    store = Store(":memory:")
    session = _session_with_user_message(store)
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[
                    ToolCall(id="call_first", name="first"),
                    ToolCall(id="call_second", name="second"),
                ]
            ),
            ProviderResponse.message("Both reads complete."),
        ]
    )

    outcome = run_agent(
        agent=Agent(tools=["first", "second"]),
        session=session,
        provider=provider,
        registry=ToolRegistry([first, second]),
        context=ExecutionContext(workspace=tmp_path),
        store=store,
    )

    assert outcome.run.status == "finished"
    tool_messages = [
        message for message in store.list_messages(session.id) if message.role == "tool"
    ]
    assert [message.name for message in tool_messages] == ["first", "second"]
    assert [message.content for message in tool_messages] == ["first", "second"]
    assert [message.name for message in provider.requests[1].messages[-2:]] == [
        "first",
        "second",
    ]


def test_run_agent_uses_non_parallel_tool_as_batch_barrier(tmp_path) -> None:
    before_barrier = Barrier(2)
    after_barrier = Barrier(2)
    before_finished: set[str] = set()
    barrier_finished = ThreadEvent()

    @tool(permission="read", parallel_safe=True)
    def before_one() -> str:
        _ = before_barrier.wait(timeout=1)
        before_finished.add("one")
        return "before-one"

    @tool(permission="read", parallel_safe=True)
    def before_two() -> str:
        _ = before_barrier.wait(timeout=1)
        before_finished.add("two")
        return "before-two"

    @tool(permission="read")
    def ordered_step() -> str:
        if before_finished != {"one", "two"}:
            raise RuntimeError("ordered step crossed the preceding read batch")
        barrier_finished.set()
        return "ordered"

    @tool(permission="read", parallel_safe=True)
    def after_one() -> str:
        if not barrier_finished.is_set():
            raise RuntimeError("read crossed the ordered step")
        _ = after_barrier.wait(timeout=1)
        return "after-one"

    @tool(permission="read", parallel_safe=True)
    def after_two() -> str:
        if not barrier_finished.is_set():
            raise RuntimeError("read crossed the ordered step")
        _ = after_barrier.wait(timeout=1)
        return "after-two"

    tools = [before_one, before_two, ordered_step, after_one, after_two]
    store = Store(":memory:")
    session = _session_with_user_message(store)
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[
                    ToolCall(id="call_before_one", name="before_one"),
                    ToolCall(id="call_before_two", name="before_two"),
                    ToolCall(id="call_ordered", name="ordered_step"),
                    ToolCall(id="call_after_one", name="after_one"),
                    ToolCall(id="call_after_two", name="after_two"),
                ]
            ),
            ProviderResponse.message("Ordered work complete."),
        ]
    )

    outcome = run_agent(
        agent=Agent(tools=[item.name for item in tools]),
        session=session,
        provider=provider,
        registry=ToolRegistry(tools),
        context=ExecutionContext(workspace=tmp_path),
        store=store,
    )

    assert outcome.run.status == "finished"
    tool_messages = [
        message for message in store.list_messages(session.id) if message.role == "tool"
    ]
    assert [message.name for message in tool_messages] == [
        "before_one",
        "before_two",
        "ordered_step",
        "after_one",
        "after_two",
    ]
    assert all(message.metadata["ok"] is True for message in tool_messages)


def test_run_agent_preserves_other_parallel_results_when_one_tool_fails(
    tmp_path,
) -> None:
    both_started = Barrier(2)

    @tool(permission="read", parallel_safe=True)
    def failing_read() -> str:
        _ = both_started.wait(timeout=1)
        raise RuntimeError("read failed")

    @tool(permission="read", parallel_safe=True)
    def successful_read() -> str:
        _ = both_started.wait(timeout=1)
        return "available"

    store = Store(":memory:")
    session = _session_with_user_message(store)
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[
                    ToolCall(id="call_failure", name="failing_read"),
                    ToolCall(id="call_success", name="successful_read"),
                ]
            ),
            ProviderResponse.message("Handled partial results."),
        ]
    )

    outcome = run_agent(
        agent=Agent(tools=["failing_read", "successful_read"]),
        session=session,
        provider=provider,
        registry=ToolRegistry([failing_read, successful_read]),
        context=ExecutionContext(workspace=tmp_path),
        store=store,
    )

    assert outcome.run.status == "finished"
    tool_messages = [
        message for message in store.list_messages(session.id) if message.role == "tool"
    ]
    assert [message.metadata["ok"] for message in tool_messages] == [False, True]
    assert [message.content for message in tool_messages] == [
        "read failed",
        "available",
    ]


def test_run_agent_waits_for_approval_and_continues(tmp_path) -> None:
    executions: list[str] = []

    @tool(permission="read")
    def inspect_workspace() -> str:
        executions.append("read")
        return "read"

    @tool(permission="write")
    def update_workspace() -> str:
        executions.append("write")
        return "write"

    store = Store(":memory:")
    session = _session_with_user_message(store)
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[
                    ToolCall(id="call_read", name="inspect_workspace"),
                    ToolCall(id="call_write", name="update_workspace"),
                ]
            ),
            ProviderResponse.message("Updates complete."),
        ]
    )
    requested: list[ApprovalRequest] = []

    def approve(approval: ApprovalRequest) -> ApprovalDecision:
        requested.append(approval)
        assert executions == []
        persisted_run = store.get_run(approval.run_id)
        assert persisted_run is not None
        assert persisted_run.status == "blocked"
        return "approved"

    outcome = run_agent(
        agent=Agent(tools=["inspect_workspace", "update_workspace"]),
        session=session,
        provider=provider,
        registry=ToolRegistry([inspect_workspace, update_workspace]),
        context=ExecutionContext(workspace=tmp_path),
        store=store,
        approval_callback=approve,
    )

    assert outcome.run.status == "finished"
    assert executions == ["read", "write"]
    assert len(requested) == 1
    assert requested[0].tool_call.id == "call_write"
    persisted_approval = store.get_approval(requested[0].id)
    assert persisted_approval is not None
    assert persisted_approval.status == "approved"
    assert len(provider.requests) == 2
    assert [message.role for message in store.list_messages(session.id)] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert [event.type for event in store.list_events(run_id=outcome.run.id)] == [
        "run.started",
        "message.created",
        "approval.requested",
        "run.blocked",
        "approval.resolved",
        "run.resumed",
        "tool.started",
        "message.created",
        "tool.finished",
        "tool.started",
        "message.created",
        "tool.finished",
        "message.created",
        "run.finished",
    ]


def test_run_agent_approves_workspace_write_before_mutation(tmp_path) -> None:
    store = Store(":memory:")
    session = _session_with_user_message(store)
    provider = FakeProvider(
        [
            ProviderResponse.tool(
                "write_file",
                {"path": "result.txt", "content": "approved\n"},
                tool_call_id="call_write_file",
            ),
            ProviderResponse.message("File written."),
        ]
    )

    def approve(_: ApprovalRequest) -> ApprovalDecision:
        assert (tmp_path / "result.txt").exists() is False
        return "approved"

    outcome = run_agent(
        agent=Agent(tools=["write_file"]),
        session=session,
        provider=provider,
        registry=ToolRegistry([write_file]),
        context=ExecutionContext(workspace=tmp_path),
        store=store,
        approval_callback=approve,
    )

    assert outcome.run.status == "finished"
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "approved\n"


def test_run_agent_returns_user_denial_to_provider(tmp_path) -> None:
    executions: list[str] = []

    @tool(permission="write")
    def update_workspace() -> str:
        executions.append("write")
        return "write"

    store = Store(":memory:")
    session = _session_with_user_message(store)
    provider = FakeProvider(
        [
            ProviderResponse.tool(
                "update_workspace",
                {},
                tool_call_id="call_write",
            ),
            ProviderResponse.message("The update was denied."),
        ]
    )

    outcome = run_agent(
        agent=Agent(tools=["update_workspace"]),
        session=session,
        provider=provider,
        registry=ToolRegistry([update_workspace]),
        context=ExecutionContext(workspace=tmp_path),
        store=store,
        approval_callback=lambda _: "denied",
    )

    assert outcome.run.status == "finished"
    assert executions == []
    tool_message = store.list_messages(session.id)[2]
    assert tool_message.content == "Tool call denied by user"
    assert tool_message.metadata["approval_status"] == "denied"
    assert provider.requests[1].messages[-1].tool_call_id == "call_write"
    event_types = [event.type for event in store.list_events(run_id=outcome.run.id)]
    assert "run.blocked" in event_types
    assert "run.resumed" in event_types
    assert "tool.started" not in event_types


def test_run_agent_cleans_up_approvals_when_callback_fails(tmp_path) -> None:
    executions: list[str] = []

    @tool(permission="write")
    def update_first() -> str:
        executions.append("first")
        return "first"

    @tool(permission="write")
    def update_second() -> str:
        executions.append("second")
        return "second"

    store = Store(":memory:")
    session = _session_with_user_message(store)
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[
                    ToolCall(id="call_first", name="update_first"),
                    ToolCall(id="call_second", name="update_second"),
                ]
            ),
        ]
    )
    callback_calls = 0

    def fail_on_second(_: ApprovalRequest) -> ApprovalDecision:
        nonlocal callback_calls
        callback_calls += 1
        if callback_calls == 1:
            return "approved"
        raise RuntimeError("approval client disconnected")

    with pytest.raises(RuntimeError, match="approval client disconnected"):
        _ = run_agent(
            agent=Agent(tools=["update_first", "update_second"]),
            session=session,
            provider=provider,
            registry=ToolRegistry([update_first, update_second]),
            context=ExecutionContext(workspace=tmp_path),
            store=store,
            approval_callback=fail_on_second,
        )

    assert executions == []
    run = store.list_runs(session.id)[0]
    assert run.status == "failed"
    assert run.error == "approval client disconnected"
    assert [approval.status for approval in store.list_approvals(run_id=run.id)] == [
        "approved",
        "denied",
    ]
    assert [event.type for event in store.list_events(run_id=run.id)] == [
        "run.started",
        "message.created",
        "approval.requested",
        "approval.requested",
        "run.blocked",
        "approval.resolved",
        "approval.resolved",
        "run.failed",
    ]


def test_run_agent_denies_approval_without_callback(tmp_path) -> None:
    executions: list[str] = []

    @tool(permission="write")
    def update_workspace() -> str:
        executions.append("write")
        return "write"

    store = Store(":memory:")
    session = _session_with_user_message(store)
    provider = FakeProvider(
        [
            ProviderResponse.tool("update_workspace", {}),
            ProviderResponse.message("No approval handler was available."),
        ]
    )

    outcome = run_agent(
        agent=Agent(tools=["update_workspace"]),
        session=session,
        provider=provider,
        registry=ToolRegistry([update_workspace]),
        context=ExecutionContext(workspace=tmp_path),
        store=store,
    )

    assert outcome.run.status == "finished"
    assert executions == []
    assert store.list_approvals(run_id=outcome.run.id)[0].status == "denied"
    tool_message = store.list_messages(session.id)[2]
    assert tool_message.content == (
        "Tool call denied because no approval callback is configured"
    )


def test_run_agent_returns_policy_denial_to_provider(tmp_path) -> None:
    store = Store(":memory:")
    session = _session_with_user_message(store)
    provider = FakeProvider(
        [
            ProviderResponse.tool("missing", {}, tool_call_id="call_missing"),
            ProviderResponse.message("I cannot use that tool."),
        ]
    )

    outcome = run_agent(
        agent=Agent(),
        session=session,
        provider=provider,
        registry=ToolRegistry(),
        context=ExecutionContext(workspace=tmp_path),
        store=store,
    )

    assert outcome.run.status == "finished"
    messages = store.list_messages(session.id)
    assert messages[2].role == "tool"
    assert messages[2].content == "Unknown tool: missing"
    assert messages[2].metadata == {
        "ok": False,
        "policy_action": "deny",
    }
    assert provider.requests[1].messages[-1].tool_call_id == "call_missing"


def test_run_agent_fails_at_iteration_limit(tmp_path) -> None:
    @tool(permission="read")
    def echo(value: str) -> str:
        return value

    store = Store(":memory:")
    session = _session_with_user_message(store)
    provider = FakeProvider(
        [
            ProviderResponse.tool("echo", {"value": "again"}),
        ]
    )

    outcome = run_agent(
        agent=Agent(tools=["echo"]),
        session=session,
        provider=provider,
        registry=ToolRegistry([echo]),
        context=ExecutionContext(workspace=tmp_path),
        store=store,
        max_iterations=1,
    )

    assert outcome.run.status == "failed"
    assert outcome.run.error == "Maximum iterations exceeded: 1"
    assert outcome.iterations == 1
    assert store.list_events(run_id=outcome.run.id)[-1].type == "run.failed"


def test_run_agent_persists_provider_failure(tmp_path) -> None:
    class FailingProvider:
        name = "failing"

        def model_limits(self, model: str | None = None) -> ModelLimits:
            return ModelLimits(200_000, 4_096)

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            raise RuntimeError("provider unavailable")

        def close(self) -> None:
            pass

    store = Store(":memory:")
    session = _session_with_user_message(store)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        run_agent(
            agent=Agent(),
            session=session,
            provider=FailingProvider(),
            registry=ToolRegistry(),
            context=ExecutionContext(workspace=tmp_path),
            store=store,
        )

    run = store.list_runs(session.id)[0]
    assert run.status == "failed"
    assert run.error == "provider unavailable"
    assert store.list_events(run_id=run.id)[-1].type == "run.failed"

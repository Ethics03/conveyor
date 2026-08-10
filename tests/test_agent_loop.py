from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.approvals import DefaultApprovalPolicy, PolicyDecision, ToolCallDecision
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
    ProviderResponse,
    Session,
    ToolCall,
)
from providers.base import ProviderRequest
from providers.fake import FakeProvider
from storage.store import Store
from tools.base import ExecutionContext, tool
from tools.registry import ToolRegistry


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
    provider = FakeProvider([
        ProviderResponse.tool(
            "echo",
            {"value": "hello"},
            tool_call_id="call_echo",
        ),
        ProviderResponse.message("Tool complete."),
    ])

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
    provider = FakeProvider([
        ProviderResponse(
            tool_calls=[
                ToolCall(id="call_read", name="inspect_workspace"),
                ToolCall(id="call_write", name="update_workspace"),
            ]
        ),
        ProviderResponse.message("Updates complete."),
    ])
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


def test_run_agent_returns_user_denial_to_provider(tmp_path) -> None:
    executions: list[str] = []

    @tool(permission="write")
    def update_workspace() -> str:
        executions.append("write")
        return "write"

    store = Store(":memory:")
    session = _session_with_user_message(store)
    provider = FakeProvider([
        ProviderResponse.tool(
            "update_workspace",
            {},
            tool_call_id="call_write",
        ),
        ProviderResponse.message("The update was denied."),
    ])

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
    provider = FakeProvider([
        ProviderResponse(
            tool_calls=[
                ToolCall(id="call_first", name="update_first"),
                ToolCall(id="call_second", name="update_second"),
            ]
        ),
    ])
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
    provider = FakeProvider([
        ProviderResponse.tool("update_workspace", {}),
        ProviderResponse.message("No approval handler was available."),
    ])

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
    provider = FakeProvider([
        ProviderResponse.tool("missing", {}, tool_call_id="call_missing"),
        ProviderResponse.message("I cannot use that tool."),
    ])

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
    provider = FakeProvider([
        ProviderResponse.tool("echo", {"value": "again"}),
    ])

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

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            raise RuntimeError("provider unavailable")

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

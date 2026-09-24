from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from typing import cast

import pytest

from agent.models import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    Event,
    Message,
    Run,
    Session,
    ToolCall,
    utc_now,
)
from storage.store import Store


@pytest.fixture
async def store() -> AsyncIterator[Store]:
    opened = await Store.open(":memory:")
    try:
        yield opened
    finally:
        await opened.close()


async def test_session_roundtrip(store: Store) -> None:
    session = Session(title="hello", title_source="user")
    await store.save_session(session)

    loaded = await store.get_session(session.id)
    assert loaded == session


async def test_session_upsert_updates(store: Store) -> None:
    session = Session(title="before")
    await store.save_session(session)

    session.title = "after"
    session.status = "archived"
    await store.save_session(session)

    loaded = await store.get_session(session.id)
    assert loaded is not None
    assert loaded.title == "before"
    assert loaded.status == "archived"
    assert len(await store.list_sessions()) == 1


async def test_auto_title_only_replaces_default_title(store: Store) -> None:
    session = Session()
    await store.save_session(session)

    assert await store.set_auto_title(session.id, "  Fix   provider timeout ") is True
    loaded = await store.get_session(session.id)
    assert loaded is not None
    assert loaded.title == "Fix provider timeout"
    assert loaded.title_source == "auto"
    assert await store.set_auto_title(session.id, "Replace it again") is False
    unchanged = await store.get_session(session.id)
    assert unchanged is not None
    assert unchanged.title == "Fix provider timeout"


async def test_user_title_replaces_auto_title_and_cannot_be_overwritten(
    store: Store,
) -> None:
    session = Session()
    await store.save_session(session)
    assert await store.set_auto_title(session.id, "Automatic title") is True

    assert await store.set_session_title(session.id, "User title") is True
    assert await store.set_auto_title(session.id, "Late automatic title") is False

    loaded = await store.get_session(session.id)
    assert loaded is not None
    assert loaded.title == "User title"
    assert loaded.title_source == "user"


async def test_set_session_title_returns_false_for_unknown_session(
    store: Store,
) -> None:
    assert await store.set_session_title("ses_missing", "Missing") is False


async def test_store_methods_are_awaitable(store: Store) -> None:
    session = Session(title="worker")

    await store.save_session(session)
    loaded = await store.get_session(session.id)

    assert loaded == session


async def test_store_serializes_concurrent_writes(store: Store) -> None:
    sessions = [Session(title=f"worker-{index}") for index in range(20)]

    await asyncio.gather(*(store.save_session(session) for session in sessions))

    assert await store.list_sessions() == sessions


async def test_cancelled_write_settles_before_returning(
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session()
    started = asyncio.Event()
    release = asyncio.Event()
    insert_session = store._execute

    async def delayed_execute(
        statement: str,
        parameters: tuple[object, ...] = (),
    ) -> int:
        if "INSERT INTO sessions" in statement:
            started.set()
            await release.wait()
        return await insert_session(statement, parameters)

    monkeypatch.setattr(store, "_execute", delayed_execute)
    save = asyncio.create_task(store.save_session(session))
    await started.wait()
    save.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await save

    assert await store.get_session(session.id) == session


async def test_run_roundtrip_with_parent(store: Store) -> None:
    session = Session()
    await store.save_session(session)
    parent = Run(session_id=session.id, agent_id="agent_1")
    await store.save_run(parent)
    child = Run(session_id=session.id, agent_id="agent_1", parent_run_id=parent.id)
    await store.save_run(child)

    loaded = await store.get_run(child.id)
    assert loaded is not None
    assert loaded == child
    assert loaded.parent_run_id == parent.id
    assert await store.list_runs(session.id) == [parent, child]


async def test_message_ordering_and_metadata(store: Store) -> None:
    session = Session()
    await store.save_session(session)
    first = Message(session_id=session.id, role="user", content="hi", metadata={"a": 1})
    second = Message(session_id=session.id, role="assistant", content="hello")
    await store.save_message(first)
    await store.save_message(second)

    loaded = await store.list_messages(session.id)
    assert loaded == [first, second]
    assert loaded[0].metadata == {"a": 1}


async def test_tool_messages_roundtrip(store: Store) -> None:
    session = Session()
    await store.save_session(session)
    tool_call = ToolCall(
        id="call_readme",
        name="read_file",
        arguments={"path": "README.md"},
    )
    assistant = Message(
        session_id=session.id,
        role="assistant",
        content="I will inspect the file.",
        tool_calls=[tool_call],
    )
    tool_result = Message(
        session_id=session.id,
        role="tool",
        content='{"content": "Conveyor"}',
        name="read_file",
        tool_call_id=tool_call.id,
    )

    await store.save_message(assistant)
    await store.save_message(tool_result)

    assert await store.list_messages(session.id) == [assistant, tool_result]


async def test_events_append_and_filter(store: Store) -> None:
    session = Session()
    await store.save_session(session)
    run = Run(session_id=session.id, agent_id="agent_1")
    await store.save_run(run)

    session_event = Event(type="session.created", session_id=session.id)
    run_event = Event(type="run.started", session_id=session.id, run_id=run.id)
    await store.append_event(session_event)
    await store.append_event(run_event)

    assert await store.list_events(session_id=session.id) == [session_event, run_event]
    assert await store.list_events(run_id=run.id) == [run_event]


async def test_event_foreign_keys_enforced(store: Store) -> None:
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        await store.append_event(Event(type="run.started", run_id="run_missing"))


async def test_approval_roundtrip(store: Store) -> None:
    session = Session()
    await store.save_session(session)
    run = Run(session_id=session.id, agent_id="agent_1")
    await store.save_run(run)

    approval = ApprovalRequest(
        session_id=session.id,
        run_id=run.id,
        tool_call=ToolCall(name="run_command", arguments={"command": "rm -rf /tmp/x"}),
        reason="dangerous tool",
    )
    await store.save_approval(approval)

    loaded = await store.get_approval(approval.id)
    assert loaded is not None
    assert loaded == approval
    assert loaded.status == "pending"
    assert loaded.tool_call is not None
    assert loaded.tool_call.arguments == {"command": "rm -rf /tmp/x"}


@pytest.mark.parametrize("status", ["approved", "denied"])
async def test_save_approval_rejects_resolved_status(
    store: Store,
    status: ApprovalStatus,
) -> None:
    session = Session()
    await store.save_session(session)
    run = Run(session_id=session.id, agent_id="agent_1")
    await store.save_run(run)
    approval = ApprovalRequest(
        session_id=session.id,
        run_id=run.id,
        tool_call=ToolCall(name="write_file"),
        status=status,
    )

    with pytest.raises(ValueError, match="must have pending status"):
        await store.save_approval(approval)


async def test_save_approval_rejects_resolved_at(store: Store) -> None:
    session = Session()
    await store.save_session(session)
    run = Run(session_id=session.id, agent_id="agent_1")
    await store.save_run(run)
    approval = ApprovalRequest(
        session_id=session.id,
        run_id=run.id,
        tool_call=ToolCall(name="write_file"),
        resolved_at=utc_now(),
    )

    with pytest.raises(ValueError, match="cannot have resolved_at"):
        await store.save_approval(approval)


async def test_block_run_rolls_back_all_state_on_failure(store: Store) -> None:
    session = Session()
    await store.save_session(session)
    run = Run(session_id=session.id, agent_id="agent_1", status="running")
    await store.save_run(run)
    approval = ApprovalRequest(
        session_id=session.id,
        run_id=run.id,
        tool_call=ToolCall(name="write_file"),
        reason="write access requires approval",
    )
    run.status = "blocked"
    duplicate_id = "evt_duplicate"

    with pytest.raises(sqlite3.IntegrityError):
        await store.block_run(
            run=run,
            approvals=[approval],
            events=[
                Event(id=duplicate_id, type="approval.requested", run_id=run.id),
                Event(id=duplicate_id, type="run.blocked", run_id=run.id),
            ],
        )

    persisted_run = await store.get_run(run.id)
    assert persisted_run is not None
    assert persisted_run.status == "running"
    assert await store.list_approvals(run_id=run.id) == []
    assert await store.list_events(run_id=run.id) == []


async def test_resume_run_updates_state_and_appends_event(store: Store) -> None:
    session = Session()
    await store.save_session(session)
    run = Run(session_id=session.id, agent_id="agent_1", status="blocked")
    await store.save_run(run)
    run.status = "running"
    event = Event(type="run.resumed", session_id=session.id, run_id=run.id)

    await store.resume_run(run=run, event=event)

    assert await store.get_run(run.id) == run
    assert await store.list_events(run_id=run.id) == [event]


async def test_resume_run_rejects_non_blocked_run(store: Store) -> None:
    session = Session()
    await store.save_session(session)
    run = Run(session_id=session.id, agent_id="agent_1", status="running")
    await store.save_run(run)

    with pytest.raises(ValueError, match="is not blocked"):
        await store.resume_run(
            run=run,
            event=Event(type="run.resumed", run_id=run.id),
        )

    assert await store.list_events(run_id=run.id) == []


@pytest.mark.parametrize("decision", ["approved", "denied"])
async def test_resolve_approval_is_atomic_and_idempotent(
    store: Store,
    decision: ApprovalDecision,
) -> None:
    session = Session()
    await store.save_session(session)
    run = Run(session_id=session.id, agent_id="agent_1")
    await store.save_run(run)
    approval = ApprovalRequest(
        session_id=session.id,
        run_id=run.id,
        tool_call=ToolCall(name="write_file"),
    )
    await store.save_approval(approval)

    resolved = await store.resolve_approval(approval.id, decision)

    assert resolved.status == decision
    assert resolved.resolved_at is not None
    assert await store.resolve_approval(approval.id, decision) == resolved
    events = await store.list_events(run_id=run.id)
    assert len(events) == 1
    assert events[0].type == "approval.resolved"
    assert events[0].payload == {
        "approval_id": approval.id,
        "decision": decision,
        "tool_call_id": approval.tool_call.id,
    }

    conflicting: ApprovalDecision = "denied" if decision == "approved" else "approved"
    with pytest.raises(ValueError, match=f"already resolved as {decision}"):
        _ = await store.resolve_approval(approval.id, conflicting)

    assert await store.get_approval(approval.id) == resolved


async def test_resolve_approval_rejects_unknown_id(store: Store) -> None:
    with pytest.raises(KeyError, match="Unknown approval: appr_missing"):
        _ = await store.resolve_approval("appr_missing", "approved")


async def test_resolve_approval_rejects_invalid_decision(store: Store) -> None:
    session = Session()
    await store.save_session(session)
    run = Run(session_id=session.id, agent_id="agent_1")
    await store.save_run(run)
    approval = ApprovalRequest(
        session_id=session.id,
        run_id=run.id,
        tool_call=ToolCall(name="write_file"),
    )
    await store.save_approval(approval)

    invalid = cast(ApprovalDecision, "invalid")
    with pytest.raises(ValueError, match="Invalid approval decision"):
        _ = await store.resolve_approval(approval.id, invalid)

    persisted = await store.get_approval(approval.id)
    assert persisted is not None
    assert persisted.status == "pending"
    assert await store.list_events(run_id=run.id) == []


async def test_list_approvals_filters_by_run_and_status(store: Store) -> None:
    session = Session()
    await store.save_session(session)
    first_run = Run(session_id=session.id, agent_id="agent_1")
    second_run = Run(session_id=session.id, agent_id="agent_1")
    await store.save_run(first_run)
    await store.save_run(second_run)

    pending_first = ApprovalRequest(
        id="appr_a",
        session_id=session.id,
        run_id=first_run.id,
        tool_call=ToolCall(id="call_a", name="write_file"),
        reason="first pending",
    )
    approved_first = ApprovalRequest(
        id="appr_b",
        session_id=session.id,
        run_id=first_run.id,
        tool_call=ToolCall(id="call_b", name="write_file"),
        reason="first approved",
    )
    pending_second = ApprovalRequest(
        id="appr_c",
        session_id=session.id,
        run_id=second_run.id,
        tool_call=ToolCall(id="call_c", name="write_file"),
        reason="second pending",
    )
    for approval in (pending_first, approved_first, pending_second):
        approval.created_at = pending_first.created_at
        await store.save_approval(approval)
    approved_first = await store.resolve_approval(approved_first.id, "approved")

    assert await store.list_approvals() == [
        pending_first,
        approved_first,
        pending_second,
    ]
    assert await store.list_approvals(run_id=first_run.id) == [
        pending_first,
        approved_first,
    ]
    assert await store.list_approvals(status="pending") == [
        pending_first,
        pending_second,
    ]
    assert await store.list_approvals(
        run_id=first_run.id,
        status="approved",
    ) == [approved_first]


async def test_persistence_across_reopen(tmp_path) -> None:
    db_path = tmp_path / "conveyor.db"
    store = await Store.open(db_path)
    session = Session(title="durable")
    await store.save_session(session)
    await store.append_event(Event(type="session.created", session_id=session.id))
    await store.close()

    reopened = await Store.open(db_path)
    assert await reopened.get_session(session.id) == session
    events = await reopened.list_events(session_id=session.id)
    assert len(events) == 1
    assert events[0].type == "session.created"
    await reopened.close()

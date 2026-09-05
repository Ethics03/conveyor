from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
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
def store() -> Store:
    return Store(":memory:")


def test_session_roundtrip(store: Store) -> None:
    session = Session(title="hello")
    store.save_session(session)

    loaded = store.get_session(session.id)
    assert loaded == session


def test_session_upsert_updates(store: Store) -> None:
    session = Session(title="before")
    store.save_session(session)

    session.title = "after"
    session.status = "archived"
    store.save_session(session)

    loaded = store.get_session(session.id)
    assert loaded is not None
    assert loaded.title == "after"
    assert loaded.status == "archived"
    assert len(store.list_sessions()) == 1


def test_store_can_be_used_from_worker_thread(store: Store) -> None:
    session = Session(title="worker")

    with ThreadPoolExecutor(max_workers=1) as executor:
        saved = executor.submit(store.save_session, session)
        saved.result()
        loaded = executor.submit(store.get_session, session.id).result()

    assert loaded == session


def test_store_serializes_concurrent_writes(store: Store) -> None:
    sessions = [Session(title=f"worker-{index}") for index in range(20)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(store.save_session, session) for session in sessions]
        for future in futures:
            future.result()

    assert store.list_sessions() == sessions


def test_run_roundtrip_with_parent(store: Store) -> None:
    session = Session()
    store.save_session(session)
    parent = Run(session_id=session.id, agent_id="agent_1")
    store.save_run(parent)
    child = Run(session_id=session.id, agent_id="agent_1", parent_run_id=parent.id)
    store.save_run(child)

    loaded = store.get_run(child.id)
    assert loaded is not None
    assert loaded == child
    assert loaded.parent_run_id == parent.id
    assert store.list_runs(session.id) == [parent, child]


def test_message_ordering_and_metadata(store: Store) -> None:
    session = Session()
    store.save_session(session)
    first = Message(session_id=session.id, role="user", content="hi", metadata={"a": 1})
    second = Message(session_id=session.id, role="assistant", content="hello")
    store.save_message(first)
    store.save_message(second)

    loaded = store.list_messages(session.id)
    assert loaded == [first, second]
    assert loaded[0].metadata == {"a": 1}


def test_tool_messages_roundtrip(store: Store) -> None:
    session = Session()
    store.save_session(session)
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

    store.save_message(assistant)
    store.save_message(tool_result)

    assert store.list_messages(session.id) == [assistant, tool_result]


def test_events_append_and_filter(store: Store) -> None:
    session = Session()
    store.save_session(session)
    run = Run(session_id=session.id, agent_id="agent_1")
    store.save_run(run)

    session_event = Event(type="session.created", session_id=session.id)
    run_event = Event(type="run.started", session_id=session.id, run_id=run.id)
    store.append_event(session_event)
    store.append_event(run_event)

    assert store.list_events(session_id=session.id) == [session_event, run_event]
    assert store.list_events(run_id=run.id) == [run_event]


def test_event_foreign_keys_enforced(store: Store) -> None:
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        store.append_event(Event(type="run.started", run_id="run_missing"))


def test_approval_roundtrip(store: Store) -> None:
    session = Session()
    store.save_session(session)
    run = Run(session_id=session.id, agent_id="agent_1")
    store.save_run(run)

    approval = ApprovalRequest(
        session_id=session.id,
        run_id=run.id,
        tool_call=ToolCall(name="run_command", arguments={"command": "rm -rf /tmp/x"}),
        reason="dangerous tool",
    )
    store.save_approval(approval)

    loaded = store.get_approval(approval.id)
    assert loaded is not None
    assert loaded == approval
    assert loaded.status == "pending"
    assert loaded.tool_call is not None
    assert loaded.tool_call.arguments == {"command": "rm -rf /tmp/x"}


@pytest.mark.parametrize("status", ["approved", "denied"])
def test_save_approval_rejects_resolved_status(
    store: Store,
    status: ApprovalStatus,
) -> None:
    session = Session()
    store.save_session(session)
    run = Run(session_id=session.id, agent_id="agent_1")
    store.save_run(run)
    approval = ApprovalRequest(
        session_id=session.id,
        run_id=run.id,
        tool_call=ToolCall(name="write_file"),
        status=status,
    )

    with pytest.raises(ValueError, match="must have pending status"):
        store.save_approval(approval)


def test_save_approval_rejects_resolved_at(store: Store) -> None:
    session = Session()
    store.save_session(session)
    run = Run(session_id=session.id, agent_id="agent_1")
    store.save_run(run)
    approval = ApprovalRequest(
        session_id=session.id,
        run_id=run.id,
        tool_call=ToolCall(name="write_file"),
        resolved_at=utc_now(),
    )

    with pytest.raises(ValueError, match="cannot have resolved_at"):
        store.save_approval(approval)


def test_block_run_rolls_back_all_state_on_failure(store: Store) -> None:
    session = Session()
    store.save_session(session)
    run = Run(session_id=session.id, agent_id="agent_1", status="running")
    store.save_run(run)
    approval = ApprovalRequest(
        session_id=session.id,
        run_id=run.id,
        tool_call=ToolCall(name="write_file"),
        reason="write access requires approval",
    )
    run.status = "blocked"
    duplicate_id = "evt_duplicate"

    with pytest.raises(sqlite3.IntegrityError):
        store.block_run(
            run=run,
            approvals=[approval],
            events=[
                Event(id=duplicate_id, type="approval.requested", run_id=run.id),
                Event(id=duplicate_id, type="run.blocked", run_id=run.id),
            ],
        )

    persisted_run = store.get_run(run.id)
    assert persisted_run is not None
    assert persisted_run.status == "running"
    assert store.list_approvals(run_id=run.id) == []
    assert store.list_events(run_id=run.id) == []


def test_resume_run_updates_state_and_appends_event(store: Store) -> None:
    session = Session()
    store.save_session(session)
    run = Run(session_id=session.id, agent_id="agent_1", status="blocked")
    store.save_run(run)
    run.status = "running"
    event = Event(type="run.resumed", session_id=session.id, run_id=run.id)

    store.resume_run(run=run, event=event)

    assert store.get_run(run.id) == run
    assert store.list_events(run_id=run.id) == [event]


def test_resume_run_rejects_non_blocked_run(store: Store) -> None:
    session = Session()
    store.save_session(session)
    run = Run(session_id=session.id, agent_id="agent_1", status="running")
    store.save_run(run)

    with pytest.raises(ValueError, match="is not blocked"):
        store.resume_run(
            run=run,
            event=Event(type="run.resumed", run_id=run.id),
        )

    assert store.list_events(run_id=run.id) == []


@pytest.mark.parametrize("decision", ["approved", "denied"])
def test_resolve_approval_is_atomic_and_idempotent(
    store: Store,
    decision: ApprovalDecision,
) -> None:
    session = Session()
    store.save_session(session)
    run = Run(session_id=session.id, agent_id="agent_1")
    store.save_run(run)
    approval = ApprovalRequest(
        session_id=session.id,
        run_id=run.id,
        tool_call=ToolCall(name="write_file"),
    )
    store.save_approval(approval)

    resolved = store.resolve_approval(approval.id, decision)

    assert resolved.status == decision
    assert resolved.resolved_at is not None
    assert store.resolve_approval(approval.id, decision) == resolved
    events = store.list_events(run_id=run.id)
    assert len(events) == 1
    assert events[0].type == "approval.resolved"
    assert events[0].payload == {
        "approval_id": approval.id,
        "decision": decision,
        "tool_call_id": approval.tool_call.id,
    }

    conflicting: ApprovalDecision = (
        "denied" if decision == "approved" else "approved"
    )
    with pytest.raises(ValueError, match=f"already resolved as {decision}"):
        _ = store.resolve_approval(approval.id, conflicting)

    assert store.get_approval(approval.id) == resolved


def test_resolve_approval_rejects_unknown_id(store: Store) -> None:
    with pytest.raises(KeyError, match="Unknown approval: appr_missing"):
        _ = store.resolve_approval("appr_missing", "approved")


def test_resolve_approval_rejects_invalid_decision(store: Store) -> None:
    session = Session()
    store.save_session(session)
    run = Run(session_id=session.id, agent_id="agent_1")
    store.save_run(run)
    approval = ApprovalRequest(
        session_id=session.id,
        run_id=run.id,
        tool_call=ToolCall(name="write_file"),
    )
    store.save_approval(approval)

    invalid = cast(ApprovalDecision, "invalid")
    with pytest.raises(ValueError, match="Invalid approval decision"):
        _ = store.resolve_approval(approval.id, invalid)

    persisted = store.get_approval(approval.id)
    assert persisted is not None
    assert persisted.status == "pending"
    assert store.list_events(run_id=run.id) == []


def test_list_approvals_filters_by_run_and_status(store: Store) -> None:
    session = Session()
    store.save_session(session)
    first_run = Run(session_id=session.id, agent_id="agent_1")
    second_run = Run(session_id=session.id, agent_id="agent_1")
    store.save_run(first_run)
    store.save_run(second_run)

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
        store.save_approval(approval)
    approved_first = store.resolve_approval(approved_first.id, "approved")

    assert store.list_approvals() == [
        pending_first,
        approved_first,
        pending_second,
    ]
    assert store.list_approvals(run_id=first_run.id) == [
        pending_first,
        approved_first,
    ]
    assert store.list_approvals(status="pending") == [
        pending_first,
        pending_second,
    ]
    assert store.list_approvals(
        run_id=first_run.id,
        status="approved",
    ) == [approved_first]


def test_persistence_across_reopen(tmp_path) -> None:
    db_path = tmp_path / "conveyor.db"
    store = Store(db_path)
    session = Session(title="durable")
    store.save_session(session)
    store.append_event(Event(type="session.created", session_id=session.id))
    store.close()

    reopened = Store(db_path)
    assert reopened.get_session(session.id) == session
    events = reopened.list_events(session_id=session.id)
    assert len(events) == 1
    assert events[0].type == "session.created"
    reopened.close()

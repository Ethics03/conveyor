from __future__ import annotations

import pytest

from agent.models import ApprovalRequest, Event, Message, Run, Session, ToolCall, utc_now
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


def test_run_roundtrip_with_parent(store: Store) -> None:
    session = Session()
    store.save_session(session)
    parent = Run(session_id=session.id, agent_id="agent_1")
    store.save_run(parent)
    child = Run(session_id=session.id, agent_id="agent_1", parent_run_id=parent.id)
    store.save_run(child)

    loaded = store.get_run(child.id)
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


def test_approval_roundtrip_and_resolution(store: Store) -> None:
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

    approval.status = "approved"
    approval.resolved_at = utc_now()
    store.save_approval(approval)

    loaded = store.get_approval(approval.id)
    assert loaded == approval
    assert loaded.status == "approved"
    assert loaded.tool_call is not None
    assert loaded.tool_call.arguments == {"command": "rm -rf /tmp/x"}


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

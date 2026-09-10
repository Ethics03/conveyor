from pathlib import Path
from sqlite3 import ProgrammingError
from unittest.mock import Mock

import pytest

from agent.models import Agent, Session
from agent.runtime import Runtime
from providers.fake import FakeProvider
from storage.store import Store
from tools.registry import ToolRegistry


@pytest.mark.parametrize("title", [None, "Research notes"])
def test_create_session_persists_after_reopen(tmp_path: Path, title: str | None) -> None:
    database = tmp_path / "runtime.db"
    provider = FakeProvider()
    with Runtime(Store(database), provider, ToolRegistry(), tmp_path) as runtime:
        session = (
            runtime.create_session()
            if title is None
            else runtime.create_session(title)
        )
        assert session.title == ("New session" if title is None else title)
        assert session.title_source == ("default" if title is None else "user")
        assert session.status == "active"
        assert provider.requests == []

    reopened = Store(database)
    try:
        assert reopened.get_session(session.id) == session
        assert reopened.list_messages(session.id) == []
        assert reopened.list_runs(session.id) == []
    finally:
        reopened.close()


def test_create_session_rejects_closed_runtime(tmp_path: Path) -> None:
    database = tmp_path / "runtime.db"
    runtime = Runtime(Store(database), FakeProvider(), ToolRegistry(), tmp_path)
    runtime.close()

    with pytest.raises(RuntimeError, match="Runtime is closed"):
        runtime.create_session("Too late")

    reopened = Store(database)
    try:
        assert reopened.list_sessions() == []
    finally:
        reopened.close()


def test_create_session_propagates_storage_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Runtime(Store(), FakeProvider(), ToolRegistry(), tmp_path) as runtime:
        failure = OSError("storage unavailable")
        monkeypatch.setattr(runtime.store, "save_session", Mock(side_effect=failure))

        with pytest.raises(OSError) as raised:
            runtime.create_session("Unsaved")

        assert raised.value is failure
        assert runtime.store.list_sessions() == []


def test_run_turn_persists_input_and_executes_agent(tmp_path: Path) -> None:
    store = Store()
    provider = FakeProvider(["Hello from Conveyor"])

    with Runtime(store, provider, ToolRegistry(), tmp_path) as runtime:
        session = runtime.create_session("Runtime test")
        outcome = runtime.run_turn(
            agent=Agent(name="Test agent"),
            session=session,
            content="  Keep this spacing.  ",
        )

        assert runtime.context.workspace == tmp_path.resolve()
        assert outcome.run.status == "finished"
        assert outcome.final_message is not None
        assert outcome.final_message.content == "Hello from Conveyor"

        messages = store.list_messages(session.id)
        assert [message.role for message in messages] == ["user", "assistant"]
        assert messages[0].content == "  Keep this spacing.  "
        assert messages[0].run_id is None
        assert messages[1].run_id == outcome.run.id

        events = store.list_events(session_id=session.id)
        assert [event.type for event in events] == [
            "message.created",
            "run.started",
            "message.created",
            "run.finished",
        ]
        assert events[0].message_id == messages[0].id
        assert provider.requests[-1].messages[-1].content == (
            f"Message timestamp (UTC): {messages[0].created_at.isoformat()}\n\n"
            "  Keep this spacing.  "
        )
        assert len(provider.requests) == 1


def test_run_turn_generates_title_for_default_session(tmp_path: Path) -> None:
    store = Store()
    provider = FakeProvider(["Your interview is at 3 PM.", '"Interview Today"'])

    with Runtime(store, provider, ToolRegistry(), tmp_path) as runtime:
        session = runtime.create_session()
        outcome = runtime.run_turn(
            agent=Agent(model="test-model"),
            session=session,
            content="Do I have an interview today?",
        )

        assert outcome.run.status == "finished"
        assert session.title == "Interview Today"
        assert session.title_source == "auto"
        assert store.get_session(session.id) == session
        assert len(provider.requests) == 2
        assert provider.requests[1].metadata == {"purpose": "session_title"}
        assert provider.requests[1].model == "test-model"


def test_run_turn_falls_back_when_generated_title_is_invalid(tmp_path: Path) -> None:
    store = Store()
    provider = FakeProvider(["I can help with that.", ""])

    with Runtime(store, provider, ToolRegistry(), tmp_path) as runtime:
        session = runtime.create_session()
        outcome = runtime.run_turn(
            agent=Agent(),
            session=session,
            content="Debug the existing ingestion pipeline",
        )

        assert outcome.run.status == "finished"
        assert session.title == "Debug the existing ingestion pipeline"
        assert session.title_source == "auto"
        assert store.get_session(session.id) == session


@pytest.mark.parametrize(
    ("session", "error"),
    [
        (Session(), "Session does not exist"),
        (Session(status="archived"), "Session is not active"),
    ],
)
def test_run_turn_rejects_unavailable_session(
    tmp_path: Path,
    session: Session,
    error: str,
) -> None:
    store = Store()
    provider = FakeProvider()

    with Runtime(store, provider, ToolRegistry(), tmp_path) as runtime:
        if session.status == "archived":
            store.save_session(session)

        with pytest.raises(ValueError, match=error):
            runtime.run_turn(
                agent=Agent(),
                session=session,
                content="Hello",
            )

        assert store.list_messages(session.id) == []
        assert provider.requests == []


def test_run_turn_rejects_empty_content(tmp_path: Path) -> None:
    store = Store()
    provider = FakeProvider()

    with Runtime(store, provider, ToolRegistry(), tmp_path) as runtime:
        session = runtime.create_session()

        with pytest.raises(ValueError, match="Message content cannot be empty"):
            runtime.run_turn(agent=Agent(), session=session, content=" \n\t")

        assert store.list_messages(session.id) == []
        assert provider.requests == []


def test_run_turn_rejects_closed_runtime(tmp_path: Path) -> None:
    runtime = Runtime(Store(), FakeProvider(), ToolRegistry(), tmp_path)
    session = runtime.create_session()
    runtime.close()

    with pytest.raises(RuntimeError, match="Runtime is closed"):
        runtime.run_turn(agent=Agent(), session=session, content="Hello")


def test_context_manager_closes_resources(tmp_path: Path) -> None:
    store = Store()
    provider = FakeProvider()
    runtime = Runtime(store, provider, ToolRegistry(), tmp_path)

    with runtime as entered:
        assert entered is runtime
        assert provider.closed is False
        assert store.list_sessions() == []

    assert provider.closed is True
    with pytest.raises(ProgrammingError, match="closed database"):
        store.list_sessions()


def test_context_manager_closes_resources_on_exception(tmp_path: Path) -> None:
    store = Store()
    provider = FakeProvider()
    error = ValueError("turn failed")

    with (
        pytest.raises(ValueError) as raised,
        Runtime(store, provider, ToolRegistry(), tmp_path),
    ):
        raise error

    assert raised.value is error
    assert provider.closed is True
    with pytest.raises(ProgrammingError, match="closed database"):
        store.list_sessions()


def test_closed_runtime_cannot_be_entered(tmp_path: Path) -> None:
    runtime = Runtime(Store(), FakeProvider(), ToolRegistry(), tmp_path)
    runtime.close()

    with pytest.raises(RuntimeError, match="Runtime is closed"), runtime:
        pytest.fail("Closed runtime entered the block")


@pytest.mark.parametrize("close_fails", [False, True])
def test_runtime_closes_store_once_even_if_provider_close_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, close_fails: bool
) -> None:
    store = Store()
    provider = FakeProvider()
    runtime = Runtime(store, provider, ToolRegistry(), tmp_path)
    provider_close = Mock(
        side_effect=RuntimeError("provider close failed") if close_fails else None
    )
    store_close = Mock(wraps=store.close)
    monkeypatch.setattr(provider, "close", provider_close)
    monkeypatch.setattr(store, "close", store_close)

    if close_fails:
        with pytest.raises(RuntimeError, match="provider close failed"), runtime:
            pass
    else:
        with runtime:
            pass

    runtime.close()
    provider_close.assert_called_once_with()
    store_close.assert_called_once_with()
    with pytest.raises(ProgrammingError, match="closed database"):
        store.list_sessions()

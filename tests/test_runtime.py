from pathlib import Path
from sqlite3 import ProgrammingError
from unittest.mock import Mock

import pytest

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

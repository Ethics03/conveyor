from pathlib import Path

import pytest

from agent.models import Session
from agent.runtime import Runtime
from providers.fake import FakeProvider
from scripts.smoke_agent import resolve_session
from storage.store import Store
from tools.registry import ToolRegistry


async def test_resolve_session_creates_session_by_default(tmp_path: Path) -> None:
    async with Runtime(
        await Store.open(), FakeProvider(), ToolRegistry(), tmp_path
    ) as runtime:
        session = await resolve_session(runtime, None)

        assert session.title == "New session"
        assert session.title_source == "default"
        assert await runtime.store.get_session(session.id) == session


async def test_resolve_session_loads_persisted_session(tmp_path: Path) -> None:
    async with Runtime(
        await Store.open(), FakeProvider(), ToolRegistry(), tmp_path
    ) as runtime:
        session = await runtime.create_session("Existing session")

        assert await resolve_session(runtime, session.id) == session
        assert await runtime.store.list_sessions() == [session]


@pytest.mark.parametrize(
    ("session", "error"),
    [
        (Session(), "Session does not exist"),
        (Session(status="archived"), "Session is not active"),
    ],
)
async def test_resolve_session_rejects_unavailable_session(
    tmp_path: Path,
    session: Session,
    error: str,
) -> None:
    async with Runtime(
        await Store.open(), FakeProvider(), ToolRegistry(), tmp_path
    ) as runtime:
        if session.status == "archived":
            await runtime.store.save_session(session)

        with pytest.raises(ValueError, match=error):
            await resolve_session(runtime, session.id)

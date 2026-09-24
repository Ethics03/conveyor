import asyncio
import time
from pathlib import Path
from threading import Event
from unittest.mock import AsyncMock

import pytest

from agent.models import Agent, ProviderResponse, Session, ToolCall
from agent.runtime import Runtime
from providers.base import ModelLimits, ProviderRequest
from providers.fake import FakeProvider
from storage.store import Store
from tools.base import ExecutionContext, tool
from tools.registry import ToolRegistry


@pytest.mark.parametrize("title", [None, "Research notes"])
async def test_create_session_persists_after_reopen(
    tmp_path: Path, title: str | None
) -> None:
    database = tmp_path / "runtime.db"
    provider = FakeProvider()
    async with Runtime(
        await Store.open(database), provider, ToolRegistry(), tmp_path
    ) as runtime:
        session = (
            await runtime.create_session()
            if title is None
            else await runtime.create_session(title)
        )
        assert session.title == ("New session" if title is None else title)
        assert session.title_source == ("default" if title is None else "user")
        assert session.status == "active"
        assert provider.requests == []

    reopened = await Store.open(database)
    try:
        assert await reopened.get_session(session.id) == session
        assert await reopened.list_messages(session.id) == []
        assert await reopened.list_runs(session.id) == []
    finally:
        await reopened.close()


async def test_create_session_rejects_closed_runtime(tmp_path: Path) -> None:
    database = tmp_path / "runtime.db"
    runtime = Runtime(
        await Store.open(database), FakeProvider(), ToolRegistry(), tmp_path
    )
    await runtime.close()

    with pytest.raises(RuntimeError, match="Runtime is closed"):
        await runtime.create_session("Too late")

    reopened = await Store.open(database)
    try:
        assert await reopened.list_sessions() == []
    finally:
        await reopened.close()


async def test_create_session_propagates_storage_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with Runtime(
        await Store.open(), FakeProvider(), ToolRegistry(), tmp_path
    ) as runtime:
        failure = OSError("storage unavailable")
        monkeypatch.setattr(
            runtime.store,
            "save_session",
            AsyncMock(side_effect=failure),
        )

        with pytest.raises(OSError) as raised:
            await runtime.create_session("Unsaved")

        assert raised.value is failure
        assert await runtime.store.list_sessions() == []


async def test_run_turn_persists_input_and_executes_agent(tmp_path: Path) -> None:
    store = await Store.open()
    provider = FakeProvider(["Hello from Conveyor"])

    async with Runtime(store, provider, ToolRegistry(), tmp_path) as runtime:
        session = await runtime.create_session("Runtime test")
        outcome = await runtime.run_turn(
            agent=Agent(name="Test agent"),
            session=session,
            content="  Keep this spacing.  ",
        )

        assert runtime.context.workspace == tmp_path.resolve()
        assert outcome.run.status == "finished"
        assert outcome.final_message is not None
        assert outcome.final_message.content == "Hello from Conveyor"

        messages = await store.list_messages(session.id)
        assert [message.role for message in messages] == ["user", "assistant"]
        assert messages[0].content == "  Keep this spacing.  "
        assert messages[0].run_id is None
        assert messages[1].run_id == outcome.run.id

        events = await store.list_events(session_id=session.id)
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


async def test_interrupt_session_cancels_active_turn(tmp_path: Path) -> None:
    tool_started = Event()

    @tool(permission="read")
    def wait_for_cancellation(context: ExecutionContext) -> str:
        tool_started.set()
        while True:
            context.cancellation.raise_if_cancelled()
            time.sleep(0.01)

    store = await Store.open()
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[ToolCall(name="wait_for_cancellation")],
                finish_reason="tool_use",
            ),
            ProviderResponse.message("Continued after cancellation."),
        ]
    )
    registry = ToolRegistry([wait_for_cancellation])

    async with Runtime(store, provider, registry, tmp_path) as runtime:
        session = await runtime.create_session("Cancellation test")

        turn = asyncio.create_task(
            runtime.run_turn(
                agent=Agent(tools=["wait_for_cancellation"]),
                session=session,
                content="Wait until I stop the turn.",
            )
        )
        assert await asyncio.to_thread(tool_started.wait, 1)
        assert runtime.interrupt_session(session.id, "Stopped from Escape") is True
        outcome = await asyncio.wait_for(turn, timeout=2)

        assert outcome.run.status == "cancelled"
        assert outcome.run.error == "Stopped from Escape"
        assert session.status == "active"
        cancelled_messages = await store.list_messages(session.id)
        assert [message.role for message in cancelled_messages] == [
            "user",
            "assistant",
            "tool",
        ]
        assert cancelled_messages[-1].tool_call_id == (
            cancelled_messages[-2].tool_calls[0].id
        )
        assert cancelled_messages[-1].metadata == {
            "ok": False,
            "cancelled": True,
        }
        assert [
            event.type for event in await store.list_events(run_id=outcome.run.id)
        ] == [
            "run.started",
            "message.created",
            "tool.started",
            "message.created",
            "tool.cancelled",
            "run.cancelled",
        ]

        continued = await runtime.run_turn(
            agent=Agent(tools=["wait_for_cancellation"]),
            session=session,
            content="Continue with the next turn.",
        )
        assert continued.run.status == "finished"
        assert continued.final_message is not None
        assert continued.final_message.content == "Continued after cancellation."
        assert provider.requests[1].messages[-2].is_error is True
        assert provider.requests[1].messages[-2].tool_call_id == (
            cancelled_messages[-1].tool_call_id
        )
        assert runtime.interrupt_session(session.id) is False


async def test_interrupt_session_cancels_provider_request(tmp_path: Path) -> None:
    class WaitingProvider:
        name = "waiting"

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.closed = False

        def model_limits(self, model: str | None = None) -> ModelLimits:
            return ModelLimits(200_000, 4_096)

        async def generate(self, request: ProviderRequest) -> ProviderResponse:
            self.started.set()
            await asyncio.Event().wait()
            return ProviderResponse.message("unreachable")

        async def close(self) -> None:
            self.closed = True

    store = await Store.open()
    provider = WaitingProvider()
    async with Runtime(store, provider, ToolRegistry(), tmp_path) as runtime:
        session = await runtime.create_session("Provider cancellation test")
        turn = asyncio.create_task(
            runtime.run_turn(
                agent=Agent(),
                session=session,
                content="Wait for the provider.",
            )
        )

        await asyncio.wait_for(provider.started.wait(), timeout=1)
        assert runtime.interrupt_session(session.id, "Stop provider") is True
        outcome = await asyncio.wait_for(turn, timeout=1)

        assert outcome.run.status == "cancelled"
        assert outcome.run.error == "Stop provider"
        assert [
            event.type for event in await store.list_events(run_id=outcome.run.id)
        ] == [
            "run.started",
            "run.cancelled",
        ]


async def test_close_cancels_active_turn_before_closing_resources(
    tmp_path: Path,
) -> None:
    class WaitingProvider:
        name = "waiting"

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.closed = False

        def model_limits(self, model: str | None = None) -> ModelLimits:
            return ModelLimits(200_000, 4_096)

        async def generate(self, request: ProviderRequest) -> ProviderResponse:
            self.started.set()
            await asyncio.Event().wait()
            return ProviderResponse.message("unreachable")

        async def close(self) -> None:
            self.closed = True

    store = await Store.open()
    provider = WaitingProvider()
    runtime = Runtime(store, provider, ToolRegistry(), tmp_path)
    session = await runtime.create_session("Shutdown cancellation test")
    turn = asyncio.create_task(
        runtime.run_turn(
            agent=Agent(),
            session=session,
            content="Wait for shutdown.",
        )
    )

    await asyncio.wait_for(provider.started.wait(), timeout=1)
    await runtime.close()
    outcome = await asyncio.wait_for(turn, timeout=1)

    assert outcome.run.status == "cancelled"
    assert outcome.run.error == "Runtime is shutting down"
    assert provider.closed is True
    with pytest.raises(RuntimeError, match="Store is closed"):
        await store.list_sessions()


async def test_run_turn_generates_title_for_default_session(tmp_path: Path) -> None:
    store = await Store.open()
    provider = FakeProvider(["Your interview is at 3 PM.", '"Interview Today"'])

    async with Runtime(store, provider, ToolRegistry(), tmp_path) as runtime:
        session = await runtime.create_session()
        outcome = await runtime.run_turn(
            agent=Agent(model="test-model"),
            session=session,
            content="Do I have an interview today?",
        )

        assert outcome.run.status == "finished"
        assert session.title == "Interview Today"
        assert session.title_source == "auto"
        assert await store.get_session(session.id) == session
        assert len(provider.requests) == 2
        assert provider.requests[1].metadata == {"purpose": "session_title"}
        assert provider.requests[1].model == "test-model"


async def test_run_turn_falls_back_when_generated_title_is_invalid(
    tmp_path: Path,
) -> None:
    store = await Store.open()
    provider = FakeProvider(["I can help with that.", ""])

    async with Runtime(store, provider, ToolRegistry(), tmp_path) as runtime:
        session = await runtime.create_session()
        outcome = await runtime.run_turn(
            agent=Agent(),
            session=session,
            content="Debug the existing ingestion pipeline",
        )

        assert outcome.run.status == "finished"
        assert session.title == "Debug the existing ingestion pipeline"
        assert session.title_source == "auto"
        assert await store.get_session(session.id) == session


@pytest.mark.parametrize(
    ("session", "error"),
    [
        (Session(), "Session does not exist"),
        (Session(status="archived"), "Session is not active"),
    ],
)
async def test_run_turn_rejects_unavailable_session(
    tmp_path: Path,
    session: Session,
    error: str,
) -> None:
    store = await Store.open()
    provider = FakeProvider()

    async with Runtime(store, provider, ToolRegistry(), tmp_path) as runtime:
        if session.status == "archived":
            await store.save_session(session)

        with pytest.raises(ValueError, match=error):
            await runtime.run_turn(
                agent=Agent(),
                session=session,
                content="Hello",
            )

        assert await store.list_messages(session.id) == []
        assert provider.requests == []


async def test_run_turn_rejects_empty_content(tmp_path: Path) -> None:
    store = await Store.open()
    provider = FakeProvider()

    async with Runtime(store, provider, ToolRegistry(), tmp_path) as runtime:
        session = await runtime.create_session()

        with pytest.raises(ValueError, match="Message content cannot be empty"):
            await runtime.run_turn(agent=Agent(), session=session, content=" \n\t")

        assert await store.list_messages(session.id) == []
        assert provider.requests == []


async def test_run_turn_rejects_closed_runtime(tmp_path: Path) -> None:
    runtime = Runtime(await Store.open(), FakeProvider(), ToolRegistry(), tmp_path)
    session = await runtime.create_session()
    await runtime.close()

    with pytest.raises(RuntimeError, match="Runtime is closed"):
        await runtime.run_turn(agent=Agent(), session=session, content="Hello")


async def test_context_manager_closes_resources(tmp_path: Path) -> None:
    store = await Store.open()
    provider = FakeProvider()
    runtime = Runtime(store, provider, ToolRegistry(), tmp_path)

    async with runtime as entered:
        assert entered is runtime
        assert provider.closed is False
        assert await store.list_sessions() == []

    assert provider.closed is True
    with pytest.raises(RuntimeError, match="Store is closed"):
        await store.list_sessions()


async def test_context_manager_closes_resources_on_exception(tmp_path: Path) -> None:
    store = await Store.open()
    provider = FakeProvider()
    error = ValueError("turn failed")

    with pytest.raises(ValueError) as raised:
        async with Runtime(store, provider, ToolRegistry(), tmp_path):
            raise error

    assert raised.value is error
    assert provider.closed is True
    with pytest.raises(RuntimeError, match="Store is closed"):
        await store.list_sessions()


async def test_closed_runtime_cannot_be_entered(tmp_path: Path) -> None:
    runtime = Runtime(await Store.open(), FakeProvider(), ToolRegistry(), tmp_path)
    await runtime.close()

    with pytest.raises(RuntimeError, match="Runtime is closed"):
        async with runtime:
            pytest.fail("Closed runtime entered the block")


@pytest.mark.parametrize("close_fails", [False, True])
async def test_runtime_closes_store_once_even_if_provider_close_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, close_fails: bool
) -> None:
    store = await Store.open()
    provider = FakeProvider()
    runtime = Runtime(store, provider, ToolRegistry(), tmp_path)
    provider_close = AsyncMock(
        side_effect=RuntimeError("provider close failed") if close_fails else None
    )
    store_close = AsyncMock(wraps=store.close)
    monkeypatch.setattr(provider, "close", provider_close)
    monkeypatch.setattr(store, "close", store_close)

    if close_fails:
        with pytest.raises(RuntimeError, match="provider close failed"):
            async with runtime:
                pass
    else:
        async with runtime:
            pass

    await runtime.close()
    provider_close.assert_awaited_once_with()
    store_close.assert_awaited_once_with()
    with pytest.raises(RuntimeError, match="Store is closed"):
        await store.list_sessions()

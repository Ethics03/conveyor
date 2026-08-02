from __future__ import annotations

import pytest

from agent.loop import run_agent
from agent.models import Agent, Message, ProviderResponse, Session
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

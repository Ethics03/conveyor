from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from agent.loop import run_agent
from agent.models import Agent, Event, Message, Session
from providers.anthropic_provider import AnthropicProvider
from storage.store import Store
from tools.base import ExecutionContext
from tools.defaults import build_default_registry


class ConsoleStore(Store):
    def append_event(self, event: Event) -> None:
        super().append_event(event)
        if event.type == "tool.started":
            name = event.payload.get("name", "unknown")
            arguments = json.dumps(event.payload.get("arguments", {}))
            print(f"\ntool> starting {name} {arguments}")
        elif event.type == "tool.finished":
            name = event.payload.get("name", "unknown")
            status = "ok" if event.payload.get("ok") else "failed"
            print(f"tool> finished {name} ({status})")


def save_user_message(store: Store, session: Session, content: str) -> None:
    message = Message(
        session_id=session.id,
        role="user",
        content=content,
    )
    store.save_message(message)
    store.append_event(
        Event(
            type="message.created",
            session_id=session.id,
            message_id=message.id,
            payload={"role": message.role},
        )
    )


def print_messages(store: Store, session: Session) -> None:
    for message in store.list_messages(session.id):
        content = message.content
        if message.tool_calls:
            calls = ", ".join(call.name for call in message.tool_calls)
            content = content or f"[tool calls: {calls}]"
        print(f"{message.role}> {content}")


def print_events(store: Store, session: Session) -> None:
    for event in store.list_events(session_id=session.id):
        print(json.dumps(asdict(event), default=str))


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required in the environment or .env")

    workspace = Path(sys.argv[1]).expanduser().resolve()
    if not workspace.is_dir():
        raise SystemExit(f"Workspace does not exist: {workspace}")

    store = ConsoleStore(":memory:")
    session = Session(title="Anthropic smoke session")
    store.save_session(session)
    store.append_event(Event(type="session.created", session_id=session.id))

    registry = build_default_registry()
    agent = Agent(
        name="Conveyor",
        instructions=(
            "You are operating inside a workspace. Use the available tools when "
            "you need to inspect files, and answer concisely."
        ),
        model=os.environ.get("CONVEYOR_MODEL"),
        tools=registry.names(),
    )
    provider = AnthropicProvider()
    context = ExecutionContext(workspace=workspace)

    print(f"workspace: {workspace}")
    print("commands: /messages, /events, /exit")

    try:
        while True:
            try:
                prompt = input("\nyou> ").strip()
            except EOFError:
                break

            if not prompt:
                continue
            if prompt in {"/exit", "/quit"}:
                break
            if prompt == "/messages":
                print_messages(store, session)
                continue
            if prompt == "/events":
                print_events(store, session)
                continue

            save_user_message(store, session, prompt)

            try:
                outcome = run_agent(
                    agent=agent,
                    session=session,
                    provider=provider,
                    registry=registry,
                    context=context,
                    store=store,
                )
            except Exception as exc:
                print(f"error> {exc}")
                continue

            if outcome.final_message is not None:
                print(f"\nassistant> {outcome.final_message.content}")
            if outcome.run.status != "finished":
                print(f"run> {outcome.run.status}: {outcome.run.error}")
    except KeyboardInterrupt:
        print()
    finally:
        store.close()


main()

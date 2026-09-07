from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from agent.models import (
    Agent,
    ApprovalDecision,
    ApprovalRequest,
    Event,
    Session,
)
from agent.runtime import Runtime
from providers.anthropic_provider import AnthropicProvider
from storage.store import Store
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


def prompt_for_approval(approval: ApprovalRequest) -> ApprovalDecision:
    arguments = json.dumps(approval.tool_call.arguments, indent=2)
    print(f"\napproval> {approval.tool_call.name}")
    print(f"reason> {approval.reason}")
    print(arguments)

    while True:
        try:
            answer = input("allow this tool call? [y/N] ").strip().lower()
        except EOFError:
            return "denied"
        if answer in {"y", "yes", "approve"}:
            return "approved"
        if answer in {"", "n", "no", "deny"}:
            return "denied"
        print("answer with y or n")


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


def resolve_session(runtime: Runtime, session_id: str | None) -> Session:
    if session_id is None:
        return runtime.create_session()

    session = runtime.store.get_session(session_id)
    if session is None:
        raise ValueError(f"Session does not exist: {session_id}")
    if session.status != "active":
        raise ValueError(f"Session is not active: {session_id}")
    return session


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run an interactive Conveyor agent")
    parser.add_argument(
        "workspace",
        nargs="?",
        default=str(repo_root),
        help="workspace exposed to agent tools",
    )
    parser.add_argument(
        "--database",
        default=str(repo_root / ".conveyor" / "smoke-agent.db"),
        help="SQLite database used for durable sessions",
    )
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="resume an active session from the database",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required in the environment or .env")

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        parser.error(f"Workspace does not exist: {workspace}")

    database = Path(args.database).expanduser().resolve()
    database.parent.mkdir(parents=True, exist_ok=True)

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

    try:
        with Runtime(
            store=ConsoleStore(database),
            provider=AnthropicProvider(),
            registry=registry,
            workspace=workspace,
        ) as runtime:
            try:
                session = resolve_session(runtime, args.resume)
            except ValueError as exc:
                parser.error(str(exc))

            print(f"workspace: {workspace}")
            print(f"database: {database}")
            print(f"session: {session.id}")
            print("commands: /messages, /events, /exit")
            if args.resume:
                print("\nhistory:")
                print_messages(runtime.store, session)

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
                    print_messages(runtime.store, session)
                    continue
                if prompt == "/events":
                    print_events(runtime.store, session)
                    continue

                title_before_turn = session.title
                try:
                    outcome = runtime.run_turn(
                        agent=agent,
                        session=session,
                        content=prompt,
                        approval_callback=prompt_for_approval,
                    )
                except Exception as exc:  # noqa: BLE001 - keep the REPL alive per turn
                    print(f"error> {exc}")
                    continue

                if outcome.final_message is not None:
                    print(f"\nassistant> {outcome.final_message.content}")
                if session.title != title_before_turn:
                    print(f"session title> {session.title}")
                if outcome.run.status != "finished":
                    print(f"run> {outcome.run.status}: {outcome.run.error}")
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()

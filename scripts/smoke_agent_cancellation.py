from __future__ import annotations

import argparse
import asyncio
import json
import os
import select
import sys
import tempfile
import termios
import tty
from pathlib import Path

from agent.models import (
    Agent,
    ApprovalDecision,
    ApprovalRequest,
    Event,
    Message,
    Run,
    RunOutcome,
)
from agent.runtime import Runtime
from providers.anthropic_provider import AnthropicProvider
from storage.store import Store
from tools.registry import ToolRegistry
from tools.shell import bash

DEFAULT_MESSAGE = (
    "Use the bash tool to run exactly `sleep 30 && echo completed`. "
    "Wait for the command to finish before responding."
)


class ConsoleStore(Store):
    @staticmethod
    def _print_events(events: list[Event]) -> None:
        for event in events:
            line = _event_line(event)
            if line is not None:
                print(f"\n{line}", flush=True)

    async def append_event(self, event: Event) -> None:
        await super().append_event(event)
        self._print_events([event])

    async def save_run_with_events(self, run: Run, events: list[Event]) -> None:
        await super().save_run_with_events(run, events)
        self._print_events(events)

    async def save_message_with_events(
        self,
        message: Message,
        events: list[Event],
    ) -> None:
        await super().save_message_with_events(message, events)
        self._print_events(events)


def _event_line(event: Event) -> str | None:
    if event.type == "run.started":
        return "run> started"
    if event.type == "tool.started":
        name = event.payload.get("name", "unknown")
        arguments = json.dumps(event.payload.get("arguments", {}))
        return f"tool> started {name} {arguments}"
    if event.type == "tool.finished":
        name = event.payload.get("name", "unknown")
        status = "ok" if event.payload.get("ok") else "failed"
        return f"tool> finished {name} ({status})"
    if event.type == "tool.cancelled":
        name = event.payload.get("name", "unknown")
        return f"tool> cancelled {name}"
    if event.type == "run.cancelled":
        return f"run> cancelled: {event.payload.get('reason', 'cancelled')}"
    if event.type == "run.finished":
        return "run> finished"
    if event.type == "run.failed":
        return f"run> failed: {event.payload.get('error', 'unknown error')}"
    return None


def _approve_bash(approval: ApprovalRequest) -> ApprovalDecision:
    print(f"\napproval> auto-approved {approval.tool_call.name}", flush=True)
    return "approved"


def _interrupt(runtime: Runtime, session_id: str) -> bool:
    accepted = runtime.interrupt_session(
        session_id,
        "Stopped by user",
    )
    status = "accepted" if accepted else "no active turn"
    print(f"\ninterrupt> {status}", flush=True)
    return accepted


def _wait_for_escape(
    turn: asyncio.Task[RunOutcome],
    *,
    runtime: Runtime,
    session_id: str,
) -> RunOutcome:
    if not sys.stdin.isatty():
        raise RuntimeError("Interactive cancellation requires a terminal")

    input_fd = sys.stdin.fileno()
    previous_terminal = termios.tcgetattr(input_fd)
    interrupted = False
    try:
        tty.setcbreak(input_fd)
        while not turn.done():
            readable, _, _ = select.select([input_fd], [], [], 0.1)
            if not readable:
                continue
            key = os.read(input_fd, 1)
            if key == b"\x1b" and not interrupted:
                interrupted = _interrupt(runtime, session_id)
    except KeyboardInterrupt:
        if not interrupted:
            interrupted = _interrupt(runtime, session_id)
    finally:
        termios.tcsetattr(input_fd, termios.TCSADRAIN, previous_terminal)

    return turn.result()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real agent turn and cancel it by pressing Escape",
    )
    parser.add_argument(
        "workspace",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="workspace exposed to the Bash tool",
    )
    parser.add_argument(
        "--message",
        default=DEFAULT_MESSAGE,
        help="message sent to the agent",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CONVEYOR_MODEL"),
        help="Anthropic model override",
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="persist the demo in this SQLite database instead of a temporary one",
    )
    return parser.parse_args()


async def _run_demo(args: argparse.Namespace, database: Path) -> None:
    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        raise SystemExit(f"Workspace does not exist: {workspace}")

    provider = (
        AnthropicProvider()
        if args.model is None
        else AnthropicProvider(model=args.model)
    )
    registry = ToolRegistry([bash])
    agent = Agent(
        name="Cancellation demo",
        instructions=(
            "Use the Bash tool when the user asks you to run a command. "
            "Do not claim a command completed until its tool result is available."
        ),
        model=args.model,
        tools=registry.names(),
    )

    async with Runtime(
        store=await ConsoleStore.open(database),
        provider=provider,
        registry=registry,
        workspace=workspace,
    ) as runtime:
        session = await runtime.create_session("Interactive cancellation demo")
        print(f"workspace: {workspace}")
        print(f"database: {database}")
        print(f"session: {session.id}")
        print(f"\nyou> {args.message}")
        print("\nPress Esc while the turn is running to cancel it.")
        print("For immediate process cancellation, press Esc after `tool> started`.")

        turn = asyncio.create_task(
            runtime.run_turn(
                agent=agent,
                session=session,
                content=args.message,
                approval_callback=_approve_bash,
            )
        )

        outcome = await asyncio.to_thread(
            _wait_for_escape,
            turn,
            runtime=runtime,
            session_id=session.id,
        )

        if outcome.run.status == "finished" and outcome.final_message is not None:
            print(f"\nassistant> {outcome.final_message.content}")
        elif outcome.run.status == "cancelled":
            print(f"\nresult> cancelled ({outcome.run.error})")
        else:
            print(f"\nresult> {outcome.run.status} ({outcome.run.error})")

        events = await runtime.store.list_events(run_id=outcome.run.id)
        print("events> " + ", ".join(event.type for event in events))


async def main() -> None:
    args = parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required in the environment or .env")

    if args.database is not None:
        database = args.database.expanduser().resolve()
        database.parent.mkdir(parents=True, exist_ok=True)
        await _run_demo(args, database)
        return

    with tempfile.TemporaryDirectory(prefix="conveyor-agent-cancel-") as temp_dir:
        await _run_demo(args, Path(temp_dir) / "runtime.db")


if __name__ == "__main__":
    asyncio.run(main())

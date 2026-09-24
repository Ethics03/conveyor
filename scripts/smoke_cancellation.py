from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import tempfile
import time
from pathlib import Path

from agent.models import (
    Agent,
    ApprovalDecision,
    ApprovalRequest,
    ProviderResponse,
    RunOutcome,
    ToolCall,
)
from agent.runtime import Runtime
from providers.fake import FakeProvider
from storage.store import Store
from tools.registry import ToolRegistry
from tools.shell import bash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test turn cancellation")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="workspace exposed to the Bash tool",
    )
    parser.add_argument(
        "--command",
        default="sleep 30",
        help="long-running command to cancel",
    )
    parser.add_argument(
        "--cancel-after",
        type=float,
        default=0.25,
        help="seconds to wait after the command starts before cancelling",
    )
    return parser.parse_args()


async def _wait_for_process(marker: Path, turn: asyncio.Task[RunOutcome]) -> int:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if marker.exists():
            return int(marker.read_text().strip())
        if turn.done():
            _ = turn.result()
            raise RuntimeError("Agent turn finished before Bash started")
        await asyncio.sleep(0.02)
    raise TimeoutError("Bash did not start within 5 seconds")


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _approve_smoke_command(_: ApprovalRequest) -> ApprovalDecision:
    return "approved"


async def main() -> None:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        raise SystemExit(f"Workspace does not exist: {workspace}")
    if args.cancel_after < 0:
        raise SystemExit("--cancel-after must not be negative")

    with tempfile.TemporaryDirectory(prefix="conveyor-cancellation-") as temp_dir:
        temporary = Path(temp_dir)
        database = temporary / "runtime.db"
        process_marker = temporary / "process.pid"
        command = (
            f"printf '%s' \"$$\" > {shlex.quote(str(process_marker))}; {args.command}"
        )
        provider = FakeProvider(
            [
                ProviderResponse(
                    tool_calls=[
                        ToolCall(
                            id="call_cancel_bash",
                            name="bash",
                            arguments={"command": command},
                        )
                    ],
                    finish_reason="tool_use",
                )
            ]
        )

        async with Runtime(
            store=await Store.open(database),
            provider=provider,
            registry=ToolRegistry([bash]),
            workspace=workspace,
        ) as runtime:
            session = await runtime.create_session("Cancellation smoke test")

            turn = asyncio.create_task(
                runtime.run_turn(
                    agent=Agent(name="Cancellation agent", tools=["bash"]),
                    session=session,
                    content="Run the command and wait for it to finish.",
                    approval_callback=_approve_smoke_command,
                )
            )

            process_id = await _wait_for_process(process_marker, turn)
            await asyncio.sleep(args.cancel_after)

            cancelled_at = time.monotonic()
            interrupt_accepted = runtime.interrupt_session(
                session.id,
                "Stopped by cancellation smoke test",
            )
            outcome = await asyncio.wait_for(turn, timeout=5)
            cancellation_seconds = time.monotonic() - cancelled_at

            continuation = await runtime.run_turn(
                agent=Agent(name="Cancellation agent", tools=["bash"]),
                session=session,
                content="Continue after the cancelled command.",
            )

        verified = await Store.open(database)
        try:
            persisted_session = await verified.get_session(session.id)
            events = await verified.list_events(run_id=outcome.run.id)
            messages = await verified.list_messages(session.id)
        finally:
            await verified.close()

        process_alive = _process_exists(process_id)
        assert interrupt_accepted
        assert outcome.run.status == "cancelled"
        assert persisted_session is not None
        assert persisted_session.status == "active"
        assert events[-1].type == "run.cancelled"
        assert continuation.run.status == "finished"
        assert continuation.final_message is not None
        assert [message.role for message in messages[:3]] == [
            "user",
            "assistant",
            "tool",
        ]
        assert messages[2].metadata.get("cancelled") is True
        assert not process_alive

        print(
            json.dumps(
                {
                    "workspace": str(workspace),
                    "session_id": session.id,
                    "run_id": outcome.run.id,
                    "run_status": outcome.run.status,
                    "cancellation_reason": outcome.run.error,
                    "interrupt_accepted": interrupt_accepted,
                    "cancel_after_seconds": args.cancel_after,
                    "cancellation_elapsed_seconds": round(cancellation_seconds, 3),
                    "session_status": persisted_session.status,
                    "continuation_status": continuation.run.status,
                    "continuation_response": continuation.final_message.content,
                    "process_id": process_id,
                    "process_alive": process_alive,
                    "message_roles": [message.role for message in messages],
                    "events": [event.type for event in events],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())

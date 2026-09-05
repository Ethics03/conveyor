from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from agent.models import Agent
from agent.runtime import Runtime
from providers.fake import FakeProvider
from storage.store import Store
from tools.registry import ToolRegistry


def _database_path(value: str | None) -> tuple[Path, bool]:
    if value is not None:
        path = Path(value).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path, False

    descriptor, raw_path = tempfile.mkstemp(prefix="conveyor-runtime-", suffix=".db")
    os.close(descriptor)
    return Path(raw_path), True


def _remove_database(path: Path) -> None:
    for candidate in (path, Path(f"{path}-shm"), Path(f"{path}-wal")):
        candidate.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Runtime.run_turn")
    parser.add_argument(
        "--workspace",
        default=".",
        help="workspace exposed through the runtime execution context",
    )
    parser.add_argument(
        "--database",
        help="optional SQLite path; temporary storage is used by default",
    )
    parser.add_argument(
        "--message",
        default="Verify the runtime turn pipeline.",
        help="user message persisted by run_turn",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        parser.error(f"Workspace does not exist: {workspace}")

    database, temporary = _database_path(args.database)
    provider = FakeProvider(["Runtime turn completed."])

    try:
        with Runtime(
            store=Store(database),
            provider=provider,
            registry=ToolRegistry(),
            workspace=workspace,
        ) as runtime:
            session = runtime.create_session("Runtime smoke test")
            outcome = runtime.run_turn(
                agent=Agent(name="Runtime smoke agent"),
                session=session,
                content=args.message,
            )

        verified = Store(database)
        try:
            persisted_session = verified.get_session(session.id)
            messages = verified.list_messages(session.id)
            runs = verified.list_runs(session.id)
            events = verified.list_events(session_id=session.id)
        finally:
            verified.close()

        assert persisted_session == session
        assert runs == [outcome.run]
        assert [message.role for message in messages] == ["user", "assistant"]
        assert messages[0].content == args.message
        assert outcome.final_message is not None
        assert messages[1] == outcome.final_message
        assert provider.closed

        print(
            json.dumps(
                {
                    "workspace": str(workspace),
                    "database": str(database),
                    "temporary": temporary,
                    "session_id": session.id,
                    "run_id": outcome.run.id,
                    "run_status": outcome.run.status,
                    "final_response": outcome.final_message.content,
                    "message_roles": [message.role for message in messages],
                    "events": [event.type for event in events],
                    "provider_requests": len(provider.requests),
                    "provider_closed": provider.closed,
                    "reopened": True,
                },
                indent=2,
            )
        )
    finally:
        if temporary:
            _remove_database(database)


if __name__ == "__main__":
    main()

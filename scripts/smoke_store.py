from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from agent.models import (
    ApprovalRequest,
    Event,
    Message,
    Run,
    Session,
    ToolCall,
    utc_now,
)
from storage.store import Store


def _database_path(value: str | None) -> tuple[Path, bool]:
    if value is not None:
        path = Path(value).expanduser().resolve()
        if path.exists():
            raise ValueError(f"Refusing to overwrite existing database: {path}")
        return path, False

    descriptor, raw_path = tempfile.mkstemp(prefix="conveyor-store-", suffix=".db")
    os.close(descriptor)
    return Path(raw_path), True


def _remove_database(path: Path) -> None:
    for candidate in (path, Path(f"{path}-shm"), Path(f"{path}-wal")):
        candidate.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the durable SQLite store")
    parser.add_argument(
        "database",
        nargs="?",
        help="optional new database path; temporary storage is used by default",
    )
    parser.add_argument(
        "--decision",
        choices=("approved", "denied"),
        default="approved",
        help="approval decision to persist",
    )
    args = parser.parse_args()

    try:
        database, temporary = _database_path(args.database)
    except ValueError as exc:
        parser.error(str(exc))

    session = Session(title="Store smoke test")
    run = Run(session_id=session.id, agent_id="smoke_agent", status="running")
    tool_call = ToolCall(
        id="call_smoke_write",
        name="write_file",
        arguments={"path": "smoke.txt", "content": "hello"},
    )
    message = Message(
        session_id=session.id,
        run_id=run.id,
        role="assistant",
        tool_calls=[tool_call],
    )
    approval = ApprovalRequest(
        session_id=session.id,
        run_id=run.id,
        tool_call=tool_call,
        reason="write_file can modify workspace files",
    )

    try:
        store = Store(database)
        store.save_session(session)
        store.save_run(run)
        store.append_event(
            Event(type="run.started", session_id=session.id, run_id=run.id)
        )
        store.save_message(message)

        run.status = "blocked"
        store.block_run(
            run=run,
            approvals=[approval],
            events=[
                Event(
                    type="approval.requested",
                    session_id=session.id,
                    run_id=run.id,
                    message_id=message.id,
                    payload={
                        "approval_id": approval.id,
                        "tool_call_id": tool_call.id,
                    },
                ),
                Event(
                    type="run.blocked",
                    session_id=session.id,
                    run_id=run.id,
                    message_id=message.id,
                    payload={
                        "approval_ids": [approval.id],
                        "approval_count": 1,
                    },
                ),
            ],
        )
        persisted_run = store.get_run(run.id)
        persisted_approval = store.get_approval(approval.id)
        assert persisted_run is not None and persisted_run.status == "blocked"
        assert persisted_approval is not None
        assert persisted_approval.status == "pending"
        assert persisted_approval.tool_call == tool_call

        resolved = store.resolve_approval(approval.id, args.decision)
        run.status = "running"
        run.updated_at = utc_now()
        store.resume_run(
            run=run,
            event=Event(
                type="run.resumed",
                session_id=session.id,
                run_id=run.id,
                message_id=message.id,
                payload={"approval_ids": [approval.id]},
            ),
        )
        store.close()

        verified = Store(database)
        final_run = verified.get_run(run.id)
        final_approval = verified.get_approval(approval.id)
        events = verified.list_events(run_id=run.id)
        messages = verified.list_messages(session.id)
        verified.close()

        assert final_approval == resolved
        assert final_approval is not None
        assert final_approval.status == args.decision
        assert final_approval.resolved_at is not None
        assert final_run is not None and final_run.status == "running"
        assert messages == [message]
        assert [event.type for event in events] == [
            "run.started",
            "approval.requested",
            "run.blocked",
            "approval.resolved",
            "run.resumed",
        ]

        print(
            json.dumps(
                {
                    "database": str(database),
                    "temporary": temporary,
                    "session_id": session.id,
                    "run_id": run.id,
                    "run_status": final_run.status,
                    "approval": {
                        "id": final_approval.id,
                        "status": final_approval.status,
                        "tool_call_id": final_approval.tool_call.id,
                    },
                    "events": [event.type for event in events],
                    "reopen_checks": 1,
                },
                indent=2,
            )
        )
    finally:
        if temporary:
            _remove_database(database)


if __name__ == "__main__":
    main()

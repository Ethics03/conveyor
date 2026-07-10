from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.models import (
    ApprovalRequest,
    Event,
    Message,
    Run,
    Session,
    ToolCall,
)
from storage.schema import SCHEMA, SCHEMA_VERSION


class Store:
    """Synchronous SQLite store. The only component allowed to touch the database.

    Events are append-only: there is deliberately no update path for them.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

        # sessions  

    def save_session(self, session: Session) -> None:
        self._conn.execute(
            """
            INSERT INTO sessions (id, title, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            (
                session.id,
                session.title,
                session.status,
                _dump_dt(session.created_at),
                _dump_dt(session.updated_at),
            ),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> Session | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return Session(
            id=row["id"],
            title=row["title"],
            status=row["status"],
            created_at=_load_dt(row["created_at"]),
            updated_at=_load_dt(row["updated_at"]),
        )

    def list_sessions(self, status: str | None = None) -> list[Session]:
        query = "SELECT * FROM sessions"
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY created_at, id"
        rows = self._conn.execute(query, params).fetchall()
        return [
            Session(
                id=row["id"],
                title=row["title"],
                status=row["status"],
                created_at=_load_dt(row["created_at"]),
                updated_at=_load_dt(row["updated_at"]),
            )
            for row in rows
        ]

    # runs 

    def save_run(self, run: Run) -> None:
        self._conn.execute(
            """
            INSERT INTO runs (id, session_id, agent_id, parent_run_id, status,
                              error, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                error = excluded.error,
                updated_at = excluded.updated_at
            """,
            (
                run.id,
                run.session_id,
                run.agent_id,
                run.parent_run_id,
                run.status,
                run.error,
                _dump_dt(run.created_at),
                _dump_dt(run.updated_at),
            ),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> Run | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_run(row)

    def list_runs(self, session_id: str) -> list[Run]:
        rows = self._conn.execute(
            "SELECT * FROM runs WHERE session_id = ? ORDER BY created_at, id",
            (session_id,),
        ).fetchall()
        return [_row_to_run(row) for row in rows]

    # messages 

    def save_message(self, message: Message) -> None:
        self._conn.execute(
            """
            INSERT INTO messages (id, session_id, role, content, run_id, name,
                                  metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.id,
                message.session_id,
                message.role,
                message.content,
                message.run_id,
                message.name,
                json.dumps(message.metadata),
                _dump_dt(message.created_at),
            ),
        )
        self._conn.commit()

    def list_messages(self, session_id: str) -> list[Message]:
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at, id",
            (session_id,),
        ).fetchall()
        return [
            Message(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                run_id=row["run_id"],
                name=row["name"],
                metadata=json.loads(row["metadata"]),
                created_at=_load_dt(row["created_at"]),
            )
            for row in rows
        ]

    # events -> append-only 

    def append_event(self, event: Event) -> None:
        self._conn.execute(
            """
            INSERT INTO events (id, type, session_id, run_id, message_id,
                                payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.type,
                event.session_id,
                event.run_id,
                event.message_id,
                json.dumps(event.payload),
                _dump_dt(event.created_at),
            ),
        )
        self._conn.commit()

    def list_events(
        self,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> list[Event]:
        conditions: list[str] = []
        params: list[Any] = []
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)
        if run_id is not None:
            conditions.append("run_id = ?")
            params.append(run_id)
        query = "SELECT * FROM events"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at, id"
        rows = self._conn.execute(query, params).fetchall()
        return [
            Event(
                id=row["id"],
                type=row["type"],
                session_id=row["session_id"],
                run_id=row["run_id"],
                message_id=row["message_id"],
                payload=json.loads(row["payload"]),
                created_at=_load_dt(row["created_at"]),
            )
            for row in rows
        ]

    # approvals 

    def save_approval(self, approval: ApprovalRequest) -> None:
        tool_call = None
        if approval.tool_call is not None:
            tool_call = json.dumps(
                {
                    "id": approval.tool_call.id,
                    "name": approval.tool_call.name,
                    "arguments": approval.tool_call.arguments,
                }
            )
        self._conn.execute(
            """
            INSERT INTO approvals (id, session_id, run_id, tool_call, reason,
                                   status, created_at, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                resolved_at = excluded.resolved_at
            """,
            (
                approval.id,
                approval.session_id,
                approval.run_id,
                tool_call,
                approval.reason,
                approval.status,
                _dump_dt(approval.created_at),
                _dump_dt(approval.resolved_at) if approval.resolved_at else None,
            ),
        )
        self._conn.commit()

    def get_approval(self, approval_id: str) -> ApprovalRequest | None:
        row = self._conn.execute(
            "SELECT * FROM approvals WHERE id = ?", (approval_id,)
        ).fetchone()
        if row is None:
            return None
        tool_call = None
        if row["tool_call"]:
            data = json.loads(row["tool_call"])
            tool_call = ToolCall(
                id=data["id"], name=data["name"], arguments=data["arguments"]
            )
        return ApprovalRequest(
            id=row["id"],
            session_id=row["session_id"],
            run_id=row["run_id"],
            tool_call=tool_call,
            reason=row["reason"],
            status=row["status"],
            created_at=_load_dt(row["created_at"]),
            resolved_at=_load_dt(row["resolved_at"]) if row["resolved_at"] else None,
        )


def _row_to_run(row: sqlite3.Row) -> Run:
    return Run(
        id=row["id"],
        session_id=row["session_id"],
        agent_id=row["agent_id"],
        parent_run_id=row["parent_run_id"],
        status=row["status"],
        error=row["error"],
        created_at=_load_dt(row["created_at"]),
        updated_at=_load_dt(row["updated_at"]),
    )


def _dump_dt(value: datetime) -> str:
    return value.isoformat()


def _load_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any, Concatenate, ParamSpec, TypeVar

from agent.models import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    Event,
    Message,
    Run,
    Session,
    ToolCall,
    utc_now,
)
from storage.schema import SCHEMA, SCHEMA_VERSION


P = ParamSpec("P")
R = TypeVar("R")


def _serialized(
    method: Callable[Concatenate[Store, P], R],
) -> Callable[Concatenate[Store, P], R]:
    @wraps(method)
    def wrapper(self: Store, *args: P.args, **kwargs: P.kwargs) -> R:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class Store:
    """Synchronous SQLite store. The only component allowed to touch the database.

    Events are append-only: there is deliberately no update path for them.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._lock = RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    @_serialized
    def close(self) -> None:
        self._conn.close()

    # sessions

    @_serialized
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

    @_serialized
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

    @_serialized
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

    @_serialized
    def save_run(self, run: Run) -> None:
        self._save_run(run)
        self._conn.commit()

    def _save_run(self, run: Run) -> None:
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

    @_serialized
    def get_run(self, run_id: str) -> Run | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_run(row)

    @_serialized
    def list_runs(self, session_id: str) -> list[Run]:
        rows = self._conn.execute(
            "SELECT * FROM runs WHERE session_id = ? ORDER BY created_at, id",
            (session_id,),
        ).fetchall()
        return [_row_to_run(row) for row in rows]

    # messages

    @_serialized
    def save_message(self, message: Message) -> None:
        self._conn.execute(
            """
            INSERT INTO messages (id, session_id, role, content, run_id, name,
                                  tool_calls, tool_call_id, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.id,
                message.session_id,
                message.role,
                message.content,
                message.run_id,
                message.name,
                json.dumps([
                    {
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }
                    for tool_call in message.tool_calls
                ]),
                message.tool_call_id,
                json.dumps(message.metadata),
                _dump_dt(message.created_at),
            ),
        )
        self._conn.commit()

    @_serialized
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
                tool_calls=[
                    ToolCall(
                        id=tool_call["id"],
                        name=tool_call["name"],
                        arguments=tool_call["arguments"],
                    )
                    for tool_call in json.loads(row["tool_calls"])
                ],
                tool_call_id=row["tool_call_id"],
                metadata=json.loads(row["metadata"]),
                created_at=_load_dt(row["created_at"]),
            )
            for row in rows
        ]

    # events -> append-only

    @_serialized
    def append_event(self, event: Event) -> None:
        self._insert_event(event)
        self._conn.commit()

    def _insert_event(self, event: Event) -> None:
        _ = self._conn.execute(
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

    @_serialized
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

    @_serialized
    def save_approval(self, approval: ApprovalRequest) -> None:
        self._insert_approval(approval)
        self._conn.commit()

    def _insert_approval(self, approval: ApprovalRequest) -> None:
        if approval.status != "pending":
            raise ValueError("New approvals must have pending status")
        if approval.resolved_at is not None:
            raise ValueError("New approvals cannot have resolved_at")

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

    @_serialized
    def block_run(
        self,
        *,
        run: Run,
        approvals: list[ApprovalRequest],
        events: list[Event],
    ) -> None:
        try:
            for approval in approvals:
                self._insert_approval(approval)
            self._save_run(run)
            for event in events:
                self._insert_event(event)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    @_serialized
    def resume_run(self, *, run: Run, event: Event) -> None:
        if run.status != "running":
            raise ValueError("Resumed runs must have running status")
        try:
            cursor = self._conn.execute(
                """
                UPDATE runs
                SET status = ?, error = ?, updated_at = ?
                WHERE id = ? AND status = 'blocked'
                """,
                (
                    run.status,
                    run.error,
                    _dump_dt(run.updated_at),
                    run.id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Run {run.id} is not blocked")
            self._insert_event(event)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    @_serialized
    def get_approval(self, approval_id: str) -> ApprovalRequest | None:
        row = self._conn.execute(
            "SELECT * FROM approvals WHERE id = ?",
            (approval_id,),
        ).fetchone()

        if row is None:
            return None

        return _row_to_approval(row)

    @_serialized
    def list_approvals(
        self,
        *,
        run_id: str | None = None,
        status: ApprovalStatus | None = None,
    ) -> list[ApprovalRequest]:
        conditions: list[str] = []
        parameters: list[str] = []

        if run_id is not None:
            conditions.append("run_id = ?")
            parameters.append(run_id)

        if status is not None:
            conditions.append("status = ?")
            parameters.append(status)

        query = "SELECT * FROM approvals"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at, id"

        rows = self._conn.execute(query, parameters).fetchall()
        return [_row_to_approval(row) for row in rows]

    @_serialized
    def resolve_approval(
        self,
        approval_id: str,
        decision: ApprovalDecision,
    ) -> ApprovalRequest:
        decision = _validate_approval_decision(decision)
        resolved_at = utc_now()
        try:
            row = self._conn.execute(
                """
                UPDATE approvals
                SET status = ?, resolved_at = ?
                WHERE id = ? AND status = 'pending'
                RETURNING *
                """,
                (decision, _dump_dt(resolved_at), approval_id),
            ).fetchone()

            if row is not None:
                approval = _row_to_approval(row)
                tool_call = approval.tool_call
                self._insert_event(
                    Event(
                        type="approval.resolved",
                        session_id=approval.session_id,
                        run_id=approval.run_id,
                        payload={
                            "approval_id": approval.id,
                            "decision": decision,
                            "tool_call_id": tool_call.id,
                        },
                    )
                )
                self._conn.commit()
                return approval

            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        approval = self.get_approval(approval_id)
        if approval is None:
            raise KeyError(f"Unknown approval: {approval_id}")
        if approval.status == decision:
            return approval
        if approval.status != "pending":
            raise ValueError(
                f"Approval {approval_id} already resolved as {approval.status}"
            )

        raise RuntimeError(f"Approval could not be resolved: {approval_id}")


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


def _row_to_approval(row: sqlite3.Row) -> ApprovalRequest:
    raw_tool_call = row["tool_call"]
    if not raw_tool_call:
        raise ValueError(f"Approval {row['id']} is missing its tool call")
    data = json.loads(raw_tool_call)
    tool_call = ToolCall(
        id=data["id"],
        name=data["name"],
        arguments=data["arguments"],
    )

    return ApprovalRequest(
        id=row["id"],
        session_id=row["session_id"],
        run_id=row["run_id"],
        tool_call=tool_call,
        reason=row["reason"],
        status=row["status"],
        created_at=_load_dt(row["created_at"]),
        resolved_at=(
            _load_dt(row["resolved_at"])
            if row["resolved_at"]
            else None
        ),
    )


def _validate_approval_decision(value: object) -> ApprovalDecision:
    if value == "approved":
        return "approved"
    if value == "denied":
        return "denied"
    raise ValueError(f"Invalid approval decision: {value!r}")

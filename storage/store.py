from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from sqlite3 import Row
from typing import Any, Concatenate, Literal, Self

import aiosqlite

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
from agent.titles import clean_session_title
from storage.schema import SCHEMA, SCHEMA_VERSION


def _serialized[**P, R](
    method: Callable[Concatenate[Store, P], Coroutine[object, object, R]],
) -> Callable[Concatenate[Store, P], Coroutine[object, object, R]]:
    @wraps(method)
    async def wrapper(self: Store, *args: P.args, **kwargs: P.kwargs) -> R:
        async def invoke() -> R:
            async with self._lock:
                self._ensure_open()
                return await method(self, *args, **kwargs)

        task = asyncio.create_task(invoke())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Let the queued SQLite operation commit or roll back before the
            # caller observes cancellation.
            _ = await asyncio.gather(task, return_exceptions=True)
            raise

    return wrapper


class Store:
    """Async SQLite store. The only component allowed to touch the database.

    Events are append-only: there is deliberately no update path for them.
    """

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._conn = connection
        self._lock = asyncio.Lock()
        self._closed = False

    @classmethod
    async def open(cls, path: str | Path = ":memory:") -> Self:
        connection = await aiosqlite.connect(
            str(path),
            isolation_level=None,
        )
        connection.row_factory = Row
        store = cls(connection)
        try:
            await store._initialize()
        except BaseException:
            await connection.close()
            raise
        return store

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def _initialize(self) -> None:
        await self._execute_pragma("PRAGMA journal_mode=WAL")
        await self._execute_pragma("PRAGMA foreign_keys=ON")
        await self._execute_pragma("PRAGMA busy_timeout=5000")
        cursor = await self._conn.executescript(SCHEMA)
        await cursor.close()
        cursor = await self._conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        await cursor.close()

    async def _execute_pragma(self, statement: str) -> None:
        cursor = await self._conn.execute(statement)
        await cursor.close()

    async def _execute(
        self,
        statement: str,
        parameters: tuple[object, ...] = (),
    ) -> int:
        cursor = await self._conn.execute(statement, parameters)
        try:
            return cursor.rowcount
        finally:
            await cursor.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Store is closed")

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        await self._execute("BEGIN IMMEDIATE")
        try:
            yield
            await self._conn.commit()
        except BaseException:
            await self._conn.rollback()
            raise

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            close_task = asyncio.create_task(self._conn.close())
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError:
                _ = await asyncio.gather(close_task, return_exceptions=True)
                self._closed = True
                raise
            self._closed = True

    # sessions

    @_serialized
    async def save_session(self, session: Session) -> None:
        async with self._transaction():
            await self._execute(
                """
                INSERT INTO sessions (
                    id, title, title_source, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    session.id,
                    session.title,
                    session.title_source,
                    session.status,
                    _dump_dt(session.created_at),
                    _dump_dt(session.updated_at),
                ),
            )

    @_serialized
    async def set_session_title(self, session_id: str, title: str) -> bool:
        """Set a user-authored title that automatic naming cannot replace."""
        async with self._transaction():
            return await self._set_session_title(
                session_id=session_id,
                title=title,
                source="user",
            )

    @_serialized
    async def set_auto_title(self, session_id: str, title: str) -> bool:
        """Set a derived title only while the session still has its default."""
        async with self._transaction():
            return await self._set_session_title(
                session_id=session_id,
                title=title,
                source="auto",
            )

    async def _set_session_title(
        self,
        *,
        session_id: str,
        title: str,
        source: Literal["auto", "user"],
    ) -> bool:
        cleaned = clean_session_title(title)
        query = (
            "UPDATE sessions "
            "SET title = ?, title_source = ?, updated_at = ? "
            "WHERE id = ?"
        )
        if source == "auto":
            query += " AND title_source = 'default'"

        async with self._conn.execute(
            query, (cleaned, source, _dump_dt(utc_now()), session_id)
        ) as cursor:
            return cursor.rowcount > 0

    @_serialized
    async def get_session(self, session_id: str) -> Session | None:
        async with self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return Session(
            id=row["id"],
            title=row["title"],
            title_source=row["title_source"],
            status=row["status"],
            created_at=_load_dt(row["created_at"]),
            updated_at=_load_dt(row["updated_at"]),
        )

    @_serialized
    async def list_sessions(self, status: str | None = None) -> list[Session]:
        query = "SELECT * FROM sessions"
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY created_at, id"
        async with self._conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        return [
            Session(
                id=row["id"],
                title=row["title"],
                title_source=row["title_source"],
                status=row["status"],
                created_at=_load_dt(row["created_at"]),
                updated_at=_load_dt(row["updated_at"]),
            )
            for row in rows
        ]

    # runs

    @_serialized
    async def save_run(self, run: Run) -> None:
        async with self._transaction():
            await self._save_run(run)

    @_serialized
    async def save_run_with_events(self, run: Run, events: list[Event]) -> None:
        async with self._transaction():
            await self._save_run(run)
            for event in events:
                await self._insert_event(event)

    async def _save_run(self, run: Run) -> None:
        await self._execute(
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
    async def get_run(self, run_id: str) -> Run | None:
        async with self._conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_run(row)

    @_serialized
    async def list_runs(self, session_id: str) -> list[Run]:
        async with self._conn.execute(
            "SELECT * FROM runs WHERE session_id = ? ORDER BY created_at, id",
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_run(row) for row in rows]

    # messages

    @_serialized
    async def save_message(self, message: Message) -> None:
        async with self._transaction():
            await self._insert_message(message)

    @_serialized
    async def save_message_with_events(
        self,
        message: Message,
        events: list[Event],
    ) -> None:
        async with self._transaction():
            await self._insert_message(message)
            for event in events:
                await self._insert_event(event)

    async def _insert_message(self, message: Message) -> None:
        await self._execute(
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
                json.dumps(
                    [
                        {
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        }
                        for tool_call in message.tool_calls
                    ]
                ),
                message.tool_call_id,
                json.dumps(message.metadata),
                _dump_dt(message.created_at),
            ),
        )

    @_serialized
    async def list_messages(self, session_id: str) -> list[Message]:
        async with self._conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at, id",
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
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
    async def append_event(self, event: Event) -> None:
        async with self._transaction():
            await self._insert_event(event)

    async def _insert_event(self, event: Event) -> None:
        await self._execute(
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
    async def list_events(
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
        async with self._conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
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
    async def save_approval(self, approval: ApprovalRequest) -> None:
        async with self._transaction():
            await self._insert_approval(approval)

    async def _insert_approval(self, approval: ApprovalRequest) -> None:
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
        await self._execute(
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
    async def block_run(
        self,
        *,
        run: Run,
        approvals: list[ApprovalRequest],
        events: list[Event],
    ) -> None:
        async with self._transaction():
            for approval in approvals:
                await self._insert_approval(approval)
            await self._save_run(run)
            for event in events:
                await self._insert_event(event)

    @_serialized
    async def resume_run(self, *, run: Run, event: Event) -> None:
        if run.status != "running":
            raise ValueError("Resumed runs must have running status")
        async with self._transaction():
            rowcount = await self._execute(
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
            if rowcount != 1:
                raise ValueError(f"Run {run.id} is not blocked")
            await self._insert_event(event)

    @_serialized
    async def get_approval(self, approval_id: str) -> ApprovalRequest | None:
        return await self._get_approval(approval_id)

    async def _get_approval(self, approval_id: str) -> ApprovalRequest | None:
        async with self._conn.execute(
            "SELECT * FROM approvals WHERE id = ?",
            (approval_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return _row_to_approval(row)

    @_serialized
    async def list_approvals(
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

        async with self._conn.execute(query, parameters) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_approval(row) for row in rows]

    @_serialized
    async def resolve_approval(
        self,
        approval_id: str,
        decision: ApprovalDecision,
    ) -> ApprovalRequest:
        decision = _validate_approval_decision(decision)
        resolved_at = utc_now()
        async with self._transaction():
            async with self._conn.execute(
                """
                UPDATE approvals
                SET status = ?, resolved_at = ?
                WHERE id = ? AND status = 'pending'
                RETURNING *
                """,
                (decision, _dump_dt(resolved_at), approval_id),
            ) as cursor:
                row = await cursor.fetchone()

            if row is not None:
                approval = _row_to_approval(row)
                tool_call = approval.tool_call
                await self._insert_event(
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
                return approval

        approval = await self._get_approval(approval_id)
        if approval is None:
            raise KeyError(f"Unknown approval: {approval_id}")
        if approval.status == decision:
            return approval
        if approval.status != "pending":
            raise ValueError(
                f"Approval {approval_id} already resolved as {approval.status}"
            )

        raise RuntimeError(f"Approval could not be resolved: {approval_id}")


def _row_to_run(row: Row) -> Run:
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


def _row_to_approval(row: Row) -> ApprovalRequest:
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
        resolved_at=(_load_dt(row["resolved_at"]) if row["resolved_at"] else None),
    )


def _validate_approval_decision(value: object) -> ApprovalDecision:
    if value == "approved":
        return "approved"
    if value == "denied":
        return "denied"
    raise ValueError(f"Invalid approval decision: {value!r}")

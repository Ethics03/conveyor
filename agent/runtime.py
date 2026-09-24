from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Self

from agent.approvals import ApprovalCallback
from agent.cancellation import DEFAULT_CANCELLATION_REASON, CancellationToken
from agent.loop import run_agent
from agent.models import Agent, Event, Message, RunOutcome, Session
from agent.titles import (
    clean_session_title,
    derive_session_title,
    generate_session_title,
)
from providers.base import Provider
from storage.store import Store
from tools.base import ExecutionContext
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
RUNTIME_SHUTDOWN_REASON = "Runtime is shutting down"


@dataclass(frozen=True, slots=True)
class _ActiveTurn:
    cancellation: CancellationToken
    task: asyncio.Task[object]
    loop: asyncio.AbstractEventLoop


@dataclass(slots=True)
class Runtime:
    """Owns the shared resources used to execute agent runs."""

    store: Store
    provider: Provider
    registry: ToolRegistry
    workspace: Path
    context: ExecutionContext = field(init=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _active_turns: dict[str, _ActiveTurn] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _active_turns_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.workspace = self.workspace.expanduser().resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"Workspace does not exist: {self.workspace}")
        self.context = ExecutionContext(workspace=self.workspace)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Runtime is closed")

    async def create_session(self, title: str | None = None) -> Session:
        self._ensure_open()
        session = (
            Session()
            if title is None
            else Session(title=clean_session_title(title), title_source="user")
        )
        await self.store.save_session(session)
        return session

    def interrupt_session(
        self,
        session_id: str,
        reason: str = DEFAULT_CANCELLATION_REASON,
    ) -> bool:
        """Request cancellation of the active turn without closing its session."""
        self._ensure_open()
        with self._active_turns_lock:
            active_turn = self._active_turns.get(session_id)
        if active_turn is None:
            return False
        _ = active_turn.cancellation.cancel(reason)
        active_turn.loop.call_soon_threadsafe(active_turn.task.cancel, reason)
        return True

    def _begin_turn(self, session_id: str) -> _ActiveTurn:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("run_turn requires an active asyncio task")
        cancellation = CancellationToken()
        active_turn = _ActiveTurn(
            cancellation=cancellation,
            task=task,
            loop=asyncio.get_running_loop(),
        )
        with self._active_turns_lock:
            if session_id in self._active_turns:
                raise RuntimeError(f"Session already has an active turn: {session_id}")
            self._active_turns[session_id] = active_turn
        return active_turn

    def _end_turn(
        self,
        session_id: str,
        active_turn: _ActiveTurn,
    ) -> None:
        with self._active_turns_lock:
            if self._active_turns.get(session_id) is active_turn:
                del self._active_turns[session_id]

    async def run_turn(
        self,
        *,
        agent: Agent,
        session: Session,
        content: str,
        approval_callback: ApprovalCallback | None = None,
    ) -> RunOutcome:
        self._ensure_open()
        if not content.strip():
            raise ValueError("Message content cannot be empty")

        persisted_session = await self.store.get_session(session.id)
        if persisted_session is None:
            raise ValueError(f"Session does not exist: {session.id}")
        if persisted_session.status != "active":
            raise ValueError(f"Session is not active: {session.id}")

        active_turn = self._begin_turn(persisted_session.id)
        try:
            user_message = Message(
                session_id=persisted_session.id,
                role="user",
                content=content,
            )
            await self.store.save_message_with_events(
                user_message,
                [
                    Event(
                        type="message.created",
                        session_id=persisted_session.id,
                        message_id=user_message.id,
                        payload={"role": user_message.role},
                    )
                ],
            )

            outcome = await run_agent(
                agent=agent,
                session=persisted_session,
                provider=self.provider,
                registry=self.registry,
                context=ExecutionContext(
                    workspace=self.workspace,
                    cancellation=active_turn.cancellation,
                ),
                store=self.store,
                approval_callback=approval_callback,
            )
        finally:
            self._end_turn(persisted_session.id, active_turn)

        if (
            persisted_session.title_source == "default"
            and outcome.run.status == "finished"
            and outcome.final_message is not None
        ):
            await self._title_session(
                agent=agent,
                session=session,
                user_message=content,
                assistant_response=outcome.final_message.content,
            )
        return outcome

    async def _title_session(
        self,
        *,
        agent: Agent,
        session: Session,
        user_message: str,
        assistant_response: str,
    ) -> None:
        fallback = derive_session_title(user_message)
        if fallback is None:
            return

        title = fallback
        try:
            title = await generate_session_title(
                self.provider,
                user_message=user_message,
                assistant_response=assistant_response,
                model=agent.model,
            )
        except Exception:
            logger.warning(
                "Session title generation failed; using fallback", exc_info=True
            )

        try:
            updated = await self.store.set_auto_title(session.id, title)
            persisted = await self.store.get_session(session.id) if updated else None
        except Exception:
            logger.warning("Session title persistence failed", exc_info=True)
            return

        if persisted is not None:
            session.title = persisted.title
            session.title_source = persisted.title_source
            session.updated_at = persisted.updated_at

    async def __aenter__(self) -> Self:
        self._ensure_open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        current_task = asyncio.current_task()
        with self._active_turns_lock:
            active_turns = list(self._active_turns.values())

        for active_turn in active_turns:
            _ = active_turn.cancellation.cancel(RUNTIME_SHUTDOWN_REASON)
            if active_turn.task is not current_task:
                active_turn.task.cancel(RUNTIME_SHUTDOWN_REASON)

        pending = [
            active_turn.task
            for active_turn in active_turns
            if active_turn.task is not current_task
        ]
        if pending:
            _ = await asyncio.gather(*pending, return_exceptions=True)

        try:
            await self.provider.close()
        finally:
            await self.store.close()

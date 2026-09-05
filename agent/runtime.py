from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Self

from agent.approvals import ApprovalCallback
from agent.loop import run_agent
from agent.models import Agent, Event, Message, RunOutcome, Session
from providers.base import Provider
from storage.store import Store
from tools.base import ExecutionContext
from tools.registry import ToolRegistry


@dataclass(slots=True)
class Runtime:
    """Owns the shared resources used to execute agent runs."""

    store: Store
    provider: Provider
    registry: ToolRegistry
    workspace: Path
    context: ExecutionContext = field(init=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.workspace = self.workspace.expanduser().resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"Workspace does not exist: {self.workspace}")
        self.context = ExecutionContext(workspace=self.workspace)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Runtime is closed")

    def create_session(self, title: str = "New session") -> Session:
        self._ensure_open()
        session = Session(title=title)
        self.store.save_session(session)
        return session

    def run_turn(
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

        persisted_session = self.store.get_session(session.id)
        if persisted_session is None:
            raise ValueError(f"Session does not exist: {session.id}")
        if persisted_session.status != "active":
            raise ValueError(f"Session is not active: {session.id}")

        user_message = Message(
            session_id=persisted_session.id,
            role="user",
            content=content,
        )
        self.store.save_message(user_message)
        self.store.append_event(
            Event(
                type="message.created",
                session_id=persisted_session.id,
                message_id=user_message.id,
                payload={"role": user_message.role},
            )
        )

        return run_agent(
            agent=agent,
            session=persisted_session,
            provider=self.provider,
            registry=self.registry,
            context=self.context,
            store=self.store,
            approval_callback=approval_callback,
        )

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        try:
            self.provider.close()
        finally:
            self.store.close()

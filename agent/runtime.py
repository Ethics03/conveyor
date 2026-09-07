from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Self

from agent.approvals import ApprovalCallback
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

    def create_session(self, title: str | None = None) -> Session:
        self._ensure_open()
        session = (
            Session()
            if title is None
            else Session(title=clean_session_title(title), title_source="user")
        )
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

        outcome = run_agent(
            agent=agent,
            session=persisted_session,
            provider=self.provider,
            registry=self.registry,
            context=self.context,
            store=self.store,
            approval_callback=approval_callback,
        )
        if (
            persisted_session.title_source == "default"
            and outcome.run.status == "finished"
            and outcome.final_message is not None
        ):
            self._title_session(
                agent=agent,
                session=session,
                user_message=content,
                assistant_response=outcome.final_message.content,
            )
        return outcome

    def _title_session(
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
            title = generate_session_title(
                self.provider,
                user_message=user_message,
                assistant_response=assistant_response,
                model=agent.model,
            )
        except Exception:
            logger.warning("Session title generation failed; using fallback", exc_info=True)

        try:
            updated = self.store.set_auto_title(session.id, title)
            persisted = self.store.get_session(session.id) if updated else None
        except Exception:
            logger.warning("Session title persistence failed", exc_info=True)
            return

        if persisted is not None:
            session.title = persisted.title
            session.title_source = persisted.title_source
            session.updated_at = persisted.updated_at

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

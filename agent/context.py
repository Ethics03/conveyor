from __future__ import annotations

from agent.models import Agent, Message, ProviderMessage, Run, Session


def _to_provider_message(message: Message) -> ProviderMessage:
    return ProviderMessage(
        role=message.role,
        content=message.content,
        name=message.name,
        tool_calls=list(message.tool_calls),
        tool_call_id=message.tool_call_id,
    )


def build_provider_messages(
    agent: Agent,
    messages: list[Message],
    *,
    session: Session | None = None,
    run: Run | None = None,
) -> list[ProviderMessage]:
    if (session is None) != (run is None):
        raise ValueError("Session and run temporal context must be provided together")

    provider_messages: list[ProviderMessage] = []

    if agent.instructions:
        provider_messages.append(
            ProviderMessage(
                role="system",
                content=agent.instructions,
            )
        )

    if session is not None and run is not None:
        provider_messages.append(
            ProviderMessage(
                role="system",
                content=(
                    "Temporal context (UTC):\n"
                    f"- Session created at: {session.created_at.isoformat()}\n"
                    f"- Current run started at: {run.created_at.isoformat()}"
                ),
            )
        )

    provider_messages.extend(
        _to_provider_message(message)
        for message in messages
    )
    return provider_messages

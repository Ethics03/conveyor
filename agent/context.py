from __future__ import annotations

from agent.models import Agent, Message, ProviderMessage


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
) -> list[ProviderMessage]:
    provider_messages: list[ProviderMessage] = []

    if agent.instructions:
        provider_messages.append(
            ProviderMessage(
                role="system",
                content=agent.instructions,
            )
        )

    provider_messages.extend(
        _to_provider_message(message)
        for message in messages
    )
    return provider_messages

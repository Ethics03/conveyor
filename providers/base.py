from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.models import ProviderMessage, ProviderResponse


@dataclass(slots=True)
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderRequest:
    messages: list[ProviderMessage]
    tools: list[ToolSchema] = field(default_factory=list)
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    name: str

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        pass

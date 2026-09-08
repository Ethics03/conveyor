from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.models import ProviderMessage, ProviderResponse


@dataclass(frozen=True, slots=True)
class ModelLimits:
    context_window_tokens: int
    max_output_tokens: int

    def __post_init__(self) -> None:
        if self.context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.max_output_tokens >= self.context_window_tokens:
            raise ValueError(
                "max_output_tokens must be smaller than the context window"
            )


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

    def model_limits(self, model: str | None = None) -> ModelLimits: ...

    def generate(self, request: ProviderRequest) -> ProviderResponse: ...

    def close(self) -> None: ...

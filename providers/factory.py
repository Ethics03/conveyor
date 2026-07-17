from __future__ import annotations

from providers.anthropic import DEFAULT_ANTHROPIC_MODEL, AnthropicProvider
from providers.base import Provider
from providers.fake import FakeProvider


def create_provider(
    name: str = "fake",
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> Provider:
    if name == "fake":
        return FakeProvider()
    if name == "anthropic":
        return AnthropicProvider(
            model=model or DEFAULT_ANTHROPIC_MODEL,
            api_key=api_key,
        )
    raise ValueError(f"Unknown provider: {name}") 


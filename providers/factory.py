from __future__ import annotations

from providers.anthropic import AnthropicProvider
from providers.base import Provider
from providers.fake import FakeProvider


def create_provider(name: str = "fake", **kwargs: object) -> Provider:
    if name == "fake":
        return FakeProvider()
    if name == "anthropic":
        return AnthropicProvider(**kwargs)
    raise ValueError(f"Unknown provider: {name}") 


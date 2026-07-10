from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from agent.models import ProviderMessage, ProviderResponse, ToolCall
from providers.base import ProviderRequest


DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"


@dataclass(slots=True)
class AnthropicProvider:
    model: str = DEFAULT_ANTHROPIC_MODEL
    api_key: str | None = None
    name: str = "anthropic"

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        try:
            anthropic_module = import_module("anthropic")
        except ImportError as exc:
            raise RuntimeError("Install Anthropic support with: uv add anthropic") from exc

        model_name = request.model or self.model
        async_anthropic = getattr(anthropic_module, "AsyncAnthropic")
        client = (
            async_anthropic(api_key=self.api_key)
            if self.api_key
            else async_anthropic()
        )
        params: dict[str, Any] = {
            "model": model_name,
            "max_tokens": request.max_tokens or 4096,
            "messages": _anthropic_messages(request.messages),
        }
        if request.temperature is not None:
            params["temperature"] = request.temperature
        system = _system_instructions(request.messages)
        if system:
            params["system"] = system
        tools = _anthropic_tools(request)
        if tools:
            params["tools"] = tools

        response = await client.messages.create(**params)

        for block in response.content:
            if block.type == "tool_use":
                return ProviderResponse(
                    type="tool_call",
                    tool_call=ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input),
                    ),
                    raw={"provider": self.name, "model": model_name},
                )

        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        return ProviderResponse.message(
            text, raw={"provider": self.name, "model": model_name}
        )


def _anthropic_messages(messages: list[ProviderMessage]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": message.content}],
                }
            )
            continue
        role = "assistant" if message.role == "assistant" else "user"
        converted.append({"role": role, "content": message.content})
    return converted or [{"role": "user", "content": ""}]


def _anthropic_tools(request: ProviderRequest) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters or {"type": "object", "properties": {}},
        }
        for tool in request.tools
    ]


def _system_instructions(messages: list[ProviderMessage]) -> str:
    instructions = [message.content for message in messages if message.role == "system"]
    return "\n\n".join(instructions)

from __future__ import annotations

from dataclasses import dataclass

from anthropic import Anthropic, omit
from anthropic.types import (
    ContentBlockParam,
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)

from agent.models import ProviderMessage, ProviderResponse, ToolCall
from providers.base import ProviderRequest


DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"


@dataclass(slots=True)
class AnthropicProvider:
    model: str = DEFAULT_ANTHROPIC_MODEL
    api_key: str | None = None
    name: str = "anthropic"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        model_name = request.model or self.model
        client = (
            Anthropic(api_key=self.api_key)
            if self.api_key
            else Anthropic()
        )
        system = _system_instructions(request.messages)
        tools = _anthropic_tools(request)
        response = client.messages.create(
            model=model_name,
            max_tokens=request.max_tokens or 4096,
            messages=_anthropic_messages(request.messages),
            temperature=(
                request.temperature
                if request.temperature is not None
                else omit
            ),
            system=system if system else omit,
            tools=tools if tools else omit,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input),
                    )
                )

        return ProviderResponse(
            content="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            finish_reason=response.stop_reason,
            raw={"provider": self.name, "model": model_name},
        )


def _anthropic_messages(messages: list[ProviderMessage]) -> list[MessageParam]:
    converted: list[MessageParam] = []
    pending_tool_results: list[ToolResultBlockParam] = []

    for message in messages:
        if message.role == "system":
            continue

        if message.role == "tool":
            if message.tool_call_id is None:
                raise ValueError("Tool result message requires tool_call_id")
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content,
            })
            continue

        if pending_tool_results:
            converted.append({
                "role": "user",
                "content": pending_tool_results,
            })
            pending_tool_results = []

        role = "assistant" if message.role == "assistant" else "user"
        if message.role == "assistant" and message.tool_calls:
            content: list[ContentBlockParam] = []
            if message.content:
                text_block: TextBlockParam = {
                    "type": "text",
                    "text": message.content,
                }
                content.append(text_block)
            for tool_call in message.tool_calls:
                tool_use_block: ToolUseBlockParam = {
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": tool_call.arguments,
                }
                content.append(tool_use_block)
            converted.append({"role": role, "content": content})
            continue

        converted.append({"role": role, "content": message.content})

    if pending_tool_results:
        converted.append({
            "role": "user",
            "content": pending_tool_results,
        })

    return converted or [{"role": "user", "content": ""}]


def _anthropic_tools(request: ProviderRequest) -> list[ToolParam]:
    converted: list[ToolParam] = []
    for tool in request.tools:
        input_schema: dict[str, object] = dict(tool.parameters)
        converted.append({
            "name": tool.name,
            "description": tool.description,
            "input_schema": input_schema or {
                "type": "object",
                "properties": {},
            },
        })
    return converted


def _system_instructions(messages: list[ProviderMessage]) -> str:
    instructions = [message.content for message in messages if message.role == "system"]
    return "\n\n".join(instructions)

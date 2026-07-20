from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from agent.models import ToolCall, ToolPermission, ToolResult
from providers.base import ToolSchema
from tools.base import (
    ExecutionContext,
    JsonObject,
    Tool,
    ToolOutput,
    _output_content_type,
    _stringify_output,
)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        self.register_many(tools)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def register_many(self, tools: Iterable[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def permissions(self) -> dict[str, ToolPermission]:
        return {name: self._tools[name].permission for name in self.names()}

    def schemas(self) -> list[ToolSchema]:
        return [self._tools[name].schema for name in self.names()]

    async def execute(
        self, tool_call: ToolCall, context: ExecutionContext
    ) -> ToolResult:
        tool = self.get(tool_call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                ok=False,
                content=f"Unknown tool: {tool_call.name}",
            )

        try:
            output = await tool.execute(_json_object(tool_call.arguments), context)
        except Exception as exc:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool.name,
                ok=False,
                content=str(exc),
            )

        return _successful_result(tool_call, tool, output)


def _successful_result(tool_call: ToolCall, tool: Tool, output: ToolOutput) -> ToolResult:
    return ToolResult(
        tool_call_id=tool_call.id,
        name=tool.name,
        ok=True,
        content=_stringify_output(output),
        metadata={
            "permission": tool.permission,
            "content_type": _output_content_type(output),
        },
    )


def _json_object(value: dict[str, Any]) -> JsonObject:
    return cast(JsonObject, value)

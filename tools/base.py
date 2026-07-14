from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.models import ToolPermission, ToolResult
from providers.base import ToolSchema

ToolExecutor = Callable[[dict[str, Any], "ExecutionContext"], Awaitable[ToolResult]]


@dataclass(slots=True)
class ExecutionContext:
    workspace: Path


@dataclass(slots=True)
class Tool:
    schema: ToolSchema
    permission: ToolPermission
    execute: ToolExecutor

    @property
    def name(self) -> str:
        return self.schema.name


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[ToolSchema]:
        return [tool.schema for tool in self._tools.values()]

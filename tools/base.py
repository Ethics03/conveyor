from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

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


_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) != 1:
            raise TypeError(f"Unsupported union type: {annotation}")
        return _json_type(args[0])
    if origin is not None:
        annotation = origin
    if annotation in _TYPE_MAP:
        return _TYPE_MAP[annotation]
    raise TypeError(f"Unsupported parameter type: {annotation}")


def tool(
    *,
    permission: ToolPermission,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Awaitable[ToolResult]]], Tool]:
    def decorator(fn: Callable[..., Awaitable[ToolResult]]) -> Tool:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"Tool function must be async: {fn.__name__}")

        hints = get_type_hints(fn)
        signature = inspect.signature(fn)
        properties: dict[str, Any] = {}
        required: list[str] = []
        wants_context = False

        for param_name, param in signature.parameters.items():
            annotation = hints.get(param_name, str)
            if annotation is ExecutionContext:
                wants_context = True
                continue
            properties[param_name] = {"type": _json_type(annotation)}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schema = ToolSchema(
            name=name or fn.__name__,
            description=description or inspect.getdoc(fn) or "",
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
        )

        async def execute(
            arguments: dict[str, Any], context: ExecutionContext
        ) -> ToolResult:
            kwargs = dict(arguments)
            if wants_context:
                kwargs["context"] = context
            return await fn(**kwargs)

        return Tool(schema=schema, permission=permission, execute=execute)

    return decorator


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

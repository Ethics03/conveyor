from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

from agent.models import ToolPermission
from providers.base import ToolSchema

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]
ToolOutput = JsonValue
ToolExecutor = Callable[[JsonObject, "ExecutionContext"], Awaitable[ToolOutput]]


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


def _json_type(annotation: object) -> str:
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
) -> Callable[[Callable[..., Awaitable[ToolOutput]]], Tool]:
    def decorator(fn: Callable[..., Awaitable[ToolOutput]]) -> Tool:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"Tool function must be async: {fn.__name__}")

        hints: dict[str, object] = get_type_hints(fn)
        signature = inspect.signature(fn)
        properties: JsonObject = {}
        required: list[str] = []
        wants_context = False

        for param_name, param in signature.parameters.items():
            annotation: object = hints.get(param_name, str)
            if annotation is ExecutionContext:
                wants_context = True
                continue
            properties[param_name] = {"type": _json_type(annotation)}
            default: object = param.default
            if default is inspect.Parameter.empty:
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
            arguments: JsonObject, context: ExecutionContext
        ) -> ToolOutput:
            kwargs: dict[str, object] = dict(arguments)
            if wants_context:
                kwargs["context"] = context
            return await fn(**cast(dict[str, Any], kwargs))

        return Tool(schema=schema, permission=permission, execute=execute)

    return decorator


def _stringify_output(output: ToolOutput) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    return repr(output)

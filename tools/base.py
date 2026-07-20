from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, cast, get_args, get_origin

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
    if isinstance(annotation, UnionType):
        union_args = cast(tuple[object, ...], get_args(annotation))
        args = [arg for arg in union_args if arg is not NoneType]
        if len(args) != 1:
            raise TypeError(f"Unsupported union type: {annotation}")
        return _json_type(args[0])
    origin = cast(object | None, get_origin(annotation))
    if origin is not None:
        return _json_type(origin)
    if isinstance(annotation, type) and annotation in _TYPE_MAP:
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

        signature = inspect.signature(fn)
        properties: JsonObject = {}
        required: list[str] = []
        wants_context = False

        for param_name, param in signature.parameters.items():
            annotation = _resolve_parameter_annotation(fn, param)
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
    return json.dumps(output, ensure_ascii=False)


def _output_content_type(output: ToolOutput) -> str:
    if isinstance(output, str):
        return "text/plain"
    return "application/json"


def _resolve_parameter_annotation(
    fn: Callable[..., Awaitable[ToolOutput]], param: inspect.Parameter
) -> object:
    annotation = param.annotation
    if annotation is inspect.Parameter.empty:
        return str
    if isinstance(annotation, str):
        globalns = getattr(fn, "__globals__", {})
        return eval(annotation, globalns)
    return annotation

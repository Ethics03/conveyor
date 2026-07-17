from __future__ import annotations

import pytest

from agent.models import ToolCall
from tools.base import ExecutionContext, ToolRegistry, tool


@pytest.mark.anyio
async def test_tool_decorator_builds_schema_and_executes(tmp_path) -> None:
    @tool(permission="read", description="Echo a value.")
    async def echo(value: str, count: int = 1) -> str:
        return value * count

    registry = ToolRegistry()
    registry.register(echo)

    assert echo.schema.name == "echo"
    assert echo.schema.description == "Echo a value."
    assert echo.schema.parameters["properties"] == {
        "value": {"type": "string"},
        "count": {"type": "integer"},
    }
    assert echo.schema.parameters["required"] == ["value"]

    result = await registry.execute(
        ToolCall(name="echo", arguments={"value": "ha", "count": 2}),
        ExecutionContext(workspace=tmp_path),
    )

    assert result.ok is True
    assert result.name == "echo"
    assert result.content == "haha"
    assert result.metadata == {"permission": "read"}


@pytest.mark.anyio
async def test_tool_can_receive_execution_context(tmp_path) -> None:
    @tool(permission="read")
    async def current_workspace(context: ExecutionContext) -> str:
        return str(context.workspace)

    registry = ToolRegistry()
    registry.register(current_workspace)

    result = await registry.execute(
        ToolCall(name="current_workspace"),
        ExecutionContext(workspace=tmp_path),
    )

    assert result.ok is True
    assert result.content == str(tmp_path)


@pytest.mark.anyio
async def test_unknown_tool_returns_failed_result(tmp_path) -> None:
    registry = ToolRegistry()
    tool_call = ToolCall(name="missing")

    result = await registry.execute(tool_call, ExecutionContext(workspace=tmp_path))

    assert result.ok is False
    assert result.tool_call_id == tool_call.id
    assert result.name == "missing"
    assert result.content == "Unknown tool: missing"


@pytest.mark.anyio
async def test_tool_exception_returns_failed_result(tmp_path) -> None:
    @tool(permission="read")
    async def explode() -> str:
        raise ValueError("bad input")

    registry = ToolRegistry()
    registry.register(explode)

    result = await registry.execute(
        ToolCall(name="explode"),
        ExecutionContext(workspace=tmp_path),
    )

    assert result.ok is False
    assert result.name == "explode"
    assert result.content == "bad input"

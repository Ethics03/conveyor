from __future__ import annotations

import asyncio
import time

import pytest

from agent.cancellation import RunCancelled
from agent.models import ToolCall
from tools.base import ExecutionContext
from tools.registry import ToolRegistry
from tools.shell import MAX_COMMAND_OUTPUT_CHARS, bash


async def test_bash_runs_command_in_workspace(tmp_path) -> None:
    result = await bash.execute(
        {"command": "printf 'hello'"},
        ExecutionContext(workspace=tmp_path),
    )

    assert result == {
        "command": "printf 'hello'",
        "cwd": ".",
        "exit_code": 0,
        "stdout": "hello",
        "stderr": "",
        "timed_out": False,
        "truncated": False,
    }


async def test_bash_runs_in_workspace_subdirectory(tmp_path) -> None:
    directory = tmp_path / "src"
    directory.mkdir()

    result = await bash.execute(
        {"command": "pwd", "cwd": "src"},
        ExecutionContext(workspace=tmp_path),
    )

    assert result["cwd"] == "src"
    assert result["stdout"].strip() == str(directory)


async def test_bash_rejects_cwd_outside_workspace(tmp_path) -> None:
    registry = ToolRegistry([bash])

    result = await registry.execute(
        ToolCall(name="bash", arguments={"command": "pwd", "cwd": ".."}),
        ExecutionContext(workspace=tmp_path),
    )

    assert result.ok is False
    assert "Path escapes workspace" in result.content


async def test_bash_returns_nonzero_exit_and_stderr(tmp_path) -> None:
    result = await bash.execute(
        {"command": "printf 'failed' >&2; exit 7"},
        ExecutionContext(workspace=tmp_path),
    )

    assert result["exit_code"] == 7
    assert result["stdout"] == ""
    assert result["stderr"] == "failed"
    assert result["timed_out"] is False


async def test_bash_terminates_timed_out_command(tmp_path) -> None:
    result = await bash.execute(
        {"command": "sleep 5", "timeout_seconds": 0.1},
        ExecutionContext(workspace=tmp_path),
    )

    assert result["timed_out"] is True
    assert result["exit_code"] != 0


async def test_bash_terminates_cancelled_command(tmp_path) -> None:
    context = ExecutionContext(workspace=tmp_path)
    marker = tmp_path / "started"

    task = asyncio.create_task(
        bash.execute(
            {"command": "touch started; sleep 30"},
            context,
        )
    )
    deadline = time.monotonic() + 1
    while not marker.exists() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert marker.exists()

    started_at = time.monotonic()
    _ = context.cancellation.cancel("Stop command")
    with pytest.raises(RunCancelled, match="Stop command"):
        _ = await asyncio.wait_for(task, timeout=2)

    assert time.monotonic() - started_at < 2


async def test_bash_kills_command_that_ignores_termination(tmp_path) -> None:
    result = await bash.execute(
        {
            "command": "trap '' TERM; sleep 5",
            "timeout_seconds": 0.1,
        },
        ExecutionContext(workspace=tmp_path),
    )

    assert result["timed_out"] is True
    assert result["exit_code"] != 0


async def test_bash_strips_sensitive_environment_variables(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CONVEYOR_TEST_API_KEY", "secret-value")

    result = await bash.execute(
        {"command": "printf '%s' \"${CONVEYOR_TEST_API_KEY-missing}\""},
        ExecutionContext(workspace=tmp_path),
    )

    assert result["stdout"] == "missing"


async def test_bash_does_not_inherit_stdin(tmp_path) -> None:
    result = await bash.execute(
        {
            "command": (
                "if read -r value; then printf 'read:%s' \"$value\"; "
                "else printf 'eof'; fi"
            )
        },
        ExecutionContext(workspace=tmp_path),
    )

    assert result["stdout"] == "eof"


async def test_bash_bounds_output_while_command_is_running(tmp_path) -> None:
    result = await bash.execute(
        {
            "command": "printf 'start'; head -c 200000 /dev/zero | tr '\\0' x; printf 'end'"
        },
        ExecutionContext(workspace=tmp_path),
    )

    assert len(result["stdout"]) == MAX_COMMAND_OUTPUT_CHARS
    assert result["stdout"].startswith("start")
    assert result["stdout"].endswith("end")
    assert result["truncated"] is True


async def test_bash_cleans_up_background_descendants(tmp_path) -> None:
    started_at = time.monotonic()

    result = await bash.execute(
        {"command": "sleep 10 &", "timeout_seconds": 2},
        ExecutionContext(workspace=tmp_path),
    )

    assert time.monotonic() - started_at < 2
    assert result["exit_code"] == 0
    assert result["timed_out"] is False

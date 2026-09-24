from __future__ import annotations

import asyncio
import codecs
import os
import signal
from collections import deque
from math import isfinite
from pathlib import Path
from shutil import which

from tools.base import ExecutionContext, JsonObject, tool
from tools.workspace import (
    WorkspacePathError,
    WorkspaceToolError,
    resolve_workspace_path,
)

DEFAULT_COMMAND_TIMEOUT_SECONDS = 120.0
MAX_COMMAND_TIMEOUT_SECONDS = 600.0
MAX_COMMAND_OUTPUT_CHARS = 100_000
COMMAND_POLL_INTERVAL_SECONDS = 0.1
_OUTPUT_TRUNCATION_MARKER = "\n... output truncated ...\n"
_SENSITIVE_ENV_FRAGMENTS = (
    "ACCESS_KEY",
    "API_KEY",
    "AUTH_TOKEN",
    "BEARER_TOKEN",
    "CLIENT_SECRET",
    "CREDENTIAL",
    "PASSWORD",
    "PRIVATE_KEY",
)
_SENSITIVE_ENV_SUFFIXES = ("_SECRET", "_TOKEN")


def require_bash() -> str:
    executable = which("bash")
    if executable is None:
        raise WorkspaceToolError("bash is required for the bash tool")
    return executable


def _command_environment() -> dict[str, str]:
    environment: dict[str, str] = {}
    for name, value in os.environ.items():
        normalized = name.upper()
        if any(fragment in normalized for fragment in _SENSITIVE_ENV_FRAGMENTS):
            continue
        if normalized.endswith(_SENSITIVE_ENV_SUFFIXES):
            continue
        environment[name] = value
    return environment


class _BoundedOutput:
    def __init__(self, max_chars: int) -> None:
        self.max_chars: int = max_chars
        self._head_limit: int = max_chars // 2
        self._tail_limit: int = max_chars - self._head_limit
        self._head: list[str] = []
        self._tail: deque[str] = deque()
        self._head_chars: int = 0
        self._tail_chars: int = 0
        self._total_chars: int = 0

    def append(self, value: str) -> None:
        if not value:
            return

        self._total_chars += len(value)
        offset = 0
        if self._head_chars < self._head_limit:
            take = min(self._head_limit - self._head_chars, len(value))
            self._head.append(value[:take])
            self._head_chars += take
            offset = take

        remainder = value[offset:]
        if not remainder:
            return
        if len(remainder) >= self._tail_limit:
            self._tail.clear()
            self._tail.append(remainder[-self._tail_limit :])
            self._tail_chars = self._tail_limit
            return

        self._tail.append(remainder)
        self._tail_chars += len(remainder)
        while self._tail_chars > self._tail_limit:
            excess = self._tail_chars - self._tail_limit
            first = self._tail[0]
            if len(first) <= excess:
                self._tail.popleft()
                self._tail_chars -= len(first)
            else:
                self._tail[0] = first[excess:]
                self._tail_chars -= excess

    def render(self) -> tuple[str, bool]:
        head = "".join(self._head)
        tail = "".join(self._tail)
        if self._total_chars <= self.max_chars:
            return head + tail, False

        available = self.max_chars - len(_OUTPUT_TRUNCATION_MARKER)
        head_chars = available // 2
        tail_chars = available - head_chars
        return (
            head[:head_chars] + _OUTPUT_TRUNCATION_MARKER + tail[-tail_chars:],
            True,
        )


async def _drain_output(
    stream: asyncio.StreamReader,
    output: _BoundedOutput,
) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        while chunk := await stream.read(8192):
            output.append(decoder.decode(chunk))
        output.append(decoder.decode(b"", final=True))
    except OSError, ValueError, asyncio.CancelledError:
        return


async def _terminate_process(
    process: asyncio.subprocess.Process,
    wait_task: asyncio.Task[int],
) -> None:
    if os.name != "posix":
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(asyncio.shield(wait_task), timeout=1)
            except TimeoutError:
                process.kill()
        await wait_task
        return

    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        await wait_task
        return

    loop = asyncio.get_running_loop()
    deadline = loop.time() + 1
    while loop.time() < deadline:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            await wait_task
            return
        await asyncio.sleep(0.05)

    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass

    try:
        await asyncio.wait_for(asyncio.shield(wait_task), timeout=1)
    except TimeoutError:
        process.kill()
        await wait_task


async def _wait_for_process(
    process: asyncio.subprocess.Process,
    wait_task: asyncio.Task[int],
    *,
    context: ExecutionContext,
    timeout_seconds: float,
) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while process.returncode is None:
        context.cancellation.raise_if_cancelled()
        remaining = deadline - loop.time()
        if remaining <= 0:
            return True
        _, _ = await asyncio.wait(
            {wait_task},
            timeout=min(COMMAND_POLL_INTERVAL_SECONDS, remaining),
        )
    context.cancellation.raise_if_cancelled()
    return False


def _relative_cwd(workspace: Path, cwd: Path) -> str:
    relative = cwd.relative_to(workspace)
    return relative.as_posix() if relative.parts else "."


@tool(
    permission="dangerous",
    description=(
        "Run a foreground Bash command with an initial working directory "
        "inside the workspace."
    ),
)
async def bash(
    command: str,
    context: ExecutionContext,
    cwd: str = ".",
    timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> JsonObject:
    if not command.strip():
        raise WorkspaceToolError("Command cannot be empty")

    resolved_cwd = resolve_workspace_path(context, cwd)
    if not resolved_cwd.is_dir():
        raise WorkspacePathError(f"Not a directory: {cwd}")

    if not isfinite(timeout_seconds):
        raise WorkspaceToolError("Command timeout must be finite")
    normalized_timeout = max(
        0.1,
        min(float(timeout_seconds), MAX_COMMAND_TIMEOUT_SECONDS),
    )
    process = await asyncio.create_subprocess_exec(
        require_bash(),
        "-c",
        command,
        cwd=resolved_cwd,
        env=_command_environment(),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        _ = await process.wait()
        raise WorkspaceToolError("Command output pipes were not created")

    stdout_output = _BoundedOutput(MAX_COMMAND_OUTPUT_CHARS)
    stderr_output = _BoundedOutput(MAX_COMMAND_OUTPUT_CHARS)
    stdout_task = asyncio.create_task(_drain_output(process.stdout, stdout_output))
    stderr_task = asyncio.create_task(_drain_output(process.stderr, stderr_output))
    wait_task = asyncio.create_task(process.wait())
    try:
        timed_out = await _wait_for_process(
            process,
            wait_task,
            context=context,
            timeout_seconds=normalized_timeout,
        )
    except BaseException:
        await _terminate_process(process, wait_task)
        await asyncio.gather(stdout_task, stderr_task)
        raise

    # A foreground command must not leave background descendants running.
    await _terminate_process(process, wait_task)
    try:
        await asyncio.wait_for(
            asyncio.gather(stdout_task, stderr_task),
            timeout=3,
        )
    except TimeoutError as exc:
        stdout_task.cancel()
        stderr_task.cancel()
        raise WorkspaceToolError("Command output streams did not close") from exc

    if process.returncode is None:
        raise WorkspaceToolError("Command process did not terminate")

    stdout, stdout_truncated = stdout_output.render()
    stderr, stderr_truncated = stderr_output.render()
    workspace = context.workspace.resolve()

    return {
        "command": command,
        "cwd": _relative_cwd(workspace, resolved_cwd),
        "exit_code": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "truncated": stdout_truncated or stderr_truncated,
    }

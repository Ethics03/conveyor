from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections import deque
from math import isfinite
from pathlib import Path
from shutil import which
from typing import TextIO

from tools.base import ExecutionContext, JsonObject, tool
from tools.workspace import (
    WorkspacePathError,
    WorkspaceToolError,
    resolve_workspace_path,
)

DEFAULT_COMMAND_TIMEOUT_SECONDS = 120.0
MAX_COMMAND_TIMEOUT_SECONDS = 600.0
MAX_COMMAND_OUTPUT_CHARS = 100_000
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
            head[:head_chars]
            + _OUTPUT_TRUNCATION_MARKER
            + tail[-tail_chars:],
            True,
        )


def _drain_output(stream: TextIO, output: _BoundedOutput) -> None:
    try:
        while chunk := stream.read(8192):
            output.append(chunk)
    except (OSError, ValueError):
        pass


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if os.name != "posix":
        if process.poll() is None:
            process.terminate()
            try:
                _ = process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
        return

    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        if process.poll() is None:
            try:
                _ = process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                pass
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)

    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return

    try:
        _ = process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        _ = process.wait(timeout=1)


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
def bash(
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
    process = subprocess.Popen(
        [require_bash(), "-c", command],
        cwd=resolved_cwd,
        env=_command_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    if process.stdout is None or process.stderr is None:
        _terminate_process(process)
        raise WorkspaceToolError("Command output pipes were not created")

    stdout_output = _BoundedOutput(MAX_COMMAND_OUTPUT_CHARS)
    stderr_output = _BoundedOutput(MAX_COMMAND_OUTPUT_CHARS)
    stdout_thread = threading.Thread(
        target=_drain_output,
        args=(process.stdout, stdout_output),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_drain_output,
        args=(process.stderr, stderr_output),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    timed_out = False
    try:
        _ = process.wait(timeout=normalized_timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process(process)
    except BaseException:
        _terminate_process(process)
        raise
    else:
        # A foreground command must not leave background descendants running.
        _terminate_process(process)

    stdout_thread.join(timeout=2)
    stderr_thread.join(timeout=2)
    process.stdout.close()
    process.stderr.close()
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)

    if process.returncode is None:
        raise WorkspaceToolError("Command process did not terminate")
    if stdout_thread.is_alive() or stderr_thread.is_alive():
        raise WorkspaceToolError("Command output streams did not close")

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

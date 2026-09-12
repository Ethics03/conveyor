from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Collection, Iterator
from typing import IO


class ProcessExecutionError(RuntimeError):
    pass


class DelimitedTextReader:
    """Read bounded records from a text stream without unbounded readline calls."""

    def __init__(self, stream: IO[str]) -> None:
        self._stream = stream
        self._buffer = ""

    def read_until(self, delimiter: str, max_chars: int) -> str | None:
        while True:
            delimiter_index = self._buffer.find(delimiter)
            if delimiter_index >= 0:
                value = self._buffer[:delimiter_index]
                self._buffer = self._buffer[delimiter_index + len(delimiter) :]
                return value
            if len(self._buffer) > max_chars:
                raise ProcessExecutionError("process emitted an oversized record")

            chunk = self._stream.read(8192)
            if not chunk:
                if not self._buffer:
                    return None
                value = self._buffer
                self._buffer = ""
                return value
            self._buffer += chunk


def terminate_process(process: subprocess.Popen[str]) -> None:
    """Terminate a subprocess and its process group without waiting indefinitely."""
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


def _drain_bounded_text(stream: IO[str], chunks: list[str], max_chars: int) -> None:
    remaining = max_chars
    try:
        while chunk := stream.read(8192):
            if remaining <= 0:
                continue
            retained = chunk[:remaining]
            chunks.append(retained)
            remaining -= len(retained)
    except (OSError, ValueError):
        pass


def collect_process_page[T](
    *,
    command: list[str],
    cwd: os.PathLike[str],
    offset: int,
    limit: int,
    records: Callable[[IO[str]], Iterator[T]],
    timeout_seconds: float,
    max_error_chars: int,
    success_returncodes: Collection[int] = (0,),
) -> tuple[list[T], int, bool]:
    """Collect one result page and one lookahead item from a subprocess."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    if process.stdout is None or process.stderr is None:
        terminate_process(process)
        raise ProcessExecutionError("process output pipes were not created")
    stdout = process.stdout
    stderr = process.stderr

    page: list[T] = []
    observed = 0
    stop_after = offset + limit + 1
    worker_errors: list[Exception] = []
    stderr_chunks: list[str] = []

    def collect_stdout() -> None:
        nonlocal observed
        try:
            for item in records(stdout):
                observed += 1
                if observed > offset and len(page) < limit:
                    page.append(item)
                if observed >= stop_after:
                    return
        except Exception as exc:  # noqa: BLE001 - propagate failures from worker thread
            worker_errors.append(exc)

    stdout_thread = threading.Thread(target=collect_stdout, daemon=True)
    stderr_thread = threading.Thread(
        target=_drain_bounded_text,
        args=(stderr, stderr_chunks, max_error_chars),
        daemon=True,
    )
    started_at = time.monotonic()
    stdout_thread.start()
    stderr_thread.start()
    timed_out = False
    stopped_early = False
    try:
        stdout_thread.join(timeout=timeout_seconds)

        timed_out = stdout_thread.is_alive()
        stopped_early = observed >= stop_after
        if timed_out or stopped_early or worker_errors:
            terminate_process(process)
        else:
            remaining = max(0.1, timeout_seconds - (time.monotonic() - started_at))
            try:
                _ = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_process(process)
    except BaseException:
        terminate_process(process)
        raise
    finally:
        stdout.close()
        stderr.close()
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)

    if timed_out:
        raise ProcessExecutionError("process timed out")
    if worker_errors:
        error = worker_errors[0]
        raise ProcessExecutionError(f"invalid process output: {error}") from error
    if not stopped_early and process.returncode not in success_returncodes:
        message = "".join(stderr_chunks).strip() or "process failed"
        raise ProcessExecutionError(message)

    return page, observed, stopped_early

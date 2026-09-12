from __future__ import annotations

import fnmatch
from collections.abc import Callable, Iterator
from pathlib import Path
from shutil import which
from typing import IO, Literal, cast

from tools.base import ExecutionContext, JsonObject, JsonValue, tool
from tools.process import (
    DelimitedTextReader,
    ProcessExecutionError,
    collect_process_page,
)

DEFAULT_READ_OFFSET = 1
DEFAULT_READ_LIMIT = 500
MAX_READ_LIMIT = 2000
MAX_READ_FILE_CHARS = 100_000
MAX_READ_LINE_CHARS = 2_000
MAX_READ_MANY_FILES = 20
DEFAULT_READ_MANY_TOTAL_CHARS = 100_000
MAX_READ_MANY_TOTAL_CHARS = 250_000
DEFAULT_SEARCH_LIMIT = 50
MAX_SEARCH_LIMIT = 500
SEARCH_TIMEOUT_SECONDS = 30.0
MAX_SEARCH_ERROR_CHARS = 10_000
MAX_SEARCH_MATCH_CHARS = 4_000
MAX_SEARCH_RECORD_CHARS = MAX_SEARCH_MATCH_CHARS + 1_000
MAX_SEARCH_PATH_CHARS = 32_768
_LINE_TRUNCATION_MARKER = " ... [truncated]"


class WorkspacePathError(ValueError):
    pass


class WorkspaceToolError(RuntimeError):
    pass


def require_ripgrep() -> str:
    rg = which("rg")
    if rg is None:
        raise WorkspaceToolError("ripgrep is required for search_files")
    return rg


def resolve_workspace_path(context: ExecutionContext, path: str = ".") -> Path:
    workspace = context.workspace.resolve()
    candidate = (workspace / path).resolve()

    try:
        _ = candidate.relative_to(workspace)
    except ValueError as exc:
        raise WorkspacePathError(f"Path escapes workspace: {path}") from exc

    return candidate


def relative_workspace_path(context: ExecutionContext, path: Path) -> str:
    workspace = context.workspace.resolve()
    candidate = path.resolve()

    try:
        relative = candidate.relative_to(workspace)
    except ValueError as exc:
        raise WorkspacePathError(f"Path escapes workspace: {path}") from exc

    return relative.as_posix()


def _normalize_search_pagination(offset: int, limit: int) -> tuple[int, int]:
    page_offset = max(0, int(offset))
    page_limit = max(1, min(int(limit), MAX_SEARCH_LIMIT))
    return page_offset, page_limit


def _file_search_glob(pattern: str) -> str:
    normalized = pattern.strip() or "*"
    if "/" not in normalized and not normalized.startswith("*"):
        return f"*{normalized}*"
    return normalized


def _file_matches_pattern(path: str, pattern: str) -> bool:
    glob_pattern = _file_search_glob(pattern)
    return fnmatch.fnmatch(path, glob_pattern) or fnmatch.fnmatch(
        Path(path).name, glob_pattern
    )


def _normalize_ripgrep_file(path: str) -> str:
    if path == ".":
        return path
    return path.removeprefix("./")


def _file_records(stream: IO[str]) -> Iterator[str]:
    reader = DelimitedTextReader(stream)
    while (path := reader.read_until("\0", MAX_SEARCH_PATH_CHARS)) is not None:
        if path:
            yield _normalize_ripgrep_file(path)


def _content_records(stream: IO[str]) -> Iterator[JsonObject]:
    reader = DelimitedTextReader(stream)
    while (path := reader.read_until("\0", MAX_SEARCH_PATH_CHARS)) is not None:
        record = reader.read_until("\n", MAX_SEARCH_RECORD_CHARS)
        if record is None:
            raise WorkspaceToolError("ripgrep emitted an incomplete match")

        line_text, separator, text = record.partition(":")
        if not separator:
            raise WorkspaceToolError("ripgrep emitted an invalid match")
        try:
            line_number = int(line_text)
        except ValueError as exc:
            raise WorkspaceToolError("ripgrep emitted an invalid line number") from exc

        text = text.rstrip("\r")
        text_truncated = len(text) > MAX_SEARCH_MATCH_CHARS
        match: JsonObject = {
            "path": _normalize_ripgrep_file(path),
            "line": line_number,
            "text": text[:MAX_SEARCH_MATCH_CHARS],
        }
        if text_truncated:
            match["text_truncated"] = True
        yield match


def _run_ripgrep_page[T](
    *,
    command: list[str],
    workspace: Path,
    offset: int,
    limit: int,
    records: Callable[[IO[str]], Iterator[T]],
) -> tuple[list[T], int, bool]:
    try:
        return collect_process_page(
            command=command,
            cwd=workspace,
            offset=offset,
            limit=limit,
            records=records,
            timeout_seconds=SEARCH_TIMEOUT_SECONDS,
            max_error_chars=MAX_SEARCH_ERROR_CHARS,
            success_returncodes=(0, 1),
        )
    except ProcessExecutionError as exc:
        raise WorkspaceToolError(f"ripgrep failed: {exc}") from exc


def _ripgrep_files(
    *,
    rg: str,
    workspace: Path,
    root: str,
    glob_pattern: str,
    offset: int,
    limit: int,
) -> tuple[list[str], int, bool]:
    return _run_ripgrep_page(
        command=[
            rg,
            "--files",
            "--null",
            "--sort",
            "path",
            "-g",
            glob_pattern,
            root,
        ],
        workspace=workspace,
        offset=offset,
        limit=limit,
        records=_file_records,
    )


def _ripgrep_content(
    *,
    rg: str,
    workspace: Path,
    root: str,
    pattern: str,
    offset: int,
    limit: int,
) -> tuple[list[JsonObject], int, bool]:
    return _run_ripgrep_page(
        command=[
            rg,
            "--null",
            "--with-filename",
            "--no-heading",
            "--line-number",
            "--max-columns",
            str(MAX_SEARCH_MATCH_CHARS),
            "--max-columns-preview",
            "--sort",
            "path",
            "--color",
            "never",
            "--",
            pattern,
            root,
        ],
        workspace=workspace,
        offset=offset,
        limit=limit,
        records=_content_records,
    )


@tool(
    permission="read",
    description="Search workspace files by name or content using ripgrep.",
    parallel_safe=True,
)
def search_files(
    pattern: str,
    context: ExecutionContext,
    path: str = ".",
    target: Literal["files", "content"] = "files",
    offset: int = 0,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> JsonObject:
    resolved = resolve_workspace_path(context, path)
    if not resolved.exists():
        raise WorkspacePathError(f"Path not found: {path}")

    workspace = context.workspace.resolve()
    root = relative_workspace_path(context, resolved)
    rg = require_ripgrep()
    page_offset, page_limit = _normalize_search_pagination(offset, limit)

    if target == "files":
        if resolved.is_file():
            files = [root] if _file_matches_pattern(root, pattern) else []
            end = page_offset + page_limit
            file_page = files[page_offset:end]
            total_count = len(files)
            truncated = end < total_count
            total_count_is_exact = True
        else:
            file_page, total_count, truncated = _ripgrep_files(
                rg=rg,
                workspace=workspace,
                root=root,
                glob_pattern=_file_search_glob(pattern),
                offset=page_offset,
                limit=page_limit,
            )
            total_count_is_exact = not truncated

        response: JsonObject = {
            "target": "files",
            "pattern": pattern,
            "path": root,
            "files": cast(JsonValue, file_page),
            "offset": page_offset,
            "limit": page_limit,
            "total_count": total_count,
            "total_count_is_exact": total_count_is_exact,
            "truncated": truncated,
        }
        return response

    if target == "content":
        match_page, total_count, truncated = _ripgrep_content(
            rg=rg,
            workspace=workspace,
            root=root,
            pattern=pattern,
            offset=page_offset,
            limit=page_limit,
        )

        response: JsonObject = {
            "target": "content",
            "pattern": pattern,
            "path": root,
            "matches": cast(JsonValue, match_page),
            "offset": page_offset,
            "limit": page_limit,
            "total_count": total_count,
            "total_count_is_exact": not truncated,
            "truncated": truncated,
        }
        return response

    raise WorkspaceToolError("target must be 'files' or 'content'")


def _bounded_text_lines(file: IO[str]) -> Iterator[tuple[str, bool]]:
    parts: list[str] = []
    retained_chars = 0
    truncated = False
    pending_line = False

    while chunk := file.readline(8192):
        pending_line = True
        line_ended = chunk.endswith("\n")
        text = chunk[:-1] if line_ended else chunk
        remaining = MAX_READ_LINE_CHARS - retained_chars
        if remaining > 0:
            retained = text[:remaining]
            parts.append(retained)
            retained_chars += len(retained)
        if len(text) > max(0, remaining):
            truncated = True

        if line_ended:
            line = "".join(parts)
            if truncated:
                line += _LINE_TRUNCATION_MARKER
            yield line, truncated
            parts = []
            retained_chars = 0
            truncated = False
            pending_line = False

    if pending_line:
        line = "".join(parts)
        if truncated:
            line += _LINE_TRUNCATION_MARKER
        yield line, truncated


def _read_text_file(
    *,
    path: str,
    context: ExecutionContext,
    offset: int,
    limit: int,
    max_chars: int = MAX_READ_FILE_CHARS,
) -> tuple[JsonObject, bool]:
    resolved = resolve_workspace_path(context, path)
    if not resolved.is_file():
        raise WorkspacePathError(f"Not a file: {path}")

    page_offset = max(1, int(offset))
    page_limit = max(1, min(int(limit), MAX_READ_LIMIT))
    char_limit = max(1, min(int(max_chars), MAX_READ_FILE_CHARS))
    page_end = page_offset + page_limit - 1
    output_lines: list[str] = []
    output_chars = 0
    total_lines = 0
    total_lines_is_exact = False
    char_truncated = False
    line_truncated = False
    partial_line_truncated = False
    next_offset: int | None = None

    with resolved.open("r", encoding="utf-8") as file:
        for line_number, (line, was_truncated) in enumerate(
            _bounded_text_lines(file),
            start=1,
        ):
            total_lines = line_number
            if line_number < page_offset:
                continue

            numbered = f"{line_number}|{line}"
            separator_chars = 1 if output_lines else 0
            if output_chars + separator_chars + len(numbered) > char_limit:
                char_truncated = True
                if not output_lines:
                    partial_line_truncated = True
                    output_lines.append(numbered[:char_limit])
                    output_chars = len(output_lines[0])
                    next_char = file.read(1)
                    if next_char:
                        total_lines += 1
                        next_offset = line_number + 1
                    else:
                        total_lines_is_exact = True
                else:
                    next_offset = line_number
                break

            output_lines.append(numbered)
            output_chars += separator_chars + len(numbered)
            line_truncated = line_truncated or was_truncated

            if line_number >= page_end:
                next_char = file.read(1)
                if next_char:
                    total_lines += 1
                    next_offset = line_number + 1
                else:
                    total_lines_is_exact = True
                break
        else:
            total_lines_is_exact = True

    if total_lines_is_exact and total_lines > 0 and page_offset > total_lines:
        raise WorkspaceToolError(
            f"Offset {page_offset} exceeds the file's {total_lines} lines"
        )

    truncated = char_truncated or line_truncated or next_offset is not None
    result: JsonObject = {
        "path": relative_workspace_path(context, resolved),
        "content": "\n".join(output_lines),
        "offset": page_offset,
        "limit": page_limit,
        "total_lines": total_lines,
        "total_lines_is_exact": total_lines_is_exact,
        "truncated": truncated,
    }
    hints: list[str] = []
    if next_offset is not None:
        result["next_offset"] = next_offset
        hints.append(f"Continue with offset={next_offset}.")
    if line_truncated:
        result["line_truncated"] = True
        hints.append(
            f"Lines longer than {MAX_READ_LINE_CHARS:,} characters were clipped."
        )
    if partial_line_truncated:
        hints.append(
            "The first selected line exceeded the output budget; its omitted "
            "remainder cannot be retrieved with a line offset."
        )
    if hints:
        result["hint"] = " ".join(hints)
    return result, truncated


@tool(
    permission="read",
    description="Read a UTF-8 text file from the workspace with line pagination.",
    parallel_safe=True,
)
def read_file(
    path: str,
    context: ExecutionContext,
    offset: int = DEFAULT_READ_OFFSET,
    limit: int = DEFAULT_READ_LIMIT,
) -> JsonObject:
    result, _ = _read_text_file(
        path=path,
        context=context,
        offset=offset,
        limit=limit,
    )
    return result


@tool(
    permission="read",
    description="Read multiple UTF-8 workspace files in one bounded batch.",
    parallel_safe=True,
)
def read_many(
    paths: list[str],
    context: ExecutionContext,
    offset: int = DEFAULT_READ_OFFSET,
    limit: int = DEFAULT_READ_LIMIT,
    max_total_chars: int = DEFAULT_READ_MANY_TOTAL_CHARS,
) -> JsonObject:
    selected_paths = paths[:MAX_READ_MANY_FILES]
    char_limit = max(
        1,
        min(int(max_total_chars), MAX_READ_MANY_TOTAL_CHARS),
    )

    files: list[JsonObject] = []
    errors: list[JsonObject] = []
    total_chars = 0
    budget_truncated = False

    for path in selected_paths:
        remaining_chars = char_limit - total_chars
        if remaining_chars <= 0:
            break

        try:
            result, file_truncated = _read_text_file(
                path=path,
                context=context,
                offset=offset,
                limit=limit,
                max_chars=remaining_chars,
            )
        except (OSError, UnicodeError, WorkspacePathError) as exc:
            errors.append(
                {
                    "path": path,
                    "error": str(exc),
                }
            )
            continue

        content = result.get("content")
        if not isinstance(content, str):
            errors.append(
                {
                    "path": path,
                    "error": "File reader returned invalid content",
                }
            )
            continue

        if file_truncated:
            budget_truncated = True

        files.append(result)
        total_chars += len(content)

        if total_chars >= char_limit:
            break

    processed_count = len(files) + len(errors)

    return {
        "files": cast(JsonValue, files),
        "errors": cast(JsonValue, errors),
        "requested_count": len(paths),
        "processed_count": processed_count,
        "total_chars": total_chars,
        "max_total_chars": char_limit,
        "truncated": (
            budget_truncated
            or len(paths) > len(selected_paths)
            or processed_count < len(selected_paths)
        ),
    }


@tool(
    permission="write",
    description="Create or replace a UTF-8 text file inside the workspace.",
)
def write_file(
    path: str,
    content: str,
    context: ExecutionContext,
) -> JsonObject:
    resolved = resolve_workspace_path(context, path)
    if resolved.exists() and not resolved.is_file():
        raise WorkspacePathError(f"Not a file: {path}")

    created = not resolved.exists()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")

    return {
        "path": relative_workspace_path(context, resolved),
        "created": created,
        "bytes_written": len(content.encode("utf-8")),
    }

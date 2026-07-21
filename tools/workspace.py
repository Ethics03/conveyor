from __future__ import annotations

import fnmatch
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from shutil import which
from typing import cast

from tools.base import ExecutionContext
from tools.base import JsonObject
from tools.base import JsonValue
from tools.base import tool


DEFAULT_READ_OFFSET = 1
DEFAULT_READ_LIMIT = 500
MAX_READ_LIMIT = 2000
DEFAULT_SEARCH_LIMIT = 50
MAX_SEARCH_LIMIT = 500


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
    normalized_offset = max(0, int(offset))
    normalized_limit = max(1, min(int(limit), MAX_SEARCH_LIMIT))
    return normalized_offset, normalized_limit


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


def _as_object_mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return cast(Mapping[str, object], value)


def _content_match_sort_key(match: JsonObject) -> tuple[str, int]:
    path = match.get("path")
    line = match.get("line")
    path_key = path if isinstance(path, str) else ""
    line_key = line if isinstance(line, int) else 0
    return path_key, line_key


def _ripgrep_files(
    *,
    rg: str,
    workspace: Path,
    root: str,
    glob_pattern: str,
) -> list[str]:
    result = subprocess.run(
        [rg, "--files", "-g", glob_pattern, root],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode not in {0, 1}:
        message = result.stderr.strip() or "file search failed"
        raise WorkspaceToolError(message)

    return sorted(
        _normalize_ripgrep_file(line) for line in result.stdout.splitlines() if line
    )


def _ripgrep_content(
    *,
    rg: str,
    workspace: Path,
    root: str,
    pattern: str,
) -> list[JsonObject]:
    result = subprocess.run(
        [rg, "--json", "--color", "never", "--", pattern, root],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode not in {0, 1}:
        message = result.stderr.strip() or "content search failed"
        raise WorkspaceToolError(message)

    matches: list[JsonObject] = []
    for line in result.stdout.splitlines():
        event = _as_object_mapping(cast(object, json.loads(line)))
        if event is None:
            continue
        if event.get("type") != "match":
            continue

        data = _as_object_mapping(event.get("data"))
        if data is None:
            continue

        path_data = _as_object_mapping(data.get("path"))
        lines_data = _as_object_mapping(data.get("lines"))
        line_number = data.get("line_number")
        if path_data is None:
            continue
        if lines_data is None:
            continue

        path_text = path_data.get("text")
        lines_text = lines_data.get("text")
        if not isinstance(path_text, str):
            continue
        if not isinstance(lines_text, str):
            continue
        if not isinstance(line_number, int):
            continue

        match: JsonObject = {
            "path": _normalize_ripgrep_file(path_text),
            "line": line_number,
            "text": lines_text.rstrip("\n"),
        }
        matches.append(match)

    matches.sort(key=_content_match_sort_key)
    return matches


@tool(
    permission="read",
    description="Search workspace files by name or content using ripgrep.",
)
async def search_files(
    pattern: str,
    context: ExecutionContext,
    path: str = ".",
    target: str = "files",
    offset: int = 0,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> JsonObject:
    resolved = resolve_workspace_path(context, path)
    if not resolved.exists():
        raise WorkspacePathError(f"Path not found: {path}")

    workspace = context.workspace.resolve()
    root = relative_workspace_path(context, resolved)
    rg = require_ripgrep()
    normalized_offset, normalized_limit = _normalize_search_pagination(offset, limit)

    if target == "files":
        if resolved.is_file():
            files = [root] if _file_matches_pattern(root, pattern) else []
        else:
            files = _ripgrep_files(
                rg=rg,
                workspace=workspace,
                root=root,
                glob_pattern=_file_search_glob(pattern),
        )

        end = normalized_offset + normalized_limit
        file_page: list[str] = files[normalized_offset:end]

        response: JsonObject = {
            "target": "files",
            "pattern": pattern,
            "path": root,
            "files": cast(JsonValue, file_page),
            "offset": normalized_offset,
            "limit": normalized_limit,
            "total_count": len(files),
            "truncated": end < len(files),
        }
        return response

    if target == "content":
        matches = _ripgrep_content(
            rg=rg,
            workspace=workspace,
            root=root,
            pattern=pattern,
        )
        end = normalized_offset + normalized_limit
        match_page: list[JsonObject] = matches[normalized_offset:end]

        response: JsonObject = {
            "target": "content",
            "pattern": pattern,
            "path": root,
            "matches": cast(JsonValue, match_page),
            "offset": normalized_offset,
            "limit": normalized_limit,
            "total_count": len(matches),
            "truncated": end < len(matches),
        }
        return response

    raise WorkspaceToolError("target must be 'files' or 'content'")


@tool(
    permission="read",
    description="Read a UTF-8 text file from the workspace with line pagination.",
)
async def read_file(
    path: str,
    context: ExecutionContext,
    offset: int = DEFAULT_READ_OFFSET,
    limit: int = DEFAULT_READ_LIMIT,
) -> JsonObject:
    resolved = resolve_workspace_path(context, path)
    if not resolved.is_file():
        raise WorkspacePathError(f"Not a file: {path}")

    normalized_offset = max(1, int(offset))
    normalized_limit = max(1, min(int(limit), MAX_READ_LIMIT))
    lines = resolved.read_text(encoding="utf-8").splitlines()
    start = normalized_offset - 1
    end = start + normalized_limit
    selected = lines[start:end]
    numbered = [
        f"{line_number}|{line}"
        for line_number, line in enumerate(selected, start=normalized_offset)
    ]

    return {
        "path": relative_workspace_path(context, resolved),
        "content": "\n".join(numbered),
        "offset": normalized_offset,
        "limit": normalized_limit,
        "total_lines": len(lines),
        "truncated": end < len(lines),
    }

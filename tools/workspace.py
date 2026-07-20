from __future__ import annotations

from pathlib import Path
from shutil import which

from tools.base import ExecutionContext
from tools.base import JsonObject
from tools.base import tool


DEFAULT_READ_OFFSET = 1
DEFAULT_READ_LIMIT = 500
MAX_READ_LIMIT = 2000


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

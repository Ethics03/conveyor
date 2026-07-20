from __future__ import annotations

import pytest
from shutil import which

from agent.models import ToolCall
from tools.base import ExecutionContext
from tools.registry import ToolRegistry
from tools.workspace import (
    WorkspacePathError,
    read_file,
    relative_workspace_path,
    require_ripgrep,
    resolve_workspace_path,
)


def test_require_ripgrep_returns_rg_path() -> None:
    if which("rg") is None:
        pytest.skip("ripgrep is not installed")

    rg = require_ripgrep()

    assert rg


def test_resolve_workspace_path_allows_relative_inside_path(tmp_path) -> None:
    context = ExecutionContext(workspace=tmp_path)

    resolved = resolve_workspace_path(context, "notes.txt")

    assert resolved == (tmp_path / "notes.txt").resolve()


def test_resolve_workspace_path_allows_absolute_inside_path(tmp_path) -> None:
    context = ExecutionContext(workspace=tmp_path)
    inside = tmp_path / "notes.txt"

    resolved = resolve_workspace_path(context, str(inside))

    assert resolved == inside.resolve()


def test_resolve_workspace_path_rejects_parent_escape(tmp_path) -> None:
    context = ExecutionContext(workspace=tmp_path)

    with pytest.raises(WorkspacePathError, match="Path escapes workspace"):
        resolve_workspace_path(context, "../outside.txt")


def test_resolve_workspace_path_rejects_absolute_outside_path(tmp_path) -> None:
    context = ExecutionContext(workspace=tmp_path)
    outside = tmp_path.parent / "outside.txt"

    with pytest.raises(WorkspacePathError, match="Path escapes workspace"):
        resolve_workspace_path(context, str(outside))


def test_resolve_workspace_path_rejects_symlink_escape(tmp_path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(outside)
    context = ExecutionContext(workspace=tmp_path)

    with pytest.raises(WorkspacePathError, match="Path escapes workspace"):
        resolve_workspace_path(context, "link")


def test_relative_workspace_path_returns_posix_relative_path(tmp_path) -> None:
    context = ExecutionContext(workspace=tmp_path)
    nested = tmp_path / "agent" / "models.py"

    relative = relative_workspace_path(context, nested)

    assert relative == "agent/models.py"


def test_relative_workspace_path_rejects_outside_path(tmp_path) -> None:
    context = ExecutionContext(workspace=tmp_path)
    outside = tmp_path.parent / "outside.txt"

    with pytest.raises(WorkspacePathError, match="Path escapes workspace"):
        relative_workspace_path(context, outside)


@pytest.mark.anyio
async def test_read_file_returns_numbered_lines_with_metadata(tmp_path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    result = await read_file.execute(
        {"path": "notes.txt"},
        ExecutionContext(workspace=tmp_path),
    )

    assert result == {
        "path": "notes.txt",
        "content": "1|alpha\n2|beta\n3|gamma",
        "offset": 1,
        "limit": 500,
        "total_lines": 3,
        "truncated": False,
    }


@pytest.mark.anyio
async def test_read_file_supports_line_pagination(tmp_path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

    result = await read_file.execute(
        {"path": "notes.txt", "offset": 2, "limit": 2},
        ExecutionContext(workspace=tmp_path),
    )

    assert result["content"] == "2|two\n3|three"
    assert result["offset"] == 2
    assert result["limit"] == 2
    assert result["total_lines"] == 4
    assert result["truncated"] is True


@pytest.mark.anyio
async def test_read_file_rejects_paths_outside_workspace(tmp_path) -> None:
    registry = ToolRegistry([read_file])
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    result = await registry.execute(
        ToolCall(name="read_file", arguments={"path": str(outside)}),
        ExecutionContext(workspace=tmp_path),
    )

    assert result.ok is False
    assert "Path escapes workspace" in result.content

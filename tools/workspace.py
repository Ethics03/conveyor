from __future__ import annotations
from pathlib import Path
from tools.base import ExecutionContext


class WorkspacePathError(ValueError):
    pass 


def resolve_workspace_path(context: ExecutionContext, path: str = ".") -> Path:
    workspace = context.workspace.resolve()
    candidate = (workspace / path).resolve()

    try:
        _ = candidate.relative_to(workspace)
    except ValueError as exc:
        raise WorkspacePathError(f"Path escapes workspace: {path}") from exc 

    return candidate 

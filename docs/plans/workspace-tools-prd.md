# Workspace Tools PRD

## Summary

Workspace tools are Conveyor's built-in capability layer for inspecting and later modifying files inside a user-selected workspace. They provide the safe boundary between model-requested tool calls and local filesystem access.

The first version should be read-only: path resolution, file listing, file reading, and text search. Write and shell tools should come later after approval and policy handling are designed.

## Product Requirements

- Tools operate only inside the active workspace from `ExecutionContext`.
- Path handling must reject `..` escapes, absolute paths outside the workspace, and symlink escapes.
- Tool outputs should use workspace-relative paths.
- File listings and searches should be deterministic, sorted, capped, and skip noisy directories.
- File reads should enforce a byte limit and report truncation.
- Search should return structured matches with path, line number, and text.
- Tool failures should surface as failed `ToolResult`s through `ToolRegistry.execute`, not unhandled runtime crashes.

## Architecture

```text
agent loop
  -> receives ToolCall
  -> calls ToolRegistry.execute(...)
  -> registry finds Tool
  -> workspace tool receives ExecutionContext
  -> workspace helper resolves path safely
  -> filesystem operation runs
  -> registry wraps output in ToolResult
```

Responsibilities:

- `tools/base.py`: generic tool protocol, schema inference, registry execution.
- `tools/workspace.py`: concrete workspace filesystem tools and path safety helpers.
- `agent/loop.py`: orchestration only; it should not directly read or write files.
- `storage/`: persistence for messages, events, and tool results.

## V0 Tool Set

- `list_files`
  - permission: `read`
  - returns relative file paths
  - supports root, pattern, and limit
  - skips ignored directories

- `read_file`
  - permission: `read`
  - reads UTF-8 text
  - supports max byte limit
  - returns path, content, and truncation flag

- `search_files`
  - permission: `read`
  - searches literal text
  - supports root, pattern, and limit
  - returns path, line number, and matching line text

## Safety Rules

- All paths resolve against `ExecutionContext.workspace`.
- Absolute paths are allowed only if they resolve inside the workspace.
- Symlinks are followed during resolution so symlink escapes are blocked.
- No tool should return files from `.git`, `.venv`, `__pycache__`, `node_modules`, build directories, or cache directories.
- Binary or undecodable files should be skipped during search.
- Read limits and result limits are mandatory, not optional safeguards.

## Step-by-Step Implementation Plan

1. Add `tools/workspace.py` with path safety constants and helper functions.
2. Implement `resolve_workspace_path(context, path)`.
3. Test inside paths, absolute inside paths, relative escapes, and symlink escapes.
4. Implement `relative_workspace_path(context, path)`.
5. Implement deterministic workspace file iteration with ignore rules and limit handling.
6. Implement and test `list_files`.
7. Implement and test `read_file`.
8. Implement and test `search_files`.
9. Add a helper that registers all default workspace tools into a `ToolRegistry`.
10. Keep `write_file` and `run_command` deferred until approval flow exists.

## Acceptance Criteria

- Tests pass for all path containment cases.
- `list_files` returns sorted workspace-relative file paths and honors limits.
- `read_file` cannot read outside the workspace and reports truncation.
- `search_files` finds literal matches and skips binary/unreadable files.
- Tool functions remain small and rely on shared helpers for path safety.
- The agent loop can later use these tools without knowing filesystem details.

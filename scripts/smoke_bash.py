from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from agent.models import ToolCall
from tools.base import ExecutionContext
from tools.registry import ToolRegistry
from tools.shell import bash

DEFAULT_COMMAND = (
    "printf 'stdout: bash works\\n'; "
    "printf 'stderr: sample\\n' >&2; "
    "printf 'cwd: '; pwd"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the Conveyor bash tool")
    parser.add_argument(
        "--command",
        default=DEFAULT_COMMAND,
        help="Bash command to execute",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="workspace root exposed through ExecutionContext",
    )
    parser.add_argument(
        "--cwd",
        default=".",
        help="initial directory relative to the workspace",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="command timeout in seconds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    result = ToolRegistry([bash]).execute(
        ToolCall(
            name="bash",
            arguments={
                "command": args.command,
                "cwd": args.cwd,
                "timeout_seconds": args.timeout,
            },
        ),
        ExecutionContext(workspace=workspace),
    )

    execution = json.loads(result.content) if result.ok else None
    print(
        json.dumps(
            {
                "workspace": str(workspace),
                "tool_result": asdict(result),
                "execution": execution,
            },
            indent=2,
        )
    )

    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

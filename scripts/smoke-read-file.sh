#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="${1:-$repo_root}"
path="${2:-README.md}"
offset="${3:-1}"
limit="${4:-40}"

cd "$repo_root"

uv run python - "$workspace" "$path" "$offset" "$limit" <<'PY'
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from agent.models import ToolCall
from tools.base import ExecutionContext
from tools.registry import ToolRegistry
from tools.workspace import read_file


async def main() -> None:
    workspace = Path(sys.argv[1])
    path = sys.argv[2]
    offset = int(sys.argv[3])
    limit = int(sys.argv[4])

    registry = ToolRegistry([read_file])
    result = await registry.execute(
        ToolCall(
            name="read_file",
            arguments={"path": path, "offset": offset, "limit": limit},
        ),
        ExecutionContext(workspace=workspace),
    )

    print(json.dumps(asdict(result), indent=2))


asyncio.run(main())
PY

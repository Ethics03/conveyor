#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="${1:-$repo_root}"
pattern="${2:-*.py}"
path="${3:-.}"
target="${4:-files}"
offset="${5:-0}"
limit="${6:-20}"

cd "$repo_root"

uv run python - "$workspace" "$pattern" "$path" "$target" "$offset" "$limit" <<'PY'
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from agent.models import ToolCall
from tools.base import ExecutionContext
from tools.registry import ToolRegistry
from tools.workspace import search_files


async def main() -> None:
    workspace = Path(sys.argv[1])
    pattern = sys.argv[2]
    path = sys.argv[3]
    target = sys.argv[4]
    offset = int(sys.argv[5])
    limit = int(sys.argv[6])

    registry = ToolRegistry([search_files])
    result = await registry.execute(
        ToolCall(
            name="search_files",
            arguments={
                "pattern": pattern,
                "path": path,
                "target": target,
                "offset": offset,
                "limit": limit,
            },
        ),
        ExecutionContext(workspace=workspace),
    )

    print(json.dumps(asdict(result), indent=2))


asyncio.run(main())
PY

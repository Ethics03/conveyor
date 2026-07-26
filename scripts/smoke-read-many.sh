#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="${1:-$repo_root}"

if (( $# > 0 )); then
  shift
fi

paths=("$@")
if (( ${#paths[@]} == 0 )); then
  paths=("README.md" "pyproject.toml")
fi

cd "$repo_root"

uv run python - "$workspace" "${paths[@]}" <<'PY'
import json
import sys
from dataclasses import asdict
from pathlib import Path

from agent.models import ToolCall
from tools.base import ExecutionContext
from tools.registry import ToolRegistry
from tools.workspace import read_many


def main() -> None:
    workspace = Path(sys.argv[1])
    paths = sys.argv[2:]

    registry = ToolRegistry([read_many])
    result = registry.execute(
        ToolCall(
            name="read_many",
            arguments={"paths": paths},
        ),
        ExecutionContext(workspace=workspace),
    )

    print(json.dumps(asdict(result), indent=2))


main()
PY

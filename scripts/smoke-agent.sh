#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="${1:-$repo_root}"

uv_args=()
if [[ -f "$repo_root/.env" ]]; then
  uv_args+=(--env-file "$repo_root/.env")
fi

cd "$repo_root"

uv run "${uv_args[@]}" python -m scripts.smoke_agent "$workspace"

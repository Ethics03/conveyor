#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv_args=()
if [[ -f "$repo_root/.env" ]]; then
  uv_args+=(--env-file "$repo_root/.env")
fi

cd "$repo_root"

uv run "${uv_args[@]}" python -m scripts.smoke_session_titles "$@"

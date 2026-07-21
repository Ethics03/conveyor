#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="${1:-$repo_root}"
pattern="${2:-ToolRegistry}"
path="${3:-.}"
offset="${4:-0}"
limit="${5:-10}"

"$repo_root/scripts/smoke-search-files.sh" \
  "$workspace" \
  "$pattern" \
  "$path" \
  content \
  "$offset" \
  "$limit"

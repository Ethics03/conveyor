from __future__ import annotations

from tools.registry import ToolRegistry
from tools.workspace import read_file, read_many, search_files


def build_default_registry() -> ToolRegistry:
    return ToolRegistry([
        read_file,
        read_many,
        search_files,
    ])

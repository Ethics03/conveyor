from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from providers.base import Provider
from storage.store import Store
from tools.registry import ToolRegistry


@dataclass(slots=True)
class Runtime:
    """Owns the shared resources used to execute agent runs."""

    store: Store
    provider: Provider
    registry: ToolRegistry
    workspace: Path

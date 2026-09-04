from __future__ import annotations

from dataclasses import dataclass, field
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
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.workspace = self.workspace.expanduser().resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"Workspace does not exist: {self.workspace}")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Runtime is closed")

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        try:
            self.provider.close()
        finally:
            self.store.close()

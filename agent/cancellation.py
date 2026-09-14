from __future__ import annotations

import threading

DEFAULT_CANCELLATION_REASON = "Run cancelled by user"


class RunCancelled(RuntimeError):
    """Raised at a safe execution boundary when cancellation is requested."""


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = DEFAULT_CANCELLATION_REASON

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def cancel(self, reason: str = DEFAULT_CANCELLATION_REASON) -> bool:
        normalized = reason.strip() or DEFAULT_CANCELLATION_REASON
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = normalized
            self._event.set()
            return True

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise RunCancelled(self.reason)

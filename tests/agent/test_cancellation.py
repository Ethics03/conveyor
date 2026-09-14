from __future__ import annotations

import pytest

from agent.cancellation import CancellationToken, RunCancelled


def test_cancellation_token_preserves_first_reason() -> None:
    cancellation = CancellationToken()

    assert cancellation.cancel("Stopped by user") is True
    assert cancellation.cancel("Later reason") is False
    assert cancellation.cancelled is True
    assert cancellation.reason == "Stopped by user"

    with pytest.raises(RunCancelled, match="Stopped by user"):
        cancellation.raise_if_cancelled()

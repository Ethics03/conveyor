from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from agent.models import ProviderResponse
from providers.base import ProviderRequest


class FakeProvider:
    name = "fake"

    def __init__(self, responses: Iterable[ProviderResponse | str] | None = None) -> None:
        self._responses: deque[ProviderResponse | str] = deque(
            responses or [ProviderResponse.message("ok")]
        )

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        if not self._responses:
            return ProviderResponse.message("ok")

        response = self._responses.popleft()
        if isinstance(response, str):
            return ProviderResponse.message(response)
        return response

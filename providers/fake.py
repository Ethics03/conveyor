from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from agent.models import ProviderResponse
from providers.base import ModelLimits, ProviderRequest

DEFAULT_FAKE_MODEL_LIMITS = ModelLimits(
    context_window_tokens=200_000,
    max_output_tokens=4_096,
)


class FakeProvider:
    name = "fake"

    def __init__(
        self,
        responses: Iterable[ProviderResponse | str] | None = None,
        *,
        model_limits: ModelLimits | None = None,
    ) -> None:
        self._responses: deque[ProviderResponse | str] = deque(
            responses or [ProviderResponse.message("ok")]
        )
        self.requests: list[ProviderRequest] = []
        self.closed = False
        self._model_limits = model_limits or DEFAULT_FAKE_MODEL_LIMITS

    def model_limits(self, model: str | None = None) -> ModelLimits:
        return self._model_limits

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        if self.closed:
            raise RuntimeError("Provider is closed")

        self.requests.append(request)
        if not self._responses:
            return ProviderResponse.message("ok")

        response = self._responses.popleft()
        if isinstance(response, str):
            return ProviderResponse.message(response)
        return response

    def close(self) -> None:
        self.closed = True

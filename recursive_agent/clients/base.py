"""Small model-client interface shared by real and test clients."""

from __future__ import annotations

from typing import Any, Protocol

from ..types import ModelResponse


class ModelClient(Protocol):
    model_name: str

    def completion(
        self, messages: list[dict[str, Any]], *, timeout: float | None = None
    ) -> ModelResponse: ...

    def close(self) -> None: ...


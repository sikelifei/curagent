from __future__ import annotations

import copy
import threading
from typing import Any, Callable

from recursive_agent.types import ModelCallUsage, ModelResponse


class FakeClient:
    model_name = "fake-model"

    def __init__(self, factory: "FakeFactory") -> None:
        self.factory = factory
        self.closed = False

    def completion(
        self, messages: list[dict[str, Any]], *, timeout: float | None = None
    ) -> ModelResponse:
        snapshot = copy.deepcopy(messages)
        with self.factory.lock:
            self.factory.calls.append(snapshot)
        content = self.factory.handler(snapshot, timeout)
        return ModelResponse(
            content=content,
            usage=ModelCallUsage(
                model=self.model_name,
                input_tokens=3,
                output_tokens=2,
            ),
        )

    def close(self) -> None:
        self.closed = True
        with self.factory.lock:
            self.factory.closed_clients += 1


class FakeFactory:
    def __init__(
        self,
        handler: Callable[[list[dict[str, Any]], float | None], str],
    ) -> None:
        self.handler = handler
        self.calls: list[list[dict[str, Any]]] = []
        self.created = 0
        self.closed_clients = 0
        self.lock = threading.Lock()

    def __call__(self, backend: str, kwargs: dict[str, Any]) -> FakeClient:
        assert backend == "openai"
        with self.lock:
            self.created += 1
        return FakeClient(self)


def initial_task(messages: list[dict[str, Any]]) -> str:
    content = messages[1]["content"]
    for marker in ("Task:\n", "Delegated task:\n"):
        if content.startswith(marker):
            return content[len(marker) :].split("\n\n", 1)[0]
    raise AssertionError(f"Unexpected initial user prompt: {content!r}")

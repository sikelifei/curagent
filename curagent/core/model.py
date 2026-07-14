"""Model interface used by every node."""

from __future__ import annotations

from typing import Protocol, Sequence

from curagent.core.types import ModelResponse, ToolSchema


class ToolCallingModel(Protocol):
    async def generate(self, prompt: str, tools: Sequence[ToolSchema]) -> ModelResponse:
        """Return an unmodified native-tool or strict-JSON response."""

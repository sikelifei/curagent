"""Azure OpenAI adapter, based on rlm's provider client."""

from __future__ import annotations

import os
from typing import Any

import openai

from ..exceptions import ConfigurationError
from ..types import ModelCallUsage, ModelResponse


class AzureOpenAIClient:
    def __init__(
        self,
        *,
        model_name: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        azure_deployment: str | None = None,
        timeout: float = 300.0,
        sampling_args: dict[str, Any] | None = None,
        **client_kwargs: Any,
    ) -> None:
        self.model_name = model_name or model or azure_deployment or ""
        if not self.model_name:
            raise ConfigurationError("Azure backend requires model_name or azure_deployment")
        endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ConfigurationError("Azure backend requires azure_endpoint")
        self.timeout = float(timeout)
        self.sampling_args = {
            key: value for key, value in (sampling_args or {}).items() if value is not None
        }
        allowed = {"organization", "project", "default_headers", "default_query", "max_retries"}
        constructor_kwargs = {
            key: value for key, value in client_kwargs.items() if key in allowed
        }
        self.client = openai.AzureOpenAI(
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=endpoint,
            api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_deployment=azure_deployment,
            timeout=self.timeout,
            **constructor_kwargs,
        )

    def completion(
        self, messages: list[dict[str, Any]], *, timeout: float | None = None
    ) -> ModelResponse:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            timeout=min(self.timeout, timeout) if timeout is not None else self.timeout,
            **self.sampling_args,
        )
        usage = getattr(response, "usage", None)
        return ModelResponse(
            content=response.choices[0].message.content or "",
            usage=ModelCallUsage(
                model=self.model_name,
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            ),
        )

    def close(self) -> None:
        self.client.close()


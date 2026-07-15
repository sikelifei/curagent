"""Anthropic Messages API adapter, based on rlm's provider client."""

from __future__ import annotations

import os
from typing import Any

from ..exceptions import ConfigurationError
from ..types import ModelCallUsage, ModelResponse


class AnthropicClient:
    def __init__(
        self,
        *,
        model_name: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
        max_tokens: int = 32768,
        sampling_args: dict[str, Any] | None = None,
        **client_kwargs: Any,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ConfigurationError(
                "The anthropic backend requires: pip install 'curagent[anthropic]'"
            ) from exc
        self.model_name = model_name or model or ""
        if not self.model_name:
            raise ConfigurationError("Anthropic backend requires model_name")
        self.timeout = float(timeout)
        args = dict(sampling_args or {})
        self.max_tokens = int(args.pop("max_tokens", max_tokens))
        if "stop" in args:
            args["stop_sequences"] = args.pop("stop")
        allowed_sampling = {"temperature", "top_p", "top_k", "stop_sequences"}
        self.sampling_args = {
            key: value for key, value in args.items() if key in allowed_sampling and value is not None
        }
        allowed_client = {"base_url", "max_retries", "default_headers"}
        constructor_kwargs = {
            key: value for key, value in client_kwargs.items() if key in allowed_client
        }
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            timeout=self.timeout,
            **constructor_kwargs,
        )

    def completion(
        self, messages: list[dict[str, Any]], *, timeout: float | None = None
    ) -> ModelResponse:
        system: str | None = None
        provider_messages = []
        for message in messages:
            if message.get("role") == "system":
                system = str(message.get("content", ""))
            else:
                provider_messages.append(message)
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": provider_messages,
            **self.sampling_args,
        }
        if system:
            kwargs["system"] = system
        if timeout is not None:
            kwargs["timeout"] = min(self.timeout, timeout)
        response = self.client.messages.create(**kwargs)
        usage = response.usage
        text = "".join(getattr(block, "text", "") for block in response.content)
        return ModelResponse(
            content=text,
            usage=ModelCallUsage(
                model=self.model_name,
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            ),
        )

    def close(self) -> None:
        self.client.close()


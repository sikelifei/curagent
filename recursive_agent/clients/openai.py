"""OpenAI-compatible chat completions client, adapted from rlm."""

from __future__ import annotations

import os
from typing import Any

import openai

from ..exceptions import ConfigurationError
from ..types import ModelCallUsage, ModelResponse


class OpenAIClient:
    def __init__(
        self,
        *,
        model_name: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 300.0,
        sampling_args: dict[str, Any] | None = None,
        **client_kwargs: Any,
    ) -> None:
        self.model_name = model_name or model or ""
        if not self.model_name:
            raise ConfigurationError("backend_kwargs must include model_name or model")
        self.timeout = float(timeout)
        self.sampling_args = {
            key: value for key, value in (sampling_args or {}).items() if value is not None
        }
        if api_key is None:
            api_key = _default_api_key(base_url)
        allowed_client_keys = {
            "organization",
            "project",
            "default_headers",
            "default_query",
            "max_retries",
        }
        constructor_kwargs = {
            key: value for key, value in client_kwargs.items() if key in allowed_client_keys
        }
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout,
            **constructor_kwargs,
        )

    def completion(
        self, messages: list[dict[str, Any]], *, timeout: float | None = None
    ) -> ModelResponse:
        request_args = dict(self.sampling_args)
        extra_body = request_args.pop("extra_body", None)
        if extra_body is not None:
            request_args["extra_body"] = extra_body
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            timeout=min(self.timeout, timeout) if timeout is not None else self.timeout,
            **request_args,
        )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = _extract_cost(usage)
        return ModelResponse(
            content=content,
            usage=ModelCallUsage(
                model=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
            ),
        )

    def close(self) -> None:
        self.client.close()


def _extract_cost(usage: Any) -> float | None:
    if usage is None:
        return None
    direct = getattr(usage, "cost", None)
    if direct is not None:
        return float(direct)
    extra = getattr(usage, "model_extra", None) or {}
    cost = extra.get("cost")
    if cost is None:
        cost = (extra.get("cost_details") or {}).get("upstream_inference_cost")
    return float(cost) if cost is not None else None


def _default_api_key(base_url: str | None) -> str | None:
    environment_keys = {
        "https://openrouter.ai/api/v1": "OPENROUTER_API_KEY",
        "https://ai-gateway.vercel.sh/v1": "AI_GATEWAY_API_KEY",
        "https://api.portkey.ai/v1": "PORTKEY_API_KEY",
    }
    key = environment_keys.get((base_url or "").rstrip("/"))
    return os.getenv(key) if key else os.getenv("OPENAI_API_KEY")

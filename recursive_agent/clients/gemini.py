"""Google Gemini adapter, based on rlm's provider client."""

from __future__ import annotations

import os
from typing import Any

from ..exceptions import ConfigurationError
from ..types import ModelCallUsage, ModelResponse


class GeminiClient:
    def __init__(
        self,
        *,
        model_name: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
        sampling_args: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ConfigurationError(
                "The gemini backend requires: pip install 'curagent[gemini]'"
            ) from exc
        self._types = types
        self.model_name = model_name or model or ""
        if not self.model_name:
            raise ConfigurationError("Gemini backend requires model_name")
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ConfigurationError("Gemini backend requires an API key")
        self.sampling_args = dict(sampling_args or {})
        self.client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=int(float(timeout) * 1000)),
        )

    def completion(
        self, messages: list[dict[str, Any]], *, timeout: float | None = None
    ) -> ModelResponse:
        system: str | None = None
        contents = []
        for message in messages:
            role = message.get("role")
            content = str(message.get("content", ""))
            if role == "system":
                system = content
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(
                self._types.Content(
                    role=gemini_role,
                    parts=[self._types.Part(text=content)],
                )
            )
        config_args = dict(self.sampling_args)
        if "max_tokens" in config_args:
            config_args["max_output_tokens"] = config_args.pop("max_tokens")
        if "stop" in config_args:
            config_args["stop_sequences"] = config_args.pop("stop")
        allowed_config = {
            "temperature",
            "top_p",
            "top_k",
            "candidate_count",
            "seed",
            "frequency_penalty",
            "presence_penalty",
            "stop_sequences",
            "max_output_tokens",
        }
        config_args = {
            key: value
            for key, value in config_args.items()
            if key in allowed_config and value is not None
        }
        config = self._types.GenerateContentConfig(
            system_instruction=system,
            **config_args,
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        usage = getattr(response, "usage_metadata", None)
        return ModelResponse(
            content=response.text or "",
            usage=ModelCallUsage(
                model=self.model_name,
                input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
                output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            ),
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

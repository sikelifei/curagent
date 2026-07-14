"""OpenAI-compatible chat-completions tool-calling adapter."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from curagent.core.errors import ModelServiceError
from curagent.core.types import ModelResponse, ToolSchema


class OpenAICompatibleModel:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 120.0,
        native_tools: bool = True,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.native_tools = native_tools
        self.chat_template_kwargs = dict(chat_template_kwargs or {})

    async def generate(self, prompt: str, tools: Sequence[ToolSchema]) -> ModelResponse:
        payload = self._build_payload(prompt, tools)
        response = await asyncio.to_thread(self._request, payload)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("chat-completions response has no choices")
        message = choices[0].get("message")
        if not isinstance(message, Mapping):
            raise RuntimeError("chat-completions choice has no assistant message")
        if self.native_tools:
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                raise RuntimeError("assistant tool_calls is not a list")
            return ModelResponse(raw_response=dict(message), tool_calls=tuple(tool_calls), protocol="native")
        return ModelResponse(raw_response=message.get("content"), protocol="json")

    def _build_payload(self, prompt: str, tools: Sequence[ToolSchema]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.native_tools:
            payload["tools"] = [tool.to_model_dict() for tool in tools]
            payload["tool_choice"] = "required"
            payload["parallel_tool_calls"] = False
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        return payload

    def _request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            retryable = exc.code in {408, 409, 425, 429} or exc.code >= 500
            raise ModelServiceError(
                f"model HTTP {exc.code}: {detail}", retryable=retryable
            ) from exc
        except urllib.error.URLError as exc:
            raise ModelServiceError(f"model connection error: {exc}", retryable=True) from exc
        if not isinstance(value, Mapping):
            raise RuntimeError("chat-completions response is not an object")
        return value


def load_model_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load a YAML model config") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise ValueError("model config must be an object")
    planner = value.get("planner", value)
    if not isinstance(planner, Mapping):
        raise ValueError("planner config must be an object")
    planner_type = str(planner.get("type", "api"))
    section = planner.get(planner_type, planner)
    if not isinstance(section, Mapping):
        raise ValueError(f"planner.{planner_type} must be an object")
    result = dict(section)
    if "api_key_env" in result:
        result["api_key"] = os.environ.get(str(result.pop("api_key_env")), "")
    allowed = {
        "base_url",
        "model",
        "api_key",
        "temperature",
        "max_tokens",
        "timeout",
        "native_tools",
        "chat_template_kwargs",
    }
    return {key: item for key, item in result.items() if key in allowed}

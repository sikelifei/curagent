"""Model client factory, retaining the provider routes from rlm."""

from .base import ModelClient
from .openai import OpenAIClient


def get_client(backend: str, backend_kwargs: dict):
    kwargs = dict(backend_kwargs)
    if backend == "openai":
        return OpenAIClient(**kwargs)
    if backend == "vllm":
        if not kwargs.get("base_url"):
            raise ValueError("base_url is required for the vllm backend")
        kwargs.setdefault("api_key", "not-needed")
        return OpenAIClient(**kwargs)
    if backend == "openrouter":
        kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        return OpenAIClient(**kwargs)
    if backend == "vercel":
        kwargs.setdefault("base_url", "https://ai-gateway.vercel.sh/v1")
        return OpenAIClient(**kwargs)
    if backend == "portkey":
        kwargs.setdefault("base_url", "https://api.portkey.ai/v1")
        return OpenAIClient(**kwargs)
    if backend == "azure_openai":
        from .azure_openai import AzureOpenAIClient

        return AzureOpenAIClient(**kwargs)
    if backend == "anthropic":
        from .anthropic import AnthropicClient

        return AnthropicClient(**kwargs)
    if backend == "gemini":
        from .gemini import GeminiClient

        return GeminiClient(**kwargs)
    raise ValueError(f"Unsupported backend: {backend}")


__all__ = ["ModelClient", "OpenAIClient", "get_client"]

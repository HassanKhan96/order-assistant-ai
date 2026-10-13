"""Create a provider-neutral adapter from configuration."""

import os
from pathlib import Path
from typing import Any, Mapping

from .base import AIAdapter, AIError, AIQuotaError, AIResponseError


def create_ai_adapter(provider: str | None = None, *, env: Mapping[str, str] | None = None,
                      client: Any = None, **overrides: Any) -> AIAdapter:
    """Read project .env when env is omitted; explicit overrides win.

    Only the selected SDK is imported. Passing env makes tests/configuration
    independent of the process environment and the local .env file.
    """
    if env is None:
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
        except ImportError:
            pass
        env = os.environ
    selected = (provider if provider is not None else env.get("AI_PROVIDER", "gemini")).strip().lower()
    if selected not in {"gemini", "openai", "qwen"}:
        raise ValueError("AI_PROVIDER must be 'gemini', 'openai', or 'qwen'")
    prefix = selected.upper()
    kwargs: dict[str, Any] = {
        "client": client,
        "dimensions": int(env.get("AI_EMBEDDING_DIMENSIONS", "768")),
        "batch_size": int(env.get("AI_EMBEDDING_BATCH_SIZE", "32")),
    }
    if env.get(f"{prefix}_EMBEDDING_MODEL"):
        kwargs["embedding_model"] = env[f"{prefix}_EMBEDDING_MODEL"]
    if selected == "qwen":
        # Local and keyless: no API key, text model, or request pacing applies.
        kwargs.update(overrides)
        from .qwen import QwenAdapter
        return QwenAdapter(**kwargs)
    kwargs["api_key"] = env.get(f"{prefix}_API_KEY")
    kwargs["requests_per_minute"] = float(
        env.get("AI_REQUESTS_PER_MINUTE", "10" if selected == "gemini" else "60"))
    if env.get(f"{prefix}_TEXT_MODEL"):
        kwargs["text_model"] = env[f"{prefix}_TEXT_MODEL"]
    kwargs.update(overrides)
    if kwargs.get("client") is None and not kwargs.get("api_key"):
        raise AIError(f"{prefix}_API_KEY is required")
    if selected == "gemini":
        from .gemini import GeminiAdapter
        return GeminiAdapter(**kwargs)
    from .openai import OpenAIAdapter
    return OpenAIAdapter(**kwargs)


__all__ = ["AIAdapter", "AIError", "AIQuotaError", "AIResponseError", "create_ai_adapter"]

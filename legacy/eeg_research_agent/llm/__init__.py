"""LLM access layer: provider-agnostic, credential-safe, cacheable.

Providers
---------
``AnthropicProvider``   Claude models (``anthropic``)
``DeepSeekProvider``    DeepSeek models (``deepseek``)
``OfflineProvider``     deterministic template renderer (default; needs no key)

Selection is by ``LLM_PROVIDER`` (or inferred from the model name).  A model id
the provider does not offer is a hard error - the pipeline never silently falls
back to a different model.  The deterministic EEG pipeline does not depend on
any of this: the offline provider is the default, so every stage runs with zero
credentials.
"""

from __future__ import annotations

import os
from typing import Any

from .offline import OfflineProvider
from .provider import (
    AnthropicProvider,
    DeepSeekProvider,
    LLMClient,
    LLMMessage,
    LLMProvider,
    LLMReply,
    LLMRequest,
    LLMUnavailableError,
    ModelNotAvailableError,
    build_provider,
    resolve_provider,
)

__all__ = [
    "AnthropicProvider",
    "DeepSeekProvider",
    "OfflineProvider",
    "LLMClient",
    "LLMMessage",
    "LLMProvider",
    "LLMReply",
    "LLMRequest",
    "LLMUnavailableError",
    "ModelNotAvailableError",
    "build_provider",
    "resolve_provider",
    "build_llm_client",
    "provider_status",
]


def provider_status(provider: str = "offline", model: str | None = None) -> dict[str, Any]:
    """Describe what *would* happen for a given provider without calling it."""
    resolved, resolved_model = resolve_provider(provider, model or os.environ.get("LLM_MODEL"))
    key_var = {"anthropic": "ANTHROPIC_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}.get(resolved)
    return {
        "provider": resolved,
        "model": resolved_model,
        "requires_api_key": resolved != "offline",
        "api_key_present": bool(key_var and os.environ.get(key_var)),
        "api_key_env_var": key_var,
        "networked": resolved != "offline",
    }


def build_llm_client(
    config: Any | None = None,
    provider: str | None = None,
    model: str | None = None,
    allow_fallback: bool | None = None,
) -> LLMClient:
    """Build a client from the project config, preferring the config's provider.

    When the config selects a networked provider the offline provider is attached
    as a *fallback* (unless fallback is explicitly disabled), and every fallback
    reply is tagged so the report can state which sections used model reasoning.
    """
    settings = getattr(getattr(config, "settings", None), "llm", None)
    provider = (
        provider
        or (settings.provider if settings else None)
        or os.environ.get("LLM_PROVIDER", "offline")
    )
    model = model or (settings.model if settings else None) or os.environ.get("LLM_MODEL")
    if allow_fallback is None:
        allow_fallback = bool(settings.allow_offline_fallback) if settings else True
    cache_path = None
    if config is not None and settings is not None and settings.request_cache:
        cache_path = config.path("logs") / "llm_cache.json"
    fallback: OfflineProvider | None = OfflineProvider()
    try:
        active = build_provider(provider=provider, model=model)
    except LLMUnavailableError:
        if not allow_fallback:
            raise
        active, fallback, allow_fallback = OfflineProvider(), None, False
    if isinstance(active, OfflineProvider):
        fallback = None
    return LLMClient(
        provider=active,
        fallback=fallback,
        allow_fallback=bool(allow_fallback),
        cache_path=cache_path,
    )

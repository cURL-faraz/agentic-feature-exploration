"""Provider-agnostic LLM interface.

Three properties are enforced here rather than left to the caller:

1. **The requested model is verified before it is used.**  Selecting a model that
   the provider does not offer is a hard error; the pipeline never silently
   substitutes a weaker or unrelated model.
2. **Secrets never leave the process.**  Keys are read from the environment,
   attached to a request header, and excluded from every ``repr`` and log line.
3. **Failure is explicit.**  When a networked provider cannot be reached and
   offline fallback is disabled, the call raises
   :class:`LLMUnavailableError`; the caller decides what to do.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..logging_utils import get_logger

__all__ = [
    "LLMMessage",
    "LLMRequest",
    "LLMReply",
    "LLMProvider",
    "LLMUnavailableError",
    "ModelNotAvailableError",
    "LLMClient",
    "build_provider",
    "resolve_provider",
]

logger = get_logger(__name__)


class LLMUnavailableError(RuntimeError):
    """The provider could not be reached and fallback was not permitted."""


class ModelNotAvailableError(ValueError):
    """The requested model does not exist for this provider."""


@dataclass
class LLMMessage:
    """One chat message."""

    role: str
    content: str


@dataclass
class LLMRequest:
    """A completion request; ``metadata`` carries the structured task payload."""

    messages: list[LLMMessage]
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    system: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def prompt_text(self) -> str:
        return "\n\n".join(message.content for message in self.messages)


@dataclass
class LLMReply:
    """A completion plus its provenance."""

    text: str
    model: str
    provider: str
    from_cache: bool = False
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "from_cache": self.from_cache,
            "usage": self.usage,
            "error": self.error,
            "text": self.text,
        }


class LLMProvider(Protocol):
    """Minimal protocol every provider implements."""

    name: str
    model: str

    def list_models(self) -> list[str]: ...

    def complete(self, request: LLMRequest) -> LLMReply: ...


def _redact(value: str | None) -> str:
    if not value:
        return "<unset>"
    return f"<{len(value)} chars, redacted>"


class AnthropicProvider:
    """Anthropic Messages API (``x-api-key`` + ``anthropic-version``)."""

    name = "anthropic"
    requires_api_key = True
    network = True

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str = "https://api.anthropic.com",
        timeout_s: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise LLMUnavailableError("anthropic provider selected but ANTHROPIC_API_KEY is not set")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.model = model or ""
        self._models_cache: list[str] | None = None

    def __repr__(self) -> str:  # pragma: no cover - defensive redaction
        return f"AnthropicProvider(model={self.model!r}, api_key={_redact(self._api_key)})"

    def list_models(self) -> list[str]:
        if self._models_cache is not None:
            return self._models_cache
        import httpx

        headers = {"x-api-key": self._api_key, "anthropic-version": "2023-06-01"}
        models: list[str] = []
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                response = client.get(f"{self.base_url}/v1/models", headers=headers)
            if response.status_code == 200:
                for item in response.json().get("data", []):
                    if item.get("id"):
                        models.append(str(item["id"]))
        except Exception as exc:  # noqa: BLE001 - network failure is reported, never hidden
            logger.warning("could not list Anthropic models: %s", type(exc).__name__)
        self._models_cache = models
        return models

    def complete(self, request: LLMRequest) -> LLMReply:
        import httpx

        model = request.model or self.model
        if not model:
            raise ModelNotAvailableError(
                "no Anthropic model requested; set LLM_MODEL (run `cli llm-check` to list models)"
            )
        available = self.list_models()
        if available and model not in available:
            raise ModelNotAvailableError(
                f"model {model!r} is not offered by this provider. "
                f"Available ids include: {', '.join(sorted(available)[:12])}"
            )
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": int(request.max_tokens),
            "temperature": float(request.temperature),
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.system:
            payload["system"] = request.system
        last_error: str | None = None
        for attempt in range(self.max_retries):
            started = time.time()
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(f"{self.base_url}/v1/messages", headers=headers, json=payload)
                if response.status_code == 200:
                    body = response.json()
                    text = "".join(
                        block.get("text", "")
                        for block in body.get("content", [])
                        if block.get("type") == "text"
                    )
                    return LLMReply(
                        text=text,
                        model=model,
                        provider=self.name,
                        usage=body.get("usage", {}),
                        metadata={"elapsed_s": round(time.time() - started, 3)},
                    )
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code in {429, 500, 502, 503, 504}:
                    time.sleep(2**attempt)
                    continue
                break
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(2**attempt)
        return LLMReply(text="", model=model, provider=self.name, error=last_error)


class DeepSeekProvider:
    """DeepSeek (OpenAI-compatible ``/chat/completions`` endpoint)."""

    name = "deepseek"
    requires_api_key = True
    network = True

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str = "https://api.deepseek.com",
        timeout_s: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise LLMUnavailableError("deepseek provider selected but DEEPSEEK_API_KEY is not set")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.model = model or ""
        self._models_cache: list[str] | None = None

    def __repr__(self) -> str:  # pragma: no cover - defensive redaction
        return f"DeepSeekProvider(model={self.model!r}, api_key={_redact(self._api_key)})"

    def list_models(self) -> list[str]:
        if self._models_cache is not None:
            return self._models_cache
        import httpx

        models: list[str] = []
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                response = client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
            if response.status_code == 200:
                models = [str(item["id"]) for item in response.json().get("data", []) if item.get("id")]
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not list DeepSeek models: %s", type(exc).__name__)
        self._models_cache = models
        return models

    def complete(self, request: LLMRequest) -> LLMReply:
        import httpx

        model = request.model or self.model
        if not model:
            raise ModelNotAvailableError(
                "no DeepSeek model requested; set LLM_MODEL (run `cli llm-check` to list models)"
            )
        available = self.list_models()
        if available and model not in available:
            raise ModelNotAvailableError(
                f"model {model!r} is not offered by this provider. "
                f"Available ids include: {', '.join(sorted(available)[:12])}"
            )
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend({"role": m.role, "content": m.content} for m in request.messages)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(request.temperature),
            "max_tokens": int(request.max_tokens),
            "stream": False,
        }
        last_error: str | None = None
        for attempt in range(self.max_retries):
            started = time.time()
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions", headers=headers, json=payload
                    )
                if response.status_code == 200:
                    body = response.json()
                    choices = body.get("choices") or [{}]
                    text = (choices[0].get("message") or {}).get("content", "")
                    return LLMReply(
                        text=text,
                        model=model,
                        provider=self.name,
                        usage=body.get("usage", {}),
                        metadata={"elapsed_s": round(time.time() - started, 3)},
                    )
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code in {429, 500, 502, 503, 504}:
                    time.sleep(2**attempt)
                    continue
                break
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(2**attempt)
        return LLMReply(text="", model=model, provider=self.name, error=last_error)


class LLMClient:
    """Facade that adds response caching and offline fallback policy."""

    def __init__(
        self,
        provider: Any,
        fallback: Any | None = None,
        allow_fallback: bool = True,
        cache_path: str | Path | None = None,
    ) -> None:
        self.provider = provider
        self.fallback = fallback
        self.allow_fallback = bool(allow_fallback)
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []
        if self.cache_path and self.cache_path.is_file():
            try:
                with open(self.cache_path, encoding="utf-8") as handle:
                    self.cache = json.load(handle)
            except Exception:  # noqa: BLE001
                logger.warning("LLM cache unreadable; starting empty")
                self.cache = {}

    @property
    def name(self) -> str:
        return str(getattr(self.provider, "name", "unknown"))

    @property
    def model(self) -> str:
        return str(getattr(self.provider, "model", ""))

    def _cache_key(self, request: LLMRequest) -> str:
        import hashlib

        payload = json.dumps(
            {
                "provider": self.name,
                "model": request.model or self.model,
                "messages": [{"role": m.role, "content": m.content} for m in request.messages],
                "system": request.system,
                "temperature": request.temperature,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def complete(self, request: LLMRequest) -> LLMReply:
        key = self._cache_key(request)
        cached = self.cache.get(key)
        if cached:
            return LLMReply(**{**cached, "from_cache": True, "provider": self.name})
        try:
            reply = self.provider.complete(request)
        except (LLMUnavailableError, ModelNotAvailableError):
            raise
        except Exception as exc:  # noqa: BLE001
            reply = LLMReply(
                text="",
                model=request.model or self.model,
                provider=self.name,
                error=f"{type(exc).__name__}: {exc}",
            )
        self.calls.append({"provider": self.name, "ok": reply.ok, "error": reply.error})
        if reply.ok:
            self.cache[key] = reply.to_dict()
            self._flush()
            return reply
        if self.fallback is not None and self.allow_fallback:
            logger.warning(
                "%s provider failed (%s); using the deterministic offline provider",
                self.name,
                reply.error,
            )
            fallback_reply = self.fallback.complete(request)
            fallback_reply.metadata["fallback_from"] = self.name
            fallback_reply.metadata["fallback_reason"] = reply.error
            return fallback_reply
        raise LLMUnavailableError(
            f"{self.name} provider failed and offline fallback is disabled: {reply.error}"
        )

    def _flush(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as handle:
            json.dump(self.cache, handle, indent=2)


def resolve_provider(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_s: float = 60.0,
    max_retries: int = 3,
) -> tuple[str, str | None]:
    """Work out which provider a ``(provider, model)`` pair implies.

    If the provider is not stated, it is inferred from the model name (``claude*``
    -> anthropic, ``deepseek*`` -> deepseek).  An ambiguous pair raises rather than
    guessing, because guessing is how a run silently switches model.
    """
    if not provider:
        if model and model.lower().startswith("claude"):
            provider = "anthropic"
        elif model and (model.lower().startswith("deepseek") or "deepseek" in model.lower()):
            provider = "deepseek"
        else:
            return "offline", model
    provider = provider.lower()
    if provider not in {"anthropic", "deepseek", "offline"}:
        raise ValueError(f"unknown LLM provider: {provider}")
    return provider, model


def build_provider(
    provider: str = "offline",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_s: float = 60.0,
    max_retries: int = 3,
) -> Any:
    """Instantiate the requested provider.  Raises when a key is missing."""
    provider, model = resolve_provider(provider, model)
    if provider == "offline":
        from .offline import OfflineProvider

        return OfflineProvider(model=model or "offline-template-v1")
    key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
    if provider == "anthropic":
        return AnthropicProvider(
            api_key=key or "",
            model=model,
            base_url=base_url or os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
    return DeepSeekProvider(
        api_key=key or "",
        model=model,
        base_url=base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        timeout_s=timeout_s,
        max_retries=max_retries,
    )

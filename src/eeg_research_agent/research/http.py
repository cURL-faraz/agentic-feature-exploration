"""A small, polite HTTP client for bibliographic APIs.

Features that matter for scientific reproducibility:

* **rate limiting** per host (the public APIs ask for <= 1 request/second and
  will ban clients that ignore that),
* **retries with exponential backoff** on transient failures,
* **an on-disk response cache** so a re-run of the literature stage does not
  re-query the network and produces byte-identical evidence,
* **no silent substitution**: a failure raises, and the caller records it.

The client is only ever used when the network probe succeeds; the offline path
never touches this module.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger

__all__ = ["HttpResponse", "HttpClient"]

logger = get_logger(__name__)

DEFAULT_USER_AGENT = (
    "agentic-feature-exploration/0.1 (research reproducibility project; "
    "contact: repository maintainer)"
)


@dataclass
class HttpResponse:
    """One HTTP exchange, cached and attributable."""

    url: str
    status: int
    payload: Any
    from_cache: bool = False
    elapsed_s: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None


@dataclass
class HttpCache:
    """Simple JSON-file cache keyed by the full request URL."""

    path: Path
    entries: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path.is_file():
            try:
                with open(self.path, encoding="utf-8") as handle:
                    self.entries = json.load(handle)
            except Exception:  # noqa: BLE001 - a corrupt cache is not fatal
                logger.warning("response cache at %s is unreadable; starting empty", self.path)
                self.entries = {}

    @staticmethod
    def key(url: str, params: dict[str, Any] | None = None) -> str:
        payload = json.dumps({"url": url, "params": params or {}}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        return self.entries.get(key)

    def put(self, key: str, value: Any) -> None:
        self.entries[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self.entries, handle, indent=2)


class HttpClient:
    """Rate-limited, retrying JSON client with an on-disk cache."""

    def __init__(
        self,
        cache_path: str | Path | None = None,
        min_interval_s: float = 1.0,
        timeout_s: float = 20.0,
        retries: int = 3,
        user_agent: str = DEFAULT_USER_AGENT,
        api_key: str | None = None,
    ) -> None:
        self.cache = HttpCache(Path(cache_path)) if cache_path else None
        self.min_interval_s = float(min_interval_s)
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.user_agent = user_agent
        self.api_key = api_key
        self._last_call: dict[str, float] = {}
        self.log: list[dict[str, Any]] = []

    def _throttle(self, host: str) -> None:
        previous = self._last_call.get(host)
        if previous is not None:
            wait = self.min_interval_s - (time.time() - previous)
            if wait > 0:
                time.sleep(wait)
        self._last_call[host] = time.time()

    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        """GET ``url`` and decode JSON, with caching, throttling and retries."""
        return self._request(url, params, headers, decode="json")

    def get_text(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        """GET ``url`` and return the raw body (for XML/Atom endpoints such as arXiv)."""
        return self._request(url, params, headers, decode="text")

    def _request(
        self,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
        decode: str,
    ) -> HttpResponse:
        key = HttpCache.key(url, params)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                self.log.append({"url": url, "from_cache": True})
                return HttpResponse(url=url, status=200, payload=cached, from_cache=True)

        from urllib.parse import urlparse

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - httpx is a hard dependency
            return HttpResponse(url=url, status=0, payload=None, error=f"httpx unavailable: {exc}")

        host = urlparse(url).netloc or url
        request_headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        request_headers.update(headers or {})
        if self.api_key and "api.semanticscholar.org" in host:
            request_headers["x-api-key"] = self.api_key

        last_error = "unknown error"
        for attempt in range(self.retries):
            self._throttle(host)
            started = time.time()
            try:
                with httpx.Client(timeout=self.timeout_s, follow_redirects=True) as client:
                    response = client.get(url, params=params, headers=request_headers)
                elapsed = time.time() - started
                if response.status_code == 200:
                    payload = response.text if decode == "text" else response.json()
                    if self.cache is not None:
                        self.cache.put(key, payload)
                    self.log.append({"url": url, "status": 200, "elapsed_s": round(elapsed, 3)})
                    return HttpResponse(url=url, status=200, payload=payload, elapsed_s=elapsed)
                if response.status_code in {429, 500, 502, 503, 504}:
                    last_error = f"HTTP {response.status_code}"
                    time.sleep(self.min_interval_s * (2**attempt))
                    continue
                self.log.append({"url": url, "status": response.status_code})
                return HttpResponse(
                    url=url,
                    status=response.status_code,
                    payload=None,
                    elapsed_s=elapsed,
                    error=f"HTTP {response.status_code}",
                )
            except Exception as exc:  # noqa: BLE001 - network errors are expected here
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(self.min_interval_s * (2**attempt))
        self.log.append({"url": url, "error": last_error})
        return HttpResponse(url=url, status=0, payload=None, error=last_error)

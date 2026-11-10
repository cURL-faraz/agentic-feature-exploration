"""LLM access through OpenRouter (free models), plus a scripted offline client.

Production concerns that matter more than the prompt itself:

* **Model routing per role.**  Each agent role has its own ordered model list.
  The critic deliberately uses a *different model family* from the inventor, so
  the two do not share the same blind spots.
* **Fallback.**  Free models disappear, get rate-limited or return garbage.  On
  any failure the client walks down the role's list, then (optionally) any model
  that is free today.  The run degrades instead of dying.
* **Rate limit + daily cap.**  The free tier allows roughly 20 requests/minute and
  ~50/day (more with account credits).  Calls are spaced and counted per day.
* **Cache.**  Identical requests are answered from disk: re-runs are free and a
  crashed run can be resumed without paying again.
* **Keys never leave this module**: read from the environment, not logged, and
  stripped from the environment of the process that runs generated code.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


class LLMError(RuntimeError):
    pass


class DailyCapReached(LLMError):
    pass


@dataclass
class LLMResult:
    text: str
    model: str
    cached: bool = False
    seconds: float = 0.0
    usage: dict = field(default_factory=dict)


def list_free_models(timeout: float = 30) -> list[dict]:
    """Models whose prompt and completion price are both zero today (no key needed)."""
    import urllib.request

    with urllib.request.urlopen(OPENROUTER_MODELS_URL, timeout=timeout) as response:
        data = json.loads(response.read().decode())["data"]
    free = []
    for model in data:
        pricing = model.get("pricing") or {}
        if str(pricing.get("prompt")) in {"0", "0.0"} and str(pricing.get("completion")) in {"0", "0.0"}:
            free.append({"id": model["id"], "context_length": model.get("context_length"),
                         "name": model.get("name")})
    return sorted(free, key=lambda m: m["id"])


class _Usage:
    """Per-day call counter persisted to disk (survives restarts)."""

    def __init__(self, path: Path):
        self.path = path

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except Exception:  # noqa: BLE001
            return {}

    def today(self) -> int:
        return int(self._load().get(dt.date.today().isoformat(), 0))

    def add(self) -> None:
        data = self._load()
        key = dt.date.today().isoformat()
        data[key] = int(data.get(key, 0)) + 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data))


class OpenRouterClient:
    def __init__(self, config: dict, on_call: Callable[[dict], None] | None = None):
        self.cfg = config["llm"]
        root = Path(config["data"]["root"])
        self.cache_path = root / "llm_cache.jsonl"
        self.usage = _Usage(root / "llm_usage.json")
        self.on_call = on_call or (lambda record: None)
        self._last_call = 0.0
        self._cache = self._load_cache()
        self._free: list[str] | None = None
        self._client = None

    # -------------------------------------------------------------- plumbing #
    def _load_cache(self) -> dict:
        cache = {}
        if self.cache_path.is_file():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                    cache[item["key"]] = item
                except Exception:  # noqa: BLE001
                    continue
        return cache

    def _sdk(self):
        if self._client is None:
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise LLMError("OPENROUTER_API_KEY is not set (put it in .env; see README)")
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.cfg["base_url"], api_key=key, timeout=float(self.cfg["timeout_s"]),
                max_retries=0, default_headers={"X-Title": "featurelab"},
            )
        return self._client

    def _models_for(self, role: str) -> list[str]:
        models = list(self.cfg["roles"].get(role) or [])
        if self.cfg.get("auto_free_fallback"):
            if self._free is None:
                try:
                    self._free = [m["id"] for m in list_free_models()]
                except Exception:  # noqa: BLE001
                    self._free = []
            preferred = [m for m in models if m in self._free] if self._free else models
            models = preferred + [m for m in models if m not in preferred] + [
                m for m in self._free if m not in models]
        return models

    # ------------------------------------------------------------------ call #
    def chat(self, role: str, system: str, user: str) -> LLMResult:
        temperature = float((self.cfg.get("temperature") or {}).get(role, 0.3))
        errors = []
        for model in self._models_for(role)[:8]:
            key = hashlib.sha256(json.dumps([model, system, user, temperature]).encode()).hexdigest()
            if key in self._cache:
                hit = self._cache[key]
                self.on_call({"role": role, "model": model, "ok": True, "cached": True, "seconds": 0})
                return LLMResult(hit["text"], model, cached=True)
            if self.usage.today() >= int(self.cfg["daily_call_cap"]):
                raise DailyCapReached(f"daily cap of {self.cfg['daily_call_cap']} LLM calls reached")
            wait = float(self.cfg["min_seconds_between_calls"]) - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            started = time.time()
            self._last_call = started
            self.usage.add()
            try:
                response = self._sdk().chat.completions.create(
                    model=model, temperature=temperature, max_tokens=int(self.cfg["max_tokens"]),
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                text = (response.choices[0].message.content or "").strip()
                if not text:
                    raise LLMError("empty response")
            except Exception as exc:  # noqa: BLE001 - try the next model
                message = f"{model}: {type(exc).__name__}: {str(exc)[:200]}"
                errors.append(message)
                self.on_call({"role": role, "model": model, "ok": False, "error": message,
                              "seconds": round(time.time() - started, 2)})
                continue
            usage = getattr(response, "usage", None)
            usage = {"prompt_tokens": getattr(usage, "prompt_tokens", None),
                     "completion_tokens": getattr(usage, "completion_tokens", None)}
            record = {"key": key, "model": model, "role": role, "text": text}
            self._cache[key] = record
            with open(self.cache_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            seconds = round(time.time() - started, 2)
            self.on_call({"role": role, "model": model, "ok": True, "cached": False, "seconds": seconds, **usage})
            return LLMResult(text, model, seconds=seconds, usage=usage)
        raise LLMError("all models failed for role " + role + ": " + " | ".join(errors[-4:]))


class ScriptedClient:
    """Deterministic stand-in: returns queued answers per role.  For tests and the offline demo."""

    def __init__(self, responses: dict[str, list[str]], on_call=None):
        self.responses = {role: list(items) for role, items in responses.items()}
        self.counters = {role: 0 for role in responses}
        self.on_call = on_call or (lambda record: None)

    def chat(self, role: str, system: str, user: str) -> LLMResult:
        items = self.responses.get(role) or []
        if not items:
            raise LLMError(f"scripted client has no responses for role {role}")
        i = self.counters[role]
        self.counters[role] = i + 1
        text = items[i % len(items)]
        if callable(text):
            text = text(system, user)
        self.on_call({"role": role, "model": "scripted", "ok": True, "cached": False, "seconds": 0})
        return LLMResult(text, "scripted")


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (fences, <think> blocks, prose)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates = [fenced.group(1)] if fenced else []
    start = text.find("{")
    while start != -1:
        depth, in_str, escape = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
        start = text.find("{", start + 1) if not candidates else -1
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return json.loads(candidate, strict=False)
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found in the reply")


def build_client(config: dict, on_call=None, provider: str | None = None):
    provider = provider or config["llm"]["provider"]
    if provider == "openrouter":
        return OpenRouterClient(config, on_call=on_call)
    if provider == "scripted":
        from ..agents.scripted import demo_responses

        return ScriptedClient(demo_responses(), on_call=on_call)
    raise ValueError(f"unknown provider {provider}")

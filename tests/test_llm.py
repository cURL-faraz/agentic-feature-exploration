"""LLM-layer tests: provider resolution, secret handling, caching and fallback.

No test here performs a network call.  The networked providers are exercised only
through their *pre-flight* behaviour (missing key, unknown model), which is where
the safety properties live: never silently substituting a different model, never
logging a secret.
"""

from __future__ import annotations

import pytest

from eeg_research_agent.llm import (
    AnthropicProvider,
    DeepSeekProvider,
    LLMClient,
    LLMMessage,
    LLMReply,
    LLMRequest,
    LLMUnavailableError,
    ModelNotAvailableError,
    OfflineProvider,
    build_llm_client,
    provider_status,
    resolve_provider,
)


def test_provider_resolution_infers_from_the_model_name():
    assert resolve_provider(None, "claude-opus-4-6")[0] == "anthropic"
    assert resolve_provider(None, "deepseek-chat")[0] == "deepseek"
    assert resolve_provider("offline", None)[0] == "offline"
    with pytest.raises(ValueError):
        resolve_provider("openai", "gpt-4")


def test_provider_status_reports_key_presence_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    status = provider_status("anthropic")
    assert status["requires_api_key"] and not status["api_key_present"]
    assert status["api_key_env_var"] == "ANTHROPIC_API_KEY"
    offline = provider_status("offline")
    assert offline["requires_api_key"] is False and offline["networked"] is False


def test_networked_provider_refuses_to_construct_without_a_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(LLMUnavailableError):
        DeepSeekProvider(api_key="")


def test_unknown_model_is_a_hard_error_not_a_silent_substitution(monkeypatch):
    provider = AnthropicProvider(api_key="sk-test-not-a-real-key")
    monkeypatch.setattr(provider, "list_models", lambda: ["claude-3-5-sonnet-20241022"])
    with pytest.raises(ModelNotAvailableError) as error:
        provider.complete(
            LLMRequest(messages=[LLMMessage("user", "hi")], model="claude-does-not-exist")
        )
    assert "claude-3-5-sonnet-20241022" in str(error.value)


def test_secrets_are_redacted_from_repr():
    secret = "sk-ant-super-secret-value"
    provider = AnthropicProvider(api_key=secret)
    assert secret not in repr(provider)
    assert "redacted" in repr(provider)


def test_offline_provider_renders_every_task_deterministically():
    provider = OfflineProvider()
    payloads = {
        "dataset_summary": {
            "dataset_summary": {
                "dataset": "d",
                "data_origin": "synthetic",
                "subject_count": 2,
                "epoch_count": 4,
                "class_distribution": {"N2": 4},
            }
        },
        "search_topics": {"queries": [{"query": "q", "topic": "t"}]},
        "feature_hypotheses": {"candidates": [{"feature_id": "abs_power_delta"}], "selected": ["abs_power_delta"]},
        "interpretation": {"question": "q", "findings": [{"feature": "f", "p_adjusted": 0.01}]},
        "report_section": {"title": "T", "body_markdown": "body"},
    }
    for task, payload in payloads.items():
        request = LLMRequest(messages=[LLMMessage("user", "x")], metadata={"task": task, "payload": payload})
        first = provider.complete(request)
        second = provider.complete(request)
        assert first.ok and first.text == second.text, task
        assert first.metadata["generated_by"] == "offline-template"
        assert first.metadata["reasoning"] is False


def test_offline_provider_never_invents_numbers():
    provider = OfflineProvider()
    reply = provider.complete(
        LLMRequest(
            messages=[LLMMessage("user", "x")],
            metadata={"task": "dataset_summary", "payload": {"dataset_summary": {"dataset": "d"}}},
        )
    )
    assert "unknown" in reply.text, "absent fields must be reported as unknown, not filled in"


class _FlakyProvider:
    name = "flaky"
    model = "flaky-1"

    def __init__(self, succeed: bool = False) -> None:
        self.succeed = succeed
        self.calls = 0

    def list_models(self):
        return ["flaky-1"]

    def complete(self, request):
        self.calls += 1
        if self.succeed:
            return LLMReply(text="ok", model=self.model, provider=self.name)
        return LLMReply(text="", model=self.model, provider=self.name, error="HTTP 500")


def test_client_falls_back_to_the_offline_provider_and_labels_the_reply():
    client = LLMClient(provider=_FlakyProvider(), fallback=OfflineProvider(), allow_fallback=True)
    reply = client.complete(
        LLMRequest(messages=[LLMMessage("user", "x")], metadata={"task": "interpretation", "payload": {}})
    )
    assert reply.provider == "offline"
    assert reply.metadata["fallback_from"] == "flaky"
    assert reply.metadata["fallback_reason"] == "HTTP 500"


def test_client_raises_when_fallback_is_disabled():
    client = LLMClient(provider=_FlakyProvider(), fallback=None, allow_fallback=False)
    with pytest.raises(LLMUnavailableError):
        client.complete(LLMRequest(messages=[LLMMessage("user", "x")]))


def test_client_caches_successful_responses():
    provider = _FlakyProvider(succeed=True)
    client = LLMClient(provider=provider)
    request = LLMRequest(messages=[LLMMessage("user", "same")])
    assert not client.complete(request).from_cache
    second = client.complete(request)
    assert second.from_cache
    assert provider.calls == 1


def test_build_client_defaults_to_offline_without_credentials(base_config, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    client = build_llm_client(base_config)
    assert client.name == "offline"

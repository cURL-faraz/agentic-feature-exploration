import pytest

from featurelab.agents.roles import _parse_proposal
from featurelab.agents.scripted import THETA_CODE, THETA_META, _proposal
from featurelab.llm.client import extract_json


def test_extract_json_from_prose_and_think_blocks():
    text = '<think>hmm {"no": 1}</think> Sure! Here it is:\n```json\n{"a": "b \\" }", "c": [1, 2]}\n```'
    assert extract_json(text) == {"a": 'b " }', "c": [1, 2]}


def test_extract_json_raises_without_json():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_proposal_parses_json_plus_code_block():
    proposal = _parse_proposal(_proposal(THETA_META, THETA_CODE))
    assert proposal.name == "theta_burstiness"
    assert proposal.prediction.higher_in == "REM"
    assert "def feature" in proposal.code


def test_stage_names_are_normalised():
    meta = dict(THETA_META, prediction={"higher_in": "R", "lower_in": "stage 1"}.copy())
    meta["prediction"]["lower_in"] = "S1"
    assert _parse_proposal(_proposal(meta, THETA_CODE)).prediction.lower_in == "N1"


def test_model_override_puts_pinned_models_first(monkeypatch):
    from featurelab.config import load_config

    monkeypatch.setenv("FEATURELAB_LLM_MODEL", "vendor/model-a:free, vendor/model-b:free")
    config = load_config()
    for models in config["llm"]["roles"].values():
        assert models[:2] == ["vendor/model-a:free", "vendor/model-b:free"]
        assert len(models) > 2  # configured models stay as fallbacks


class _FakeCompletions:
    def __init__(self, behaviour):
        self.behaviour, self.calls = behaviour, []

    def create(self, **kwargs):
        from types import SimpleNamespace as NS

        self.calls.append(kwargs)
        outcome = self.behaviour(kwargs["model"])
        if isinstance(outcome, Exception):
            raise outcome
        usage = NS(prompt_tokens=10, completion_tokens=8000,
                   completion_tokens_details=NS(reasoning_tokens=8000))
        return NS(choices=[NS(message=NS(content=outcome), finish_reason="length" if not outcome else "stop")],
                  usage=usage)


def _client(tmp_path, behaviour, **llm):
    from types import SimpleNamespace as NS

    from featurelab.config import load_config
    from featurelab.llm.client import OpenRouterClient

    config = load_config()
    config["data"]["root"] = str(tmp_path)
    config["llm"].update(auto_free_fallback=False, min_seconds_between_calls=0, **llm)
    config["llm"]["roles"]["inventor"] = ["a:free", "b:free", "c:free"]
    records = []
    client = OpenRouterClient(config, on_call=records.append)
    fake = _FakeCompletions(behaviour)
    client._client = NS(chat=NS(completions=fake))
    return client, fake, records


def test_daily_quota_stops_at_once_instead_of_walking_every_model(tmp_path):
    import pytest

    from featurelab.llm.client import DailyCapReached

    quota = RuntimeError("Error code: 429 - Rate limit exceeded: free-models-per-day")
    client, fake, _ = _client(tmp_path, lambda model: quota)
    with pytest.raises(DailyCapReached):
        client.chat("inventor", "s", "u")
    assert len(fake.calls) == 1


def test_empty_reply_is_explained_and_reasoning_is_capped(tmp_path):
    client, fake, records = _client(tmp_path, lambda m: "" if m == "a:free" else '{"ok": 1}',
                                    reasoning_max_tokens=3000)
    result = client.chat("inventor", "s", "u")
    assert result.model == "b:free"
    assert "finish_reason=length" in records[0]["error"] and "reasoning_tokens=8000" in records[0]["error"]
    assert fake.calls[0]["extra_body"]["reasoning"]["max_tokens"] == 3000

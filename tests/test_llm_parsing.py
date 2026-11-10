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

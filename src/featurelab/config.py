"""Configuration: packaged defaults, deep-merged with an optional project file.

Resolution order (later wins):
1. ``featurelab/default_config.yaml`` (shipped with the package - the protocol)
2. ``./featurelab.yaml`` or the file passed with ``--config``
3. environment: ``FEATURELAB_DATA`` (data root), ``OPENROUTER_API_KEY`` (never stored in config)
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).with_name("default_config.yaml")


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    from dotenv import load_dotenv

    load_dotenv(override=False)  # picks up .env in the working directory, if any
    with open(DEFAULT_PATH, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    candidate = Path(path) if path else Path("featurelab.yaml")
    if candidate.is_file():
        with open(candidate, encoding="utf-8") as handle:
            config = _merge(config, yaml.safe_load(handle) or {})
    elif path:
        raise FileNotFoundError(candidate)
    if os.environ.get("FEATURELAB_DATA"):
        config["data"]["root"] = os.environ["FEATURELAB_DATA"]
    config["data"]["root"] = str(Path(config["data"]["root"]).expanduser().resolve())
    _apply_model_override(config, os.environ.get("FEATURELAB_LLM_MODEL", ""))
    return config


def _apply_model_override(config: dict[str, Any], value: str) -> None:
    """``FEATURELAB_LLM_MODEL=model-a[,model-b]`` puts these models first for every role.

    The configured per-role models stay behind them as fallbacks, so one flaky
    free model does not stop the run.
    """
    pinned = [m.strip() for m in value.split(",") if m.strip()]
    if not pinned:
        return
    roles = config["llm"]["roles"]
    for role, models in roles.items():
        roles[role] = pinned + [m for m in (models or []) if m not in pinned]


def protocol_hash(config: dict[str, Any]) -> str:
    """Fingerprint of everything that decides acceptance (data + evaluation).

    Stored with every ledger row, so a result can always be tied to the exact
    protocol that produced it - and results from different protocols are never mixed.
    """
    relevant = {k: config[k] for k in ("protocol_version", "data", "evaluation")}
    relevant = json.loads(json.dumps(relevant, sort_keys=True))
    relevant["data"].pop("root", None)  # where the data lives does not change the science
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:12]


def data_root(config: dict[str, Any]) -> Path:
    return Path(config["data"]["root"])


def runs_dir(config: dict[str, Any]) -> Path:
    path = data_root(config) / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path

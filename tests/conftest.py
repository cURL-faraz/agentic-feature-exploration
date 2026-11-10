"""Shared fixtures: a small synthetic world, fast model settings, no network, no API key."""

from pathlib import Path

import pytest

from featurelab.config import load_config
from featurelab.data import synthetic


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("featurelab")
    synthetic.generate(root, "synthetic", n_subjects=14, epochs=120, seed=7)
    synthetic.generate(root, "synthetic_lock", n_subjects=14, epochs=120, seed=99)
    config = load_config()
    config["data"]["root"] = str(root)
    config["data"]["cohorts"] = {"discovery": "synthetic", "lockbox": "synthetic_lock", "external": "none"}
    config["evaluation"]["model"]["max_iter"] = 60
    config["evaluation"]["incremental"]["hypothesis_budget"] = 5     # 14 subjects: keep the test powered
    config["evaluation"]["incremental"]["null_shuffles"] = 1
    config["evaluation"]["confirm"]["max_looks"] = 2                 # small lockbox: fewer looks, more power
    config["sandbox"]["workers"] = 2
    config["literature"]["queries"] = []                             # no network in tests
    return config

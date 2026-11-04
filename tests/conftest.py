"""Shared fixtures.

The suite is designed to run **offline and without an LLM key**, using a small
synthetic corpus generated deterministically.  Every fixture writes into a
temporary project root so that a test run never touches the committed artifacts
or the real ``data/`` directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def base_config(tmp_path_factory: pytest.TempPathFactory):
    """A configuration whose paths live in a temporary directory.

    Feature toggles are left at their defaults so the tests exercise the real
    registry; only the corpus size and the output location are reduced.
    """
    import os

    from eeg_research_agent.config import load_config

    root = tmp_path_factory.mktemp("eeg-agent-test")
    config = load_config()
    paths = config.settings.paths.model_copy(
        update={
            "data_root": str(root / "data"),
            "raw": str(root / "data/raw"),
            "interim": str(root / "data/interim"),
            "processed": str(root / "data/processed"),
            "features": str(root / "data/features"),
            "metadata": str(root / "data/metadata"),
            "artifacts": str(root / "artifacts"),
            "figures": str(root / "artifacts/figures"),
            "tables": str(root / "artifacts/tables"),
            "logs": str(root / "artifacts/execution_logs"),
            "reports": str(root / "reports"),
        }
    )
    execution = config.settings.execution.model_copy(update={"mode": "smoke", "max_subjects": 4})
    dataset = config.dataset.model_copy(
        update={
            "data_origin": "synthetic",
            "fallback": config.dataset.fallback.model_copy(
                # The smoke corpus is capped at 3 subjects / 40 epochs upstream;
                # 40 epochs is enough for the *real* statistical minimum group
                # size, so the comparisons exercise the production code path.
                update={"subjects": 4, "epochs_per_subject": 40, "seed": 4242}
            ),
            "signal": config.dataset.signal.model_copy(
                update={"channel_preference": ["EEG Fpz-Cz", "EEG"]}
            ),
            "subset": config.dataset.subset.model_copy(update={"max_recording_seconds": 720}),
        }
    )
    analysis = config.analysis.model_copy(
        update={
            "statistics": config.analysis.statistics.model_copy(
                update={"permutation_iterations": 60, "bootstrap_iterations": 60}
            ),
            "artifacts": config.analysis.artifacts.model_copy(
                update={"save_figures": False, "top_feature_plots": 2}
            ),
        }
    )
    tuned = config.model_copy(
        update={
            "root": root,
            "settings": config.settings.model_copy(
                update={"paths": paths, "execution": execution, "offline_mode": "always"}
            ),
            "dataset": dataset,
            "analysis": analysis,
        }
    )
    os.environ.setdefault("MPLCONFIGDIR", str(root / "mplcache"))
    tuned.ensure_directories()
    return tuned


@pytest.fixture(scope="session")
def small_corpus(base_config):
    """Recordings for the small synthetic corpus (no network, deterministic)."""
    from eeg_research_agent.data.acquisition import acquire_dataset

    result = acquire_dataset(base_config, origin="synthetic")
    return list(result.recordings)


@pytest.fixture(scope="session")
def preprocessed(small_corpus, base_config):
    from eeg_research_agent.data.preprocessing import preprocess_recording

    cleaned = []
    reports = []
    for recording in small_corpus:
        recording_out, report = preprocess_recording(recording, base_config)
        cleaned.append(recording_out)
        reports.append(report)
    return cleaned, reports


@pytest.fixture(scope="session")
def epoch_sets(preprocessed, base_config):
    from eeg_research_agent.eeg.segmentation import segment_recordings

    cleaned, _ = preprocessed
    sets, reports = segment_recordings(cleaned, base_config)
    return list(sets), list(reports)


@pytest.fixture(scope="session")
def feature_table(epoch_sets, base_config):
    """The full feature table for the small corpus (extracted once per session)."""
    from eeg_research_agent.eeg.feature_extraction import add_sequence_features, extract_features
    from eeg_research_agent.data.splits import assign_splits, build_epoch_index, make_subject_splits

    sets, _ = epoch_sets
    subjects = sorted({record.subject_id for record in sets})
    splits = make_subject_splits(subjects, seed=int(base_config.seed))
    rows = assign_splits(build_epoch_index(sets), splits)
    lookup = {f"{row['subject_id']}|{row['recording_id']}|{row['epoch_id']}": row["split"] for row in rows}
    table_out = extract_features(sets, base_config)
    frame = table_out.table.copy()
    frame["split"] = [
        lookup.get(f"{subject}|{recording}|{epoch}", "unknown")
        for subject, recording, epoch in zip(frame.subject_id, frame.recording_id, frame.epoch_id)
    ]
    frame = add_sequence_features(frame)
    return frame, table_out, splits


@pytest.fixture(scope="session")
def registry():
    from eeg_research_agent.eeg.feature_registry import load_default_registry

    return load_default_registry()


@pytest.fixture()
def offline_llm(base_config):
    """An explicitly offline LLM client (no network, no key)."""
    from eeg_research_agent.llm import build_llm_client

    return build_llm_client(base_config, provider="offline")

"""Sources named explicitly in the project brief.

These are **not** retrieval results.  They are the anchor documents the brief
itself points at (the AttnSleep repository, the BIOT paper, the Sleep-EDF
dataset landing page and the MNE sleep tutorial).  They are recorded with
``evidence_level: "unverified-offline"`` and ``source: "project_brief"`` so that

* the literature stage has something concrete to start from,
* no downstream report can mistake them for independently verified citations,
* :func:`..verification.research_validation.validate_citations` lists them as
  unverified rather than certifying them.

Their metadata is intentionally minimal - exactly the title and URL the brief
supplies.  Nothing has been added from memory, because an invented DOI or author
list is precisely the failure mode this project must avoid.
"""

from __future__ import annotations

from typing import Any

__all__ = ["SEED_SOURCES", "SEARCH_QUERIES"]

#: Anchor documents supplied by the project brief.  Metadata is limited to what
#: the brief states verbatim.
SEED_SOURCES: list[dict[str, Any]] = [
    {
        "title": "AttnSleep: An Efficient Sleep Staging Model Based on Attention Mechanisms",
        "authors": [],
        "year": None,
        "source": "project_brief",
        "venue": None,
        "identifier": "https://github.com/emadeldeen24/AttnSleep",
        "identifier_type": "url",
        "abstract": (
            "Reference implementation named in the project brief for attention-based sleep "
            "staging from single-channel EEG (multi-resolution convolution, feature "
            "recalibration, temporal attention). Full text was not retrieved."
        ),
        "access": "reference-implementation-repository",
        "full_text_available": False,
        "evidence_level": "unverified-offline",
        "datasets": ["Sleep-EDF", "SHHS"],
        "notes": "Named in the project brief; metadata not independently verified in this run.",
    },
    {
        "title": "BIOT: Biosignal Transformer for Cross-data Learning in the Wild",
        "authors": [],
        "year": 2023,
        "source": "project_brief",
        "venue": "NeurIPS 2023 (as stated in the project brief)",
        "identifier": "https://arxiv.org/abs/2305.10351",
        "identifier_type": "arxiv",
        "abstract": (
            "Biosignal transformer with channel-independent tokenisation, named in the project "
            "brief as the cross-dataset transfer baseline. The abstract was not retrieved in "
            "this run; only the arXiv identifier supplied by the brief is recorded."
        ),
        "access": "preprint-listing",
        "full_text_available": False,
        "evidence_level": "unverified-offline",
        "datasets": ["multiple (not verified in this run)"],
        "notes": "arXiv identifier taken verbatim from the project brief; not verified online.",
    },
    {
        "title": "Sleep-EDF Expanded (version 1.0.0)",
        "authors": [],
        "year": 2013,
        "source": "project_brief",
        "venue": "PhysioNet",
        "identifier": "https://physionet.org/content/sleep-edfx/1.0.0/",
        "identifier_type": "url",
        "abstract": (
            "The dataset this project is designed around (Sleep-EDF Expanded, PhysioNet). "
            "Named in the project brief as the recommended starting dataset."
        ),
        "access": "dataset-landing-page",
        "full_text_available": False,
        "evidence_level": "unverified-offline",
        "datasets": ["Sleep-EDF Expanded"],
        "notes": "Landing page taken from the project brief. The dataset files were not downloaded in this run.",
    },
    {
        "title": "MNE-Python sleep staging tutorial (clinical tutorial 60)",
        "authors": [],
        "year": None,
        "source": "project_brief",
        "venue": "MNE-Python documentation",
        "identifier": "https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html",
        "identifier_type": "url",
        "abstract": (
            "Implementation reference named in the project brief for loading Sleep-EDF "
            "recordings, handling annotations and constructing epochs."
        ),
        "access": "documentation",
        "full_text_available": False,
        "evidence_level": "unverified-offline",
        "datasets": ["Sleep-EDF"],
        "notes": "Documentation page taken from the project brief.",
    },
]


#: Search plan for the live literature stage.  Each entry is a topic from the
#: brief mapped to a ready-to-run query; covering the brief's list explicitly
#: prevents the agent from silently narrowing the search.
SEARCH_QUERIES: list[dict[str, str]] = [
    {
        "topic": "single_channel_sleep_staging",
        "query": "single-channel EEG automatic sleep staging",
        "why": "core research question: which EEG features separate sleep stages",
    },
    {
        "topic": "n1_classification",
        "query": "N1 sleep stage classification difficulty automatic scoring",
        "why": "N1 is the hardest stage and the brief asks specifically about N1/N2/REM",
    },
    {
        "topic": "spectral_features",
        "query": "spectral power features delta theta alpha sigma sleep stage EEG",
        "why": "conventional frequency-domain features in category A",
    },
    {
        "topic": "nonlinear_features",
        "query": "entropy fractal dimension EEG sleep stage classification",
        "why": "nonlinear features in category A",
    },
    {
        "topic": "spindle_detection",
        "query": "automated sleep spindle detection algorithm validation",
        "why": "event-based features must be independently documented and validated",
    },
    {
        "topic": "k_complex_detection",
        "query": "K-complex detection EEG automated algorithm",
        "why": "event-based features in category A",
    },
    {
        "topic": "slow_wave_activity",
        "query": "slow wave activity EEG N3 homeostatic",
        "why": "N3 characterisation and slow-wave density/amplitude features",
    },
    {
        "topic": "feature_selection",
        "query": "EEG feature selection sleep staging machine learning interpretation",
        "why": "feature selection and interaction analysis",
    },
    {
        "topic": "transfer_and_domain_adaptation",
        "query": "EEG sleep staging domain adaptation transfer learning cross-dataset",
        "why": "generalisation across recordings and datasets",
    },
    {
        "topic": "deep_learning_and_interpretability",
        "query": "deep learning interpretable sleep staging EEG attention",
        "why": "deep-learning comparison and interpretability",
    },
    {
        "topic": "mixed_effects_and_clustered_inference",
        "query": "mixed effects models clustered EEG epochs statistical inference",
        "why": "the statistical methodology must respect the repeated-measures structure",
    },
    {
        "topic": "dataset_benchmark",
        "query": "Sleep-EDF expanded benchmark single channel sleep staging baseline",
        "why": "dataset provenance and comparable baselines",
    },
]

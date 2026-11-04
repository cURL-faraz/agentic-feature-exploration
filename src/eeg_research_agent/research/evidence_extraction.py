"""Turn paper records into structured evidence.

The extraction is **deliberately conservative**.  Numbers such as the number of
subjects or the sampling frequency are only filled in when they can be matched
against the abstract by an explicit regular expression; otherwise the field stays
``None``.  That is the opposite of what a generative model does by default, and
it is the whole point: a missing value is recoverable, a hallucinated one is not.

An LLM may be used to *summarise* an abstract into the ``relevance`` field, but
it is never allowed to write a numeric field, and every generated sentence is
tagged with the evidence level it is based on.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from ..logging_utils import get_logger
from .paper_search import PaperRecord

__all__ = [
    "extract_evidence",
    "annotate_relevance",
    "EVIDENCE_FIELDS",
    "TOPIC_KEYWORDS",
]

logger = get_logger(__name__)

EVIDENCE_FIELDS = (
    "datasets",
    "n_subjects",
    "channels_used",
    "sampling_frequency_hz",
    "epoch_seconds",
    "feature_methods",
    "statistical_methods",
    "main_findings",
    "limitations",
)

TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "spectral": (
        "spectral",
        "power",
        "delta",
        "theta",
        "alpha",
        "sigma",
        "spindle",
        "frequency",
        "welch",
        "fft",
        "spectrogram",
    ),
    "nonlinear": (
        "entropy",
        "fractal",
        "higuchi",
        "petrosian",
        "detrended fluctuation",
        "sample entropy",
        "permutation entropy",
        "complexity",
        "lyapunov",
    ),
    "events": ("spindle", "k-complex", "k complex", "slow wave", "transient", "event detection"),
    "time_domain": ("hjorth", "zero crossing", "line length", "variance", "amplitude", "rms"),
    "wavelet": ("wavelet", "time-frequency", "multiresolution", "decomposition"),
    "time_frequency": ("time-frequency", "wavelet", "spectrogram", "hilbert", "envelope"),
    "feature_selection": ("feature selection", "recursive", "mutual information", "lasso", "shap", "importance"),
    "deep_learning": (
        "deep learning",
        "neural network",
        "convolutional",
        "transformer",
        "attention",
        "cnn",
        "lstm",
        "recurrent",
    ),
    "transfer_learning": ("transfer learning", "domain adaptation", "cross-dataset", "fine-tun", "pretrain"),
    "statistics": ("mixed effect", "anova", "kruskal", "permutation", "cross-validation", "significance"),
    "interpretability": ("interpretab", "explainab", "attention weight", "saliency", "shap"),
}

_CHANNEL_PATTERNS = (
    r"\b(Fpz-?Cz|Pz-?Oz|Fpz|Cz|C3|C4|F3|F4|O1|O2|Fz|Pz|A1|A2)\b",
)
_SFREQ_RE = re.compile(r"(\d{2,4})\s*(?:Hz|hz)\b")
_SUBJECTS_RE = re.compile(
    r"\b(\d{1,5})\s*(?:subjects|participants|patients|individuals|recordings|people)\b",
    re.IGNORECASE,
)
_EPOCH_RE = re.compile(r"\b(\d{1,3})\s*(?:-|\s)?(?:s|second|seconds|sec)\b\s*epoch", re.IGNORECASE)
_N1_RE = re.compile(r"\bN1\b")
_STAT_METHODS = (
    "anova",
    "kruskal",
    "mann-whitney",
    "wilcoxon",
    "t-test",
    "chi-square",
    "mixed effect",
    "mixed-effects",
    "permutation test",
    "cross-validation",
    "bootstrap",
    "regression",
    "fisher",
    "effect size",
)
_LIMITATION_MARKERS = (
    "limitation",
    "however",
    "nevertheless",
    "small sample",
    "single-centre",
    "single center",
    "generaliz",
    "generalis",
    "class imbalance",
    "label noise",
    "inter-scorer",
    "interscorer",
    "cannot capture",
    "does not capture",
    "future work",
)


def extract_evidence(record: PaperRecord) -> PaperRecord:
    """Fill the structured evidence fields from the retrievable text only."""
    text = f"{record.title}. {record.abstract}".strip()
    lowered = text.lower()
    if not record.abstract.strip():
        record.notes = (
            (record.notes + " " if record.notes else "")
            + "No abstract was retrievable; structured evidence fields are intentionally left empty."
        ).strip()
        record.evidence_level = record.evidence_level or "metadata-only"
        return record

    if not record.text_retrieved:
        # The text exists but came from the project brief rather than from a
        # provider, so the record may not be promoted to "abstract-only" - doing
        # so would let an unverified source masquerade as a retrieved one.
        record.evidence_level = "unverified-offline"

    # -- channels ---------------------------------------------------------- #
    channels: set[str] = set()
    for pattern in _CHANNEL_PATTERNS:
        channels.update(match.upper().replace("-", "-") for match in re.findall(pattern, text))
    record.channels_used = sorted(channels)

    # -- sampling frequency ------------------------------------------------ #
    frequencies = [int(value) for value in _SFREQ_RE.findall(text) if 20 <= int(value) <= 2000]
    if len(frequencies) == 1:
        record.sampling_frequency_hz = float(frequencies[0])
    elif frequencies:
        record.extra["sampling_frequencies_mentioned"] = sorted(set(frequencies))

    # -- subject count ----------------------------------------------------- #
    subjects = [int(value) for value in _SUBJECTS_RE.findall(text) if 1 <= int(value) <= 100000]
    if len(subjects) == 1:
        record.n_subjects = subjects[0]
    elif subjects:
        record.extra["subject_counts_mentioned"] = sorted(set(subjects))

    # -- epoch length ------------------------------------------------------ #
    epochs = [int(value) for value in _EPOCH_RE.findall(text)]
    if len(epochs) == 1:
        record.epoch_seconds = float(epochs[0])

    # -- datasets ---------------------------------------------------------- #
    datasets: set[str] = set()
    for name in ("Sleep-EDF", "Sleep EDF", "sleep-edfx", "SHHS", "MASS", "PhysioNet", "DREAMS", "ISRUC", "CAP", "MROS"):
        if name.lower() in lowered:
            datasets.add(name)
    record.datasets = sorted(datasets)

    # -- feature methods ---------------------------------------------------- #
    record.feature_methods = sorted(
        topic for topic, keywords in TOPIC_KEYWORDS.items() if any(word in lowered for word in keywords)
    )
    record.statistical_methods = sorted({name for name in _STAT_METHODS if name in lowered})
    record.limitations = sorted(
        sentence.strip()
        for sentence in _sentences(text)
        if any(marker in sentence.lower() for marker in _LIMITATION_MARKERS)
    )[:4]
    record.main_findings = _findings(text)
    if record.text_retrieved:
        record.evidence_level = "abstract-only"
    return record


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part for part in parts if len(part.split()) > 4]


_FINDING_MARKERS = (
    "we found",
    "we show",
    "results show",
    "results indicate",
    "our results",
    "achiev",
    "outperform",
    "significantly",
    "demonstrates",
    "conclude",
    "suggest",
    "improve",
)


def _findings(text: str) -> list[str]:
    """Sentences that state a result - extracted verbatim, never paraphrased."""
    findings = [
        sentence.strip()
        for sentence in _sentences(text)
        if any(marker in sentence.lower() for marker in _FINDING_MARKERS)
    ]
    return findings[:4]


def annotate_relevance(records: Sequence[PaperRecord], research_question: str) -> list[PaperRecord]:
    """Attach a deterministic relevance note to every record.

    No LLM is required: the note states which of the brief's topics the paper
    touches and whether it addresses the N1/N2/REM question, which is exactly the
    information a reviewer needs to triage the list.
    """
    question = research_question.lower()
    for record in records:
        text = f"{record.title} {record.abstract}".lower()
        topics = sorted(
            topic for topic, keywords in TOPIC_KEYWORDS.items() if any(word in text for word in keywords)
        )
        addresses_transitional = bool(_N1_RE.search(record.title + " " + record.abstract)) or any(
            marker in text for marker in ("n1", "transitional", "light sleep", "rem")
        )
        parts = []
        if topics:
            parts.append(f"touches {', '.join(topics)}")
        if addresses_transitional:
            parts.append("relevant to the N1/N2/REM discrimination question")
        if record.identifier_type in {"doi", "pmid", "arxiv"}:
            parts.append(f"traceable via {record.identifier_type.upper()} {record.identifier}")
        else:
            parts.append("no persistent identifier available yet")
        record.relevance = "; ".join(parts) if parts else "no explicit topical overlap detected"
        if "sleep staging" in text or "sleep stage" in text:
            record.relevance += "; directly on sleep staging"
        if "single channel" in text or "single-channel" in text:
            record.relevance += "; single-channel EEG"
        _ = question  # the question is recorded in the report; matching is keyword-based
    return list(records)

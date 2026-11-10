"""In-memory representation shared by every data origin.

Whether the signal comes from a real EDF file (Sleep-EDF) or from the synthetic
generator, downstream code only ever sees a :class:`Recording`, which keeps the
provenance of the data explicit instead of silently mixing sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["StageAnnotation", "Recording", "EpochSet", "CANONICAL_STAGES", "UNKNOWN_STAGE"]

CANONICAL_STAGES: tuple[str, ...] = ("W", "N1", "N2", "N3", "REM")
UNKNOWN_STAGE = "UNKNOWN"


@dataclass(slots=True)
class StageAnnotation:
    """One sleep-stage annotation (label valid for ``duration`` seconds from ``onset``)."""

    onset: float
    duration: float
    stage: str
    description: str | None = None

    @property
    def offset(self) -> float:
        return self.onset + self.duration


@dataclass
class Recording:
    """A single-channel EEG recording plus its sleep-stage annotations.

    Notes
    -----
    ``signal`` is stored in **volts** (the MNE convention) as a 1-D array of
    length ``n_samples``.  The unit never changes between modules; only the
    reporting layer converts to microvolts.
    """

    signal: np.ndarray
    sfreq: float
    ch_name: str
    annotations: list[StageAnnotation]
    subject_id: str
    recording_id: str
    data_origin: str = "unknown"
    source_file: str | None = None
    channel_candidates: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.signal = np.asarray(self.signal, dtype=np.float64).ravel()
        if self.signal.size == 0:
            raise ValueError("recording signal is empty")
        if not np.all(np.isfinite(self.signal)):
            raise ValueError("recording signal contains non-finite values")
        if self.sfreq <= 0:
            raise ValueError("sampling frequency must be positive")

    @property
    def n_samples(self) -> int:
        return int(self.signal.size)

    @property
    def duration_s(self) -> float:
        return float(self.signal.size / self.sfreq)

    @property
    def stages_present(self) -> list[str]:
        seen = {ann.stage for ann in self.annotations}
        return [stage for stage in CANONICAL_STAGES if stage in seen]

    def stage_for_time(self, onset: float, duration: float) -> str:
        """Return the stage covering ``[onset, onset + duration)`` (majority overlap)."""
        best_stage, best_overlap = UNKNOWN_STAGE, 0.0
        for ann in self.annotations:
            overlap = min(ann.offset, onset + duration) - max(ann.onset, onset)
            if overlap > best_overlap + 1e-9:
                best_stage, best_overlap = ann.stage, overlap
        return best_stage


@dataclass
class EpochSet:
    """Epoched, staged, single-channel EEG ready for feature extraction."""

    data: np.ndarray                # (n_epochs, n_samples), volts
    sfreq: float
    ch_name: str
    stages: list[str]
    onsets: np.ndarray              # (n_epochs,) seconds from recording start
    subject_id: str
    recording_id: str
    epoch_seconds: float
    data_origin: str = "unknown"
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.data.ndim != 2:
            raise ValueError("epoch data must be 2-D (n_epochs, n_samples)")
        if len(self.stages) != self.data.shape[0] or len(self.onsets) != self.data.shape[0]:
            raise ValueError("stages/onsets must have one entry per epoch")
        self.onsets = np.asarray(self.onsets, dtype=np.float64)

    @property
    def n_epochs(self) -> int:
        return int(self.data.shape[0])

    def stage_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for stage in self.stages:
            counts[stage] = counts.get(stage, 0) + 1
        return counts

    def identities(self) -> list[str]:
        return [
            f"{self.subject_id}|{self.recording_id}|{idx}"
            for idx in range(self.n_epochs)
        ]

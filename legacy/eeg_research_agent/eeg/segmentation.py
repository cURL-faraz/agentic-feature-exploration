"""Fixed-window epoching with an explicit record of every exclusion.

Sleep staging is defined on non-overlapping scoring epochs (30 s for Sleep-EDF,
which follows the R&K/AASM convention).  An epoch is kept only when

* one annotation covers at least ``min_annotation_coverage`` of the window,
* that annotation maps to a supported sleep stage, and
* the per-epoch quality checks pass.

Anything else is recorded in ``EpochSet.exclusions`` together with the reason and
the epoch's amplitude metrics.  Nothing is dropped silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import ProjectConfig
from ..data.preprocessing import epoch_quality_metrics
from ..data.recording import EpochSet, Recording, UNKNOWN_STAGE

__all__ = ["SegmentationReport", "segment_recording", "segment_recordings"]


@dataclass
class SegmentationReport:
    """Summary of one segmentation pass."""

    subject_id: str
    recording_id: str
    epoch_seconds: float
    n_windows: int = 0
    n_kept: int = 0
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    stage_counts: dict[str, int] = field(default_factory=dict)

    @property
    def exclusion_fraction(self) -> float:
        return len(self.exclusions) / self.n_windows if self.n_windows else 0.0

    def to_dict(self) -> dict[str, Any]:
        reasons: dict[str, int] = {}
        for item in self.exclusions:
            reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
        return {
            "subject_id": self.subject_id,
            "recording_id": self.recording_id,
            "epoch_seconds": self.epoch_seconds,
            "n_windows": self.n_windows,
            "n_kept": self.n_kept,
            "exclusion_fraction": round(self.exclusion_fraction, 6),
            "exclusion_reasons": reasons,
            "stage_counts": self.stage_counts,
        }


def _annotation_coverage(
    recording: Recording, onset: float, duration: float
) -> tuple[str, float, list[dict[str, Any]]]:
    """Stage with the largest overlap, its coverage fraction, and all overlaps."""
    overlaps: list[dict[str, Any]] = []
    for ann in recording.annotations:
        overlap = min(ann.offset, onset + duration) - max(ann.onset, onset)
        if overlap > 1e-9:
            overlaps.append({"stage": ann.stage, "overlap_s": float(overlap), "description": ann.description})
    if not overlaps:
        return UNKNOWN_STAGE, 0.0, overlaps
    best = max(overlaps, key=lambda item: item["overlap_s"])
    return best["stage"], float(best["overlap_s"] / duration), overlaps


def segment_recording(
    recording: Recording, config: ProjectConfig
) -> tuple[EpochSet, SegmentationReport]:
    """Segment one recording into labelled, quality-controlled epochs."""
    epoch_seconds = float(config.dataset.signal.epoch_seconds)
    sfreq = float(recording.sfreq)
    samples_per_epoch = int(round(epoch_seconds * sfreq))
    supported = set(config.dataset.stages.supported)
    min_coverage = float(config.dataset.signal.min_annotation_coverage)
    total_samples = recording.n_samples
    n_windows = total_samples // samples_per_epoch

    report = SegmentationReport(
        subject_id=recording.subject_id,
        recording_id=recording.recording_id,
        epoch_seconds=epoch_seconds,
        n_windows=int(n_windows),
    )

    epochs: list[np.ndarray] = []
    stages: list[str] = []
    onsets: list[float] = []
    kept_metadata: list[dict[str, Any]] = []

    for index in range(n_windows):
        onset = index * epoch_seconds
        window = recording.signal[index * samples_per_epoch : (index + 1) * samples_per_epoch]
        stage, coverage, overlaps = _annotation_coverage(recording, onset, epoch_seconds)
        quality = epoch_quality_metrics(window, sfreq, config)
        record: dict[str, Any] = {
            "subject_id": recording.subject_id,
            "recording_id": recording.recording_id,
            "epoch_id": index,
            "onset_s": onset,
            "stage": stage,
            "stage_coverage": round(coverage, 4),
            "amplitude_uv": quality["amplitude_uv"],
            "peak_to_peak_uv": quality["peak_to_peak_uv"],
            "n_overlapping_annotations": len(overlaps),
        }

        reason: str | None = None
        if stage == UNKNOWN_STAGE or stage not in supported:
            reason = "unsupported_stage"
        elif coverage < min_coverage:
            reason = "insufficient_annotation_coverage"
        elif not quality["ok"]:
            reason = f"quality:{quality['reason']}"

        if reason is not None:
            record["reason"] = reason
            record["included"] = False
            report.exclusions.append(record)
            continue

        record["included"] = True
        kept_metadata.append(record)
        epochs.append(np.asarray(window, dtype=np.float64))
        stages.append(stage)
        onsets.append(onset)
        report.stage_counts[stage] = report.stage_counts.get(stage, 0) + 1

    report.n_kept = len(epochs)
    data = np.vstack(epochs) if epochs else np.zeros((0, samples_per_epoch), dtype=np.float64)

    epoch_set = EpochSet(
        data=data,
        sfreq=sfreq,
        ch_name=recording.ch_name,
        stages=stages,
        onsets=np.asarray(onsets, dtype=np.float64),
        subject_id=recording.subject_id,
        recording_id=recording.recording_id,
        epoch_seconds=epoch_seconds,
        data_origin=recording.data_origin,
        exclusions=list(report.exclusions),
        metadata={
            "segmentation": report.to_dict(),
            "epoch_records": kept_metadata,
            "recording_metadata": recording.metadata,
        },
    )
    return epoch_set, report


def segment_recordings(
    recordings: list[Recording], config: ProjectConfig
) -> tuple[list[EpochSet], list[SegmentationReport]]:
    """Segment every recording; empty recordings are skipped but reported."""
    epoch_sets: list[EpochSet] = []
    reports: list[SegmentationReport] = []
    for recording in recordings:
        epoch_set, report = segment_recording(recording, config)
        reports.append(report)
        if epoch_set.n_epochs == 0:
            continue
        epoch_sets.append(epoch_set)
    return epoch_sets, reports

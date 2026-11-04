"""Dataset intelligence: schema, quality and class distribution.

This is the deterministic core of Agent 1 (Dataset Intelligence).  The agent may
narrate the result, but every number below is computed here.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np

from ..config import ProjectConfig
from .annotations import summarise_annotations
from .recording import CANONICAL_STAGES, Recording

__all__ = ["pick_channel", "inspect_recording", "inspect_dataset"]


def _normalise(name: str) -> str:
    return "".join(str(name).lower().replace("eeg", "").replace(" ", "").replace("-", "").replace("_", ""))


def pick_channel(available: Iterable[str], preference: list[str]) -> str | None:
    """Pick the first preferred channel that exists (case/format-insensitive)."""
    available = list(available)
    lookup = {_normalise(name): name for name in available}
    for candidate in preference:
        if candidate in available:
            return candidate
        key = _normalise(candidate)
        if key in lookup:
            return lookup[key]
    for key, original in lookup.items():
        if "eeg" in original.lower():
            return original
    return available[0] if available else None


def inspect_recording(recording: Recording) -> dict[str, Any]:
    """Structural + quality summary for a single recording."""
    signal_uv = recording.signal * 1e6
    diff = np.abs(np.diff(signal_uv))
    summary = {
        "subject_id": recording.subject_id,
        "recording_id": recording.recording_id,
        "data_origin": recording.data_origin,
        "source_file": recording.source_file,
        "channel": recording.ch_name,
        "channel_candidates": list(recording.channel_candidates),
        "sampling_frequency": float(recording.sfreq),
        "n_samples": recording.n_samples,
        "duration_s": round(recording.duration_s, 3),
        "amplitude_uv": {
            "mean": float(np.mean(signal_uv)),
            "std": float(np.std(signal_uv)),
            "min": float(signal_uv.min()),
            "max": float(signal_uv.max()),
            "ptp": float(np.ptp(signal_uv)),
            "median_abs": float(np.median(np.abs(signal_uv))),
        },
        "artifacts": {
            "fraction_samples_above_5x_std": float(np.mean(np.abs(signal_uv) > 5 * np.std(signal_uv))),
            "fraction_samples_flat": float(np.mean(np.abs(signal_uv) < 1e-6)),
            "max_abs_step_uv": float(diff.max()) if diff.size else 0.0,
            "n_missing": 0 if np.all(np.isfinite(recording.signal)) else int(np.sum(~np.isfinite(recording.signal))),
        },
        "annotations": summarise_annotations(recording.annotations),
    }
    return summary


def inspect_dataset(
    recordings: list[Recording],
    config: ProjectConfig,
    epoch_metadata: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Dataset-level summary used by the dataset-intelligence agent and reports."""
    if not recordings:
        raise ValueError("no recordings to inspect")

    per_recording = [inspect_recording(rec) for rec in recordings]
    stage_counts: Counter = Counter()
    for rec in recordings:
        for ann in rec.annotations:
            stage_counts[ann.stage] += 1
    supported = [stage for stage in config.dataset.stages.supported]
    available = [stage for stage in supported if stage_counts.get(stage, 0) > 0]

    sfreqs = sorted({float(rec.sfreq) for rec in recordings})
    channels = sorted({rec.ch_name for rec in recordings})
    origins = sorted({rec.data_origin for rec in recordings})
    subjects = sorted({rec.subject_id for rec in recordings})

    quality: dict[str, Any] = {
        "recording_count": len(recordings),
        "channel_consistency": len(channels) == 1,
        "sampling_frequency_consistent": len(sfreqs) == 1,
        "mean_std_uv": float(np.mean([r["amplitude_uv"]["std"] for r in per_recording])),
        "max_abs_step_uv": float(np.max([r["artifacts"]["max_abs_step_uv"] for r in per_recording])),
        "fraction_flat_samples": float(
            np.mean([r["artifacts"]["fraction_samples_flat"] for r in per_recording])
        ),
    }
    if epoch_metadata:
        excluded = [meta for meta in epoch_metadata if not meta.get("included", True)]
        reasons = Counter(meta.get("exclusion_reason") for meta in excluded)
        quality["epochs_total"] = len(epoch_metadata)
        quality["epochs_excluded"] = len(excluded)
        quality["epoch_exclusion_fraction"] = round(len(excluded) / len(epoch_metadata), 6)
        quality["epoch_exclusion_reasons"] = {str(k): v for k, v in reasons.items()}

    summary = {
        "dataset": config.dataset.dataset_name,
        "data_origin": origins[0] if len(origins) == 1 else origins,
        "selected_channel": channels[0] if len(channels) == 1 else channels,
        "channel_candidates": sorted({c for rec in recordings for c in rec.channel_candidates}),
        "sampling_frequency": sfreqs[0] if len(sfreqs) == 1 else sfreqs,
        "epoch_duration": float(config.dataset.signal.epoch_seconds),
        "available_stages": available,
        "missing_stages": [stage for stage in supported if stage not in available],
        "subject_count": len(subjects),
        "subjects": subjects,
        "recording_count": len(recordings),
        "epoch_count": int(sum(len(rec.annotations) for rec in recordings)),
        "class_distribution": {stage: int(stage_counts.get(stage, 0)) for stage in CANONICAL_STAGES},
        "quality_report": quality,
        "per_recording": per_recording,
        "caveats": _caveats(origins, available, config),
    }
    return summary


def _caveats(origins: list[str], available: list[str], config: ProjectConfig) -> list[str]:
    caveats: list[str] = []
    if "synthetic" in origins:
        caveats.append(
            "Data were generated by the deterministic synthetic PSG generator because the real "
            "Sleep-EDF Expanded files were not available (no network access). Results validate the "
            "pipeline, not human sleep biology."
        )
    if "sleep-edf" in origins:
        caveats.append(
            "Sleep-EDF Expanded contains 30-second scoring epochs; stage boundaries follow the "
            "original R&K-derived annotations, which are known to differ between scorers."
        )
    missing = [stage for stage in config.dataset.stages.supported if stage not in available]
    if missing:
        caveats.append(f"Missing stages in the acquired subset: {', '.join(missing)}.")
    return caveats

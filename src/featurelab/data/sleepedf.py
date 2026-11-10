"""Sleep-EDF Expanded -> labelled, wake-trimmed 30 s epochs.

Choices (each one is standard in the single-channel literature, and each one
changes the numbers, so they are explicit):

* Channel ``EEG Fpz-Cz`` at its native 100 Hz, no extra filtering.
* R&K stages 3 and 4 are merged into N3 (AASM convention).
* "Movement time" and "Sleep stage ?" epochs are dropped, not guessed.
* Only ``wake_margin_min`` minutes of wake are kept before the first and after
  the last sleep epoch.  Sleep-Cassette recordings last ~20 h; without trimming,
  ~70 % of epochs are daytime wake and every metric is inflated by easy W epochs.
* A *subject* (not a night) is the grouping unit: the two nights of one person
  must never sit on both sides of a train/test split.
"""

from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .cohort import STAGE_CODE, write_cohort

ANNOTATION_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}

COHORT_RULES = {
    "sc20": ("SC", range(0, 20)),
    "sc_lockbox": ("SC", range(20, 40)),
    "sc78": ("SC", range(0, 100)),
    "st": ("ST", range(0, 100)),
}


def find_recordings(raw_root: Path) -> list[tuple[str, Path, Path]]:
    """Pair each ``*-PSG.edf`` with its ``*-Hypnogram.edf`` (same 6-char prefix + night)."""
    psgs = sorted(Path(raw_root).rglob("*-PSG.edf"))
    hyps = {p.name[:7]: p for p in Path(raw_root).rglob("*-Hypnogram.edf")}
    pairs = []
    for psg in psgs:
        key = psg.name[:7]  # e.g. SC4001E / ST7011J  (subject + night)
        if key in hyps:
            pairs.append((psg.name[:6] + psg.name[6], psg, hyps[key]))
    return pairs


def subject_id(recording_name: str) -> tuple[str, int]:
    match = re.match(r"(SC|ST)\d(\d\d)", recording_name)
    if not match:
        raise ValueError(recording_name)
    return match.group(1), int(match.group(2))


def load_recording(psg: Path, hyp: Path, channel: str, epoch_s: float, wake_margin_min: float):
    import mne

    raw = mne.io.read_raw_edf(psg, include=[channel], preload=True, verbose="error")
    sfreq = float(raw.info["sfreq"])
    signal = raw.get_data()[0] * 1e6  # volts -> microvolts
    annotations = mne.read_annotations(hyp)

    spe = int(round(epoch_s * sfreq))
    n_epochs = signal.size // spe
    labels = np.full(n_epochs, -1, dtype=np.int16)
    for onset, duration, description in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        stage = ANNOTATION_MAP.get(str(description))
        if stage is None:
            continue
        first = int(round(onset / epoch_s))
        count = int(round(duration / epoch_s))
        labels[first : min(first + count, n_epochs)] = STAGE_CODE[stage]

    sleep = np.flatnonzero(labels >= 1)
    if sleep.size == 0:
        return None
    margin = int(round(wake_margin_min * 60 / epoch_s))
    start, stop = max(0, sleep[0] - margin), min(n_epochs, sleep[-1] + margin + 1)
    keep = np.arange(start, stop)
    keep = keep[labels[keep] >= 0]
    epochs = signal[: n_epochs * spe].reshape(n_epochs, spe)[keep].astype(np.float32)
    return epochs, labels[keep].astype(np.int8), keep.astype(np.int32), sfreq


def _load_one(args):
    name, psg, hyp, channel, epoch_s, margin = args
    try:
        return name, load_recording(psg, hyp, channel, epoch_s, margin), None
    except Exception as exc:  # noqa: BLE001 - reported per file, others continue
        return name, None, f"{type(exc).__name__}: {exc}"


def build_cohort(config: dict, cohort: str, workers: int = 4) -> dict:
    root = Path(config["data"]["root"])
    prefix, subjects = COHORT_RULES[cohort]
    pairs = [
        (name, psg, hyp)
        for name, psg, hyp in find_recordings(root / "raw")
        if subject_id(name)[0] == prefix and subject_id(name)[1] in subjects
    ]
    if not pairs:
        raise FileNotFoundError(f"no raw files for cohort {cohort}; run `featurelab download {cohort}`")
    d = config["data"]
    jobs = [(n, p, h, d["channel"], float(d["epoch_s"]), float(d["wake_margin_min"])) for n, p, h in pairs]
    xs, ys, subj, rec, idx, errors = [], [], [], [], [], []
    sfreq = None
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, result, error in pool.map(_load_one, jobs):
            if error or result is None:
                errors.append(f"{name}: {error or 'no sleep epochs'}")
                continue
            epochs, labels, positions, sfreq = result
            kind, number = subject_id(name)
            xs.append(epochs)
            ys.append(labels)
            subj += [f"{kind}{number:02d}"] * len(labels)
            rec += [name] * len(labels)
            idx.append(positions)
    X = np.concatenate(xs)
    y = np.concatenate(ys)
    path = write_cohort(root, cohort, X, y, subj, rec, np.concatenate(idx), sfreq)
    counts = {stage: int((y == code).sum()) for stage, code in STAGE_CODE.items()}
    return {
        "cohort": cohort, "path": str(path), "recordings": len(xs), "subjects": len(set(subj)),
        "epochs": int(len(y)), "stage_counts": counts, "errors": errors,
    }

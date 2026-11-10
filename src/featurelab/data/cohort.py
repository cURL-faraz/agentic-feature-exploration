"""A *cohort* is the unit every other module works with.

On disk (``<root>/cohorts/<name>/``)::

    X.npy      float32 (n_epochs, n_samples)  raw EEG in microvolts, memory-mapped
    meta.npz   y (int8 stage code), subject (str), recording (str), epoch_idx (int32)
    psd.npy    float32 (n_epochs, n_freqs)    Welch PSD, shared by all features
    freqs.npy  float32 (n_freqs,)

Why a plain-array format instead of a database or MNE objects?  Three consumers
must read it: the sandboxed feature runner (a separate process), the statistics
code, and teammates' deep-learning code.  ``np.load(mmap_mode="r")`` works for
all three, costs no RAM until touched, and needs no extra dependency.

Bring-your-own data (e.g. ear-EEG): write the same four arrays with
:func:`write_cohort` and every command in featurelab works on it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal as sps

STAGES = ("W", "N1", "N2", "N3", "REM")
STAGE_CODE = {name: code for code, name in enumerate(STAGES)}


@dataclass
class Cohort:
    name: str
    X: np.ndarray          # (n, samples) memory-mapped
    y: np.ndarray          # (n,) int stage codes
    subject: np.ndarray    # (n,) str
    recording: np.ndarray  # (n,) str
    epoch_idx: np.ndarray  # (n,) position of the epoch inside its recording
    psd: np.ndarray        # (n, f)
    freqs: np.ndarray      # (f,)
    sfreq: float
    path: Path

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    def subjects(self) -> list[str]:
        return sorted(set(self.subject.tolist()))

    def stage_counts(self) -> dict[str, int]:
        return {STAGES[c]: int((self.y == c).sum()) for c in range(len(STAGES))}


def cohort_dir(root: Path, name: str) -> Path:
    return Path(root) / "cohorts" / name


def welch_psd(X: np.ndarray, sfreq: float, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """4-second Welch windows, 50 % overlap -> 0.25 Hz resolution up to Nyquist."""
    nperseg = int(4 * sfreq)
    out = []
    freqs = None
    for start in range(0, X.shape[0], chunk):
        freqs, psd = sps.welch(
            np.asarray(X[start : start + chunk], dtype=np.float64),
            fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1,
        )
        out.append(psd.astype(np.float32))
    return freqs.astype(np.float32), np.concatenate(out, axis=0)


def write_cohort(
    root: Path, name: str, X: np.ndarray, y: np.ndarray, subject, recording, epoch_idx, sfreq: float
) -> Path:
    path = cohort_dir(root, name)
    path.mkdir(parents=True, exist_ok=True)
    X = np.ascontiguousarray(X, dtype=np.float32)
    np.save(path / "X.npy", X)
    np.savez(
        path / "meta.npz",
        y=np.asarray(y, dtype=np.int8),
        subject=np.asarray(subject, dtype=str),
        recording=np.asarray(recording, dtype=str),
        epoch_idx=np.asarray(epoch_idx, dtype=np.int32),
        sfreq=np.float32(sfreq),
    )
    freqs, psd = welch_psd(X, sfreq)
    np.save(path / "psd.npy", psd)
    np.save(path / "freqs.npy", freqs)
    return path


def load_cohort(root: Path, name: str) -> Cohort:
    path = cohort_dir(root, name)
    if not (path / "X.npy").is_file():
        raise FileNotFoundError(
            f"cohort '{name}' not found under {path}. Run `featurelab prepare` first."
        )
    meta = np.load(path / "meta.npz")
    return Cohort(
        name=name,
        X=np.load(path / "X.npy", mmap_mode="r"),
        y=meta["y"].astype(int),
        subject=meta["subject"],
        recording=meta["recording"],
        epoch_idx=meta["epoch_idx"],
        psd=np.load(path / "psd.npy", mmap_mode="r"),
        freqs=np.load(path / "freqs.npy"),
        sfreq=float(meta["sfreq"]),
        path=path,
    )


def available_cohorts(root: Path) -> list[str]:
    base = Path(root) / "cohorts"
    return sorted(p.name for p in base.iterdir() if (p / "X.npy").is_file()) if base.is_dir() else []

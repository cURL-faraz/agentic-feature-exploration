"""A small synthetic cohort with a *planted* effect - for testing the judge itself.

Rule of thumb in ML evaluation: before you trust an evaluator, show it can
(a) find an effect you know is there and (b) reject one you know is not.

Here N1 and REM have the **same average spectrum** (theta-dominated), so standard
band-power features cannot tell them apart.  The planted difference: REM theta
comes in *bursts* (like sawtooth waves), N1 theta is continuous.  A feature that
measures burstiness of the theta envelope (positive control) must pass; a feature
of random noise (negative control) must fail.

Never use this cohort for scientific claims - it validates the machinery only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal as sps

from .cohort import STAGE_CODE, write_cohort

FS = 100.0
N = 3000

# stage -> relative amplitude of (delta, theta, alpha, sigma, beta)
PROFILES = {
    "W": (0.6, 0.6, 1.6, 0.4, 1.2),
    "N1": (0.8, 1.4, 0.5, 0.4, 0.6),
    "N2": (1.0, 1.1, 0.4, 0.7, 0.5),
    "N3": (2.6, 0.7, 0.3, 0.4, 0.3),
    "REM": (0.8, 1.4, 0.5, 0.4, 0.6),
}
BANDS = ((0.5, 4), (4, 8), (8, 12), (12, 16), (16, 30))
TRANSITIONS = {
    "W": {"W": 0.85, "N1": 0.15},
    "N1": {"N1": 0.6, "N2": 0.3, "W": 0.05, "REM": 0.05},
    "N2": {"N2": 0.85, "N3": 0.07, "REM": 0.05, "N1": 0.03},
    "N3": {"N3": 0.85, "N2": 0.15},
    "REM": {"REM": 0.88, "N1": 0.07, "N2": 0.05},
}


def _band_noise(rng, lo, hi):
    sos = sps.butter(4, [lo, hi], btype="band", fs=FS, output="sos")
    x = sps.sosfiltfilt(sos, rng.standard_normal(N + 400))[200:-200]
    return x / (x.std() + 1e-12)


def _epoch(rng, stage: str, subject_gain: float) -> np.ndarray:
    x = 0.3 * np.cumsum(rng.standard_normal(N)) / np.sqrt(N)  # slow drift
    for (lo, hi), amp in zip(BANDS, PROFILES[stage]):
        component = _band_noise(rng, lo, hi) * amp
        if stage == "REM" and lo == 4:  # planted: bursty theta with the same average power
            envelope = 1.0 + 0.9 * np.sin(2 * np.pi * 0.15 * np.arange(N) / FS + rng.uniform(0, 6.3))
            component = component * envelope / np.sqrt(np.mean(envelope**2))
        x = x + component
    x = x + 1.2 * _band_noise(rng, 0.5, 40)  # broadband noise: makes the task realistically hard
    return (x * 20.0 * subject_gain).astype(np.float32)


def generate(root: Path, name: str = "synthetic", n_subjects: int = 8, epochs: int = 240, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    X, y, subj, rec, idx = [], [], [], [], []
    for s in range(n_subjects):
        gain = rng.uniform(0.7, 1.4)  # subjects differ in amplitude (as real people do)
        stage = "W"
        for e in range(epochs):
            X.append(_epoch(rng, stage, gain))
            y.append(STAGE_CODE[stage])
            subj.append(f"SY{s:02d}")
            rec.append(f"SY{s:02d}n1")
            idx.append(e)
            options = TRANSITIONS[stage]
            stage = rng.choice(list(options), p=np.array(list(options.values())) / sum(options.values()))
    return write_cohort(root, name, np.stack(X), np.array(y), subj, rec, np.array(idx), FS)


POSITIVE_CONTROL = '''
import numpy as np
from scipy import signal

def feature(x, fs, freqs, psd):
    """Burstiness of the theta envelope: coefficient of variation of 2-s theta power."""
    sos = signal.butter(4, [4.0, 8.0], btype="band", fs=fs, output="sos")
    theta = signal.sosfiltfilt(sos, x)
    win = int(2 * fs)
    power = (theta[: len(theta) // win * win].reshape(-1, win) ** 2).mean(axis=1)
    return float(power.std() / (power.mean() + 1e-12))
'''

NEGATIVE_CONTROL = '''
import numpy as np

def feature(x, fs, freqs, psd):
    """Pseudo-random value derived from sample digits - carries no stage information."""
    return float((abs(x[17]) * 1e4) % 1.0)
'''

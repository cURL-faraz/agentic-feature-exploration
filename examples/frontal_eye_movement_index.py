"""Frontal eye-movement proxy.

Hypothesis: Fpz-Cz sits close to the eyes, so the rapid eye movements of REM leak
into it as steep, large low-frequency deflections. N1 has slow rolling eye
movements instead. Target confusion: N1 vs REM.

    featurelab evaluate examples/frontal_eye_movement_index.py --higher-in REM --lower-in N1
"""
import numpy as np
from scipy import signal


def feature(x, fs, freqs, psd):
    """Fraction of samples whose low-passed slope exceeds 4x the epoch's median slope."""
    sos = signal.butter(4, 5.0, btype="low", fs=fs, output="sos")
    slope = np.abs(np.diff(signal.sosfiltfilt(sos, x))) * fs
    return float(np.mean(slope > 4 * np.median(slope) + 1e-9))

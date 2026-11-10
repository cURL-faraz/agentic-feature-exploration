"""Theta burstiness.

Hypothesis: REM sawtooth waves come in bursts, so 2-second theta power fluctuates
more inside a REM epoch than inside an N1 epoch. Scale-invariant by construction.

    featurelab evaluate examples/theta_burstiness.py --higher-in REM --lower-in N1
"""
import numpy as np
from scipy import signal


def feature(x, fs, freqs, psd):
    """Coefficient of variation of theta (4-8 Hz) power across 2-s windows."""
    sos = signal.butter(4, [4.0, 8.0], btype="band", fs=fs, output="sos")
    theta = signal.sosfiltfilt(sos, x)
    win = int(2 * fs)
    power = (theta[: len(theta) // win * win].reshape(-1, win) ** 2).mean(axis=1)
    return float(power.std() / (power.mean() + 1e-12))

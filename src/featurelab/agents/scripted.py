"""Canned agent replies: run the whole loop offline, with no API key.

The sequence deliberately exercises every path of the graph:
1. a proposal whose code breaks the contract (imports ``os``) -> gate -> repair
2. the repaired proposal (a frontal eye-movement proxy: Fpz-Cz picks up REMs)
3. a theta-burstiness proposal (sawtooth waves come in bursts during REM)
4. a deliberately useless proposal -> must be rejected by the statistics

That makes the scripted run a *system test* as well as a demo.
"""

from __future__ import annotations

import json
import re

EYE_META = {
    "name": "frontal_eye_movement_index",
    "hypothesis": "Fpz-Cz is close to the eyes, so rapid eye movements in REM leak in as steep, "
                  "large low-frequency deflections; N1 has slow rolling eye movements instead.",
    "formula": "fraction of samples where |d/dt lowpass_5Hz(x)| > 4 * median(|d/dt lowpass_5Hz(x)|)",
    "target_confusion": "N1 vs REM",
    "prediction": {"higher_in": "REM", "lower_in": "N1"},
    "inspired_by": [],
}

EYE_BUGGY = '''
import os
import numpy as np
from scipy import signal

def feature(x, fs, freqs, psd):
    """Steep low-frequency deflections (eye-movement proxy)."""
    sos = signal.butter(4, 5.0, btype="low", fs=fs, output="sos")
    slope = np.abs(np.diff(signal.sosfiltfilt(sos, x))) * fs
    return float(np.mean(slope > 4 * np.median(slope) + 1e-9))
'''

EYE_FIXED = EYE_BUGGY.replace("import os\n", "")

THETA_META = {
    "name": "theta_burstiness",
    "hypothesis": "REM sawtooth waves occur in bursts, so 2-s theta power fluctuates more within an "
                  "epoch in REM than in N1, where theta is more continuous.",
    "formula": "std(P_theta(w)) / mean(P_theta(w)) over consecutive 2-s windows w",
    "target_confusion": "N1 vs REM",
    "prediction": {"higher_in": "REM", "lower_in": "N1"},
    "inspired_by": [],
}

THETA_CODE = '''
import numpy as np
from scipy import signal

def feature(x, fs, freqs, psd):
    """Coefficient of variation of theta power across 2-second windows."""
    sos = signal.butter(4, [4.0, 8.0], btype="band", fs=fs, output="sos")
    theta = signal.sosfiltfilt(sos, x)
    win = int(2 * fs)
    power = (theta[: len(theta) // win * win].reshape(-1, win) ** 2).mean(axis=1)
    return float(power.std() / (power.mean() + 1e-12))
'''

NOISE_META = {
    "name": "sample_digit_entropy",
    "hypothesis": "(negative control) the fractional digits of the signal carry no stage information.",
    "formula": "frac(|x[17]| * 1e4)",
    "target_confusion": "N1 vs N2",
    "prediction": {"higher_in": "N2", "lower_in": "N1"},
    "inspired_by": [],
}

NOISE_CODE = '''
import numpy as np

def feature(x, fs, freqs, psd):
    """Fractional digits of one sample - a negative control."""
    return float((abs(x[17]) * 1e4) % 1.0)
'''


def _proposal(meta: dict, code: str) -> str:
    return f"```json\n{json.dumps(meta, indent=1)}\n```\n\n```python\n{code.strip()}\n```"


def _scout(system: str, user: str) -> str:
    ids = re.findall(r"paper_id: (\S+)", user)
    cards = [{"paper_id": ids[0], "idea": "eye-movement artifacts on frontal EEG mark REM",
              "mechanism": "Fpz is near the eyes", "target_confusion": "N1 vs REM"}] if ids else []
    return json.dumps({"cards": cards})


APPROVE = json.dumps({"verdict": "approve", "code_matches_formula": True, "issues": [], "suggestion": ""})


def demo_responses() -> dict:
    return {
        "scout": [_scout],
        "inventor": [
            _proposal(EYE_META, EYE_BUGGY),
            _proposal(EYE_META, EYE_FIXED),
            _proposal(THETA_META, THETA_CODE),
            _proposal(NOISE_META, NOISE_CODE),
        ],
        "critic": [APPROVE],
    }
